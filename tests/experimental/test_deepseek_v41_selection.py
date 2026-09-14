# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

from flashinfer.deepseek_v41 import (
    deepseek_v41_candidate_blocks,
    deepseek_v41_candidate_workspace,
    deepseek_v41_token_topk,
    deepseek_v41_token_workspace,
)


def certificate(scores, visible, selected):
    """Verify exact top-k values and set validity, allowing tied-boundary IDs."""
    nq, nk = scores.shape
    nb = (nk + 7) // 8
    columns = torch.arange(nk, device=scores.device)
    masked = scores.masked_fill(columns[None, :] >= visible[:, None], -torch.inf)
    values = (
        torch.nn.functional.pad(masked, (0, nb * 8 - nk), value=-torch.inf)
        .view(nq, nb, 8)
        .amax(-1)
    )
    for row in range(nq):
        length = int(visible[row])
        valid_blocks = (length + 7) // 8
        count = min(selected.shape[1], valid_blocks)
        ids = selected[row]
        valid_ids = ids[ids >= 0].long()
        assert valid_ids.numel() == count
        assert bool((ids[ids < 0] == -1).all())
        assert bool((valid_ids < valid_blocks).all())
        assert valid_ids.unique().numel() == count
        assert bool((valid_ids[1:] > valid_ids[:-1]).all())
        if count:
            assert bool((valid_ids == (length - 1) // 8).any())
            values[row, (length - 1) // 8] = torch.inf
            actual = values[row, valid_ids].sort().values
            expected = values[row, :valid_blocks].topk(count).values.sort().values
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("nk,topk", [(641, 8), (65537, 2048)])
def test_causal_candidate_newest_tail_and_changed_graph(nk, topk):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100/SM103")
    pytest.importorskip("deep_select")
    torch.manual_seed(41 + nk)
    scores = torch.randn(4, nk, device="cuda", dtype=torch.bfloat16)
    visible = torch.tensor([0, 5, nk - 3, nk], device="cuda", dtype=torch.int32)
    # Latest block must survive despite being the worst-scoring reachable one.
    scores[:, -8:] = -100
    workspace = deepseek_v41_candidate_workspace(scores, topk_blocks=topk)
    selected = deepseek_v41_candidate_blocks(
        scores, visible, topk_blocks=topk, workspace=workspace
    )
    certificate(scores, visible, selected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        deepseek_v41_candidate_blocks(
            scores, visible, topk_blocks=topk, workspace=workspace
        )
    scores.mul_(-1)
    visible.copy_(torch.tensor([nk, nk - 1, 1, 17], device="cuda", dtype=torch.int32))
    graph.replay()
    certificate(scores, visible, selected)
    torch.cuda.set_sync_debug_mode("error")
    try:
        deepseek_v41_candidate_blocks(
            scores, visible, topk_blocks=topk, workspace=workspace
        )
    finally:
        torch.cuda.set_sync_debug_mode("default")


def token_certificate(scores, visible, selected, candidates=None):
    if candidates is None:
        positions = torch.arange(scores.shape[1], device=scores.device).expand_as(
            scores
        )
    else:
        positions = (
            candidates.long()[..., None] * 8 + torch.arange(8, device=scores.device)
        ).flatten(1)
    valid = (positions >= 0) & (positions < visible[:, None])
    masked = scores.masked_fill(~valid, -torch.inf)
    for row in range(scores.shape[0]):
        ids = selected[row]
        actual_ids = ids[ids >= 0].long()
        count = min(int(valid[row].sum()), selected.shape[1])
        assert actual_ids.numel() == count
        assert bool((ids[ids < 0] == -1).all())
        assert bool((actual_ids[1:] > actual_ids[:-1]).all())
        if count:
            matches = positions[row, :, None] == actual_ids[None, :]
            assert bool((matches.sum(0) == 1).all())
            actual = masked[row, matches.long().argmax(0)].sort().values
            expected = masked[row].topk(count).values.sort().values
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "sparse,width", [(False, 641), (False, 65537), (True, 104), (True, 16384)]
)
@pytest.mark.parametrize("aligned", [False, True])
def test_token_topk_dense_candidate_mapping_and_graph(sparse, width, aligned):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100/SM103")
    pytest.importorskip("deep_select")
    torch.manual_seed(41 + width)
    # Exercise both the copying path and direct aligned provider scores.
    stride = (width + 511) // 512 * 512 if aligned else width + 3
    scores = torch.randn(4, stride, device="cuda", dtype=torch.bfloat16)[:, :width]
    if sparse:
        nc = width // 8
        candidates = (
            torch.arange(nc, device="cuda", dtype=torch.int32) * 3 + 2
        ).repeat(4, 1)
        candidates[0] = -1
        candidates[1, nc // 2 :] = -1
        length = nc * 3 * 8 + 11
        visible = torch.tensor(
            [0, length, length - 17, length], device="cuda", dtype=torch.int32
        )
        scores[0] = torch.nan  # Undefined provider scores outside visibility.
        scores[1, nc // 2 * 8 :] = torch.nan
    else:
        candidates = None
        visible = torch.tensor(
            [0, 5, width - 3, width], device="cuda", dtype=torch.int32
        )
        scores[0] = torch.nan
        scores[1, 5:] = torch.nan
    workspace = deepseek_v41_token_workspace(scores)

    def native():
        return deepseek_v41_token_topk(
            scores, visible, candidates=candidates, workspace=workspace
        )

    selected = native()
    token_certificate(scores, visible, selected, candidates)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        native()
    scores.copy_(torch.randn_like(scores))
    if sparse:
        candidates.copy_(
            (torch.arange(width // 8, device="cuda", dtype=torch.int32) * 2).repeat(
                4, 1
            )
        )
        visible.copy_(
            torch.tensor([width * 2, 17, 1, width], device="cuda", dtype=torch.int32)
        )
    else:
        visible.copy_(
            torch.tensor([width, width - 1, 1, 17], device="cuda", dtype=torch.int32)
        )
    graph.replay()
    token_certificate(scores, visible, selected, candidates)
    torch.cuda.set_sync_debug_mode("error")
    try:
        native()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    overlapping = dict(workspace, local_ids=workspace["out"])
    with pytest.raises(ValueError, match="overlap"):
        deepseek_v41_token_topk(
            scores, visible, candidates=candidates, workspace=overlapping
        )
