# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
import pytest
import torch

pytest.importorskip("triton")
from flashinfer.testing.sparse_mla_metadata import (
    map_sparse_indices,
    prepare_causal_indices,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def test_vllm_mapping_compacts_holes_and_refreshes_tail():
    device = "cuda"
    table = torch.tensor([[2, 0], [1, 3]], device=device, dtype=torch.int32)
    req = torch.tensor([0, 1, 0], device=device, dtype=torch.int32)
    idx = torch.tensor(
        [[0, -1, 5, 7], [6, 3, -1, 0], [0, 1, 2, 3]], device=device, dtype=torch.int32
    )
    lens = torch.tensor([4, 4, 0], device=device, dtype=torch.int32)
    out, counts = map_sparse_indices(
        idx,
        lens,
        page_size=4,
        page_stride_rows=6,
        block_table=table,
        token_to_request=req,
    )
    torch.testing.assert_close(
        out,
        torch.tensor(
            [[12, 1, 3, -1], [20, 9, 6, -1], [-1, -1, -1, -1]],
            device=device,
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        counts, torch.tensor([3, 3, 0], device=device, dtype=torch.int32)
    )
    lens.zero_()
    map_sparse_indices(
        idx,
        lens,
        page_size=4,
        page_stride_rows=6,
        block_table=table,
        token_to_request=req,
        out=out,
        counts=counts,
    )
    assert (out == -1).all() and (counts == 0).all()


@pytest.mark.parametrize("swa,ratio", [(True, 1), (False, 2), (False, 128)])
def test_vllm_causal_metadata(swa, ratio):
    pos = torch.tensor([0, 7, 15, -1], device="cuda", dtype=torch.int32)
    req = torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.int32)
    table = torch.tensor([[2, 0, 5, 1], [1, 3, 6, 4]], device="cuda", dtype=torch.int32)
    width = 4 if swa else 8
    out, lengths = prepare_causal_indices(
        pos,
        req,
        table,
        max_topk=width,
        page_size=4,
        page_stride_rows=6,
        swa=swa,
        compression_ratio=ratio,
    )
    reference = torch.full_like(out, -1)
    counts = torch.zeros_like(lengths)
    for row, position in enumerate(pos.cpu().tolist()):
        if position < 0:
            continue
        tokens = (
            list(range(max(0, position + 1 - width), position + 1))
            if swa
            else list(range(min((position + 1) // ratio, width)))
        )
        counts[row] = len(tokens)
        for col, token in enumerate(tokens):
            reference[row, col] = table[req[row], token // 4] * 6 + token % 4
    torch.testing.assert_close(out, reference)
    torch.testing.assert_close(lengths, counts)
