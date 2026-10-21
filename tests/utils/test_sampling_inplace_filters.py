"""Aliased filtering preserves row policy and scratch state through captured replays."""

import pytest
import torch

from flashinfer import sampling


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows,vocabulary", [(1, 257), (7, 4097), (192, 129280)])
@torch.no_grad()
def test_top_k_exact_alias_matches_disjoint_output_across_row_policy_changes(
    dtype, rows, vocabulary
):
    torch.manual_seed(4319)
    source = torch.randn((rows, vocabulary), device="cuda", dtype=dtype)
    source[:, ::23] = 0
    source[:, 0] = float("inf")
    source[:, 1] = -float("inf")
    inplace = torch.empty_like(source)
    separate = torch.empty_like(source)
    top_k = torch.full((rows,), vocabulary, dtype=torch.int32, device="cuda")
    states = [
        torch.zeros(1024 * 1024, dtype=torch.uint8, device="cuda") for _ in range(2)
    ]
    sampling.get_sampling_module()
    operation = torch.ops.flashinfer.top_k_mask_logits_out.default

    def forward():
        inplace.copy_(source)
        operation(inplace, inplace, top_k, 0, states[0])
        operation(source, separate, top_k, 0, states[1])

    forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    try:
        for turn in range(9):
            source.normal_()
            source[:, ::23] = 0
            source[:, 0] = float("inf")
            source[:, 1] = -float("inf")
            values = (
                [vocabulary, 1, 17, vocabulary + 1] if turn % 3 else [vocabulary] * 4
            )
            policy = torch.tensor(
                [values[(row + turn) % 4] for row in range(rows)],
                device="cuda",
                dtype=torch.int32,
            )
            top_k.copy_(policy)
            inplace.fill_(float("nan"))
            separate.fill_(float("nan"))
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            assert torch.equal(inplace, separate)
            assert torch.equal(states[0], states[1])
            assert torch.equal(
                source[:, 0], torch.full_like(source[:, 0], float("inf"))
            )
            for row, k in enumerate(policy.cpu().tolist()):
                if k >= vocabulary:
                    expected = source[row]
                else:
                    threshold = source[row].topk(k).values[-1]
                    expected = torch.where(
                        source[row] >= threshold, source[row], -float("inf")
                    )
                assert torch.equal(inplace[row], expected)
    finally:
        graph.reset()


@torch.no_grad()
def test_top_k_exact_alias_survives_compilation_and_capture():
    from torch.fx.experimental.proxy_tensor import make_fx

    sampling.get_sampling_module()
    scores = torch.randn((8, 129280), device="cuda")
    original = scores.clone()
    top_k = torch.tensor([1, 17, 129280, 129281] * 2, dtype=torch.int32, device="cuda")
    states = torch.zeros(1024 * 1024, dtype=torch.uint8, device="cuda")

    def forward(scores, top_k, states):
        torch.ops.flashinfer.top_k_mask_logits_out(scores, scores, top_k, 0, states)

    traced = make_fx(forward)(scores, top_k, states)
    compiled = torch.compile(traced, fullgraph=True, dynamic=False)
    compiled(scores, top_k, states)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(scores, top_k, states)
    try:
        for turn in range(6):
            original.normal_()
            scores.copy_(original)
            top_k.copy_(
                torch.tensor(
                    ([129280] * 8 if turn % 2 else [1, 17, 129280, 129281] * 2),
                    device="cuda",
                    dtype=torch.int32,
                )
            )
            expected = sampling.top_k_mask_logits(original, top_k)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(scores, expected)
    finally:
        graph.reset()


@pytest.mark.parametrize(
    "rows,vocabulary,deterministic,expected",
    [
        (1, 257, True, 27136),
        (1, 129280, True, 57856),
        (1, 129280, False, 49664),
        (192, 129280, True, 11083776),
    ],
)
def test_top_p_workspace_capacity_matches_native_storage_extents(
    rows, vocabulary, deterministic, expected
):
    assert (
        sampling._top_p_renorm_workspace_size(rows, vocabulary, deterministic)
        == expected
    )
