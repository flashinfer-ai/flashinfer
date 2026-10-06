"""Sparse MLA decode in the regimes a serving planner produces.

Top-k rows are padded with -1 past each token's live length, live lengths can
be far shorter than the plan's width (so most split CTAs have no active
chunk), and the plan is replayed under a CUDA graph with mutated lengths.
"""

from __future__ import annotations

import pytest
import torch

from b12x.attention import sparse_mla
from b12x.attention._shared.mla.reference import (
    pack_mla_kv_cache_reference,
    sparse_mla_reference,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for sparse MLA decode"
)

_HEADS = 8
_WIDTH = 2048
_QK_DIM = 576
_V_DIM = 512
_SM_SCALE = _QK_DIM**-0.5


def _plan(device: torch.device, rows: int):
    plan = sparse_mla.plan(
        sparse_mla.Caps(
            device=device,
            num_q_heads=_HEADS,
            max_q_rows=rows,
            max_width=_WIDTH,
            softmax_scale=_SM_SCALE,
            kv_dtype=torch.uint8,
        )
    )
    (spec,) = plan.scratch_specs()[:1]
    return plan, torch.empty(spec.shape, dtype=spec.dtype, device=device)


def _kv_cache(device: torch.device, kv_rows: int, generator: torch.Generator):
    k_nope = (torch.randn(kv_rows, 512, generator=generator) / 4).to(torch.bfloat16)
    k_rope = (torch.randn(kv_rows, 64, generator=generator) / 4).to(torch.bfloat16)
    return pack_mla_kv_cache_reference(k_nope.to(device), k_rope.to(device))


def _padded_selection(
    lengths: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sorted valid candidates per token, then -1 past the token's active count."""
    rows = int(lengths.shape[0])
    active = torch.minimum(lengths, torch.tensor(_WIDTH, dtype=torch.int32))
    selected = torch.full((rows, _WIDTH), -1, dtype=torch.int32)
    for row in range(rows):
        count = int(active[row])
        selected[row, :count] = (
            torch.randperm(int(lengths[row]), generator=generator)[:count]
            .sort()
            .values.to(torch.int32)
        )
    return selected, active


def _assert_matches_reference(output, q, kv, kv_rows, selected, active) -> None:
    expected = sparse_mla_reference(
        q_all=q,
        kv_cache=kv.reshape(kv_rows, 1, -1),
        page_table_1=selected,
        active_token_counts=active,
        sm_scale=_SM_SCALE,
        v_head_dim=_V_DIM,
    )
    assert torch.isfinite(output.float()).all()
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), expected.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999, cosine


@requires_cuda
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_decode_padded_rows_and_short_lengths_match_reference(seed: int) -> None:
    device = torch.device("cuda")
    kv_rows, rows = 8192, 32
    generator = torch.Generator(device="cpu").manual_seed(seed)
    plan, scratch = _plan(device, rows)
    kv = _kv_cache(device, kv_rows, generator)
    lengths = torch.cat(
        [
            torch.randint(1, 70, (rows // 2,), generator=generator),
            torch.randint(1, kv_rows, (rows - rows // 2,), generator=generator),
        ]
    ).to(torch.int32)
    selected, active = _padded_selection(lengths, generator)
    q = (torch.randn((rows, _HEADS, _QK_DIM), generator=generator) / 4).to(
        torch.bfloat16
    )
    q, selected, lengths, active = (
        t.to(device) for t in (q, selected, lengths, active)
    )
    binding = sparse_mla.bind(
        plan,
        scratch=scratch,
        q=q,
        kv_cache=kv,
        selected_indices=selected,
        cache_lengths=lengths,
        selected_lengths=active,
    )
    output = sparse_mla.run(binding)
    torch.cuda.synchronize(device)
    _assert_matches_reference(output, q, kv, kv_rows, selected, active)


@requires_cuda
def test_decode_wide_plan_with_short_live_lengths_replays_under_graph() -> None:
    """A plan sized for the full width leaves most split CTAs without a chunk."""
    device = torch.device("cuda")
    kv_rows, rows = 4096, 8
    generator = torch.Generator(device="cpu").manual_seed(11)
    plan, scratch = _plan(device, rows)
    kv = _kv_cache(device, kv_rows, generator)
    q = (torch.randn((rows, _HEADS, _QK_DIM), generator=generator) / 4).to(
        torch.bfloat16
    ).to(device)
    lengths_a = torch.tensor([3, 64, 65, 200, 330, 1, 127, 512], dtype=torch.int32)
    lengths_b = torch.tensor([1, 2, 130, 64, 4096, 5, 63, 66], dtype=torch.int32)
    selected_a, active_a = _padded_selection(lengths_a, generator)
    selected_b, active_b = _padded_selection(lengths_b, generator)
    selected = selected_a.to(device)
    lengths = lengths_a.to(device)
    active = active_a.to(device)
    binding = sparse_mla.bind(
        plan,
        scratch=scratch,
        q=q,
        kv_cache=kv,
        selected_indices=selected,
        cache_lengths=lengths,
        selected_lengths=active,
    )
    output = sparse_mla.run(binding)
    torch.cuda.synchronize(device)
    _assert_matches_reference(output, q, kv, kv_rows, selected, active)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = sparse_mla.run(binding)
    # Replay with different live lengths written in place: no pointer, shape,
    # or plan changes.
    selected.copy_(selected_b.to(device))
    lengths.copy_(lengths_b.to(device))
    active.copy_(active_b.to(device))
    graph.replay()
    torch.cuda.synchronize(device)
    _assert_matches_reference(captured, q, kv, kv_rows, selected, active)
