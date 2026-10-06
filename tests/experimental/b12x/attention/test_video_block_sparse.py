# SPDX-License-Identifier: Apache-2.0
"""Block-list (video) sparse attention: MiniMax-H3 / FastH3-style learned block selection.

The kernel walks an authoritative CSR list of K blocks per q tile instead of the contiguous
[n_block_min, n_block_max) range. These tests are the contract:

- a FULL list must reproduce dense attention EXACTLY (not merely within tolerance);
- a sparse list must equal the fp32 oracle computed with the same mask;
- an EMPTY list must attend nothing -- the "sparse is not silently dense" check. This is the
  important one: a full list is indistinguishable from dense, so a sparse kernel that never
  runs its walk still passes the equality test. Only the empty/degenerate case proves the walk
  is live.
"""
from __future__ import annotations

import pytest
import torch

from ..conftest import require_b12x as require_sm120

TILE_M = 128
BLOCK_K = 64  # FastH3 VSA's logical block size; b12x's varlen tile_n is also 64


def _require_backend():
    require_sm120()
    pytest.importorskip("cutlass")
    pytest.importorskip("cuda.bindings.driver")
    # An indexed device: the prepared plan records an ordinal, and an unindexed
    # torch.device("cuda") compares unequal against it in the preparation layer.
    return torch.device("cuda", torch.cuda.current_device())


def _list(S: int, radius: int | None, device):
    """CSR list per q tile. radius=None -> every tile lists every block (the full list)."""
    num_blocks = (S + BLOCK_K - 1) // BLOCK_K
    num_tiles = (S + TILE_M - 1) // TILE_M
    idx, off = [], [0]
    for m in range(num_tiles):
        if radius is None:
            blocks = range(num_blocks)
        else:
            c = (m * TILE_M) // BLOCK_K
            blocks = range(max(0, c - radius), min(num_blocks, c + radius + 1))
        idx.extend(blocks)
        off.append(len(idx))
    return (torch.tensor(idx, device=device, dtype=torch.int32),
            torch.tensor(off, device=device, dtype=torch.int32))


def _mask(S: int, block_indices, block_offsets, device):
    mask = torch.zeros(S, S, dtype=torch.bool)
    indices, offsets = block_indices.cpu().tolist(), block_offsets.cpu().tolist()
    for m in range(len(offsets) - 1):
        for block in indices[offsets[m]:offsets[m + 1]]:
            mask[m * TILE_M:min((m + 1) * TILE_M, S),
                 block * BLOCK_K:min((block + 1) * BLOCK_K, S)] = True
    return mask.to(device)


def _oracle(q, k, v, mask):
    qq = q.transpose(0, 1).unsqueeze(0).float()
    kk = k.transpose(0, 1).unsqueeze(0).float()
    vv = v.transpose(0, 1).unsqueeze(0).float()
    scores = (qq @ kk.transpose(-1, -2)) * q.shape[-1] ** -0.5
    scores.masked_fill_(~mask, float("-inf"))
    out = torch.nan_to_num(scores.softmax(-1)) @ vv
    return out.squeeze(0).transpose(0, 1).to(torch.bfloat16)


def _case(S, H, D, radius, device, seed=17):
    torch.manual_seed(seed)
    q = torch.randn(S, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(S, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(S, H, D, device=device, dtype=torch.bfloat16)
    cu = torch.tensor([0, S], device=device, dtype=torch.int32)
    bi, bo = _list(S, radius, device)
    return q, k, v, cu, bi, bo


def _run(S, H, D, radius, device):
    """Execute the sparse path for the given list; returns (output, mask)."""
    from b12x.attention import varlen
    from b12x.preparation import PreparationSession, PreparedCall, require_prepared

    q, k, v, cu, bi, bo = _case(S, H, D, radius, device)
    num_tiles = (S + TILE_M - 1) // TILE_M
    declaration = varlen.plan(
        q, k, v, cu, cu, max_seqlen_q=S, max_seqlen_k=S, causal=False,
        block_sparse=True, num_q_tiles=num_tiles,
        total_blocks_cap=max(1, int(bo[-1].item())),
    )
    with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
        def prepare(state):
            spec, = state.scratch_plan.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            binding = state.bind(scratch=scratch, q=q, k=k, v=v, cu_seqlens_q=cu,
                                 cu_seqlens_k=cu, block_indices=bi, block_offsets=bo)
            return PreparedCall(run=lambda: state.run(binding), owners=(scratch,))

        session.prepare((declaration.request(name="video-block-sparse", prepare_call=prepare),))
        state = require_prepared(declaration, "attention.varlen", device)
        spec, = state.scratch_plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        binding = varlen.bind(declaration, scratch=scratch, q=q, k=k, v=v, cu_seqlens_q=cu,
                              cu_seqlens_k=cu, max_seqlen_q=S, max_seqlen_k=S,
                              block_indices=bi, block_offsets=bo)
        result = state.run(binding)
        torch.cuda.synchronize()
    out = result[0] if isinstance(result, tuple) else result
    return out, _mask(S, bi, bo, device), (q, k, v, state, binding), declaration


def _dense(S, H, D, device):
    from b12x.attention import varlen
    from b12x.preparation import PreparationSession, PreparedCall, require_prepared

    q, k, v, cu, _, _ = _case(S, H, D, None, device)
    declaration = varlen.plan(q, k, v, cu, cu, max_seqlen_q=S, max_seqlen_k=S, causal=False)
    with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
        def prepare(state):
            spec, = state.scratch_plan.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            binding = state.bind(scratch=scratch, q=q, k=k, v=v, cu_seqlens_q=cu, cu_seqlens_k=cu)
            return PreparedCall(run=lambda: state.run(binding), owners=(scratch,))

        session.prepare((declaration.request(name="video-dense", prepare_call=prepare),))
        state = require_prepared(declaration, "attention.varlen", device)
        spec, = state.scratch_plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        binding = varlen.bind(declaration, scratch=scratch, q=q, k=k, v=v, cu_seqlens_q=cu,
                              cu_seqlens_k=cu, max_seqlen_q=S, max_seqlen_k=S)
        result = state.run(binding)
        torch.cuda.synchronize()
    return (result[0] if isinstance(result, tuple) else result), q, k, v


def test_full_list_matches_dense_exactly():
    """A list that admits every block must reproduce dense attention bit for bit."""
    device = _require_backend()
    sparse, _, _, _ = _run(1024, 8, 128, None, device)
    dense, _, _, _ = _dense(1024, 8, 128, device)
    assert torch.equal(sparse, dense), (
        f"full list != dense: max|d|={(sparse.float() - dense.float()).abs().max().item()}"
    )


def test_sparse_matches_masked_oracle():
    """A partial list must equal the fp32 oracle restricted to the same mask."""
    device = _require_backend()
    out, mask, (q, k, v, _, _), _ = _run(1024, 8, 128, 3, device)
    assert 0.0 < mask.float().mean().item() < 1.0, "the test list must be genuinely sparse"
    ref = _oracle(q, k, v, mask)
    cosine = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    assert cosine >= 0.9999, f"cosine {cosine} vs masked oracle"


def test_empty_list_attends_nothing():
    """The walk must be live: with no blocks listed the output cannot be dense.

    This is the gate that catches an inert sparse path -- a full list passes the equality test
    even when the walk never runs, because the full list IS dense.
    """
    device = _require_backend()
    S, H, D = 256, 4, 128
    from b12x.attention import varlen
    from b12x.preparation import PreparationSession, PreparedCall, require_prepared

    q, k, v, cu, _, _ = _case(S, H, D, None, device)
    empty_idx = torch.zeros(1, device=device, dtype=torch.int32)
    empty_off = torch.zeros((S + TILE_M - 1) // TILE_M + 1, device=device, dtype=torch.int32)
    declaration = varlen.plan(
        q, k, v, cu, cu, max_seqlen_q=S, max_seqlen_k=S, causal=False,
        block_sparse=True, num_q_tiles=(S + TILE_M - 1) // TILE_M, total_blocks_cap=1,
    )
    with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
        def prepare(state):
            spec, = state.scratch_plan.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            binding = state.bind(scratch=scratch, q=q, k=k, v=v, cu_seqlens_q=cu,
                                 cu_seqlens_k=cu, block_indices=empty_idx,
                                 block_offsets=empty_off)
            return PreparedCall(run=lambda: state.run(binding), owners=(scratch,))

        session.prepare((declaration.request(name="video-empty", prepare_call=prepare),))
        state = require_prepared(declaration, "attention.varlen", device)
        spec, = state.scratch_plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        binding = varlen.bind(declaration, scratch=scratch, q=q, k=k, v=v, cu_seqlens_q=cu,
                              cu_seqlens_k=cu, max_seqlen_q=S, max_seqlen_k=S,
                              block_indices=empty_idx, block_offsets=empty_off)
        result = state.run(binding)
        torch.cuda.synchronize()
    out = result[0] if isinstance(result, tuple) else result

    dense_ref, _, _, _ = _dense(S, H, D, device)
    cosine = torch.nn.functional.cosine_similarity(
        out.float().flatten(), dense_ref.float().flatten(), dim=0
    ).item()
    assert (out != 0).float().mean().item() < 0.5, "empty list must not return a dense result"
    assert cosine < 0.9, f"empty list still resembles dense (cosine {cosine}) -- the walk is inert"


@pytest.mark.parametrize("S", [1000, 4097])
def test_partial_last_block(S):
    """S not a multiple of BLOCK_K leaves a partial final block that must be masked."""
    device = _require_backend()
    out, mask, (q, k, v, _, _), _ = _run(S, 8, 128, 2, device)
    ref = _oracle(q, k, v, mask)
    cosine = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    assert cosine >= 0.9999, f"S={S} cosine {cosine}"


@pytest.mark.parametrize("block", [0, 1])
def test_list_index_mapping(block):
    """Listing block k must attend exactly block k."""
    device = _require_backend()
    from b12x.attention import varlen
    from b12x.preparation import PreparationSession, PreparedCall, require_prepared

    S, H, D = 512, 4, 128
    q, k, v, cu, _, _ = _case(S, H, D, None, device)
    num_tiles = (S + TILE_M - 1) // TILE_M
    idx, off = [], [0]
    for _ in range(num_tiles):
        idx.append(block)
        off.append(len(idx))
    bi = torch.tensor(idx, device=device, dtype=torch.int32)
    bo = torch.tensor(off, device=device, dtype=torch.int32)
    declaration = varlen.plan(q, k, v, cu, cu, max_seqlen_q=S, max_seqlen_k=S, causal=False,
                              block_sparse=True, num_q_tiles=num_tiles,
                              total_blocks_cap=int(bo[-1].item()))
    with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
        def prepare(state):
            spec, = state.scratch_plan.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            binding = state.bind(scratch=scratch, q=q, k=k, v=v, cu_seqlens_q=cu,
                                 cu_seqlens_k=cu, block_indices=bi, block_offsets=bo)
            return PreparedCall(run=lambda: state.run(binding), owners=(scratch,))

        session.prepare((declaration.request(name=f"video-map-{block}", prepare_call=prepare),))
        state = require_prepared(declaration, "attention.varlen", device)
        spec, = state.scratch_plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        binding = varlen.bind(declaration, scratch=scratch, q=q, k=k, v=v, cu_seqlens_q=cu,
                              cu_seqlens_k=cu, max_seqlen_q=S, max_seqlen_k=S,
                              block_indices=bi, block_offsets=bo)
        result = state.run(binding)
        torch.cuda.synchronize()
    out = result[0] if isinstance(result, tuple) else result
    ref = _oracle(q, k, v, _mask(S, bi, bo, device))
    cosine = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    assert cosine >= 0.9999, f"block {block} cosine {cosine}"


def test_bind_allocates_no_device_memory():
    """Bindings must not allocate: the vLLM path has to be CUDA-graph capturable."""
    device = _require_backend()
    _, _, (q, k, v, state, binding), declaration = _run(2048, 8, 128, 3, device)
    spec, = state.scratch_plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
    # refresh the CSR tensors from the same list the case used
    bi, bo = _list(2048, 3, device)
    from b12x.attention import varlen
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    varlen.bind(declaration, scratch=scratch, q=q, k=k, v=v,
                cu_seqlens_q=torch.tensor([0, 2048], device=device, dtype=torch.int32),
                cu_seqlens_k=torch.tensor([0, 2048], device=device, dtype=torch.int32),
                max_seqlen_q=2048, max_seqlen_k=2048, block_indices=bi, block_offsets=bo)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before, "bind allocated device memory"


def test_cuda_graph_capture_and_replay():
    """The sparse path must be CUDA-graph capturable and replay to the same output."""
    device = _require_backend()
    _, _, (q, k, v, state, binding), _ = _run(2048, 8, 128, 3, device)
    reference = state.run(binding)
    reference = (reference[0] if isinstance(reference, tuple) else reference).clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    state.run(binding)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        captured = state.run(binding)
    graph.replay()
    torch.cuda.synchronize()
    captured = captured[0] if isinstance(captured, tuple) else captured
    assert torch.equal(captured, reference), (
        f"graph replay differs: max|d|={(captured.float() - reference.float()).abs().max().item()}"
    )


def _binding(q, k, v, cu, bi, bo, **kwargs):
    from b12x.attention import varlen
    from b12x.attention.varlen import VarlenAttentionConfig
    from b12x.preparation import require_prepared

    length = int((cu[1:] - cu[:-1]).max())
    plan = varlen.plan(
        q, k, v, cu, cu, max_seqlen_q=length, max_seqlen_k=length,
        block_sparse=True, num_q_tiles=bo.numel() - 1, total_blocks_cap=bi.numel(),
        override=VarlenAttentionConfig(tile_m=TILE_M, tile_n=BLOCK_K), **kwargs,
    )
    state = require_prepared(plan, "attention.varlen", q.device)
    spec, = state.scratch_plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=q.device)
    args = dict(scratch=scratch, q=q, k=k, v=v, cu_seqlens_q=cu,
                cu_seqlens_k=cu, block_indices=bi, block_offsets=bo)
    return state, state.bind(**args), args


@pytest.mark.parametrize("causal,window", [(True, None), (False, (1, 1))])
def test_sparse_list_is_authoritative_over_causal_and_local_masks(causal, window):
    device = _require_backend()
    q, k, v, cu, bi, bo = _case(256, 4, 128, None, device)
    state, binding, _ = _binding(q, k, v, cu, bi, bo, causal=causal, window_size=window)
    actual = state.run(binding)[0]
    expected = _oracle(q, k, v, _mask(256, bi, bo, device))
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)


def test_sparse_partial_block_masks_nonfinite_values_from_next_segment():
    device = _require_backend()
    length = 129
    q = torch.randn(length * 2, 2, 128, device=device, dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    v[length:].fill_(float("nan"))
    cu = torch.tensor([0, length, 2 * length], device=device, dtype=torch.int32)
    # Reverse traversal visits block zero before the partial block.
    bi = torch.tensor([2, 0, 2, 0], device=device, dtype=torch.int32)
    bo = torch.tensor([0, 2, 4], device=device, dtype=torch.int32)
    state, binding, _ = _binding(q, k, v, cu, bi, bo)
    actual = state.run(binding)[0][:length]
    assert torch.isfinite(actual).all()
    expected = _oracle(q[:length], k[:length], v[:length], _mask(length, bi, bo, device))
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)


def test_sparse_grouped_query_heads_keep_logical_query_tiles():
    device = _require_backend()
    q, k, v, cu, bi, bo = _case(256, 4, 128, 0, device)
    k, v = k[:, :2].contiguous(), v[:, :2].contiguous()
    state, binding, _ = _binding(q, k, v, cu, bi, bo)
    expected = _oracle(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1),
                       _mask(256, bi, bo, device))
    torch.testing.assert_close(state.run(binding)[0], expected, rtol=0.02, atol=0.002)


@pytest.mark.parametrize("field,malformation", [
    (field, kind) for field in ("block_indices", "block_offsets")
    for kind in ("missing", "dtype", "device", "short", "strided", "rank")
])
def test_sparse_bind_rejects_invalid_csr_storage(field, malformation):
    device = _require_backend()
    state, _, args = _binding(*_case(256, 2, 128, None, device))
    value = args[field]
    args[field] = {
        "missing": lambda: None,
        "dtype": lambda: value.to(torch.int64),
        "device": lambda: value.cpu(),
        "short": lambda: value[:-1],
        "strided": lambda: value.repeat_interleave(2)[::2],
        "rank": lambda: value.reshape(1, -1),
    }[malformation]()
    with pytest.raises(ValueError, match=field):
        state.bind(**args)


def test_sparse_empty_storage():
    device = _require_backend()
    q, k, v, cu, _, bo = _case(256, 2, 128, None, device)
    bi = torch.empty(0, device=device, dtype=torch.int32)
    bo.zero_()
    state, binding, _ = _binding(q, k, v, cu, bi, bo)
    assert not torch.count_nonzero(state.run(binding)[0])


def test_sparse_tuning_preserves_csr_geometry():
    from b12x.attention import varlen
    from b12x.attention.varlen import VarlenAttentionConfig
    from b12x.attention.varlen._tuning import TUNING

    device = _require_backend()
    q, k, v, cu, bi, bo = _case(256, 2, 128, None, device)
    declaration = varlen.plan(
        q, k, v, cu, cu, max_seqlen_q=256, max_seqlen_k=256,
        block_sparse=True, num_q_tiles=bo.numel()-1, total_blocks_cap=bi.numel(),
    )
    candidates = [config for _, config in TUNING.eligible_plan(declaration.query, None).candidates]
    assert candidates == [VarlenAttentionConfig(tile_m=TILE_M, tile_n=BLOCK_K)]
    with pytest.raises(ValueError, match="CSR block geometry"):
        TUNING.validate_config(declaration.query, VarlenAttentionConfig(tile_m=64, tile_n=64), None)


@pytest.mark.parametrize("tiles,capacity", [(0, 1), (1, 1), (2, -1), (True, 1)])
def test_sparse_plan_rejects_invalid_capacities(tiles, capacity):
    from b12x.attention import varlen

    device = _require_backend()
    q, k, v, cu, _, _ = _case(256, 2, 128, None, device)
    with pytest.raises(ValueError, match="Sparse attention"):
        varlen.plan(q, k, v, cu, cu, max_seqlen_q=256, max_seqlen_k=256,
                    block_sparse=True, num_q_tiles=tiles, total_blocks_cap=capacity)


def test_sparse_graph_replay_reads_mutated_lists_without_allocating():
    from b12x.preparation._measurement import no_compilation

    device = _require_backend()
    q, k, v, cu, bi, bo = _case(256, 2, 128, 0, device)
    state, binding, _ = _binding(q, k, v, cu, bi, bo)
    output = state.run(binding)[0]
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with no_compilation(), torch.cuda.graph(graph):
        state.run(binding)
    for block in (1, 3, 0):
        bi.fill_(block)
        expected = _oracle(q, k, v, _mask(256, bi, bo, device))
        output.fill_(float("nan"))
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        with no_compilation():
            graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.002)
    bo.zero_()
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.count_nonzero(output)
    graph.reset()
