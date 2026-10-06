# SPDX-License-Identifier: Apache-2.0
"""Per-segment block lists: one CSR per packed segment, not one for the whole batch.

The block-list walk indexes ``mBlockOffsets[m_block]`` with a segment-local ``m_block``,
so by default every segment reads the same rows and a per-head selection is not
expressible. With ``per_segment_tiles=True`` the CSR is indexed by the global q tile of
the segment, ``offset_q // tile_m + m_block``, which requires tile-aligned segments
(every ``cu_seqlens_q`` value a multiple of ``tile_m``) -- what a head-packed layout
produces natively.

Each case is small enough to check against an exact per-head reference.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from ..conftest import require_b12x as require_sm120

TILE_M = 64
BLOCK_K = 64
HEAD_DIM = 128


def _require_backend():
    require_sm120()
    pytest.importorskip("cutlass")
    pytest.importorskip("cuda.bindings.driver")
    return torch.device("cuda", torch.cuda.current_device())


def _csr(rows, device):
    """rows: one block-id list per q tile, laid out segment-major."""
    flat, off = [], [0]
    for r in rows:
        flat.extend(r)
        off.append(len(flat))
    return (torch.tensor(flat, dtype=torch.int32, device=device),
            torch.tensor(off, dtype=torch.int32, device=device))


def _reference(q, k, v, head, blocks, seq_len, scale):
    rows = (torch.tensor(blocks, device=q.device)[:, None] * BLOCK_K
            + torch.arange(BLOCK_K, device=q.device)).reshape(-1)
    scores = (q[head].float() @ k[head][rows].float().T) * scale
    out = (scores.softmax(-1) @ v[head][rows].float()).to(q.dtype)
    return out.view(seq_len, HEAD_DIM)


def _run(list_a, list_b, device, per_segment_tiles=True):
    from b12x.attention import varlen
    from b12x.attention.varlen import VarlenAttentionConfig
    from b12x.preparation import require_prepared

    heads, tiles, seq_len = 2, 8, 8 * BLOCK_K
    torch.manual_seed(0)
    q = torch.randn(heads, seq_len, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = HEAD_DIM ** -0.5

    rows = [list_a] * tiles + [list_b] * tiles
    bi, bo = _csr(rows, device)
    qt = q.reshape(heads * seq_len, 1, HEAD_DIM).contiguous()
    kt = k.reshape(heads * seq_len, 1, HEAD_DIM).contiguous()
    vt = v.reshape(heads * seq_len, 1, HEAD_DIM).contiguous()
    cu = torch.arange(0, (heads + 1) * seq_len, seq_len, dtype=torch.int32, device=device)

    plan = varlen.plan(qt, kt, vt, cu, cu, max_seqlen_q=seq_len, max_seqlen_k=seq_len,
                       causal=False, block_sparse=True, num_q_tiles=heads * tiles,
                       total_blocks_cap=max(1, int(bo[-1].item())),
                       per_segment_tiles=per_segment_tiles,
                       override=VarlenAttentionConfig(tile_m=TILE_M, tile_n=BLOCK_K))
    state = require_prepared(plan, "attention.varlen", device)
    spec, = state.scratch_plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
    binding = varlen.bind(plan, scratch=scratch, q=qt, k=kt, v=vt, cu_seqlens_q=cu,
                          cu_seqlens_k=cu, max_seqlen_q=seq_len, max_seqlen_k=seq_len,
                          block_indices=bi, block_offsets=bo, softmax_scale=scale)
    out = state.run(binding)[0].reshape(heads, seq_len, HEAD_DIM)
    return out, (q, k, v, scale)


@pytest.mark.parametrize("list_a,list_b", [([0], [1]), ([0], [0, 1]), ([0, 1], [5, 6]),
                                           ([0, 1, 2], [5, 6, 7]), ([3], [3])])
def test_per_segment_lists_are_exact(list_a, list_b):
    """Each segment must attend exactly its own list, all lists within one call."""
    device = _require_backend()
    out, (q, k, v, scale) = _run(list_a, list_b, device)
    seq_len = 8 * BLOCK_K
    for head, blocks in ((0, list_a), (1, list_b)):
        ref = _reference(q, k, v, head, blocks, seq_len, scale)
        cos = F.cosine_similarity(out[head].flatten().float(), ref.flatten().float(), dim=0)
        assert cos.item() > 0.99999, (
            f"head {head}: list {blocks} cosine {cos.item():.8f} -- a segment did not read "
            "its own list")


def test_segment_zero_is_unaffected_by_segment_one():
    """Changing segment 1's list must not move segment 0's output."""
    device = _require_backend()
    first, _ = _run([0], [1], device)
    second, _ = _run([0], [5, 6], device)
    assert torch.equal(first[0], second[0]), (
        "segment 0 changed when only segment 1's list changed")


def test_shared_list_path_unchanged():
    """per_segment_tiles=False keeps the original contract: every segment reads the
    first tiles' rows, so a uniform list still reproduces each segment's own blocks."""
    device = _require_backend()
    out, (q, k, v, scale) = _run([0], [0], device, per_segment_tiles=False)
    for head in (0, 1):
        ref = _reference(q, k, v, head, [0], 8 * BLOCK_K, scale)
        cos = F.cosine_similarity(out[head].flatten().float(), ref.flatten().float(), dim=0)
        assert cos.item() > 0.99999
