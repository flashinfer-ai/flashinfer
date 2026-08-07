"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math

import pytest
import torch

from flashinfer.cute_dsl.sparse.bsa_attn_blk64 import bsa_attn_blk64_fwd
from flashinfer.utils import is_sm100a_supported

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")),
    reason="the blk64 BSA kernel is SM100-only",
)

BLK = 64  # kSparseBlockSize == kRows
HEAD_DIM = 128  # the kernel is compiled for head_dim 128 only


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------
def _reference(q, k, v, block_index, block_nums, block_sizes, scale):
    """Dense fp32 attention restricted to the selected KV blocks.

    Mirrors what the kernel computes, padding included: K/V are zero-padded up
    to a whole number of 64-token blocks, and those padded positions stay
    *visible* (they score q @ 0 == 0) unless ``block_sizes`` declares the real
    length of the tail block.
    """
    batch, seq_q, heads, dim = q.shape
    seq_k = k.shape[1]
    n_blocks = (seq_k + BLK - 1) // BLK
    k_pad = q.new_zeros((batch, n_blocks * BLK, heads, dim))
    v_pad = q.new_zeros((batch, n_blocks * BLK, heads, dim))
    k_pad[:, :seq_k] = k
    v_pad[:, :seq_k] = v

    qf = q.float().permute(0, 2, 1, 3)  # (B, H, Sq, D)
    kf = k_pad.float().permute(0, 2, 1, 3)
    vf = v_pad.float().permute(0, 2, 1, 3)

    out = torch.zeros_like(qf)
    n_q_tiles = (seq_q + BLK - 1) // BLK
    for b in range(batch):
        for h in range(heads):
            for qt in range(n_q_tiles):
                lo, hi = qt * BLK, min((qt + 1) * BLK, seq_q)
                n_sel = (
                    int(block_nums[b, h, qt])
                    if block_nums is not None
                    else block_index.shape[3]
                )
                sel = [int(x) for x in block_index[b, h, qt, :n_sel].tolist()]
                sel = sorted(set(sel))  # phantom slots repeat the last real block
                if not sel:
                    continue
                cols = []
                for blk in sel:
                    n_valid = BLK if block_sizes is None else int(block_sizes[blk])
                    cols.extend(range(blk * BLK, blk * BLK + n_valid))
                cols = torch.tensor(cols, device=q.device, dtype=torch.long)
                scores = (qf[b, h, lo:hi] @ kf[b, h, cols].T) * scale
                out[b, h, lo:hi] = torch.softmax(scores, dim=-1) @ vf[b, h, cols]
    return out.permute(0, 2, 1, 3).to(q.dtype)  # back to (B, Sq, H, D)


def _make_selection(batch, heads, seq_q, seq_k, keep, device, variable, generator):
    """Build (block_index, block_nums, block_sparse_num) the way a router would."""
    n_q_tiles = (seq_q + BLK - 1) // BLK
    n_kv = (seq_k + BLK - 1) // BLK
    keep = min(keep, n_kv)
    idx = torch.empty((batch, heads, n_q_tiles, keep), dtype=torch.int32, device=device)
    for b in range(batch):
        for h in range(heads):
            for qt in range(n_q_tiles):
                perm = torch.randperm(n_kv, generator=generator, device=device)
                idx[b, h, qt] = perm[:keep].sort().values.to(torch.int32)
    if not variable:
        return idx, None, keep
    # Variable path: each row keeps a different number of blocks; the unused
    # slots repeat the last real block, exactly as the kernel documents.
    nums = torch.randint(
        1,
        keep + 1,
        (batch, heads, n_q_tiles),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    for b in range(batch):
        for h in range(heads):
            for qt in range(n_q_tiles):
                n = int(nums[b, h, qt])
                idx[b, h, qt, n:] = idx[b, h, qt, n - 1]
    return idx, nums, keep


def _qkv(batch, seq_q, seq_k, heads, device, layout, generator):
    """Identical values in three different memory layouts."""

    def rand(s):
        return torch.randn(
            (batch, s, heads, HEAD_DIM), device=device, generator=generator
        ).bfloat16()

    q, k, v = rand(seq_q), rand(seq_k), rand(seq_k)
    if layout == "contiguous":
        return q, k, v
    if layout == "packed":
        # Ulysses all-to-all shape: q/k/v are last-dim slices of one buffer, so
        # every one of them is strided (this is what MiniMax-H3 feeds in).
        assert seq_q == seq_k
        packed = torch.empty(
            (batch, seq_q, heads, 3 * HEAD_DIM), device=device, dtype=torch.bfloat16
        )
        packed[..., :HEAD_DIM] = q
        packed[..., HEAD_DIM : 2 * HEAD_DIM] = k
        packed[..., 2 * HEAD_DIM :] = v
        qs, ks, vs = packed.split(HEAD_DIM, dim=-1)
        assert not qs.is_contiguous()
        return qs, ks, vs
    if layout == "bhsd":
        # Head dim no longer innermost-major: transposed from BHSD.
        return tuple(
            t.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3) for t in (q, k, v)
        )
    raise AssertionError(layout)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "batch,seq_q,seq_k,heads",
    [
        (1, 256, 256, 4),  # everything block-aligned
        (1, 300, 300, 4),  # ragged Q tail and ragged K tail
        (1, 64, 64, 1),  # exactly one block, single head
        (1, 40, 40, 2),  # shorter than one block: whole_blocks == 0
        (2, 192, 192, 3),  # batch > 1: narrow() leaves a non-natural batch stride
        (3, 200, 328, 2),  # batch > 1 and seq_q != seq_k, both ragged
        (1, 128, 576, 4),  # cross-attention shape, many more keys than queries
        (2, 577, 129, 2),  # ragged on both sides, K tail of 1 token
    ],
)
@pytest.mark.parametrize("variable", [False, True])
def test_matches_reference(batch, seq_q, seq_k, heads, variable):
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(1234)
    q, k, v = _qkv(batch, seq_q, seq_k, heads, device, "contiguous", gen)
    idx, nums, n_sel = _make_selection(
        batch, heads, seq_q, seq_k, 3, device, variable, gen
    )
    n_kv = (seq_k + BLK - 1) // BLK
    sizes = torch.full((n_kv,), BLK, dtype=torch.int32, device=device)
    sizes[-1] = seq_k - (n_kv - 1) * BLK
    scale = 1.0 / math.sqrt(HEAD_DIM)

    out, _ = bsa_attn_blk64_fwd(
        q,
        k,
        v,
        idx,
        n_sel,
        block_sizes=sizes,
        q2k_block_nums=nums,
        softmax_scale=scale,
    )
    ref = _reference(q, k, v, idx, nums, sizes, scale)

    assert out.shape == (batch, seq_q, heads, HEAD_DIM)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("layout", ["packed", "bhsd"])
@pytest.mark.parametrize("batch,seq,heads", [(1, 320, 4), (2, 300, 2)])
def test_strided_inputs_match_contiguous(layout, batch, seq, heads):
    """The launch template consumes strided BSHD directly; that must not change
    a single bit of the result versus the same values laid out contiguously."""
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(7)
    q, k, v = _qkv(batch, seq, seq, heads, device, "contiguous", gen)
    gen.manual_seed(7)
    qs, ks, vs = _qkv(batch, seq, seq, heads, device, layout, gen)
    assert torch.equal(q, qs) and torch.equal(k, ks) and torch.equal(v, vs)

    idx, nums, n_sel = _make_selection(batch, heads, seq, seq, 3, device, True, gen)
    n_kv = (seq + BLK - 1) // BLK
    sizes = torch.full((n_kv,), BLK, dtype=torch.int32, device=device)
    sizes[-1] = seq - (n_kv - 1) * BLK

    ref, _ = bsa_attn_blk64_fwd(
        q, k, v, idx, n_sel, block_sizes=sizes, q2k_block_nums=nums
    )
    got, _ = bsa_attn_blk64_fwd(
        qs, ks, vs, idx, n_sel, block_sizes=sizes, q2k_block_nums=nums
    )
    assert torch.equal(ref, got)


@pytest.mark.parametrize("seq", [256, 300])
def test_mqa_single_kv_head(seq):
    """A single KV head is shared by every query head."""
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(5)
    batch, heads = 1, 4
    q = torch.randn(
        (batch, seq, heads, HEAD_DIM), device=device, generator=gen
    ).bfloat16()
    k1 = torch.randn((batch, seq, 1, HEAD_DIM), device=device, generator=gen).bfloat16()
    v1 = torch.randn((batch, seq, 1, HEAD_DIM), device=device, generator=gen).bfloat16()

    idx, nums, n_sel = _make_selection(batch, heads, seq, seq, 3, device, True, gen)
    n_kv = (seq + BLK - 1) // BLK
    sizes = torch.full((n_kv,), BLK, dtype=torch.int32, device=device)
    sizes[-1] = seq - (n_kv - 1) * BLK

    got, _ = bsa_attn_blk64_fwd(
        q, k1, v1, idx, n_sel, block_sizes=sizes, q2k_block_nums=nums
    )
    # Same thing spelled out: the KV head replicated to every query head.
    ref, _ = bsa_attn_blk64_fwd(
        q,
        k1.expand(-1, -1, heads, -1).contiguous(),
        v1.expand(-1, -1, heads, -1).contiguous(),
        idx,
        n_sel,
        block_sizes=sizes,
        q2k_block_nums=nums,
    )
    assert torch.equal(got, ref)


@pytest.mark.parametrize("seq", [512, 384, 320, 256])
def test_full_budget_matches_dense(seq):
    """Selecting every KV block must reproduce dense attention.

    ``seq`` is varied so that the KV block count is both a multiple of the
    kernel's internal 8-block alignment (512 -> 8) and not (384 -> 6, 320 -> 5,
    256 -> 4); the phantom slots the kernel pads with must not contribute.
    """
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(11)
    batch, heads = 1, 4
    q, k, v = _qkv(batch, seq, seq, heads, device, "contiguous", gen)
    n_kv = seq // BLK
    idx = (
        torch.arange(n_kv, dtype=torch.int32, device=device)
        .view(1, 1, 1, n_kv)
        .expand(batch, heads, seq // BLK, n_kv)
        .contiguous()
    )
    out, _ = bsa_attn_blk64_fwd(q, k, v, idx, n_kv)
    dense = torch.nn.functional.scaled_dot_product_attention(
        *(t.permute(0, 2, 1, 3).float() for t in (q, k, v))
    ).permute(0, 2, 1, 3)
    torch.testing.assert_close(out.float(), dense, rtol=2e-2, atol=2e-2)


def test_block_sizes_masks_the_ragged_tail():
    """With a truthful block_sizes the zero-padded tail keys must not leak in."""
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(3)
    batch, seq, heads = 1, 130, 2  # tail block holds 2 real tokens of 64
    q, k, v = _qkv(batch, seq, seq, heads, device, "contiguous", gen)
    n_kv = (seq + BLK - 1) // BLK
    idx = (
        torch.arange(n_kv, dtype=torch.int32, device=device)
        .view(1, 1, 1, n_kv)
        .expand(batch, heads, (seq + BLK - 1) // BLK, n_kv)
        .contiguous()
    )
    sizes = torch.tensor([BLK, BLK, seq - 2 * BLK], dtype=torch.int32, device=device)
    out, _ = bsa_attn_blk64_fwd(q, k, v, idx, n_kv, block_sizes=sizes)
    dense = torch.nn.functional.scaled_dot_product_attention(
        *(t.permute(0, 2, 1, 3).float() for t in (q, k, v))
    ).permute(0, 2, 1, 3)
    torch.testing.assert_close(out.float(), dense, rtol=2e-2, atol=2e-2)
