# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Triton fused MoE finalize for the W4A8 top_k>=2 path: gather each token's top_k
expert outputs and weighted-sum them back to a per-token output.

For top_k>=2 the fused token-scatter epilogue must accumulate (multiple experts write
the same token), which forces a per-element FP32 atomicAdd. ncu shows that atomic is
latency-bound (the persistent kernel's low occupancy can't hide the RMW round-trip), so
even a vectorized v2.f32 atomic only saves ~8%. The faster path is to *un-fuse* it (the
same lesson as un-fusing the cp.async gather): GEMM2 writes per-expert outputs into one
contiguous buffer (routed order, plain store) and this kernel reduces them in ONE
non-atomic, memory-bound pass -- gather the token's top_k permuted rows, weighted-sum,
write once.

Measured (top_k=2, e=8, m=8192, H=4096, I=1024, H100): the finalize drops from a ~740us
scalar-atomicAdd scatter to ~78us (the BW floor), making the full GEMM2 ~2.6x faster.
"""

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _moe_reduce_kernel(
        c_ptr,  # [num_permuted, H] per-expert outputs in routed order
        out_ptr,  # [num_tokens, H] per-token weighted sum
        idx_ptr,  # [num_tokens, top_k] int32: permuted-row index (-1 = masked)
        w_ptr,  # [num_tokens, top_k] f32: routing weight
        H,
        TOPK: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        # One program = one (token, H-block): gather the token's TOPK permuted rows,
        # weighted-sum, write once. Non-atomic, coalesced along H, memory-bound.
        t = tl.program_id(0)
        h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = h < H
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for s in range(TOPK):
            row = tl.load(idx_ptr + t * TOPK + s).to(tl.int64)
            valid = row >= 0
            # Sanitize the index before the address math: the masked load already skips
            # invalid lanes, but clamping row to 0 keeps the computed address in-bounds
            # (defensive, no functional change -- the lane is still masked out).
            safe_row = tl.where(valid, row, 0)
            w = tl.load(w_ptr + t * TOPK + s)
            v = tl.load(c_ptr + safe_row * H + h, mask=mask & valid, other=0.0).to(
                tl.float32
            )
            acc += tl.where(valid, w, 0.0) * v
        tl.store(out_ptr + t * H + h, acc.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _w4a8_fill_meta_kernel(
        counts_ptr,  # [E] int32: tokens routed to each expert
        sizes1_ptr,  # [E, 4] int32 GEMM1 (M,N,K,L); cols 1..3 prefilled by the host
        ptrs1_ptr,  # [E, 4] int64 GEMM1 (A, B, C, scale) base addresses
        sizes2_ptr,  # [E, 4] int32 GEMM2 sizes; cols 1..3 prefilled
        ptrs2_ptr,  # [E, N_OPS2] int64 GEMM2 addresses (+ route/weight if scatter)
        clusters1_ptr,  # [1] int32: GEMM1 total cluster tiles
        clusters2_ptr,  # [1] int32: GEMM2 total cluster tiles
        a1_base,
        b1_base,
        c1_base,
        s1_base,  # GEMM1 operand base addresses (bytes)
        sr1_base,  # GEMM1 per-row A-scale base (f32 [total routed rows]; 0 if unused)
        a2_base,
        b2_base,
        c2_base,
        s2_base,  # GEMM2 operand base addresses
        r2_base,
        w2_base,  # GEMM2 scatter route-map / routing-weight bases (0 if unused)
        a1_row,
        c1_row,  # GEMM1 per-routed-row byte strides of A / C
        b1_exp,
        s1_exp,  # GEMM1 per-EXPERT byte strides of B / scale
        a2_row,
        c2_row,  # GEMM2 row strides (c2_row == 0 in scatter mode: shared C base)
        b2_exp,
        s2_exp,
        tile_m1,
        nn1,  # GEMM1 cluster-tile M extent / N tile count
        tile_m2,
        nn2,
        E,
        N_OPS1: tl.constexpr,
        N_OPS2: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # ONE program builds, from the per-expert token counts, everything the two
        # grouped GEMM launches need: per-group M sizes, per-group operand pointers
        # (base + exclusive-cumsum(counts) * row stride for the routed
        # activation/output operands; base + expert index * expert stride for the
        # weight/scale operands), and the two cluster-tile totals the persistent
        # scheduler reads from device memory. Replaces the per-expert Python loops
        # (~6 x E tensor slices + 2 x E small casts) and the host->device metadata
        # copies of the list-based path with a single launch, and keeps the host
        # entirely blind to the routing -- no device->host sync anywhere.
        e = tl.arange(0, BLOCK)
        mask = e < E
        cnt = tl.load(counts_ptr + e, mask=mask, other=0)
        offs = (tl.cumsum(cnt, 0) - cnt).to(tl.int64)  # exclusive prefix sum (rows)
        e64 = e.to(tl.int64)
        tl.store(sizes1_ptr + e * 4, cnt, mask=mask)
        tl.store(sizes2_ptr + e * 4, cnt, mask=mask)
        tl.store(ptrs1_ptr + e * N_OPS1 + 0, a1_base + offs * a1_row, mask=mask)
        tl.store(ptrs1_ptr + e * N_OPS1 + 1, b1_base + e64 * b1_exp, mask=mask)
        tl.store(ptrs1_ptr + e * N_OPS1 + 2, c1_base + offs * c1_row, mask=mask)
        tl.store(ptrs1_ptr + e * N_OPS1 + 3, s1_base + e64 * s1_exp, mask=mask)
        if N_OPS1 == 5:  # per-row A scale (f32 [M] per group, swiglu epilogue)
            tl.store(ptrs1_ptr + e * N_OPS1 + 4, sr1_base + offs * 4, mask=mask)
        tl.store(ptrs2_ptr + e * N_OPS2 + 0, a2_base + offs * a2_row, mask=mask)
        tl.store(ptrs2_ptr + e * N_OPS2 + 1, b2_base + e64 * b2_exp, mask=mask)
        tl.store(ptrs2_ptr + e * N_OPS2 + 2, c2_base + offs * c2_row, mask=mask)
        tl.store(ptrs2_ptr + e * N_OPS2 + 3, s2_base + e64 * s2_exp, mask=mask)
        if N_OPS2 == 6:  # token-scatter mode: route map / weights are [M] i32/f32
            tl.store(ptrs2_ptr + e * N_OPS2 + 4, r2_base + offs * 4, mask=mask)
            tl.store(ptrs2_ptr + e * N_OPS2 + 5, w2_base + offs * 4, mask=mask)
        nm1 = (cnt + tile_m1 - 1) // tile_m1
        nm2 = (cnt + tile_m2 - 1) // tile_m2
        tl.store(clusters1_ptr, tl.sum(nm1 * nn1).to(tl.int32))
        tl.store(clusters2_ptr, tl.sum(nm2 * nn2).to(tl.int32))

    @triton.jit
    def _per_token_quant_fp8_kernel(
        x_ptr,  # [*, N] input (bf16/fp16/fp32)
        out_ptr,  # [*, N] FP8 e4m3 output
        scale_ptr,  # [*] f32 per-row scale (= row amax / FP8_MAX)
        N,
        GROUP_ROWS,  # logical rows per group (= physical when no grouping)
        GROUP_STRIDE,  # physical row stride between groups
        FP8_MAX: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # One program = one LOGICAL row: per-token dynamic FP8 quant. Pass 1 finds the
        # row amax, pass 2 stores x/scale as FP8 (scale = amax / FP8_MAX, so values land
        # in [-FP8_MAX, FP8_MAX]). Downstream dequant is value * scale.
        # Logical row m maps to physical row (m // GROUP_ROWS) * GROUP_STRIDE +
        # (m % GROUP_ROWS): callers whose per-group rows sit at a fixed physical
        # stride (e.g. capacity-padded buffers) quantize only the first GROUP_ROWS
        # of each group. Identity when GROUP_ROWS == GROUP_STRIDE == total rows.
        m = tl.program_id(0)
        pr = (m // GROUP_ROWS) * GROUP_STRIDE + (m % GROUP_ROWS)
        row_x = x_ptr + pr * N
        amax = 0.0
        for off in range(0, N, BLOCK_N):
            n = off + tl.arange(0, BLOCK_N)
            v = tl.load(row_x + n, mask=n < N, other=0.0).to(tl.float32)
            amax = tl.maximum(amax, tl.max(tl.abs(v)))
        scale = amax / FP8_MAX
        scale = tl.where(scale > 0.0, scale, 1.0)  # all-zero row -> scale 1 (output 0)
        tl.store(scale_ptr + pr, scale)
        inv = 1.0 / scale
        row_o = out_ptr + pr * N
        for off in range(0, N, BLOCK_N):
            n = off + tl.arange(0, BLOCK_N)
            v = tl.load(row_x + n, mask=n < N, other=0.0).to(tl.float32) * inv
            v = tl.minimum(tl.maximum(v, -FP8_MAX), FP8_MAX)
            tl.store(row_o + n, v.to(out_ptr.dtype.element_ty), mask=n < N)


def moe_reduce(
    permuted_out: torch.Tensor,
    output: torch.Tensor,
    permuted_idx: torch.Tensor,
    topk_scales: torch.Tensor,
    top_k: int,
    block_h: int = 512,
) -> None:
    """Reduce per-expert GEMM2 outputs to per-token: ``output[t] = sum_k scales[t,k] *
    permuted_out[permuted_idx[t,k]]`` (``permuted_idx[t,k] == -1`` skips a masked expert).

    :param permuted_out: ``[num_permuted, H]`` per-expert outputs (routed order).
    :param output: ``[num_tokens, H]`` destination (written, not accumulated).
    :param permuted_idx: ``[num_tokens, top_k]`` int32 permuted-row index per token.
    :param topk_scales: ``[num_tokens, top_k]`` f32 routing weight per token.
    """
    if not _HAS_TRITON:
        raise RuntimeError("moe_reduce requires triton")
    num_tokens, H = output.shape
    assert permuted_idx.shape == (num_tokens, top_k)
    grid = (num_tokens, triton.cdiv(H, block_h))
    _moe_reduce_kernel[grid](
        permuted_out,
        output,
        permuted_idx,
        topk_scales,
        H,
        TOPK=top_k,
        BLOCK_H=block_h,
    )


def build_reduce_index(
    route_maps: "list[torch.Tensor]",
    weights: "list[torch.Tensor]",
    num_tokens: int,
    top_k: int,
    device: Optional[torch.device] = None,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """From per-group route maps (local row -> token) + weights, build the
    ``(permuted_idx, topk_scales)`` of shape ``[num_tokens, top_k]`` that ``moe_reduce``
    consumes. Assumes the per-group GEMM2 outputs are concatenated in group order (group
    g's rows occupy ``[offset_g, offset_g + len(route_maps[g]))`` of ``permuted_out``).
    Every token must appear exactly ``top_k`` times across the route maps.
    """
    if not route_maps:
        # Empty batch (no active experts): num_tokens is 0 here, so return empty
        # [0, top_k] tensors rather than indexing route_maps[0] / torch.cat([]).
        dev = device or torch.device("cuda")
        return (
            torch.zeros(num_tokens, top_k, dtype=torch.int32, device=dev),
            torch.zeros(num_tokens, top_k, dtype=torch.float32, device=dev),
        )
    device = device or route_maps[0].device
    all_tok = torch.cat([r.long() for r in route_maps])  # [num_permuted]
    all_pos = torch.arange(all_tok.numel(), device=device)  # permuted row index
    all_w = torch.cat([w.float() for w in weights])
    order = torch.argsort(all_tok, stable=True)  # group rows by token
    permuted_idx = (
        all_pos[order].reshape(num_tokens, top_k).to(torch.int32).contiguous()
    )
    topk_scales = all_w[order].reshape(num_tokens, top_k).contiguous()
    return permuted_idx, topk_scales


def per_token_quant_fp8(
    x: torch.Tensor,
    block_n: int = 1024,
    out: Optional[torch.Tensor] = None,
    scale_out: Optional[torch.Tensor] = None,
    group_rows: Optional[int] = None,
    group_stride: Optional[int] = None,
    num_groups: Optional[int] = None,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Per-token (per-row) dynamic FP8 e4m3 quantization. Returns ``(x_fp8 [M, N],
    scale [M] f32)`` with ``x ≈ x_fp8.float() * scale[:, None]`` (``scale = row amax /
    448``). Used to requantize the W4A8 MoE BF16 SwiGLU intermediate before GEMM2 -- the
    per-row scale is folded into the routing weight, so GEMM2 needs no change (the scale
    is constant along the GEMM2 contraction, so it pulls out to a per-output-row factor).

    ``out`` / ``scale_out`` (preallocated ``[M, N]`` FP8 / ``[M]`` f32) let the caller
    quantize into reused buffers instead of allocating fresh outputs each call (e.g. to
    keep pointers stable / avoid per-call allocation in a steady-state serving loop).

    Grouped mode (all three of ``group_rows`` / ``group_stride`` / ``num_groups`` set):
    ``x`` is a ``[num_groups * group_stride, N]`` backing buffer in which each group
    occupies ``group_stride`` physical rows but only the first ``group_rows`` are live;
    exactly the live rows are quantized (in place into the same-layout ``out`` /
    ``scale_out``). Used by the capacity-bucketed W4A8 serving runner, whose per-expert
    buffers are laid out at a fixed capacity-cap stride."""
    if not _HAS_TRITON:
        raise RuntimeError("per_token_quant_fp8 requires triton")
    assert x.dim() == 2, "per_token_quant_fp8 expects a 2D [M, N] tensor"
    M, N = x.shape
    grouped = group_rows is not None
    if grouped:
        assert group_stride is not None and num_groups is not None
        assert group_rows <= group_stride and num_groups * group_stride <= M
        total = num_groups * group_rows  # logical rows to quantize
        g_rows, g_stride = group_rows, group_stride
    else:
        total, g_rows, g_stride = M, M, M  # identity mapping
    x = x.contiguous()
    x_fp8 = (
        out
        if out is not None
        else torch.empty(M, N, device=x.device, dtype=torch.float8_e4m3fn)
    )
    scale = (
        scale_out
        if scale_out is not None
        else torch.empty(M, device=x.device, dtype=torch.float32)
    )
    if total > 0:
        _per_token_quant_fp8_kernel[(total,)](
            x, x_fp8, scale, N, g_rows, g_stride, FP8_MAX=448.0, BLOCK_N=block_n
        )
    return x_fp8, scale


def fill_w4a8_moe_meta(
    counts: torch.Tensor,
    sizes1: torch.Tensor,
    ptrs1: torch.Tensor,
    clusters1: torch.Tensor,
    sizes2: torch.Tensor,
    ptrs2: torch.Tensor,
    clusters2: torch.Tensor,
    g1_bases: "tuple[int, int, int, int, int]",
    g1_rows: "tuple[int, int]",
    g1_experts: "tuple[int, int]",
    g2_bases: "tuple[int, int, int, int, int, int]",
    g2_rows: "tuple[int, int]",
    g2_experts: "tuple[int, int]",
    tile_m1: int,
    nn1: int,
    tile_m2: int,
    nn2: int,
) -> None:
    """Fill BOTH grouped-GEMM metadata sets of one W4A8 MoE call from the per-expert
    token counts, in a single kernel launch (see ``_w4a8_fill_meta_kernel``).

    :param counts: ``[E]`` int32 tokens per expert (device).
    :param sizes1/ptrs1/clusters1: GEMM1 metadata device tensors -- ``[E, 4]`` int32
        (cols 1..3 prefilled with N/K/L), ``[E, 4|5]`` int64 (5 with the per-row
        A-scale column), ``[1]`` int32.
    :param sizes2/ptrs2/clusters2: GEMM2 metadata; ``ptrs2`` is ``[E, 4]`` or
        ``[E, 6]`` (token-scatter mode appends route-map/weight pointers).
    :param g*_bases: operand base addresses in bytes -- GEMM1 ``(A, B, C, scale,
        a_row_scale)`` (last 0 when unused), GEMM2 ``(A, B, C, scale, route_map,
        weights)`` (last two 0 when unused).
    :param g*_rows: per-routed-row byte strides ``(A_row, C_row)``; GEMM2's C_row
        is 0 in scatter mode (every group scatters into the shared output base).
    :param g*_experts: per-expert byte strides ``(B_expert, scale_expert)``.
    :param tile_m*/nn*: cluster-tile M extent and N-direction tile count per GEMM,
        for the device-side cluster totals the persistent scheduler consumes.
    """
    if not _HAS_TRITON:
        raise RuntimeError("fill_w4a8_moe_meta requires triton")
    E = counts.numel()
    n_ops1 = ptrs1.shape[1]
    n_ops2 = ptrs2.shape[1]
    _w4a8_fill_meta_kernel[(1,)](
        counts,
        sizes1,
        ptrs1,
        sizes2,
        ptrs2,
        clusters1,
        clusters2,
        g1_bases[0],
        g1_bases[1],
        g1_bases[2],
        g1_bases[3],
        g1_bases[4],
        g2_bases[0],
        g2_bases[1],
        g2_bases[2],
        g2_bases[3],
        g2_bases[4],
        g2_bases[5],
        g1_rows[0],
        g1_rows[1],
        g1_experts[0],
        g1_experts[1],
        g2_rows[0],
        g2_rows[1],
        g2_experts[0],
        g2_experts[1],
        tile_m1,
        nn1,
        tile_m2,
        nn2,
        E,
        N_OPS1=n_ops1,
        N_OPS2=n_ops2,
        BLOCK=triton.next_power_of_2(E),
    )
