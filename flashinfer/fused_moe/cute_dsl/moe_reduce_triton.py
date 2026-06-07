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
