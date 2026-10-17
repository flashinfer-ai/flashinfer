#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
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
"""Video Sparse Attention (VSA) for the FlashInfer WAN example.

Implements the two-stage VSA attention of `Faster Video Diffusion with
Trainable Sparse Attention <https://arxiv.org/abs/2505.13389>`_ on top of
FlashInfer's ``bsa_attn_blk64_fwd`` block-sparse kernel (SM100, 64-token
blocks, bf16, head_dim=128).

The semantics deliberately mirror FastVideo's ``video_sparse_attn`` so a
VSA-finetuned checkpoint (e.g. ``FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers``)
runs unmodified:

1. **Tile.** The ``(T, H, W)`` post-patch token grid is permuted so that every
   ``(4, 4, 4)`` spatio-temporal cube becomes 64 contiguous tokens, then padded
   up to a whole number of 64-token blocks. Edge cubes are partial; the real
   token count per block is carried in ``variable_block_sizes``. Padding slots
   read token 0 rather than zeros — see :func:`_tile` — and are masked
   wherever they would be observed.
2. **Coarse stage.** Q/K/V are mean-pooled inside each cube (masked mean, so
   padding does not bias the average), dense attention runs over the ~1.4K
   pooled tokens, and the result is broadcast back over each cube.
3. **Selection.** The same pooled scores drive a per-(head, q-block) top-k over
   KV blocks, which becomes the block-sparse index tensor.
4. **Fine stage.** ``bsa_attn_blk64_fwd`` runs token-level attention restricted
   to the selected blocks, with ``block_sizes`` masking the padded tail of each
   partial block.
5. **Combine.** ``out = out_coarse * gate_compress + out_fine``, where
   ``gate_compress`` is the extra per-token ``to_gate_compress`` projection that
   VSA training adds to the model.

Only the ``block_sizes``-aware ``bsa_attn_blk64_fwd`` entry point is used (not
``BlockSparseAttentionWrapper``): the wrapper converts the selection on the host
via ``.cpu()`` every ``plan()``, which would add a device sync per layer per
denoising step, and it cannot forward per-block valid-token counts.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

# VSA tile shape over the post-patch (T, H, W) token grid. 4*4*4 = 64 matches
# the kernel's block granularity, which is the point: one cube == one block.
VSA_TILE_SIZE: Tuple[int, int, int] = (4, 4, 4)
VSA_BLOCK_SIZE: int = VSA_TILE_SIZE[0] * VSA_TILE_SIZE[1] * VSA_TILE_SIZE[2]


def vsa_supported(head_dim: int, device: Optional[torch.device] = None) -> bool:
    """Can ``bsa_attn_blk64_fwd`` run here? SM100 and head_dim 128 only.

    (The kernel is also bf16-only, which the WAN example always satisfies.)
    Mirrors the capability gates the GEMM and attention backends already apply,
    so an unsupported GPU gets a warning and dense attention rather than a
    kernel error deep in the forward pass. Defaults to the current CUDA device,
    since models are typically constructed before being moved onto one.
    """
    if not torch.cuda.is_available() or head_dim != 128:
        return False
    return torch.cuda.get_device_capability(device)[0] == 10


@dataclass(frozen=True)
class VSAMetadata:
    """Layout tables for one ``(T, H, W)`` token grid. Built once per shape."""

    dit_seq_shape: Tuple[int, int, int]
    num_tiles: Tuple[int, int, int]
    tile_partition_indices: torch.Tensor  # [seq_len] int64, original -> tile order
    non_pad_index: torch.Tensor  # [seq_len] int64, tile order -> padded buffer
    untile_combined_index: torch.Tensor  # [seq_len] int64, padded buffer -> original
    variable_block_sizes: torch.Tensor  # [num_blocks] int32, valid tokens per block
    # Padded-buffer slot -> source token, so tiling is a single gather instead of
    # zero-fill + gather + scatter. Padding slots read token 0; that value is
    # never observed because block_valid_mask masks it in the pooled mean and
    # variable_block_sizes masks it inside the sparse kernel.
    padded_src_index: torch.Tensor  # [padded_seq_len] int64
    block_valid_mask: torch.Tensor  # [1, num_blocks, 64, 1, 1] bf16, 1 = real

    @property
    def seq_len(self) -> int:
        """Unpadded token count."""
        return math.prod(self.dit_seq_shape)

    @property
    def num_blocks(self) -> int:
        return self.variable_block_sizes.numel()

    @property
    def padded_seq_len(self) -> int:
        return self.num_blocks * VSA_BLOCK_SIZE

    def topk_for_sparsity(self, sparsity: float) -> int:
        """KV blocks kept per query block, matching FastVideo's convention."""
        topk = math.ceil((1.0 - sparsity) * self.num_blocks)
        return max(1, min(topk, self.num_blocks))


@functools.lru_cache(maxsize=8)
def build_vsa_metadata(
    dit_seq_shape: Tuple[int, int, int],
    device: torch.device,
) -> VSAMetadata:
    """Build the tile/untile permutations and per-block valid-token counts."""
    t, h, w = dit_seq_shape
    ts, hs, ws = VSA_TILE_SIZE
    n_t, n_h, n_w = (
        math.ceil(t / ts),
        math.ceil(h / hs),
        math.ceil(w / ws),
    )

    # Original -> tile order: walk cubes in (t, h, w) order and emit the
    # flattened token ids inside each cube. Built on CPU (once per shape) to
    # keep the Python loop off the device timeline, then moved over.
    ids = torch.arange(t * h * w, dtype=torch.int64).reshape(t, h, w)
    chunks = [
        ids[
            ti * ts : ti * ts + ts, hi * hs : hi * hs + hs, wi * ws : wi * ws + ws
        ].flatten()
        for ti in range(n_t)
        for hi in range(n_h)
        for wi in range(n_w)
    ]
    tile_partition_indices = torch.cat(chunks).to(device)

    # Valid token count per cube = product of the (possibly clipped) extents.
    def _extents(dim_len: int, tile: int, n_tiles: int) -> torch.Tensor:
        sizes = torch.full((n_tiles,), tile, dtype=torch.int64)
        # n_tiles = ceil(dim_len / tile), so the last tile holds 1..tile tokens.
        sizes[-1] = dim_len - (n_tiles - 1) * tile
        return sizes

    variable_block_sizes = (
        _extents(t, ts, n_t)[:, None, None]
        * _extents(h, hs, n_h)[None, :, None]
        * _extents(w, ws, n_w)[None, None, :]
    ).reshape(-1)

    # Tile order -> slot in the padded [num_blocks * 64] buffer: each cube's
    # tokens go to the front of its block, the tail is padding.
    num_blocks = variable_block_sizes.numel()
    offsets = torch.arange(num_blocks, dtype=torch.int64) * VSA_BLOCK_SIZE
    within = torch.arange(VSA_BLOCK_SIZE, dtype=torch.int64)
    slots = offsets[:, None] + within[None, :]
    valid = within[None, :] < variable_block_sizes[:, None]
    non_pad_index = slots[valid].to(device)

    # Padded buffer -> original order, fusing the two hops so untiling is a
    # single gather instead of materializing the intermediate.
    reverse = torch.argsort(tile_partition_indices)
    untile_combined_index = non_pad_index[reverse]

    padded_src_index = torch.zeros(
        num_blocks * VSA_BLOCK_SIZE, dtype=torch.int64, device=device
    )
    padded_src_index[non_pad_index] = tile_partition_indices
    # Stored in the kernel's compute dtype so the multiply below never casts.
    block_valid_mask = valid.to(device=device, dtype=torch.bfloat16).view(
        1, num_blocks, VSA_BLOCK_SIZE, 1, 1
    )

    return VSAMetadata(
        dit_seq_shape=(t, h, w),
        num_tiles=(n_t, n_h, n_w),
        tile_partition_indices=tile_partition_indices,
        non_pad_index=non_pad_index,
        untile_combined_index=untile_combined_index,
        variable_block_sizes=variable_block_sizes.to(device=device, dtype=torch.int32),
        padded_src_index=padded_src_index,
        block_valid_mask=block_valid_mask,
    )


def _tile(x: torch.Tensor, meta: VSAMetadata) -> torch.Tensor:
    """``[B, S, H, D] -> [B, S_padded, H, D]`` in cube order.

    A single gather per tensor. The obvious formulation — allocate zeros, gather
    into tile order, scatter into the non-padding slots — moves roughly 2.5x the
    bytes for the same result. Here every padded slot instead reads token 0;
    padding is masked downstream (``block_valid_mask`` for the pooled mean,
    ``variable_block_sizes`` for the sparse kernel) so it never reaches the
    output. See wan/BENCHMARK.md for the measured difference.

    Tiling q/k/v/gate as one stacked tensor would need a full copy of all four
    to build the stack, which costs more than the extra kernel launches saves.
    """
    return x[:, meta.padded_src_index]


def _block_masked_mean(x: torch.Tensor, meta: VSAMetadata) -> torch.Tensor:
    """``[B, S_padded, H, D] -> [B, H, num_blocks, D]``, averaging valid tokens only.

    Padded slots hold arbitrary tokens (see :func:`_tile`), so they are zeroed
    by ``block_valid_mask`` before summing. Accumulate in fp32 because a block
    sums up to 64 bf16 values.
    """
    b, _, h, d = x.shape
    # Mask in the input dtype (the mask is exactly 0/1, so this is lossless) and
    # let the reduction accumulate in fp32 via ``dtype=``. Promoting the whole
    # padded tensor to fp32 first costs a full-size temporary and is measurably
    # slower for an identical result.
    blocks = x.view(
        b, meta.num_blocks, VSA_BLOCK_SIZE, h, d
    ) * meta.block_valid_mask.to(x.dtype)
    blocks = blocks.sum(dim=2, dtype=torch.float32)
    blocks = blocks / meta.variable_block_sizes.view(1, -1, 1, 1)
    return blocks.permute(0, 2, 1, 3).contiguous()


def vsa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate_compress: torch.Tensor,
    meta: VSAMetadata,
    sparsity: float,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Two-stage VSA attention.

    Args:
        query, key, value, gate_compress: ``[B, S, H, D]`` bf16, ``S ==
            meta.seq_len``, in the model's natural (un-tiled) token order.
        meta: layout tables from :func:`build_vsa_metadata`.
        sparsity: fraction of KV blocks to drop, e.g. ``0.9`` keeps the top 10%.
        softmax_scale: defaults to ``1 / sqrt(head_dim)``.

    Returns:
        ``[B, S, H, D]`` in the same token order and dtype as ``query``.
    """
    from flashinfer.cute_dsl.sparse import bsa_attn_blk64_fwd

    b, s, h, d = query.shape
    if s != meta.seq_len:
        raise ValueError(
            f"sequence length {s} does not match VSA metadata ({meta.seq_len}); "
            f"metadata was built for dit_seq_shape={meta.dit_seq_shape}"
        )
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    q_t, k_t, v_t, gate_t = (_tile(x, meta) for x in (query, key, value, gate_compress))

    # --- coarse stage: dense attention over one pooled token per cube --------
    q_c = _block_masked_mean(q_t, meta)
    k_c = _block_masked_mean(k_t, meta)
    v_c = _block_masked_mean(v_t, meta)

    scores = torch.matmul(q_c, k_c.transpose(-2, -1)) * softmax_scale

    # --- selection: top-k KV blocks per (head, query block) -----------------
    # Taken before the coarse output below so ``scores`` can be freed early; at
    # this shape it is a 331 MB fp32 tensor. Left unsorted on purpose: the kernel
    # accepts any order and runs at the same speed either way, so sorting only
    # costs a kernel launch. It would change the result by bf16 rounding alone
    # (same blocks, different accumulation order).
    topk = meta.topk_for_sparsity(sparsity)
    block_index = torch.topk(scores, topk, dim=-1).indices.to(torch.int32)

    out_coarse = torch.matmul(torch.softmax(scores, dim=-1), v_c)  # [B,H,NB,D]

    # --- fine stage: token-level attention inside the selected blocks -------
    out_fine, _ = bsa_attn_blk64_fwd(
        q_t,
        k_t,
        v_t,
        block_index,
        topk,
        block_sizes=meta.variable_block_sizes,
        softmax_scale=softmax_scale,
    )

    # --- combine, then drop padding and restore the original token order ----
    # In place into the (dead) gate buffer and then into the kernel's freshly
    # allocated output: same two roundings in the same order as
    # ``out + coarse * gate``, one allocation instead of three. Not ``addcmul_``,
    # which fuses in fp32 and rounds once, changing the low bits.
    out = out_fine.view(b, meta.num_blocks, VSA_BLOCK_SIZE, h, d)
    gate_view = gate_t.view(b, meta.num_blocks, VSA_BLOCK_SIZE, h, d)
    coarse_view = out_coarse.permute(0, 2, 1, 3).unsqueeze(2).to(out.dtype)
    torch.mul(coarse_view, gate_view, out=gate_view)
    out.add_(gate_view)
    return out.view(b, meta.padded_seq_len, h, d)[:, meta.untile_combined_index]
