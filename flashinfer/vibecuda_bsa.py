"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

VibeCUDA block-sparse attention (SM100/SM103)
=============================================

GQA block-sparse attention forward driven by a per-query-head boolean block
mask.  A token pair (query row ``i``, key column ``j`` of query head ``h``) is
admitted iff ``block_mask[h, i // block_size, j // block_size]`` is true;
non-admitted scores are ``-inf``.  The KV head used by query head ``h`` is
``h // (num_qo_heads // num_kv_heads)``; ``sm_scale`` defaults to
``1 / sqrt(head_dim)``.

The CUDA kernel (``include/flashinfer/vibecuda/bsa_fwd.cuh``) uses HMMA
(``mma.sync.m16n8k16``) for Q@K^T and P@V with a stable FP32 max/sum online
softmax over admitted 64-key chunks.  TMA (``cp.async.bulk.tensor``) stages
Q/K/V into swizzled shared-memory panels.  A runtime dispatch ladder picks
between in-CTA split kernels for grid-underfilled shapes (64/32/16-row
tiles), a GQA head-packed kernel and an 8-warp wide kernel for large GQA
grid-saturated shapes, and a cross-CTA split kernel plus a PDL-coupled merge
kernel for dense rows (``split_g > 1``).
"""

import math
from typing import Optional, Tuple, Union

import torch

from .api_logging import flashinfer_api
from .jit.vibecuda import VibeCUDABSATarget, get_vibecuda_bsa_module
from .utils import get_compute_capability

_SUPPORTED_CC = ((10, 0), (10, 3))
_SUPPORTED_HEAD_DIMS = (64, 96, 128)
_MAX_SELECTED_BLOCKS = 8
_MAX_QUERY_SEQUENCE_LENGTH = 1024


def _vibecuda_bsa_target(device: torch.device) -> VibeCUDABSATarget:
    """Resolve the JIT target for a device, failing loudly when unsupported."""
    cc = get_compute_capability(device)
    arch = cc[0] * 10 + cc[1]
    if cc == (10, 0):
        return "sm100a"
    if cc == (10, 3):
        return "sm103a"
    raise RuntimeError(
        f"vibecuda block-sparse attention requires SM100/SM103 (compute "
        f"capability (10, 0) or (10, 3)), current device has compute capability "
        f"{cc} (SM{arch})"
    )


def vibecuda_bsa_split_g(max_selected_blocks: int, block_size: int, n: int) -> int:
    """Return the cross-CTA split count for a block-sparse problem.

    The split count ``split_g > 1`` routes to the cross-CTA split kernel plus
    merge kernel (best when many 64-key chunks are admitted per row tile);
    ``split_g == 1`` keeps the whole row's chunks in one CTA with a fused
    normalize epilogue.  The heuristic mirrors the tuned policy: below ~8
    estimated admitted 64-key chunks per (64-row tile, head) the fused single
    split wins; denser rows target ~2 chunks per split CTA.

    Parameters
    ----------
    max_selected_blocks : int
        Maximum number of admitted (true) blocks in any
        ``(query head, query block)`` row of the block mask.
    block_size : int
        The square block size (multiple of 64).
    n : int
        Number of key/value tokens ``N``.
    """
    cap = max(1, (n + 63) // 64)
    est = max_selected_blocks * (block_size // 64) if max_selected_blocks > 0 else cap
    return min(max(est // 2, 2), 16, cap) if est >= 8 else 1


def vibecuda_bsa_workspace_numel(
    m: int, num_qo_heads: int, head_dim: int, split_g: int
) -> int:
    """Return the required float32 workspace size for a split count.

    ``split_g == 1`` needs no workspace (returns 0).  Otherwise the
    cross-CTA split kernel stages ``split_g`` partials of
    ``(padded rows, head)`` float accumulators plus ``(m, l, aux)`` slots.
    """
    if split_g <= 1:
        return 0
    rows_pad = ((m + 63) // 64) * 64
    return split_g * rows_pad * num_qo_heads * (head_dim + 4)


def _check_vibecuda_bsa_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int,
) -> None:
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"vibecuda block-sparse attention requires bfloat16 or float16 "
            f"inputs (got {q.dtype})"
        )
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(
            "vibecuda block-sparse attention requires matching Q/K/V dtypes "
            f"(got q={q.dtype}, k={k.dtype}, v={v.dtype})"
        )
    if block_mask.dtype != torch.bool:
        raise ValueError(f"block_mask must have dtype bool (got {block_mask.dtype})")
    M, HQ, D = q.shape
    N, HKV, _ = k.shape
    if M > _MAX_QUERY_SEQUENCE_LENGTH:
        raise ValueError(
            "vibecuda block-sparse attention currently supports query "
            f"sequence lengths up to {_MAX_QUERY_SEQUENCE_LENGTH} (got M={M})"
        )
    if D not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(
            f"vibecuda block-sparse attention requires head_dim in "
            f"{_SUPPORTED_HEAD_DIMS} (got {D})"
        )
    if block_size % 64 != 0:
        raise ValueError(
            f"vibecuda block-sparse attention requires block_size to be a "
            f"multiple of 64 (got {block_size})"
        )
    if HKV == 0 or HQ % HKV != 0:
        raise ValueError(
            f"num_qo_heads ({HQ}) must be a multiple of num_kv_heads ({HKV})"
        )
    MB = (M + block_size - 1) // block_size
    NB = (N + block_size - 1) // block_size
    if block_mask.shape != (HQ, MB, NB):
        raise ValueError(
            f"block_mask must have shape (num_qo_heads={HQ}, ceil(M/block_size)="
            f"{MB}, ceil(N/block_size)={NB}), got {tuple(block_mask.shape)}"
        )
    max_selected = (
        int(block_mask.to(torch.int32).sum(dim=-1).max().item())
        if block_mask.numel()
        else 0
    )
    if max_selected > _MAX_SELECTED_BLOCKS:
        raise ValueError(
            "vibecuda block-sparse attention currently supports at most "
            f"{_MAX_SELECTED_BLOCKS} selected blocks per (head, query block) "
            f"row (got {max_selected})"
        )


@flashinfer_api
def vibecuda_block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int,
    sm_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    workspace: Optional[torch.Tensor] = None,
    split_g: Optional[int] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""GQA block-sparse attention forward (VibeCUDA SM100/SM103 kernels).

    Parameters
    ----------
    q : torch.Tensor
        Query tensor with shape ``(M, num_qo_heads, head_dim)``, bfloat16 or
        float16, CUDA, contiguous.
    k : torch.Tensor
        Key tensor with shape ``(N, num_kv_heads, head_dim)``, same dtype as
        ``q``.
    v : torch.Tensor
        Value tensor with shape ``(N, num_kv_heads, head_dim)``, same dtype as
        ``q``.
    block_mask : torch.Tensor
        Per-query-head boolean block mask with shape
        ``(num_qo_heads, ceil(M / block_size), ceil(N / block_size))``.
        ``block_mask[h, i, j] == True`` admits every token pair between query
        block ``i`` and key block ``j`` of query head ``h``.
    block_size : int
        Square block size of the mask.  Must be a multiple of 64.
    sm_scale : Optional[float]
        Softmax scale.  If ``None``, defaults to ``1 / sqrt(head_dim)``.
    out : Optional[torch.Tensor]
        Optional preallocated output with the same shape/dtype as ``q``.
        Freshly allocated when ``None``.
    lse : Optional[torch.Tensor]
        Optional preallocated FP32 LSE output with shape ``(M, num_qo_heads)``.
        Freshly allocated when ``return_lse`` is true and this is ``None``.
    return_lse : bool
        Whether to return the log-sum-exp of the scaled, admitted scores.
    workspace : Optional[torch.Tensor]
        Optional preallocated float32 scratch for the cross-CTA split path.
        Must hold at least
        :func:`vibecuda_bsa_workspace_numel` elements when ``split_g > 1``.
        Freshly allocated when ``split_g > 1`` and this is ``None``.
    split_g : Optional[int]
        Cross-CTA split count in ``[1, 16]``.  When ``None``, defaults to 1
        (single-CTA per-row fused path); use
        :func:`vibecuda_bsa_split_g` for the tuned heuristic on dense rows.

    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        If ``return_lse`` is ``False``, the attention output, shape
        ``(M, num_qo_heads, head_dim)``.  Otherwise a tuple ``(output, lse)``
        where ``lse`` has shape ``(M, num_qo_heads)`` and dtype float32.
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or block_mask.ndim != 3:
        raise ValueError(
            "q, k, v and block_mask must all be 3D tensors "
            f"(got {q.ndim}, {k.ndim}, {v.ndim}, {block_mask.ndim})"
        )
    _check_vibecuda_bsa_inputs(q, k, v, block_mask, block_size)
    target = _vibecuda_bsa_target(q.device)
    module = get_vibecuda_bsa_module(target)

    M, HQ, D = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    if out is None:
        out = torch.empty_like(q)
    if return_lse:
        if lse is None:
            lse = torch.empty((M, HQ), dtype=torch.float32, device=q.device)
    else:
        if lse is None:
            lse = torch.empty(0, dtype=torch.float32, device=q.device)
        elif lse.numel() != 0 or lse.dtype != torch.float32 or lse.device != q.device:
            raise ValueError(
                "unused lse must be an empty float32 tensor on the query device"
            )
    g = 1 if split_g is None else int(split_g)
    if not (1 <= g <= 16):
        raise ValueError(f"split_g must be in [1, 16] (got {g})")
    if g > 1:
        if workspace is None:
            workspace = torch.zeros(
                vibecuda_bsa_workspace_numel(M, HQ, D, g),
                dtype=torch.float32,
                device=q.device,
            )
    else:
        workspace = torch.empty(0, dtype=torch.float32, device=q.device)

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    module.vibecuda_bsa_fwd(
        out,
        lse,
        q_c,
        k_c,
        v_c,
        block_mask.contiguous(),
        workspace,
        block_size,
        sm_scale,
        g,
        return_lse,
    )
    return (out, lse) if return_lse else out


__all__ = [
    "vibecuda_block_sparse_attention",
    "vibecuda_bsa_split_g",
    "vibecuda_bsa_workspace_numel",
]
