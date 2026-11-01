# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block-scaled Q/K precisions of the SM100 indexer kernels.

``BLOCK_SCALED_FORMATS`` maps each precision to its Q/K and scale dtypes, its
scale vector size along head_dim and its kernel class. The indexer forward
(dense and compressed-logits) and the dense indexer score recompute validate
against it; the MXFP8 dense attention score reuses the MXFP8 entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.runtime import ceil_div
from .indexer_score_unified_sm100 import IndexerScoreUnifiedSm100
from .indexer_score_unified_sm100_mxfp4 import IndexerScoreUnifiedSm100Mxfp4
from .indexer_score_unified_sm100_mxfp8 import IndexerScoreUnifiedSm100Mxfp8


@dataclass(frozen=True)
class BlockScaledFormat:
    """Q/K storage contract of a block-scaled indexer precision.

    ``data_dtype`` / ``scale_dtype`` resolve the named torch dtypes of Q/K and
    their scales, or are ``None`` when the installed torch lacks them
    (``validate_indexer_precision`` reports that). One stored element holds
    ``elems_per_storage`` logical head_dim values (2 for packed FP4, whose last
    dim is ``head_dim // 2``); each scale covers ``sf_vec_size`` of them.
    """

    data_dtype_name: str
    scale_dtype_name: str
    sf_vec_size: int
    elems_per_storage: int
    kernel_cls: type[IndexerScoreUnifiedSm100]

    @property
    def data_dtype(self) -> Optional[torch.dtype]:
        return getattr(torch, self.data_dtype_name, None)

    @property
    def scale_dtype(self) -> Optional[torch.dtype]:
        return getattr(torch, self.scale_dtype_name, None)


BLOCK_SCALED_FORMATS = {
    "mxfp8": BlockScaledFormat(
        data_dtype_name="float8_e4m3fn",
        scale_dtype_name="float8_e8m0fnu",
        sf_vec_size=32,
        elems_per_storage=1,
        kernel_cls=IndexerScoreUnifiedSm100Mxfp8,
    ),
    "mxfp4": BlockScaledFormat(
        data_dtype_name="float4_e2m1fn_x2",
        scale_dtype_name="float8_e8m0fnu",
        sf_vec_size=32,
        elems_per_storage=2,
        kernel_cls=IndexerScoreUnifiedSm100Mxfp4,
    ),
}
SM100_INDEXER_PRECISIONS = ("bf16", *BLOCK_SCALED_FORMATS)
# Block-scaled precision names for error messages: "'mxfp8' or 'mxfp4'".
BLOCK_SCALED_PRECISION_NAMES = " or ".join(repr(name) for name in BLOCK_SCALED_FORMATS)
# The only logical head dim the block-scaled indexer kernels support.
BLOCK_SCALED_HEAD_DIM = 128


def validate_indexer_precision(precision: str) -> str:
    """Lowercase ``precision`` and reject one the SM100 indexer or the installed torch does not support."""
    precision = precision.lower()
    if precision not in SM100_INDEXER_PRECISIONS:
        raise ValueError(f"precision must be one of {SM100_INDEXER_PRECISIONS}, got {precision!r}")
    fmt = BLOCK_SCALED_FORMATS.get(precision)
    if fmt is not None:
        missing = [name for name in (fmt.data_dtype_name, fmt.scale_dtype_name) if getattr(torch, name, None) is None]
        if missing:
            raise NotImplementedError(f"precision={precision!r} requires a torch build with " + " and ".join(f"torch.{name}" for name in missing))
    return precision


def validate_sf_vec_size(precision: str, sf_vec_size: int) -> None:
    """A block-scaled ``precision`` requires its format's scale vector size (``ValueError`` otherwise)."""
    fmt = BLOCK_SCALED_FORMATS.get(precision)
    if fmt is not None and sf_vec_size != fmt.sf_vec_size:
        raise ValueError(f"precision={precision!r} currently requires sf_vec_size={fmt.sf_vec_size}")


def validate_block_scaled_inputs(
    precision: str,
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    q_scale: Optional[torch.Tensor],
    k_scale: Optional[torch.Tensor],
) -> None:
    """Check the dtype and device contract of a block-scaled precision's Q/K/W and scales."""
    fmt = BLOCK_SCALED_FORMATS[precision]
    if q.dtype != fmt.data_dtype or k.dtype != fmt.data_dtype:
        raise TypeError(f"precision={precision!r} requires q and k to be {fmt.data_dtype}")
    if w.dtype != torch.bfloat16:
        raise TypeError(f"precision={precision!r} requires w to be torch.bfloat16")
    if q_scale is None or k_scale is None:
        raise ValueError(f"precision={precision!r} requires q_scale and k_scale")
    if q_scale.dtype != fmt.scale_dtype or k_scale.dtype != fmt.scale_dtype:
        raise TypeError(f"precision={precision!r} requires q_scale and k_scale to be {fmt.scale_dtype}")
    if q_scale.device != q.device or k_scale.device != k.device:
        raise ValueError("q_scale/k_scale must be on the same device as q/k")


def logical_head_dim(precision: str, storage_head_dim: int) -> int:
    """Logical head dim of Q/K whose last dim is ``storage_head_dim``.

    A block-scaled precision must hold ``BLOCK_SCALED_HEAD_DIM`` logical values;
    otherwise this raises ``ValueError`` naming the stored last dim it needs.
    """
    fmt = BLOCK_SCALED_FORMATS.get(precision)
    if fmt is None:
        return storage_head_dim
    head_dim = storage_head_dim * fmt.elems_per_storage
    if head_dim != BLOCK_SCALED_HEAD_DIM:
        stored = BLOCK_SCALED_HEAD_DIM // fmt.elems_per_storage
        packing = f" ({fmt.elems_per_storage} values per torch.{fmt.data_dtype_name} element)" if fmt.elems_per_storage > 1 else ""
        raise ValueError(f"precision={precision!r} requires head_dim={BLOCK_SCALED_HEAD_DIM}, stored as q/k last dim {stored}{packing}; got {storage_head_dim}")
    return head_dim


def kernel_qk_view(precision: str, t: torch.Tensor) -> torch.Tensor:
    """Q or K as the ``precision`` kernel is compiled and launched with.

    A packed format (more than one value per stored element) passes its raw bytes
    as a ``torch.uint8`` view, which the kernel recasts to its element type: CuTe
    DSL before 4.8 binds a ``torch.float4_e2m1fn_x2`` tensor with its packed shape
    and strides, not the E2M1 ones the kernel is compiled for. Other precisions
    pass ``t`` unchanged.
    """
    fmt = BLOCK_SCALED_FORMATS.get(precision)
    if fmt is None or fmt.elems_per_storage == 1:
        return t
    return t.view(torch.uint8)


def bshd_packed_scale_shape(
    *,
    bs: int,
    seqlen: int,
    n_heads_kv: int,
    sf_groups: int,
    pack_q_heads: int = 1,
) -> tuple[int, int, int]:
    """Shape ``(L, padded_MN, padded_sf_groups)`` of BSHD blockscaled-packed scales.

    ``MN = seqlen * pack_q_heads`` rounds up to 128 rows and the scale groups
    to 4 columns.
    """
    mn = seqlen * pack_q_heads
    return (
        bs * n_heads_kv,
        ceil_div(mn, 128) * 128,
        ceil_div(sf_groups, 4) * 4,
    )


def validate_thd_scale_contract(
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    cu_seqlens_q_scale_padded: Optional[torch.Tensor],
    cu_seqlens_k_scale_padded: Optional[torch.Tensor],
    *,
    precision: str,
    bs: int,
    n_heads_kv: int,
    sf_groups: int,
    device: torch.device,
) -> None:
    """Check the THD scale padded prefixes and the packed ``(n_heads_kv, multiple_of_128, padded_sf_groups)`` scales."""
    if cu_seqlens_q_scale_padded is None or cu_seqlens_k_scale_padded is None:
        raise ValueError(f"THD {precision.upper()} requires Q/K scale padded cu_seqlens")

    for prefix, name in (
        (cu_seqlens_q_scale_padded, "cu_seqlens_q_scale_padded"),
        (cu_seqlens_k_scale_padded, "cu_seqlens_k_scale_padded"),
    ):
        if prefix.dtype != torch.int32 or prefix.ndim != 1 or prefix.shape[0] != bs + 1 or not prefix.is_contiguous() or prefix.device != device:
            raise ValueError(f"{name} must be contiguous int32 shape ({bs + 1},) on {device}")

    sf_padded = ceil_div(sf_groups, 4) * 4
    for scale, name in ((q_scale, "q_scale"), (k_scale, "k_scale")):
        if scale.ndim != 3:
            raise ValueError(f"{name} must be a 3D packed scale tensor")
        if scale.shape[0] != n_heads_kv or scale.shape[1] % 128 != 0 or scale.shape[2] != sf_padded:
            raise ValueError(f"THD {name} must have shape ({n_heads_kv}, multiple_of_128, " f"{sf_padded}), got {tuple(scale.shape)}")
        if scale.device != device:
            raise ValueError("q_scale/k_scale must be on the same device as q/k")
