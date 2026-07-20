# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/integration/residual_kernels.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""CuTeDSL kernels for the mHC residual path."""

from __future__ import annotations

from functools import lru_cache
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as cutlass_utils
import cutlass.utils.hopper_helpers as sm90_utils_basic
import torch
from cutlass import Float32, Int32, Uint32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum

from flashinfer.experimental.sm12x._lib.compiler import (
    DimKey,
    KernelCompileSpec,
    tensor_key,
)
from flashinfer.experimental.sm12x._lib.compiler import (
    launch as sm12x_launch,
)
from flashinfer.experimental.sm12x._lib.intrinsics import (
    bf16_mma_m16n8k16_f32,
    bfloat2_to_float2_scaled,
    f32_to_raw_bits,
    get_ptr_as_int64,
    ld_global_nc_u32,
    ldmatrix_m8n8x4_b16,
    pack_f32x2_to_bfloat2,
    shared_ptr_to_u32,
    st_shared_u32,
    tf32_mma_m16n8k8_f32,
)
from flashinfer.experimental.sm12x._lib.utils import current_cuda_stream

_MHC_MULT = 4
_TOKENS = 1
_HIDDEN = 4096
_TOTAL_K = _MHC_MULT * _HIDDEN
_SPLIT_K = 64
_SOURCE_TILE_H = 128
_SOURCE_TILES = _HIDDEN // _SOURCE_TILE_H
_SUPPORTED_HIDDEN_SIZES = (_HIDDEN, 7168)
_MIXES = 24
_PARTIALS = 1 + _MIXES
_PARTIALS_PER_CTA = 2
_MHC_PDL = os.getenv("FLASHINFER_EXP_SM12X_MHC_PDL", "0") != "0"
# Partials handled per post_pre-partial CTA. mix_groups = ceil(25/this), so the
# partial-kernel grid is (32 source tiles x mix_groups). 4 (-> 7 groups, 224
# CTAs) maximizes fn-read parallelism without excess grid-scheduling overhead.
_POST_PRE_PARTIALS_PER_CTA = 4
_THREADS = 128
_PREFILL_THREADS = int(os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_THREADS", "512"))
_PREFILL_GRAM_THREADS = int(
    os.getenv(
        "FLASHINFER_EXP_SM12X_MHC_PREFILL_GRAM_THREADS",
        os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_THREADS", "1024"),
    )
)
_PREFILL_BLOCK_M = 2
_PREFILL_BLOCK_TILE_N = 24
_PREFILL_MMA_THREADS = 32
_PREFILL_MMA_TILE_M = 16
_PREFILL_MMA_TILE_N = 8
_PREFILL_MMA_TILE_K = 16
_PREFILL_TMA_COMPUTE_WARPS = int(
    os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TMA_WARPS", "8")
)
_PREFILL_TMA_THREADS = (_PREFILL_TMA_COMPUTE_WARPS + 1) * 32
_PREFILL_TMA_TILE_M = int(
    os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TMA_TILE_M", "128")
)
_PREFILL_TMA_TILE_N = int(
    os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TMA_TILE_N", "16")
)
_PREFILL_TMA_TILE_K = int(
    os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TMA_TILE_K", "64")
)
_PREFILL_TMA_STAGES = int(os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TMA_STAGES", "3"))
_PREFILL_TF32_TMA_M_WARPS = int(
    os.getenv(
        "FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_M_WARPS",
        os.getenv(
            "FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_WARPS",
            os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TMA_WARPS", "1"),
        ),
    )
)
_PREFILL_TF32_TMA_N_WARPS = int(
    os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_N_WARPS", "1")
)
_PREFILL_TF32_TMA_COMPUTE_WARPS = _PREFILL_TF32_TMA_M_WARPS * _PREFILL_TF32_TMA_N_WARPS
_PREFILL_TF32_TMA_THREADS = (_PREFILL_TF32_TMA_COMPUTE_WARPS + 1) * 32
_PREFILL_TF32_TMA_TILE_M = int(
    os.getenv(
        "FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_TILE_M",
        os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TMA_TILE_M", "16"),
    )
)
_PREFILL_TF32_TMA_TILE_N = int(
    os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_TILE_N", "8")
)
_PREFILL_TF32_TMA_TILE_K = int(
    os.getenv(
        "FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_TILE_K",
        os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TMA_TILE_K", "256"),
    )
)
_PREFILL_TF32_TMA_STAGES = int(
    os.getenv(
        "FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_STAGES",
        os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TMA_STAGES", "1"),
    )
)
_PREFILL_TF32_TMA_CHUNK_MIN_TOKENS = int(
    os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_CHUNK_MIN_TOKENS", "4096")
)
_PREFILL_TF32_TMA_LONG_MIN_TOKENS = int(
    os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_LONG_MIN_TOKENS", "8192")
)
# At hidden=4096, one CTA owns all 24 mix columns so A is loaded once instead
# of once per 8-column N tile. M192/K64 gives the best 4096-token balance of B
# reuse and SM coverage; K splitting supplies enough CTAs for all SMs.
# Keep the previous geometry for hidden=7168, where wide-N regresses.
_PREFILL_TF32_TMA_CHUNK_4096_M_WARPS = int(
    os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_CHUNK_M_WARPS", "12")
)
_PREFILL_TF32_TMA_CHUNK_OTHER_M_WARPS = int(
    os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_CHUNK_M_WARPS", "2")
)
_PREFILL_TF32_TMA_CHUNK_4096_N_WARPS = int(
    os.getenv(
        "FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_CHUNK_N_WARPS",
        os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_N_WARPS", "1"),
    )
)
_PREFILL_TF32_TMA_CHUNK_OTHER_N_WARPS = int(
    os.getenv(
        "FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_CHUNK_N_WARPS",
        os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_N_WARPS", "1"),
    )
)
_PREFILL_TF32_TMA_CHUNK_4096_TILE_M = int(
    os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_CHUNK_TILE_M", "192")
)
_PREFILL_TF32_TMA_CHUNK_OTHER_TILE_M = int(
    os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_CHUNK_TILE_M", "32")
)
_PREFILL_TF32_TMA_CHUNK_4096_TILE_N = int(
    os.getenv(
        "FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_CHUNK_TILE_N",
        os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_TILE_N", "24"),
    )
)
_PREFILL_TF32_TMA_CHUNK_OTHER_TILE_N = int(
    os.getenv(
        "FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_CHUNK_TILE_N",
        os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_TILE_N", "8"),
    )
)
_PREFILL_TF32_TMA_CHUNK_4096_TILE_K = int(
    os.getenv(
        "FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_CHUNK_TILE_K",
        os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_TILE_K", "64"),
    )
)
_PREFILL_TF32_TMA_CHUNK_4096_STAGES = int(
    os.getenv(
        "FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_CHUNK_STAGES",
        os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_STAGES", "2"),
    )
)
_PREFILL_TF32_TMA_CHUNK_4096_K_SPLITS = int(
    os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_CHUNK_K_SPLITS", "8")
)
# At 8192+ tokens, four K splits give the same 256-CTA launch while doubling
# useful work per CTA relative to the 4096-token specialization.
_PREFILL_TF32_TMA_LONG_4096_M_WARPS = int(
    os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_LONG_M_WARPS", "8")
)
_PREFILL_TF32_TMA_LONG_4096_N_WARPS = int(
    os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_LONG_N_WARPS", "1")
)
_PREFILL_TF32_TMA_LONG_4096_TILE_M = int(
    os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_LONG_TILE_M", "128")
)
_PREFILL_TF32_TMA_LONG_4096_TILE_N = int(
    os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_LONG_TILE_N", "24")
)
_PREFILL_TF32_TMA_LONG_4096_TILE_K = int(
    os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_LONG_TILE_K", "64")
)
_PREFILL_TF32_TMA_LONG_4096_STAGES = int(
    os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_LONG_STAGES", "2")
)
_PREFILL_TF32_TMA_LONG_4096_K_SPLITS = int(
    os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_TF32_TMA_LONG_K_SPLITS", "4")
)
_PREFILL_FINALIZE_THREADS = int(
    os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_FINALIZE_THREADS", "256")
)
_POST_PRE_CHUNK = 12

# --- Gram-trick split finalize (multi-CTA fuse_norm, no per-h norm reduction) -
# The post_pre-partial kernel additionally reduces the 4x4 Gram of residual_out
# G[m,m'] = sum_h ro[m,h] ro[m',h] (10 unique entries) into free partials rows
# [32, 64) (one row per source tile). The finalize then gets sum_h y^2 =
# pre^T G pre as a scalar, so it no longer reduces over hidden and can run
# multi-CTA (one block per hidden tile) like the no-norm path. 10 packed pairs:
#   0:(0,0) 1:(1,1) 2:(2,2) 3:(3,3) 4:(0,1) 5:(0,2) 6:(0,3) 7:(1,2) 8:(1,3) 9:(2,3)
_GRAM_PAIRS = 10
_GRAM_ROW0 = 32  # gram[tile] stored at partials[token, 32 + tile, 0:10]
# 1024 threads cover one hidden tile per loop iteration.
_GRAM_BLOCK_H = 1024


@dsl_user_op
def _materialize_residual_gram_f32(value, *, loc=None, ip=None):
    """Keep rounded BF16 residual values in the F32 arithmetic domain."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _source_tiles_for_hidden(hidden_size: int) -> int:
    hidden_size = int(hidden_size)
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    if hidden_size not in _SUPPORTED_HIDDEN_SIZES:
        raise ValueError(
            f"hidden_size={hidden_size} is not supported by the mHC kernels; "
            f"supported hidden sizes are {_SUPPORTED_HIDDEN_SIZES}"
        )
    if hidden_size % _SOURCE_TILE_H != 0:
        raise ValueError(
            f"hidden_size={hidden_size} must be divisible by source tile "
            f"{_SOURCE_TILE_H}"
        )
    return hidden_size // _SOURCE_TILE_H


def _split_k_for_hidden(hidden_size: int) -> int:
    return 2 * _source_tiles_for_hidden(hidden_size)


def _validate_split_k(hidden_size: int, split_k: int) -> None:
    required = _split_k_for_hidden(hidden_size)
    if int(split_k) != required:
        raise ValueError(
            f"split_k={split_k} does not match hidden_size={hidden_size}; "
            f"expected {required}"
        )


def _hidden_specialization_name(hidden_size: int) -> str:
    hidden_size = int(hidden_size)
    if hidden_size not in _SUPPORTED_HIDDEN_SIZES:
        raise ValueError(
            f"hidden_size={hidden_size} is not supported by the mHC kernels; "
            f"supported hidden sizes are {_SUPPORTED_HIDDEN_SIZES}"
        )
    return f"hidden{hidden_size}"


def _convert_layout_acc_mn(
    acc_layout: cute.Layout,
    transpose: bool = False,
) -> cute.Layout:
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    shape = (
        (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
        (
            acc_layout_col_major.shape[0][0],
            *acc_layout_col_major.shape[0][2:],
            acc_layout_col_major.shape[2],
        ),
        *acc_layout_col_major.shape[3:],
    )
    stride = (
        (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
        (
            acc_layout_col_major.stride[0][0],
            *acc_layout_col_major.stride[0][2:],
            acc_layout_col_major.stride[2],
        ),
        *acc_layout_col_major.stride[3:],
    )
    if const_expr(transpose):
        shape = (shape[1], shape[0], *shape[2:])
        stride = (stride[1], stride[0], *stride[2:])
    return cute.composition(acc_layout, cute.make_layout(shape, stride=stride))


def _reshape_acc_to_mn(
    acc: cute.Tensor,
    transpose: bool = False,
) -> cute.Tensor:
    return cute.make_tensor(
        acc.iterator,
        _convert_layout_acc_mn(acc.layout, transpose=transpose),
    )


def _to_kernel_tensor(
    tensor: torch.Tensor,
    dtype: type[cutlass.Numeric],
    *,
    assumed_align: int = 16,
    dynamic_layout: bool = False,
) -> cutlass.cute.Tensor:
    tensor = tensor.detach()
    cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
    cute_tensor.element_type = dtype
    if dynamic_layout and tensor.ndim >= 1:
        leading_dim = next(
            (idx for idx, stride in enumerate(tensor.stride()) if stride == 1),
            None,
        )
        if leading_dim is not None:
            cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
            cute_tensor.element_type = dtype
    return cute_tensor


def _assume_tma_source_aligned(tensor: cute.Tensor) -> cute.Tensor:
    divby = 128 // tensor.element_type.width
    strides = []
    for dim, stride in enumerate(tensor.stride):
        if dim == 1 or isinstance(stride, int):
            strides.append(stride)
        else:
            strides.append(cute.assume(stride, divby=divby))
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout(tensor.shape, stride=tuple(strides)),
    )


def _tensor_meta_key(
    tensor: torch.Tensor,
) -> tuple[tuple[int, ...], tuple[int, ...], str, tuple[str, int | None]]:
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        (tensor.device.type, tensor.device.index),
    )


def _validate_tensor_shape(
    name: str,
    tensor: torch.Tensor,
    expected: tuple[int, ...],
) -> None:
    actual = tuple(int(dim) for dim in tensor.shape)
    if actual != expected:
        raise ValueError(f"{name} must have shape {expected}, got {actual}")


def _norm_weight_kernel_tensor(
    norm_weight: torch.Tensor | None,
    fallback: torch.Tensor,
) -> cutlass.cute.Tensor:
    if norm_weight is None:
        return _to_kernel_tensor(fallback, cutlass.BFloat16)
    if norm_weight.dtype == torch.bfloat16:
        return _to_kernel_tensor(norm_weight, cutlass.BFloat16)
    if norm_weight.dtype == torch.float32:
        return _to_kernel_tensor(norm_weight, cutlass.Float32)
    raise ValueError(f"norm_weight must be bf16 or fp32, got {norm_weight.dtype}")


@lru_cache(maxsize=8)
def _finalize_storage_cls(num_threads: int, include_y: bool):
    class FinalizeStorage:
        pass

    annotations = {
        "partials": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, num_threads],
            16,
        ],
        "pre": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _MHC_MULT],
            16,
        ],
        "post": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _MHC_MULT],
            16,
        ],
        "comb": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _MHC_MULT * _MHC_MULT],
            16,
        ],
    }
    if include_y:
        annotations["y"] = cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, _HIDDEN],
            16,
        ]
    FinalizeStorage.__annotations__ = annotations
    return cute.struct(FinalizeStorage)


def _validate_post_pre_partials_per_cta(partials_per_cta: int) -> int:
    partials_per_cta = int(partials_per_cta)
    if partials_per_cta <= 0 or partials_per_cta > _PARTIALS:
        raise ValueError(
            "FLASHINFER_EXP_SM12X_MHC_PARTIALS_PER_CTA must be in "
            f"[1, {_PARTIALS}], got {partials_per_cta}"
        )
    return partials_per_cta


def _selected_post_pre_decode_split_n(
    *,
    num_tokens: int,
    hidden_size: int,
    compute_capability: tuple[int, int] | None = None,
) -> tuple[int, int]:
    raw_splits = os.environ.get("FLASHINFER_EXP_SM12X_MHC_DECODE_SPLITS")
    if raw_splits is not None and raw_splits != "":
        try:
            splits = int(raw_splits)
            tile_n = int(os.environ.get("FLASHINFER_EXP_SM12X_MHC_DECODE_TILE_N", "3"))
        except ValueError as exc:
            raise ValueError(
                "FLASHINFER_EXP_SM12X_MHC_DECODE_SPLITS and FLASHINFER_EXP_SM12X_MHC_DECODE_TILE_N must be integers"
            ) from exc
        if splits == 0:
            return 0, 0
    else:
        if compute_capability is None and torch.cuda.is_available():
            compute_capability = tuple(torch.cuda.get_device_capability())
        if compute_capability != (12, 1) or int(hidden_size) != _HIDDEN:
            return 0, 0
        if int(num_tokens) >= 10:
            splits, tile_n = 8, 6
        elif int(num_tokens) >= 8:
            splits, tile_n = 4, 6
        else:
            return 0, 0
    if splits <= 0 or splits > _SOURCE_TILES or int(hidden_size) % splits != 0:
        raise ValueError(
            "FLASHINFER_EXP_SM12X_MHC_DECODE_SPLITS must be a positive divisor of hidden_size "
            f"no larger than {_SOURCE_TILES}, got {splits}"
        )
    if tile_n <= 0 or _MIXES % tile_n != 0:
        raise ValueError(
            f"FLASHINFER_EXP_SM12X_MHC_DECODE_TILE_N must divide {_MIXES}, got {tile_n}"
        )
    return splits, tile_n


def _selected_mhc_decode_finalize_threads(
    *,
    num_tokens: int,
    hidden_size: int,
    compute_capability: tuple[int, int] | None = None,
) -> int:
    raw = os.environ.get("FLASHINFER_EXP_SM12X_MHC_DECODE_FINALIZE_THREADS")
    if raw is not None and raw != "":
        try:
            threads = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"invalid FLASHINFER_EXP_SM12X_MHC_DECODE_FINALIZE_THREADS={raw!r}"
            ) from exc
    else:
        if compute_capability is None and torch.cuda.is_available():
            compute_capability = tuple(torch.cuda.get_device_capability())
        if compute_capability != (12, 1) or int(hidden_size) != _HIDDEN:
            return 0
        if int(num_tokens) >= 16:
            threads = 128
        elif int(num_tokens) >= 13:
            threads = 512
        elif int(num_tokens) >= 10:
            threads = 128
        elif int(num_tokens) >= 8:
            threads = 512
        else:
            return 0
    if threads == 0:
        return 0
    if (
        threads <= 0
        or threads > 1024
        or threads % 32 != 0
        or int(hidden_size) % (2 * threads) != 0
    ):
        raise ValueError(
            "FLASHINFER_EXP_SM12X_MHC_DECODE_FINALIZE_THREADS must be a positive multiple of "
            "32 that evenly vectorizes hidden_size, got "
            f"threads={threads}, hidden_size={hidden_size}"
        )
    return threads


def _selected_post_pre_partials_per_cta(
    *,
    num_tokens: int,
    hidden_size: int,
    compute_capability: tuple[int, int] | None = None,
) -> int:
    raw = os.environ.get("FLASHINFER_EXP_SM12X_MHC_PARTIALS_PER_CTA")
    if raw is not None and raw != "":
        try:
            return _validate_post_pre_partials_per_cta(int(raw))
        except ValueError as exc:
            raise ValueError(
                f"invalid FLASHINFER_EXP_SM12X_MHC_PARTIALS_PER_CTA={raw!r}"
            ) from exc

    if compute_capability is None and torch.cuda.is_available():
        compute_capability = tuple(torch.cuda.get_device_capability())
    if compute_capability == (12, 1) and int(hidden_size) == _HIDDEN:
        if int(num_tokens) >= 8:
            return 25
        if int(num_tokens) >= 4:
            return 9
    return _POST_PRE_PARTIALS_PER_CTA


@lru_cache(maxsize=16)
def _post_pre_partial_group_storage_cls(
    partials_per_cta: int,
    compute_gram: bool = False,
):
    partials_per_cta = _validate_post_pre_partials_per_cta(partials_per_cta)

    class PostPrePartialGroupStorage:
        pass

    annotations = {
        "warp_sums": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, partials_per_cta * (_THREADS // 32)],
            16,
        ],
    }
    if compute_gram:
        annotations["gram_sums"] = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _GRAM_PAIRS * (_THREADS // 32)],
            16,
        ]
    PostPrePartialGroupStorage.__annotations__ = annotations
    return cute.struct(PostPrePartialGroupStorage)


@lru_cache(maxsize=16)
def _post_pre_decode_split_n_storage_cls(
    tile_n: int,
    compute_gram: bool = False,
):
    nwarps = 256 // 32

    class PostPreDecodeSplitNStorage:
        pass

    annotations = {
        "warp_sums": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, (tile_n + 1) * nwarps],
            16,
        ],
        "post": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _MHC_MULT],
            16,
        ],
        "comb": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _MHC_MULT * _MHC_MULT],
            16,
        ],
    }
    if compute_gram:
        annotations["gram_sums"] = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _GRAM_PAIRS * nwarps],
            16,
        ]
    PostPreDecodeSplitNStorage.__annotations__ = annotations
    return cute.struct(PostPreDecodeSplitNStorage)


@lru_cache(maxsize=4)
def _post_pre_prefill_storage_cls(compute_gram: bool = False):
    class PostPrePrefillStorage:
        pass

    nwarps = _PREFILL_THREADS // 32
    annotations = {
        "warp_sums": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _PARTIALS * nwarps],
            16,
        ],
    }
    if compute_gram:
        annotations["gram_sums"] = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _GRAM_PAIRS * nwarps],
            16,
        ]
    PostPrePrefillStorage.__annotations__ = annotations
    return cute.struct(PostPrePrefillStorage)


@lru_cache(maxsize=1)
def _post_pre_prefill_gram_storage_cls():
    class PostPrePrefillGramStorage:
        pass

    nwarps = _PREFILL_GRAM_THREADS // 32
    PostPrePrefillGramStorage.__annotations__ = {
        "gram_sums": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _GRAM_PAIRS * nwarps],
            16,
        ],
        "post_coeff": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _MHC_MULT],
            16,
        ],
        "comb_coeff": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _MHC_MULT * _MHC_MULT],
            16,
        ],
    }
    return cute.struct(PostPrePrefillGramStorage)


@lru_cache(maxsize=1)
def _prefill_bf16_mma_storage_cls():
    class PrefillBf16MmaStorage:
        pass

    PrefillBf16MmaStorage.__annotations__ = {
        "a_tile": cute.struct.Align[
            cute.struct.MemRange[
                cutlass.BFloat16,
                _PREFILL_MMA_TILE_M * _PREFILL_MMA_TILE_K,
            ],
            128,
        ],
    }
    return cute.struct(PrefillBf16MmaStorage)


@lru_cache(maxsize=8)
def _post_pre_prefill_block_m_storage_cls(
    block_m: int,
    compute_gram: bool = False,
):
    block_m = int(block_m)
    if block_m <= 0:
        raise ValueError(f"prefill block_m must be positive, got {block_m}")

    class PostPrePrefillBlockMStorage:
        pass

    nwarps = _PREFILL_THREADS // 32
    annotations = {
        "warp_sums": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, block_m * _PARTIALS * nwarps],
            16,
        ],
    }
    if compute_gram:
        annotations["gram_sums"] = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, block_m * _GRAM_PAIRS * nwarps],
            16,
        ]
    PostPrePrefillBlockMStorage.__annotations__ = annotations
    return cute.struct(PostPrePrefillBlockMStorage)


@cute.jit
def _warp_allreduce_sum(value: Float32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=1 << shift))
    return value


@cute.jit
def _warp_quad_allreduce_sum(value: Float32) -> Float32:
    value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=1))
    value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=2))
    return value


@cute.jit
def _warp_quad_allreduce_max(value: Float32) -> Float32:
    peer = Float32(cute.arch.shuffle_sync_bfly(value, offset=1))
    if peer > value:
        value = peer
    peer = Float32(cute.arch.shuffle_sync_bfly(value, offset=2))
    if peer > value:
        value = peer
    return value


@cute.jit
def _warp_column4_allreduce_sum(value: Float32) -> Float32:
    value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=4))
    value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=8))
    return value


class MHCPostPrePartialKernel:
    num_threads = _THREADS
    hidden_size = _HIDDEN
    total_k = _TOTAL_K
    split_k = _SPLIT_K
    source_tile_h = _SOURCE_TILE_H
    source_tiles = _SOURCE_TILES
    source_warps = (_SOURCE_TILES + 31) // 32
    gram_row0 = _GRAM_ROW0
    partials = _PARTIALS
    partials_per_cta = _POST_PRE_PARTIALS_PER_CTA

    def __init__(
        self,
        *,
        hidden_size: int = _HIDDEN,
        split_k: int | None = None,
        compute_gram: bool = False,
        pre_only: bool = False,
        post_only: bool = False,
        partials_per_cta: int = _POST_PRE_PARTIALS_PER_CTA,
    ):
        self.hidden_size = int(hidden_size)
        self.total_k = _MHC_MULT * self.hidden_size
        self.source_tiles = _source_tiles_for_hidden(self.hidden_size)
        self.source_warps = (self.source_tiles + 31) // 32
        self.split_k = (
            _split_k_for_hidden(self.hidden_size) if split_k is None else int(split_k)
        )
        _validate_split_k(self.hidden_size, self.split_k)
        self.gram_row0 = self.source_tiles
        self.partials_per_cta = _validate_post_pre_partials_per_cta(partials_per_cta)
        # When True, the partial_group==0 CTAs also reduce the 4x4 Gram of
        # residual_out into partials rows [32, 64) for the Gram-trick finalize.
        self.compute_gram = bool(compute_gram)
        # When True, this is the standalone pre path: compute fn/residual
        # partials directly from residual and do not materialize post_pre's
        # residual_out side.
        self.pre_only = bool(pre_only)
        # When True, this is the standalone post path: materialize residual_out
        # and skip the fn/residual partial reductions.
        self.post_only = bool(post_only)

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        prev_post: cute.Tensor,
        prev_comb: cute.Tensor,
        fn: cute.Tensor,
        partials: cute.Tensor,
        out: cute.Tensor,
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        if const_expr((not self.pre_only) and x.element_type != cutlass.BFloat16):
            raise TypeError("x must be BFloat16")
        if const_expr(residual.element_type != cutlass.BFloat16):
            raise TypeError("residual must be BFloat16")
        if const_expr(
            (not self.pre_only) and prev_post.element_type != cutlass.Float32
        ):
            raise TypeError("prev_post must be Float32")
        if const_expr(
            (not self.pre_only) and prev_comb.element_type != cutlass.Float32
        ):
            raise TypeError("prev_comb must be Float32")
        if const_expr((not self.post_only) and fn.element_type != cutlass.Float32):
            raise TypeError("fn must be Float32")
        if const_expr(
            (not self.post_only) and partials.element_type != cutlass.Float32
        ):
            raise TypeError("partials must be Float32")
        if const_expr(out.element_type != cutlass.BFloat16):
            raise TypeError("out must be BFloat16")
        partial_groups = (
            1
            if self.post_only
            else (self.partials + self.partials_per_cta - 1) // self.partials_per_cta
        )
        self.kernel(x, residual, prev_post, prev_comb, fn, partials, out).launch(
            grid=(
                self.source_tiles,
                partial_groups,
                num_tokens,
            ),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        prev_post: cute.Tensor,
        prev_comb: cute.Tensor,
        fn: cute.Tensor,
        partials: cute.Tensor,
        out: cute.Tensor,
    ):
        hidden_tile, partial_group, token = cute.arch.block_idx()
        tidx = cute.arch.thread_idx()[0]
        lane = tidx % Int32(32)
        warp = tidx // Int32(32)
        nwarps = self.num_threads // 32
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(
            _post_pre_partial_group_storage_cls(
                self.partials_per_cta,
                self.compute_gram,
            )
        )
        warp_sums = storage.warp_sums.get_tensor(
            cute.make_layout(
                (self.partials_per_cta, self.num_threads // 32),
                stride=(self.num_threads // 32, 1),
            )
        )
        if const_expr(self.compute_gram):
            gram_sums = storage.gram_sums.get_tensor(
                cute.make_layout((_GRAM_PAIRS, nwarps), stride=(nwarps, 1))
            )

        partial0 = partial_group * Int32(self.partials_per_cta)
        h = hidden_tile * Int32(self.source_tile_h) + tidx
        if const_expr(self.pre_only):
            r0 = Float32(residual[token, h])
            r1 = r0
            r2 = r0
            r3 = r0
            if partial_group == Int32(0):
                out[token, Int32(0), h] = r0.to(cutlass.BFloat16)
                out[token, Int32(1), h] = r0.to(cutlass.BFloat16)
                out[token, Int32(2), h] = r0.to(cutlass.BFloat16)
                out[token, Int32(3), h] = r0.to(cutlass.BFloat16)
        else:
            r0 = Float32(residual[token, Int32(0), h])
            r1 = Float32(residual[token, Int32(1), h])
            r2 = Float32(residual[token, Int32(2), h])
            r3 = Float32(residual[token, Int32(3), h])

        if const_expr(not self.pre_only):
            xh = Float32(x[token, h])
            o0 = (
                Float32(prev_post[token, Int32(0)]) * xh
                + Float32(prev_comb[token, Int32(0), Int32(0)]) * r0
                + Float32(prev_comb[token, Int32(1), Int32(0)]) * r1
                + Float32(prev_comb[token, Int32(2), Int32(0)]) * r2
                + Float32(prev_comb[token, Int32(3), Int32(0)]) * r3
            ).to(cutlass.BFloat16)
            o1 = (
                Float32(prev_post[token, Int32(1)]) * xh
                + Float32(prev_comb[token, Int32(0), Int32(1)]) * r0
                + Float32(prev_comb[token, Int32(1), Int32(1)]) * r1
                + Float32(prev_comb[token, Int32(2), Int32(1)]) * r2
                + Float32(prev_comb[token, Int32(3), Int32(1)]) * r3
            ).to(cutlass.BFloat16)
            o2 = (
                Float32(prev_post[token, Int32(2)]) * xh
                + Float32(prev_comb[token, Int32(0), Int32(2)]) * r0
                + Float32(prev_comb[token, Int32(1), Int32(2)]) * r1
                + Float32(prev_comb[token, Int32(2), Int32(2)]) * r2
                + Float32(prev_comb[token, Int32(3), Int32(2)]) * r3
            ).to(cutlass.BFloat16)
            o3 = (
                Float32(prev_post[token, Int32(3)]) * xh
                + Float32(prev_comb[token, Int32(0), Int32(3)]) * r0
                + Float32(prev_comb[token, Int32(1), Int32(3)]) * r1
                + Float32(prev_comb[token, Int32(2), Int32(3)]) * r2
                + Float32(prev_comb[token, Int32(3), Int32(3)]) * r3
            ).to(cutlass.BFloat16)

            if partial_group == Int32(0):
                out[token, Int32(0), h] = o0
                out[token, Int32(1), h] = o1
                out[token, Int32(2), h] = o2
                out[token, Int32(3), h] = o3

            r0 = Float32(o0)
            r1 = Float32(o1)
            r2 = Float32(o2)
            r3 = Float32(o3)

        if const_expr(not self.post_only):
            # Optional 4x4 Gram of residual_out (only the residual_out-owning group).
            if const_expr(self.compute_gram):
                if partial_group == Int32(0):
                    gvals = cute.make_rmem_tensor(
                        cute.make_layout((_GRAM_PAIRS,), stride=(1,)),
                        Float32,
                    )
                    gvals[0] = r0 * r0
                    gvals[1] = r1 * r1
                    gvals[2] = r2 * r2
                    gvals[3] = r3 * r3
                    gvals[4] = r0 * r1
                    gvals[5] = r0 * r2
                    gvals[6] = r0 * r3
                    gvals[7] = r1 * r2
                    gvals[8] = r1 * r3
                    gvals[9] = r2 * r3
                    for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                        gvals[gp] = _warp_allreduce_sum(gvals[gp])
                    if lane == Int32(0):
                        for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                            gram_sums[gp, warp] = gvals[gp]

            values = cute.make_rmem_tensor(
                cute.make_layout((self.partials_per_cta,), stride=(1,)),
                Float32,
            )
            for slot in cutlass.range_constexpr(self.partials_per_cta):
                partial = partial0 + Int32(slot)
                value = Float32(0.0)
                if partial == Int32(0):
                    value = r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3
                elif const_expr(self.pre_only):
                    if partial < Int32(self.partials):
                        mix = partial - Int32(1)
                        value = Float32(fn[mix, h]) * r0
                elif partial < Int32(self.partials):
                    mix = partial - Int32(1)
                    value = (
                        Float32(fn[mix, h]) * r0
                        + Float32(fn[mix, Int32(self.hidden_size) + h]) * r1
                        + Float32(fn[mix, Int32(2 * self.hidden_size) + h]) * r2
                        + Float32(fn[mix, Int32(3 * self.hidden_size) + h]) * r3
                    )
                values[slot] = _warp_allreduce_sum(value)
            if lane == Int32(0):
                for slot in cutlass.range_constexpr(self.partials_per_cta):
                    warp_sums[slot, warp] = values[slot]
            cute.arch.sync_threads()

            if tidx == Int32(0):
                for slot in cutlass.range_constexpr(self.partials_per_cta):
                    total = Float32(0.0)
                    src_warp = Int32(0)
                    while src_warp < Int32(self.num_threads // 32):
                        total += Float32(warp_sums[slot, src_warp])
                        src_warp += Int32(1)
                    partial = partial0 + Int32(slot)
                    if partial < Int32(self.partials):
                        partials[token, hidden_tile, partial] = total

            if const_expr(self.compute_gram):
                if partial_group == Int32(0):
                    if tidx == Int32(0):
                        for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                            gtotal = Float32(0.0)
                            src_warp = Int32(0)
                            while src_warp < Int32(nwarps):
                                gtotal += Float32(gram_sums[gp, src_warp])
                                src_warp += Int32(1)
                            partials[token, Int32(self.gram_row0) + hidden_tile, gp] = (
                                gtotal
                            )

        if const_expr(_MHC_PDL):
            cute.arch.sync_threads()
            cute.arch.griddepcontrol_launch_dependents()


class MHCPostPreDecodeSplitNPartialKernel:
    """Fused decode partial using split hidden slices and narrow output tiles."""

    num_threads = 256

    def __init__(
        self,
        *,
        hidden_size: int = _HIDDEN,
        split_k: int | None = None,
        source_splits: int = 4,
        tile_n: int = 3,
        bf16x2: bool = False,
        compute_gram: bool = False,
    ):
        self.hidden_size = int(hidden_size)
        self.source_tiles = _source_tiles_for_hidden(self.hidden_size)
        self.split_k = (
            _split_k_for_hidden(self.hidden_size) if split_k is None else int(split_k)
        )
        _validate_split_k(self.hidden_size, self.split_k)
        self.source_splits = int(source_splits)
        if (
            self.source_splits <= 0
            or self.source_splits > self.source_tiles
            or self.hidden_size % self.source_splits != 0
        ):
            raise ValueError(f"invalid decode source_splits={self.source_splits}")
        self.hidden_per_split = self.hidden_size // self.source_splits
        if self.hidden_per_split % self.num_threads != 0:
            raise ValueError(
                f"hidden_size/source_splits={self.hidden_per_split} must be "
                f"divisible by {self.num_threads} threads"
            )
        self.hidden_iters = self.hidden_per_split // self.num_threads
        self.tile_n = int(tile_n)
        if self.tile_n <= 0 or _MIXES % self.tile_n != 0:
            raise ValueError(f"decode tile_n must divide {_MIXES}, got {self.tile_n}")
        self.n_tiles = _MIXES // self.tile_n
        self.bf16x2 = bool(bf16x2)
        if self.bf16x2 and self.hidden_per_split % (2 * self.num_threads) != 0:
            raise ValueError(
                f"hidden_size/source_splits={self.hidden_per_split} must be "
                f"divisible by {2 * self.num_threads} for bf16x2"
            )
        self.hidden_pair_iters = self.hidden_per_split // (2 * self.num_threads)
        self.compute_gram = bool(compute_gram)

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        prev_post: cute.Tensor,
        prev_comb: cute.Tensor,
        fn: cute.Tensor,
        partials: cute.Tensor,
        out: cute.Tensor,
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        if const_expr(x.element_type != cutlass.BFloat16):
            raise TypeError("x must be BFloat16")
        if const_expr(residual.element_type != cutlass.BFloat16):
            raise TypeError("residual must be BFloat16")
        if const_expr(prev_post.element_type != cutlass.Float32):
            raise TypeError("prev_post must be Float32")
        if const_expr(prev_comb.element_type != cutlass.Float32):
            raise TypeError("prev_comb must be Float32")
        if const_expr(fn.element_type != cutlass.Float32):
            raise TypeError("fn must be Float32")
        if const_expr(partials.element_type != cutlass.Float32):
            raise TypeError("partials must be Float32")
        if const_expr(out.element_type != cutlass.BFloat16):
            raise TypeError("out must be BFloat16")
        self.kernel(x, residual, prev_post, prev_comb, fn, partials, out).launch(
            grid=(num_tokens, self.n_tiles, self.source_splits),
            block=[self.num_threads, 1, 1],
            min_blocks_per_mp=3,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        prev_post: cute.Tensor,
        prev_comb: cute.Tensor,
        fn: cute.Tensor,
        partials: cute.Tensor,
        out: cute.Tensor,
    ):
        token, n_tile, source_split = cute.arch.block_idx()
        tidx = Int32(cute.arch.thread_idx()[0])
        lane = tidx % Int32(32)
        warp_idx = tidx // Int32(32)
        nwarps = self.num_threads // 32
        mix0 = n_tile * Int32(self.tile_n)

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(
            _post_pre_decode_split_n_storage_cls(
                self.tile_n,
                self.compute_gram,
            )
        )
        warp_sums = storage.warp_sums.get_tensor(
            cute.make_layout((self.tile_n + 1, nwarps), stride=(nwarps, 1))
        )
        s_post = storage.post.get_tensor(cute.make_layout((_MHC_MULT,), stride=(1,)))
        s_comb = storage.comb.get_tensor(
            cute.make_layout((_MHC_MULT, _MHC_MULT), stride=(_MHC_MULT, 1))
        )
        if const_expr(self.compute_gram):
            gram_sums = storage.gram_sums.get_tensor(
                cute.make_layout((_GRAM_PAIRS, nwarps), stride=(nwarps, 1))
            )

        if tidx < Int32(_MHC_MULT):
            s_post[tidx] = Float32(prev_post[token, tidx])
        if tidx < Int32(_MHC_MULT * _MHC_MULT):
            row = tidx // Int32(_MHC_MULT)
            col = tidx - row * Int32(_MHC_MULT)
            s_comb[row, col] = Float32(prev_comb[token, row, col])

        cute.arch.sync_threads()

        pm = cute.make_rmem_tensor(cute.make_layout((_MHC_MULT,), stride=(1,)), Float32)
        cm = cute.make_rmem_tensor(
            cute.make_layout((_MHC_MULT, _MHC_MULT), stride=(_MHC_MULT, 1)),
            Float32,
        )
        for j in cutlass.range_constexpr(_MHC_MULT):
            pm[j] = Float32(s_post[j])
            for k in cutlass.range_constexpr(_MHC_MULT):
                cm[k, j] = Float32(s_comb[k, j])

        acc = cute.make_rmem_tensor(
            cute.make_layout((self.tile_n,), stride=(1,)), Float32
        )
        for ni in cutlass.range_constexpr(self.tile_n):
            acc[ni] = Float32(0.0)
        sqr = Float32(0.0)
        if const_expr(self.compute_gram):
            gvals = cute.make_rmem_tensor(
                cute.make_layout((_GRAM_PAIRS,), stride=(1,)), Float32
            )
            for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                gvals[gp] = Float32(0.0)

        h_split0 = source_split * Int32(self.hidden_per_split)
        if const_expr(self.bf16x2):
            out_u32 = cute.recast_tensor(out, Uint32)
            for hidden_pair_iter in cutlass.range_constexpr(self.hidden_pair_iters):
                h = (
                    h_split0
                    + Int32(2 * hidden_pair_iter * self.num_threads)
                    + tidx * Int32(2)
                )
                x_pair = ld_global_nc_u32(
                    get_ptr_as_int64(
                        x,
                        token * Int32(self.hidden_size) + h,
                    )
                )
                rin0_pair = ld_global_nc_u32(
                    get_ptr_as_int64(
                        residual,
                        token * Int32(_MHC_MULT * self.hidden_size) + h,
                    )
                )
                rin1_pair = ld_global_nc_u32(
                    get_ptr_as_int64(
                        residual,
                        token * Int32(_MHC_MULT * self.hidden_size)
                        + Int32(self.hidden_size)
                        + h,
                    )
                )
                rin2_pair = ld_global_nc_u32(
                    get_ptr_as_int64(
                        residual,
                        token * Int32(_MHC_MULT * self.hidden_size)
                        + Int32(2 * self.hidden_size)
                        + h,
                    )
                )
                rin3_pair = ld_global_nc_u32(
                    get_ptr_as_int64(
                        residual,
                        token * Int32(_MHC_MULT * self.hidden_size)
                        + Int32(3 * self.hidden_size)
                        + h,
                    )
                )
                x0, x1 = bfloat2_to_float2_scaled(x_pair, Float32(1.0))
                rin00, rin01 = bfloat2_to_float2_scaled(rin0_pair, Float32(1.0))
                rin10, rin11 = bfloat2_to_float2_scaled(rin1_pair, Float32(1.0))
                rin20, rin21 = bfloat2_to_float2_scaled(rin2_pair, Float32(1.0))
                rin30, rin31 = bfloat2_to_float2_scaled(rin3_pair, Float32(1.0))

                o_values = cute.make_rmem_tensor(
                    cute.make_layout((_MHC_MULT, 2), stride=(2, 1)),
                    Float32,
                )
                for pair_lane in cutlass.range_constexpr(2):
                    xh = x0
                    rin0 = rin00
                    rin1 = rin10
                    rin2 = rin20
                    rin3 = rin30
                    if const_expr(pair_lane == 1):
                        xh = x1
                        rin0 = rin01
                        rin1 = rin11
                        rin2 = rin21
                        rin3 = rin31
                    o_values[0, pair_lane] = (
                        pm[0] * xh
                        + cm[0, 0] * rin0
                        + cm[1, 0] * rin1
                        + cm[2, 0] * rin2
                        + cm[3, 0] * rin3
                    )
                    o_values[1, pair_lane] = (
                        pm[1] * xh
                        + cm[0, 1] * rin0
                        + cm[1, 1] * rin1
                        + cm[2, 1] * rin2
                        + cm[3, 1] * rin3
                    )
                    o_values[2, pair_lane] = (
                        pm[2] * xh
                        + cm[0, 2] * rin0
                        + cm[1, 2] * rin1
                        + cm[2, 2] * rin2
                        + cm[3, 2] * rin3
                    )
                    o_values[3, pair_lane] = (
                        pm[3] * xh
                        + cm[0, 3] * rin0
                        + cm[1, 3] * rin1
                        + cm[2, 3] * rin2
                        + cm[3, 3] * rin3
                    )

                o0_pair = pack_f32x2_to_bfloat2(o_values[0, 0], o_values[0, 1])
                o1_pair = pack_f32x2_to_bfloat2(o_values[1, 0], o_values[1, 1])
                o2_pair = pack_f32x2_to_bfloat2(o_values[2, 0], o_values[2, 1])
                o3_pair = pack_f32x2_to_bfloat2(o_values[3, 0], o_values[3, 1])
                r00, r01 = bfloat2_to_float2_scaled(o0_pair, Float32(1.0))
                r10, r11 = bfloat2_to_float2_scaled(o1_pair, Float32(1.0))
                r20, r21 = bfloat2_to_float2_scaled(o2_pair, Float32(1.0))
                r30, r31 = bfloat2_to_float2_scaled(o3_pair, Float32(1.0))

                if n_tile == Int32(0):
                    out_h = h // Int32(2)
                    out_u32[token, Int32(0), out_h] = o0_pair
                    out_u32[token, Int32(1), out_h] = o1_pair
                    out_u32[token, Int32(2), out_h] = o2_pair
                    out_u32[token, Int32(3), out_h] = o3_pair
                    sqr += r00 * r00 + r10 * r10 + r20 * r20 + r30 * r30
                    sqr += r01 * r01 + r11 * r11 + r21 * r21 + r31 * r31
                    if const_expr(self.compute_gram):
                        gvals[0] += r00 * r00 + r01 * r01
                        gvals[1] += r10 * r10 + r11 * r11
                        gvals[2] += r20 * r20 + r21 * r21
                        gvals[3] += r30 * r30 + r31 * r31
                        gvals[4] += r00 * r10 + r01 * r11
                        gvals[5] += r00 * r20 + r01 * r21
                        gvals[6] += r00 * r30 + r01 * r31
                        gvals[7] += r10 * r20 + r11 * r21
                        gvals[8] += r10 * r30 + r11 * r31
                        gvals[9] += r20 * r30 + r21 * r31

                for pair_lane in cutlass.range_constexpr(2):
                    hp = h + Int32(pair_lane)
                    r0 = r00
                    r1 = r10
                    r2 = r20
                    r3 = r30
                    if const_expr(pair_lane == 1):
                        r0 = r01
                        r1 = r11
                        r2 = r21
                        r3 = r31
                    for ni in cutlass.range_constexpr(self.tile_n):
                        mix = mix0 + Int32(ni)
                        acc[ni] += (
                            Float32(fn[mix, hp]) * r0
                            + Float32(fn[mix, Int32(self.hidden_size) + hp]) * r1
                            + Float32(fn[mix, Int32(2 * self.hidden_size) + hp]) * r2
                            + Float32(fn[mix, Int32(3 * self.hidden_size) + hp]) * r3
                        )
        else:
            for hidden_iter in cutlass.range_constexpr(self.hidden_iters):
                h = h_split0 + Int32(hidden_iter * self.num_threads) + tidx
                xh = Float32(x[token, h])
                rin0 = Float32(residual[token, Int32(0), h])
                rin1 = Float32(residual[token, Int32(1), h])
                rin2 = Float32(residual[token, Int32(2), h])
                rin3 = Float32(residual[token, Int32(3), h])
                o0 = (
                    pm[0] * xh
                    + cm[0, 0] * rin0
                    + cm[1, 0] * rin1
                    + cm[2, 0] * rin2
                    + cm[3, 0] * rin3
                ).to(cutlass.BFloat16)
                o1 = (
                    pm[1] * xh
                    + cm[0, 1] * rin0
                    + cm[1, 1] * rin1
                    + cm[2, 1] * rin2
                    + cm[3, 1] * rin3
                ).to(cutlass.BFloat16)
                o2 = (
                    pm[2] * xh
                    + cm[0, 2] * rin0
                    + cm[1, 2] * rin1
                    + cm[2, 2] * rin2
                    + cm[3, 2] * rin3
                ).to(cutlass.BFloat16)
                o3 = (
                    pm[3] * xh
                    + cm[0, 3] * rin0
                    + cm[1, 3] * rin1
                    + cm[2, 3] * rin2
                    + cm[3, 3] * rin3
                ).to(cutlass.BFloat16)
                r0 = Float32(o0)
                r1 = Float32(o1)
                r2 = Float32(o2)
                r3 = Float32(o3)

                if n_tile == Int32(0):
                    out[token, Int32(0), h] = o0
                    out[token, Int32(1), h] = o1
                    out[token, Int32(2), h] = o2
                    out[token, Int32(3), h] = o3
                    sqr += r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3
                    if const_expr(self.compute_gram):
                        gvals[0] += r0 * r0
                        gvals[1] += r1 * r1
                        gvals[2] += r2 * r2
                        gvals[3] += r3 * r3
                        gvals[4] += r0 * r1
                        gvals[5] += r0 * r2
                        gvals[6] += r0 * r3
                        gvals[7] += r1 * r2
                        gvals[8] += r1 * r3
                        gvals[9] += r2 * r3

                for ni in cutlass.range_constexpr(self.tile_n):
                    mix = mix0 + Int32(ni)
                    acc[ni] += (
                        Float32(fn[mix, h]) * r0
                        + Float32(fn[mix, Int32(self.hidden_size) + h]) * r1
                        + Float32(fn[mix, Int32(2 * self.hidden_size) + h]) * r2
                        + Float32(fn[mix, Int32(3 * self.hidden_size) + h]) * r3
                    )

        for ni in cutlass.range_constexpr(self.tile_n):
            acc[ni] = _warp_allreduce_sum(acc[ni])
        if n_tile == Int32(0):
            sqr = _warp_allreduce_sum(sqr)
            if const_expr(self.compute_gram):
                for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                    gvals[gp] = _warp_allreduce_sum(gvals[gp])

        if lane == Int32(0):
            for ni in cutlass.range_constexpr(self.tile_n):
                warp_sums[ni, warp_idx] = acc[ni]
            if n_tile == Int32(0):
                warp_sums[self.tile_n, warp_idx] = sqr
                if const_expr(self.compute_gram):
                    for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                        gram_sums[gp, warp_idx] = gvals[gp]
        cute.arch.sync_threads()

        if warp_idx == Int32(0):
            if lane < Int32(self.tile_n):
                total = Float32(0.0)
                src_warp = Int32(0)
                while src_warp < Int32(nwarps):
                    total += Float32(warp_sums[lane, src_warp])
                    src_warp += Int32(1)
                partials[token, source_split, mix0 + lane + Int32(1)] = total
            if n_tile == Int32(0):
                if lane == Int32(0):
                    total = Float32(0.0)
                    src_warp = Int32(0)
                    while src_warp < Int32(nwarps):
                        total += Float32(warp_sums[self.tile_n, src_warp])
                        src_warp += Int32(1)
                    partials[token, source_split, Int32(0)] = total
                if const_expr(self.compute_gram):
                    if lane < Int32(_GRAM_PAIRS):
                        total = Float32(0.0)
                        src_warp = Int32(0)
                        while src_warp < Int32(nwarps):
                            total += Float32(gram_sums[lane, src_warp])
                            src_warp += Int32(1)
                        partials[
                            token,
                            Int32(self.source_tiles) + source_split,
                            lane,
                        ] = total

        if const_expr(_MHC_PDL):
            cute.arch.sync_threads()
            cute.arch.griddepcontrol_launch_dependents()


class MHCPostPrePrefillPartialKernel:
    """Full-hidden post_pre partial kernel for large prefill shapes.

    One CTA owns one token, loops over the full hidden dimension, and writes a
    compact partial layout:
      partials[token, 0, 0:25] = scalar partials
      partials[token, 1, 0:10] = residual_out Gram entries
    """

    num_threads = _PREFILL_THREADS
    partials = _PARTIALS
    mixes = _MIXES
    gram_pairs = _GRAM_PAIRS

    def __init__(
        self,
        *,
        hidden_size: int = _HIDDEN,
        split_k: int | None = None,
        compute_gram: bool = True,
    ):
        self.hidden_size = int(hidden_size)
        self.total_k = _MHC_MULT * self.hidden_size
        self.split_k = (
            _split_k_for_hidden(self.hidden_size) if split_k is None else int(split_k)
        )
        _validate_split_k(self.hidden_size, self.split_k)
        if self.hidden_size % self.num_threads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by "
                f"prefill threads={self.num_threads}"
            )
        self.hidden_iters = self.hidden_size // self.num_threads
        self.compute_gram = bool(compute_gram)

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        prev_post: cute.Tensor,
        prev_comb: cute.Tensor,
        fn: cute.Tensor,
        partials: cute.Tensor,
        out: cute.Tensor,
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        if const_expr(x.element_type != cutlass.BFloat16):
            raise TypeError("x must be BFloat16")
        if const_expr(residual.element_type != cutlass.BFloat16):
            raise TypeError("residual must be BFloat16")
        if const_expr(prev_post.element_type != cutlass.Float32):
            raise TypeError("prev_post must be Float32")
        if const_expr(prev_comb.element_type != cutlass.Float32):
            raise TypeError("prev_comb must be Float32")
        if const_expr(fn.element_type != cutlass.Float32):
            raise TypeError("fn must be Float32")
        if const_expr(partials.element_type != cutlass.Float32):
            raise TypeError("partials must be Float32")
        if const_expr(out.element_type != cutlass.BFloat16):
            raise TypeError("out must be BFloat16")
        self.kernel(x, residual, prev_post, prev_comb, fn, partials, out).launch(
            grid=(num_tokens, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        prev_post: cute.Tensor,
        prev_comb: cute.Tensor,
        fn: cute.Tensor,
        partials: cute.Tensor,
        out: cute.Tensor,
    ):
        token, _, _ = cute.arch.block_idx()
        tidx = Int32(cute.arch.thread_idx()[0])
        lane = tidx % Int32(32)
        warp = tidx // Int32(32)
        nwarps = self.num_threads // 32
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_post_pre_prefill_storage_cls(self.compute_gram))
        warp_sums = storage.warp_sums.get_tensor(
            cute.make_layout((_PARTIALS, nwarps), stride=(nwarps, 1))
        )
        if const_expr(self.compute_gram):
            gram_sums = storage.gram_sums.get_tensor(
                cute.make_layout((_GRAM_PAIRS, nwarps), stride=(nwarps, 1))
            )

        values = cute.make_rmem_tensor(
            cute.make_layout((_PARTIALS,), stride=(1,)),
            Float32,
        )
        for slot in cutlass.range_constexpr(_PARTIALS):
            values[slot] = Float32(0.0)

        if const_expr(self.compute_gram):
            gvals = cute.make_rmem_tensor(
                cute.make_layout((_GRAM_PAIRS,), stride=(1,)),
                Float32,
            )
            for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                gvals[gp] = Float32(0.0)

        for hidden_iter in cutlass.range_constexpr(self.hidden_iters):
            h = Int32(hidden_iter * self.num_threads) + tidx
            xh = Float32(x[token, h])
            rin0 = Float32(residual[token, Int32(0), h])
            rin1 = Float32(residual[token, Int32(1), h])
            rin2 = Float32(residual[token, Int32(2), h])
            rin3 = Float32(residual[token, Int32(3), h])
            o0 = (
                Float32(prev_post[token, Int32(0)]) * xh
                + Float32(prev_comb[token, Int32(0), Int32(0)]) * rin0
                + Float32(prev_comb[token, Int32(1), Int32(0)]) * rin1
                + Float32(prev_comb[token, Int32(2), Int32(0)]) * rin2
                + Float32(prev_comb[token, Int32(3), Int32(0)]) * rin3
            ).to(cutlass.BFloat16)
            o1 = (
                Float32(prev_post[token, Int32(1)]) * xh
                + Float32(prev_comb[token, Int32(0), Int32(1)]) * rin0
                + Float32(prev_comb[token, Int32(1), Int32(1)]) * rin1
                + Float32(prev_comb[token, Int32(2), Int32(1)]) * rin2
                + Float32(prev_comb[token, Int32(3), Int32(1)]) * rin3
            ).to(cutlass.BFloat16)
            o2 = (
                Float32(prev_post[token, Int32(2)]) * xh
                + Float32(prev_comb[token, Int32(0), Int32(2)]) * rin0
                + Float32(prev_comb[token, Int32(1), Int32(2)]) * rin1
                + Float32(prev_comb[token, Int32(2), Int32(2)]) * rin2
                + Float32(prev_comb[token, Int32(3), Int32(2)]) * rin3
            ).to(cutlass.BFloat16)
            o3 = (
                Float32(prev_post[token, Int32(3)]) * xh
                + Float32(prev_comb[token, Int32(0), Int32(3)]) * rin0
                + Float32(prev_comb[token, Int32(1), Int32(3)]) * rin1
                + Float32(prev_comb[token, Int32(2), Int32(3)]) * rin2
                + Float32(prev_comb[token, Int32(3), Int32(3)]) * rin3
            ).to(cutlass.BFloat16)
            out[token, Int32(0), h] = o0
            out[token, Int32(1), h] = o1
            out[token, Int32(2), h] = o2
            out[token, Int32(3), h] = o3

            r0 = Float32(o0)
            r1 = Float32(o1)
            r2 = Float32(o2)
            r3 = Float32(o3)
            values[0] += r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3
            for mix in cutlass.range_constexpr(_MIXES):
                values[mix + 1] += (
                    Float32(fn[mix, h]) * r0
                    + Float32(fn[mix, Int32(self.hidden_size) + h]) * r1
                    + Float32(fn[mix, Int32(2 * self.hidden_size) + h]) * r2
                    + Float32(fn[mix, Int32(3 * self.hidden_size) + h]) * r3
                )

            if const_expr(self.compute_gram):
                gvals[0] += r0 * r0
                gvals[1] += r1 * r1
                gvals[2] += r2 * r2
                gvals[3] += r3 * r3
                gvals[4] += r0 * r1
                gvals[5] += r0 * r2
                gvals[6] += r0 * r3
                gvals[7] += r1 * r2
                gvals[8] += r1 * r3
                gvals[9] += r2 * r3

        for slot in cutlass.range_constexpr(_PARTIALS):
            values[slot] = _warp_allreduce_sum(values[slot])
        if const_expr(self.compute_gram):
            for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                gvals[gp] = _warp_allreduce_sum(gvals[gp])

        if lane == Int32(0):
            for slot in cutlass.range_constexpr(_PARTIALS):
                warp_sums[slot, warp] = values[slot]
            if const_expr(self.compute_gram):
                for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                    gram_sums[gp, warp] = gvals[gp]
        cute.arch.sync_threads()

        if tidx == Int32(0):
            for slot in cutlass.range_constexpr(_PARTIALS):
                total = Float32(0.0)
                src_warp = Int32(0)
                while src_warp < Int32(nwarps):
                    total += Float32(warp_sums[slot, src_warp])
                    src_warp += Int32(1)
                partials[token, Int32(0), slot] = total

            if const_expr(self.compute_gram):
                for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                    gtotal = Float32(0.0)
                    src_warp = Int32(0)
                    while src_warp < Int32(nwarps):
                        gtotal += Float32(gram_sums[gp, src_warp])
                        src_warp += Int32(1)
                    partials[token, Int32(1), gp] = gtotal


class MHCPostPrePrefillBlockMPartialKernel:
    """Block-M scalar prefill kernel for large post_pre shapes.

    One CTA owns two tokens and twelve projection rows. Compared with the
    compact one-token CTA, this shares fn loads across two tokens while still
    emitting compact partials for the existing Gram/RMSNorm finalize.
    """

    num_threads = _PREFILL_THREADS
    block_m = _PREFILL_BLOCK_M
    tile_n = _PREFILL_BLOCK_TILE_N
    partials = _PARTIALS
    mixes = _MIXES
    gram_pairs = _GRAM_PAIRS

    def __init__(
        self,
        *,
        hidden_size: int = _HIDDEN,
        split_k: int | None = None,
        block_m: int = _PREFILL_BLOCK_M,
        tile_n: int = _PREFILL_BLOCK_TILE_N,
        compute_gram: bool = True,
    ):
        self.hidden_size = int(hidden_size)
        self.total_k = _MHC_MULT * self.hidden_size
        self.split_k = (
            _split_k_for_hidden(self.hidden_size) if split_k is None else int(split_k)
        )
        _validate_split_k(self.hidden_size, self.split_k)
        if self.hidden_size % self.num_threads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by "
                f"prefill threads={self.num_threads}"
            )
        self.hidden_iters = self.hidden_size // self.num_threads
        self.block_m = int(block_m)
        if self.block_m <= 0:
            raise ValueError(f"block_m must be positive, got {self.block_m}")
        self.tile_n = int(tile_n)
        if self.tile_n <= 0:
            raise ValueError(f"tile_n must be positive, got {self.tile_n}")
        self.n_tiles = (self.mixes + self.tile_n - 1) // self.tile_n
        self.compute_gram = bool(compute_gram)

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        prev_post: cute.Tensor,
        prev_comb: cute.Tensor,
        fn: cute.Tensor,
        partials: cute.Tensor,
        out: cute.Tensor,
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        if const_expr(x.element_type != cutlass.BFloat16):
            raise TypeError("x must be BFloat16")
        if const_expr(residual.element_type != cutlass.BFloat16):
            raise TypeError("residual must be BFloat16")
        if const_expr(prev_post.element_type != cutlass.Float32):
            raise TypeError("prev_post must be Float32")
        if const_expr(prev_comb.element_type != cutlass.Float32):
            raise TypeError("prev_comb must be Float32")
        if const_expr(fn.element_type != cutlass.Float32):
            raise TypeError("fn must be Float32")
        if const_expr(partials.element_type != cutlass.Float32):
            raise TypeError("partials must be Float32")
        if const_expr(out.element_type != cutlass.BFloat16):
            raise TypeError("out must be BFloat16")
        m_tiles = (num_tokens + Int32(self.block_m - 1)) // Int32(self.block_m)
        self.kernel(
            x, residual, prev_post, prev_comb, fn, partials, out, num_tokens
        ).launch(
            grid=(m_tiles, self.n_tiles, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        prev_post: cute.Tensor,
        prev_comb: cute.Tensor,
        fn: cute.Tensor,
        partials: cute.Tensor,
        out: cute.Tensor,
        num_tokens: Int32,
    ):
        m_tile, n_tile, _ = cute.arch.block_idx()
        tidx = Int32(cute.arch.thread_idx()[0])
        lane = tidx % Int32(32)
        warp = tidx // Int32(32)
        nwarps = self.num_threads // 32
        mix0 = Int32(n_tile * self.tile_n)

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(
            _post_pre_prefill_block_m_storage_cls(self.block_m, self.compute_gram)
        )
        warp_sums = storage.warp_sums.get_tensor(
            cute.make_layout(
                (self.block_m, _PARTIALS, nwarps),
                stride=(_PARTIALS * nwarps, nwarps, 1),
            )
        )
        if const_expr(self.compute_gram):
            gram_sums = storage.gram_sums.get_tensor(
                cute.make_layout(
                    (self.block_m, _GRAM_PAIRS, nwarps),
                    stride=(_GRAM_PAIRS * nwarps, nwarps, 1),
                )
            )

        acc = cute.make_rmem_tensor(
            cute.make_layout((self.block_m, self.tile_n), stride=(self.tile_n, 1)),
            Float32,
        )
        sqr = cute.make_rmem_tensor(
            cute.make_layout((self.block_m,), stride=(1,)),
            Float32,
        )
        oval = cute.make_rmem_tensor(
            cute.make_layout((self.block_m, _MHC_MULT), stride=(_MHC_MULT, 1)),
            Float32,
        )
        for mi in cutlass.range_constexpr(self.block_m):
            sqr[mi] = Float32(0.0)
            for c in cutlass.range_constexpr(_MHC_MULT):
                oval[mi, c] = Float32(0.0)
            for ni in cutlass.range_constexpr(self.tile_n):
                acc[mi, ni] = Float32(0.0)

        if const_expr(self.compute_gram):
            gvals = cute.make_rmem_tensor(
                cute.make_layout((self.block_m, _GRAM_PAIRS), stride=(_GRAM_PAIRS, 1)),
                Float32,
            )
            for mi in cutlass.range_constexpr(self.block_m):
                for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                    gvals[mi, gp] = Float32(0.0)

        for hidden_iter in cutlass.range_constexpr(self.hidden_iters):
            h = Int32(hidden_iter * self.num_threads) + tidx

            for mi in cutlass.range_constexpr(self.block_m):
                token = Int32(m_tile * self.block_m + mi)
                oval[mi, 0] = Float32(0.0)
                oval[mi, 1] = Float32(0.0)
                oval[mi, 2] = Float32(0.0)
                oval[mi, 3] = Float32(0.0)
                if token < num_tokens:
                    xh = Float32(x[token, h])
                    rin0 = Float32(residual[token, Int32(0), h])
                    rin1 = Float32(residual[token, Int32(1), h])
                    rin2 = Float32(residual[token, Int32(2), h])
                    rin3 = Float32(residual[token, Int32(3), h])
                    o0 = (
                        Float32(prev_post[token, Int32(0)]) * xh
                        + Float32(prev_comb[token, Int32(0), Int32(0)]) * rin0
                        + Float32(prev_comb[token, Int32(1), Int32(0)]) * rin1
                        + Float32(prev_comb[token, Int32(2), Int32(0)]) * rin2
                        + Float32(prev_comb[token, Int32(3), Int32(0)]) * rin3
                    ).to(cutlass.BFloat16)
                    o1 = (
                        Float32(prev_post[token, Int32(1)]) * xh
                        + Float32(prev_comb[token, Int32(0), Int32(1)]) * rin0
                        + Float32(prev_comb[token, Int32(1), Int32(1)]) * rin1
                        + Float32(prev_comb[token, Int32(2), Int32(1)]) * rin2
                        + Float32(prev_comb[token, Int32(3), Int32(1)]) * rin3
                    ).to(cutlass.BFloat16)
                    o2 = (
                        Float32(prev_post[token, Int32(2)]) * xh
                        + Float32(prev_comb[token, Int32(0), Int32(2)]) * rin0
                        + Float32(prev_comb[token, Int32(1), Int32(2)]) * rin1
                        + Float32(prev_comb[token, Int32(2), Int32(2)]) * rin2
                        + Float32(prev_comb[token, Int32(3), Int32(2)]) * rin3
                    ).to(cutlass.BFloat16)
                    o3 = (
                        Float32(prev_post[token, Int32(3)]) * xh
                        + Float32(prev_comb[token, Int32(0), Int32(3)]) * rin0
                        + Float32(prev_comb[token, Int32(1), Int32(3)]) * rin1
                        + Float32(prev_comb[token, Int32(2), Int32(3)]) * rin2
                        + Float32(prev_comb[token, Int32(3), Int32(3)]) * rin3
                    ).to(cutlass.BFloat16)

                    r0 = Float32(o0)
                    r1 = Float32(o1)
                    r2 = Float32(o2)
                    r3 = Float32(o3)
                    oval[mi, 0] = r0
                    oval[mi, 1] = r1
                    oval[mi, 2] = r2
                    oval[mi, 3] = r3

                    if n_tile == Int32(0):
                        out[token, Int32(0), h] = o0
                        out[token, Int32(1), h] = o1
                        out[token, Int32(2), h] = o2
                        out[token, Int32(3), h] = o3
                        g0 = _materialize_residual_gram_f32(r0)
                        g1 = _materialize_residual_gram_f32(r1)
                        g2 = _materialize_residual_gram_f32(r2)
                        g3 = _materialize_residual_gram_f32(r3)
                        if const_expr(self.compute_gram):
                            gvals[mi, 0] += g0 * g0
                            gvals[mi, 1] += g1 * g1
                            gvals[mi, 2] += g2 * g2
                            gvals[mi, 3] += g3 * g3
                            gvals[mi, 4] += g0 * g1
                            gvals[mi, 5] += g0 * g2
                            gvals[mi, 6] += g0 * g3
                            gvals[mi, 7] += g1 * g2
                            gvals[mi, 8] += g1 * g3
                            gvals[mi, 9] += g2 * g3
                        else:
                            sqr[mi] += g0 * g0 + g1 * g1 + g2 * g2 + g3 * g3

            for ni in cutlass.range_constexpr(self.tile_n):
                mix = mix0 + Int32(ni)
                f0 = Float32(0.0)
                f1 = Float32(0.0)
                f2 = Float32(0.0)
                f3 = Float32(0.0)
                if mix < Int32(_MIXES):
                    f0 = Float32(fn[mix, h])
                    f1 = Float32(fn[mix, Int32(self.hidden_size) + h])
                    f2 = Float32(fn[mix, Int32(2 * self.hidden_size) + h])
                    f3 = Float32(fn[mix, Int32(3 * self.hidden_size) + h])
                for mi in cutlass.range_constexpr(self.block_m):
                    token = Int32(m_tile * self.block_m + mi)
                    if token < num_tokens:
                        acc[mi, ni] += (
                            f0 * oval[mi, 0]
                            + f1 * oval[mi, 1]
                            + f2 * oval[mi, 2]
                            + f3 * oval[mi, 3]
                        )

        if const_expr(self.compute_gram):
            for mi in cutlass.range_constexpr(self.block_m):
                sqr[mi] = gvals[mi, 0] + gvals[mi, 1] + gvals[mi, 2] + gvals[mi, 3]

        for mi in cutlass.range_constexpr(self.block_m):
            for ni in cutlass.range_constexpr(self.tile_n):
                acc[mi, ni] = _warp_allreduce_sum(acc[mi, ni])
            if n_tile == Int32(0):
                sqr[mi] = _warp_allreduce_sum(sqr[mi])
                if const_expr(self.compute_gram):
                    for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                        gvals[mi, gp] = _warp_allreduce_sum(gvals[mi, gp])

        if lane == Int32(0):
            for mi in cutlass.range_constexpr(self.block_m):
                for ni in cutlass.range_constexpr(self.tile_n):
                    mix = mix0 + Int32(ni)
                    if mix < Int32(_MIXES):
                        warp_sums[mi, mix + Int32(1), warp] = acc[mi, ni]
                if n_tile == Int32(0):
                    warp_sums[mi, Int32(0), warp] = sqr[mi]
                    if const_expr(self.compute_gram):
                        for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                            gram_sums[mi, gp, warp] = gvals[mi, gp]
        cute.arch.sync_threads()

        if warp == Int32(0):
            for mi in cutlass.range_constexpr(self.block_m):
                token = Int32(m_tile * self.block_m + mi)
                if token < num_tokens:
                    mix = mix0 + lane
                    if lane < Int32(self.tile_n) and mix < Int32(_MIXES):
                        total = Float32(0.0)
                        src_warp = Int32(0)
                        while src_warp < Int32(nwarps):
                            total += Float32(warp_sums[mi, mix + Int32(1), src_warp])
                            src_warp += Int32(1)
                        partials[token, Int32(0), mix + Int32(1)] = total
                    if n_tile == Int32(0):
                        if lane == Int32(0):
                            total_sqr = Float32(0.0)
                            src = Int32(0)
                            while src < Int32(nwarps):
                                total_sqr += Float32(warp_sums[mi, Int32(0), src])
                                src += Int32(1)
                            partials[token, Int32(0), Int32(0)] = total_sqr
                        if const_expr(self.compute_gram):
                            if lane < Int32(_GRAM_PAIRS):
                                gtotal = Float32(0.0)
                                src = Int32(0)
                                while src < Int32(nwarps):
                                    gtotal += Float32(gram_sums[mi, lane, src])
                                    src += Int32(1)
                                partials[token, Int32(1), lane] = gtotal


class MHCPostPrePrefillGramKernel:
    """Full-hidden post + compact Gram kernel for tensor-core prefill paths."""

    num_threads = _PREFILL_GRAM_THREADS
    gram_pairs = _GRAM_PAIRS

    def __init__(
        self,
        *,
        hidden_size: int = _HIDDEN,
        split_k: int | None = None,
    ):
        self.hidden_size = int(hidden_size)
        self.split_k = (
            _split_k_for_hidden(self.hidden_size) if split_k is None else int(split_k)
        )
        _validate_split_k(self.hidden_size, self.split_k)
        if self.hidden_size % self.num_threads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by "
                f"prefill threads={self.num_threads}"
            )
        self.hidden_pair_iters = (
            self.hidden_size // 2 + self.num_threads - 1
        ) // self.num_threads

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        prev_post: cute.Tensor,
        prev_comb: cute.Tensor,
        partials: cute.Tensor,
        out: cute.Tensor,
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        if const_expr(x.element_type != cutlass.BFloat16):
            raise TypeError("x must be BFloat16")
        if const_expr(residual.element_type != cutlass.BFloat16):
            raise TypeError("residual must be BFloat16")
        if const_expr(prev_post.element_type != cutlass.Float32):
            raise TypeError("prev_post must be Float32")
        if const_expr(prev_comb.element_type != cutlass.Float32):
            raise TypeError("prev_comb must be Float32")
        if const_expr(partials.element_type != cutlass.Float32):
            raise TypeError("partials must be Float32")
        if const_expr(out.element_type != cutlass.BFloat16):
            raise TypeError("out must be BFloat16")
        self.kernel(x, residual, prev_post, prev_comb, partials, out).launch(
            grid=(num_tokens, 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        prev_post: cute.Tensor,
        prev_comb: cute.Tensor,
        partials: cute.Tensor,
        out: cute.Tensor,
    ):
        token, _, _ = cute.arch.block_idx()
        tidx = Int32(cute.arch.thread_idx()[0])
        lane = tidx % Int32(32)
        warp = tidx // Int32(32)
        nwarps = self.num_threads // 32
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_post_pre_prefill_gram_storage_cls())
        gram_sums = storage.gram_sums.get_tensor(
            cute.make_layout((_GRAM_PAIRS, nwarps), stride=(nwarps, 1))
        )
        post_coeff = storage.post_coeff.get_tensor(
            cute.make_layout((_MHC_MULT,), stride=(1,))
        )
        comb_coeff = storage.comb_coeff.get_tensor(
            cute.make_layout((_MHC_MULT, _MHC_MULT), stride=(_MHC_MULT, 1))
        )
        if tidx < Int32(_MHC_MULT):
            post_coeff[tidx] = prev_post[token, tidx]
        if tidx < Int32(_MHC_MULT * _MHC_MULT):
            source = tidx // Int32(_MHC_MULT)
            target = tidx - source * Int32(_MHC_MULT)
            comb_coeff[source, target] = prev_comb[token, source, target]
        cute.arch.sync_threads()

        gvals = cute.make_rmem_tensor(
            cute.make_layout((_GRAM_PAIRS,), stride=(1,)),
            Float32,
        )
        for gp in cutlass.range_constexpr(_GRAM_PAIRS):
            gvals[gp] = Float32(0.0)

        token = Int32(token)
        out_u32 = cute.recast_tensor(out, Uint32)
        for hidden_pair_iter in cutlass.range_constexpr(self.hidden_pair_iters):
            h = Int32(2 * hidden_pair_iter * self.num_threads) + tidx * Int32(2)
            if h < Int32(self.hidden_size):
                x_pair = ld_global_nc_u32(
                    get_ptr_as_int64(
                        x,
                        token * Int32(self.hidden_size) + h,
                    )
                )
                rin0_pair = ld_global_nc_u32(
                    get_ptr_as_int64(
                        residual,
                        token * Int32(_MHC_MULT * self.hidden_size) + h,
                    )
                )
                rin1_pair = ld_global_nc_u32(
                    get_ptr_as_int64(
                        residual,
                        token * Int32(_MHC_MULT * self.hidden_size)
                        + Int32(self.hidden_size)
                        + h,
                    )
                )
                rin2_pair = ld_global_nc_u32(
                    get_ptr_as_int64(
                        residual,
                        token * Int32(_MHC_MULT * self.hidden_size)
                        + Int32(2 * self.hidden_size)
                        + h,
                    )
                )
                rin3_pair = ld_global_nc_u32(
                    get_ptr_as_int64(
                        residual,
                        token * Int32(_MHC_MULT * self.hidden_size)
                        + Int32(3 * self.hidden_size)
                        + h,
                    )
                )
                x0, x1 = bfloat2_to_float2_scaled(x_pair, Float32(1.0))
                rin00, rin01 = bfloat2_to_float2_scaled(rin0_pair, Float32(1.0))
                rin10, rin11 = bfloat2_to_float2_scaled(rin1_pair, Float32(1.0))
                rin20, rin21 = bfloat2_to_float2_scaled(rin2_pair, Float32(1.0))
                rin30, rin31 = bfloat2_to_float2_scaled(rin3_pair, Float32(1.0))

                o_values = cute.make_rmem_tensor(
                    cute.make_layout((_MHC_MULT, 2), stride=(2, 1)),
                    Float32,
                )
                for pair_lane in cutlass.range_constexpr(2):
                    xh = x0
                    rin0 = rin00
                    rin1 = rin10
                    rin2 = rin20
                    rin3 = rin30
                    if const_expr(pair_lane == 1):
                        xh = x1
                        rin0 = rin01
                        rin1 = rin11
                        rin2 = rin21
                        rin3 = rin31
                    o_values[0, pair_lane] = (
                        Float32(post_coeff[0]) * xh
                        + Float32(comb_coeff[0, 0]) * rin0
                        + Float32(comb_coeff[1, 0]) * rin1
                        + Float32(comb_coeff[2, 0]) * rin2
                        + Float32(comb_coeff[3, 0]) * rin3
                    )
                    o_values[1, pair_lane] = (
                        Float32(post_coeff[1]) * xh
                        + Float32(comb_coeff[0, 1]) * rin0
                        + Float32(comb_coeff[1, 1]) * rin1
                        + Float32(comb_coeff[2, 1]) * rin2
                        + Float32(comb_coeff[3, 1]) * rin3
                    )
                    o_values[2, pair_lane] = (
                        Float32(post_coeff[2]) * xh
                        + Float32(comb_coeff[0, 2]) * rin0
                        + Float32(comb_coeff[1, 2]) * rin1
                        + Float32(comb_coeff[2, 2]) * rin2
                        + Float32(comb_coeff[3, 2]) * rin3
                    )
                    o_values[3, pair_lane] = (
                        Float32(post_coeff[3]) * xh
                        + Float32(comb_coeff[0, 3]) * rin0
                        + Float32(comb_coeff[1, 3]) * rin1
                        + Float32(comb_coeff[2, 3]) * rin2
                        + Float32(comb_coeff[3, 3]) * rin3
                    )

                o0_pair = pack_f32x2_to_bfloat2(o_values[0, 0], o_values[0, 1])
                o1_pair = pack_f32x2_to_bfloat2(o_values[1, 0], o_values[1, 1])
                o2_pair = pack_f32x2_to_bfloat2(o_values[2, 0], o_values[2, 1])
                o3_pair = pack_f32x2_to_bfloat2(o_values[3, 0], o_values[3, 1])
                out_h = h // Int32(2)
                out_u32[token, Int32(0), out_h] = o0_pair
                out_u32[token, Int32(1), out_h] = o1_pair
                out_u32[token, Int32(2), out_h] = o2_pair
                out_u32[token, Int32(3), out_h] = o3_pair

                r00, r01 = bfloat2_to_float2_scaled(o0_pair, Float32(1.0))
                r10, r11 = bfloat2_to_float2_scaled(o1_pair, Float32(1.0))
                r20, r21 = bfloat2_to_float2_scaled(o2_pair, Float32(1.0))
                r30, r31 = bfloat2_to_float2_scaled(o3_pair, Float32(1.0))
                gvals[0] += r00 * r00
                gvals[1] += r10 * r10
                gvals[2] += r20 * r20
                gvals[3] += r30 * r30
                gvals[4] += r00 * r10
                gvals[5] += r00 * r20
                gvals[6] += r00 * r30
                gvals[7] += r10 * r20
                gvals[8] += r10 * r30
                gvals[9] += r20 * r30
                gvals[0] += r01 * r01
                gvals[1] += r11 * r11
                gvals[2] += r21 * r21
                gvals[3] += r31 * r31
                gvals[4] += r01 * r11
                gvals[5] += r01 * r21
                gvals[6] += r01 * r31
                gvals[7] += r11 * r21
                gvals[8] += r11 * r31
                gvals[9] += r21 * r31

        for gp in cutlass.range_constexpr(_GRAM_PAIRS):
            gvals[gp] = _warp_allreduce_sum(gvals[gp])

        if lane == Int32(0):
            for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                gram_sums[gp, warp] = gvals[gp]
        cute.arch.sync_threads()

        if tidx == Int32(0):
            total_sq = Float32(0.0)
            for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                gtotal = Float32(0.0)
                src = Int32(0)
                while src < Int32(nwarps):
                    gtotal += Float32(gram_sums[gp, src])
                    src += Int32(1)
                partials[token, Int32(1), gp] = gtotal
                if gp < 4:
                    total_sq += gtotal
            partials[token, Int32(0), Int32(0)] = total_sq

        if const_expr(_MHC_PDL):
            cute.arch.sync_threads()
            cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _mhc_warp_mma_gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsA: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_A: cute.TiledCopy,
    smem_thr_copy_B: cute.TiledCopy,
):
    tCrA_copy_view = smem_thr_copy_A.retile(tCrA)
    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
    cute.copy(smem_thr_copy_A, tCsA[None, None, 0], tCrA_copy_view[None, None, 0])
    cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])
    for k in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
        if k < cute.size(tCsA.shape[2]) - 1:
            cute.copy(
                smem_thr_copy_A,
                tCsA[None, None, k + 1],
                tCrA_copy_view[None, None, k + 1],
            )
            cute.copy(
                smem_thr_copy_B,
                tCsB[None, None, k + 1],
                tCrB_copy_view[None, None, k + 1],
            )
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


class MHCPrefillBf16ProjectTmaKernel:
    """TMA-fed BF16 tensor-core projection for compact mHC prefill partials."""

    num_threads = _PREFILL_TMA_THREADS
    num_compute_warps = _PREFILL_TMA_COMPUTE_WARPS
    producer_warp = _PREFILL_TMA_COMPUTE_WARPS
    tile_m = _PREFILL_TMA_TILE_M
    tile_n = _PREFILL_TMA_TILE_N
    tile_k = _PREFILL_TMA_TILE_K
    num_stages = _PREFILL_TMA_STAGES
    buffer_align_bytes = 1024

    def __init__(
        self,
        *,
        hidden_size: int = _HIDDEN,
        split_k: int | None = None,
    ):
        self.hidden_size = int(hidden_size)
        self.total_k = _MHC_MULT * self.hidden_size
        self.split_k = (
            _split_k_for_hidden(self.hidden_size) if split_k is None else int(split_k)
        )
        _validate_split_k(self.hidden_size, self.split_k)
        if self.total_k % self.tile_k != 0:
            raise ValueError(
                f"total_k={self.total_k} must be divisible by TMA tile_k={self.tile_k}"
            )
        m_per_mma_group = self.num_compute_warps * 16
        if self.tile_m % m_per_mma_group != 0:
            raise ValueError(
                f"tile_m={self.tile_m} must be divisible by "
                f"num_compute_warps*16={m_per_mma_group}"
            )
        self.k_tiles = self.total_k // self.tile_k
        self.n_tiles = (_MIXES + self.tile_n - 1) // self.tile_n

    def _get_tiled_mma(self) -> cute.TiledMma:
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.BFloat16, Float32, (16, 8, 16)),
            (self.num_compute_warps, 1, 1),
            permutation_mnk=(self.num_compute_warps * 16, self.tile_n, 16),
        )

    def _get_smem_layouts(self) -> tuple[cute.ComposedLayout, cute.ComposedLayout]:
        a_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR,
                cutlass.BFloat16,
                self.tile_k,
            ),
            cutlass.BFloat16,
        )
        b_layout_atom = a_layout_atom
        sA_layout = cute.tile_to_shape(
            a_layout_atom,
            (self.tile_m, self.tile_k, self.num_stages),
            order=(0, 1, 2),
        )
        sB_layout = cute.tile_to_shape(
            b_layout_atom,
            (self.tile_n, self.tile_k, self.num_stages),
            order=(0, 1, 2),
        )
        return sA_layout, sB_layout

    def _get_shared_storage_cls(
        self,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
    ):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "mbar_ptr": cute.struct.MemRange[
                cutlass.Int64,
                self.num_stages * 2,
            ],
            "sA": cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(sA_layout)],
                self.buffer_align_bytes,
            ],
            "sB": cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(sB_layout)],
                self.buffer_align_bytes,
            ],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        out_flat: cute.Tensor,
        fn_bf16: cute.Tensor,
        partials: cute.Tensor,
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        if const_expr(out_flat.element_type != cutlass.BFloat16):
            raise TypeError("out_flat must be BFloat16")
        if const_expr(fn_bf16.element_type != cutlass.BFloat16):
            raise TypeError("fn_bf16 must be BFloat16")
        if const_expr(partials.element_type != cutlass.Float32):
            raise TypeError("partials must be Float32")

        sA_layout, sB_layout = self._get_smem_layouts()
        tiled_mma = self._get_tiled_mma()
        SharedStorage = self._get_shared_storage_cls(sA_layout, sB_layout)
        sA_tma_layout = cute.slice_(sA_layout, (None, None, 0))
        sB_tma_layout = cute.slice_(sB_layout, (None, None, 0))
        out_tma = _assume_tma_source_aligned(out_flat)
        fn_tma = _assume_tma_source_aligned(fn_bf16)
        tma_atom_A, tma_tensor_A = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            out_tma,
            sA_tma_layout,
            (self.tile_m, self.tile_k),
            num_multicast=1,
        )
        tma_atom_B, tma_tensor_B = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            fn_tma,
            sB_tma_layout,
            (self.tile_n, self.tile_k),
            num_multicast=1,
        )
        grid_m = (num_tokens + Int32(self.tile_m - 1)) // Int32(self.tile_m)
        self.kernel(
            tma_tensor_A,
            tma_tensor_B,
            partials,
            tma_atom_A,
            tma_atom_B,
            sA_layout,
            sB_layout,
            tiled_mma,
            SharedStorage,
            num_tokens,
        ).launch(
            grid=(grid_m, self.n_tiles, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        out_flat: cute.Tensor,
        fn_bf16: cute.Tensor,
        partials: cute.Tensor,
        tma_atom_A: cute.CopyAtom,
        tma_atom_B: cute.CopyAtom,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        num_tokens: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_tile, n_tile, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_A)
            cpasync.prefetch_descriptor(tma_atom_B)

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
        sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)

        tma_copy_bytes = (
            (self.tile_m + self.tile_n) * self.tile_k * cutlass.BFloat16.width // 8
        )
        load_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_compute_warps,
            ),
            tx_count=tma_copy_bytes,
            barrier_storage=storage.mbar_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        cute.arch.sync_threads()

        gA = cute.local_tile(
            out_flat,
            (self.tile_m, self.tile_k),
            (None, None),
        )
        gB = cute.local_tile(
            fn_bf16,
            (self.tile_n, self.tile_k),
            (None, None),
        )
        cta_layout = cute.make_layout(1)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_A,
            0,
            cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_B,
            0,
            cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )

        if warp_idx < Int32(self.num_compute_warps):
            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_stages,
            )
            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = thr_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = thr_mma.make_fragment_B(tCsB[None, None, None, 0])
            acc_shape = thr_mma.partition_shape_C((self.tile_m, self.tile_n))
            acc = cute.make_rmem_tensor(acc_shape, Float32)
            acc.fill(0.0)
            smem_copy_atom_A = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                cutlass.BFloat16,
            )
            smem_copy_atom_B = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                cutlass.BFloat16,
            )
            smem_thr_copy_A = cute.make_tiled_copy_A(
                smem_copy_atom_A,
                tiled_mma,
            ).get_slice(tidx)
            smem_thr_copy_B = cute.make_tiled_copy_B(
                smem_copy_atom_B,
                tiled_mma,
            ).get_slice(tidx)
            tSsA = smem_thr_copy_A.partition_S(sA)
            tSsB = smem_thr_copy_B.partition_S(sB)

            for _k_tile in cutlass.range_constexpr(self.k_tiles):
                load_pipeline.consumer_wait(consumer_state)
                _mhc_warp_mma_gemm(
                    thr_mma,
                    acc,
                    tCrA,
                    tCrB,
                    tSsA[None, None, None, consumer_state.index],
                    tSsB[None, None, None, consumer_state.index],
                    smem_thr_copy_A,
                    smem_thr_copy_B,
                )
                load_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

            acc_mn = _reshape_acc_to_mn(acc)
            coord_mn = _reshape_acc_to_mn(
                thr_mma.partition_C(
                    cute.make_identity_tensor((self.tile_m, self.tile_n))
                )
            )
            for acc_m in cutlass.range_constexpr(cute.size(acc_mn.shape[0])):
                for acc_n in cutlass.range_constexpr(cute.size(acc_mn.shape[1])):
                    coord = coord_mn[acc_m, acc_n]
                    token = m_tile * Int32(self.tile_m) + coord[0]
                    mix = n_tile * Int32(self.tile_n) + coord[1]
                    if token < num_tokens and mix < Int32(_MIXES):
                        partials[token, Int32(0), mix + Int32(1)] = acc_mn[
                            acc_m,
                            acc_n,
                        ]

        elif warp_idx == Int32(self.producer_warp):
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.num_stages,
            )
            for k_tile in cutlass.range_constexpr(self.k_tiles):
                load_pipeline.producer_acquire(producer_state)
                cute.copy(
                    tma_atom_A,
                    tAgA[(None, m_tile, k_tile)],
                    tAsA[(None, producer_state.index)],
                    tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
                )
                cute.copy(
                    tma_atom_B,
                    tBgB[(None, n_tile, k_tile)],
                    tBsB[(None, producer_state.index)],
                    tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
                )
                load_pipeline.producer_commit(producer_state)
                producer_state.advance()
            load_pipeline.producer_tail(producer_state)


class MHCPrefillTf32ProjectTmaKernel:
    """TMA-fed TF32 tensor-core projection for compact mHC prefill partials."""

    num_threads = _PREFILL_TF32_TMA_THREADS
    num_m_warps = _PREFILL_TF32_TMA_M_WARPS
    num_n_warps = _PREFILL_TF32_TMA_N_WARPS
    num_compute_warps = _PREFILL_TF32_TMA_COMPUTE_WARPS
    producer_warp = _PREFILL_TF32_TMA_COMPUTE_WARPS
    tile_m = _PREFILL_TF32_TMA_TILE_M
    tile_n = _PREFILL_TF32_TMA_TILE_N
    tile_k = _PREFILL_TF32_TMA_TILE_K
    num_stages = _PREFILL_TF32_TMA_STAGES
    buffer_align_bytes = 1024

    def __init__(
        self,
        *,
        hidden_size: int = _HIDDEN,
        split_k: int | None = None,
        chunk_geometry: bool = False,
        long_geometry: bool = False,
    ):
        self.hidden_size = int(hidden_size)
        use_4096_chunk_geometry = chunk_geometry and self.hidden_size == _HIDDEN
        use_4096_long_geometry = long_geometry and self.hidden_size == _HIDDEN
        if use_4096_long_geometry:
            self.num_m_warps = _PREFILL_TF32_TMA_LONG_4096_M_WARPS
            self.num_n_warps = _PREFILL_TF32_TMA_LONG_4096_N_WARPS
            self.tile_m = _PREFILL_TF32_TMA_LONG_4096_TILE_M
            self.tile_n = _PREFILL_TF32_TMA_LONG_4096_TILE_N
            self.tile_k = _PREFILL_TF32_TMA_LONG_4096_TILE_K
            self.num_stages = _PREFILL_TF32_TMA_LONG_4096_STAGES
            self.k_splits = _PREFILL_TF32_TMA_LONG_4096_K_SPLITS
        elif use_4096_chunk_geometry:
            self.num_m_warps = _PREFILL_TF32_TMA_CHUNK_4096_M_WARPS
            self.num_n_warps = _PREFILL_TF32_TMA_CHUNK_4096_N_WARPS
            self.tile_m = _PREFILL_TF32_TMA_CHUNK_4096_TILE_M
            self.tile_n = _PREFILL_TF32_TMA_CHUNK_4096_TILE_N
            self.tile_k = _PREFILL_TF32_TMA_CHUNK_4096_TILE_K
            self.num_stages = _PREFILL_TF32_TMA_CHUNK_4096_STAGES
            self.k_splits = _PREFILL_TF32_TMA_CHUNK_4096_K_SPLITS
        elif chunk_geometry:
            self.num_m_warps = _PREFILL_TF32_TMA_CHUNK_OTHER_M_WARPS
            self.num_n_warps = _PREFILL_TF32_TMA_CHUNK_OTHER_N_WARPS
            self.tile_m = _PREFILL_TF32_TMA_CHUNK_OTHER_TILE_M
            self.tile_n = _PREFILL_TF32_TMA_CHUNK_OTHER_TILE_N
            self.tile_k = _PREFILL_TF32_TMA_TILE_K
            self.num_stages = _PREFILL_TF32_TMA_STAGES
            self.k_splits = 1
        else:
            self.num_m_warps = _PREFILL_TF32_TMA_M_WARPS
            self.num_n_warps = _PREFILL_TF32_TMA_N_WARPS
            self.tile_m = _PREFILL_TF32_TMA_TILE_M
            self.tile_n = _PREFILL_TF32_TMA_TILE_N
            self.tile_k = _PREFILL_TF32_TMA_TILE_K
            self.num_stages = _PREFILL_TF32_TMA_STAGES
            self.k_splits = 1
        self.num_compute_warps = self.num_m_warps * self.num_n_warps
        self.producer_warp = self.num_compute_warps
        self.num_threads = (self.num_compute_warps + 1) * 32
        self.total_k = _MHC_MULT * self.hidden_size
        self.split_k = (
            _split_k_for_hidden(self.hidden_size) if split_k is None else int(split_k)
        )
        _validate_split_k(self.hidden_size, self.split_k)
        if self.total_k % self.tile_k != 0:
            raise ValueError(
                f"total_k={self.total_k} must be divisible by TF32 TMA tile_k={self.tile_k}"
            )
        if self.tile_k % 8 != 0:
            raise ValueError(f"TF32 TMA tile_k={self.tile_k} must be divisible by 8")
        if self.tile_n <= 0 or self.tile_n % 8 != 0:
            raise ValueError(
                f"TF32 TMA tile_n={self.tile_n} must be a positive multiple of 8 "
                "for m16n8k8 warp MMA"
            )
        if self.num_n_warps <= 0 or (self.tile_n // 8) % self.num_n_warps != 0:
            raise ValueError(
                f"TF32 TMA N warps={self.num_n_warps} must divide "
                f"tile_n/8={self.tile_n // 8}"
            )
        m_per_mma_group = self.num_m_warps * 16
        if self.tile_m != m_per_mma_group:
            raise ValueError(
                f"TF32 TMA tile_m={self.tile_m} must equal "
                f"num_compute_warps*16={m_per_mma_group}"
            )
        self.k_tiles = self.total_k // self.tile_k
        if self.k_splits <= 0 or self.k_tiles % self.k_splits != 0:
            raise ValueError(
                f"TF32 TMA k_splits={self.k_splits} must be positive and divide "
                f"k_tiles={self.k_tiles}"
            )
        if self.k_splits >= self.split_k:
            raise ValueError(
                f"TF32 TMA k_splits={self.k_splits} must be less than "
                f"partials split_k={self.split_k}"
            )
        self.k_tiles_per_split = self.k_tiles // self.k_splits
        self.n_tiles = (_MIXES + self.tile_n - 1) // self.tile_n
        self.n_mma_tiles = self.tile_n // 8
        self.n_mma_tiles_per_warp = self.n_mma_tiles // self.num_n_warps

    def _get_smem_layouts(
        self,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout]:
        a_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR,
                cutlass.BFloat16,
                self.tile_k,
            ),
            cutlass.BFloat16,
        )
        b_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR,
                cutlass.Float32,
                self.tile_k,
            ),
            cutlass.Float32,
        )
        sA_layout = cute.tile_to_shape(
            a_layout_atom,
            (self.tile_m, self.tile_k, self.num_stages),
            order=(0, 1, 2),
        )
        sB_layout = cute.tile_to_shape(
            b_layout_atom,
            (self.tile_n, self.tile_k, self.num_stages),
            order=(0, 1, 2),
        )
        return sA_layout, sB_layout

    def _get_shared_storage_cls(
        self,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
    ):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "mbar_ptr": cute.struct.MemRange[
                cutlass.Int64,
                self.num_stages * 2,
            ],
            "sA": cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(sA_layout)],
                self.buffer_align_bytes,
            ],
            "sB": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(sB_layout)],
                self.buffer_align_bytes,
            ],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        out_flat: cute.Tensor,
        fn: cute.Tensor,
        partials: cute.Tensor,
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        if const_expr(out_flat.element_type != cutlass.BFloat16):
            raise TypeError("out_flat must be BFloat16")
        if const_expr(fn.element_type != cutlass.Float32):
            raise TypeError("fn must be Float32")
        if const_expr(partials.element_type != cutlass.Float32):
            raise TypeError("partials must be Float32")

        sA_layout, sB_layout = self._get_smem_layouts()
        SharedStorage = self._get_shared_storage_cls(sA_layout, sB_layout)
        sA_tma_layout = cute.slice_(sA_layout, (None, None, 0))
        sB_tma_layout = cute.slice_(sB_layout, (None, None, 0))
        out_tma = _assume_tma_source_aligned(out_flat)
        fn_tma = _assume_tma_source_aligned(fn)
        tma_atom_A, tma_tensor_A = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            out_tma,
            sA_tma_layout,
            (self.tile_m, self.tile_k),
            num_multicast=1,
        )
        tma_atom_B, tma_tensor_B = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            fn_tma,
            sB_tma_layout,
            (self.tile_n, self.tile_k),
            num_multicast=1,
        )
        grid_m = (num_tokens + Int32(self.tile_m - 1)) // Int32(self.tile_m)
        self.kernel(
            tma_tensor_A,
            tma_tensor_B,
            partials,
            tma_atom_A,
            tma_atom_B,
            sA_layout,
            sB_layout,
            SharedStorage,
            num_tokens,
        ).launch(
            grid=(grid_m, self.n_tiles, self.k_splits),
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=_MHC_PDL,
        )

    @cute.kernel
    def kernel(
        self,
        out_flat: cute.Tensor,
        fn: cute.Tensor,
        partials: cute.Tensor,
        tma_atom_A: cute.CopyAtom,
        tma_atom_B: cute.CopyAtom,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        SharedStorage: cutlass.Constexpr,
        num_tokens: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_tile, n_tile, k_split = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if const_expr(_MHC_PDL):
            cute.arch.griddepcontrol_wait()

        partial_row = k_split + Int32(1)
        if k_split == Int32(0):
            partial_row = Int32(0)

        if tidx == 0:
            cpasync.prefetch_descriptor(tma_atom_A)
            cpasync.prefetch_descriptor(tma_atom_B)

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
        sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)

        tma_copy_bytes = (
            self.tile_m * self.tile_k * cutlass.BFloat16.width // 8
            + self.tile_n * self.tile_k * cutlass.Float32.width // 8
        )
        load_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_compute_warps,
            ),
            tx_count=tma_copy_bytes,
            barrier_storage=storage.mbar_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )

        gA = cute.local_tile(
            out_flat,
            (self.tile_m, self.tile_k),
            (None, None),
        )
        gB = cute.local_tile(
            fn,
            (self.tile_n, self.tile_k),
            (None, None),
        )
        cta_layout = cute.make_layout(1)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_A,
            0,
            cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_B,
            0,
            cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )

        if warp_idx < Int32(self.num_compute_warps):
            warp_m = warp_idx // Int32(self.num_n_warps)
            warp_n = warp_idx % Int32(self.num_n_warps)
            lane = tidx & Int32(31)
            lane_group = lane >> Int32(2)
            lane_in_group = lane & Int32(3)
            lane_pair_base = lane_in_group * Int32(2)
            warp_m_base = warp_m * Int32(16)
            row0 = warp_m_base + lane_group
            row1 = row0 + Int32(8)
            token0 = m_tile * Int32(self.tile_m) + row0
            token1 = token0 + Int32(8)
            acc = cute.make_rmem_tensor(
                cute.make_layout((self.n_mma_tiles_per_warp, 4), stride=(4, 1)),
                Float32,
            )
            acc.fill(0.0)
            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_stages,
            )
            for _k_tile in range(0, self.k_tiles_per_split, 1, unroll=1):
                load_pipeline.consumer_wait(consumer_state)
                for kk in cutlass.range_constexpr(self.tile_k // 8):
                    k0 = Int32(kk * 8)
                    a_k0 = k0 + lane_in_group
                    a_k1 = a_k0 + Int32(4)

                    # TMA zero-fills out-of-bounds rows, so all compute lanes
                    # can load shared memory without a per-MMA token branch.
                    a0_f = Float32(sA[row0, a_k0, consumer_state.index])
                    a1_f = Float32(sA[row1, a_k0, consumer_state.index])
                    a2_f = Float32(sA[row0, a_k1, consumer_state.index])
                    a3_f = Float32(sA[row1, a_k1, consumer_state.index])

                    # These values originate from BF16, so their promoted FP32
                    # representation is already exact in TF32.  A raw bitcast
                    # avoids four redundant cvt.rna.tf32.f32 instructions per
                    # MMA step without changing the operand value.
                    a0_tf32 = f32_to_raw_bits(a0_f)
                    a1_tf32 = f32_to_raw_bits(a1_f)
                    a2_tf32 = f32_to_raw_bits(a2_f)
                    a3_tf32 = f32_to_raw_bits(a3_f)
                    for warp_mma_n in cutlass.range_constexpr(
                        self.n_mma_tiles_per_warp
                    ):
                        mma_n = Int32(warp_mma_n * self.num_n_warps) + warp_n
                        b_mix_local = Int32(mma_n * 8) + lane_group
                        # TMA likewise zero-fills a partial final N tile.
                        b0_f = Float32(sB[b_mix_local, a_k0, consumer_state.index])
                        b1_f = Float32(sB[b_mix_local, a_k1, consumer_state.index])

                        d0, d1, d2, d3 = tf32_mma_m16n8k8_f32(
                            acc[warp_mma_n, 0],
                            acc[warp_mma_n, 1],
                            acc[warp_mma_n, 2],
                            acc[warp_mma_n, 3],
                            a0_tf32,
                            a1_tf32,
                            a2_tf32,
                            a3_tf32,
                            f32_to_raw_bits(b0_f),
                            f32_to_raw_bits(b1_f),
                        )
                        acc[warp_mma_n, 0] = d0
                        acc[warp_mma_n, 1] = d1
                        acc[warp_mma_n, 2] = d2
                        acc[warp_mma_n, 3] = d3
                load_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

            for warp_mma_n in cutlass.range_constexpr(self.n_mma_tiles_per_warp):
                mma_n = Int32(warp_mma_n * self.num_n_warps) + warp_n
                mix0 = n_tile * Int32(self.tile_n) + Int32(mma_n * 8) + lane_pair_base
                mix1 = mix0 + Int32(1)
                if token0 < num_tokens:
                    if mix0 < Int32(_MIXES):
                        partials[token0, partial_row, mix0 + Int32(1)] = acc[
                            warp_mma_n, 0
                        ]
                    if mix1 < Int32(_MIXES):
                        partials[token0, partial_row, mix1 + Int32(1)] = acc[
                            warp_mma_n, 1
                        ]
                if token1 < num_tokens:
                    if mix0 < Int32(_MIXES):
                        partials[token1, partial_row, mix0 + Int32(1)] = acc[
                            warp_mma_n, 2
                        ]
                    if mix1 < Int32(_MIXES):
                        partials[token1, partial_row, mix1 + Int32(1)] = acc[
                            warp_mma_n, 3
                        ]

        elif warp_idx == Int32(self.producer_warp):
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.num_stages,
            )
            for _k_tile in range(0, self.k_tiles_per_split, 1, unroll=1):
                load_pipeline.producer_acquire(producer_state)
                k_tile = k_split * Int32(self.k_tiles_per_split) + producer_state.count
                cute.copy(
                    tma_atom_A,
                    tAgA[(None, m_tile, k_tile)],
                    tAsA[(None, producer_state.index)],
                    tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
                )
                cute.copy(
                    tma_atom_B,
                    tBgB[(None, n_tile, k_tile)],
                    tBsB[(None, producer_state.index)],
                    tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
                )
                load_pipeline.producer_commit(producer_state)
                producer_state.advance()
            load_pipeline.producer_tail(producer_state)

        if const_expr(_MHC_PDL):
            cute.arch.sync_threads()
            cute.arch.griddepcontrol_launch_dependents()


class MHCPrefillBf16ProjectKernel:
    """BF16 tensor-core projection for compact mHC prefill partials."""

    num_threads = _PREFILL_MMA_THREADS
    tile_m = _PREFILL_MMA_TILE_M
    tile_n = _PREFILL_MMA_TILE_N
    tile_k = _PREFILL_MMA_TILE_K

    def __init__(
        self,
        *,
        hidden_size: int = _HIDDEN,
        split_k: int | None = None,
    ):
        self.hidden_size = int(hidden_size)
        self.total_k = _MHC_MULT * self.hidden_size
        self.split_k = (
            _split_k_for_hidden(self.hidden_size) if split_k is None else int(split_k)
        )
        _validate_split_k(self.hidden_size, self.split_k)
        if self.total_k % self.tile_k != 0:
            raise ValueError(
                f"total_k={self.total_k} must be divisible by MMA tile_k={self.tile_k}"
            )
        self.k_iters = self.total_k // self.tile_k
        self.n_tiles = (_MIXES + self.tile_n - 1) // self.tile_n

    @cute.jit
    def __call__(
        self,
        out: cute.Tensor,
        fn_bf16: cute.Tensor,
        partials: cute.Tensor,
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        if const_expr(out.element_type != cutlass.BFloat16):
            raise TypeError("out must be BFloat16")
        if const_expr(fn_bf16.element_type != cutlass.BFloat16):
            raise TypeError("fn_bf16 must be BFloat16")
        if const_expr(partials.element_type != cutlass.Float32):
            raise TypeError("partials must be Float32")
        grid_m = (num_tokens + Int32(self.tile_m - 1)) // Int32(self.tile_m)
        self.kernel(out, fn_bf16, partials, num_tokens).launch(
            grid=(grid_m, self.n_tiles, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        out: cute.Tensor,
        fn_bf16: cute.Tensor,
        partials: cute.Tensor,
        num_tokens: Int32,
    ):
        m_tile, n_tile, _ = cute.arch.block_idx()
        lane = Int32(cute.arch.thread_idx()[0])
        lane_group = lane >> Int32(2)
        lane_pair_base = (lane & Int32(3)) * Int32(2)
        b_mix = n_tile * Int32(self.tile_n) + lane_group
        b_tid = lane & Int32(3)
        a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
        a_col = (lane >> Int32(4)) * Int32(8)

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_prefill_bf16_mma_storage_cls())
        a_base_addr = shared_ptr_to_u32(storage.a_tile.data_ptr())

        d0 = Float32(0.0)
        d1 = Float32(0.0)
        d2 = Float32(0.0)
        d3 = Float32(0.0)

        for k_iter in cutlass.range_constexpr(self.k_iters):
            k0 = Int32(k_iter * self.tile_k)
            linear = lane
            while linear < Int32((self.tile_m * self.tile_k) // 2):
                row = linear // Int32(self.tile_k // 2)
                pair = linear - row * Int32(self.tile_k // 2)
                token = m_tile * Int32(self.tile_m) + row
                kval = k0 + pair * Int32(2)
                value = Uint32(0)
                if token < num_tokens:
                    value = ld_global_nc_u32(
                        get_ptr_as_int64(
                            out,
                            token * Int32(self.total_k) + kval,
                        )
                    )
                st_shared_u32(
                    a_base_addr
                    + (row * Int32(self.tile_k) + pair * Int32(2)) * Int32(2),
                    value,
                )
                linear += Int32(self.num_threads)
            cute.arch.sync_warp()

            a_byte = (a_row * Int32(self.tile_k) + a_col) * Int32(2)
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_base_addr + a_byte)
            b0 = ld_global_nc_u32(
                get_ptr_as_int64(
                    fn_bf16,
                    b_mix * Int32(self.total_k) + k0 + b_tid * Int32(2),
                )
            )
            b1 = ld_global_nc_u32(
                get_ptr_as_int64(
                    fn_bf16,
                    b_mix * Int32(self.total_k) + k0 + Int32(8) + b_tid * Int32(2),
                )
            )
            d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                d0, d1, d2, d3, a0, a1, a2, a3, b0, b1
            )
            cute.arch.sync_warp()

        for reg_id in cutlass.range_constexpr(4):
            row_slot = Int32(reg_id // 2)
            token = m_tile * Int32(self.tile_m) + lane_group + row_slot * Int32(8)
            mix = n_tile * Int32(self.tile_n) + lane_pair_base + Int32(reg_id % 2)
            value = d0
            if const_expr(reg_id == 1):
                value = d1
            elif const_expr(reg_id == 2):
                value = d2
            elif const_expr(reg_id == 3):
                value = d3
            if token < num_tokens and mix < Int32(_MIXES):
                partials[token, Int32(0), mix + Int32(1)] = value


class MHCFinalizeGramKernel:
    """Multi-CTA fuse_norm finalize using the residual_out Gram matrix.

    The partial kernel (compute_gram=True) provides G[m,m'] in partials rows
    [32, 64), so sum_h y^2 = pre^T G pre is a scalar -- no per-h norm reduction.
    Each CTA owns one or more hidden tiles, redundantly reduces partials+Gram and
    runs the Sinkhorn (cheap), then writes its y tiles without cross-CTA sync.
    Compact prefill partials use one CTA per token because prefill has enough
    token parallelism and this avoids repeating the scalar finalize per tile.
    """

    num_threads = _GRAM_BLOCK_H
    block_h = _GRAM_BLOCK_H
    hidden_size = _HIDDEN
    source_tiles = _SOURCE_TILES
    source_warps = (_SOURCE_TILES + 31) // 32
    total_k = _TOTAL_K
    split_k = _SPLIT_K
    mixes = _MIXES
    partials = _PARTIALS
    gram_row0 = _GRAM_ROW0
    gram_pairs = _GRAM_PAIRS

    def __init__(
        self,
        *,
        hidden_size: int = _HIDDEN,
        split_k: int | None = None,
        rms_eps: float,
        hc_eps: float,
        sinkhorn_iters: int,
        norm_eps: float,
        fuse_norm: bool = True,
        compact_partials: bool = False,
        compact_projection_splits: int = 1,
        single_cta: bool = False,
        single_cta_threads: int = _PREFILL_FINALIZE_THREADS,
        single_cta_groups: int = 1,
        active_source_splits: int = 0,
    ):
        self.hidden_size = int(hidden_size)
        self.single_cta = bool(single_cta)
        self.single_cta_threads = int(single_cta_threads)
        self.single_cta_groups = int(single_cta_groups)
        if self.single_cta_groups <= 0:
            raise ValueError("single_cta_groups must be positive")
        if not self.single_cta and self.single_cta_groups != 1:
            raise ValueError("single_cta_groups requires single_cta=True")
        self.num_threads = (
            self.single_cta_threads
            if self.single_cta
            else (_PREFILL_FINALIZE_THREADS if compact_partials else _GRAM_BLOCK_H)
        )
        self.block_h = self.num_threads
        if self.hidden_size % self.block_h != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by "
                f"finalize block_h={self.block_h}"
            )
        if self.single_cta and (
            self.hidden_size % (2 * self.num_threads * self.single_cta_groups) != 0
        ):
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by "
                "2 * single-CTA threads * groups="
                f"{2 * self.num_threads * self.single_cta_groups}"
            )
        self.source_tiles = _source_tiles_for_hidden(self.hidden_size)
        self.hidden_tiles = self.hidden_size // self.block_h
        if self.hidden_tiles % self.single_cta_groups != 0:
            raise ValueError(
                f"hidden tiles={self.hidden_tiles} must be divisible by "
                f"single_cta_groups={self.single_cta_groups}"
            )
        self.active_source_splits = int(active_source_splits)
        if self.active_source_splits == 0:
            self.active_source_splits = self.source_tiles
        if (
            self.active_source_splits <= 0
            or self.active_source_splits > self.source_tiles
        ):
            raise ValueError(
                f"active_source_splits must be in [1, {self.source_tiles}], "
                f"got {self.active_source_splits}"
            )
        self.source_warps = (self.active_source_splits + 31) // 32
        self.total_k = _MHC_MULT * self.hidden_size
        self.split_k = (
            _split_k_for_hidden(self.hidden_size) if split_k is None else int(split_k)
        )
        _validate_split_k(self.hidden_size, self.split_k)
        self.gram_row0 = self.source_tiles
        self.rms_eps = float(rms_eps)
        self.hc_eps = float(hc_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.norm_eps = float(norm_eps)
        # When False, norm_weight is ignored: y is the raw collapsed activation
        # (no Gram reduction, no RMSNorm). The partial then skips the Gram.
        self.fuse_norm = bool(fuse_norm)
        self.compact_partials = bool(compact_partials)
        self.compact_projection_splits = int(compact_projection_splits)
        if self.compact_projection_splits <= 0:
            raise ValueError("compact_projection_splits must be positive")
        if not self.compact_partials and self.compact_projection_splits != 1:
            raise ValueError("compact_projection_splits requires compact_partials=True")
        if self.compact_partials and self.active_source_splits != self.source_tiles:
            raise ValueError(
                "active_source_splits is only valid for standard partial layouts"
            )
        if self.compact_projection_splits >= self.split_k:
            raise ValueError(
                "compact_projection_splits must be less than partials split_k"
            )
        self.tiles_per_cta = (
            self.hidden_tiles // self.single_cta_groups
            if self.single_cta
            else self.hidden_tiles
            if self.compact_partials
            else 1
        )
        if (self.compact_partials or self.single_cta) and self.hidden_size % (
            2 * self.num_threads
        ) != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by "
                f"2 * vectorized finalize threads={2 * self.num_threads}"
            )

    @cute.jit
    def __call__(
        self,
        residual: cute.Tensor,
        partials: cute.Tensor,
        scale: cute.Tensor,
        bias: cute.Tensor,
        y: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
        norm_weight: cute.Tensor,
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        if const_expr(residual.element_type != cutlass.BFloat16):
            raise TypeError("residual must be BFloat16")
        if const_expr(partials.element_type != cutlass.Float32):
            raise TypeError("partials must be Float32")
        if const_expr(y.element_type != cutlass.BFloat16):
            raise TypeError("y must be BFloat16")
        if const_expr(
            self.fuse_norm
            and norm_weight.element_type != cutlass.BFloat16
            and norm_weight.element_type != cutlass.Float32
        ):
            raise TypeError("norm_weight must be BFloat16 or Float32")
        self.kernel(residual, partials, scale, bias, y, post, comb, norm_weight).launch(
            grid=(self.hidden_tiles // self.tiles_per_cta, num_tokens, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
            use_pdl=_MHC_PDL,
        )

    @cute.kernel
    def kernel(
        self,
        residual: cute.Tensor,
        partials: cute.Tensor,
        scale: cute.Tensor,
        bias: cute.Tensor,
        y: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
        norm_weight: cute.Tensor,
    ):
        tile_group, token, _ = cute.arch.block_idx()
        tile_group = Int32(tile_group)
        tidx = Int32(cute.arch.thread_idx()[0])

        if const_expr(_MHC_PDL):
            cute.arch.griddepcontrol_wait()

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_finalize_storage_cls(self.num_threads, False))
        s_pre = storage.pre.get_tensor(cute.make_layout((_MHC_MULT,), stride=(1,)))
        s_post = storage.post.get_tensor(cute.make_layout((_MHC_MULT,), stride=(1,)))
        s_comb = storage.comb.get_tensor(
            cute.make_layout((_MHC_MULT, _MHC_MULT), stride=(_MHC_MULT, 1))
        )

        sums = cute.make_rmem_tensor(
            cute.make_layout((self.partials,), stride=(1,)), Float32
        )
        gram = cute.make_rmem_tensor(
            cute.make_layout((self.gram_pairs,), stride=(1,)), Float32
        )
        if const_expr(self.single_cta):
            source_values = storage.partials.get_tensor(
                cute.make_layout((self.num_threads,), stride=(1,))
            )
            if tidx < Int32(_PARTIALS):
                total = Float32(0.0)
                for source_split in cutlass.range_constexpr(self.active_source_splits):
                    total += Float32(partials[token, source_split, tidx])
                source_values[tidx] = total
            if const_expr(self.fuse_norm):
                if tidx < Int32(_GRAM_PAIRS):
                    total = Float32(0.0)
                    for source_split in cutlass.range_constexpr(
                        self.active_source_splits
                    ):
                        total += Float32(
                            partials[
                                token,
                                Int32(self.gram_row0 + source_split),
                                tidx,
                            ]
                        )
                    source_values[Int32(32) + tidx] = total
            cute.arch.sync_threads()
        elif const_expr(self.compact_partials):
            if tidx == Int32(0):
                for column in cutlass.range_constexpr(_PARTIALS):
                    sums[column] = Float32(partials[token, Int32(0), column])
                    if column > 0:
                        for projection_split in cutlass.range_constexpr(
                            1, self.compact_projection_splits
                        ):
                            sums[column] += Float32(
                                partials[
                                    token,
                                    Int32(projection_split + 1),
                                    column,
                                ]
                            )
                if const_expr(self.fuse_norm):
                    for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                        gram[gp] = Float32(partials[token, Int32(1), gp])
        elif const_expr(self.source_warps == 1):
            if tidx < Int32(32):
                for column in cutlass.range_constexpr(_PARTIALS):
                    value = Float32(0.0)
                    if tidx < Int32(self.active_source_splits):
                        value = Float32(partials[token, tidx, column])
                    sums[column] = _warp_allreduce_sum(value)
                if const_expr(self.fuse_norm):
                    for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                        gvalue = Float32(0.0)
                        if tidx < Int32(self.active_source_splits):
                            gvalue = Float32(
                                partials[token, Int32(self.gram_row0) + tidx, gp]
                            )
                        gram[gp] = _warp_allreduce_sum(gvalue)
        elif const_expr(self.source_warps == 2):
            lane = tidx % Int32(32)
            warp = tidx // Int32(32)
            source_tile = warp * Int32(32) + lane
            source_sums = storage.partials.get_tensor(
                cute.make_layout((self.num_threads,), stride=(1,))
            )
            if tidx < Int32(64):
                for column in cutlass.range_constexpr(_PARTIALS):
                    value = Float32(0.0)
                    if source_tile < Int32(self.active_source_splits):
                        value = Float32(partials[token, source_tile, column])
                    value = _warp_allreduce_sum(value)
                    if lane == Int32(0):
                        source_sums[Int32(column * 2) + warp] = value
                if const_expr(self.fuse_norm):
                    for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                        gvalue = Float32(0.0)
                        if source_tile < Int32(self.active_source_splits):
                            gvalue = Float32(
                                partials[token, Int32(self.gram_row0) + source_tile, gp]
                            )
                        gvalue = _warp_allreduce_sum(gvalue)
                        if lane == Int32(0):
                            source_sums[Int32((_PARTIALS + gp) * 2) + warp] = gvalue
            cute.arch.sync_threads()
            if tidx == Int32(0):
                for column in cutlass.range_constexpr(_PARTIALS):
                    sums[column] = Float32(source_sums[Int32(column * 2)]) + Float32(
                        source_sums[Int32(column * 2 + 1)]
                    )
                if const_expr(self.fuse_norm):
                    for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                        gram[gp] = Float32(
                            source_sums[Int32((_PARTIALS + gp) * 2)]
                        ) + Float32(source_sums[Int32((_PARTIALS + gp) * 2 + 1)])
        else:
            lane = tidx % Int32(32)
            warp = tidx // Int32(32)
            source_sums = storage.partials.get_tensor(
                cute.make_layout((self.num_threads,), stride=(1,))
            )
            source_warp_threads = Int32(self.source_warps * 32)
            for column in cutlass.range_constexpr(_PARTIALS):
                value = Float32(0.0)
                if tidx < source_warp_threads:
                    if tidx < Int32(self.active_source_splits):
                        value = Float32(partials[token, tidx, column])
                    value = _warp_allreduce_sum(value)
                    if lane == Int32(0):
                        source_sums[warp] = value
                cute.arch.sync_threads()
                if tidx == Int32(0):
                    total = Float32(0.0)
                    src_warp = Int32(0)
                    while src_warp < Int32(self.source_warps):
                        total += Float32(source_sums[src_warp])
                        src_warp += Int32(1)
                    sums[column] = total
                cute.arch.sync_threads()
            if const_expr(self.fuse_norm):
                for gp in cutlass.range_constexpr(_GRAM_PAIRS):
                    gvalue = Float32(0.0)
                    if tidx < source_warp_threads:
                        if tidx < Int32(self.active_source_splits):
                            gvalue = Float32(
                                partials[token, Int32(self.gram_row0) + tidx, gp]
                            )
                        gvalue = _warp_allreduce_sum(gvalue)
                        if lane == Int32(0):
                            source_sums[warp] = gvalue
                    cute.arch.sync_threads()
                    if tidx == Int32(0):
                        gtotal = Float32(0.0)
                        src_warp = Int32(0)
                        while src_warp < Int32(self.source_warps):
                            gtotal += Float32(source_sums[src_warp])
                            src_warp += Int32(1)
                        gram[gp] = gtotal
                    cute.arch.sync_threads()

        if const_expr(self.single_cta):
            lane = tidx % Int32(32)
            if tidx == Int32(0):
                s_post[0] = cute.math.rsqrt(
                    Float32(source_values[0]) / Float32(self.total_k)
                    + Float32(self.rms_eps),
                    fastmath=True,
                )
            cute.arch.sync_threads()

            mix_value = Float32(0.0)
            one_value = Float32(1.0)
            value = Float32(0.0)
            if tidx < Int32(_MIXES):
                mix_value = Float32(source_values[tidx + Int32(1)]) * Float32(s_post[0])
                if tidx < Int32(4):
                    value = one_value / (
                        one_value
                        + cute.math.exp(
                            -(mix_value * Float32(scale[0]) + Float32(bias[tidx])),
                            fastmath=True,
                        )
                    ) + Float32(self.hc_eps)
                    s_pre[tidx] = value
                elif tidx < Int32(8):
                    value = Float32(2.0) / (
                        one_value
                        + cute.math.exp(
                            -(mix_value * Float32(scale[1]) + Float32(bias[tidx])),
                            fastmath=True,
                        )
                    )
                    if tile_group == Int32(0):
                        post[token, tidx - Int32(4)] = value
                else:
                    cidx = tidx - Int32(8)
                    row = cidx // Int32(4)
                    col = cidx - row * Int32(4)
                    s_comb[row, col] = mix_value * Float32(scale[2]) + Float32(
                        bias[tidx]
                    )
            cute.arch.sync_threads()

            if tidx < Int32(32):
                cvalue = Float32(0.0)
                if lane < Int32(16):
                    row = lane // Int32(4)
                    col = lane - row * Int32(4)
                    cvalue = Float32(s_comb[row, col])
                row_max = _warp_quad_allreduce_max(cvalue)
                cvalue = cute.math.exp(cvalue - row_max, fastmath=True)
                row_sum = _warp_quad_allreduce_sum(cvalue)
                cvalue = cvalue * cute.arch.rcp_approx(row_sum) + Float32(self.hc_eps)
                col_sum = _warp_column4_allreduce_sum(cvalue) + Float32(self.hc_eps)
                cvalue = cvalue * cute.arch.rcp_approx(col_sum)
                for _ in cutlass.range_constexpr(self.sinkhorn_iters - 1):
                    row_sum = _warp_quad_allreduce_sum(cvalue) + Float32(self.hc_eps)
                    cvalue = cvalue * cute.arch.rcp_approx(row_sum)
                    col_sum = _warp_column4_allreduce_sum(cvalue) + Float32(self.hc_eps)
                    cvalue = cvalue * cute.arch.rcp_approx(col_sum)

                if lane < Int32(16):
                    row = lane // Int32(4)
                    col = lane - row * Int32(4)
                    s_comb[row, col] = cvalue
                    if tile_group == Int32(0):
                        comb[token, row, col] = cvalue

                if const_expr(self.fuse_norm):
                    p0 = Float32(s_pre[0])
                    p1 = Float32(s_pre[1])
                    p2 = Float32(s_pre[2])
                    p3 = Float32(s_pre[3])
                    term = Float32(0.0)
                    if lane == Int32(0):
                        term = p0 * p0 * Float32(source_values[Int32(32)])
                    elif lane == Int32(1):
                        term = p1 * p1 * Float32(source_values[Int32(33)])
                    elif lane == Int32(2):
                        term = p2 * p2 * Float32(source_values[Int32(34)])
                    elif lane == Int32(3):
                        term = p3 * p3 * Float32(source_values[Int32(35)])
                    elif lane == Int32(4):
                        term = (
                            Float32(2.0) * p0 * p1 * Float32(source_values[Int32(36)])
                        )
                    elif lane == Int32(5):
                        term = (
                            Float32(2.0) * p0 * p2 * Float32(source_values[Int32(37)])
                        )
                    elif lane == Int32(6):
                        term = (
                            Float32(2.0) * p0 * p3 * Float32(source_values[Int32(38)])
                        )
                    elif lane == Int32(7):
                        term = (
                            Float32(2.0) * p1 * p2 * Float32(source_values[Int32(39)])
                        )
                    elif lane == Int32(8):
                        term = (
                            Float32(2.0) * p1 * p3 * Float32(source_values[Int32(40)])
                        )
                    elif lane == Int32(9):
                        term = (
                            Float32(2.0) * p2 * p3 * Float32(source_values[Int32(41)])
                        )
                    sy2 = _warp_allreduce_sum(term)
                    if lane == Int32(0):
                        s_post[0] = cute.math.rsqrt(
                            sy2 / Float32(self.hidden_size) + Float32(self.norm_eps),
                            fastmath=True,
                        )

        if tidx == Int32(0) and const_expr(not self.single_cta):
            total_sqsum = Float32(sums[0])
            mixes = cute.make_rmem_tensor(
                cute.make_layout((self.mixes,), stride=(1,)), Float32
            )
            for mix in cutlass.range_constexpr(_MIXES):
                mixes[mix] = Float32(sums[mix + 1])
            inv_rms = cute.math.rsqrt(
                total_sqsum / Float32(self.total_k) + Float32(self.rms_eps),
                fastmath=True,
            )
            for mix in cutlass.range_constexpr(_MIXES):
                mixes[mix] = mixes[mix] * inv_rms

            s0 = Float32(scale[0])
            s1 = Float32(scale[1])
            s2 = Float32(scale[2])
            one = Float32(1.0)
            two = Float32(2.0)
            eps = Float32(self.hc_eps)

            pre0 = (
                one
                / (
                    one
                    + cute.math.exp(-(mixes[0] * s0 + Float32(bias[0])), fastmath=True)
                )
                + eps
            )
            pre1 = (
                one
                / (
                    one
                    + cute.math.exp(-(mixes[1] * s0 + Float32(bias[1])), fastmath=True)
                )
                + eps
            )
            pre2 = (
                one
                / (
                    one
                    + cute.math.exp(-(mixes[2] * s0 + Float32(bias[2])), fastmath=True)
                )
                + eps
            )
            pre3 = (
                one
                / (
                    one
                    + cute.math.exp(-(mixes[3] * s0 + Float32(bias[3])), fastmath=True)
                )
                + eps
            )
            s_pre[0] = pre0
            s_pre[1] = pre1
            s_pre[2] = pre2
            s_pre[3] = pre3

            post0 = two / (
                one + cute.math.exp(-(mixes[4] * s1 + Float32(bias[4])), fastmath=True)
            )
            post1 = two / (
                one + cute.math.exp(-(mixes[5] * s1 + Float32(bias[5])), fastmath=True)
            )
            post2 = two / (
                one + cute.math.exp(-(mixes[6] * s1 + Float32(bias[6])), fastmath=True)
            )
            post3 = two / (
                one + cute.math.exp(-(mixes[7] * s1 + Float32(bias[7])), fastmath=True)
            )
            if tile_group == Int32(0):
                post[token, 0] = post0
                post[token, 1] = post1
                post[token, 2] = post2
                post[token, 3] = post3

            c00 = mixes[8] * s2 + Float32(bias[8])
            c01 = mixes[9] * s2 + Float32(bias[9])
            c02 = mixes[10] * s2 + Float32(bias[10])
            c03 = mixes[11] * s2 + Float32(bias[11])
            c10 = mixes[12] * s2 + Float32(bias[12])
            c11 = mixes[13] * s2 + Float32(bias[13])
            c12 = mixes[14] * s2 + Float32(bias[14])
            c13 = mixes[15] * s2 + Float32(bias[15])
            c20 = mixes[16] * s2 + Float32(bias[16])
            c21 = mixes[17] * s2 + Float32(bias[17])
            c22 = mixes[18] * s2 + Float32(bias[18])
            c23 = mixes[19] * s2 + Float32(bias[19])
            c30 = mixes[20] * s2 + Float32(bias[20])
            c31 = mixes[21] * s2 + Float32(bias[21])
            c32 = mixes[22] * s2 + Float32(bias[22])
            c33 = mixes[23] * s2 + Float32(bias[23])

            m0 = c00
            if c01 > m0:
                m0 = c01
            if c02 > m0:
                m0 = c02
            if c03 > m0:
                m0 = c03
            m1 = c10
            if c11 > m1:
                m1 = c11
            if c12 > m1:
                m1 = c12
            if c13 > m1:
                m1 = c13
            m2 = c20
            if c21 > m2:
                m2 = c21
            if c22 > m2:
                m2 = c22
            if c23 > m2:
                m2 = c23
            m3 = c30
            if c31 > m3:
                m3 = c31
            if c32 > m3:
                m3 = c32
            if c33 > m3:
                m3 = c33

            c00 = cute.math.exp(c00 - m0, fastmath=True)
            c01 = cute.math.exp(c01 - m0, fastmath=True)
            c02 = cute.math.exp(c02 - m0, fastmath=True)
            c03 = cute.math.exp(c03 - m0, fastmath=True)
            c10 = cute.math.exp(c10 - m1, fastmath=True)
            c11 = cute.math.exp(c11 - m1, fastmath=True)
            c12 = cute.math.exp(c12 - m1, fastmath=True)
            c13 = cute.math.exp(c13 - m1, fastmath=True)
            c20 = cute.math.exp(c20 - m2, fastmath=True)
            c21 = cute.math.exp(c21 - m2, fastmath=True)
            c22 = cute.math.exp(c22 - m2, fastmath=True)
            c23 = cute.math.exp(c23 - m2, fastmath=True)
            c30 = cute.math.exp(c30 - m3, fastmath=True)
            c31 = cute.math.exp(c31 - m3, fastmath=True)
            c32 = cute.math.exp(c32 - m3, fastmath=True)
            c33 = cute.math.exp(c33 - m3, fastmath=True)

            r0s = c00 + c01 + c02 + c03
            r1s = c10 + c11 + c12 + c13
            r2s = c20 + c21 + c22 + c23
            r3s = c30 + c31 + c32 + c33
            inv_r0 = cute.arch.rcp_approx(r0s)
            inv_r1 = cute.arch.rcp_approx(r1s)
            inv_r2 = cute.arch.rcp_approx(r2s)
            inv_r3 = cute.arch.rcp_approx(r3s)
            c00 = c00 * inv_r0 + eps
            c01 = c01 * inv_r0 + eps
            c02 = c02 * inv_r0 + eps
            c03 = c03 * inv_r0 + eps
            c10 = c10 * inv_r1 + eps
            c11 = c11 * inv_r1 + eps
            c12 = c12 * inv_r1 + eps
            c13 = c13 * inv_r1 + eps
            c20 = c20 * inv_r2 + eps
            c21 = c21 * inv_r2 + eps
            c22 = c22 * inv_r2 + eps
            c23 = c23 * inv_r2 + eps
            c30 = c30 * inv_r3 + eps
            c31 = c31 * inv_r3 + eps
            c32 = c32 * inv_r3 + eps
            c33 = c33 * inv_r3 + eps

            col0 = c00 + c10 + c20 + c30 + eps
            col1 = c01 + c11 + c21 + c31 + eps
            col2 = c02 + c12 + c22 + c32 + eps
            col3 = c03 + c13 + c23 + c33 + eps
            inv_col0 = cute.arch.rcp_approx(col0)
            inv_col1 = cute.arch.rcp_approx(col1)
            inv_col2 = cute.arch.rcp_approx(col2)
            inv_col3 = cute.arch.rcp_approx(col3)
            c00 = c00 * inv_col0
            c10 = c10 * inv_col0
            c20 = c20 * inv_col0
            c30 = c30 * inv_col0
            c01 = c01 * inv_col1
            c11 = c11 * inv_col1
            c21 = c21 * inv_col1
            c31 = c31 * inv_col1
            c02 = c02 * inv_col2
            c12 = c12 * inv_col2
            c22 = c22 * inv_col2
            c32 = c32 * inv_col2
            c03 = c03 * inv_col3
            c13 = c13 * inv_col3
            c23 = c23 * inv_col3
            c33 = c33 * inv_col3

            for _ in cutlass.range_constexpr(self.sinkhorn_iters - 1):
                r0s = c00 + c01 + c02 + c03 + eps
                r1s = c10 + c11 + c12 + c13 + eps
                r2s = c20 + c21 + c22 + c23 + eps
                r3s = c30 + c31 + c32 + c33 + eps
                inv_r0 = cute.arch.rcp_approx(r0s)
                inv_r1 = cute.arch.rcp_approx(r1s)
                inv_r2 = cute.arch.rcp_approx(r2s)
                inv_r3 = cute.arch.rcp_approx(r3s)
                c00 = c00 * inv_r0
                c01 = c01 * inv_r0
                c02 = c02 * inv_r0
                c03 = c03 * inv_r0
                c10 = c10 * inv_r1
                c11 = c11 * inv_r1
                c12 = c12 * inv_r1
                c13 = c13 * inv_r1
                c20 = c20 * inv_r2
                c21 = c21 * inv_r2
                c22 = c22 * inv_r2
                c23 = c23 * inv_r2
                c30 = c30 * inv_r3
                c31 = c31 * inv_r3
                c32 = c32 * inv_r3
                c33 = c33 * inv_r3

                col0 = c00 + c10 + c20 + c30 + eps
                col1 = c01 + c11 + c21 + c31 + eps
                col2 = c02 + c12 + c22 + c32 + eps
                col3 = c03 + c13 + c23 + c33 + eps
                inv_col0 = cute.arch.rcp_approx(col0)
                inv_col1 = cute.arch.rcp_approx(col1)
                inv_col2 = cute.arch.rcp_approx(col2)
                inv_col3 = cute.arch.rcp_approx(col3)
                c00 = c00 * inv_col0
                c10 = c10 * inv_col0
                c20 = c20 * inv_col0
                c30 = c30 * inv_col0
                c01 = c01 * inv_col1
                c11 = c11 * inv_col1
                c21 = c21 * inv_col1
                c31 = c31 * inv_col1
                c02 = c02 * inv_col2
                c12 = c12 * inv_col2
                c22 = c22 * inv_col2
                c32 = c32 * inv_col2
                c03 = c03 * inv_col3
                c13 = c13 * inv_col3
                c23 = c23 * inv_col3
                c33 = c33 * inv_col3

            if tile_group == Int32(0):
                comb[token, 0, 0] = c00
                comb[token, 0, 1] = c01
                comb[token, 0, 2] = c02
                comb[token, 0, 3] = c03
                comb[token, 1, 0] = c10
                comb[token, 1, 1] = c11
                comb[token, 1, 2] = c12
                comb[token, 1, 3] = c13
                comb[token, 2, 0] = c20
                comb[token, 2, 1] = c21
                comb[token, 2, 2] = c22
                comb[token, 2, 3] = c23
                comb[token, 3, 0] = c30
                comb[token, 3, 1] = c31
                comb[token, 3, 2] = c32
                comb[token, 3, 3] = c33

            if const_expr(self.fuse_norm):
                # sum_h y^2 = pre^T G pre  (G symmetric, 10 packed entries)
                sy2 = (
                    pre0 * pre0 * Float32(gram[0])
                    + pre1 * pre1 * Float32(gram[1])
                    + pre2 * pre2 * Float32(gram[2])
                    + pre3 * pre3 * Float32(gram[3])
                    + two
                    * (
                        pre0 * pre1 * Float32(gram[4])
                        + pre0 * pre2 * Float32(gram[5])
                        + pre0 * pre3 * Float32(gram[6])
                        + pre1 * pre2 * Float32(gram[7])
                        + pre1 * pre3 * Float32(gram[8])
                        + pre2 * pre3 * Float32(gram[9])
                    )
                )
                s_post[0] = cute.math.rsqrt(
                    sy2 / Float32(self.hidden_size) + Float32(self.norm_eps),
                    fastmath=True,
                )

        cute.arch.sync_threads()

        p0 = Float32(s_pre[0])
        p1 = Float32(s_pre[1])
        p2 = Float32(s_pre[2])
        p3 = Float32(s_pre[3])
        if const_expr(self.compact_partials or self.single_cta):
            y_u32 = cute.recast_tensor(y, Uint32)
            first_pair = Int32(0)
            if const_expr(self.single_cta and self.single_cta_groups > 1):
                first_pair = tile_group * Int32(
                    self.hidden_size // (2 * self.single_cta_groups)
                )
            for pair_iter in cutlass.range_constexpr(
                self.hidden_size // (2 * self.num_threads * self.single_cta_groups)
            ):
                h = (first_pair + Int32(pair_iter * self.num_threads) + tidx) * Int32(2)
                residual_base = token * Int32(_MHC_MULT * self.hidden_size) + h
                ro0_pair = ld_global_nc_u32(get_ptr_as_int64(residual, residual_base))
                ro1_pair = ld_global_nc_u32(
                    get_ptr_as_int64(
                        residual,
                        residual_base + Int32(self.hidden_size),
                    )
                )
                ro2_pair = ld_global_nc_u32(
                    get_ptr_as_int64(
                        residual,
                        residual_base + Int32(2 * self.hidden_size),
                    )
                )
                ro3_pair = ld_global_nc_u32(
                    get_ptr_as_int64(
                        residual,
                        residual_base + Int32(3 * self.hidden_size),
                    )
                )
                ro00, ro01 = bfloat2_to_float2_scaled(ro0_pair, Float32(1.0))
                ro10, ro11 = bfloat2_to_float2_scaled(ro1_pair, Float32(1.0))
                ro20, ro21 = bfloat2_to_float2_scaled(ro2_pair, Float32(1.0))
                ro30, ro31 = bfloat2_to_float2_scaled(ro3_pair, Float32(1.0))
                norm0 = Float32(1.0)
                norm1 = Float32(1.0)
                if const_expr(self.fuse_norm):
                    if const_expr(norm_weight.element_type == cutlass.BFloat16):
                        norm_pair = ld_global_nc_u32(get_ptr_as_int64(norm_weight, h))
                        norm0, norm1 = bfloat2_to_float2_scaled(
                            norm_pair,
                            Float32(1.0),
                        )
                    else:
                        norm0 = Float32(norm_weight[h])
                        norm1 = Float32(norm_weight[h + Int32(1)])
                y_pre0 = (p0 * ro00 + p1 * ro10 + p2 * ro20 + p3 * ro30).to(
                    cutlass.BFloat16
                )
                y_pre1 = (p0 * ro01 + p1 * ro11 + p2 * ro21 + p3 * ro31).to(
                    cutlass.BFloat16
                )
                y0 = Float32(y_pre0)
                y1 = Float32(y_pre1)
                if const_expr(self.fuse_norm):
                    rms = Float32(s_post[0])
                    y0 = y0 * rms * norm0
                    y1 = y1 * rms * norm1
                y_u32[token, h // Int32(2)] = pack_f32x2_to_bfloat2(y0, y1)
        else:
            first_tile_h = tile_group * Int32(self.tiles_per_cta)
            for tile_iter in cutlass.range_constexpr(self.tiles_per_cta):
                h = (first_tile_h + Int32(tile_iter)) * Int32(self.block_h) + tidx
                ro0 = Float32(residual[token, 0, h])
                ro1 = Float32(residual[token, 1, h])
                ro2 = Float32(residual[token, 2, h])
                ro3 = Float32(residual[token, 3, h])
                # Round y_prenorm to bf16 before applying the norm, matching the
                # reference (and vLLM), so the only difference is the (negligible)
                # fp32-vs-bf16 sum-of-squares used for rms.
                y_pre = (p0 * ro0 + p1 * ro1 + p2 * ro2 + p3 * ro3).to(cutlass.BFloat16)
                if const_expr(self.fuse_norm):
                    rms = Float32(s_post[0])
                    y[token, h] = (Float32(y_pre) * rms * Float32(norm_weight[h])).to(
                        cutlass.BFloat16
                    )
                else:
                    y[token, h] = y_pre


@lru_cache(maxsize=64)
def _post_pre_partial_kernel(
    hidden_size: int,
    split_k: int,
    compute_gram: bool = False,
    pre_only: bool = False,
    post_only: bool = False,
    partials_per_cta: int = _POST_PRE_PARTIALS_PER_CTA,
) -> MHCPostPrePartialKernel:
    return MHCPostPrePartialKernel(
        hidden_size=hidden_size,
        split_k=split_k,
        compute_gram=compute_gram,
        pre_only=pre_only,
        post_only=post_only,
        partials_per_cta=partials_per_cta,
    )


@lru_cache(maxsize=32)
def _post_pre_decode_split_n_partial_kernel(
    hidden_size: int,
    split_k: int,
    source_splits: int,
    tile_n: int,
    bf16x2: bool = False,
    compute_gram: bool = False,
) -> MHCPostPreDecodeSplitNPartialKernel:
    return MHCPostPreDecodeSplitNPartialKernel(
        hidden_size=hidden_size,
        split_k=split_k,
        source_splits=source_splits,
        tile_n=tile_n,
        bf16x2=bf16x2,
        compute_gram=compute_gram,
    )


@lru_cache(maxsize=16)
def _post_pre_prefill_partial_kernel(
    hidden_size: int,
    split_k: int,
    compute_gram: bool = True,
) -> MHCPostPrePrefillPartialKernel:
    return MHCPostPrePrefillPartialKernel(
        hidden_size=hidden_size,
        split_k=split_k,
        compute_gram=compute_gram,
    )


@lru_cache(maxsize=16)
def _post_pre_prefill_block_m_partial_kernel(
    hidden_size: int,
    split_k: int,
    block_m: int = _PREFILL_BLOCK_M,
    tile_n: int = _PREFILL_BLOCK_TILE_N,
    compute_gram: bool = True,
) -> MHCPostPrePrefillBlockMPartialKernel:
    return MHCPostPrePrefillBlockMPartialKernel(
        hidden_size=hidden_size,
        split_k=split_k,
        block_m=block_m,
        tile_n=tile_n,
        compute_gram=compute_gram,
    )


@lru_cache(maxsize=16)
def _post_pre_prefill_gram_kernel(
    hidden_size: int,
    split_k: int,
) -> MHCPostPrePrefillGramKernel:
    return MHCPostPrePrefillGramKernel(
        hidden_size=hidden_size,
        split_k=split_k,
    )


@lru_cache(maxsize=16)
def _prefill_bf16_project_kernel(
    hidden_size: int,
    split_k: int,
) -> MHCPrefillBf16ProjectKernel:
    return MHCPrefillBf16ProjectKernel(
        hidden_size=hidden_size,
        split_k=split_k,
    )


@lru_cache(maxsize=16)
def _prefill_bf16_project_tma_kernel(
    hidden_size: int,
    split_k: int,
) -> MHCPrefillBf16ProjectTmaKernel:
    return MHCPrefillBf16ProjectTmaKernel(
        hidden_size=hidden_size,
        split_k=split_k,
    )


@lru_cache(maxsize=16)
def _prefill_tf32_project_kernel(
    hidden_size: int,
    split_k: int,
    chunk_geometry: bool = False,
    long_geometry: bool = False,
) -> MHCPrefillTf32ProjectTmaKernel:
    return MHCPrefillTf32ProjectTmaKernel(
        hidden_size=hidden_size,
        split_k=split_k,
        chunk_geometry=chunk_geometry,
        long_geometry=long_geometry,
    )


@lru_cache(maxsize=64)
def _finalize_gram_kernel(
    hidden_size: int,
    split_k: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
    fuse_norm: bool,
    compact_partials: bool = False,
    compact_projection_splits: int = 1,
    single_cta: bool = False,
    single_cta_threads: int = _PREFILL_FINALIZE_THREADS,
    single_cta_groups: int = 1,
    active_source_splits: int = 0,
) -> MHCFinalizeGramKernel:
    return MHCFinalizeGramKernel(
        hidden_size=hidden_size,
        split_k=split_k,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        norm_eps=norm_eps,
        fuse_norm=fuse_norm,
        compact_partials=compact_partials,
        compact_projection_splits=compact_projection_splits,
        single_cta=single_cta,
        single_cta_threads=single_cta_threads,
        single_cta_groups=single_cta_groups,
        active_source_splits=active_source_splits,
    )


def _run_mhc_post_pre_partial_launch(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
    compute_gram: bool = False,
) -> None:
    tokens = int(x.shape[0])
    hidden_size = int(residual.shape[2])
    split_k = int(partials.shape[1])
    _validate_split_k(hidden_size, split_k)
    decode_source_splits, decode_tile_n = _selected_post_pre_decode_split_n(
        num_tokens=tokens,
        hidden_size=hidden_size,
    )
    raw_bf16x2 = os.environ.get("FLASHINFER_EXP_SM12X_MHC_DECODE_BF16X2")
    decode_bf16x2 = (
        raw_bf16x2 != "0"
        if raw_bf16x2 is not None
        else tokens == 16 and hidden_size == _HIDDEN and decode_source_splits > 0
    )
    partials_per_cta = _selected_post_pre_partials_per_cta(
        num_tokens=tokens,
        hidden_size=hidden_size,
    )
    _validate_tensor_shape("x", x, (tokens, hidden_size))
    _validate_tensor_shape("residual", residual, (tokens, _MHC_MULT, hidden_size))
    _validate_tensor_shape("prev_post", prev_post, (tokens, _MHC_MULT))
    _validate_tensor_shape("prev_comb", prev_comb, (tokens, _MHC_MULT, _MHC_MULT))
    _validate_tensor_shape("fn", fn, (_MIXES, _MHC_MULT * hidden_size))
    _validate_tensor_shape("partials", partials, (tokens, split_k, _PARTIALS))
    _validate_tensor_shape("out", out, (tokens, _MHC_MULT, hidden_size))
    decode_bf16x2 = decode_bf16x2 and all(
        tensor.is_contiguous() and tensor.data_ptr() % 4 == 0
        for tensor in (x, residual, out)
    )
    compute_gram = bool(compute_gram)
    args = (
        _to_kernel_tensor(x, cutlass.BFloat16, dynamic_layout=True),
        _to_kernel_tensor(residual, cutlass.BFloat16, dynamic_layout=True),
        _to_kernel_tensor(
            prev_post,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        _to_kernel_tensor(
            prev_comb,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        _to_kernel_tensor(fn, cutlass.Float32),
        _to_kernel_tensor(
            partials,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        _to_kernel_tensor(out, cutlass.BFloat16, dynamic_layout=True),
        Int32(tokens),
        current_cuda_stream(),
    )
    cache_key = (
        tensor_key(
            "x",
            x,
            dims=(DimKey.dynamic(), DimKey.exact(hidden_size)),
        ),
        tensor_key(
            "residual",
            residual,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(_MHC_MULT),
                DimKey.exact(hidden_size),
            ),
        ),
        tensor_key(
            "prev_post",
            prev_post,
            dims=(DimKey.dynamic(), DimKey.exact(_MHC_MULT)),
        ),
        tensor_key(
            "prev_comb",
            prev_comb,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(_MHC_MULT),
                DimKey.exact(_MHC_MULT),
            ),
        ),
        tensor_key(
            "fn",
            fn,
            dims=(DimKey.exact(_MIXES), DimKey.exact(_MHC_MULT * hidden_size)),
        ),
        tensor_key(
            "partials",
            partials,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(split_k),
                DimKey.exact(_PARTIALS),
            ),
        ),
        tensor_key(
            "out",
            out,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(_MHC_MULT),
                DimKey.exact(hidden_size),
            ),
        ),
    )
    hidden_specialization = _hidden_specialization_name(hidden_size)
    if decode_source_splits > 0:
        compile_name = (
            "integration.residual.mhc_post_pre_decode_split_n_"
            f"{hidden_specialization}_s{decode_source_splits}n{decode_tile_n}"
            f"x{2 if decode_bf16x2 else 1}"
        )
        compile_key = (
            ("hidden_size", hidden_size),
            ("split_k", split_k),
            ("source_splits", decode_source_splits),
            ("tile_n", decode_tile_n),
            ("bf16x2", decode_bf16x2),
            ("compute_gram", compute_gram),
            ("pdl", _MHC_PDL),
            cache_key,
        )
        kernel = _post_pre_decode_split_n_partial_kernel(
            hidden_size,
            split_k,
            decode_source_splits,
            decode_tile_n,
            decode_bf16x2,
            compute_gram,
        )
    elif hidden_size == _HIDDEN and split_k == _SPLIT_K:
        compile_name = (
            "integration.residual.mhc_post_pre_partial_hidden4096_hctile128"
            f"_all{partials_per_cta}"
        )
        compile_key = (
            ("partials_per_cta", partials_per_cta),
            ("compute_gram", compute_gram),
            ("pdl", _MHC_PDL),
            cache_key,
        )
        kernel = _post_pre_partial_kernel(
            hidden_size,
            split_k,
            compute_gram,
            False,
            False,
            partials_per_cta,
        )
    else:
        compile_name = (
            "integration.residual.mhc_post_pre_partial_"
            f"{hidden_specialization}_hctile128_all{partials_per_cta}"
        )
        compile_key = (
            ("hidden_size", hidden_size),
            ("split_k", split_k),
            ("source_tiles", hidden_size // _SOURCE_TILE_H),
            ("partials_per_cta", partials_per_cta),
            ("compute_gram", compute_gram),
            ("pdl", _MHC_PDL),
            cache_key,
        )
        kernel = _post_pre_partial_kernel(
            hidden_size,
            split_k,
            compute_gram,
            False,
            False,
            partials_per_cta,
        )
    sm12x_launch(
        kernel,
        compile_spec=KernelCompileSpec.from_key(compile_name, 4, compile_key),
        compile_args=args,
        runtime_args=args,
    )


@torch.library.custom_op(
    "flashinfer_sm12x::mhc_post_pre_partial_launch",
    mutates_args=("partials", "out"),
)
def _mhc_post_pre_partial_launch_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
    compute_gram: bool,
) -> None:
    _run_mhc_post_pre_partial_launch(
        x=x,
        residual=residual,
        prev_post=prev_post,
        prev_comb=prev_comb,
        fn=fn,
        partials=partials,
        out=out,
        compute_gram=compute_gram,
    )


@_mhc_post_pre_partial_launch_op.register_fake
def _mhc_post_pre_partial_launch_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
    compute_gram: bool,
) -> None:
    return None


def run_mhc_post_pre_partial(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
    compute_gram: bool = False,
) -> None:
    torch.ops.flashinfer_sm12x.mhc_post_pre_partial_launch(
        x,
        residual,
        prev_post,
        prev_comb,
        fn,
        partials,
        out,
        bool(compute_gram),
    )


def _run_mhc_post_pre_prefill_partial_launch(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
    compute_gram: bool = True,
) -> None:
    tokens = int(x.shape[0])
    hidden_size = int(residual.shape[2])
    split_k = int(partials.shape[1])
    _validate_split_k(hidden_size, split_k)
    _validate_tensor_shape("x", x, (tokens, hidden_size))
    _validate_tensor_shape("residual", residual, (tokens, _MHC_MULT, hidden_size))
    _validate_tensor_shape("prev_post", prev_post, (tokens, _MHC_MULT))
    _validate_tensor_shape("prev_comb", prev_comb, (tokens, _MHC_MULT, _MHC_MULT))
    _validate_tensor_shape("fn", fn, (_MIXES, _MHC_MULT * hidden_size))
    _validate_tensor_shape("partials", partials, (tokens, split_k, _PARTIALS))
    _validate_tensor_shape("out", out, (tokens, _MHC_MULT, hidden_size))
    compute_gram = bool(compute_gram)
    args = (
        _to_kernel_tensor(x, cutlass.BFloat16, dynamic_layout=True),
        _to_kernel_tensor(residual, cutlass.BFloat16, dynamic_layout=True),
        _to_kernel_tensor(
            prev_post,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        _to_kernel_tensor(
            prev_comb,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        _to_kernel_tensor(fn, cutlass.Float32),
        _to_kernel_tensor(
            partials,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        _to_kernel_tensor(out, cutlass.BFloat16, dynamic_layout=True),
        Int32(tokens),
        current_cuda_stream(),
    )
    cache_key = (
        tensor_key(
            "x",
            x,
            dims=(DimKey.dynamic(), DimKey.exact(hidden_size)),
        ),
        tensor_key(
            "residual",
            residual,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(_MHC_MULT),
                DimKey.exact(hidden_size),
            ),
        ),
        tensor_key(
            "prev_post",
            prev_post,
            dims=(DimKey.dynamic(), DimKey.exact(_MHC_MULT)),
        ),
        tensor_key(
            "prev_comb",
            prev_comb,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(_MHC_MULT),
                DimKey.exact(_MHC_MULT),
            ),
        ),
        tensor_key(
            "fn",
            fn,
            dims=(DimKey.exact(_MIXES), DimKey.exact(_MHC_MULT * hidden_size)),
        ),
        tensor_key(
            "partials",
            partials,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(split_k),
                DimKey.exact(_PARTIALS),
            ),
        ),
        tensor_key(
            "out",
            out,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(_MHC_MULT),
                DimKey.exact(hidden_size),
            ),
        ),
    )
    hidden_specialization = _hidden_specialization_name(hidden_size)
    compile_name = (
        "integration.residual.mhc_post_pre_prefill_partial_"
        f"{hidden_specialization}_threads{_PREFILL_THREADS}"
    )
    compile_key = (
        ("hidden_size", hidden_size),
        ("split_k", split_k),
        ("threads", _PREFILL_THREADS),
        ("compute_gram", compute_gram),
        cache_key,
    )
    sm12x_launch(
        _post_pre_prefill_partial_kernel(
            hidden_size,
            split_k,
            compute_gram,
        ),
        compile_spec=KernelCompileSpec.from_key(compile_name, 3, compile_key),
        compile_args=args,
        runtime_args=args,
    )


@torch.library.custom_op(
    "flashinfer_sm12x::mhc_post_pre_prefill_partial_launch",
    mutates_args=("partials", "out"),
)
def _mhc_post_pre_prefill_partial_launch_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
    compute_gram: bool,
) -> None:
    _run_mhc_post_pre_prefill_partial_launch(
        x=x,
        residual=residual,
        prev_post=prev_post,
        prev_comb=prev_comb,
        fn=fn,
        partials=partials,
        out=out,
        compute_gram=compute_gram,
    )


@_mhc_post_pre_prefill_partial_launch_op.register_fake
def _mhc_post_pre_prefill_partial_launch_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
    compute_gram: bool,
) -> None:
    return None


def run_mhc_post_pre_prefill_partial(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
    compute_gram: bool = True,
) -> None:
    torch.ops.flashinfer_sm12x.mhc_post_pre_prefill_partial_launch(
        x,
        residual,
        prev_post,
        prev_comb,
        fn,
        partials,
        out,
        bool(compute_gram),
    )


def _run_mhc_post_pre_prefill_block_m_partial_launch(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
    compute_gram: bool = True,
    block_m: int = _PREFILL_BLOCK_M,
    tile_n: int = _PREFILL_BLOCK_TILE_N,
) -> None:
    tokens = int(x.shape[0])
    hidden_size = int(residual.shape[2])
    split_k = int(partials.shape[1])
    block_m = int(block_m)
    tile_n = int(tile_n)
    _validate_split_k(hidden_size, split_k)
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    if tile_n <= 0:
        raise ValueError(f"tile_n must be positive, got {tile_n}")
    _validate_tensor_shape("x", x, (tokens, hidden_size))
    _validate_tensor_shape("residual", residual, (tokens, _MHC_MULT, hidden_size))
    _validate_tensor_shape("prev_post", prev_post, (tokens, _MHC_MULT))
    _validate_tensor_shape("prev_comb", prev_comb, (tokens, _MHC_MULT, _MHC_MULT))
    _validate_tensor_shape("fn", fn, (_MIXES, _MHC_MULT * hidden_size))
    _validate_tensor_shape("partials", partials, (tokens, split_k, _PARTIALS))
    _validate_tensor_shape("out", out, (tokens, _MHC_MULT, hidden_size))
    compute_gram = bool(compute_gram)
    args = (
        _to_kernel_tensor(x, cutlass.BFloat16, dynamic_layout=True),
        _to_kernel_tensor(residual, cutlass.BFloat16, dynamic_layout=True),
        _to_kernel_tensor(
            prev_post,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        _to_kernel_tensor(
            prev_comb,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        _to_kernel_tensor(fn, cutlass.Float32),
        _to_kernel_tensor(
            partials,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        _to_kernel_tensor(out, cutlass.BFloat16, dynamic_layout=True),
        Int32(tokens),
        current_cuda_stream(),
    )
    cache_key = (
        tensor_key(
            "x",
            x,
            dims=(DimKey.dynamic(), DimKey.exact(hidden_size)),
        ),
        tensor_key(
            "residual",
            residual,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(_MHC_MULT),
                DimKey.exact(hidden_size),
            ),
        ),
        tensor_key(
            "prev_post",
            prev_post,
            dims=(DimKey.dynamic(), DimKey.exact(_MHC_MULT)),
        ),
        tensor_key(
            "prev_comb",
            prev_comb,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(_MHC_MULT),
                DimKey.exact(_MHC_MULT),
            ),
        ),
        tensor_key(
            "fn",
            fn,
            dims=(DimKey.exact(_MIXES), DimKey.exact(_MHC_MULT * hidden_size)),
        ),
        tensor_key(
            "partials",
            partials,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(split_k),
                DimKey.exact(_PARTIALS),
            ),
        ),
        tensor_key(
            "out",
            out,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(_MHC_MULT),
                DimKey.exact(hidden_size),
            ),
        ),
    )
    hidden_specialization = _hidden_specialization_name(hidden_size)
    compile_name = (
        "integration.residual.mhc_post_pre_prefill_block_m_partial_"
        f"{hidden_specialization}_threads{_PREFILL_THREADS}_m{block_m}_n{tile_n}"
    )
    compile_key = (
        ("hidden_size", hidden_size),
        ("split_k", split_k),
        ("threads", _PREFILL_THREADS),
        ("block_m", block_m),
        ("tile_n", tile_n),
        ("compute_gram", compute_gram),
        cache_key,
    )
    sm12x_launch(
        _post_pre_prefill_block_m_partial_kernel(
            hidden_size,
            split_k,
            block_m,
            tile_n,
            compute_gram,
        ),
        compile_spec=KernelCompileSpec.from_key(compile_name, 3, compile_key),
        compile_args=args,
        runtime_args=args,
    )


@torch.library.custom_op(
    "flashinfer_sm12x::mhc_post_pre_prefill_block_m_partial_launch",
    mutates_args=("partials", "out"),
)
def _mhc_post_pre_prefill_block_m_partial_launch_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
    compute_gram: bool,
    block_m: int,
    tile_n: int,
) -> None:
    _run_mhc_post_pre_prefill_block_m_partial_launch(
        x=x,
        residual=residual,
        prev_post=prev_post,
        prev_comb=prev_comb,
        fn=fn,
        partials=partials,
        out=out,
        compute_gram=compute_gram,
        block_m=block_m,
        tile_n=tile_n,
    )


@_mhc_post_pre_prefill_block_m_partial_launch_op.register_fake
def _mhc_post_pre_prefill_block_m_partial_launch_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
    compute_gram: bool,
    block_m: int,
    tile_n: int,
) -> None:
    return None


def run_mhc_post_pre_prefill_block_m_partial(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
    compute_gram: bool = True,
    block_m: int = _PREFILL_BLOCK_M,
    tile_n: int = _PREFILL_BLOCK_TILE_N,
) -> None:
    torch.ops.flashinfer_sm12x.mhc_post_pre_prefill_block_m_partial_launch(
        x,
        residual,
        prev_post,
        prev_comb,
        fn,
        partials,
        out,
        bool(compute_gram),
        int(block_m),
        int(tile_n),
    )


def _run_mhc_post_pre_prefill_gram_launch(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
) -> None:
    tokens = int(x.shape[0])
    hidden_size = int(residual.shape[2])
    split_k = int(partials.shape[1])
    _validate_split_k(hidden_size, split_k)
    _validate_tensor_shape("x", x, (tokens, hidden_size))
    _validate_tensor_shape("residual", residual, (tokens, _MHC_MULT, hidden_size))
    _validate_tensor_shape("prev_post", prev_post, (tokens, _MHC_MULT))
    _validate_tensor_shape("prev_comb", prev_comb, (tokens, _MHC_MULT, _MHC_MULT))
    _validate_tensor_shape("partials", partials, (tokens, split_k, _PARTIALS))
    _validate_tensor_shape("out", out, (tokens, _MHC_MULT, hidden_size))
    args = (
        _to_kernel_tensor(x, cutlass.BFloat16, dynamic_layout=True),
        _to_kernel_tensor(residual, cutlass.BFloat16, dynamic_layout=True),
        _to_kernel_tensor(
            prev_post,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        _to_kernel_tensor(
            prev_comb,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        _to_kernel_tensor(
            partials,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        _to_kernel_tensor(out, cutlass.BFloat16, dynamic_layout=True),
        Int32(tokens),
        current_cuda_stream(),
    )
    cache_key = (
        tensor_key(
            "x",
            x,
            dims=(DimKey.dynamic(), DimKey.exact(hidden_size)),
        ),
        tensor_key(
            "residual",
            residual,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(_MHC_MULT),
                DimKey.exact(hidden_size),
            ),
        ),
        tensor_key(
            "prev_post",
            prev_post,
            dims=(DimKey.dynamic(), DimKey.exact(_MHC_MULT)),
        ),
        tensor_key(
            "prev_comb",
            prev_comb,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(_MHC_MULT),
                DimKey.exact(_MHC_MULT),
            ),
        ),
        tensor_key(
            "partials",
            partials,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(split_k),
                DimKey.exact(_PARTIALS),
            ),
        ),
        tensor_key(
            "out",
            out,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(_MHC_MULT),
                DimKey.exact(hidden_size),
            ),
        ),
    )
    hidden_specialization = _hidden_specialization_name(hidden_size)
    compile_name = (
        "integration.residual.mhc_post_pre_prefill_gram_"
        f"{hidden_specialization}_threads{_PREFILL_GRAM_THREADS}"
    )
    compile_key = (
        ("impl", "bf16x2_io_coeff_smem_v2"),
        ("hidden_size", hidden_size),
        ("split_k", split_k),
        ("threads", _PREFILL_GRAM_THREADS),
        ("pdl", _MHC_PDL),
        cache_key,
    )
    sm12x_launch(
        _post_pre_prefill_gram_kernel(hidden_size, split_k),
        compile_spec=KernelCompileSpec.from_key(compile_name, 2, compile_key),
        compile_args=args,
        runtime_args=args,
    )


@torch.library.custom_op(
    "flashinfer_sm12x::mhc_post_pre_prefill_gram_launch",
    mutates_args=("partials", "out"),
)
def _mhc_post_pre_prefill_gram_launch_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
) -> None:
    _run_mhc_post_pre_prefill_gram_launch(
        x=x,
        residual=residual,
        prev_post=prev_post,
        prev_comb=prev_comb,
        partials=partials,
        out=out,
    )


@_mhc_post_pre_prefill_gram_launch_op.register_fake
def _mhc_post_pre_prefill_gram_launch_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
) -> None:
    return None


def run_mhc_post_pre_prefill_gram(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
) -> None:
    torch.ops.flashinfer_sm12x.mhc_post_pre_prefill_gram_launch(
        x,
        residual,
        prev_post,
        prev_comb,
        partials,
        out,
    )


def _run_mhc_prefill_bf16_project_launch(
    *,
    out: torch.Tensor,
    fn_bf16: torch.Tensor,
    partials: torch.Tensor,
) -> None:
    tokens = int(out.shape[0])
    hidden_size = int(out.shape[2])
    split_k = int(partials.shape[1])
    _validate_split_k(hidden_size, split_k)
    _validate_tensor_shape("out", out, (tokens, _MHC_MULT, hidden_size))
    _validate_tensor_shape("fn_bf16", fn_bf16, (_MIXES, _MHC_MULT * hidden_size))
    _validate_tensor_shape("partials", partials, (tokens, split_k, _PARTIALS))
    if fn_bf16.dtype != torch.bfloat16:
        raise ValueError(f"fn_bf16 must be torch.bfloat16, got {fn_bf16.dtype}")
    if fn_bf16.device != out.device or partials.device != out.device:
        raise ValueError("fn_bf16, partials, and out must be on the same device")
    if not fn_bf16.is_cuda:
        raise ValueError("fn_bf16 must be CUDA")
    if not fn_bf16.is_contiguous():
        raise ValueError("fn_bf16 must be contiguous")
    if os.getenv("FLASHINFER_EXP_SM12X_MHC_PREFILL_BF16_TMA", "1") != "0":
        if not out.is_contiguous():
            raise ValueError("out must be contiguous for TMA BF16 prefill projection")
        out_flat = out.view(tokens, _MHC_MULT * hidden_size)
        args = (
            _to_kernel_tensor(out_flat, cutlass.BFloat16, dynamic_layout=True),
            _to_kernel_tensor(fn_bf16, cutlass.BFloat16),
            _to_kernel_tensor(
                partials,
                cutlass.Float32,
                assumed_align=4,
                dynamic_layout=True,
            ),
            Int32(tokens),
            current_cuda_stream(),
        )
        cache_key = (
            tensor_key(
                "out_flat",
                out_flat,
                dims=(DimKey.dynamic(), DimKey.exact(_MHC_MULT * hidden_size)),
            ),
            tensor_key(
                "fn_bf16",
                fn_bf16,
                dims=(DimKey.exact(_MIXES), DimKey.exact(_MHC_MULT * hidden_size)),
            ),
            tensor_key(
                "partials",
                partials,
                dims=(
                    DimKey.dynamic(),
                    DimKey.exact(split_k),
                    DimKey.exact(_PARTIALS),
                ),
            ),
        )
        hidden_specialization = _hidden_specialization_name(hidden_size)
        compile_name = (
            "integration.residual.mhc_prefill_bf16_project_tma_"
            f"{hidden_specialization}_m{_PREFILL_TMA_TILE_M}_n{_PREFILL_TMA_TILE_N}"
        )
        compile_key = (
            ("hidden_size", hidden_size),
            ("split_k", split_k),
            ("tile_m", _PREFILL_TMA_TILE_M),
            ("tile_n", _PREFILL_TMA_TILE_N),
            ("tile_k", _PREFILL_TMA_TILE_K),
            ("stages", _PREFILL_TMA_STAGES),
            ("compute_warps", _PREFILL_TMA_COMPUTE_WARPS),
            ("threads", _PREFILL_TMA_THREADS),
            cache_key,
        )
        sm12x_launch(
            _prefill_bf16_project_tma_kernel(hidden_size, split_k),
            compile_spec=KernelCompileSpec.from_key(compile_name, 1, compile_key),
            compile_args=args,
            runtime_args=args,
        )
        return
    args = (
        _to_kernel_tensor(out, cutlass.BFloat16, dynamic_layout=True),
        _to_kernel_tensor(fn_bf16, cutlass.BFloat16),
        _to_kernel_tensor(
            partials,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        Int32(tokens),
        current_cuda_stream(),
    )
    cache_key = (
        tensor_key(
            "out",
            out,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(_MHC_MULT),
                DimKey.exact(hidden_size),
            ),
        ),
        tensor_key(
            "fn_bf16",
            fn_bf16,
            dims=(DimKey.exact(_MIXES), DimKey.exact(_MHC_MULT * hidden_size)),
        ),
        tensor_key(
            "partials",
            partials,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(split_k),
                DimKey.exact(_PARTIALS),
            ),
        ),
    )
    hidden_specialization = _hidden_specialization_name(hidden_size)
    compile_name = (
        "integration.residual.mhc_prefill_bf16_project_"
        f"{hidden_specialization}_m{_PREFILL_MMA_TILE_M}_n{_PREFILL_MMA_TILE_N}"
    )
    compile_key = (
        ("hidden_size", hidden_size),
        ("split_k", split_k),
        ("tile_m", _PREFILL_MMA_TILE_M),
        ("tile_n", _PREFILL_MMA_TILE_N),
        ("tile_k", _PREFILL_MMA_TILE_K),
        ("sync", "warp_v2"),
        cache_key,
    )
    sm12x_launch(
        _prefill_bf16_project_kernel(hidden_size, split_k),
        compile_spec=KernelCompileSpec.from_key(compile_name, 1, compile_key),
        compile_args=args,
        runtime_args=args,
    )


@torch.library.custom_op(
    "flashinfer_sm12x::mhc_prefill_bf16_project_launch",
    mutates_args=("partials",),
)
def _mhc_prefill_bf16_project_launch_op(
    out: torch.Tensor,
    fn_bf16: torch.Tensor,
    partials: torch.Tensor,
) -> None:
    _run_mhc_prefill_bf16_project_launch(
        out=out,
        fn_bf16=fn_bf16,
        partials=partials,
    )


@_mhc_prefill_bf16_project_launch_op.register_fake
def _mhc_prefill_bf16_project_launch_fake(
    out: torch.Tensor,
    fn_bf16: torch.Tensor,
    partials: torch.Tensor,
) -> None:
    return None


def run_mhc_prefill_bf16_project(
    *,
    out: torch.Tensor,
    fn_bf16: torch.Tensor,
    partials: torch.Tensor,
) -> None:
    torch.ops.flashinfer_sm12x.mhc_prefill_bf16_project_launch(
        out,
        fn_bf16,
        partials,
    )


def _run_mhc_prefill_tf32_project_launch(
    *,
    out: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
) -> None:
    tokens = int(out.shape[0])
    hidden_size = int(out.shape[2])
    split_k = int(partials.shape[1])
    _validate_split_k(hidden_size, split_k)
    _validate_tensor_shape("out", out, (tokens, _MHC_MULT, hidden_size))
    _validate_tensor_shape("fn", fn, (_MIXES, _MHC_MULT * hidden_size))
    _validate_tensor_shape("partials", partials, (tokens, split_k, _PARTIALS))
    if fn.dtype != torch.float32:
        raise ValueError(f"fn must be torch.float32, got {fn.dtype}")
    if fn.device != out.device or partials.device != out.device:
        raise ValueError("fn, partials, and out must be on the same device")
    if not fn.is_cuda:
        raise ValueError("fn must be CUDA")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous for TF32 prefill projection")
    if not fn.is_contiguous():
        raise ValueError("fn must be contiguous")
    out_flat = out.view(tokens, _MHC_MULT * hidden_size)
    chunk_geometry = tokens >= _PREFILL_TF32_TMA_CHUNK_MIN_TOKENS
    long_geometry = (
        hidden_size == _HIDDEN and tokens >= _PREFILL_TF32_TMA_LONG_MIN_TOKENS
    )
    kernel = _prefill_tf32_project_kernel(
        hidden_size,
        split_k,
        chunk_geometry,
        long_geometry,
    )
    args = (
        _to_kernel_tensor(out_flat, cutlass.BFloat16, dynamic_layout=True),
        _to_kernel_tensor(fn, cutlass.Float32),
        _to_kernel_tensor(
            partials,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        Int32(tokens),
        current_cuda_stream(),
    )
    cache_key = (
        tensor_key(
            "out_flat",
            out_flat,
            dims=(DimKey.dynamic(), DimKey.exact(_MHC_MULT * hidden_size)),
        ),
        tensor_key(
            "fn",
            fn,
            dims=(DimKey.exact(_MIXES), DimKey.exact(_MHC_MULT * hidden_size)),
        ),
        tensor_key(
            "partials",
            partials,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(split_k),
                DimKey.exact(_PARTIALS),
            ),
        ),
    )
    hidden_specialization = _hidden_specialization_name(hidden_size)
    compile_name = (
        "integration.residual.mhc_prefill_tf32_project_tma_"
        f"{hidden_specialization}_m{kernel.tile_m}_n{kernel.tile_n}"
    )
    compile_key = (
        ("hidden_size", hidden_size),
        ("split_k", split_k),
        ("chunk_geometry", chunk_geometry),
        ("chunk_min_tokens", _PREFILL_TF32_TMA_CHUNK_MIN_TOKENS),
        ("long_geometry", long_geometry),
        ("long_min_tokens", _PREFILL_TF32_TMA_LONG_MIN_TOKENS),
        ("tile_m", kernel.tile_m),
        ("tile_n", kernel.tile_n),
        ("tile_k", kernel.tile_k),
        ("num_stages", kernel.num_stages),
        ("num_m_warps", kernel.num_m_warps),
        ("num_n_warps", kernel.num_n_warps),
        ("num_compute_warps", kernel.num_compute_warps),
        ("threads", kernel.num_threads),
        ("k_splits", kernel.k_splits),
        ("pdl", _MHC_PDL),
        (
            "operand_layout",
            "tma_swizzled_branchless_a_bf16_b_f32_rawtf32_m16n8k8_v7",
        ),
        cache_key,
    )
    sm12x_launch(
        kernel,
        compile_spec=KernelCompileSpec.from_key(compile_name, 6, compile_key),
        compile_args=args,
        runtime_args=args,
    )


def mhc_prefill_tf32_project_splits(*, tokens: int, hidden_size: int) -> int:
    """Return the projection split count selected by the TF32 prefill kernel."""
    chunk_geometry = int(tokens) >= _PREFILL_TF32_TMA_CHUNK_MIN_TOKENS
    long_geometry = (
        int(hidden_size) == _HIDDEN and int(tokens) >= _PREFILL_TF32_TMA_LONG_MIN_TOKENS
    )
    return _prefill_tf32_project_kernel(
        int(hidden_size),
        _split_k_for_hidden(int(hidden_size)),
        chunk_geometry,
        long_geometry,
    ).k_splits


@torch.library.custom_op(
    "flashinfer_sm12x::mhc_prefill_tf32_project_launch",
    mutates_args=("partials",),
)
def _mhc_prefill_tf32_project_launch_op(
    out: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
) -> None:
    _run_mhc_prefill_tf32_project_launch(
        out=out,
        fn=fn,
        partials=partials,
    )


@_mhc_prefill_tf32_project_launch_op.register_fake
def _mhc_prefill_tf32_project_launch_fake(
    out: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
) -> None:
    return None


def run_mhc_prefill_tf32_project(
    *,
    out: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
) -> None:
    torch.ops.flashinfer_sm12x.mhc_prefill_tf32_project_launch(
        out,
        fn,
        partials,
    )


@torch.library.custom_op(
    "flashinfer_sm12x::mhc_post_pre_partial_alloc",
    mutates_args=(),
)
def _mhc_post_pre_partial_alloc_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    compute_gram: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Functional (allocate + return) post-pre partial. Allocating BOTH partials
    # and residual_out internally and returning them gives this op ZERO mutated
    # args, so it is never auto_functionalized. That avoids the
    # decompose_auto_functionalized node-count assertion that fires for an
    # auto_functionalized op carrying TWO mutated args sharing a symbolic dim
    # (the second clone's as_strided sym_size gets CSE-collapsed on re-trace).
    # residual_out is allocated contiguous.
    tokens = int(residual.shape[0])
    hidden_size = int(residual.shape[2])
    split_k = _split_k_for_hidden(hidden_size)
    partials = torch.empty(
        (tokens, split_k, _PARTIALS), dtype=torch.float32, device=residual.device
    )
    out = torch.empty(residual.shape, dtype=residual.dtype, device=residual.device)
    _run_mhc_post_pre_partial_launch(
        x=x,
        residual=residual,
        prev_post=prev_post,
        prev_comb=prev_comb,
        fn=fn,
        partials=partials,
        out=out,
        compute_gram=compute_gram,
    )
    return partials, out


@_mhc_post_pre_partial_alloc_op.register_fake
def _mhc_post_pre_partial_alloc_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    compute_gram: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = residual.shape[0]
    split_k = residual.shape[2] // (_SOURCE_TILE_H // 2)
    partials = torch.empty(
        (tokens, split_k, _PARTIALS), dtype=torch.float32, device=residual.device
    )
    out = torch.empty(residual.shape, dtype=residual.dtype, device=residual.device)
    return partials, out


def run_mhc_post_pre_partial_alloc(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    compute_gram: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.flashinfer_sm12x.mhc_post_pre_partial_alloc(
        x, residual, prev_post, prev_comb, fn, bool(compute_gram)
    )


def _run_mhc_post_launch(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    out: torch.Tensor,
) -> None:
    tokens = int(x.shape[0])
    hidden_size = int(residual.shape[2])
    split_k = _split_k_for_hidden(hidden_size)
    _validate_tensor_shape("x", x, (tokens, hidden_size))
    _validate_tensor_shape("residual", residual, (tokens, _MHC_MULT, hidden_size))
    _validate_tensor_shape("prev_post", prev_post, (tokens, _MHC_MULT))
    _validate_tensor_shape("prev_comb", prev_comb, (tokens, _MHC_MULT, _MHC_MULT))
    _validate_tensor_shape("out", out, (tokens, _MHC_MULT, hidden_size))
    args = (
        _to_kernel_tensor(x, cutlass.BFloat16, dynamic_layout=True),
        _to_kernel_tensor(residual, cutlass.BFloat16, dynamic_layout=True),
        _to_kernel_tensor(
            prev_post,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        _to_kernel_tensor(
            prev_comb,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        _to_kernel_tensor(
            prev_comb,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        _to_kernel_tensor(
            prev_post,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        _to_kernel_tensor(out, cutlass.BFloat16, dynamic_layout=True),
        Int32(tokens),
        current_cuda_stream(),
    )
    cache_key = (
        tensor_key(
            "x",
            x,
            dims=(DimKey.dynamic(), DimKey.exact(hidden_size)),
        ),
        tensor_key(
            "residual",
            residual,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(_MHC_MULT),
                DimKey.exact(hidden_size),
            ),
        ),
        tensor_key(
            "prev_post",
            prev_post,
            dims=(DimKey.dynamic(), DimKey.exact(_MHC_MULT)),
        ),
        tensor_key(
            "prev_comb",
            prev_comb,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(_MHC_MULT),
                DimKey.exact(_MHC_MULT),
            ),
        ),
        tensor_key(
            "out",
            out,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(_MHC_MULT),
                DimKey.exact(hidden_size),
            ),
        ),
    )
    hidden_specialization = _hidden_specialization_name(hidden_size)
    if hidden_size == _HIDDEN:
        compile_name = "integration.residual.mhc_post_hidden4096_hctile128_all4"
        compile_key = (
            ("post_only", True),
            cache_key,
        )
    else:
        compile_name = (
            f"integration.residual.mhc_post_{hidden_specialization}_hctile128_all4"
        )
        compile_key = (
            ("hidden_size", hidden_size),
            ("post_only", True),
            cache_key,
        )
    sm12x_launch(
        _post_pre_partial_kernel(hidden_size, split_k, False, False, True),
        compile_spec=KernelCompileSpec.from_key(compile_name, 2, compile_key),
        compile_args=args,
        runtime_args=args,
    )


@torch.library.custom_op(
    "flashinfer_sm12x::mhc_post_launch",
    mutates_args=("out",),
)
def _mhc_post_launch_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    out: torch.Tensor,
) -> None:
    _run_mhc_post_launch(
        x=x,
        residual=residual,
        prev_post=prev_post,
        prev_comb=prev_comb,
        out=out,
    )


@_mhc_post_launch_op.register_fake
def _mhc_post_launch_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    out: torch.Tensor,
) -> None:
    return None


def run_mhc_post(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    out: torch.Tensor,
) -> None:
    torch.ops.flashinfer_sm12x.mhc_post_launch(
        x,
        residual,
        prev_post,
        prev_comb,
        out,
    )


def _run_mhc_pre_partial_launch(
    *,
    residual: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
    compute_gram: bool = False,
) -> None:
    tokens = int(residual.shape[0])
    hidden_size = int(residual.shape[1])
    split_k = int(partials.shape[1])
    _validate_split_k(hidden_size, split_k)
    partials_per_cta = _selected_post_pre_partials_per_cta(
        num_tokens=tokens,
        hidden_size=hidden_size,
    )
    _validate_tensor_shape("residual", residual, (tokens, hidden_size))
    _validate_tensor_shape("fn", fn, (_MIXES, hidden_size))
    _validate_tensor_shape("partials", partials, (tokens, split_k, _PARTIALS))
    _validate_tensor_shape("out", out, (tokens, _MHC_MULT, hidden_size))
    compute_gram = bool(compute_gram)
    args = (
        _to_kernel_tensor(residual, cutlass.BFloat16, dynamic_layout=True),
        _to_kernel_tensor(residual, cutlass.BFloat16, dynamic_layout=True),
        _to_kernel_tensor(
            partials,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        _to_kernel_tensor(
            partials,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        _to_kernel_tensor(fn, cutlass.Float32),
        _to_kernel_tensor(
            partials,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        _to_kernel_tensor(out, cutlass.BFloat16, dynamic_layout=True),
        Int32(tokens),
        current_cuda_stream(),
    )
    cache_key = (
        tensor_key(
            "residual",
            residual,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(hidden_size),
            ),
        ),
        tensor_key(
            "fn",
            fn,
            dims=(DimKey.exact(_MIXES), DimKey.exact(hidden_size)),
        ),
        tensor_key(
            "partials",
            partials,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(split_k),
                DimKey.exact(_PARTIALS),
            ),
        ),
        tensor_key(
            "out",
            out,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(_MHC_MULT),
                DimKey.exact(hidden_size),
            ),
        ),
    )
    hidden_specialization = _hidden_specialization_name(hidden_size)
    if hidden_size == _HIDDEN and split_k == _SPLIT_K:
        compile_name = (
            "integration.residual.mhc_pre_partial_hidden4096_hctile128"
            f"_all{partials_per_cta}"
        )
        compile_key = (
            ("partials_per_cta", partials_per_cta),
            ("compute_gram", compute_gram),
            ("pre_only", True),
            cache_key,
        )
    else:
        compile_name = (
            "integration.residual.mhc_pre_partial_"
            f"{hidden_specialization}_hctile128_all{partials_per_cta}"
        )
        compile_key = (
            ("hidden_size", hidden_size),
            ("split_k", split_k),
            ("source_tiles", hidden_size // _SOURCE_TILE_H),
            ("partials_per_cta", partials_per_cta),
            ("compute_gram", compute_gram),
            ("pre_only", True),
            cache_key,
        )
    sm12x_launch(
        _post_pre_partial_kernel(
            hidden_size,
            split_k,
            compute_gram,
            True,
            False,
            partials_per_cta,
        ),
        compile_spec=KernelCompileSpec.from_key(compile_name, 2, compile_key),
        compile_args=args,
        runtime_args=args,
    )


@torch.library.custom_op(
    "flashinfer_sm12x::mhc_pre_partial_launch",
    mutates_args=("partials", "out"),
)
def _mhc_pre_partial_launch_op(
    residual: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
    compute_gram: bool,
) -> None:
    _run_mhc_pre_partial_launch(
        residual=residual,
        fn=fn,
        partials=partials,
        out=out,
        compute_gram=compute_gram,
    )


@_mhc_pre_partial_launch_op.register_fake
def _mhc_pre_partial_launch_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
    compute_gram: bool,
) -> None:
    return None


def run_mhc_pre_partial(
    *,
    residual: torch.Tensor,
    fn: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
    compute_gram: bool = False,
) -> None:
    torch.ops.flashinfer_sm12x.mhc_pre_partial_launch(
        residual,
        fn,
        partials,
        out,
        bool(compute_gram),
    )


def _run_mhc_finalize_gram_launch(
    *,
    residual: torch.Tensor,
    partials: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    y: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor,
    norm_eps: float,
    fuse_norm: bool,
    compact_partials: bool = False,
    compact_projection_splits: int = 1,
    active_source_splits: int = 0,
) -> None:
    rms_eps = float(rms_eps)
    hc_eps = float(hc_eps)
    sinkhorn_iters = int(sinkhorn_iters)
    norm_eps = float(norm_eps)
    fuse_norm = bool(fuse_norm)
    compact_partials = bool(compact_partials)
    compact_projection_splits = int(compact_projection_splits)
    active_source_splits = int(active_source_splits)
    norm_weight_tensor = _norm_weight_kernel_tensor(
        norm_weight if fuse_norm else None,
        y,
    )
    tokens = int(residual.shape[0])
    hidden_size = int(residual.shape[2])
    split_k = int(partials.shape[1])
    _validate_split_k(hidden_size, split_k)
    single_cta_threads = (
        0
        if compact_partials
        else _selected_mhc_decode_finalize_threads(
            num_tokens=tokens,
            hidden_size=hidden_size,
        )
    )
    single_cta = single_cta_threads > 0
    raw_single_cta_groups = os.environ.get(
        "FLASHINFER_EXP_SM12X_MHC_DECODE_FINALIZE_CTAS"
    )
    single_cta_groups = 1
    if single_cta:
        single_cta_groups = (
            int(raw_single_cta_groups)
            if raw_single_cta_groups is not None
            else 8
            if single_cta_threads == 128 and tokens >= 10
            else 1
        )
    _validate_tensor_shape("residual", residual, (tokens, _MHC_MULT, hidden_size))
    _validate_tensor_shape("partials", partials, (tokens, split_k, _PARTIALS))
    _validate_tensor_shape("scale", scale, (3,))
    _validate_tensor_shape("bias", bias, (_MIXES,))
    _validate_tensor_shape("y", y, (tokens, hidden_size))
    _validate_tensor_shape("post", post, (tokens, _MHC_MULT))
    _validate_tensor_shape("comb", comb, (tokens, _MHC_MULT, _MHC_MULT))
    if fuse_norm:
        _validate_tensor_shape("norm_weight", norm_weight, (hidden_size,))
    args = (
        _to_kernel_tensor(residual, cutlass.BFloat16, dynamic_layout=True),
        _to_kernel_tensor(
            partials,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        _to_kernel_tensor(scale, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(bias, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(y, cutlass.BFloat16, dynamic_layout=True),
        _to_kernel_tensor(
            post,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        _to_kernel_tensor(
            comb,
            cutlass.Float32,
            assumed_align=4,
            dynamic_layout=True,
        ),
        norm_weight_tensor,
        Int32(tokens),
        current_cuda_stream(),
    )
    norm_weight_key = (
        tensor_key(
            "norm_weight",
            norm_weight,
            dims=(DimKey.exact(hidden_size),),
        )
        if fuse_norm
        else ("norm_weight", None)
    )
    common_key_tail = (
        (
            "impl",
            "prefill_finalize_gram_bf16x2_io_v3"
            if compact_partials
            else (
                "finalize_gram_single_cta_bf16x2_io_v2"
                if single_cta
                else "finalize_gram_multicta_v2"
            ),
        ),
        ("math", "fast_exp_exact_sigmoid_rcp_approx_sinkhorn"),
        ("fuse_norm", fuse_norm),
        ("compact_partials", compact_partials),
        ("compact_projection_splits", compact_projection_splits),
        ("single_cta", single_cta),
        ("single_cta_threads", single_cta_threads),
        ("single_cta_groups", single_cta_groups),
        ("active_source_splits", active_source_splits),
        ("pdl", _MHC_PDL),
        ("norm_eps", norm_eps if fuse_norm else 0.0),
        rms_eps,
        hc_eps,
        sinkhorn_iters,
        tensor_key(
            "residual",
            residual,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(_MHC_MULT),
                DimKey.exact(hidden_size),
            ),
        ),
        tensor_key(
            "partials",
            partials,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(split_k),
                DimKey.exact(_PARTIALS),
            ),
        ),
        tensor_key("scale", scale, dims=(DimKey.exact(3),)),
        tensor_key("bias", bias, dims=(DimKey.exact(_MIXES),)),
        tensor_key(
            "y",
            y,
            dims=(DimKey.dynamic(), DimKey.exact(hidden_size)),
        ),
        tensor_key(
            "post",
            post,
            dims=(DimKey.dynamic(), DimKey.exact(_MHC_MULT)),
        ),
        tensor_key(
            "comb",
            comb,
            dims=(
                DimKey.dynamic(),
                DimKey.exact(_MHC_MULT),
                DimKey.exact(_MHC_MULT),
            ),
        ),
        norm_weight_key,
    )
    hidden_specialization = _hidden_specialization_name(hidden_size)
    if hidden_size == _HIDDEN and split_k == _SPLIT_K:
        suffix = "_compact" if compact_partials else ""
        if single_cta:
            suffix = f"_single_cta_t{single_cta_threads}g{single_cta_groups}"
        compile_name = f"integration.residual.mhc_finalize_gram_hidden4096{suffix}"
        compile_key = (
            (
                "block_h",
                single_cta_threads
                if single_cta
                else (_PREFILL_FINALIZE_THREADS if compact_partials else _GRAM_BLOCK_H),
            ),
            ("source_tiles", _SOURCE_TILES),
            ("gram_row0", _GRAM_ROW0),
            *common_key_tail,
        )
    else:
        suffix = "_compact" if compact_partials else ""
        if single_cta:
            suffix = f"_single_cta_t{single_cta_threads}g{single_cta_groups}"
        compile_name = (
            f"integration.residual.mhc_finalize_gram_{hidden_specialization}{suffix}"
        )
        compile_key = (
            ("hidden_size", hidden_size),
            ("split_k", split_k),
            (
                "block_h",
                single_cta_threads
                if single_cta
                else (_PREFILL_FINALIZE_THREADS if compact_partials else _GRAM_BLOCK_H),
            ),
            ("source_tiles", hidden_size // _SOURCE_TILE_H),
            ("gram_row0", hidden_size // _SOURCE_TILE_H),
            *common_key_tail,
        )
    sm12x_launch(
        _finalize_gram_kernel(
            hidden_size,
            split_k,
            rms_eps,
            hc_eps,
            sinkhorn_iters,
            norm_eps,
            fuse_norm,
            compact_partials,
            compact_projection_splits,
            single_cta,
            single_cta_threads if single_cta else _PREFILL_FINALIZE_THREADS,
            single_cta_groups,
            active_source_splits,
        ),
        compile_spec=KernelCompileSpec.from_key(compile_name, 3, compile_key),
        compile_args=args,
        runtime_args=args,
    )


@torch.library.custom_op(
    "flashinfer_sm12x::mhc_finalize_gram_launch",
    mutates_args=("y", "post", "comb"),
)
def _mhc_finalize_gram_launch_op(
    residual: torch.Tensor,
    partials: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    y: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
    fuse_norm: bool,
    compact_partials: bool,
    compact_projection_splits: int,
    active_source_splits: int,
) -> None:
    _run_mhc_finalize_gram_launch(
        residual=residual,
        partials=partials,
        scale=scale,
        bias=bias,
        y=y,
        post=post,
        comb=comb,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        fuse_norm=fuse_norm,
        compact_partials=compact_partials,
        compact_projection_splits=compact_projection_splits,
        active_source_splits=active_source_splits,
    )


@_mhc_finalize_gram_launch_op.register_fake
def _mhc_finalize_gram_launch_fake(
    residual: torch.Tensor,
    partials: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    y: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
    fuse_norm: bool,
    compact_partials: bool,
    compact_projection_splits: int,
    active_source_splits: int,
) -> None:
    return None


def run_mhc_finalize_gram(
    *,
    residual: torch.Tensor,
    partials: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    y: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    compact_partials: bool = False,
    compact_projection_splits: int = 1,
    active_source_splits: int = 0,
) -> None:
    # When norm_weight is None the kernel ignores it (fuse_norm=False), but it
    # still needs a valid tensor arg. Do NOT alias `y` here: `y` is a mutated arg
    # of this op, and passing a mutated arg a second time as a read-only arg makes
    # auto_functionalized's decomposition fail under torch.compile (the
    # replace_by_example node-count assertion). Use a fresh, non-mutated
    # placeholder with y's (kernel-proven) shape/dtype instead.
    norm_weight_for_kernel = (
        norm_weight if norm_weight is not None else torch.empty_like(y)
    )
    torch.ops.flashinfer_sm12x.mhc_finalize_gram_launch(
        residual,
        partials,
        scale,
        bias,
        y,
        post,
        comb,
        norm_weight_for_kernel,
        float(rms_eps),
        float(hc_eps),
        int(sinkhorn_iters),
        float(norm_eps),
        norm_weight is not None,
        bool(compact_partials),
        int(compact_projection_splits),
        int(active_source_splits),
    )


@torch.library.custom_op(
    "flashinfer_sm12x::mhc_post_launch_functional",
    mutates_args=(),
)
def _mhc_post_launch_functional_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(residual)
    if int(x.shape[0]) != 0:
        _run_mhc_post_launch(
            x=x,
            residual=residual,
            prev_post=prev_post,
            prev_comb=prev_comb,
            out=out,
        )
    return out


@_mhc_post_launch_functional_op.register_fake
def _mhc_post_launch_functional_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


def run_mhc_post_functional(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.flashinfer_sm12x.mhc_post_launch_functional(
        x,
        residual,
        prev_post,
        prev_comb,
    )


@torch.library.custom_op(
    "flashinfer_sm12x::mhc_pre_launch_functional",
    mutates_args=(),
)
def _mhc_pre_launch_functional_op(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
    fuse_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = int(residual.shape[0])
    hidden_size = int(residual.shape[1])
    split_k = _split_k_for_hidden(hidden_size)
    partials = torch.empty(
        (tokens, split_k, _PARTIALS),
        dtype=torch.float32,
        device=residual.device,
    )
    y = torch.empty(
        (tokens, hidden_size),
        dtype=residual.dtype,
        device=residual.device,
    )
    post = torch.empty(
        (tokens, _MHC_MULT),
        dtype=torch.float32,
        device=residual.device,
    )
    comb = torch.empty(
        (tokens, _MHC_MULT, _MHC_MULT),
        dtype=torch.float32,
        device=residual.device,
    )
    out = torch.empty(
        (tokens, _MHC_MULT, hidden_size),
        dtype=residual.dtype,
        device=residual.device,
    )
    if tokens != 0:
        _run_mhc_pre_partial_launch(
            residual=residual,
            fn=fn,
            partials=partials,
            out=out,
            compute_gram=fuse_norm,
        )
        _run_mhc_finalize_gram_launch(
            residual=out,
            partials=partials,
            scale=scale,
            bias=bias,
            y=y,
            post=post,
            comb=comb,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            sinkhorn_iters=sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            fuse_norm=fuse_norm,
        )
    return out, post, comb, y


@_mhc_pre_launch_functional_op.register_fake
def _mhc_pre_launch_functional_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
    fuse_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = residual.shape[0]
    hidden_size = residual.shape[1]
    y = torch.empty(
        (tokens, hidden_size),
        dtype=residual.dtype,
        device=residual.device,
    )
    post = torch.empty(
        (tokens, _MHC_MULT),
        dtype=torch.float32,
        device=residual.device,
    )
    comb = torch.empty(
        (tokens, _MHC_MULT, _MHC_MULT),
        dtype=torch.float32,
        device=residual.device,
    )
    out = torch.empty(
        (tokens, _MHC_MULT, hidden_size),
        dtype=residual.dtype,
        device=residual.device,
    )
    return out, post, comb, y


def run_mhc_pre_functional(
    *,
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    norm_weight_for_kernel = norm_weight if norm_weight is not None else residual
    return torch.ops.flashinfer_sm12x.mhc_pre_launch_functional(
        residual,
        fn,
        scale,
        bias,
        norm_weight_for_kernel,
        float(rms_eps),
        float(hc_eps),
        int(sinkhorn_iters),
        float(norm_eps),
        norm_weight is not None,
    )


@torch.library.custom_op(
    "flashinfer_sm12x::mhc_post_pre_launch_functional",
    mutates_args=(),
)
def _mhc_post_pre_launch_functional_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
    fuse_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = int(residual.shape[0])
    hidden_size = int(residual.shape[2])
    split_k = _split_k_for_hidden(hidden_size)
    partials = torch.empty(
        (tokens, split_k, _PARTIALS),
        dtype=torch.float32,
        device=residual.device,
    )
    residual_out = torch.empty_like(residual)
    y = torch.empty(
        (tokens, hidden_size),
        dtype=residual.dtype,
        device=residual.device,
    )
    post = torch.empty(
        (tokens, _MHC_MULT),
        dtype=torch.float32,
        device=residual.device,
    )
    comb = torch.empty(
        (tokens, _MHC_MULT, _MHC_MULT),
        dtype=torch.float32,
        device=residual.device,
    )
    if tokens != 0:
        _run_mhc_post_pre_partial_launch(
            x=x,
            residual=residual,
            prev_post=prev_post,
            prev_comb=prev_comb,
            fn=fn,
            partials=partials,
            out=residual_out,
            compute_gram=fuse_norm,
        )
        decode_source_splits, _ = _selected_post_pre_decode_split_n(
            num_tokens=tokens,
            hidden_size=hidden_size,
        )
        _run_mhc_finalize_gram_launch(
            residual=residual_out,
            partials=partials,
            scale=scale,
            bias=bias,
            y=y,
            post=post,
            comb=comb,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            sinkhorn_iters=sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            fuse_norm=fuse_norm,
            active_source_splits=decode_source_splits,
        )
    return residual_out, post, comb, y


@_mhc_post_pre_launch_functional_op.register_fake
def _mhc_post_pre_launch_functional_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
    fuse_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = residual.shape[0]
    hidden_size = residual.shape[2]
    residual_out = torch.empty_like(residual)
    y = torch.empty(
        (tokens, hidden_size),
        dtype=residual.dtype,
        device=residual.device,
    )
    post = torch.empty(
        (tokens, _MHC_MULT),
        dtype=torch.float32,
        device=residual.device,
    )
    comb = torch.empty(
        (tokens, _MHC_MULT, _MHC_MULT),
        dtype=torch.float32,
        device=residual.device,
    )
    return residual_out, post, comb, y


def run_mhc_post_pre_functional(
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    norm_weight_for_kernel = norm_weight if norm_weight is not None else residual
    return torch.ops.flashinfer_sm12x.mhc_post_pre_launch_functional(
        x,
        residual,
        prev_post,
        prev_comb,
        fn,
        scale,
        bias,
        norm_weight_for_kernel,
        float(rms_eps),
        float(hc_eps),
        int(sinkhorn_iters),
        float(norm_eps),
        norm_weight is not None,
    )


__all__ = [
    "run_mhc_finalize_gram",
    "run_mhc_post",
    "run_mhc_post_functional",
    "run_mhc_pre_functional",
    "run_mhc_post_pre_functional",
    "run_mhc_pre_partial",
    "run_mhc_post_pre_partial",
    "run_mhc_post_pre_partial_alloc",
]
