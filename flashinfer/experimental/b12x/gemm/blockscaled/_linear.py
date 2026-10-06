"""Packed-weight adapters for the public :mod:`b12x.gemm.blockscaled` API."""

from __future__ import annotations

from b12x._lib.quant.block_codec import BLOCK_CODECS

from b12x.preparation import Plan
from b12x.preparation.types import plan_from_handle, require_prepared

from dataclasses import dataclass
from typing import Any, TypeAlias

import cutlass.cute as cute
import torch

from b12x._lib.dense_gemm import (
    _dense_spark_policy_for_sm_count,
    dense_gemm,
)
from b12x._lib.intrinsics import as_grouped_scale_view, as_grouped_scale_view_mx
from b12x._lib.utils import cuda_stream_to_int, get_num_sm
from b12x.gemm._shared.wo_mxfp8 import (
    MXFP8Rows,
    MXFP8_SCALE_VEC_SIZE,
    _check_gpu_tensor,
    pack_mxfp8_scales_for_dense_gemm,
)
from ._a16 import NVFP4LinearWeight, pack_nvfp4_weight
from ._iq2_xs import IQ2XSLinearWeight, pack_iq2_xs_weight


@dataclass(frozen=True)
class MXFP8LinearWeight:
    """ModelOpt-style MXFP8 weight packed for ``blockscaled.mm``."""

    weight: MXFP8Rows
    in_features: int
    padded_in_features: int
    out_features: int


@dataclass(frozen=True)
class TensorFP8LinearWeight:
    """Tensor-scaled E4M3 weight packed for ``blockscaled.mm``."""

    values: torch.Tensor
    scale_mma: torch.Tensor
    block_scale: torch.Tensor
    output_scale: torch.Tensor
    in_features: int
    padded_in_features: int
    out_features: int


Weight: TypeAlias = MXFP8LinearWeight | TensorFP8LinearWeight | NVFP4LinearWeight | IQ2XSLinearWeight


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _output_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    raise ValueError(f"blockscaled linear output must be bf16/fp16, got {dtype}")


def _output_dtype(c_dtype: str) -> torch.dtype:
    if c_dtype == "bfloat16":
        return torch.bfloat16
    if c_dtype == "float16":
        return torch.float16
    raise ValueError(f"blockscaled output must be bfloat16/float16, got {c_dtype!r}")


def _source_2d(source: torch.Tensor) -> torch.Tensor:
    if source.ndim < 2:
        raise ValueError(f"source must have at least 2 dims, got {tuple(source.shape)}")
    return source.reshape(-1, source.shape[-1]).contiguous()


def _pad_k(tensor: torch.Tensor, padded_k: int) -> torch.Tensor:
    rows, width = map(int, tensor.shape)
    if width == padded_k:
        return tensor.contiguous()
    padded = tensor.new_zeros((rows, padded_k))
    padded[:, :width] = tensor
    return padded.contiguous()


def _scale_rows_to_u8(scale_rows: torch.Tensor) -> torch.Tensor:
    if scale_rows.dtype == torch.uint8:
        return scale_rows.contiguous()
    if scale_rows.dtype == torch.float8_e8m0fnu:
        return scale_rows.view(torch.uint8).contiguous()
    raise ValueError(f"weight_scale must be uint8/e8m0, got {scale_rows.dtype}")


def _pad_scale_rows_k(
    scale_rows_u8: torch.Tensor,
    padded_sf_k: int,
) -> torch.Tensor:
    rows, sf_k = map(int, scale_rows_u8.shape)
    if sf_k == padded_sf_k:
        return scale_rows_u8.contiguous().view(torch.float8_e8m0fnu)
    padded = torch.full(
        (rows, padded_sf_k),
        127,
        dtype=torch.uint8,
        device=scale_rows_u8.device,
    )
    padded[:, :sf_k] = scale_rows_u8
    return padded.contiguous().view(torch.float8_e8m0fnu)


def _mxfp8_scale_mma_from_input(
    scale: torch.Tensor,
    *,
    rows: int,
    width: int,
    logical_width: int,
) -> torch.Tensor:
    """Normalize compact or F8_128x4 scales to the dense-GEMM MMA view."""

    if scale.dtype == torch.uint8:
        scale_u8 = scale
    elif scale.dtype != torch.float8_e8m0fnu:
        raise ValueError(
            f"MXFP8 activation scale must be uint8/e8m0, got {scale.dtype}"
        )
    else:
        scale_u8 = scale.view(torch.uint8)

    compact_shape = (rows, logical_width // MXFP8_SCALE_VEC_SIZE)
    if scale.ndim == 2 and tuple(scale.shape) == compact_shape:
        padded_scale = _pad_scale_rows_k(
            scale_u8,
            width // MXFP8_SCALE_VEC_SIZE,
        )
        return pack_mxfp8_scales_for_dense_gemm(
            padded_scale,
            m=rows,
            k=width,
            num_groups=1,
        )

    m_tiles = _align_up(rows, 128) // 128
    k_tiles = _align_up(width, 128) // 128
    expected_shape = (32, 4, m_tiles, 4, k_tiles, 1)
    if scale.ndim == 6:
        if tuple(scale.shape) != expected_shape:
            raise ValueError(
                "MXFP8 MMA scale has the wrong shape: expected "
                f"{expected_shape}, got {tuple(scale.shape)}"
            )
        return scale

    expected_numel = m_tiles * k_tiles * 32 * 4 * 4
    if not scale.is_contiguous() or scale.numel() != expected_numel:
        raise ValueError(
            "MXFP8 activation scale must use contiguous F8_128x4 swizzled "
            f"storage with {expected_numel} elements for M={rows}, K={width}; "
            f"got shape={tuple(scale.shape)}, contiguous={scale.is_contiguous()}"
        )
    physical = scale.view(m_tiles, k_tiles, 32, 4, 4)
    return physical.permute(2, 3, 0, 4, 1).unsqueeze(-1)


def _unit_scale_mma(rows: int, width: int, device: torch.device) -> torch.Tensor:
    scale_rows = torch.full(
        (rows, width // MXFP8_SCALE_VEC_SIZE),
        127,
        dtype=torch.uint8,
        device=device,
    )
    return pack_mxfp8_scales_for_dense_gemm(
        scale_rows,
        m=rows,
        k=width,
        num_groups=1,
    )


def _unit_block_scale(rows: int, width: int, device: torch.device) -> torch.Tensor:
    return torch.ones(
        (rows // 128, width // 128),
        dtype=torch.float32,
        device=device,
    )


_UNIT_SCALE_MMA_CACHE: dict[tuple, torch.Tensor] = {}
_UNIT_BLOCK_SCALE_CACHE: dict[tuple, torch.Tensor] = {}


def _cached_unit_scale_mma(
    device_type: str,
    device_index: int | None,
    rows: int,
    width: int,
) -> torch.Tensor:
    key = (device_type, device_index, rows, width)
    value = _UNIT_SCALE_MMA_CACHE.get(key)
    if value is None:
        value = _unit_scale_mma(rows, width, torch.device(device_type, device_index))
        _UNIT_SCALE_MMA_CACHE[key] = value
    return value


def _activation_scale_mma(
    source: torch.Tensor,
    rows: int,
    width: int,
) -> torch.Tensor:
    device_index = source.device.index
    if source.device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    return _cached_unit_scale_mma(
        source.device.type,
        device_index,
        int(rows),
        int(width),
    )


def _cached_unit_activation_block_scale(
    device_type: str,
    device_index: int | None,
    rows: int,
    width: int,
) -> torch.Tensor:
    key = (device_type, device_index, rows, width)
    value = _UNIT_BLOCK_SCALE_CACHE.get(key)
    if value is None:
        value = torch.ones((rows, width // 128), dtype=torch.float32,
                           device=torch.device(device_type, device_index))
        _UNIT_BLOCK_SCALE_CACHE[key] = value
    return value


def _activation_block_scale(
    source: torch.Tensor,
    rows: int,
    width: int,
) -> torch.Tensor:
    device_index = source.device.index
    if source.device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    return _cached_unit_activation_block_scale(
        source.device.type,
        device_index,
        int(rows),
        int(width),
    )


def _use_block_fp8_recipe(
    *,
    live_m: int,
    expected_m: int,
    out_features: int,
    padded_in_features: int,
    sm_count: int,
) -> bool:
    """Select the measured degenerate K128 recipe for aligned decode GEMMs."""

    return (
        live_m <= 8
        and expected_m <= 8
        and out_features % 128 == 0
        and padded_in_features % 128 == 0
        and _dense_spark_policy_for_sm_count(sm_count)
    )


def is_mxfp8_linear_supported() -> tuple[bool, str | None]:
    if not hasattr(cute.nvgpu.warp, "MmaMXF8Op"):
        return False, "CUTLASS DSL does not expose cute.nvgpu.warp.MmaMXF8Op"
    return True, None


def pack_mxfp8_linear_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> MXFP8LinearWeight:
    """Pack ModelOpt ``[N,K]`` MXFP8 values and compact ``[N,K/32]`` scales."""

    _check_gpu_tensor("weight", weight)
    _check_gpu_tensor("weight_scale", weight_scale)
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [N,K], got {tuple(weight.shape)}")
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"weight must be float8_e4m3fn, got {weight.dtype}")
    if weight_scale.ndim != 2:
        raise ValueError(
            f"weight_scale must have shape [N,K/32], got {tuple(weight_scale.shape)}"
        )

    out_features, in_features = map(int, weight.shape)
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if in_features <= 0 or in_features % MXFP8_SCALE_VEC_SIZE != 0:
        raise ValueError(
            "ModelOpt MXFP8 weight K must be a positive multiple of "
            f"{MXFP8_SCALE_VEC_SIZE}, got {in_features}"
        )

    scale_k = in_features // MXFP8_SCALE_VEC_SIZE
    if (
        int(weight_scale.shape[0]) < out_features
        or int(weight_scale.shape[1]) < scale_k
    ):
        raise ValueError(
            "weight_scale must have at least shape "
            f"{(out_features, scale_k)}, got {tuple(weight_scale.shape)}"
        )

    padded_in_features = _align_up(in_features, 128)
    padded_scale_k = padded_in_features // MXFP8_SCALE_VEC_SIZE
    weight_values = _pad_k(
        weight[:out_features, :in_features],
        padded_in_features,
    )
    scale_rows_u8 = _scale_rows_to_u8(weight_scale[:out_features, :scale_k])
    scale_rows = _pad_scale_rows_k(scale_rows_u8, padded_scale_k)
    scale_mma = pack_mxfp8_scales_for_dense_gemm(
        scale_rows,
        m=out_features,
        k=padded_in_features,
        num_groups=1,
    )
    return MXFP8LinearWeight(
        weight=MXFP8Rows(
            values=weight_values,
            scale_rows=scale_rows.reshape(1, out_features, padded_scale_k),
            scale_mma=scale_mma,
        ),
        in_features=in_features,
        padded_in_features=padded_in_features,
        out_features=out_features,
    )


def is_tensor_fp8_linear_supported() -> tuple[bool, str | None]:
    if not hasattr(cute.nvgpu.warp, "MmaMXF8Op"):
        return False, "CUTLASS DSL does not expose cute.nvgpu.warp.MmaMXF8Op"
    return True, None


def pack_tensor_fp8_linear_weight(
    weight: torch.Tensor,
    output_scale: torch.Tensor,
) -> TensorFP8LinearWeight:
    """Pack an E4M3 ``[N,K]`` weight with one combined dequantization scale."""

    _check_gpu_tensor("weight", weight)
    _check_gpu_tensor("output_scale", output_scale)
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [N,K], got {tuple(weight.shape)}")
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"weight must be float8_e4m3fn, got {weight.dtype}")
    if output_scale.dtype != torch.float32 or output_scale.numel() != 1:
        raise ValueError(
            "output_scale must be one float32 value, got "
            f"dtype={output_scale.dtype}, shape={tuple(output_scale.shape)}"
        )
    if output_scale.device != weight.device:
        raise ValueError("weight and output_scale must be on the same device")
    if not bool(torch.isfinite(output_scale).all()) or bool((output_scale < 0).any()):
        raise ValueError("output_scale must be finite and non-negative")

    out_features, in_features = map(int, weight.shape)
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if in_features <= 0 or in_features % MXFP8_SCALE_VEC_SIZE != 0:
        raise ValueError(
            "tensor FP8 weight K must be a positive multiple of "
            f"{MXFP8_SCALE_VEC_SIZE}, got {in_features}"
        )

    padded_in_features = _align_up(in_features, 128)
    values = _pad_k(weight, padded_in_features)
    scale_mma = _unit_scale_mma(
        out_features,
        padded_in_features,
        weight.device,
    )
    return TensorFP8LinearWeight(
        values=values,
        scale_mma=scale_mma,
        block_scale=_unit_block_scale(
            out_features,
            padded_in_features,
            weight.device,
        ),
        output_scale=output_scale.reshape(1).contiguous(),
        in_features=in_features,
        padded_in_features=padded_in_features,
        out_features=out_features,
    )


def pack_weight(
    weight: torch.Tensor,
    scale: torch.Tensor | None = None,
    *,
    recipe: str | None = None,
    global_scale: torch.Tensor | None = None,
    global_scale_kind: str = "multiplier",
) -> Weight:
    """Pack a serialized dense weight for ``blockscaled.mm``.

    A scalar FP32 ``scale`` selects tensor-scaled FP8 and represents the
    combined activation/weight dequantization scale. A 2D uint8/UE8M0 scale
    selects ModelOpt MXFP8.

    ``recipe='nvfp4'`` borrows packed uint8 values and swizzled E4M3 scales.
    Pass the weight-only ``global_scale`` and its ``global_scale_kind``
    (``'multiplier'`` or ``'reciprocal'``); no scale transformation occurs.

    ``recipe='iq2_xs'`` takes CUDA uint8 ``[N,K/256,74]`` safetensors block
    payloads with embedded scales; ``iq2_xxs`` takes 66-byte blocks.
    ``q8_0`` takes ``[N,K/32,34]`` blocks containing INT8 values and FP16 scales.
    Omit ``scale`` and ``global_scale``.
    Preparation losslessly rearranges descriptors and metadata for A16 GEMM.
    """

    if recipe in BLOCK_CODECS:
        if scale is not None or global_scale is not None or global_scale_kind != "multiplier":
            raise ValueError("IQ2_XS scales are embedded in its block payload")
        return pack_iq2_xs_weight(weight, codec=recipe)
    if scale is None:
        raise ValueError("this weight recipe requires a scale tensor")
    if recipe == "nvfp4":
        return pack_nvfp4_weight(weight, scale, global_scale,
                                 global_scale_kind=global_scale_kind)
    if recipe not in (None, "mxfp8", "tensor_fp8") or global_scale is not None or global_scale_kind != "multiplier":
        raise ValueError("global_scale/kind are specific to recipe='nvfp4'")
    if scale.dtype == torch.float32 and scale.numel() == 1 and recipe != "mxfp8":
        return pack_tensor_fp8_linear_weight(weight, scale)
    if scale.dtype in (torch.uint8, torch.float8_e8m0fnu) and scale.ndim == 2 and recipe != "tensor_fp8":
        return pack_mxfp8_linear_weight(weight, scale)
    raise ValueError(
        "unsupported blockscaled weight scale: expected one FP32 combined "
        "tensor scale or a 2D uint8/UE8M0 MXFP8 scale; got "
        f"dtype={scale.dtype}, shape={tuple(scale.shape)}"
    )


@torch.library.custom_op(
    "b12x::blockscaled_serialized",
    mutates_args=(),
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def _blockscaled_serialized_op(
    lhs_values: torch.Tensor,
    lhs_scale_storage: torch.Tensor,
    rhs_values: torch.Tensor,
    rhs_scale_storage: torch.Tensor,
    alpha: torch.Tensor | None,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    sf_vec_size: int,
    block_fp8: bool,
    plan_handle: int,
    stream_int: int | None,
    expected_m: int | None = None,
) -> torch.Tensor:
    declaration = plan_from_handle(plan_handle) if plan_handle else None
    if declaration is None:
        from ._preparation import heuristic_plan, query_from_call
        declaration = heuristic_plan(query_from_call(
            (lhs_values, lhs_scale_storage), (rhs_values, rhs_scale_storage),
            alpha=alpha, ab_dtype=ab_dtype, sf_dtype=sf_dtype, c_dtype=c_dtype,
            sf_vec_size=sf_vec_size, block_fp8=block_fp8, expected_m=expected_m,
        ), lhs_values.device)
    state = require_prepared(declaration, "gemm.blockscaled.fixed", lhs_values.device)
    return state.run_serialized(
        lhs_values,
        lhs_scale_storage,
        rhs_values,
        rhs_scale_storage,
        alpha,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
        block_fp8=block_fp8,
        stream=stream_int,
    )


@_blockscaled_serialized_op.register_fake
def _blockscaled_serialized_fake(
    lhs_values: torch.Tensor,
    lhs_scale_storage: torch.Tensor,
    rhs_values: torch.Tensor,
    rhs_scale_storage: torch.Tensor,
    alpha: torch.Tensor | None,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    sf_vec_size: int,
    block_fp8: bool,
    plan_handle: int,
    stream_int: int | None,
    expected_m: int | None = None,
) -> torch.Tensor:
    del lhs_scale_storage, rhs_scale_storage, alpha
    del ab_dtype, sf_dtype, sf_vec_size, block_fp8, plan_handle, stream_int
    return torch.empty(
        (lhs_values.shape[0], rhs_values.shape[0]),
        dtype=_output_dtype(c_dtype),
        device=lhs_values.device,
    )


@torch.library.custom_op(
    "b12x::blockscaled_packed_mxfp8",
    mutates_args=(),
)
def _packed_mxfp8_op(
    source_2d: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    plan_handle: int,
    stream_int: int | None,
    expected_m: int | None = None,
) -> torch.Tensor:
    declaration = _fixed_plan(plan_handle, source_2d, weight_values, "mxfp8", source_2d.dtype, expected_m=expected_m)
    state = require_prepared(declaration, "gemm.blockscaled.fixed", source_2d.device)
    return state.run_mxfp8(
        source_2d,
        weight_values,
        weight_scale_mma,
        out_dtype=source_2d.dtype,
        stream=stream_int,
    )


@_packed_mxfp8_op.register_fake
def _packed_mxfp8_fake(
    source_2d: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    plan_handle: int,
    stream_int: int | None,
    expected_m: int | None = None,
) -> torch.Tensor:
    del weight_scale_mma, plan_handle, stream_int
    return torch.empty(
        (source_2d.shape[0], weight_values.shape[0]),
        dtype=source_2d.dtype,
        device=source_2d.device,
    )


@torch.library.custom_op(
    "b12x::blockscaled_packed_mxfp8_prequantized",
    mutates_args=(),
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def _packed_mxfp8_prequantized_op(
    source_values: torch.Tensor,
    source_scale_storage: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    plan_handle: int,
    out_dtype: torch.dtype,
    stream_int: int | None,
    expected_m: int | None = None,
) -> torch.Tensor:
    declaration = _fixed_plan(plan_handle, source_values, weight_values, "mxfp8", out_dtype,
                              source_scale_storage, expected_m=expected_m)
    state = require_prepared(declaration, "gemm.blockscaled.fixed", source_values.device)
    return state.run_mxfp8(
        source_values,
        weight_values,
        weight_scale_mma,
        source_scale=source_scale_storage,
        out_dtype=out_dtype,
        stream=stream_int,
    )


@_packed_mxfp8_prequantized_op.register_fake
def _packed_mxfp8_prequantized_fake(
    source_values: torch.Tensor,
    source_scale_storage: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    plan_handle: int,
    out_dtype: torch.dtype,
    stream_int: int | None,
    expected_m: int | None = None,
) -> torch.Tensor:
    del source_scale_storage, weight_scale_mma, plan_handle, stream_int
    return torch.empty(
        (source_values.shape[0], weight_values.shape[0]),
        dtype=out_dtype,
        device=source_values.device,
    )


def _fixed_plan(handle, source, weight, recipe, out_dtype, source_scale=None, *, expected_m=None):
    if handle:
        return plan_from_handle(handle)
    from ._preparation import heuristic_plan
    from ._tuning import FixedBlockscaledQuery
    m, k = source.shape
    form = "none"
    if source_scale is not None:
        form = ("compact" if source_scale.ndim == 2 and source_scale.shape == (m, k // 32)
                else "mma" if source_scale.ndim == 6 else "swizzled")
    return heuristic_plan(FixedBlockscaledQuery(
        recipe=recipe, call_kind="packed", max_rows=m, in_features=k,
        padded_in_features=weight.shape[1], out_features=weight.shape[0],
        input_dtype=str(source.dtype).removeprefix("torch."),
        output_dtype=str(out_dtype).removeprefix("torch."), expected_m=expected_m,
        source_scale_form=form,
    ), source.device)


def _validate_bias(
    bias: torch.Tensor | None,
    *,
    out_features: int,
    out_dtype: torch.dtype,
    device: torch.device,
) -> None:
    if bias is None:
        return
    _check_gpu_tensor("bias", bias)
    if bias.device != device:
        raise ValueError("bias must be on the same device as source")
    if bias.dtype != out_dtype or bias.shape != (out_features,):
        raise ValueError(
            f"bias must have shape {(out_features,)} and dtype {out_dtype}, "
            f"got shape={tuple(bias.shape)}, dtype={bias.dtype}"
        )


def mxfp8_linear(
    source: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    packed_weight: MXFP8LinearWeight,
    *,
    plan: Plan | None = None,
    expected_m: int | None = None,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    stream: object = None,
) -> torch.Tensor:
    """Run plain or prequantized activations through MXFP8 ``blockscaled.mm``."""

    if not isinstance(packed_weight, MXFP8LinearWeight):
        raise TypeError("packed_weight must be an MXFP8LinearWeight")

    prequantized = isinstance(source, tuple)
    if prequantized:
        if len(source) != 2:
            raise ValueError("prequantized source must be a (values, scale) pair")
        source_values, source_scale = source
    else:
        source_values = source
        source_scale = None
    _check_gpu_tensor("source", source_values)
    source_2d = _source_2d(source_values)
    tokens, in_features = map(int, source_2d.shape)
    if in_features != int(packed_weight.in_features):
        raise ValueError(
            f"input K={in_features} does not match packed weight K="
            f"{packed_weight.in_features}"
        )
    if packed_weight.weight.values.device != source_2d.device:
        raise ValueError("source and packed weight must be on the same device")

    if prequantized:
        if source_2d.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"prequantized MXFP8 source must be float8_e4m3fn, got {source_2d.dtype}"
            )
        assert source_scale is not None
        _check_gpu_tensor("source_scale", source_scale)
        if source_scale.device != source_2d.device:
            raise ValueError("source and source_scale must be on the same device")
        resolved_out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    else:
        if source_2d.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"source dtype must be bf16/fp16, got {source_2d.dtype}")
        resolved_out_dtype = source_2d.dtype if out_dtype is None else out_dtype
        if resolved_out_dtype != source_2d.dtype:
            raise ValueError(
                "plain MXFP8 output dtype must match the BF16/FP16 source dtype"
            )
    _output_dtype_name(resolved_out_dtype)

    if not prequantized and source_2d.dtype == torch.bfloat16:
        return blockscaled_mm(
            source_values,
            packed_weight,
            plan=plan,
            expected_m=expected_m,
            bias=bias,
            out_dtype=resolved_out_dtype,
            stream=stream,
        )

    out_features = int(packed_weight.out_features)
    _validate_bias(
        bias,
        out_features=out_features,
        out_dtype=resolved_out_dtype,
        device=source_2d.device,
    )
    if tokens == 0:
        output = torch.empty(
            (0, out_features),
            dtype=resolved_out_dtype,
            device=source_2d.device,
        )
    elif prequantized:
        assert source_scale is not None
        output = torch.ops.b12x.blockscaled_packed_mxfp8_prequantized(
            source_2d,
            source_scale,
            packed_weight.weight.values,
            packed_weight.weight.scale_mma,
            0 if plan is None else plan.handle,
            resolved_out_dtype,
            cuda_stream_to_int(stream),
            expected_m,
        )
    else:
        output = torch.ops.b12x.blockscaled_packed_mxfp8(
            source_2d,
            packed_weight.weight.values,
            packed_weight.weight.scale_mma,
            0 if plan is None else plan.handle,
            cuda_stream_to_int(stream),
            expected_m,
        )
    if bias is not None:
        output = output + bias
    return output.view(*source_values.shape[:-1], out_features)


@torch.library.custom_op(
    "b12x::blockscaled_packed_tensor_fp8",
    mutates_args=(),
)
def _packed_tensor_fp8_op(
    source_2d: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    weight_block_scale: torch.Tensor,
    output_scale: torch.Tensor,
    plan_handle: int,
    out_dtype: torch.dtype,
    stream_int: int | None,
    expected_m: int | None = None,
) -> torch.Tensor:
    declaration = _fixed_plan(plan_handle, source_2d, weight_values, "tensor_fp8", out_dtype, expected_m=expected_m)
    state = require_prepared(declaration, "gemm.blockscaled.fixed", source_2d.device)
    return state.run_tensor_fp8(
        source_2d,
        weight_values,
        weight_scale_mma,
        weight_block_scale,
        output_scale,
        out_dtype=out_dtype,
        stream=stream_int,
    )


@_packed_tensor_fp8_op.register_fake
def _packed_tensor_fp8_fake(
    source_2d: torch.Tensor,
    weight_values: torch.Tensor,
    weight_scale_mma: torch.Tensor,
    weight_block_scale: torch.Tensor,
    output_scale: torch.Tensor,
    plan_handle: int,
    out_dtype: torch.dtype,
    stream_int: int | None,
    expected_m: int | None = None,
) -> torch.Tensor:
    del weight_scale_mma, weight_block_scale, output_scale, plan_handle, stream_int
    return torch.empty(
        (source_2d.shape[0], weight_values.shape[0]),
        dtype=out_dtype,
        device=source_2d.device,
    )


def tensor_fp8_linear(
    source: torch.Tensor,
    packed_weight: TensorFP8LinearWeight,
    *,
    plan: Plan | None = None,
    expected_m: int | None = None,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    stream: object = None,
) -> torch.Tensor:
    """Run static per-tensor E4M3 operands through the prepared fixed route."""

    _check_gpu_tensor("source", source)
    if not isinstance(packed_weight, TensorFP8LinearWeight):
        raise TypeError("packed_weight must be a TensorFP8LinearWeight")
    source_2d = _source_2d(source)
    tokens, in_features = map(int, source_2d.shape)
    if source_2d.dtype != torch.float8_e4m3fn:
        raise ValueError(f"source must be float8_e4m3fn, got {source_2d.dtype}")
    if in_features != int(packed_weight.in_features):
        raise ValueError(
            f"input K={in_features} does not match packed weight K="
            f"{packed_weight.in_features}"
        )
    if packed_weight.values.device != source_2d.device:
        raise ValueError("source and packed weight must be on the same device")
    _output_dtype_name(out_dtype)

    out_features = int(packed_weight.out_features)
    _validate_bias(
        bias,
        out_features=out_features,
        out_dtype=out_dtype,
        device=source_2d.device,
    )
    if tokens == 0:
        output = torch.empty(
            (0, out_features),
            dtype=out_dtype,
            device=source_2d.device,
        )
    else:
        output = torch.ops.b12x.blockscaled_packed_tensor_fp8(
            source_2d,
            packed_weight.values,
            packed_weight.scale_mma,
            packed_weight.block_scale,
            packed_weight.output_scale,
            0 if plan is None else plan.handle,
            out_dtype,
            cuda_stream_to_int(stream),
            expected_m,
        )
    if bias is not None:
        output = output + bias
    return output.view(*source.shape[:-1], out_features)


def blockscaled_mm(
    lhs: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    rhs: Weight | tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor | None = None,
    *,
    plan: Plan | None = None,
    expected_m: int | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Dispatch packed linear weights or preserve the raw ``dense_gemm`` API."""

    if plan is None and isinstance(rhs, IQ2XSLinearWeight):
        raise TypeError("block-quantized linear weights require a prepared Plan")

    if isinstance(rhs, (NVFP4LinearWeight, IQ2XSLinearWeight)) or (
        isinstance(rhs, MXFP8LinearWeight) and isinstance(lhs, torch.Tensor)
        and lhs.dtype == torch.bfloat16
    ):
        if not isinstance(lhs, torch.Tensor):
            raise TypeError("packed linear requires a BF16 source tensor")
        options = dict(kwargs)
        bias = options.pop("bias", None)
        out_dtype = options.pop("out_dtype", None)
        if out_dtype not in (None, torch.bfloat16):
            raise ValueError("BF16 blockscaled linear output must be BF16")
        _validate_bias(bias, out_features=rhs.out_features, out_dtype=torch.bfloat16,
                       device=lhs.device)
        if bias is not None and out is not None:
            from ._a16 import _overlap
            if _overlap(out, bias):
                raise ValueError("out must not overlap bias")
        if lhs.shape[-1] != rhs.in_features:
            raise ValueError("source logical K does not match the packed weight")
        from ._ops import linear
        from ._a16 import _weight_parts
        values, scales, global_scale, fp4 = _weight_parts(rhs)
        result = linear(
            lhs, values, scales, global_scale, plan=plan,
            global_scale_kind=rhs.global_scale_kind if fp4 else "none",
            out=out, expected_m=expected_m, **options,
        )
        if bias is not None:
            from ._a16 import _stream_context
            with torch.cuda.device(lhs.device), _stream_context(options.get("stream"), lhs.device):
                result.add_(bias)
        return result
    if isinstance(rhs, MXFP8LinearWeight):
        if out is not None:
            raise ValueError("packed MXFP8 blockscaled.mm does not accept out")
        return mxfp8_linear(lhs, rhs, plan=plan, expected_m=expected_m, **kwargs)
    if isinstance(rhs, TensorFP8LinearWeight):
        if out is not None:
            raise ValueError("packed tensor-FP8 blockscaled.mm does not accept out")
        if isinstance(lhs, tuple):
            raise TypeError(
                "tensor-scaled FP8 blockscaled.mm accepts its prequantized "
                "values tensor directly; its static scale is already folded "
                "into the packed weight"
            )
        return tensor_fp8_linear(lhs, rhs, plan=plan, expected_m=expected_m, **kwargs)
    if not isinstance(lhs, tuple) or not isinstance(rhs, tuple):
        raise TypeError(
            "raw blockscaled.mm operands must be (values, scale) pairs, or rhs "
            "must be a weight returned by blockscaled.pack_weight"
        )
    lhs_values, lhs_scale = lhs
    rhs_values, rhs_scale = rhs
    if lhs_values.ndim == 2 or rhs_values.ndim == 2:
        if lhs_values.ndim != 2 or rhs_values.ndim != 2:
            raise ValueError(
                "serialized blockscaled values must either both be 2D or both "
                "use the native 3D dense-GEMM layout"
            )
        if out is not None:
            raise ValueError("serialized blockscaled.mm does not accept out")
        recipe = dict(kwargs)
        try:
            ab_dtype = recipe.pop("ab_dtype")
            sf_dtype = recipe.pop("sf_dtype")
            c_dtype = recipe.pop("c_dtype")
            sf_vec_size = recipe.pop("sf_vec_size")
        except KeyError as exc:
            raise TypeError(
                f"serialized blockscaled.mm requires {exc.args[0]}"
            ) from None
        alpha = recipe.pop("alpha", None)
        stream = recipe.pop("stream", None)
        block_fp8 = bool(recipe.pop("block_fp8", False))
        if recipe:
            names = ", ".join(sorted(recipe))
            raise TypeError(
                "serialized blockscaled.mm does not support these raw-engine "
                f"options: {names}"
            )
        return torch.ops.b12x.blockscaled_serialized(
            lhs_values,
            lhs_scale,
            rhs_values,
            rhs_scale,
            alpha,
            str(ab_dtype),
            str(sf_dtype),
            str(c_dtype),
            int(sf_vec_size),
            block_fp8,
            0 if plan is None else plan.handle,
            cuda_stream_to_int(stream),
            expected_m,
        )
    if plan is None:
        return dense_gemm(lhs, rhs, out, expected_m=expected_m, **kwargs)
    from b12x.gemm._preparation import mm as prepared_mm
    return prepared_mm(lhs, rhs, out, plan=plan, **kwargs)






__all__ = [
    "MXFP8LinearWeight",
    "TensorFP8LinearWeight",
    "Weight",
    "blockscaled_mm",
    "is_mxfp8_linear_supported",
    "is_tensor_fp8_linear_supported",
    "mxfp8_linear",
    "pack_mxfp8_linear_weight",
    "pack_weight",
    "pack_tensor_fp8_linear_weight",
    "tensor_fp8_linear",
]
