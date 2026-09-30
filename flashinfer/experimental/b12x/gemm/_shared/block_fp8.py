from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Sequence

import torch

from b12x._lib.scratch import (
    ScratchBufferSpec,
    scratch_buffer_spec,
    scratch_tensor,
)
from b12x._lib.utils import cuda_stream_to_int
from b12x.gemm._shared.wo_mxfp8 import (
    MXFP8_SCALE_K_TILE,
    MXFP8_SCALE_ROW_TILE,
    MXFP8_SCALE_VEC_SIZE,
    MXFP8Rows,
    _check_gpu_tensor,
    _check_mxfp8_k,
    _check_mxfp8_rows_storage,
    empty_dense_gemm_mnl_view,
    empty_mxfp8_rows_bases,
    mxfp8_rows_from_bases,
    pack_fp8_block_scaled_weight_mxfp8,
)
from b12x.preparation import Plan
from b12x.preparation.types import require_prepared

_SCRATCH_ALIGN_BYTES = 1024


@dataclass(frozen=True)
class BlockFP8LinearWeight:
    weight: MXFP8Rows
    in_features: int
    out_features: int
    block_size: tuple[int, int]


def _physical_mxfp8_k(logical_k: int) -> int:
    logical_k = int(logical_k)
    if logical_k <= 0 or logical_k % MXFP8_SCALE_VEC_SIZE:
        raise ValueError(
            "block FP8 linear logical K must be a positive multiple of "
            f"{MXFP8_SCALE_VEC_SIZE}, got {logical_k}"
        )
    return _align_up(logical_k, 128)



def _packed_physical_mxfp8_k(packed_weight: BlockFP8LinearWeight) -> int:
    physical_k = int(packed_weight.weight.values.shape[1])
    _check_mxfp8_k(physical_k)
    return physical_k



@dataclass(frozen=True, kw_only=True)
class BlockFP8LinearBinding:
    plan: Plan
    source: torch.Tensor
    packed_weight: BlockFP8LinearWeight
    x_q: MXFP8Rows
    output: torch.Tensor
    workspace: torch.Tensor | None = None
    bias: torch.Tensor | None = None
    expected_m: int | None = None
    mma_tiler_mn: tuple[int, int] | None = None

    def run(self, *, stream: object = None) -> torch.Tensor:
        return block_fp8_linear_mxfp8(binding=self, stream=stream)


@dataclass(frozen=True, kw_only=True)
class BlockFP8LinearScratchCaps:
    device: torch.device | str
    max_tokens: int
    in_features: int
    out_features: int
    source_dtype: torch.dtype = torch.bfloat16
    output_dtype: torch.dtype = torch.bfloat16
    output_mode: str = "provided"
    block_size: tuple[int, int] = (128, 128)

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "max_tokens", max(int(self.max_tokens), 1))
        object.__setattr__(self, "in_features", max(int(self.in_features), 1))
        object.__setattr__(self, "out_features", max(int(self.out_features), 1))
        _physical_mxfp8_k(self.in_features)
        object.__setattr__(self, "block_size", _check_block_size(self.block_size))
        if self.source_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"source_dtype must be bf16/fp16, got {self.source_dtype}")
        if self.output_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"output_dtype must be bf16/fp16, got {self.output_dtype}")
        if self.output_mode not in ("functional", "provided"):
            raise ValueError("block-FP8 output_mode must be functional or provided")


@dataclass(frozen=True)
class _BlockFP8LinearScratchPlan:
    """Private scratch-view mapper retained by a prepared plan state."""
    caps: BlockFP8LinearScratchCaps
    _scratch_specs: tuple[ScratchBufferSpec, ...]
    mma_tiler_mn: tuple[int, int]
    workspace_nbytes: int = 0

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def bind(
        self,
        *,
        plan: Plan,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        source: torch.Tensor,
        packed_weight: BlockFP8LinearWeight,
        output: torch.Tensor,
        bias: torch.Tensor | None = None,
        expected_m: int | None = None,
    ) -> BlockFP8LinearBinding:
        source_2d = _source_2d(source)
        tokens, in_features = map(int, source_2d.shape)
        if tokens > self.caps.max_tokens or in_features != self.caps.in_features:
            raise ValueError("block-FP8 binding geometry differs from its prepared capacity")
        if packed_weight.out_features != self.caps.out_features:
            raise ValueError("packed weight output geometry differs from preparation")
        if packed_weight.block_size != self.caps.block_size:
            raise ValueError("packed weight block size differs from preparation")
        if source_2d.dtype != self.caps.source_dtype:
            raise ValueError("source dtype differs from preparation")
        scratch = scratch_tensor(scratch, self._scratch_specs, owner="block FP8 linear")
        workspace = None
        if self.workspace_nbytes:
            offset = _align_up(_block_fp8_linear_scratch_layout(
                tokens=self.caps.max_tokens, in_features=self.caps.in_features,
                out_features=self.caps.out_features, output_dtype=self.caps.output_dtype,
            ).nbytes, _SCRATCH_ALIGN_BYTES)
            workspace = scratch.narrow(0, offset, self.workspace_nbytes).view(torch.float32)
        return build_block_fp8_linear_binding(
            plan=plan, source=source, packed_weight=packed_weight,
            x_q=_block_fp8_linear_x_q_from_scratch(
                scratch, tokens=tokens, in_features=self.caps.in_features,
                output_dtype=self.caps.output_dtype,
            ),
            output=output, workspace=workspace, bias=bias,
            expected_m=self.caps.max_tokens if expected_m is None else expected_m,
            mma_tiler_mn=self.mma_tiler_mn,
        )


@dataclass(frozen=True, kw_only=True)
class _BlockFP8LinearScratchLayout:
    nbytes: int
    x_values_offset_bytes: int
    x_scale_rows_offset_bytes: int
    x_scale_mma_offset_bytes: int
    x_scale_mma_physical_shape: tuple[int, int, int, int, int, int]


def _check_block_size(block_size: Sequence[int]) -> tuple[int, int]:
    if len(block_size) != 2:
        raise ValueError(f"block_size must have two elements, got {block_size}")
    block_n, block_k = int(block_size[0]), int(block_size[1])
    if (block_n, block_k) not in ((128, 128), (32, 32)):
        raise ValueError(
            f"b12x block FP8 linear supports 128x128 or 32x32 weight blocks, got {block_size}"
        )
    return block_n, block_k


def _c_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    raise ValueError(
        f"b12x block FP8 linear output dtype must be bf16/fp16, got {dtype}"
    )


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return dtype.itemsize


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _shape_numel(shape: Sequence[int]) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _block_fp8_linear_scratch_layout(
    *,
    tokens: int,
    in_features: int,
    out_features: int,
    output_dtype: torch.dtype,
) -> _BlockFP8LinearScratchLayout:
    tokens = max(int(tokens), 1)
    in_features = max(int(in_features), 1)
    del out_features
    physical_in_features = _physical_mxfp8_k(in_features)
    _c_dtype_name(output_dtype)

    offset = 0
    offset = _align_up(offset, _SCRATCH_ALIGN_BYTES)
    x_values_offset_bytes = offset
    offset += tokens * physical_in_features * _dtype_nbytes(torch.float8_e4m3fn)

    offset = _align_up(offset, _SCRATCH_ALIGN_BYTES)
    x_scale_rows_offset_bytes = offset
    offset += (
        tokens
        * (physical_in_features // MXFP8_SCALE_VEC_SIZE)
        * _dtype_nbytes(torch.float8_e8m0fnu)
    )

    offset = _align_up(offset, _SCRATCH_ALIGN_BYTES)
    x_scale_mma_offset_bytes = offset
    sf_k = physical_in_features // MXFP8_SCALE_VEC_SIZE
    x_scale_mma_physical_shape = (
        1,
        math.ceil(tokens / MXFP8_SCALE_ROW_TILE),
        math.ceil(sf_k / MXFP8_SCALE_K_TILE),
        32,
        4,
        4,
    )
    offset += _shape_numel(x_scale_mma_physical_shape) * _dtype_nbytes(torch.uint8)

    return _BlockFP8LinearScratchLayout(
        nbytes=max(int(offset), 1),
        x_values_offset_bytes=x_values_offset_bytes,
        x_scale_rows_offset_bytes=x_scale_rows_offset_bytes,
        x_scale_mma_offset_bytes=x_scale_mma_offset_bytes,
        x_scale_mma_physical_shape=x_scale_mma_physical_shape,
    )


def _scratch_view(
    scratch: torch.Tensor,
    *,
    offset_bytes: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    offset_bytes = _align_up(
        offset_bytes, max(_SCRATCH_ALIGN_BYTES, _dtype_nbytes(dtype))
    )
    nbytes = _shape_numel(shape) * _dtype_nbytes(dtype)
    return scratch.narrow(0, offset_bytes, nbytes).view(dtype).view(shape)


def _block_fp8_linear_x_q_from_scratch(
    scratch: torch.Tensor,
    *,
    tokens: int,
    in_features: int,
    output_dtype: torch.dtype,
) -> MXFP8Rows:
    layout = _block_fp8_linear_scratch_layout(
        tokens=tokens,
        in_features=in_features,
        out_features=1,
        output_dtype=output_dtype,
    )
    if scratch.dtype != torch.uint8:
        raise TypeError(
            f"block FP8 linear scratch must have dtype torch.uint8, got {scratch.dtype}"
        )
    if not scratch.is_contiguous():
        raise ValueError("block FP8 linear scratch must be contiguous")
    if int(scratch.numel()) < int(layout.nbytes):
        raise ValueError(
            f"block FP8 linear scratch has {int(scratch.numel())} bytes, requires {layout.nbytes}"
        )
    x_values = _scratch_view(
        scratch,
        offset_bytes=layout.x_values_offset_bytes,
        shape=(int(tokens), _physical_mxfp8_k(in_features)),
        dtype=torch.float8_e4m3fn,
    )
    x_scale_rows_u8 = _scratch_view(
        scratch,
        offset_bytes=layout.x_scale_rows_offset_bytes,
        shape=(
            1,
            int(tokens),
            _physical_mxfp8_k(in_features) // MXFP8_SCALE_VEC_SIZE,
        ),
        dtype=torch.uint8,
    )
    x_scale_mma_u8 = _scratch_view(
        scratch,
        offset_bytes=layout.x_scale_mma_offset_bytes,
        shape=layout.x_scale_mma_physical_shape,
        dtype=torch.uint8,
    )
    # Quantization overwrites every scale contributing to a logical output row.
    # Leave M128 padding unspecified to avoid two CUDA fills when binding scratch.
    x_scale_mma = x_scale_mma_u8.view(torch.float8_e8m0fnu).permute(
        3,
        4,
        1,
        5,
        2,
        0,
    )
    return MXFP8Rows(
        values=x_values,
        scale_rows=x_scale_rows_u8.view(torch.float8_e8m0fnu),
        scale_mma=x_scale_mma,
    )


def _source_2d(source: torch.Tensor) -> torch.Tensor:
    if source.ndim == 0:
        raise ValueError("source must have at least one dimension")
    return source.view(-1, source.shape[-1])


def _check_block_fp8_linear_tensors(
    x_q: MXFP8Rows,
    output: torch.Tensor,
    *,
    tokens: int,
    packed_weight: BlockFP8LinearWeight,
    output_dtype: torch.dtype,
) -> None:
    _check_mxfp8_rows_storage(
        x_q,
        m=tokens,
        k=_packed_physical_mxfp8_k(packed_weight),
        num_groups=1,
    )
    if output.shape != (tokens, packed_weight.out_features, 1):
        raise ValueError(
            "output must have shape "
            f"{(tokens, packed_weight.out_features, 1)}, got {tuple(output.shape)}"
        )
    if output.dtype != output_dtype:
        raise ValueError(
            f"output dtype {output.dtype} does not match input {output_dtype}"
        )


def build_block_fp8_linear_binding(
    *,
    plan: Plan,
    source: torch.Tensor,
    packed_weight: BlockFP8LinearWeight,
    x_q: MXFP8Rows,
    output: torch.Tensor,
    workspace: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    expected_m: int | None = None,
    mma_tiler_mn: tuple[int, int] | None = None,
) -> BlockFP8LinearBinding:
    if not isinstance(packed_weight, BlockFP8LinearWeight):
        raise TypeError("packed_weight must be a BlockFP8LinearWeight")
    source_2d = _source_2d(source)
    tokens, in_features = map(int, source_2d.shape)
    if in_features != packed_weight.in_features:
        raise ValueError("input K does not match packed weight")
    if source_2d.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("source dtype must be bf16/fp16")
    _check_block_fp8_linear_tensors(
        x_q, output, tokens=tokens, packed_weight=packed_weight, output_dtype=output.dtype,
    )
    return BlockFP8LinearBinding(
        plan=plan, source=source, packed_weight=packed_weight, x_q=x_q,
        output=output, workspace=workspace, bias=bias, expected_m=expected_m,
        mma_tiler_mn=mma_tiler_mn,
    )


def _scratch_plan(
    caps: BlockFP8LinearScratchCaps, mma_tiler_mn: tuple[int, int], *,
    workspace_nbytes: int = 0,
) -> _BlockFP8LinearScratchPlan:
    workspace_nbytes = int(workspace_nbytes)
    if workspace_nbytes < 0 or workspace_nbytes % 4:
        raise ValueError("block-FP8 fused workspace must be a nonnegative FP32 byte count")
    layout = _block_fp8_linear_scratch_layout(
        tokens=caps.max_tokens, in_features=caps.in_features,
        out_features=caps.out_features, output_dtype=caps.output_dtype,
    )
    total = _align_up(layout.nbytes, _SCRATCH_ALIGN_BYTES) + workspace_nbytes
    return _BlockFP8LinearScratchPlan(
        caps=caps,
        _scratch_specs=(scratch_buffer_spec(
            "block_fp8_linear.scratch", nbytes=total, device=caps.device,
        ),),
        mma_tiler_mn=mma_tiler_mn, workspace_nbytes=workspace_nbytes,
    )


def pack_block_fp8_linear_weight_mxfp8(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    block_size: Sequence[int] = (128, 128),
) -> BlockFP8LinearWeight:
    """Pack serialized block-FP8 linear weights for the native b12x MXFP8 GEMM.

    The checkpoint weight stays in E4M3 for UE8M0 scales. The 128x128 DSV4 or
    32x32 DSV4.1 scales expand once into the row/32-column SM120 MMA layout.
    """

    _check_gpu_tensor("weight", weight)
    _check_gpu_tensor("weight_scale", weight_scale)
    block_size = _check_block_size(block_size)
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [N,K], got {tuple(weight.shape)}")
    out_features, in_features = map(int, weight.shape)
    physical_in_features = _physical_mxfp8_k(in_features)
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    packed_weight = weight.detach()
    packed_scale = weight_scale.detach()
    if physical_in_features != in_features:
        padded_weight = torch.zeros(
            (out_features, physical_in_features),
            dtype=weight.dtype,
            device=weight.device,
        )
        padded_weight[:, :in_features] = packed_weight
        packed_weight = padded_weight
        logical_scale_tiles = math.ceil(in_features / block_size[1])
        physical_scale_tiles = math.ceil(physical_in_features / block_size[1])
        if (
            packed_scale.ndim > 0
            and int(packed_scale.shape[-1]) == logical_scale_tiles
            and physical_scale_tiles != logical_scale_tiles
        ):
            neutral_scale = 127 if packed_scale.dtype == torch.uint8 else 1.0
            padded_scale = torch.full(
                (*packed_scale.shape[:-1], physical_scale_tiles),
                neutral_scale,
                dtype=packed_scale.dtype,
                device=packed_scale.device,
            )
            padded_scale[..., :logical_scale_tiles] = packed_scale
            packed_scale = padded_scale
    packed = pack_fp8_block_scaled_weight_mxfp8(
        packed_weight,
        packed_scale,
        m=out_features,
        k=physical_in_features,
        num_groups=1,
        block_size=block_size,
    )
    return BlockFP8LinearWeight(
        weight=packed,
        in_features=in_features,
        out_features=out_features,
        block_size=block_size,
    )




def quantize_block_fp8_linear_input_mxfp8(
    source_tk: torch.Tensor,
    *,
    plan: Plan,
    out: MXFP8Rows | None = None,
) -> MXFP8Rows:
    """Quantize through the launcher retained by a prepared block-FP8 plan."""
    state = require_prepared(plan, "gemm.block_fp8_linear", source_tk.device)
    return state.quantize_input(source_tk, out=out)


def block_fp8_linear_mxfp8(
    source: torch.Tensor | None = None,
    packed_weight: BlockFP8LinearWeight | None = None,
    *,
    plan: Plan | None = None,
    bias: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    binding: BlockFP8LinearBinding | None = None,
    stream: object = None,
) -> torch.Tensor:
    """Execute only a session-prepared block-FP8 declaration."""
    if binding is not None:
        if any(value is not None for value in (source, packed_weight, plan, bias, workspace)):
            raise ValueError("block-FP8 binding owns inputs and prepared plan")
        state = require_prepared(binding.plan, "gemm.block_fp8_linear", binding.source.device)
        return state.run_binding(binding, stream=stream)
    if source is None or packed_weight is None or plan is None:
        raise TypeError(
            "block_fp8_linear_mxfp8 requires source, packed_weight, and a prepared plan"
        )
    state = require_prepared(plan, "gemm.block_fp8_linear", source.device)
    return state.run(source, packed_weight, bias=bias, workspace=workspace, stream=stream)


__all__ = [
    "BlockFP8LinearBinding",
    "BlockFP8LinearScratchCaps",
    "BlockFP8LinearWeight",
    "build_block_fp8_linear_binding",
    "block_fp8_linear_mxfp8",
    "pack_block_fp8_linear_weight_mxfp8",
    "quantize_block_fp8_linear_input_mxfp8",
]
