"""Prepared plan for the fixed CuTe MXFP8 row quantizer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x._lib.quant.mxfp8_rows import (
    _get_compiled_mxfp8_rows_quant,
    mxfp8_rows_quant_launch_options,
)
from b12x.preparation import (
    FrozenMapping,
    MemoryRequirements,
    Plan,
)
from b12x.preparation.types import plan_from_handle, require_prepared
from ._tuning import Mxfp8Config, Mxfp8Query, TUNING


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float16:
        return "float16"
    raise TypeError(f"MXFP8 CuTe quantizer requires BF16 or FP16 input, got {dtype}")


def _scale_storage_name(dtype: torch.dtype) -> str:
    if dtype is torch.uint8:
        return "uint8"
    if dtype is torch.float8_e8m0fnu:
        return "float8_e8m0fnu"
    raise ValueError("MXFP8 scale storage must be uint8 or float8_e8m0fnu")


def _check_pointer_alignment(name: str, tensor: torch.Tensor) -> None:
    if tensor.data_ptr() % 16:
        raise ValueError(f"MXFP8 {name} must be 16-byte aligned")


def _scale_rows_layout(scale_rows: torch.Tensor, rows: int, columns: int) -> str:
    groups = columns // 32
    if scale_rows.ndim == 2 and tuple(scale_rows.shape) == (rows, groups):
        return "row_major"
    if scale_rows.ndim == 3 and tuple(scale_rows.shape) == (1, rows, groups):
        return "grouped"
    raise ValueError("scale_rows must have shape [M,K/32] or [1,M,K/32] for MXFP8 rows")


def _scale_mma_layout(scale_mma: torch.Tensor, rows: int, columns: int) -> str:
    groups = columns // 32
    m_tiles = (rows + 127) // 128
    k_tiles = (groups + 3) // 4
    physical_nbytes = m_tiles * k_tiles * 512
    if (
        scale_mma.ndim == 1
        and scale_mma.is_contiguous()
        and scale_mma.numel() >= physical_nbytes
    ):
        return "linear_storage"
    expected_shape = (32, 4, m_tiles, 4, k_tiles, 1)
    expected_stride = (16, 4, k_tiles * 512, 1, 512, m_tiles * k_tiles * 512)
    if (
        tuple(scale_mma.shape) == expected_shape
        and scale_mma.stride() == expected_stride
    ):
        return "dense_gemm_swizzled"
    raise ValueError(
        "scale_mma must be contiguous linear physical storage or the canonical "
        f"dense-GEMM MXFP8 swizzled layout {expected_shape}"
    )


def query_from_call(
    source: torch.Tensor,
    values: torch.Tensor,
    scale_rows: torch.Tensor,
    scale_mma: torch.Tensor,
    *,
    value_order: str = "linear",
    expected_m: int | None = None,
) -> Mxfp8Query:
    """Normalize all immutable source/output layout semantics for a declaration."""
    if source.ndim != 2 or not source.is_contiguous():
        raise ValueError("CuTe MXFP8 quantizer requires contiguous [M,K] input")
    rows, columns = (int(dimension) for dimension in source.shape)
    planned_rows = rows if expected_m is None else expected_m
    if type(planned_rows) is not int or planned_rows < rows or planned_rows <= 0:
        raise ValueError("expected_m must be a positive row bound covering the source")
    if (
        values.ndim != 2
        or tuple(values.shape) != (rows, columns)
        or not values.is_contiguous()
    ):
        raise ValueError("values must be contiguous [M,K] MXFP8 storage")
    if values.dtype is not torch.float8_e4m3fn:
        raise ValueError("values must use float8_e4m3fn storage")
    if not scale_rows.is_contiguous():
        raise ValueError("scale_rows must be contiguous MXFP8 storage")
    _check_pointer_alignment("source", source)
    _check_pointer_alignment("values", values)
    _check_pointer_alignment("scale_rows", scale_rows)
    _check_pointer_alignment("scale_mma", scale_mma)
    scale_rows_storage = _scale_storage_name(scale_rows.dtype)
    scale_mma_storage = _scale_storage_name(scale_mma.dtype)
    return Mxfp8Query(
        rows=planned_rows,
        columns=columns,
        dtype=_dtype_name(source.dtype),
        value_order=value_order,
        scale_rows_layout=_scale_rows_layout(scale_rows, rows, columns),
        scale_rows_storage=scale_rows_storage,
        scale_mma_layout=_scale_mma_layout(scale_mma, rows, columns),
        scale_mma_storage=scale_mma_storage,
    )


@program_cache(scope="preparation")
def compile_mxfp8_rows(query_payload, ordinal, sm_count):
    """Compile the actual fixed CuTe quantizer from metadata only."""
    query = Mxfp8Query(**dict(query_payload))
    source_dtype = getattr(torch, query.dtype)
    subgroup_width, threads = mxfp8_rows_quant_launch_options(
        query.rows, query.value_order
    )
    return _get_compiled_mxfp8_rows_quant(
        query.columns,
        source_dtype,
        subgroup_width,
        threads,
        query.value_order,
        device_ordinal=ordinal,
        sm_count=sm_count,
    )


@dataclass(frozen=True)
class _Mxfp8RowsExecutionState:
    query: Mxfp8Query
    device: torch.device
    quantizer: object

    def _check(self, source, values, scale_rows, scale_mma):
        query = self.query
        if source.device != self.device or source.dtype is not getattr(
            torch, query.dtype
        ):
            raise ValueError("MXFP8 source device/dtype differs from preparation")
        if source.ndim != 2 or not source.is_contiguous():
            raise ValueError("CuTe MXFP8 quantizer requires contiguous [M,K] input")
        _check_pointer_alignment("source", source)
        rows, columns = (int(dimension) for dimension in source.shape)
        if columns != query.columns or rows > query.rows:
            raise ValueError("MXFP8 source geometry exceeds prepared invocation")
        for name, tensor in (
            ("values", values),
            ("scale_rows", scale_rows),
            ("scale_mma", scale_mma),
        ):
            if tensor.device != self.device:
                raise ValueError(f"MXFP8 {name} device differs from preparation")
            _check_pointer_alignment(name, tensor)
        if values.dtype is not torch.float8_e4m3fn:
            raise ValueError("values must use float8_e4m3fn storage")
        if _scale_storage_name(scale_rows.dtype) != query.scale_rows_storage:
            raise ValueError("scale_rows storage differs from preparation")
        if _scale_storage_name(scale_mma.dtype) != query.scale_mma_storage:
            raise ValueError("scale_mma storage differs from preparation")
        if (
            values.ndim != 2
            or tuple(values.shape) != (rows, columns)
            or not values.is_contiguous()
        ):
            raise ValueError("values must have the prepared contiguous [M,K] layout")
        expected_rows = (
            (rows, columns // 32)
            if query.scale_rows_layout == "row_major"
            else (1, rows, columns // 32)
        )
        if tuple(scale_rows.shape) != expected_rows or not scale_rows.is_contiguous():
            raise ValueError("scale_rows layout differs from preparation")
        m_tiles = (rows + 127) // 128
        k_tiles = ((columns // 32) + 3) // 4
        physical_nbytes = m_tiles * k_tiles * 512
        if query.scale_mma_layout == "linear_storage":
            if (
                scale_mma.ndim != 1
                or not scale_mma.is_contiguous()
                or scale_mma.numel() < physical_nbytes
            ):
                raise ValueError("scale_mma linear storage differs from preparation")
        else:
            expected_mma = (32, 4, m_tiles, 4, k_tiles, 1)
            expected_stride = (16, 4, k_tiles * 512, 1, 512, m_tiles * k_tiles * 512)
            if (
                tuple(scale_mma.shape) != expected_mma
                or scale_mma.stride() != expected_stride
            ):
                raise ValueError("scale_mma swizzled layout differs from preparation")

    def run(self, source, values, scale_rows, scale_mma):
        self._check(source, values, scale_rows, scale_mma)
        self.quantizer(source, values, scale_rows, scale_mma)


def plan(
    query: Mxfp8Query,
    *,
    invocation: FrozenMapping = FrozenMapping(),
    override: Mxfp8Config | None = None,
) -> Plan:
    """Declare a fixed MXFP8 row-quantization invocation without compiling it."""
    if not isinstance(query, Mxfp8Query):
        raise TypeError("query must be Mxfp8Query")
    if invocation:
        raise ValueError("MXFP8 row-quantization semantics belong in Mxfp8Query")

    def materialize(selection, device):
        if device.ordinal is None:
            raise RuntimeError("MXFP8 row quantization requires a CUDA device")
        quantizer = compile_mxfp8_rows(
            TUNING.encode_query(query), device.ordinal, device.identity.sm_count
        )
        return _Mxfp8RowsExecutionState(
            query, torch.device("cuda", device.ordinal), quantizer
        )

    return Plan(
        contract=TUNING,
        query=query,
        invocation=FrozenMapping(invocation),
        override=override,
        shared=True,
        _compile_jobs=lambda config, device: (
            CompileJob.create(
                "b12x.quantization.mxfp8._preparation:compile_mxfp8_rows",
                TUNING.encode_query(query),
                device.ordinal,
                device.identity.sm_count,
            ),
        ),
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=materialize,
    )


@torch.library.custom_op(
    "b12x::mxfp8_rows_prepared",
    mutates_args=("values", "scale_rows", "scale_mma"),
)
def _quantize_rows_prepared(
    source: torch.Tensor,
    values: torch.Tensor,
    scale_rows: torch.Tensor,
    scale_mma: torch.Tensor,
    plan_handle: int,
) -> None:
    state = require_prepared(
        plan_from_handle(plan_handle), "quantization.mxfp8", source.device
    )
    state.run(source, values, scale_rows, scale_mma)


@_quantize_rows_prepared.register_fake
def _quantize_rows_prepared_fake(
    source: torch.Tensor,
    values: torch.Tensor,
    scale_rows: torch.Tensor,
    scale_mma: torch.Tensor,
    plan_handle: int,
) -> None:
    del source, values, scale_rows, scale_mma, plan_handle


def quantize_rows(
    source: torch.Tensor,
    values: torch.Tensor,
    scale_rows: torch.Tensor,
    scale_mma: torch.Tensor,
    *,
    plan: Plan,
) -> None:
    """Quantize through a session-prepared CuTe plan only."""
    _quantize_rows_prepared(source, values, scale_rows, scale_mma, plan.handle)


__all__ = ["Mxfp8Config", "Mxfp8Query", "plan", "query_from_call", "quantize_rows"]
