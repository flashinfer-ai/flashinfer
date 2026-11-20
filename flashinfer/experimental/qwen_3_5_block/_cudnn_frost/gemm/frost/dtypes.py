# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared dtype conversion helpers for the GEMM engine."""

from __future__ import annotations

import functools
from typing import Any

import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

from .fusion_ir import Dtype, FusionChain, MoeSwapAbSpec

DTYPE_TO_CUTLASS: dict[Dtype, str] = {
    "bf16": "cutlass.BFloat16",
    "fp16": "cutlass.Float16",
    "fp32": "cutlass.Float32",
    "int8": "cutlass.Int8",
    "fp8_e4m3": "cutlass.Float8E4M3FN",
    "fp8_e5m2": "cutlass.Float8E5M2",
    "fp8_e8m0": "cutlass.Float8E8M0FNU",
    "fp8_e5m3": "cutlass.Uint8",
    "fp4_e2m1": "cutlass.Float4E2M1FNx2",
    "uint8": "cutlass.Uint8",
    "int32": "cutlass.Int32",
    "int64": "cutlass.Int64",
}

# internal dtype -> element size in bytes (FP4 packs 2/byte; counted as 1 here).
DTYPE_BYTES: dict[Dtype, int] = {
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
    "int8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "fp8_e8m0": 1,
    "fp8_e5m3": 1,
    "fp4_e2m1": 1,
    "uint8": 1,
    "int32": 4,
    "int64": 8,
}

DTYPE_BITS: dict[Dtype, int] = {**{dt: nbytes * 8 for dt, nbytes in DTYPE_BYTES.items()}, "fp4_e2m1": 4}

DTYPE_GPU_ARCH_RANGES: dict[Dtype, tuple[tuple[int, int], ...]] = {
    "fp8_e5m3": ((107, 110),),
}


def _fmt_ranges(ranges: tuple[tuple[int, int], ...]) -> str:
    return " or ".join(f"{lo} <= SM < {hi}" for lo, hi in ranges)


def dtype_arch_reject(chain: FusionChain, arch: "int | None") -> "str | None":
    """Why the active GPU cannot run this chain's dtypes, or ``None``."""
    if arch is None:
        return None
    for dtype in sorted(chain.dtypes_used()):
        ranges = DTYPE_GPU_ARCH_RANGES.get(dtype)
        if ranges is not None and not any(lo <= arch < hi for lo, hi in ranges):
            return f"dtype {dtype!r} exists only on {_fmt_ranges(ranges)}, but the active GPU is sm_{arch}"
    return None


# input dtype -> tcgen05 MMA kind.
DTYPE_TO_MMA_KIND: dict[Dtype, str] = {
    "bf16": "nvvm.Tcgen05MMAKind.F16",
    "fp16": "nvvm.Tcgen05MMAKind.F16",
    "fp8_e4m3": "nvvm.Tcgen05MMAKind.F8F6F4",
    "fp8_e5m2": "nvvm.Tcgen05MMAKind.F8F6F4",
    "int8": "nvvm.Tcgen05MMAKind.INT8",
}

# cudnn.data_type <-> internal dtype.
DTYPE_FROM_CUDNN: dict[Any, Dtype] = {
    cudnn.data_type.BFLOAT16: "bf16",
    cudnn.data_type.HALF: "fp16",
    cudnn.data_type.FLOAT: "fp32",
    cudnn.data_type.INT8: "int8",
    cudnn.data_type.FP8_E4M3: "fp8_e4m3",
    cudnn.data_type.FP8_E5M2: "fp8_e5m2",
    cudnn.data_type.FP8_E8M0: "fp8_e8m0",
    cudnn.data_type.FP8_E5M3: "fp8_e5m3",
    cudnn.data_type.FP4_E2M1: "fp4_e2m1",
    cudnn.data_type.UINT8: "uint8",
    cudnn.data_type.INT32: "int32",
    cudnn.data_type.INT64: "int64",
}

CUDNN_FROM_DTYPE: dict[Dtype, Any] = {v: k for k, v in DTYPE_FROM_CUDNN.items()}


MAX_MEM_ACCESS_BYTES = 32

MAX_EPI_CHUNK_ELEMS = 32


def _pow2_floor(x: int, cap: int = MAX_MEM_ACCESS_BYTES) -> int:
    """Largest power of 2 (<= cap) dividing ``x``; ``cap`` when ``x`` is 0 (no
    constraint) or divisible by a higher power than the cap."""
    if x <= 0:
        return cap
    return min(x & (-x), cap)  # x & -x isolates the lowest set bit


def tensor_alignment(shape, stride, elem_bytes: int, ptr: "int | None" = None, cap: int = MAX_MEM_ACCESS_BYTES) -> int:
    """Byte alignment of a tensor = ``min(A_ptr, A_stride, A_shape)`` capped at ``cap``."""
    a = _layout_alignment(tuple(shape), tuple(stride), elem_bytes, cap)
    if ptr is not None:
        a = min(a, _pow2_floor(int(ptr), cap))
    return a


@functools.lru_cache(maxsize=256)
def _layout_alignment(shape: tuple, stride: tuple, elem_bytes: int, cap: int) -> int:
    """``min(A_stride, A_shape)`` -- the half the pointer does not enter."""
    stride_align = cap
    for sh, st in zip(shape, stride):
        if sh != 1 and st != 1:
            stride_align = min(stride_align, _pow2_floor(int(st) * elem_bytes, cap))

    shape_align = None
    for sh, st in zip(shape, stride):
        if st == 1 and sh != 1:
            shape_align = _pow2_floor(int(sh) * elem_bytes, cap)
            break
    if shape_align is None:
        shape_align = min(elem_bytes, cap)
    return min(stride_align, shape_align)


def allowed_store_vsize(dim, stride, dtype: str) -> int:
    """Return the widest store vector allowed by the output layout."""
    elem_bytes = DTYPE_BYTES[dtype]
    align_bytes = tensor_alignment(dim, stride, elem_bytes)
    return align_bytes // elem_bytes


def dense_output_layout(chain: FusionChain, dtype: "str", dim, stride) -> tuple:
    """Resolve the declared or default dense output layout."""
    mm = chain.matmul
    if not stride:
        return (mm.batch, mm.M, mm.N), (mm.M * mm.N, mm.N, 1)
    if not dim:
        n = mm.N // 2 if dtype == "fp4_e2m1" else mm.N
        dim = (mm.batch, mm.M, n)
    return dim, stride


def _allowed_vsize(chain: FusionChain, dtype: "str", dim=None, stride=None) -> int:
    """This dense output's widest STG vector (elements) = its OWN declared-layout
    alignment."""
    dim, stride = dense_output_layout(chain, dtype, dim, stride)
    vsize = allowed_store_vsize(dim, stride, dtype)
    if isinstance(chain.moe, MoeSwapAbSpec):
        vsize = min(vsize, _pow2_floor(chain.moe.offset_multiple), _pow2_floor(chain.matmul.N))
    return vsize


def _compute_output_vec_bytes(chain: FusionChain, tile_cols: "int | None" = None) -> int:
    """Epilogue chunk width in bytes of the first dense output's dtype."""
    elem_bytes = DTYPE_BYTES[chain.output_dtype]
    widths = [_allowed_vsize(chain, spec.dtype, spec.dim, spec.stride) for spec in chain.output_specs if spec.dtype != "fp4_e2m1"]
    if chain.quants:
        vsize = max(q.block_size for q in chain.quants)
    elif chain.out_major == "m" or not widths:
        vsize = MAX_EPI_CHUNK_ELEMS
    else:
        vsize = min(widths)
    vsize = min(vsize, MAX_EPI_CHUNK_ELEMS)
    if not chain.quants and (chain.out_major == "m" or not widths):
        vsize = min(vsize, _pow2_floor(chain.matmul.N, cap=MAX_EPI_CHUNK_ELEMS))
    if tile_cols is not None:
        vsize = min(vsize, _pow2_floor(tile_cols, cap=MAX_EPI_CHUNK_ELEMS))
    if isinstance(chain.moe, MoeSwapAbSpec) and not chain.quants:
        vsize = min(vsize, _pow2_floor(chain.moe.offset_multiple), _pow2_floor(chain.matmul.N))
    return vsize * elem_bytes


def _aux_align_reqs(chain: FusionChain, vec_bytes: "int | None" = None) -> dict:
    """Return the required byte alignment for each auxiliary input."""
    reqs: dict = {}
    if vec_bytes is None:
        vec_bytes = _compute_output_vec_bytes(chain)
    vsize = vec_bytes // DTYPE_BYTES[chain.output_dtype]
    for aux in chain.aux_tensors:
        aeb = DTYPE_BYTES[aux.dtype]
        if aux.bcast_mode in ("per_col", "per_elem"):
            reqs[aux.name] = min(tensor_alignment(aux.dim, aux.stride, aeb), vsize * aeb)
            if isinstance(chain.moe, MoeSwapAbSpec):
                reqs[aux.name] = min(reqs[aux.name], _pow2_floor(chain.moe.offset_multiple) * aeb) if aux.stride[-1] == 1 else aeb
        else:
            reqs[aux.name] = aeb
    return reqs


def _output_align_reqs(chain: FusionChain, tma_slots: "frozenset[int]", vec_bytes: "int | None" = None) -> "list[int]":
    """Return the required byte alignment for each output."""
    if chain.has_moe and chain.moe.mode == "combine":
        return [DTYPE_BYTES[out.dtype] for out in chain.outputs]
    if vec_bytes is None:
        vec_bytes = _compute_output_vec_bytes(chain)
    vsize = vec_bytes // DTYPE_BYTES[chain.output_dtype]
    reqs = []
    for i, out in enumerate(chain.outputs):
        eb = DTYPE_BYTES[out.dtype]
        if out.is_reduction or out.is_quant_scale:
            reqs.append(1)
        elif chain.output_specs[i].major == "m":
            reqs.append(16 if i in tma_slots else eb)
        elif i in tma_slots:
            reqs.append(16)
        elif out.dtype == "fp4_e2m1":
            reqs.append(max(vsize // 2, 4))
        else:
            reqs.append(min(vsize, _allowed_vsize(chain, out.dtype, out.dim, out.stride)) * eb)
    return reqs
