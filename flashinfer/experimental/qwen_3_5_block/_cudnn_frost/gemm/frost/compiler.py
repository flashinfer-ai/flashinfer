# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared GEMM compiler helpers and active-family entry points."""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.util
import logging
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Generic, Protocol, TypeVar

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost import buffers
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.workspace import Workspace

from .arch_family import active_family
from .dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS, _pow2_floor, tensor_alignment
from .fusion_ir import Axis, FusionChain, MoeSwapAbSpec, TensorRef, segmented_row_scale_capacity_rows
from .recipe import AX_MN, FROM_M, FROM_N, REDUCTION_INIT_VALUE, _output_rule, contiguous_modulus, expected_shape


class _RuntimeConfig(Protocol):
    @property
    def swap_ab(self) -> bool: ...

    @property
    def split_k_slices(self) -> int: ...

    @property
    def workspace_split_k(self) -> bool: ...


class _RuntimeBinding(Protocol):
    a_operands: list[Any]
    b_operands: list[Any]
    outputs: list[Any]
    aux: list[Any]
    sfa_operands: list[Any]
    sfb_operands: list[Any]
    first_token_offset: Any
    token_index: Any
    token_ks: Any
    top_k_scores: Any

    def bound_tensors(self) -> list[Any]: ...


_ConfigT = TypeVar("_ConfigT", bound=_RuntimeConfig)
_BindingT = TypeVar("_BindingT", bound=_RuntimeBinding)

_CTAMMA_SUFFIX = re.compile(r"_[12]ctamma$")

_INJECT_MARKER_LINE = re.compile(r"^([ \t]*)# *@@([A-Z0-9_]+)@@[ \t]*\n", flags=re.MULTILINE)


def _aux_kernel_shape(aux: TensorRef) -> tuple[int | Axis, ...]:
    """Lower logical extents to the existing minimum-rank-2 aux ABI."""
    shape = aux.symbolic_shape
    return (1,) * max(0, 2 - len(shape)) + shape


def _aux_fake_shape_code(aux: TensorRef) -> str:
    """Render lowered extents without reconstructing broadcasts."""
    symbols = {Axis.M: "sym_m", Axis.N: "sym_n", Axis.K: "sym_k", Axis.BATCH: "sym_l", Axis.GROUP: "sym_g"}
    return "(" + ", ".join(symbols[extent] if isinstance(extent, Axis) else str(extent) for extent in _aux_kernel_shape(aux)) + ")"


def _aux_can_use_explicit_fake_stride(aux: TensorRef) -> bool:
    """Check whether auxiliary strides can be encoded directly."""
    # Unit-axis strides also change when dynamic K is rebound; do not bake
    # the originally declared K-sized strides into a per-K fake tensor.
    if aux.bcast_mode == "per_k" or len(aux.dim) not in (2, 3):
        return False
    stride1_dims = [i for i, stride in enumerate(aux.stride) if stride == 1]
    if len(stride1_dims) <= 1:
        return True
    nontrivial = [i for i in stride1_dims if aux.dim[i] != 1]
    return len(nontrivial) == 1


def _aux_fake_block(aux_tensors: list[TensorRef], *, dynamic_strides: bool = False, align_reqs: "dict | None" = None) -> str:
    """Declare `fake_<name>` for each aux tensor."""
    lines = []
    for aux in aux_tensors:
        shape = _aux_fake_shape_code(aux)
        dtype = DTYPE_TO_CUTLASS[aux.dtype]
        _align = align_reqs[aux.name] if align_reqs is not None else 16
        if dynamic_strides and _aux_can_use_explicit_fake_stride(aux):
            stride = "(" + ", ".join(str(s) for s in aux.stride) + ")"
            lines.append(f"fake_{aux.name} = cute.runtime.make_fake_tensor({dtype}, {shape}, " f"stride={stride}, assumed_align={_align})")
        else:
            _rank = len(_aux_kernel_shape(aux))
            stride = "(" + ", ".join("cute.sym_int64()" for _ in range(_rank)) + ")"
            lines.append(f"fake_{aux.name} = cute.runtime.make_fake_tensor({dtype}, {shape}, " f"stride={stride}, assumed_align={_align})")
    return "\n".join(lines) if lines else "pass"


def _aux_signature_block(aux_tensors: list[TensorRef]) -> str:
    """Comma-separated signature params (one per line). No baked indent."""
    if not aux_tensors:
        return ""
    return ",\n".join(f"{aux.name}: cute.Tensor" for aux in aux_tensors) + ","


def _aux_call_block(aux_tensors: list[TensorRef], prefix: str = "") -> str:
    """Comma-separated args (one per line) for a call list. No baked indent."""
    if not aux_tensors:
        return ""
    return ",\n".join(f"{prefix}{aux.name}" for aux in aux_tensors) + ","


def _reduction_stride_kernel_params(chain: FusionChain) -> str:
    """Declare reduction output stride parameters."""
    params: list[str] = []
    for i in range(len(chain.output_specs)):
        params.extend(
            [
                f"out_stride_m_{i}: cutlass.Int64",
                f"out_stride_n_{i}: cutlass.Int64",
                f"out_stride_l_{i}: cutlass.Int64",
            ]
        )
    for i in range(len(chain.reductions)):
        params.extend(
            [
                f"red_stride_m_{i}: cutlass.Int64",
                f"red_stride_n_{i}: cutlass.Int64",
                f"red_stride_l_{i}: cutlass.Int64",
            ]
        )
    for i in range(len(chain.quants)):
        params.extend(
            [
                f"quant_scale_stride_m_{i}: cutlass.Int64",
                f"quant_scale_stride_n_{i}: cutlass.Int64",
                f"quant_scale_stride_l_{i}: cutlass.Int64",
            ]
        )
    return ",\n".join(params) + "," if params else ""


def _reduction_stride_host_unpack(chain: FusionChain) -> str:
    """Unpack reduction output strides from the launch recipe."""
    if not chain.output_specs and not chain.reductions and not chain.quants:
        return ""
    lines = []
    for i in range(len(chain.output_specs)):
        lines.extend(
            [
                f"out_stride_m_{i} = problem_size[_stride_idx]",
                f"out_stride_n_{i} = problem_size[_stride_idx + 1]",
                f"out_stride_l_{i} = problem_size[_stride_idx + 2]",
                "_stride_idx += 3",
            ]
        )
    for i in range(len(chain.reductions)):
        lines.extend(
            [
                f"red_stride_m_{i} = problem_size[_stride_idx]",
                f"red_stride_n_{i} = problem_size[_stride_idx + 1]",
                f"red_stride_l_{i} = problem_size[_stride_idx + 2]",
                "_stride_idx += 3",
            ]
        )
    for i in range(len(chain.quants)):
        lines.extend(
            [
                f"quant_scale_stride_m_{i} = problem_size[_stride_idx]",
                f"quant_scale_stride_n_{i} = problem_size[_stride_idx + 1]",
                f"quant_scale_stride_l_{i} = problem_size[_stride_idx + 2]",
                "_stride_idx += 3",
            ]
        )
    return "\n".join(lines)


def _reduction_stride_host_unpack_from(chain: FusionChain, start_index: int) -> str:
    """Unpack reduction strides from a selected argument offset."""
    if not chain.output_specs and not chain.reductions and not chain.quants:
        return ""
    lines: list[str] = []
    n_dense = len(chain.output_specs)
    for i in range(n_dense):
        base = start_index + 3 * i
        lines.extend(
            [
                f"out_stride_m_{i} = problem_size[{base}]",
                f"out_stride_n_{i} = problem_size[{base + 1}]",
                f"out_stride_l_{i} = problem_size[{base + 2}]",
            ]
        )
    for i in range(len(chain.reductions)):
        base = start_index + 3 * (n_dense + i)
        lines.extend(
            [
                f"red_stride_m_{i} = problem_size[{base}]",
                f"red_stride_n_{i} = problem_size[{base + 1}]",
                f"red_stride_l_{i} = problem_size[{base + 2}]",
            ]
        )
    for i in range(len(chain.quants)):
        base = start_index + 3 * (n_dense + len(chain.reductions) + i)
        lines.extend(
            [
                f"quant_scale_stride_m_{i} = problem_size[{base}]",
                f"quant_scale_stride_n_{i} = problem_size[{base + 1}]",
                f"quant_scale_stride_l_{i} = problem_size[{base + 2}]",
            ]
        )
    return "\n".join(lines)


def _reduction_stride_names(chain: FusionChain) -> list[str]:
    """The runtime stride scalars, in the order every kernel signature takes them."""
    names: list[str] = []
    for i in range(len(chain.output_specs)):
        names.extend([f"out_stride_m_{i}", f"out_stride_n_{i}", f"out_stride_l_{i}"])
    for i in range(len(chain.reductions)):
        names.extend([f"red_stride_m_{i}", f"red_stride_n_{i}", f"red_stride_l_{i}"])
    for i in range(len(chain.quants)):
        names.extend([f"quant_scale_stride_m_{i}", f"quant_scale_stride_n_{i}", f"quant_scale_stride_l_{i}"])
    return names


def _reduction_stride_host_pass(chain: FusionChain) -> str:
    """Pass reduction output strides to the kernel."""
    args = _reduction_stride_names(chain)
    return ",\n".join(args) + "," if args else ""


def _splitk_epilogue_bindings(chain: FusionChain, n_taps: int) -> str:
    """Bind split-K reducer arguments to epilogue names."""
    lines = []
    for i in range(n_taps):
        lines.append(f"gC_tap_{i}_ptr = taps[{i}].iterator.raw_ptr()")
        lines.append(f"VEC_BYTES_TAP_{i} = vec_bytes_tap_{i}")
    for i, name in enumerate(_reduction_stride_names(chain)):
        lines.append(f"{name} = strides[{i}]")
    for i, aux in enumerate(chain.aux_tensors):
        lines.append(f"{aux.name} = aux[{i}]")
    return "\n".join(lines) if lines else "pass"


def _reduction_stride_compile_decls(chain: FusionChain) -> str:
    """Declare symbolic reduction strides for compilation."""
    lines: list[str] = []
    for i in range(len(chain.output_specs)):
        lines.extend(
            [
                f"sym_out_stride_m_{i} = cute.sym_int64()",
                f"sym_out_stride_n_{i} = cute.sym_int64()",
                f"sym_out_stride_l_{i} = cute.sym_int64()",
            ]
        )
    for i in range(len(chain.reductions)):
        lines.extend(
            [
                f"sym_red_stride_m_{i} = cute.sym_int64()",
                f"sym_red_stride_n_{i} = cute.sym_int64()",
                f"sym_red_stride_l_{i} = cute.sym_int64()",
            ]
        )
    for i in range(len(chain.quants)):
        lines.extend(
            [
                f"sym_quant_scale_stride_m_{i} = cute.sym_int64()",
                f"sym_quant_scale_stride_n_{i} = cute.sym_int64()",
                f"sym_quant_scale_stride_l_{i} = cute.sym_int64()",
            ]
        )
    return "\n".join(lines)


def _reduction_stride_compile_symbols(chain: FusionChain) -> str:
    """Collect symbolic reduction stride arguments."""
    args: list[str] = []
    for i in range(len(chain.output_specs)):
        args.extend(
            [
                f"sym_out_stride_m_{i}",
                f"sym_out_stride_n_{i}",
                f"sym_out_stride_l_{i}",
            ]
        )
    for i in range(len(chain.reductions)):
        args.extend(
            [
                f"sym_red_stride_m_{i}",
                f"sym_red_stride_n_{i}",
                f"sym_red_stride_l_{i}",
            ]
        )
    for i in range(len(chain.quants)):
        args.extend(
            [
                f"sym_quant_scale_stride_m_{i}",
                f"sym_quant_scale_stride_n_{i}",
                f"sym_quant_scale_stride_l_{i}",
            ]
        )
    return ",\n".join(args) + "," if args else ""


def _cvt_f32_to_fp8_scale_bits(fn_name: str, dsl_dtype: str, rnd: str) -> list[str]:
    """Generate fp32-to-scale conversion code."""
    return [
        "",
        "",
        f"def {fn_name}_x2(a, b):",
        "    lo = cutlass.Float32(a).ir_value()",
        "    hi = cutlass.Float32(b).ir_value()",
        "    packed = _frost_nvvm.convert_f32x2_to_f8x2(",
        "        _frost_ir.VectorType.get([2], cutlass.Int8.mlir_type),",
        "        hi,",
        "        lo,",
        f"        _frost_ir.TypeAttr.get(cutlass.{dsl_dtype}.mlir_type),",
        f"        rnd=_frost_nvvm.FPRoundingMode.{rnd},",
        "        sat=_frost_nvvm.SaturationMode.SATFINITE,",
        "    )",
        "    return cutlass.Int32(_frost_llvm.zext(_frost_T.i32(), _frost_llvm.bitcast(cutlass.Int16.mlir_type, packed)))",
        "",
        "",
        f"def {fn_name}(x):",
        f"    return {fn_name}_x2(x, x) & 0xFF",
    ]


def _cvt_e5m3_bits_to_f32() -> list[str]:
    """The inverse of :func:`_cvt_f32_to_fp8_scale_bits` for ue5m3."""
    return [
        "",
        "",
        "def _frost_e5m3_bits_to_f32_x2(a, b):",
        "    lo = _frost_llvm.trunc(_frost_T.i8(), cutlass.Int32(a).ir_value(), _frost_llvm.IntegerOverflowFlags.none)",
        "    hi = _frost_llvm.trunc(_frost_T.i8(), cutlass.Int32(b).ir_value(), _frost_llvm.IntegerOverflowFlags.none)",
        "    pair = _frost_vector.from_elements(_frost_ir.VectorType.get([2], cutlass.Int8.mlir_type), [lo, hi])",
        "    widened = _frost_nvvm.convert_f8x2_to_bf16x2(",
        "        _frost_ir.VectorType.get([2], cutlass.BFloat16.mlir_type),",
        "        pair,",
        "        _frost_ir.TypeAttr.get(cutlass.FloatNV8E5M3FNU.mlir_type),",
        "    )",
        "    return (",
        "        cutlass.Float32(cutlass.BFloat16(_frost_vector.extract(widened, dynamic_position=[], static_position=[0]))),",
        "        cutlass.Float32(cutlass.BFloat16(_frost_vector.extract(widened, dynamic_position=[], static_position=[1]))),",
        "    )",
        "",
        "",
        "def _frost_e5m3_bits_to_f32(b):",
        "    return _frost_e5m3_bits_to_f32_x2(b, b)[0]",
    ]


def _quant_device_imports(chain: FusionChain) -> list[str]:
    """Generate device scale converters."""
    kinds = {q.scale_dtype for q in chain.quants}
    lines: list[str] = []
    if "fp8_e8m0" in kinds:
        lines += _cvt_f32_to_fp8_scale_bits("_frost_cvt_f32_to_e8m0_bits", "Float8E8M0FNU", "RP")
    if "fp8_e5m3" in kinds:
        lines += _cvt_f32_to_fp8_scale_bits("_frost_cvt_f32_to_e5m3_bits", "FloatNV8E5M3FNU", "RP")
        lines += _cvt_e5m3_bits_to_f32()
    if not lines:
        return []
    return [
        "from cutlass.cutlass_dsl import T as _frost_T",
        "from cutlass._mlir import ir as _frost_ir",
        "from cutlass._mlir.dialects import llvm as _frost_llvm, nvvm as _frost_nvvm, vector as _frost_vector",
    ] + lines


def _template_stem(file: str) -> str:
    """Return the template stem used in kernel symbols."""
    return _CTAMMA_SUFFIX.sub("", file.removesuffix(".py"))


def _resolve_path_blocks(src: str, use_tma_store_epi: bool) -> str:
    """Strip the dead epilogue-mode block from the template before markers fill."""
    src = _resolve_blocks(src, "TMA_STORE_ONLY", use_tma_store_epi)
    src = _resolve_blocks(src, "STG_ONLY", not use_tma_store_epi)
    return src


def _resolve_blocks(src: str, marker: str, keep: bool) -> str:
    """Keep or drop a template's `# @@<marker>:BEGIN/END@@` blocks."""
    pat = re.compile(
        rf"^[ \t]*# *@@{marker}:BEGIN@@[ \t]*\n(.*?)" rf"^[ \t]*# *@@{marker}:END@@[ \t]*\n",
        flags=re.MULTILINE | re.DOTALL,
    )
    return pat.sub(r"\1" if keep else "", src)


def _replace_marker_lines(src: str, replacements: dict[str, str], *, template_kind: str = "template") -> str:
    """Replace all requested injection-marker lines in one template scan."""
    found: set[str] = set()

    def repl(match: re.Match) -> str:
        marker = match.group(2)
        if marker not in replacements:
            return match.group(0)
        found.add(marker)
        replacement = replacements[marker]
        if not replacement:
            return ""
        indent = match.group(1)
        return indent + replacement.replace("\n", "\n" + indent) + "\n"

    rendered = _INJECT_MARKER_LINE.sub(repl, src)
    for marker in replacements:
        if marker not in found:
            raise RuntimeError(f"{template_kind} missing marker @@{marker}@@")
    return rendered


_MOE_SCHED_COUNTER_SLOTS = 1

_MOE_DESC_SLOT_BYTES = 128


def _is_layout_ambiguous(t: object) -> bool:
    """True iff cute's auto-deduce of ``leading_dim`` would fault on this tensor."""
    strides = getattr(t, "stride", None)
    shape = getattr(t, "shape", None)
    if strides is None or shape is None:
        return False
    try:
        strides = list(strides())
    except TypeError:
        strides = list(strides)
    sizes = list(shape)
    stride1_dims = [i for i, s in enumerate(strides) if s == 1]
    if len(stride1_dims) <= 1:
        return False

    big = [i for i in stride1_dims if sizes[i] > 1]
    return len(big) != 1


def _aux_fake_real_axes(ref: TensorRef) -> list[bool]:
    """Identify symbolic axes in an auxiliary tensor."""
    row = ref.bcast_mode in ("per_row", "per_elem") or (ref.bcast_mode == "per_k" and ref.k_axis == 1)
    col = ref.bcast_mode in ("per_col", "per_elem") or (ref.bcast_mode == "per_k" and ref.k_axis == 2)
    if len(ref.dim) == 3:
        batch_real = ref.grouped_by_moe or ref.dim[0] != 1
        return [batch_real, row, col]
    return [row, col]


def _swap_mn_view(t: object) -> object:
    """Transpose the logical M/N axes without moving the runtime storage."""
    # A contiguous DeviceView needs no strided transpose when one axis is unit,
    # as for per-K scales. Its reshape preserves the pointer and owns no memory.
    if isinstance(t, buffers.DeviceView) and len(t.shape) == 3 and 1 in t.shape[1:]:
        return t.reshape(t.shape[0], t.shape[2], t.shape[1])
    if len(t.shape) == 3:
        return t.permute(0, 2, 1)
    if len(t.shape) == 2:
        return t.permute(1, 0)
    return t


def _offset_vector(t):
    """View ``[G+1,1,1]`` offsets as a vector."""
    shape = tuple(t.shape)
    if len(shape) == 3 and shape[1:] == (1, 1):
        return t.reshape([shape[0]])
    return t


def _moe_num_groups(offsets, num_experts: int) -> int:
    """Resolve the number of grouped GEMM intervals."""
    count = int(offsets.shape[0]) - 1
    if num_experts < 1 or count < 1 or count % num_experts:
        raise ValueError("FROST MoE requires G+1 explicit first_token_offset boundaries, where G is a positive multiple of num_experts")
    return count


def _expected_output_shape(spec, chain: FusionChain, mnk, top_k_scores=None) -> tuple[int, int, int]:
    """Resolve an output shape from runtime GEMM dimensions."""
    if chain.has_moe and chain.moe.mode == "combine":
        tokens = int(top_k_scores.shape[0])
        return (1, int(mnk[0]), tokens) if isinstance(chain.moe, MoeSwapAbSpec) else (1, tokens, int(mnk[1]))
    return expected_shape(_output_rule(spec, chain), int(mnk[0]), int(mnk[1]))


def _initialize_reduction_outputs(chain: FusionChain, outputs, stream=None) -> None:
    """Seed each reduction output with its identity, before the kernel runs."""
    fills = []
    for spec, tensor in zip(chain.outputs, outputs):
        if not spec.is_reduction:
            continue
        red = chain.reductions[int(spec.source.rsplit("_", 1)[1])]
        value = REDUCTION_INIT_VALUE[red.compute_dtype][red.mode]
        shape, strides = tuple(tensor.shape), tuple(tensor.stride())
        word = buffers.init_word(red.compute_dtype, value)
        if tensor.element_size() != 4:
            raise ValueError("MoE reduction outputs require 4-byte elements")
        plan = buffers.strided_fill_plan(shape, strides)
        if plan is None:
            raise ValueError(f"MoE reduction output cannot write an element twice (shape {shape} stride {strides})")
        fills.append((tensor.data_ptr(), plan, word))

    for ptr, plan, word in fills:
        buffers.apply_fill_plan(ptr, plan, word, stream)


def _tma_alignment_reject(
    a_dtype: str,
    b_dtype: str,
    a_major: str,
    b_major: str,
    M: int,
    N: int,
    K: int,
) -> str | None:
    """Check input extents required by TMA."""
    bad: list[str] = []
    for name, dtype, major, extent, dim in (
        ("A", a_dtype, a_major, K if a_major == "k" else M, "K" if a_major == "k" else "M"),
        ("B", b_dtype, b_major, K if b_major == "k" else N, "K" if b_major == "k" else "N"),
    ):
        modulus, pack = contiguous_modulus(dtype, major == "k")
        logical = modulus * pack
        if extent % logical:
            bad.append(f"{name} ({major}-major, {dtype}) requires {dim} % {logical} == 0, " f"got {dim}={extent}")
    if bad:
        return "TMA input contiguous dimensions must be 16-byte aligned: " + "; ".join(bad)
    return None


def _partial_scale_block_reject(K: int, block_size: int) -> str | None:
    """Reject a block-scale K that ends inside a scale block."""
    if K % block_size == 0:
        return None
    return (
        f"block-scale K={K} is not a multiple of the scale block size {block_size}: the kernel sizes its F8_128x4 "
        f"scale-factor view from the {K // block_size} whole blocks, so the scales of the partial last block can be "
        f"read as zero or from another 128-row block; zero-pad K to {-(-K // block_size) * block_size}"
    )


def _alignment_reject(named_buffers) -> "str | None":
    """Every runtime buffer's alignment must be >= the kernel's compiled requirement for its role."""
    bad = []
    for role, buf, required, mode in named_buffers:
        if buf is None:
            continue
        ptr = int(buf.data_ptr())
        if mode == "ptr":
            align = _pow2_floor(ptr)
        else:
            align = tensor_alignment(tuple(buf.shape), tuple(buf.stride()), buf.element_size(), ptr=ptr)
        if align < required:
            bad.append(f"{role}: alignment {align}B < required {required}B (ptr=0x{ptr:x})")
    if bad:
        return "runtime tensor alignment is below the kernel's compiled requirement: " + "; ".join(bad)
    return None


def _sf_blob_reject(named_blobs) -> "str | None":
    """Validate scale-factor storage layout and capacity."""
    bad = []
    for role, buf, required in named_blobs:
        if buf is None:
            continue
        span = 1 + sum((int(s) - 1) * int(st) for s, st in zip(buf.shape, buf.stride()))
        if int(buf.numel()) != span:
            bad.append(f"{role} shape {tuple(buf.shape)} stride {tuple(buf.stride())} is not a dense byte run")
            continue
        have = int(buf.numel()) * buf.element_size()
        if have < required:
            bad.append(f"{role} is {have}B but the kernel reads {required}B ({required // 512} atoms of 128 rows x 4 SF-K) — was it produced by to_blocked()?")
    if bad:
        return "block-scale F8_128x4 scale factors must be a packed blob: " + "; ".join(bad)
    return None


def _grouped_row_quant_scale_blob_reject(
    chain: FusionChain,
    outputs,
    total_rows: int,
    runtime_n: int,
    num_groups: int,
) -> str | None:
    """Reject a strided view where grouped-row quant stores raw blob offsets."""
    blobs = []
    swapped = isinstance(chain.moe, MoeSwapAbSpec)
    if swapped:
        total_rows, runtime_n = runtime_n, total_rows
    for spec, buf in zip(chain.outputs, outputs):
        if not spec.is_quant_scale:
            continue
        quant_idx = int(spec.source.rsplit("_", 1)[1])
        quant = chain.quants[quant_idx]
        if not quant.grouped_by_moe or (quant.axis == 1) != swapped:
            continue
        padded_n_blocks = ((runtime_n // quant.block_size + 3) // 4) * 4
        required_rows = segmented_row_scale_capacity_rows(total_rows, num_groups)
        required = required_rows * padded_n_blocks * DTYPE_BYTES[quant.scale_dtype]
        blobs.append((f"grouped row quant scale output[{quant_idx}]", buf, required))
    return _sf_blob_reject(blobs) if blobs else None


def _moe_dense_layout_bad(t) -> bool:
    """A MoE dense output must be contiguous in one of the two inner dims."""
    return t.stride(-1) != 1 and t.stride(-2) != 1


def _moe_operand_layout_bad(chain, token, weight) -> bool:
    """Check MoE operand major-axis contiguity."""
    a_unit = -1 if chain.matmul.a_major == "k" else -2
    b_unit = -1 if chain.matmul.b_major == "k" else -2
    return token.stride(a_unit) != 1 or weight.stride(b_unit) != 1


def _declared_storage(t, memo: dict):
    """Read a bound tensor storage geometry."""
    key = id(t)
    if key not in memo:
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.graph_types import storage_geometry

        try:
            memo[key] = storage_geometry(t.get_dim(), t.get_stride(), t.get_data_type())
        except Exception:  # noqa: BLE001 -- an analyzer-synthesized ref has no dims
            memo[key] = None
    return memo[key]


def _kernel_order(buf, t, memo: "dict | None" = None):
    """View graph-declared B storage in kernel axis order."""
    declared = _declared_storage(t, {} if memo is None else memo)
    if declared is not None and declared[0] and (tuple(buf.shape), tuple(buf.stride())) == declared:
        return buf.permute(0, 2, 1)
    return buf


def _check_moe_combine_n(compiled, n: int) -> None:
    """Enforce the compiled N divisibility contract before output memset."""
    chain = compiled.chain
    if chain.moe.mode != "combine":
        return
    multiple = compiled.vec_bytes_epi // DTYPE_BYTES[chain.output_dtype]
    if isinstance(chain.moe, MoeSwapAbSpec):
        multiple = min(multiple, chain.moe.offset_multiple)
    if n % multiple:
        raise ValueError(f"MoE COMBINE runtime kernel N must be divisible by {multiple} for the compiled epilogue; got N={n}")


def _check_moe_operand_shapes(chain, a_bufs, b_bufs) -> None:
    """Validate grouped operand shapes and logical K extents."""
    swapped = isinstance(chain.moe, MoeSwapAbSpec)
    tokens, weights = (b_bufs, a_bufs) if swapped else (a_bufs, b_bufs)
    token_dtype = chain.matmul.b_dtype if swapped else chain.matmul.a_dtype
    weight_dtype = chain.matmul.a_dtype if swapped else chain.matmul.b_dtype
    for role, operands in (("token", tokens), ("weight", weights)):
        for index, value in enumerate(operands):
            if len(value.shape) != 3:
                raise ValueError(f"MoE {role} operand[{index}] must be rank-3; got {tuple(value.shape)}")
    rows = tokens[0].shape[1]
    k = tokens[0].shape[2] * (2 if token_dtype == "fp4_e2m1" else 1)
    experts, n = weights[0].shape[:2]
    for index, value in enumerate(tokens):
        if tuple(value.shape[:2]) != (1, rows):
            raise ValueError("MoE token operands must share shape [1,S,K]")
        actual = value.shape[2] * (2 if token_dtype == "fp4_e2m1" else 1)
        if actual != k:
            raise ValueError(f"MoE token operand[{index}] must have logical K={k}; got {actual}")
    for index, value in enumerate(weights):
        if tuple(value.shape[:2]) != (experts, n):
            raise ValueError("MoE weight operands must share shape [E,N,K] in kernel order")
        actual = value.shape[2] * (2 if weight_dtype == "fp4_e2m1" else 1)
        if actual != k:
            raise ValueError(f"MoE weight operand[{index}] must have logical K={k}; got {actual}")


def _resolve_moe_variant_pack(compiled, variant_pack: dict):
    """Resolve MoE buffers and runtime dimensions."""
    b = compiled.binding
    if b is None:
        raise NotImplementedError("variant-pack call is not yet wired up for this graph type")
    resolved = compiled._compiler.resolve_variant_pack(variant_pack, b)

    memo = getattr(compiled, "_declared_storage_memo", None)
    if memo is None:
        memo = {}
        try:
            compiled._declared_storage_memo = memo
        except AttributeError:
            # Keep the memo local for slot-only launchers.
            pass

    def pull(t, role):
        if t is None or id(t) not in resolved:
            raise KeyError(f"variant pack is missing a buffer for {role}")
        return resolved[id(t)]

    top_k_scores = None
    if compiled.chain.moe.mode == "combine":
        scores = pull(b.top_k_scores, "top_k_scores")
        shape = tuple(scores.shape)
        top_k = compiled.chain.moe.top_k
        if len(shape) != 3 or shape[0] != 1 or shape[1] < 1 or shape[2] != top_k:
            raise ValueError("MoE COMBINE top_k_scores must have shape [1, tokens, top_k]")
        if buffers.dtype_name(scores) != "float32" or tuple(scores.stride())[1:] != (top_k, 1) or scores.data_ptr() % 4:
            raise ValueError("MoE COMBINE top_k_scores must be aligned compact FP32")
        top_k_scores = scores.reshape([shape[1], shape[2]])
        for spec, tensor in zip(compiled.chain.outputs, b.outputs):
            output = pull(tensor, "output")
            if buffers.dtype_name(output) != {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[spec.dtype]:
                raise ValueError(f"MoE COMBINE output requires {spec.dtype}")
            if buffers.strided_fill_plan(tuple(output.shape), tuple(output.stride())) is None:
                raise ValueError("MoE COMBINE output must have non-overlapping positive strides")

    def routing_indices(tokens):
        mode = compiled.chain.moe.mode
        if mode == "none":
            return None, None
        indices = []
        for tensor, role in [(b.token_index, "token_index")] + ([(b.token_ks, "token_ks")] if mode in ("scatter", "combine") else []):
            buf = pull(tensor, role)
            shape = tuple(buf.shape)
            if len(shape) != 3 or shape[0] != 1 or shape[2] != 1 or shape[1] < 1:
                raise ValueError(f"MoE {mode.upper()} {role} must have shape [1, routed_rows, 1]")
            if buffers.dtype_name(buf) != "int32" or buf.stride(1) != 1 or buffers.data_ptr(buf) % 4:
                raise ValueError(f"MoE {mode.upper()} {role} must be aligned INT32 with contiguous routed rows")
            if mode in ("scatter", "combine") and (shape[1] != tokens[0].shape[1] or (mode == "scatter" and shape[1] % compiled.chain.moe.top_k)):
                raise ValueError(f"MoE {mode.upper()} {role} must match token rows" + (" divisible by top_k" if mode == "scatter" else ""))
            indices.append(buf.reshape([shape[1]]))
        if any(len(t.shape) != 3 or t.shape[0] != 1 or tuple(t.shape) != tuple(tokens[0].shape) for t in tokens):
            raise ValueError(f"MoE {mode.upper()} token operands must share shape [1, rows, K]")
        return indices[0], indices[1] if mode in ("scatter", "combine") else None

    if isinstance(compiled.chain.moe, MoeSwapAbSpec):
        a_bufs = [_kernel_order(pull(t, "weight"), t, memo) for t in b.a_operands]
        b_bufs = [pull(t, "token") for t in b.b_operands]
        out_bufs = [
            (
                pull(t, "output")
                if spec.is_quant_scale and compiled.chain.quants[int(spec.source.rsplit("_", 1)[1])].scale_reorder == "F8_128x4"
                else _swap_mn_view(pull(t, "output"))
            )
            for spec, t in zip(compiled.chain.outputs, b.outputs)
        ]
        aux_bufs = [_swap_mn_view(pull(t, "aux")) for t in b.aux]
        fto = pull(b.first_token_offset, "first_token_offset")
        sfa = [_kernel_order(pull(t, "SFA"), t, memo) for t in b.sfa_operands]
        sfb = [pull(t, "SFB") for t in b.sfb_operands]
        mm = compiled.chain.matmul
        _check_moe_operand_shapes(compiled.chain, a_bufs, b_bufs)
        token_index, token_ks = routing_indices(b_bufs)
        m = a_bufs[0].shape[1]
        n = token_index.shape[0] if token_index is not None else b_bufs[0].shape[1]
        k = a_bufs[0].shape[2] * (2 if mm.a_dtype == "fp4_e2m1" else 1)
        reason = _tma_alignment_reject(mm.a_dtype, mm.b_dtype, mm.a_major, mm.b_major, m, n, k)
        if reason is not None:
            raise ValueError(reason)
        _check_moe_combine_n(compiled, n)
        return a_bufs, b_bufs, out_bufs, aux_bufs, fto, sfa, sfb, token_index, token_ks, top_k_scores, (m, n, k)

    a_bufs = [pull(t, "token") for t in b.a_operands]
    b_bufs = [_kernel_order(pull(t, "weight"), t, memo) for t in b.b_operands]
    out_bufs = [pull(t, "output") for t in b.outputs]
    aux_bufs = [pull(t, "aux") for t in b.aux]
    fto = pull(b.first_token_offset, "first_token_offset")
    sfa = [pull(t, "SFA") for t in b.sfa_operands]
    sfb = [_kernel_order(pull(t, "SFB"), t, memo) for t in b.sfb_operands]
    _check_moe_operand_shapes(compiled.chain, a_bufs, b_bufs)
    k_factor = 2 if compiled.chain.matmul.a_dtype == "fp4_e2m1" else 1
    token_index, token_ks = routing_indices(a_bufs)
    S = token_index.shape[0] if token_index is not None else a_bufs[0].shape[1]
    K = a_bufs[0].shape[2] * k_factor

    N = b_bufs[0].shape[1]
    mm = compiled.chain.matmul
    _align_reason = _tma_alignment_reject(mm.a_dtype, mm.b_dtype, mm.a_major, mm.b_major, S, N, K)
    if _align_reason is not None:
        raise ValueError(_align_reason)
    _check_moe_combine_n(compiled, N)
    return a_bufs, b_bufs, out_bufs, aux_bufs, fto, sfa, sfb, token_index, token_ks, top_k_scores, (S, N, K)


def _register_legacy_device_view_adapter() -> None:
    """Teach the LEGACY (pre-tvm-ffi) DSL executor to accept a raw ``DeviceView`` launch arg."""
    try:
        from cutlass.base_dsl.runtime.jit_arg_adapters import JitArgAdapterRegistry
    except ImportError:
        return
    if buffers.DeviceView in JitArgAdapterRegistry.jit_arg_adapter_registry:
        return
    from cutlass.cute.runtime import from_dlpack

    @JitArgAdapterRegistry.register_jit_arg_adapter(buffers.DeviceView)
    def _adapt_device_view(view):
        ptr = view.data_ptr()
        return from_dlpack(view, assumed_align=min(ptr & -ptr, _MOE_DESC_SLOT_BYTES))


def _moe_carve_workspace(caller, n_slots: int, plan: str):
    """Carve MoE descriptor scratch from workspace."""
    _register_legacy_device_view_adapter()
    need = n_slots * _MOE_DESC_SLOT_BYTES
    ws = caller if isinstance(caller, Workspace) else Workspace(caller, need, plan, align=_MOE_DESC_SLOT_BYTES)
    return ws.view(0, "int64", (need // 8,))


def _moe_launch_tail(cs, aux=(), *, tma_slots: "frozenset[int]" = frozenset()) -> tuple:
    """Order output and auxiliary launch arguments."""
    cs = list(cs)
    taps = [c for i, c in enumerate(cs) if i not in tma_slots]
    tmas = [cs[i] for i in sorted(tma_slots) if i < len(cs)]
    return (*taps, *aux, *tmas)


_LOG = logging.getLogger(__name__)


def _fallback_cache_dir() -> Path:
    """A writable per-user cache under the system temp dir."""
    suffix = f"-{os.getuid()}" if hasattr(os, "getuid") else ""
    return Path(tempfile.gettempdir()) / f"cudnn_gemm_kernel_cache{suffix}"


@functools.lru_cache(maxsize=None)
def _usable_cache_dir(base: str) -> Path:
    """Resolve a writable kernel cache directory."""
    try:
        p = Path(base)
        p.mkdir(parents=True, exist_ok=True)
        if not os.access(p, os.W_OK):
            raise PermissionError(f"{p} is not writable")
        return p
    except OSError as exc:
        fallback = _fallback_cache_dir()
        fallback.mkdir(parents=True, exist_ok=True)
        _LOG.warning(
            "cudnn.gemm.frost: kernel cache %s is unusable (%s); falling back to %s. Set CUDNN_FRONTEND_GEMM_KERNEL_CACHE to choose another location.",
            base,
            exc,
            fallback,
        )
        return fallback


def _cache_dir() -> Path:
    """Select the generated-kernel source cache directory."""
    base = os.environ.get("CUDNN_FRONTEND_GEMM_KERNEL_CACHE")
    if not base:
        xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        base = os.path.join(xdg, "cudnn_gemm", "kernel_cache")
    return _usable_cache_dir(base)


def _write_atomic(path: Path, src: str) -> None:
    """Publish ``src`` at ``path`` in one rename."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(src)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            # Preserve the original write failure.
            pass
        raise


def _import_kernel(src: str) -> object:
    """Import a generated kernel from the source cache."""
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    out_dir = _cache_dir() / f"gen_{digest}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "generated_kernel.py"
    _write_atomic(out_file, src)

    mod_name = f"_cudnn_gemm_generated_{digest}"
    spec = importlib.util.spec_from_file_location(mod_name, out_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    mod.FROST_SOURCE_DIGEST = digest
    return mod


class _CompiledRuntime:
    _compiler: ClassVar[Any]
    _validate_runtime_quant: ClassVar[bool] = False
    _validate_combine: ClassVar[bool] = False
    # Per call: refuse a block-scale K that ends inside a scale block (_partial_scale_block_reject).
    _validate_whole_scale_blocks: ClassVar[bool] = False
    accepts_stream: ClassVar[bool] = True

    @property
    def tma_slots(self) -> "frozenset[int]":
        """Which dense output slots ride the TMA-C surface."""
        return frozenset((i for i, m in enumerate(self.store_modes) if m == "tma"))


@dataclass
class _CompiledMoeBase(_CompiledRuntime, Generic[_ConfigT, _BindingT]):
    chain: FusionChain
    config: _ConfigT
    generated_path: Path
    _launchable: Callable
    _grid_ctas: int = 0
    device: int = 0
    aux_names: list = field(default_factory=list)
    binding: _BindingT | None = None
    vec_bytes_epi: "int | None" = None
    _desc_slots_per_cta: int = 0
    store_modes: tuple = ()
    use_tma_store: bool = False

    @property
    def workspace_bytes(self) -> int:
        """Per-CTA tensormap scratch: one 128-byte slot per CTA per patched descriptor."""
        backend = self._compiler
        return (self._grid_ctas * self._desc_slots_per_cta + backend._MOE_SCHED_COUNTER_SLOTS) * backend._MOE_DESC_SLOT_BYTES

    def _make_workspace(self, n_slots, caller=None):
        """The per-CTA tensormap GMEM workspace (16 int64/slot, 128-byte aligned)."""
        backend = self._compiler
        if caller is None:
            raise ValueError(
                f"{type(self).__name__} requires a {n_slots * backend._MOE_DESC_SLOT_BYTES}-byte workspace but execute() received none; size it with workspace_bytes and pass workspace= (a plan owns no device memory, Rule 8)"
            )
        return backend._moe_carve_workspace(caller, n_slots, type(self).__name__)

    def __call__(self, variant_pack, workspace=None, stream=None):
        """Validate the target device and dispatch a grouped GEMM call."""
        backend = self._compiler
        if not isinstance(variant_pack, dict):
            raise TypeError(f"compiled kernels are called with a variant-pack dict {{cuDNN tensor | uid | name: buffer}}; got {type(variant_pack).__name__}")
        backend._check_plan_device(self.device)
        return self._call_variant_pack(variant_pack, workspace, stream)

    def _launch_moe(self, problem, offsets, workspace, operands, outputs, aux, stream, token_index, token_ks, top_k_scores):
        """Assemble and launch the grouped GEMM arguments."""
        return self._launchable(
            problem,
            offsets,
            workspace,
            *((top_k_scores,) if top_k_scores is not None else ()),
            *((token_ks,) if token_ks is not None else ()),
            *((token_index,) if token_index is not None else ()),
            *operands,
            *_moe_launch_tail(outputs, aux, tma_slots=self.tma_slots),
            stream=self._compiler._as_custream(stream),
        )


@dataclass
class _CompiledFusedGemm(_CompiledRuntime, Generic[_ConfigT, _BindingT]):
    """Execute a compiled fused GEMM."""

    chain: FusionChain
    config: _ConfigT
    aux_names: list[str]
    generated_path: Path
    _launchable: Callable
    block_scale: bool = False
    device: int = 0
    binding: _BindingT | None = None
    store_modes: tuple = ()
    use_tma_store: bool = False
    vec_bytes_epi: "int | None" = None
    workspace_bytes: int = 0

    def workspace_bytes_for(self, batch: int, m: int, n: int) -> int:
        """workspace for a given shape."""
        backend = self._compiler
        if not self.config.workspace_split_k:
            return 0
        return backend.align_up(self.config.split_k_slices * batch * m * n * 4)

    # The partials span the call's M and N, which the launch path reads off A and B.
    # Output 0 is no stand-in: a graph whose only outputs are reductions has a reduction there.
    def workspace_bytes_for_operands(self, operands, graph_order=None) -> int:
        """workspace for one call's A and B buffers, in the axis order the launch path reads."""
        r = self.recipe
        if r is None:
            return self.workspace_bytes
        mn = []
        for op in (r.a, r.b):
            v = operands[op.index]
            sh = v.shape
            axes = op.axes(sh, v.stride(), bool(graph_order) and graph_order[op.index]) if len(sh) == 3 else None
            if axes is None:
                return 0  # the launch path refuses this layout before it reads the workspace
            mn.append(int(sh[axes[AX_MN]]))
        return self.workspace_bytes_for(r.batch, *mn)

    def workspace_bytes_for_overrides(self, override_uids, override_shapes) -> int:
        """workspace for an override-shape call: A and B dims it names (graph order), else an output's, else the plan's."""
        r = self.recipe
        if r is None:
            return self.workspace_bytes
        dims = {uid: shape for uid, shape in zip(override_uids or (), override_shapes or ()) if len(shape) == 3}
        mn = [None, None]
        for i, op in enumerate((r.a, r.b)):
            shape = dims.get(self.bound[op.index].get_uid())
            if shape is not None:
                mn[i] = int(shape[op.declared[AX_MN]])
        # A list may leave A or B out while a strided view of it (which keeps its own description) runs at
        # new extents. The launch path holds every output to A's M and B's N, so a named output carries them.
        for out in r.outputs:
            shape = dims.get(self.bound[out.index].get_uid())
            if shape is None:
                continue
            for axis, (source, _) in enumerate(out.rule):
                at = {FROM_M: 0, FROM_N: 1}.get(source)
                if at is not None and mn[at] is None:
                    mn[at] = int(shape[axis])  # override dims count elements: no fp4 n // 2 to undo
        planned = (self.chain.matmul.M, self.chain.matmul.N)
        return self.workspace_bytes_for(r.batch, *(p if v is None else v for v, p in zip(mn, planned)))

    recipe: Any = field(default=None, init=False, repr=False, compare=False)
    lowered: Any = field(default=None, init=False, repr=False, compare=False)
    bound: Any = field(default=(), init=False, repr=False, compare=False)
    declined: "str | None" = field(default=None, init=False, repr=False, compare=False)
    deferrals: Any = field(default_factory=Counter, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Build the immutable launch recipe and lowering closure."""
        backend = self._compiler
        if self.binding is not None:
            self.bound = tuple(self.binding.bound_tensors())
            self.recipe = backend.build_recipe(self)
            self.lowered = self._lower()

    def __call__(self, variant_pack, stream=None, workspace=None):
        """Resolve the variant pack and execute the compiled GEMM."""
        backend = self._compiler
        if not isinstance(variant_pack, dict):
            raise TypeError(f"compiled kernels are called with a variant-pack dict {{cuDNN tensor | uid | name: buffer}}; got {type(variant_pack).__name__}")
        if self.binding is None:
            raise NotImplementedError("variant-pack call is not wired up for this graph type")
        return self.run_resolved(backend.resolve_variant_pack(variant_pack, self.binding), stream=stream, workspace=workspace)

    def run_resolved(self, resolved, stream=None, workspace=None):
        """Execute resolved buffers."""
        operands = []
        for i, t in enumerate(self.bound):
            buf = resolved.get(id(t))
            if buf is None:
                raise KeyError(f"variant pack is missing a buffer for {self.recipe.roles[i]}")
            operands.append(buf)
        if self.lowered is None:
            raise NotImplementedError(f"cudnn.frost gemm: this kernel has no launch path -- {self.declined}")
        return self.lowered(operands, stream=stream, workspace=workspace)

    def _lower(self):
        """Build the validated launch closure."""
        backend = self._compiler

        def decline(reason: str) -> None:
            self.declined = reason
            return None

        r = self.recipe
        if r is None:
            return decline("no binding")
        try:
            backend._check_executable(self.chain)
        except NotImplementedError as exc:
            return decline(str(exc))
        if bool(r.sf) != bool(r.block_size):
            return decline("scale factors without a block size")
        renorm = tuple(
            (
                (j, tuple(op.declared), op.declared_layout[0] if len(op.declared_layout[0]) == 3 else None, op.declared_layout[1])
                for j, op in enumerate(r.inputs)
                if op.declared != backend.KERNEL_AXES
            )
        )
        stride_ins = r.stride_ins
        in_slots = tuple((op.index for op in r.inputs))
        ins = tuple(((op.kc, op.modulus, op.batch, op.kpack, op.is_b, i in stride_ins) for i, op in enumerate(r.inputs)))
        outs = tuple(((o.index, *(x for axis in o.rule for x in axis), o.align) for o in r.outputs))
        auxs = tuple((x.index, x.align, x.mainloop_scale) for x in r.aux)
        sfs = tuple(((s.index, s.is_a, r.inputs[s.operand_at].batch) for s in r.sf))
        tail = r.arg_plan[len(r.inputs) :]
        shared, seeds = (r.shared_layout, r.seeds)
        a_pos, b_pos, a_kpack = (r.a_at, r.b_at, r.a.kpack)
        block_size, batch = (r.block_size, r.batch)
        whole_scale_blocks = self._validate_whole_scale_blocks
        device, launchable, refuse = (r.device, self._launchable, self.explain)
        fill_word, fill_plan, apply_fill = (backend.buffers.fill_word_async, backend.buffers.strided_fill_plan, backend.buffers.apply_fill_plan)
        is_contiguous = backend.buffers.is_contiguous
        needs_workspace = bool(r.workspace_bytes)
        split_k_slices = self.config.split_k_slices
        swap_ab = self.config.swap_ab
        packed_scale_slots = frozenset(
            (o.index for o in r.outputs if o.role.startswith("quant_scale_") and self.chain.quants[int(o.role.rsplit("_", 1)[1])].scale_reorder == "F8_128x4")
        )
        mn_view_slots = (frozenset((o.index for o in r.outputs)) | frozenset((x.index for x in r.aux))) - packed_scale_slots
        if split_k_slices > 1:
            cta_k_elems = backend._cta_k_elems(self.chain, self.config)
        if needs_workspace:
            reduce_elems = backend._splitk_reduce_elems(self.chain)
        quant_mn = (self.chain.matmul.M, self.chain.matmul.N) if self._validate_runtime_quant and self.chain.quants else None
        quant_reason = backend._runtime_quant_reason if quant_mn is not None else None
        quant_chain, quant_cfg = (self.chain, self.config)
        gave_up = self.deferrals

        def lowered(operands, graph_order=None, stream=None, workspace=None):
            """Validate resolved buffers and launch the already compiled GEMM plan."""
            backend._check_plan_device(device)
            vs = [operands[i] for i in in_slots]
            for j, perm, dsh, dst in renorm:
                v = vs[j]
                sh = v.shape
                if len(sh) != 3:
                    continue
                declared = bool(graph_order) and graph_order[in_slots[j]]
                if not declared and dsh is not None:
                    declared = sh[0] == dsh[0] and sh[1] == dsh[1] and (sh[2] == dsh[2])
                    if declared:
                        st = v.stride()
                        declared = st[0] == dst[0] and st[1] == dst[1] and (st[2] == dst[2])
                if declared:
                    vs[j] = v.permute(*perm)
            a_sh, b_sh = (vs[a_pos].shape, vs[b_pos].shape)
            if len(a_sh) != 3 or len(b_sh) != 3:
                gave_up["A or B is not rank 3"] += 1
                return refuse(operands, graph_order)
            m, n, k = (a_sh[1], b_sh[1], a_sh[2] * a_kpack)
            problem = [m, n, k, batch]
            for (kc, mod, ebatch, kpack, is_b, takes_stride), v in zip(ins, vs):
                sh, st = (v.shape, v.stride())
                if (
                    len(sh) != 3
                    or st[kc] != 1
                    or sh[kc] % mod
                    or (sh[0] != ebatch)
                    or (sh[1] != (n if is_b else m))
                    or (sh[2] * kpack != k)
                    or (backend._pow2_floor(v.data_ptr()) < 16)
                ):
                    gave_up["input layout"] += 1
                    return refuse(operands, graph_order)
                if takes_stride:
                    problem += (st[1], st[2], st[0])
            if quant_mn is not None and (m, n) != quant_mn and (quant_reason(quant_chain, quant_cfg, m, n) is not None):
                gave_up["quant extents"] += 1
                return refuse(operands, graph_order)
            for idx, k0, v0, k1, v1, k2, v2, align in outs:
                v = operands[idx]
                sh, st = (v.shape, v.stride())
                if (
                    len(sh) != 3
                    or sh[0] != (v0 if k0 == backend.CONST else m if k0 == backend.FROM_M else n // v0)
                    or sh[1] != (v1 if k1 == backend.CONST else m if k1 == backend.FROM_M else n // v1)
                    or (sh[2] != (v2 if k2 == backend.CONST else m if k2 == backend.FROM_M else n // v2))
                    or (backend.tensor_alignment(tuple(sh), tuple(st), v.element_size(), ptr=v.data_ptr()) < align)
                ):
                    gave_up["output layout"] += 1
                    return refuse(operands, graph_order)
                problem += (st[2], st[1], st[0]) if swap_ab and idx in mn_view_slots else (st[1], st[2], st[0])
            for idx, align, scale_rule in auxs:
                v = operands[idx]
                if scale_rule is not None:
                    scale_rule.check(v, k)
                if backend.tensor_alignment(tuple(v.shape), tuple(v.stride()), v.element_size(), ptr=v.data_ptr()) < align:
                    gave_up["aux alignment"] += 1
                    return refuse(operands, graph_order)
            if sfs:
                if whole_scale_blocks and k % block_size:
                    gave_up["partial scale block"] += 1
                    return refuse(operands, graph_order)
                k4 = (k // block_size + 3) // 4
                for idx, is_a, sf_batch in sfs:
                    v = operands[idx]
                    count = int(v.numel())
                    if (
                        len(v.shape) != 3
                        or backend._pow2_floor(v.data_ptr()) < 16
                        or count != 1 + sum(((int(s) - 1) * int(st) for s, st in zip(v.shape, v.stride())))
                        or (count * v.element_size() < 512 * k4 * (((m if is_a else n) + 127) // 128) * sf_batch)
                    ):
                        gave_up["scale-factor blob"] += 1
                        return refuse(operands, graph_order)
            for lead, followers in shared:
                st = tuple(vs[lead].stride())
                for j in followers:
                    if tuple(vs[j].stride()) != st:
                        gave_up["operands do not share a layout"] += 1
                        return refuse(operands, graph_order)
            if split_k_slices > 1:
                backend.check_split_k_shape(k, n, batch, split_k_slices, cta_k_elems, reduce_elems if needs_workspace else 1)
            extra = ()
            if needs_workspace:
                if workspace is None:
                    raise ValueError(
                        "cudnn.frost gemm: this plan needs a workspace; allocate graph.get_workspace_size() bytes and pass the buffer to execute()"
                    )
                extra = (workspace.view(0, "float32", (split_k_slices * batch * m * n,)),)
            if seeds:
                fills = []
                for idx, word, elem_bytes in seeds:
                    v = operands[idx]
                    if v.element_size() != elem_bytes:
                        gave_up["reduction seed dtype"] += 1
                        return refuse(operands, graph_order)
                    sh, st = (v.shape, v.stride())
                    if is_contiguous(sh, st):
                        fills.append((v.data_ptr(), None, int(v.numel()), word))
                        continue
                    plan = fill_plan(sh, st)
                    if plan is None:
                        gave_up["reduction output writes an element twice"] += 1
                        return refuse(operands, graph_order)
                    fills.append((v.data_ptr(), plan, 0, word))
                for ptr, plan, count, word in fills:
                    if plan is None:
                        fill_word(ptr, count, word, stream)
                    else:
                        apply_fill(ptr, plan, word, stream)
            return launchable(
                tuple(problem),
                *(v.permute(1, 2, 0) for v in vs),
                *(
                    (
                        (backend._swap_mn_view(operands[i]) if swap_ab and i in mn_view_slots else operands[i]).permute(1, 2, 0)
                        if ref is None
                        else backend._reshape_aux_to_fake(backend._swap_mn_view(operands[i]) if swap_ab else operands[i], ref)
                    )
                    for i, ref in tail
                ),
                *extra,
                stream=backend._as_custream(stream),
            )

        return lowered

    def explain(self, operands, graph_order=None):
        """Say what is wrong with a call the launch path refused, and raise."""
        backend = self._compiler
        recipe = self.recipe
        backend._check_plan_device(recipe.device)
        mnk, axes = recipe.problem(operands, graph_order)
        backend.check_shapes(recipe, operands, mnk, axes)
        backend.check_alignment(recipe, operands, axes)
        if self._validate_whole_scale_blocks and self.chain.block_scale is not None:
            reason = _partial_scale_block_reject(int(mnk[2]), self.chain.block_scale.block_size)
            if reason is not None:
                raise ValueError(reason)
        if self.config.split_k_slices > 1:
            backend.check_split_k_shape(
                int(mnk[2]),
                int(mnk[1]),
                recipe.batch,
                self.config.split_k_slices,
                backend._cta_k_elems(self.chain, self.config),
                backend._splitk_reduce_elems(self.chain) if self.config.workspace_split_k else 1,
            )
        if self._validate_runtime_quant:
            m, n = (int(mnk[0]), int(mnk[1]))
            if self.config.swap_ab and self.chain.quants:
                from .fusion_ir import swap_ab

                try:
                    graph_chain = swap_ab(self.chain)
                except (NotImplementedError, ValueError, AssertionError):
                    graph_chain = None
                if graph_chain is not None:
                    try:
                        backend._check_block_quant_semantics(backend.replace(graph_chain, matmul=backend.replace(graph_chain.matmul, M=n, N=m)))
                    except NotImplementedError as exc:
                        raise ValueError(str(exc)) from exc
            try:
                backend._check_runtime_quant(self.chain, self.config, m, n)
            except ValueError as exc:
                if not self.config.swap_ab:
                    raise
                raise ValueError(
                    f"{exc} [in the kernel frame of this swap-AB plan, which runs the graph transposed: its M = {m} and N = {n} are the graph's N and M, and its row / col quant axes the graph's col / row]"
                ) from exc
        raise RuntimeError(
            f"cudnn.frost gemm: the launch path refused this call and no rule explains why -- its per-call guards and the diagnostics here have drifted apart. Guards that fired: {dict(self.deferrals)}."
        )


@dataclass
class _CompiledMoeGemm(_CompiledMoeBase[_ConfigT, _BindingT]):
    """Execute a compiled grouped GEMM."""

    def _launch_single(self, token, weight, first_token_offset, output, snke, workspace=None, stream=None, token_index=None, token_ks=None, top_k_scores=None):
        """Validate and launch one grouped GEMM."""
        backend = self._compiler
        first_token_offset = backend._offset_vector(first_token_offset)
        if len(snke) < 3:
            raise ValueError(f"MoE call needs problem_size (S, N, K[, ...]); got {snke!r}")
        S, N, K = (int(snke[0]), int(snke[1]), int(snke[2]))
        if self._validate_runtime_quant:
            backend._check_runtime_quant(self.chain, self.config, S, N)
        outputs_spec = self.chain.outputs
        outputs = list(output) if isinstance(output, (list, tuple)) else [output]
        if len(outputs) != len(outputs_spec):
            raise ValueError(
                f"this MoE graph has {len(outputs_spec)} output(s) ({[o.source for o in outputs_spec]}); got {len(outputs)}. Pass outputs in slot order."
            )
        for name, t, rank in (("token", token, 3), ("weight", weight, 3), ("first_token_offset", first_token_offset, 1)):
            if len(t.shape) != rank:
                raise ValueError(f"MoE {name} must be rank-{rank}; got shape {tuple(t.shape)}")
        for spec, t in zip(outputs_spec, outputs):
            if len(t.shape) != 3 or tuple(t.shape) != backend._expected_output_shape(spec, self.chain, (S, N, K), top_k_scores):
                raise ValueError(
                    f"MoE output {spec.source!r} must have shape {backend._expected_output_shape(spec, self.chain, (S, N, K), top_k_scores)}; got {tuple(t.shape)}"
                )
        num_experts = int(weight.shape[0])
        num_groups = backend._moe_num_groups(first_token_offset, num_experts)
        a_perm = token.permute(1, 2, 0)
        b_perm = weight.permute(1, 2, 0)
        c_perms = [t.permute(1, 2, 0) for t in outputs]
        dense_bad = self.chain.output_specs and backend._moe_dense_layout_bad(outputs[0])
        if backend._moe_operand_layout_bad(self.chain, token, weight) or dense_bad:
            raise ValueError(
                f"MoE non-packed tensors require contiguous innermost dimensions: got token stride {tuple(token.stride())}, weight stride {tuple(weight.stride())}, output stride {tuple(outputs[0].stride())}"
            )
        output_strides = tuple((stride for _spec, ci in zip(outputs_spec, c_perms) for stride in ci.stride()))
        problem_size = (S, N, K, num_experts, num_groups, *tuple(a_perm.stride()), *tuple(b_perm.stride()), *output_strides)
        a = backend._maybe_wrap_layout(a_perm, backend._LEADING_DIM_A)
        b = backend._maybe_wrap_layout(b_perm, backend._LEADING_DIM_B)
        cs = [
            backend._wrap_raw_tensor(ci) if spec.is_reduction or spec.is_quant_scale else backend._maybe_wrap_layout(ci, backend._LEADING_DIM_C)
            for spec, ci in zip(outputs_spec, c_perms)
        ]
        workspace = self._make_workspace(self._grid_ctas * self._desc_slots_per_cta + backend._MOE_SCHED_COUNTER_SLOTS, workspace)
        if self._validate_combine:
            backend._check_moe_combine_operands(
                self.chain, [token], [weight], first_token_offset, [], [], (token_index, token_ks, top_k_scores), outputs, K, workspace
            )
        backend._initialize_reduction_outputs(self.chain, outputs, stream)
        backend._initialize_combine_output(self.chain, outputs, stream)
        return self._launch_moe(problem_size, first_token_offset, workspace, (a, b), cs, (), stream, token_index, token_ks, top_k_scores)

    def _call_variant_pack(self, variant_pack: dict, workspace=None, stream=None):
        """Bind grouped GEMM inputs and select the launch path."""
        backend = self._compiler
        a_bufs, b_bufs, out_bufs, aux_bufs, fto, _sfa, _sfb, token_index, token_ks, top_k_scores, snk = backend._resolve_moe_variant_pack(self, variant_pack)
        _out_reqs = backend._output_align_reqs(self.chain, self.tma_slots, vec_bytes=self.vec_bytes_epi)
        _aux_reqs = backend._aux_align_reqs(self.chain, vec_bytes=self.vec_bytes_epi)
        _named = (
            [("token", x, 16, "ptr") for x in a_bufs]
            + [("weight", x, 16, "ptr") for x in b_bufs]
            + [(f"output[{i}]", x, _out_reqs[i], "full") for i, x in enumerate(out_bufs)]
            + [(f"aux {self.chain.aux_tensors[i].name!r}", x, _aux_reqs[self.chain.aux_tensors[i].name], "full") for i, x in enumerate(aux_bufs)]
        )
        _r = backend._alignment_reject(_named)
        if _r is not None:
            raise ValueError(_r)
        if isinstance(self.chain.moe, backend.MoeSwapAbSpec):
            return backend._launch_moe_swap_ab(
                self,
                a_bufs,
                b_bufs,
                out_bufs,
                aux_bufs,
                fto,
                [],
                [],
                snk,
                workspace,
                stream,
                token_index=token_index,
                token_ks=token_ks,
                top_k_scores=top_k_scores,
            )
        out = out_bufs if len(out_bufs) > 1 else out_bufs[0]
        if self.chain.is_multi_gemm or self.chain.ops:
            pairs = [(a_bufs[ai], b_bufs[bi]) for ai, bi in self.chain.gemm_operands]
            return self._call_multi_gemm(
                pairs, fto, out, snk, *aux_bufs, workspace=workspace, stream=stream, token_index=token_index, token_ks=token_ks, top_k_scores=top_k_scores
            )
        return self._launch_single(
            a_bufs[0], b_bufs[0], fto, out, snk, workspace=workspace, stream=stream, token_index=token_index, token_ks=token_ks, top_k_scores=top_k_scores
        )

    def _call_multi_gemm(
        self, gemm_pairs, first_token_offset, output, snke, *aux, workspace=None, stream=None, token_index=None, token_ks=None, top_k_scores=None
    ):
        """Launch parallel grouped GEMMs."""
        backend = self._compiler
        first_token_offset = backend._offset_vector(first_token_offset)
        chain = self.chain
        if not isinstance(gemm_pairs, (list, tuple)) or not all((isinstance(p, (list, tuple)) and len(p) == 2 for p in gemm_pairs)):
            raise ValueError(f"multi-GEMM MoE call expects a list of (token, weight) pairs as the first argument; got {type(gemm_pairs).__name__}")
        if len(gemm_pairs) != chain.num_gemms:
            raise ValueError(f"this graph has {chain.num_gemms} grouped matmul(s); got {len(gemm_pairs)} (token, weight) pair(s)")
        if len(snke) < 3:
            raise ValueError(f"MoE call needs problem_size (S, N, K[, ...]); got {snke!r}")
        S, N, K = (int(snke[0]), int(snke[1]), int(snke[2]))
        if self._validate_runtime_quant:
            backend._check_runtime_quant(chain, self.config, S, N)
        na, nb = (chain.num_a_operands, chain.num_b_operands)
        a_slots: list = [None] * na
        b_slots: list = [None] * nb
        for (tok, w), (ai, bi) in zip(gemm_pairs, chain.gemm_operands):
            for slots, idx, t, role in ((a_slots, ai, tok, "token"), (b_slots, bi, w, "weight")):
                if slots[idx] is None:
                    slots[idx] = t
                elif slots[idx].data_ptr() != t.data_ptr():
                    raise ValueError(
                        f"multi-GEMM MoE operand sharing mismatch: distinct {role} slot {idx} was given two different tensors. The runtime sharing pattern must match the compiled graph."
                    )
        if any((s is None for s in a_slots)) or any((s is None for s in b_slots)):
            raise ValueError("multi-GEMM MoE: not every distinct token/weight slot was filled")
        outputs_spec = chain.outputs
        outs = list(output) if isinstance(output, (list, tuple)) else [output]
        if len(outs) != len(outputs_spec):
            raise ValueError(f"fused multi-GEMM MoE has {len(outputs_spec)} output(s); got {len(outs)}")
        out = outs[0]
        for name, t, rank in (("output", out, 3), ("first_token_offset", first_token_offset, 1)):
            if len(t.shape) != rank:
                raise ValueError(f"MoE {name} must be rank-{rank}; got shape {tuple(t.shape)}")
        for role, slots, major in (("token", a_slots, chain.matmul.a_major), ("weight", b_slots, chain.matmul.b_major)):
            unit_dim = -1 if major == "k" else -2
            for t in slots:
                if len(t.shape) != 3 or t.stride(unit_dim) != 1:
                    raise ValueError(
                        f"multi-GEMM MoE {role} must be rank-3 and contiguous along its {major}-major dimension; got shape {tuple(t.shape)} stride {tuple(t.stride())}"
                    )
        if backend._moe_dense_layout_bad(out):
            raise ValueError("multi-GEMM MoE output requires contiguous innermost dim")
        for spec, ci in zip(outputs_spec, outs):
            if len(ci.shape) != 3 or tuple(ci.shape) != backend._expected_output_shape(spec, chain, (S, N, K), top_k_scores):
                raise ValueError(
                    f"multi-GEMM MoE output {spec.source!r} must have shape {backend._expected_output_shape(spec, chain, (S, N, K), top_k_scores)}; got {tuple(ci.shape)}"
                )
        num_experts = int(b_slots[0].shape[0])
        num_groups = backend._moe_num_groups(first_token_offset, num_experts)
        a_stride_perms = [t.permute(1, 2, 0) for t in a_slots]
        b_stride_perms = [t.permute(1, 2, 0) for t in b_slots]
        c_perms = [ci.permute(1, 2, 0) for ci in outs]
        output_strides = tuple((stride for _spec, ci in zip(outputs_spec, c_perms) for stride in ci.stride()))
        problem_size = (
            S,
            N,
            K,
            num_experts,
            num_groups,
            *(x for t in a_stride_perms for x in t.stride()),
            *(x for t in b_stride_perms for x in t.stride()),
            *output_strides,
        )
        a_wrapped = [backend._maybe_wrap_layout(t.permute(1, 2, 0), backend._LEADING_DIM_A) for t in a_slots]
        b_wrapped = [backend._maybe_wrap_layout(t.permute(1, 2, 0), backend._LEADING_DIM_B) for t in b_slots]
        cs = [
            backend._wrap_raw_tensor(ci) if spec.is_reduction or spec.is_quant_scale else backend._maybe_wrap_layout(ci, backend._LEADING_DIM_C)
            for spec, ci in zip(outputs_spec, c_perms)
        ]
        for ref, t in zip(chain.aux_tensors, aux):
            if ref.grouped_by_moe and (len(t.shape) != 3 or int(t.shape[0]) != num_groups):
                raise ValueError(f"per-group aux {ref.name!r} must be rank-3 with leading dim {num_groups} (the number of groups); got shape {tuple(t.shape)}")
        aux = tuple((backend._maybe_wrap_layout(backend._reshape_aux_to_fake(t, ref), backend._LEADING_DIM_AUX) for ref, t in zip(chain.aux_tensors, aux)))
        workspace = self._make_workspace(self._grid_ctas * self._desc_slots_per_cta + backend._MOE_SCHED_COUNTER_SLOTS, workspace)
        if self._validate_combine:
            backend._check_moe_combine_operands(chain, a_slots, b_slots, first_token_offset, [], [], (token_index, token_ks, top_k_scores), outs, K, workspace)
        backend._initialize_reduction_outputs(chain, outs, stream)
        backend._initialize_combine_output(chain, outs, stream)
        return self._launch_moe(problem_size, first_token_offset, workspace, (*a_wrapped, *b_wrapped), cs, aux, stream, token_index, token_ks, top_k_scores)


@dataclass
class _CompiledMoeBlockScaleGemm(_CompiledMoeBase[_ConfigT, _BindingT]):
    """Execute a compiled grouped block-scale GEMM."""

    _check_padded_operands: ClassVar[bool] = False
    _check_all_operands: ClassVar[bool] = False

    def _launch_single(
        self, token, weight, sfa, sfb, first_token_offset, output, snke, workspace=None, stream=None, token_index=None, token_ks=None, top_k_scores=None
    ):
        """Validate and launch one block-scaled grouped GEMM."""
        backend = self._compiler
        first_token_offset = backend._offset_vector(first_token_offset)
        if len(snke) < 3:
            raise ValueError(f"MoE block-scale call needs problem_size (S, N, K[, ...]); got {snke!r}")
        S, N, K = (int(snke[0]), int(snke[1]), int(snke[2]))
        if self._validate_runtime_quant:
            backend._check_runtime_quant(self.chain, self.config, S, N)
        outputs_spec = self.chain.outputs
        outputs = list(output) if isinstance(output, (list, tuple)) else [output]
        if len(outputs) != len(outputs_spec):
            raise ValueError(
                f"this MoE block-scale graph has {len(outputs_spec)} output(s) ({[o.source for o in outputs_spec]}); got {len(outputs)}. Pass outputs in slot order."
            )
        ranked_inputs = [("token", token, 3), ("weight", weight, 3), ("first_token_offset", first_token_offset, 1)]
        if sfa is not None:
            ranked_inputs.append(("sfa", sfa, 3))
        if sfb is not None:
            ranked_inputs.append(("sfb", sfb, 3))
        for name, t, rank in ranked_inputs:
            if len(t.shape) != rank:
                raise ValueError(f"MoE block-scale {name} must be rank-{rank}; got shape {tuple(t.shape)}")
        for spec, t in zip(outputs_spec, outputs):
            if len(t.shape) != 3 or tuple(t.shape) != backend._expected_output_shape(spec, self.chain, (S, N, K), top_k_scores):
                raise ValueError(
                    f"MoE block-scale output {spec.source!r} must have shape {backend._expected_output_shape(spec, self.chain, (S, N, K), top_k_scores)}; got {tuple(t.shape)}"
                )
        num_experts = int(weight.shape[0])
        num_groups = backend._moe_num_groups(first_token_offset, num_experts)
        scale_blob_reason = backend._grouped_row_quant_scale_blob_reject(self.chain, outputs, S, N, num_groups)
        if scale_blob_reason is not None:
            raise ValueError(scale_blob_reason)
        a_perm = token.permute(1, 2, 0)
        b_perm = weight.permute(1, 2, 0)
        c_perms = [t.permute(1, 2, 0) for t in outputs]
        dense_bad = self.chain.output_specs and backend._moe_dense_layout_bad(outputs[0])
        if backend._moe_operand_layout_bad(self.chain, token, weight) or dense_bad:
            raise ValueError(
                f"MoE block-scale non-packed tensors require contiguous innermost dimensions: got token stride {tuple(token.stride())}, weight stride {tuple(weight.stride())}, output stride {tuple(outputs[0].stride())}"
            )
        output_strides = tuple((stride for _spec, ci in zip(outputs_spec, c_perms) for stride in ci.stride()))
        problem_size = (S, N, K, num_experts, num_groups, *tuple(a_perm.stride()), *tuple(b_perm.stride()), *output_strides)
        a = backend._maybe_wrap_layout(a_perm, backend._LEADING_DIM_A)
        b = backend._maybe_wrap_layout(b_perm, backend._LEADING_DIM_B)
        cs = [
            backend._wrap_raw_tensor(ci) if spec.is_reduction or spec.is_quant_scale else backend._maybe_wrap_layout(ci, backend._LEADING_DIM_C)
            for spec, ci in zip(outputs_spec, c_perms)
        ]
        sf_args = []
        if sfa is not None:
            sf_args.append(backend._maybe_wrap_layout(sfa.permute(1, 2, 0), backend._LEADING_DIM_AUX))
        if sfb is not None:
            sf_args.append(backend._maybe_wrap_layout(sfb.permute(1, 2, 0), backend._LEADING_DIM_AUX))
        workspace = self._make_workspace(self._grid_ctas * self._desc_slots_per_cta + backend._MOE_SCHED_COUNTER_SLOTS, workspace)
        if self._validate_combine:
            sfs = ([sfa] if sfa is not None else [], [sfb] if sfb is not None else [])
            backend._check_moe_combine_operands(
                self.chain, [token], [weight], first_token_offset, *sfs, (token_index, token_ks, top_k_scores), outputs, K, workspace
            )
        backend._initialize_reduction_outputs(self.chain, outputs, stream)
        backend._initialize_combine_output(self.chain, outputs, stream)
        return self._launch_moe(problem_size, first_token_offset, workspace, (a, b, *sf_args), cs, (), stream, token_index, token_ks, top_k_scores)

    def _call_variant_pack(self, variant_pack: dict, workspace=None, stream=None):
        """Bind block-scaled grouped GEMM inputs and select the launch path."""
        backend = self._compiler
        a_bufs, b_bufs, out_bufs, aux_bufs, fto, sfa, sfb, token_index, token_ks, top_k_scores, snk = backend._resolve_moe_variant_pack(self, variant_pack)
        _out_reqs = backend._output_align_reqs(self.chain, self.tma_slots, vec_bytes=self.vec_bytes_epi)
        _aux_reqs = backend._aux_align_reqs(self.chain, vec_bytes=self.vec_bytes_epi)
        _padded_a, _padded_b = (False, False)
        if self._check_padded_operands:
            _mixed = backend._mixed_width_operand_reject(self.chain, a_bufs, b_bufs)
            if _mixed is not None:
                raise ValueError(_mixed)
            from .kernel_registry import sm120_padded_e2m1_sides

            _padded_a, _padded_b = sm120_padded_e2m1_sides(self.chain)
            _padded = backend._padded_e2m1_operand_reject(
                [("token", x) for x in (a_bufs if _padded_a else ())] + [("weight", x) for x in (b_bufs if _padded_b else ())]
            )
            if _padded is not None:
                raise ValueError(_padded)
        _named = (
            [("token", x, 32 if _padded_a else 16, "ptr") for x in a_bufs]
            + [("weight", x, 32 if _padded_b else 16, "ptr") for x in b_bufs]
            + [(f"output[{i}]", x, _out_reqs[i], "full") for i, x in enumerate(out_bufs)]
            + [(f"aux {self.chain.aux_tensors[i].name!r}", x, _aux_reqs[self.chain.aux_tensors[i].name], "full") for i, x in enumerate(aux_bufs)]
            + [("SFA", x, 16, "ptr") for x in sfa or []]
            + [("SFB", x, 16, "ptr") for x in sfb or []]
        )
        _r = backend._alignment_reject(_named)
        if _r is not None:
            raise ValueError(_r)
        if sfa or sfb:
            _S, _N, _K = (snk[0], snk[1], snk[2])
            if self._validate_whole_scale_blocks:
                _r_k = _partial_scale_block_reject(int(_K), self.chain.block_scale.block_size)
                if _r_k is not None:
                    raise ValueError(_r_k)
            token_sf, weight_sf, weights = (sfa, sfb, b_bufs)
            if isinstance(self.chain.moe, backend.MoeSwapAbSpec):
                _S, _N = (_N, _S)
                token_sf, weight_sf, weights = (sfb, sfa, a_bufs)
            if self.chain.moe.mode == "gather":
                from .kernel_registry import linear_token_sf_reject

                source_rows = (b_bufs if isinstance(self.chain.moe, backend.MoeSwapAbSpec) else a_bufs)[0].shape[1]
                for sf in token_sf:
                    reason = linear_token_sf_reject(sf.shape, sf.stride(), source_rows, _K // self.chain.block_scale.block_size)
                    if reason is not None:
                        raise ValueError(reason)
                token_sf = []
            _sf_k4 = (_K // self.chain.block_scale.block_size + 3) // 4
            _segmented_sfa_rows = backend.segmented_row_scale_capacity_rows(int(_S), backend._moe_num_groups(fto, int(weights[0].shape[0])))
            _r_sf = backend._sf_blob_reject(
                [(f"token SF[{i}]", x, 4 * _sf_k4 * _segmented_sfa_rows) for i, x in enumerate(token_sf or [])]
            ) or backend.moe_weight_sf_reject(
                [(f"weight SF[{j}]", x, 512 * _sf_k4 * ((_N + 127) // 128), int(weights[j].shape[0])) for j, x in enumerate(weight_sf or [])]
            )
            if _r_sf is not None:
                raise ValueError(_r_sf)
        if isinstance(self.chain.moe, backend.MoeSwapAbSpec):
            return backend._launch_moe_swap_ab(
                self,
                a_bufs,
                b_bufs,
                out_bufs,
                aux_bufs,
                fto,
                sfa,
                sfb,
                snk,
                workspace,
                stream,
                token_index=token_index,
                token_ks=token_ks,
                top_k_scores=top_k_scores,
            )
        out = out_bufs if len(out_bufs) > 1 else out_bufs[0]
        fake_a = self.chain.block_scale.fake_dequant_a
        fake_b = self.chain.block_scale.fake_dequant_b
        if self.chain.is_multi_gemm or self.chain.ops:
            pairs = [((a_bufs[ai], None if fake_a else sfa[ai]), (b_bufs[bi], None if fake_b else sfb[bi])) for ai, bi in self.chain.gemm_operands]
            return self._call_multi_gemm(
                pairs, fto, out, snk, *aux_bufs, workspace=workspace, stream=stream, token_index=token_index, token_ks=token_ks, top_k_scores=top_k_scores
            )
        return self._launch_single(
            a_bufs[0],
            b_bufs[0],
            None if fake_a else sfa[0],
            None if fake_b else sfb[0],
            fto,
            out,
            snk,
            workspace=workspace,
            stream=stream,
            token_index=token_index,
            token_ks=token_ks,
            top_k_scores=top_k_scores,
        )

    def _call_multi_gemm(
        self, gemm_pairs, first_token_offset, output, snke, *aux, workspace=None, stream=None, token_index=None, token_ks=None, top_k_scores=None
    ):
        """Launch parallel grouped GEMMs."""
        backend = self._compiler
        first_token_offset = backend._offset_vector(first_token_offset)
        chain = self.chain
        ok = (
            isinstance(gemm_pairs, (list, tuple))
            and gemm_pairs
            and all((isinstance(p, (list, tuple)) and len(p) == 2 and all((isinstance(o, (list, tuple)) and len(o) == 2 for o in p)) for p in gemm_pairs))
        )
        if not ok:
            raise ValueError("multi-GEMM MoE block-scale call expects a list of ((token,sfa),(weight,sfb)) pairs as the first argument")
        if len(gemm_pairs) != chain.num_gemms:
            raise ValueError(f"this graph has {chain.num_gemms} grouped matmul(s); got {len(gemm_pairs)} pair(s)")
        if len(snke) < 3:
            raise ValueError(f"MoE block-scale call needs problem_size (S, N, K[, ...]); got {snke!r}")
        S, N, K = (int(snke[0]), int(snke[1]), int(snke[2]))
        if self._validate_runtime_quant:
            backend._check_runtime_quant(chain, self.config, S, N)
        na, nb = (chain.num_a_operands, chain.num_b_operands)
        a_slots: list = [None] * na
        b_slots: list = [None] * nb
        for ((tok, sfa), (w, sfb)), (ai, bi) in zip(gemm_pairs, chain.gemm_operands):
            for slots, idx, data, sf, role in ((a_slots, ai, tok, sfa, "token"), (b_slots, bi, w, sfb, "weight")):
                if slots[idx] is None:
                    slots[idx] = (data, sf)
                elif slots[idx][0].data_ptr() != data.data_ptr():
                    raise ValueError(f"multi-GEMM MoE block-scale {role} sharing mismatch: distinct slot {idx} was given two different packed tensors.")
        if any((s is None for s in a_slots)) or any((s is None for s in b_slots)):
            raise ValueError("multi-GEMM MoE block-scale: not every distinct operand slot was filled")
        outputs_spec = chain.outputs
        outs = list(output) if isinstance(output, (list, tuple)) else [output]
        if len(outs) != len(outputs_spec):
            raise ValueError(f"fused multi-GEMM MoE block-scale has {len(outputs_spec)} output(s); got {len(outs)}")
        out = outs[0]
        for name, t, rank in (("output", out, 3), ("first_token_offset", first_token_offset, 1)):
            if len(t.shape) != rank:
                raise ValueError(f"MoE block-scale {name} must be rank-{rank}; got {tuple(t.shape)}")
        for spec, ci in zip(outputs_spec, outs):
            if len(ci.shape) != 3 or tuple(ci.shape) != backend._expected_output_shape(spec, chain, (S, N, K), top_k_scores):
                raise ValueError(
                    f"multi-GEMM MoE block-scale output {spec.source!r} must have shape {backend._expected_output_shape(spec, chain, (S, N, K), top_k_scores)}; got {tuple(ci.shape)}"
                )
        num_experts = int(b_slots[0][0].shape[0])
        num_groups = backend._moe_num_groups(first_token_offset, num_experts)
        scale_blob_reason = backend._grouped_row_quant_scale_blob_reject(chain, outs, S, N, num_groups)
        if scale_blob_reason is not None:
            raise ValueError(scale_blob_reason)
        a0, b0 = (a_slots[0][0], b_slots[0][0])
        a_stride_perms = [t.permute(1, 2, 0) for t, _sf in a_slots]
        b_stride_perms = [t.permute(1, 2, 0) for t, _sf in b_slots]
        c_perms = [ci.permute(1, 2, 0) for ci in outs]
        dense_bad = chain.output_specs and backend._moe_dense_layout_bad(out)
        operand_bad = (
            any((backend._moe_operand_layout_bad(chain, t, b0) for t, _sf in a_slots))
            or any((backend._moe_operand_layout_bad(chain, a0, w) for w, _sf in b_slots))
            if self._check_all_operands
            else backend._moe_operand_layout_bad(chain, a0, b0)
        )
        if operand_bad or dense_bad:
            raise ValueError("multi-GEMM MoE block-scale tensors require contiguous innermost dim")
        output_strides = tuple((stride for _spec, ci in zip(outputs_spec, c_perms) for stride in ci.stride()))
        problem_size = (
            S,
            N,
            K,
            num_experts,
            num_groups,
            *(x for t in a_stride_perms for x in t.stride()),
            *(x for t in b_stride_perms for x in t.stride()),
            *output_strides,
        )
        a_wrapped = [backend._maybe_wrap_layout(t.permute(1, 2, 0), backend._LEADING_DIM_A) for t, _sf in a_slots]
        b_wrapped = [backend._maybe_wrap_layout(t.permute(1, 2, 0), backend._LEADING_DIM_B) for t, _sf in b_slots]
        sfa_wrapped = [backend._maybe_wrap_layout(sf.permute(1, 2, 0), backend._LEADING_DIM_AUX) for _t, sf in a_slots if sf is not None]
        sfb_wrapped = [backend._maybe_wrap_layout(sf.permute(1, 2, 0), backend._LEADING_DIM_AUX) for _t, sf in b_slots if sf is not None]
        cs = [
            backend._wrap_raw_tensor(ci) if spec.is_reduction or spec.is_quant_scale else backend._maybe_wrap_layout(ci, backend._LEADING_DIM_C)
            for spec, ci in zip(outputs_spec, c_perms)
        ]
        for ref, t in zip(chain.aux_tensors, aux):
            if ref.grouped_by_moe and (len(t.shape) != 3 or int(t.shape[0]) != num_groups):
                raise ValueError(f"per-group aux {ref.name!r} must be rank-3 with leading dim {num_groups} (the number of groups); got shape {tuple(t.shape)}")
        aux = tuple((backend._maybe_wrap_layout(backend._reshape_aux_to_fake(t, ref), backend._LEADING_DIM_AUX) for ref, t in zip(chain.aux_tensors, aux)))
        workspace = self._make_workspace(self._grid_ctas * self._desc_slots_per_cta + backend._MOE_SCHED_COUNTER_SLOTS, workspace)
        if self._validate_combine:
            tokens, weights = ([t for t, _sf in a_slots], [w for w, _sf in b_slots])
            sfs = ([sf for _t, sf in a_slots if sf is not None], [sf for _w, sf in b_slots if sf is not None])
            backend._check_moe_combine_operands(chain, tokens, weights, first_token_offset, *sfs, (token_index, token_ks, top_k_scores), outs, K, workspace)
        backend._initialize_reduction_outputs(chain, outs, stream)
        backend._initialize_combine_output(chain, outs, stream)
        return self._launch_moe(
            problem_size,
            first_token_offset,
            workspace,
            (*a_wrapped, *b_wrapped, *sfa_wrapped, *sfb_wrapped),
            cs,
            aux,
            stream,
            token_index,
            token_ks,
            top_k_scores,
        )


_SHARED_EXPORTS = {
    "_ConfigT": _ConfigT,
    "_BindingT": _BindingT,
    "_CompiledRuntime": _CompiledRuntime,
    "_CompiledMoeBase": _CompiledMoeBase,
    "_fallback_cache_dir": _fallback_cache_dir,
    "_usable_cache_dir": _usable_cache_dir,
    "_cache_dir": _cache_dir,
    "_write_atomic": _write_atomic,
    "_import_kernel": _import_kernel,
    "_CompiledFusedGemm": _CompiledFusedGemm,
    "_CompiledMoeGemm": _CompiledMoeGemm,
    "_CompiledMoeBlockScaleGemm": _CompiledMoeBlockScaleGemm,
    "_CTAMMA_SUFFIX": _CTAMMA_SUFFIX,
    "_INJECT_MARKER_LINE": _INJECT_MARKER_LINE,
    "_aux_fake_shape_code": _aux_fake_shape_code,
    "_aux_can_use_explicit_fake_stride": _aux_can_use_explicit_fake_stride,
    "_aux_fake_block": _aux_fake_block,
    "_aux_signature_block": _aux_signature_block,
    "_aux_call_block": _aux_call_block,
    "_reduction_stride_kernel_params": _reduction_stride_kernel_params,
    "_reduction_stride_host_unpack": _reduction_stride_host_unpack,
    "_reduction_stride_host_unpack_from": _reduction_stride_host_unpack_from,
    "_reduction_stride_names": _reduction_stride_names,
    "_reduction_stride_host_pass": _reduction_stride_host_pass,
    "_splitk_epilogue_bindings": _splitk_epilogue_bindings,
    "_reduction_stride_compile_decls": _reduction_stride_compile_decls,
    "_reduction_stride_compile_symbols": _reduction_stride_compile_symbols,
    "_cvt_f32_to_fp8_scale_bits": _cvt_f32_to_fp8_scale_bits,
    "_cvt_e5m3_bits_to_f32": _cvt_e5m3_bits_to_f32,
    "_quant_device_imports": _quant_device_imports,
    "_template_stem": _template_stem,
    "_resolve_path_blocks": _resolve_path_blocks,
    "_resolve_blocks": _resolve_blocks,
    "_replace_marker_lines": _replace_marker_lines,
    "_MOE_SCHED_COUNTER_SLOTS": _MOE_SCHED_COUNTER_SLOTS,
    "_MOE_DESC_SLOT_BYTES": _MOE_DESC_SLOT_BYTES,
    "_is_layout_ambiguous": _is_layout_ambiguous,
    "_aux_fake_real_axes": _aux_fake_real_axes,
    "_swap_mn_view": _swap_mn_view,
    "_offset_vector": _offset_vector,
    "_moe_num_groups": _moe_num_groups,
    "_expected_output_shape": _expected_output_shape,
    "_initialize_reduction_outputs": _initialize_reduction_outputs,
    "_tma_alignment_reject": _tma_alignment_reject,
    "_partial_scale_block_reject": _partial_scale_block_reject,
    "_alignment_reject": _alignment_reject,
    "_sf_blob_reject": _sf_blob_reject,
    "_grouped_row_quant_scale_blob_reject": _grouped_row_quant_scale_blob_reject,
    "_moe_dense_layout_bad": _moe_dense_layout_bad,
    "_moe_operand_layout_bad": _moe_operand_layout_bad,
    "_declared_storage": _declared_storage,
    "_kernel_order": _kernel_order,
    "_check_moe_combine_n": _check_moe_combine_n,
    "_resolve_moe_variant_pack": _resolve_moe_variant_pack,
    "_register_legacy_device_view_adapter": _register_legacy_device_view_adapter,
    "_moe_carve_workspace": _moe_carve_workspace,
    "_moe_launch_tail": _moe_launch_tail,
}

_impl = importlib.import_module(f"{__package__}.{active_family()}.compiler")
_impl.__dict__.update(_SHARED_EXPORTS)
sys.modules[__name__] = _impl
