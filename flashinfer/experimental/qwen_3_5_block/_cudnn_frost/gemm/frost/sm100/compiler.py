# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compiler driver: cudnn graph -> rendered GEMM kernel.py -> compiled callable.

Ties together graph_analyzer + epilogue_codegen + a kernel template: analyze,
codegen, render the template's @@INJECT_*@@ markers, cache-write, import, compile.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
from collections import Counter
import importlib.util
import logging
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Callable, ClassVar

import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn
from cuda.bindings import driver as _cuda
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost import buffers
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import current_device
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.workspace import Workspace, align_up

_LOG = logging.getLogger(__name__)

# The arch tree this file IS. Each tree's compiler renders the templates of its
# own pipeline families only (PIPELINE_FAMILY in kernel_registry); a template of
# the other tree is declined, never rendered with the wrong tile constants.
_FAMILY = "sm100"


def _check_own_family(tmpl) -> None:
    """Decline ``tmpl`` unless this tree's compiler renders its pipeline. Runs
    AFTER the semantic gates of a precheck (so a graph-level reason still wins
    the message) and at the top of the tile-constant renderers (so a direct
    render of the other family's template fails clearly, not with the wrong
    constants). The message is the one the frost test conftest reports as a
    skip; keep the two in step."""
    if tmpl.family != _FAMILY:
        raise NotImplementedError(
            f"template {tmpl.file} ({tmpl.pipeline} pipeline) is served by the {tmpl.family} arch tree, but this process "
            f"runs the {_FAMILY} tree (cudnn.gemm.frost.{_FAMILY}.compiler): run it on a GPU that tree serves, or set "
            f"CUDNN_FRONTEND_GEMM_ARCH_FAMILY={tmpl.family} before cudnn.gemm.frost.compiler is first imported"
        )


def _as_custream(stream):
    # The execute-time stream from cudnn.get_stream(handle) is a raw CUstream int
    # (None/0 -> the default stream). Resolved per call, never cached, so a plan
    # runs on whichever stream the handle carries at execute time. Shared by every
    # compiled-GEMM executor (dense/block-scale + MoE-grouped).
    return _cuda.CUstream(stream or 0)


from ..dtypes import (
    DTYPE_BITS,
    DTYPE_BYTES,
    DTYPE_TO_CUTLASS,
    DTYPE_TO_MMA_KIND,
    MAX_EPI_CHUNK_ELEMS,
    _allowed_vsize,
    _aux_align_reqs,
    _compute_output_vec_bytes,
    _output_align_reqs,
    _pow2_floor,
    allowed_store_vsize,
    dtype_arch_reject,
    dense_output_layout,
    tensor_alignment,
)
from .epilogue_codegen import EpilogueSnippets, generate, tma_out_ready_marker, tma_out_value
from ..fusion_ir import (
    ZERO_PRESERVING_OPS,
    FusionChain,
    MoeSwapAbSpec,
    TensorRef,
    segmented_row_scale_capacity_rows,
)
from ..recipe import (
    CONST,
    FROM_M,
    KERNEL_AXES,
    REDUCTION_INIT_VALUE,
    _output_rule,
    build as build_recipe,
    check_alignment,
    check_shapes,
    contiguous_modulus,
    expected_shape,
    moe_weight_sf_reject,
)
from ..graph_analyzer import (
    GemmBinding,
    analyze_with_binding,
    resolve_variant_pack,
    swap_ab_binding,
)
from ..arch_family import current_arch
from ..tile_config import DEFAULT_CONFIG, TileConfig, _sm_count

# tvm-ffi front door: the compiled kernel gets a C++ argument-validation entry
# point instead of the Python launch path (~4.3x lower host dispatch). Guarded
# because CompileOptions.enable_tvm_ffi raises when tvm_ffi is absent; find_spec
# is the DSL's own predicate, so the two cannot disagree. Baked into the rendered
# source (frost_compile_options) so it is part of the content-addressed cache key.
_TVM_FFI_OK = importlib.util.find_spec("tvm_ffi") is not None
_FROST_COMPILE_OPTIONS = "--enable-tvm-ffi" if _TVM_FFI_OK else ""


def _supports_gpu_arch_option() -> bool:
    """Whether this cutedsl threads ``--gpu-arch`` from the cute.compile() options
    string to the compile target — dsl.compile_and_cache / get_arch_enum consult
    ``compile_options.gpu_arch`` before the (import-time, ambient-detected) env
    arch. That landed in the public wheel at 4.7 (frost's ``CUTEDSL_MIN_VERSION``),
    so the ONLY build that lacks it is a public ``nvidia-cutlass-dsl`` wheel below
    the floor. Reuses ``buffers.cutedsl_too_old`` so an internal RC (its own 0.x
    numbering) is judged new, not old — matching how the rest of frost gates it."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_state, cutedsl_too_old

    return not cutedsl_too_old(cutedsl_state()[1])


def _sm_target_string(arch: int) -> str:
    """``major*10+minor`` -> the cute ``sm_XXX[a]`` target, matching cutedsl's own
    ``detect_gpu_arch`` (the arch-specific ``a`` suffix appears from major >= 9)."""
    major, minor = divmod(arch, 10)
    return f"sm_{major}{minor}{'a' if major >= 9 else ''}"


def _frost_compile_options() -> str:
    """The ``cute.compile()`` options string for a build in the current
    ``build_device()`` scope. Pins ``--gpu-arch`` to the scope's GPU so the
    compile target follows the handle instead of the ambient CUDA device. On a
    public wheel below the floor the option is inert, so a handle-scoped build is
    refused rather than silently mis-targeted; an unscoped build needs no pin.
    (frost declines such wheels as too-old before reaching here, so the refusal
    is belt-and-suspenders.)

    Called at render time (inside the scope) so the arch is baked into the
    generated source, keeping it part of the content-addressed cache key."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import build_scope_device as _build_scope_device

    opts = [_FROST_COMPILE_OPTIONS] if _FROST_COMPILE_OPTIONS else []
    arch = _current_arch()  # scope-aware: the handle's GPU inside build_device()
    if arch is None:
        return " ".join(opts)  # render-only / no GPU visible
    if _supports_gpu_arch_option():
        opts.append(f"--gpu-arch {_sm_target_string(arch)}")
    elif _build_scope_device() is not None:
        # A handle pinned a GPU (build_device scope) but this wheel cannot pin the
        # compile target: cutedsl resolves it from an arch captured at IMPORT time,
        # which we can neither set NOR reliably read here (comparing against the
        # live device would miss an import-on-B, build-on-A process). We cannot
        # honor the scope, so refuse rather than bake scope constants into a
        # possibly-mis-targeted kernel. (frost's cutedsl_too_old gate already
        # declines such wheels, so this is belt-and-suspenders.) An unscoped build
        # makes no cross-device promise and falls through unchanged.
        major, minor = divmod(arch, 10)
        raise NotImplementedError(
            f"cudnn.frost: a handle-scoped build for sm_{major}{minor} needs an nvidia-cutlass-dsl "
            f"that pins the compile target (public >= 4.7 or an internal RC); this wheel cannot, so "
            f"the build cannot follow the handle. Upgrade cutedsl, or build with the handle's GPU "
            f"already current (no build_device scope)."
        )
    return " ".join(opts)


# ---------------------------------------------------------------------------
# Symbolic-shape helpers for aux fake tensors
# ---------------------------------------------------------------------------


def _aux_fake_shape_code(aux: TensorRef) -> str:
    """Shape-tuple expression for an aux fake tensor (sym_l/m/n where the dim
    matches the matmul, concrete 1s elsewhere)."""
    if len(aux.dim) == 3:
        if aux.grouped_by_moe:
            batch = "sym_g"
        else:
            batch = "1" if aux.dim[0] == 1 else "sym_l"
        if aux.bcast_mode == "scalar":
            return f"({batch}, 1, 1)"
        if aux.bcast_mode == "per_row":
            return f"({batch}, sym_m, 1)"
        if aux.bcast_mode == "per_col":
            return f"({batch}, 1, sym_n)"
        if aux.bcast_mode == "per_elem":
            return f"({batch}, sym_m, sym_n)"
    else:
        if aux.bcast_mode == "scalar":
            return "(1, 1)"
        if aux.bcast_mode == "per_row":
            return "(sym_m, 1)"
        if aux.bcast_mode == "per_col":
            return "(1, sym_n)"
        if aux.bcast_mode == "per_elem":
            return "(sym_m, sym_n)"
    raise AssertionError(f"unknown bcast_mode {aux.bcast_mode!r}")


def _aux_can_use_explicit_fake_stride(aux: TensorRef) -> bool:
    # Rank-1 aux is represented as a rank-2 broadcastable fake at compile, so
    # its raw rank-1 stride is not a valid fake stride.
    if len(aux.dim) not in (2, 3):
        return False
    stride1_dims = [i for i, stride in enumerate(aux.stride) if stride == 1]
    if len(stride1_dims) <= 1:
        return True
    nontrivial = [i for i in stride1_dims if aux.dim[i] != 1]
    return len(nontrivial) == 1


def _aux_fake_block(aux_tensors: list[TensorRef], *, dynamic_strides: bool = False, align_reqs: "dict | None" = None) -> str:
    """Declare `fake_<name>` for each aux tensor. No baked indent — the
    marker-replacement layer re-applies the marker line's indent (baking it here
    would double it). The host consumes each aux via explicit runtime strides, so
    the truthful branch declares its real strides and the fallback declares all
    strides free (never a compact over-claim). assumed_align is the aux's true LDG
    width from _aux_align_reqs, not a hardcoded 16."""
    lines = []
    for aux in aux_tensors:
        shape = _aux_fake_shape_code(aux)
        dtype = DTYPE_TO_CUTLASS[aux.dtype]
        _align = align_reqs[aux.name] if align_reqs is not None else 16
        if dynamic_strides and _aux_can_use_explicit_fake_stride(aux):
            stride = "(" + ", ".join(str(s) for s in aux.stride) + ")"
            lines.append(f"fake_{aux.name} = cute.runtime.make_fake_tensor({dtype}, {shape}, " f"stride={stride}, assumed_align={_align})")
        else:
            # Fake shape is rank-3 for len(dim)==3, else rank-2 (rank-1 aux is
            # rendered rank-2 broadcastable), so match the stride count to that.
            _rank = 3 if len(aux.dim) == 3 else 2
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


def _tma_c_plumbing(chain: FusionChain, tma_slots: "frozenset[int]" = frozenset({0}), *, splitk: bool = False) -> dict[str, str]:
    """Kernel / host / compile plumbing for the outputs on the TMA-C surface.

    One C descriptor per TMA slot, numbered 0..k-1 in SLOT order -- the kernel
    never sees the slot number, only its own index into `tma_c_descs`. The count
    is a property of the STORE PLAN, not of the chain: an output that takes STG
    passes as an ordinary tap instead."""

    # For splitK, kernel 1 only has the fp32 partials buffer as its C descriptor.
    if splitk:
        return {
            "INJECT_KERNEL_TMA_C_PARAMS": "tma_c_desc_0: cutlass.GridConstant[_tma.TensorMap],",
            "INJECT_TMA_C_LISTS": "tma_c_descs = [tma_c_desc_0]",
            "INJECT_HOST_TMA_C_PARAMS": "",
            "INJECT_HOST_TMA_C_LISTS": "",
            "INJECT_HOST_TMA_C_PASS": "tma_c_desc_list[0],",
            "INJECT_COMPILE_TMA_C_FAKES": "",
            "INJECT_COMPILE_TMA_C_PASS": "",
        }
    n_out = max(1, len(tma_slots))
    c_lists = "tma_c_descs = [" + ", ".join(f"tma_c_desc_{i}" for i in range(n_out)) + "]"
    if chain.has_moe:
        # Group clipping follows M, which is dim 0 only on M-major descriptors.
        c_lists += f"\ntma_c_m_major = {tuple(chain.output_specs[slot].major == 'm' for slot in sorted(tma_slots))!r}"
    return {
        "INJECT_KERNEL_TMA_C_PARAMS": ",\n".join(f"tma_c_desc_{i}: cutlass.GridConstant[_tma.TensorMap]" for i in range(n_out)) + ",",
        "INJECT_TMA_C_LISTS": c_lists,
        "INJECT_HOST_TMA_C_PARAMS": ",\n".join(f"c_{i}: cute.Tensor" for i in range(n_out)) + ",",
        "INJECT_HOST_TMA_C_LISTS": "_tma_c_outputs = [" + ", ".join(f"c_{i}" for i in range(n_out)) + "]",
        "INJECT_HOST_TMA_C_PASS": ",\n".join(f"tma_c_desc_list[{i}]" for i in range(n_out)) + ",",
        "INJECT_COMPILE_TMA_C_FAKES": "\n".join(
            f"fake_c_{j} = _make_fake_c({_cd_cutlass(dt)}, {2 if dt == 'fp4_e2m1' else 1}, {_fake_c_major(chain, slot)})"
            for j, (slot, dt) in enumerate(_tma_out_dtypes(chain, tma_slots))
        ),
        "INJECT_COMPILE_TMA_C_PASS": ",\n".join(f"fake_c_{i}" for i in range(n_out)) + ",",
    }


def _tma_out_dtypes(chain: FusionChain, tma_slots: "frozenset[int]") -> "list[tuple[int, str]]":
    """(dense slot, dtype) for each output on the TMA-C surface, in slot order."""
    return [(i, chain.output_specs[i].dtype) for i in sorted(tma_slots) if i < len(chain.output_specs)]


def _fake_c_major(chain, slot: int) -> bool:
    return chain.output_specs[slot].major == "m" if slot < len(chain.output_specs) else chain.out_major == "m"


def _mmajor_atom_m(dt: str) -> int:
    """Rows in a 128-byte M-contiguous column. BITS: `128 // DTYPE_BYTES` is 2x off sub-byte."""
    return 1024 // DTYPE_BITS[dt]


def _cd_cutlass(dt: str) -> str:
    return "cutlass.Int8" if dt == "fp4_e2m1" else DTYPE_TO_CUTLASS[dt]


def _cd_view_bits(dt: str) -> int:
    """Width of the carrier the C descriptor is declared with; sub-byte rides Int8."""
    return 8 if dt == "fp4_e2m1" else DTYPE_BITS[dt]


def _epi_row_bytes(dt: str, epi_n: int) -> int:
    """`epi_n` LOGICAL columns in bytes, packed."""
    return epi_n * DTYPE_BITS[dt] // 8


def _epi_row_elems(dt: str, epi_n: int) -> int:
    """The same row in carrier elements."""
    return _epi_row_bytes(dt, epi_n) * 8 // _cd_view_bits(dt)


def _tma_store_issue(chain, cfg, j: int, coord: str, ptr: str) -> "list[str]":
    # MoE rewrites D's `global_dim` per routed group to clip the ragged tail, so
    # it stores through its per-CTA workspace copy -- unless the offsets promise
    # says no tile crosses a group, in which case the original descriptor serves.
    desc = f"tma_c_descs[{j}].get_ptr()" if (not chain.has_moe or _moe_aligned_offsets(chain, cfg)) else f"d_desc_ptr_list[{j}]"
    return [
        "if warp_idx == 0:",
        "    if elect_one:",
        "        nvvm.cp_async_bulk_tensor_global_shared_cta(",
        f"            {desc},",
        f"            {ptr},",
        f"            {coord},",
        "        )",
    ]


def _tma_store_coord(chain, dim0: str, dim1: str) -> str:
    """Coordinate tuple matching the output TMA descriptor rank.

    Routed MoE outputs are one flat ``(S, N)`` surface: their logical batch is
    fixed to one and every store used to carry a redundant third coordinate.
    Keep ordinary GEMM descriptors rank-3, but let MoE issue the cheaper 2-D
    TMA-store form.
    """
    return f"({dim0}, {dim1})" if chain.has_moe else f"({dim0}, {dim1}, tile_l)"


def _tma_scatter_issue(chain, j: int, feature: str, row: str, ptr: str, rows: int, pitch: int) -> list[str]:
    extent = "N" if isinstance(chain.moe, MoeSwapAbSpec) else "M"
    return [
        "if warp_idx == 0:",
        "    if elect_one:",
        f"        for _sr in cutlass.range_constexpr({rows // 4}):",
        *[f"            _dst{k} = moe_scatter_row(token_index, token_ks, {row} + _sr * 4 + {k}, group_end, {extent}, moe_top_k)" for k in range(4)],
        f"            tma_scatter4(tma_c_descs[{j}].get_ptr(), {ptr} + _sr * {4 * pitch}, {feature}, _dst0, _dst1, _dst2, _dst3)",
    ]


def _tma_store_one(chain, cfg, epi_n: int, j: int, dt: str, major: str) -> "list[str]":
    """One TMA-stored output's store stage: stage this lane's fragment into the
    shared ring slot, then issue from it.

    Both layouts consume the SAME row-per-lane fragment; only the SMEM image
    differs. N-major stages a row and hands TMA one box. M-major stages a
    COLUMN -- lane `t` owns row `t`, so its `epi_n` elements land strided by the
    128-byte column pitch -- and hands TMA one box per M block.
    """
    v = f"_tsv_{j}"
    m_rows = cfg.epi_tile_m
    packed = _epi_packed_lanes(cfg)
    scatter = chain.has_moe and chain.moe.mode == "scatter"
    lines = [
        "epi_stage_idx = (epi_stage_idx + 1) % EPI_SMEM_STAGES",
        f"{v} = cutlass.Array(base=smem_d_ptr.data_ptr(epi_stage_idx * epi_subtile_elems), shape={_epi_stage_rows(cfg) * _epi_row_elems(dt, epi_n)}, dtype={_cd_cutlass(dt)})",
    ]
    val = tma_out_value(j)
    if major == "m":
        atom_m = _mmajor_atom_m(dt)
        elem_bytes = DTYPE_BYTES[dt]
        # The s128b XOR lands on bits the COLUMN index alone sets here, so the 8
        # XORed row bases hoist and every store offset is an immediate.
        assert 7 * (16 // elem_bytes) < atom_m
        # As N-major: `tidx` is the slot everywhere but packed. Under 2x2-DP
        # `tidx // atom_m` happens to name the COLUMN HALF, which is what it needs.
        slot = "(row - coord_m)" if packed else "tidx"
        lines += [
            f"_mrow_{j} = {slot} % {atom_m}",
            f"_mblk_{j} = ({slot} // {atom_m}) * {atom_m * epi_n}",
        ]
        lines += [f"_mx{x}_{j} = _mrow_{j} ^ {x * (16 // elem_bytes)}" for x in range(8)]
        for k in range(epi_n):
            st = f"{v}.data_ptr(_mblk_{j} + {k * atom_m} + _mx{k % 8}_{j}).store({val}[{k} : {k + 1}], alignment={elem_bytes})"
            lines.append(f"if row_active:\n    {st}" if packed else st)
        lines += [
            "cute.arch.fence_view_async_shared()",
            "nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)",
        ]
        for mb in range(m_rows // atom_m):
            if scatter:
                lines += _tma_scatter_issue(chain, j, f"coord_m + {mb * atom_m}", "col", f"{v}.data_ptr({mb * atom_m * epi_n})", epi_n, atom_m)
                continue
            lines += _tma_store_issue(
                chain,
                cfg,
                j,
                _tma_store_coord(chain, f"coord_m + {mb * atom_m}", "col"),
                f"{v}.data_ptr({mb * atom_m * epi_n})",
            )
        if _epi_dp22(cfg):
            # the other COLUMN half, not another M block: same rows, further along N
            lines += (
                _tma_scatter_issue(chain, j, "coord_m", "col + epi_cols_per_mma_m", f"{v}.data_ptr({atom_m * epi_n})", epi_n, atom_m)
                if scatter
                else _tma_store_issue(
                    chain,
                    cfg,
                    j,
                    _tma_store_coord(chain, "coord_m", "col + epi_cols_per_mma_m"),
                    f"{v}.data_ptr({atom_m * epi_n})",
                )
            )
    else:
        row_bytes = _epi_row_bytes(dt, epi_n)
        row_elems = _epi_row_elems(dt, epi_n)
        b, _tma_sw = _EPI_SWIZZLE_BY_ROW_BYTES[row_bytes]
        # sub-byte packs 2/byte: ring row and store coordinate are the packed ones.
        col_c = "col // 2" if DTYPE_BITS[dt] < 8 else "col"
        # `row - coord_m` is the row within the tile, and is `tidx` except packed.
        store = f"{v}.data_ptr({'(row - coord_m)' if packed else 'tidx'} * {row_elems}).store_swizzled({val}, alignment={row_bytes}, swizzle=cutlass.Swizzle({b}, 4, 3))"
        lines += [
            f"if row_active:\n    {store}" if packed else store,
            "cute.arch.fence_view_async_shared()",
            "nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)",
        ]
        lines += (
            _tma_scatter_issue(chain, j, col_c, "coord_m", f"{v}.data_ptr()", m_rows, row_elems)
            if scatter
            else _tma_store_issue(chain, cfg, j, _tma_store_coord(chain, col_c, "coord_m"), f"{v}.data_ptr()")
        )
        if _epi_dp22(cfg):
            # warps 2/3's column half, staged behind the first tile
            half = "(col + epi_cols_per_mma_m) // 2" if DTYPE_BITS[dt] < 8 else "col + epi_cols_per_mma_m"
            lines += (
                _tma_scatter_issue(chain, j, half, "coord_m", f"{v}.data_ptr({m_rows * row_elems})", m_rows, row_elems)
                if scatter
                else _tma_store_issue(
                    chain,
                    cfg,
                    j,
                    _tma_store_coord(chain, half, "coord_m"),
                    f"{v}.data_ptr({m_rows * row_elems})",
                )
            )
    lines += [
        "    if elect_one:",
        "        nvvm.cp_async_bulk_commit_group()",
        "    nvvm.cp_async_bulk_wait_group(EPI_SMEM_STAGES - 1, read=True)",
        "nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)",
    ]
    return lines


def _tma_store_sequence(chain, cfg, tma_slots: "frozenset[int]", epi_n: int) -> str:
    """The TMA arm, unrolled once per TMA-stored output.

    All of them walk the SAME SMEM ring: the slot is sized for the widest, and
    each output takes its OWN typed view of it (`Array(base=...)`), so N outputs
    cost no extra SMEM. `epi_n` is a COLUMN count and is shared; what differs per
    output is its dtype and its layout, both compile-time here."""
    lines: list[str] = []
    if cfg.split_k_slices > 1:
        lines += _tma_store_one(chain, cfg, epi_n, 0, "fp32", "n")
    else:
        for j, (slot, dt) in enumerate(_tma_out_dtypes(chain, tma_slots)):
            lines += _tma_store_one(chain, cfg, epi_n, j, dt, chain.output_specs[slot].major)
    return "\n".join(lines) if lines else "pass"


def _place_tma_stores(epilogue: str, chain, cfg, tma_slots: "frozenset[int]", epi_n: int) -> tuple[str, str]:
    """Inline each multi-output TMA store at its fragment's last use.

    Single-output graphs retain the aggregate trailing sequence, preserving
    their existing instruction scheduling.
    """
    # For splitK, kernel 1 only stores the fp32 partials, and kernel 2 handles the outputs.
    if cfg.split_k_slices > 1:
        return epilogue.replace(tma_out_ready_marker(0), ""), _tma_store_sequence(chain, cfg, tma_slots, epi_n)
    outputs = _tma_out_dtypes(chain, tma_slots)
    stream = len(outputs) > 1
    for j, (slot, dt) in enumerate(outputs):
        marker = tma_out_ready_marker(j)
        if epilogue.count(marker) != 1:
            raise RuntimeError(f"epilogue missing unique TMA-output marker {marker}")
        replacement = ""
        if stream:
            replacement = "\n".join(_tma_store_one(chain, cfg, epi_n, j, dt, chain.output_specs[slot].major))
        epilogue = epilogue.replace(marker, replacement)
    return epilogue, "pass" if stream else _tma_store_sequence(chain, cfg, tma_slots, epi_n)


def _host_tma_c_descs(chain, cfg, tma_slots: "frozenset[int]", epi_n: int) -> str:
    """Build one C descriptor per TMA-stored output. Each carries its OWN dtype,
    strides, box and swizzle, so outputs of different dtypes and different
    layouts can share one epilogue."""
    if cfg.split_k_slices > 1:
        # workspace: [batch * split][M][N].
        sw = _EPI_SWIZZLE_BY_ROW_BYTES[_epi_row_bytes("fp32", epi_n)][1]
        return "\n".join(
            [
                "tma_c_desc_0 = _tma.create_tensor_map_tiled(",
                "    global_address=splitk_partials.iterator.toint(),",
                "    dtype=cutlass.Float32,",
                "    global_dims=[n, m, batch * split_k_slices],",
                "    global_strides=[",
                "        n * 32 // 128,",
                "        m * n * 32 // 128,",
                "    ],",
                f"    box_dims=[{_epi_row_elems('fp32', epi_n)}, epi_tile_mn[0], 1],",
                f"    swizzle=_tma.TensorMapSwizzle.{sw},",
                ")",
                "tma_c_desc_list = [tma_c_desc_0]",
            ]
        )
    outs = _tma_out_dtypes(chain, tma_slots)
    lines: list[str] = []
    for j, (slot, dt) in enumerate(outs):
        width = _cd_view_bits(dt)
        if chain.output_specs[slot].major == "m":
            dims = "[m, n]" if chain.has_moe else "[m, n, batch]"
            outer = f"out_stride_n_{slot}"
            box = f"[{_mmajor_atom_m(dt)}, {epi_n}]" if chain.has_moe else f"[{_mmajor_atom_m(dt)}, {epi_n}, 1]"
            sw = "s128b"
        else:
            n_dim = "n // 2" if DTYPE_BITS[dt] < 8 else "n"
            dims = f"[{n_dim}, m]" if chain.has_moe else f"[{n_dim}, m, batch]"
            outer = f"out_stride_m_{slot}"
            box = f"[{_epi_row_elems(dt, epi_n)}, epi_tile_mn[0]]" if chain.has_moe else f"[{_epi_row_elems(dt, epi_n)}, epi_tile_mn[0], 1]"
            sw = _EPI_SWIZZLE_BY_ROW_BYTES[_epi_row_bytes(dt, epi_n)][1]
        strides = [f"        {outer} * {width} // 128,"]
        if chain.has_moe and chain.moe.mode == "scatter":
            box = f"[{_mmajor_atom_m(dt) if chain.output_specs[slot].major == 'm' else _epi_row_elems(dt, epi_n)}, 1]"
        if not chain.has_moe:
            strides.append(f"        out_stride_l_{slot} * {width} // 128,")
        lines += [
            f"_c{j} = _tma_c_outputs[{j}]",
            f"tma_c_desc_{j} = _tma.create_tensor_map_tiled(",
            f"    global_address=_c{j}.iterator.toint(),",
            f"    dtype={_cd_cutlass(dt)},",
            f"    global_dims={dims},",
            "    global_strides=[",
            *strides,
            "    ],",
            f"    box_dims={box},",
            f"    swizzle=_tma.TensorMapSwizzle.{sw},",
            ")",
        ]
    lines.append("tma_c_desc_list = [" + ", ".join(f"tma_c_desc_{j}" for j in range(len(outs))) + "]")
    return "\n".join(lines)


def _reduction_stride_kernel_params(chain: FusionChain) -> str:
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
    args = _reduction_stride_names(chain)
    return ",\n".join(args) + "," if args else ""


def _splitk_epilogue_bindings(chain: FusionChain, n_taps: int) -> str:
    """Unpack the reducer's ``taps`` / ``strides`` / ``aux`` tuples into the names the
    epilogue snippet uses (``gC_tap_0_ptr``, ``out_stride_m_0``, ``bias``, ...)."""
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


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


def _epi_tile_cols(
    config: TileConfig,
) -> int:
    """Per-CTA epilogue drain width in accumulator columns, for ONE MMA-M block
    (the templates' ``epi_cols_per_mma_m``; with ``mma_size_m > 1`` the per-GEMM
    ``cols_per_acc_stage`` is ``mma_size_m`` times this).

    Under 2-CTA MMA a per-CTA ``mma_inst_m`` of 64 means cluster-MMA m=128, whose
    2x2-DP drain splits the N range across the two 64-lane halves — so each CTA
    drains N/2. Like the LDTM shape (foot-gun #18) this keys on the MMA
    INSTRUCTION's M, not the CTA tile's: they agree only at mma_size_m == 1."""
    cols = config.cta_tile_n
    if config.is_cta_pair_mma and config.mma_tile_m == 64:
        cols //= 2
    return cols


# Widest epilogue chunk in ELEMENTS per pipeline family. The tcgen05 drains
# take the global cap; sm120's transposed-STG epilogue stores at most the
# 8-column fragment row run each compute warp owns (template: 8 % _STG_V == 0).
_EPI_CHUNK_ELEMS_BY_PIPELINE = {
    "sm100": MAX_EPI_CHUNK_ELEMS,
    "sm103": MAX_EPI_CHUNK_ELEMS,
    "sm120": 8,
}


def _epi_vec_bytes(
    chain: FusionChain,
    config: TileConfig,
) -> int:
    """The epilogue chunk width the kernel is rendered with: the chain-derived
    width additionally clamped so it divides every power-of-2 subtile span of
    this config's N-tile (see ``_compute_output_vec_bytes``), capped at the
    pipeline's widest store run (:data:`_EPI_CHUNK_ELEMS_BY_PIPELINE`)."""
    vec = _compute_output_vec_bytes(chain, tile_cols=_epi_tile_cols(config))
    return min(vec, _EPI_CHUNK_ELEMS_BY_PIPELINE[config.pipeline] * DTYPE_BYTES[chain.output_dtype])


def _epi_store_dtype(chain: FusionChain, config: TileConfig) -> str:
    return "fp32" if config.split_k_slices > 1 else chain.output_dtype


def _epi_chunk_elems(chain: FusionChain, config: TileConfig, use_tma_store: bool) -> int:
    if not use_tma_store:
        return _epi_vec_bytes(chain, config) // DTYPE_BYTES[chain.output_dtype]
    return _epi_n_for_chain(config, chain)


def _epi_chunk_bytes(chain: FusionChain, config: TileConfig, use_tma_store: bool) -> int:
    return _epi_chunk_elems(chain, config, use_tma_store) * DTYPE_BYTES[chain.output_dtype]


def _splitk_reduce_elems(chain: FusionChain) -> int:
    """Number fp32 elems per thread"""
    elems = 4
    while chain.matmul.N % elems:
        elems //= 2
    return elems


def _out_vec_bytes(chain: FusionChain, config: TileConfig, use_tma_store: bool) -> int:
    """The vector width for the output stores."""
    if config.split_k_slices > 1:
        return _splitk_reduce_elems(chain) * DTYPE_BYTES[chain.output_dtype]
    return _epi_chunk_bytes(chain, config, use_tma_store)


def _splitk_reduce_snippets(chain: FusionChain, config: TileConfig) -> EpilogueSnippets:
    """Two kernel splitK's reduce snippets"""
    return generate(
        chain,
        vec_bytes_epi=_out_vec_bytes(chain, config, False),
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        tma_slots=frozenset(),
        packed_lanes=False,
        split_k_slices=1,
    )


def _mainloop_chain_zero_preserving(ops) -> bool:
    """True iff applying this mainloop op chain to 0 provably yields 0.

    Per-op zero-preservation is judged by ``ZERO_PRESERVING_OPS`` (fusion_ir,
    single source of truth); an unlisted op is conservatively NON-preserving.
    A false negative only fires the K-OOB mask more often (still correct)."""
    zero = True
    for op in ops:
        if not zero:
            return False
        zero = op.op in ZERO_PRESERVING_OPS and not op.attrs
    return zero


def _l2_swizzle_budget_bytes() -> int:
    """Operand bytes the adaptive N-super-block rasterization may treat as L2-resident."""
    from ..tile_config import l2_swizzle_budget_bytes

    return l2_swizzle_budget_bytes()


def _render_tile_constants(
    cfg: TileConfig,
    chain: FusionChain,
    tmpl,
) -> str:
    """Emit module-level tile + dtype constants for a tcgen05 (sm100-family)
    template, appended below the template's defaults (last assignment wins).
    TileConfig geometry is dtype-agnostic (K in bytes); resolved to element
    counts via the chain's A dtype. Dtype constants derive from
    chain.matmul.{a,b}_dtype + output_dtype. Alongside the geometry this is where
    the TMEM budget (acc_stages, the alloc size, the drain layouts) and the
    tcgen05 SMEM-descriptor strides are decided -- none of which exists on the
    warp-MMA family, whose renderer lives in the sm120 tree
    (``cudnn.gemm.frost.sm120.compiler``).
    """
    _check_own_family(tmpl)
    # Both are facts of the TEMPLATE, not of the geometry: whether it consumes
    # the mixed-CGA fallback constant, and how much SMEM it holds back.
    if tmpl.warps_per_cta is not None:
        cfg = replace(cfg, warps_per_cta=tmpl.warps_per_cta)
    fallback_cluster = _mixed_cga_fallback(cfg, tmpl.file)
    smem_fixed_reserve = tmpl.smem_fixed_reserve
    _check_mma_k_dim(chain, cfg)
    # CTAs one MMA instruction spans; 1 on a family whose MMA has no pair.
    ctas_per_mma = cfg.ctas_per_mma
    # a_dt/b_dt: GMEM dtypes (what TMA loads). mma_a_dt/mma_b_dt: the MMA
    # instruction dtype — equal to the GMEM dtype (no implicit cast).
    a_dt = chain.matmul.a_dtype
    b_dt = chain.matmul.b_dtype
    mma_a_dt = _mma_a_dtype(chain)
    mma_b_dt = _mma_b_dtype(chain)
    accum_dt = chain.matmul.accum_dtype
    out_dt = chain.output_dtype
    epi_store_dt = _epi_store_dtype(chain, cfg)
    # Always the MMA dtype bytes (e.g. 2 for BF16 even when GMEM A is fp32);
    # drives K_TILE, swizzle, and MMA instruction sizing.
    elem_bytes = DTYPE_BYTES[mma_a_dt]
    # SMEM-row swizzle keyed by K-tile width in bytes; stride_byte_offset =
    # 8 * K_bytes (8 rows/chunk from the 8×16B tcgen05 core matrix).
    # A 16-byte row needs no swizzle: the XOR exists to spread a WIDE row across
    # the banks, and at 16 B a warp already covers all 32 banks conflict-free.
    # Same rule as the epilogue staging row (_EPI_SWIZZLE_BY_ROW_BYTES).
    _SWIZZLE_TABLE = {
        128: ("SWIZZLE_128B", "s128b"),
        64: ("SWIZZLE_64B", "s64b"),
        32: ("SWIZZLE_32B", "s32b"),
        16: ("NONE", "none"),
    }
    if cfg.cta_tile_k_bytes not in _SWIZZLE_TABLE:
        raise ValueError(f"TileConfig {cfg.name!r}: unsupported cta_tile_k_bytes=" f"{cfg.cta_tile_k_bytes} (supported: {sorted(_SWIZZLE_TABLE)})")
    smem_swizzle_name, tma_swizzle_name = _SWIZZLE_TABLE[cfg.cta_tile_k_bytes]
    smem_swizzle_bytes = cfg.cta_tile_k_bytes
    smem_desc_stride_byte_offset = 8 * cfg.cta_tile_k_bytes
    mma_inst_k_bytes = cfg.mma_tile_mnk(elem_bytes)[2] * elem_bytes
    cta_smem_m, cta_smem_n, _cta_smem_k = cfg.cta_smem_tile_mnk(elem_bytes)
    mn_group_elems = smem_swizzle_bytes // elem_bytes

    def _smem_desc_params(
        is_mn_major: bool,
        mn_extent: int,
        operand_name: str,
        num_mma: int = 1,
    ) -> tuple[int, int, int, int]:
        if not is_mn_major:
            return 16, smem_desc_stride_byte_offset, mma_inst_k_bytes, 1
        # Each MMA instruction's operand descriptor starts at its own MN
        # sub-block, so an MN-major operand needs the SLICE (not just the whole
        # SMEM extent) to be a whole number of swizzle groups.
        mn_slice = mn_extent // num_mma
        if mn_slice < mn_group_elems or mn_slice % mn_group_elems != 0:
            what = f"per-MMA SMEM extent {mn_slice}" if num_mma > 1 else f"SMEM extent {mn_slice}"
            raise ValueError(
                f"TileConfig {cfg.name!r} cannot use {operand_name}-major input: " f"{what} is not a multiple of the " f"{mn_group_elems}-element swizzle group"
            )
        group_elems = mn_group_elems
        return (
            cfg.cta_tile_k_bytes * group_elems,
            8 * group_elems * elem_bytes,
            mma_inst_k_bytes * group_elems,
            group_elems,
        )

    _desc_num_mma = 1 if getattr(type(cfg), "FIXED_MMA_TILE_MN", None) is not None else cfg.mma_size_m
    a_lbo, a_sbo, a_k_step, a_tma_group_elems = _smem_desc_params(chain.matmul.a_major == "m", cta_smem_m, "M", _desc_num_mma)
    b_lbo, b_sbo, b_k_step, b_tma_group_elems = _smem_desc_params(chain.matmul.b_major == "n", cta_smem_n, "N")
    # Byte step from one MMA sub-block to the next inside the SMEM tile. Same
    # formula for both majors: a K-major tile is (MN x K) rows of
    # cta_tile_k_bytes, and an MN-major tile is MN/group_elems groups of
    # group_elems * cta_tile_k_bytes — both give cta_tile_k_bytes per MN element.
    a_smem_m_step_bytes = (cta_smem_m // cfg.mma_size_m) * cfg.cta_tile_k_bytes
    a_mcast_slices, b_mcast_slices, ab_empty_full_mask = _mcast_slice_plan(chain, cfg)

    lines = [
        f"# Tile config: {cfg.name}",
        f"mma_inst_shape_mnk = {cfg.mma_tile_mnk(elem_bytes)}",
        f"mma_k_dim = {int(cfg.mma_tile_k_bytes == 64)}",
        f"cta_group = {ctas_per_mma}",
        f"cgrp_tile_mnk = {cfg.cga_tile_mnk(elem_bytes)}",
        # Template `cta_tile_mnk` = per-CTA SMEM/TMA box dims (B's N halved under
        # 2-CTA MMA), NOT the logical per-CTA tile from TileConfig.
        f"cta_tile_mnk = {cfg.cta_smem_tile_mnk(elem_bytes)}",
        f"epi_tile_mn = {(cfg.epi_tile_m, _epi_n_for_chain(cfg, chain))}",
        f"threads_per_cta = {cfg.threads_per_cta}",
        f"cluster_shape_mnk = {cfg.cluster_shape}",
        f"matmul_a_batch = {chain.matmul.a_batch}",
        f"matmul_b_batch = {chain.matmul.b_batch}",
        f"a_is_m_major = {chain.matmul.a_major == 'm'}",
        f"b_is_n_major = {chain.matmul.b_major == 'n'}",
        f"mma_a_major = {1 if chain.matmul.a_major == 'm' else 0}",
        f"mma_b_major = {1 if chain.matmul.b_major == 'n' else 0}",
        f"ab_stages = {cfg.max_ab_stages(smem_fixed_reserve=smem_fixed_reserve)}",
        f"b_collector_ok = {_b_collector_ok(cfg)}",
        f"multicast_a = {cfg.multicast_a}",
        f"multicast_b = {cfg.multicast_b}",
        f"a_mcast_slices = {a_mcast_slices}",
        f"a_tma_box_m = {_a_tma_box_m(cfg, a_mcast_slices)}",
        f"b_mcast_slices = {b_mcast_slices}",
        f"ab_empty_full_mask = {ab_empty_full_mask}",
        f"ab_smem_swizzle = cutlass.experimental.primitives.Tcgen05SmemSwizzle.{smem_swizzle_name}",
        f"a_smem_desc_leading_byte_offset = {a_lbo}",
        f"a_smem_desc_stride_byte_offset = {a_sbo}",
        f"a_smem_k_step_bytes = {a_k_step}",
        f"a_smem_m_step_bytes = {a_smem_m_step_bytes}",
        f"a_tma_group_elems = {a_tma_group_elems}",
        f"b_smem_desc_leading_byte_offset = {b_lbo}",
        f"b_smem_desc_stride_byte_offset = {b_sbo}",
        f"b_smem_k_step_bytes = {b_k_step}",
        f"b_tma_group_elems = {b_tma_group_elems}",
        # MMA instructions the WARP tile spans per axis -- the kernel's loop
        # counts, read off the config instead of recomputed from the tiles. The
        # MMA warp issues mma_size_m of them per K-block and the epilogue drains
        # one M block per pass.
        f"mma_size_m = {cfg.mma_size_m}",
        f"mma_size_n = {cfg.mma_size_n}",
        f"mma_size_k = {cfg.mma_size_k}",
        f"ab_tma_swizzle = _tma.TensorMapSwizzle.{tma_swizzle_name}",
        "",
        f"# Dtype family: A={a_dt}->MMA{mma_a_dt}, B={b_dt}->MMA{mma_b_dt}, out={out_dt} (K_BYTES={cfg.cta_tile_k_bytes})",
        # ab_dtype: MMA operand dtype (SMEM holds / MMA reads).
        f"ab_dtype = {DTYPE_TO_CUTLASS[mma_a_dt]}",
        f"cd_dtype = {'cutlass.Int8' if out_dt == 'fp4_e2m1' else DTYPE_TO_CUTLASS[out_dt]}",
        f"epi_store_dtype = {DTYPE_TO_CUTLASS[epi_store_dt]}",
        f"mma_a_dtype = {DTYPE_TO_CUTLASS[mma_a_dt]}",
        f"mma_b_dtype = {DTYPE_TO_CUTLASS[mma_b_dt]}",
        f"mma_c_dtype = {DTYPE_TO_CUTLASS[accum_dt]}",
        f"acc_widen_to_fp32 = {accum_dt == 'int32' and out_dt != 'int32'}",
        # ab_tma_dtype: A/B TMA-descriptor element dtype (same for A and B — TMA
        # only cares about element byte width, identical across an a/b pair).
        f"ab_tma_dtype = {DTYPE_TO_CUTLASS[mma_a_dt]}",
        f"mma_kind = {DTYPE_TO_MMA_KIND[mma_a_dt]}",
        *_epi_swizzle_lines(cfg, epi_store_dt, chain),
    ]
    # Persistent kernel always: double-TMEM + L2 N-super-block swizzle.
    # (acc_stages is emitted below, once the TMEM budget is known.)
    lines.append(f"tile_swizzle_n = {1 if fallback_cluster is not None else 0}")
    lines.append(f"swizzle_l2_budget_bytes = {_l2_swizzle_budget_bytes()}")
    # Multi-GEMM (parallel matmuls sharing the epilogue). Always emitted;
    # single-GEMM = (1, 1, 1). gemm_a_idx[g]/gemm_b_idx[g] pick GEMM g's operand
    # from the distinct A/B pools. acc_stages resolved below per TMEM budget.
    lines.append(f"num_gemms = {chain.num_gemms}")
    lines.append(f"num_a_operands = {chain.num_a_operands}")
    lines.append(f"num_b_operands = {chain.num_b_operands}")
    lines.append(f"gemm_a_idx = {tuple(a for a, _ in chain.gemm_operands)}")
    lines.append(f"gemm_b_idx = {tuple(b for _, b in chain.gemm_operands)}")
    total_tmem = _tmem_cols_for_arch()
    lines.append(f"num_tmem_alloc_cols = {total_tmem}")
    lines.append(f"tmem_alloc_exclusive = {total_tmem > _MAX_NON_EXCLUSIVE_TMEM_COLS}")
    # TMEM accumulator budget. One acc stage holds, per GEMM, one region of
    # `mma_size_m` MMA-M blocks each `_epi_tile_cols` columns wide (the N-direction
    # MMAs subdivide that width, they do not add to it). `total_tmem == 0` means
    # no GPU is visible (render-only / CI) — keep the config's own depth then.
    cols_per_mma_m = _epi_tile_cols(cfg)
    acc_region_cols = chain.num_gemms * cfg.mma_size_m * cols_per_mma_m
    acc_stages = _ACC_STAGES_MAX
    if total_tmem:
        if acc_region_cols > total_tmem:
            raise NotImplementedError(
                f"accumulators need {acc_region_cols} TMEM columns "
                f"({chain.num_gemms} GEMM(s) × {cfg.mma_size_m} MMA-M block(s) × "
                f"{cols_per_mma_m} cols) but only {total_tmem} exist even at a "
                f"single acc stage. Pick a smaller cta_tile_n / mma_size_m or "
                f"fewer GEMMs."
            )
        acc_stages = _ACC_STAGES_MAX if 2 * acc_region_cols <= total_tmem else 1
    lines.append(f"acc_stages = {acc_stages}  # {acc_region_cols} acc cols/stage")
    # Multi-GEMM holds one SMEM buffer per DISTINCT operand, not the single A+B
    # `max_ab_stages` assumes. Carry that as a per-stage surcharge so the
    # ONE override below decides ab_stages -- computing it here would be silently
    # clobbered by that override, which sizes a single A+B (SMEM overflow at
    # launch once multi-GEMM can also take the TMA-store epilogue).
    mg_extra_per_stage = 0
    if chain.is_multi_gemm:
        # Per-CTA SMEM B-tile N is halved under 2-CTA MMA (the pair splits B's N).
        smem_n = cfg.cta_smem_tile_n
        base_mn = cfg.cta_tile_m + smem_n
        mg_mn = chain.num_a_operands * cfg.cta_tile_m + chain.num_b_operands * smem_n
        mg_extra_per_stage = (mg_mn - base_mn) * cfg.cta_tile_k_bytes
    # MoE grouped matmul: grouped persistent scheduler launches a FIXED cluster
    # count (≈ NUM_SMS / cluster_size); host grid and kernel stride share it.
    if chain.has_moe:
        lines.append(f"grid_num_clusters = {_grid_num_clusters(cfg)}")
        # first_token_offset dtype (int32/int64) drives the compile() fake; the
        # scheduler casts reads to Int32 internally.
        lines.append(f"offset_cutlass_dtype = {DTYPE_TO_CUTLASS[chain.moe.offset_dtype]}")
    # Mainloop fusion: the template's 4 extra warps are already in the config's
    # warps_per_cta (re-stamped above), so only the count itself is emitted.
    if chain.has_mainloop_fusion:
        lines.append("num_mainloop_warps = 4")
        # Per-operand fusion flags (const_expr-gated in the template).
        lines.append(f"mainloop_fuse_a = {chain.has_mainloop_fusion_a}")
        lines.append(f"mainloop_fuse_b = {chain.has_mainloop_fusion_b}")
        # K-OOB mask: when BOTH operands are fused and NEITHER chain maps 0->0,
        # the transform corrupts the TMA K-tail zero-fill (e.g. cos(0)=1) → OOB
        # K adds f_a(0)*f_b(0) to every accumulator. Then zero A's OOB K (via the
        # swizzle-aware load/store below) so the product with B's OOB is 0.
        koob_fix = (
            chain.has_mainloop_fusion_a
            and chain.has_mainloop_fusion_b
            and not _mainloop_chain_zero_preserving(chain.mainloop_a_ops)
            and not _mainloop_chain_zero_preserving(chain.mainloop_b_ops)
        )
        lines.append(f"mainloop_koob_fix = {koob_fix}")
        # CuTe XOR swizzle matching the MMA-dtype SMEM layout; bbits =
        # log2(K_BYTES/16) (s128b=Swizzle(3,4,3), s64b=(2,4,3), s32b=(1,4,3)).
        _bbits = (cfg.cta_tile_k_bytes // 16).bit_length() - 1
        lines.append(f"ab_swizzle = cutlass.Swizzle({_bbits}, 4, 3)")
        # Mixed-input mainloop (dtype cast): a fused operand may be LOADED
        # narrower than the MMA reads (e.g. int8 A -> bf16 MMA). TMA loads the
        # narrow tile into a separate LOAD SMEM buffer; the mainloop warps widen
        # it into the wide MMA buffer via store_swizzled. load==MMA ⇒ no cast.
        lines.append(f"mainloop_a_cast = {chain.mainloop_a_cast}")
        lines.append(f"mainloop_b_cast = {chain.mainloop_b_cast}")
        load_a_dt = chain.mainloop_a_load_dtype or chain.matmul.a_dtype
        load_b_dt = chain.mainloop_b_load_dtype or chain.matmul.b_dtype
        lines.append(f"ab_load_a_dtype = {DTYPE_TO_CUTLASS[load_a_dt]}")
        lines.append(f"ab_load_b_dtype = {DTYPE_TO_CUTLASS[load_b_dt]}")

    # Epilogue store vector width (see _compute_output_vec_bytes; clamped so it
    # divides every power-of-2 subtile span of this config's N-tile).
    vec_bytes_epi = _epi_vec_bytes(chain, cfg)
    lines.append(f"vec_bytes_epi = {vec_bytes_epi}")
    lines.append(f"split_k_slices = {cfg.split_k_slices}")
    if cfg.split_k_slices > 1:
        lines.append(f"splitk_reduce_elems = {_splitk_reduce_elems(chain)}")
    lines.append(f"frost_compile_options = {_frost_compile_options()!r}")
    # Epilogue store mode: TMA-store-via-SMEM (preferred) vs per-thread STG
    # (fallback). See _use_tma_store_epi() for gating.
    use_tma = _use_tma_store_epi(chain, cfg)
    # only 1 tma output for splitK
    lines.append(f"n_tma_outputs = {1 if (cfg.split_k_slices > 1 and use_tma) else len(_tma_slots_for(chain, cfg))}")
    lines.append(f"moe_aligned_offsets = {_moe_aligned_offsets(chain, cfg)}")
    lines.append(f"moe_gather = {chain.has_moe and chain.moe.mode == 'gather'}")
    lines.append(f"moe_scatter = {chain.has_moe and chain.moe.mode == 'scatter'}")
    lines.append(f"moe_top_k = {chain.moe.top_k if chain.has_moe else 1}")
    lines.append(f"moe_combine = {chain.has_moe and chain.moe.mode == 'combine'}")
    lines.append(f"moe_token_alignment = {_pow2_floor(math.gcd(chain.moe.offset_multiple, chain.matmul.N)) if isinstance(chain.moe, MoeSwapAbSpec) else 1}")
    lines.append(f"epi_slot_widen = {_epi_slot_widen(chain, cfg)}")
    # The three drain layouts, decided host-side so the templates can key on the
    # LAYOUT rather than re-deriving it from the mode: packed lane<16 (hardware
    # M=64 under 1-CTA MMA, foot-gun #18), the 2-CTA 2x2-DP split, or the plain
    # thread-per-row one. Spelling them out is what lets one drain body serve a
    # 1ctamma, a 2ctamma and a merged template unchanged.
    lines.append(f"epi_packed_lanes = {_epi_packed_lanes(cfg)}")
    lines.append(f"epi_dp22 = {_epi_dp22(cfg)}")
    lines.append(f"epi_stage_rows = {_epi_stage_rows(cfg)}")
    lines.append(f"epi_chunk_elems = {_epi_chunk_elems(chain, cfg, use_tma)}")
    # Final ab_stages override: account for the TMA-D SMEM buffer (fixed, when
    # TMA-store is active) AND a mixed-input mainloop's narrow LOAD buffer
    # (per-stage). Otherwise leave the plain max.
    smem_d_bytes = _smem_d_bytes(cfg, chain) if use_tma else 0
    cast_extra_per_stage = 0
    if chain.has_mainloop_fusion and (chain.mainloop_a_cast or chain.mainloop_b_cast):
        smem_n = cfg.cta_smem_tile_n
        k_elems = cfg.cta_tile_k_bytes // DTYPE_BYTES[chain.matmul.a_dtype]
        if chain.mainloop_a_cast:
            cast_extra_per_stage += cfg.cta_tile_m * k_elems * DTYPE_BYTES[chain.mainloop_a_load_dtype]
        if chain.mainloop_b_cast:
            cast_extra_per_stage += smem_n * k_elems * DTYPE_BYTES[chain.mainloop_b_load_dtype]
    if smem_d_bytes > 0 or cast_extra_per_stage > 0 or mg_extra_per_stage > 0:
        try:
            new_ab = cfg.max_ab_stages(
                smem_fixed_reserve=smem_fixed_reserve,
                extra_smem_bytes=smem_d_bytes,
                extra_per_stage_bytes=cast_extra_per_stage + mg_extra_per_stage,
            )
        except ValueError as exc:
            if chain.is_multi_gemm:
                raise NotImplementedError(
                    f"multi-GEMM: {chain.num_a_operands} A + {chain.num_b_operands} B " f"operand tiles per stage exceed SMEM budget at this geometry"
                ) from exc
            raise
        lines.append(
            f"ab_stages = {new_ab}  # SMEM-D {smem_d_bytes}B fixed" f" + cast LOAD {cast_extra_per_stage}B/stage" f" + multi-GEMM {mg_extra_per_stage}B/stage"
        )
    lines.extend(_quant_device_imports(chain))
    lines.extend(_mixed_cga_constants(cfg, fallback_cluster))
    return "\n".join(lines)


def _cvt_f32_to_fp8_scale_bits(fn_name: str, dsl_dtype: str, rnd: str) -> list[str]:
    """A ``fp32 -> <8-bit scale> byte`` helper, emitted into the generated kernel
    so it stays self-contained. The cvt unit is natively x2, so the pair form is
    the primitive and the scalar one converts a value against itself and drops
    the high byte -- col-quant, which has one scale per COLUMN, calls the pair."""
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
    """The inverse of :func:`_cvt_f32_to_fp8_scale_bits` for ue5m3. It goes
    through **bf16**, not fp16: bf16 carries 8 exponent bits, so it holds the
    whole E5M3 range including bytes 248..254 (up to 114688), which are finite
    because the format is canonical-NaN-only. fp16 would turn those into inf."""
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
    """Device-side converters between fp32 and the two scale formats whose
    user-level DSL cast is not usable here. These reach the same hardware cvt
    unit through the typed NVVM ops the cast itself is built on, which is what
    lets them ask for the two things the cast does not do:

    * ``sat=SATFINITE`` on the narrowing. The plain cast overflows to byte 255
      (NaN); a NaN scale poisons its whole block on dequantize. Measured on
      sm_107, that saturation is the ONLY way the cast differs here.
    * ``ue5m3 -> bf16`` on the widening. The cast widens through a type that
      reads E == 31 as inf, but E5M3 is canonical-NaN-only, so bytes 248..253
      are finite (up to 114688) and come back as inf.

    Both round UP: a scale rounded DOWN makes ``amax / scale`` exceed the output
    format's max, clamping the block's largest element.

    ``ue8m0`` needs no widening helper — its exponent bits are widened directly,
    with byte 0 handled as the FP32 subnormal 2**-127. ``ue5m3``'s cvt exists ONLY on sm_107, see the arch gate in
    :func:`_check_block_quant_supported`.

    Both take ``x == 0`` to byte 0: 2**-127 for ue8m0, 0.0 for ue5m3."""
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


# TMEM columns the GPU has — a HARDWARE property, so every pipeline running on
# a given arch gets the same budget
_TMEM_COLS_BY_ARCH: tuple[tuple[tuple[int, int], int], ...] = (
    ((100, 107), 512),
    ((107, 110), 576),
    ((110, 120), 512),
)

# Past this, tcgen05.alloc must ask for the exclusive mode (and the count stops
# being a power of two, so it goes through as a register operand).
_MAX_NON_EXCLUSIVE_TMEM_COLS = 512


_B_COLLECTOR_ARCH_RANGES: tuple[tuple[int, int], ...] = ((107, 110),)


def _b_collector_supported(arch: int | None = None) -> bool:
    """Whether this GPU's MMA can hold B in a collector buffer across MMAs."""
    a = _current_arch() if arch is None else arch
    return a is not None and any(lo <= a < hi for lo, hi in _B_COLLECTOR_ARCH_RANGES)


def _b_collector_ok(cfg, arch: int | None = None) -> bool:
    """...and whether THIS geometry may ask for it. The 64-row MMA may not: measured
    `cudaErrorIllegalInstruction` at `mma_tile_m == 64` under both CTA groups, so it
    is the per-CTA instruction M that decides, not the hardware M. Block-scale never
    reaches it (`validate_block_scale_config_sm100` requires `mma_tile_m % 128 == 0`)."""
    return _b_collector_supported(arch) and cfg.mma_tile_m == 128


def _tmem_cols_for_arch(arch: int | None = None) -> int:
    if arch is None:
        arch = _current_arch()
    for (lo, hi), cols in _TMEM_COLS_BY_ARCH:
        if arch is not None and lo <= arch < hi:
            return cols
    return 0


def _current_arch(device=None) -> int | None:
    """SM version of ``device`` (default: the current one) as ``major*10+minor``,
    or ``None`` when no GPU is visible (render-only / CI). A module-level name
    (the registry and the tests pin it here) over the family-neutral probe in
    ``arch_family`` -- the same one the facade picked this tree with."""
    return current_arch(device)


def _plan_device() -> int:
    """The GPU a plan compiled right now targets. Every device-derived constant
    baked into the kernel (ab_stages, grid_num_clusters, the target SM) is read
    for this device, so the plan records it and re-checks it at execute time."""
    return current_device()


def _check_plan_device(plan_device: int) -> None:
    """A plan's SMEM depth / cluster count / target SM are baked for ONE GPU;
    refuse to launch it anywhere else.

    Asks where the launch is going, not where each operand lives. The operands
    were never the question — a kernel built for one arch produces garbage on
    another whoever owns the memory — and the backend does not look at operand
    devices either: ``create_variant_pack`` sets only pointers, uids and the
    workspace, and ``graph_interface.h`` takes its ordinal from
    ``cuda_get_device``. Reading the current device once is 0.74 us against
    1.45 per operand for the walk this replaces -- and the import is at module
    scope because doing it here costs 1.1 us of the 1.7 this function takes.
    """
    device = current_device()
    if device != plan_device:
        raise ValueError(
            f"cudnn.frost: this FROST plan was built for cuda:{plan_device} but cuda:{device} is "
            f"current. The kernel's SMEM pipeline depth, cluster count and target SM are baked at "
            f"build time, so a plan cannot move between GPUs — rebuild it with cuda:{device} current."
        )


def _grid_num_clusters(cfg: TileConfig, device=None) -> int:
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.occupancy import max_active_clusters

    return max_active_clusters(cfg.cga_size_m * cfg.cga_size_n, device)


_SMEM_SWIZZLE_ATOM_ROWS = 8


def _mcast_slice_plan(chain: FusionChain, cfg: TileConfig) -> tuple[int, int, bool]:
    """(a_slices, b_slices, needs_full_empty_mask) for the cluster's TMA multicast.

    Every geometric input is the config's own -- the cluster, how many CTAs one
    MMA spans, the per-CTA SMEM tile -- so it is asked, not unpacked by callers.
    """
    cluster_m, cluster_n = cfg.cga_size_mn
    # A mainloop-fused operand is loaded PER CTA under a multi-CTA MMA, so its
    # group is one and it couples nothing.
    per_cta_a = cfg.is_cta_pair_mma and bool(chain.mainloop_a_ops)
    per_cta_b = cfg.is_cta_pair_mma and bool(chain.mainloop_b_ops)
    a_group = 1 if per_cta_a else cluster_n
    b_group = 1 if per_cta_b else cluster_m // cfg.ctas_per_mma
    atom = _SMEM_SWIZZLE_ATOM_ROWS
    a_slices = a_group if (chain.matmul.a_major == "k" and a_group > 1 and cfg.cta_tile_m % (a_group * atom) == 0) else 1
    b_slices = b_group if (chain.matmul.b_major == "k" and b_group > 1 and cfg.cta_smem_tile_n % (b_group * atom) == 0) else 1
    a_closed = cluster_n == 1 or a_slices > 1
    b_closed = cluster_m // cfg.ctas_per_mma == 1 or b_slices > 1
    return a_slices, b_slices, not (a_closed and b_closed)


def _cluster_mcast_patterns(cfg: TileConfig, cga_size_mnk: tuple[int, int, int] | None = None) -> tuple[int, int]:
    """(A, B) multicast bit patterns at CTA rank 0, for the config's own cluster
    or -- under mixed CGA -- for the fallback shape passed instead.

    A is shared along N (one bit per n_rank, stride cluster_m); B along the M
    groups an MMA spans (one bit each, stride ctas_per_mma). The kernel shifts
    each by its own rank.
    """
    cluster_m, cluster_n, _ = cga_size_mnk if cga_size_mnk is not None else cfg.cga_size_mnk
    ctas_per_mma = cfg.ctas_per_mma
    a_pattern = 0
    for n_idx in range(cluster_n):
        a_pattern |= 1 << (n_idx * cluster_m)
    b_pattern = 0
    for mma_idx in range(cluster_m // ctas_per_mma):
        b_pattern |= 1 << (mma_idx * ctas_per_mma)
    return a_pattern, b_pattern


# Preferred-cluster substitution (CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION)
# is a property of the PART, like the B collector above — but unlike it, every
# part from SM 10.0 up can do it, so this is a FLOOR, not a range: there is no
# known ceiling to encode. (No driver check rides along: the attribute and
# Blackwell support shipped together, so a part this new implies a driver that
# knows it.)
_MIXED_CGA_MIN_ARCH = 100


def _mixed_cga_supported(arch: int | None = None) -> bool:
    """Whether this GPU can group blocks into a preferred cluster and fall back
    to a smaller one where the preferred does not fit — SM 10.0 and up."""
    a = _current_arch() if arch is None else arch
    return a is not None and a >= _MIXED_CGA_MIN_ARCH


@functools.lru_cache(maxsize=None)
def _template_reads_fallback_cluster(template_file: str) -> bool:
    """Whether a template implements mixed CGA — i.e. whether it consumes the
    ``fallback_cluster_shape_mnk`` constant. Read off the source rather than a
    hand-kept capability flag, so it cannot drift from what the template does."""
    from ..kernel_registry import template_path

    return "fallback_cluster_shape_mnk" in template_path(template_file).read_text()


def _mixed_cga_fallback(cfg: TileConfig, template_file: str) -> tuple[int, int, int] | None:
    """The fallback cluster to attach to this launch, or ``None`` for a plain
    fixed-cluster launch — byte-for-byte the pre-mixed-CGA behavior.

    Nothing here is a caller knob: the shape is the config's own `fallback_cga_size_mnk`
    and the rest is facts. It is OFF when the GPU cannot substitute clusters,
    when the template has not been ported (its cluster constants are baked to the
    preferred shape, so a CTA landing in a smaller cluster would wait on arrivals
    that never come — a hang, not a lost optimization), when the config's cluster
    is ALREADY the minimum, when the preferred cluster is not an integer multiple
    of it per dim (what the driver requires of the pair), and when the config pins
    the N-super-block walk (that rasterization is not invariant across the two
    cluster shapes).
    """
    if os.environ.get("CUDNN_FROST_DISABLE_MIXED_CGA"):
        return None
    if not _mixed_cga_supported():
        return None
    if not _template_reads_fallback_cluster(template_file):
        return None
    fallback = cfg.fallback_cga_size_mnk
    return None if fallback == cfg.cga_size_mnk else fallback


def _mixed_cga_constants(cfg: TileConfig, fallback_cluster: tuple[int, int, int] | None) -> list[str]:
    """Constants for the kernel's runtime cluster-shape select.

    The config's own cluster is the PREFERRED (wide) shape; ``fallback_cluster``
    is the smaller one the device substitutes when a preferred cluster does not
    fit. Everything else the kernel needs follows arithmetically from the cluster
    dims it reads at runtime — only the multicast bit patterns are loop-built, so
    both are precomputed here.
    """
    a_pref, b_pref = _cluster_mcast_patterns(cfg)
    if fallback_cluster is None:
        a_fb, b_fb = a_pref, b_pref
        shape = "None"
    else:
        a_fb, b_fb = _cluster_mcast_patterns(cfg, fallback_cluster)
        shape = str(fallback_cluster)
    return [
        f"fallback_cluster_shape_mnk = {shape}",
        f"mixed_a_pattern_pref = {a_pref}",
        f"mixed_b_pattern_pref = {b_pref}",
        f"mixed_a_pattern_fb = {a_fb}",
        f"mixed_b_pattern_fb = {b_fb}",
    ]


# `Tcgen05SmemDesc.build()` lowers through the NON-versioned `_tcgen05_mma_smem_desc`
# intrinsic (cutlass-dsl experimental/primitives/descriptors.py:513-522), which keeps
# 14 bits of `addr >> 4`: a start address at or above 256 KiB wraps to the bottom of
# SMEM with no error (the SF UTCCP then copies A-operand bytes into the SF TMEM
# columns -> NaN/inf on every output, even at a single K-tile).  Only the ring ROOTS
# go through build(); per-stage / per-k-step offsets ride `advance_start_address`
# (:413-425), an unmasked encoded add that is correct past the line.  Reachable on
# sm107 alone -- its 327 KiB carveout puts rings past 256 KiB; sm100 / sm103 cap at
# 227 KiB.  The block-scale templates therefore declare their SMALL scale-factor
# rings before the big A/B rings; the renderer trims `ab_stages` until every root
# clears the line and refuses a layout a single stage still cannot fit (a typed
# failure instead of silent NaN).
_TCGEN05_DESC_BUILD_ADDR_LIMIT = 262144
_SMEM_RING_ALIGN = 1024
# Templates that declare the TMA-store staging buffer AHEAD of the A/B/SF rings, so
# its bytes push every ring root up (the plain block-scale template puts it after).
_SMEM_D_BEFORE_RINGS_TEMPLATES = frozenset({"sm100_moe_grouped_block_scale_matmul_fwd.py"})


def _block_scale_smem_desc_roots(*, header_bytes: int, rings: Sequence[tuple[str, int]]) -> dict[str, int]:
    """Byte offset of every ring ROOT: the 1024-aligned `cutlass.Array` rings laid out
    in DECLARATION order behind ``header_bytes``.  The template's fixed reserve bounds
    everything declared before the rings, so the modelled roots are >= the real ones
    (conservative: a root this says is safe is safe)."""
    off = header_bytes
    roots: dict[str, int] = {}
    for name, nbytes in rings:
        off = -(-off // _SMEM_RING_ALIGN) * _SMEM_RING_ALIGN
        roots[name] = off
        off += nbytes
    return roots


def _check_block_scale_desc_roots(cfg: TileConfig, ab_stages: int, roots: dict[str, int]) -> None:
    """Refuse a SMEM layout whose `Tcgen05SmemDesc.build()` root would wrap the 14-bit field."""
    bad = {name: off for name, off in roots.items() if off >= _TCGEN05_DESC_BUILD_ADDR_LIMIT}
    if bad:
        where = ", ".join(f"{name} at {off} B" for name, off in bad.items())
        raise ValueError(
            f"block-scale {cfg.name!r}: SMEM ring root(s) {where} sit at or above {_TCGEN05_DESC_BUILD_ADDR_LIMIT} B "
            f"with ab_stages={ab_stages}. `Tcgen05SmemDesc.build()` keeps 14 bits of the start address, so such a root "
            f"wraps to the bottom of SMEM and the MMA / UTCCP reads the wrong buffer (NaN/inf output, no fault). "
            f"Reduce the stage count, or build that root with a version-1 descriptor (`_tcgen05_mma_smem_desc_v2`)."
        )


def _render_block_scale_tile_constants(
    cfg: TileConfig,
    chain: FusionChain,
    tmpl,
) -> str:
    """Emit module-level constants for the tcgen05 block-scale matmul templates
    (the sm100 family, and sm103 through its ``is_sm103`` branches).

    Bypasses the generic dtype-byte machinery (FP4 is 0.5 B/elem); resolves
    everything from the chain's BlockScaleSpec + geometry, incl. SF SMEM/TMEM
    sizing and the SMEM→TMEM (utccp) copy schedules.
    """
    _check_own_family(tmpl)
    from ..tile_config import CtaPairTileConfig, validate_block_scale_config_sm100

    if tmpl.warps_per_cta is not None:
        cfg = replace(cfg, warps_per_cta=tmpl.warps_per_cta)
    use_tma_store_epi = _use_tma_store_epi(chain, cfg)
    fallback_cluster = _mixed_cga_fallback(cfg, tmpl.file)
    # Only the CTA-pair families have a block-scale template, so the axis is
    # readable here -- assert it rather than assume it.
    assert isinstance(cfg, CtaPairTileConfig), f"{cfg.pipeline} has no block-scale template"
    cta_group = cfg.cta_group

    bs = chain.block_scale
    assert bs is not None
    both_fp4 = bs.both_fp4
    mixed_width = bs.mixed_width
    is_sm103 = cfg.pipeline == "sm103"
    # The 64-byte block-scale MMA. A pipeline no longer implies it: sm100
    # configs carry both widths and the ACTIVE arch decides (validate_block_scale_config_sm100).
    mma_k64 = cfg.mma_tile_k_bytes == 64
    if is_sm103 and not both_fp4:
        raise NotImplementedError("the sm103 block-scale pipeline is fp4-only; " f"{bs.a_dtype} data with {bs.sf_dtype} scales runs the sm100 templates")
    # Baseline SM100 mixed block-scale MMA uses K32 and the padded E2M1 SMEM format
    # (16 FP4 lanes in 8 payload bytes + 8 padding bytes). Rubin's K64 form
    # consumes native packed FP4 instead.
    padded_fp4 = mixed_width and not mma_k64

    a_data_elem_bits = DTYPE_BITS[bs.a_dtype]
    b_data_elem_bits = DTYPE_BITS[bs.b_dtype]
    a_smem_elem_bits = 8 if padded_fp4 and a_data_elem_bits == 4 else a_data_elem_bits
    b_smem_elem_bits = 8 if padded_fp4 and b_data_elem_bits == 4 else b_data_elem_bits
    # K_BYTES names the byte width of the widest operand.  In a mixed-width MMA
    # both sides span the same logical K, while the native packed FP4 SMEM row
    # occupies half the bytes of its FP8 peer.
    data_elem_bits = max(a_data_elem_bits, b_data_elem_bits)
    cta_k_elems = cfg.cta_tile_k_bytes * 8 // data_elem_bits
    validate_block_scale_config_sm100(cfg, bs.block_size, cta_k_elems)

    cta_m = cfg.cta_tile_m
    cta_n = cfg.cta_tile_n
    bs_a_mcast_slices, bs_b_mcast_slices, bs_ab_empty_full_mask = _mcast_slice_plan(chain, cfg)
    # MMA K-instruction width (sm100 → 32 bytes): fp4 → 64 elems, fp8 → 32.
    mma_inst_k_bytes = cfg.mma_tile_k_bytes
    mma_inst_k_elems = mma_inst_k_bytes * 8 // data_elem_bits
    num_kblocks = cta_k_elems // mma_inst_k_elems
    a_cta_k_bytes = cta_k_elems * a_smem_elem_bits // 8
    b_cta_k_bytes = cta_k_elems * b_smem_elem_bits // 8
    a_mma_inst_k_bytes = mma_inst_k_elems * a_smem_elem_bits // 8
    b_mma_inst_k_bytes = mma_inst_k_elems * b_smem_elem_bits // 8

    # --- Operand major (K- / M- / N-major) -----------------------------------
    # FP4 (nvfp4/mxfp4) is K-major only — sub-byte (Float4E2M1FNx2) packing
    # mis-strides an M/N-contiguous descriptor. mxfp8 (1 B/elem) may be M-major
    # (A) / N-major (B). SF layout is unchanged regardless of data major.
    a_major = chain.matmul.a_major
    b_major = chain.matmul.b_major
    if (a_data_elem_bits == 4 and a_major != "k") or (b_data_elem_bits == 4 and b_major != "k"):
        raise ValueError(
            f"FP4 block-scaled inputs must be K-major (got A={a_major}-major, " f"B={b_major}-major); only an MXFP8 side supports M/N-major layout."
        )
    # Major-dependent SMEM descriptor params, mirroring _smem_desc_params.
    cta_smem_m = cta_m
    cta_smem_n = cta_n // cta_group

    def _bs_smem_desc_params(is_mn_major, mn_extent, name, cta_k_bytes, inst_k_bytes, elem_bits):
        if not is_mn_major:
            return 16, 8 * min(cta_k_bytes, 128), inst_k_bytes, 1
        elem_bytes = elem_bits // 8
        mn_group_elems = cta_k_bytes // elem_bytes
        if mn_extent < mn_group_elems or mn_extent % mn_group_elems != 0:
            raise ValueError(
                f"block-scale config {cfg.name!r} cannot use {name}-major input: "
                f"SMEM extent {mn_extent} is not a multiple of the "
                f"{mn_group_elems}-element swizzle group"
            )
        g = mn_group_elems
        return (
            cta_k_bytes * g,
            8 * g * elem_bytes,
            inst_k_bytes * g,
            g,
        )

    a_lbo, a_sbo, a_k_step, a_tma_group_elems = _bs_smem_desc_params(a_major == "m", cta_smem_m, "M", a_cta_k_bytes, a_mma_inst_k_bytes, a_data_elem_bits)
    b_lbo, b_sbo, b_k_step, b_tma_group_elems = _bs_smem_desc_params(b_major == "n", cta_smem_n, "N", b_cta_k_bytes, b_mma_inst_k_bytes, b_data_elem_bits)

    # Packed (Float4E2M1FNx2 / Float8) element count per row for SMEM.
    a_pack = 8 // a_smem_elem_bits
    b_pack = 8 // b_smem_elem_bits
    a_packed_per_row = cta_k_elems // a_pack
    b_packed_per_row = cta_k_elems // b_pack
    sA_packed_elems = cta_m * a_packed_per_row
    sB_packed_elems = (cta_n // cta_group) * b_packed_per_row

    # --- Scale factors --------------------------------------------------------
    sf_k = cta_k_elems // bs.block_size  # SF values along K per tile
    sf_k4 = sf_k // 4  # 4 SF-K per utccp atom
    nb_m = cta_m // 128
    nb_n = cta_n // 128
    mma_nb_m = cfg.mma_tile_m // 128
    mma_nb_n = cfg.mma_tile_n // 128
    sfa_nb_m = mma_nb_m * cfg.mma_size_m

    _REGISTERS_PER_ATOM = 4  # cols per 128×4 utccp atom
    scales_per_inst = mma_inst_k_elems // bs.block_size
    word_scales = max(_REGISTERS_PER_ATOM, scales_per_inst)  # cols per block-word
    word_atoms = -(-word_scales // _REGISTERS_PER_ATOM)  # atoms copied per word (ceil: a partial atom still costs one)
    insts_per_word = max(_REGISTERS_PER_ATOM // scales_per_inst, 1)
    _REGISTERS_PER_BLOCK = word_scales  # SF word width per block
    if is_sm103:
        # A whole K-tile of SF is TMEM-resident, so a region is sf_k wide.
        num_sf_words = sf_k4
        sfa_tmem_cols = sfa_nb_m * sf_k
        sfb_tmem_cols = mma_nb_n * sf_k
    else:
        # One SF word per 128-row / 128-column block, packed back to back and
        # refreshed per word. SFB must stay contiguous (one instruction's SFB read
        # extent grows with n_dim, so a single scale_b reads every N block as one
        # span); SFA is one block per M sub-block, so its M instructions stay
        # independent.
        num_sf_words = max(num_kblocks // insts_per_word, 1)  # utccp refreshes / k-tile
        if num_sf_words * insts_per_word != num_kblocks:
            raise NotImplementedError(
                f"block-scale {cfg.name!r}: {num_sf_words} SF word(s) x {insts_per_word} "
                f"instruction(s) per word covers {num_sf_words * insts_per_word} K-blocks, "
                f"but the K-tile has {num_kblocks}. The MMA would read scale bytes that were "
                f"never staged, or skip the trailing K-blocks. Reachable only if the "
                f"cta_tile_k_bytes == 128 requirement is relaxed."
            )
        sfa_tmem_cols = sfa_nb_m * _REGISTERS_PER_BLOCK
        sfb_tmem_cols = mma_nb_n * _REGISTERS_PER_BLOCK
    # utccp SMEM-source offsets (16-byte units). One 128×4 atom = 512 B = 32;
    # consecutive K-atoms 1 atom apart; each M/N-block of 128 rows is sf_k4 atoms
    # further along the SF SMEM tile.
    sf_atom_desc_stride = 32  # K-atom stride
    sf_block_desc_stride = 32 * sf_k4  # M/N-block stride

    # SF SMEM bytes per stage (sf_dtype 1 byte; whole k-tile loaded by TMA).
    sfa_smem_bytes = cta_m * sf_k
    sfb_smem_bytes = cta_n * sf_k
    # SF TMA box (fp16-recast trick): inner 256 fp16 (=512 B = one 128×4 SF atom
    # block, hardcoded in the template), box-K = sf_k4 atoms.
    sf_tma_box_k = sf_k4

    # --- TMEM budget → acc_stages (+ optional overlap trick) -----------------
    # acc needs cta_n cols/stage; SF needs sf_total_cols. Two non-overlapping acc
    # buffers = 2*cta_n + sf. When that doesn't fit (e.g. cta_n=256) use the
    # OVERLAP trick: the two acc buffers overlap by acc_overlap_cols (multiple of
    # the 32-col epilogue load); the epilogue drains the overlap subtile FIRST
    # and arrives acc_empty early, so the next MMA reuses it — preserving
    # double-TMEM pipelining at a single mbar.
    total_tmem = _tmem_cols_for_arch()

    # --- TMEM acc-stage + overlap (per-GEMM budget; arch- & count-agnostic) ---
    # SF = one fixed word PER DISTINCT OPERAND (shared A → one SFA word).
    # Each GEMM's acc gets its OWN region; the 2 tile-stage buffers overlap
    # WITHIN it. Budget SF, split the rest across GEMMs, then run the single-GEMM
    # stage/overlap decision per GEMM:
    #   per_gemm = (total_tmem - sf) // num_gemms          (>= cta_n, else reject)
    #   2*cta_n <= per_gemm        -> acc_stages = 2 (double-buffer)
    #   else overlap = ceil32(2*cta_n - per_gemm):
    #         overlap < cta_n      -> acc_stages = 1 + overlap (parity toggle)
    #         else                 -> acc_stages = 1, no overlap
    # GEMM g occupies [g*acc_gemm_stride, (g+1)*acc_gemm_stride); stage s at
    # +s*acc_stage_stride. Single-GEMM collapses to legacy behaviour.
    num_gemms = chain.num_gemms
    mma_size_m = cfg.mma_size_m
    # Same per-M-block drain width the plain path uses (block-scale pins
    # mma_inst_m to a multiple of 128, so the 2x2-DP halving never fires here).
    epi_cols_per_mma_m = _epi_tile_cols(cfg)
    acc_cols_per_stage = mma_size_m * epi_cols_per_mma_m
    na, nb = chain.num_a_operands, chain.num_b_operands
    real_na = 0 if bs.fake_dequant_a else na
    real_nb = 0 if bs.fake_dequant_b else nb
    sf_total_cols = na * sfa_tmem_cols + nb * sfb_tmem_cols
    # Columns each instruction reads from its scale base -- ISA opUTCHMMA,
    # "Load A/B scale factors" (sf{a,b}_tmem_cols).
    if is_sm103:
        # The resident K-tile means the scale pointer advances per k-block, so a
        # read window can reach past the operand's own region.
        sf_ids = [scales_per_inst * j % 4 for j in range(num_kblocks)]
        wide_vec = bs.block_size == 16
        sfb_extra = 4 if cta_n <= 128 else 8
        sfa_off = [scales_per_inst * j // 4 * 4 * sfa_nb_m for j in range(num_kblocks)]
        sfb_off = [scales_per_inst * j // 4 * 4 * mma_nb_n for j in range(num_kblocks)]
        extra = [sfb_extra if (wide_vec or sf_ids[j] >= 2) else 0 for j in range(num_kblocks)]
        sfa_spans = [(sfa_off[j], 4 if (not wide_vec and sf_ids[j] < 2) else 8) for j in range(num_kblocks)]
        sfb_spans = [(sfb_off[j], 2 * ((cta_n + 63) // 64) + extra[j]) for j in range(num_kblocks)]
        sf_reserved_cols = max(
            sf_total_cols,
            (na - 1) * sfa_tmem_cols + max(off + cols for off, cols in sfa_spans),
            na * sfa_tmem_cols + (nb - 1) * sfb_tmem_cols + max(off + cols for off, cols in sfb_spans),
        )
    else:
        # Every SF word is re-utccp'd into the same columns and the scale base is fixed,
        # so an instruction reads exactly its own operand's region.
        sf_reserved_cols = sf_total_cols
    per_gemm = (total_tmem - sf_reserved_cols) // num_gemms
    if per_gemm < acc_cols_per_stage:
        raise NotImplementedError(
            f"block-scale {cfg.name!r}: per-GEMM TMEM budget {per_gemm} < one acc "
            f"({acc_cols_per_stage}) for {num_gemms} GEMMs + SF({sf_total_cols}). "
            f"Smaller cta_n / fewer GEMMs."
        )
    acc_overlap_cols = 0
    if 2 * acc_cols_per_stage <= per_gemm:
        acc_stages = 2  # full per-GEMM double-buffer
    else:
        acc_stages = 1
        gran = _epi_n_for_chain(cfg, chain)  # epilogue TMEM-load drain unit (cols)
        ov = ((2 * acc_cols_per_stage - per_gemm + gran - 1) // gran) * gran
        if ov < acc_cols_per_stage:  # else no room -> plain 1-stage
            acc_overlap_cols = ov
    use_acc_overlap = acc_overlap_cols > 0
    # within-GEMM per-stage stride + per-GEMM region size:
    acc_stage_stride = (acc_cols_per_stage - acc_overlap_cols) if use_acc_overlap else acc_cols_per_stage
    if acc_stages == 2:
        acc_gemm_stride = 2 * acc_cols_per_stage
    elif use_acc_overlap:
        acc_gemm_stride = 2 * acc_cols_per_stage - acc_overlap_cols
    else:
        acc_gemm_stride = acc_cols_per_stage
    acc_overlap_subtiles = acc_overlap_cols // _epi_n_for_chain(cfg, chain)
    acc_region_cols = acc_cols_per_stage  # per-stage stride WITHIN a GEMM

    sf_region_base = num_gemms * acc_gemm_stride
    # Per-distinct-operand SF word col bases (single-GEMM → length-1 lists).
    sfa_col_bases = [sf_region_base + i * sfa_tmem_cols for i in range(na)]
    sfb_col_bases = [sf_region_base + na * sfa_tmem_cols + j * sfb_tmem_cols for j in range(nb)]
    # tcgen05.alloc requires a power-of-2 column count; allocate the full TMEM.
    used_cols = sf_region_base + sf_total_cols
    num_tmem_alloc_cols = total_tmem
    if used_cols > num_tmem_alloc_cols:
        raise NotImplementedError(
            f"block-scale {cfg.name!r}: the accumulator + SF regions need {used_cols} TMEM columns but only {num_tmem_alloc_cols} are allocated"
        )

    if is_sm103:
        for _label, _bases, _spans in (("SFA", sfa_col_bases, sfa_spans), ("SFB", sfb_col_bases, sfb_spans)):
            _end = max(b + off + cols for b in _bases for off, cols in _spans)
            if _end > num_tmem_alloc_cols:
                raise NotImplementedError(
                    f"block-scale {cfg.name!r}: the hardware {_label} TMEM span reaches "
                    f"column {_end} but only {num_tmem_alloc_cols} are allocated (the MMA "
                    f"would fault OOR_ADDR)"
                )

    # --- AB SMEM pipeline depth ----------------------------------------------
    from ..tile_config import smem_ab_budget_bytes, smem_ab_stages

    # TMA-store stages output through a fixed SMEM-D buffer; reserve it before
    # sizing the AB pipeline (else SMEM overflows the cap).
    epi_reserved = _smem_d_bytes(cfg, chain) if use_tma_store_epi else _combine_smem_bytes(chain, cfg)
    ab_reserved = epi_reserved
    if chain.has_moe and chain.moe.mode == "gather":
        token_rows, token_sfs = (cta_n, real_nb) if isinstance(chain.moe, MoeSwapAbSpec) else (cta_m, real_na)
        ab_reserved += token_rows * 32 * token_sfs
    if is_sm103:
        # CUTLASS-style sm103 pipeline: an AB stage is ONE 128-B-K chunk (a
        # third of the 384-B K-tile), and SF rides its OWN ring (own warp,
        # own mbars) at 12-SF-per-row group granularity — 4 groups per K-tile
        # at VS16, 2 at VS32 (both 12 SFs/row). Data-only AB stages + a fixed
        # SF ring, instead of one combined per-K-tile stage.
        sf_stages = 6
        a_chunk_bytes = cta_m * 128
        b_chunk_bytes = (cta_n // cta_group) * 128
        sfa_group_bytes = cta_m * 12
        # SFB is loaded FULL per CTA (the pair MMA reads each CTA's own TMEM
        # SFB across the whole pair-N range) — same convention as sm100.
        sfb_group_bytes = cta_n * 12
        sf_ring_bytes = sf_stages * (na * sfa_group_bytes + nb * sfb_group_bytes)
        per_ab_stage = na * a_chunk_bytes + nb * b_chunk_bytes
        ab_stages = smem_ab_stages(per_ab_stage, smem_fixed_reserve=tmpl.smem_fixed_reserve, extra_smem_bytes=ab_reserved + sf_ring_bytes)
        if ab_stages < 3:
            raise NotImplementedError(
                f"block-scale {cfg.name!r}: only {ab_stages} 128-B AB chunk " f"stages fit in SMEM — the sm103 pipeline needs >= 3 (one " f"K-tile in flight)"
            )
        desc_roots = None  # 227 KiB part: no ring root can reach _TCGEN05_DESC_BUILD_ADDR_LIMIT
        budget_stages = ab_stages
    else:
        # Each DISTINCT operand has its own 1024-aligned ring, in template
        # allocation order -- the SF rings FIRST, then A, then B (the order the
        # Tcgen05SmemDesc.build() root guard below relies on). Only real dequant
        # operands allocate an SF ring.
        ring_stage_bytes = [sfa_smem_bytes] * real_na + [sfb_smem_bytes] * real_nb + [cta_m * a_cta_k_bytes] * na + [(cta_n // cta_group) * b_cta_k_bytes] * nb
        ab_stages = smem_ab_stages(sum(ring_stage_bytes), smem_fixed_reserve=tmpl.smem_fixed_reserve, extra_smem_bytes=ab_reserved)
        smem_budget = smem_ab_budget_bytes(tmpl.smem_fixed_reserve)
        while ab_stages:
            # The fixed reserve covers control storage and the aligned start
            # of the first ring. Pad whole rings, not individual stages.
            ring_bytes = 0
            for stage_bytes in ring_stage_bytes:
                ring_bytes = align_up(ring_bytes, 1024) + stage_bytes * ab_stages
            # Dense puts the 1024-aligned D buffer AFTER the SF rings. MoE
            # puts it before them; its payload is a multiple of 1024 bytes
            # and is already covered by ab_reserved. STG needs no final pad.
            if use_tma_store_epi and chain.moe is None:
                ring_bytes = align_up(ring_bytes, 1024)
            if ring_bytes + ab_reserved <= smem_budget:
                break
            ab_stages -= 1
        if ab_stages < 1:
            raise NotImplementedError(f"block-scale {cfg.name!r}: no AB stage fits in SMEM including buffer alignment padding")

        # Every ring ROOT is a `Tcgen05SmemDesc.build()` operand (the template advances
        # per stage / per k-step from it) and build() keeps 14 bits of the address --
        # _TCGEN05_DESC_BUILD_ADDR_LIMIT.  Mirror the template's declaration order
        # (SF rings, then A, then B; MoE stages its TMA-store buffer ahead of them),
        # trim the depth until every root clears the line -- the dense template never
        # needs it at any catalog geometry; the MoE template does on sm107 at 512x128
        # nvfp4 (3 -> 2: a 192 KiB A ring behind its 32 KiB staging buffer) and 256x128
        # 2-CTA mxfp8 (7 -> 6) -- then refuse anything a single stage still cannot fit.
        def _desc_roots(stages: int) -> dict[str, int]:
            """Mirror shared-memory allocation order for descriptor validation."""
            return _block_scale_smem_desc_roots(
                header_bytes=tmpl.smem_fixed_reserve
                + (epi_reserved if tmpl.file in _SMEM_D_BEFORE_RINGS_TEMPLATES or _use_swap_bulk_combine(chain, cfg) else 0),
                rings=[
                    *((f"SFA[{i}]", sfa_smem_bytes * stages) for i in range(real_na)),
                    *((f"SFB[{j}]", sfb_smem_bytes * stages) for j in range(real_nb)),
                    *((f"A[{i}]", cta_m * a_cta_k_bytes * stages) for i in range(na)),
                    *((f"B[{j}]", (cta_n // cta_group) * b_cta_k_bytes * stages) for j in range(nb)),
                ],
            )

        budget_stages = ab_stages
        desc_roots = _desc_roots(ab_stages)
        while ab_stages > 1 and max(desc_roots.values()) >= _TCGEN05_DESC_BUILD_ADDR_LIMIT:
            ab_stages -= 1
            desc_roots = _desc_roots(ab_stages)
        if ab_stages != budget_stages:
            _LOG.debug(
                "block-scale %r: ab_stages %d -> %d so every Tcgen05SmemDesc.build() root stays below %d B (roots %s)",
                cfg.name,
                budget_stages,
                ab_stages,
                _TCGEN05_DESC_BUILD_ADDR_LIMIT,
                desc_roots,
            )
        _check_block_scale_desc_roots(cfg, ab_stages, desc_roots)

    out_dt = chain.output_dtype
    epi_store_dt = _epi_store_dtype(chain, cfg)
    vec_bytes_epi = _epi_vec_bytes(chain, cfg)

    # Instruction-descriptor operand dtype. On sm100 the fp4 MMA rides
    # Tcgen05MxInstrDesc with the E5M2 piggy-back; the K=64B fp4 MMA is an
    # OMMA and takes the real fp4 dtype. fp8 always uses its real dtype.
    if both_fp4 and not mma_k64:
        idesc_a = idesc_b = "cutlass.Float8E5M2"
    elif both_fp4:
        idesc_a = idesc_b = "cutlass.Float4E2M1FN"
    else:
        idesc_a = DTYPE_TO_CUTLASS[bs.a_dtype]
        idesc_b = DTYPE_TO_CUTLASS[bs.b_dtype]

    def _tma_side(dtype):
        is_fp4 = dtype == "fp4_e2m1"
        if is_fp4 and padded_fp4:
            return DTYPE_TO_CUTLASS[dtype], "_tma.TensorMapDataFormat.B4X16_P64"
        return (
            "cutlass.Float4E2M1FN" if is_fp4 else DTYPE_TO_CUTLASS[dtype],
            "_tma.TensorMapDataFormat.B4X16" if is_fp4 else "None",
        )

    a_tma_desc_dtype, a_tma_format = _tma_side(bs.a_dtype)
    b_tma_desc_dtype, b_tma_format = _tma_side(bs.b_dtype)

    def _swizzle_side(row_bytes):
        if row_bytes % 128 == 0:
            return "s128b", "SWIZZLE_128B"
        if row_bytes % 64 == 0:
            return "s64b", "SWIZZLE_64B"
        if row_bytes % 32 == 0:
            return "s32b", "SWIZZLE_32B"
        return "none", "NONE"

    a_tma_swizzle, a_smem_swizzle = _swizzle_side(a_cta_k_bytes)
    b_tma_swizzle, b_smem_swizzle = _swizzle_side(b_cta_k_bytes)

    lines = [
        f"# Block-scale config: {cfg.name} data={bs.a_dtype}x{bs.b_dtype} sf={bs.sf_dtype} block={bs.block_size}",
        f"cta_group = {cta_group}",
        f"cta_tile_mnk = ({cta_m}, {cta_n // cta_group}, {cta_k_elems})",
        # MMA instructions the WARP tile spans per axis -- the kernel's loop
        # counts, read off the config instead of recomputed from the tiles.
        f"mma_size_m = {cfg.mma_size_m}",
        f"mma_size_n = {cfg.mma_size_n}",
        f"mma_size_k = {cfg.mma_size_k}",
        f"cgrp_tile_mnk = ({cta_m * cfg.cga_size_m}, {cta_n * cfg.cga_size_n}, {cta_k_elems})",
        f"cgrp_tile_m = {cta_m * cfg.cga_size_m}",
        f"cgrp_tile_n = {cta_n * cfg.cga_size_n}",
        f"epi_tile_mn = {(cfg.epi_tile_m, _epi_n_for_chain(cfg, chain))}",
        f"threads_per_cta = {cfg.threads_per_cta}",
        f"cluster_shape_mnk = {cfg.cluster_shape}",
        f"matmul_a_batch = {chain.matmul.a_batch}",
        f"matmul_b_batch = {chain.matmul.b_batch}",
        f"ab_stages = {ab_stages}",
        *(
            [
                f"# tcgen05 SMEM descriptor roots (bytes, declaration order; Tcgen05SmemDesc.build() keeps 14 bits of addr>>4, "
                f"so each must stay < {_TCGEN05_DESC_BUILD_ADDR_LIMIT}): " + " ".join(f"{name}={off}" for name, off in desc_roots.items())
            ]
            if desc_roots is not None
            else []
        ),
        *(
            [
                f"# ab_stages trimmed from {budget_stages} to {ab_stages}: the deeper ring put a Tcgen05SmemDesc.build() root at or above {_TCGEN05_DESC_BUILD_ADDR_LIMIT} B"
            ]
            if ab_stages != budget_stages
            else []
        ),
        f"acc_stages = {acc_stages}",
        f"use_acc_overlap = {use_acc_overlap}",
        f"acc_stage_stride = {acc_stage_stride}",
        f"acc_overlap_subtiles = {acc_overlap_subtiles}",
        # Multi-GEMM (parallel block-scale matmuls sharing the epilogue). Always
        # emitted; single-GEMM = (1,1,1). Each GEMM owns cta_n acc cols/stage;
        # each distinct operand owns one SF word.
        f"num_gemms = {num_gemms}",
        f"num_a_operands = {na}",
        f"num_b_operands = {nb}",
        f"num_sfa_operands = {real_na}",
        f"num_sfb_operands = {real_nb}",
        f"fake_dequant_a = {bs.fake_dequant_a}",
        f"fake_dequant_b = {bs.fake_dequant_b}",
        f"gemm_a_idx = {tuple(a for a, _ in chain.gemm_operands)}",
        f"gemm_b_idx = {tuple(b for _, b in chain.gemm_operands)}",
        f"acc_region_cols = {acc_region_cols}",
        # GEMM g's acc lives at base + g*acc_gemm_stride (disjoint regions; the 2
        # tile-stage buffers overlap WITHIN a region).
        f"acc_gemm_stride = {acc_gemm_stride}",
        f"sfa_col_bases = {tuple(sfa_col_bases)}",
        f"sfb_col_bases = {tuple(sfb_col_bases)}",
        f"sfa_tmem_cols = {sfa_tmem_cols}",
        f"sfb_tmem_cols = {sfb_tmem_cols}",
        # Mixed CGA pins the walk to the identity map: the super-block
        # rasterization is not invariant across the two cluster shapes.
        f"tile_swizzle_n = {1 if fallback_cluster is not None else 0}",
        f"swizzle_l2_budget_bytes = {_l2_swizzle_budget_bytes()}",
        # Read off the PREFERRED cluster; a fallback cluster is a divisor of it,
        # so these flags dominate and the multicast code path degenerates to the
        # plain load when the runtime pattern names a single peer.
        f"multicast_a = {cfg.multicast_a}",
        f"multicast_b = {cfg.multicast_b}",
        f"a_mcast_slices = {bs_a_mcast_slices}",
        f"a_tma_box_m = {_a_tma_box_m(cfg, bs_a_mcast_slices)}",
        f"b_mcast_slices = {bs_b_mcast_slices}",
        f"ab_empty_full_mask = {bs_ab_empty_full_mask}",
        "",
        f"# packed data SMEM",
        f"a_dtype = {DTYPE_TO_CUTLASS[bs.a_dtype]}",
        f"b_dtype = {DTYPE_TO_CUTLASS[bs.b_dtype]}",
        f"a_smem_dtype = {'cutlass.Uint8' if padded_fp4 and a_data_elem_bits == 4 else DTYPE_TO_CUTLASS[bs.a_dtype]}",
        f"b_smem_dtype = {'cutlass.Uint8' if padded_fp4 and b_data_elem_bits == 4 else DTYPE_TO_CUTLASS[bs.b_dtype]}",
        f"ab_max_data_bits = {data_elem_bits}",
        # Fake-tensor dtypes: A and B may differ (mxfp8 e4m3xe5m2); NOT idesc_a/b (fp4 forces E5M2).
        f"a_fake_dtype = {DTYPE_TO_CUTLASS[bs.a_dtype]}",
        f"b_fake_dtype = {DTYPE_TO_CUTLASS[bs.b_dtype]}",
        f"a_packed_per_row = {a_packed_per_row}",
        f"b_packed_per_row = {b_packed_per_row}",
        f"sA_packed_elems = {sA_packed_elems}",
        f"sB_packed_elems = {sB_packed_elems}",
        # TMA completion counts GMEM payload bytes, not the padded SMEM
        # footprint.  They differ for B4X16_P64: 128 logical FP4 values read
        # 64 bytes from GMEM but occupy 128 bytes in SMEM.
        f"sA_tma_bytes = {cta_m * cta_k_elems * a_data_elem_bits // 8}",
        f"sB_tma_bytes = {(cta_n // cta_group) * cta_k_elems * b_data_elem_bits // 8}",
        # Native FP4 TMA uses logical-width Float4E2M1FN + B4X16.  Padded FP4
        # uses packed-pair Float4E2M1FNx2 + B4X16_P64, whose builder derives
        # logical FP4 strides while expanding the SMEM destination. FP8 uses
        # its ordinary data dtype/format.
        f"a_tma_desc_dtype = {a_tma_desc_dtype}",
        f"b_tma_desc_dtype = {b_tma_desc_dtype}",
        f"a_tma_format = {a_tma_format}",
        f"b_tma_format = {b_tma_format}",
        f"a_tma_swizzle = _tma.TensorMapSwizzle.{a_tma_swizzle}",
        f"b_tma_swizzle = _tma.TensorMapSwizzle.{b_tma_swizzle}",
        f"a_smem_swizzle = cutlass.experimental.primitives.Tcgen05SmemSwizzle.{a_smem_swizzle}",
        f"b_smem_swizzle = cutlass.experimental.primitives.Tcgen05SmemSwizzle.{b_smem_swizzle}",
        f"a_smem_desc_leading_byte_offset = {a_lbo}",
        f"a_smem_desc_stride_byte_offset = {a_sbo}",
        f"a_smem_k_step_bytes = {a_k_step}",
        f"a_tma_group_elems = {a_tma_group_elems}",
        f"b_smem_desc_leading_byte_offset = {b_lbo}",
        f"b_smem_desc_stride_byte_offset = {b_sbo}",
        f"b_smem_k_step_bytes = {b_k_step}",
        f"b_tma_group_elems = {b_tma_group_elems}",
        # A/B operand major. FP4 K-major only (rejected above); mxfp8 may be
        # M-major (A) / N-major (B). SF layout is unchanged regardless.
        f"a_is_m_major = {a_major == 'm'}",
        f"b_is_n_major = {b_major == 'n'}",
        # MMA-idesc major flags (1 = MN-major operand) so tcgen05 reads the SMEM
        # matrix descriptor with the right operand orientation.
        f"mma_a_major = {1 if a_major == 'm' else 0}",
        f"mma_b_major = {1 if b_major == 'n' else 0}",
        "",
        f"# output",
        f"cd_dtype = {'cutlass.Int8' if out_dt == 'fp4_e2m1' else DTYPE_TO_CUTLASS[out_dt]}",
        f"epi_store_dtype = {DTYPE_TO_CUTLASS[epi_store_dt]}",
        f"vec_bytes_epi = {vec_bytes_epi}",
        f"split_k_slices = {cfg.split_k_slices}",
        *([f"splitk_reduce_elems = {_splitk_reduce_elems(chain)}"] if cfg.split_k_slices > 1 else []),
        f"frost_compile_options = {_frost_compile_options()!r}",
        # The COUNT, not the flag: a block-scale graph can put more than one
        # output on the TMA-C surface, and on MoE this number also sizes the
        # tensormap scratch and the per-CTA workspace stride.
        f"n_tma_outputs = {1 if (cfg.split_k_slices > 1 and use_tma_store_epi) else len(_tma_slots_for(chain, cfg))}",
        f"moe_aligned_offsets = {_moe_aligned_offsets(chain, cfg)}",
        f"moe_token_alignment = {_pow2_floor(math.gcd(chain.moe.offset_multiple, chain.matmul.N)) if isinstance(chain.moe, MoeSwapAbSpec) else 1}",
        f"epi_slot_widen = {_epi_slot_widen(chain, cfg)}",
        f"epi_stage_rows = {_epi_stage_rows(cfg)}",
        f"epi_chunk_elems = {_epi_chunk_elems(chain, cfg, use_tma_store_epi)}",
        *_epi_swizzle_lines(cfg, epi_store_dt, chain),
        "",
        f"# block-scale MMA",
        f"mma_block_scale_kind = nvvm.MMABlockScaleKind.{bs.mma_block_scale_kind}",
        f"scale_vec_size = nvvm.Tcgen05MMABlockScale.{bs.scale_vec_size}",
        f"idesc_a_dtype = {idesc_a}",
        f"idesc_b_dtype = {idesc_b}",
        f"sf_scale_format = {bs.sf_scale_format}",
        # Four packed copies of the scale format's identity byte. E4M3 uses
        # bias 7 (0x38), E8M0 bias 127 (0x7f), and unsigned E5M3 bias 15
        # (0x78).  The MMA consumes these as opaque scale bytes from TMEM.
        f"sf_one_word = { {'fp8_e4m3': 0x38383838, 'fp8_e8m0': 0x7F7F7F7F, 'fp8_e5m3': 0x78787878}[bs.sf_dtype] }",
        # The idesc M/N are the INSTRUCTION's, not the CTA tile's. They agreed while
        # one instruction covered the tile; with mma_size_m > 1 a CTA-tile M of 256 would
        # encode the M=256 enum on a 1-CTA MMA, which does not exist -> illegal instruction.
        f"mma_m_dim = {cfg.mma_tile_m * cta_group}",
        f"mma_n_dim = {cfg.mma_tile_n}",
        "",
        f"# scale factors",
        f"block_size = {bs.block_size}",
        f"sf_cutlass_dtype = {DTYPE_TO_CUTLASS[bs.sf_dtype]}",
        f"sf_scales_per_inst = {scales_per_inst}",
        f"sf_insts_per_atom = {insts_per_word}",
        f"num_sf_atoms = {num_sf_words}",
        f"word_atoms = {word_atoms}",
        f"num_blocks_m = {nb_m}",
        f"num_blocks_n = {nb_n}",
        f"registers_per_block = {_REGISTERS_PER_BLOCK}",
        f"epi_cols_per_mma_m = {epi_cols_per_mma_m}",
        f"mma_c_dtype = {DTYPE_TO_CUTLASS[chain.matmul.accum_dtype]}",
        # Byte step from one MMA M sub-block to the next inside the SMEM tile.
        # sm103 stages ONE 128-B K chunk per AB stage, not the whole K-tile, so
        # its per-M-row width is the chunk's, not cta_tile_k_bytes.
        f"a_smem_m_step_bytes = {(cta_m // mma_size_m) * (128 if is_sm103 else a_cta_k_bytes)}",
        f"registers_per_atom = {_REGISTERS_PER_ATOM}",
        f"sf_atom_desc_stride = {sf_atom_desc_stride}",
        f"sf_block_desc_stride = {sf_block_desc_stride}",
        f"num_tmem_alloc_cols = {num_tmem_alloc_cols}",
        f"tmem_alloc_exclusive = {num_tmem_alloc_cols > _MAX_NON_EXCLUSIVE_TMEM_COLS}",
        f"b_collector_ok = {_b_collector_ok(cfg)}",
        f"sfa_smem_bytes = {sfa_smem_bytes}",
        f"sfb_smem_bytes = {sfb_smem_bytes}",
        f"sf_tma_box_k = {sf_tma_box_k}",
        f"sfa_tma_box_mn = {nb_m}",
        f"sfb_tma_box_mn = {nb_n}",
    ]
    if is_sm103:
        kstep = mma_inst_k_bytes
        spi = scales_per_inst
        sf_groups = 4 if bs.block_size == 16 else 2  # 12 SFs/row per group either way
        lines += [
            "",
            # The sm103 template is homogeneous FP4-only and still uses the
            # shared AB names.  Keep those aliases local to that pipeline; the
            # sm100 templates intentionally consume the side-specific names so
            # a mixed-width operand cannot accidentally inherit its peer's
            # packing, descriptor dtype, or swizzle.
            "ab_dtype = a_dtype",
            "ab_packed_per_row = a_packed_per_row",
            "ab_tma_desc_dtype = a_tma_desc_dtype",
            "ab_tma_format = a_tma_format",
            "ab_tma_swizzle = a_tma_swizzle",
            "ab_smem_swizzle = a_smem_swizzle",
            "",
            f"# sm103 K=48B UTCOMMA pipeline: {num_kblocks} MMAs per K-tile; an AB " f"stage is one 128-B chunk (3 per K-tile); SF rides its own ring",
            f"chunks_per_ktile = {cfg.cta_tile_k_bytes // 128}",
            f"ab_tma_box_k_elems = {128 * 8 // data_elem_bits}",
            f"a_chunk_packed_elems = {cta_m * 128}",
            f"b_chunk_packed_elems = {(cta_n // cta_group) * 128}",
            f"sf_stages = {sf_stages}",
            f"sf_groups_per_ktile = {sf_groups}",
            f"mmas_per_sf_group = {num_kblocks // sf_groups}",
            f"sf_atoms_per_group = {sf_k4 // sf_groups}",
            f"sfa_group_bytes = {sfa_group_bytes}",
            f"sfb_group_bytes = {sfb_group_bytes}",
            f"sf_group_block_desc_stride = {32 * (sf_k4 // sf_groups)}",
            f"num_kblocks = {num_kblocks}",
            f"mma_chunk_by_j = {tuple((kstep * j) // 128 for j in range(num_kblocks))}",
            f"mma_next_chunk_by_j = {tuple((kstep * j + kstep - 1) // 128 for j in range(num_kblocks))}",
            f"mma_phase16_by_j = {tuple((kstep * j) % 128 // 16 for j in range(num_kblocks))}",
            f"sf_id_by_j = {tuple(spi * j % 4 for j in range(num_kblocks))}",
            f"sfa_mma_col_off_by_j = {tuple(spi * j // 4 * 4 * sfa_nb_m for j in range(num_kblocks))}",
            f"sfb_mma_col_off_by_j = {tuple(spi * j // 4 * 4 * mma_nb_n for j in range(num_kblocks))}",
        ]
    # Instruction-descriptor K mode. The MX descriptor's k_dim is 0 for a 32-byte
    # K and 1 for 64; fp4 at 64 leaves it for the OMMA descriptor, whose k_dim 2
    # means 128 fp4 elements. At 32 fp4 rides the MX descriptor's E5M2
    # piggy-back, so it is NOT an OMMA there.
    lines += [
        "",
        f"# block-scale MMA: {num_kblocks} MMAs per K-tile at mma_inst_k_bytes={cfg.mma_tile_k_bytes}",
        f"idesc_is_omma = {both_fp4 and mma_k64}",
        f"mma_k_dim_mode = {(2 if both_fp4 else 1) if mma_k64 else 0}",
    ]
    # MoE grouped block-scale: grouped persistent scheduler launches a FIXED
    # cluster count (≈ NUM_SMS / cluster_size); host grid and stride share it.
    # first_token_offset dtype (int32/int64) drives the compile() fake.
    if chain.has_moe:
        lines.append(f"grid_num_clusters = {_grid_num_clusters(cfg)}")
        lines.append(f"offset_cutlass_dtype = {DTYPE_TO_CUTLASS[chain.moe.offset_dtype]}")
        # A's per-row byte stride = k * ab_data_elem_bits / 8 (FP4: k/2 bytes);
        # routed-group base offset = group_begin rows × this. NOT ab_dtype.width
        # (that is the packed Float4E2M1FNx2 8-bit type).
    lines.extend(_mixed_cga_constants(cfg, fallback_cluster))
    lines.extend(_quant_device_imports(chain))
    return "\n".join(lines)


_CTAMMA_SUFFIX = re.compile(r"_[12]ctamma$")


def _template_stem(file: str) -> str:
    """Template stem with any ``_<N>ctamma`` token removed -- the kernel symbol
    spells cta_group itself, so a template that still encodes it in its filename
    must not spell it twice."""
    return _CTAMMA_SUFFIX.sub("", file.removesuffix(".py"))


def _resolve_path_blocks(src: str, use_tma_store_epi: bool) -> str:
    """Strip the dead epilogue-mode block from the template before markers fill.

    `cutlass.const_expr` picks a branch's IR, but cute type-checks BOTH branches
    at parse time. The TMA vs STG paths bind ``vec_f32`` to different shapes, so
    leaving both would trip cute's type-consistency check; stripping the dead
    branch avoids it. Block syntax: `# @@{TMA_STORE,STG}_ONLY:BEGIN@@ ... :END@@`
    (one per pair, no nesting)."""
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


_INJECT_MARKER_LINE = re.compile(r"^([ \t]*)# *@@([A-Z0-9_]+)@@[ \t]*\n", flags=re.MULTILINE)


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


def _render_template(
    chain: FusionChain,
    snippets: EpilogueSnippets,
    config: TileConfig,
) -> str:
    # Template selected by the kernel registry from the pure-geometry config +
    # execution strategy (cta_group); mainloop/graph_type from chain.
    from ..kernel_registry import select_template

    tmpl = select_template(chain, config)
    template_path = tmpl.path
    src = template_path.read_text()
    # Strip the unused epilogue path FIRST so its @@INJECT_EPILOGUE@@ marker
    # doesn't survive into the marker-replacement step.
    store_modes = _store_modes(chain, config)
    tma_slots = frozenset(i for i, m in enumerate(store_modes) if m == "tma")
    use_tma = _use_tma_store_epi(chain, config)
    src = _resolve_path_blocks(src, use_tma)
    # Split-K blocks
    splitk = config.split_k_slices > 1
    if "@@SPLITK_ONLY:BEGIN@@" in src:
        src = _resolve_blocks(src, "SPLITK_ONLY", splitk)
    elif splitk:
        raise NotImplementedError(f"split_k_slices={config.split_k_slices}: template {tmpl.file} has no split-K support")
    # For splitK, `snippets` is is kernel 1's partials store. Re-generate epilogue for reduce kernel
    plumb = _splitk_reduce_snippets(chain, config) if splitk else snippets

    aux_tensors = chain.aux_tensors

    # ---- Multi-GEMM A/B operand plumbing (1ctamma only) ------------------
    # One TMA descriptor + SMEM buffer + runtime tensor per DISTINCT operand.
    # Single-GEMM = 1 A + 1 B (suffix _0) → length-1 loops == legacy behavior.
    na, nb = chain.num_a_operands, chain.num_b_operands
    kernel_ab_desc_params = (
        ",\n".join(
            [f"tma_a_desc_{i}: cutlass.GridConstant[_tma.TensorMap]" for i in range(na)]
            + [f"tma_b_desc_{j}: cutlass.GridConstant[_tma.TensorMap]" for j in range(nb)]
        )
        + ","
    )
    ab_desc_lists = (
        "tma_a_descs = [" + ", ".join(f"tma_a_desc_{i}" for i in range(na)) + "]\n" "tma_b_descs = [" + ", ".join(f"tma_b_desc_{j}" for j in range(nb)) + "]"
    )
    host_ab_params = ",\n".join([f"a_{i}: cute.Tensor" for i in range(na)] + [f"b_{j}: cute.Tensor" for j in range(nb)]) + ","
    host_ab_lists = "_a_operands = [" + ", ".join(f"a_{i}" for i in range(na)) + "]\n" "_b_operands = [" + ", ".join(f"b_{j}" for j in range(nb)) + "]"
    host_kernel_desc_pass = ",\n".join([f"tma_a_desc_list[{i}]" for i in range(na)] + [f"tma_b_desc_list[{j}]" for j in range(nb)]) + ","
    compile_ab_fakes = "\n".join([f"fake_a_{i} = _make_fake_a()" for i in range(na)] + [f"fake_b_{j} = _make_fake_b()" for j in range(nb)])
    compile_ab_pass = ",\n".join([f"fake_a_{i}" for i in range(na)] + [f"fake_b_{j}" for j in range(nb)]) + ","
    # Per-GEMM register-vector bindings in the STG inner loop: GEMM 0 is bound by
    # the template (vec_f32); the rest (vec_f32_1, ...) are injected here.
    stg_vec_bindings = "\n".join(f"vec_f32_{g} = c_rmem_vecs[{g}][j * vsize : (j + 1) * vsize]" for g in range(1, chain.num_gemms)) or "pass"
    # MoE multi-GEMM: the kernel also takes the raw A (token) tensor per distinct
    # A operand (for the per-group patched base address) — same tensors the host
    # uses to build the A descriptors.
    moe_kernel_ma_params = ",\n".join([f"mA_{i}: cute.Tensor" for i in range(na)] + [f"a_stride_m_{i}: cutlass.Int64" for i in range(na)])
    if moe_kernel_ma_params:
        moe_kernel_ma_params += ","
    moe_ma_list = "mA_list = [" + ", ".join(f"mA_{i}" for i in range(na)) + "]\n" "a_stride_m_list = [" + ", ".join(f"a_stride_m_{i}" for i in range(na)) + "]"
    moe_host_ma_pass = ",\n".join([f"a_{i}" for i in range(na)] + [f"_a_stride_sets[{i}][0]" for i in range(na)])
    if moe_host_ma_pass:
        moe_host_ma_pass += ","

    if isinstance(chain.moe, MoeSwapAbSpec):
        moe_kernel_ma_params = ",\n".join([f"mB_{i}: cute.Tensor" for i in range(nb)] + [f"b_stride_n_{i}: cutlass.Int64" for i in range(nb)]) + ","
        moe_ma_list = (
            "mB_list = [" + ", ".join(f"mB_{i}" for i in range(nb)) + "]\n" + "b_stride_n_list = [" + ", ".join(f"b_stride_n_{i}" for i in range(nb)) + "]"
        )
        moe_host_ma_pass = ",\n".join([f"b_{i}" for i in range(nb)] + [f"_b_stride_sets[{i}][0]" for i in range(nb)]) + ","

    if chain.has_moe and chain.moe.mode in ("gather", "scatter", "combine"):
        routed_rows = "sym_n" if isinstance(chain.moe, MoeSwapAbSpec) else "sym_m"
        host_ab_params = "token_index: cute.Tensor,\n" + host_ab_params
        compile_ab_fakes += f"\nfake_token_index = make_fake_compact_tensor(cutlass.Int32, ({routed_rows},), stride_order=(0,), assumed_align=4)"
        compile_ab_pass = "fake_token_index,\n" + compile_ab_pass
        moe_kernel_ma_params += "\ntoken_index: cute.Tensor,"
        moe_host_ma_pass += "\ntoken_index,"
        if chain.moe.mode in ("scatter", "combine"):
            host_ab_params = "token_ks: cute.Tensor,\n" + host_ab_params
            compile_ab_fakes += f"\nfake_token_ks = make_fake_compact_tensor(cutlass.Int32, ({routed_rows},), stride_order=(0,), assumed_align=4)"
            compile_ab_pass = "fake_token_ks,\n" + compile_ab_pass
            moe_kernel_ma_params += "\ntoken_ks: cute.Tensor,"
            moe_host_ma_pass += "\ntoken_ks,"
        if chain.moe.mode == "combine":
            host_ab_params = "top_k_scores: cute.Tensor,\n" + host_ab_params
            compile_ab_fakes += (
                f"\nfake_top_k_scores = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(), {chain.moe.top_k}), stride_order=(1, 0), assumed_align=4)"
            )
            compile_ab_pass = "fake_top_k_scores,\n" + compile_ab_pass
            moe_kernel_ma_params += "\ntop_k_scores: cute.Tensor,"
            moe_host_ma_pass += "\ntop_k_scores,"

    # Indentation matches the marker's column in the template (8 spaces inside
    # _kernel/_host signatures, 4 inside compile() body).
    kernel_aux_params = _aux_signature_block(aux_tensors)
    host_aux_params = _aux_signature_block(aux_tensors)
    host_aux_pass = _aux_call_block(aux_tensors)
    compile_aux_fakes = _aux_fake_block(
        aux_tensors,
        dynamic_strides=True,
        align_reqs=_aux_align_reqs(chain, vec_bytes=_out_vec_bytes(chain, config, use_tma)),
    )
    compile_aux_pass = _aux_call_block(aux_tensors, prefix="fake_")
    tile_constants = _render_tile_constants(config, chain, tmpl)
    if plumb.tap_constants:
        tile_constants += "\n" + "\n".join(plumb.tap_constants)
    # Multi-output tap plumbing. Empty lists → markers expand to nothing (kernel
    # signature shrinks back to single-output form).
    kernel_tap_params = ",\n".join(plumb.tap_kernel_params)
    if kernel_tap_params:
        kernel_tap_params += ","
    host_tap_params = ",\n".join(plumb.tap_host_params)
    if host_tap_params:
        host_tap_params += ","
    host_tap_pass = ",\n".join(plumb.tap_host_pass)
    if host_tap_pass:
        host_tap_pass += ","
    compile_tap_fakes = "\n".join(plumb.tap_compile_fakes)
    compile_tap_pass = ",\n".join(plumb.tap_compile_pass)
    if compile_tap_pass:
        compile_tap_pass += ","
    tap_ptr_binds = "\n".join(plumb.tap_ptr_binds) if (plumb.tap_ptr_binds and not splitk) else "pass"
    red_kernel_stride_params = _reduction_stride_kernel_params(chain)
    red_host_stride_unpack = _reduction_stride_host_unpack(chain)
    red_host_stride_pass = _reduction_stride_host_pass(chain)
    red_compile_stride_decls = _reduction_stride_compile_decls(chain)
    red_compile_stride_symbols = _reduction_stride_compile_symbols(chain)

    replacements = {
        "INJECT_TILE_CONSTANTS": tile_constants,
        "INJECT_KERNEL_AUX_PARAMS": kernel_aux_params,
        "INJECT_HOST_AUX_PARAMS": host_aux_params,
        "INJECT_HOST_AUX_PASS": host_aux_pass,
        "INJECT_COMPILE_AUX_FAKES": compile_aux_fakes,
        "INJECT_COMPILE_AUX_PASS": compile_aux_pass,
        "INJECT_KERNEL_TAP_PARAMS": kernel_tap_params,
        "INJECT_HOST_TAP_PARAMS": host_tap_params,
        "INJECT_HOST_TAP_PASS": host_tap_pass,
        "INJECT_COMPILE_TAP_FAKES": compile_tap_fakes,
        "INJECT_COMPILE_TAP_PASS": compile_tap_pass,
        "INJECT_TAP_PTRS": tap_ptr_binds,
        # Under split-K kernel 1 only stores partials: no aux reads, and the
        # graph's epilogue runs in kernel 2 instead.
        "INJECT_AUX_VIEWS": "pass" if splitk else snippets.aux_views,
        "INJECT_EPILOGUE": snippets.epilogue,
    }
    if splitk:
        replacements["INJECT_SPLITK_EPILOGUE_BINDINGS"] = _splitk_epilogue_bindings(chain, len(plumb.tap_kernel_params))
        replacements["INJECT_REDUCE_AUX_VIEWS"] = plumb.aux_views
        replacements["INJECT_REDUCE_EPILOGUE"] = plumb.epilogue
    for marker, replacement in (
        ("INJECT_KERNEL_REDUCTION_STRIDE_PARAMS", red_kernel_stride_params),
        ("INJECT_HOST_REDUCTION_STRIDES", red_host_stride_unpack),
        ("INJECT_HOST_REDUCTION_STRIDE_PASS", red_host_stride_pass),
        ("INJECT_COMPILE_REDUCTION_STRIDE_DECLS", red_compile_stride_decls),
        ("INJECT_COMPILE_REDUCTION_STRIDE_SYMBOLS", red_compile_stride_symbols),
    ):
        if f"@@{marker}@@" in src:
            replacements[marker] = replacement
        elif chain.reductions or chain.quants:
            raise RuntimeError(f"template missing marker @@{marker}@@")
    # Multi-GEMM A/B operand plumbing — filled only when the template carries
    # the markers (single-GEMM renders get length-1 operand lists).
    if "@@INJECT_KERNEL_AB_DESC_PARAMS@@" in src:
        replacements.update(
            {
                "INJECT_KERNEL_AB_DESC_PARAMS": kernel_ab_desc_params,
                "INJECT_AB_DESC_LISTS": ab_desc_lists,
                "INJECT_HOST_AB_PARAMS": host_ab_params,
                "INJECT_HOST_AB_LISTS": host_ab_lists,
                "INJECT_HOST_KERNEL_DESC_PASS": host_kernel_desc_pass,
                "INJECT_COMPILE_AB_FAKES": compile_ab_fakes,
                "INJECT_COMPILE_AB_PASS": compile_ab_pass,
            }
        )
    if "@@INJECT_KERNEL_TMA_C_PARAMS@@" in src:
        replacements.update(_tma_c_plumbing(chain, tma_slots, splitk=splitk))
    if "@@INJECT_TMA_STORE_SEQUENCE@@" in src:
        _epi = _epi_n_for_chain(config, chain)
        replacements["INJECT_EPILOGUE"], replacements["INJECT_TMA_STORE_SEQUENCE"] = _place_tma_stores(snippets.epilogue, chain, config, tma_slots, _epi)
        replacements["INJECT_HOST_TMA_C_DESCS"] = _host_tma_c_descs(chain, config, tma_slots, _epi)
    # Per-GEMM STG vector bindings — on every STG-epilogue template (mainloop
    # included; single-GEMM → `pass`).
    if "@@INJECT_STG_VEC_BINDINGS@@" in src:
        replacements["INJECT_STG_VEC_BINDINGS"] = stg_vec_bindings
    # MoE raw-A-tensor plumbing — only MoE grouped-matmul templates carry these
    # (the kernel patches each A descriptor per routed group).
    if "@@INJECT_MOE_KERNEL_MA_PARAMS@@" in src:
        replacements.update(
            {
                "INJECT_MOE_KERNEL_MA_PARAMS": moe_kernel_ma_params,
                "INJECT_MOE_MA_LIST": moe_ma_list,
                "INJECT_MOE_HOST_MA_PASS": moe_host_ma_pass,
            }
        )
    # Mainloop-fusion transforms — only the 12-warp templates carry these.
    if chain.has_mainloop_fusion:
        replacements.update(
            {
                "INJECT_MAINLOOP_A": snippets.mainloop_transform_a,
                "INJECT_MAINLOOP_B": snippets.mainloop_transform_b,
            }
        )

    src = _replace_marker_lines(src, replacements)

    # Tag the kernel fn name with template + geometry so nsys gives each
    # (template, config) a distinct GPU kernel symbol.
    tag = re.sub(r"[^A-Za-z0-9_]", "_", f"{_template_stem(tmpl.file)}_{config.geometry_name}{_store_mode_tag(store_modes)}")
    src = re.sub(r"\b_kernel(?=\(|\.set_name_prefix\b)", f"frost_{tag}", src)

    return src


def _render_block_scale_template(
    chain: FusionChain,
    snippets: EpilogueSnippets,
    config: TileConfig,
) -> str:
    """Render the block-scale matmul template. Picks TMA-store when
    _use_tma_store_epi allows, else STG; SF TMA descriptors are hardcoded in the
    template (not injected). Epilogue aux/tap markers still work."""
    from ..kernel_registry import select_template

    tmpl = select_template(chain, config)
    template_path = tmpl.path
    src = template_path.read_text()
    store_modes = _store_modes(chain, config)
    use_tma = _use_tma_store_epi(chain, config)
    splitk = config.split_k_slices > 1
    tma_slots = frozenset({0}) if (splitk and use_tma) else frozenset(i for i, m in enumerate(store_modes) if m == "tma")
    src = _resolve_path_blocks(src, use_tma_store_epi=use_tma)
    if "@@SPLITK_ONLY:BEGIN@@" in src:
        src = _resolve_blocks(src, "SPLITK_ONLY", splitk)
    elif splitk:
        raise NotImplementedError(f"split_k_slices={config.split_k_slices}: template {tmpl.file} has no split-K support")
    # Under split-K `snippets` is kernel 1's partials store; kernel 2 renders the graph's epilogue.
    plumb = _splitk_reduce_snippets(chain, config) if splitk else snippets

    aux_tensors = chain.aux_tensors
    kernel_aux_params = _aux_signature_block(aux_tensors)
    host_aux_params = _aux_signature_block(aux_tensors)
    host_aux_pass = _aux_call_block(aux_tensors)
    compile_aux_fakes = _aux_fake_block(aux_tensors, dynamic_strides=True, align_reqs=_aux_align_reqs(chain, vec_bytes=_out_vec_bytes(chain, config, use_tma)))
    compile_aux_pass = _aux_call_block(aux_tensors, prefix="fake_")
    tile_constants = _render_block_scale_tile_constants(config, chain, tmpl)
    tile_constants += f"\nmoe_gather = {chain.has_moe and chain.moe.mode == 'gather'}\nmoe_scatter = {chain.has_moe and chain.moe.mode == 'scatter'}\nmoe_top_k = {chain.moe.top_k if chain.has_moe else 1}"
    tile_constants += f"\nmoe_combine = {chain.has_moe and chain.moe.mode == 'combine'}"
    tile_constants += f"\nmoe_combine_bulk_enabled = {_use_bulk_combine(chain, config)}"
    tile_constants += f"\nmoe_combine_fixed_descriptors = {_use_fixed_combine_descriptors(chain, config)}"
    tile_constants += f"\nmoe_combine_swap_bulk_enabled = {_use_swap_bulk_combine(chain, config)}"
    if plumb.tap_constants:
        tile_constants += "\n" + "\n".join(plumb.tap_constants)

    kernel_tap_params = ",\n".join(plumb.tap_kernel_params)
    if kernel_tap_params:
        kernel_tap_params += ","
    host_tap_params = ",\n".join(plumb.tap_host_params)
    if host_tap_params:
        host_tap_params += ","
    host_tap_pass = ",\n".join(plumb.tap_host_pass)
    if host_tap_pass:
        host_tap_pass += ","
    compile_tap_fakes = "\n".join(plumb.tap_compile_fakes)
    compile_tap_pass = ",\n".join(plumb.tap_compile_pass)
    if compile_tap_pass:
        compile_tap_pass += ","
    tap_ptr_binds = "\n".join(plumb.tap_ptr_binds) if (plumb.tap_ptr_binds and not splitk) else "pass"
    red_kernel_stride_params = _reduction_stride_kernel_params(chain)
    red_host_stride_unpack = _reduction_stride_host_unpack(chain) if chain.has_moe else _reduction_stride_host_unpack_from(chain, 10)
    red_host_stride_pass = _reduction_stride_host_pass(chain)
    red_compile_stride_decls = _reduction_stride_compile_decls(chain)
    red_compile_stride_symbols = _reduction_stride_compile_symbols(chain)

    # ---- Multi-GEMM A/B + SF operand plumbing (block-scale) ----------------
    # One (packed data + SF) descriptor pair per DISTINCT operand (SF travels
    # with its data). GROUPED BY KIND — all A, all B, all SFA, all SFB — so
    # single-GEMM (na=nb=1) is exactly (a, b, sfa, sfb), the legacy call order.
    na, nb = chain.num_a_operands, chain.num_b_operands
    bs = chain.block_scale
    assert bs is not None
    nsa = 0 if bs.fake_dequant_a else na
    nsb = 0 if bs.fake_dequant_b else nb
    _G = "cutlass.GridConstant[_tma.TensorMap]"
    kernel_ab_desc_params = (
        ",\n".join(
            [f"tma_a_desc_{i}: {_G}" for i in range(na)]
            + [f"tma_b_desc_{j}: {_G}" for j in range(nb)]
            + [f"tma_sfa_desc_{i}: {_G}" for i in range(nsa)]
            + [f"tma_sfb_desc_{j}: {_G}" for j in range(nsb)]
        )
        + ","
    )
    ab_desc_lists = (
        "tma_a_descs = [" + ", ".join(f"tma_a_desc_{i}" for i in range(na)) + "]\n"
        "tma_b_descs = [" + ", ".join(f"tma_b_desc_{j}" for j in range(nb)) + "]\n"
        "tma_sfa_descs = [" + ", ".join(f"tma_sfa_desc_{i}" for i in range(nsa)) + "]\n"
        "tma_sfb_descs = [" + ", ".join(f"tma_sfb_desc_{j}" for j in range(nsb)) + "]"
    )
    host_ab_params = (
        ",\n".join(
            [f"a_{i}: cute.Tensor" for i in range(na)]
            + [f"b_{j}: cute.Tensor" for j in range(nb)]
            + [f"sfa_{i}: cute.Tensor" for i in range(nsa)]
            + [f"sfb_{j}: cute.Tensor" for j in range(nsb)]
        )
        + ","
    )
    host_ab_lists = (
        "_a_operands = [" + ", ".join(f"a_{i}" for i in range(na)) + "]\n"
        "_b_operands = [" + ", ".join(f"b_{j}" for j in range(nb)) + "]\n"
        "_sfa_operands = [" + ", ".join(f"sfa_{i}" for i in range(nsa)) + "]\n"
        "_sfb_operands = [" + ", ".join(f"sfb_{j}" for j in range(nsb)) + "]"
    )
    host_kernel_desc_pass = (
        ",\n".join(
            [f"tma_a_desc_list[{i}]" for i in range(na)]
            + [f"tma_b_desc_list[{j}]" for j in range(nb)]
            + [f"tma_sfa_desc_list[{i}]" for i in range(nsa)]
            + [f"tma_sfb_desc_list[{j}]" for j in range(nsb)]
        )
        + ","
    )
    compile_ab_fakes = "\n".join(
        [f"fake_a_{i} = _make_fake_a()" for i in range(na)]
        + [f"fake_b_{j} = _make_fake_b()" for j in range(nb)]
        + [f"fake_sfa_{i} = _make_fake_sfa()" for i in range(nsa)]
        + [f"fake_sfb_{j} = _make_fake_sfb()" for j in range(nsb)]
    )
    compile_ab_pass = (
        ",\n".join(
            [f"fake_a_{i}" for i in range(na)]
            + [f"fake_b_{j}" for j in range(nb)]
            + [f"fake_sfa_{i}" for i in range(nsa)]
            + [f"fake_sfb_{j}" for j in range(nsb)]
        )
        + ","
    )
    stg_vec_bindings = "\n".join(f"vec_f32_{g} = c_rmem_vecs[{g}][j * vsize : (j + 1) * vsize]" for g in range(1, chain.num_gemms)) or "pass"
    # MoE block-scale: raw A (token) tensor per distinct A operand for the
    # per-routed-group descriptor patch.
    moe_kernel_ma_params = ",\n".join([f"mA_{i}: cute.Tensor" for i in range(na)] + [f"a_stride_m_{i}: cutlass.Int64" for i in range(na)])
    if moe_kernel_ma_params:
        moe_kernel_ma_params += ","
    moe_ma_list = "mA_list = [" + ", ".join(f"mA_{i}" for i in range(na)) + "]\n" "a_stride_m_list = [" + ", ".join(f"a_stride_m_{i}" for i in range(na)) + "]"
    moe_host_ma_pass = ",\n".join([f"a_{i}" for i in range(na)] + [f"_a_stride_sets[{i}][0]" for i in range(na)])
    if moe_host_ma_pass:
        moe_host_ma_pass += ","
    moe_kernel_msfa_params = ",\n".join(f"mSFA_{i}: cute.Tensor" for i in range(nsa))
    if moe_kernel_msfa_params:
        moe_kernel_msfa_params += ","
    moe_msfa_list = "mSFA_list = [" + ", ".join(f"mSFA_{i}" for i in range(nsa)) + "]"
    moe_host_msfa_pass = ",\n".join(f"_sfa_operands[{i}]" for i in range(nsa))
    if moe_host_msfa_pass:
        moe_host_msfa_pass += ","

    if isinstance(chain.moe, MoeSwapAbSpec):
        moe_kernel_ma_params = ",\n".join([f"mB_{i}: cute.Tensor" for i in range(nb)] + [f"b_stride_n_{i}: cutlass.Int64" for i in range(nb)]) + ","
        moe_ma_list = (
            "mB_list = [" + ", ".join(f"mB_{i}" for i in range(nb)) + "]\n" + "b_stride_n_list = [" + ", ".join(f"b_stride_n_{i}" for i in range(nb)) + "]"
        )
        moe_host_ma_pass = ",\n".join([f"b_{i}" for i in range(nb)] + [f"_b_stride_sets[{i}][0]" for i in range(nb)]) + ","
        moe_kernel_msfa_params = ",\n".join(f"mSFB_{i}: cute.Tensor" for i in range(nsb)) + ("," if nsb else "")
        moe_msfa_list = "mSFB_list = [" + ", ".join(f"mSFB_{i}" for i in range(nsb)) + "]"
        moe_host_msfa_pass = ",\n".join(f"_sfb_operands[{i}]" for i in range(nsb)) + ("," if nsb else "")

    if chain.has_moe and chain.moe.mode in ("gather", "scatter", "combine"):
        routed_rows = "sym_n" if isinstance(chain.moe, MoeSwapAbSpec) else "sym_m"
        host_ab_params = "token_index: cute.Tensor,\n" + host_ab_params
        compile_ab_fakes += f"\nfake_token_index = make_fake_compact_tensor(cutlass.Int32, ({routed_rows},), stride_order=(0,), assumed_align=4)"
        compile_ab_pass = "fake_token_index,\n" + compile_ab_pass
        moe_kernel_ma_params += "\ntoken_index: cute.Tensor,"
        moe_host_ma_pass += "\ntoken_index,"
        if chain.moe.mode in ("scatter", "combine"):
            host_ab_params = "token_ks: cute.Tensor,\n" + host_ab_params
            compile_ab_fakes += f"\nfake_token_ks = make_fake_compact_tensor(cutlass.Int32, ({routed_rows},), stride_order=(0,), assumed_align=4)"
            compile_ab_pass = "fake_token_ks,\n" + compile_ab_pass
            moe_kernel_ma_params += "\ntoken_ks: cute.Tensor,"
            moe_host_ma_pass += "\ntoken_ks,"
        if chain.moe.mode == "combine":
            host_ab_params = "top_k_scores: cute.Tensor,\n" + host_ab_params
            compile_ab_fakes += (
                f"\nfake_top_k_scores = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(), {chain.moe.top_k}), stride_order=(1, 0), assumed_align=4)"
            )
            compile_ab_pass = "fake_top_k_scores,\n" + compile_ab_pass
            moe_kernel_ma_params += "\ntop_k_scores: cute.Tensor,"
            moe_host_ma_pass += "\ntop_k_scores,"

    replacements = {
        "INJECT_TILE_CONSTANTS": tile_constants,
        "INJECT_KERNEL_AUX_PARAMS": kernel_aux_params,
        "INJECT_HOST_AUX_PARAMS": host_aux_params,
        "INJECT_HOST_AUX_PASS": host_aux_pass,
        "INJECT_COMPILE_AUX_FAKES": compile_aux_fakes,
        "INJECT_COMPILE_AUX_PASS": compile_aux_pass,
        "INJECT_KERNEL_TAP_PARAMS": kernel_tap_params,
        "INJECT_HOST_TAP_PARAMS": host_tap_params,
        "INJECT_HOST_TAP_PASS": host_tap_pass,
        "INJECT_COMPILE_TAP_FAKES": compile_tap_fakes,
        "INJECT_COMPILE_TAP_PASS": compile_tap_pass,
        "INJECT_TAP_PTRS": tap_ptr_binds,
        "INJECT_AUX_VIEWS": "pass" if splitk else snippets.aux_views,
        "INJECT_EPILOGUE": snippets.epilogue,
    }
    if splitk:
        replacements["INJECT_SPLITK_EPILOGUE_BINDINGS"] = _splitk_epilogue_bindings(chain, len(plumb.tap_kernel_params))
        replacements["INJECT_REDUCE_AUX_VIEWS"] = plumb.aux_views
        replacements["INJECT_REDUCE_EPILOGUE"] = plumb.epilogue
    for marker, replacement in (
        ("INJECT_KERNEL_REDUCTION_STRIDE_PARAMS", red_kernel_stride_params),
        ("INJECT_HOST_REDUCTION_STRIDES", red_host_stride_unpack),
        ("INJECT_HOST_REDUCTION_STRIDE_PASS", red_host_stride_pass),
        ("INJECT_COMPILE_REDUCTION_STRIDE_DECLS", red_compile_stride_decls),
        ("INJECT_COMPILE_REDUCTION_STRIDE_SYMBOLS", red_compile_stride_symbols),
    ):
        if f"@@{marker}@@" in src:
            replacements[marker] = replacement
        elif chain.reductions or chain.quants:
            raise RuntimeError(f"block-scale template missing marker @@{marker}@@")
    # Multi-GEMM A/B + SF operand plumbing — only block-scale templates that
    # carry these markers (MoE block-scale lacks them).
    if "@@INJECT_KERNEL_AB_DESC_PARAMS@@" in src:
        replacements.update(
            {
                "INJECT_KERNEL_AB_DESC_PARAMS": kernel_ab_desc_params,
                "INJECT_AB_DESC_LISTS": ab_desc_lists,
                "INJECT_HOST_AB_PARAMS": host_ab_params,
                "INJECT_HOST_AB_LISTS": host_ab_lists,
                "INJECT_HOST_KERNEL_DESC_PASS": host_kernel_desc_pass,
                "INJECT_COMPILE_AB_FAKES": compile_ab_fakes,
                "INJECT_COMPILE_AB_PASS": compile_ab_pass,
            }
        )
    if "@@INJECT_STG_VEC_BINDINGS@@" in src:
        replacements["INJECT_STG_VEC_BINDINGS"] = stg_vec_bindings
    if "@@INJECT_KERNEL_TMA_C_PARAMS@@" in src:
        replacements.update(_tma_c_plumbing(chain, tma_slots, splitk=splitk))
    if "@@INJECT_TMA_STORE_SEQUENCE@@" in src:
        _epi = _epi_n_for_chain(config, chain)
        replacements["INJECT_EPILOGUE"], replacements["INJECT_TMA_STORE_SEQUENCE"] = _place_tma_stores(snippets.epilogue, chain, config, tma_slots, _epi)
        replacements["INJECT_HOST_TMA_C_DESCS"] = _host_tma_c_descs(chain, config, tma_slots, _epi)
    # MoE block-scale raw-A-tensor plumbing (per-routed-group descriptor patch).
    if "@@INJECT_MOE_KERNEL_MA_PARAMS@@" in src:
        replacements.update(
            {
                "INJECT_MOE_KERNEL_MA_PARAMS": moe_kernel_ma_params,
                "INJECT_MOE_MA_LIST": moe_ma_list,
                "INJECT_MOE_HOST_MA_PASS": moe_host_ma_pass,
                "INJECT_MOE_KERNEL_MSFA_PARAMS": moe_kernel_msfa_params,
                "INJECT_MOE_MSFA_LIST": moe_msfa_list,
                "INJECT_MOE_HOST_MSFA_PASS": moe_host_msfa_pass,
            }
        )

    if _use_bulk_combine(chain, config):
        # The template stages COMBINE fragments and reduces each complete row.
        replacements["INJECT_EPILOGUE"] = "pass"

    if "@@INJECT_SWAP_COMBINE_LAUNCH@@" in src:
        replacements["INJECT_SWAP_COMBINE_LAUNCH"] = "min_blocks_per_mp=1," if _use_swap_bulk_combine(chain, config) else ""
    src = _replace_marker_lines(src, replacements, template_kind="block-scale template")

    tag = re.sub(r"[^A-Za-z0-9_]", "_", f"{_template_stem(tmpl.file)}_{config.geometry_name}{_store_mode_tag(store_modes)}")
    src = re.sub(r"\b_kernel(?=\(|\.set_name_prefix\b)", f"frost_{tag}", src)
    return src


# ---------------------------------------------------------------------------
# Cache / import
# ---------------------------------------------------------------------------


def _fallback_cache_dir() -> Path:
    """A writable per-user cache under the system temp dir."""
    suffix = f"-{os.getuid()}" if hasattr(os, "getuid") else ""
    return Path(tempfile.gettempdir()) / f"cudnn_gemm_kernel_cache{suffix}"


@functools.lru_cache(maxsize=None)
def _usable_cache_dir(base: str) -> Path:
    """``base`` if it can be created and written, else the temp-dir fallback.

    Cached per ``base`` so an unwritable location is diagnosed (and warned
    about) once rather than on every compile."""
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
    base = os.environ.get("CUDNN_FRONTEND_GEMM_KERNEL_CACHE")
    if not base:
        # Per-user cache OUTSIDE the source tree / installed package (never write
        # into the project or site-packages). Honor XDG_CACHE_HOME, else ~/.cache.
        xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        base = os.path.join(xdg, "cudnn_gemm", "kernel_cache")
    return _usable_cache_dir(base)


def _write_atomic(path: Path, src: str) -> None:
    """Publish ``src`` at ``path`` in one rename.

    Concurrent JITs of the same kernel (the CI runs pytest -n 4) land on the same
    content-addressed path; a plain write truncates in place, so a peer can
    import a half-written file. Writing to a private temp file and renaming means
    the path only ever holds a complete module."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(src)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _import_kernel(src: str) -> object:
    """Write rendered source to a content-addressed dir and dynamic-import it."""
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
    # The template's compile() keys its persistent object on this digest: the
    # rendered source carries the tile config, dtypes, fusion chain and the
    # --gpu-arch option, so it IS the kernel's identity.
    mod.FROST_SOURCE_DIGEST = digest
    return mod


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Layout-ambiguity helper — see CompiledFusedGemm.__call__ for the policy
# ---------------------------------------------------------------------------


def _is_layout_ambiguous(t: object) -> bool:
    """True iff cute's auto-deduce of ``leading_dim`` would fault on this tensor.

    cute is OK with exactly one stride-1 dim, or multiple stride-1 dims where
    exactly one has size>1; otherwise it raises. The fault case is most commonly
    a (1, 1) scalar aux with stride (1, 1) (two stride-1 dims, none size>1).
    """
    strides = getattr(t, "stride", None)
    shape = getattr(t, "shape", None)
    if strides is None or shape is None:
        return False
    try:
        strides = list(strides())  # torch.Tensor.stride is a method
    except TypeError:
        strides = list(strides)  # numpy etc.
    sizes = list(shape)
    stride1_dims = [i for i, s in enumerate(strides) if s == 1]
    if len(stride1_dims) <= 1:
        return False
    # Multiple stride-1 dims -> safe only if exactly one has size > 1.
    big = [i for i in stride1_dims if sizes[i] > 1]
    return len(big) != 1


# Project convention for explicit leading_dim, applied when cute's auto-deduce
# would fault. Batch is permuted to the outermost dim before the kernel sees it,
# so these refer to the inner plane: A (batch,M,K) -> -1, B (batch,N,K) -> -2,
# C (batch,M,N) row-major -> -1, aux (broadcasts onto (...,M,N), C-like) -> -1.
_LEADING_DIM_A = -1
_LEADING_DIM_B = -2
_LEADING_DIM_C = -1
_LEADING_DIM_AUX = -1


def _maybe_wrap_layout(t: object, leading_dim: int) -> object:
    """If ``t``'s layout defeats cute's auto-deduce, pre-wrap via DLPack +
    explicit ``mark_layout_dynamic(leading_dim)`` (the result exposes
    ``__c_pointers__`` so the JIT executor bypasses ``TensorAdapter``, which
    would fault here). Non-ambiguous tensors pass through (fast path)."""
    # The tvm-ffi front door reads the raw DLTensor and validates it against the
    # fake's layout, so the cute _Tensor wrapper (whose __tvm_ffi_object__ raises)
    # is both unnecessary and rejected; pass the torch tensor through.
    if _TVM_FFI_OK:
        return t
    if not _is_layout_ambiguous(t):
        return t
    # Lazy import — cute is only needed on the runtime fault path (verify's
    # smoke tier runs without a GPU).
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t).mark_layout_dynamic(leading_dim)


def _wrap_raw_tensor(t: object) -> object:
    """Wrap a tensor when the kernel only consumes its raw pointer."""
    if _TVM_FFI_OK:
        return t
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t)


def _aux_fake_real_axes(ref: TensorRef) -> list[bool]:
    """Per fake-tensor axis: True where the axis carries a real (matmul) extent,
    False where the fake pins a unit dim. Mirrors ``_aux_fake_shape_code``."""
    row = ref.bcast_mode in ("per_row", "per_elem")
    col = ref.bcast_mode in ("per_col", "per_elem")
    if len(ref.dim) == 3:
        batch_real = ref.grouped_by_moe or ref.dim[0] != 1
        return [batch_real, row, col]
    return [row, col]


def _reshape_aux_to_fake(t: object, ref: TensorRef) -> object:
    """View a broadcast-aux runtime tensor to the fake's rank/unit-dim layout so
    the tvm-ffi front door's ndim/shape check accepts it. Free view: the aux is
    consumed via raw pointer + injected strides, so only the wrapper rank/shape is
    at stake. No-op off the front door, so the plain path stays bit-identical.

    The rank comes from ``len(t.shape)``, which every buffer answers. Reading it
    off an ``ndim`` attribute with a default meant a buffer that does not carry
    one -- the pack's, so every operand arriving through ``graph.execute()`` --
    was taken to match already, and a bias declared ``[1, 1, N]`` but handed over
    as ``[N]`` was refused where the backend accepts it."""
    if not _TVM_FFI_OK:
        return t
    real = _aux_fake_real_axes(ref)
    if len(t.shape) == len(real):
        return t
    extents = [int(e) for e in t.shape if int(e) != 1]
    if sum(real) != len(extents):
        return t  # extents don't map cleanly onto the fake axes; leave as-is
    it = iter(extents)
    shape = tuple(next(it) if r else 1 for r in real)
    return t.reshape(shape)


def _swap_mn_view(t: object) -> object:
    """Transpose the logical M/N axes without moving the runtime storage."""
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
    count = int(offsets.shape[0]) - 1
    if num_experts < 1 or count < 1 or count % num_experts:
        raise ValueError("FROST MoE requires G+1 explicit first_token_offset boundaries, where G is a positive multiple of num_experts")
    return count


def _expected_output_shape(spec, chain: FusionChain, mnk, top_k_scores=None) -> tuple[int, int, int]:
    if chain.has_moe and chain.moe.mode == "combine":
        tokens = int(top_k_scores.shape[0])
        return (1, int(mnk[0]), tokens) if isinstance(chain.moe, MoeSwapAbSpec) else (1, tokens, int(mnk[1]))
    return expected_shape(_output_rule(spec, chain), int(mnk[0]), int(mnk[1]))


def _initialize_combine_output(chain: FusionChain, outputs, stream=None) -> None:
    """Zero COMBINE's final output on the launch stream after runtime checks."""
    if chain.moe.mode != "combine":
        return
    output = outputs[0]  # COMBINE accepts exactly one final output.
    shape, strides = tuple(output.shape), tuple(output.stride())
    major = chain.output_specs[0].major
    if strides[-1 if major == "n" else -2] != 1:
        raise ValueError(f"MoE COMBINE output must match the compiled {major.upper()}-major layout")
    plan = buffers.strided_fill_plan(shape, strides)
    if plan is None:
        raise ValueError("MoE COMBINE output must have non-overlapping positive strides")
    buffers.apply_zero_fill_plan(output.data_ptr(), plan, DTYPE_BYTES[chain.outputs[0].dtype], stream)


def _initialize_reduction_outputs(chain: FusionChain, outputs, stream=None) -> None:
    """Seed each reduction output with its identity, before the kernel runs.

    Through the driver rather than the buffer: an engine that reaches for
    ``tensor.fill_()`` works only while the caller happened to pass a torch
    tensor, and the variant pack exists so that it does not have to.
    """
    fills = []
    for spec, tensor in zip(chain.outputs, outputs):
        if not spec.is_reduction:
            continue
        red = chain.reductions[int(spec.source.rsplit("_", 1)[1])]
        value = REDUCTION_INIT_VALUE[red.compute_dtype][red.mode]
        # The driver, on the stream the kernel will run on -- for a padded output
        # too. tensor.fill_() would queue on torch's current stream instead,
        # which is the same stream only by luck, and only exists at all while
        # the caller happened to pass a torch tensor. The pattern is packed as
        # the OUTPUT's dtype, not as float: int32's identities are the ends of
        # its range.
        shape, strides = tuple(tensor.shape), tuple(tensor.stride())
        word = buffers.init_word(red.compute_dtype, value)
        if tensor.element_size() != 4:
            raise ValueError("MoE reduction outputs require 4-byte elements")
        plan = buffers.strided_fill_plan(shape, strides)
        if plan is None:
            raise ValueError(f"MoE reduction output cannot write an element twice (shape {shape} stride {strides})")
        fills.append((tensor.data_ptr(), plan, word))
    # Validate every output before the first write, including contiguous ones.
    for ptr, plan, word in fills:
        buffers.apply_fill_plan(ptr, plan, word, stream)


@dataclass
class CompiledFusedGemm:
    """One compiled fused-GEMM, directly callable with runtime tensors.

    M, N, K are symbolic in the kernel — this one object handles any valid
    problem size (including non-cluster-tile multiples). TMA OOB-fill zero-fills
    elements past `global_dims`, so K/M/N tail tiles contribute 0 to the fp32
    accumulator; the STG path also gates per-element on `row < M` /
    `col + vsize <= N`, the TMA-store path relies on HW dropping OOB coords.

    Construct via :func:`jit_from_cudnn_graph`, then call with a variant-pack
    dict. ``lowered`` is the launch, and the only one: a graph it cannot serve
    was declined by :func:`_check_executable` before a plan existed, and a call
    it refuses goes to ``explain``, which raises rather than running anything.

    Do NOT set `oob_fill=nan_request_zero_fma` on sm100: the "NONE" enum name is
    misleading (bit 0 = zero-fill), and NaN-request is harmful because
    tcgen05.mma propagates the NaN straight through the accumulator.
    """

    chain: FusionChain
    config: TileConfig
    aux_names: list[str]  # in aux-tensor order
    generated_path: Path
    _launchable: Callable  # cute-compiled, accepts (a, b, c, mnk, *aux)
    block_scale: bool = False  # block-scaled matmul (FP4/FP8 + SF)
    device: int = 0  # CUDA device this plan's baked constants describe
    binding: "GemmBinding | None" = None  # role -> cuDNN tensor (variant-pack call)
    # Per-dense-output store mode ("tma" | "stg"). `use_tma_store` is the derived
    # "did the kernel render the TMA arm" -- the template question, not the
    # output's. A TMA slot binds the template's trailing TMA-only params (passed
    # LAST); every other output passes as a tap.
    store_modes: tuple = ()
    use_tma_store: bool = False
    # The epilogue chunk width (bytes) the kernel was RENDERED with (tile-
    # clamped); drives the runtime output/aux alignment requirements. None →
    # fall back to the chain-derived width.
    vec_bytes_epi: "int | None" = None
    # workspace for split-K
    workspace_bytes: int = 0

    def workspace_bytes_for(self, batch: int, m: int, n: int) -> int:
        """workspace for a given shape."""
        if self.config.split_k_slices == 1:
            return 0
        return align_up(self.config.split_k_slices * batch * m * n * 4)

    @property
    def tma_slots(self) -> "frozenset[int]":
        """Which dense output slots ride the TMA-C surface."""
        return frozenset(i for i, m in enumerate(self.store_modes) if m == "tma")

    # Opt in to stream-aware dispatch: frost/dispatch.py resolves the stream
    # from the execute-time cuDNN handle and forwards it as `stream=`. Engines
    # that do not carry the param stay on the default stream (see dispatch).
    accepts_stream: ClassVar[bool] = True
    # Everything the call path needs that a runtime value cannot change, read
    # once here rather than rebuilt per execute. None when this object was
    # constructed without a binding and so has no call path.
    recipe: Any = field(default=None, init=False, repr=False, compare=False)
    # The recipe as one loop: this kernel's launch path, and the only one.
    lowered: Any = field(default=None, init=False, repr=False, compare=False)
    bound: Any = field(default=(), init=False, repr=False, compare=False)
    # Why this object has no launch path, or None when it has one. Set at build.
    declined: "str | None" = field(default=None, init=False, repr=False, compare=False)
    # Reason -> count for calls the launch path refused. Refusing is the answer
    # now, not a slower route, so this is the invariant rather than observability:
    # a legal call of any flavor must leave it empty. Counted only on the path
    # that is already raising, so a served call pays nothing for it.
    deferrals: Any = field(default_factory=Counter, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.binding is not None:
            self.bound = tuple(self.binding.bound_tensors())
            self.recipe = build_recipe(self)
            self.lowered = self._lower()

    def __call__(self, variant_pack, stream=None, workspace=None):
        # The runtime call is a variant-pack dict keyed by cuDNN tensor object
        # (or uid / name) -> buffer; (M, N, K) is inferred from the buffer shapes.
        if not isinstance(variant_pack, dict):
            raise TypeError(
                "compiled kernels are called with a variant-pack dict " "{cuDNN tensor | uid | name: buffer}; got " f"{type(variant_pack).__name__}"
            )
        if self.binding is None:
            raise NotImplementedError("variant-pack call is not wired up for this graph type")
        return self.run_resolved(resolve_variant_pack(variant_pack, self.binding), stream=stream, workspace=workspace)

    def run_resolved(self, resolved, stream=None, workspace=None):
        """Launch over ``{id(bound_tensor): buffer}``, already resolved."""
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
        """The recipe as one loop over flat tuples: this kernel's only call path.

        Every branch a per-call walk would take -- multi-GEMM or not, block
        scale or not, how many outputs, which are reductions, which axis of each
        operand carries M/N/K, what order the kernel's parameters come in -- was
        settled when the kernel compiled. Taking them again per call is most of
        what a python execute path costs: 44 us for the walk that rebuilt them,
        35 reading a recipe through its objects, 20 here.

        What is left per call is the same loop for every flavor, over tuples
        flat enough that the body does no attribute lookup and calls nothing it
        does not have to. Measured against a hand-unrolled straight line for the
        plain flavor, the loop gives back 12% (19.7 us against 17.5) and costs
        one body instead of one per shape; source codegen off this same table is
        how to get that 12% back for every flavor at once.

        The body never raises. What it refuses it hands to ``explain``, which
        owns every rejection message and does not run anything -- so the goal is
        that the set of calls this refuses EQUALS the set of illegal calls, and
        a legal one it will not serve is a bug rather than a slower route.

        ``deferrals`` gets one direction of that, and only one: a legal call of
        every flavor must leave it empty, so nothing legal is refused. It says
        nothing about the other direction, because a call this ACCEPTS never
        reaches the counter -- an illegal call slipping through is caught, if at
        all, by the per-case rejection tests and by the backend differentials.
        Closing it properly means generating both the fused guards and the
        readable diagnostics from one ordered list of checks.

        None means this object has no call path at all, which the engine's
        support gate already refused; ``declined`` says why.
        """

        def decline(reason: str) -> None:
            self.declined = reason
            return None

        r = self.recipe
        if r is None:
            return decline("no binding")
        # The engine's support gate, read again rather than restated: a graph
        # this cannot run was declined before a plan existed, so reaching here
        # means a caller built the kernel directly.
        try:
            _check_executable(self.chain)
        except NotImplementedError as exc:
            return decline(str(exc))
        # Scale factors come with a block size to size their blob against.
        if bool(r.sf) != bool(r.block_size):
            return decline("scale factors without a block size")

        # Operands whose graph axis order is not the kernel's -- B, which cuDNN
        # declares [b, K, N] where the kernel reads (b, N, K). Re-labelling one
        # into kernel order is a permute, so the checks below read one order and
        # the second axis map costs nothing on a call that does not use it.
        # Keyed by POSITION in `inputs`, never by operand index: matmul(A, A)
        # binds ONE slot as both operands, and re-labelling the slot would hand
        # the other role a transposed view of the same memory.
        renorm = tuple(
            (j, tuple(op.declared), op.declared_layout[0] if len(op.declared_layout[0]) == 3 else None, op.declared_layout[1])
            for j, op in enumerate(r.inputs)
            if op.declared != KERNEL_AXES
        )

        # The recipe flattened into the loop headers: everything the body reads
        # is unpacked by the `for`, so it costs no attribute lookup. `kc` is both
        # the axis whose stride must be 1 and the axis whose extent enters the
        # TMA rule -- the same axis, by the definition of major.
        stride_ins = r.stride_ins
        in_slots = tuple(op.index for op in r.inputs)
        ins = tuple((op.kc, op.modulus, op.batch, op.kpack, op.is_b, i in stride_ins) for i, op in enumerate(r.inputs))
        outs = tuple((o.index, *(x for axis in o.rule for x in axis), o.align) for o in r.outputs)
        auxs = tuple((x.index, x.align) for x in r.aux)
        sfs = tuple((s.index, s.is_a, r.inputs[s.operand_at].batch) for s in r.sf)
        # Every input is one of the FIRST arguments, in order, so the plan only
        # has to carry the ones after them.
        tail = r.arg_plan[len(r.inputs) :]
        shared, seeds = r.shared_layout, r.seeds
        a_pos, b_pos, a_kpack = r.a_at, r.b_at, r.a.kpack
        block_size, batch = r.block_size, r.batch
        device, launchable, refuse = r.device, self._launchable, self.explain
        fill_word, fill_plan, apply_fill = buffers.fill_word_async, buffers.strided_fill_plan, buffers.apply_fill_plan
        is_contiguous = buffers.is_contiguous
        # split-K slices for the kernel
        needs_workspace = bool(r.workspace_bytes)
        split_k_slices = self.config.split_k_slices
        swap_ab = self.config.swap_ab
        packed_scale_slots = frozenset(
            o.index for o in r.outputs if o.role.startswith("quant_scale_") and self.chain.quants[int(o.role.rsplit("_", 1)[1])].scale_reorder == "F8_128x4"
        )
        mn_view_slots = (frozenset(o.index for o in r.outputs) | frozenset(x.index for x in r.aux)) - packed_scale_slots
        if needs_workspace:
            cta_k_elems = _cta_k_elems(self.chain, self.config)
            reduce_elems = _splitk_reduce_elems(self.chain)
        # Named so a test can assert which rule refused a call, and that a legal
        # call trips none. Incremented only on the path that is already raising.
        gave_up = self.deferrals

        def lowered(operands, graph_order=None, stream=None, workspace=None):
            _check_plan_device(device)
            # Which axis order each input arrived in, by the backend's own rule:
            # the descriptor defines the tensor and the pack supplies a pointer,
            # so a slot the pack described FROM the graph, or a buffer reporting
            # exactly the declared (dim, stride), is the declaration. Everything
            # else is the caller's own labelling.
            #
            # One view per ROLE. Re-labelling the SLOT would be cheaper and
            # wrong: matmul(A, A) binds a single buffer as both operands, and
            # the two roles read it through different axis maps.
            vs = [operands[i] for i in in_slots]
            for j, perm, dsh, dst in renorm:
                v = vs[j]
                sh = v.shape
                if len(sh) != 3:
                    continue  # no axis map fits it; the loop below is what refuses it
                declared = bool(graph_order) and graph_order[in_slots[j]]
                if not declared and dsh is not None:
                    declared = sh[0] == dsh[0] and sh[1] == dsh[1] and sh[2] == dsh[2]
                    if declared:
                        st = v.stride()
                        declared = st[0] == dst[0] and st[1] == dst[1] and st[2] == dst[2]
                if declared:
                    vs[j] = v.permute(*perm)
            a_sh, b_sh = vs[a_pos].shape, vs[b_pos].shape
            if len(a_sh) != 3 or len(b_sh) != 3:
                gave_up["A or B is not rank 3"] += 1
                return refuse(operands, graph_order)
            m, n, k = a_sh[1], b_sh[1], a_sh[2] * a_kpack
            # permute(1, 2, 0) relabels axes, so the strides the kernel wants are
            # that rotation of the ones each check already read -- no second read.
            problem = [m, n, k, batch]
            for (kc, mod, ebatch, kpack, is_b, takes_stride), v in zip(ins, vs):
                sh, st = v.shape, v.stride()
                if (
                    len(sh) != 3
                    or st[kc] != 1
                    or sh[kc] % mod
                    or sh[0] != ebatch
                    or sh[1] != (n if is_b else m)
                    or sh[2] * kpack != k
                    or _pow2_floor(v.data_ptr()) < 16
                ):
                    gave_up["input layout"] += 1
                    return refuse(operands, graph_order)
                if takes_stride:
                    problem += (st[1], st[2], st[0])
            # Each output axis is a constant, M, or N over a divisor (fp4 packs
            # two along N) -- the rule the build recorded, read back per axis.
            for idx, k0, v0, k1, v1, k2, v2, align in outs:
                v = operands[idx]
                sh, st = v.shape, v.stride()
                if (
                    len(sh) != 3
                    or sh[0] != (v0 if k0 == CONST else m if k0 == FROM_M else n // v0)
                    or sh[1] != (v1 if k1 == CONST else m if k1 == FROM_M else n // v1)
                    or sh[2] != (v2 if k2 == CONST else m if k2 == FROM_M else n // v2)
                    or tensor_alignment(tuple(sh), tuple(st), v.element_size(), ptr=v.data_ptr()) < align
                ):
                    gave_up["output layout"] += 1
                    return refuse(operands, graph_order)
                problem += (st[2], st[1], st[0]) if swap_ab and idx in mn_view_slots else (st[1], st[2], st[0])
            for idx, align in auxs:
                v = operands[idx]
                if tensor_alignment(tuple(v.shape), tuple(v.stride()), v.element_size(), ptr=v.data_ptr()) < align:
                    gave_up["aux alignment"] += 1
                    return refuse(operands, graph_order)
            if sfs:
                k4 = ((k // block_size) + 3) // 4
                for idx, is_a, sf_batch in sfs:
                    v = operands[idx]
                    count = int(v.numel())
                    if (
                        # A scale factor rides the launch as a rank-3 permute
                        # like every other head, so its rank is a rule here and
                        # not something for the permute to discover.
                        len(v.shape) != 3
                        or _pow2_floor(v.data_ptr()) < 16
                        or count != 1 + sum((int(s) - 1) * int(st) for s, st in zip(v.shape, v.stride()))
                        or count * v.element_size() < 512 * k4 * (((m if is_a else n) + 127) // 128) * sf_batch
                    ):
                        gave_up["scale-factor blob"] += 1
                        return refuse(operands, graph_order)
            for lead, followers in shared:
                st = tuple(vs[lead].stride())
                for j in followers:
                    if tuple(vs[j].stride()) != st:
                        gave_up["operands do not share a layout"] += 1
                        return refuse(operands, graph_order)
            # Seeding is the one thing here that WRITES, so it is planned in full
            # before any of it is issued: a second reduction output that turns
            # out to be unseedable must not find the first already filled, and a
            # 32-bit word into a narrower element would run off the end of the
            # caller's allocation before the launch could reject the dtype.
            if seeds:
                fills = []
                for idx, word, elem_bytes in seeds:
                    v = operands[idx]
                    if v.element_size() != elem_bytes:
                        gave_up["reduction seed dtype"] += 1
                        return refuse(operands, graph_order)
                    sh, st = v.shape, v.stride()
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
            extra = ()
            if needs_workspace:
                if workspace is None:
                    raise ValueError(
                        "cudnn.frost gemm: this plan needs a workspace; allocate " "graph.get_workspace_size() bytes and pass the buffer to execute()"
                    )
                # Every slice must own at least one whole CTA-K tile.
                if (k + cta_k_elems - 1) // cta_k_elems < split_k_slices:
                    raise ValueError(
                        f"cudnn.frost gemm: split_k_slices={split_k_slices} exceeds the "
                        f"{(k + cta_k_elems - 1) // cta_k_elems} K tile(s) of K={k}; "
                        f"the plan was built for a deeper K — rebuild for this shape"
                    )
                if batch * split_k_slices > 65535:
                    raise ValueError(f"cudnn.frost gemm: batch={batch} * split_k_slices={split_k_slices} " f"exceeds the CUDA grid.z limit of 65535")
                # The reducer's row chunk was sized for the plan's N.
                if n % reduce_elems:
                    raise ValueError(
                        f"cudnn.frost gemm: N={n} is not a multiple of the reducer chunk ({reduce_elems} elements) "
                        f"this plan was built with — rebuild for this shape"
                    )
                extra = (workspace.view(0, "float32", (split_k_slices * batch * m * n,)),)
            return launchable(
                tuple(problem),
                *(v.permute(1, 2, 0) for v in vs),
                *(
                    (
                        ((_swap_mn_view(operands[i]) if swap_ab and i in mn_view_slots else operands[i]).permute(1, 2, 0))
                        if ref is None
                        else _reshape_aux_to_fake(_swap_mn_view(operands[i]) if swap_ab else operands[i], ref)
                    )
                    for i, ref in tail
                ),
                *extra,
                stream=_as_custream(stream),
            )

        return lowered

    def explain(self, operands, graph_order=None):
        """Say what is wrong with a call the launch path refused, and raise.

        The rules are written twice on purpose: once fused into ``lowered``'s
        guards, where the cost is per call and the answer is a bool, and once
        here, where the cost is nothing -- this only ever runs on a call that
        has already failed -- and the answer names the operand and both numbers.

        What this is NOT is a second way to run the kernel. Two executors are
        two answers to what the graph computes, and the differential that
        policed them could never catch a misconception they shared, which is
        exactly how the axis-order bug survived one.

        Falling off the end means the two readings have drifted apart, which is
        the one failure mode splitting them introduces. Loud beats silent.
        """
        recipe = self.recipe
        _check_plan_device(recipe.device)
        mnk, axes = recipe.problem(operands, graph_order)
        check_shapes(recipe, operands, mnk, axes)
        check_alignment(recipe, operands, axes)
        raise RuntimeError(
            "cudnn.frost gemm: the launch path refused this call and no rule explains why -- its "
            "per-call guards and the diagnostics here have drifted apart. Guards that fired: "
            f"{dict(self.deferrals)}."
        )


def _mma_a_dtype(chain: FusionChain) -> str:
    """MMA A dtype = the graph-declared operand dtype (no implicit cast;
    mainloop transforms are dtype-preserving)."""
    return chain.matmul.a_dtype


def _mma_b_dtype(chain: FusionChain) -> str:
    """MMA-instruction B dtype = the graph-declared operand dtype."""
    return chain.matmul.b_dtype


def _check_supported(chain: FusionChain, config: TileConfig) -> None:
    """Reject a plain-matmul (input/acc dtype combo × active arch) we can't run.
    Delegates to ``kernel_registry``'s unified MMA-type support table (single
    source of truth). ``config`` is unused (support is config-independent)."""
    from ..kernel_registry import GraphType, mma_arch_reject

    reason = mma_arch_reject(chain, GraphType.MATMUL, config.pipeline)
    if reason is not None:
        raise NotImplementedError(reason)


def _check_cta_group_geometry(
    config: TileConfig,
) -> None:
    """2-CTA MMA structural constraints on the geometry. The MMA pair spans two
    M-direction CTAs, so an odd cgrp_size_m breaks the kernel's barrier/B-split
    math — this is impossible to render, not a known-bad-but-probeable case."""
    if not config.is_cta_pair_mma:
        return
    if config.cga_size_m % 2 != 0:
        raise NotImplementedError(f"2-CTA MMA needs cgrp_size_m % 2 == 0; " f"config {config.name!r} has cgrp_size_m={config.cga_size_m}")
    if config.mma_tile_n % 16 != 0:
        # Empirically (B200): tcgen05.mma cta_group::2 with n_dim not a
        # multiple of 16 raises an illegal-instruction fault (n_dim=8/24/40
        # fault, 16/32/48/240 run); 1-CTA MMA accepts any multiple of 8.
        raise NotImplementedError(
            f"2-CTA MMA needs mma_inst_n % 16 == 0 (pair MMA instruction n_dim); " f"config {config.name!r} has mma_inst_n={config.mma_tile_n}"
        )
    if config.cta_tile_n % 16 != 0:
        # The pair splits B's N across the two CTAs: per-CTA SMEM/TMA N is
        # cta_tile_n // 2, which must stay a multiple of 8 (the 8-row tcgen05
        # core-matrix chunk / TileConfig's N granularity).
        raise NotImplementedError(
            f"2-CTA MMA needs cta_tile_n % 16 == 0 (pair splits B's N in SMEM); " f"config {config.name!r} has cta_tile_n={config.cta_tile_n}"
        )


def _check_mma_n_dim(
    chain: FusionChain,
    config: TileConfig,
) -> None:
    """MMA n_dim rules that depend on the dtype / operand layout, not just geometry.

    Neither is visible to the geometry guards. The rule is on the MMA
    instruction's n_dim (``mma_inst_n``); ``cta_tile_n`` is checked too because
    it equals it (N is never split across instructions) — cheap belt-and-braces,
    not a second rule.
    """
    ns = (("mma_inst_n", config.mma_tile_n), ("cta_tile_n", config.cta_tile_n))

    # The int8 MMA kind is narrower than the others above N=32.
    if DTYPE_TO_MMA_KIND.get(_mma_a_dtype(chain)) == "nvvm.Tcgen05MMAKind.INT8":
        for label, n in ns:
            if n >= 40 and (n // 8) % 2 != 0:
                raise NotImplementedError(f"int8 MMA needs N ≤ 32 or a multiple of 16; " f"config {config.name!r} has {label}={n}")

    # An 8-bit TRANSPOSED (N-major) B tightens N for every non-fp16/bf16 kind:
    # 1CTA 16..256 step 16, 2CTA 32..256 step 32.
    bs = chain.block_scale
    b_dt = bs.b_dtype if bs is not None else chain.matmul.b_dtype
    if _dtype_bits(b_dt) == 8 and chain.matmul.b_major == "n":
        step = 32 if config.is_cta_pair_mma else 16
        for label, n in ns:
            if n < step or n % step != 0:
                raise NotImplementedError(
                    f"8-bit transposed (N-major) B needs N ≥ {step} and a multiple "
                    f"of {step} under a {config.ctas_per_mma}-CTA MMA; config {config.name!r} "
                    f"has {label}={n}"
                )


# cuTensorMapEncodeTiled caps each boxDim at 256 elements.
_TMA_BOX_DIM_MAX = 256


def _a_tma_box_m(cfg: TileConfig, mcast_slices: int) -> int:
    """Rows per K-major A copy, subdividing each multicast slice as needed.

    The boxes partition the same SMEM tile and complete the same byte-counted
    barrier. A 512-row tile uses two 256-row boxes; non-power-of-two forced
    tiles also need an exact divisor so the last copy cannot cross a tile edge.
    """
    rows = cfg.cta_tile_m // mcast_slices
    return rows if rows <= _TMA_BOX_DIM_MAX else math.gcd(rows, _TMA_BOX_DIM_MAX)


def _check_mma_k_dim(chain: FusionChain, config: TileConfig) -> None:
    """Dense MMA K width: shared by candidate enumeration, precheck, and render."""
    if config.mma_tile_k_bytes == 32:
        return
    if config.mma_tile_k_bytes != 64 or config.pipeline != "sm100":
        raise NotImplementedError(f"plain matmul supports mma_tile_k_bytes=32, or 64 for sm100 FP8; config {config.name!r}")
    fp8_dtypes = ("fp8_e4m3", "fp8_e5m2")
    if _mma_a_dtype(chain) not in fp8_dtypes or _mma_b_dtype(chain) not in fp8_dtypes:
        raise NotImplementedError(f"plain matmul at mma_tile_k_bytes=64 requires FP8 E4M3/E5M2 A and B; config {config.name!r}")
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gemm.frost.kernel_registry import MMA_INST_K64_ARCH_RANGES

    arch = _current_arch()
    if arch is None or not any(lo <= arch < hi for lo, hi in MMA_INST_K64_ARCH_RANGES):
        spans = " or ".join(f"{lo} <= SM < {hi}" for lo, hi in MMA_INST_K64_ARCH_RANGES)
        active = "unknown" if arch is None else f"sm_{arch}"
        raise NotImplementedError(f"plain FP8 at mma_tile_k_bytes=64 needs {spans}, but the active GPU is {active}; config {config.name!r}")
    # Dense K64 uses full M datapaths: hardware M=128 (1CTA) or 256 (2CTA).
    if config.mma_tile_m != 128:
        raise NotImplementedError(f"plain FP8 mma_tile_k_bytes=64 requires mma_tile_m=128 per CTA; config {config.name!r}")


def _check_dtype_config_compat(
    chain: FusionChain,
    config: TileConfig,
) -> None:
    """Reject (chain, config) where the config K_BYTES isn't a multiple of the
    MMA dtype's element width. The config's own SMEM N drives the N-major-B
    swizzle-group check."""
    _check_mma_k_dim(chain, config)
    mma_dt = _mma_a_dtype(chain)
    elem_bytes = DTYPE_BYTES.get(mma_dt)
    if elem_bytes is None:
        raise ValueError(f"unsupported MMA a_dtype {mma_dt!r}")
    if config.cta_tile_k_bytes % elem_bytes != 0:
        raise ValueError(
            f"TileConfig {config.name!r} has cta_tile_k_bytes="
            f"{config.cta_tile_k_bytes} which is not divisible by "
            f"elem_bytes={elem_bytes} for dtype {chain.matmul.a_dtype!r}."
        )
    # sm100 partitions K-major A into boxes of _a_tma_box_m rows (M-major
    # already walks swizzle groups). Other operands still need the whole
    # extent to fit cuTensorMapEncodeTiled's per-dimension ceiling.
    smem_m, smem_n, _ = config.cta_smem_tile_mnk(elem_bytes)
    box_extents = [("per-CTA SMEM N", smem_n)]
    if config.pipeline != "sm100":
        box_extents.append(("cta_tile_m", smem_m))
    for label, extent in box_extents:
        if extent > _TMA_BOX_DIM_MAX:
            raise NotImplementedError(
                f"TileConfig {config.name!r}: {label}={extent} exceeds the TMA "
                f"box_dim ceiling of {_TMA_BOX_DIM_MAX}; the CTA tile is loaded "
                f"in one box per operand"
            )
    mn_group_elems = config.cta_tile_k_bytes // elem_bytes
    # Each tcgen05 MMA instruction's operand descriptor starts at its own MN
    # sub-block, so the check is on the PER-MMA slice. A warp-scoped family
    # (fixed mma tile, e.g. sm120) has no per-MMA descriptors -- only the whole
    # extent must cut into whole groups (the TMA group walk) -- so num_mma == 1.
    _per_mma = 1 if getattr(type(config), "FIXED_MMA_TILE_MN", None) is not None else config.mma_size_m
    if chain.matmul.a_major == "m":
        slice_m = config.cta_tile_m // _per_mma
        what = f"per-MMA M={slice_m}" if _per_mma > 1 else f"M={slice_m}"
        if slice_m < mn_group_elems:
            raise ValueError(
                f"TileConfig {config.name!r} cannot use M-major A for "
                f"dtype={chain.matmul.a_dtype!r}: {what} "
                f"is smaller than the {mn_group_elems}-element swizzle group"
            )
        if slice_m % mn_group_elems != 0:
            raise ValueError(f"TileConfig {config.name!r} cannot use M-major A: " f"{what} is not divisible by " f"swizzle group {mn_group_elems}")
    if chain.matmul.b_major == "n":
        slice_n = config.cta_smem_tile_mnk(elem_bytes)[1]
        if slice_n < mn_group_elems:
            raise ValueError(
                f"TileConfig {config.name!r} cannot use N-major B for "
                f"dtype={chain.matmul.b_dtype!r}: per-MMA per-CTA SMEM N={slice_n} "
                f"is smaller than the {mn_group_elems}-element swizzle group"
            )
        if slice_n % mn_group_elems != 0:
            raise ValueError(
                f"TileConfig {config.name!r} cannot use N-major B: " f"per-MMA per-CTA SMEM N={slice_n} is not divisible by " f"swizzle group {mn_group_elems}"
            )


def _dtype_bits(dtype: str) -> int:
    """Element width in BITS (handles sub-byte FP4)."""
    return 4 if dtype == "fp4_e2m1" else DTYPE_BYTES[dtype] * 8


def _tma_alignment_reject(
    a_dtype: str,
    b_dtype: str,
    a_major: str,
    b_major: str,
    M: int,
    N: int,
    K: int,
) -> str | None:
    """TMA encodes the contiguous input dimension's stride in 16-byte (128-bit)
    units; a misaligned extent silently mis-strides every row past the first.
    One rule for every pipeline (plain + block-scale, sm100 + sm103) and for
    both the graph-time and runtime dims. ``None`` = aligned."""
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


def _alignment_reject(named_buffers) -> "str | None":
    """Every runtime buffer's alignment must be >= the kernel's compiled
    requirement for its role. Entries are ``(role, buffer, required_bytes,
    mode)``; ``None`` buffers are skipped. Two modes:

    - ``"ptr"`` (TMA-loaded operands / SF): only the BASE POINTER must meet the
      requirement (16, the TMA global-address rule). TMA tiles the tensor via a
      descriptor, so the contiguous extent is irrelevant (an mxfp8 SF with an
      8-element inner run is fine) and the row stride is checked separately by
      ``_tma_alignment_reject``.
    - ``"full"`` (STG/LDG outputs / aux): the full ``tensor_alignment`` =
      ``min(ptr, stride, shape)`` (bytes, cap 32) bounds the widest vector the
      store/load can issue, so all three must meet the baked vector width.

    Below-requirement buffers otherwise fault deep in the kernel with an opaque
    ``cudaErrorInvalidValue``. Returns a reason string, or ``None``."""
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
    """A block-scale SF operand reaches the kernel as a BASE POINTER plus a
    layout the template re-synthesizes from M/N/K — the F8_128x4 packed blob of
    512-byte atoms (128 rows x 4 SF-K, fp8). The runtime tensor's own shape and
    strides are never read, so a blob that is not one dense byte run of at least
    the required size is silently read out of bounds (wrong numerics, no fault).

    The graph cannot carry this check: it declares the LOGICAL scale factors
    (``dim=[1, M, K//block_size]``) while the runtime buffer is the reordered
    blob, so the two shapes legitimately disagree and only the call site sees
    the real one. Entries are ``(role, buffer, required_bytes)``; ``None``
    buffers are skipped. Returns a reason string, or ``None``."""
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
    """Reject a strided view where grouped-row quant stores raw blob offsets.

    The F8_128x4 epilogue deliberately addresses the physical atom stream and
    does not apply the runtime tensor strides. Reuse the input-SF packed-blob
    contract so every accepted destination is one dense byte run.
    """
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


def _check_input_alignment(chain: FusionChain) -> None:
    """Graph-time TMA input-alignment gate (the runtime dims are re-checked in
    ``CompiledFusedGemm.__call__`` — the kernel is shape-agnostic, so the call
    may carry different M/N/K than the graph)."""
    mm = chain.matmul
    reason = _tma_alignment_reject(mm.a_dtype, mm.b_dtype, mm.a_major, mm.b_major, mm.M, mm.N, mm.K)
    if reason is not None:
        raise ValueError(reason)


# SMEM-D double-buffer depth (TMA store of one subtile overlaps the sts of the
# next). 2 is the minimum useful value; more helps only if TMA-store latency
# exceeds one subtile's sts cost, not generally the case on B200.
_EPI_SMEM_STAGES = 2


def _moe_aligned_offsets(chain, cfg) -> bool:
    if not chain.has_moe or chain.moe is None:
        return False
    required = cfg.cga_tile_mn[1 if isinstance(chain.moe, MoeSwapAbSpec) else 0]
    if chain.block_scale is not None:
        required = math.lcm(required, 128)
    return chain.moe.offset_multiple % required == 0


# TMEM accumulator stages: 2 = MMA of tile N+1 overlaps the epilogue of tile N.
# The TMEM budget is the only thing that can take it back down to 1.
_ACC_STAGES_MAX = 2


# A 16-byte staging row needs no swizzle: the XOR exists to spread a WIDE row
# across the banks, and at 16 B/thread an STS.128 warp already covers all 32
# banks conflict-free per phase. `Swizzle(0, ...)` has a zero-width mask, so
# `store_swizzled` degenerates to a plain store and the descriptor declares
# `none`, which is the one mode the DSL exempts from the box-width check.
_EPI_SWIZZLE_BY_ROW_BYTES = {16: (0, "none"), 32: (1, "s32b"), 64: (2, "s64b"), 128: (3, "s128b")}


_EPI_ROW_BYTES_MAX = 128  # widest TMA store swizzle
_EPI_N_BASE = 32  # drain width when the epilogue is already hidden behind the MMA
_EPI_N_MAX = 64  # per-lane fp32 registers the drain can hold

# Families whose templates the compiler renders with the TMA-store epilogue;
# the sm120 warp kernel always takes its transposed-STG path.
_TMA_STORE_EPI_PIPELINES = ("sm100", "sm103")


def _epi_n(cfg, out_dt: str) -> int:
    cols = _epi_tile_cols(cfg)
    cap = _EPI_N_BASE
    if cfg.mma_size_m > 1 and 2 * cfg.mma_size_m * cols > _tmem_cols_for_arch():
        cap = _EPI_N_MAX
    # Must DIVIDE `cols`, not floor it: a partial subtile would TMA-store past the
    # tile edge, which TMA clips only at the GLOBAL extent. No separate gate --
    # this is the only place that can violate it (test_epi_n_divides_the_drain_width).
    return min(_EPI_ROW_BYTES_MAX * 8 // DTYPE_BITS[out_dt], cap, _pow2_floor(cols, cap=cols))


def _epi_n_for_chain(cfg, chain: FusionChain) -> int:
    """Chain-aware drain width.

    Column block quantization pays one warp reduction per column regardless of
    subtile width.  A 64-column drain therefore keeps the same reduction work
    while halving subtile-loop and output-store overhead.  Other epilogues keep
    the measured 32-column default and its lower register footprint.
    """
    if _use_swap_bulk_combine(chain, cfg):
        # Logical token batch, staged across two 64-column TMEM loads.
        return 128
    if _use_bulk_combine(chain, cfg):
        cols = _epi_tile_cols(cfg)
        # A full row is staged across drains before one bulk reduction. Keep
        # register pressure bounded and uniform spans for accumulator overlap.
        return min(256 // DTYPE_BYTES[chain.output_dtype], cols & -cols)
    if not any(q.axis == 1 for q in chain.quants):
        return _epi_n(cfg, _epi_store_dtype(chain, cfg))
    cols = _epi_tile_cols(cfg)
    return min(
        _EPI_ROW_BYTES_MAX * 8 // DTYPE_BITS[chain.output_dtype],
        _EPI_N_MAX,
        _pow2_floor(cols, cap=cols),
    )


def _epi_swizzle_lines(cfg, out_dt: str, chain: FusionChain | None = None) -> list[str]:
    epi_n = _epi_n_for_chain(cfg, chain) if chain is not None else _epi_n(cfg, out_dt)
    return [
        f"epi_n = {epi_n}",
        f"epi_row_elems = {_epi_row_elems(out_dt, epi_n)}",
    ]


def _epi_slot_widen(
    chain,
    cfg,
) -> int:
    """One shared slot spans the widest TMA-stored row. An STG output never
    touches the ring, so it must not widen it."""
    if cfg.split_k_slices > 1:
        return 1
    epi_n = _epi_n_for_chain(cfg, chain)
    tma = _tma_slots_for(chain, cfg)
    widths = [_epi_row_bytes(chain.output_specs[i].dtype, epi_n) for i in sorted(tma) if i < len(chain.output_specs)]
    if not widths:
        return 1
    return max(1, max(widths) // _epi_row_bytes(chain.output_dtype, epi_n))


_EPI_WARPS = 4  # every template's `num_epilogue_warps`
_EPI_STAGE_ROWS = _EPI_WARPS * 32


def _use_bulk_combine(chain: FusionChain, cfg: TileConfig) -> bool:
    """Use the qualified single-CTA block-scaled COMBINE epilogue.

    The normal 2-CTA BF16 route fails memcheck on SM100 (#1942). Keep
    normal 2-CTA configurations on atomics until independently qualified.
    The separately qualified SM107 swapped route has its own predicate.
    """
    return (
        not _FORCE_STG_EPI
        and chain.has_moe
        and chain.moe.mode == "combine"
        and chain.has_block_scale
        and not isinstance(chain.moe, MoeSwapAbSpec)
        and not cfg.swap_ab
        and cfg.pipeline == "sm100"
        and cfg.cta_group == 1
        and cfg.mma_tile_m == 128
        and chain.output_specs[0].major == "n"
        and chain.output_dtype in ("fp32", "fp16", "bf16")
    )


def _combine_smem_bytes(chain: FusionChain, cfg: TileConfig) -> int:
    """Reserve the weighted staging tile and its alignment padding."""
    if _use_swap_bulk_combine(chain, cfg):
        return 128 * cfg.cta_tile_m * DTYPE_BYTES[chain.output_dtype] + 16
    if not _use_bulk_combine(chain, cfg):
        return 0
    return _EPI_STAGE_ROWS * _epi_tile_cols(cfg) * DTYPE_BYTES[chain.output_dtype] + 16


def _use_swap_bulk_combine(chain: FusionChain, cfg: TileConfig) -> bool:
    """Qualified weight-by-token COMBINE: transpose fragments, then bulk-reduce.

    The tile owns 256 features and a pair of 128-token CTA partitions. Stage
    one 128-token batch per CTA; 64-column TMEM loads bound live registers.
    Immutable B/SFB maps address compact tokens and segmented scales separately,
    just as the normal COMBINE path does for A/SFA. No workload shape is baked in.
    """
    return (
        not _FORCE_STG_EPI
        and chain.has_moe
        and isinstance(chain.moe, MoeSwapAbSpec)
        and chain.moe.mode == "combine"
        and chain.has_block_scale
        and _current_arch() == 107
        and cfg.pipeline == "sm100"
        and cfg.swap_ab
        and cfg.cta_group == 2
        and (cfg.cga_size_m, cfg.cga_size_n, cfg.cga_size_k) == (2, 1, 1)
        and (cfg.cta_tile_m, cfg.cta_tile_n, cfg.cta_tile_k_bytes) == (256, 256, 128)
        and (cfg.mma_tile_m, cfg.mma_tile_n, cfg.mma_tile_k_bytes) == (128, 256, 64)
        and cfg.mma_size_m == 2
        and cfg.split_k_slices == 1
        and chain.num_gemms == 1
        and chain.output_specs[0].major == "m"
        and chain.output_dtype == "bf16"
        and chain.matmul.b_major == "k"
        and chain.matmul.a_dtype == chain.matmul.b_dtype == "fp4_e2m1"
        and chain.block_scale.block_size == 16
        and not chain.block_scale.fake_dequant_a
        and not chain.block_scale.fake_dequant_b
        and not _moe_aligned_offsets(chain, cfg)
    )


def _use_fixed_combine_descriptors(chain: FusionChain, cfg: TileConfig) -> bool:
    """Read compact tokens and segmented scales through immutable tensor maps.

    COMBINE masks group-tail rows before reading routing metadata or reducing
    outputs. Those rows need not be zero-filled at each expert boundary. The
    token and SF coordinates differ because only SF segments are 128-row padded.
    Keep the first implementation on the qualified Rubin NVFP4 singleton path.
    """
    bs = chain.block_scale
    return (
        _use_bulk_combine(chain, cfg)
        and _current_arch() == 107
        and cfg.cta_group == 1
        and (cfg.cga_size_m, cfg.cga_size_n, cfg.cga_size_k) == (1, 1, 1)
        and cfg.cta_tile_m % 128 == 0
        and chain.matmul.a_major == "k"
        and chain.matmul.a_dtype == chain.matmul.b_dtype == "fp4_e2m1"
        and bs.block_size == 16
        and not bs.fake_dequant_a
        and not bs.fake_dequant_b
        and not _moe_aligned_offsets(chain, cfg)
    )


def _epi_packed_lanes(
    cfg,
) -> bool:
    """The packed `lane < 16` drain (hardware M=64, foot-gun #18): half the lanes
    carry nothing and a row is `warp_idx * 16 + lane`, not `tidx`."""
    return cfg.mma_tile_m == 64 and not cfg.is_cta_pair_mma


def _epi_dp22(
    cfg,
) -> bool:
    """The 2-CTA 2x2-DP drain: two column halves per stage, so two TMA boxes.
    Trades short K for long K vs STG (4096^2, `64x128` cluster2x1): +5.8 % at
    K=512, -2.5 % at 4096."""
    return cfg.mma_tile_m == 64 and cfg.is_cta_pair_mma


def _epi_stage_rows(
    cfg,
) -> int:
    """Distinct row slots per ring stage: one per thread, except packed."""
    return cfg.epi_tile_m if _epi_packed_lanes(cfg) else _EPI_STAGE_ROWS


def _smem_d_bytes(
    cfg,
    chain,
) -> int:
    """SMEM-D buffer bytes for the TMA-store epilogue: `_EPI_SMEM_STAGES` slots of
    one epilogue subtile (one MMA-M block x epi_n) + a 16-byte alignment pad. With
    mma_size_m > 1 the M blocks reuse the same slots. epi_n MUST be the same value
    the kernel renders, or the reserve under-counts and the launch is rejected."""
    store_dt = _epi_store_dtype(chain, cfg)
    row_bytes = _epi_row_bytes(store_dt, _epi_n_for_chain(cfg, chain))
    return _EPI_SMEM_STAGES * _epi_stage_rows(cfg) * row_bytes * _epi_slot_widen(chain, cfg) + 16


def _output_store_mode(
    out,
    chain,
    cfg,
) -> str:
    """Store mode for ONE output, from the output itself plus the geometry it is
    stored with -- never from what the others need."""
    # Families whose templates render the TMA-store arm at all; the sm120 warp
    # kernel always takes its transposed-STG path.
    if cfg.pipeline not in _TMA_STORE_EPI_PIPELINES or (chain.has_moe and chain.moe.mode == "combine"):
        return "stg"
    if chain.has_moe and chain.moe.mode == "scatter":
        # Scatter4 selects the descriptor's outer rows. The token axis must
        # therefore be non-contiguous (public N-major, also after swapAB).
        token_major = "n" if isinstance(chain.moe, MoeSwapAbSpec) else "m"
        if out.major == token_major:
            return "stg"
        if isinstance(chain.moe, MoeSwapAbSpec) and _epi_n_for_chain(cfg, chain) % 4:
            return "stg"
    # TMA addresses its contiguous dim in 16-byte units, and truncates. The next
    # two rejections are that granule at a different extent.
    epi_n = _epi_n_for_chain(cfg, chain)

    # The staged SMEM row; transposed stages the 128-byte M column instead. The
    # ceiling is the widest swizzle, not the granule.
    row = _epi_row_bytes(out.dtype, _mmajor_atom_m(out.dtype) if out.major == "m" else epi_n)
    if row < 16 or row % 16 or row > _EPI_ROW_BYTES_MAX:
        return "stg"

    # The strides the descriptor encodes -- never the contiguous dim's.
    carrier = _cd_view_bits(out.dtype) // 8
    dim, stride = dense_output_layout(chain, out.dtype, out.dim, out.stride)
    if tensor_alignment(dim, stride, carrier) < 16:
        return "stg"
    encoded = [stride[1] if out.major == "n" else stride[2]]
    if dim[0] > 1:
        encoded.append(stride[0])
    if any(x * carrier < 16 or x * carrier % 16 for x in encoded):
        return "stg"

    # The chunk is SHARED, so an output that FOLDS it constrains the arm for all.
    # `epi_n % block_size == 0` needs no check: `_check_block_quant_supported`
    # already forces it through `_pow2_floor(cols)`.
    # Quant and ordinary reduction emitters guard a whole chunk. N-grouped
    # reductions split it at the promised token boundary alignment instead.
    quants = [chain.quants[q.quant_idx] for q in chain.output_specs if q.quant_idx is not None]
    if (quants or chain.reductions) and not isinstance(chain.moe, MoeSwapAbSpec) and chain.matmul.N % epi_n:
        return "stg"

    if isinstance(chain.moe, MoeSwapAbSpec) and out.major == "n":
        if chain.moe.offset_multiple * carrier % 16 or chain.matmul.N * carrier % 16:
            return "stg"
    if out.major == "m":
        if chain.has_moe and not isinstance(chain.moe, MoeSwapAbSpec) and (chain.moe.offset_multiple * carrier % 16 or chain.matmul.M * carrier % 16):
            return "stg"
        # A block taller than the drain emits ZERO stores.
        if cfg.epi_tile_m % _mmajor_atom_m(out.dtype):
            return "stg"
    return "tma"


def _store_mode_tag(store_modes: "tuple[str, ...]") -> str:
    """Suffix for the kernel symbol naming its store arm, empty for the TMA one.

    Two builds of the SAME geometry that differ only in store mode -- `force_stg_epi`
    against the gate's own choice -- otherwise carry an IDENTICAL symbol, so nsys
    reports them as one kernel."""
    if not store_modes or all(m == "tma" for m in store_modes):
        return ""
    return "_stg" if all(m == "stg" for m in store_modes) else "_mixedstore"


def _store_modes(
    chain,
    cfg,
) -> tuple[str, ...]:
    """One store mode per output, in `chain.outputs` slot order.

    Every TMA-stored output shares ONE SMEM ring -- the slot is sized for the
    widest of them and each takes its own typed view -- so there is no budget to
    spend: an output takes the surface iff it is eligible. MoE additionally gives
    each one a workspace slot and a scratch copy of the D descriptor, which it
    re-dimensions per routed group.
    """
    outs = chain.outputs
    if _FORCE_STG_EPI:
        return ("stg",) * len(outs)
    # For splitK, reduce kernel always use STG
    if cfg.split_k_slices > 1:
        return ("stg",) * len(outs)
    modes = ["stg"] * len(outs)
    for i in range(len(chain.output_specs)):
        modes[i] = _output_store_mode(outs[i], chain, cfg)
    return tuple(modes)


def _tma_slots_for(
    chain,
    cfg,
) -> "frozenset[int]":
    return frozenset(i for i, m in enumerate(_store_modes(chain, cfg)) if m == "tma")


def _use_tma_store_epi(
    chain,
    cfg,
) -> bool:
    """Does the kernel render the TMA-store arm at all? True iff some output
    takes it -- the renderer deletes one of `@@STG_ONLY@@` / `@@TMA_STORE_ONLY@@`,
    so this is a property of the TEMPLATE, while `_store_modes` is the per-output
    answer the emitters consume."""
    # use tma for splitK's fp32 partials when N % 4 == 0
    if cfg.split_k_slices > 1:
        return not _FORCE_STG_EPI and cfg.pipeline in _TMA_STORE_EPI_PIPELINES and chain.matmul.N % 4 == 0
    return "tma" in _store_modes(chain, cfg)


def _check_block_quant_supported(
    chain: FusionChain,
    vec_bytes_epi: int,
    config: TileConfig,
) -> None:
    if not chain.quants:
        return
    if chain.has_mainloop_fusion:
        raise NotImplementedError("block_scale_quantize epilogue is not supported with mainloop fusion")
    elem_bytes = DTYPE_BYTES[chain.output_dtype]
    vsize = vec_bytes_epi // elem_bytes
    cols_per_acc_stage = _epi_tile_cols(config)
    # A quant pins the chunk to its block size, so — unlike every other chain,
    # where the chunk is derived from the outputs' own layouts — the two
    # divisibility properties the epilogue relies on have to be asserted here:
    # the drain walks whole chunks (`for j in range(subtile_w // vsize)`) and
    # stores only whole chunks (`if col_j + vsize <= N`).
    subtile_w = _pow2_floor(cols_per_acc_stage, MAX_EPI_CHUNK_ELEMS)
    if subtile_w % vsize != 0:
        raise NotImplementedError(
            f"block_scale_quantize epilogue chunk ({vsize} elements, from block sizes "
            f"{sorted({q.block_size for q in chain.quants})}) must divide the {subtile_w}-column "
            f"drain subtile of cols_per_acc_stage={cols_per_acc_stage} "
            f"(config={config.name})"
        )
    swapped = isinstance(chain.moe, MoeSwapAbSpec)
    if not swapped and chain.matmul.N % vsize != 0:
        raise NotImplementedError(
            f"block_scale_quantize requires N % chunk == 0 — the epilogue stores whole "
            f"chunks only, so a partial trailing chunk is dropped; got N={chain.matmul.N}, "
            f"chunk={vsize} elements"
        )
    for q in chain.quants:
        if q.scale_reorder is None and q.scale_dim is not None:
            expected_scale_dim = (
                (chain.matmul.batch, chain.matmul.M // q.block_size, chain.matmul.N)
                if q.axis == 1
                else (chain.matmul.batch, chain.matmul.M, chain.matmul.N // q.block_size)
            )
            if q.scale_dim != expected_scale_dim:
                raise NotImplementedError(f"block_scale_quantize requires scale_dim={expected_scale_dim}; got {q.scale_dim}")
        # Applies to col quants too: `_emit_block_quant_col` maps chunk column
        # i to lane i % block_size, so a chunk narrower than the block leaves
        # the upper lanes storing a scale they never computed.
        if vsize % q.block_size != 0:
            raise NotImplementedError(
                "block_scale_quantize requires block_size to divide the epilogue chunk "
                f"width (= the largest quant block, clamped to the lowest set bit of "
                f"cols_per_acc_stage={cols_per_acc_stage} and to {MAX_EPI_CHUNK_ELEMS} "
                f"elements); got block_size={q.block_size}, vsize={vsize}"
            )
        segmented_rows = q.grouped_by_moe and ((q.axis == 1) == swapped)
        if segmented_rows:
            if not chain.has_moe:
                raise NotImplementedError("grouped row block_scale_quantize requires a MoE grouped matmul graph")
            if not chain.has_block_scale:
                raise NotImplementedError(
                    "grouped row block_scale_quantize currently requires a block-scaled MoE "
                    "matmul whose scheduler supplies the per-group 128-row scale prefix"
                )
        if swapped and q.axis != 1:
            boundary = 4 * q.block_size if q.grouped_by_moe else q.block_size
            if chain.moe.offset_multiple % boundary or chain.matmul.N % q.block_size:
                raise NotImplementedError(
                    f"MoE token-axis block_scale_quantize requires first_token_offset value alignment divisible by {boundary} "
                    f"and S divisible by block_size={q.block_size}"
                )
        if segmented_rows:
            tokens = chain.matmul.N if swapped else chain.matmul.M
            features = chain.matmul.M if swapped else chain.matmul.N
            required_rows = segmented_row_scale_capacity_rows(tokens, chain.moe.num_groups)
            padded_blocks = ((features // q.block_size + 3) // 4) * 4
            if q.scale_dim is None or q.scale_dim[0] != 1 or q.scale_dim[1] < required_rows or q.scale_dim[1] % 128 or q.scale_dim[2] != padded_blocks:
                raise NotImplementedError(
                    f"grouped F8_128x4 feature-axis block_scale_quantize requires scale_dim=(1, segmented_rows, {padded_blocks}), "
                    f"with segmented_rows a multiple of 128 and >= {required_rows}; got {q.scale_dim}"
                )
        if q.axis == 1:
            # Col quant: a warp (block 32) or half-warp (block 16) of rows is
            # one M block; the redux needs every row guard uniform across the
            # reducing lanes. MoE CONTRACT (cannot be checked at JIT — fto is
            # runtime device data): every first_token_offset value must be a
            # multiple of block_size, so no column block spans a routed-group
            # boundary (mirrors cutedsl's FIX_PAD_SIZE padding contract);
            # grouped_by_moe tightens it to fto % (4 * block_size) == 0.
            if q.grouped_by_moe and not chain.has_moe:
                raise NotImplementedError("grouped (per-group segmented) col block_scale_quantize requires " "a MoE grouped matmul graph")
            if q.block_size not in (16, 32):
                raise NotImplementedError(f"col block_scale_quantize requires block_size 32 (one warp of rows) or " f"16 (one half-warp); got {q.block_size}")
            if chain.matmul.M % q.block_size != 0:
                raise NotImplementedError(
                    f"col block_scale_quantize requires M % block_size == 0 for a "
                    f"reduction-uniform row guard; got M={chain.matmul.M}, "
                    f"block_size={q.block_size}"
                )
            if config.mma_tile_m == 64 and not config.is_cta_pair_mma and q.block_size != 16:
                raise NotImplementedError(
                    "col block_scale_quantize on the mma_inst_m=64 1-CTA-MMA epilogue " "(lane<16 packed layout, 16 rows per warp) supports only block_size 16"
                )
            if q.scale_reorder == "F8_128x4" and not segmented_rows:
                expected_scale_dim = (
                    chain.matmul.batch,
                    ((chain.matmul.N + 127) // 128) * 128,
                    (((chain.matmul.M // q.block_size) + 3) // 4) * 4,
                )
                if q.scale_dim != expected_scale_dim:
                    raise NotImplementedError(
                        "F8_128x4 col block_scale_quantize scale output currently requires " f"scale_dim={expected_scale_dim}; got {q.scale_dim}"
                    )
            continue
        if cols_per_acc_stage < q.block_size or cols_per_acc_stage % q.block_size != 0:
            raise NotImplementedError(
                "block_scale_quantize epilogue requires each CTA epilogue drain to "
                "cover whole quantization blocks; got "
                f"cols_per_acc_stage={cols_per_acc_stage}, block_size={q.block_size}, "
                f"config={config.name}"
            )
        if q.scale_reorder == "F8_128x4" and not segmented_rows:
            padded_n_blocks = (((chain.matmul.N // q.block_size) + 3) // 4) * 4
            expected_scale_dim = (
                chain.matmul.batch,
                ((chain.matmul.M + 127) // 128) * 128,
                padded_n_blocks,
            )
            if q.scale_dim != expected_scale_dim:
                raise NotImplementedError("F8_128x4 block_scale_quantize scale output currently requires " f"scale_dim={expected_scale_dim}; got {q.scale_dim}")


_FORCE_STG_EPI = False


@contextlib.contextmanager
def force_stg_epi(enabled: bool = True):
    """Make every output take the STG store, even where the TMA gate accepts it.

    A/B harness only -- nothing on the shipping path sets it. The flag is module
    state rather than an argument because `_store_modes` is reached from the
    bottom of four separate render paths; this is the ONE place that saves and
    restores it.
    """
    global _FORCE_STG_EPI
    prev = _FORCE_STG_EPI
    _FORCE_STG_EPI = enabled
    try:
        yield
    finally:
        _FORCE_STG_EPI = prev


def _check_executable(chain: FusionChain) -> None:
    """Can this engine RUN the graph, or only render a kernel for it?

    A dense or block-scale chain has ONE call path -- the one lowered from the
    recipe -- so what that path cannot serve is a graph this engine declines,
    not a call that takes a slower route. Declining sends it to the backend,
    which is where it would have gone had this engine never been asked; keeping
    a second executor for it is how the two drift apart.

    MoE is the exception and stays on its own launchers: it is >= 2 launches
    with a workspace, and has no recipe to lower from.
    """
    if any(red.mode == "norm2" for red in chain.reductions):
        raise NotImplementedError("a norm2 reduction takes a square root after the kernel, which is a device operation this engine does not own")
    if chain.has_moe and chain.moe.mode == "combine":
        if chain.num_gemms != 1 or chain.ops or chain.quants or chain.reductions or len(chain.output_specs) != 1:
            raise NotImplementedError("MoE COMBINE requires one GEMM and one final output without post-combine operations")
        if chain.matmul.accum_dtype != "fp32" or chain.output_dtype not in ("fp32", "fp16", "bf16"):
            raise NotImplementedError("MoE COMBINE requires FP32 MMA accumulation and FP32/FP16/BF16 output")
    if chain.has_moe and chain.moe.mode == "scatter":
        if chain.quants or chain.reductions:
            raise NotImplementedError("MoE SCATTER currently supports dense pointwise outputs without quantization or reduction")
        if any(DTYPE_BITS[out.dtype] < 8 for out in chain.outputs):
            raise NotImplementedError("MoE SCATTER packed sub-byte outputs are not implemented")
    if chain.has_moe:
        return
    if not _TVM_FFI_OK:
        # Not a narrowing of the install surface: apache-tvm-ffi ships in the
        # same `cutedsl` extra as the DSL these kernels are written in, so a
        # build without it has no DSL either and was already declining.
        raise NotImplementedError("the tvm-ffi front door is not installed, and the launch path this engine has needs it (pip install apache-tvm-ffi)")


def _cta_k_elems(chain: FusionChain, config: TileConfig) -> int:
    """Elements of K per CTA tile; fp4 packs two per byte."""
    return config.cta_tile_k_bytes * 8 // DTYPE_BITS[_mma_a_dtype(chain)]


def _splitk_workspace_bytes(chain: FusionChain, config: TileConfig) -> int:
    """fp32 partials: split_k_slices * B * M * N * 4B."""
    if config.split_k_slices == 1:
        return 0
    return align_up(config.split_k_slices * chain.matmul.batch * chain.matmul.M * chain.matmul.N * 4)


def _auto_split_k(chain: FusionChain, config: TileConfig, sm_count: "int | None" = None) -> TileConfig:
    """Layer split-K onto the selected config when the output grid underfills
    the GPU and K is deep.

    Enable only when BOTH hold:
    - output CTA tiles < SM count (otherwise the grid already fills the GPU);
    - K >= 2048 bytes (below that the reducer overhead outweighs the split).

    S = min over four bounds:
    - one full wave of CTAs (sm // output_tiles): fractional waves leave a
      ragged tail, and partial traffic grows with S;
    - every slice keeps at least max(64 elements, 2 CTA-K tiles) of K;
    - 32, the reducer's trace-time unroll bound;
    - the CUDA grid.z limit (kernel 1's z = batch * S)."""
    if config.split_k_slices != 1:
        return config
    if _splitk_reject_reason(chain, replace(config, split_k_slices=2)) is not None:
        return config
    mm = chain.matmul
    if mm.K * DTYPE_BITS[_mma_a_dtype(chain)] < 2048 * 8:
        return config
    sm = sm_count if sm_count is not None else _sm_count()
    output_tiles = mm.batch * -(-mm.M // config.cta_tile_m) * -(-mm.N // config.cta_tile_n)
    # For nvfp4 block-scale the one-wave target overshoots: measured split-K
    # curves pay only below 1/8 grid fill, near 64 total CTAs, with
    # power-of-two S >= 4. First measured on MMA_INST_K64_ARCH_RANGES silicon
    # and gated there; the weekly sweep shows the same curve on sm100/sm103
    # (+15 layers / -5, net +0.5% fp4 geomean on both), so the gate is now
    # dtype-driven, not arch-driven. fp8 keeps one-wave sizing.
    _fp4_bs = chain.has_block_scale and mm.a_dtype == mm.b_dtype == "fp4_e2m1"
    eager_fill = not _fp4_bs
    if output_tiles >= (sm if eager_fill else -(-sm // 8)):
        return config
    cta_k_elems = _cta_k_elems(chain, config)
    num_k_tiles = -(-mm.K // cta_k_elems)
    min_k_tiles_per_slice = max(-(-64 // cta_k_elems), 2)
    slices = min(
        sm // output_tiles,  # one full wave of CTAs
        num_k_tiles // min_k_tiles_per_slice,  # per-slice K floor
        32,  # reducer chain-unroll limit; S>32 runs one serial dynamic loop
        65535 // mm.batch,  # CUDA grid.z hard limit
    )
    if not eager_fill:
        slices = min(slices, max(1, 64 // output_tiles))
        if slices < 4:
            # the bounds cannot support four slices; S=2/3 measure as losses
            return config
        slices = 1 << (slices.bit_length() - 1)  # snap down to a power of two
    if slices <= 1:
        return config
    # A two-way split does not amortize the partial-output traffic and the
    # reduction launch: the weekly exhaustive sweep measures every dense S=2
    # pick as a loss on every architecture measured, while S>=3 keeps its
    # wins, matching what was already measured for matched FP4 operands.
    # Larger automatic splits and explicitly supplied knobs remain available.
    if slices == 2:
        return config
    return replace(config, split_k_slices=slices)


_TERMINAL_QUANT_ONE_CTA_MAX_M = 1408


def _graph_dynamic_shapes(graph) -> bool:
    """Whether the frontend graph declared its shapes dynamic."""
    return bool(getattr(graph, "_cpp_graph_kwargs", {}).get("is_dynamic_shape_enabled", False))


def _baseline_config(chain: FusionChain, *, dynamic_shapes: bool = False, knobs=None) -> TileConfig:
    if knobs is not None:
        try:
            config = knobs.to_config()
        except (KeyError, ValueError, NotImplementedError) as exc:
            raise NotImplementedError(f"frost_gemm: knobs do not name a canonical tile config: {exc}") from exc
        if dynamic_shapes and config.split_k_slices != 1:
            # Same rule as the automatic pick below: the slice count is fixed at
            # plan time and a runtime K may not tile it, so a replayed record
            # does not get to bypass it.
            raise NotImplementedError("frost_gemm: split-K plans do not support dynamic shapes")
        return config
    from ..kernel_registry import preferred_strategy
    from ..tile_config import select_config

    tile_m = chain.matmul.M
    if chain.moe is not None:
        tile_m = (chain.matmul.M + chain.moe.num_groups - 1) // chain.moe.num_groups
    declared_rows = chain.matmul.M if chain.moe is not None else chain.matmul.M * chain.matmul.batch
    force_cta_group = None
    if chain.num_gemms == 1 and chain.has_block_scale and chain.quants and declared_rows <= _TERMINAL_QUANT_ONE_CTA_MAX_M:
        # A terminal block-scale quantizer adds a substantial epilogue drain.
        # Keep both M tiles independently scheduled over the measured row
        # envelope instead of coupling them in a 2-CTA MMA pair. For dense
        # batches, every batch contributes M rows; for MoE, total M is the only
        # plan-time upper bound on a runtime expert group -- the average in
        # tile_m cannot bound device-resident offsets.
        # select_config still scores the cluster after this strategy choice;
        # no cluster shape is pinned here.
        force_cta_group = 1
    config = select_config(
        tile_m,
        chain.matmul.N,
        chain.num_gemms,
        K=chain.matmul.K,
        block_scale=chain.has_block_scale,
        b_n_major=chain.matmul.b_major == "n",
        b_elem_bytes=DTYPE_BYTES[chain.matmul.b_dtype],
        force_cta_group=force_cta_group,
        m_is_group_average=chain.moe is not None,
    )
    # Re-target at the preferred family and MMA-inst K width; cta_group rides
    # the geometry and only moves when the family cannot serve it (sm120 is
    # warp-scoped MMA, 1-CTA only).
    config = preferred_strategy(chain, config)
    # skip splitK for MoE and dynamic shapes
    if chain.has_moe or dynamic_shapes:
        return config
    return _auto_split_k(chain, config)


def plan_config(chain: FusionChain, *, dynamic_shapes: bool = False, knobs=None) -> TileConfig:
    """Choose one strategy, or replay the exact supplied knobs."""
    config = _baseline_config(chain, dynamic_shapes=dynamic_shapes, knobs=knobs)
    if knobs is not None or dynamic_shapes:
        return config
    from ..planning import select_strategy

    return select_strategy(chain, config, probe=probe_chain)


def plan_configs(chain: FusionChain, *, dynamic_shapes: bool = False) -> list[TileConfig]:
    """Choose a bounded list with the single-plan recommendation first."""
    config = _baseline_config(chain, dynamic_shapes=dynamic_shapes)
    if dynamic_shapes:
        return [config]
    from ..planning import select_strategies

    return select_strategies(chain, config, probe=probe_chain)


def _precheck_plain(
    chain: FusionChain,
    config: TileConfig,
) -> None:
    from ..kernel_registry import select_template

    _check_supported(chain, config)
    _arch_reason = select_template(chain, config).active_reject(config, chain)
    if _arch_reason is not None:
        raise NotImplementedError(_arch_reason)
    _check_dtype_config_compat(chain, config)
    _check_input_alignment(chain)
    _compute_output_vec_bytes(chain)
    _check_block_quant_supported(chain, _epi_vec_bytes(chain, config), config)
    _check_own_family(select_template(chain, config))


def _precheck_moe(
    chain: FusionChain,
    config: TileConfig,
) -> None:
    from ..kernel_registry import GraphType, mma_arch_reject, select_template

    _check_executable(chain)
    reason = mma_arch_reject(chain, GraphType.MOE, config.pipeline)
    if reason is not None:
        raise NotImplementedError(reason)
    token_major = chain.matmul.b_major if isinstance(chain.moe, MoeSwapAbSpec) else chain.matmul.a_major
    if token_major != "k":
        raise NotImplementedError(f"MoE grouped matmul supports only K-major token (got token {token_major}-major)")
    _arch_reason = select_template(chain, config).accepts(chain, config)
    if _arch_reason is not None:
        raise NotImplementedError(_arch_reason)
    _check_dtype_config_compat(chain, config)
    _check_input_alignment(chain)
    _compute_output_vec_bytes(chain)
    _check_block_quant_supported(chain, _epi_vec_bytes(chain, config), config)
    _check_own_family(select_template(chain, config))


def _precheck_block_scale(
    chain: FusionChain,
    config: TileConfig,
) -> None:
    from ..kernel_registry import select_template

    _check_block_scale_supported(chain, config.pipeline)
    _check_input_alignment(chain)
    _epi_pipelines = ("sm100",)
    if chain.reductions:
        if config.pipeline not in _epi_pipelines:
            raise NotImplementedError(f"block-scale reduction is not supported on {config.pipeline} templates")
        for red in chain.reductions:
            if red.compute_dtype != "fp32" or red.dtype != "fp32":
                raise NotImplementedError("block-scale reduction supports only fp32 compute/output")
    if chain.quants and config.pipeline not in _epi_pipelines:
        raise NotImplementedError(f"block-scale quant epilogue is not supported on {config.pipeline} " "templates (not yet validated on sm103)")
    _tmpl = select_template(chain, config)
    if chain.is_multi_gemm and not _tmpl.supports_multi_gemm:
        raise NotImplementedError(f"block-scale multi-GEMM ({chain.num_gemms} GEMMs) is not supported by " f"{_tmpl.file}.")
    _arch_reason = _tmpl.active_reject(config)
    if _arch_reason is not None:
        raise NotImplementedError(_arch_reason)
    _compute_output_vec_bytes(chain)
    _check_block_quant_supported(chain, _epi_vec_bytes(chain, config), config)
    _check_own_family(select_template(chain, config))


def _precheck_moe_block_scale(
    chain: FusionChain,
    config: TileConfig,
) -> None:
    from ..kernel_registry import GraphType, mma_arch_reject, select_template

    _check_executable(chain)
    reason = mma_arch_reject(chain, GraphType.MOE_BLOCK_SCALE, config.pipeline)
    if reason is not None:
        raise NotImplementedError(reason)
    token_major = chain.matmul.b_major if isinstance(chain.moe, MoeSwapAbSpec) else chain.matmul.a_major
    if token_major != "k":
        raise NotImplementedError(f"MoE block-scale matmul supports only K-major token (got token {token_major}-major)")
    if chain.reductions and not isinstance(chain.moe, MoeSwapAbSpec):
        for red in chain.reductions:
            if red.compute_dtype != "fp32" or red.dtype != "fp32":
                raise NotImplementedError("MoE block-scale reduction supports only fp32 compute/output")
    _check_input_alignment(chain)
    _arch_reason = select_template(chain, config).accepts(chain, config)
    if _arch_reason is not None:
        raise NotImplementedError(_arch_reason)
    _compute_output_vec_bytes(chain)
    _check_block_quant_supported(chain, _epi_vec_bytes(chain, config), config)
    _check_own_family(select_template(chain, config))


def precheck_path(
    chain: FusionChain,
    config: TileConfig,
) -> None:
    """Every gate the matching ``_jit_*`` path runs before it renders anything.
    ``probe_supported`` calls this so a graph the build will decline is never
    reported eligible -- otherwise the engine is listed, ``build_plans`` drops
    it, native cuDNN serves the graph and no test fails."""
    if chain.has_moe and chain.has_block_scale:
        _precheck_moe_block_scale(chain, config)
    elif chain.has_block_scale:
        _precheck_block_scale(chain, config)
    elif chain.has_moe:
        _precheck_moe(chain, config)
    else:
        _precheck_plain(chain, config)


def probe_supported(graph: cudnn.pygraph, config: "TileConfig | None" = None) -> None:
    """Cheap eligibility check — the :func:`jit_from_cudnn_graph` gates WITHOUT
    ``cute.compile``. Raises if the engine can't run the graph. This is
    ``FrostGemmEngine.check_support`` (see ``cudnn/gemm/frost/engine.py``), so it
    runs for every candidate graph and must stay cheap.

    With no explicit config it probes the (config, cta_group) ``build_gemm_plan``
    will pick, and runs the same gate prefix that path runs -- probe and build
    must agree or an ineligible graph is listed, dropped at build, and silently
    served by native cuDNN."""
    # frost is written in the cutedsl these kernels compile through; below its
    # floor the engine cannot build (same gate the linear-attention engines
    # apply). Decline early so a too-old wheel falls back to the backend instead
    # of faulting deep in cute -- and so the --gpu-arch target pin, which lands in
    # cutedsl at the floor, is always available by the time a plan compiles. An
    # internal RC passes: cutedsl_too_old judges only the public wheel.
    probe_cutedsl()
    chain, _binding = analyze_with_binding(graph)
    if config is None:
        config = plan_config(chain, dynamic_shapes=_graph_dynamic_shapes(graph))
    probe_chain(chain, config)


def probe_cutedsl() -> None:
    """The cutedsl floor gate shared by check_support and the family heuristics."""
    installed, version = buffers.cutedsl_state()
    if not installed:
        raise NotImplementedError("frost_gemm requires the cutedsl extra (nvidia-cutlass-dsl)")
    if buffers.cutedsl_too_old(version):
        want = ".".join(str(v) for v in buffers.CUTEDSL_MIN_VERSION)
        raise NotImplementedError(f"frost_gemm requires nvidia-cutlass-dsl >= {want}; found {version[1]}")


def probe_chain(chain: FusionChain, config: TileConfig) -> None:
    """The chain-level gates of :func:`probe_supported` for one explicit
    ``config``: what the family heuristics run before listing a plan, so a plan
    in the ranked list is one the engine will build (facts in, no graph needed)."""
    if config.swap_ab:
        from ..fusion_ir import swap_ab

        chain = swap_ab(chain)
    _dtype_reason = dtype_arch_reject(chain, _current_arch())
    if _dtype_reason is not None:
        raise NotImplementedError(_dtype_reason)
    _check_executable(chain)
    _check_splitk_supported(chain, config)
    if chain.is_multi_gemm and not (chain.has_moe or chain.has_block_scale):
        from ..kernel_registry import select_template

        tmpl = select_template(chain, config)
        if not tmpl.supports_multi_gemm:
            raise NotImplementedError(
                f"multi-GEMM ({chain.num_gemms} parallel GEMMs) is only supported "
                f"by the 1-CTA-MMA CLC template this pass; got {config.name!r}, "
                f"→ {tmpl.file}."
            )
    precheck_path(chain, config)


def _check_splitk_supported(chain: FusionChain, config: TileConfig) -> None:
    reason = _splitk_reject_reason(chain, config)
    if reason is not None:
        raise NotImplementedError(reason)


def _splitk_reject_reason(chain: FusionChain, config: TileConfig) -> "str | None":
    """Two-kernel split-K applicability"""
    if config.split_k_slices == 1:
        return None
    reasons = []
    if chain.has_moe:
        reasons.append("MoE grouped matmul")
    if chain.is_multi_gemm:
        reasons.append("multi-GEMM")
    if chain.quants:
        reasons.append("block-scale quantize")
    if chain.output_dtype == "fp4_e2m1":
        reasons.append("packed fp4 output")
    if chain.matmul.accum_dtype != "fp32":
        reasons.append(f"{chain.matmul.accum_dtype} accumulation (fp32 partials only)")
    if chain.matmul.batch * config.split_k_slices > 65535:
        reasons.append(f"grid.z={chain.matmul.batch * config.split_k_slices} exceeds CUDA's 65535 limit")
    if config.split_k_slices > 32:
        # The reducer unrolls its accumulation at trace time; 32 bounds the
        # unroll and is where the auto-selector caps S anyway.
        reasons.append(f"more than 32 slices ({config.split_k_slices})")
    if reasons:
        return f"split_k_slices={config.split_k_slices} is not supported with: {', '.join(reasons)}"
    return None


def jit_from_cudnn_graph(
    graph: cudnn.pygraph,
    config: TileConfig = DEFAULT_CONFIG,
) -> CompiledFusedGemm:
    """End-to-end: cuDNN frontend graph -> rendered + cute-compiled GEMM kernel.

    Eagerly analyze → codegen → render → import → cute.compile; returns a
    directly-callable :class:`CompiledFusedGemm`.

    `graph` is a ``cudnn.pygraph`` built after ``import cudnn.gemm.frost`` (the
    import installs the op-recording hook). `config` is a PURE-GEOMETRY tile from
    `tile_config.CATALOG`. ``config.swap_ab`` logically lowers
    ``C.T = B.T @ A.T`` without changing runtime buffers. Execution strategy:
    ``cta_group`` ∈ {1, 2} and picks the template (mainloop auto-detected).

    Mixed CGA needs no argument and no caller change: where the GPU and the
    template both support it, the launch carries ``config``'s cluster as the
    PREFERRED shape plus the smallest fallback the MMA mode allows, so the device
    fills the SMs a wide fixed cluster leaves idle (:func:`_mixed_cga_fallback`).
    Everywhere else the launch is the plain fixed cluster it always was.
    """
    chain, binding = analyze_with_binding(graph)
    if config.swap_ab:
        from ..fusion_ir import swap_ab

        chain = swap_ab(chain)
        binding = swap_ab_binding(binding)
    _dtype_reason = dtype_arch_reject(chain, _current_arch())
    if _dtype_reason is not None:
        raise NotImplementedError(_dtype_reason)
    _check_cta_group_geometry(config)
    _check_mma_n_dim(chain, config)
    _check_splitk_supported(chain, config)
    # MoE grouped block-scale = both matches at once (dequant + moe_grouped);
    # check BEFORE the single-feature gates.
    if chain.has_moe and chain.has_block_scale:
        return _jit_moe_block_scale(chain, config, binding=binding)
    # Block-scale is gated independently (own per-side case table).
    if chain.has_block_scale:
        return _jit_block_scale(chain, config, binding=binding)
    # MoE grouped matmul: own template (grouped persistent scheduler + per-group
    # A TMA descriptor replacement).
    if chain.has_moe:
        return _jit_moe(chain, config, binding=binding)
    # Multi-GEMM is only in the 1ctamma CLC template. select_template skips
    # capability gates, so reject unsupported strategy here with a clear message
    # rather than fault deep in cute on a missing vec_f32_<g> binding.
    if chain.is_multi_gemm:
        from ..kernel_registry import select_template

        tmpl = select_template(chain, config)
        if not tmpl.supports_multi_gemm:
            raise NotImplementedError(
                f"multi-GEMM ({chain.num_gemms} parallel GEMMs) is only supported "
                f"by the 1-CTA-MMA CLC template this pass; got {config.name!r}, "
                f"→ {tmpl.file}. Use cta_group=1."
            )
    # Plain-matmul (pipeline × input/acc dtype combo [× GPU for the rare
    # special-case combos]) gate, then the template family's active-GPU gate.
    _precheck_plain(chain, config)
    store_modes = _store_modes(chain, config)
    use_tma = _use_tma_store_epi(chain, config)
    splitk = config.split_k_slices > 1
    snippets = generate(
        chain,
        vec_bytes_epi=_epi_chunk_bytes(chain, config, use_tma),
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        # For splitK, only the partials buffer (slot 0) uses the TMA
        tma_slots=frozenset({0}) if (splitk and use_tma) else frozenset(i for i, m in enumerate(store_modes) if m == "tma"),
        packed_lanes=_epi_packed_lanes(config),
        split_k_slices=config.split_k_slices,
    )
    src = _render_template(chain, snippets, config)
    mod = _import_kernel(src)
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    return CompiledFusedGemm(
        chain=chain,
        config=config,
        device=_plan_device(),
        aux_names=[aux.name for aux in chain.aux_tensors],
        generated_path=_cache_dir() / f"gen_{digest}" / "generated_kernel.py",
        _launchable=mod.compile(),  # one-shot cute.compile (lru_cached in mod)
        binding=binding,
        store_modes=store_modes,
        use_tma_store=use_tma,
        vec_bytes_epi=_out_vec_bytes(chain, config, use_tma),
        workspace_bytes=_splitk_workspace_bytes(chain, config),
    )


# --- sm100_block_scale_matmul: arch/dtype support -------------------------
# Supported per-side cases live in `kernel_registry.MMA_TYPE_SUPPORT` (single
# source of truth); this gate delegates to it.


def _check_block_scale_supported(chain: FusionChain, template_pipeline: str) -> None:
    """Reject a block-scale matmul the ``template_pipeline`` family can't run —
    FULL per-side match (data/SF dtype, block size, reorder, dequant
    compute/out) + the rare family×combo×GPU special cases, via
    `kernel_registry.MMA_TYPE_SUPPORT` / `MMA_GPU_ARCH_SPECIAL_CASES`."""
    from ..kernel_registry import GraphType, mma_arch_reject

    reason = mma_arch_reject(chain, GraphType.BLOCK_SCALE_MATMUL, template_pipeline)
    if reason is not None:
        raise NotImplementedError(reason)


def _moe_dense_layout_bad(t) -> bool:
    """A MoE dense output must be contiguous in one of the two inner dims."""
    return t.stride(-1) != 1 and t.stride(-2) != 1


def _moe_operand_layout_bad(chain, token, weight) -> bool:
    """True when a MoE token / weight buffer is not contiguous along its major
    dim. Both are passed shaped (batch, M|N, K); the major lives in the strides."""
    a_unit = -1 if chain.matmul.a_major == "k" else -2
    b_unit = -1 if chain.matmul.b_major == "k" else -2
    return token.stride(a_unit) != 1 or weight.stride(b_unit) != 1


def _declared_storage(t, memo: dict):
    """``storage_geometry`` of a bound cuDNN tensor, computed once per binding:
    three pybind reads per operand per call otherwise."""
    key = id(t)
    if key not in memo:
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.graph_types import storage_geometry

        try:
            memo[key] = storage_geometry(t.get_dim(), t.get_stride(), t.get_data_type())
        except Exception:  # noqa: BLE001 -- an analyzer-synthesized ref has no dims
            memo[key] = None
    return memo[key]


def _kernel_order(buf, t, memo: "dict | None" = None):
    """A B-side buffer described AS its declaration is in the graph's
    ``[b, k, n]`` axis order (the variant pack lends a bare address, or a buffer
    that disagrees with the declaration, exactly that geometry); the launch
    reads ``(b, n, k)``. The declaration is compared in STORAGE slots -- an fp4
    declaration spells elements, the slot spells x2 pairs -- with the same
    tie-break ``recipe.Operand.axes`` applies on the dense path."""
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


def _resolve_moe_variant_pack(compiled, variant_pack: dict):
    """Resolve a MoE variant-pack dict into the positional-call buffers,
    inferring (S, N, K) from shapes. Returns ``(a_bufs, b_bufs, out_bufs,
    aux_bufs, fto, sfa, sfb, token_index, token_ks, top_k_scores, (S, N, K))``."""
    b = compiled.binding
    if b is None:
        raise NotImplementedError("variant-pack call is not yet wired up for this graph type")
    resolved = resolve_variant_pack(variant_pack, b)
    # the binding's tensors are fixed, so their declared geometry is too
    memo = getattr(compiled, "_declared_storage_memo", None)
    if memo is None:
        memo = {}
        try:
            compiled._declared_storage_memo = memo
        except AttributeError:  # a compiled object that refuses new attributes: per-call memo
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
    k_factor = 2 if compiled.chain.matmul.a_dtype == "fp4_e2m1" else 1
    token_index, token_ks = routing_indices(a_bufs)
    S = token_index.shape[0] if token_index is not None else a_bufs[0].shape[1]
    K = a_bufs[0].shape[2] * k_factor
    # N from the weight: a dense output may be FP4-packed (N/2 bytes).
    N = b_bufs[0].shape[1]
    # Unified runtime TMA-alignment gate (S plays the M role). An aligned K
    # also keeps every per-routed-group patched A descriptor base 16-byte
    # aligned (group base = token + group_begin*K elements).
    mm = compiled.chain.matmul
    _align_reason = _tma_alignment_reject(mm.a_dtype, mm.b_dtype, mm.a_major, mm.b_major, S, N, K)
    if _align_reason is not None:
        raise ValueError(_align_reason)
    _check_moe_combine_n(compiled, N)
    return a_bufs, b_bufs, out_bufs, aux_bufs, fto, sfa, sfb, token_index, token_ks, top_k_scores, (S, N, K)


# One 128-byte TMA tensormap slot per CTA per distinct A operand.
_MOE_SCHED_COUNTER_SLOTS = 1
_MOE_DESC_SLOT_BYTES = 128


def _register_legacy_device_view_adapter() -> None:
    """Teach the LEGACY (pre-tvm-ffi) DSL executor to accept a raw
    ``DeviceView`` launch arg. That executor marshals only recognized types
    (its own tensors, or types with a registered adapter such as
    torch.Tensor) and silently DROPS the rest from the packed launch args —
    shifting every later argument one slot left and crashing the compiled
    host. The adapter wraps the view via DLPack at launch time; the executor
    keeps the wrapper alive for the call, and the view (held in the caller's
    args) owns the capsule's backing structs. The tvm-ffi executor consumes
    DLPack args natively — and a pre-wrapped cute tensor breaks its MoE
    compilation — so when the legacy registry is absent this is a no-op."""
    try:
        from cutlass.base_dsl.runtime.jit_arg_adapters import JitArgAdapterRegistry
    except ImportError:
        return
    if buffers.DeviceView in JitArgAdapterRegistry.jit_arg_adapter_registry:
        return
    from cutlass.cute.runtime import from_dlpack

    @JitArgAdapterRegistry.register_jit_arg_adapter(buffers.DeviceView)
    def _adapt_device_view(view):
        # Truthful alignment claim from the actual pointer, capped at the
        # 128-byte tensormap slot the MoE workspace fake declares.
        ptr = view.data_ptr()
        return from_dlpack(view, assumed_align=min(ptr & -ptr, _MOE_DESC_SLOT_BYTES))


def _moe_carve_workspace(caller, n_slots: int, plan: str):
    """View a workspace buffer as the int64 A-descriptor scratch the kernel
    patches (16 int64 = one tensormap slot). Carving from the caller's buffer
    instead of allocating keeps the pointer stable across executes, which is
    what makes the MoE plan safe to capture in a CUDA graph.

    Returned as the raw ``DeviceView``: the tvm-ffi executor converts DLPack
    launch args natively (pre-wrapping as a cute tensor breaks it); the legacy
    executor converts through the one-time adapter registered above."""
    _register_legacy_device_view_adapter()
    need = n_slots * _MOE_DESC_SLOT_BYTES
    # already carved when the engine handed one down; a raw buffer only when
    # the plan allocated its own (the direct jit_from_cudnn_graph path)
    ws = caller if isinstance(caller, Workspace) else Workspace(caller, need, plan, align=_MOE_DESC_SLOT_BYTES)
    return ws.view(0, "int64", (need // 8,))


@dataclass
class CompiledMoeGemm:
    """A compiled MoE grouped matmul forward pass, directly callable.

    Layouts (rank-3):
      * ``token``  — (1, T, K) row-major (A); T == S for NONE.
      * ``weight`` — (E, N, K) row-major (B, per-expert; bit-identical to cuDNN's
        ``[E, H, N]`` column-major-in-H×N layout).
      * ``first_token_offset`` — (G+1,) explicit boundaries:
        group g spans ``[fto[g], fto[g+1])``.
      * ``output`` — (1, S, N) row-major."""

    chain: FusionChain
    config: TileConfig
    generated_path: Path
    _launchable: Callable
    _grid_ctas: int = 0
    device: int = 0  # CUDA device this plan's baked constants describe
    aux_names: list = field(default_factory=list)
    binding: "GemmBinding | None" = None  # role -> cuDNN tensor (variant-pack call)
    # The epilogue chunk width (bytes) the kernel was RENDERED with (tile-
    # clamped); drives the runtime output/aux alignment requirements.
    vec_bytes_epi: "int | None" = None
    # Tensormap slots each CTA patches: one per distinct A operand, plus the
    # output descriptor when the TMA-store epilogue is on (re-dimensioned per
    # routed group so the hardware clips the ragged tail).
    _desc_slots_per_cta: int = 0
    store_modes: tuple = ()
    use_tma_store: bool = False
    accepts_stream: ClassVar[bool] = True  # stream-aware dispatch (see CompiledFusedGemm)

    @property
    def tma_slots(self) -> "frozenset[int]":
        """Which dense output slots ride the TMA-C surface."""
        return frozenset(i for i, m in enumerate(self.store_modes) if m == "tma")

    @property
    def workspace_bytes(self) -> int:
        """Per-CTA tensormap scratch: one 128-byte slot per CTA per patched
        descriptor. The persistent grid is shape-independent, so this is
        constant for the plan — which is why override-shape needs no re-query."""
        return (self._grid_ctas * self._desc_slots_per_cta + _MOE_SCHED_COUNTER_SLOTS) * _MOE_DESC_SLOT_BYTES

    def _make_workspace(self, n_slots, caller=None):
        """The per-CTA tensormap GMEM workspace (16 int64/slot, 128-byte
        aligned). ``n_slots`` = grid_ctas * _desc_slots_per_cta. Carved from the
        CALLER's buffer; a call without one is a contract error (Rule 8)."""
        if caller is None:
            raise ValueError(
                f"{type(self).__name__} requires a {n_slots * _MOE_DESC_SLOT_BYTES}-byte workspace but execute() received "
                "none; size it with workspace_bytes and pass workspace= (a plan owns no device memory, Rule 8)"
            )
        return _moe_carve_workspace(caller, n_slots, type(self).__name__)

    def __call__(self, variant_pack, workspace=None, stream=None):
        # Variant-pack dict {cuDNN tensor | uid | name: buffer}; (S, N, K) is
        # inferred from buffer shapes.
        if not isinstance(variant_pack, dict):
            raise TypeError(
                "compiled kernels are called with a variant-pack dict " "{cuDNN tensor | uid | name: buffer}; got " f"{type(variant_pack).__name__}"
            )
        _check_plan_device(self.device)
        return self._call_variant_pack(variant_pack, workspace, stream)

    def _launch_single(self, token, weight, first_token_offset, output, snke, workspace=None, stream=None, token_index=None, token_ks=None, top_k_scores=None):
        first_token_offset = _offset_vector(first_token_offset)
        if len(snke) < 3:
            raise ValueError("MoE call needs problem_size (S, N, K[, ...]); " f"got {snke!r}")
        S, N, K = int(snke[0]), int(snke[1]), int(snke[2])
        outputs_spec = self.chain.outputs
        outputs = list(output) if isinstance(output, (list, tuple)) else [output]
        if len(outputs) != len(outputs_spec):
            raise ValueError(
                f"this MoE graph has {len(outputs_spec)} output(s) " f"({[o.source for o in outputs_spec]}); got {len(outputs)}. " "Pass outputs in slot order."
            )
        for name, t, rank in (
            ("token", token, 3),
            ("weight", weight, 3),
            ("first_token_offset", first_token_offset, 1),
        ):
            if len(t.shape) != rank:
                raise ValueError(f"MoE {name} must be rank-{rank}; got shape {tuple(t.shape)}")
        for spec, t in zip(outputs_spec, outputs):
            if len(t.shape) != 3 or tuple(t.shape) != _expected_output_shape(spec, self.chain, (S, N, K), top_k_scores):
                raise ValueError(
                    f"MoE output {spec.source!r} must have shape "
                    f"{_expected_output_shape(spec, self.chain, (S, N, K), top_k_scores)}; "
                    f"got {tuple(t.shape)}"
                )
        num_experts = int(weight.shape[0])
        num_groups = _moe_num_groups(first_token_offset, num_experts)
        _initialize_reduction_outputs(self.chain, outputs, stream)
        # Permute to the kernel's (S,K,1)/(N,K,E)/(S,N,1) layout.
        a_perm = token.permute(1, 2, 0)
        b_perm = weight.permute(1, 2, 0)
        c_perms = [t.permute(1, 2, 0) for t in outputs]
        dense_bad = self.chain.output_specs and _moe_dense_layout_bad(outputs[0])
        if _moe_operand_layout_bad(self.chain, token, weight) or dense_bad:
            raise ValueError(
                "MoE non-packed tensors require contiguous innermost dimensions: "
                f"got token stride {tuple(token.stride())}, "
                f"weight stride {tuple(weight.stride())}, "
                f"output stride {tuple(outputs[0].stride())}"
            )
        output_strides = tuple(stride for _spec, ci in zip(outputs_spec, c_perms) for stride in ci.stride())
        problem_size = (
            S,
            N,
            K,
            num_experts,
            num_groups,
            *tuple(a_perm.stride()),
            *tuple(b_perm.stride()),
            *output_strides,
        )
        a = _maybe_wrap_layout(a_perm, _LEADING_DIM_A)
        b = _maybe_wrap_layout(b_perm, _LEADING_DIM_B)
        cs = [
            (_wrap_raw_tensor(ci) if (spec.is_reduction or spec.is_quant_scale) else _maybe_wrap_layout(ci, _LEADING_DIM_C))
            for spec, ci in zip(outputs_spec, c_perms)
        ]
        # Tensormap workspace: one 128-byte slot per CTA per patched descriptor.
        workspace = self._make_workspace(self._grid_ctas * self._desc_slots_per_cta + _MOE_SCHED_COUNTER_SLOTS, workspace)
        _initialize_combine_output(self.chain, outputs, stream)
        return self._launchable(
            problem_size,
            first_token_offset,
            workspace,
            *((top_k_scores,) if top_k_scores is not None else ()),
            *((token_ks,) if token_ks is not None else ()),
            *((token_index,) if token_index is not None else ()),
            a,
            b,
            *_moe_launch_tail(cs, tma_slots=self.tma_slots),
            stream=_as_custream(stream),
        )

    def _call_variant_pack(self, variant_pack: dict, workspace=None, stream=None):
        a_bufs, b_bufs, out_bufs, aux_bufs, fto, _sfa, _sfb, token_index, token_ks, top_k_scores, snk = _resolve_moe_variant_pack(self, variant_pack)
        _out_reqs = _output_align_reqs(self.chain, self.tma_slots, vec_bytes=self.vec_bytes_epi)
        _aux_reqs = _aux_align_reqs(self.chain, vec_bytes=self.vec_bytes_epi)
        _named = (
            [("token", x, 16, "ptr") for x in a_bufs]
            + [("weight", x, 16, "ptr") for x in b_bufs]
            + [(f"output[{i}]", x, _out_reqs[i], "full") for i, x in enumerate(out_bufs)]
            + [(f"aux {self.chain.aux_tensors[i].name!r}", x, _aux_reqs[self.chain.aux_tensors[i].name], "full") for i, x in enumerate(aux_bufs)]
        )
        _r = _alignment_reject(_named)
        if _r is not None:
            raise ValueError(_r)
        if isinstance(self.chain.moe, MoeSwapAbSpec):
            return _launch_moe_swap_ab(
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
        """Multi-GEMM MoE call:
        ``compiled([(tok, w0), (tok, w1), ...], fto, out, (S, N, K[, ...]), *aux)``.

        (token, weight) pairs deduped by tensor identity into the JIT-fixed A/B
        slots (shared token → one A operand). All matmuls share ``fto``; ``out``
        is the single fused output."""
        first_token_offset = _offset_vector(first_token_offset)
        chain = self.chain
        if not isinstance(gemm_pairs, (list, tuple)) or not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in gemm_pairs):
            raise ValueError("multi-GEMM MoE call expects a list of (token, weight) pairs as " f"the first argument; got {type(gemm_pairs).__name__}")
        if len(gemm_pairs) != chain.num_gemms:
            raise ValueError(f"this graph has {chain.num_gemms} grouped matmul(s); got " f"{len(gemm_pairs)} (token, weight) pair(s)")
        if len(snke) < 3:
            raise ValueError(f"MoE call needs problem_size (S, N, K[, ...]); got {snke!r}")
        S, N, K = int(snke[0]), int(snke[1]), int(snke[2])

        na, nb = chain.num_a_operands, chain.num_b_operands
        a_slots: list = [None] * na
        b_slots: list = [None] * nb
        for (tok, w), (ai, bi) in zip(gemm_pairs, chain.gemm_operands):
            for slots, idx, t, role in (
                (a_slots, ai, tok, "token"),
                (b_slots, bi, w, "weight"),
            ):
                if slots[idx] is None:
                    slots[idx] = t
                elif slots[idx].data_ptr() != t.data_ptr():
                    raise ValueError(
                        f"multi-GEMM MoE operand sharing mismatch: distinct {role} "
                        f"slot {idx} was given two different tensors. The runtime "
                        "sharing pattern must match the compiled graph."
                    )
        if any(s is None for s in a_slots) or any(s is None for s in b_slots):
            raise ValueError("multi-GEMM MoE: not every distinct token/weight slot was filled")

        outputs_spec = chain.outputs
        outs = list(output) if isinstance(output, (list, tuple)) else [output]
        if len(outs) != len(outputs_spec):
            raise ValueError(f"fused multi-GEMM MoE has {len(outputs_spec)} output(s); got {len(outs)}")
        out = outs[0]
        for name, t, rank in (
            ("output", out, 3),
            ("first_token_offset", first_token_offset, 1),
        ):
            if len(t.shape) != rank:
                raise ValueError(f"MoE {name} must be rank-{rank}; got shape {tuple(t.shape)}")
        for role, slots, major in (
            ("token", a_slots, chain.matmul.a_major),
            ("weight", b_slots, chain.matmul.b_major),
        ):
            unit_dim = -1 if major == "k" else -2
            for t in slots:
                if len(t.shape) != 3 or t.stride(unit_dim) != 1:
                    raise ValueError(
                        f"multi-GEMM MoE {role} must be rank-3 and contiguous "
                        f"along its {major}-major dimension; got shape "
                        f"{tuple(t.shape)} stride {tuple(t.stride())}"
                    )
        if _moe_dense_layout_bad(out):
            raise ValueError("multi-GEMM MoE output requires contiguous innermost dim")
        for spec, ci in zip(outputs_spec, outs):
            if len(ci.shape) != 3 or tuple(ci.shape) != _expected_output_shape(spec, chain, (S, N, K), top_k_scores):
                raise ValueError(
                    f"multi-GEMM MoE output {spec.source!r} must have shape "
                    f"{_expected_output_shape(spec, chain, (S, N, K), top_k_scores)}; got {tuple(ci.shape)}"
                )
        num_experts = int(b_slots[0].shape[0])
        num_groups = _moe_num_groups(first_token_offset, num_experts)
        _initialize_reduction_outputs(chain, outs, stream)
        a_stride_perms = [t.permute(1, 2, 0) for t in a_slots]
        b_stride_perms = [t.permute(1, 2, 0) for t in b_slots]
        c_perms = [ci.permute(1, 2, 0) for ci in outs]
        output_strides = tuple(stride for _spec, ci in zip(outputs_spec, c_perms) for stride in ci.stride())
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
        a_wrapped = [_maybe_wrap_layout(t.permute(1, 2, 0), _LEADING_DIM_A) for t in a_slots]
        b_wrapped = [_maybe_wrap_layout(t.permute(1, 2, 0), _LEADING_DIM_B) for t in b_slots]
        cs = [
            (_wrap_raw_tensor(ci) if (spec.is_reduction or spec.is_quant_scale) else _maybe_wrap_layout(ci, _LEADING_DIM_C))
            for spec, ci in zip(outputs_spec, c_perms)
        ]
        for ref, t in zip(chain.aux_tensors, aux):
            if ref.grouped_by_moe and (len(t.shape) != 3 or int(t.shape[0]) != num_groups):
                raise ValueError(
                    f"per-group aux {ref.name!r} must be rank-3 with leading dim " f"{num_groups} (the number of groups); got shape {tuple(t.shape)}"
                )
        aux = tuple(_maybe_wrap_layout(_reshape_aux_to_fake(t, ref), _LEADING_DIM_AUX) for ref, t in zip(chain.aux_tensors, aux))
        # Workspace: one 128-B tensormap slot per patched descriptor per CTA.
        workspace = self._make_workspace(self._grid_ctas * self._desc_slots_per_cta + _MOE_SCHED_COUNTER_SLOTS, workspace)
        _initialize_combine_output(chain, outs, stream)
        return self._launchable(
            problem_size,
            first_token_offset,
            workspace,
            *((top_k_scores,) if top_k_scores is not None else ()),
            *((token_ks,) if token_ks is not None else ()),
            *((token_index,) if token_index is not None else ()),
            *a_wrapped,
            *b_wrapped,
            *_moe_launch_tail(cs, aux, tma_slots=self.tma_slots),
            stream=_as_custream(stream),
        )


def _moe_launch_tail(cs, aux=(), *, tma_slots: "frozenset[int]" = frozenset()) -> tuple:
    """Outputs + auxes in the host signature's order -- TAPS, AUX, then the
    trailing TMA-C slots in slot order. An output on the TMA surface binds a
    TMA-only parameter and so goes LAST; every other one rides a tap slot. MoE
    has no recipe (``_check_executable`` returns early for it), so this is the
    only place that order is written down."""
    cs = list(cs)
    taps = [c for i, c in enumerate(cs) if i not in tma_slots]
    tmas = [cs[i] for i in sorted(tma_slots) if i < len(cs)]
    return (*taps, *aux, *tmas)


def _launch_moe_swap_ab(
    compiled, weights, tokens, outputs, aux, offsets, weight_sf, token_sf, mnk, workspace, stream, token_index=None, token_ks=None, top_k_scores=None
):
    chain, cfg = compiled.chain, compiled.config
    m, n, k = map(int, mnk)
    e = int(weights[0].shape[0])
    offsets = _offset_vector(offsets)
    g = _moe_num_groups(offsets, e)
    if chain.quants and (m, n) != (chain.matmul.M, chain.matmul.N):
        runtime_chain = replace(chain, matmul=replace(chain.matmul, M=m, N=n))
        try:
            _check_block_quant_supported(runtime_chain, _epi_vec_bytes(chain, cfg), cfg)
        except NotImplementedError as exc:
            raise ValueError(str(exc)) from exc
    if n % math.gcd(_epi_vec_bytes(chain, cfg) // DTYPE_BYTES[chain.output_dtype], chain.moe.offset_multiple, chain.matmul.N):
        raise ValueError("MoE weight-by-token runtime S violates the compiled STG chunk alignment")
    for quant in chain.quants:
        if (m if quant.axis == 1 else n) % quant.block_size:
            raise ValueError("MoE weight-by-token quantization axis must contain whole runtime blocks")
    for spec, out in zip(chain.outputs, outputs):
        expected = _expected_output_shape(spec, chain, (m, n, k), top_k_scores)
        if tuple(out.shape) != expected:
            raise ValueError(f"MoE weight-by-token output {spec.source!r} must have lowered shape {expected}; got {tuple(out.shape)}")
        if not spec.is_reduction and not spec.is_quant_scale and out.stride(-1 if spec.major == "n" else -2) != 1:
            raise ValueError("MoE weight-by-token output must match the lowered graph's [1,N,S] shape and major")
        if spec.is_quant_scale:
            qi = int(spec.source.rsplit("_", 1)[1])
            if chain.quants[qi].scale_reorder == "F8_128x4":
                reason = _sf_blob_reject([(f"quant scale output[{qi}]", out, math.prod(expected) * DTYPE_BYTES[spec.dtype])])
                if reason is not None:
                    raise ValueError(reason)
    scale_blob_reason = _grouped_row_quant_scale_blob_reject(chain, outputs, m, n, g)
    if scale_blob_reason is not None:
        raise ValueError(scale_blob_reason)
    for i, spec in enumerate(chain.output_specs):
        if i in compiled.tma_slots and spec.major == "n" and n * DTYPE_BYTES[spec.dtype] % 16:
            raise ValueError("MoE weight-by-token runtime S violates the TMA token boundary alignment")
    token_rows = tokens[0].shape[1] if token_index is not None else n
    for tok in tokens:
        if len(tok.shape) != 3 or tuple(tok.shape[:2]) != (1, token_rows) or int(tok.shape[2]) * (2 if chain.matmul.b_dtype == "fp4_e2m1" else 1) != k:
            raise ValueError("MoE weight-by-token token operands must share [1,S,K]")
    for weight in weights:
        if tuple(weight.shape[:2]) != (e, m) or int(weight.shape[2]) * (2 if chain.matmul.a_dtype == "fp4_e2m1" else 1) != k:
            raise ValueError("MoE weight-by-token weight operands must share [E,N,K] in kernel order")
    if any(_moe_operand_layout_bad(chain, w, t) for t in tokens for w in weights):
        raise ValueError("MoE weight-by-token operands must have the graph's contiguous major dimension")
    for ref, buf in zip(chain.aux_tensors, aux):
        if ref.grouped_by_moe and (len(buf.shape) != 3 or int(buf.shape[0]) != g):
            raise ValueError(f"per-group aux {ref.name!r} must have leading dimension {g}")
    a = [w.permute(1, 2, 0) for w in weights]
    b = [t.permute(1, 2, 0) for t in tokens]
    c = [out.permute(1, 2, 0) for out in outputs]
    problem = (m, n, k, e, g, *(x for buf in a + b + c for x in buf.stride()))
    wrap = _maybe_wrap_layout
    a = [wrap(buf, _LEADING_DIM_A) for buf in a]
    b = [wrap(buf, _LEADING_DIM_B) for buf in b]
    c = [_wrap_raw_tensor(buf) if spec.is_reduction or spec.is_quant_scale else wrap(buf, _LEADING_DIM_C) for spec, buf in zip(chain.outputs, c)]
    sf = [wrap(buf.permute(1, 2, 0), _LEADING_DIM_AUX) for buf in weight_sf + token_sf]
    aux = [wrap(_reshape_aux_to_fake(buf, ref), _LEADING_DIM_AUX) for ref, buf in zip(chain.aux_tensors, aux)]
    slots = compiled._grid_ctas * compiled._desc_slots_per_cta
    workspace = compiled._make_workspace(slots + _MOE_SCHED_COUNTER_SLOTS, workspace)
    _initialize_reduction_outputs(chain, outputs, stream)
    _initialize_combine_output(chain, outputs, stream)
    return compiled._launchable(
        problem,
        offsets,
        workspace,
        *((top_k_scores,) if top_k_scores is not None else ()),
        *((token_ks,) if token_ks is not None else ()),
        *((token_index,) if token_index is not None else ()),
        *a,
        *b,
        *sf,
        *_moe_launch_tail(c, aux, tma_slots=compiled.tma_slots),
        stream=_as_custream(stream),
    )


def _jit_moe(
    chain: FusionChain,
    config: TileConfig,
    *,
    binding: "GemmBinding | None" = None,
) -> CompiledMoeGemm:
    """JIT path for a MoE grouped matmul forward pass (NONE or GATHER)."""
    _precheck_moe(chain, config)
    store_modes = _store_modes(chain, config)
    use_tma = "tma" in store_modes
    snippets = generate(
        chain,
        vec_bytes_epi=_epi_chunk_bytes(chain, config, use_tma),
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        tma_slots=frozenset(i for i, m in enumerate(store_modes) if m == "tma"),
        packed_lanes=_epi_packed_lanes(config),
    )
    src = _render_template(chain, snippets, config)
    mod = _import_kernel(src)
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    cluster_m, cluster_n = config.cga_size_m, config.cga_size_n
    grid_ctas = _grid_num_clusters(config) * cluster_m * cluster_n
    token_operands = chain.num_b_operands if isinstance(chain.moe, MoeSwapAbSpec) else chain.num_a_operands
    return CompiledMoeGemm(
        chain=chain,
        config=config,
        device=_plan_device(),
        generated_path=_cache_dir() / f"gen_{digest}" / "generated_kernel.py",
        _launchable=mod.compile(),
        _grid_ctas=grid_ctas,
        aux_names=[aux.name for aux in chain.aux_tensors],
        binding=binding,
        vec_bytes_epi=_epi_chunk_bytes(chain, config, use_tma),
        _desc_slots_per_cta=token_operands + len([m for m in store_modes if m == "tma"]),
        store_modes=store_modes,
        use_tma_store=use_tma,
    )


def _jit_block_scale(
    chain: FusionChain,
    config: TileConfig,
    *,
    binding: "GemmBinding | None" = None,
) -> CompiledFusedGemm:
    """JIT path for block-scaled (FP4 / FP8 + per-block SF) matmul.

    Bypasses the generic dtype-byte checks (FP4 is 0.5 B/elem); routes to the
    block-scale template. (config, block_size) validation happens in the
    tile-constant renderer (``validate_block_scale_config``)."""
    # Exact per-side match against the supported cases (+ arch); subsumes the
    # both-sided requirement (single-sided matches no case).
    from ..kernel_registry import select_template

    _precheck_block_scale(chain, config)
    store_modes = _store_modes(chain, config)
    use_tma = _use_tma_store_epi(chain, config)
    splitk = config.split_k_slices > 1
    snippets = generate(
        chain,
        vec_bytes_epi=_epi_chunk_bytes(chain, config, use_tma),
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        tma_slots=frozenset({0}) if (splitk and use_tma) else frozenset(i for i, m in enumerate(store_modes) if m == "tma"),
        packed_lanes=_epi_packed_lanes(config),
        split_k_slices=config.split_k_slices,
    )
    src = _render_block_scale_template(chain, snippets, config)
    mod = _import_kernel(src)
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    return CompiledFusedGemm(
        chain=chain,
        config=config,
        device=_plan_device(),
        aux_names=[aux.name for aux in chain.aux_tensors],
        generated_path=_cache_dir() / f"gen_{digest}" / "generated_kernel.py",
        _launchable=mod.compile(),
        store_modes=store_modes,
        use_tma_store=use_tma,
        block_scale=True,
        binding=binding,
        vec_bytes_epi=_out_vec_bytes(chain, config, use_tma),
        workspace_bytes=_splitk_workspace_bytes(chain, config),
    )


@dataclass
class CompiledMoeBlockScaleGemm:
    """A compiled MoE grouped *block-scale* (FP4/FP8 + per-block SF) matmul fwd.

    Layouts (rank-3):
      * ``token``  — (1, S, Kp) packed FP4/FP8 (A).
      * ``weight`` — (E, N, Kp) packed FP4/FP8 (B, per-expert).
      * ``sfa``    — token SF, F8_128x4-reordered + padded to 128 rows PER GROUP,
        then concatenated (Σ ceil(group_m/128) blocks).
      * ``sfb``    — weight SF, F8_128x4-reordered dense byte planes with E
        outermost; the runtime expert stride is 16-byte aligned and may include padding.
      * ``first_token_offset`` — (num_groups+1,) int32/int64; group g spans token
        rows ``[fto[g], fto[g+1])``. Group sizes arbitrary (NOT
        128-aligned); the scheduler tracks each group's start SF-block.
      * ``output`` — (1, S, N) row-major.

    The per-CTA A-descriptor workspace is allocated/owned here; its size follows
    the FIXED persistent grid (shape-independent), so one allocation serves every
    problem size (override-shape needs no workspace accounting)."""

    chain: FusionChain
    config: TileConfig
    generated_path: Path
    _launchable: Callable
    _grid_ctas: int = 0
    device: int = 0  # CUDA device this plan's baked constants describe
    aux_names: list = field(default_factory=list)
    binding: "GemmBinding | None" = None  # role -> cuDNN tensor (variant-pack call)
    # The epilogue chunk width (bytes) the kernel was RENDERED with (tile-
    # clamped); drives the runtime output/aux alignment requirements.
    vec_bytes_epi: "int | None" = None
    # Tensormap slots each CTA patches: one per distinct A operand, one per SFA
    # (its base carries start_sf_block_m and its m extent bounds the group), plus
    # the output descriptor when the TMA-store epilogue re-dimensions it per group.
    _desc_slots_per_cta: int = 0
    store_modes: tuple = ()
    use_tma_store: bool = False
    accepts_stream: ClassVar[bool] = True  # stream-aware dispatch (see CompiledFusedGemm)

    @property
    def tma_slots(self) -> "frozenset[int]":
        """Which dense output slots ride the TMA-C surface."""
        return frozenset(i for i, m in enumerate(self.store_modes) if m == "tma")

    @property
    def workspace_bytes(self) -> int:
        """Per-CTA scratch for dynamic A, real-SFA, and output tensormaps.

        A fake SFA needs no slot. The persistent grid is shape-independent, so
        this is constant for the plan and override-shape needs no re-query.
        """
        return (self._grid_ctas * self._desc_slots_per_cta + _MOE_SCHED_COUNTER_SLOTS) * _MOE_DESC_SLOT_BYTES

    def _make_workspace(self, n_slots, caller=None):
        """The per-CTA dynamic-descriptor workspace (16 int64/slot, 128-byte
        aligned). ``n_slots`` covers every dynamic descriptor. Carved from the
        CALLER's buffer when execute() supplied one; otherwise from one this plan
        owns (the direct jit_from_cudnn_graph path passes no workspace)."""
        if caller is None:
            raise ValueError(
                f"{type(self).__name__} requires a {n_slots * _MOE_DESC_SLOT_BYTES}-byte workspace but execute() received "
                "none; size it with workspace_bytes and pass workspace= (a plan owns no device memory, Rule 8)"
            )
        return _moe_carve_workspace(caller, n_slots, type(self).__name__)

    def __call__(self, variant_pack, workspace=None, stream=None):
        # Variant-pack dict {cuDNN tensor | uid | name: buffer}; (S, N, K) is
        # inferred from buffer shapes.
        if not isinstance(variant_pack, dict):
            raise TypeError(
                "compiled kernels are called with a variant-pack dict " "{cuDNN tensor | uid | name: buffer}; got " f"{type(variant_pack).__name__}"
            )
        _check_plan_device(self.device)
        return self._call_variant_pack(variant_pack, workspace, stream)

    def _launch_single(
        self, token, weight, sfa, sfb, first_token_offset, output, snke, workspace=None, stream=None, token_index=None, token_ks=None, top_k_scores=None
    ):
        first_token_offset = _offset_vector(first_token_offset)
        if len(snke) < 3:
            raise ValueError("MoE block-scale call needs problem_size (S, N, K[, ...]); " f"got {snke!r}")
        S, N, K = int(snke[0]), int(snke[1]), int(snke[2])
        outputs_spec = self.chain.outputs
        outputs = list(output) if isinstance(output, (list, tuple)) else [output]
        if len(outputs) != len(outputs_spec):
            raise ValueError(
                f"this MoE block-scale graph has {len(outputs_spec)} output(s) "
                f"({[o.source for o in outputs_spec]}); got {len(outputs)}. "
                "Pass outputs in slot order."
            )
        ranked_inputs = [
            ("token", token, 3),
            ("weight", weight, 3),
            ("first_token_offset", first_token_offset, 1),
        ]
        if sfa is not None:
            ranked_inputs.append(("sfa", sfa, 3))
        if sfb is not None:
            ranked_inputs.append(("sfb", sfb, 3))
        for name, t, rank in ranked_inputs:
            if len(t.shape) != rank:
                raise ValueError(f"MoE block-scale {name} must be rank-{rank}; " f"got shape {tuple(t.shape)}")
        for spec, t in zip(outputs_spec, outputs):
            if len(t.shape) != 3 or tuple(t.shape) != _expected_output_shape(spec, self.chain, (S, N, K), top_k_scores):
                raise ValueError(
                    f"MoE block-scale output {spec.source!r} must have shape "
                    f"{_expected_output_shape(spec, self.chain, (S, N, K), top_k_scores)}; "
                    f"got {tuple(t.shape)}"
                )
        num_experts = int(weight.shape[0])
        num_groups = _moe_num_groups(first_token_offset, num_experts)
        scale_blob_reason = _grouped_row_quant_scale_blob_reject(
            self.chain,
            outputs,
            S,
            N,
            num_groups,
        )
        if scale_blob_reason is not None:
            raise ValueError(scale_blob_reason)
        _initialize_reduction_outputs(self.chain, outputs, stream)
        # Permute to inner-plane layouts (batch last). SF descriptors rebuild
        # blocked row/K geometry and retain the validated weight expert stride.
        a_perm = token.permute(1, 2, 0)
        b_perm = weight.permute(1, 2, 0)
        c_perms = [t.permute(1, 2, 0) for t in outputs]
        dense_bad = self.chain.output_specs and _moe_dense_layout_bad(outputs[0])
        if _moe_operand_layout_bad(self.chain, token, weight) or dense_bad:
            raise ValueError(
                "MoE block-scale non-packed tensors require contiguous innermost "
                f"dimensions: got token stride {tuple(token.stride())}, "
                f"weight stride {tuple(weight.stride())}, "
                f"output stride {tuple(outputs[0].stride())}"
            )
        output_strides = tuple(stride for _spec, ci in zip(outputs_spec, c_perms) for stride in ci.stride())
        problem_size = (
            S,
            N,
            K,
            num_experts,
            num_groups,
            *tuple(a_perm.stride()),
            *tuple(b_perm.stride()),
            *output_strides,
        )
        a = _maybe_wrap_layout(a_perm, _LEADING_DIM_A)
        b = _maybe_wrap_layout(b_perm, _LEADING_DIM_B)
        cs = [
            (_wrap_raw_tensor(ci) if (spec.is_reduction or spec.is_quant_scale) else _maybe_wrap_layout(ci, _LEADING_DIM_C))
            for spec, ci in zip(outputs_spec, c_perms)
        ]
        sf_args = []
        if sfa is not None:
            sf_args.append(_maybe_wrap_layout(sfa.permute(1, 2, 0), _LEADING_DIM_AUX))
        if sfb is not None:
            sf_args.append(_maybe_wrap_layout(sfb.permute(1, 2, 0), _LEADING_DIM_AUX))
        workspace = self._make_workspace(self._grid_ctas * self._desc_slots_per_cta + _MOE_SCHED_COUNTER_SLOTS, workspace)
        _initialize_combine_output(self.chain, outputs, stream)
        return self._launchable(
            problem_size,
            first_token_offset,
            workspace,
            *((top_k_scores,) if top_k_scores is not None else ()),
            *((token_ks,) if token_ks is not None else ()),
            *((token_index,) if token_index is not None else ()),
            a,
            b,
            *sf_args,
            *_moe_launch_tail(cs, tma_slots=self.tma_slots),
            stream=_as_custream(stream),
        )

    def _call_variant_pack(self, variant_pack: dict, workspace=None, stream=None):
        a_bufs, b_bufs, out_bufs, aux_bufs, fto, sfa, sfb, token_index, token_ks, top_k_scores, snk = _resolve_moe_variant_pack(self, variant_pack)
        _out_reqs = _output_align_reqs(self.chain, self.tma_slots, vec_bytes=self.vec_bytes_epi)
        _aux_reqs = _aux_align_reqs(self.chain, vec_bytes=self.vec_bytes_epi)
        _named = (
            [("token", x, 16, "ptr") for x in a_bufs]
            + [("weight", x, 16, "ptr") for x in b_bufs]
            + [(f"output[{i}]", x, _out_reqs[i], "full") for i, x in enumerate(out_bufs)]
            + [(f"aux {self.chain.aux_tensors[i].name!r}", x, _aux_reqs[self.chain.aux_tensors[i].name], "full") for i, x in enumerate(aux_bufs)]
            + [("SFA", x, 16, "ptr") for x in (sfa or [])]
            + [("SFB", x, 16, "ptr") for x in (sfb or [])]
        )
        _r = _alignment_reject(_named)
        if _r is not None:
            raise ValueError(_r)
        if sfa or sfb:
            _S, _N, _K = snk[0], snk[1], snk[2]
            token_sf, weight_sf, weights = sfa, sfb, b_bufs
            if isinstance(self.chain.moe, MoeSwapAbSpec):
                _S, _N = _N, _S
                token_sf, weight_sf, weights = sfb, sfa, a_bufs
            if self.chain.moe.mode == "gather":
                from ..kernel_registry import linear_token_sf_reject

                source_rows = (b_bufs if isinstance(self.chain.moe, MoeSwapAbSpec) else a_bufs)[0].shape[1]
                for sf in token_sf:
                    reason = linear_token_sf_reject(sf.shape, sf.stride(), source_rows, _K // self.chain.block_scale.block_size)
                    if reason is not None:
                        raise ValueError(reason)
                token_sf = []  # Linear inputs do not use segmented F8_128x4 capacity.
            _sf_k4 = ((_K // self.chain.block_scale.block_size) + 3) // 4
            _segmented_sfa_rows = segmented_row_scale_capacity_rows(int(_S), _moe_num_groups(fto, int(weights[0].shape[0])))
            _r_sf = _sf_blob_reject([(f"token SF[{i}]", x, 4 * _sf_k4 * _segmented_sfa_rows) for i, x in enumerate(token_sf or [])]) or moe_weight_sf_reject(
                [(f"weight SF[{j}]", x, 512 * _sf_k4 * ((_N + 127) // 128), int(weights[j].shape[0])) for j, x in enumerate(weight_sf or [])]
            )
            if _r_sf is not None:
                raise ValueError(_r_sf)
        if isinstance(self.chain.moe, MoeSwapAbSpec):
            return _launch_moe_swap_ab(
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
            pairs = [
                (
                    (a_bufs[ai], None if fake_a else sfa[ai]),
                    (b_bufs[bi], None if fake_b else sfb[bi]),
                )
                for ai, bi in self.chain.gemm_operands
            ]
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
        """Multi-GEMM MoE block-scale call:
        ``compiled([((tok,sfa),(w0,sfb0)), ((tok,sfa),(w1,sfb1)), ...], fto,
        out, (S, N, K[, ...]), *aux)``.

        Each GEMM is a ((token, sfa), (weight, sfb)) pair; dedup by PACKED-data
        identity (SF travels with its data → shared token+sfa collapses to one A
        operand). All matmuls share ``fto``; ``out`` is the fused output."""
        first_token_offset = _offset_vector(first_token_offset)
        chain = self.chain
        ok = (
            isinstance(gemm_pairs, (list, tuple))
            and gemm_pairs
            and all(isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(o, (list, tuple)) and len(o) == 2 for o in p) for p in gemm_pairs)
        )
        if not ok:
            raise ValueError("multi-GEMM MoE block-scale call expects a list of " "((token,sfa),(weight,sfb)) pairs as the first argument")
        if len(gemm_pairs) != chain.num_gemms:
            raise ValueError(f"this graph has {chain.num_gemms} grouped matmul(s); got " f"{len(gemm_pairs)} pair(s)")
        if len(snke) < 3:
            raise ValueError(f"MoE block-scale call needs problem_size (S, N, K[, ...]); got {snke!r}")
        S, N, K = int(snke[0]), int(snke[1]), int(snke[2])

        na, nb = chain.num_a_operands, chain.num_b_operands
        a_slots: list = [None] * na  # (packed token, sfa)
        b_slots: list = [None] * nb  # (packed weight, sfb)
        for ((tok, sfa), (w, sfb)), (ai, bi) in zip(gemm_pairs, chain.gemm_operands):
            for slots, idx, data, sf, role in (
                (a_slots, ai, tok, sfa, "token"),
                (b_slots, bi, w, sfb, "weight"),
            ):
                if slots[idx] is None:
                    slots[idx] = (data, sf)
                elif slots[idx][0].data_ptr() != data.data_ptr():
                    raise ValueError(f"multi-GEMM MoE block-scale {role} sharing mismatch: distinct " f"slot {idx} was given two different packed tensors.")
        if any(s is None for s in a_slots) or any(s is None for s in b_slots):
            raise ValueError("multi-GEMM MoE block-scale: not every distinct operand slot was filled")

        outputs_spec = chain.outputs
        outs = list(output) if isinstance(output, (list, tuple)) else [output]
        if len(outs) != len(outputs_spec):
            raise ValueError(f"fused multi-GEMM MoE block-scale has {len(outputs_spec)} output(s); got {len(outs)}")
        out = outs[0]
        for name, t, rank in (
            ("output", out, 3),
            ("first_token_offset", first_token_offset, 1),
        ):
            if len(t.shape) != rank:
                raise ValueError(f"MoE block-scale {name} must be rank-{rank}; got {tuple(t.shape)}")
        for spec, ci in zip(outputs_spec, outs):
            if len(ci.shape) != 3 or tuple(ci.shape) != _expected_output_shape(spec, chain, (S, N, K), top_k_scores):
                raise ValueError(
                    f"multi-GEMM MoE block-scale output {spec.source!r} must have shape "
                    f"{_expected_output_shape(spec, chain, (S, N, K), top_k_scores)}; got {tuple(ci.shape)}"
                )
        num_experts = int(b_slots[0][0].shape[0])
        num_groups = _moe_num_groups(first_token_offset, num_experts)
        scale_blob_reason = _grouped_row_quant_scale_blob_reject(
            chain,
            outs,
            S,
            N,
            num_groups,
        )
        if scale_blob_reason is not None:
            raise ValueError(scale_blob_reason)
        _initialize_reduction_outputs(chain, outs, stream)
        a0, b0 = a_slots[0][0], b_slots[0][0]
        a_stride_perms = [t.permute(1, 2, 0) for (t, _sf) in a_slots]
        b_stride_perms = [t.permute(1, 2, 0) for (t, _sf) in b_slots]
        c_perms = [ci.permute(1, 2, 0) for ci in outs]
        dense_bad = chain.output_specs and _moe_dense_layout_bad(out)
        if _moe_operand_layout_bad(chain, a0, b0) or dense_bad:
            raise ValueError("multi-GEMM MoE block-scale tensors require contiguous innermost dim")
        output_strides = tuple(stride for _spec, ci in zip(outputs_spec, c_perms) for stride in ci.stride())
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
        # Grouped by kind (all A, all B, all SFA, all SFB); single-GEMM → a,b,sfa,sfb.
        a_wrapped = [_maybe_wrap_layout(t.permute(1, 2, 0), _LEADING_DIM_A) for (t, _sf) in a_slots]
        b_wrapped = [_maybe_wrap_layout(t.permute(1, 2, 0), _LEADING_DIM_B) for (t, _sf) in b_slots]
        sfa_wrapped = [_maybe_wrap_layout(sf.permute(1, 2, 0), _LEADING_DIM_AUX) for (_t, sf) in a_slots if sf is not None]
        sfb_wrapped = [_maybe_wrap_layout(sf.permute(1, 2, 0), _LEADING_DIM_AUX) for (_t, sf) in b_slots if sf is not None]
        cs = [
            (_wrap_raw_tensor(ci) if (spec.is_reduction or spec.is_quant_scale) else _maybe_wrap_layout(ci, _LEADING_DIM_C))
            for spec, ci in zip(outputs_spec, c_perms)
        ]
        for ref, t in zip(chain.aux_tensors, aux):
            if ref.grouped_by_moe and (len(t.shape) != 3 or int(t.shape[0]) != num_groups):
                raise ValueError(
                    f"per-group aux {ref.name!r} must be rank-3 with leading dim " f"{num_groups} (the number of groups); got shape {tuple(t.shape)}"
                )
        aux = tuple(_maybe_wrap_layout(_reshape_aux_to_fake(t, ref), _LEADING_DIM_AUX) for ref, t in zip(chain.aux_tensors, aux))
        workspace = self._make_workspace(self._grid_ctas * self._desc_slots_per_cta + _MOE_SCHED_COUNTER_SLOTS, workspace)
        _initialize_combine_output(chain, outs, stream)
        return self._launchable(
            problem_size,
            first_token_offset,
            workspace,
            *((top_k_scores,) if top_k_scores is not None else ()),
            *((token_ks,) if token_ks is not None else ()),
            *((token_index,) if token_index is not None else ()),
            *a_wrapped,
            *b_wrapped,
            *sfa_wrapped,
            *sfb_wrapped,
            *_moe_launch_tail(cs, aux, tma_slots=self.tma_slots),
            stream=_as_custream(stream),
        )


def _jit_moe_block_scale(
    chain: FusionChain,
    config: TileConfig,
    *,
    binding: "GemmBinding | None" = None,
) -> CompiledMoeBlockScaleGemm:
    """JIT path for a MoE grouped block-scale matmul (dequant + moe_grouped).

    Block-scale SF machinery + MoE grouped persistent scheduler + per-group A
    TMA descriptor replacement. STG epilogue only."""
    from ..kernel_registry import (
        GraphType,
        mma_arch_reject,
        select_template,
    )

    _precheck_moe_block_scale(chain, config)
    _tmpl = select_template(chain, config)
    store_modes = _store_modes(chain, config)
    use_tma = "tma" in store_modes
    snippets = generate(
        chain,
        vec_bytes_epi=_epi_chunk_bytes(chain, config, use_tma),
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        tma_slots=frozenset(i for i, m in enumerate(store_modes) if m == "tma"),
        packed_lanes=_epi_packed_lanes(config),
    )
    src = _render_block_scale_template(chain, snippets, config)
    mod = _import_kernel(src)
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    cluster_m, cluster_n = config.cga_size_m, config.cga_size_n
    grid_ctas = _grid_num_clusters(config) * cluster_m * cluster_n
    token_operands = chain.num_b_operands if isinstance(chain.moe, MoeSwapAbSpec) else chain.num_a_operands
    return CompiledMoeBlockScaleGemm(
        chain=chain,
        config=config,
        device=_plan_device(),
        generated_path=_cache_dir() / f"gen_{digest}" / "generated_kernel.py",
        _launchable=mod.compile(),
        _grid_ctas=grid_ctas,
        binding=binding,
        vec_bytes_epi=_epi_chunk_bytes(chain, config, use_tma),
        _desc_slots_per_cta=token_operands
        + (0 if (chain.block_scale.fake_dequant_b if isinstance(chain.moe, MoeSwapAbSpec) else chain.block_scale.fake_dequant_a) else token_operands)
        + len([m for m in store_modes if m == "tma"]),
        store_modes=store_modes,
        use_tma_store=use_tma,
    )
