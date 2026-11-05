# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compiler driver: cudnn graph -> rendered GEMM kernel.py -> compiled callable."""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib.util
import logging
import math
import os
import re
import sys
from dataclasses import replace
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Callable, ClassVar

import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn
from cuda.bindings import driver as _cuda
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost import buffers
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import current_device
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.workspace import align_up

_LOG = logging.getLogger(__name__)

_FAMILY = "sm100"


def _check_own_family(tmpl) -> None:
    """Decline ``tmpl`` unless this tree's compiler renders its pipeline."""
    if tmpl.family != _FAMILY:
        raise NotImplementedError(
            f"template {tmpl.file} ({tmpl.pipeline} pipeline) is served by the {tmpl.family} arch tree, but this process "
            f"runs the {_FAMILY} tree (cudnn.gemm.frost.{_FAMILY}.compiler): run it on a GPU that tree serves, or set "
            f"CUDNN_FRONTEND_GEMM_ARCH_FAMILY={tmpl.family} before cudnn.gemm.frost.compiler is first imported"
        )


def _as_custream(stream):
    return _cuda.CUstream(stream or 0)


from ..dtypes import (
    DTYPE_BITS,
    DTYPE_BYTES,
    DTYPE_TO_CUTLASS,
    DTYPE_TO_MMA_KIND,
    MAX_EPI_CHUNK_ELEMS,
    _aux_align_reqs,
    _compute_output_vec_bytes,
    _output_align_reqs,
    _pow2_floor,
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
    _output_rule,
    build as build_recipe,
    check_alignment,
    check_shapes,
    check_split_k_shape,
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

from ..compiler import (
    _fallback_cache_dir,
    _usable_cache_dir,
    _cache_dir,
    _write_atomic,
    _import_kernel,
    _CompiledFusedGemm,
    _CompiledMoeGemm,
    _CompiledMoeBlockScaleGemm,
    _CTAMMA_SUFFIX,
    _INJECT_MARKER_LINE,
    _aux_fake_shape_code,
    _aux_can_use_explicit_fake_stride,
    _aux_fake_block,
    _aux_signature_block,
    _aux_call_block,
    _reduction_stride_kernel_params,
    _reduction_stride_host_unpack,
    _reduction_stride_host_unpack_from,
    _reduction_stride_names,
    _reduction_stride_host_pass,
    _splitk_epilogue_bindings,
    _reduction_stride_compile_decls,
    _reduction_stride_compile_symbols,
    _cvt_f32_to_fp8_scale_bits,
    _cvt_e5m3_bits_to_f32,
    _quant_device_imports,
    _template_stem,
    _resolve_path_blocks,
    _resolve_blocks,
    _replace_marker_lines,
    _MOE_SCHED_COUNTER_SLOTS,
    _MOE_DESC_SLOT_BYTES,
    _is_layout_ambiguous,
    _aux_fake_real_axes,
    _swap_mn_view,
    _offset_vector,
    _moe_num_groups,
    _expected_output_shape,
    _initialize_reduction_outputs,
    _tma_alignment_reject,
    _alignment_reject,
    _sf_blob_reject,
    _grouped_row_quant_scale_blob_reject,
    _moe_dense_layout_bad,
    _moe_operand_layout_bad,
    _declared_storage,
    _kernel_order,
    _check_moe_combine_n,
    _resolve_moe_variant_pack,
    _register_legacy_device_view_adapter,
    _moe_carve_workspace,
    _moe_launch_tail,
)

_TVM_FFI_OK = importlib.util.find_spec("tvm_ffi") is not None
_FROST_COMPILE_OPTIONS = "--enable-tvm-ffi" if _TVM_FFI_OK else ""


def _supports_gpu_arch_option() -> bool:
    """Check whether the DSL supports an explicit GPU target."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_state, cutedsl_too_old

    return not cutedsl_too_old(cutedsl_state()[1])


def _sm_target_string(arch: int) -> str:
    """``major*10+minor`` -> the cute ``sm_XXX[a]`` target, matching cutedsl's own
    ``detect_gpu_arch`` (the arch-specific ``a`` suffix appears from major >= 9)."""
    major, minor = divmod(arch, 10)
    return f"sm_{major}{minor}{'a' if major >= 9 else ''}"


def _frost_compile_options() -> str:
    """The ``cute.compile()`` options string for a build in the current ``build_device()`` scope."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import build_scope_device as _build_scope_device

    opts = [_FROST_COMPILE_OPTIONS] if _FROST_COMPILE_OPTIONS else []
    arch = _current_arch()  # scope-aware: the handle's GPU inside build_device()
    if arch is None:
        return " ".join(opts)  # render-only / no GPU visible
    if _supports_gpu_arch_option():
        opts.append(f"--gpu-arch {_sm_target_string(arch)}")
    elif _build_scope_device() is not None:
        major, minor = divmod(arch, 10)
        raise NotImplementedError(
            f"cudnn.frost: a handle-scoped build for sm_{major}{minor} needs an nvidia-cutlass-dsl "
            f"that pins the compile target (public >= 4.7 or an internal RC); this wheel cannot, so "
            f"the build cannot follow the handle. Upgrade cutedsl, or build with the handle's GPU "
            f"already current (no build_device scope)."
        )
    return " ".join(opts)


def _tma_c_plumbing(chain: FusionChain, tma_slots: "frozenset[int]" = frozenset({0}), *, splitk: bool = False) -> dict[str, str]:
    """Kernel / host / compile plumbing for the outputs on the TMA-C surface."""

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
    """Coordinate tuple matching the output TMA descriptor rank."""
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
    """One TMA-stored output's store stage: stage this lane's fragment into the shared ring slot, then issue from it."""
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
        assert 7 * (16 // elem_bytes) < atom_m
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
    """The TMA arm, unrolled once per TMA-stored output."""
    lines: list[str] = []
    if cfg.workspace_split_k:
        lines += _tma_store_one(chain, cfg, epi_n, 0, "fp32", "n")
    else:
        for j, (slot, dt) in enumerate(_tma_out_dtypes(chain, tma_slots)):
            lines += _tma_store_one(chain, cfg, epi_n, j, dt, chain.output_specs[slot].major)
    return "\n".join(lines) if lines else "pass"


def _place_tma_stores(epilogue: str, chain, cfg, tma_slots: "frozenset[int]", epi_n: int) -> tuple[str, str]:
    """Inline each multi-output TMA store at its fragment's last use."""
    # For splitK, kernel 1 only stores the fp32 partials, and kernel 2 handles the outputs.
    if cfg.workspace_split_k:
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
    """Build one C descriptor per TMA-stored output."""
    if cfg.workspace_split_k:
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


def _epi_tile_cols(
    config: TileConfig,
) -> int:
    """Return accumulator columns drained per MMA-M block."""
    cols = config.cta_tile_n
    if config.is_cta_pair_mma and config.mma_tile_m == 64:
        cols //= 2
    return cols


_EPI_CHUNK_ELEMS_BY_PIPELINE = {
    "sm100": MAX_EPI_CHUNK_ELEMS,
    "sm103": MAX_EPI_CHUNK_ELEMS,
    "sm120": 8,
}


def _epi_vec_bytes(
    chain: FusionChain,
    config: TileConfig,
) -> int:
    """Return the compiled epilogue chunk width in bytes."""
    vec = _compute_output_vec_bytes(chain, tile_cols=_epi_tile_cols(config))
    return min(vec, _EPI_CHUNK_ELEMS_BY_PIPELINE[config.pipeline] * DTYPE_BYTES[chain.output_dtype])


def _epi_store_dtype(chain: FusionChain, config: TileConfig) -> str:
    """Use FP32 for workspace partials and the graph output dtype for final stores."""
    return "fp32" if config.workspace_split_k else chain.output_dtype


def _epi_chunk_elems(chain: FusionChain, config: TileConfig, use_tma_store: bool) -> int:
    if not use_tma_store:
        return _epi_vec_bytes(chain, config) // DTYPE_BYTES[chain.output_dtype]
    return _epi_n_for_chain(config, chain)


def _epi_chunk_bytes(chain: FusionChain, config: TileConfig, use_tma_store: bool) -> int:
    return _epi_chunk_elems(chain, config, use_tma_store) * DTYPE_BYTES[chain.output_dtype]


def _splitk_reduce_elems(chain: FusionChain) -> int:
    """Return FP32 elements reduced by each thread."""
    elems = 4
    while chain.matmul.N % elems:
        elems //= 2
    return elems


def _out_vec_bytes(chain: FusionChain, config: TileConfig, use_tma_store: bool) -> int:
    """The vector width for the output stores."""
    if config.workspace_split_k:
        return _splitk_reduce_elems(chain) * DTYPE_BYTES[chain.output_dtype]
    return _epi_chunk_bytes(chain, config, use_tma_store)


def _splitk_reduce_snippets(chain: FusionChain, config: TileConfig) -> EpilogueSnippets:
    """Generate the split-K reducer epilogue."""
    return generate(
        chain,
        vec_bytes_epi=_out_vec_bytes(chain, config, False),
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        tma_slots=frozenset(),
        packed_lanes=False,
    )


def _mainloop_chain_zero_preserving(ops) -> bool:
    """True iff applying this mainloop op chain to 0 provably yields 0."""
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
    """Generate module-level tile and dtype constants."""
    _check_own_family(tmpl)
    if tmpl.warps_per_cta is not None:
        cfg = replace(cfg, warps_per_cta=tmpl.warps_per_cta)
    fallback_cluster = _mixed_cga_fallback(cfg, tmpl.file)
    smem_fixed_reserve = tmpl.smem_fixed_reserve
    _check_mma_k_dim(chain, cfg)
    # CTAs one MMA instruction spans; 1 on a family whose MMA has no pair.
    ctas_per_mma = cfg.ctas_per_mma
    a_dt = chain.matmul.a_dtype
    b_dt = chain.matmul.b_dtype
    mma_a_dt = _mma_a_dtype(chain)
    mma_b_dt = _mma_b_dtype(chain)
    accum_dt = chain.matmul.accum_dtype
    out_dt = chain.output_dtype
    epi_store_dt = _epi_store_dtype(chain, cfg)
    elem_bytes = DTYPE_BYTES[mma_a_dt]
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
    a_smem_m_step_bytes = (cta_smem_m // cfg.mma_size_m) * cfg.cta_tile_k_bytes
    a_mcast_slices, b_mcast_slices, ab_empty_full_mask = _mcast_slice_plan(chain, cfg)

    lines = [
        f"# Tile config: {cfg.name}",
        f"mma_inst_shape_mnk = {cfg.mma_tile_mnk(elem_bytes)}",
        f"mma_k_dim = {int(cfg.mma_tile_k_bytes == 64)}",
        f"cta_group = {ctas_per_mma}",
        f"cgrp_tile_mnk = {cfg.cga_tile_mnk(elem_bytes)}",
        f"cta_tile_mnk = {cfg.cta_smem_tile_mnk(elem_bytes)}",
        f"epi_tile_mn = {(cfg.epi_tile_m, _epi_n_for_chain(cfg, chain))}",
        f"threads_per_cta = {cfg.threads_per_cta}",
        f"cluster_shape_mnk = {cfg.cluster_shape}",
        f"cluster_k_reduce = {cfg.cluster_k_reduce}",
        f"cluster_reduce_elems = {_cluster_split_k_chunk_elems(chain, cfg) if cfg.cluster_k_reduce else 0}",
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
        f"acc_widen_to_fp32 = {chain.widen_integer_accumulators}",
        f"ab_tma_dtype = {DTYPE_TO_CUTLASS[mma_a_dt]}",
        f"mma_kind = {DTYPE_TO_MMA_KIND[mma_a_dt]}",
        *_epi_swizzle_lines(cfg, epi_store_dt, chain),
    ]
    lines.append(f"tile_swizzle_n = {1 if fallback_cluster is not None else 0}")
    lines.append(f"swizzle_l2_budget_bytes = {_l2_swizzle_budget_bytes()}")
    lines.append(f"num_gemms = {chain.num_gemms}")
    lines.append(f"num_a_operands = {chain.num_a_operands}")
    lines.append(f"num_b_operands = {chain.num_b_operands}")
    lines.append(f"gemm_a_idx = {tuple(a for a, _ in chain.gemm_operands)}")
    lines.append(f"gemm_b_idx = {tuple(b for _, b in chain.gemm_operands)}")
    total_tmem = _tmem_cols_for_arch()
    lines.append(f"num_tmem_alloc_cols = {total_tmem}")
    lines.append(f"tmem_alloc_exclusive = {total_tmem > _MAX_NON_EXCLUSIVE_TMEM_COLS}")
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
    mg_extra_per_stage = 0
    if chain.is_multi_gemm:
        # Per-CTA SMEM B-tile N is halved under 2-CTA MMA (the pair splits B's N).
        smem_n = cfg.cta_smem_tile_n
        base_mn = cfg.cta_tile_m + smem_n
        mg_mn = chain.num_a_operands * cfg.cta_tile_m + chain.num_b_operands * smem_n
        mg_extra_per_stage = (mg_mn - base_mn) * cfg.cta_tile_k_bytes
    if chain.has_moe:
        lines.append(f"grid_num_clusters = {_grid_num_clusters(cfg)}")
        lines.append(f"offset_cutlass_dtype = {DTYPE_TO_CUTLASS[chain.moe.offset_dtype]}")
    if chain.has_mainloop_fusion:
        lines.append("num_mainloop_warps = 4")
        # Per-operand fusion flags (const_expr-gated in the template).
        lines.append(f"mainloop_fuse_a = {chain.has_mainloop_fusion_a}")
        lines.append(f"mainloop_fuse_b = {chain.has_mainloop_fusion_b}")
        # Per-K aux indexing needs ml_L in logical order, not the physical
        # swizzled SMEM address. The K-OOB path already provides that access.
        koob_fix = (
            any(aux.bcast_mode == "per_k" for aux in chain.aux_tensors)
            or not _mainloop_chain_zero_preserving(chain.mainloop_a_ops)
            or not _mainloop_chain_zero_preserving(chain.mainloop_b_ops)
        )
        lines.append(f"mainloop_koob_fix = {koob_fix}")
        _bbits = (cfg.cta_tile_k_bytes // 16).bit_length() - 1
        lines.append(f"ab_swizzle = cutlass.Swizzle({_bbits}, 4, 3)")
        lines.append(f"mainloop_a_cast = {chain.mainloop_a_cast}")
        lines.append(f"mainloop_b_cast = {chain.mainloop_b_cast}")
        load_a_dt = chain.mainloop_a_load_dtype or chain.matmul.a_dtype
        load_b_dt = chain.mainloop_b_load_dtype or chain.matmul.b_dtype
        lines.append(f"ab_load_a_dtype = {DTYPE_TO_CUTLASS[load_a_dt]}")
        lines.append(f"ab_load_b_dtype = {DTYPE_TO_CUTLASS[load_b_dt]}")

    vec_bytes_epi = _epi_vec_bytes(chain, cfg)
    lines.append(f"vec_bytes_epi = {vec_bytes_epi}")
    lines.append(f"split_k_slices = {cfg.split_k_slices}")
    if cfg.workspace_split_k:
        lines.append(f"splitk_reduce_elems = {_splitk_reduce_elems(chain)}")
    lines.append(f"frost_compile_options = {_frost_compile_options()!r}")
    use_tma = _use_tma_store_epi(chain, cfg)
    # only 1 tma output for splitK
    lines.append(f"n_tma_outputs = {1 if (cfg.workspace_split_k and use_tma) else len(_tma_slots_for(chain, cfg))}")
    lines.append(f"moe_aligned_offsets = {_moe_aligned_offsets(chain, cfg)}")
    lines.append(f"moe_gather = {chain.has_moe and chain.moe.mode == 'gather'}")
    lines.append(f"moe_scatter = {chain.has_moe and chain.moe.mode == 'scatter'}")
    lines.append(f"moe_top_k = {chain.moe.top_k if chain.has_moe else 1}")
    lines.append(f"moe_combine = {chain.has_moe and chain.moe.mode == 'combine'}")
    lines.append(f"moe_token_alignment = {_pow2_floor(math.gcd(chain.moe.offset_multiple, chain.matmul.N)) if isinstance(chain.moe, MoeSwapAbSpec) else 1}")
    lines.append(f"epi_slot_widen = {_epi_slot_widen(chain, cfg)}")
    lines.append(f"epi_packed_lanes = {_epi_packed_lanes(cfg)}")
    lines.append(f"epi_dp22 = {_epi_dp22(cfg)}")
    lines.append(f"epi_stage_rows = {_epi_stage_rows(cfg)}")
    lines.append(f"epi_chunk_elems = {_epi_chunk_elems(chain, cfg, use_tma)}")
    smem_d_bytes = (_smem_d_bytes(cfg, chain) if use_tma else 0) + _cluster_split_k_smem_bytes(chain, cfg)
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


_TMEM_COLS_BY_ARCH: tuple[tuple[tuple[int, int], int], ...] = (
    ((100, 107), 512),
    ((107, 110), 576),
    ((110, 120), 512),
)

_MAX_NON_EXCLUSIVE_TMEM_COLS = 512


_B_COLLECTOR_ARCH_RANGES: tuple[tuple[int, int], ...] = ((107, 110),)


def _b_collector_supported(arch: int | None = None) -> bool:
    """Whether this GPU's MMA can hold B in a collector buffer across MMAs."""
    a = _current_arch() if arch is None else arch
    return a is not None and any(lo <= a < hi for lo, hi in _B_COLLECTOR_ARCH_RANGES)


def _b_collector_ok(cfg, arch: int | None = None) -> bool:
    """...and whether THIS geometry may ask for it."""
    return _b_collector_supported(arch) and cfg.mma_tile_m == 128


def _tmem_cols_for_arch(arch: int | None = None) -> int:
    if arch is None:
        arch = _current_arch()
    for (lo, hi), cols in _TMEM_COLS_BY_ARCH:
        if arch is not None and lo <= arch < hi:
            return cols
    return 0


def _current_arch(device=None) -> int | None:
    """SM version of ``device`` (default: the current one) as ``major*10+minor``, or ``None`` when no GPU is visible (render-only / CI)."""
    return current_arch(device)


def _plan_device() -> int:
    """The GPU a plan compiled right now targets."""
    return current_device()


def _check_plan_device(plan_device: int) -> None:
    """A plan's SMEM depth / cluster count / target SM are baked for ONE GPU; refuse to launch it anywhere else."""
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
    """(a_slices, b_slices, needs_full_empty_mask) for the cluster's TMA multicast."""
    cluster_m, cluster_n = cfg.cga_size_mn
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
    """(A, B) multicast bit patterns at CTA rank 0, for the config's own cluster or -- under mixed CGA -- for the fallback shape passed instead."""
    cluster_m, cluster_n, _ = cga_size_mnk if cga_size_mnk is not None else cfg.cga_size_mnk
    ctas_per_mma = cfg.ctas_per_mma
    a_pattern = 0
    for n_idx in range(cluster_n):
        a_pattern |= 1 << (n_idx * cluster_m)
    b_pattern = 0
    for mma_idx in range(cluster_m // ctas_per_mma):
        b_pattern |= 1 << (mma_idx * ctas_per_mma)
    return a_pattern, b_pattern


_MIXED_CGA_MIN_ARCH = 100


def _mixed_cga_supported(arch: int | None = None) -> bool:
    """Whether this GPU can group blocks into a preferred cluster and fall back
    to a smaller one where the preferred does not fit — SM 10.0 and up."""
    a = _current_arch() if arch is None else arch
    return a is not None and a >= _MIXED_CGA_MIN_ARCH


@functools.lru_cache(maxsize=None)
def _template_reads_fallback_cluster(template_file: str) -> bool:
    """Whether a template implements mixed CGA — i.e. whether it consumes the ``fallback_cluster_shape_mnk`` constant."""
    from ..kernel_registry import template_path

    return "fallback_cluster_shape_mnk" in template_path(template_file).read_text()


def _mixed_cga_fallback(cfg: TileConfig, template_file: str) -> tuple[int, int, int] | None:
    """The fallback cluster to attach to this launch, or ``None`` for a plain fixed-cluster launch — byte-for-byte the pre-mixed-CGA behavior."""
    if cfg.cluster_k_reduce or os.environ.get("CUDNN_FROST_DISABLE_MIXED_CGA"):
        return None
    if not _mixed_cga_supported():
        return None
    if not _template_reads_fallback_cluster(template_file):
        return None
    fallback = cfg.fallback_cga_size_mnk
    return None if fallback == cfg.cga_size_mnk else fallback


def _mixed_cga_constants(cfg: TileConfig, fallback_cluster: tuple[int, int, int] | None) -> list[str]:
    """Constants for the kernel's runtime cluster-shape select."""
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


_TCGEN05_DESC_BUILD_ADDR_LIMIT = 262144
_SMEM_RING_ALIGN = 1024
_SMEM_D_BEFORE_RINGS_TEMPLATES = frozenset({"sm100_moe_grouped_block_scale_matmul_fwd.py"})


def _block_scale_smem_desc_roots(*, header_bytes: int, rings: Sequence[tuple[str, int]]) -> dict[str, int]:
    """Byte offset of every ring ROOT: the 1024-aligned `cutlass.Array` rings laid out in DECLARATION order behind ``header_bytes``."""
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
    """Generate block-scale tile and resource constants."""
    _check_own_family(tmpl)
    from ..tile_config import CtaPairTileConfig, validate_block_scale_config_sm100

    if tmpl.warps_per_cta is not None:
        cfg = replace(cfg, warps_per_cta=tmpl.warps_per_cta)
    use_tma_store_epi = _use_tma_store_epi(chain, cfg)
    fallback_cluster = _mixed_cga_fallback(cfg, tmpl.file)
    assert isinstance(cfg, CtaPairTileConfig), f"{cfg.pipeline} has no block-scale template"
    cta_group = cfg.cta_group

    bs = chain.block_scale
    assert bs is not None
    both_fp4 = bs.both_fp4
    mixed_width = bs.mixed_width
    is_sm103 = cfg.pipeline == "sm103"
    mma_k64 = cfg.mma_tile_k_bytes == 64
    if is_sm103 and not both_fp4:
        raise NotImplementedError("the sm103 block-scale pipeline is fp4-only; " f"{bs.a_dtype} data with {bs.sf_dtype} scales runs the sm100 templates")
    padded_fp4 = mixed_width and not mma_k64

    a_data_elem_bits = DTYPE_BITS[bs.a_dtype]
    b_data_elem_bits = DTYPE_BITS[bs.b_dtype]
    a_smem_elem_bits = 8 if padded_fp4 and a_data_elem_bits == 4 else a_data_elem_bits
    b_smem_elem_bits = 8 if padded_fp4 and b_data_elem_bits == 4 else b_data_elem_bits
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

    sf_k = cta_k_elems // bs.block_size  # SF values along K per tile
    sf_k4 = sf_k // 4  # 4 SF-K per utccp atom
    nb_m = cta_m // 128
    nb_n = max(1, cta_n // 128)
    sfb_sub_tile_n = cta_n if cta_n < 128 else 0
    sfb_load_blocks = -(-(cta_n + 128 - math.gcd(cta_n, 128)) // 128)
    mma_nb_m = cfg.mma_tile_m // 128
    mma_nb_n = max(1, cfg.mma_tile_n // 128)
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
    sf_atom_desc_stride = 32  # K-atom stride
    sf_block_desc_stride = 32 * sf_k4  # M/N-block stride

    # SF SMEM bytes per stage (sf_dtype 1 byte; whole k-tile loaded by TMA).
    sfa_smem_bytes = cta_m * sf_k
    sfb_smem_bytes = sfb_load_blocks * 128 * sf_k
    sf_tma_box_k = sf_k4

    total_tmem = _tmem_cols_for_arch()

    num_gemms = chain.num_gemms
    mma_size_m = cfg.mma_size_m
    epi_cols_per_mma_m = _epi_tile_cols(cfg)
    acc_cols_per_stage = mma_size_m * epi_cols_per_mma_m
    na, nb = chain.num_a_operands, chain.num_b_operands
    real_na = 0 if bs.fake_dequant_a else na
    real_nb = 0 if bs.fake_dequant_b else nb
    sf_total_cols = na * sfa_tmem_cols + nb * sfb_tmem_cols
    if is_sm103:
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
    sf_word_stride_cols = 0
    if not is_sm103 and num_sf_words > 1 and sf_region_base + num_sf_words * sf_total_cols <= total_tmem:
        sf_word_stride_cols = sf_total_cols
        used_cols += (num_sf_words - 1) * sf_total_cols
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

    from ..tile_config import smem_ab_budget_bytes, smem_ab_stages

    epi_reserved = _smem_d_bytes(cfg, chain) if use_tma_store_epi else _combine_smem_bytes(chain, cfg)
    ab_reserved = epi_reserved + _cluster_split_k_smem_bytes(chain, cfg)
    if chain.has_moe and chain.moe.mode == "gather":
        token_rows, token_sfs = (cta_n, real_nb) if isinstance(chain.moe, MoeSwapAbSpec) else (cta_m, real_na)
        ab_reserved += token_rows * 32 * token_sfs
    if is_sm103:
        sf_stages = 6
        a_chunk_bytes = cta_m * 128
        b_chunk_bytes = (cta_n // cta_group) * 128
        sfa_group_bytes = cta_m * 12
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
        ring_stage_bytes = [sfa_smem_bytes] * real_na + [sfb_smem_bytes] * real_nb + [cta_m * a_cta_k_bytes] * na + [(cta_n // cta_group) * b_cta_k_bytes] * nb
        ab_stages = smem_ab_stages(sum(ring_stage_bytes), smem_fixed_reserve=tmpl.smem_fixed_reserve, extra_smem_bytes=ab_reserved)
        smem_budget = smem_ab_budget_bytes(tmpl.smem_fixed_reserve)
        while ab_stages:
            ring_bytes = 0
            for stage_bytes in ring_stage_bytes:
                ring_bytes = align_up(ring_bytes, 1024) + stage_bytes * ab_stages
            if use_tma_store_epi and chain.moe is None:
                ring_bytes = align_up(ring_bytes, 1024)
            if ring_bytes + ab_reserved <= smem_budget:
                break
            ab_stages -= 1
        if ab_stages < 1:
            raise NotImplementedError(f"block-scale {cfg.name!r}: no AB stage fits in SMEM including buffer alignment padding")

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
        f"mma_size_m = {cfg.mma_size_m}",
        f"mma_size_n = {cfg.mma_size_n}",
        f"mma_size_k = {cfg.mma_size_k}",
        f"cgrp_tile_mnk = ({cta_m * cfg.cga_size_m}, {cta_n * cfg.cga_size_n}, {cta_k_elems * cfg.cga_size_k})",
        f"cgrp_tile_m = {cta_m * cfg.cga_size_m}",
        f"cgrp_tile_n = {cta_n * cfg.cga_size_n}",
        f"epi_tile_mn = {(cfg.epi_tile_m, _epi_n_for_chain(cfg, chain))}",
        f"threads_per_cta = {cfg.threads_per_cta}",
        f"cluster_shape_mnk = {cfg.cluster_shape}",
        f"cluster_k_reduce = {cfg.cluster_k_reduce}",
        f"cluster_reduce_elems = {_cluster_split_k_chunk_elems(chain, cfg) if cfg.cluster_k_reduce else 0}",
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
        f"acc_gemm_stride = {acc_gemm_stride}",
        f"sfa_col_bases = {tuple(sfa_col_bases)}",
        f"sfb_col_bases = {tuple(sfb_col_bases)}",
        f"sfa_tmem_cols = {sfa_tmem_cols}",
        f"sfb_tmem_cols = {sfb_tmem_cols}",
        f"sf_word_stride_cols = {sf_word_stride_cols}",
        f"tile_swizzle_n = {1 if fallback_cluster is not None else 0}",
        f"swizzle_l2_budget_bytes = {_l2_swizzle_budget_bytes()}",
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
        f"sA_tma_bytes = {cta_m * cta_k_elems * a_data_elem_bits // 8}",
        f"sB_tma_bytes = {(cta_n // cta_group) * cta_k_elems * b_data_elem_bits // 8}",
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
        f"a_is_m_major = {a_major == 'm'}",
        f"b_is_n_major = {b_major == 'n'}",
        f"mma_a_major = {1 if a_major == 'm' else 0}",
        f"mma_b_major = {1 if b_major == 'n' else 0}",
        "",
        f"# output",
        f"cd_dtype = {'cutlass.Int8' if out_dt == 'fp4_e2m1' else DTYPE_TO_CUTLASS[out_dt]}",
        f"epi_store_dtype = {DTYPE_TO_CUTLASS[epi_store_dt]}",
        f"vec_bytes_epi = {vec_bytes_epi}",
        f"split_k_slices = {cfg.split_k_slices}",
        *([f"splitk_reduce_elems = {_splitk_reduce_elems(chain)}"] if cfg.workspace_split_k else []),
        f"frost_compile_options = {_frost_compile_options()!r}",
        f"n_tma_outputs = {1 if (cfg.workspace_split_k and use_tma_store_epi) else len(_tma_slots_for(chain, cfg))}",
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
        f"sf_one_word = { {'fp8_e4m3': 0x38383838, 'fp8_e8m0': 0x7F7F7F7F, 'fp8_e5m3': 0x78787878}[bs.sf_dtype] }",
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
        f"sfb_sub_tile_n = {sfb_sub_tile_n}",
        f"registers_per_block = {_REGISTERS_PER_BLOCK}",
        f"epi_cols_per_mma_m = {epi_cols_per_mma_m}",
        f"mma_c_dtype = {DTYPE_TO_CUTLASS[chain.matmul.accum_dtype]}",
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
        f"sfb_tma_box_mn = {sfb_load_blocks}",
    ]
    if is_sm103:
        kstep = mma_inst_k_bytes
        spi = scales_per_inst
        sf_groups = 4 if bs.block_size == 16 else 2  # 12 SFs/row per group either way
        lines += [
            "",
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
    lines += [
        "",
        f"# block-scale MMA: {num_kblocks} MMAs per K-tile at mma_inst_k_bytes={cfg.mma_tile_k_bytes}",
        f"idesc_is_omma = {both_fp4 and mma_k64}",
        f"mma_k_dim_mode = {(2 if both_fp4 else 1) if mma_k64 else 0}",
    ]
    if chain.has_moe:
        lines.append(f"grid_num_clusters = {_grid_num_clusters(cfg)}")
        lines.append(f"offset_cutlass_dtype = {DTYPE_TO_CUTLASS[chain.moe.offset_dtype]}")
    lines.extend(_mixed_cga_constants(cfg, fallback_cluster))
    lines.extend(_quant_device_imports(chain))
    return "\n".join(lines)


def _render_template(
    chain: FusionChain,
    snippets: EpilogueSnippets,
    config: TileConfig,
) -> str:
    """Render the selected dense or mainloop template with its exact config."""
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
    splitk = config.workspace_split_k
    if "@@SPLITK_ONLY:BEGIN@@" in src:
        src = _resolve_blocks(src, "SPLITK_ONLY", splitk)
    elif splitk:
        raise NotImplementedError(f"split_k_slices={config.split_k_slices}: template {tmpl.file} has no split-K support")
    # For splitK, `snippets` is is kernel 1's partials store. Re-generate epilogue for reduce kernel
    plumb = _splitk_reduce_snippets(chain, config) if splitk else snippets

    aux_tensors = chain.aux_tensors

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
    stg_vec_bindings = "\n".join(f"vec_f32_{g} = c_rmem_vecs[{g}][j * vsize : (j + 1) * vsize]" for g in range(1, chain.num_gemms)) or "pass"
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
    if "@@INJECT_STG_VEC_BINDINGS@@" in src:
        replacements["INJECT_STG_VEC_BINDINGS"] = stg_vec_bindings
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

    tag = re.sub(r"[^A-Za-z0-9_]", "_", f"{_template_stem(tmpl.file)}_{config.geometry_name}{_store_mode_tag(store_modes)}")
    src = re.sub(r"\b_kernel(?=\(|\.set_name_prefix\b)", f"frost_{tag}", src)

    return src


def _render_block_scale_template(
    chain: FusionChain,
    snippets: EpilogueSnippets,
    config: TileConfig,
) -> str:
    """Render the block-scale matmul template."""
    from ..kernel_registry import select_template

    tmpl = select_template(chain, config)
    template_path = tmpl.path
    src = template_path.read_text()
    store_modes = _store_modes(chain, config)
    use_tma = _use_tma_store_epi(chain, config)
    splitk = config.workspace_split_k
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


_LEADING_DIM_A = -1
_LEADING_DIM_B = -2
_LEADING_DIM_C = -1
_LEADING_DIM_AUX = -1


def _maybe_wrap_layout(t: object, leading_dim: int) -> object:
    """Wrap layouts that the DSL cannot infer automatically."""
    if _TVM_FFI_OK:
        return t
    if not _is_layout_ambiguous(t):
        return t
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t).mark_layout_dynamic(leading_dim)


def _wrap_raw_tensor(t: object) -> object:
    """Wrap a tensor when the kernel only consumes its raw pointer."""
    if _TVM_FFI_OK:
        return t
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t)


def _reshape_aux_to_fake(t: object, ref: TensorRef) -> object:
    """View a broadcast-aux runtime tensor to the fake's rank/unit-dim layout so the tvm-ffi front door's ndim/shape check accepts it."""
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


class CompiledFusedGemm(_CompiledFusedGemm[TileConfig, GemmBinding]):
    """Execute a compiled fused GEMM."""

    _compiler = sys.modules[__name__]


def _mma_a_dtype(chain: FusionChain) -> str:
    """MMA A dtype = the graph-declared operand dtype (no implicit cast;
    mainloop transforms are dtype-preserving)."""
    return chain.matmul.a_dtype


def _mma_b_dtype(chain: FusionChain) -> str:
    """MMA-instruction B dtype = the graph-declared operand dtype."""
    return chain.matmul.b_dtype


def _check_supported(chain: FusionChain, config: TileConfig) -> None:
    """Reject a plain-matmul (input/acc dtype combo × active arch) we can't run."""
    from ..kernel_registry import GraphType, mma_arch_reject

    reason = mma_arch_reject(chain, GraphType.MATMUL, config.pipeline)
    if reason is not None:
        raise NotImplementedError(reason)


def _check_cta_group_geometry(
    config: TileConfig,
) -> None:
    """2-CTA MMA structural constraints on the geometry."""
    if not config.is_cta_pair_mma:
        return
    if config.cga_size_m % 2 != 0:
        raise NotImplementedError(f"2-CTA MMA needs cgrp_size_m % 2 == 0; " f"config {config.name!r} has cgrp_size_m={config.cga_size_m}")
    if config.mma_tile_n % 16 != 0:
        raise NotImplementedError(
            f"2-CTA MMA needs mma_inst_n % 16 == 0 (pair MMA instruction n_dim); " f"config {config.name!r} has mma_inst_n={config.mma_tile_n}"
        )
    if config.cta_tile_n % 16 != 0:
        raise NotImplementedError(
            f"2-CTA MMA needs cta_tile_n % 16 == 0 (pair splits B's N in SMEM); " f"config {config.name!r} has cta_tile_n={config.cta_tile_n}"
        )


def _check_mma_n_dim(
    chain: FusionChain,
    config: TileConfig,
) -> None:
    """MMA n_dim rules that depend on the dtype / operand layout, not just geometry."""
    ns = (("mma_inst_n", config.mma_tile_n), ("cta_tile_n", config.cta_tile_n))

    # The int8 MMA kind is narrower than the others above N=32.
    if DTYPE_TO_MMA_KIND.get(_mma_a_dtype(chain)) == "nvvm.Tcgen05MMAKind.INT8":
        for label, n in ns:
            if n >= 40 and (n // 8) % 2 != 0:
                raise NotImplementedError(f"int8 MMA needs N ≤ 32 or a multiple of 16; " f"config {config.name!r} has {label}={n}")

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
    """Rows per K-major A copy, subdividing each multicast slice as needed."""
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
    """Reject (chain, config) where the config K_BYTES isn't a multiple of the MMA dtype's element width."""
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


def _check_input_alignment(chain: FusionChain) -> None:
    """Validate graph-declared TMA input alignment."""
    mm = chain.matmul
    reason = _tma_alignment_reject(mm.a_dtype, mm.b_dtype, mm.a_major, mm.b_major, mm.M, mm.N, mm.K)
    if reason is not None:
        raise ValueError(reason)


_EPI_SMEM_STAGES = 2


def _moe_aligned_offsets(chain, cfg) -> bool:
    if not chain.has_moe or chain.moe is None:
        return False
    required = cfg.cga_tile_mn[1 if isinstance(chain.moe, MoeSwapAbSpec) else 0]
    if chain.block_scale is not None:
        required = math.lcm(required, 128)
    return chain.moe.offset_multiple % required == 0


_ACC_STAGES_MAX = 2


_EPI_SWIZZLE_BY_ROW_BYTES = {16: (0, "none"), 32: (1, "s32b"), 64: (2, "s64b"), 128: (3, "s128b")}


_EPI_ROW_BYTES_MAX = 128  # widest TMA store swizzle
_EPI_N_BASE = 32  # drain width when the epilogue is already hidden behind the MMA
_EPI_N_MAX = 64  # per-lane fp32 registers the drain can hold

_TMA_STORE_EPI_PIPELINES = ("sm100", "sm103")


def _epi_n(cfg, out_dt: str) -> int:
    cols = _epi_tile_cols(cfg)
    cap = _EPI_N_BASE
    if cfg.mma_size_m > 1 and 2 * cfg.mma_size_m * cols > _tmem_cols_for_arch():
        cap = _EPI_N_MAX
    return min(_EPI_ROW_BYTES_MAX * 8 // DTYPE_BITS[out_dt], cap, _pow2_floor(cols, cap=cols))


def _epi_n_for_chain(cfg, chain: FusionChain) -> int:
    """Chain-aware drain width."""
    if _use_swap_bulk_combine(chain, cfg):
        # Logical token batch, staged across two 64-column TMEM loads.
        return 128
    if _use_bulk_combine(chain, cfg):
        cols = _epi_tile_cols(cfg)
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
    if cfg.workspace_split_k:
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
    """Use the qualified single-CTA block-scaled COMBINE epilogue."""
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
    """Qualified weight-by-token COMBINE: transpose fragments, then bulk-reduce."""
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
    """Read compact tokens and segmented scales through immutable tensor maps."""
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
    """The 2-CTA 2x2-DP drain: two column halves per stage, so two TMA boxes."""
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
    """SMEM-D buffer bytes for the TMA-store epilogue: `_EPI_SMEM_STAGES` slots of one epilogue subtile (one MMA-M block x epi_n) + a 16-byte alignment pad."""
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
    if cfg.pipeline not in _TMA_STORE_EPI_PIPELINES or (chain.has_moe and chain.moe.mode == "combine"):
        return "stg"
    if chain.has_moe and chain.moe.mode == "scatter":
        token_major = "n" if isinstance(chain.moe, MoeSwapAbSpec) else "m"
        if out.major == token_major:
            return "stg"
        if isinstance(chain.moe, MoeSwapAbSpec) and _epi_n_for_chain(cfg, chain) % 4:
            return "stg"
        if out.major == "n" and 4 * _epi_row_bytes(out.dtype, _epi_n_for_chain(cfg, chain)) % 128:
            return "stg"
    epi_n = _epi_n_for_chain(cfg, chain)

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
    """Suffix for the kernel symbol naming its store arm, empty for the TMA one."""
    if not store_modes or all(m == "tma" for m in store_modes):
        return ""
    return "_stg" if all(m == "stg" for m in store_modes) else "_mixedstore"


def _store_modes(
    chain,
    cfg,
) -> tuple[str, ...]:
    """One store mode per output, in `chain.outputs` slot order."""
    outs = chain.outputs
    if _FORCE_STG_EPI:
        return ("stg",) * len(outs)
    # For splitK, reduce kernel always use STG
    if cfg.workspace_split_k:
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
    """Does the kernel render the TMA-store arm at all?"""
    # use tma for splitK's fp32 partials when N % 4 == 0
    if cfg.workspace_split_k:
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
    """Make every output take the STG store, even where the TMA gate accepts it."""
    global _FORCE_STG_EPI
    prev = _FORCE_STG_EPI
    _FORCE_STG_EPI = enabled
    try:
        yield
    finally:
        _FORCE_STG_EPI = prev


def _check_executable(chain: FusionChain) -> None:
    """Can this engine RUN the graph, or only render a kernel for it?"""
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
        raise NotImplementedError("the tvm-ffi front door is not installed, and the launch path this engine has needs it (pip install apache-tvm-ffi)")


def _cta_k_elems(chain: FusionChain, config: TileConfig) -> int:
    """Elements of K per CTA tile; fp4 packs two per byte."""
    bits = max(DTYPE_BITS[chain.block_scale.a_dtype], DTYPE_BITS[chain.block_scale.b_dtype]) if chain.has_block_scale else DTYPE_BITS[_mma_a_dtype(chain)]
    return config.cta_tile_k_bytes * 8 // bits


def _cluster_split_k_chunk_elems(chain: FusionChain, config: TileConfig) -> int:
    """Size one mailbox fragment from the selected epilogue drain."""
    # One fragment per epilogue lane, reused over all N spans and MMA-M blocks.
    return 128 * _epi_n_for_chain(config, chain)


def _cluster_split_k_smem_bytes(chain: FusionChain, config: TileConfig) -> int:
    """Reserve reusable fragment slots and barriers for cluster reduction."""
    # S-1 receive slots on the root plus one CTA-local bulk-copy source slot.
    return 4 * config.split_k_slices * _cluster_split_k_chunk_elems(chain, config) + 16 if config.cluster_k_reduce else 0


def _splitk_workspace_bytes(chain: FusionChain, config: TileConfig) -> int:
    """fp32 partials: split_k_slices * B * M * N * 4B."""
    if not config.workspace_split_k:
        return 0
    return align_up(config.split_k_slices * chain.matmul.batch * chain.matmul.M * chain.matmul.N * 4)


def _auto_split_k(chain: FusionChain, config: TileConfig, sm_count: "int | None" = None) -> TileConfig:
    """Layer split-K onto the selected config when the output grid underfills the GPU and K is deep."""
    if config.split_k_slices != 1:
        return config
    if _splitk_reject_reason(chain, replace(config, split_k_slices=2)) is not None:
        return config
    mm = chain.matmul
    if mm.K * DTYPE_BITS[_mma_a_dtype(chain)] < 2048 * 8:
        return config
    sm = sm_count if sm_count is not None else _sm_count()
    output_tiles = mm.batch * -(-mm.M // config.cta_tile_m) * -(-mm.N // config.cta_tile_n)
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
    if slices == 2:
        return config
    return replace(config, split_k_slices=slices)


_TERMINAL_QUANT_ONE_CTA_MAX_M = 1408


def _graph_dynamic_shapes(graph) -> bool:
    """Whether the frontend graph declared its shapes dynamic."""
    return bool(getattr(graph, "_cpp_graph_kwargs", {}).get("is_dynamic_shape_enabled", False))


def _graph_shapes_fixed(graph) -> bool:
    """Whether planning may specialize geometry to the declared token count."""
    options = getattr(graph, "_cpp_graph_kwargs", {})
    return not any(options.get(key, False) for key in ("is_dynamic_shape_enabled", "is_override_shape_enabled"))


def _baseline_config(chain: FusionChain, *, dynamic_shapes: bool = False, knobs=None) -> TileConfig:
    if knobs is not None:
        try:
            config = knobs.to_config()
        except (KeyError, ValueError, NotImplementedError) as exc:
            raise NotImplementedError(f"frost_gemm: knobs do not name a canonical tile config: {exc}") from exc
        if dynamic_shapes and config.split_k_slices != 1:
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
    config = preferred_strategy(chain, config)
    # skip splitK for MoE and dynamic shapes
    if chain.has_moe or dynamic_shapes:
        return config
    return _auto_split_k(chain, config)


def plan_config(chain: FusionChain, *, dynamic_shapes: bool = False, knobs=None, shapes_fixed: bool = False) -> TileConfig:
    """Choose one strategy, or replay the exact supplied knobs."""
    config = _baseline_config(chain, dynamic_shapes=dynamic_shapes, knobs=knobs)
    if knobs is not None or dynamic_shapes:
        return config
    from ..planning import select_strategy

    return select_strategy(chain, config, probe=probe_chain, shapes_fixed=shapes_fixed)


def plan_configs(chain: FusionChain, *, dynamic_shapes: bool = False, shapes_fixed: bool = False) -> list[TileConfig]:
    """Choose a bounded list with the single-plan recommendation first."""
    config = _baseline_config(chain, dynamic_shapes=dynamic_shapes)
    if dynamic_shapes:
        return [config]
    from ..planning import select_strategies

    return select_strategies(chain, config, probe=probe_chain, shapes_fixed=shapes_fixed)


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
    """Every gate the matching ``_jit_*`` path runs before it renders anything."""
    if chain.has_moe and chain.has_block_scale:
        _precheck_moe_block_scale(chain, config)
    elif chain.has_block_scale:
        _precheck_block_scale(chain, config)
    elif chain.has_moe:
        _precheck_moe(chain, config)
    else:
        _precheck_plain(chain, config)


def probe_supported(graph: cudnn.pygraph, config: "TileConfig | None" = None) -> None:
    """Cheap eligibility check — the :func:`jit_from_cudnn_graph` gates WITHOUT ``cute.compile``."""
    probe_cutedsl()
    chain, _binding = analyze_with_binding(graph)
    if config is None:
        config = plan_config(chain, dynamic_shapes=_graph_dynamic_shapes(graph), shapes_fixed=_graph_shapes_fixed(graph))
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
    """Check support for one fusion chain and explicit configuration."""
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
    """Reject split-K modes that exceed template, geometry or device limits."""
    reason = _splitk_reject_reason(chain, config)
    if reason is not None:
        raise NotImplementedError(reason)
    if config.cluster_k_reduce:
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.occupancy import max_active_clusters

        max_active_clusters(math.prod(config.cluster_shape), _plan_device())


def _splitk_reject_reason(chain: FusionChain, config: TileConfig) -> "str | None":
    """Check split-K strategy constraints."""
    if config.split_k_slices == 1:
        return None
    reasons = []
    if config.cluster_k_reduce:
        from ..kernel_registry import select_template

        template = select_template(chain, config)
        if not template.supports_cluster_k_reduce:
            reasons.append(f"template {template.file} does not support cluster_k_reduce")
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
        reasons.append(f"more than 32 slices ({config.split_k_slices})")
    if reasons:
        return f"split_k_slices={config.split_k_slices} is not supported with: {', '.join(reasons)}"
    return None


def jit_from_cudnn_graph(
    graph: cudnn.pygraph,
    config: TileConfig = DEFAULT_CONFIG,
) -> CompiledFusedGemm:
    """End-to-end: cuDNN frontend graph -> rendered + cute-compiled GEMM kernel."""
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
    if chain.has_moe and chain.has_block_scale:
        return _jit_moe_block_scale(chain, config, binding=binding)
    # Block-scale is gated independently (own per-side case table).
    if chain.has_block_scale:
        return _jit_block_scale(chain, config, binding=binding)
    if chain.has_moe:
        return _jit_moe(chain, config, binding=binding)
    if chain.is_multi_gemm:
        from ..kernel_registry import select_template

        tmpl = select_template(chain, config)
        if not tmpl.supports_multi_gemm:
            raise NotImplementedError(
                f"multi-GEMM ({chain.num_gemms} parallel GEMMs) is only supported "
                f"by the 1-CTA-MMA CLC template this pass; got {config.name!r}, "
                f"→ {tmpl.file}. Use cta_group=1."
            )
    _precheck_plain(chain, config)
    return _compile_fused_gemm(chain, config, binding=binding)


def _check_block_scale_supported(chain: FusionChain, template_pipeline: str) -> None:
    """Validate block-scale MMA support for the pipeline."""
    from ..kernel_registry import GraphType, mma_arch_reject

    reason = mma_arch_reject(chain, GraphType.BLOCK_SCALE_MATMUL, template_pipeline)
    if reason is not None:
        raise NotImplementedError(reason)


# One 128-byte TMA tensormap slot per CTA per distinct A operand.


class CompiledMoeGemm(_CompiledMoeGemm[TileConfig, GemmBinding]):
    """Execute a compiled grouped GEMM."""

    _compiler = sys.modules[__name__]


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
    return compiled._launch_moe(problem, offsets, workspace, (*a, *b, *sf), c, aux, stream, token_index, token_ks, top_k_scores)


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


def _compile_fused_gemm(chain: FusionChain, config: TileConfig, *, binding=None) -> CompiledFusedGemm:
    """Build exactly the reduction strategy named by the config."""
    store_modes = _store_modes(chain, config)
    use_tma = _use_tma_store_epi(chain, config)
    snippets = generate(
        chain,
        vec_bytes_epi=_epi_chunk_bytes(chain, config, use_tma),
        output_elem_bytes=DTYPE_BYTES[chain.output_dtype],
        tma_slots=frozenset({0}) if (config.workspace_split_k and use_tma) else frozenset(i for i, mode in enumerate(store_modes) if mode == "tma"),
        packed_lanes=_epi_packed_lanes(config),
        workspace_split_k=config.workspace_split_k,
    )
    render = _render_block_scale_template if chain.has_block_scale else _render_template
    src = render(chain, snippets, config)
    mod = _import_kernel(src)
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    return CompiledFusedGemm(
        chain=chain,
        config=config,
        device=_plan_device(),
        aux_names=[aux.name for aux in chain.aux_tensors],
        generated_path=_cache_dir() / f"gen_{digest}" / "generated_kernel.py",
        _launchable=mod.compile(),
        binding=binding,
        store_modes=store_modes,
        use_tma_store=use_tma,
        block_scale=chain.has_block_scale,
        vec_bytes_epi=_out_vec_bytes(chain, config, use_tma),
        workspace_bytes=_splitk_workspace_bytes(chain, config),
    )


def _jit_block_scale(
    chain: FusionChain,
    config: TileConfig,
    *,
    binding: "GemmBinding | None" = None,
) -> CompiledFusedGemm:
    """JIT path for block-scaled (FP4 / FP8 + per-block SF) matmul."""
    _precheck_block_scale(chain, config)
    return _compile_fused_gemm(chain, config, binding=binding)


class CompiledMoeBlockScaleGemm(_CompiledMoeBlockScaleGemm[TileConfig, GemmBinding]):
    """Execute a compiled grouped block-scale GEMM."""

    _compiler = sys.modules[__name__]


def _jit_moe_block_scale(
    chain: FusionChain,
    config: TileConfig,
    *,
    binding: "GemmBinding | None" = None,
) -> CompiledMoeBlockScaleGemm:
    """JIT path for a MoE grouped block-scale matmul (dequant + moe_grouped)."""
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
