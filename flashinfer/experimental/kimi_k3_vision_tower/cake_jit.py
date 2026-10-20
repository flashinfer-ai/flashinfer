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
"""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ...jit import env as jit_env
from ...jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

# Explicit target-owned registration of the generated programs of the
# Kimi-K3 vision tower (MoonViT-3D encoder + PatchMergerV2; SM100 / SM103).
#
# ``_PROGRAMS`` holds one row per generated program (a kernel plus its host
# binding), keyed by the program digest: the architectures the source is built
# for (``arches``; the loader compiles it with the exact flag set of the device
# it runs on), the extra compile flags and the name of its argument plan in
# ``ARG_PLANS``.  ``_module`` expands a row into the ``MODULES`` record the
# host launcher reads: the two translation units under ``csrc/`` follow from
# the program name and every program exports the ``run`` entry.  One source
# serves every architecture it lists; a program listed for one architecture
# only is a form the host plan selects on that architecture alone (the plain
# one-tile and two-tile-split attention forms exist on SM103 only).  The five
# attention forms both architectures share are one source each with
# compile-time architecture arms (``#if __CUDA_ARCH__ == 1000``; SM103 drains
# its scores with ``tcgen05.ld.red`` in the other arm), so their SM100 build
# carries the SM103 kernel symbol.
#
# ``ARG_PLANS`` lists each distinct argument plan once (``[kind, name]`` in
# launch order; ``kind`` is ``tma_buffer`` / ``buffer`` / ``parameter`` /
# ``grid``), keyed by the program kind that uses it.
#
# ``KERNELS`` maps ``"<arch>"`` to the logical kernel key -> program
# assignment the host launcher resolves at preparation, expanded from the
# table both architectures share (``_KERNELS_SHARED``) and the entries of
# one architecture (``_KERNELS_ARCH``):
#
# * ``gemm:<variant>:<tile>``   the fused tcgen05 BF16 GEMM family
#   (``pos``, ``norm_qkv_rope``, ``residual_wo``, ``norm_gelu``,
#   ``residual_fc1``, ``gelu_erf``, ``rmsnorm``) on the production tile
#   configuration ``select_tile_config`` picks for the token count in its
#   production PDL form;
# * ``gemm:<variant>:<tile>:pdle`` the same tile's ``PDL_EARLY`` binary (the
#   grid-dependency wait moved into the load / epilogue roles), which the host
#   launches only inside the form's census window (``cake_backend.pdl_early_on``)
#   and never on the stream-K / tail / multicast / ``pos`` tiles
#   (``cake_backend.pdl_early_selected``); same kernel parameters, same
#   numerics;
# * ``gemm:<variant>:<tile>`` with a ``*_h`` tile: the half-N tail twin of a
#   plain persistent tile, routed where the tile census leaves a tail that
#   fits one half-round (``cake_backend.half_tail_split``);
# * ``attention:<layout>[:wide][:split]``   the packed-varlen BF16 attention
#   kernel: ``<layout>`` = ``tiles2`` / ``tiles1`` / ``ring3`` (two-tile and
#   SPLIT_KV unit layouts, the host plan rule chooses per ``grid_thws`` batch;
#   ``ring3`` is the SPLIT_KV form with the shared-O three-deep score ring,
#   ``cake_backend.ring3_selected``); ``:wide`` = the build reading the
#   four-word unit table with the next-unit prefetch, selected on short rows
#   (``cake_backend.unit_prefetch_selected``); ``:split`` = the partial-output
#   build of the rows whose tail-round units the plan splits into K/V ranges
#   (``cake_backend.select_layout_and_split``), followed by
# * ``attention_merge``           the exact fixed-order merge of those split
#   units (one launch after every attention launch of a split row);
# * ``merge``                     final RMSNorm + 2x2 spatial / temporal-mean merge;
# * ``rmsnorm_apply``             the post-projector RMSNorm apply pass.
#
# The rows, ``ARG_PLANS`` and the two ``KERNELS`` tables are exporter output,
# hand-folded here pending exporter support: the export writes one verbose
# record per program; the row form and the builders below are the fold, and the
# five attention rows that list both architectures are the former SM100-only
# attention programs folded into their SM103 twins (``KERNELS["sm_100a"]``
# names the twins for those keys).  The JIT library of a program is named after
# a content digest of its sources, architecture and compile flags
# (``spec_content_digest``), so an edited source or flag set builds into a
# fresh library and no stale library is served by name.
ARCHES = ("sm_100a", "sm_103a")
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}

# Row vocabulary of ``_PROGRAMS``: the architecture lists and the extra
# compile-flag sets of the family.  ``_PROGRAM_PREFIX`` + digest is the
# program name; its kernel symbol is ``kernel_`` + that name.
_SM103 = ("sm_103a",)
_BOTH = ("sm_100a", "sm_103a")
_NO_FLAGS: tuple[str, ...] = ()
_GEMM_FLAGS = ("--use_fast_math",)
_ATTENTION_FLAGS = ("--use_fast_math", "--ptxas-options=--opt-level=1")
_PROGRAM_PREFIX = "cake_kimi_k3_vision_tower_"
_SOURCE_DIR = "cake_kimi_k3_vision_tower"

# One row per generated program, keyed by its digest:
# ``(arches, compile_flags, arg_plan)``.
_Row = tuple[tuple[str, ...], tuple[str, ...], str]
_PROGRAMS: dict[str, _Row] = {
    "05528121bdb20ac63f4a": (_BOTH, _ATTENTION_FLAGS, "attention"),
    "08d980e26f7723727971": (_BOTH, _GEMM_FLAGS, "gemm"),
    "0edb981e0644cf0bc327": (_BOTH, _GEMM_FLAGS, "gemm"),
    "1111425111141a4c76aa": (_BOTH, _GEMM_FLAGS, "gemm"),
    "153c43aa8701296c68b2": (_BOTH, _ATTENTION_FLAGS, "attention"),
    "1d473f131d0bf9fe2508": (_BOTH, _GEMM_FLAGS, "gemm"),
    "22ce0ac3a66ecbb5f1de": (_BOTH, _GEMM_FLAGS, "gemm"),
    "23237a8f58b5d594d271": (_BOTH, _ATTENTION_FLAGS, "attention"),
    "233a1176df61a9a0c965": (_BOTH, _GEMM_FLAGS, "gemm"),
    "271dc4ac70c5d911e926": (_BOTH, _GEMM_FLAGS, "gemm"),
    "27e643774380ec1f4d0d": (_BOTH, _GEMM_FLAGS, "gemm"),
    "2c6f70951276ee4ad16a": (_BOTH, _GEMM_FLAGS, "gemm"),
    "2d6aef50287c307414cf": (_BOTH, _GEMM_FLAGS, "gemm"),
    "305b1af2834f5df4a215": (_BOTH, _GEMM_FLAGS, "gemm"),
    "3192e923fe894b0e2b76": (_BOTH, _GEMM_FLAGS, "gemm"),
    "34fb28355074f71ead09": (_BOTH, _GEMM_FLAGS, "gemm"),
    "37082353724ecfc06b4b": (_BOTH, _GEMM_FLAGS, "gemm"),
    "383aff1079a05f46ee76": (_BOTH, _GEMM_FLAGS, "gemm"),
    "3bce5c6a1f5bf7453e9c": (_BOTH, _GEMM_FLAGS, "gemm"),
    "3c66a23ea8d562de0cca": (_BOTH, _GEMM_FLAGS, "gemm"),
    "3fa4eee3bd95517c6334": (_BOTH, _GEMM_FLAGS, "gemm"),
    "444f24ac65b233f0870e": (_BOTH, _GEMM_FLAGS, "gemm"),
    "500195ea53ac5c8b3237": (_BOTH, _GEMM_FLAGS, "gemm"),
    "500a20583d1bf09bca00": (_BOTH, _GEMM_FLAGS, "gemm"),
    "549562cf9e312aa6b3cd": (_BOTH, _GEMM_FLAGS, "gemm"),
    "578a6bdda6a300f3ad5b": (_BOTH, _GEMM_FLAGS, "gemm"),
    "588033a85f785e5ab47e": (_BOTH, _GEMM_FLAGS, "gemm"),
    "5932b4a4ba51246a1b70": (_BOTH, _GEMM_FLAGS, "gemm"),
    "5d0c411d41b3acc90942": (_BOTH, _GEMM_FLAGS, "gemm"),
    "5f015168b8e10231ef00": (_BOTH, _GEMM_FLAGS, "gemm"),
    "6493b5e12e1dff606c6d": (_BOTH, _GEMM_FLAGS, "gemm"),
    "6a51ed9304f30c8815cc": (_BOTH, _ATTENTION_FLAGS, "attention"),
    "71ff4f073d1a66f79bd1": (_BOTH, _GEMM_FLAGS, "gemm"),
    "7315eace6540f14b3ad9": (_BOTH, _GEMM_FLAGS, "gemm"),
    "756759066a3ea781e5d0": (_BOTH, _GEMM_FLAGS, "gemm"),
    "77bc888412b990ccbb9b": (_BOTH, _GEMM_FLAGS, "gemm"),
    "7994f90134f7dadf4bd4": (_BOTH, _GEMM_FLAGS, "gemm"),
    "7bd7dea39285bd36a559": (_BOTH, _GEMM_FLAGS, "gemm"),
    "7d1fd2b79d6a381fc4ac": (_BOTH, _NO_FLAGS, "rmsnorm_apply"),
    "7d601111390a7f44c612": (_BOTH, _GEMM_FLAGS, "gemm"),
    "872157e810f7de49c890": (_BOTH, _GEMM_FLAGS, "gemm"),
    "8b3fb47bc4fd2f09dde3": (_SM103, _ATTENTION_FLAGS, "attention"),
    "901428864fefe2c56b14": (_SM103, _ATTENTION_FLAGS, "attention"),
    "91336598d9b2fdc0508a": (_BOTH, _GEMM_FLAGS, "gemm"),
    "938f1e0c1b33d616fea7": (_BOTH, _NO_FLAGS, "merge"),
    "989472145506999c2ebb": (_BOTH, _GEMM_FLAGS, "gemm"),
    "99833ffa0cdf69c0acc7": (_BOTH, _GEMM_FLAGS, "gemm"),
    "a4adf8330da8ca78e7a3": (_BOTH, _GEMM_FLAGS, "gemm"),
    "a7c57693ffc21df68ada": (_BOTH, _GEMM_FLAGS, "gemm"),
    "b2eeaf27a8eaffecafa2": (_BOTH, _GEMM_FLAGS, "gemm"),
    "b8e7e62f4c62c1cc15de": (_BOTH, _GEMM_FLAGS, "gemm"),
    "b9768b899c32133e295e": (_BOTH, _GEMM_FLAGS, "gemm"),
    "baca276f87b855ba86ee": (_BOTH, _ATTENTION_FLAGS, "attention"),
    "c13cb49b638bcff4e175": (_BOTH, _GEMM_FLAGS, "gemm"),
    "c55f4fa4ba71d511955e": (_BOTH, _GEMM_FLAGS, "gemm"),
    "c69ccc304ef4503e90fe": (_BOTH, _GEMM_FLAGS, "gemm"),
    "c907c55838c4fa0f4292": (_SM103, _ATTENTION_FLAGS, "attention"),
    "cdba842167bb848c365a": (_BOTH, _GEMM_FLAGS, "gemm"),
    "ceb72031b5bae82ea07e": (_BOTH, _GEMM_FLAGS, "gemm"),
    "d027933e3830a35dbcaa": (_BOTH, _GEMM_FLAGS, "gemm"),
    "d8e69a7d4433d7fe81a5": (_BOTH, _GEMM_FLAGS, "gemm"),
    "db2e4e6986f3081ba04c": (_BOTH, _GEMM_FLAGS, "gemm"),
    "dfe112d51921ab20f27d": (_BOTH, _GEMM_FLAGS, "gemm"),
    "e0b196f00aea91ddbf42": (_BOTH, _NO_FLAGS, "attention_merge"),
    "e0ffb5db037008a9ae4f": (_BOTH, _GEMM_FLAGS, "gemm"),
    "e23e007914f3e62dc498": (_BOTH, _GEMM_FLAGS, "gemm"),
    "e29b6e2b82f24c223c3c": (_SM103, _ATTENTION_FLAGS, "attention"),
    "e4470b785885e4cf8e34": (_BOTH, _GEMM_FLAGS, "gemm"),
    "ec6b4fe4671efd0cf5b4": (_BOTH, _GEMM_FLAGS, "gemm"),
    "ee92e52df4965a9d956c": (_BOTH, _GEMM_FLAGS, "gemm"),
    "eff7e7abe0f9d89c58d7": (_BOTH, _GEMM_FLAGS, "gemm"),
    "f3a513bb9c3f5376a578": (_BOTH, _GEMM_FLAGS, "gemm"),
    "f4b3b1373781bf4578ad": (_BOTH, _GEMM_FLAGS, "gemm"),
    "f6147e75f57721f530db": (_BOTH, _GEMM_FLAGS, "gemm"),
    "f6e7c197edcd927c1ff4": (_BOTH, _GEMM_FLAGS, "gemm"),
    "fb3646f34c27b5490c69": (_BOTH, _GEMM_FLAGS, "gemm"),
}


def _sources(name: str) -> list[str]:
    """The translation units of program ``name`` under ``csrc/``: its kernel
    and its host binding, compiled together into one library."""
    return [f"{_SOURCE_DIR}/{name}_kernel.cu", f"{_SOURCE_DIR}/{name}_binding.cu"]


def _module(digest: str, row: _Row) -> dict[str, Any]:
    """The ``MODULES`` record of the program with ``digest`` from its row."""
    arches, compile_flags, arg_plan = row
    return {
        "role": "kernel",
        "sources": _sources(f"{_PROGRAM_PREFIX}{digest}"),
        "arches": list(arches),
        "compile_flags": list(compile_flags),
        "ffi_entry": "run",
        "arg_plan": arg_plan,
    }


MODULES: dict[str, dict[str, Any]] = {
    f"{_PROGRAM_PREFIX}{digest}": _module(digest, row)
    for digest, row in _PROGRAMS.items()
}

ARG_PLANS: dict[str, list[list[str]]] = {
    "attention": [
        ["tma_buffer", "Q"],
        ["buffer", "Q_raw"],
        ["tma_buffer", "K"],
        ["tma_buffer", "V"],
        ["buffer", "O"],
        ["tma_buffer", "O_tma"],
        ["buffer", "seg_begin"],
        ["buffer", "seg_len"],
        ["buffer", "unit_table"],
        ["buffer", "probe"],
        ["parameter", "total_tiles"],
        ["parameter", "num_heads"],
        ["parameter", "softmax_scale_log2"],
        ["buffer", "partial_O"],
        ["buffer", "partial_ML"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "attention_merge": [
        ["buffer", "partial_O"],
        ["buffer", "partial_ML"],
        ["buffer", "merge_table"],
        ["buffer", "seg_begin"],
        ["buffer", "seg_len"],
        ["buffer", "O"],
        ["parameter", "num_heads"],
        ["parameter", "part_rows"],
        ["parameter", "softmax_scale_log2"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "gemm": [
        ["tma_buffer", "A"],
        ["tma_buffer", "A2"],
        ["tma_buffer", "B"],
        ["buffer", "C"],
        ["buffer", "C2"],
        ["buffer", "C3"],
        ["buffer", "R"],
        ["buffer", "COS"],
        ["buffer", "SIN"],
        ["buffer", "CS"],
        ["buffer", "SQ"],
        ["buffer", "XW"],
        ["buffer", "WN"],
        ["buffer", "WS"],
        ["buffer", "FLAGS"],
        ["tma_buffer", "RT"],
        ["tma_buffer", "CT"],
        ["tma_buffer", "XWT"],
        ["parameter", "M"],
        ["parameter", "m_tiles"],
        ["parameter", "full_tiles"],
        ["parameter", "tail_split"],
        ["parameter", "pf_l2"],
        ["parameter", "eps"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "merge": [
        ["buffer", "x"],
        ["buffer", "norm_weight"],
        ["buffer", "merge_table"],
        ["buffer", "m_out"],
        ["parameter", "eps"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "rmsnorm_apply": [
        ["buffer", "y"],
        ["buffer", "weight"],
        ["buffer", "rowsumsq"],
        ["parameter", "M"],
        ["parameter", "eps"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
}

# Logical kernel key -> program digest.  ``_KERNELS_SHARED`` holds the 72
# assignments both architectures share; ``_KERNELS_ARCH`` the keys whose
# program differs per architecture (none) and the 4 keys SM103 alone has (the
# plain one-tile and two-tile-split attention forms, selected there alone).
_KERNELS_SHARED: dict[str, str] = {
    "attention:ring3": "baca276f87b855ba86ee",
    "attention:ring3:split": "6a51ed9304f30c8815cc",
    "attention:ring3:wide": "23237a8f58b5d594d271",
    "attention:tiles2": "05528121bdb20ac63f4a",
    "attention:tiles2:wide": "153c43aa8701296c68b2",
    "attention_merge": "e0b196f00aea91ddbf42",
    "gemm:gelu_erf:l": "f4b3b1373781bf4578ad",
    "gemm:gelu_erf:l_h": "db2e4e6986f3081ba04c",
    "gemm:gelu_erf:l_sk": "fb3646f34c27b5490c69",
    "gemm:gelu_erf:s": "71ff4f073d1a66f79bd1",
    "gemm:gelu_erf:s:pdle": "d027933e3830a35dbcaa",
    "gemm:gelu_erf:s_h": "3192e923fe894b0e2b76",
    "gemm:gelu_erf:s_h:pdle": "ceb72031b5bae82ea07e",
    "gemm:gelu_erf:xs": "3bce5c6a1f5bf7453e9c",
    "gemm:gelu_erf:xs:pdle": "ec6b4fe4671efd0cf5b4",
    "gemm:norm_gelu:l_e8": "7bd7dea39285bd36a559",
    "gemm:norm_gelu:l_e8:pdle": "500a20583d1bf09bca00",
    "gemm:norm_gelu:l_e8_h": "d8e69a7d4433d7fe81a5",
    "gemm:norm_gelu:l_e8_h:pdle": "1111425111141a4c76aa",
    "gemm:norm_gelu:m_e8:pdle": "34fb28355074f71ead09",
    "gemm:norm_gelu:m_e8_h:pdle": "383aff1079a05f46ee76",
    "gemm:norm_gelu:s:pdle": "5f015168b8e10231ef00",
    "gemm:norm_gelu:s_h:pdle": "578a6bdda6a300f3ad5b",
    "gemm:norm_gelu:xs:pdle": "b2eeaf27a8eaffecafa2",
    "gemm:norm_qkv_rope:l_e8_cs": "271dc4ac70c5d911e926",
    "gemm:norm_qkv_rope:l_e8_cs:pdle": "77bc888412b990ccbb9b",
    "gemm:norm_qkv_rope:l_e8_cs_h": "e0ffb5db037008a9ae4f",
    "gemm:norm_qkv_rope:l_e8_cs_h:pdle": "f3a513bb9c3f5376a578",
    "gemm:norm_qkv_rope:m_e8_cs:pdle": "7994f90134f7dadf4bd4",
    "gemm:norm_qkv_rope:m_e8_cs_h:pdle": "756759066a3ea781e5d0",
    "gemm:norm_qkv_rope:s_cs:pdle": "37082353724ecfc06b4b",
    "gemm:norm_qkv_rope:s_cs_h:pdle": "c55f4fa4ba71d511955e",
    "gemm:norm_qkv_rope:xs_cs_pf:pdle": "99833ffa0cdf69c0acc7",
    "gemm:pos_sqxw:s_e8_pf": "1d473f131d0bf9fe2508",
    "gemm:pos_sqxw:xs_pf": "7d601111390a7f44c612",
    "gemm:residual_fc1:l_e8_pf:pdle": "989472145506999c2ebb",
    "gemm:residual_fc1:l_e8_pf_h:pdle": "27e643774380ec1f4d0d",
    "gemm:residual_fc1:l_sk": "ee92e52df4965a9d956c",
    "gemm:residual_fc1:m_p": "6493b5e12e1dff606c6d",
    "gemm:residual_fc1:m_p:pdle": "7315eace6540f14b3ad9",
    "gemm:residual_fc1:m_p_h:pdle": "305b1af2834f5df4a215",
    "gemm:residual_fc1:m_sk": "08d980e26f7723727971",
    "gemm:residual_fc1:s_e8_pf:pdle": "444f24ac65b233f0870e",
    "gemm:residual_fc1:xs_k4_pf:pdle": "2d6aef50287c307414cf",
    "gemm:residual_fc1:xs_pf:pdle": "549562cf9e312aa6b3cd",
    "gemm:residual_fc1_sqxw:l_e8_pf:pdle": "500195ea53ac5c8b3237",
    "gemm:residual_fc1_sqxw:l_e8_pf_h:pdle": "e4470b785885e4cf8e34",
    "gemm:residual_fc1_sqxw:l_sk": "c69ccc304ef4503e90fe",
    "gemm:residual_fc1_sqxw:m_p": "f6147e75f57721f530db",
    "gemm:residual_fc1_sqxw:m_p:pdle": "22ce0ac3a66ecbb5f1de",
    "gemm:residual_fc1_sqxw:m_p_h:pdle": "5932b4a4ba51246a1b70",
    "gemm:residual_fc1_sqxw:m_sk": "5d0c411d41b3acc90942",
    "gemm:residual_fc1_sqxw:s_e8_pf:pdle": "dfe112d51921ab20f27d",
    "gemm:residual_fc1_sqxw:xs_k4_pf:pdle": "91336598d9b2fdc0508a",
    "gemm:residual_fc1_sqxw:xs_pf:pdle": "0edb981e0644cf0bc327",
    "gemm:residual_wo_sqxw:m_tma1": "e23e007914f3e62dc498",
    "gemm:residual_wo_sqxw:m_tma1:pdle": "b9768b899c32133e295e",
    "gemm:residual_wo_sqxw:m_tma1_h": "b8e7e62f4c62c1cc15de",
    "gemm:residual_wo_sqxw:m_tma1_h:pdle": "c13cb49b638bcff4e175",
    "gemm:residual_wo_sqxw:s_e8_pf:pdle": "eff7e7abe0f9d89c58d7",
    "gemm:residual_wo_sqxw:s_e8_pf_h:pdle": "a7c57693ffc21df68ada",
    "gemm:residual_wo_sqxw:xs_pf:pdle": "3c66a23ea8d562de0cca",
    "gemm:rmsnorm:l": "2c6f70951276ee4ad16a",
    "gemm:rmsnorm:l:pdle": "588033a85f785e5ab47e",
    "gemm:rmsnorm:l_h": "872157e810f7de49c890",
    "gemm:rmsnorm:l_sk": "a4adf8330da8ca78e7a3",
    "gemm:rmsnorm:s": "cdba842167bb848c365a",
    "gemm:rmsnorm:s:pdle": "f6e7c197edcd927c1ff4",
    "gemm:rmsnorm:s_h": "233a1176df61a9a0c965",
    "gemm:rmsnorm:s_h:pdle": "3fa4eee3bd95517c6334",
    "merge": "938f1e0c1b33d616fea7",
    "rmsnorm_apply": "7d1fd2b79d6a381fc4ac",
}
_KERNELS_ARCH: dict[str, dict[str, str]] = {
    "sm_100a": {},
    "sm_103a": {
        "attention:tiles1": "c907c55838c4fa0f4292",
        "attention:tiles1:split": "901428864fefe2c56b14",
        "attention:tiles1:wide": "8b3fb47bc4fd2f09dde3",
        "attention:tiles2:split": "e29b6e2b82f24c223c3c",
    },
}
KERNELS: dict[str, dict[str, str]] = {
    arch: {
        key: f"{_PROGRAM_PREFIX}{digest}"
        for key, digest in sorted({**_KERNELS_SHARED, **_KERNELS_ARCH[arch]}.items())
    }
    for arch in ARCHES
}


PDL_EARLY_KEY_SUFFIX = "pdle"


def gemm_kernel_key(variant: str, tile: str, pdl_early: bool = False) -> str:
    """``gemm:<variant>:<tile>`` for the production PDL binary of the tile,
    ``gemm:<variant>:<tile>:pdle`` for its PDL_EARLY binary."""
    key = f"gemm:{variant}:{tile}"
    return f"{key}:{PDL_EARLY_KEY_SUFFIX}" if pdl_early else key


ATTENTION_WIDE_KEY_SUFFIX = (
    "wide"  # UNIT_PREFETCH build (wide unit table, next-unit prefetch)
)
ATTENTION_SPLIT_KEY_SUFFIX = (
    "split"  # KV_PARTIAL build (partial O / (m, l) workspace, merge launch)
)


def attention_kernel_key(
    tiles_per_cta: int, ring3: bool = False, *, wide: bool = False, split: bool = False
) -> str:
    """``attention:<layout>[:wide][:split]``: ``tiles<n>`` for the plain layouts, ``ring3`` for the
    production SPLIT_KV form with the shared-O score ring (selected per arch / longest segment by the
    plan), ``:wide`` for the prefetching wide-table build of the short rows, ``:split`` for the
    partial-output build of the rows with split units."""
    if ring3:
        if int(tiles_per_cta) != 1:
            raise ValueError(
                "the ring3 attention form is the SPLIT_KV (one-tile) layout"
            )
        key = "attention:ring3"
    else:
        key = f"attention:tiles{int(tiles_per_cta)}"
    if wide:
        key += f":{ATTENTION_WIDE_KEY_SUFFIX}"
    if split:
        key += f":{ATTENTION_SPLIT_KEY_SUFFIX}"
    return key


def parse_attention_kernel_key(key: str) -> dict[str, Any]:
    """``dict(tiles_per_cta, ring3, wide, split)`` of an ``attention:`` key."""
    parts = key.split(":")
    if (
        parts[0] != "attention"
        or len(parts) < 2
        or parts[1] not in ("tiles2", "tiles1", "ring3")
    ):
        raise ValueError(f"not an attention kernel key: {key!r}")
    flags = parts[2:]
    allowed = [ATTENTION_WIDE_KEY_SUFFIX, ATTENTION_SPLIT_KEY_SUFFIX]
    if flags != [f for f in allowed if f in flags] or len(set(flags)) != len(flags):
        raise ValueError(f"malformed attention kernel key: {key!r}")
    return dict(
        tiles_per_cta=1 if parts[1] in ("tiles1", "ring3") else 2,
        ring3=parts[1] == "ring3",
        wide=ATTENTION_WIDE_KEY_SUFFIX in flags,
        split=ATTENTION_SPLIT_KEY_SUFFIX in flags,
    )


ATTENTION_MERGE_KERNEL_KEY = "attention_merge"
MERGE_KERNEL_KEY = "merge"
RMSNORM_APPLY_KERNEL_KEY = "rmsnorm_apply"


def route_available(arch: str, required_keys: tuple[str, ...] = ()) -> bool:
    """True when ``arch`` is registered and carries every key in ``required_keys``."""
    table = KERNELS.get(arch)
    return table is not None and all(key in table for key in required_keys)


def kernel_module_name(arch: str, key: str) -> str:
    """Return the registered program for ``key`` on ``arch``."""
    table = KERNELS.get(arch)
    if table is None:
        raise NotImplementedError(
            f"The generated Kimi-K3 vision tower programs for {arch} are not "
            "registered in this checkout yet (see flashinfer-ai/flashinfer#4568)"
        )
    name = table.get(key)
    if name is None:
        raise NotImplementedError(
            f"The generated Kimi-K3 vision tower kernel {key!r} for {arch} is not "
            "registered in this checkout (see flashinfer-ai/flashinfer#4568)"
        )
    if arch not in MODULES[name]["arches"]:
        raise RuntimeError(
            f"registered program {name!r} is not built for {arch} "
            f"(arches {MODULES[name]['arches']})"
        )
    return name


def _header_dirs():
    installed = [jit_env.FLASHINFER_CSRC_DIR, jit_env.FLASHINFER_INCLUDE_DIR]
    if (installed[0] / "tvm_ffi_utils.h").is_file() and (
        installed[1] / "flashinfer/layout.cuh"
    ).is_file():
        return installed
    checkout = Path(__file__).resolve().parents[3]
    source = [checkout / "csrc", checkout / "include"]
    if (source[0] / "tvm_ffi_utils.h").is_file() and (
        source[1] / "flashinfer/layout.cuh"
    ).is_file():
        return source
    raise FileNotFoundError("FlashInfer binding headers were not found")


def spec_content_digest(
    source_bytes: Sequence[bytes], arch: str, compile_flags: Sequence[str]
) -> str:
    """Content identity of a JIT spec: the first 12 hex digits of the SHA-256
    over the bytes of its sources (in compile order), the architecture and the
    program's extra compile flags, each item framed by its length.  A pure
    function of its arguments: a changed source byte, architecture or flag
    gives another spec name, so a stale library is never served by name."""
    digest = hashlib.sha256()
    items = (*source_bytes, arch.encode(), *(f.encode() for f in compile_flags))
    for item in items:
        digest.update(len(item).to_bytes(8, "little"))
        digest.update(item)
    return digest.hexdigest()[:12]


@functools.cache
def gen_cake_kimi_k3_vision_tower_module(name: str, arch: str):
    """JIT spec of program ``name`` built for ``arch``: the shared source compiled with the
    exact flag set of that architecture; the architecture and the content digest of the
    sources and flags are part of the spec name, so two architectures never share one cached
    library and an edited source builds into a fresh one."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise ValueError(f"program {name!r} is not built for {arch}")
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    digest = spec_content_digest(
        [source.read_bytes() for source in sources], arch, record["compile_flags"]
    )
    return gen_jit_spec(
        name=f"{name}_{arch}_{digest}",
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[arch],
            *record["compile_flags"],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_kimi_k3_vision_tower_module(name: str, arch: str):
    return gen_cake_kimi_k3_vision_tower_module(name, arch).build_and_load()
