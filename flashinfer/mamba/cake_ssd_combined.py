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
import os
import shutil
import subprocess
from collections import OrderedDict
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple, Optional, Tuple

import torch
from filelock import FileLock
from tvm_ffi import cpp

from ..jit import env as jit_env

_CHUNK_SIZE = 128
_HEADDIM = 64
_DSTATE = 128

_TARGET_ARCHS = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}
_ARCH_CAPABILITIES = {arch: capability for capability, arch in _TARGET_ARCHS.items()}
# The Rubin target (capability (10, 7)): the main kernels regenerated with a
# Rubin geometry are separate device modules (``_SM107_MODULES``, looked up
# by :func:`_main_module`; every other program builds its fold module) and its
# routing is SM103's (``_ONE_LAUNCH_CAPABILITIES``: the seq_idx triple form
# without a caller table runs the one-launch ``exactone_*`` render, which
# derives the sequence table in-kernel; the fused varlen calls with one run
# the ``exactfused_*`` render, as on SM103).
_SM107_ARCH = "sm_107a"
_DEVICE_DIR = Path("generated") / "device"
# Prepared two-launch host launchers, one per kernel family: the same nine
# ``CAKE_SSD_*`` placeholders and namespace scheme; the look-back chunk-parallel
# main kernel takes six more arguments than the exact scan
# (``_MAIN_ARGS_CHUNKPAR``), the three-phase chunk-parallel main kernel four
# more (``_MAIN_ARGS_CHUNKPAR3``).
# The start two-kernel exact-scan programs (``exact_*``: the retained files,
# 41 main arguments ``_MAIN_ARGS_START``) keep their original launcher; the
# fused launcher serves ``exactfused_*``, ``exactone_*`` and the two-kernel
# ``exactpre_*`` programs (44 main arguments ``_MAIN_ARGS``).  Its preprocess
# stage runs when the preprocess grid is nonzero; the loader zeroes that grid
# only for ``exactfused_*`` / ``exactone_*`` (``_plan``), so the ``exactpre_*``
# programs always run the segment preprocess their scan reads.
_HOST_TEMPLATE = Path("generated") / "host" / "mamba_ssd_combined_sequence.cpp"
_EXACTFUSED_HOST_TEMPLATE = (
    Path("generated") / "host" / "mamba_ssd_combined_fused_sequence.cpp"
)
_CHUNKPAR_HOST_TEMPLATE = (
    Path("generated") / "host" / "mamba_ssd_combined_chunk_parallel_sequence.cpp"
)
# The fused exact-scan programs of the seq_idx triple form without a trusted
# caller ``seq_chunk_cumsum`` (``exactmeta_*``): the ``exactfused_*`` varlen
# main kernels behind the light packed-varlen metadata kernel, whose stage
# publishes the sequence table in place of the full preprocess (12 arguments,
# ``_METADATA_ARGS``) and always runs.
_EXACTMETA_HOST_TEMPLATE = (
    Path("generated") / "host" / "mamba_ssd_combined_fused_metadata_sequence.cpp"
)
_CHUNKPAR3_HOST_TEMPLATE = (
    Path("generated") / "host" / "mamba_ssd_combined_chunk_parallel3_sequence.cpp"
)


@dataclass(frozen=True)
class _Kernel:
    """One compiled device source: ``generated/device/<module>.cu``."""

    module: str
    kernel: str
    threads: int
    fast_math: bool

    @property
    def source(self) -> str:
        return f"{self.module}.cu"

    @property
    def compile_flags(self) -> tuple[str, ...]:
        return ("--use_fast_math",) if self.fast_math else ()


@dataclass(frozen=True)
class _Program:
    """A two-launch program: metadata preprocess followed by the main kernel.

    ``family`` is ``exactfused`` (the fused persistent scan, one CTA per
    (sequence, head) item, delta / cumsum computed in-CTA; its preprocess
    stage is skipped with a grid of 0 unless it publishes packed-varlen
    metadata), ``exactmeta`` (the same fused scan behind the light
    packed-varlen metadata kernel, which publishes ``seq_chunk_cumsum`` from
    the seq_idx triple in place of the full preprocess: the triple form
    without a trusted caller table on SM100, two launches),
    ``exactone`` (the SM103 / SM107 render of the fused scan whose CTA
    prologue derives that table itself into its private row of the
    ``seq_chunk_cumsum`` scratch: the same form on SM103 and SM107, one
    launch, the stage skipped with a grid of 0),
    ``exactpre`` (the same persistent scan reading the delta /
    cumsum its preprocess wrote to the HBM workspace: the start two-launch
    form, same main arguments, with the round-6 state update), ``exact`` (the
    start two-launch program retained byte for byte for one-chunk items, see
    :func:`two_kernel_family`; 41 main arguments), ``chunkpar`` (the look-back chunk-parallel
    program, one CTA per SM over item-major tile ranges) or ``chunkpar3``
    (the three-phase chunk-parallel program, one CTA per (chunk, head)
    tile); it fixes the host template and both kernels' argument orders
    (``preprocess_args``: ``_PREPROCESS_ARGS``, or ``_METADATA_ARGS`` for the
    light metadata stage).
    ``variant`` is the schedule variant of the exact-scan families: ``""``
    (the ring program), ``"noring"``, ``"tail"`` (the two-kernel family's
    2-CTA pair program, ``cluster_dims`` ``(2, 1, 1)``, launched on the even
    grid :func:`_tail_grid_size`), ``"tailone"`` (the two-kernel family's
    sm_107a varlen pair program launched as one-CTA clusters on the same
    grid, :func:`tail_one_cta_selected`), ``"claim"`` (the two-kernel family's
    sm_107a varlen ring program that draws its work items at run time,
    :func:`claim_selected`) or ``"ibulk"`` (the two-kernel family's sm_107a
    packed-varlen ring program loading a work item's initial state as one
    bulk copy, :func:`ibulk_selected`; the fused family's sm_107a
    packed-varlen ring program doing the same, :func:`fused_ibulk_selected`).
    """

    family: str
    preprocess: _Kernel
    main: _Kernel
    state_dtype_code: int
    state_dtype_bits: int
    main_smem_bytes: int
    host_template: Path
    main_args: tuple[str, ...]
    preprocess_args: tuple[str, ...]
    variant: str = ""
    cluster_dims: tuple[int, int, int] = (1, 1, 1)

    @property
    def kernels(self) -> tuple[_Kernel, _Kernel]:
        return (self.preprocess, self.main)


# Generated-source identities, ``generated/device/<module>.cu``: the Cake
# kernel symbol followed by the export's module identity hash.  This block is
# the single place the Cake export refreshes; every program binding below
# derives from it.  Seven kernel families ship, each x {bf16, f16, f32 state}
# x {batched, varlen} (``exactmeta`` and ``exactone`` varlen only), plus the
# preprocess shared by the two-launch programs and the light packed-varlen
# metadata kernel: the fused exact scan (``exactfused_*``: delta / cumsum
# computed in-CTA, one launch), the same fused scan behind the metadata kernel
# (``exactmeta_*``: the seq_idx triple form deriving ``seq_chunk_cumsum`` on
# SM100, two launches), the SM103 / SM107 fused scan deriving that table in
# its CTA prologue (``exactone_*``: the same form on SM103 and SM107, one
# launch), the
# two-kernel exact scan (``exactpre_*``: the start pre+scan
# form with the round-6 single-rounding state update, which the fused route
# keeps for saturated / long-item calls, see :func:`fused_selected`), the
# START two-kernel exact scan (``exact_*``: the
# round-5 programs retained byte for byte -- the export never refreshes
# ``_SCAN_MODULES`` -- which :func:`two_kernel_family` keeps for one-chunk
# items), the look-back chunk-parallel program
# (``chunkpar_*``, round 6 item 1a) and the three-phase chunk-parallel
# program (``chunkpar3_*``, the three-phase program kept as the second
# chunk-parallel family).  The three exact-scan families ship two schedule
# variants of a program only where :func:`_exact_scan_variant` can select the
# ``_noring`` one from host-known shape quantities: the ring program
# (``<family>_<state>_<mode>``) of every (state, mode) and, for those shapes
# (one-chunk items, so never the two-kernel family; never an f32 state on
# the varlen path), the ``_noring`` program (the same kernel built with
# ``epilogue_ring=False``, bitwise-identical outputs).  The two-kernel
# family also ships the ``_tail`` pair program (the ring kernel launched as
# 2-CTA clusters that split an odd pair's first item at its middle chunk,
# bitwise-identical outputs), selected by :func:`tail_selected` for long-item
# two-kernel calls whose balanced persistent grid leaves a tail wave; no call
# inside the fused window reaches the pair program's chunk floor, so the fused
# family ships none.
_SEGMENT_PREPROCESS_MODULE = "factorized_persistent_segment_preprocess_2904ce6fee"
# The sm_107a (Rubin) render of the segment preprocess: the same kernel symbol,
# arguments, launch grid and bitwise outputs, with the Rubin load and store
# schedule; the loader builds it on capability (10, 7) in place of the shared
# module (:func:`_arch_program`).  Refreshed by the Cake export.
_SM107_SEGMENT_PREPROCESS_MODULE = "factorized_persistent_segment_preprocess_3fdd933bec"
# Its twin that triggers the dependent launch at its start
# (``griddepcontrol.launch_dependents``), the sm_107a preprocess of the
# ``_PDL_TRIGGER_FAMILIES`` programs.  Refreshed by the Cake export.
_SM107_SEGMENT_PREPROCESS_TRIGGER_MODULE = (
    "factorized_persistent_segment_preprocess_4a4134cc42"
)
# Retained start programs (the FlashInfer 36fd76859292 files): a literal the
# Cake export does not fill.
_SCAN_MODULES = {
    "exact_bf16_batched": "mamba_ssd_q_tmem_alias_bf16_batched_3876f7208d",
    "exact_f16_batched": "mamba_ssd_q_tmem_alias_f16_batched_f074fc128b",
    "exact_f32_batched": "mamba_ssd_q_tmem_alias_f32_batched_2dc39c9955",
    "exact_bf16_varlen": "mamba_ssd_q_tmem_alias_bf16_varlen_636342f6cc",
    "exact_f16_varlen": "mamba_ssd_q_tmem_alias_f16_varlen_9759edcc11",
    "exact_f32_varlen": "mamba_ssd_q_tmem_alias_f32_varlen_6037fe869d",
    "exact_bf16_batched_noring": "mamba_ssd_q_tmem_alias_bf16_batched_noring_77156893a6",
    "exact_bf16_varlen_noring": "mamba_ssd_q_tmem_alias_bf16_varlen_noring_0f475595b9",
    "exact_f16_batched_noring": "mamba_ssd_q_tmem_alias_f16_batched_noring_d19039c163",
    "exact_f16_varlen_noring": "mamba_ssd_q_tmem_alias_f16_varlen_noring_d11774ac8e",
    "exact_f32_batched_noring": "mamba_ssd_q_tmem_alias_f32_batched_noring_bbd73b5202",
}
_EXACTFUSED_MODULES = {
    "exactfused_bf16_batched": "mamba_ssd_q_tmem_alias_bf16_batched_b9b1004bf9",
    "exactfused_f16_batched": "mamba_ssd_q_tmem_alias_f16_batched_ca0e86b85e",
    "exactfused_f32_batched": "mamba_ssd_q_tmem_alias_f32_batched_c0c70bdc72",
    "exactfused_bf16_varlen": "mamba_ssd_q_tmem_alias_bf16_varlen_2e60cd811b",
    "exactfused_f16_varlen": "mamba_ssd_q_tmem_alias_f16_varlen_9f26678a99",
    "exactfused_f32_varlen": "mamba_ssd_q_tmem_alias_f32_varlen_33156e24a3",
    "exactfused_bf16_batched_noring": "mamba_ssd_q_tmem_alias_bf16_batched_noring_b23fda648d",
    "exactfused_bf16_varlen_noring": "mamba_ssd_q_tmem_alias_bf16_varlen_noring_96d2060d32",
    "exactfused_f16_batched_noring": "mamba_ssd_q_tmem_alias_f16_batched_noring_1b6ebc1d72",
    "exactfused_f16_varlen_noring": "mamba_ssd_q_tmem_alias_f16_varlen_noring_b13ad21c49",
    "exactfused_f32_batched_noring": "mamba_ssd_q_tmem_alias_f32_batched_noring_079bcfe4b5",
}
_CHUNKPAR_MODULES = {
    "chunkpar_bf16_batched": "mamba_ssd_chunk_parallel_v3_bf16_batched_22e35cf4ea",
    "chunkpar_f16_batched": "mamba_ssd_chunk_parallel_v3_f16_batched_f383a3a64e",
    "chunkpar_f32_batched": "mamba_ssd_chunk_parallel_v3_f32_batched_84131101e6",
    "chunkpar_bf16_varlen": "mamba_ssd_chunk_parallel_v3_bf16_varlen_2a29ec0757",
    "chunkpar_f16_varlen": "mamba_ssd_chunk_parallel_v3_f16_varlen_247f589535",
    "chunkpar_f32_varlen": "mamba_ssd_chunk_parallel_v3_f32_varlen_bbe331b2df",
}
_EXACTPRE_MODULES = {
    "exactpre_bf16_batched": "mamba_ssd_q_tmem_alias_bf16_batched_precomputed_42e1969ba8",
    "exactpre_bf16_varlen": "mamba_ssd_q_tmem_alias_bf16_varlen_precomputed_da7c5004dc",
    "exactpre_f16_batched": "mamba_ssd_q_tmem_alias_f16_batched_precomputed_9df4f49711",
    "exactpre_f16_varlen": "mamba_ssd_q_tmem_alias_f16_varlen_precomputed_6a9d60dc45",
    "exactpre_f32_batched": "mamba_ssd_q_tmem_alias_f32_batched_precomputed_d2d517f5a8",
    "exactpre_f32_varlen": "mamba_ssd_q_tmem_alias_f32_varlen_precomputed_4d4d183c65",
    "exactpre_bf16_batched_tail": "mamba_ssd_q_tmem_alias_bf16_batched_precomputed_tail_83e285944e",
    "exactpre_bf16_varlen_tail": "mamba_ssd_q_tmem_alias_bf16_varlen_precomputed_tail_920a72bca9",
    "exactpre_f16_batched_tail": "mamba_ssd_q_tmem_alias_f16_batched_precomputed_tail_8c871220a8",
    "exactpre_f16_varlen_tail": "mamba_ssd_q_tmem_alias_f16_varlen_precomputed_tail_58fa3281d0",
    "exactpre_f32_batched_tail": "mamba_ssd_q_tmem_alias_f32_batched_precomputed_tail_da634e043c",
    "exactpre_f32_varlen_tail": "mamba_ssd_q_tmem_alias_f32_varlen_precomputed_tail_3db510fd81",
}
_CHUNKPAR3_MODULES = {
    "chunkpar3_bf16_batched": "mamba_ssd_chunk_parallel_bf16_batched_2d0d9e4b72",
    "chunkpar3_f16_batched": "mamba_ssd_chunk_parallel_f16_batched_e1869cbee5",
    "chunkpar3_f32_batched": "mamba_ssd_chunk_parallel_f32_batched_4b726bb26f",
    "chunkpar3_bf16_varlen": "mamba_ssd_chunk_parallel_bf16_varlen_3b884ec056",
    "chunkpar3_f16_varlen": "mamba_ssd_chunk_parallel_f16_varlen_a3f2b783ec",
    "chunkpar3_f32_varlen": "mamba_ssd_chunk_parallel_f32_varlen_d031378b2d",
}
# The fused programs of the seq_idx triple form deriving its table
# (``exactmeta_*``): the ``exactfused_*`` varlen main kernels -- the same
# device sources, bound here once more so the table names every program --
# behind the light packed-varlen metadata kernel instead of the full
# preprocess.  Refreshed by the Cake export together with the exactfused
# varlen entries (an exactmeta entry always equals its exactfused twin).
_EXACTMETA_MODULES = {
    "exactmeta_bf16_varlen": "mamba_ssd_q_tmem_alias_bf16_varlen_2e60cd811b",
    "exactmeta_f16_varlen": "mamba_ssd_q_tmem_alias_f16_varlen_9f26678a99",
    "exactmeta_f32_varlen": "mamba_ssd_q_tmem_alias_f32_varlen_33156e24a3",
    "exactmeta_bf16_varlen_noring": "mamba_ssd_q_tmem_alias_bf16_varlen_noring_96d2060d32",
    "exactmeta_f16_varlen_noring": "mamba_ssd_q_tmem_alias_f16_varlen_noring_b13ad21c49",
}
# The one-launch fused varlen programs (``exactone_*``): the exact scan's fused
# varlen entries as SM103 renders them -- the main kernel's CTA prologue
# derives the sequence table of the seq_idx triple form itself, so no stage
# runs ahead of it -- with their own device sources (the ``exactfused_*``
# varlen programs keep the SM100 render) behind the fused host template.
# SM107 ships the same one-launch form beside its ``exactfused_*`` varlen
# render: its sources are the ``exactone_*`` entries of ``_SM107_MODULES``.
# Refreshed by the Cake export; seed identities until it fills them.
_EXACTONE_MODULES = {
    "exactone_bf16_varlen": "mamba_ssd_q_tmem_alias_bf16_varlen_2e93602110",
    "exactone_f16_varlen": "mamba_ssd_q_tmem_alias_f16_varlen_417e094e67",
    "exactone_f32_varlen": "mamba_ssd_q_tmem_alias_f32_varlen_71ead26645",
    "exactone_bf16_varlen_noring": "mamba_ssd_q_tmem_alias_bf16_varlen_noring_1100526c71",
    "exactone_f16_varlen_noring": "mamba_ssd_q_tmem_alias_f16_varlen_noring_948223063c",
}
# sm_107a (Rubin) device modules, filled by the Cake export: every
# ``exactfused_*``, ``exactone_*``, ``exactpre_*`` (tail included),
# ``chunkpar_*`` and ``chunkpar3_*`` program builds its own sm_107a render for
# capability (10, 7) (:func:`_main_module`).  The retained start ``exact_*``
# programs and the ``exactmeta_*`` programs (SM100 only,
# ``_LIGHT_STAGE_CAPABILITIES``) build their fold module on every
# architecture; the fold source selects the tcgen05 SMEM descriptor address
# mask by ``__CUDA_ARCH__`` (15 bits on sm_107a, ``0x7FFF``; 14 bits,
# ``0x3FFF``, on sm_100a and sm_103a).  The metadata kernel is
# shared; the segment preprocess builds its own sm_107a render
# (``_SM107_SEGMENT_PREPROCESS``, or its early-trigger twin
# ``_SM107_SEGMENT_PREPROCESS_TRIGGER`` for the ``_PDL_TRIGGER_FAMILIES``
# programs).  A fused varlen suffix has two sm_107a
# sources, as SM103 has two texts: the ``exactfused_*`` varlen entry names
# the render of the SM100 form (the caller-table and cu_seqlens forms) and
# the ``exactone_*`` entry the one-launch render, which derives the sequence
# table in-kernel.  The two-kernel claim programs
# (``exactpre_<state>_varlen_claim``, :func:`claim_selected`), initial-state
# bulk programs (``exactpre_<state>_varlen_ibulk``, :func:`ibulk_selected`)
# and one-CTA pair programs (``exactpre_<state>_varlen_tailone``,
# :func:`tail_one_cta_selected`) exist on sm_107a only
# (``_SM107_ONLY_PROGRAMS``),
# as do the fused initial-state bulk programs
# (``exactfused_<state>_varlen_ibulk``, :func:`fused_ibulk_selected`):
# their entries are their sole sources (no fold module, never built for
# another architecture).
_SM107_MODULES = {
    "exactfused_bf16_varlen": "mamba_ssd_q_tmem_alias_bf16_varlen_2cbbfc3dd6",
    "exactfused_f16_varlen": "mamba_ssd_q_tmem_alias_f16_varlen_9a77db3ad6",
    "exactfused_f32_varlen": "mamba_ssd_q_tmem_alias_f32_varlen_ca9cc524b2",
    "exactfused_bf16_varlen_noring": "mamba_ssd_q_tmem_alias_bf16_varlen_noring_ad5e3246eb",
    "exactfused_f16_varlen_noring": "mamba_ssd_q_tmem_alias_f16_varlen_noring_622868a1e1",
    "chunkpar_bf16_batched": "mamba_ssd_chunk_parallel_v3_bf16_batched_0d36ddcff6",
    "chunkpar_bf16_varlen": "mamba_ssd_chunk_parallel_v3_bf16_varlen_cc9360d18a",
    "chunkpar_f16_batched": "mamba_ssd_chunk_parallel_v3_f16_batched_74041d25de",
    "chunkpar_f16_varlen": "mamba_ssd_chunk_parallel_v3_f16_varlen_fd7ee521e3",
    "chunkpar_f32_batched": "mamba_ssd_chunk_parallel_v3_f32_batched_77be0da401",
    "chunkpar_f32_varlen": "mamba_ssd_chunk_parallel_v3_f32_varlen_b7f2b1f440",
    "exactpre_bf16_varlen": "mamba_ssd_q_tmem_alias_bf16_varlen_precomputed_d143ae24ea",
    "exactpre_f16_varlen": "mamba_ssd_q_tmem_alias_f16_varlen_precomputed_ae77ddc912",
    "exactpre_f32_varlen": "mamba_ssd_q_tmem_alias_f32_varlen_precomputed_ee7d6500e5",
    "exactone_bf16_varlen": "mamba_ssd_q_tmem_alias_bf16_varlen_8cf6f8fef2",
    "exactone_f16_varlen": "mamba_ssd_q_tmem_alias_f16_varlen_7e8ffa5e3b",
    "exactone_f32_varlen": "mamba_ssd_q_tmem_alias_f32_varlen_e9ab92eb70",
    "exactone_bf16_varlen_noring": "mamba_ssd_q_tmem_alias_bf16_varlen_noring_afc73d515a",
    "exactone_f16_varlen_noring": "mamba_ssd_q_tmem_alias_f16_varlen_noring_36f7d06ed4",
    "exactfused_bf16_batched": "mamba_ssd_q_tmem_alias_bf16_batched_a926c4b548",
    "exactfused_f16_batched": "mamba_ssd_q_tmem_alias_f16_batched_20a770476a",
    "exactfused_f32_batched": "mamba_ssd_q_tmem_alias_f32_batched_52f18e0520",
    "exactfused_bf16_batched_noring": "mamba_ssd_q_tmem_alias_bf16_batched_noring_3c6cf7b844",
    "exactfused_f16_batched_noring": "mamba_ssd_q_tmem_alias_f16_batched_noring_d33767ff60",
    "exactfused_f32_batched_noring": "mamba_ssd_q_tmem_alias_f32_batched_noring_f26078282e",
    "exactpre_bf16_batched": "mamba_ssd_q_tmem_alias_bf16_batched_precomputed_542047a878",
    "exactpre_f16_batched": "mamba_ssd_q_tmem_alias_f16_batched_precomputed_8a0418850b",
    "exactpre_f32_batched": "mamba_ssd_q_tmem_alias_f32_batched_precomputed_5bb1e2274f",
    "exactpre_bf16_batched_tail": "mamba_ssd_q_tmem_alias_bf16_batched_precomputed_tail_a0ea2820d8",
    "exactpre_f16_batched_tail": "mamba_ssd_q_tmem_alias_f16_batched_precomputed_tail_48e80d8704",
    "exactpre_bf16_varlen_tail": "mamba_ssd_q_tmem_alias_bf16_varlen_precomputed_tail_8edaef9799",
    "exactpre_f16_varlen_tail": "mamba_ssd_q_tmem_alias_f16_varlen_precomputed_tail_4ff710621d",
    "exactpre_f32_batched_tail": "mamba_ssd_q_tmem_alias_f32_batched_precomputed_tail_fc388577dc",
    "exactpre_f32_varlen_tail": "mamba_ssd_q_tmem_alias_f32_varlen_precomputed_tail_7fd866a0ea",
    "chunkpar3_bf16_batched": "mamba_ssd_chunk_parallel_bf16_batched_39bb8597d5",
    "chunkpar3_f16_batched": "mamba_ssd_chunk_parallel_f16_batched_35f6e66f75",
    "chunkpar3_bf16_varlen": "mamba_ssd_chunk_parallel_bf16_varlen_3449e8c0f2",
    "chunkpar3_f16_varlen": "mamba_ssd_chunk_parallel_f16_varlen_ebf7470866",
    "chunkpar3_f32_batched": "mamba_ssd_chunk_parallel_f32_batched_43dad44ea0",
    "chunkpar3_f32_varlen": "mamba_ssd_chunk_parallel_f32_varlen_5d526eacf5",
    "exactpre_bf16_varlen_claim": "mamba_ssd_q_tmem_alias_bf16_varlen_precomputed_claim_57675d132c",
    "exactpre_f32_varlen_claim": "mamba_ssd_q_tmem_alias_f32_varlen_precomputed_claim_8ae380b2cd",
    "exactpre_bf16_varlen_ibulk": "mamba_ssd_q_tmem_alias_bf16_varlen_precomputed_ibulk_272db07349",
    "exactpre_f16_varlen_ibulk": "mamba_ssd_q_tmem_alias_f16_varlen_precomputed_ibulk_ae9645a9b3",
    "exactone_bf16_varlen_tablefirst": "mamba_ssd_q_tmem_alias_bf16_varlen_tablefirst_76a8f20d0d",
    "exactone_f16_varlen_tablefirst": "mamba_ssd_q_tmem_alias_f16_varlen_tablefirst_494f203d00",
    "exactone_f32_varlen_tablefirst": "mamba_ssd_q_tmem_alias_f32_varlen_tablefirst_13c07efd3e",
    "exactone_bf16_varlen_tableroles": "mamba_ssd_q_tmem_alias_bf16_varlen_tableroles_cd9bb4215b",
    "exactone_f16_varlen_tableroles": "mamba_ssd_q_tmem_alias_f16_varlen_tableroles_b7f24c6017",
    "exactone_f32_varlen_tableroles": "mamba_ssd_q_tmem_alias_f32_varlen_tableroles_fd7c260936",
    "exactfused_bf16_varlen_ibulk": "mamba_ssd_q_tmem_alias_bf16_varlen_ibulk_7e3e039507",
    "exactfused_f16_varlen_ibulk": "mamba_ssd_q_tmem_alias_f16_varlen_ibulk_ad95ebc012",
    "exactpre_bf16_varlen_tailone": "mamba_ssd_q_tmem_alias_bf16_varlen_precomputed_tailone_cb5978f317",
    "exactpre_f16_varlen_tailone": "mamba_ssd_q_tmem_alias_f16_varlen_precomputed_tailone_ed8277f475",
    "exactpre_f32_varlen_tailone": "mamba_ssd_q_tmem_alias_f32_varlen_precomputed_tailone_a6f49c4f67",
}

_SEGMENT_PREPROCESS = _Kernel(
    _SEGMENT_PREPROCESS_MODULE,
    "kernel_factorized_persistent_segment_preprocess",
    threads=256,
    fast_math=False,
)
_SM107_SEGMENT_PREPROCESS = _Kernel(
    _SM107_SEGMENT_PREPROCESS_MODULE,
    "kernel_factorized_persistent_segment_preprocess",
    threads=256,
    fast_math=False,
)
_SM107_SEGMENT_PREPROCESS_TRIGGER = _Kernel(
    _SM107_SEGMENT_PREPROCESS_TRIGGER_MODULE,
    "kernel_factorized_persistent_segment_preprocess",
    threads=256,
    fast_math=False,
)
# The light packed-varlen metadata kernel, the preprocess stage of the
# ``exactmeta_*`` programs: one thread per logical segment publishes
# ``seq_chunk_cumsum`` (``sequence_offsets``) and the segment tables from the
# caller's seq_idx triple (the sequence-boundary rule, the same publication the
# full preprocess makes with ``write_seq_chunk_cumsum``); its launch grid is
# ``ceil(num_segments / threads)``.  Module identity refreshed by the Cake
# export; built with the loader's ``--use_fast_math`` (an integer-only kernel).
_PACKED_METADATA_MODULE = "factorized_persistent_packed_varlen_metadata_1ac82c1448"
_PACKED_METADATA = _Kernel(
    _PACKED_METADATA_MODULE,
    "kernel_factorized_persistent_packed_varlen_metadata",
    threads=256,
    fast_math=True,
)
# (segment, head) tiles one preprocess CTA scans (one warp per tile, so
# ``threads / 32`` of them); the launch grid is ``ceil(num_segments * nheads
# / tiles_per_block)``.  Both literals describe the shipped preprocess source
# above and are refreshed by the Cake export together with its module
# identity.
_SEGMENT_PREPROCESS_TILES_PER_BLOCK = 8
# DLDataType (code, bits) of the state tensors: kDLFloat=2, kDLBfloat=4.
_STATE_DTYPE_CODES = {"bf16": (4, 16), "f16": (2, 16), "f32": (2, 32)}
_STATE_DTYPE_KEYS = {
    torch.bfloat16: "bf16",
    torch.float16: "f16",
    torch.float32: "f32",
}
# Dynamic shared memory (bytes) of each family's main kernel; refreshed by the
# Cake export.
_EXACTFUSED_SMEM_BYTES = 232320
_CHUNKPAR_SMEM_BYTES = 231808
_CHUNKPAR3_SMEM_BYTES = 232448
# Main dynamic SMEM of the fused ``_noring`` programs (the start schedule
# keeps the prefix control placement); refreshed by the Cake export with the
# ``exactfused_*_noring`` table entries.
_EXACTFUSED_NORING_SMEM_BYTES = 232192
# Main dynamic SMEM of the one-launch fused varlen programs (``exactone_*``)
# and their ``_noring`` variants; refreshed by the Cake export with the
# ``exactone_*`` table entries (the SM103 render's literals).
_EXACTONE_SMEM_BYTES = 232320
_EXACTONE_NORING_SMEM_BYTES = 232192
# Main dynamic SMEM of the retained start programs (``exact_*`` and
# ``exact_*_noring``): literals of the round-5 files, not refreshed.
_EXACT_SMEM_BYTES = 231808
_EXACT_NORING_SMEM_BYTES = 232448
# Main dynamic SMEM of the two-kernel exact-scan programs (``exactpre_*``);
# refreshed by the Cake export with the ``exactpre_*`` table entries.
_EXACTPRE_SMEM_BYTES = 232320
# Main dynamic SMEM of the tail-wave pair programs (``exactpre_*_tail``: the
# two-kernel ring programs launched as 2-CTA clusters, with the hand-off
# mbarrier); refreshed by the Cake export with the ``exactpre_*_tail`` table
# entries.
_EXACTPRE_TAIL_SMEM_BYTES = 232320
# Main dynamic SMEM of the sm_107a varlen tail-pair programs (the
# ``_SM107_MODULES`` entries of ``exactpre_*_varlen_tail``): they keep each
# chunk's segment bounds in shared memory, 8 bytes past the pool of the other
# tail programs, which keep ``_EXACTPRE_TAIL_SMEM_BYTES``; the Cake export
# reads it for the launch of those table entries.
_EXACTPRE_TAIL_VARLEN_SM107_SMEM_BYTES = 232448
# Main dynamic SMEM of the sm_107a claim programs (``exactpre_*_claim``: the
# two-kernel varlen ring programs with the 8-byte item hand-off slots and the
# two claim mbarrier pairs); the Cake export reads it for the launch of the
# ``exactpre_*_claim`` table entries.
_EXACTPRE_CLAIM_SMEM_BYTES = 232448
# Main dynamic SMEM of the sm_107a initial-state bulk programs
# (``exactpre_*_ibulk``: the varlen ring programs with one more mbarrier, the
# initial-state bulk copy's); the Cake export reads it for the launch of the
# ``exactpre_*_ibulk`` table entries.
_EXACTPRE_IBULK_SMEM_BYTES = 232448
# Main dynamic SMEM of the sm_107a fused initial-state bulk programs
# (``exactfused_*_ibulk``: the fused varlen ring programs with the
# initial-state bulk copy's mbarrier); the Cake export reads it for the launch
# of the ``exactfused_*_ibulk`` table entries.
_EXACTFUSED_IBULK_SMEM_BYTES = 232448
# Main dynamic SMEM of the sm_107a one-CTA pair programs
# (``exactpre_*_varlen_tailone``: the varlen tail-pair programs launched
# without the 2-CTA cluster, with the rank ticket's slot past the segment
# bounds); the Cake export reads it for the launch of the
# ``exactpre_*_tailone`` table entries.
_EXACTPRE_TAILONE_SMEM_BYTES = 232448


# Positional launcher ABI shared by every program: preprocess arguments, its
# grid, main arguments (the family's order), its grid, then the stream.  The
# preprocess also derives ``seq_chunk_cumsum`` from the packed-varlen metadata
# (``write_seq_chunk_cumsum``), so one launcher call covers the whole forward;
# with ``metadata_from_cu_seqlens`` it derives the whole segment metadata
# (``chunk_indices`` / ``chunk_offsets`` / ``seq_chunk_cumsum`` + sentinel)
# from ``cu_seqlens`` into runner-owned tables (CAKE-934 item 2), inserting
# the chunk-unaligned ``checkpoint_token_indices`` boundaries when
# ``checkpoint_state_count > 0``; ``preprocess_status`` is the runner-owned
# int32 word it sets to 1 when a packed-sequence id is out of range or
# non-monotonic (CAKE-990) or ``cu_seqlens`` is invalid.  The order is the
# kernel's parameter order (``preprocess_arg_plan`` / ``main_arg_plan`` of
# the export).
_PREPROCESS_ARGS = (
    "dt",
    "A",
    "dt_bias",
    "segment_starts",
    "segment_lengths",
    "chunk_indices",
    "chunk_offsets",
    "delta",
    "cumsum",
    "num_segments",
    "nheads",
    "seqlen",
    "direct_varlen_metadata",
    "dt_softplus",
    "dt_min",
    "dt_max",
    "seq_idx_i32",
    "seq_idx_i64",
    "seq_idx_int64",
    "seq_chunk_cumsum",
    "num_sequences",
    "write_seq_chunk_cumsum",
    "cu_seqlens",
    "checkpoint_token_indices",
    "metadata_from_cu_seqlens",
    "checkpoint_state_count",
    "preprocess_status",
)
# Light packed-varlen metadata stage of the ``exactmeta_*`` programs, in the
# kernel's parameter order: the caller's seq_idx (one of the two pointers is
# read, ``seq_idx_int64`` says which) and chunk tables, the runner's segment
# tables and the published ``sequence_offsets`` (the call's
# ``seq_chunk_cumsum`` vector), the counts and the status word.
_METADATA_ARGS = (
    "seq_idx_i32",
    "seq_idx_i64",
    "chunk_indices",
    "chunk_offsets",
    "segment_starts",
    "segment_lengths",
    "sequence_offsets",
    "num_segments",
    "num_sequences",
    "seqlen",
    "seq_idx_int64",
    "preprocess_status",
)
# Exact-scan main kernel, both families (the fused ``exactfused_*`` programs and
# the two-kernel ``exactpre_*`` programs): the 41 start arguments followed by the
# three inputs the fused kernel takes over from the preprocess -- ``cu_seqlens``
# and ``metadata_from_cu_seqlens`` (it derives the cu_seqlens segment tables
# in-CTA) and the ``preprocess_status`` word it flags an invalid ``cu_seqlens``
# in.  The two-kernel program carries the same signature and leaves the three
# unused (``metadata_from_cu_seqlens`` is 0: it reads the tables its preprocess
# published).
_MAIN_ARGS = (
    "x_map",
    "b_map",
    "c_map",
    "out_map",
    "x",
    "dt",
    "delta_precomputed",
    "cumsum_precomputed",
    "A",
    "B_tensor",
    "C",
    "D",
    "z",
    "dt_bias",
    "initial_states",
    "final_states",
    "checkpoint_states",
    "checkpoint_token_indices",
    "checkpoint_state_slots",
    "seq_idx_i32",
    "seq_idx_i64",
    "chunk_indices",
    "chunk_offsets",
    "seq_chunk_cumsum",
    "out_native",
    "nheads",
    "ngroups",
    "batch",
    "seqlen",
    "nchunks",
    "sequence_count",
    "num_logical_chunks",
    "mode_varlen",
    "D_mode",
    "has_z",
    "has_initial",
    "dt_softplus",
    "dt_min",
    "dt_max",
    "write_final_states",
    "checkpoint_state_count",
    "cu_seqlens",
    "metadata_from_cu_seqlens",
    "preprocess_status",
)
# The fused-only inputs that close ``_MAIN_ARGS``: both exact-scan families
# bind them (only the fused program reads them); the chunk-parallel
# signature has none of them.
_FUSED_ONLY_MAIN_ARGS = ("cu_seqlens", "metadata_from_cu_seqlens", "preprocess_status")
# ``metadata_from_cu_seqlens`` of the fused main kernel names its sequence-table
# source: the published / caller ``seq_chunk_cumsum`` vector (every program
# but the two below; the two-kernel and chunk-parallel programs pass 0), the
# ``cu_seqlens`` closed form derived per item (``exactfused`` in the cu_seqlens
# form), or -- the ``exactone`` programs only -- the table the CTA prologue
# derives from the int32 / int64 seq_idx triple into its private row of the
# ``seq_chunk_cumsum`` scratch.
_TABLE_PUBLISHED = 0
_TABLE_CU_SEQLENS = 1
_TABLE_IN_KERNEL_I32 = 2
_TABLE_IN_KERNEL_I64 = 3
# The retained start two-kernel program (``exact_*``) predates the three
# fused-only inputs: its launcher binds the 41 arguments before them.
_MAIN_ARGS_START = _MAIN_ARGS[: len(_MAIN_ARGS) - len(_FUSED_ONLY_MAIN_ARGS)]
# Chunk-parallel main kernel (the ``chunkpar_*`` programs, the round-6
# single-pass look-back program): the exact-scan order with the state-operand
# tensor map ``h_map`` after ``out_map``, the workspace pointers ``s_work``
# (f32 per-tile state increments), ``agg_work`` (f32 published range
# aggregates), ``h_words`` (u32 view of the bf16 state operand behind
# ``h_map``) and ``grid_barrier`` (u32 ``[ticket, epoch, pad, pad, flags]``)
# after ``out_native``, and the virtual CTA count ``vcount`` last.
_MAIN_ARGS_CHUNKPAR = (
    "x_map",
    "b_map",
    "c_map",
    "out_map",
    "h_map",
    "x",
    "dt",
    "delta_precomputed",
    "cumsum_precomputed",
    "A",
    "B_tensor",
    "C",
    "D",
    "z",
    "dt_bias",
    "initial_states",
    "final_states",
    "checkpoint_states",
    "checkpoint_token_indices",
    "checkpoint_state_slots",
    "seq_idx_i32",
    "seq_idx_i64",
    "chunk_indices",
    "chunk_offsets",
    "seq_chunk_cumsum",
    "out_native",
    "s_work",
    "agg_work",
    "h_words",
    "grid_barrier",
    "nheads",
    "ngroups",
    "batch",
    "seqlen",
    "nchunks",
    "sequence_count",
    "num_logical_chunks",
    "mode_varlen",
    "D_mode",
    "has_z",
    "has_initial",
    "dt_softplus",
    "dt_min",
    "dt_max",
    "write_final_states",
    "checkpoint_state_count",
    "vcount",
)
# Three-phase chunk-parallel main kernel (the ``chunkpar3_*`` programs, the
# round-4 program kept as the second chunk-parallel family): the exact-scan
# order with the state-operand tensor map ``h_map`` after ``out_map`` and the
# workspace pointers ``s_work``, ``h_words`` and ``grid_barrier`` (u32
# ``[arrive, generation]``) after ``out_native``.
_MAIN_ARGS_CHUNKPAR3 = (
    "x_map",
    "b_map",
    "c_map",
    "out_map",
    "h_map",
    "x",
    "dt",
    "delta_precomputed",
    "cumsum_precomputed",
    "A",
    "B_tensor",
    "C",
    "D",
    "z",
    "dt_bias",
    "initial_states",
    "final_states",
    "checkpoint_states",
    "checkpoint_token_indices",
    "checkpoint_state_slots",
    "seq_idx_i32",
    "seq_idx_i64",
    "chunk_indices",
    "chunk_offsets",
    "seq_chunk_cumsum",
    "out_native",
    "s_work",
    "h_words",
    "grid_barrier",
    "nheads",
    "ngroups",
    "batch",
    "seqlen",
    "nchunks",
    "sequence_count",
    "num_logical_chunks",
    "mode_varlen",
    "D_mode",
    "has_z",
    "has_initial",
    "dt_softplus",
    "dt_min",
    "dt_max",
    "write_final_states",
    "checkpoint_state_count",
)
# Per family: host template, main-kernel argument order, main dynamic SMEM.
_FAMILY_HOST = {
    "exact": (_HOST_TEMPLATE, _MAIN_ARGS_START, _EXACT_SMEM_BYTES),
    "exactfused": (_EXACTFUSED_HOST_TEMPLATE, _MAIN_ARGS, _EXACTFUSED_SMEM_BYTES),
    "exactmeta": (_EXACTMETA_HOST_TEMPLATE, _MAIN_ARGS, _EXACTFUSED_SMEM_BYTES),
    "exactone": (_EXACTFUSED_HOST_TEMPLATE, _MAIN_ARGS, _EXACTONE_SMEM_BYTES),
    "chunkpar": (_CHUNKPAR_HOST_TEMPLATE, _MAIN_ARGS_CHUNKPAR, _CHUNKPAR_SMEM_BYTES),
    "exactpre": (_EXACTFUSED_HOST_TEMPLATE, _MAIN_ARGS, _EXACTPRE_SMEM_BYTES),
    "chunkpar3": (
        _CHUNKPAR3_HOST_TEMPLATE,
        _MAIN_ARGS_CHUNKPAR3,
        _CHUNKPAR3_SMEM_BYTES,
    ),
}


def _scan(module: str) -> _Kernel:
    """The main kernel of ``generated/device/<module>.cu`` (either family)."""

    symbol = module.rsplit("_", 1)[0]
    return _Kernel(module, f"kernel_{symbol}", threads=512, fast_math=True)


# Main dynamic SMEM of a family's non-default schedule variants.
_VARIANT_SMEM_BYTES = {
    ("exact", "noring"): _EXACT_NORING_SMEM_BYTES,
    ("exactfused", "noring"): _EXACTFUSED_NORING_SMEM_BYTES,
    ("exactmeta", "noring"): _EXACTFUSED_NORING_SMEM_BYTES,
    ("exactone", "noring"): _EXACTONE_NORING_SMEM_BYTES,
    ("exactpre", "tail"): _EXACTPRE_TAIL_SMEM_BYTES,
    ("exactpre", "claim"): _EXACTPRE_CLAIM_SMEM_BYTES,
    ("exactpre", "ibulk"): _EXACTPRE_IBULK_SMEM_BYTES,
    ("exactpre", "tailone"): _EXACTPRE_TAILONE_SMEM_BYTES,
    ("exactfused", "ibulk"): _EXACTFUSED_IBULK_SMEM_BYTES,
    ("exactone", "tablefirst"): _EXACTONE_SMEM_BYTES,
    ("exactone", "tableroles"): _EXACTONE_SMEM_BYTES,
}
# Main dynamic SMEM of the sm_107a renders (``_SM107_MODULES`` entries) whose
# pool differs from their family and variant literal, by (family, mode,
# variant); every other program, and every program on sm_100a and sm_103a,
# launches with the literal above.
_SM107_VARIANT_SMEM_BYTES = {
    ("exactpre", "varlen", "tail"): _EXACTPRE_TAIL_VARLEN_SM107_SMEM_BYTES,
}
# Cluster shape of the main kernels (the Cake export checks the generated
# sources against it): the ``_tail`` pair programs launch as 2-CTA clusters
# (``__cluster_dims__(2, 1, 1)`` in the device source, honored by a plain
# launch, so the host templates are unchanged); every other program launches
# without clusters.
_VARIANT_CLUSTER_DIMS = {
    ("exactpre", "tail"): (2, 1, 1),
}


def _program(name: str, module: str) -> _Program:
    """Bind program ``<family>_<state>_<mode>[_<variant>]`` to its generated main source."""

    family, state_key, _mode, *variant = name.split("_")
    code, bits = _STATE_DTYPE_CODES[state_key]
    host_template, main_args, smem_bytes = _FAMILY_HOST[family]
    variant_name = ""
    if variant:
        (variant_name,) = variant
        smem_bytes = _VARIANT_SMEM_BYTES[(family, variant_name)]
    preprocess, preprocess_args = (
        (_PACKED_METADATA, _METADATA_ARGS)
        if family == "exactmeta"
        else (_SEGMENT_PREPROCESS, _PREPROCESS_ARGS)
    )
    return _Program(
        family,
        preprocess,
        _scan(module),
        code,
        bits,
        smem_bytes,
        host_template,
        main_args,
        preprocess_args,
        variant_name,
        _VARIANT_CLUSTER_DIMS.get((family, variant_name), (1, 1, 1)),
    )


_PROGRAMS: dict[str, _Program] = {
    name: _program(name, module)
    for name, module in (
        *_SCAN_MODULES.items(),
        *_EXACTFUSED_MODULES.items(),
        *_CHUNKPAR_MODULES.items(),
        *_EXACTPRE_MODULES.items(),
        *_CHUNKPAR3_MODULES.items(),
        *_EXACTMETA_MODULES.items(),
        *_EXACTONE_MODULES.items(),
    )
}
# The sm_107a-only programs (the claim, initial-state bulk and one-CTA pair
# programs, and the one-launch programs' table forms): their
# ``_SM107_MODULES`` entry is their main module on every architecture; only
# capability (10, 7) routes to them (:func:`claim_selected`,
# :func:`ibulk_selected`, :func:`tail_one_cta_selected`,
# :func:`fused_ibulk_selected`, :func:`exactone_table_variant`).
_SM107_ONLY_PROGRAMS = frozenset(_SM107_MODULES) - frozenset(_PROGRAMS)
_PROGRAMS.update(
    (name, _program(name, _SM107_MODULES[name]))
    for name in sorted(_SM107_ONLY_PROGRAMS)
)


def _program_name(
    family: str, state_dtype: torch.dtype, mode_varlen: bool, variant: str = ""
) -> str:
    """The program of ``family`` serving a state dtype in batched or
    packed-varlen mode; ``variant`` names a non-default schedule variant of the
    family (``"noring"`` for the exact-scan families other than the two-kernel
    one, ``"tail"``, ``"tailone"``, ``"claim"`` or ``"ibulk"`` for the
    two-kernel exact-scan family, ``"ibulk"`` for the fused one,
    ``"tablefirst"`` or ``"tableroles"`` for the one-launch family)."""

    mode_key = "varlen" if mode_varlen else "batched"
    suffix = f"_{variant}" if variant else ""
    return f"{family}_{_STATE_DTYPE_KEYS[state_dtype]}_{mode_key}{suffix}"


def _main_module(name: str, arch: str) -> str:
    """The main-kernel module program ``name`` builds for ``arch``: its
    ``_SM107_MODULES`` entry on sm_107a where the table has one, else the
    program's fold module."""

    module = _PROGRAMS[name].main.module
    if arch == _SM107_ARCH:
        return _SM107_MODULES.get(name, module)
    return module


def _arch_program(name: str, arch: str) -> _Program:
    """Program ``name`` as built for ``arch``: the table binding with its main
    kernel rebound to :func:`_main_module` and, on sm_107a, its main dynamic
    SMEM to ``_SM107_VARIANT_SMEM_BYTES`` for a render of a kind it lists and
    its segment preprocess rebound to ``_SM107_SEGMENT_PREPROCESS`` -- or to its
    early-trigger twin ``_SM107_SEGMENT_PREPROCESS_TRIGGER`` when the main
    kernel launches with programmatic dependent launch (:func:`_main_pdl`) and
    the family is in ``_PDL_TRIGGER_FAMILIES`` (the same symbol, thread count
    and build flags; only the module identity, hence the device source,
    differs)."""

    program = _PROGRAMS[name]
    module = _main_module(name, arch)
    main = program.main if module == program.main.module else _scan(module)
    preprocess = program.preprocess
    if arch == _SM107_ARCH and program.preprocess is _SEGMENT_PREPROCESS:
        preprocess = (
            _SM107_SEGMENT_PREPROCESS_TRIGGER
            if _main_pdl(name, arch) and program.family in _PDL_TRIGGER_FAMILIES
            else _SM107_SEGMENT_PREPROCESS
        )
    smem_bytes = program.main_smem_bytes
    if arch == _SM107_ARCH and name in _SM107_MODULES:
        mode = name.split("_")[2]
        smem_bytes = _SM107_VARIANT_SMEM_BYTES.get(
            (program.family, mode, program.variant), smem_bytes
        )
    if (
        main is program.main
        and preprocess is program.preprocess
        and smem_bytes == program.main_smem_bytes
    ):
        return program
    return replace(
        program, main=main, preprocess=preprocess, main_smem_bytes=smem_bytes
    )


# The sm_107a two-kernel programs whose main kernel waits on the preprocess
# grid (``griddepcontrol.wait`` ahead of its first access to the preprocess
# outputs): their main stage launches with programmatic dependent launch, so
# the main grid's launch and CTA prologue overlap the preprocess tail.  Every
# other program, and every program on sm_100a and sm_103a, launches its main
# kernel stream-ordered as before.
_PDL_FAMILIES = frozenset({"exactpre", "chunkpar", "chunkpar3"})
# The PDL families whose preprocess triggers the dependent launch at its start
# (``_SM107_SEGMENT_PREPROCESS_TRIGGER``), so the main grid launches while the
# preprocess still runs: the chunk-parallel three-phase programs.  The
# ``exactpre`` and look-back (``chunkpar``) main kernels measured no gain with
# the early launch and keep the preprocess that triggers at CTA exit.
_PDL_TRIGGER_FAMILIES = frozenset({"chunkpar3"})


def _main_pdl(name: str, arch: str) -> bool:
    """Whether program ``name`` launches its main kernel with programmatic
    dependent launch on ``arch``: the sm_107a render (``_SM107_MODULES``) of a
    two-kernel family."""

    return (
        arch == _SM107_ARCH
        and _PROGRAMS[name].family in _PDL_FAMILIES
        and name in _SM107_MODULES
    )


def _direct_preprocess_inputs(
    *,
    dt: object,
    A: object,
    dt_bias: object,
    segment_starts: object,
    segment_lengths: object,
    chunk_indices: object,
    chunk_offsets: object,
    delta: object,
    cumsum: object,
    num_segments: int,
    nheads: int,
    seqlen: int,
    mode_varlen: bool,
    dt_softplus: bool,
    dt_limit: Tuple[float, float],
    tiles_per_block: int,
    seq_idx_i32: object,
    seq_idx_i64: object,
    seq_idx_int64: bool,
    seq_chunk_cumsum: object,
    num_sequences: int,
    write_seq_chunk_cumsum: bool,
    cu_seqlens: object,
    checkpoint_token_indices: object,
    metadata_from_cu_seqlens: bool,
    checkpoint_state_count: int,
    preprocess_status: object,
) -> tuple[dict[str, object], tuple[int, int, int]]:
    """Build the metadata-fused preprocess values and launch grid.

    ``num_segments`` is the real segment count of the triple / batched forms
    and the host bound ``_segment_bound`` of the ``cu_seqlens`` form (the
    kernel derives the real count and never reads past its sentinel).
    """

    if tiles_per_block <= 0:
        raise ValueError(
            f"preprocess tiles per block must be positive, got {tiles_per_block}"
        )
    dt_min, dt_max = (float(value) for value in dt_limit)
    values: dict[str, object] = {
        "dt": dt,
        "A": A,
        "dt_bias": dt_bias,
        "segment_starts": segment_starts,
        "segment_lengths": segment_lengths,
        "chunk_indices": chunk_indices,
        "chunk_offsets": chunk_offsets,
        "delta": delta,
        "cumsum": cumsum,
        "num_segments": num_segments,
        "nheads": nheads,
        "seqlen": seqlen,
        "direct_varlen_metadata": int(mode_varlen),
        "dt_softplus": int(dt_softplus),
        "dt_min": dt_min,
        "dt_max": dt_max,
        "seq_idx_i32": seq_idx_i32,
        "seq_idx_i64": seq_idx_i64,
        "seq_idx_int64": int(seq_idx_int64),
        "seq_chunk_cumsum": seq_chunk_cumsum,
        "num_sequences": num_sequences,
        "write_seq_chunk_cumsum": int(write_seq_chunk_cumsum),
        "cu_seqlens": cu_seqlens,
        "checkpoint_token_indices": checkpoint_token_indices,
        "metadata_from_cu_seqlens": int(metadata_from_cu_seqlens),
        "checkpoint_state_count": int(checkpoint_state_count),
        "preprocess_status": preprocess_status,
    }
    total_tiles = num_segments * nheads
    return values, (
        max(1, (total_tiles + tiles_per_block - 1) // tiles_per_block),
        1,
        1,
    )


def _metadata_preprocess_inputs(
    *,
    seq_idx_i32: object,
    seq_idx_i64: object,
    chunk_indices: object,
    chunk_offsets: object,
    segment_starts: object,
    segment_lengths: object,
    sequence_offsets: object,
    num_segments: int,
    num_sequences: int,
    seqlen: int,
    seq_idx_int64: bool,
    preprocess_status: object,
    threads: int,
) -> tuple[dict[str, object], tuple[int, int, int]]:
    """Build the light metadata stage's values (``_METADATA_ARGS``) and launch
    grid, one thread per logical segment: ``ceil(num_segments / threads)``
    CTAs of ``threads``.  ``num_segments`` is the triple form's real segment
    count (``chunk_indices.numel()``)."""

    if threads <= 0:
        raise ValueError(f"metadata threads per block must be positive, got {threads}")
    values: dict[str, object] = {
        "seq_idx_i32": seq_idx_i32,
        "seq_idx_i64": seq_idx_i64,
        "chunk_indices": chunk_indices,
        "chunk_offsets": chunk_offsets,
        "segment_starts": segment_starts,
        "segment_lengths": segment_lengths,
        "sequence_offsets": sequence_offsets,
        "num_segments": int(num_segments),
        "num_sequences": int(num_sequences),
        "seqlen": int(seqlen),
        "seq_idx_int64": int(seq_idx_int64),
        "preprocess_status": preprocess_status,
    }
    return values, (max(1, (int(num_segments) + threads - 1) // threads), 1, 1)


def _segment_bound(seqlen: int, num_sequences: int) -> int:
    """Segment-count bound of the ``cu_seqlens`` form: every 128-token chunk
    of the packed stream plus one unaligned start and one unaligned checkpoint
    per sequence.  Sizes the preprocess grid, the delta/cumsum workspace and
    the ``[bound + 1]`` metadata tables; the real count stays on the device."""

    return -(-int(seqlen) // _CHUNK_SIZE) + 2 * int(num_sequences)


# Exact-scan launch grid.  This function mirrors the generator's host-side
# grid rule: the Cake export refuses a loader whose body or values over the
# generator's probe vector differ from that rule, and the source test
# ``test_source_persistent_grid_mirrors_the_seed_rule`` pins the values on
# the shipped SM counts.  The grid is a schedule knob only: every item's
# arithmetic is the same at any grid (grid-stride loop), so the rule changes
# the time of a call, never its output.
def _persistent_grid_size(*, total_work: int, sm_count: int) -> int:
    """Balanced persistent grid for ``total_work`` (sequence, head) items:
    one CTA per item up to one wave; beyond it the smallest count of two to
    four items per CTA that divides ``total_work`` and keeps at least four
    fifths of the SMs busy; else one CTA per SM."""

    full_grid = min(total_work, sm_count)
    if total_work <= sm_count:
        return full_grid
    for items_per_cta in range(2, 5):
        if total_work % items_per_cta:
            continue
        balanced_grid = total_work // items_per_cta
        if balanced_grid <= sm_count and balanced_grid * 5 >= sm_count * 4:
            return balanced_grid
    return full_grid


# ---------------------------------------------------------------------------
# Kernel-family selection.  The constants below are the calibrated main-kernel
# cost model of the Cake chunk-parallel seed module
# (``CHUNK_PARALLEL_COST_MODEL_US`` and the ``CHUNK_PARALLEL_*`` rule constants
# of loom/examples/weave/
# flashinfer_blackwell_mamba_ssd_combined_chunk_parallel_seed_v1.py), copied
# verbatim: both copies must be refreshed together whenever the seed is
# recalibrated (CAKE-956 round 5, 2026-10-11: B200 step
# b74734547976789d7eb7036b, B300 step 8ae022395365a025939cf937).
# Least-squares fits of the main-kernel time in microseconds:
#   exact scan     = serial_fixed + per_chunk(work_items) * chunks
#     per_chunk is flat up to 32 concurrent (sequence, head) items and larger
#     at 128 items (128 CTAs streaming x/B/C through L2); item counts in
#     between interpolate linearly, above 128 use the large value.
#   chunk-parallel = cp_fixed + cp_per_tile * max(0, tiles - sm_count)
#     the first wave of sm_count tiles and the grid-barrier-separated phases
#     are inside cp_fixed; each further (chunk, head) tile costs cp_per_tile.
_CHUNK_PARALLEL_COST_MODEL_US = {
    # (major, minor) compute capability -> calibrated constants.
    (10, 0): {  # B200, sm_100a
        "serial_fixed": 4.69,
        "serial_per_chunk_small": 2.594,
        "serial_per_chunk_large": 2.809,
        "cp_fixed": 23.3,
        "cp_per_tile": 0.0513,
    },
    (10, 3): {  # B300 / GB300, sm_103a
        "serial_fixed": 4.33,
        "serial_per_chunk_small": 2.395,
        "serial_per_chunk_large": 2.51,
        "cp_fixed": 21.6,
        "cp_per_tile": 0.0459,
    },
    (10, 7): {  # Rubin, sm_107a (212 SMs); calibrated on the same 24-cell
        # protocol as the Blackwell rows (main-kernel medians, cold L2).
        "serial_fixed": 4.07,
        "serial_per_chunk_small": 1.841,
        "serial_per_chunk_large": 1.962,
        "cp_fixed": 20.0,
        "cp_per_tile": 0.0302,
    },
}
# Capabilities without a calibrated row use the B200 constants.
_CHUNK_PARALLEL_COST_MODEL_DEFAULT = (10, 0)
_CHUNK_PARALLEL_SMALL_ITEMS = 32
_CHUNK_PARALLEL_LARGE_ITEMS = 128
_CHUNK_PARALLEL_SELECTION_MARGIN = 1.05
_CHUNK_PARALLEL_WORKSPACE_CAP_BYTES = 256 << 20
# Per-capability cap: sm_107a admits 8192-tile calls (384 MiB) -- 1 sequence x
# 1024 chunks or 4 x 256 chunks at 8 heads -- which the exact scan serves 2-7x
# slower; the workspace is grow-only per device, so the cap bounds that growth.
_CHUNK_PARALLEL_WORKSPACE_CAP_BYTES_BY_CAPABILITY = {(10, 7): 512 << 20}


def chunk_parallel_workspace_cap_bytes(capability) -> int:
    """Workspace cap of the chunk-parallel regime for ``capability``."""

    return _CHUNK_PARALLEL_WORKSPACE_CAP_BYTES_BY_CAPABILITY.get(
        (int(capability[0]), int(capability[1])), _CHUNK_PARALLEL_WORKSPACE_CAP_BYTES
    )


# Third cost arm: the three-phase program's own
# calibrated constants (``CHUNK_PARALLEL_COST_MODEL_US`` of the Cake seed
# module flashinfer_blackwell_mamba_ssd_combined_chunk_parallel_seed_v1, round
# 4, copied verbatim for the Blackwell rows; the (10, 7) row was fit on Rubin
# with the same protocol); they take part in the exact-vs-chunk-parallel gate
# only (the faster of the two chunk-parallel predictions is compared with the
# exact scan's).
_THREE_PHASE_COST_MODEL_US = {
    (10, 0): {"cp_fixed": 25.0, "cp_per_tile": 0.0491},
    (10, 3): {"cp_fixed": 24.1, "cp_per_tile": 0.0429},
    (10, 7): {"cp_fixed": 21.0, "cp_per_tile": 0.0266},
}
# Chunk-parallel FAMILY rule inside the chunk-parallel regime, mirrored from
# the look-back seed module (``three_phase_preferred``): the three-phase
# program serves batched calls with at most ``_THREE_PHASE_MAX_TILES`` tiles
# (at most ~1.7 tiles per look-back range, where the look-back's per-range
# aggregate publish and fold dominate; measured look-back / three-phase
# 1.05-1.32 at 128-256 tiles on B200 and GB300) or with at least
# ``_THREE_PHASE_MIN_ITEMS`` work items and ``_THREE_PHASE_MIN_CHUNKS`` chunks
# per item (32 items x 64 chunks: 1.02-1.03); every other chunk-parallel call
# runs the look-back program (32 items x 16 chunks: 0.91-0.95; 16 items x
# 32-256 chunks: 0.92-0.99).  Varlen calls stay on the look-back (three-phase
# varlen rows measured 7-13 % slower).
_THREE_PHASE_MAX_TILES = 256
_THREE_PHASE_MIN_ITEMS = 32
_THREE_PHASE_MIN_CHUNKS = 64
# Rubin (sm_107a, 212 SMs) measured the opposite ordering over most of the
# chunk-parallel regime (three-phase / look-back 0.90-0.97 at 8 items x
# 64-128 chunks, 16 items x 32-128 chunks and 32 items x 32 chunks, main
# kernel, cold L2); the look-back stays ahead only at 32 items x 16 chunks
# (1.01-1.02) and 8 items x 256 chunks (1.01).  With the PDL-triggered
# preprocess (forced routes on R200, whole call, cold
# L2) the look-back leads between ``_RUBIN_LOOKBACK_TILES`` tiles (more than
# 512, at most 2048: 1.04-1.14x at 8-32 items x 32-128 chunks) and the
# three-phase program on both sides (512 tiles at 32 chunks 0.97, 4096
# tiles 0.96).  Its rule: three-phase for batched calls with at most
# ``_THREE_PHASE_MAX_TILES`` tiles or at least
# ``_RUBIN_THREE_PHASE_MIN_CHUNKS`` chunks per item, except calls inside
# the ``_RUBIN_LOOKBACK_TILES`` band and calls of at most
# ``_RUBIN_LOOKBACK_MAX_ITEMS`` items with at least
# ``_RUBIN_LOOKBACK_MIN_CHUNKS`` chunks; varlen calls stay on the look-back.
_RUBIN_THREE_PHASE_CAPABILITIES = ((10, 7),)
_RUBIN_THREE_PHASE_MIN_CHUNKS = 32
_RUBIN_LOOKBACK_MAX_ITEMS = 8
_RUBIN_LOOKBACK_MIN_CHUNKS = 256
_RUBIN_LOOKBACK_TILES = (512, 2048)  # exclusive lower, inclusive upper bound
# Chunk-parallel workspace per (chunk, head) tile: the f32 [64, 128] state
# increments (32 KB) plus the bf16 [64, 128] state operand (16 KB).
_CHUNK_PARALLEL_WORKSPACE_BYTES_PER_TILE = _HEADDIM * _DSTATE * (4 + 2)
# Look-back device state, mirrored from the Cake seed module
# (flashinfer_blackwell_mamba_ssd_combined_chunk_parallel_lookback_seed_v1.py):
# the item-major tile range is split into ``vcount`` virtual CTA ranges (V =
# the SM count the program balances against per capability; the B200 value
# for an unmeasured capability; every floating-point result depends on
# (inputs, V) only, never on the launch grid).  The host passes the table
# value capped at the device's SM count (``_chunk_parallel_vcount``): every
# CTA claims ranges until none is left, so a board with fewer SMs than V
# runs a second round of ranges on some CTAs -- the 208-SM R200 board
# variant against 212 took 1.35-1.52x the 216-SM boards' time; capped at
# 208 it measures 0.93-0.98x of those boards' times, and a board with at
# least V SMs keeps the table value.  ``grid_barrier`` is u32
# ``[TICKET, EPOCH, pad, pad, FLAG[0..SEGMENT_CAP)]`` (zeroed once per device;
# the kernel's claim counter wraps itself, the epoch free-runs and the flags
# carry the epoch).  ``agg_work`` holds V f32 [64, 128] range aggregates
# followed by V f32 log-decay scalars.
_CHUNK_PARALLEL_SEGMENT_CAP = 1024
_CHUNK_PARALLEL_GRID_BARRIER_WORDS = 4 + _CHUNK_PARALLEL_SEGMENT_CAP
_CHUNK_PARALLEL_VCOUNT_BY_ARCH = {(10, 0): 148, (10, 3): 152, (10, 7): 212}
_CHUNK_PARALLEL_VCOUNT_DEFAULT = (10, 0)
_CHUNK_PARALLEL_AGG_ELEMS_PER_RANGE = _HEADDIM * _DSTATE + 1
# Three-phase device state: u32 ``[arrive, generation]`` of its software grid
# barrier (zeroed once per device; the arrive counter wraps to zero after
# ``gridDim.x`` arrivals and the generation word free-runs).  Its word
# protocol differs from the look-back's ``[ticket, epoch, ...]``, so the two
# families never share a barrier buffer.
_CHUNKPAR3_GRID_BARRIER_WORDS = 2


def _chunk_parallel_vcount(
    capability: tuple[int, int], sm_count: Optional[int] = None
) -> int:
    """The look-back program's virtual CTA count V: the compute capability's
    table value (the SM count of the part the program balances its tiles
    against), capped at ``sm_count`` when the device's SM count is given.
    The cap keeps one range per CTA on a board with fewer SMs than the table
    value and leaves a board with at least that many SMs unchanged.  The
    result depends on (inputs, V) only, so every board of one SM count
    computes the same values; board variants of one capability differ within
    the precision gate.  Mirrors the Cake kernel-tests host rule
    ``chunk_parallel_vcount``."""

    key = (int(capability[0]), int(capability[1]))
    vcount = _CHUNK_PARALLEL_VCOUNT_BY_ARCH.get(
        key, _CHUNK_PARALLEL_VCOUNT_BY_ARCH[_CHUNK_PARALLEL_VCOUNT_DEFAULT]
    )
    if sm_count is not None:
        vcount = min(int(vcount), int(sm_count))
    return max(1, min(int(vcount), _CHUNK_PARALLEL_SEGMENT_CAP))


# ``auto`` applies the rule; ``always`` forces the look-back family,
# ``three_phase`` the three-phase family, ``never`` the exact scan.  Read on
# every unprepared eager call; ``CakeSSDCombined.prepare`` reads it once and
# freezes it for the call shape (a captured call never reads it).
_CHUNK_PARALLEL_ENV = "FLASHINFER_CAKE_SSD_CHUNK_PARALLEL"
_CHUNK_PARALLEL_MODES = ("auto", "always", "three_phase", "never")
_CHUNK_PARALLEL_MODE_TEXT = ", ".join(_CHUNK_PARALLEL_MODES)


def _chunk_parallel_mode() -> str:
    """The ``FLASHINFER_CAKE_SSD_CHUNK_PARALLEL`` override of the current call."""

    value = os.environ.get(_CHUNK_PARALLEL_ENV, "auto")
    mode = value.strip().lower()
    if mode not in _CHUNK_PARALLEL_MODES:
        raise ValueError(
            f"{_CHUNK_PARALLEL_ENV} must be one of {_CHUNK_PARALLEL_MODE_TEXT}; got {value!r}"
        )
    return mode


def _stream_capturing(device: torch.device) -> bool:
    """Whether the current CUDA stream of ``device`` -- the stream ``run``
    launches on -- is recording a CUDA graph.

    ``False`` where CUDA is unavailable or ``device`` is not a CUDA device,
    so device-less hosts (the CPU-only loader tests) take the eager path
    without touching the CUDA runtime.
    """

    if not torch.cuda.is_available() or device.type != "cuda":
        return False
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


class CakeSSDCombinedCaptureError(RuntimeError):
    """A :meth:`CakeSSDCombined.run` inside CUDA-graph capture would have done
    work a graph cannot record: materialise the workspace of a call shape
    this runner has never run or prepared, or load (possibly build) a
    program that has not been launched in this process; or
    :meth:`CakeSSDCombined.prepare` was called under capture.  Run the call
    shape eagerly once (or prepare it and run it once) before capturing it.
    Allocations under capture (``out``, final states, packed-input copies)
    are legal: PyTorch serves them from the graph's private pool."""


class _WorkspaceKey(NamedTuple):
    """Identity of a per-call-shape workspace.  The family override is part
    of it because it selects the program and therefore the values the
    workspace binds; the device index because the buffers live on it.
    ``from_cu_seqlens``, ``caller_cumsum`` and ``publish_table`` are the three
    packed-varlen metadata facts the exact-scan route reads
    (``_materialize_workspace``): the ``cu_seqlens`` form; the ``seq_idx``
    triple form with a caller ``seq_chunk_cumsum`` the preprocess does not
    rewrite (``update_seq_chunk_cumsum=False``), where the fused program is
    one launch; and the triple form publishing the table into the caller's
    buffer (``update_seq_chunk_cumsum=True``), which needs a publishing stage
    (the one-launch ``exactone`` program keeps its table private).  All three
    are derived from the call's arguments alone, so ``prepare`` and ``run``
    of the same call resolve the same key."""

    device_index: Optional[int]
    batch: int
    seqlen: int
    nchunks: int
    num_segments: int
    num_sequences: int
    state_dtype: torch.dtype
    from_cu_seqlens: bool
    caller_cumsum: bool
    publish_table: bool
    chunk_parallel_mode: str


# ``_WorkspaceKey`` without the family override: what ``prepare`` freezes the
# override for.
_ShapeKey = Tuple[Optional[int], int, int, int, int, int, torch.dtype, bool, bool, bool]


@dataclass(frozen=True)
class PreparedSSDCombined:
    """What :meth:`CakeSSDCombined.prepare` resolved for one call shape: the
    program it launches, the workspace key it binds and both launch grids.
    Whether a CUDA graph has since been captured with the shape is a runner
    query (:meth:`CakeSSDCombined.is_captured`)."""

    program_name: str
    workspace_key: _WorkspaceKey
    grid: tuple[int, int, int]
    preprocess_grid: tuple[int, int, int]


@dataclass
class _WorkspaceEntry:
    """One registered workspace; ``captured`` (a CUDA graph replays into it)
    and ``prepared`` (:meth:`CakeSSDCombined.prepare` handed out its static
    buffers) each pin it (``_WorkspaceRegistry``)."""

    workspace: dict[str, Any]
    captured: bool = False
    prepared: bool = False


class _WorkspaceRegistry:
    """The keyed workspaces of one runner.

    Unpinned entries form a bounded LRU (``capacity`` of them, the least
    recently used evicted first), so returning to a recent call shape
    re-allocates nothing while a sweep over many shapes cannot hoard device
    memory.  An entry a CUDA graph was captured with is pinned: its tensors
    are the addresses the graph replays, so evicting (or re-zeroing) it would
    make the replays touch freed storage; an entry ``prepare`` materialised
    is pinned too, because the static buffers it handed out must stay the
    ones a later eager or captured call binds.  Pinned entries never count
    against the capacity.  Pure host bookkeeping: keys and workspaces are
    opaque here.
    """

    def __init__(self, capacity: int) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError(
                f"workspace capacity must be a positive int, got {capacity!r}"
            )
        self.capacity = capacity
        self._entries: OrderedDict[Hashable, _WorkspaceEntry] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._entries

    def keys(self) -> list[Hashable]:
        """Registered keys, least recently used first."""

        return list(self._entries)

    def get(self, key: Hashable) -> Optional[_WorkspaceEntry]:
        """The entry of ``key`` (now the most recently used) or ``None``."""

        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
        return entry

    def add(self, key: Hashable, workspace: dict[str, Any]) -> _WorkspaceEntry:
        """Register ``workspace`` under the new ``key`` as the most recently
        used entry and evict the least recently used unpinned entries beyond
        the capacity (never the one just added)."""

        if key in self._entries:
            raise KeyError(f"workspace {key!r} is already registered")
        entry = _WorkspaceEntry(workspace)
        self._entries[key] = entry
        evictable = [
            k for k, e in self._entries.items() if not (e.captured or e.prepared)
        ]
        for stale in evictable[: max(0, len(evictable) - self.capacity)]:
            del self._entries[stale]
        return entry

    def hold(self, key: Hashable) -> _WorkspaceEntry:
        """Mark ``key`` prepared: it is never evicted from now on."""

        entry = self._entries[key]
        entry.prepared = True
        return entry

    def pin(self, key: Hashable) -> _WorkspaceEntry:
        """Mark ``key`` captured: it is never evicted from now on."""

        entry = self._entries[key]
        entry.captured = True
        return entry

    def captured(self, key: Hashable) -> bool:
        return key in self._entries and self._entries[key].captured


@dataclass
class _LaunchPlan:
    """Everything one call binds, resolved before any device work: the
    stage values in launcher order, both grids and the packed-input copies
    to issue first.  ``run`` executes it; ``prepare`` only builds it."""

    program_name: str
    arch: str
    workspace_key: _WorkspaceKey
    workspace: dict[str, Any]
    preprocess: dict[str, object]
    preprocess_grid: tuple[int, int, int]
    main: dict[str, object]
    grid: tuple[int, int, int]
    pending_copies: list[tuple[torch.Tensor, torch.Tensor]]
    out: Optional[torch.Tensor]
    final: Optional[torch.Tensor]
    # The chunk-parallel buffers the call binds (``None`` for the exact scan);
    # a captured call pins them into its workspace.
    chunk_parallel: Optional[dict[str, torch.Tensor]]


def selection_quantities(
    *,
    nheads: int,
    num_sequences: int,
    num_segments: int,
    nchunks: int,
    mode_varlen: bool,
    sm_count: int,
    capability: tuple[int, int],
) -> dict[str, Any]:
    """The quantities the family rule reads and the three programs' predicted
    main-kernel times (microseconds) for one call.

    ``work_items`` is the exact scan's concurrency (one CTA per (sequence,
    head) item); ``tiles`` the chunk-parallel tile bound (``num_segments *
    nheads``; varlen calls pass the host segment bound ``_segment_bound`` in
    both metadata forms, so the rule is independent of the form that carries
    the geometry); ``chunks`` the serial chunk count per item: ``nchunks``
    when batched, the mean bound per packed sequence when varlen (the exact
    per-sequence counts live on the device).
    """

    key = (int(capability[0]), int(capability[1]))
    model = _CHUNK_PARALLEL_COST_MODEL_US.get(
        key, _CHUNK_PARALLEL_COST_MODEL_US[_CHUNK_PARALLEL_COST_MODEL_DEFAULT]
    )
    three_phase = _THREE_PHASE_COST_MODEL_US.get(
        key, _THREE_PHASE_COST_MODEL_US[_CHUNK_PARALLEL_COST_MODEL_DEFAULT]
    )
    work_items = int(num_sequences) * int(nheads)
    tiles = int(num_segments) * int(nheads)
    if mode_varlen:
        chunks = -(-int(num_segments) // max(1, int(num_sequences)))
    else:
        chunks = int(nchunks)
    span = _CHUNK_PARALLEL_LARGE_ITEMS - _CHUNK_PARALLEL_SMALL_ITEMS
    weight = min(1.0, max(0.0, (work_items - _CHUNK_PARALLEL_SMALL_ITEMS) / span))
    per_chunk = model["serial_per_chunk_small"] + weight * (
        model["serial_per_chunk_large"] - model["serial_per_chunk_small"]
    )
    extra_tiles = max(0, tiles - int(sm_count))
    return {
        "work_items": work_items,
        "sm_count": int(sm_count),
        "capability": (int(capability[0]), int(capability[1])),
        "chunks": chunks,
        "tiles": tiles,
        "mode_varlen": bool(mode_varlen),
        "workspace_bytes": tiles * _CHUNK_PARALLEL_WORKSPACE_BYTES_PER_TILE,
        "predicted_us": {
            "exact_scan": model["serial_fixed"] + per_chunk * chunks,
            "chunk_parallel": model["cp_fixed"] + model["cp_per_tile"] * extra_tiles,
            "three_phase": three_phase["cp_fixed"]
            + three_phase["cp_per_tile"] * extra_tiles,
        },
    }


def three_phase_preferred(
    *,
    work_items: int,
    chunks: int,
    tiles: int,
    mode_varlen: bool,
    capability: tuple[int, int],
) -> bool:
    """Inside the chunk-parallel regime: the three-phase family for batched
    calls with at most ``_THREE_PHASE_MAX_TILES`` tiles or with at least
    ``_THREE_PHASE_MIN_ITEMS`` work items and ``_THREE_PHASE_MIN_CHUNKS``
    chunks per item; the look-back family otherwise.  On the
    ``_RUBIN_THREE_PHASE_CAPABILITIES`` the three-phase family also serves
    every batched call of at least ``_RUBIN_THREE_PHASE_MIN_CHUNKS`` chunks
    per item, except calls whose tiles lie in the ``_RUBIN_LOOKBACK_TILES``
    band and at most ``_RUBIN_LOOKBACK_MAX_ITEMS`` items with at least
    ``_RUBIN_LOOKBACK_MIN_CHUNKS`` chunks."""

    if mode_varlen:
        return False
    if int(tiles) <= _THREE_PHASE_MAX_TILES:
        return True
    if tuple(capability) in _RUBIN_THREE_PHASE_CAPABILITIES:
        low, high = _RUBIN_LOOKBACK_TILES
        if low < int(tiles) <= high:
            return False
        if (
            int(work_items) <= _RUBIN_LOOKBACK_MAX_ITEMS
            and int(chunks) >= _RUBIN_LOOKBACK_MIN_CHUNKS
        ):
            return False
        return int(chunks) >= _RUBIN_THREE_PHASE_MIN_CHUNKS
    return (
        int(work_items) >= _THREE_PHASE_MIN_ITEMS
        and int(chunks) >= _THREE_PHASE_MIN_CHUNKS
    )


def chunk_parallel_family(
    *,
    nheads: int,
    num_sequences: int,
    num_segments: int,
    nchunks: int,
    mode_varlen: bool,
    sm_count: int,
    capability: tuple[int, int],
    mode: str,
) -> str:
    """Family rule: ``"exact"``, ``"chunkpar"`` (look-back) or ``"chunkpar3"``
    (three-phase).  Chunk-parallel iff fewer work items than SMs, the S/H
    workspace fits the cap, and the faster of the two chunk-parallel
    predictions times the selection margin still beats the exact scan's; the
    chunk-parallel family then follows :func:`three_phase_preferred`.
    ``mode`` is the environment override: ``always`` forces the look-back,
    ``three_phase`` the three-phase program, ``never`` the exact scan,
    ``auto`` applies the rule."""

    if mode not in _CHUNK_PARALLEL_MODES:
        raise ValueError(
            f"chunk-parallel mode must be one of {_CHUNK_PARALLEL_MODE_TEXT}; got {mode!r}"
        )
    if mode == "always":
        return "chunkpar"
    if mode == "three_phase":
        return "chunkpar3"
    if mode == "never":
        return "exact"
    quantities = selection_quantities(
        nheads=nheads,
        num_sequences=num_sequences,
        num_segments=num_segments,
        nchunks=nchunks,
        mode_varlen=mode_varlen,
        sm_count=sm_count,
        capability=capability,
    )
    predicted = quantities["predicted_us"]
    chunk_parallel = (
        quantities["work_items"] < quantities["sm_count"]
        and quantities["workspace_bytes"]
        <= chunk_parallel_workspace_cap_bytes(capability)
        and min(predicted["chunk_parallel"], predicted["three_phase"])
        * _CHUNK_PARALLEL_SELECTION_MARGIN
        <= predicted["exact_scan"]
    )
    if not chunk_parallel:
        return "exact"
    if three_phase_preferred(
        work_items=quantities["work_items"],
        chunks=quantities["chunks"],
        tiles=quantities["tiles"],
        mode_varlen=quantities["mode_varlen"],
        capability=capability,
    ):
        return "chunkpar3"
    return "chunkpar"


def chunk_parallel_selected(
    *,
    nheads: int,
    num_sequences: int,
    num_segments: int,
    nchunks: int,
    mode_varlen: bool,
    sm_count: int,
    capability: tuple[int, int],
    mode: str,
) -> bool:
    """Whether :func:`chunk_parallel_family` picks either chunk-parallel
    family (the look-back or the three-phase program) for the call."""

    return (
        chunk_parallel_family(
            nheads=nheads,
            num_sequences=num_sequences,
            num_segments=num_segments,
            nchunks=nchunks,
            mode_varlen=mode_varlen,
            sm_count=sm_count,
            capability=capability,
            mode=mode,
        )
        != "exact"
    )


# Fused-program route, mirrored from ``fused_route`` of the Cake backend's
# exact-scan seed module
# (``flashinfer_blackwell_mamba_ssd_combined_exact_scan_seed_v1``)
# and read on the same host-known quantities: ``items = sequences x heads`` is
# the persistent work count; batched calls take the fused program when
# ``items <= 1024``, chunks per sequence ``<= 16`` and ``items x chunks <=
# 2048``; packed-varlen calls when ``items <= 256`` and the packed chunk count
# times ``nheads`` is ``<= 512``.  Every other call keeps the two-kernel
# program (``exactpre_*``): the fused kernel's in-CTA preprocess serialises
# with the scan and loses on saturated grids (93-row calibration, both archs:
# 50 routed rows win beyond the band, no routed row loses).
# The seq_idx triple form deriving its table inside the fused window, by
# compute capability (``_materialize_workspace``): the light metadata stage
# ahead of the fused main kernel (``exactmeta_*``) on SM100, and the
# one-launch ``exactone_*`` render on SM103 and SM107 -- the Rubin fused
# varlen render derives the table in-kernel like SM103's, and its host (a
# Grace CPU like GB300's) hides a lighter stage in the launch interval too.
_LIGHT_STAGE_CAPABILITIES = ((10, 0),)
_ONE_LAUNCH_CAPABILITIES = ((10, 3), (10, 7))
_FUSED_ROUTE_MAX_ITEMS_BATCHED = 1024
_FUSED_ROUTE_MAX_CHUNKS_BATCHED = 16
_FUSED_ROUTE_MAX_CHUNK_ITEMS_BATCHED = 2048
_FUSED_ROUTE_MAX_ITEMS_VARLEN = 256
_FUSED_ROUTE_MAX_CHUNK_ITEMS_VARLEN = 512
# Two-kernel program choice (host route; mirrors the seed
# module's ``two_kernel_route`` / ``TWO_KERNEL_ROUTE_MIN_CHUNKS``): the round-6
# ``exactpre_*`` render carries lever 1b (the per-item decay-factor table and
# the single-rounding FMA state update), whose fixed cost per work item is
# repaid only where the item spans several chunks; one-chunk items keep the
# retained start program ``exact_*``.
_TWO_KERNEL_ROUTE_MIN_CHUNKS = 2
# sm_107a (R200, 212 SMs) only: one-chunk items take ``exactpre`` too where
# the call has more items than SMs and at most this many (sequence, head)
# items and runs the ring schedule (the ``noring`` start schedule exists for
# the retained program only).  Forced routes on R200:
# 512-2048 items 1.015-1.06x faster than ``exact``; 8192-20480 items
# 0.99-1.02x (kept).
_TWO_KERNEL_ONE_CHUNK_MAX_ITEMS = {(10, 7): 2048}
# sm_107a: one-chunk ring calls with these state dtypes take ``exactpre``
# above the item bound too (fp32 state, 8192-20480 items: 1.016-1.031x
# faster than ``exact``).
_ONE_CHUNK_EXACTPRE_STATE_DTYPES = {(10, 7): frozenset({torch.float32})}
# sm_107a: a call the retained ``exact_*_noring`` program would serve takes
# the fused one-kernel program instead (bf16 state, 8 x 64: 1.34x).
_NORING_RETAINED_FUSED = frozenset({(10, 7)})
# sm_107a: a packed-varlen call outside the fused window that the start
# schedule (``noring``) would serve runs the ring schedule instead where it
# has more than this many (sequence, head) items per SM and its ring
# two-kernel family is ``exactpre`` (one-chunk items up to
# ``_TWO_KERNEL_ONE_CHUNK_MAX_ITEMS``): there the ring program is faster than
# the fused start schedule in every call form, outputs bitwise identical.
# With fewer items per SM the two are close and the cu_seqlens form loses;
# above ``_TWO_KERNEL_ONE_CHUNK_MAX_ITEMS`` items the ring family is the
# retained ``exact`` program, which is slower, so those calls keep the fused
# start schedule.
_NORING_RING_EXACTPRE_MIN_ITEMS_PER_SM = {(10, 7): 4}
# sm_107a: one-chunk varlen ring calls with more than this many (sequence,
# head) items, at least ``_CLAIM_MIN_ITEMS_PER_CTA`` items per CTA of the
# persistent grid and a state dtype listed in ``_CLAIM_STATE_DTYPES`` take the
# claim program (``exactpre_<state>_varlen_claim``, :func:`claim_selected`):
# the two-kernel ring program drawing each CTA's next item at run time from a
# self-resetting counter (``preprocess_status[1]``) instead of the static
# grid-stride order, so the CTAs on faster SMs take more items; the bf16
# program updates the state as the retained start program does (bitwise its
# outputs), the f32 program as ``exactpre`` does (bitwise its outputs).  On
# R200, whose SMs do not run at one speed, both are faster than the programs
# those calls ran before.
_CLAIM_MIN_ITEMS = {(10, 7): 2048}
_CLAIM_MIN_ITEMS_PER_CTA = 8
_CLAIM_STATE_DTYPES = {(10, 7): frozenset({torch.bfloat16, torch.float32})}
# sm_107a: packed-varlen two-kernel ring calls of many short items take the
# ``_ibulk`` program (``exactpre_*_varlen_ibulk``, bf16 / f16 state), whose
# pre_inter warps read each work item's initial state as one 16 KB bulk copy
# into a state stage (the CTA's first item at role start, every later item
# while the previous item's last chunk runs) instead of 64 scalar global
# loads per thread; that per-item load sits on pre_inter, the binding role of
# short items.  The final state keeps the scalar store.  Taken with at least
# ``_IBULK_MIN_ITEMS_PER_SM`` items per SM (a CTA runs consecutive items) of
# at most ``_IBULK_MAX_CHUNKS_PER_ITEM`` host-known chunks; a call with z
# only with one host-known chunk per item and at least
# ``_IBULK_Z_MIN_ITEMS_PER_SM`` items per SM, since with fewer items or
# longer items its epilogue binds and the bulk load gains nothing.  Same
# outputs bit for bit as the ring program.  Batched calls keep the ring
# program (no gain measured).
_IBULK_CAPABILITIES = frozenset({(10, 7)})
_IBULK_STATE_DTYPES = frozenset({torch.bfloat16, torch.float16})
_IBULK_MIN_ITEMS_PER_SM = 2
_IBULK_MAX_CHUNKS_PER_ITEM = 8
_IBULK_Z_MIN_ITEMS_PER_SM = 8


def fused_selected(
    *,
    nheads: int,
    num_sequences: int,
    nchunks: int,
    mode_varlen: bool,
    total_tokens: int,
) -> bool:
    """Exact-scan shape window: the fused program (``exactfused_*``) for
    one-wave / short-item calls, the two-kernel form otherwise (``exactpre_*``
    or the retained start ``exact_*``, see :func:`two_kernel_family`).
    ``nchunks`` is the chunk count per sequence (batched); ``total_tokens`` the
    packed token count (varlen; the per-sequence lengths live on the device
    and are not read).  The runner applies the window to every call form:
    batched calls, the ``cu_seqlens`` form and the ``seq_idx`` triple form
    with a caller ``seq_chunk_cumsum`` the preprocess does not rewrite take
    ``exactfused_*`` (preprocess grid 0, one launch); the triple form
    deriving the table takes ``exactmeta_*`` on SM100 (the light metadata
    kernel then the same fused main kernel, two launches) and ``exactone_*``
    on SM103 and SM107 (the fused main kernel derives the table in its CTA
    prologue, one launch) unless the table is published into the caller's
    buffer, which keeps the two-kernel program on SM103 and SM107
    (``_materialize_workspace`` decides per compute capability)."""

    items = int(num_sequences) * int(nheads)
    if not mode_varlen:
        return (
            items <= _FUSED_ROUTE_MAX_ITEMS_BATCHED
            and int(nchunks) <= _FUSED_ROUTE_MAX_CHUNKS_BATCHED
            and items * int(nchunks) <= _FUSED_ROUTE_MAX_CHUNK_ITEMS_BATCHED
        )
    packed_chunks = -(-int(total_tokens) // _CHUNK_SIZE)
    return (
        items <= _FUSED_ROUTE_MAX_ITEMS_VARLEN
        and packed_chunks * int(nheads) <= _FUSED_ROUTE_MAX_CHUNK_ITEMS_VARLEN
    )


def host_chunks_per_item(*, mode_varlen: bool, num_sequences: int, seqlen: int) -> int:
    """Host-known chunks per (sequence, head) work item: batched calls
    ``ceil(seqlen / 128)`` -- every item spans the same ``seqlen`` tokens;
    varlen calls ``ceil(ceil(seqlen / num_sequences) / 128)`` with ``seqlen``
    the packed token count -- the mean sequence length is the host-known
    quantity (the per-sequence lengths live on the device).  A
    ``num_sequences`` of 0 counts as 1."""

    if not mode_varlen:
        return -(-int(seqlen) // _CHUNK_SIZE)
    mean_tokens = -(-int(seqlen) // max(int(num_sequences), 1))
    return -(-mean_tokens // _CHUNK_SIZE)


def two_kernel_family(
    *,
    mode_varlen: bool,
    num_sequences: int,
    seqlen: int,
    nheads: Optional[int] = None,
    sm_count: Optional[int] = None,
    capability: Optional[tuple[int, int]] = None,
    variant: str = "",
    state_dtype: Optional[torch.dtype] = None,
) -> str:
    """Two-kernel program family of a call the chunk-parallel and fused routes
    left on the two-kernel form: ``"exactpre"`` (the round-6 render) where the
    host-known chunks per work item reach ``_TWO_KERNEL_ROUTE_MIN_CHUNKS``,
    else ``"exact"`` (the retained start program).

    Chunks per item: batched calls ``ceil(seqlen / 128)`` -- every item of a
    batched call spans the same ``seqlen`` tokens; varlen calls
    ``ceil(ceil(seqlen / num_sequences) / 128)`` with ``seqlen`` the packed
    token count -- the mean sequence length is the host-known quantity (the
    per-sequence lengths live in ``cu_seqlens`` / ``seq_idx`` on the device
    and are not read), and the mean chunk count per item is what the table's
    fixed cost is amortized over.  A ``num_sequences`` of 0 counts as 1.
    On a ``capability`` of ``_TWO_KERNEL_ONE_CHUNK_MAX_ITEMS`` one-chunk
    items take ``exactpre`` as well when the call's ``num_sequences x
    nheads`` items exceed ``sm_count`` and are at most that bound and its
    scan ``variant`` is the ring schedule (``""``); on a ``capability`` of
    ``_ONE_CHUNK_EXACTPRE_STATE_DTYPES`` a ``state_dtype`` it lists takes
    ``exactpre`` above that bound too.
    """

    chunks = host_chunks_per_item(
        mode_varlen=mode_varlen, num_sequences=num_sequences, seqlen=seqlen
    )
    if chunks >= _TWO_KERNEL_ROUTE_MIN_CHUNKS:
        return "exactpre"
    max_items = (
        None
        if capability is None or nheads is None or sm_count is None
        else _TWO_KERNEL_ONE_CHUNK_MAX_ITEMS.get(
            (int(capability[0]), int(capability[1]))
        )
    )
    items = max(int(num_sequences), 1) * int(nheads or 0)
    if max_items is not None and not variant and sm_count < items <= max_items:
        return "exactpre"
    if (
        max_items is not None
        and not variant
        and sm_count < items
        and state_dtype
        in _ONE_CHUNK_EXACTPRE_STATE_DTYPES.get(
            (int(capability[0]), int(capability[1])), ()
        )
    ):
        return "exactpre"
    return "exact"


# sm_107a (R200, 212 SMs) only: calls without ``z`` whose fused program runs
# without a stage ahead of it -- batched calls, the cu_seqlens form and the
# seq_idx triple form with a caller ``seq_chunk_cumsum`` -- take the fused
# program beyond the seed window (:func:`fused_selected`) up to these
# (sequence, head) item bounds, where the fused kernel's in-CTA dt work costs
# less than the segment preprocess launch it replaces: batched one-chunk
# calls up to the seed's 2048 chunk-items, packed-varlen calls up to 512
# items within the seed's 512 packed chunk-items (sequences of at most one
# chunk); outputs are bitwise those of the two-kernel ``exactpre`` program.
# Past these bounds the fused program stops winning.  With ``z`` it loses on
# most shapes, so calls with ``z`` keep the seed window.
_FUSED_NO_Z_MAX_ITEMS = {(10, 7): (2048, 512)}  # (batched, varlen)


def fused_no_z_extended(
    *,
    capability,
    nheads: int,
    num_sequences: int,
    nchunks: int,
    mode_varlen: bool,
    total_tokens: int,
) -> bool:
    """The fused-window extension of ``capability`` for calls without ``z``
    (``_FUSED_NO_Z_MAX_ITEMS``; ``False`` on capabilities it does not list):
    batched one-chunk calls (``nchunks`` 1) with at most its batched item
    bound and ``_FUSED_ROUTE_MAX_CHUNK_ITEMS_BATCHED`` chunk-items;
    packed-varlen calls with at most its varlen item bound and
    ``_FUSED_ROUTE_MAX_CHUNK_ITEMS_VARLEN`` packed chunk-items.  The runner
    applies it only where the fused program needs no stage ahead of it and
    the call has no ``z``."""

    bounds = _FUSED_NO_Z_MAX_ITEMS.get((int(capability[0]), int(capability[1])))
    if bounds is None:
        return False
    max_items_batched, max_items_varlen = bounds
    items = max(int(num_sequences), 1) * int(nheads)
    if not mode_varlen:
        return (
            int(nchunks) == 1
            and items <= max_items_batched
            and items <= _FUSED_ROUTE_MAX_CHUNK_ITEMS_BATCHED
        )
    packed_chunks = -(-int(total_tokens) // _CHUNK_SIZE)
    return (
        items <= max_items_varlen
        and packed_chunks * int(nheads) <= _FUSED_ROUTE_MAX_CHUNK_ITEMS_VARLEN
    )


def claim_selected(
    *,
    mode_varlen: bool,
    num_sequences: int,
    seqlen: int,
    nheads: int,
    sm_count: int,
    capability: Optional[tuple[int, int]],
    variant: str,
    state_dtype: torch.dtype,
) -> bool:
    """Whether a two-kernel call runs the claim program
    (``exactpre_<state>_varlen_claim``): on a ``capability`` of
    ``_CLAIM_MIN_ITEMS``, a packed-varlen call of the ring schedule
    (``variant`` ``""``) whose host-known chunks per item
    (:func:`host_chunks_per_item`, ``seqlen`` the packed token count) is one,
    with more than that many ``num_sequences x nheads`` items, at least
    ``_CLAIM_MIN_ITEMS_PER_CTA`` items per CTA of its persistent grid
    (:func:`_persistent_grid_size`) and a ``state_dtype`` of
    ``_CLAIM_STATE_DTYPES``."""

    if capability is None or not mode_varlen or variant:
        return False
    key = (int(capability[0]), int(capability[1]))
    min_items = _CLAIM_MIN_ITEMS.get(key)
    if min_items is None or state_dtype not in _CLAIM_STATE_DTYPES.get(key, ()):
        return False
    chunks = host_chunks_per_item(
        mode_varlen=True, num_sequences=num_sequences, seqlen=seqlen
    )
    items = max(int(num_sequences), 1) * int(nheads)
    grid = _persistent_grid_size(total_work=items, sm_count=int(sm_count))
    return (
        chunks == 1 and items > min_items and items >= _CLAIM_MIN_ITEMS_PER_CTA * grid
    )


def ibulk_selected(
    *,
    capability: tuple[int, int],
    state_dtype: torch.dtype,
    mode_varlen: bool,
    total_work: int,
    chunks_per_item: int,
    sm_count: int,
    has_z: bool,
) -> bool:
    """Whether a two-kernel ring call (``exactpre``, variant ``""``) takes the
    ``_ibulk`` program: a packed-varlen call on a ``_IBULK_CAPABILITIES``
    device with a ``_IBULK_STATE_DTYPES`` state and at least
    ``_IBULK_MIN_ITEMS_PER_SM`` x ``sm_count`` (sequence, head) items of at
    most ``_IBULK_MAX_CHUNKS_PER_ITEM`` host-known chunks
    (:func:`host_chunks_per_item`); a call with z only with one chunk per item
    and at least ``_IBULK_Z_MIN_ITEMS_PER_SM`` x ``sm_count`` items."""

    if (int(capability[0]), int(capability[1])) not in _IBULK_CAPABILITIES:
        return False
    if not mode_varlen or state_dtype not in _IBULK_STATE_DTYPES:
        return False
    if has_z and (
        chunks_per_item > 1 or total_work < _IBULK_Z_MIN_ITEMS_PER_SM * sm_count
    ):
        return False
    return (
        total_work >= _IBULK_MIN_ITEMS_PER_SM * sm_count
        and chunks_per_item <= _IBULK_MAX_CHUNKS_PER_ITEM
    )


# sm_107a: packed-varlen fused ring calls (``exactfused_*`` varlen: the
# cu_seqlens form and the seq_idx triple form with a caller
# ``seq_chunk_cumsum``) with initial states, items of one host-known chunk
# and more (sequence, head) items than SMs take the fused ``_ibulk`` program
# (``exactfused_*_varlen_ibulk``, bf16 / f16 state): its pre_inter warps
# read each work item's initial state as one 16 KB bulk copy through the
# state stage -- the CTA's first item's at role start, every later item's
# while the previous item's last chunk runs -- instead of 64 scalar global
# loads per thread.  Same outputs bit for bit as the ring program.  Calls
# with at most one item per SM (no CTA runs a second item), items of several
# chunks and calls without initial states keep the ring program (no gain
# measured).
_FUSED_IBULK_CAPABILITIES = frozenset({(10, 7)})
_FUSED_IBULK_MAX_CHUNKS_PER_ITEM = 1


def fused_ibulk_selected(
    *,
    capability: tuple[int, int],
    state_dtype: torch.dtype,
    has_initial_states: bool,
    total_work: int,
    chunks_per_item: int,
    sm_count: int,
) -> bool:
    """Whether a packed-varlen fused ring call (``exactfused``, variant
    ``""``) takes the fused ``_ibulk`` program: on a
    ``_FUSED_IBULK_CAPABILITIES`` device, with a ``_IBULK_STATE_DTYPES``
    state, initial states, more than ``sm_count`` (sequence, head) items and
    at most ``_FUSED_IBULK_MAX_CHUNKS_PER_ITEM`` host-known chunks per item
    (:func:`host_chunks_per_item`)."""

    if (int(capability[0]), int(capability[1])) not in _FUSED_IBULK_CAPABILITIES:
        return False
    if not has_initial_states or state_dtype not in _IBULK_STATE_DTYPES:
        return False
    return total_work > sm_count and chunks_per_item <= _FUSED_IBULK_MAX_CHUNKS_PER_ITEM


# sm_107a: the one-launch ring programs (``exactone_<state>_varlen``) in
# two further prologue forms, each a program of its own chosen by the
# launch grid (:func:`_persistent_grid_size`).  Below
# ``_EXACTONE_TABLE_ROLES_MIN_GRID`` CTAs ``tablefirst``: the CTA derives
# its private sequence table before the barrier setup, so the derivation's
# two dependent table loads overlap the mbarrier init and the TMEM alloc.
# From it on ``tableroles``: each role passes the table barrier right
# before its work loop instead of in the prologue, so its entry code runs
# while the derivation's loads are in flight.  Integer table plumbing only:
# the outputs are bitwise those of the ``exactone`` program.  The
# ``noring`` calls keep their program.
_EXACTONE_TABLE_ROLES_MIN_GRID = {(10, 7): 128}
_EXACTONE_TABLE_STATE_DTYPES = frozenset({torch.bfloat16, torch.float16, torch.float32})


def exactone_table_variant(
    *,
    capability: Optional[tuple[int, int]],
    variant: str,
    state_dtype: torch.dtype,
    grid: int,
) -> str:
    """The variant of a one-launch (``exactone``) call: on a capability of
    ``_EXACTONE_TABLE_ROLES_MIN_GRID``, a ring call (``variant`` ``""``)
    of a ``_EXACTONE_TABLE_STATE_DTYPES`` state takes ``"tablefirst"`` at a
    launch ``grid`` below that many CTAs and ``"tableroles"`` from it on;
    every other call keeps ``variant``."""

    if capability is None or variant or state_dtype not in _EXACTONE_TABLE_STATE_DTYPES:
        return variant
    roles_min = _EXACTONE_TABLE_ROLES_MIN_GRID.get(
        (int(capability[0]), int(capability[1]))
    )
    if roles_min is None:
        return variant
    return "tablefirst" if grid < roles_min else "tableroles"


# Tail-wave pair program (``_tail``): the two-kernel exact-scan kernel launched
# as 2-CTA clusters on an even grid; a pair with an odd item count splits its
# first item at the middle chunk and hands the scan state to its partner, so
# the half wave a balanced persistent grid leaves idle is filled.  The cluster
# launch and the pair plan cost 2-5 us per launch and the hand-off about one
# chunk per split item, so the pair program is taken only where the chunks per
# item amortize both (measured floor: 16 chunks) and the pair makespan in
# items beats the persistent program's.  Mirrors the Cake kernel-tests
# ``tail_route`` for the two-kernel family; its fused clause (32 chunks, SM103
# only) lies outside the fused window (:func:`fused_selected`), so the fused
# family ships no pair program and the fused route is judged first.
_TAIL_MIN_CHUNKS = 16


@dataclass(frozen=True)
class _TailFullGridRule:
    """Admission of a two-kernel call whose balanced persistent grid is the
    full device (``_persistent_grid_size`` returned ``sm_count``: on 212 SMs
    every item count above 212 except multiples the balancing rule accepts,
    e.g. 256, 512, 640) to the ``_tail`` pair program.  The persistent program
    then runs ``ceil(items / sm_count)`` item waves, the last one ragged; the
    pair program runs ``ceil(items / pairs) / 2`` half-item rounds and saves
    the difference.  Measured on R200 (CAKE-2549 round 9, lease 799171: the
    item ladder 128..256 and the (items x chunks x form) grid, main-kernel
    medians, cold L2): the pair render costs more per item than the ring
    render -- ``render_penalty`` is that measured ratio minus one (batched
    ~1.5-5 us per launch, up to 1/12 of an item at 16 chunks; varlen 4-11 us
    per item, up to 1/5) -- and the split item of an odd pair hands its scan
    state over at the cost of about one chunk step.  The route is taken only
    where the saved waves exceed both: 256 items (2 waves vs 1.5 rounds) win
    on every form (1.04-1.26x), 512 and 640 items win on the batched form
    only (1.06-1.13x; the varlen pair render loses 0.90-0.99x there), and
    equal makespans (384, 768, 1024, 2048 items) lose by the overhead.  The
    varlen chunk floor is 8: 256-item varlen calls of 8 chunks win 1.03-1.08x
    (perf_varlen_32x8, nem_v1024x2) and no varlen call above one item per SM
    with 8 chunks lies inside the fused window; the batched floor stays 16
    (an 8-chunk batched call up to 256 items is a fused-window call).

    The varlen pair render's cost is per item, so varlen calls without ``z``
    whose items span at least ``long_chunks_varlen`` chunks spread it over
    more chunk steps and take the smaller ``render_penalty_varlen_long``:
    with the current varlen pair render those calls run faster on the pair
    program in every call form, head count and state dtype measured, outputs
    bitwise identical.  8-chunk items keep the larger penalty, and so do
    calls with ``z``, whose pair epilogue gains little or nothing there.  The
    rule reads the mean chunks per item: random sequence lengths win as much
    as equal ones, but lengths that alternate long and short by sequence load
    whole pairs unevenly and run slower on the pair program."""

    min_chunks_batched: int
    min_chunks_varlen: int
    render_penalty_batched: float
    render_penalty_varlen: float
    long_chunks_varlen: int
    render_penalty_varlen_long: float

    def min_chunks(self, mode_varlen: bool) -> int:
        return self.min_chunks_varlen if mode_varlen else self.min_chunks_batched

    def render_penalty(
        self, mode_varlen: bool, chunks_per_item: int = 0, has_z: bool = False
    ) -> float:
        """The render penalty of the call form; a varlen call without ``z``
        of at least ``long_chunks_varlen`` chunks per item takes the long-item
        penalty."""

        if not mode_varlen:
            return self.render_penalty_batched
        if not has_z and chunks_per_item >= self.long_chunks_varlen:
            return self.render_penalty_varlen_long
        return self.render_penalty_varlen


# sm_107a (R200, 212 SMs) only; the Blackwell rule keeps the balanced-grid
# condition (no entry: a full grid never takes the pair program).
_TAIL_FULL_GRID_RULES: dict[tuple[int, int], _TailFullGridRule] = {
    (10, 7): _TailFullGridRule(
        min_chunks_batched=16,
        min_chunks_varlen=8,
        render_penalty_batched=1 / 12,
        render_penalty_varlen=1 / 5,
        long_chunks_varlen=16,
        render_penalty_varlen_long=1 / 8,
    ),
}


def tail_full_grid_rule(capability) -> Optional[_TailFullGridRule]:
    """The full-grid admission rule of ``capability`` (``None``: none)."""

    return _TAIL_FULL_GRID_RULES.get((int(capability[0]), int(capability[1])))


def _full_grid_tail_selected(
    *,
    rule: _TailFullGridRule,
    total_work: int,
    chunks_per_item: int,
    sm_count: int,
    mode_varlen: bool,
    has_z: bool = False,
) -> bool:
    """:class:`_TailFullGridRule` on a call whose balanced grid is the full
    device: the pair makespan (half-item rounds) grown by the measured render
    penalty, plus one chunk step of hand-off when the pair with the most items
    splits one, must stay below the persistent program's item waves."""

    if chunks_per_item < rule.min_chunks(mode_varlen):
        return False
    pairs = sm_count // 2
    pair_rounds = -(-total_work // pairs)
    persistent_makespan = -(-total_work // sm_count)
    pair_makespan = pair_rounds / 2
    hand_off = (pair_rounds % 2) / chunks_per_item
    return (
        pair_makespan * (1.0 + rule.render_penalty(mode_varlen, chunks_per_item, has_z))
        + hand_off
        < persistent_makespan
    )


def _tail_grid_size(*, sm_count: int) -> int:
    """Grid of a ``_tail`` program (one 2-CTA cluster per SM pair) and of its
    one-CTA form ``_tailone`` (CTAs ``2 p`` and ``2 p + 1`` pair ``p``)."""

    return 2 * (sm_count // 2)


def tail_selected(
    *,
    total_work: int,
    chunks_per_item: int,
    sm_count: int,
    mode_varlen: bool = False,
    full_grid_rule: Optional[_TailFullGridRule] = None,
    has_z: bool = False,
) -> bool:
    """Whether a two-kernel exact-scan call of ``total_work`` (sequence, head)
    items with ``chunks_per_item`` host-known chunks per item
    (:func:`host_chunks_per_item`) runs the ``_tail`` pair program on a device
    of ``sm_count`` SMs: a tail wave exists (the balanced persistent grid is
    below one CTA per SM), the chunk floor is met and ``ceil(items / pairs) /
    2`` (half items count) is below the persistent program's ``ceil(items /
    grid)``.  A call whose balanced grid is the full device takes the pair
    program only under ``full_grid_rule`` (:class:`_TailFullGridRule`, the
    sm_107a admission, which also reads whether the call has ``z``; ``None``
    -- the Blackwell default -- never)."""

    pairs = sm_count // 2
    if pairs < 1 or total_work <= sm_count:
        return False
    balanced_grid = _persistent_grid_size(total_work=total_work, sm_count=sm_count)
    if balanced_grid >= sm_count:
        return full_grid_rule is not None and _full_grid_tail_selected(
            rule=full_grid_rule,
            total_work=total_work,
            chunks_per_item=chunks_per_item,
            sm_count=sm_count,
            mode_varlen=mode_varlen,
            has_z=has_z,
        )
    if chunks_per_item < _TAIL_MIN_CHUNKS:
        return False
    persistent_makespan = -(-total_work // balanced_grid)
    pair_makespan = -(-total_work // pairs) / 2
    return pair_makespan < persistent_makespan


# The one-CTA form of the varlen pair program (``_tailone``, sm_107a only):
# the same pair plan, scan and hand-off over global memory as ``_tail``,
# launched on the same even grid without the 2-CTA cluster.  CTAs ``2 p`` and
# ``2 p + 1`` form pair ``p``; each draws its rank from the pair's ticket word
# in ``preprocess_status`` (the first to draw runs the split item's head), and
# the receiver of a split item waits on the pair's flag bit, which the head's
# CTA sets once the parked state is in global memory.  The receiver only waits
# on a partner that has already drawn its ticket (so it is running), and the
# head's CTA waits on no other CTA: no co-residency is assumed.  Every complete
# launch leaves the ticket word and the flag bit 0.  Measured on R200: faster
# than the 2-CTA cluster launch on the 256-item varlen calls of 8 chunks
# (perf_varlen_32x8 without z, nem_v1024x2 with z), slower on varlen calls of
# 32 chunks, which keep the cluster program.
_TAIL_ONE_CTA_CAPABILITIES = frozenset({(10, 7)})


def tail_one_cta_selected(
    *,
    capability: Optional[Tuple[int, int]],
    mode_varlen: bool,
    chunks_per_item: int,
) -> bool:
    """Whether a call :func:`tail_selected` sends to the ``_tail`` pair
    program runs its one-CTA form ``_tailone`` instead: on SM107, packed
    varlen, with or without ``z``, items of fewer host-known chunks
    (:func:`host_chunks_per_item`) than the full-grid admission's
    ``long_chunks_varlen`` (:class:`_TailFullGridRule`).  Outputs are bitwise
    those of the pair program."""

    if capability is None:
        return False
    key = (int(capability[0]), int(capability[1]))
    rule = _TAIL_FULL_GRID_RULES.get(key)
    return (
        key in _TAIL_ONE_CTA_CAPABILITIES
        and rule is not None
        and mode_varlen
        and chunks_per_item < rule.long_chunks_varlen
    )


def _sequence_arguments(
    main_args: tuple[str, ...],
    preprocess: Mapping[str, object],
    preprocess_grid: tuple[int, int, int],
    main: Mapping[str, object],
    main_grid: tuple[int, int, int],
    cuda_stream: int,
    preprocess_args: tuple[str, ...] = _PREPROCESS_ARGS,
) -> tuple[object, ...]:
    """Order the name-keyed stage values into the launcher's positional ABI;
    ``main_args`` is the program's main-kernel argument order and
    ``preprocess_args`` its preprocess stage's (``_METADATA_ARGS`` for the
    ``exactmeta`` programs)."""

    return (
        *(preprocess[name] for name in preprocess_args),
        *preprocess_grid,
        *(main[name] for name in main_args),
        *main_grid,
        cuda_stream,
    )


# Programs launched in this process, by (name, arch).  Loading or building a
# program is host work a CUDA graph cannot record, and the host shim resolves
# its kernel handles on the first launch, so a captured call requires one
# eager launch of its program first.
_LOADED_PROGRAMS: set[tuple[str, str]] = set()


def _launch_program(
    name: str,
    arch: str,
    *,
    preprocess: Mapping[str, object],
    preprocess_grid: tuple[int, int, int],
    main: Mapping[str, object],
    main_grid: tuple[int, int, int],
    cuda_stream: int,
) -> None:
    """Launch one program's preprocess and main kernel on the explicit stream."""

    _load_generated_program(name, arch).run(
        *_sequence_arguments(
            _PROGRAMS[name].main_args,
            preprocess,
            preprocess_grid,
            main,
            main_grid,
            cuda_stream,
            preprocess_args=_PROGRAMS[name].preprocess_args,
        )
    )
    _LOADED_PROGRAMS.add((name, arch))


def _source_dir() -> Path:
    packaged = jit_env.FLASHINFER_CSRC_DIR / "cake_mamba_ssd_combined"
    if packaged.is_dir():
        return packaged
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "cake_mamba_ssd_combined"
    return checkout


@functools.cache
def _target_arch(device_index: int) -> str:
    capability = torch.cuda.get_device_capability(device_index)
    arch = _TARGET_ARCHS.get(tuple(capability))
    if arch is None:
        raise ValueError(
            "Cake SSDCombined requires SM100, SM103 or SM107, got "
            f"SM{capability[0]}{capability[1]}"
        )
    return arch


@functools.cache
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _cuda_device_index(tensor: torch.Tensor) -> int:
    device = tensor.device
    if device.type != "cuda" or device.index is None:
        raise ValueError("Cake SSDCombined inputs must be on a CUDA device")
    return device.index


def _nvcc() -> Path:
    candidate = shutil.which("nvcc")
    if candidate is None:
        cuda_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_root:
            path = Path(cuda_root) / "bin" / "nvcc"
            if path.is_file():
                candidate = str(path)
    if candidate is None:
        raise RuntimeError("nvcc is required to build the Cake SSDCombined backend")
    return Path(candidate).resolve()


def _render_host_source(
    template: str, name: str, program: _Program, *, main_pdl: bool = False
) -> str:
    """Substitute one program's table values into the shared launcher.

    The program name becomes part of the launcher namespace so the inline
    launch helpers (and their function-local static kernel handles) of one
    loaded program library are never unified with another's.  A template
    with the ``CAKE_SSD_MAIN_PDL_LAUNCH`` statement ahead of its stream-ordered
    main launch gets, with ``main_pdl`` (:func:`_main_pdl`), the same launch
    through the extended launch API with the programmatic stream
    serialization attribute (and returns), else a comment.
    """

    values = {
        "CAKE_SSD_PROGRAM": name,
        "CAKE_SSD_PREPROCESS_MODULE": program.preprocess.module,
        "CAKE_SSD_PREPROCESS_KERNEL": program.preprocess.kernel,
        "CAKE_SSD_PREPROCESS_THREADS": str(program.preprocess.threads),
        "CAKE_SSD_MAIN_MODULE": program.main.module,
        "CAKE_SSD_MAIN_KERNEL": program.main.kernel,
        "CAKE_SSD_STATE_DTYPE_CODE": str(program.state_dtype_code),
        "CAKE_SSD_STATE_DTYPE_BITS": str(program.state_dtype_bits),
        "CAKE_SSD_MAIN_SMEM_BYTES": str(program.main_smem_bytes),
    }
    if main_pdl or "CAKE_SSD_MAIN_PDL_LAUNCH" in template:
        values["CAKE_SSD_MAIN_PDL_LAUNCH"] = (
            "{\n"
            "    // Programmatic dependent launch after the preprocess stage: the main\n"
            "    // kernel's griddepcontrol.wait orders its accesses to the preprocess outputs.\n"
            "    tvm::ffi::cuda_api::LaunchConfig config{};\n"
            "    tvm::ffi::cuda_api::LaunchAttrType attr{};\n"
            "    tvm::ffi::cuda_api::ConstructLaunchConfig(prepared.kernel->GetHandle(), "
            f"stream, {program.main_smem_bytes}u, prepared.grid, prepared.block, 1, "
            "config, attr);\n"
            "    attr.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;\n"
            "    attr.value.programmaticStreamSerializationAllowed = 1;\n"
            "    config.attrs = &attr;\n"
            "    config.numAttrs = 1;\n"
            "    TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(\n"
            "        prepared.kernel->LaunchEx(prepared.kargs, config));\n"
            "    return;\n"
            "  }"
            if main_pdl
            else "// Stream-ordered main launch."
        )
    source = template
    for placeholder in sorted(values, key=len, reverse=True):
        if placeholder not in source:
            raise RuntimeError(
                f"Cake SSDCombined host template lacks placeholder {placeholder}"
            )
        source = source.replace(placeholder, values[placeholder])
    return source


# Host selection between the two exact-scan schedule variants (CAKE-956 round 5,
# route lane).  Both variants compute bitwise-identical outputs; the ring
# program (``exact_*``) is faster on every calibrated geometry except the ones
# below, where its epilogue z refill queues behind the partial-segment scalar
# stores of one-chunk work items.  Calibration = scan-kernel time ring / start
# (CUPTI, cold L2, 3 rounds, arms alternating in one step) on NVIDIA B200
# (nsc-svg-slurm-1-gpu-104) and GB300 (nvl72d423-T07), Nemotron-H geometry
# nheads 128 / ngroups 8, bf16 state, D per head, initial states; steps
# a64287477fdbbd7e1393d6c2 / aa3374884f3fb4da09ecd0d8 (84 cells) and
# 54d7ca23fc6a62e86d35f518 / ec121ae14d1b35e5ba452d77 (28 boundary cells).
# Kernel band 0.0035.  GB300 / B200 ring / start:
#
#   varlen, z, nseq x tokens      16      32      48      56      64      72      96     128
#     nseq 1                   0.940   0.933      -       -   0.937      -   0.959   0.958   (B200 0.933-0.969)
#     nseq 2                   0.974   0.984      -       -   0.999      -   0.978   0.976   (B200 0.959-0.998)
#     nseq 3 / 4 (64 tok)          -       -      -       -  0.986 / 1.000-1.003             (B200 0.990 / 0.999-1.002)
#     nseq 6 (64 tok)              -       -      -       -   1.009      -       -       -   (B200 1.006)
#     nseq 8                   0.984   0.988  0.960   0.963   1.019   0.959   0.991   0.915   (B200 0.978 0.985 0.952 0.952 1.013 0.957 0.967 0.917)
#     nseq 16                  0.960   0.974      -       -   1.009      -   0.979   0.917   (B200 0.963 0.977 1.011 0.965 0.918)
#     nseq 32                  0.907   0.965      -       -   1.003      -   0.986   0.911   (B200 0.925 0.971 1.006 0.973 0.913)
#     nseq 64                  0.862   0.914  0.958   0.949   1.008   0.969   0.988   0.901   (B200 0.882 0.938 0.956 0.949 1.000 0.960 0.977 0.906)
#     nseq 128                 0.836   0.860      -       -   1.004      -   0.990   0.899   (B200 0.867 0.884 1.002 0.976 0.903)
#   80 / 100 / 112 tokens (nseq 8 / 64): 0.949-0.988 on both archs; 127 tokens: 0.970-0.985 on both;
#   60 / 63 / 65 / 68 tokens (nseq 8 / 64): 0.950-0.979 on both archs (steps 0043bcdea2f8b84c94a470ad /
#   426afd144b4818c5731b89af); at 64 tokens nseq 5 1.001 / 0.994, nseq 7 1.000 / 0.999, nseq 12 0.999 / 1.005.
#   Mixtures whose consecutive pairs sum to 128 tokens (two partial segments per chunk, tokens per sequence
#   = 64 on average): [32, 96] x 4 1.006 / 1.005, [48, 80] x 4 1.007 / 1.003, [63, 65] x 4 1.008 / 1.007.
#   varlen, no z, 64 tokens: nseq 8 0.992 / 0.990, nseq 64 0.987 / 0.980 (ring wins; the loss needs z).
#   batched, z, initial states, batch x tokens 50 / 100 / 127 / 128: batch 1 0.965-0.979 (B200 0.948-0.959),
#     batch 2 0.983-1.002 (B200 0.979-0.987), batch 8 0.929-0.949 (B200 0.939-0.949), batch 64 0.898-0.921 (B200 0.908-0.920);
#     batch 2 x 25 / 75 tokens 0.988 / 0.993 (B200 0.975 / 0.977); batch 4 x 50 0.960 (B200 0.948).
#   batched, no z, initial states: batch 64 0.898-0.920 (B200 0.908-0.921); batch 4 x 50 0.973 (B200 0.960);
#     batch 2 x 25 0.990 (B200 0.991); batch 2 x 100 / 127 / 128 0.995 / 1.000 / 0.990 (B200 0.991 / 0.995 / 0.995);
#     batch 2 x 50 0.995-1.008 over six readings per arch (B200 1.000-1.005); batch 2 x 60 1.000 (B200 1.000);
#     batch 2 x 75 1.005 / 0.995 (B200 0.993 / 0.998); batch 2 x 90 0.987 (B200 0.995);
#     batch 1 x 25 / 50 / 75 / 100 0.963-0.975 (B200 0.955-0.959); batch 3 x 50 / 75 0.992 / 0.985 (B200 0.986 / 0.981);
#     batch 4 x 75 0.966 (B200 0.961); batch 8 x 50 0.944 (B200 0.948).
#
#   FI-level re-measurement of the 64-token window on the acceptance instrument (the
#   FlashInfer call, CUPTI call span split per kernel, cold L2, 5 interleaved rounds,
#   one process per arm; steps 2916e66a06685ae93d43293c GB300 / 9ac4a94a6de616b831f11b19
#   B200), ring / start scan-kernel time, nseq x 64 tokens, z, initial states (GB300 / B200):
#     bf16 state   nseq 1 0.937 / 0.940   nseq 2 1.011 / 1.001   nseq 3 0.996 / 0.989
#     f32 state    nseq 1 0.939 / 0.941   nseq 2 1.003 / 0.975   nseq 3 0.978 / 0.974
#   The 93-row gate row nem_v64_64 (2 x 64, bf16 state) read 1.018 / 1.017 / 1.013 on GB300
#   in three earlier gates against 0.997 / 0.997 on B200, and its f32 twin 0.978-0.987 /
#   0.977.  So with a 16-bit state the 2 x 64 cell is a measured loss on GB300 (four
#   readings) and a tie on B200, and the 3 x 64 cell a tie on GB300 (B200 0.989):
#   the start schedule begins at 2 sequences x 128 heads.  With an f32 state (twice
#   the final-state bytes per item) the 2 x 64 and 3 x 64 cells keep winning on both
#   archs, so its threshold stays at 4 x 128.  The kernel-level cells above read the
#   bf16 2 x 64 cell as a tie (0.999 / 0.993-1.003); the FI-level reading decides
#   because it is the acceptance instrument.  float16 state is unmeasured and
#   follows bfloat16 (same state width).
#
# Rule: a measured losing cell goes to the start schedule (``_noring``); a
# winning cell keeps the ring program; ties and unmeasured cells follow the ring
# program only where the nearest measured cells win on BOTH archs, otherwise the
# start schedule.  The varlen loss geometry is the one where consecutive
# sequence pairs tile every 128-token chunk with two partial segments (both
# one-chunk items on the scalar-store path): uniform 64-token sequences and
# the pair-sum-128 mixtures, i.e. 64 tokens per sequence on average; uniform
# 63 and 65 tokens win on both archs, so the window is exclusive of them; at
# 64 tokens the item-count threshold is 2 sequences x 128 heads for a 16-bit
# state (the FI-level 2 x 64 loss above); an f32 state never takes the start
# schedule (round-6 table below).  The batched no-z batch-2 cells at 50-75
# tokens are the only batched readings at or beyond the band (batch 1, 3, 4, 8
# and z calls win); 25 and 90 tokens win on both archs.
# Only host-known quantities enter: the runner's mode / has_z / nheads /
# state dtype and the call's batch, seqlen and sequence count
# (``cu_seqlens.numel() - 1``, ``num_seqs`` or ``initial_states.shape[0]``);
# no per-sequence lengths and no device synchronisation, so prepared / captured
# shapes keep their program.
#
# Recalibration on the current exact-scan programs
# (Cake 370cc1dc7941 + the adopted 1b(ii) prescale factor table, which changes
# the per-chunk cost of BOTH variants): scan-kernel time ring / ``_noring``,
# the two variants of the same program as the two arms, CUPTI, cold L2, 3 and 5
# interleaved rounds, kernel band 0.005.  GB300 (bf16 at 3 and 5 rounds, f32):
#   varlen, z, 64 tokens, bf16 state: nseq 1 0.940; nseq 2 / 3 / 4 / 5 1.003 /
#     1.000 / 1.002 / 1.003 (5 rounds 0.989 / 1.001 / 1.000 / 1.005: ties);
#     nseq 6 / 8 / 16 / 64 1.011 / 1.017 / 1.013 / 1.014 (5 rounds 1.008 / 1.021);
#     63 / 65 tokens at nseq 8 0.977 / 0.969; mixtures [63, 65] x 4 1.012,
#     [32, 96] x 4 1.011 -> the bf16 / f16 window above is unchanged.
#   varlen, z, 64 tokens, f32 state: nseq 1 / 2 / 3 / 4 / 5 / 6 / 8 / 16 / 64 =
#     0.945 / 0.986 / 0.984 / 0.993 / 0.990 / 0.983 / 0.997 / 0.989 / 0.981 ->
#     the ring program wins every f32 cell (nseq 8 a tie): the round-5 f32
#     threshold (4 x 128 items) is retired; an f32 state never takes ``_noring``
#     on the varlen path.
#   batched, no z, batch 2 x 25 / 50 / 60 / 75 / 90 / 100 tokens: bf16 0.997 /
#     0.995 / 0.987 / 0.995 / 0.992 / 0.995 (5 rounds 1.003 / 0.987 / 0.997 /
#     0.987 / 0.992 / 0.985), f32 0.977 / 0.982 / - / 0.990 / 0.973 / -;
#     short_batched_2x50 0.992 / 0.995 -> no batched cell loses on GB300 any more.
# B200 (bf16 at 3 and 5 rounds, f32):
#   varlen, z, 64 tokens, bf16: nseq 1 0.932; 2 / 3 / 4 / 5 1.001 / 0.985 /
#     0.998 / 0.996 (5 rounds 0.990 / 0.989 / 0.999 / 0.996); 6 / 8 / 16 / 64
#     1.017 / 1.012 / 1.012 / 1.006 (5 rounds 1.003 / 1.013); 63 / 65 tokens
#     0.966 / 0.967; mixtures 1.009 / 1.009 -> same window as GB300.
#   varlen, z, 64 tokens, f32: nseq 1 / 2 / 3 / 4 / 5 / 6 / 8 / 16 / 64 = 0.954 /
#     0.985 / 0.974 / 0.994 / 0.983 / 0.991 / 0.995 / 0.991 / 0.969 -> ring wins
#     beyond the band on both archs at every f32 cell the round-5 rule sent to
#     ``_noring`` (nseq 8 a tie on both, its neighbours win on both).
#   batched, no z, batch 2 x 25 / 50 / 60 / 75 / 90 / 100: bf16 1.000 / 1.002 /
#     0.995 / 0.998 / 0.990 / 0.990 (5 rounds 1.005 / 0.998 / 1.000 / 0.995 /
#     0.993 / 0.986), f32 0.993 / 0.995 / - / 0.995 / 0.984 / -;
#     short_batched_2x50 0.995 / 1.000 -> the 50-75-token cells are ties on
#     B200 (GB300 wins): not a two-arch win, so the batched window stays.
# Round-6 rule change: an f32 state never takes the start schedule on the
# varlen path (the bf16 / f16 varlen window and the dtype-independent batched
# window are unchanged).
_EXACT_NORING_VARLEN_MIN_ITEMS = {  # sequences x heads, by state dtype; a state
    # dtype absent here (f32, round 6) never takes the varlen start schedule
    torch.bfloat16: 2 * 128,
    torch.float16: 2 * 128,
}
# exclusive bounds on tokens per sequence (average over the call)
_EXACT_NORING_VARLEN_TOKENS = (63, 65)
_EXACT_NORING_BATCHED_BATCH = 2
_EXACT_NORING_BATCHED_SEQLEN = (25, 90)  # exclusive bounds


def _exact_scan_variant(
    *,
    mode_varlen: bool,
    has_z: bool,
    nheads: int,
    num_sequences: int,
    total_tokens: int,
    batch: int,
    seqlen: int,
    state_dtype: torch.dtype,
) -> str:
    """Schedule variant of the exact-scan program for one call shape: ``""`` (the
    ring program) or ``"noring"`` (the start schedule), from host-known
    quantities only (see the calibration tables above)."""

    if mode_varlen:
        min_items = _EXACT_NORING_VARLEN_MIN_ITEMS.get(state_dtype)
        if min_items is None or not has_z or num_sequences * nheads < min_items:
            return ""
        low, high = _EXACT_NORING_VARLEN_TOKENS
        # tokens per sequence strictly inside (low, high), in integers
        if low * num_sequences < total_tokens < high * num_sequences:
            return "noring"
        return ""
    if has_z or batch != _EXACT_NORING_BATCHED_BATCH:
        return ""
    low, high = _EXACT_NORING_BATCHED_SEQLEN
    return "noring" if low < seqlen < high else ""


@functools.cache
def _load_generated_program(name: str, arch: str):
    """Build one program for ``arch``; the loaded module serves every device."""

    program = _arch_program(name, arch)
    main_pdl = _main_pdl(name, arch)
    source_dir = _source_dir()
    host_source = _render_host_source(
        (source_dir / program.host_template).read_text(encoding="utf-8"),
        name,
        program,
        main_pdl=main_pdl,
    )
    nvcc = _nvcc()
    digest = hashlib.sha256()
    digest.update(host_source.encode("utf-8"))
    digest.update(arch.encode())
    digest.update(str(nvcc).encode())
    device_sources: list[tuple[_Kernel, Path]] = []
    for kernel in program.kernels:
        source_path = source_dir / _DEVICE_DIR / kernel.source
        source_bytes = source_path.read_bytes()
        if (
            main_pdl
            and kernel is program.main
            and b"griddepcontrol.wait" not in source_bytes
        ):
            raise RuntimeError(
                f"Cake SSDCombined {name} ({arch}) launches its main kernel "
                f"with programmatic dependent launch, but {kernel.source} "
                "never waits on the preprocess grid"
            )
        digest.update(kernel.module.encode())
        digest.update(source_bytes)
        digest.update("\0".join(kernel.compile_flags).encode())
        device_sources.append((kernel, source_path))

    key = digest.hexdigest()[:16]
    module_name = f"cake_mamba_ssd_{name}_{arch}_{key}"
    build_dir = jit_env.FLASHINFER_JIT_DIR / module_name
    build_dir.mkdir(parents=True, exist_ok=True)
    lock_path = build_dir / f"{module_name}.lock"
    with FileLock(lock_path, thread_local=False):
        cubins: dict[str, bytes] = {}
        for kernel, source_path in device_sources:
            cubin_path = build_dir / f"{kernel.module}.cubin"
            if not cubin_path.is_file():
                temporary_cubin = build_dir / f"{kernel.module}.{os.getpid()}.tmp.cubin"
                command = [
                    str(nvcc),
                    "-cubin",
                    f"-arch={arch}",
                    "--std=c++17",
                    "-O3",
                    "-I",
                    str(nvcc.parent.parent / "include"),
                    *kernel.compile_flags,
                    str(source_path),
                    "-o",
                    str(temporary_cubin),
                ]
                process = subprocess.run(command, text=True, capture_output=True)
                if process.returncode != 0:
                    temporary_cubin.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Cake SSDCombined nvcc failed for {name}/{kernel.module} "
                        f"({arch}):\n{process.stderr}"
                    )
                os.replace(temporary_cubin, cubin_path)
            cubins[kernel.module] = cubin_path.read_bytes()

        return cpp.load_inline(
            module_name,
            cpp_sources=host_source,
            embed_cubin=cubins,
            extra_include_paths=[str(nvcc.parent.parent / "include")],
            extra_cflags=["-O3", "-DTVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API=1"],
            extra_ldflags=["-lcuda"],
            build_directory=str(build_dir),
        )


class CakeSSDCombined:
    """Source-built Cake implementation of the admitted SSDCombined domain.

    The kernels require head dimension 64, state dimension 128, BF16
    inputs/outputs, BF16, FP16, or FP32 states, and SM100, SM103 or SM107
    (Rubin).  Any positive ``chunk_size`` is accepted as the caller's
    convention: the programs tile
    the token axis in 128-token chunks internally and the SSD output and
    final states do not depend on the chunk size (chunking factorises the
    same recurrence; only rounding differs), so ``chunk_size=256`` (the
    Nemotron-H engine default) runs the same kernels as 128.  Head and group
    counts are runtime values and may be any positive pair for
    which ``nheads`` is divisible by ``ngroups``.  Any positive sequence
    length is accepted: the kernels handle a partial trailing chunk, and a
    call shorter than one 128-token chunk (batched ``seqlen < 128`` or packed
    varlen ``total < 128``) binds the caller's tensors directly: the token
    axis of the x/B/C/out tensor maps admits a global extent below the
    128-row box, TMA zero-fills the rows past ``seqlen`` on loads and clips
    them on stores, so the kernel sees exactly the state of a partial
    trailing chunk.  The output
    is token-major ``[batch, seqlen, nheads, 64]`` (packed varlen:
    ``[1, total_seqlen, nheads, 64]``), written directly by the kernels.
    Every call issues one launcher call: the preprocess (which also derives
    ``seq_chunk_cumsum`` from the packed-varlen metadata unless the caller
    supplies a precomputed vector) followed by the scan.  Inside the fused
    shape window the fused exact-scan main kernel serves every form: batched
    calls, the ``cu_seqlens`` form and the ``seq_idx`` triple form with a
    caller ``seq_chunk_cumsum`` the preprocess does not rewrite
    (``update_seq_chunk_cumsum=False``; the kernel reads the caller's tables)
    run the ``exactfused_*`` program with its preprocess stage skipped (grid
    0), one kernel; the triple form without one (the table is published,
    runner-owned or into the caller's buffer with
    ``update_seq_chunk_cumsum=True``) runs the program of the device's
    compute capability: on SM100 ``exactmeta_*`` -- the light packed-varlen
    metadata kernel publishes the table, then the same fused main kernel runs
    (two launches, no dt preprocess); on SM103 and SM107 ``exactone_*`` --
    the fused main kernel derives its private table in a CTA prologue (one
    launch) -- where the table is runner-owned, and the two-kernel program
    where the caller's buffer must receive it.  A caller buffer for the
    ``cu_seqlens``-derived vector brings the fused program's full preprocess
    back to publish it (two launches).

    Seven kernel families serve every call and are selected per call.  The
    fused exact scan (``exactfused_*`` programs, ``exactmeta_*`` behind the
    metadata kernel and the SM103 / SM107 ``exactone_*`` render) runs one persistent CTA per
    (sequence, head) item over its chunks in order and computes delta /
    cumsum in-CTA; the two-kernel exact scan (``exactpre_*``) is the same
    persistent scan reading the delta / cumsum its preprocess wrote to the
    HBM workspace -- the two-launch form, kept for the calls the fused
    program loses on (:func:`fused_selected`: batched calls with more than
    1024 (sequence, head) items, more than 16 chunks per sequence or more
    than 2048 chunk-items; packed-varlen calls with more than 256 items or
    more than 512 packed chunk-items); two-kernel calls with one chunk per
    (sequence, head) item run the retained start programs (``exact_*``,
    :func:`two_kernel_family`) instead of ``exactpre_*``.  The
    look-back chunk-parallel program (``chunkpar_*``) runs one CTA per SM
    over item-major tile ranges with a decoupled look-back over the carried
    state: the same arithmetic with the carried state of later ranges
    re-associated (judged by the precision gate, not bitwise).  The
    three-phase chunk-parallel program (``chunkpar3_*``) runs one CTA per
    (chunk, head) tile in three phases separated by a software grid barrier;
    its output and final states are bitwise the round-5 exact scan's (the
    round-6 exact scan rounds differently and is judged against it by the
    precision gate).  A chunk-parallel family is selected by a calibrated
    cost rule (:func:`chunk_parallel_family`: fewer
    (sequence, head) items than SMs, the per-tile workspace within 256 MiB,
    and a predicted main-kernel time that beats the exact scan's by the
    selection margin), so long prefills with few heads leave the exact scan
    (one sequence of 32768 tokens with 8 heads runs about four times faster)
    and 128-head calls never do; inside that regime the three-phase program
    serves batched calls with at most 256 tiles or with at least 32 items and
    64 chunks per item, the look-back every other call
    (:func:`three_phase_preferred`).  The environment variable
    ``FLASHINFER_CAKE_SSD_CHUNK_PARALLEL`` (``auto``, the default;
    ``always`` = look-back; ``three_phase``; ``never`` = exact scan) forces
    one family; any other value is rejected.  It is read on every unprepared
    eager call; :meth:`prepare` reads it once and freezes the choice for the
    call shape.  The chunk-parallel workspace (32 KB f32 + 16 KB bf16 per
    tile, grown only when a call needs more tiles, shared by both
    chunk-parallel families, plus the look-back's aggregate workspace and
    1028-word barrier and the three-phase program's two-word barrier, each
    zeroed once per device) is owned by the runner.  :attr:`last_program_name`
    names the program of the most recent call.

    Workspaces are kept per call shape in a small registry
    (:attr:`workspace_capacity` uncaptured entries, least recently used
    evicted first), so alternating between recent shapes re-allocates
    nothing.  CUDA-graph capture: a call whose program has been launched in
    this process may be recorded into a :class:`torch.cuda.CUDAGraph`; the
    storage it allocates while captured (output, final states, a new
    shape's workspace, packed copies) comes from the graph's private pool
    and every replay reuses it, and the workspace it recorded is pinned --
    never evicted, and its chunk-parallel buffers are kept when the
    device's grow-only buffers are later replaced -- because the graph
    replays into exactly those addresses.  :meth:`prepare` performs every
    allocation, program build and environment read a call shape needs
    ahead of time, so :meth:`run` for that shape with a static ``out=`` is
    a pure launch (``_allocations_in_last_run`` stays 0); the prepared
    workspace is pinned like a captured one.  A captured call refuses with
    :class:`CakeSSDCombinedCaptureError` only when its call shape has never
    run or been prepared on this runner (the workspace would be built by
    recorded, not executed, kernels) or when its program has not been
    launched in this process (loading or building a program, and the host
    shim's first-launch kernel-handle resolution, cannot be recorded): run
    the shape eagerly once before capturing it.

    Packed varlen has two forms.  ``cu_seqlens`` (int32 ``[num_seqs + 1]``,
    ``cu[0] == 0``, non-decreasing, ``cu[-1] == seqlen``, ``batch == 1``):
    the preprocess derives the 128-granularity segment tables, the sequence
    prefix sum and a sentinel on the device, so the caller's chunk-size
    convention never matters, ``seq_idx`` is optional (never read) and the
    chunk-unaligned ``checkpoint_token_indices`` boundaries are inserted
    automatically; the sequence count is ``cu_seqlens.numel() - 1`` (host
    shape, no synchronisation).  An invalid ``cu_seqlens`` (``cu[0] != 0``,
    decreasing, ``cu[-1] != seqlen``, more segments than the bound) is
    memory-safe: it is flagged in ``preprocess_status`` (:meth:`seq_idx_status`),
    the untouched sequences stay exact and the offending tokens' outputs are
    undefined.  The ``seq_idx`` / ``chunk_indices`` / ``chunk_offsets`` triple
    (chunk-128 logical segments built by the caller; checkpoint tokens must
    then be logical chunk ends) is accepted with ``chunk_size == 128`` only.

    Packed-varlen ``seq_idx`` contract (CAKE-990): ids must be non-decreasing
    along the packed token axis.  An id in ``[0, num_sequences)`` without
    tokens is allowed: the preprocess gives it an empty chunk range and the
    scan writes its final state as ``initial_states[id]`` (zero when there
    are no initial states).  An id outside ``[0, num_sequences)`` or below
    its predecessor never writes outside the ``[num_sequences + 1]`` boundary
    table; the preprocess flags it in the runner-owned ``preprocess_status``
    word instead (read with :meth:`seq_idx_status`, a synchronizing debug
    accessor; nothing on the hot path reads it).  On a flagged call the
    outputs of the offending tokens are undefined (they are skipped or folded
    into a neighbouring range) and so is every sequence whose range they
    touch; the other in-range sequences are still correct.  The preprocess
    reads ``seq_idx`` only when it derives ``seq_chunk_cumsum``
    (``seq_chunk_cumsum=None``, or a caller buffer with
    ``update_seq_chunk_cumsum=True``).  A caller-supplied table with
    ``update_seq_chunk_cumsum=False`` is trusted as the sequence boundaries:
    no kernel reads ``seq_idx`` on that call, so :meth:`seq_idx_status`
    cannot report ids the caller already consumed when building the table.
    """

    #: Program of the most recent :meth:`run` (``exactfused_*``,
    #: ``exactmeta_*``, ``exactone_*``, ``exactpre_*``, ``exact_*``,
    #: ``chunkpar_*`` or ``chunkpar3_*``);
    #: ``None`` before the first call.
    last_program_name: Optional[str] = None
    #: Uncaptured per-shape workspaces a runner keeps (least recently used
    #: evicted first); captured workspaces are pinned in addition.  Read when
    #: the runner is constructed.
    workspace_capacity: int = 8

    def __init__(
        self,
        chunk_size: int,
        nheads: int,
        headdim: int,
        dstate: int,
        ngroups: int,
        *,
        io_dtype: torch.dtype,
        state_dtype: torch.dtype,
        has_d: bool,
        d_has_hdim: bool,
        has_initial_states: bool,
        has_varlen: bool,
        has_z: bool,
        seq_idx_dtype: torch.dtype,
    ) -> None:
        if headdim != _HEADDIM or dstate != _DSTATE:
            raise ValueError("Cake SSDCombined requires headdim=64 and dstate=128")
        if (
            not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or chunk_size <= 0
        ):
            raise ValueError(
                "Cake SSDCombined chunk_size must be a positive int (a caller "
                "convention; the kernels tile 128 tokens internally)"
            )
        if nheads <= 0 or ngroups <= 0 or nheads % ngroups:
            raise ValueError(
                "Cake SSDCombined requires positive nheads divisible by ngroups"
            )
        if io_dtype != torch.bfloat16:
            raise ValueError("Cake SSDCombined requires bfloat16 IO")
        if state_dtype not in _STATE_DTYPE_KEYS:
            raise ValueError(
                "Cake SSDCombined state dtype must be bfloat16, float16, or float32"
            )
        if seq_idx_dtype not in (torch.int32, torch.int64):
            raise ValueError("Cake SSDCombined seq_idx dtype must be int32 or int64")
        _target_arch(torch.cuda.current_device())
        self.chunk_size = int(chunk_size)
        self.nheads = nheads
        self.ngroups = ngroups
        self.state_dtype = state_dtype
        self.has_d = bool(has_d)
        self.d_has_hdim = bool(d_has_hdim)
        self.has_initial_states = bool(has_initial_states)
        self.has_varlen = bool(has_varlen)
        self.has_z = bool(has_z)
        self.seq_idx_dtype = seq_idx_dtype
        # Per-call-shape workspaces: a bounded LRU of uncaptured entries plus
        # the pinned entries CUDA graphs were captured with.
        self._workspaces = _WorkspaceRegistry(self.workspace_capacity)
        # The workspace of the most recent call (diagnostics: tests read the
        # preprocess-derived tables through it).
        self._workspace: Optional[dict[str, Any]] = None
        # Family override frozen per call shape by ``prepare``: ``run`` never
        # reads the environment for a prepared shape.
        self._prepared_modes: dict[_ShapeKey, str] = {}
        self._dummy_cache: dict[Tuple[Optional[int], torch.dtype], torch.Tensor] = {}
        self._preprocess_status: dict[Optional[int], torch.Tensor] = {}
        # Chunk-parallel main-kernel buffers per device (grow-only) and the
        # grid-barrier words per device (zeroed once); see
        # ``_chunk_parallel_workspace``.
        self._chunk_parallel_buffers: dict[Optional[int], dict[str, torch.Tensor]] = {}
        self._grid_barriers: dict[Optional[int], torch.Tensor] = {}
        self._grid_barriers3: dict[Optional[int], torch.Tensor] = {}
        self._agg_works: dict[Optional[int], torch.Tensor] = {}
        # Per-call bookkeeping: how many allocations the most recent call made
        # (0 for a prepared shape with a static ``out``).
        self._allocations_in_last_run = 0

    def _allocating(self, what: str) -> None:
        """Account one allocation of ``what`` by the current call.

        Every allocation site of the launch path passes through here, so the
        eager path can prove it allocated nothing
        (``_allocations_in_last_run == 0`` after :meth:`prepare`).  Under
        CUDA-graph capture an allocation is legal: PyTorch serves it from the
        graph's private pool and every replay reuses that storage (the
        behaviour callers already rely on when they capture a warm call);
        :meth:`prepare` only makes the captured call allocation-free.
        """

        del what
        self._allocations_in_last_run += 1

    def _preprocess_status_buffer(self, device: torch.device) -> torch.Tensor:
        """The runner-owned ``preprocess_status`` words (int32, per device).

        The preprocess stores 1 into word 0 when a packed-sequence id is out of
        range or non-monotonic (see the class docstring).  Word 1 is the item
        counter the sm_107a claim programs draw from (:func:`claim_selected`).
        The sm_107a one-CTA pair programs (:func:`tail_one_cta_selected`) use
        the words after them, sized by the device's tail grid
        (:func:`_tail_grid_size`): pair ``p``'s rank ticket in word ``2 + p``,
        its hand-off flag in bit ``p % 32`` of word ``2 + pairs + p // 32``
        (on 212 SMs 112 words, within one 512-byte allocator block like the
        two words alone).  Every complete launch leaves word 1 and the words
        after it 0, so a stream's launches see them 0.
        Like the runner's other per-device workspace (the delta and cumsum
        tiles, the grid barriers), the words serve one stream at a time: a
        runner is not used concurrently on two streams.  Both are allocated
        zeroed once per device and never read on the launch path.
        """

        status = self._preprocess_status.get(device.index)
        if status is None:
            self._allocating("the preprocess status words")
            pairs = _tail_grid_size(sm_count=_sm_count(device.index)) // 2
            words = 2 + pairs + (pairs + 31) // 32
            status = torch.zeros(words, dtype=torch.int32, device=device)
            self._preprocess_status[device.index] = status
        return status

    def seq_idx_status(
        self, reset: bool = False, *, device: Optional[torch.device] = None
    ) -> int:
        """Debug read of the ``seq_idx`` status word (synchronizes the device).

        Returns 1 when any call on ``device`` (default: the current CUDA
        device) since the last reset met a packed-sequence id outside
        ``[0, num_sequences)`` or below its predecessor, else 0.  Only calls
        on which the preprocess derives ``seq_chunk_cumsum`` read ``seq_idx``;
        a precomputed table (``update_seq_chunk_cumsum=False``) is trusted and
        leaves the word untouched.  With ``reset=True`` the word is cleared
        after reading.  The read is a
        device-to-host copy, so this is for tests and diagnostics only;
        ``run`` never reads the word.
        """

        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        status = self._preprocess_status_buffer(device)
        value = int(status[0].item())
        if reset:
            status.zero_()
        return value

    def _dummy(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device.index, dtype)
        value = self._dummy_cache.get(key)
        if value is None:
            self._allocating(f"a {dtype} dummy operand")
            value = torch.empty(1, dtype=dtype, device=device)
            self._dummy_cache[key] = value
        return value

    def _contiguous_input(
        self,
        workspace: dict[str, Any],
        name: str,
        value: Optional[torch.Tensor],
        align: int = 1,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Resolve graph-stable packed storage for a public strided input.

        Returns the tensor to bind and the source still to be copied into
        it, or ``None`` when the input is already contiguous.  The copies are
        issued together right before the launch so no Python runs between the
        auxiliary device work and the kernels.  One buffer per (input, shape,
        dtype) lives in the workspace for its lifetime and is never replaced,
        so the address a captured graph copies into stays valid when a later
        eager call of the same shape brings, say, a bf16 ``dt`` instead of an
        f32 one.  ``align`` is the byte alignment the program needs of the
        bound address: a contiguous input at a lesser one is copied too.
        """

        if value is None or (value.is_contiguous() and value.data_ptr() % align == 0):
            return value, None
        buffers: dict[tuple[str, tuple[int, ...], torch.dtype], torch.Tensor]
        buffers = workspace["contiguous"]
        buffer_key = (name, tuple(value.shape), value.dtype)
        buffer = buffers.get(buffer_key)
        if buffer is None:
            self._allocating(f"a packed copy of the strided input {name}")
            buffer = torch.empty(
                tuple(value.shape), dtype=value.dtype, device=value.device
            )
            buffers[buffer_key] = buffer
        return buffer, value

    def _get_workspace(
        self,
        *,
        key: _WorkspaceKey,
        device: torch.device,
        sm_count: int,
        capability: tuple[int, int],
    ) -> dict[str, Any]:
        """The workspace of ``key``: the registered one, or materialised now
        (never under capture: ``_plan`` refuses an unregistered shape there).

        The kernel family (``workspace["program"]``) is a function of the
        key, which carries the ``FLASHINFER_CAKE_SSD_CHUNK_PARALLEL`` override
        of the call, and of the device's SM count and capability (fixed per
        ``device.index``).
        """

        entry = self._workspaces.get(key)
        if entry is None:
            self._allocating("the workspace of a new call shape")
            entry = self._workspaces.add(
                key,
                self._materialize_workspace(
                    key=key, device=device, sm_count=sm_count, capability=capability
                ),
            )
        self._workspace = entry.workspace
        return entry.workspace

    def _materialize_workspace(
        self,
        *,
        key: _WorkspaceKey,
        device: torch.device,
        sm_count: int,
        capability: tuple[int, int],
    ) -> dict[str, Any]:
        """Allocate the per-call-shape workspace of ``key`` and select its
        kernel family.  The chunk-parallel buffers are per device, not per
        shape: this only grows them to the shape's tile bound (the call binds
        the device's current buffers, see ``_chunk_parallel_workspace``)."""

        batch, seqlen, nchunks = key.batch, key.seqlen, key.nchunks
        num_segments, num_sequences = key.num_segments, key.num_sequences
        ids = torch.arange(num_segments, dtype=torch.int32, device=device)
        chunks = ids % nchunks
        starts = (ids // nchunks) * seqlen + chunks * _CHUNK_SIZE
        # Batched segments are physical chunks; the trailing chunk of a
        # sequence whose length is not a multiple of 128 is partial.
        lengths = torch.clamp(seqlen - chunks * _CHUNK_SIZE, max=_CHUNK_SIZE)
        tile_count = num_segments * self.nheads
        # The chunk-parallel rule reads the same host quantities in both
        # varlen metadata forms: the cu_seqlens form only knows the host
        # segment bound, so the triple form selects on that bound too (its
        # exact segment count still sizes the workspace).  Within the
        # exact-scan family the form then decides fused vs two-kernel (below).
        selection_segments = (
            _segment_bound(seqlen, num_sequences) if self.has_varlen else num_segments
        )
        family = chunk_parallel_family(
            nheads=self.nheads,
            num_sequences=num_sequences,
            num_segments=selection_segments,
            nchunks=nchunks,
            mode_varlen=self.has_varlen,
            sm_count=sm_count,
            capability=capability,
            mode=key.chunk_parallel_mode,
        )
        chunk_parallel = family != "exact"
        # Inside the fused shape window (the FI mirror of the Cake
        # ``fused_route`` window) the fused main kernel serves every form
        # that has its sequence table: batched calls, the cu_seqlens form
        # (the kernel derives the segment tables in-CTA) and the seq_idx
        # triple form with a caller ``seq_chunk_cumsum`` the preprocess does
        # not rewrite (the kernel reads the caller's tables,
        # ``key.caller_cumsum``) run ``exactfused`` as one launch (preprocess
        # grid 0).  The triple form without one needs its sequence table
        # first, and which program provides it is decided per compute
        # capability, each measured on its own architecture: on SM100
        # ``exactmeta`` runs the light packed-varlen metadata kernel (one
        # thread per segment, no dt work) ahead of the same fused main kernel
        # (the stage drops from 4.1-4.7 us to 2.4-2.9 us; the fused main
        # kernel costs 0.7-0.9 us more than the two-kernel main kernel); on
        # SM103 -- and SM107, whose fused varlen render derives the table
        # in-kernel too and whose host is a Grace CPU like GB300's -- the host
        # interval between two launches hides any lighter stage, so
        # ``exactone`` runs that render of the fused main kernel, whose CTA
        # prologue derives a private table from the triple (one launch, no
        # stage) -- unless the caller's buffer must receive the table
        # (``key.publish_table``), which only a publishing stage writes: that
        # form keeps the two-kernel program on SM103 and SM107.
        derives_table = (
            self.has_varlen and not key.from_cu_seqlens and not key.caller_cumsum
        )
        light_stage = derives_table and tuple(capability) in _LIGHT_STAGE_CAPABILITIES
        one_launch = (
            derives_table
            and not key.publish_table
            and tuple(capability) in _ONE_LAUNCH_CAPABILITIES
        )
        fused = (
            not chunk_parallel
            and (not derives_table or light_stage or one_launch)
            and fused_selected(
                nheads=self.nheads,
                num_sequences=num_sequences,
                nchunks=nchunks,
                mode_varlen=self.has_varlen,
                total_tokens=batch * seqlen,
            )
        )
        if (
            not fused
            and not chunk_parallel
            and not derives_table
            and not self.has_z
            and fused_no_z_extended(
                capability=capability,
                nheads=self.nheads,
                num_sequences=num_sequences,
                nchunks=nchunks,
                mode_varlen=self.has_varlen,
                total_tokens=batch * seqlen,
            )
        ):
            fused = True
        # The two-kernel pair program (``tail_selected``) is judged after the
        # fused route and before the ring / noring choice; its route and the
        # retained ``exact`` family are disjoint (16 chunks per item imply
        # ``exactpre``).
        tail = (
            not chunk_parallel
            and not fused
            and tail_selected(
                total_work=num_sequences * self.nheads,
                chunks_per_item=host_chunks_per_item(
                    mode_varlen=self.has_varlen,
                    num_sequences=num_sequences,
                    # packed varlen: x is [1, total tokens, nheads, 64]
                    seqlen=batch * seqlen if self.has_varlen else seqlen,
                ),
                sm_count=sm_count,
                mode_varlen=self.has_varlen,
                full_grid_rule=tail_full_grid_rule(capability),
                has_z=self.has_z,
            )
        )
        variant = (
            ""
            if chunk_parallel
            else "tail"
            if tail
            else _exact_scan_variant(
                mode_varlen=self.has_varlen,
                has_z=self.has_z,
                nheads=self.nheads,
                num_sequences=num_sequences,
                # packed varlen: x is [1, total tokens, nheads, 64]
                total_tokens=batch * seqlen,
                batch=batch,
                seqlen=seqlen,
                state_dtype=self.state_dtype,
            )
        )
        # sm_107a: the start schedule of a packed-varlen call outside the
        # fused window gives way to the ring ``exactpre`` program above
        # ``_NORING_RING_EXACTPRE_MIN_ITEMS_PER_SM`` items per SM (judged
        # before the retained-noring rule below, on the forms that rule
        # sends to the fused program).
        ring_items_per_sm = _NORING_RING_EXACTPRE_MIN_ITEMS_PER_SM.get(
            (int(capability[0]), int(capability[1]))
        )
        if (
            ring_items_per_sm is not None
            and self.has_varlen
            and not chunk_parallel
            and not fused
            and not tail
            and variant == "noring"
            and (not derives_table or light_stage or one_launch)
            and num_sequences * self.nheads > ring_items_per_sm * sm_count
            and two_kernel_family(
                mode_varlen=self.has_varlen,
                num_sequences=num_sequences,
                seqlen=batch * seqlen,
                nheads=self.nheads,
                sm_count=sm_count,
                capability=capability,
                variant="",
                state_dtype=self.state_dtype,
            )
            == "exactpre"
        ):
            variant = ""
        if (
            tuple(capability) in _NORING_RETAINED_FUSED
            and not chunk_parallel
            and not fused
            and not tail
            and variant == "noring"
            and two_kernel_family(
                mode_varlen=self.has_varlen,
                num_sequences=num_sequences,
                seqlen=batch * seqlen if self.has_varlen else seqlen,
                nheads=self.nheads,
                sm_count=sm_count,
                capability=capability,
                variant=variant,
                state_dtype=self.state_dtype,
            )
            == "exact"
            and (not derives_table or light_stage or one_launch)
        ):
            fused = True
        if chunk_parallel:
            program_family = family
        elif fused:
            program_family = (
                "exactmeta"
                if light_stage
                else "exactone"
                if one_launch
                else "exactfused"
            )
            if (
                program_family == "exactfused"
                and self.has_varlen
                and not variant
                and fused_ibulk_selected(
                    capability=capability,
                    state_dtype=self.state_dtype,
                    has_initial_states=self.has_initial_states,
                    total_work=num_sequences * self.nheads,
                    chunks_per_item=host_chunks_per_item(
                        mode_varlen=True,
                        num_sequences=num_sequences,
                        # packed varlen: x is [1, total tokens, nheads, 64]
                        seqlen=batch * seqlen,
                    ),
                    sm_count=sm_count,
                )
            ):
                variant = "ibulk"
        else:
            program_family = two_kernel_family(
                mode_varlen=self.has_varlen,
                num_sequences=num_sequences,
                # packed varlen: x is [1, total tokens, nheads, 64]
                seqlen=batch * seqlen if self.has_varlen else seqlen,
                nheads=self.nheads,
                sm_count=sm_count,
                capability=capability,
                variant=variant,
                state_dtype=self.state_dtype,
            )
            if claim_selected(
                mode_varlen=self.has_varlen,
                num_sequences=num_sequences,
                seqlen=batch * seqlen,
                nheads=self.nheads,
                sm_count=sm_count,
                capability=capability,
                variant=variant,
                state_dtype=self.state_dtype,
            ):
                program_family, variant = "exactpre", "claim"
            # The initial-state bulk program takes the ring calls the claim
            # rule leaves: a call both rules admit keeps the claim program.
            if (
                program_family == "exactpre"
                and not variant
                and ibulk_selected(
                    capability=capability,
                    state_dtype=self.state_dtype,
                    mode_varlen=self.has_varlen,
                    total_work=num_sequences * self.nheads,
                    chunks_per_item=host_chunks_per_item(
                        mode_varlen=self.has_varlen,
                        num_sequences=num_sequences,
                        # packed varlen: x is [1, total tokens, nheads, 64]
                        seqlen=batch * seqlen if self.has_varlen else seqlen,
                    ),
                    sm_count=sm_count,
                    has_z=self.has_z,
                )
            ):
                variant = "ibulk"
            # sm_107a: the pair calls ``tail_one_cta_selected`` admits run the
            # pair program's one-CTA form.
            if variant == "tail" and tail_one_cta_selected(
                capability=capability,
                mode_varlen=self.has_varlen,
                chunks_per_item=host_chunks_per_item(
                    mode_varlen=self.has_varlen,
                    num_sequences=num_sequences,
                    # packed varlen: x is [1, total tokens, nheads, 64]
                    seqlen=batch * seqlen if self.has_varlen else seqlen,
                ),
            ):
                variant = "tailone"
        if program_family == "exactone":
            variant = exactone_table_variant(
                capability=capability,
                variant=variant,
                state_dtype=self.state_dtype,
                grid=_persistent_grid_size(
                    total_work=num_sequences * self.nheads, sm_count=sm_count
                ),
            )
        workspace: dict[str, Any] = {
            "program": _program_name(
                program_family,
                self.state_dtype,
                self.has_varlen,
                variant,
            ),
            "starts": starts,
            "lengths": lengths,
            "delta": torch.empty(
                (tile_count, _CHUNK_SIZE), dtype=torch.float16, device=device
            ),
            "cumsum": torch.empty(
                (tile_count, _CHUNK_SIZE), dtype=torch.float32, device=device
            ),
            "dt_float": torch.empty(
                (batch, seqlen, self.nheads),
                dtype=torch.float32,
                device=device,
            ),
            "dt_bias_float": torch.empty(
                self.nheads,
                dtype=torch.float32,
                device=device,
            ),
            # Bound when the caller passes no dt_bias; never written after
            # allocation, so it costs no per-call fill.
            "dt_bias_zero": torch.zeros(
                self.nheads,
                dtype=torch.float32,
                device=device,
            ),
            "d_head": torch.empty(
                self.nheads,
                dtype=torch.bfloat16,
                device=device,
            ),
            "final": torch.empty(
                (num_sequences, self.nheads, _HEADDIM, _DSTATE),
                dtype=self.state_dtype,
                device=device,
            ),
            # Packed copies of strided public inputs (``_contiguous_input``).
            "contiguous": {},
        }
        if self.has_varlen:
            # The runner-owned ``seq_chunk_cumsum`` the preprocess fills when
            # the caller supplies no vector (both packed-varlen forms).  Per
            # shape rather than per sequence count so that a captured graph's
            # vector is never replaced by a call of another shape.  The
            # one-launch ``exactone`` program owns one row per CTA instead:
            # its prologue derives this CTA's private table into row
            # ``blockIdx.x`` of a ``grid x (num_sequences + 1)`` scratch
            # (``_persistent_grid_size``, the main kernel's launch grid).
            table_rows = 1
            if program_family == "exactone":
                table_rows = _persistent_grid_size(
                    total_work=num_sequences * self.nheads, sm_count=sm_count
                )
            workspace["seq_chunk_cumsum"] = torch.empty(
                table_rows * (num_sequences + 1), dtype=torch.int32, device=device
            )
        if key.from_cu_seqlens:
            # cu_seqlens form: the preprocess publishes the derived
            # chunk_indices / chunk_offsets (+ the sentinel at the real
            # segment count) into these [bound + 1] tables; the scan
            # reads them with num_logical_chunks = bound.
            workspace.update(
                chunk_indices=torch.empty(
                    num_segments + 1, dtype=torch.int32, device=device
                ),
                chunk_offsets=torch.empty(
                    num_segments + 1, dtype=torch.int32, device=device
                ),
            )
        if chunk_parallel:
            self._chunk_parallel_workspace(
                device, tile_count, _chunk_parallel_vcount(capability, sm_count)
            )
        return workspace

    def _chunk_parallel_workspace(
        self, device: torch.device, tiles_bound: int, vcount: int
    ) -> dict[str, torch.Tensor]:
        """The chunk-parallel main kernels' workspace on ``device`` (both
        chunk-parallel families; the three-phase family binds the same S/H
        buffers and its own two-word barrier ``grid_barrier3``).

        ``s_work`` (f32 ``[tiles, 64, 128]``, the per-tile state increments,
        32 KB per tile) and ``h_work`` (bf16 ``[tiles, 64, 128]``, the state
        operand entering each tile, 16 KB per tile; bound as the ``h_map``
        tensor map and, through ``h_words``, as its u32 view) grow only: a
        call whose tile bound exceeds the allocation replaces them, smaller
        calls reuse them (the kernel addresses tiles below the call's bound
        only).  ``grid_barrier`` (u32 ``[ticket, epoch, pad, pad, flags]``) is
        allocated zeroed once per device and never re-zeroed: the kernel's
        claim counter wraps itself, its epoch word is free-running and the
        flags carry the epoch.  ``agg_work`` (f32, ``vcount`` range aggregates
        plus ``vcount`` scalars) is per device too and grows only; the kernel
        writes every slot it reads within one launch, so it needs no initial
        value, and it indexes at most the bound ``vcount`` ranges, so a buffer
        sized for a larger V (the uncapped table value) stays in bounds.  A
        workspace captured into a CUDA graph pins the buffers its graph
        recorded (``workspace["chunk_parallel"]``, see ``run``), so a later
        growth replaces the device's current buffers without freeing them; a
        captured call never grows them (its shape ran eagerly first, so the
        device's buffers already cover its tile bound).
        """

        buffers = self._chunk_parallel_buffers.get(device.index)
        if buffers is None or buffers["h_work"].shape[0] < tiles_bound:
            self._allocating(f"chunk-parallel buffers for {tiles_bound} tiles")
            shape = (tiles_bound, _HEADDIM, _DSTATE)
            h_work = torch.empty(shape, dtype=torch.bfloat16, device=device)
            buffers = {
                "s_work": torch.empty(shape, dtype=torch.float32, device=device),
                "h_work": h_work,
                "h_words": h_work.view(torch.uint32),
            }
            self._chunk_parallel_buffers[device.index] = buffers
        grid_barrier = self._grid_barriers.get(device.index)
        if grid_barrier is None:
            self._allocating("the chunk-parallel grid barrier")
            grid_barrier = torch.zeros(
                _CHUNK_PARALLEL_GRID_BARRIER_WORDS, dtype=torch.uint32, device=device
            )
            self._grid_barriers[device.index] = grid_barrier
        grid_barrier3 = self._grid_barriers3.get(device.index)
        if grid_barrier3 is None:
            self._allocating("the three-phase chunk-parallel grid barrier")
            grid_barrier3 = torch.zeros(
                _CHUNKPAR3_GRID_BARRIER_WORDS, dtype=torch.uint32, device=device
            )
            self._grid_barriers3[device.index] = grid_barrier3
        agg_elems = int(vcount) * _CHUNK_PARALLEL_AGG_ELEMS_PER_RANGE
        agg_work = self._agg_works.get(device.index)
        if agg_work is None or agg_work.shape[0] < agg_elems:
            self._allocating(
                f"the chunk-parallel aggregate workspace for {vcount} ranges"
            )
            agg_work = torch.empty(agg_elems, dtype=torch.float32, device=device)
            self._agg_works[device.index] = agg_work
        return {
            **buffers,
            "agg_work": agg_work,
            "grid_barrier": grid_barrier,
            "grid_barrier3": grid_barrier3,
        }

    def run(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        dt_softplus: bool = False,
        dt_limit: Tuple[float, float] = (0.0, float("inf")),
        initial_states: Optional[torch.Tensor] = None,
        seq_idx: Optional[torch.Tensor] = None,
        chunk_indices: Optional[torch.Tensor] = None,
        chunk_offsets: Optional[torch.Tensor] = None,
        seq_chunk_cumsum: Optional[torch.Tensor] = None,
        update_seq_chunk_cumsum: bool = False,
        checkpoint_token_indices: Optional[torch.Tensor] = None,
        checkpoint_state_slots: Optional[torch.Tensor] = None,
        checkpoint_states: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        return_final_states: bool = True,
        num_seqs: Optional[int] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        """Run one forward: plan every host-side decision, then issue only
        device work (the packed-input copies, then the single launcher call).

        Inside CUDA-graph capture the plan may allocate ``out``, final states
        and packed-input copies (from the graph's private pool) but never
        materialise a workspace this runner has not seen eagerly, nor load
        or build a program -- it refuses with
        :class:`CakeSSDCombinedCaptureError` in those cases -- and the
        workspace it binds is pinned, because the graph replays into those
        addresses.
        """

        capturing = _stream_capturing(x.device)
        plan = self._plan(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
            initial_states=initial_states,
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            seq_chunk_cumsum=seq_chunk_cumsum,
            update_seq_chunk_cumsum=update_seq_chunk_cumsum,
            checkpoint_token_indices=checkpoint_token_indices,
            checkpoint_state_slots=checkpoint_state_slots,
            checkpoint_states=checkpoint_states,
            out=out,
            return_final_states=return_final_states,
            num_seqs=num_seqs,
            cu_seqlens=cu_seqlens,
            capturing=capturing,
            preparing=False,
        )
        self.last_program_name = plan.program_name
        if capturing:
            # The graph replays into exactly these addresses: pin the
            # workspace (never evicted) and the chunk-parallel buffers it
            # recorded (a later growth must not free them).
            self._workspaces.pin(plan.workspace_key)
            if plan.chunk_parallel is not None:
                plan.workspace["chunk_parallel"] = plan.chunk_parallel
        assert plan.out is not None
        # Every host-side decision is made; from here on only device work is
        # issued: the packed-input copies, then the single launcher call that
        # runs the preprocess and the scan.
        with torch.cuda.device(x.device):
            for destination, source in plan.pending_copies:
                destination.copy_(source)
            _launch_program(
                plan.program_name,
                plan.arch,
                preprocess=plan.preprocess,
                preprocess_grid=plan.preprocess_grid,
                main=plan.main,
                main_grid=plan.grid,
                cuda_stream=int(torch.cuda.current_stream(x.device).cuda_stream),
            )
        return plan.out, plan.final

    def prepare(
        self,
        *,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        dt_softplus: bool = False,
        dt_limit: Tuple[float, float] = (0.0, float("inf")),
        initial_states: Optional[torch.Tensor] = None,
        seq_idx: Optional[torch.Tensor] = None,
        chunk_indices: Optional[torch.Tensor] = None,
        chunk_offsets: Optional[torch.Tensor] = None,
        seq_chunk_cumsum: Optional[torch.Tensor] = None,
        return_final_states: bool = True,
        num_seqs: Optional[int] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
    ) -> PreparedSSDCombined:
        """Prepare a call shape eagerly so that :meth:`run` on it is a pure
        launch, in particular inside CUDA-graph capture.

        Performs every host-side decision and device-side preparation the
        same :meth:`run` would perform, without launching a kernel: the
        validation, the workspace of the shape (including the chunk-parallel
        buffers when that family is selected), the packed copies of strided
        inputs (prepare with the same strides the captured call will bring),
        the dummy operands, the program build and load.  It reads
        ``FLASHINFER_CAKE_SSD_CHUNK_PARALLEL`` once and freezes the family
        choice for the shape: later :meth:`run` calls on it never consult the
        environment.  With ``return_final_states`` it allocates the shape's
        static final-states buffer, which such :meth:`run` calls then return
        (the same tensor every call, overwritten by each launch) instead of
        fresh storage.  The prepared shape's workspace is pinned: never
        evicted for the lifetime of the runner.  A caller-owned
        ``seq_chunk_cumsum`` given here is read as :meth:`run` reads it by
        default: in the ``seq_idx`` triple form the trusted table of the call
        (``update_seq_chunk_cumsum=False``, the fused program where its
        shape window admits it), in the ``cu_seqlens`` form the buffer the
        preprocess publishes the derived table into (that form's only
        reading, ``update_seq_chunk_cumsum=True``).  A later :meth:`run` of
        the triple form that asks the preprocess to rewrite the table
        (``update_seq_chunk_cumsum=True``) is another call shape -- its
        workspace key differs -- and is prepared by running it eagerly once.
        The checkpoint outputs are per-call options of :meth:`run`.  The
        kernel handles of the host shim resolve
        on the program's first launch, so run the shape eagerly once before
        capturing it.  A static ``out`` is not required here; a captured
        :meth:`run` without one allocates its output from the graph pool,
        with one it allocates nothing.
        """

        if _stream_capturing(x.device):
            raise CakeSSDCombinedCaptureError(
                "Cake SSDCombined prepare() allocates and may build programs: call "
                "it eagerly, outside CUDA-graph capture"
            )
        # The cu_seqlens form only publishes into a caller buffer; the triple
        # form's caller table is trusted (run's default), which keeps
        # prepare's workspace key the key of that run.
        publishes_into_buffer = cu_seqlens is not None and seq_chunk_cumsum is not None
        plan = self._plan(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
            initial_states=initial_states,
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            seq_chunk_cumsum=seq_chunk_cumsum,
            update_seq_chunk_cumsum=publishes_into_buffer,
            checkpoint_token_indices=None,
            checkpoint_state_slots=None,
            checkpoint_states=None,
            out=out,
            return_final_states=return_final_states,
            num_seqs=num_seqs,
            cu_seqlens=cu_seqlens,
            capturing=False,
            preparing=True,
        )
        _load_generated_program(plan.program_name, plan.arch)
        # Only a successful preparation freezes the family choice and pins
        # the workspace: a failed allocation or build leaves no trace.
        shape: _ShapeKey = tuple(plan.workspace_key[:-1])
        self._prepared_modes[shape] = plan.workspace_key.chunk_parallel_mode
        self._workspaces.hold(plan.workspace_key)
        return PreparedSSDCombined(
            program_name=plan.program_name,
            workspace_key=plan.workspace_key,
            grid=plan.grid,
            preprocess_grid=plan.preprocess_grid,
        )

    def is_captured(self, prepared: PreparedSSDCombined) -> bool:
        """Whether a CUDA graph has been captured with ``prepared``'s call
        shape on this runner (its workspace is then pinned as captured)."""

        return self._workspaces.captured(prepared.workspace_key)

    def _plan(
        self,
        *,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
        dt_softplus: bool,
        dt_limit: Tuple[float, float],
        initial_states: Optional[torch.Tensor],
        seq_idx: Optional[torch.Tensor],
        chunk_indices: Optional[torch.Tensor],
        chunk_offsets: Optional[torch.Tensor],
        seq_chunk_cumsum: Optional[torch.Tensor],
        update_seq_chunk_cumsum: bool,
        checkpoint_token_indices: Optional[torch.Tensor],
        checkpoint_state_slots: Optional[torch.Tensor],
        checkpoint_states: Optional[torch.Tensor],
        out: Optional[torch.Tensor],
        return_final_states: bool,
        num_seqs: Optional[int],
        cu_seqlens: Optional[torch.Tensor],
        capturing: bool,
        preparing: bool,
    ) -> _LaunchPlan:
        """Validate one call and resolve everything it binds (``_LaunchPlan``).

        ``capturing``: the call records into a CUDA graph, so nothing may be
        allocated, built or read from the environment (named refusals).
        ``preparing``: :meth:`prepare` is materialising the shape -- the
        family override is read and frozen, a static final-states buffer is
        allocated on request and a missing ``out`` leaves the output unbound.
        """

        self._allocations_in_last_run = 0
        batch, seqlen, nheads, headdim = x.shape
        if batch <= 0 or seqlen <= 0:
            raise ValueError(
                "x must have a positive batch and sequence length "
                f"(got batch={batch}, seqlen={seqlen})"
            )
        if (nheads, headdim) != (self.nheads, _HEADDIM):
            raise ValueError(f"x must have shape [batch, seqlen, {self.nheads}, 64]")
        if tuple(B.shape) != (batch, seqlen, self.ngroups, _DSTATE):
            raise ValueError(f"B must have shape [batch, seqlen, {self.ngroups}, 128]")
        if C.shape != B.shape:
            raise ValueError("C must have the same shape as B")
        if x.dtype != torch.bfloat16 or B.dtype != x.dtype or C.dtype != x.dtype:
            raise ValueError("x, B, and C must be bfloat16")
        if tuple(dt.shape) != (batch, seqlen, self.nheads):
            raise ValueError(f"dt must have shape [batch, seqlen, {self.nheads}]")
        if tuple(A.shape) != (self.nheads,):
            raise ValueError(f"A must have shape [{self.nheads}]")
        if dt.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("dt must be bfloat16 or float32")
        if A.dtype != torch.float32:
            raise ValueError("A must be float32")
        if (D is not None) != self.has_d or (z is not None) != self.has_z:
            raise ValueError("runtime D/z presence must match the constructor")
        if (initial_states is not None) != self.has_initial_states:
            raise ValueError(
                "runtime initial_states presence must match the constructor"
            )
        mode_varlen = self.has_varlen
        metadata = (seq_idx, chunk_indices, chunk_offsets)
        from_cu_seqlens = cu_seqlens is not None
        if not mode_varlen and (
            any(value is not None for value in metadata)
            or seq_chunk_cumsum is not None
            or num_seqs is not None
            or from_cu_seqlens
        ):
            raise ValueError(
                "batched mode does not accept varlen metadata, seq_chunk_cumsum, "
                "num_seqs, or cu_seqlens"
            )
        if mode_varlen and from_cu_seqlens:
            # cu_seqlens form: the device derives the segment tables; the
            # caller's chunk-128 triple is not consulted (seq_idx may ride
            # along for callers that still build it; it is never read).
            if chunk_indices is not None or chunk_offsets is not None:
                raise ValueError(
                    "cu_seqlens excludes chunk_indices / chunk_offsets: the "
                    "preprocess derives the segment tables from cu_seqlens"
                )
            if (
                cu_seqlens.dtype != torch.int32
                or cu_seqlens.ndim != 1
                or cu_seqlens.numel() < 2
            ):
                raise ValueError(
                    "cu_seqlens must be an int32 vector of num_seqs + 1 entries"
                )
            if batch != 1:
                raise ValueError("cu_seqlens describes a packed [1, total] stream")
            if seq_chunk_cumsum is not None and not update_seq_chunk_cumsum:
                raise ValueError(
                    "with cu_seqlens the preprocess derives seq_chunk_cumsum; pass "
                    "update_seq_chunk_cumsum=True to receive it in your buffer or "
                    "omit it"
                )
        elif mode_varlen:
            if self.chunk_size != _CHUNK_SIZE:
                raise ValueError(
                    f"chunk_size={self.chunk_size}: the seq_idx / chunk_indices / "
                    "chunk_offsets triple is the chunk-128 logical segmentation; "
                    "pass cu_seqlens instead (the kernels derive the segments)"
                )
            if any(value is None for value in metadata):
                raise ValueError(
                    "varlen mode requires seq_idx, chunk_indices, and chunk_offsets "
                    "(or cu_seqlens)"
                )
        if initial_states is not None and initial_states.dtype != self.state_dtype:
            raise ValueError("initial_states dtype must match state_dtype")
        # seq_idx triple form with a caller ``seq_chunk_cumsum`` the
        # preprocess does not rewrite: the kernels read the caller's tables
        # and nothing is published, so the fused program is one launch there
        # (part of the workspace key: it selects the program).
        caller_cumsum = (
            mode_varlen
            and not from_cu_seqlens
            and seq_chunk_cumsum is not None
            and not update_seq_chunk_cumsum
        )
        # seq_idx triple form publishing the table into the caller's buffer:
        # a publishing stage must write it, so the one-launch program (whose
        # table stays private to each CTA) is not an option for this shape.
        publish_table = (
            mode_varlen
            and not from_cu_seqlens
            and seq_chunk_cumsum is not None
            and bool(update_seq_chunk_cumsum)
        )

        nchunks = -(-seqlen // _CHUNK_SIZE)
        if not mode_varlen:
            num_sequences = batch
        else:
            # Every given source of the packed sequence count must agree;
            # cu_seqlens is a host shape (no synchronisation).
            counts: list[tuple[str, int]] = []
            if from_cu_seqlens:
                counts.append(("cu_seqlens", int(cu_seqlens.numel()) - 1))
            if initial_states is not None:
                counts.append(("initial_states", int(initial_states.shape[0])))
            if not counts and seq_chunk_cumsum is not None:
                # The caller's vector names the count only when nothing else
                # does; otherwise its shape is validated against the count.
                counts.append(("seq_chunk_cumsum", int(seq_chunk_cumsum.numel()) - 1))
            if num_seqs is not None:
                counts.append(("num_seqs", int(num_seqs)))
            if not counts:
                raise ValueError(
                    "varlen mode without initial_states requires seq_chunk_cumsum or "
                    "num_seqs to determine the sequence count"
                )
            source, num_sequences = counts[0]
            for name, value in counts[1:]:
                if value != num_sequences:
                    raise ValueError(
                        f"{name} ({value}) does not match the sequence count "
                        f"({num_sequences}) implied by {source}"
                    )
        if num_sequences <= 0:
            raise ValueError("the sequence count must be positive")
        if from_cu_seqlens:
            num_segments = _segment_bound(seqlen, num_sequences)
        else:
            num_segments = len(chunk_indices) if mode_varlen else batch * nchunks
        dt_min, dt_max = (float(value) for value in dt_limit)
        checkpoint_args = (
            checkpoint_token_indices,
            checkpoint_state_slots,
            checkpoint_states,
        )
        if any(value is not None for value in checkpoint_args) and not all(
            value is not None for value in checkpoint_args
        ):
            raise ValueError(
                "checkpoint_token_indices, checkpoint_state_slots, and "
                "checkpoint_states must be provided together"
            )
        checkpoint_state_count = 0
        if checkpoint_states is not None:
            assert checkpoint_token_indices is not None
            assert checkpoint_state_slots is not None
            assert checkpoint_states is not None
            if (
                tuple(checkpoint_token_indices.shape) != (num_sequences,)
                or checkpoint_token_indices.dtype != torch.int32
                or not checkpoint_token_indices.is_contiguous()
            ):
                raise ValueError(
                    "checkpoint_token_indices must be a contiguous int32 vector "
                    "with one entry per sequence"
                )
            if (
                tuple(checkpoint_state_slots.shape) != (num_sequences,)
                or checkpoint_state_slots.dtype != torch.int32
                or not checkpoint_state_slots.is_contiguous()
            ):
                raise ValueError(
                    "checkpoint_state_slots must be a contiguous int32 vector "
                    "with one entry per sequence"
                )
            if (
                checkpoint_states.ndim != 4
                or tuple(checkpoint_states.shape[1:])
                != (self.nheads, _HEADDIM, _DSTATE)
                or checkpoint_states.dtype != self.state_dtype
                or not checkpoint_states.is_contiguous()
            ):
                raise ValueError(
                    "checkpoint_states must be contiguous [num_checkpoints, "
                    f"{self.nheads}, 64, 128] with state dtype"
                )
            checkpoint_state_count = int(checkpoint_states.shape[0])
        if D is not None:
            valid_d_shapes = ((self.nheads,), (self.nheads, _HEADDIM))
            if tuple(D.shape) not in valid_d_shapes or D.dtype != torch.bfloat16:
                raise ValueError(
                    f"D must have shape [{self.nheads}] or "
                    f"[{self.nheads}, 64] and dtype bfloat16"
                )
        if z is not None and (z.shape != x.shape or z.dtype != torch.bfloat16):
            raise ValueError("z must have the same shape and dtype as x")
        if initial_states is not None:
            expected_states = (
                num_sequences,
                self.nheads,
                _HEADDIM,
                _DSTATE,
            )
            if tuple(initial_states.shape) != expected_states:
                raise ValueError(f"initial_states must have shape {expected_states}")
        if seq_idx is not None:
            if (
                tuple(seq_idx.shape) != (batch, seqlen)
                or seq_idx.dtype != self.seq_idx_dtype
            ):
                raise ValueError(
                    "seq_idx shape or dtype does not match the constructor"
                )
        if chunk_indices is not None:
            if (
                chunk_indices.dtype != torch.int32
                or chunk_offsets.dtype != torch.int32
                or chunk_indices.ndim != 1
                or chunk_offsets.shape != chunk_indices.shape
            ):
                raise ValueError(
                    "chunk_indices/chunk_offsets must be matching int32 vectors"
                )
        if seq_chunk_cumsum is not None and (
            tuple(seq_chunk_cumsum.shape) != (num_sequences + 1,)
            or seq_chunk_cumsum.dtype != torch.int32
            or not seq_chunk_cumsum.is_contiguous()
        ):
            raise ValueError(
                "seq_chunk_cumsum shape or dtype is invalid: expected a contiguous "
                f"int32 vector of {num_sequences + 1} entries"
            )
        if dt_bias is not None and (
            tuple(dt_bias.shape) != (self.nheads,)
            or dt_bias.dtype not in (torch.bfloat16, torch.float32)
        ):
            raise ValueError(
                f"dt_bias must have shape [{self.nheads}] and dtype bfloat16 or float32"
            )
        # The kernels write the token-major output directly (TMA store per
        # chunk; vectorised stores for packed-varlen segments that share a
        # physical chunk), so the public layout is also the kernel layout.
        expected_out = (batch, seqlen, self.nheads, _HEADDIM)
        if out is not None:
            if tuple(out.shape) != expected_out or out.dtype != torch.bfloat16:
                raise ValueError(
                    f"out must have shape {expected_out} and dtype bfloat16"
                )
            if not out.is_contiguous():
                raise ValueError("out must be contiguous")
        elif not preparing:
            # Match SSDCombined's ownership contract: each allocation-returning
            # call owns fresh output storage that later calls cannot overwrite.
            self._allocating("the output")
            out = torch.empty(expected_out, dtype=torch.bfloat16, device=x.device)
        # The kernel family is chosen with the workspace: it depends on the
        # workspace key, the device (SM count, capability) and the family
        # override.  ``prepare`` reads the override once and freezes it for
        # the shape; an unprepared call (eager or captured) reads it per call.
        device_index = _cuda_device_index(x)
        arch = _target_arch(device_index)
        sm_count = _sm_count(device_index)
        shape: _ShapeKey = (
            x.device.index,
            batch,
            seqlen,
            nchunks,
            num_segments,
            num_sequences,
            self.state_dtype,
            from_cu_seqlens,
            caller_cumsum,
            publish_table,
        )
        if preparing:
            mode = _chunk_parallel_mode()
        else:
            mode = self._prepared_modes.get(shape)
            if mode is None:
                mode = _chunk_parallel_mode()
        key = _WorkspaceKey(*shape, mode)
        if capturing and key not in self._workspaces:
            # Materialising a workspace records its table-building kernels
            # (segment ids, zero vectors) into the graph instead of running
            # them: the tables would stay uninitialised until the first
            # replay while the entry is pinned.
            raise CakeSSDCombinedCaptureError(
                "Cake SSDCombined call shape has not run on this runner: run it "
                "eagerly once (or prepare() it) before capturing it into a CUDA "
                "graph"
            )
        workspace = self._get_workspace(
            key=key,
            device=x.device,
            sm_count=sm_count,
            capability=_ARCH_CAPABILITIES[arch],
        )
        program_name = workspace["program"]
        program = _PROGRAMS[program_name]
        if capturing and (program_name, arch) not in _LOADED_PROGRAMS:
            # Loading (possibly building) the program is host work a graph
            # cannot record, and the host shim resolves its kernel handles
            # on the first launch.
            raise CakeSSDCombinedCaptureError(
                f"Cake SSDCombined program {program_name} ({arch}) has not been "
                "launched in this process: run the shape eagerly once before "
                "capturing it"
            )
        # The kernel needs valid storage even when final states are disabled.
        # Returned final states live in fresh caller-owned storage per eager
        # call (repeated cached-runner calls do not alias and no post-kernel
        # device copy adds another GPU activity to the measured route) unless
        # the shape was prepared with return_final_states: then the static
        # buffer ``prepare`` allocated is bound (the address a captured graph
        # replays into); a captured call without it allocates them from the
        # graph pool.
        if not return_final_states:
            final_states_arg = workspace["final"]
        else:
            final_states_arg = workspace.get("final_static")
            if final_states_arg is None:
                self._allocating("the final states")
                final_states_arg = torch.empty_like(workspace["final"])
                if preparing:
                    workspace["final_static"] = final_states_arg
        # The TMA descriptors preserve the physical strides of x/B/C,
        # including the row padding produced by framework projection splits.
        # Only inputs consumed through flat pointer indexing need packed,
        # graph-stable storage.  Resolve the storage now; copy later, together.
        pending_copies: list[tuple[torch.Tensor, torch.Tensor]] = []

        def packed(
            name: str, value: Optional[torch.Tensor], align: int = 1
        ) -> Optional[torch.Tensor]:
            buffer, source = self._contiguous_input(workspace, name, value, align)
            if source is not None:
                assert buffer is not None
                pending_copies.append((buffer, source))
            return buffer

        dt = packed("dt", dt)
        A = packed("A", A)
        D = packed("D", D)
        z = packed("z", z)
        dt_bias = packed("dt_bias", dt_bias)
        # The ``_ibulk`` programs read an item's initial state with one bulk
        # copy, which needs a 16-byte aligned address.
        initial_states = packed(
            "initial_states",
            initial_states,
            16 if program.variant == "ibulk" else 1,
        )
        seq_idx = packed("seq_idx", seq_idx)
        chunk_indices = packed("chunk_indices", chunk_indices)
        chunk_offsets = packed("chunk_offsets", chunk_offsets)
        cu_seqlens = packed("cu_seqlens", cu_seqlens)
        checkpoint_token_indices = packed(
            "checkpoint_token_indices", checkpoint_token_indices
        )
        checkpoint_state_slots = packed(
            "checkpoint_state_slots", checkpoint_state_slots
        )
        # A call shorter than one chunk (``seqlen < 128``) binds the caller's
        # x/B/C/out directly: the token axis of the four tensor maps admits a
        # global extent below the 128-row box (``allow_oob_box`` in the
        # exported programs), TMA zero-fills the rows past ``seqlen`` on
        # loads and clips them on stores.  Every argument stays logical
        # (``seqlen``, ``nchunks = 1``, dt, z, the preprocess tables, the
        # sequence metadata), so the kernel sees exactly the state of a
        # partial trailing chunk of a longer call; the pad rows contribute
        # exact zeros and the final states are unchanged.
        assert x is not None and dt is not None and A is not None
        assert B is not None and C is not None
        seq_idx_int64 = seq_idx is not None and seq_idx.dtype == torch.int64
        seq_i32 = (
            seq_idx
            if seq_idx is not None and not seq_idx_int64
            else self._dummy(x.device, torch.int32)
        )
        seq_i64 = (
            seq_idx
            if seq_idx is not None and seq_idx_int64
            else self._dummy(x.device, torch.int64)
        )
        if from_cu_seqlens:
            chunk_indices_arg = workspace["chunk_indices"]
            chunk_offsets_arg = workspace["chunk_offsets"]
        else:
            chunk_indices_arg = (
                chunk_indices
                if chunk_indices is not None
                else self._dummy(x.device, torch.int32)
            )
            chunk_offsets_arg = (
                chunk_offsets
                if chunk_offsets is not None
                else self._dummy(x.device, torch.int32)
            )
        cu_seqlens_arg = (
            cu_seqlens if from_cu_seqlens else self._dummy(x.device, torch.int32)
        )
        # Packed varlen: the preprocess writes seq_chunk_cumsum (into the
        # caller's buffer when update_seq_chunk_cumsum is set, else into the
        # runner-owned one) unless the caller passed a precomputed vector.
        # The cu_seqlens derivation always publishes it (the seq_idx
        # publisher, write_seq_chunk_cumsum, is the triple form's).
        if not mode_varlen:
            write_seq_chunk_cumsum = False
            cumsum_arg = self._dummy(x.device, torch.int32)
        elif from_cu_seqlens:
            write_seq_chunk_cumsum = False
            cumsum_arg = (
                seq_chunk_cumsum
                if seq_chunk_cumsum is not None
                else workspace["seq_chunk_cumsum"]
            )
        elif seq_chunk_cumsum is None:
            # The runner's vector: the stage publishes it -- except for the
            # one-launch program, whose main kernel derives its per-CTA rows
            # of this scratch itself (nothing is published).
            write_seq_chunk_cumsum = program.family != "exactone"
            cumsum_arg = workspace["seq_chunk_cumsum"]
        else:
            write_seq_chunk_cumsum = bool(update_seq_chunk_cumsum)
            cumsum_arg = seq_chunk_cumsum
        checkpoint_token_indices_arg = (
            checkpoint_token_indices
            if checkpoint_token_indices is not None
            else self._dummy(x.device, torch.int32)
        )

        dt_float = dt
        if dt.dtype != torch.float32:
            dt_float = workspace["dt_float"]
            pending_copies.append((dt_float, dt))
        if dt_bias is None:
            dt_bias_float = workspace["dt_bias_zero"]
        elif dt_bias.dtype == torch.float32:
            dt_bias_float = dt_bias
        else:
            dt_bias_float = workspace["dt_bias_float"]
            pending_copies.append((dt_bias_float, dt_bias))
        preprocess_status = self._preprocess_status_buffer(x.device)
        # The sm_107a preprocess loads the dt rows of an aligned segment table
        # before it reads the table, bounded by its ``seqlen`` argument; the
        # batched form uses ``seqlen`` for nothing else, so there it is the
        # token count of all rows.
        preprocess_seqlen = seqlen
        if arch == _SM107_ARCH and not mode_varlen:
            preprocess_seqlen = batch * seqlen
        preprocess_values, preprocess_grid = _direct_preprocess_inputs(
            dt=dt_float,
            A=A,
            dt_bias=dt_bias_float,
            segment_starts=workspace["starts"],
            segment_lengths=workspace["lengths"],
            chunk_indices=chunk_indices_arg,
            chunk_offsets=chunk_offsets_arg,
            delta=workspace["delta"],
            cumsum=workspace["cumsum"],
            num_segments=num_segments,
            nheads=self.nheads,
            seqlen=preprocess_seqlen,
            mode_varlen=mode_varlen,
            dt_softplus=bool(dt_softplus),
            dt_limit=(dt_min, dt_max),
            tiles_per_block=_SEGMENT_PREPROCESS_TILES_PER_BLOCK,
            seq_idx_i32=seq_i32,
            seq_idx_i64=seq_i64,
            seq_idx_int64=seq_idx_int64,
            seq_chunk_cumsum=cumsum_arg,
            num_sequences=num_sequences,
            write_seq_chunk_cumsum=write_seq_chunk_cumsum,
            cu_seqlens=cu_seqlens_arg,
            checkpoint_token_indices=checkpoint_token_indices_arg,
            metadata_from_cu_seqlens=from_cu_seqlens,
            checkpoint_state_count=checkpoint_state_count,
            preprocess_status=preprocess_status,
        )
        if program.family == "exactmeta":
            # The light metadata stage: one thread per logical segment
            # publishes ``seq_chunk_cumsum`` (into the runner's vector, or
            # the caller's buffer with ``update_seq_chunk_cumsum``) and the
            # segment tables nothing on this path reads, from the caller's
            # triple; the fused main kernel computes delta / cumsum in-CTA
            # and reads the published table like the two-kernel main kernel.
            preprocess_values, preprocess_grid = _metadata_preprocess_inputs(
                seq_idx_i32=seq_i32,
                seq_idx_i64=seq_i64,
                chunk_indices=chunk_indices_arg,
                chunk_offsets=chunk_offsets_arg,
                segment_starts=workspace["starts"],
                segment_lengths=workspace["lengths"],
                sequence_offsets=cumsum_arg,
                num_segments=num_segments,
                num_sequences=num_sequences,
                seqlen=seqlen,
                seq_idx_int64=seq_idx_int64,
                preprocess_status=preprocess_status,
                threads=_PACKED_METADATA.threads,
            )
        if program.family == "exactfused":
            # The fused program computes delta / cumsum in-CTA and, in the
            # cu_seqlens form, derives the segment tables itself (it reads
            # ``cu_seqlens`` and flags an invalid one in ``preprocess_status``),
            # so its preprocess stage is skipped -- a grid of 0 -- except where
            # the preprocess publishes packed-varlen metadata the fused kernel
            # does not: the seq_idx triple form deriving ``seq_chunk_cumsum``
            # (runner-owned or a caller buffer with
            # ``update_seq_chunk_cumsum``), or a caller buffer for the
            # cu_seqlens-derived vector.
            publishes_metadata = write_seq_chunk_cumsum or (
                from_cu_seqlens and seq_chunk_cumsum is not None
            )
            if not publishes_metadata:
                preprocess_grid = (0, 1, 1)
        if program.family in ("chunkpar", "chunkpar3"):
            # One CTA per (chunk, head) tile, at most one per SM (SMEM, TMEM
            # and registers allow one CTA per SM), so every CTA is co-resident
            # for the grid barrier.  The tile bound counts workspace rows the
            # cu_seqlens form may never address; the kernel reads the real
            # tile count from ``seq_chunk_cumsum[sequence_count]``.
            grid = max(1, min(num_segments * self.nheads, sm_count))
        elif program.variant in ("tail", "tailone"):
            # One 2-CTA cluster (``_tailone``: one CTA pair) per SM pair; the
            # pair plan is computed in-kernel from this grid (``tail_selected``
            # chose the program).
            grid = _tail_grid_size(sm_count=sm_count)
        else:
            grid = _persistent_grid_size(
                total_work=num_sequences * self.nheads, sm_count=sm_count
            )
        if program.family == "exactone":
            # One launch: no stage runs (grid 0).  The main kernel's CTA
            # prologue derives this CTA's sequence table from the seq_idx
            # triple into row ``blockIdx.x`` of the runner's scratch (the
            # workspace sized it ``grid x (num_sequences + 1)`` for this
            # shape) and flags an invalid id in the status word exactly as
            # the publishing stages do.
            preprocess_grid = (0, 1, 1)
            assert cumsum_arg.numel() == grid * (num_sequences + 1), (
                cumsum_arg.numel(),
                grid,
                num_sequences,
            )

        d_arg = D if D is not None else self._dummy(x.device, torch.bfloat16)
        if D is not None and D.ndim == 2 and not self.d_has_hdim:
            # Match CuTe's public runner: a 2D D passed to a per-head
            # constructor consumes its first column.
            d_arg = workspace["d_head"]
            pending_copies.append((d_arg, D[:, 0]))
        z_arg = z if z is not None else self._dummy(x.device, torch.bfloat16)
        # The scan epilogue reads z and D in 16-byte groups (CAKE-991). Check
        # the tensors that are bound: packed copies and workspace buffers are
        # aligned by construction, a contiguous view at an odd offset is not.
        for name, bound, given in (("D", d_arg, D), ("z", z_arg, z)):
            if given is not None and bound.data_ptr() % 16 != 0:
                raise ValueError(f"{name} must be 16-byte aligned")
        initial_arg = (
            initial_states
            if initial_states is not None
            else self._dummy(x.device, self.state_dtype)
        )
        checkpoint_states_arg = (
            checkpoint_states
            if checkpoint_states is not None
            else self._dummy(x.device, self.state_dtype)
        )
        checkpoint_state_slots_arg = (
            checkpoint_state_slots
            if checkpoint_state_slots is not None
            else self._dummy(x.device, torch.int32)
        )
        d_mode = 0 if D is None else 2 if self.d_has_hdim and D.ndim == 2 else 1
        # x/B/C are consumed through their stride-aware TMA descriptors.
        # The kernel signature retains unused raw-pointer slots for the
        # same buffers; pass a valid packed dummy so the host checks do
        # not reject the descriptor-compatible public views.
        unused_bf16 = self._dummy(x.device, torch.bfloat16)
        main_values: dict[str, object] = {
            "x_map": x,
            "b_map": B,
            "c_map": C,
            "out_map": out,
            "x": unused_bf16,
            "dt": dt_float,
            "delta_precomputed": workspace["delta"],
            "cumsum_precomputed": workspace["cumsum"],
            "A": A,
            "B_tensor": unused_bf16,
            "C": unused_bf16,
            "D": d_arg,
            "z": z_arg,
            "dt_bias": dt_bias_float,
            "initial_states": initial_arg,
            "final_states": final_states_arg,
            "checkpoint_states": checkpoint_states_arg,
            "checkpoint_token_indices": checkpoint_token_indices_arg,
            "checkpoint_state_slots": checkpoint_state_slots_arg,
            "seq_idx_i32": seq_i32,
            "seq_idx_i64": seq_i64,
            "chunk_indices": chunk_indices_arg,
            "chunk_offsets": chunk_offsets_arg,
            "seq_chunk_cumsum": cumsum_arg,
            "out_native": out,
            "nheads": self.nheads,
            "ngroups": self.ngroups,
            "batch": batch,
            "seqlen": seqlen,
            "nchunks": nchunks,
            "sequence_count": num_sequences,
            # cu_seqlens: the bound; the scan reads at most one entry past the
            # last derived segment, the sentinel.
            "num_logical_chunks": num_segments if mode_varlen else nchunks,
            "mode_varlen": int(mode_varlen),
            "D_mode": d_mode,
            "has_z": int(z is not None),
            "has_initial": int(initial_states is not None),
            "dt_softplus": int(bool(dt_softplus)),
            "dt_min": dt_min,
            "dt_max": dt_max,
            "write_final_states": int(return_final_states),
            "checkpoint_state_count": checkpoint_state_count,
            # The fused and exactpre programs bind these (``_MAIN_ARGS``; the
            # retained start program's 41-argument launcher stops before
            # them); only the fused program reads them: it derives the cu_seqlens segment
            # tables in-CTA and flags an invalid cu_seqlens in the runner's
            # status word.  The two-kernel and chunk-parallel programs read
            # the tables their preprocess published.
            "cu_seqlens": cu_seqlens_arg,
            "metadata_from_cu_seqlens": (
                (_TABLE_IN_KERNEL_I64 if seq_idx_int64 else _TABLE_IN_KERNEL_I32)
                if program.family == "exactone"
                else _TABLE_CU_SEQLENS
                if program.family == "exactfused" and from_cu_seqlens
                else _TABLE_PUBLISHED
            ),
            "preprocess_status": preprocess_status,
        }
        if program.family in ("chunkpar", "chunkpar3"):
            # Neither chunk-parallel signature (``_MAIN_ARGS_CHUNKPAR``,
            # ``_MAIN_ARGS_CHUNKPAR3``) has a fused input: bind exactly the
            # family's signature.
            for name in _FUSED_ONLY_MAIN_ARGS:
                del main_values[name]
        chunk_parallel: Optional[dict[str, torch.Tensor]] = None
        if program.family == "chunkpar":
            # The state operand workspace is bound twice: as the ``h_map``
            # tensor map (the host builds the TMA descriptor from the bf16
            # [tiles, 64, 128] tensor) and as the u32 word view the phase-2
            # chains store through.  A captured workspace binds the buffers
            # its graph recorded (pinned by ``run``); every other call binds
            # the device's current grow-only buffers.  V is the capability's
            # table value capped at this device's SM count (``_sm_count`` is
            # fixed per device index, so a captured call replays its V).
            vcount = _chunk_parallel_vcount(_ARCH_CAPABILITIES[arch], sm_count)
            chunk_parallel = workspace.get("chunk_parallel")
            if chunk_parallel is None:
                chunk_parallel = self._chunk_parallel_workspace(
                    x.device, num_segments * self.nheads, vcount
                )
            main_values.update(
                h_map=chunk_parallel["h_work"],
                s_work=chunk_parallel["s_work"],
                agg_work=chunk_parallel["agg_work"],
                h_words=chunk_parallel["h_words"],
                grid_barrier=chunk_parallel["grid_barrier"],
                vcount=vcount,
            )
        elif program.family == "chunkpar3":
            # The three-phase program binds the same S/H buffers (grow-only,
            # per device; pinned per captured workspace like the look-back's)
            # and its own two-word grid barrier.
            chunk_parallel = workspace.get("chunk_parallel")
            if chunk_parallel is None:
                chunk_parallel = self._chunk_parallel_workspace(
                    x.device,
                    num_segments * self.nheads,
                    _chunk_parallel_vcount(_ARCH_CAPABILITIES[arch], sm_count),
                )
            main_values.update(
                h_map=chunk_parallel["h_work"],
                s_work=chunk_parallel["s_work"],
                h_words=chunk_parallel["h_words"],
                grid_barrier=chunk_parallel["grid_barrier3"],
            )
        return _LaunchPlan(
            program_name=program_name,
            arch=arch,
            workspace_key=key,
            workspace=workspace,
            preprocess=preprocess_values,
            preprocess_grid=preprocess_grid,
            main=main_values,
            grid=(grid, 1, 1),
            pending_copies=pending_copies,
            out=out,
            final=final_states_arg if return_final_states else None,
            chunk_parallel=chunk_parallel,
        )


__all__ = ["CakeSSDCombined", "CakeSSDCombinedCaptureError", "PreparedSSDCombined"]
