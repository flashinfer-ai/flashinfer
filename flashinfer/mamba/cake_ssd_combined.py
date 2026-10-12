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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Optional, Tuple

import torch
from filelock import FileLock
from tvm_ffi import cpp

from ..jit import env as jit_env

_CHUNK_SIZE = 128
_HEADDIM = 64
_DSTATE = 128

_TARGET_ARCHS = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
_ARCH_CAPABILITIES = {arch: capability for capability, arch in _TARGET_ARCHS.items()}
_DEVICE_DIR = Path("generated") / "device"
# Prepared two-launch host launchers, one per kernel family: the same nine
# ``CAKE_SSD_*`` placeholders and namespace scheme; the look-back chunk-parallel
# main kernel takes six more arguments than the exact scan
# (``_MAIN_ARGS_CHUNKPAR``), the three-phase chunk-parallel main kernel four
# more (``_MAIN_ARGS_CHUNKPAR3``).
_HOST_TEMPLATE = Path("generated") / "host" / "mamba_ssd_combined_sequence.cpp"
_CHUNKPAR_HOST_TEMPLATE = (
    Path("generated") / "host" / "mamba_ssd_combined_chunk_parallel_sequence.cpp"
)
# The two-kernel exact-scan programs (``exactpre_*``, the start pre+scan form)
# bind the same main arguments as the fused programs and always run their
# preprocess stage.
_EXACTPRE_HOST_TEMPLATE = (
    Path("generated") / "host" / "mamba_ssd_combined_precomputed_sequence.cpp"
)
_CHUNKPAR3_HOST_TEMPLATE = (
    Path("generated") / "host" / "mamba_ssd_combined_chunk_parallel3_sequence.cpp"
)
# Module identities the Cake export has not filled yet carry this token; such
# a program refuses to build (``_require_exported``).
_PENDING_EXPORT = "PENDINGEXPORT"


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

    ``family`` is ``exact`` (the fused persistent scan, one CTA per
    (sequence, head) item, delta / cumsum computed in-CTA; its preprocess
    stage is skipped with a grid of 0 unless it publishes packed-varlen
    metadata), ``exactpre`` (the same persistent scan reading the delta /
    cumsum its preprocess wrote to the HBM workspace: the start two-launch
    form, same main arguments), ``chunkpar`` (the look-back chunk-parallel
    program, one CTA per SM over item-major tile ranges) or ``chunkpar3``
    (the three-phase chunk-parallel program, one CTA per (chunk, head)
    tile); it fixes the host template and the main kernel's argument order.
    """

    family: str
    preprocess: _Kernel
    main: _Kernel
    state_dtype_code: int
    state_dtype_bits: int
    main_smem_bytes: int
    host_template: Path
    main_args: tuple[str, ...]

    @property
    def kernels(self) -> tuple[_Kernel, _Kernel]:
        return (self.preprocess, self.main)


# Generated-source identities, ``generated/device/<module>.cu``: the Cake
# kernel symbol followed by the export's module identity hash.  This block is
# the single place the Cake export refreshes; every program binding below
# derives from it.  Four kernel families ship, each x {bf16, f16, f32 state}
# x {batched, varlen}, plus one preprocess shared by all: the fused exact scan
# (``exact_*``, CAKE-956 round 6 item 2: delta / cumsum computed in-CTA, one
# launch), the two-kernel exact scan (``exactpre_*``: the start pre+scan form
# the fused route keeps for saturated / long-item calls and for the seq_idx
# triple form, see :func:`fused_selected`), the look-back chunk-parallel
# program (``chunkpar_*``, round 6 item 1a) and the three-phase chunk-parallel
# program (``chunkpar3_*``, the three-phase program kept as the second
# chunk-parallel family).  Both exact-scan families ship two schedule
# variants of every program: the ring program (``<family>_<state>_<mode>``)
# and the ``_noring`` program (the same kernel built with
# ``epilogue_ring=False``, bitwise-identical outputs), selected per call by
# :func:`_exact_scan_variant` from host-known shape quantities.
_SEGMENT_PREPROCESS_MODULE = "factorized_persistent_segment_preprocess_08c5aea4d0"
_SCAN_MODULES = {
    "exact_bf16_batched": "mamba_ssd_q_tmem_alias_bf16_batched_cf0cc7f97b",
    "exact_f16_batched": "mamba_ssd_q_tmem_alias_f16_batched_e2e32e4f62",
    "exact_f32_batched": "mamba_ssd_q_tmem_alias_f32_batched_7ba9991804",
    "exact_bf16_varlen": "mamba_ssd_q_tmem_alias_bf16_varlen_5164405361",
    "exact_f16_varlen": "mamba_ssd_q_tmem_alias_f16_varlen_a9cdf1469c",
    "exact_f32_varlen": "mamba_ssd_q_tmem_alias_f32_varlen_b909ad0d33",
    "exact_bf16_batched_noring": "mamba_ssd_q_tmem_alias_bf16_batched_noring_f26d53b67a",
    "exact_bf16_varlen_noring": "mamba_ssd_q_tmem_alias_bf16_varlen_noring_e4b1e97be8",
    "exact_f16_batched_noring": "mamba_ssd_q_tmem_alias_f16_batched_noring_f1a7364d1d",
    "exact_f16_varlen_noring": "mamba_ssd_q_tmem_alias_f16_varlen_noring_e978a7b714",
    "exact_f32_batched_noring": "mamba_ssd_q_tmem_alias_f32_batched_noring_5d52af609f",
    "exact_f32_varlen_noring": "mamba_ssd_q_tmem_alias_f32_varlen_noring_3db93d8b16",
}
_CHUNKPAR_MODULES = {
    "chunkpar_bf16_batched": "mamba_ssd_chunk_parallel_v3_bf16_batched_d4923ea014",
    "chunkpar_f16_batched": "mamba_ssd_chunk_parallel_v3_f16_batched_c3c1d5ce0e",
    "chunkpar_f32_batched": "mamba_ssd_chunk_parallel_v3_f32_batched_9dece8eafe",
    "chunkpar_bf16_varlen": "mamba_ssd_chunk_parallel_v3_bf16_varlen_1901669584",
    "chunkpar_f16_varlen": "mamba_ssd_chunk_parallel_v3_f16_varlen_fd716a595d",
    "chunkpar_f32_varlen": "mamba_ssd_chunk_parallel_v3_f32_varlen_c1c31b225e",
}
_EXACTPRE_MODULES = {
    "exactpre_bf16_batched": "mamba_ssd_q_tmem_alias_bf16_batched_precomputed_4531674c62",
    "exactpre_bf16_varlen": "mamba_ssd_q_tmem_alias_bf16_varlen_precomputed_956e111919",
    "exactpre_f16_batched": "mamba_ssd_q_tmem_alias_f16_batched_precomputed_5103b26966",
    "exactpre_f16_varlen": "mamba_ssd_q_tmem_alias_f16_varlen_precomputed_7ed92cf8f5",
    "exactpre_f32_batched": "mamba_ssd_q_tmem_alias_f32_batched_precomputed_d0a3c9b35c",
    "exactpre_f32_varlen": "mamba_ssd_q_tmem_alias_f32_varlen_precomputed_2f69268561",
    "exactpre_bf16_batched_noring": "mamba_ssd_q_tmem_alias_bf16_batched_precomputed_noring_ea22a1bb6b",
    "exactpre_bf16_varlen_noring": "mamba_ssd_q_tmem_alias_bf16_varlen_precomputed_noring_dac8f82648",
    "exactpre_f16_batched_noring": "mamba_ssd_q_tmem_alias_f16_batched_precomputed_noring_3ebc7e8e13",
    "exactpre_f16_varlen_noring": "mamba_ssd_q_tmem_alias_f16_varlen_precomputed_noring_af40bd8812",
    "exactpre_f32_batched_noring": "mamba_ssd_q_tmem_alias_f32_batched_precomputed_noring_4cbe908403",
    "exactpre_f32_varlen_noring": "mamba_ssd_q_tmem_alias_f32_varlen_precomputed_noring_fd84383935",
}
_CHUNKPAR3_MODULES = {
    "chunkpar3_bf16_batched": "mamba_ssd_chunk_parallel_bf16_batched_f17431806f",
    "chunkpar3_f16_batched": "mamba_ssd_chunk_parallel_f16_batched_a1c231f3d6",
    "chunkpar3_f32_batched": "mamba_ssd_chunk_parallel_f32_batched_e2c4a6fe65",
    "chunkpar3_bf16_varlen": "mamba_ssd_chunk_parallel_bf16_varlen_1161b3473c",
    "chunkpar3_f16_varlen": "mamba_ssd_chunk_parallel_f16_varlen_d5600a56db",
    "chunkpar3_f32_varlen": "mamba_ssd_chunk_parallel_f32_varlen_6798faa560",
}

_SEGMENT_PREPROCESS = _Kernel(
    _SEGMENT_PREPROCESS_MODULE,
    "kernel_factorized_persistent_segment_preprocess",
    threads=256,
    fast_math=False,
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
# Cake export.  0 is the unfilled placeholder: the program refuses to build.
_EXACT_SMEM_BYTES = 232320
_CHUNKPAR_SMEM_BYTES = 231808
_CHUNKPAR3_SMEM_BYTES = 232448
# Main dynamic SMEM of the exact-scan ``_noring`` programs (the start schedule
# keeps the prefix control placement); refreshed by the Cake export with the
# ``exact_*_noring`` table entries.
_EXACT_NORING_SMEM_BYTES = 232192
# Main dynamic SMEM of the two-kernel exact-scan programs (``exactpre_*``) and
# of their ``_noring`` twins; refreshed by the Cake export with the
# ``exactpre_*`` table entries (0 = not exported yet).
_EXACTPRE_SMEM_BYTES = 232320
_EXACTPRE_NORING_SMEM_BYTES = 232192


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
# Exact-scan main kernel, both families (the fused ``exact_*`` programs and the
# two-kernel ``exactpre_*`` programs): the 41 start arguments followed by the
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
    "exact": (_HOST_TEMPLATE, _MAIN_ARGS, _EXACT_SMEM_BYTES),
    "chunkpar": (_CHUNKPAR_HOST_TEMPLATE, _MAIN_ARGS_CHUNKPAR, _CHUNKPAR_SMEM_BYTES),
    "exactpre": (_EXACTPRE_HOST_TEMPLATE, _MAIN_ARGS, _EXACTPRE_SMEM_BYTES),
    "chunkpar3": (_CHUNKPAR3_HOST_TEMPLATE, _MAIN_ARGS_CHUNKPAR3, _CHUNKPAR3_SMEM_BYTES),
}


def _scan(module: str) -> _Kernel:
    """The main kernel of ``generated/device/<module>.cu`` (either family)."""

    symbol = module.rsplit("_", 1)[0]
    return _Kernel(module, f"kernel_{symbol}", threads=512, fast_math=True)


# Main dynamic SMEM of a family's non-default schedule variants.
_VARIANT_SMEM_BYTES = {
    ("exact", "noring"): _EXACT_NORING_SMEM_BYTES,
    ("exactpre", "noring"): _EXACTPRE_NORING_SMEM_BYTES,
}


def _program(name: str, module: str) -> _Program:
    """Bind program ``<family>_<state>_<mode>[_<variant>]`` to its generated main source."""

    family, state_key, _mode, *variant = name.split("_")
    code, bits = _STATE_DTYPE_CODES[state_key]
    host_template, main_args, smem_bytes = _FAMILY_HOST[family]
    if variant:
        (variant_name,) = variant
        smem_bytes = _VARIANT_SMEM_BYTES[(family, variant_name)]
    return _Program(
        family,
        _SEGMENT_PREPROCESS,
        _scan(module),
        code,
        bits,
        smem_bytes,
        host_template,
        main_args,
    )


_PROGRAMS: dict[str, _Program] = {
    name: _program(name, module)
    for name, module in (
        *_SCAN_MODULES.items(),
        *_CHUNKPAR_MODULES.items(),
        *_EXACTPRE_MODULES.items(),
        *_CHUNKPAR3_MODULES.items(),
    )
}


def _program_name(
    family: str, state_dtype: torch.dtype, mode_varlen: bool, variant: str = ""
) -> str:
    """The program of ``family`` serving a state dtype in batched or
    packed-varlen mode; ``variant`` names a non-default schedule variant of the
    family (``"noring"`` for the two exact-scan families)."""

    mode_key = "varlen" if mode_varlen else "batched"
    suffix = f"_{variant}" if variant else ""
    return f"{family}_{_STATE_DTYPE_KEYS[state_dtype]}_{mode_key}{suffix}"


def _require_exported(name: str, program: _Program) -> None:
    """Refuse a program whose table entries the Cake export has not filled."""

    if _PENDING_EXPORT in program.main.module or program.main_smem_bytes <= 0:
        raise RuntimeError(
            f"Cake SSDCombined program {name} is not exported yet (main module "
            f"{program.main.module!r}, main dynamic shared memory "
            f"{program.main_smem_bytes} bytes): run the Cake export "
            "(tools/export_cake_mamba_ssd_combined.py with the chunk-parallel "
            "programs) and fill this loader's program table from its summary"
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


def _segment_bound(seqlen: int, num_sequences: int) -> int:
    """Segment-count bound of the ``cu_seqlens`` form: every 128-token chunk
    of the packed stream plus one unaligned start and one unaligned checkpoint
    per sequence.  Sizes the preprocess grid, the delta/cumsum workspace and
    the ``[bound + 1]`` metadata tables; the real count stays on the device."""

    return -(-int(seqlen) // _CHUNK_SIZE) + 2 * int(num_sequences)


def _persistent_grid_size(*, total_work: int, sm_count: int) -> int:
    """Balanced persistent grid for ``total_work`` (sequence, head) items."""

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
}
# Unmeasured capabilities use the B200 constants.
_CHUNK_PARALLEL_COST_MODEL_DEFAULT = (10, 0)
_CHUNK_PARALLEL_SMALL_ITEMS = 32
_CHUNK_PARALLEL_LARGE_ITEMS = 128
_CHUNK_PARALLEL_SELECTION_MARGIN = 1.05
_CHUNK_PARALLEL_WORKSPACE_CAP_BYTES = 256 << 20
# Third cost arm (CAKE-956 round 6 option A): the three-phase program's own
# calibrated constants (``CHUNK_PARALLEL_COST_MODEL_US`` of the Cake seed
# module flashinfer_blackwell_mamba_ssd_combined_chunk_parallel_seed_v1, round
# 4), copied verbatim; they take part in the exact-vs-chunk-parallel gate only
# (the faster of the two chunk-parallel predictions is compared with the
# exact scan's).
_THREE_PHASE_COST_MODEL_US = {
    (10, 0): {"cp_fixed": 25.0, "cp_per_tile": 0.0491},
    (10, 3): {"cp_fixed": 24.1, "cp_per_tile": 0.0429},
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
# Chunk-parallel workspace per (chunk, head) tile: the f32 [64, 128] state
# increments (32 KB) plus the bf16 [64, 128] state operand (16 KB).
_CHUNK_PARALLEL_WORKSPACE_BYTES_PER_TILE = _HEADDIM * _DSTATE * (4 + 2)
# Look-back device state, mirrored from the Cake seed module
# (flashinfer_blackwell_mamba_ssd_combined_chunk_parallel_lookback_seed_v1.py):
# the item-major tile range is split into ``vcount`` virtual CTA ranges (V =
# the SM count the program balances against per capability; the B200 value
# for an unmeasured capability; every floating-point result depends on
# (inputs, V) only, never on the launch grid).  ``grid_barrier`` is u32
# ``[TICKET, EPOCH, pad, pad, FLAG[0..SEGMENT_CAP)]`` (zeroed once per device;
# the kernel's claim counter wraps itself, the epoch free-runs and the flags
# carry the epoch).  ``agg_work`` holds V f32 [64, 128] range aggregates
# followed by V f32 log-decay scalars.
_CHUNK_PARALLEL_SEGMENT_CAP = 1024
_CHUNK_PARALLEL_GRID_BARRIER_WORDS = 4 + _CHUNK_PARALLEL_SEGMENT_CAP
_CHUNK_PARALLEL_VCOUNT_BY_ARCH = {(10, 0): 148, (10, 3): 152}
_CHUNK_PARALLEL_VCOUNT_DEFAULT = (10, 0)
_CHUNK_PARALLEL_AGG_ELEMS_PER_RANGE = _HEADDIM * _DSTATE + 1
# Three-phase device state: u32 ``[arrive, generation]`` of its software grid
# barrier (zeroed once per device; the arrive counter wraps to zero after
# ``gridDim.x`` arrivals and the generation word free-runs).  Its word
# protocol differs from the look-back's ``[ticket, epoch, ...]``, so the two
# families never share a barrier buffer.
_CHUNKPAR3_GRID_BARRIER_WORDS = 2


def _chunk_parallel_vcount(capability: tuple[int, int]) -> int:
    """The look-back program's virtual CTA count V of a compute capability."""

    key = (int(capability[0]), int(capability[1]))
    vcount = _CHUNK_PARALLEL_VCOUNT_BY_ARCH.get(
        key, _CHUNK_PARALLEL_VCOUNT_BY_ARCH[_CHUNK_PARALLEL_VCOUNT_DEFAULT]
    )
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
    workspace binds; the device index because the buffers live on it."""

    device_index: Optional[int]
    batch: int
    seqlen: int
    nchunks: int
    num_segments: int
    num_sequences: int
    state_dtype: torch.dtype
    from_cu_seqlens: bool
    chunk_parallel_mode: str


# ``_WorkspaceKey`` without the family override: what ``prepare`` freezes the
# override for.
_ShapeKey = Tuple[Optional[int], int, int, int, int, int, torch.dtype, bool]


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
    *, work_items: int, chunks: int, tiles: int, mode_varlen: bool
) -> bool:
    """Inside the chunk-parallel regime: the three-phase family for batched
    calls with at most ``_THREE_PHASE_MAX_TILES`` tiles or with at least
    ``_THREE_PHASE_MIN_ITEMS`` work items and ``_THREE_PHASE_MIN_CHUNKS``
    chunks per item; the look-back family otherwise."""

    if mode_varlen:
        return False
    return int(tiles) <= _THREE_PHASE_MAX_TILES or (
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
        and quantities["workspace_bytes"] <= _CHUNK_PARALLEL_WORKSPACE_CAP_BYTES
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


# Fused-program route (CAKE-956 round 6 item 2, lane i2 rule v4), mirrored from
# ``fused_route`` of
# ``loom.examples.weave.flashinfer_blackwell_mamba_ssd_combined_exact_scan_seed_v1``
# and read on the same host-known quantities: ``items = sequences x heads`` is
# the persistent work count; batched calls take the fused program when
# ``items <= 1024``, chunks per sequence ``<= 16`` and ``items x chunks <=
# 2048``; packed-varlen calls when ``items <= 256`` and the packed chunk count
# times ``nheads`` is ``<= 512``.  Every other call keeps the two-kernel
# program (``exactpre_*``): the fused kernel's in-CTA preprocess serialises
# with the scan and loses on saturated grids (93-row calibration, both archs:
# 50 routed rows win beyond the band, no routed row loses).
_FUSED_ROUTE_MAX_ITEMS_BATCHED = 1024
_FUSED_ROUTE_MAX_CHUNKS_BATCHED = 16
_FUSED_ROUTE_MAX_CHUNK_ITEMS_BATCHED = 2048
_FUSED_ROUTE_MAX_ITEMS_VARLEN = 256
_FUSED_ROUTE_MAX_CHUNK_ITEMS_VARLEN = 512


def fused_selected(
    *,
    nheads: int,
    num_sequences: int,
    nchunks: int,
    mode_varlen: bool,
    total_tokens: int,
) -> bool:
    """Exact-scan shape window: the fused program (``exact_*``) for one-wave /
    short-item calls, the two-kernel program (``exactpre_*``) otherwise.
    ``nchunks`` is the chunk count per sequence (batched); ``total_tokens`` the
    packed token count (varlen; the per-sequence lengths live on the device
    and are not read).  The runner applies the window only where the fused
    program's preprocess grid is 0 -- batched calls and the ``cu_seqlens``
    form; the ``seq_idx`` triple form always runs the two-kernel program
    (``_materialize_workspace``)."""

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


def _sequence_arguments(
    main_args: tuple[str, ...],
    preprocess: Mapping[str, object],
    preprocess_grid: tuple[int, int, int],
    main: Mapping[str, object],
    main_grid: tuple[int, int, int],
    cuda_stream: int,
) -> tuple[object, ...]:
    """Order the name-keyed stage values into the launcher's positional ABI;
    ``main_args`` is the program's main-kernel argument order."""

    return (
        *(preprocess[name] for name in _PREPROCESS_ARGS),
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
            "Cake SSDCombined requires SM100 or SM103, got "
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


def _render_host_source(template: str, name: str, program: _Program) -> str:
    """Substitute one program's table values into the shared launcher.

    The program name becomes part of the launcher namespace so the inline
    launch helpers (and their function-local static kernel handles) of one
    loaded program library are never unified with another's.
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
# CAKE-956 round 6 recalibration (lane i5a) on the round-6 exact-scan programs
# (Cake 370cc1dc7941 + the adopted 1b(ii) prescale factor table, which changes
# the per-chunk cost of BOTH variants): scan-kernel time ring / ``_noring``,
# the two variants of the same program as the two arms, CUPTI, cold L2, 3 and 5
# interleaved rounds, kernel band 0.005.  GB300 (nvl72d215-T02; steps
# b2dfae6f4779984e576bc87f bf16 3 rounds, e6235314fb13dc4f3ea8b931 bf16 5 rounds,
# 809b22098ffc542789a047b1 f32):
#   varlen, z, 64 tokens, bf16 state: nseq 1 0.940; nseq 2 / 3 / 4 / 5 1.003 /
#     1.000 / 1.002 / 1.003 (5 rounds 0.989 / 1.001 / 1.000 / 1.005: ties);
#     nseq 6 / 8 / 16 / 64 1.011 / 1.017 / 1.013 / 1.014 (5 rounds 1.008 / 1.021);
#     63 / 65 tokens at nseq 8 0.977 / 0.969; mixtures [63, 65] x 4 1.012,
#     [32, 96] x 4 1.011 -> the bf16 / f16 window above is unchanged.
#   varlen, z, 64 tokens, f32 state: nseq 1 / 2 / 3 / 4 / 5 / 6 / 8 / 16 / 64 =
#     0.945 / 0.986 / 0.984 / 0.993 / 0.990 / 0.983 / 0.997 / 0.989 / 0.981 ->
#     the ring program wins every f32 cell (nseq 8 a tie): the round-5 f32
#     threshold (4 x 128 items) is retired; an f32 state never takes ``_noring``.
#   batched, no z, batch 2 x 25 / 50 / 60 / 75 / 90 / 100 tokens: bf16 0.997 /
#     0.995 / 0.987 / 0.995 / 0.992 / 0.995 (5 rounds 1.003 / 0.987 / 0.997 /
#     0.987 / 0.992 / 0.985), f32 0.977 / 0.982 / - / 0.990 / 0.973 / -;
#     short_batched_2x50 0.992 / 0.995 -> no batched cell loses on GB300 any more.
# B200 (nsc-svg-slurm-1-gpu-49; steps 46baa0c96214abfe53fe148d bf16 3 rounds,
# e3377c73e383c429af13cd86 bf16 5 rounds, 4c210f860a87eeeb876dedcb f32):
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
# Round-6 rule change: an f32 state never takes the start schedule (the
# bf16 / f16 varlen window and the batched window are unchanged).
_EXACT_NORING_VARLEN_MIN_ITEMS = {  # sequences x heads, by state dtype; a state
    # dtype absent here (f32, round 6) never takes the start schedule
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

    program = _PROGRAMS[name]
    _require_exported(name, program)
    source_dir = _source_dir()
    host_source = _render_host_source(
        (source_dir / program.host_template).read_text(encoding="utf-8"),
        name,
        program,
    )
    nvcc = _nvcc()
    digest = hashlib.sha256()
    digest.update(host_source.encode("utf-8"))
    digest.update(arch.encode())
    digest.update(str(nvcc).encode())
    device_sources: list[tuple[_Kernel, Path]] = []
    for kernel in program.kernels:
        source_path = source_dir / _DEVICE_DIR / kernel.source
        digest.update(kernel.module.encode())
        digest.update(source_path.read_bytes())
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
            extra_cflags=["-O3"],
            extra_ldflags=["-lcuda"],
            build_directory=str(build_dir),
        )


class CakeSSDCombined:
    """Source-built Cake implementation of the admitted SSDCombined domain.

    The kernels require head dimension 64, state dimension 128, BF16
    inputs/outputs, BF16, FP16, or FP32 states, and SM100/SM103.  Any positive
    ``chunk_size`` is accepted as the caller's convention: the programs tile
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
    supplies a precomputed vector) followed by the scan.  The fused
    exact-scan program is selected only where its preprocess stage is skipped
    (grid 0): batched calls and the ``cu_seqlens`` form, where the call is
    one kernel; the ``seq_idx`` triple form runs the two-kernel program.  A
    caller buffer for the ``cu_seqlens``-derived vector brings the fused
    program's preprocess back to publish it (two launches).

    Four kernel families serve every call and are selected per call.  The
    fused exact scan (``exact_*`` programs) runs one persistent CTA per
    (sequence, head) item over its chunks in order and computes delta /
    cumsum in-CTA; the two-kernel exact scan (``exactpre_*``) is the same
    persistent scan reading the delta / cumsum its preprocess wrote to the
    HBM workspace -- the start two-launch form, kept for the calls the fused
    program loses on (:func:`fused_selected`: batched calls with more than
    1024 (sequence, head) items, more than 16 chunks per sequence or more
    than 2048 chunk-items; packed-varlen calls with more than 256 items or
    more than 512 packed chunk-items) and for the seq_idx triple form.  The
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

    #: Program of the most recent :meth:`run` (``exact_*``, ``exactpre_*``,
    #: ``chunkpar_*`` or ``chunkpar3_*``); ``None`` before the first call.
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
        """The runner-owned ``preprocess_status`` word (one int32 per device).

        The preprocess stores 1 into it when a packed-sequence id is out of
        range or non-monotonic (see the class docstring); it is allocated
        zeroed once per device and never read on the launch path.
        """

        status = self._preprocess_status.get(device.index)
        if status is None:
            self._allocating("the preprocess status word")
            status = torch.zeros(1, dtype=torch.int32, device=device)
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
        value = int(status.item())
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
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Resolve graph-stable packed storage for a public strided input.

        Returns the tensor to bind and the source still to be copied into
        it, or ``None`` when the input is already contiguous.  The copies are
        issued together right before the launch so no Python runs between the
        auxiliary device work and the kernels.  One buffer per (input, shape,
        dtype) lives in the workspace for its lifetime and is never replaced,
        so the address a captured graph copies into stays valid when a later
        eager call of the same shape brings, say, a bf16 ``dt`` instead of an
        f32 one.
        """

        if value is None or value.is_contiguous():
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
        # The fused program is a single launch only where its preprocess
        # grid is 0: batched calls and the cu_seqlens form (the kernel
        # derives the segment tables in-CTA).  In the seq_idx triple form the
        # preprocess still runs to publish ``seq_chunk_cumsum``, and the fused
        # main kernel then costs more than the two-kernel main kernel, so the
        # triple form keeps the two-kernel program (round-6 FI mirror of the
        # Cake ``fused_route`` window plus this grid-0 condition).
        fused = (
            not chunk_parallel
            and (not self.has_varlen or key.from_cu_seqlens)
            and fused_selected(
                nheads=self.nheads,
                num_sequences=num_sequences,
                nchunks=nchunks,
                mode_varlen=self.has_varlen,
                total_tokens=batch * seqlen,
            )
        )
        variant = (
            ""
            if chunk_parallel
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
        workspace: dict[str, Any] = {
            "program": _program_name(
                family if chunk_parallel else "exact" if fused else "exactpre",
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
            # vector is never replaced by a call of another shape.
            workspace["seq_chunk_cumsum"] = torch.empty(
                num_sequences + 1, dtype=torch.int32, device=device
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
                device, tile_count, _chunk_parallel_vcount(capability)
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
        plus ``vcount`` scalars) is per device too; the kernel writes every
        slot it reads within one launch, so it needs no initial value.  A
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
            self._allocating(f"the chunk-parallel aggregate workspace for {vcount} ranges")
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
        ``seq_chunk_cumsum`` given here is validated as :meth:`run` does with
        ``update_seq_chunk_cumsum=True``; the checkpoint outputs are per-call
        options of :meth:`run`.  The kernel handles of the host shim resolve
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
            update_seq_chunk_cumsum=seq_chunk_cumsum is not None,
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

        def packed(name: str, value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            buffer, source = self._contiguous_input(workspace, name, value)
            if source is not None:
                assert buffer is not None
                pending_copies.append((buffer, source))
            return buffer

        dt = packed("dt", dt)
        A = packed("A", A)
        D = packed("D", D)
        z = packed("z", z)
        dt_bias = packed("dt_bias", dt_bias)
        initial_states = packed("initial_states", initial_states)
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
            write_seq_chunk_cumsum = True
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
            seqlen=seqlen,
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
        if program.family == "exact":
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
        else:
            grid = _persistent_grid_size(
                total_work=num_sequences * self.nheads, sm_count=sm_count
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
            # Both exact-scan families bind these (``_MAIN_ARGS``); only the
            # fused program reads them: it derives the cu_seqlens segment
            # tables in-CTA and flags an invalid cu_seqlens in the runner's
            # status word.  The two-kernel and chunk-parallel programs read
            # the tables their preprocess published.
            "cu_seqlens": cu_seqlens_arg,
            "metadata_from_cu_seqlens": int(program.family == "exact" and from_cu_seqlens),
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
            # the device's current grow-only buffers.
            vcount = _chunk_parallel_vcount(_ARCH_CAPABILITIES[arch])
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
                    _chunk_parallel_vcount(_ARCH_CAPABILITIES[arch]),
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
