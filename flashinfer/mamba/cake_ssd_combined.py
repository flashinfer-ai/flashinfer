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
# ``CAKE_SSD_*`` placeholders and namespace scheme; the chunk-parallel main
# kernel takes four more arguments (``_MAIN_ARGS_CHUNKPAR``).
_HOST_TEMPLATE = Path("generated") / "host" / "mamba_ssd_combined_sequence.cpp"
_CHUNKPAR_HOST_TEMPLATE = (
    Path("generated") / "host" / "mamba_ssd_combined_chunk_parallel_sequence.cpp"
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

    ``family`` is ``exact`` (the persistent scan, one CTA per (sequence,
    head) item) or ``chunkpar`` (the chunk-parallel program, one CTA per
    (chunk, head) tile); it fixes the host template and the main kernel's
    argument order.
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
# derives from it.  Two kernel families ship, each x {bf16, f16, f32 state} x
# {batched, varlen}, plus one preprocess shared by both: the exact scan
# (``exact_*``) and the chunk-parallel program (``chunkpar_*``).
_SEGMENT_PREPROCESS_MODULE = "factorized_persistent_segment_preprocess_ff8a998f8c"
_SCAN_MODULES = {
    "exact_bf16_batched": "mamba_ssd_q_tmem_alias_bf16_batched_eba19916b3",
    "exact_f16_batched": "mamba_ssd_q_tmem_alias_f16_batched_b56612c764",
    "exact_f32_batched": "mamba_ssd_q_tmem_alias_f32_batched_dac68e5ef1",
    "exact_bf16_varlen": "mamba_ssd_q_tmem_alias_bf16_varlen_5332560192",
    "exact_f16_varlen": "mamba_ssd_q_tmem_alias_f16_varlen_dd75c0cd47",
    "exact_f32_varlen": "mamba_ssd_q_tmem_alias_f32_varlen_70e1bc6bef",
}
_CHUNKPAR_MODULES = {
    "chunkpar_bf16_batched": "mamba_ssd_chunk_parallel_bf16_batched_b5e0c1cc35",
    "chunkpar_f16_batched": "mamba_ssd_chunk_parallel_f16_batched_72f5ce3af3",
    "chunkpar_f32_batched": "mamba_ssd_chunk_parallel_f32_batched_f471149926",
    "chunkpar_bf16_varlen": "mamba_ssd_chunk_parallel_bf16_varlen_aa911bd5d9",
    "chunkpar_f16_varlen": "mamba_ssd_chunk_parallel_f16_varlen_32d827eb25",
    "chunkpar_f32_varlen": "mamba_ssd_chunk_parallel_f32_varlen_39f645fa37",
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
_EXACT_SMEM_BYTES = 232448
_CHUNKPAR_SMEM_BYTES = 231936


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
# Exact-scan main kernel (the ``exact_*`` programs).
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
)
# Chunk-parallel main kernel (the ``chunkpar_*`` programs): the exact-scan
# order with the state-operand tensor map ``h_map`` after ``out_map`` and the
# workspace pointers ``s_work`` (f32 per-tile state increments), ``h_words``
# (u32 view of the bf16 state operand behind ``h_map``) and ``grid_barrier``
# (u32 ``[arrive, generation]``) after ``out_native``.
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
}


def _scan(module: str) -> _Kernel:
    """The main kernel of ``generated/device/<module>.cu`` (either family)."""

    symbol = module.rsplit("_", 1)[0]
    return _Kernel(module, f"kernel_{symbol}", threads=512, fast_math=True)


def _program(name: str, module: str) -> _Program:
    """Bind program ``<family>_<state>_<mode>`` to its generated main source."""

    family, state_key, _mode = name.split("_")
    code, bits = _STATE_DTYPE_CODES[state_key]
    host_template, main_args, smem_bytes = _FAMILY_HOST[family]
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
    for name, module in (*_SCAN_MODULES.items(), *_CHUNKPAR_MODULES.items())
}


def _program_name(family: str, state_dtype: torch.dtype, mode_varlen: bool) -> str:
    """The program of ``family`` serving a state dtype in batched or
    packed-varlen mode."""

    mode_key = "varlen" if mode_varlen else "batched"
    return f"{family}_{_STATE_DTYPE_KEYS[state_dtype]}_{mode_key}"


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
# recalibrated.  Least-squares fits of the main-kernel time in microseconds:
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
        "serial_fixed": 5.4,
        "serial_per_chunk_small": 2.68,
        "serial_per_chunk_large": 4.58,
        "cp_fixed": 29.9,
        "cp_per_tile": 0.0627,
    },
    (10, 3): {  # B300 / GB300, sm_103a
        "serial_fixed": 5.2,
        "serial_per_chunk_small": 2.59,
        "serial_per_chunk_large": 4.16,
        "cp_fixed": 28.8,
        "cp_per_tile": 0.0595,
    },
}
# Unmeasured capabilities use the B200 constants.
_CHUNK_PARALLEL_COST_MODEL_DEFAULT = (10, 0)
_CHUNK_PARALLEL_SMALL_ITEMS = 32
_CHUNK_PARALLEL_LARGE_ITEMS = 128
_CHUNK_PARALLEL_SELECTION_MARGIN = 1.05
_CHUNK_PARALLEL_WORKSPACE_CAP_BYTES = 256 << 20
# Chunk-parallel workspace per (chunk, head) tile: the f32 [64, 128] state
# increments (32 KB) plus the bf16 [64, 128] state operand (16 KB).
_CHUNK_PARALLEL_WORKSPACE_BYTES_PER_TILE = _HEADDIM * _DSTATE * (4 + 2)
# Grid barrier state: u32 ``[arrive, generation]``.
_CHUNK_PARALLEL_GRID_BARRIER_WORDS = 2
# ``auto`` applies the rule; ``always`` / ``never`` force the family.  Read on
# every unprepared eager call; ``CakeSSDCombined.prepare`` reads it once and
# freezes it for the call shape (a captured call never reads it).
_CHUNK_PARALLEL_ENV = "FLASHINFER_CAKE_SSD_CHUNK_PARALLEL"
_CHUNK_PARALLEL_MODES = ("auto", "always", "never")


def _chunk_parallel_mode() -> str:
    """The ``FLASHINFER_CAKE_SSD_CHUNK_PARALLEL`` override of the current call."""

    value = os.environ.get(_CHUNK_PARALLEL_ENV, "auto")
    mode = value.strip().lower()
    if mode not in _CHUNK_PARALLEL_MODES:
        raise ValueError(
            f"{_CHUNK_PARALLEL_ENV} must be auto, always or never; got {value!r}"
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
    """The quantities the family rule reads and both programs' predicted
    main-kernel times (microseconds) for one call.

    ``work_items`` is the exact scan's concurrency (one CTA per (sequence,
    head) item); ``tiles`` the chunk-parallel tile bound (``num_segments *
    nheads``, the preprocess row count; the cu_seqlens form passes its host
    segment bound); ``chunks`` the serial chunk count per item: ``nchunks``
    when batched, the mean logical-chunk count per packed sequence when varlen
    (the per-sequence counts live on the device).
    """

    model = _CHUNK_PARALLEL_COST_MODEL_US.get(
        (int(capability[0]), int(capability[1])),
        _CHUNK_PARALLEL_COST_MODEL_US[_CHUNK_PARALLEL_COST_MODEL_DEFAULT],
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
    return {
        "work_items": work_items,
        "sm_count": int(sm_count),
        "capability": (int(capability[0]), int(capability[1])),
        "chunks": chunks,
        "tiles": tiles,
        "workspace_bytes": tiles * _CHUNK_PARALLEL_WORKSPACE_BYTES_PER_TILE,
        "predicted_us": {
            "exact_scan": model["serial_fixed"] + per_chunk * chunks,
            "chunk_parallel": model["cp_fixed"]
            + model["cp_per_tile"] * max(0, tiles - int(sm_count)),
        },
    }


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
    """Family rule: chunk-parallel iff fewer work items than SMs, the S/H
    workspace fits the cap, and its predicted main-kernel time times the
    selection margin still beats the exact scan's.  ``mode`` is the
    environment override: ``always`` / ``never`` force the answer, ``auto``
    applies the rule."""

    if mode not in _CHUNK_PARALLEL_MODES:
        raise ValueError(
            f"chunk-parallel mode must be auto, always or never; got {mode!r}"
        )
    if mode == "always":
        return True
    if mode == "never":
        return False
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
    return (
        quantities["work_items"] < quantities["sm_count"]
        and quantities["workspace_bytes"] <= _CHUNK_PARALLEL_WORKSPACE_CAP_BYTES
        and predicted["chunk_parallel"] * _CHUNK_PARALLEL_SELECTION_MARGIN
        <= predicted["exact_scan"]
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
    supplies a precomputed vector) followed by the scan.

    Two kernel families serve every call and are selected per call.  The
    exact scan (``exact_*`` programs) runs one persistent CTA per (sequence,
    head) item over its chunks in order.  The chunk-parallel program
    (``chunkpar_*``) computes the same arithmetic in a different schedule --
    one CTA per (chunk, head) tile, at most one per SM, with the inter-chunk
    state recurrence resolved across a grid barrier -- and its output and
    final states are bitwise identical to the exact scan's.  It is selected
    by a calibrated cost rule (:func:`chunk_parallel_selected`: fewer
    (sequence, head) items than SMs, the per-tile workspace within 256 MiB,
    and a predicted main-kernel time that beats the exact scan's by the
    selection margin), so it serves long prefills with few heads (one
    sequence of 32768 tokens with 8 heads runs about four times faster) and
    never 128-head calls.  The environment variable
    ``FLASHINFER_CAKE_SSD_CHUNK_PARALLEL`` (``auto``, the default;
    ``always``; ``never``) forces either family; any other value is
    rejected.  It is read on every unprepared eager call; :meth:`prepare`
    reads it once and freezes the choice for the call shape.  The
    chunk-parallel workspace (32 KB f32 + 16 KB bf16 per tile, grown only
    when a call needs more tiles, plus a two-word grid-barrier state zeroed
    once per device) is owned by the runner.  :attr:`last_program_name`
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

    #: Program of the most recent :meth:`run` (``exact_*`` or
    #: ``chunkpar_*``); ``None`` before the first call.
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
        chunk_parallel = chunk_parallel_selected(
            nheads=self.nheads,
            num_sequences=num_sequences,
            num_segments=num_segments,
            nchunks=nchunks,
            mode_varlen=self.has_varlen,
            sm_count=sm_count,
            capability=capability,
            mode=key.chunk_parallel_mode,
        )
        workspace: dict[str, Any] = {
            "program": _program_name(
                "chunkpar" if chunk_parallel else "exact",
                self.state_dtype,
                self.has_varlen,
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
            self._chunk_parallel_workspace(device, tile_count)
        return workspace

    def _chunk_parallel_workspace(
        self, device: torch.device, tiles_bound: int
    ) -> dict[str, torch.Tensor]:
        """The chunk-parallel main kernel's workspace on ``device``.

        ``s_work`` (f32 ``[tiles, 64, 128]``, the per-tile state increments,
        32 KB per tile) and ``h_work`` (bf16 ``[tiles, 64, 128]``, the state
        operand entering each tile, 16 KB per tile; bound as the ``h_map``
        tensor map and, through ``h_words``, as its u32 view) grow only: a
        call whose tile bound exceeds the allocation replaces them, smaller
        calls reuse them (the kernel addresses tiles below the call's bound
        only).  ``grid_barrier`` (u32 ``[arrive, generation]``) is allocated
        zeroed once per device and never re-zeroed: the kernel's arrive
        counter self-resets and its generation word is free-running.  A
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
        return {**buffers, "grid_barrier": grid_barrier}

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
            preprocess_status=self._preprocess_status_buffer(x.device),
        )
        if program.family == "chunkpar":
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
        }
        chunk_parallel: Optional[dict[str, torch.Tensor]] = None
        if program.family == "chunkpar":
            # The state operand workspace is bound twice: as the ``h_map``
            # tensor map (the host builds the TMA descriptor from the bf16
            # [tiles, 64, 128] tensor) and as the u32 word view the phase-2
            # chains store through.  A captured workspace binds the buffers
            # its graph recorded (pinned by ``run``); every other call binds
            # the device's current grow-only buffers.
            chunk_parallel = workspace.get("chunk_parallel")
            if chunk_parallel is None:
                chunk_parallel = self._chunk_parallel_workspace(
                    x.device, num_segments * self.nheads
                )
            main_values.update(
                h_map=chunk_parallel["h_work"],
                s_work=chunk_parallel["s_work"],
                h_words=chunk_parallel["h_words"],
                grid_barrier=chunk_parallel["grid_barrier"],
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
