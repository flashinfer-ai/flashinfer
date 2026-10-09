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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

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
    "exact_bf16_batched": "mamba_ssd_q_tmem_alias_bf16_batched_e2bea5f3e7",
    "exact_f16_batched": "mamba_ssd_q_tmem_alias_f16_batched_1d9fc0c3c8",
    "exact_f32_batched": "mamba_ssd_q_tmem_alias_f32_batched_0a79355dd1",
    "exact_bf16_varlen": "mamba_ssd_q_tmem_alias_bf16_varlen_fca9cbf4e7",
    "exact_f16_varlen": "mamba_ssd_q_tmem_alias_f16_varlen_2ac63fe21c",
    "exact_f32_varlen": "mamba_ssd_q_tmem_alias_f32_varlen_fb7365f694",
}
_CHUNKPAR_MODULES = {
    "chunkpar_bf16_batched": "mamba_ssd_chunk_parallel_bf16_batched_39deb388f5",
    "chunkpar_f16_batched": "mamba_ssd_chunk_parallel_f16_batched_73359c8907",
    "chunkpar_f32_batched": "mamba_ssd_chunk_parallel_f32_batched_289739d002",
    "chunkpar_bf16_varlen": "mamba_ssd_chunk_parallel_bf16_varlen_8877d644f7",
    "chunkpar_f16_varlen": "mamba_ssd_chunk_parallel_f16_varlen_8af2ca5619",
    "chunkpar_f32_varlen": "mamba_ssd_chunk_parallel_f32_varlen_a92577a08a",
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
_EXACT_SMEM_BYTES = 231936
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
# every call.
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
    varlen ``total < 128``) runs on zero-padded one-chunk copies of x/B/C and
    a staged output owned by the runner, because the generated host pins a
    128-row TMA box on the token axis (CAKE-1063; see ``run``).  The output
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
    ``always``; ``never``), read on every call, forces either family; any
    other value is rejected.  The chunk-parallel workspace (32 KB f32 + 16 KB
    bf16 per tile, grown only when a call needs more tiles, plus a two-word
    grid-barrier state zeroed once per device) is owned by the runner.
    :attr:`last_program_name` names the program of the most recent call.

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
        self._workspace_key: Optional[
            Tuple[Optional[int], int, int, int, int, int, torch.dtype, bool, str]
        ] = None
        self._workspace: Optional[dict[str, Any]] = None
        self._dummy_cache: dict[Tuple[Optional[int], torch.dtype], torch.Tensor] = {}
        self._seq_cumsum_key: Optional[Tuple[Optional[int], int]] = None
        self._seq_cumsum_buf: Optional[torch.Tensor] = None
        self._preprocess_status: dict[Optional[int], torch.Tensor] = {}
        # Chunk-parallel main-kernel buffers per device (grow-only) and the
        # grid-barrier words per device (zeroed once); see
        # ``_chunk_parallel_workspace``.
        self._chunk_parallel_buffers: dict[Optional[int], dict[str, torch.Tensor]] = {}
        self._grid_barriers: dict[Optional[int], torch.Tensor] = {}

    def _preprocess_status_buffer(self, device: torch.device) -> torch.Tensor:
        """The runner-owned ``preprocess_status`` word (one int32 per device).

        The preprocess stores 1 into it when a packed-sequence id is out of
        range or non-monotonic (see the class docstring); it is allocated
        zeroed once per device and never read on the launch path.
        """

        status = self._preprocess_status.get(device.index)
        if status is None:
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
            value = torch.empty(1, dtype=dtype, device=device)
            self._dummy_cache[key] = value
        return value

    @staticmethod
    def _contiguous_input(
        workspace: dict[str, Any],
        name: str,
        value: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Resolve graph-stable packed storage for a public strided input.

        Returns the tensor to bind and the source still to be copied into
        it, or ``None`` when the input is already contiguous.  The copies are
        issued together right before the launch so no Python runs between the
        auxiliary device work and the kernels.
        """

        if value is None or value.is_contiguous():
            return value, None
        buffer_name = f"contiguous_{name}"
        buffer = workspace.get(buffer_name)
        if (
            buffer is None
            or buffer.shape != value.shape
            or buffer.dtype != value.dtype
            or buffer.device != value.device
        ):
            buffer = torch.empty(
                tuple(value.shape), dtype=value.dtype, device=value.device
            )
            workspace[buffer_name] = buffer
        return buffer, value

    def _get_workspace(
        self,
        *,
        device: torch.device,
        batch: int,
        seqlen: int,
        nchunks: int,
        num_segments: int,
        num_sequences: int,
        from_cu_seqlens: bool = False,
        sm_count: int,
        capability: tuple[int, int],
        chunk_parallel_mode: str,
    ) -> dict[str, Any]:
        """The per-call-shape workspace, with the kernel family of the call.

        The family (``workspace["program"]``) is a function of the key, the
        device's SM count and capability (fixed per ``device.index``) and the
        per-call ``FLASHINFER_CAKE_SSD_CHUNK_PARALLEL`` override, which is
        therefore part of the key; the chunk-parallel buffers are bound only
        when that family is selected.
        """

        key = (
            device.index,
            batch,
            seqlen,
            nchunks,
            num_segments,
            num_sequences,
            self.state_dtype,
            from_cu_seqlens,
            chunk_parallel_mode,
        )
        if self._workspace_key != key:
            ids = torch.arange(num_segments, dtype=torch.int32, device=device)
            chunks = ids % nchunks
            starts = (ids // nchunks) * seqlen + chunks * _CHUNK_SIZE
            # Batched segments are physical chunks; the trailing chunk of a
            # sequence whose length is not a multiple of 128 is partial.
            lengths = torch.clamp(seqlen - chunks * _CHUNK_SIZE, max=_CHUNK_SIZE)
            sequence_offsets = (
                torch.arange(num_sequences + 1, dtype=torch.int32, device=device)
                * nchunks
            )
            tile_count = num_segments * self.nheads
            chunk_parallel = chunk_parallel_selected(
                nheads=self.nheads,
                num_sequences=num_sequences,
                num_segments=num_segments,
                nchunks=nchunks,
                mode_varlen=self.has_varlen,
                sm_count=sm_count,
                capability=capability,
                mode=chunk_parallel_mode,
            )
            self._workspace = {
                "program": _program_name(
                    "chunkpar" if chunk_parallel else "exact",
                    self.state_dtype,
                    self.has_varlen,
                ),
                "starts": starts,
                "lengths": lengths,
                "sequence_offsets": sequence_offsets,
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
            }
            if from_cu_seqlens:
                # cu_seqlens form: the preprocess publishes the derived
                # chunk_indices / chunk_offsets (+ the sentinel at the real
                # segment count) into these [bound + 1] tables; the scan
                # reads them with num_logical_chunks = bound.
                self._workspace.update(
                    chunk_indices=torch.empty(
                        num_segments + 1, dtype=torch.int32, device=device
                    ),
                    chunk_offsets=torch.empty(
                        num_segments + 1, dtype=torch.int32, device=device
                    ),
                )
            if seqlen < _CHUNK_SIZE:
                # CAKE-1063: a call shorter than one chunk binds the x/B/C/out
                # tensor maps to these one-chunk buffers (see ``run``).  The
                # key fixes ``seqlen``, and ``run`` only ever writes rows
                # ``[:seqlen]`` of the input buffers, so the pad rows
                # ``[seqlen, 128)`` keep the zeros of this allocation for the
                # lifetime of the workspace: no per-call re-zeroing.  The
                # output stage is never read past ``seqlen``.
                padded = (batch, _CHUNK_SIZE)
                self._workspace.update(
                    padded_x=torch.zeros(
                        (*padded, self.nheads, _HEADDIM),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    padded_B=torch.zeros(
                        (*padded, self.ngroups, _DSTATE),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    padded_C=torch.zeros(
                        (*padded, self.ngroups, _DSTATE),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    padded_out=torch.empty(
                        (*padded, self.nheads, _HEADDIM),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                )
            if chunk_parallel:
                self._workspace.update(
                    self._chunk_parallel_workspace(device, tile_count)
                )
            self._workspace_key = key
        workspace = self._workspace
        assert workspace is not None
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
        counter self-resets and its generation word is free-running.
        """

        buffers = self._chunk_parallel_buffers.get(device.index)
        if buffers is None or buffers["h_work"].shape[0] < tiles_bound:
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
            grid_barrier = torch.zeros(
                _CHUNK_PARALLEL_GRID_BARRIER_WORDS, dtype=torch.uint32, device=device
            )
            self._grid_barriers[device.index] = grid_barrier
        return {**buffers, "grid_barrier": grid_barrier}

    def _seq_chunk_cumsum_buffer(
        self, device: torch.device, num_sequences: int
    ) -> torch.Tensor:
        """The runner-owned ``seq_chunk_cumsum`` the preprocess fills."""

        size = num_sequences + 1
        key = (device.index, size)
        if self._seq_cumsum_key != key:
            self._seq_cumsum_buf = torch.empty(size, dtype=torch.int32, device=device)
            self._seq_cumsum_key = key
        output = self._seq_cumsum_buf
        assert output is not None
        return output

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
        else:
            # Match SSDCombined's ownership contract: each allocation-returning
            # call owns fresh output storage that later calls cannot overwrite.
            out = torch.empty(expected_out, dtype=torch.bfloat16, device=x.device)
        # The kernel family is chosen with the workspace: it depends on the
        # workspace key, the device (SM count, capability) and the per-call
        # environment override.
        device_index = _cuda_device_index(x)
        arch = _target_arch(device_index)
        sm_count = _sm_count(device_index)
        workspace = self._get_workspace(
            device=x.device,
            batch=batch,
            seqlen=seqlen,
            nchunks=nchunks,
            num_segments=num_segments,
            num_sequences=num_sequences,
            from_cu_seqlens=from_cu_seqlens,
            sm_count=sm_count,
            capability=_ARCH_CAPABILITIES[arch],
            chunk_parallel_mode=_chunk_parallel_mode(),
        )
        program_name = workspace["program"]
        program = _PROGRAMS[program_name]
        self.last_program_name = program_name
        # The kernel needs valid storage even when final states are disabled.
        # When they are returned, allocate caller-owned storage up front so
        # repeated cached-runner calls do not alias and no post-kernel device
        # copy adds another GPU activity to the measured route.
        final_states_arg = (
            torch.empty_like(workspace["final"])
            if return_final_states
            else workspace["final"]
        )
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
        # CAKE-1063: calls shorter than one chunk.  The x/B/C/out tensor maps
        # carry a fixed 128-row box on the token axis and the generated host
        # rejects a global extent below the box, so when ``seqlen < 128`` the
        # maps are bound to the runner-owned one-chunk buffers of the
        # workspace: x/B/C are zero padded (valid rows copied in with the
        # other pending copies), ``out`` is staged and its valid rows are
        # copied back after the launch.  Every other argument stays logical
        # (``seqlen``, ``nchunks = 1``, dt, z, the preprocess tables, the
        # sequence metadata), so the kernel sees exactly the state of a
        # partial trailing chunk of a longer call: the TMA zero fill past
        # ``seqlen`` becomes explicit zero rows, the loader publishes a flat
        # cumsum and a zero delta for those slots (batched) or clips the
        # segment to ``seqlen`` (varlen), and ``z`` -- read through its
        # pointer with the logical ``seqlen`` stride -- is guarded by
        # ``row < chunk_tokens``.  The pad rows contribute exact zeros and
        # the final states are unchanged.
        staged_out = None
        if seqlen < _CHUNK_SIZE:

            def padded(name: str, value: torch.Tensor) -> torch.Tensor:
                buffer = workspace[f"padded_{name}"]
                pending_copies.append((buffer[:, :seqlen], value))
                return buffer

            x = padded("x", x)
            B = padded("B", B)
            C = padded("C", C)
            staged_out = workspace["padded_out"]
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
                else self._seq_chunk_cumsum_buffer(x.device, num_sequences)
            )
        elif seq_chunk_cumsum is None:
            write_seq_chunk_cumsum = True
            cumsum_arg = self._seq_chunk_cumsum_buffer(x.device, num_sequences)
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
        kernel_out = out if staged_out is None else staged_out
        main_values: dict[str, object] = {
            "x_map": x,
            "b_map": B,
            "c_map": C,
            "out_map": kernel_out,
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
            "out_native": kernel_out,
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
        if program.family == "chunkpar":
            # The state operand workspace is bound twice: as the ``h_map``
            # tensor map (the host builds the TMA descriptor from the bf16
            # [tiles, 64, 128] tensor) and as the u32 word view the phase-2
            # chains store through.
            main_values.update(
                h_map=workspace["h_work"],
                s_work=workspace["s_work"],
                h_words=workspace["h_words"],
                grid_barrier=workspace["grid_barrier"],
            )
        # Every host-side decision is made; from here on only device work is
        # issued: the packed-input copies, then the single launcher call that
        # runs the preprocess and the scan, then (calls shorter than one
        # chunk only) the stream-ordered copy of the staged output rows.
        with torch.cuda.device(x.device):
            for destination, source in pending_copies:
                destination.copy_(source)
            _launch_program(
                program_name,
                arch,
                preprocess=preprocess_values,
                preprocess_grid=preprocess_grid,
                main=main_values,
                main_grid=(grid, 1, 1),
                cuda_stream=int(torch.cuda.current_stream(x.device).cuda_stream),
            )
            if staged_out is not None:
                out.copy_(staged_out[:, :seqlen])
        final = final_states_arg if return_final_states else None
        return out, final


__all__ = ["CakeSSDCombined"]
