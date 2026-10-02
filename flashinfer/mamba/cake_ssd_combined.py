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
from typing import Optional, Tuple

import torch
from filelock import FileLock
from tvm_ffi import cpp

from ..jit import env as jit_env
from ..jit.mamba.seq_chunk_cumsum import gen_seq_chunk_cumsum_module

_CHUNK_SIZE = 128
_HEADDIM = 64
_DSTATE = 128

_ROUTE_EXACT_SCAN = "exact_scan"
_ROUTE_SHALLOW_VARLEN = "shallow_varlen"
_ROUTE_PREFIX_VARLEN = "prefix_varlen"
# The packed-varlen prefix program is promoted for the 128-head / 8-group
# domain; every other shallow packed-varlen input runs the direct program.
_PREFIX_ROUTE_SELECTED = True

_TARGET_ARCHS = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
_DEVICE_DIR = Path("generated") / "device"
_HOST_TEMPLATE = Path("generated") / "host" / "mamba_ssd_combined_sequence.cpp"


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
    """A two-launch program: metadata preprocess followed by the scan."""

    preprocess: _Kernel
    main: _Kernel
    state_dtype_code: int
    main_smem_bytes: int

    @property
    def kernels(self) -> tuple[_Kernel, _Kernel]:
        return (self.preprocess, self.main)


_SEGMENT_PREPROCESS = _Kernel(
    "factorized_persistent_segment_preprocess_7ae61d5f32",
    "kernel_factorized_persistent_segment_preprocess",
    threads=128,
    fast_math=False,
)
_PREFIX_PREPROCESS = _Kernel(
    "prefix_factorized_segment_preprocess_onewarp_68ea71ca2f",
    "kernel_prefix_factorized_segment_preprocess_onewarp",
    threads=32,
    fast_math=False,
)
# DLDataType codes of the state tensors: kDLFloat=2 (float16), kDLBfloat=4.
_STATE_DTYPE_CODES = {"bf16": 4, "f16": 2}
_EXACT_SMEM_BYTES = 231936
_SHALLOW_SMEM_BYTES = 149248


def _scan(module: str) -> _Kernel:
    """The scan kernel of ``generated/device/<module>.cu``."""

    symbol = module.rsplit("_", 1)[0]
    return _Kernel(module, f"kernel_{symbol}", threads=512, fast_math=True)


_PROGRAMS: dict[str, _Program] = {
    "exact_bf16_batched": _Program(
        _SEGMENT_PREPROCESS,
        _scan("mamba_ssd_q_tmem_alias_bf16_batched_152ad01e4f"),
        _STATE_DTYPE_CODES["bf16"],
        _EXACT_SMEM_BYTES,
    ),
    "exact_f16_batched": _Program(
        _SEGMENT_PREPROCESS,
        _scan("mamba_ssd_q_tmem_alias_f16_batched_8b5ef7d7eb"),
        _STATE_DTYPE_CODES["f16"],
        _EXACT_SMEM_BYTES,
    ),
    "exact_bf16_varlen": _Program(
        _SEGMENT_PREPROCESS,
        _scan("mamba_ssd_q_tmem_alias_bf16_varlen_351b79a64d"),
        _STATE_DTYPE_CODES["bf16"],
        _EXACT_SMEM_BYTES,
    ),
    "exact_f16_varlen": _Program(
        _SEGMENT_PREPROCESS,
        _scan("mamba_ssd_q_tmem_alias_f16_varlen_1895881324"),
        _STATE_DTYPE_CODES["f16"],
        _EXACT_SMEM_BYTES,
    ),
    "shallow_bf16_varlen": _Program(
        _SEGMENT_PREPROCESS,
        _scan("mamba_ssd_direct_preprocess_warp_sync_1212_bf16_varlen_c551a2b2f0"),
        _STATE_DTYPE_CODES["bf16"],
        _SHALLOW_SMEM_BYTES,
    ),
    "shallow_f16_varlen": _Program(
        _SEGMENT_PREPROCESS,
        _scan("mamba_ssd_direct_preprocess_warp_sync_1212_f16_varlen_ff78ede5d2"),
        _STATE_DTYPE_CODES["f16"],
        _SHALLOW_SMEM_BYTES,
    ),
    "prefix_bf16_varlen": _Program(
        _PREFIX_PREPROCESS,
        _scan("mamba_ssd_prefix_warp_sync_1212_bf16_varlen_r10_v1_a872ac4eb2"),
        _STATE_DTYPE_CODES["bf16"],
        _SHALLOW_SMEM_BYTES,
    ),
    "prefix_f16_varlen": _Program(
        _PREFIX_PREPROCESS,
        _scan("mamba_ssd_prefix_warp_sync_1212_f16_varlen_r10_v1_8c373c2436"),
        _STATE_DTYPE_CODES["f16"],
        _SHALLOW_SMEM_BYTES,
    ),
}

# Positional launcher ABI shared by every program: preprocess arguments, its
# grid, main arguments, its grid, then the explicit CUDA stream.
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
)
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
    "has_seq_chunk_cumsum",
    "seq_idx_int64",
    "D_mode",
    "has_z",
    "has_initial",
    "dt_softplus",
    "dt_min",
    "dt_max",
    "write_final_states",
    "checkpoint_state_count",
)


def _select_scan_route(
    *,
    mode_varlen: bool,
    num_logical_chunks: int,
    num_sequences: int,
    nheads: int,
    ngroups: int,
    dt_min: float,
    prefix_route_selected: bool,
) -> str:
    """Resolve the semantic route from host-known shape predicates."""

    shallow_varlen = mode_varlen and num_logical_chunks <= num_sequences
    if dt_min < 0.0 or not shallow_varlen:
        return _ROUTE_EXACT_SCAN
    if prefix_route_selected and nheads == 128 and ngroups == 8:
        return _ROUTE_PREFIX_VARLEN
    return _ROUTE_SHALLOW_VARLEN


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
    threads: int,
) -> tuple[dict[str, object], tuple[int, int, int]]:
    """Build the metadata-fused preprocess values and launch grid."""

    if threads <= 0:
        raise ValueError(f"preprocess thread count must be positive, got {threads}")
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
    }
    total_tiles = num_segments * nheads
    return values, ((total_tiles + threads - 1) // threads, 1, 1)


def _persistent_grid_size(*, total_work: int, sm_count: int) -> int:
    """Match the balanced-grid policy shared by shallow and prefix routes."""

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


def _sequence_arguments(
    preprocess: Mapping[str, object],
    preprocess_grid: tuple[int, int, int],
    main: Mapping[str, object],
    main_grid: tuple[int, int, int],
    cuda_stream: int,
) -> tuple[object, ...]:
    """Order the name-keyed stage values into the launcher's positional ABI."""

    return (
        *(preprocess[name] for name in _PREPROCESS_ARGS),
        *preprocess_grid,
        *(main[name] for name in _MAIN_ARGS),
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
    """Launch one program's preprocess and scan on the explicit stream."""

    _load_generated_program(name, arch).run(
        *_sequence_arguments(preprocess, preprocess_grid, main, main_grid, cuda_stream)
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


def _render_host_source(template: str, program: _Program) -> str:
    """Substitute one program's table values into the shared launcher."""

    values = {
        "CAKE_SSD_PREPROCESS_MODULE": program.preprocess.module,
        "CAKE_SSD_PREPROCESS_KERNEL": program.preprocess.kernel,
        "CAKE_SSD_PREPROCESS_THREADS": str(program.preprocess.threads),
        "CAKE_SSD_MAIN_MODULE": program.main.module,
        "CAKE_SSD_MAIN_KERNEL": program.main.kernel,
        "CAKE_SSD_STATE_DTYPE_CODE": str(program.state_dtype_code),
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
    source_dir = _source_dir()
    host_source = _render_host_source(
        (source_dir / _HOST_TEMPLATE).read_text(encoding="utf-8"), program
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


@functools.cache
def _seq_chunk_cumsum_module():
    return gen_seq_chunk_cumsum_module().build_and_load()


class CakeSSDCombined:
    """Source-built Cake implementation of the admitted SSDCombined domain.

    The kernels require chunk size 128, head dimension 64, state dimension
    128, BF16 inputs/outputs, BF16 or FP16 states, and SM100/SM103.
    Head and group counts are runtime values and may be any positive pair for
    which ``nheads`` is divisible by ``ngroups``.
    """

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
        if chunk_size != _CHUNK_SIZE or headdim != _HEADDIM or dstate != _DSTATE:
            raise ValueError(
                "Cake SSDCombined requires chunk_size=128, headdim=64, and dstate=128"
            )
        if nheads <= 0 or ngroups <= 0 or nheads % ngroups:
            raise ValueError(
                "Cake SSDCombined requires positive nheads divisible by ngroups"
            )
        if io_dtype != torch.bfloat16:
            raise ValueError("Cake SSDCombined requires bfloat16 IO")
        if state_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Cake SSDCombined state dtype must be bfloat16 or float16")
        if seq_idx_dtype not in (torch.int32, torch.int64):
            raise ValueError("Cake SSDCombined seq_idx dtype must be int32 or int64")
        _target_arch(torch.cuda.current_device())
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
            Tuple[Optional[int], int, int, int, int, int, torch.dtype]
        ] = None
        self._workspace: Optional[dict[str, torch.Tensor]] = None
        self._dummy_cache: dict[Tuple[Optional[int], torch.dtype], torch.Tensor] = {}
        self._seq_cumsum_key: Optional[Tuple[Optional[int], int]] = None
        self._seq_cumsum_buf: Optional[torch.Tensor] = None

    def _dummy(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device.index, dtype)
        value = self._dummy_cache.get(key)
        if value is None:
            value = torch.empty(1, dtype=dtype, device=device)
            self._dummy_cache[key] = value
        return value

    @staticmethod
    def _contiguous_input(
        workspace: dict[str, torch.Tensor],
        name: str,
        value: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Materialize public strided inputs in graph-stable runner storage."""

        if value is None or value.is_contiguous():
            return value
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
        buffer.copy_(value)
        return buffer

    def _get_workspace(
        self,
        *,
        device: torch.device,
        batch: int,
        seqlen: int,
        nchunks: int,
        num_segments: int,
        num_sequences: int,
    ):
        key = (
            device.index,
            batch,
            seqlen,
            nchunks,
            num_segments,
            num_sequences,
            self.state_dtype,
        )
        if self._workspace_key != key:
            ids = torch.arange(num_segments, dtype=torch.int32, device=device)
            starts = (ids // nchunks) * seqlen + (ids % nchunks) * _CHUNK_SIZE
            lengths = torch.full(
                (num_segments,), _CHUNK_SIZE, dtype=torch.int32, device=device
            )
            sequence_offsets = (
                torch.arange(num_sequences + 1, dtype=torch.int32, device=device)
                * nchunks
            )
            tile_count = num_segments * self.nheads
            self._workspace = {
                "starts": starts,
                "lengths": lengths,
                "sequence_offsets": sequence_offsets,
                "delta": torch.empty(
                    (tile_count, _CHUNK_SIZE), dtype=torch.bfloat16, device=device
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
            self._workspace_key = key
        workspace = self._workspace
        assert workspace is not None
        return workspace

    def _compute_seq_chunk_cumsum(
        self,
        seq_idx: torch.Tensor,
        chunk_indices: torch.Tensor,
        chunk_offsets: torch.Tensor,
        num_sequences: int,
        output: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if output is None:
            size = num_sequences + 1
            key = (seq_idx.device.index, size)
            if self._seq_cumsum_key != key:
                self._seq_cumsum_buf = torch.empty(
                    size, dtype=torch.int32, device=seq_idx.device
                )
                self._seq_cumsum_key = key
            output = self._seq_cumsum_buf
            assert output is not None
        module = _seq_chunk_cumsum_module()
        workspace_bytes = module.seq_chunk_cumsum_tile_state_size(num_sequences)
        tile_state = (
            torch.empty(workspace_bytes, dtype=torch.uint8, device=seq_idx.device)
            if workspace_bytes
            else None
        )
        module.seq_chunk_cumsum(
            seq_idx,
            chunk_indices,
            chunk_offsets,
            output,
            tile_state,
            _CHUNK_SIZE,
            len(chunk_indices),
            num_sequences,
        )
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
    ):
        batch, seqlen, nheads, headdim = x.shape
        if seqlen % _CHUNK_SIZE:
            raise ValueError("seqlen must be divisible by chunk_size=128")
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
        if mode_varlen and any(value is None for value in metadata):
            raise ValueError(
                "varlen mode requires seq_idx, chunk_indices, and chunk_offsets"
            )
        if not mode_varlen and (
            any(value is not None for value in metadata) or seq_chunk_cumsum is not None
        ):
            raise ValueError(
                "batched mode does not accept varlen metadata or seq_chunk_cumsum"
            )
        if mode_varlen and initial_states is None:
            raise ValueError("varlen mode requires initial_states")
        if initial_states is not None and initial_states.dtype != self.state_dtype:
            raise ValueError("initial_states dtype must match state_dtype")

        nchunks = seqlen // _CHUNK_SIZE
        num_sequences = initial_states.shape[0] if mode_varlen else batch
        num_segments = len(chunk_indices) if mode_varlen else batch * nchunks
        dt_min, dt_max = (float(value) for value in dt_limit)
        scan_route = _select_scan_route(
            mode_varlen=mode_varlen,
            num_logical_chunks=num_segments,
            num_sequences=num_sequences,
            nheads=self.nheads,
            ngroups=self.ngroups,
            dt_min=dt_min,
            prefix_route_selected=_PREFIX_ROUTE_SELECTED,
        )
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
        ):
            raise ValueError("seq_chunk_cumsum shape or dtype is invalid")
        if out is not None:
            expected_out = (
                batch,
                self.nheads,
                _HEADDIM,
                nchunks,
                _CHUNK_SIZE,
            )
            if tuple(out.shape) != expected_out or out.dtype != torch.bfloat16:
                raise ValueError(
                    f"out must have shape {expected_out} and dtype bfloat16"
                )
            if not out.is_contiguous():
                raise ValueError("out must be contiguous")
        if out is None:
            # Match SSDCombined's ownership contract: each allocation-returning
            # call owns fresh output storage that later calls cannot overwrite.
            out = torch.empty(
                (batch, self.nheads, _HEADDIM, nchunks, _CHUNK_SIZE),
                dtype=torch.bfloat16,
                device=x.device,
            )
        workspace = self._get_workspace(
            device=x.device,
            batch=batch,
            seqlen=seqlen,
            nchunks=nchunks,
            num_segments=num_segments,
            num_sequences=num_sequences,
        )
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
        # graph-stable storage.
        dt = self._contiguous_input(workspace, "dt", dt)
        A = self._contiguous_input(workspace, "A", A)
        D = self._contiguous_input(workspace, "D", D)
        z = self._contiguous_input(workspace, "z", z)
        dt_bias = self._contiguous_input(workspace, "dt_bias", dt_bias)
        initial_states = self._contiguous_input(
            workspace, "initial_states", initial_states
        )
        seq_idx = self._contiguous_input(workspace, "seq_idx", seq_idx)
        chunk_indices = self._contiguous_input(
            workspace, "chunk_indices", chunk_indices
        )
        chunk_offsets = self._contiguous_input(
            workspace, "chunk_offsets", chunk_offsets
        )
        checkpoint_token_indices = self._contiguous_input(
            workspace, "checkpoint_token_indices", checkpoint_token_indices
        )
        checkpoint_state_slots = self._contiguous_input(
            workspace, "checkpoint_state_slots", checkpoint_state_slots
        )
        assert x is not None and dt is not None and A is not None
        assert B is not None and C is not None
        device_index = _cuda_device_index(x)
        arch = _target_arch(device_index)
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
        if mode_varlen and (seq_chunk_cumsum is None or update_seq_chunk_cumsum):
            with torch.cuda.device(x.device):
                seq_chunk_cumsum = self._compute_seq_chunk_cumsum(
                    seq_idx,
                    chunk_indices,
                    chunk_offsets,
                    num_sequences,
                    seq_chunk_cumsum,
                )

        dt_float = dt
        if dt.dtype != torch.float32:
            workspace["dt_float"].copy_(dt)
            dt_float = workspace["dt_float"]
        if dt_bias is not None and (
            tuple(dt_bias.shape) != (self.nheads,)
            or dt_bias.dtype not in (torch.bfloat16, torch.float32)
        ):
            raise ValueError(
                f"dt_bias must have shape [{self.nheads}] and dtype bfloat16 or float32"
            )
        if dt_bias is None:
            workspace["dt_bias_float"].zero_()
            dt_bias_float = workspace["dt_bias_float"]
        elif dt_bias.dtype == torch.float32:
            dt_bias_float = dt_bias
        else:
            workspace["dt_bias_float"].copy_(dt_bias)
            dt_bias_float = workspace["dt_bias_float"]
        state_key = "f16" if self.state_dtype == torch.float16 else "bf16"
        mode_key = "varlen" if mode_varlen else "batched"
        family = (
            "prefix"
            if scan_route == _ROUTE_PREFIX_VARLEN
            else "shallow"
            if scan_route == _ROUTE_SHALLOW_VARLEN
            else "exact"
        )
        program_name = f"{family}_{state_key}_{mode_key}"
        program = _PROGRAMS[program_name]
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
            threads=program.preprocess.threads,
        )
        sm_count = _sm_count(device_index)
        if scan_route == _ROUTE_PREFIX_VARLEN:
            preprocess_grid = (
                max(1, min(preprocess_grid[0], sm_count)),
                preprocess_grid[1],
                preprocess_grid[2],
            )
        grid = _persistent_grid_size(
            total_work=num_sequences * self.nheads,
            sm_count=sm_count,
        )

        d_arg = D if D is not None else self._dummy(x.device, torch.bfloat16)
        if D is not None and D.ndim == 2 and not self.d_has_hdim:
            # Match CuTe's public runner: a 2D D passed to a per-head
            # constructor consumes its first column.
            workspace["d_head"].copy_(D[:, 0])
            d_arg = workspace["d_head"]
        z_arg = z if z is not None else self._dummy(x.device, torch.bfloat16)
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
        checkpoint_token_indices_arg = (
            checkpoint_token_indices
            if checkpoint_token_indices is not None
            else self._dummy(x.device, torch.int32)
        )
        checkpoint_state_slots_arg = (
            checkpoint_state_slots
            if checkpoint_state_slots is not None
            else self._dummy(x.device, torch.int32)
        )
        cumsum_arg = (
            seq_chunk_cumsum
            if seq_chunk_cumsum is not None
            else self._dummy(x.device, torch.int32)
        )
        d_mode = 0 if D is None else 2 if self.d_has_hdim and D.ndim == 2 else 1
        with torch.cuda.device(x.device):
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
                "num_logical_chunks": num_segments if mode_varlen else nchunks,
                "mode_varlen": int(mode_varlen),
                "has_seq_chunk_cumsum": int(seq_chunk_cumsum is not None),
                "seq_idx_int64": int(seq_idx_int64),
                "D_mode": d_mode,
                "has_z": int(z is not None),
                "has_initial": int(initial_states is not None),
                "dt_softplus": int(bool(dt_softplus)),
                "dt_min": dt_min,
                "dt_max": dt_max,
                "write_final_states": int(return_final_states),
                "checkpoint_state_count": checkpoint_state_count,
            }
            _launch_program(
                program_name,
                arch,
                preprocess=preprocess_values,
                preprocess_grid=preprocess_grid,
                main=main_values,
                main_grid=(grid, 1, 1),
                cuda_stream=int(torch.cuda.current_stream(x.device).cuda_stream),
            )
        out_view = out.permute(0, 3, 4, 1, 2).reshape(
            batch, seqlen, self.nheads, _HEADDIM
        )
        final = final_states_arg if return_final_states else None
        return out_view, final


__all__ = ["CakeSSDCombined"]
