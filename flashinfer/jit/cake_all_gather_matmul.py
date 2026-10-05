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

JIT loader of the Cake Blackwell all-gather matmul (SM100 / SM103).

Every program is one generated CUDA device translation unit plus one tvm-ffi
launcher, shared by both architectures. The main kernels are tcgen05 code, so
each program is compiled once per exact architecture (``sm_100a`` or
``sm_103a``) into a library named ``<program>_<arch>``. ``PROGRAMS``,
``ROUTES`` and ``COMPILE_FLAGS`` are filled by the exporter from the program
bundle; the loader never hashes sources or interprets argument plans.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, NamedTuple

import torch

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

# Filled mechanically from the program bundle.
PROGRAMS: dict[str, dict[str, Any]] = {}
ROUTES: dict[str, str] = {}
COMPILE_FLAGS: list[str] = []

SOURCE_PACKAGE = "cake_all_gather_matmul"
# The kernels are tcgen05 / TMEM code; one source serves both capabilities,
# each compiled with its exact architecture flag set.
ARCH_FLAGS: dict[str, list[str]] = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}
CAPABILITY_ARCH: dict[tuple[int, int], str] = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
SUPPORTED_WORLD_SIZES: tuple[int, ...] = (2, 4, 8)
SUPPORTED_DTYPES: dict[torch.dtype, str] = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
}
K = 8192
BLOCK_M = 128
BLOCK_N = 256
# Rows pushed per peer chunk; the main kernel consumes chunk ``c`` of a peer
# after the matching readiness word reaches the launch epoch.
CHUNK_ROWS = 19 * BLOCK_M
# Physical layouts of the logical ``[K, N]`` weight, each with its own main
# kernel: ``n_major`` = contiguous ``[K, N]``; ``k_major`` = the ``w.t()`` view
# of a contiguous ``[N, K]`` parameter.
B_LAYOUTS: tuple[str, ...] = ("n_major", "k_major")
MAIN_THREADS = 192
# Fused SM103 peer copy: CTAs per remote peer along grid x, one peer per grid y.
FUSED_COPY_CTAS_PER_PEER = 32
# Barrier flag pad: two phase epochs plus two sender-indexed mailbox banks per
# phase and rank (``2 + 2 * 2 * world_size`` uint32 words).
BARRIER_PHASES = 2
BARRIER_BANKS = 2


class DeviceFacts(NamedTuple):
    capability: tuple[int, int]
    arch: str | None
    sm_count: int


@functools.cache
def device_facts(device_index: int) -> DeviceFacts:
    """Compute capability, exported architecture and SM count, queried once per device."""

    properties = torch.cuda.get_device_properties(device_index)
    capability = (int(properties.major), int(properties.minor))
    return DeviceFacts(
        capability=capability,
        arch=CAPABILITY_ARCH.get(capability),
        sm_count=int(properties.multi_processor_count),
    )


def barrier_flag_words(world_size: int) -> int:
    """Number of uint32 words of one rank's barrier flag pad."""

    return BARRIER_PHASES + BARRIER_PHASES * BARRIER_BANKS * int(world_size)


def padded_rows(rows: int) -> int:
    """``rows`` rounded up to the 128-row MMA tile."""

    return (int(rows) + BLOCK_M - 1) // BLOCK_M * BLOCK_M


def chunk_plan(rows: int) -> tuple[int, int, int]:
    """``(padded_rows, chunk_rows, num_chunks)`` of the push protocol for ``rows`` local rows."""

    padded = padded_rows(rows)
    chunk_rows = min(padded, CHUNK_ROWS)
    return padded, chunk_rows, (padded + chunk_rows - 1) // chunk_rows


def main_grid(rows: int, n: int, *, peer_partitions: int) -> tuple[int, int, int]:
    """Launch grid of the main kernel: one CTA per output tile of the first (padded) chunk."""

    _padded, chunk_rows, _num_chunks = chunk_plan(rows)
    return ((chunk_rows // BLOCK_M) * (int(n) // BLOCK_N), int(peer_partitions), 1)


# ``grid.y`` of the narrow (N <= 2048) rows per world size, from the same-session
# y-sweeps on 8 x B200 / 8 x B300 (lever L2 of the engine-operand work): ws8 prefers every peer
# concurrently, ws4 and ws2 prefer two partitions.
_NARROW_ROW_PARTITIONS = {2: 2, 4: 2, 8: 8}


def peer_partitions(
    *, arch: str, dtype_name: str, world_size: int, rows: int, n: int, fused: bool
) -> int:
    """``grid.y`` of the main kernel: CTA partitions sharing the cyclic peer traversal.

    ``y = 1`` is the serial local-first traversal of the wide rows; the fused
    SM103 TP8 packed-QKV route partitions over four; latency-bound shapes (few
    CTAs per chunk: ``N <= 2048`` up to 8192 local rows, or one 128-row tile)
    run every peer concurrently (``y = world_size``).
    """

    del arch, dtype_name
    if fused:
        return 4
    n_tiles = n // BLOCK_N
    m_tiles = min(padded_rows(rows), CHUNK_ROWS) // BLOCK_M
    if m_tiles <= 1:
        return world_size
    if n_tiles <= 8 and rows <= 8192:
        return _NARROW_ROW_PARTITIONS[world_size]
    return 1


def weight_layout(w: torch.Tensor) -> str:
    """Classify the logical ``[K, N]`` weight view by its strides (no copy is ever made)."""

    if w.ndim != 2 or int(w.shape[0]) != K:
        raise ValueError(f"w must be a [{K}, N] view, got shape {tuple(w.shape)}")
    n = int(w.shape[1])
    stride_k, stride_n = (int(stride) for stride in w.stride())
    if stride_n == 1 and stride_k == n:
        return "n_major"
    if stride_k == 1 and stride_n == K:
        return "k_major"
    raise ValueError(
        f"w must be a contiguous [{K}, {n}] tensor or the transposed view of a "
        f"contiguous [{n}, {K}] tensor, got strides {(stride_k, stride_n)}"
    )


def weight_tma_source(w: torch.Tensor, b_layout: str) -> torch.Tensor:
    """The rank-3 tensor the main kernel's ``B`` tensor map is encoded from."""

    n = int(w.shape[1])
    if b_layout == "n_major":
        return w.view(1, K, n)
    return w.t().view(1, n, K)


def uses_fused_peer_copy(
    *, arch: str, dtype_name: str, world_size: int, rows: int, n: int
) -> bool:
    """The SM103 TP8 packed-QKV route pushes its payload with the fused SM copy kernel."""

    return (
        arch == "sm_103a"
        and dtype_name == "bfloat16"
        and int(world_size) == 8
        and int(rows) == 512
        and int(n) == 1280
    )


def barrier_program(phase: int) -> str:
    """The barrier of one phase; its world size is a runtime argument."""

    return ROUTES[f"barrier_p{int(phase)}"]


def main_program(world_size: int, dtype_name: str, b_layout: str) -> str:
    return ROUTES[f"main_{dtype_name}_ws{int(world_size)}_{b_layout}"]


def fused_peer_copy_program() -> str:
    return ROUTES["fused_peer_copy"]


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / SOURCE_PACKAGE
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / SOURCE_PACKAGE
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "Cake all-gather matmul CUDA sources were not found. Checked:\n"
        f"  - {installed}\n  - {checkout}"
    )


def _source_path(relative: str) -> Path:
    parts = Path(relative).parts
    if parts[:2] != ("csrc", SOURCE_PACKAGE) or len(parts) != 3:
        raise ValueError(
            f"exported source path is outside the all-gather matmul package: {relative!r}"
        )
    return _source_dir() / parts[2]


@functools.cache
def spec(program: str, arch: str) -> JitSpec:
    """Build specification of one program for one exact architecture."""

    row = PROGRAMS[program]
    if arch not in row["arches"]:
        raise ValueError(
            f"program {program!r} is delivered for {row['arches']}, not {arch!r}"
        )
    return gen_jit_spec(
        name=f"{program}_{arch}",
        sources=[_source_path(relative) for relative in row["sources"]],
        extra_cuda_cflags=[*ARCH_FLAGS[arch], *COMPILE_FLAGS],
        extra_include_paths=[_source_dir().parent],
        # The source build carries every math flag in ``COMPILE_FLAGS``; the
        # default ``-use_fast_math`` would otherwise be added or dropped.
        use_fast_math="--use_fast_math" in COMPILE_FLAGS,
    )


@functools.cache
def load(program: str, arch: str):
    return spec(program, arch).build_and_load()


def launch_block(program: str) -> tuple[int, int, int]:
    block = PROGRAMS[program]["block"]
    return (int(block[0]), int(block[1]), int(block[2]))


def dynamic_smem_bytes(program: str) -> int:
    return int(PROGRAMS[program]["dynamic_smem_bytes"])


__all__ = [
    "ARCH_FLAGS",
    "B_LAYOUTS",
    "BARRIER_BANKS",
    "BARRIER_PHASES",
    "BLOCK_M",
    "BLOCK_N",
    "CAPABILITY_ARCH",
    "CHUNK_ROWS",
    "COMPILE_FLAGS",
    "FUSED_COPY_CTAS_PER_PEER",
    "K",
    "MAIN_THREADS",
    "PROGRAMS",
    "ROUTES",
    "SUPPORTED_DTYPES",
    "SUPPORTED_WORLD_SIZES",
    "DeviceFacts",
    "barrier_flag_words",
    "barrier_program",
    "chunk_plan",
    "device_facts",
    "dynamic_smem_bytes",
    "fused_peer_copy_program",
    "launch_block",
    "load",
    "main_grid",
    "main_program",
    "padded_rows",
    "peer_partitions",
    "spec",
    "uses_fused_peer_copy",
    "weight_layout",
    "weight_tma_source",
]
