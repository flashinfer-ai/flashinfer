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

JIT loader of the Cake fused row-wise FP8 GEMM + all-reduce (SM100, four ranks).

One generated program: the persistent two-CTA tcgen05 GEMM whose epilogue
stores each owned bf16 partial tile into this rank's output and publishes every
other partial tile into the chunk owner's symmetric receive slot, whose reducer
warps sum the four partials in fixed rank order and broadcast the result, and
whose drainer warps pull the non-owned chunks (``PROGRAMS`` records
its device and binding translation units, launch block, cluster shape, dynamic
shared memory and the positional argument names of its ``run`` launcher;
``ROUTES`` maps the stage key ``main_ws4_p2p`` to it). The program is compiled
once for its exact architecture together with its thin tvm-ffi launcher.
``COMM_MODE`` / ``TRANSPORT`` / ``TIMEOUT_CYCLES_LOG2`` / ``NUM_STAGES`` record
the choices frozen into the generated program (the unicast p2p carrier, the
pull broadcast, the bounded-wait exponent, the ring depth); they are not knobs.
The tables are filled by the exporter from the program bundle; the loader never
hashes sources or interprets argument plans.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, NamedTuple

import torch

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags

# Filled mechanically from the program bundle.
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_fp8_rowwise_gemm_allreduce_40d34f41bf331a3a97a7": {
        "sources": [
            "csrc/cake_fp8_rowwise_gemm_allreduce/cake_fp8_rowwise_gemm_allreduce_40d34f41bf331a3a97a7_kernel.cu",
            "csrc/cake_fp8_rowwise_gemm_allreduce/cake_fp8_rowwise_gemm_allreduce_40d34f41bf331a3a97a7_binding.cu",
        ],
        "arches": ["sm_100a"],
        "block": [384, 1, 1],
        "cluster": [2, 1, 1],
        "dynamic_smem_bytes": 222208,
        "args": [
            "A",
            "W",
            "OUT",
            "out_ptr",
            "sa",
            "sw",
            "ctl",
            "S_mc",
            "M",
            "N",
            "K",
            "grid_n",
            "pair_tiles",
            "K_tiles",
            "C",
            "C_cap",
            "S_bytes",
            "flag_words",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "S",
            "S_peers",
            "flags",
            "flags_peers",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
    },
}
ROUTES: dict[str, str] = {
    "main_ws4_p2p": "cake_fp8_rowwise_gemm_allreduce_40d34f41bf331a3a97a7",
}
COMPILE_FLAGS: list[str] = []
# Frozen trace-time choices of the generated program (documentation, never knobs).
COMM_MODE: str = "p2p"
TRANSPORT: str = "pull"
TIMEOUT_CYCLES_LOG2: int = 34
NUM_STAGES: int = 4

SOURCE_PACKAGE = "cake_fp8_rowwise_gemm_allreduce"
# The kernel is tcgen05 / TMEM / two-CTA MMA code compiled for its exact architecture.
ARCH_FLAGS: dict[str, list[str]] = {"sm_100a": sm100a_nvcc_flags}
CAPABILITY_ARCH: dict[tuple[int, int], str] = {(10, 0): "sm_100a"}
SUPPORTED_WORLD_SIZES: tuple[int, ...] = (4,)
EXCHANGE_MODE = "X16"
# Tile geometry of the generated program (mirrors the Cake export adapter).
BLOCK_M = 128
PAIR_TILE_M = 256
BLOCK_N = 256
BLOCK_K = 128
THREADS = 384
CLUSTER_DIMS: tuple[int, int, int] = (2, 1, 1)
OUT_BYTES = 2
# One 128 x 256 bf16 tile image in the symmetric receive workspace.
TILE_BYTES = BLOCK_M * BLOCK_N * OUT_BYTES
# Readiness words per chunk: ``pub_count[c]`` and ``bc[c]`` (two uint32).
FLAG_WORD_BYTES_PER_CHUNK = 8
# Local control words: ``arr[C_cap] | red_done[C_cap] | cta_arrivals[MAX_CTAS] |
# exit ticket | EP | steal[C_cap]``; the grid never exceeds ``MAX_CTAS`` CTAs.
MAX_CTAS = 1024
CTL_EXTRA_WORDS = 2
MAX_ROWS_TIMES_COLS = 1 << 30
# Shared-memory opt-in ceiling of the exported architecture (the program's
# dynamic shared memory stays below it).
MAX_DYNAMIC_SMEM_BYTES = 232448


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


def ceil128(rows: int) -> int:
    """``rows`` rounded up to the 128-row chunk."""

    rows = int(rows)
    if rows <= 0:
        raise ValueError(f"rows must be positive, got {rows}")
    return (rows + BLOCK_M - 1) // BLOCK_M * BLOCK_M


def chunk_count(m: int) -> int:
    """``C = ceil128(M) / 128`` chunks of 128 rows x N."""

    return ceil128(m) // BLOCK_M


def chunk_capacity(max_rows: int) -> int:
    """``C_cap = ceil128(max_rows) / 128``: sizes the receive workspace and the flag words."""

    return ceil128(max_rows) // BLOCK_M


def grid_n(n: int) -> int:
    """Column tiles ``N / 256``."""

    n = int(n)
    if n <= 0 or n % BLOCK_N:
        raise ValueError(f"N must be a positive multiple of {BLOCK_N}, got {n}")
    return n // BLOCK_N


def pair_tiles(m: int, n: int) -> int:
    """Two-CTA pair tiles of one call: ``ceil(M / 256) * N / 256``."""

    m = int(m)
    if m <= 0:
        raise ValueError(f"M must be positive, got {m}")
    return -(-m // PAIR_TILE_M) * grid_n(n)


def slots_per_chunk(world_size: int) -> int:
    """Slots per chunk in the symmetric workspace: one per sending peer plus the owner's result slot.

    The ``world_size - 1`` sender slots receive the peers' partial tiles; the last slot
    holds the reduced result the peers pull and is never written by a sender, so a peer's
    next-call partial cannot overwrite a result another peer is still pulling.
    """

    world_size = int(world_size)
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"world size must be one of {SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    return world_size


def workspace_bytes(max_rows: int, n: int, world_size: int) -> int:
    """Symmetric receive workspace per rank: ``C_cap x slots x grid_n x 64 KiB``.

    Equals ``2 * N * world_size * ceil128(max_rows)`` bytes (four slots per chunk
    at four ranks: three sender slots and the owner's result slot).
    """

    return (
        chunk_capacity(max_rows) * slots_per_chunk(world_size) * grid_n(n) * TILE_BYTES
    )


def flag_words(max_rows: int) -> int:
    """Readiness words of one rank: ``2 * C_cap`` uint32."""

    return 2 * chunk_capacity(max_rows)


def flag_bytes(max_rows: int) -> int:
    return FLAG_WORD_BYTES_PER_CHUNK * chunk_capacity(max_rows)


def ctl_words(c_cap: int) -> int:
    """Local control words for a chunk capacity ``c_cap``."""

    c_cap = int(c_cap)
    if c_cap <= 0:
        raise ValueError(f"C_cap must be positive, got {c_cap}")
    return 2 * c_cap + MAX_CTAS + CTL_EXTRA_WORDS + c_cap


def validate_shape(m: int, n: int, k_l: int) -> tuple[int, int, int]:
    """``M >= 1``, ``N % 256 == 0``, ``K_l % 128 == 0``, ``M * N <= 2**30``."""

    m, n, k_l = int(m), int(n), int(k_l)
    if m <= 0:
        raise ValueError(f"M must be positive, got {m}")
    grid_n(n)
    if k_l <= 0 or k_l % BLOCK_K:
        raise ValueError(f"K_l must be a positive multiple of {BLOCK_K}, got {k_l}")
    if m * n > MAX_ROWS_TIMES_COLS:
        raise ValueError(f"M * N = {m * n} exceeds 2**30")
    return m, n, k_l


def launch_grid(m: int, n: int, *, max_clusters: int) -> tuple[int, int, int]:
    """Persistent cluster grid: ``2 * min(co-resident clusters, pair tiles)`` CTAs.

    Every cluster of the grid must be co-resident (the exchange protocol waits
    on peers' tiles), so the grid never exceeds the device's co-resident
    two-CTA cluster count for this program (``cluster_capacity``).
    """

    max_clusters = int(max_clusters)
    if max_clusters <= 0:
        raise ValueError(f"max_clusters must be positive, got {max_clusters}")
    clusters = min(max_clusters, pair_tiles(m, n))
    return (2 * clusters, 1, 1)


def main_program(world_size: int) -> str:
    """The fused program of one world size (the unicast p2p carrier)."""

    key = f"main_ws{int(world_size)}_p2p"
    if key not in ROUTES:
        raise ValueError(
            f"no fused FP8 GEMM + all-reduce program is delivered for {key!r} "
            f"(world_size={int(world_size)})"
        )
    return ROUTES[key]


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / SOURCE_PACKAGE
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / SOURCE_PACKAGE
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "Cake fused FP8 GEMM + all-reduce CUDA sources were not found. Checked:\n"
        f"  - {installed}\n  - {checkout}"
    )


def _source_path(relative: str) -> Path:
    parts = Path(relative).parts
    if parts[:2] != ("csrc", SOURCE_PACKAGE) or len(parts) != 3:
        raise ValueError(
            f"exported source path is outside the fused FP8 GEMM + all-reduce package: {relative!r}"
        )
    return _source_dir() / parts[2]


@functools.cache
def spec(program: str, arch: str) -> JitSpec:
    """Build specification of one program for one exact architecture."""

    record = PROGRAMS[program]
    if arch not in record["arches"]:
        raise ValueError(
            f"program {program!r} is delivered for {record['arches']}, not {arch!r}"
        )
    return gen_jit_spec(
        name=f"{SOURCE_PACKAGE}_{arch}_{program}",
        sources=[_source_path(relative) for relative in record["sources"]],
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


def cluster_dims(program: str) -> tuple[int, int, int]:
    cluster = PROGRAMS[program]["cluster"]
    return (int(cluster[0]), int(cluster[1]), int(cluster[2]))


def dynamic_smem_bytes(program: str) -> int:
    return int(PROGRAMS[program]["dynamic_smem_bytes"])


def launcher_args(program: str) -> tuple[str, ...]:
    """Positional argument names of the program's ``run`` launcher (grid triple last)."""

    return tuple(PROGRAMS[program]["args"])


_PROBE_KERNEL = "cake_fp8_rowwise_gemm_allreduce_cluster_capacity_probe"
_PROBE_SOURCE = f'extern "C" __global__ void {_PROBE_KERNEL}() {{}}\n'


@functools.cache
def cluster_capacity(device_index: int, program: str) -> int:
    """Co-resident two-CTA clusters of ``program`` on ``device_index``.

    ``cuOccupancyMaxActiveClusters`` on an empty probe kernel launched with the
    program's block, cluster shape and dynamic shared memory. The shared memory
    pins one CTA per SM (as the program itself does), so the answer follows the
    part's GPC topology after floorsweeping, not ``SMs // 2``: it is the number
    the persistent grid must not exceed for every cluster to be resident.
    """

    from ..cuda_utils import checkCudaErrors, driver, nvrtc

    block = launch_block(program)
    cluster = cluster_dims(program)
    smem = dynamic_smem_bytes(program)
    with torch.cuda.device(device_index):
        torch.empty(1, device="cuda")  # primary context
        major, minor = torch.cuda.get_device_capability(device_index)
        prog = checkCudaErrors(
            nvrtc.nvrtcCreateProgram(_PROBE_SOURCE.encode(), b"probe.cu", 0, [], [])
        )
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode()]
        checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
        size = checkCudaErrors(nvrtc.nvrtcGetCUBINSize(prog))
        cubin = b" " * size
        checkCudaErrors(nvrtc.nvrtcGetCUBIN(prog, cubin))
        checkCudaErrors(nvrtc.nvrtcDestroyProgram(prog))
        ctx = checkCudaErrors(driver.cuDevicePrimaryCtxRetain(device_index))
        try:
            checkCudaErrors(driver.cuCtxSetCurrent(ctx))
            module = checkCudaErrors(driver.cuModuleLoadData(cubin))
            try:
                func = checkCudaErrors(
                    driver.cuModuleGetFunction(module, _PROBE_KERNEL.encode())
                )
                checkCudaErrors(
                    driver.cuFuncSetAttribute(
                        func,
                        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        smem,
                    )
                )
                attr = driver.CUlaunchAttribute()
                attr.id = (
                    driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
                )
                attr.value.clusterDim.x = cluster[0]
                attr.value.clusterDim.y = cluster[1]
                attr.value.clusterDim.z = cluster[2]
                config = driver.CUlaunchConfig()
                config.gridDimX = cluster[0]
                config.gridDimY = cluster[1]
                config.gridDimZ = cluster[2]
                config.blockDimX = block[0]
                config.blockDimY = block[1]
                config.blockDimZ = block[2]
                config.sharedMemBytes = smem
                config.attrs = [attr]
                config.numAttrs = 1
                capacity = int(
                    checkCudaErrors(driver.cuOccupancyMaxActiveClusters(func, config))
                )
            finally:
                checkCudaErrors(driver.cuModuleUnload(module))
        finally:
            checkCudaErrors(driver.cuDevicePrimaryCtxRelease(device_index))
    if capacity <= 0:
        raise RuntimeError(
            f"device {device_index} holds no co-resident {cluster} cluster of program {program!r}"
        )
    return capacity


__all__ = [
    "ARCH_FLAGS",
    "BLOCK_K",
    "BLOCK_M",
    "BLOCK_N",
    "CAPABILITY_ARCH",
    "CLUSTER_DIMS",
    "COMM_MODE",
    "COMPILE_FLAGS",
    "CTL_EXTRA_WORDS",
    "EXCHANGE_MODE",
    "FLAG_WORD_BYTES_PER_CHUNK",
    "MAX_CTAS",
    "MAX_DYNAMIC_SMEM_BYTES",
    "MAX_ROWS_TIMES_COLS",
    "NUM_STAGES",
    "OUT_BYTES",
    "PAIR_TILE_M",
    "PROGRAMS",
    "ROUTES",
    "SOURCE_PACKAGE",
    "SUPPORTED_WORLD_SIZES",
    "THREADS",
    "TILE_BYTES",
    "TIMEOUT_CYCLES_LOG2",
    "TRANSPORT",
    "DeviceFacts",
    "ceil128",
    "chunk_capacity",
    "chunk_count",
    "cluster_capacity",
    "cluster_dims",
    "ctl_words",
    "device_facts",
    "dynamic_smem_bytes",
    "flag_bytes",
    "flag_words",
    "grid_n",
    "launch_block",
    "launch_grid",
    "launcher_args",
    "load",
    "main_program",
    "pair_tiles",
    "slots_per_chunk",
    "spec",
    "validate_shape",
    "workspace_bytes",
]
