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
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_all_gather_matmul_0d2303b9569e9213c642": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_0d2303b9569e9213c642_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_0d2303b9569e9213c642_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_18d5d4b7d46bb61cc447": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_18d5d4b7d46bb61cc447_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_18d5d4b7d46bb61cc447_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [32, 1, 1],
        "dynamic_smem_bytes": 0,
    },
    "cake_all_gather_matmul_4cd953b96b2b21f3489d": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_4cd953b96b2b21f3489d_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_4cd953b96b2b21f3489d_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_790c03c0287641404c2c": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_790c03c0287641404c2c_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_790c03c0287641404c2c_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_9303668b92f5f5b46f87": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_9303668b92f5f5b46f87_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_9303668b92f5f5b46f87_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_932f3826fd5c3c9dcf30": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_932f3826fd5c3c9dcf30_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_932f3826fd5c3c9dcf30_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [32, 1, 1],
        "dynamic_smem_bytes": 0,
    },
    "cake_all_gather_matmul_93966f91063fc84be3c6": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_93966f91063fc84be3c6_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_93966f91063fc84be3c6_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
    "cake_all_gather_matmul_bda23d0402c5309b99dd": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bda23d0402c5309b99dd_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_bda23d0402c5309b99dd_binding.cu",
        ],
        "arches": ["sm_103a"],
        "block": [128, 1, 1],
        "dynamic_smem_bytes": 0,
    },
    "cake_all_gather_matmul_f5e5cb3d3bc68357235b": {
        "sources": [
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f5e5cb3d3bc68357235b_kernel.cu",
            "csrc/cake_all_gather_matmul/cake_all_gather_matmul_f5e5cb3d3bc68357235b_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "block": [192, 1, 1],
        "dynamic_smem_bytes": 197632,
    },
}
ROUTES: dict[str, str] = {
    "barrier_p0": "cake_all_gather_matmul_932f3826fd5c3c9dcf30",
    "barrier_p1": "cake_all_gather_matmul_18d5d4b7d46bb61cc447",
    "fused_peer_copy": "cake_all_gather_matmul_bda23d0402c5309b99dd",
    "main_bfloat16_ws2": "cake_all_gather_matmul_0d2303b9569e9213c642",
    "main_bfloat16_ws4": "cake_all_gather_matmul_4cd953b96b2b21f3489d",
    "main_bfloat16_ws8": "cake_all_gather_matmul_9303668b92f5f5b46f87",
    "main_float16_ws2": "cake_all_gather_matmul_93966f91063fc84be3c6",
    "main_float16_ws4": "cake_all_gather_matmul_790c03c0287641404c2c",
    "main_float16_ws8": "cake_all_gather_matmul_f5e5cb3d3bc68357235b",
}
COMPILE_FLAGS: list[str] = ["--use_fast_math"]

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


def chunk_plan(rows: int) -> tuple[int, int]:
    """``(chunk_rows, num_chunks)`` of the push protocol for ``rows`` local rows."""

    chunk_rows = min(int(rows), CHUNK_ROWS)
    return chunk_rows, (int(rows) + chunk_rows - 1) // chunk_rows


def main_grid(rows: int, n: int, *, peer_partitions: int) -> tuple[int, int, int]:
    """Launch grid of the main kernel: one CTA per output tile of the first chunk."""

    chunk_rows, _ = chunk_plan(rows)
    return ((chunk_rows // BLOCK_M) * (int(n) // BLOCK_N), int(peer_partitions), 1)


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


def main_program(world_size: int, dtype_name: str) -> str:
    return ROUTES[f"main_{dtype_name}_ws{int(world_size)}"]


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
    "spec",
    "uses_fused_peer_copy",
]
