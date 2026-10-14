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

JIT loader of the Cake fused QKV projection + norm + RoPE + head-exchange
programs (SM90 / SM100 / SM103).

``PROGRAMS`` records every delivered program: its device and binding
translation units, the architectures it compiles for, its compile flags, the
launch block, the cluster shape, the dynamic shared memory and the positional
argument names of its ``run`` launcher.  ``ROUTES`` maps, per architecture, the
logical stage keys (``barrier_p0``, ``barrier_p1``, ``wait`` and
``producer_bn<bn>`` / ``producer_bn<bn>_pair``) to programs.  Both tables are
filled by the exporter from the program bundle; the loader never hashes sources
or interprets argument plans.  A program is compiled once per architecture with
that architecture's exact flag set, and the architecture is part of the JIT
spec name so two architectures never share one cached library.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, NamedTuple

import torch

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    sm90a_nvcc_flags,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)

# Filled mechanically from the program bundle.
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_qkv_norm_rope_exchange_overlap_092734d18ca9cefe21b8": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_092734d18ca9cefe21b8_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_092734d18ca9cefe21b8_binding.cu",
        ],
        "arches": ["sm_90a"],
        "compile_flags": [],
        "args": [
            "device_index",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [32, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 0,
    },
    "cake_qkv_norm_rope_exchange_overlap_0cc6f56e47ebab3fdb82": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_0cc6f56e47ebab3fdb82_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_0cc6f56e47ebab3fdb82_binding.cu",
        ],
        "arches": ["sm_90a"],
        "compile_flags": [],
        "args": [
            "device_index",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [32, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 0,
    },
    "cake_qkv_norm_rope_exchange_overlap_502e52054565032af6ad": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_502e52054565032af6ad_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_502e52054565032af6ad_binding.cu",
        ],
        "arches": ["sm_90a"],
        "compile_flags": [],
        "args": [
            "ready",
            "num_words_total",
            "slot_base",
            "ready_target",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [128, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 0,
    },
    "cake_qkv_norm_rope_exchange_overlap_70a5b6a09569b80313de": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_70a5b6a09569b80313de_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_70a5b6a09569b80313de_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [192, 1, 1],
        "cluster": [2, 1, 1],
        "dynamic_smem_bytes": 232448,
    },
    "cake_qkv_norm_rope_exchange_overlap_849dd94cf08ca290ee35": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_849dd94cf08ca290ee35_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_849dd94cf08ca290ee35_binding.cu",
        ],
        "arches": ["sm_90a"],
        "compile_flags": [],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [384, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 131136,
    },
    "cake_qkv_norm_rope_exchange_overlap_a4c464facca189a5a803": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_a4c464facca189a5a803_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_a4c464facca189a5a803_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [192, 1, 1],
        "cluster": [2, 1, 1],
        "dynamic_smem_bytes": 232448,
    },
    "cake_qkv_norm_rope_exchange_overlap_b1937d956988deed0eda": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_b1937d956988deed0eda_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_b1937d956988deed0eda_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [192, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 232448,
    },
    "cake_qkv_norm_rope_exchange_overlap_b1d866d329f2f38fa312": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_b1d866d329f2f38fa312_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_b1d866d329f2f38fa312_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "ready",
            "num_words_total",
            "slot_base",
            "ready_target",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [128, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 0,
    },
    "cake_qkv_norm_rope_exchange_overlap_b7fcf611d8f7bdc09ca2": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_b7fcf611d8f7bdc09ca2_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_b7fcf611d8f7bdc09ca2_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "device_index",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [32, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 0,
    },
    "cake_qkv_norm_rope_exchange_overlap_e6bc2a25bf377b02cb1e": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_e6bc2a25bf377b02cb1e_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_e6bc2a25bf377b02cb1e_binding.cu",
        ],
        "arches": ["sm_90a"],
        "compile_flags": [],
        "args": [
            "A",
            "B",
            "cos_sin",
            "q_scale",
            "k_scale",
            "tiles",
            "dst_ptrs",
            "ready_ptrs",
            "counters",
            "num_tiles",
            "M",
            "K",
            "S_local",
            "head_stride",
            "batch_stride",
            "slot_elem_offset",
            "pad_base",
            "num_words",
            "rank",
            "world",
            "ready_target",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [384, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 196672,
    },
    "cake_qkv_norm_rope_exchange_overlap_ee7af16c3fae716e45d2": {
        "sources": [
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_ee7af16c3fae716e45d2_kernel.cu",
            "csrc/cake_qkv_norm_rope_exchange_overlap/cake_qkv_norm_rope_exchange_overlap_ee7af16c3fae716e45d2_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--fmad=false"],
        "args": [
            "device_index",
            "pg_world",
            "pg_rank",
            "pg_flags",
            "grid_x",
            "grid_y",
            "grid_z",
        ],
        "block": [32, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 0,
    },
}
ROUTES: dict[str, dict[str, str]] = {
    "sm_90a": {
        "barrier_p0": "cake_qkv_norm_rope_exchange_overlap_0cc6f56e47ebab3fdb82",
        "barrier_p1": "cake_qkv_norm_rope_exchange_overlap_092734d18ca9cefe21b8",
        "producer_bn128": "cake_qkv_norm_rope_exchange_overlap_849dd94cf08ca290ee35",
        "producer_bn256": "cake_qkv_norm_rope_exchange_overlap_e6bc2a25bf377b02cb1e",
        "wait": "cake_qkv_norm_rope_exchange_overlap_502e52054565032af6ad",
    },
    "sm_100a": {
        "barrier_p0": "cake_qkv_norm_rope_exchange_overlap_ee7af16c3fae716e45d2",
        "barrier_p1": "cake_qkv_norm_rope_exchange_overlap_b7fcf611d8f7bdc09ca2",
        "producer_bn128": "cake_qkv_norm_rope_exchange_overlap_b1937d956988deed0eda",
        "producer_bn128_pair": "cake_qkv_norm_rope_exchange_overlap_a4c464facca189a5a803",
        "producer_bn256_pair": "cake_qkv_norm_rope_exchange_overlap_70a5b6a09569b80313de",
        "wait": "cake_qkv_norm_rope_exchange_overlap_b1d866d329f2f38fa312",
    },
    "sm_103a": {
        "barrier_p0": "cake_qkv_norm_rope_exchange_overlap_ee7af16c3fae716e45d2",
        "barrier_p1": "cake_qkv_norm_rope_exchange_overlap_b7fcf611d8f7bdc09ca2",
        "producer_bn128": "cake_qkv_norm_rope_exchange_overlap_b1937d956988deed0eda",
        "producer_bn128_pair": "cake_qkv_norm_rope_exchange_overlap_a4c464facca189a5a803",
        "producer_bn256_pair": "cake_qkv_norm_rope_exchange_overlap_70a5b6a09569b80313de",
        "wait": "cake_qkv_norm_rope_exchange_overlap_b1d866d329f2f38fa312",
    },
}

SOURCE_PACKAGE = "cake_qkv_norm_rope_exchange_overlap"
ARCH_FLAGS: dict[str, list[str]] = {
    "sm_90a": sm90a_nvcc_flags,
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}
CAPABILITY_ARCH: dict[tuple[int, int], str] = {
    (9, 0): "sm_90a",
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}
SUPPORTED_WORLD_SIZES: tuple[int, ...] = (2, 4, 8)
# Barrier flag pad: two phase epochs plus two sender-indexed mailbox banks per
# phase and rank (``2 + 2 * 2 * world_size`` uint32 words).
BARRIER_PHASES = 2
BARRIER_BANKS = 2
STAGE_KEYS: tuple[str, ...] = ("barrier_p0", "barrier_p1", "wait")


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


def producer_key(block_n: int, pair: bool) -> str:
    return f"producer_bn{int(block_n)}{'_pair' if pair else ''}"


def program_for(arch: str, key: str) -> str:
    """The program bound to one stage key on one architecture."""

    try:
        return ROUTES[arch][key]
    except KeyError:
        raise ValueError(
            f"no fused QKV exchange program is delivered for {key!r} on {arch}"
        ) from None


def has_producer(arch: str, block_n: int, pair: bool) -> bool:
    return producer_key(block_n, pair) in ROUTES.get(arch, {})


def producer_program(arch: str, block_n: int, pair: bool) -> str:
    return program_for(arch, producer_key(block_n, pair))


def barrier_program(arch: str, phase: int) -> str:
    if phase not in (0, 1):
        raise ValueError(f"barrier phase must be 0 or 1, got {phase!r}")
    return program_for(arch, f"barrier_p{phase}")


def wait_program(arch: str) -> str:
    return program_for(arch, "wait")


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / SOURCE_PACKAGE
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / SOURCE_PACKAGE
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "Cake fused QKV exchange CUDA sources were not found. Checked:\n"
        f"  - {installed}\n  - {checkout}"
    )


def _source_path(relative: str) -> Path:
    parts = Path(relative).parts
    if parts[:2] != ("csrc", SOURCE_PACKAGE) or len(parts) != 3:
        raise ValueError(
            f"exported source path is outside the fused QKV exchange package: {relative!r}"
        )
    return _source_dir() / parts[2]


def _include_paths() -> list[str | Path]:
    checkout_root = Path(__file__).resolve().parents[2]
    include_dir = jit_env.FLASHINFER_INCLUDE_DIR
    if not include_dir.is_dir():
        include_dir = checkout_root / "include"
    # ``gen_jit_spec`` declares ``extra_include_paths: list[str | Path]``; the
    # list is built under that annotation (mypy's ``list`` is invariant).
    paths: list[str | Path] = [
        _source_dir(),
        _source_dir().parent,
        jit_env.FLASHINFER_CSRC_DIR,
        include_dir,
    ]
    return list(dict.fromkeys(paths))


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
        extra_cuda_cflags=[*ARCH_FLAGS[arch], *record["compile_flags"]],
        extra_include_paths=_include_paths(),
        # The source build carries its own math flags in ``compile_flags``
        # (no fast math); the default ``-use_fast_math`` would change them.
        use_fast_math=False,
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
    """Positional argument names of the program's ``run`` launcher."""

    return tuple(PROGRAMS[program]["args"])


__all__ = [
    "ARCH_FLAGS",
    "BARRIER_BANKS",
    "BARRIER_PHASES",
    "CAPABILITY_ARCH",
    "PROGRAMS",
    "ROUTES",
    "SOURCE_PACKAGE",
    "STAGE_KEYS",
    "SUPPORTED_WORLD_SIZES",
    "DeviceFacts",
    "barrier_flag_words",
    "barrier_program",
    "cluster_dims",
    "device_facts",
    "dynamic_smem_bytes",
    "has_producer",
    "launch_block",
    "launcher_args",
    "load",
    "producer_key",
    "producer_program",
    "program_for",
    "spec",
    "wait_program",
]
