"""
Copyright (c) 2026 by NVIDIA CORPORATION.

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
from pathlib import Path

from ...jit import env as jit_env
from ...jit.core import JitSpec, gen_jit_spec, logger, sm100a_nvcc_flags

_UNION_GQA_URI = "union_gqa_prefill_sm100a"


def _get_union_gqa_csrc_dir() -> Path:
    """Locate the in-package union-GQA source bundle."""

    csrc_dir = Path(__file__).resolve().parent / "csrc"
    if csrc_dir.is_dir():
        return csrc_dir
    raise FileNotFoundError("union-GQA CUDA sources were not found")


def _get_union_gqa_include_dirs() -> list[Path]:
    """Resolve FlashInfer and CUTLASS headers for wheels and source trees."""

    if jit_env.FLASHINFER_INCLUDE_DIR.exists() and all(
        path.exists() for path in jit_env.CUTLASS_INCLUDE_DIRS
    ):
        return [
            jit_env.FLASHINFER_CSRC_DIR,
            jit_env.FLASHINFER_INCLUDE_DIR,
            *jit_env.CUTLASS_INCLUDE_DIRS,
        ]

    repo_root = Path(__file__).resolve().parents[3]
    checkout_dirs = [
        repo_root / "csrc",
        repo_root / "include",
        repo_root / "3rdparty" / "cutlass" / "include",
        repo_root / "3rdparty" / "cutlass" / "tools" / "util" / "include",
    ]
    if all(path.exists() for path in checkout_dirs):
        return checkout_dirs

    raise FileNotFoundError("FlashInfer/CUTLASS headers for union_gqa were not found")


@functools.cache
def gen_union_gqa_module(*, benchmark_q_prefix: int = 0) -> JitSpec:
    """Generate the exact-SM100a union_gqa prefill JIT module."""

    if benchmark_q_prefix < 0 or benchmark_q_prefix % 16:
        raise ValueError("benchmark_q_prefix must be nonnegative and divisible by 16")
    name = _UNION_GQA_URI
    benchmark_flags = []
    if benchmark_q_prefix:
        name += f"_benchmark_q_prefix_{benchmark_q_prefix}"
        benchmark_flags = [
            f"-DFLASHINFER_UNION_GQA_BENCHMARK_Q_PREFIX={benchmark_q_prefix}"
        ]
    csrc_dir = _get_union_gqa_csrc_dir()
    include_dirs = _get_union_gqa_include_dirs()
    target_dir = csrc_dir / "sm100a"
    binding = target_dir / "union_gqa_prefill_binding.cu"
    body = target_dir / "union_gqa_prefill.cu"
    helpers = (
        target_dir / "union_gqa_mma_noelect.cuh",
        target_dir / "union_gqa_prefill_api.cuh",
        target_dir / "union_gqa_utils.cuh",
    )
    for source in (binding, body, *helpers):
        if not source.is_file():
            raise FileNotFoundError(f"union_gqa CUDA source not found: {source}")

    spec = gen_jit_spec(
        name=name,
        sources=[body, binding],
        extra_cuda_cflags=[
            "-std=c++20",
            "--expt-extended-lambda",
            # CUDA 13's device optimizer crashes on this translation unit at
            # its default level. ptxas -O2 preserves the measured code
            # generation while avoiding the optimizer crash.
            "-Xptxas",
            "-O2",
            "-lineinfo",
            *sm100a_nvcc_flags,
            *benchmark_flags,
        ],
        extra_include_paths=[target_dir, *include_dirs],
    )
    logger.info("Generated union_gqa SM100a prefill JIT spec")
    return spec


@functools.cache
def load_union_gqa_module():
    """Build and load the exact-SM100a union_gqa prefill module."""

    module = gen_union_gqa_module().build_and_load()
    logger.info("Loaded union_gqa SM100a prefill module")
    return module


def get_union_gqa_module():
    """Return the loaded union_gqa module used by the Python API."""

    return load_union_gqa_module()


__all__ = [
    "gen_union_gqa_module",
    "get_union_gqa_module",
    "load_union_gqa_module",
]
