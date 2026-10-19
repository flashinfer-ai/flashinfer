# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""JIT loader for the generated Cake SM90 (Hopper) variable block-sparse attention kernels.

The sources live under ``csrc/cake_vsa_sm90``: one ``sm_90a`` device
translation unit per stage (BF16 HND, head_dim 128, 64-token blocks), one
shared tvm-ffi launcher source that every stage compiles with its own ``-D``
selection, and the shared device preamble and launcher-helper headers.
:data:`_STAGES` is the table of the shipped stages with their launch geometry
and source pins.  The Python planner that produces the kernels' tile metadata
lives in :mod:`flashinfer.cake_vsa_sm90`.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import NamedTuple

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm90a_nvcc_flags

_ARCH = "sm_90a"
_BINDING = "cake_vsa_sm90_binding.cu"
_BINDING_SHA256 = "d4e1e8555b067e7365a0cf102a6a39264c560b43498df83fc4d1f9b219cf214a"
_COMPILE_FLAGS = ("--use_fast_math",)


class _Stage(NamedTuple):
    """One shipped stage: module identity, launch geometry and source pin."""

    module: str
    """Generated module identity; ``kernel_<module>`` is its kernel symbol."""
    shim: str
    """Hash of the launcher's host-shim namespace."""
    threads: int
    smem_bytes: int
    cluster_x: int
    abi: int
    """1 = small-selection plan, 2 = split-KV plan (+ partials), 3 = persistent."""
    device_sha256: str


# Persistent pair/split route (static and queue-scheduled tile lists) plus the
# small-selection route (one CTA per query block, plan in the kernel-parameter
# constant bank) in four KMAX variants, its split-KV variants for KMAX 4 and 6
# (selections sliced across CTAs, last-CTA merge; the planner never selects a
# KMAX 1 or 3 split), and the cluster variants (slices of a query block as one
# thread-block cluster, merge through distributed shared memory).
_STAGES: dict[str, _Stage] = {
    "attention": _Stage(
        module="cake_vsa_sm90_7bffc3d851bf002de78f",
        shim="be3a5d6b05b219fe",
        threads=384,
        smem_bytes=232064,
        cluster_x=1,
        abi=3,
        device_sha256="8f3f731f38a4457c0f6735c76d0284f29aff807c89f69b96145262abc9fb0caa",
    ),
    "attention_queue": _Stage(
        module="cake_vsa_sm90_efbb759a1ac3a01e0317",
        shim="0b8765d6fa18a096",
        threads=384,
        smem_bytes=232192,
        cluster_x=1,
        abi=3,
        device_sha256="8370f26ac29416b3ce643d3064d67cd42975819866b35f232b2cb8e4d605bc55",
    ),
    "small_k1": _Stage(
        module="cake_vsa_sm90_d017790969e7c56b43e2",
        shim="e04438b222d78082",
        threads=128,
        smem_bytes=50176,
        cluster_x=1,
        abi=1,
        device_sha256="4a41d221dde3bba00ec73d6bc57734ed8b7e680f54c8b7958d42232cae437324",
    ),
    "small_k3": _Stage(
        module="cake_vsa_sm90_559ec9ab2e8bf979f3ea",
        shim="295dbb471f6024e7",
        threads=256,
        smem_bytes=116736,
        cluster_x=1,
        abi=1,
        device_sha256="e3219d516135ea0b818ab2323116960f43a89b3932f853cc3ed91a7fa1944691",
    ),
    "small_k4": _Stage(
        module="cake_vsa_sm90_b02f089c3462dac30ed2",
        shim="bd56c4878bb94e84",
        threads=256,
        smem_bytes=149504,
        cluster_x=1,
        abi=1,
        device_sha256="6f5293e0b86d9cc8b772347aa96c4ceef79e738a9349d8889a03b7b6fc0f2895",
    ),
    "small_k6": _Stage(
        module="cake_vsa_sm90_f1a608b42c8170eebc6e",
        shim="f43d3e57aad5575a",
        threads=256,
        smem_bytes=215040,
        cluster_x=1,
        abi=1,
        device_sha256="06161ac8f9c6bac461981e1c7eec6956b932d54930ae19c8495db2efcc049aa0",
    ),
    "small_k4s": _Stage(
        module="cake_vsa_sm90_06ec4e60074ac4de3893",
        shim="ddb240dc74a216d6",
        threads=256,
        smem_bytes=149632,
        cluster_x=1,
        abi=2,
        device_sha256="e461b24d75aede1e6d0e4acb9d71a80fdd1aa6f851e3c967d88bd6da7c3ebbe8",
    ),
    "small_k6s": _Stage(
        module="cake_vsa_sm90_01292822ce588fb86dd4",
        shim="e6c5ce3b5a4b1223",
        threads=256,
        smem_bytes=215168,
        cluster_x=1,
        abi=2,
        device_sha256="80b3dd9ad7672ad0fd62df080b9f0f6040a95ee5aef2dff41f1d91e8c06e4d33",
    ),
    "small_k2c4": _Stage(
        module="cake_vsa_sm90_306d746ed864b400ac35",
        shim="26cf4461539d9f87",
        threads=256,
        smem_bytes=111616,
        cluster_x=4,
        abi=1,
        device_sha256="09e0403ca63cc7573e8c53602ee4a5c2956339fe153453fbb492462b3ea61a7c",
    ),
    "small_k3c3": _Stage(
        module="cake_vsa_sm90_b2dbe19d96d3b988a27b",
        shim="d35d0eca5a7b094d",
        threads=256,
        smem_bytes=151552,
        cluster_x=3,
        abi=1,
        device_sha256="f39630d7aa9d44bdc24a4e8a998f7f1c10576100afc29c346cce31c50694a0c2",
    ),
    "small_k3c6": _Stage(
        module="cake_vsa_sm90_e62576d2c73d063c1f11",
        shim="10a402374ad02093",
        threads=256,
        smem_bytes=162816,
        cluster_x=6,
        abi=1,
        device_sha256="e11c874ddbe0a68f15a021ad6844f13d49bfe404bfdf16f2db04e1e3e0ca6b78",
    ),
    "small_k4c2": _Stage(
        module="cake_vsa_sm90_0e5c59c845b71d39fc89",
        shim="c5662c2e8b8e60b0",
        threads=256,
        smem_bytes=166912,
        cluster_x=2,
        abi=1,
        device_sha256="da95c66f6da1dc06fc089e8be9315ec7a58d8bcb29908e969b8c5c75ff8c210b",
    ),
    "small_k4c4": _Stage(
        module="cake_vsa_sm90_f2127f961acc174f20a6",
        shim="504977ef53ae6453",
        threads=256,
        smem_bytes=177152,
        cluster_x=4,
        abi=1,
        device_sha256="1a9cd9c29501322cc357a390ddf0153341568abcdb10d88f8eadbb225db63437",
    ),
    "small_k6c2": _Stage(
        module="cake_vsa_sm90_cd7cc195b169fb1832a9",
        shim="885811e58f98afe7",
        threads=256,
        smem_bytes=232448,
        cluster_x=2,
        abi=1,
        device_sha256="4e67f760201795fbb30e2b21af3c3b25f3d3049a16e865a59c82c445992fc2c2",
    ),
}
STAGES = tuple(_STAGES)


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_vsa_sm90"
    if (installed / _BINDING).is_file():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "cake_vsa_sm90"
    if (checkout / _BINDING).is_file():
        return checkout
    raise FileNotFoundError("Cake SM90 VSA sources were not found")


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout
    raise FileNotFoundError("FlashInfer headers were not found")


def _stage(stage: str) -> _Stage:
    try:
        return _STAGES[stage]
    except KeyError:
        raise ValueError(
            f"unknown Cake SM90 VSA stage {stage!r}; expected one of {STAGES}"
        ) from None


def _source_path(name: str, sha256: str) -> Path:
    path = _get_csrc_dir() / name
    if not path.is_file():
        raise FileNotFoundError(
            f"generated source is absent: csrc/cake_vsa_sm90/{name}"
        )
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise RuntimeError(f"generated source hash mismatch: csrc/cake_vsa_sm90/{name}")
    return path


def _stage_defines(row: _Stage) -> list[str]:
    return [
        f"-DCAKE_VSA_SM90_KERNEL=kernel_{row.module}",
        f"-DCAKE_VSA_SM90_SHIM_NAMESPACE=cake_host_shim_{row.shim}",
        f"-DCAKE_VSA_SM90_THREADS={row.threads}u",
        f"-DCAKE_VSA_SM90_SMEM_BYTES={row.smem_bytes}u",
        f"-DCAKE_VSA_SM90_CLUSTER_X={row.cluster_x}",
        f"-DCAKE_VSA_SM90_ABI={row.abi}",
    ]


@functools.cache
def gen_cake_vsa_sm90_module(stage: str = "attention") -> JitSpec:
    row = _stage(stage)
    spec = gen_jit_spec(
        name=f"{row.module}_{_ARCH}",
        sources=[
            _source_path(f"{row.module}_kernel.cu", row.device_sha256),
            _source_path(_BINDING, _BINDING_SHA256),
        ],
        extra_cuda_cflags=[*sm90a_nvcc_flags, *_COMPILE_FLAGS, *_stage_defines(row)],
        extra_ldflags=["-lcuda"],
        # The family directory holds the shared device preamble and launcher
        # helper headers that every translation unit includes by name.
        extra_include_paths=[
            _get_csrc_dir(),
            _get_csrc_dir().parent,
            _get_include_dir(),
        ],
    )
    logger.info("Generated Cake SM90 VSA JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_vsa_sm90_module(stage: str = "attention"):
    """Build and load one ``sm_90a`` module; return ``(module, stage table row)``."""
    spec = gen_cake_vsa_sm90_module(stage)
    module = spec.build_and_load()
    return module, _stage(stage)


__all__ = [
    "STAGES",
    "gen_cake_vsa_sm90_module",
    "load_cake_vsa_sm90_module",
]
