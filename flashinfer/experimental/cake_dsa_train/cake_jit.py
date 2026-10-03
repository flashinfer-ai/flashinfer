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
from pathlib import Path
from typing import Any

from ...jit import env as jit_env
from ...jit.core import (
    gen_jit_spec,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
    sm107a_nvcc_flags,
)

# Explicit target-owned registration of the generated training program: one
# record (``PROGRAM``) shared by every architecture it compiles for
# (``arches``), the kernel ``stages`` it registers in launch order, the host
# policy that selects the key-range-pass form of the backward
# (``key_pass_policy``, see ``cake_backend.KeyPassPolicy``) and one physical
# entry per stage (translation units, compile flags, FFI entry, launch block /
# cluster and closure identity).  The positional argument order of every stage
# lives in the generated ``cake_launch`` module.  Populated verbatim by the
# generated-program export; do not edit by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_dsa_h64_train": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "stages": [
            "fwd",
            "bwd_delta",
            "bwd_main",
            "bwd_compact",
            "bwd_main_pass",
            "bwd_cast",
        ],
        "key_pass_policy": {
            "l2_budget_bytes": 104857600,
            "key_bytes": 2304,
            "workspace_budget_bytes": 671088640,
            "token_chunk_multiple": 128,
        },
        "fwd": {
            "module": "cake_dsa_h64_train_877b7797bb6b1bd6a37e",
            "sources": [
                "cake_dsa_h64_train/cake_dsa_h64_train_877b7797bb6b1bd6a37e_kernel.cu",
                "cake_dsa_h64_train/cake_dsa_h64_train_877b7797bb6b1bd6a37e_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "closure_sha256": "e22741ea0b74dd6de31912ee31dddaa56a05588af3925957d5802093c0348370",
            "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_delta": {
            "module": "cake_dsa_h64_train_dc81cb985444fde8f97f",
            "sources": [
                "cake_dsa_h64_train/cake_dsa_h64_train_dc81cb985444fde8f97f_kernel.cu",
                "cake_dsa_h64_train/cake_dsa_h64_train_dc81cb985444fde8f97f_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "closure_sha256": "2325f6b6bae2dc5b0e95cf38eced4e28eee4794b3d62874f04e106ad91a1b5bd",
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_main": {
            "module": "cake_dsa_h64_train_39efe002ad7020fd9d33",
            "sources": [
                "cake_dsa_h64_train/cake_dsa_h64_train_39efe002ad7020fd9d33_kernel.cu",
                "cake_dsa_h64_train/cake_dsa_h64_train_39efe002ad7020fd9d33_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "closure_sha256": "b0c30eb8d084022f61a179f5be6943221ae20c535502f5bf29064cef8a31124f",
            "launch": {"block": [640, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_compact": {
            "module": "cake_dsa_h64_train_f362dd10b022aa648c28",
            "sources": [
                "cake_dsa_h64_train/cake_dsa_h64_train_f362dd10b022aa648c28_kernel.cu",
                "cake_dsa_h64_train/cake_dsa_h64_train_f362dd10b022aa648c28_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "closure_sha256": "bd3c22277be41ec3928d06784bf9464dcfbb4845a3faf21c1a9d5ddcd6b4b714",
            "launch": {"block": [128, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_main_pass": {
            "module": "cake_dsa_h64_train_1ef045c316e8c838c26f",
            "sources": [
                "cake_dsa_h64_train/cake_dsa_h64_train_1ef045c316e8c838c26f_kernel.cu",
                "cake_dsa_h64_train/cake_dsa_h64_train_1ef045c316e8c838c26f_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "closure_sha256": "d55bb7aa4b309e42d6a5fed6fdbda6620642d5ec6d6efebc117cab1435148f7d",
            "launch": {"block": [640, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_cast": {
            "module": "cake_dsa_h64_train_a87a953e2d2ac436209c",
            "sources": [
                "cake_dsa_h64_train/cake_dsa_h64_train_a87a953e2d2ac436209c_kernel.cu",
                "cake_dsa_h64_train/cake_dsa_h64_train_a87a953e2d2ac436209c_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "closure_sha256": "1738785f27ee923c0002cd48d8a375374d990ee5c48f911eacf15d01efc6bf9a",
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
    },
}

PROGRAM = "cake_dsa_h64_train"
STAGES = (
    "fwd",
    "bwd_delta",
    "bwd_main",
    "bwd_compact",
    "bwd_main_pass",
    "bwd_cast",
)
FORWARD_STAGES = ("fwd",)
BACKWARD_STAGES = STAGES[1:]
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
    "sm_107a": sm107a_nvcc_flags,
}


def record() -> dict[str, Any]:
    """The registered program record."""
    try:
        return MODULES[PROGRAM]
    except KeyError:
        raise NotImplementedError(
            "The generated DSA sparse-attention training program is not registered "
            "in this checkout (see flashinfer-ai/flashinfer#5657)"
        ) from None


def toolchain_supports(arch: str) -> bool:
    """Can the nvcc this checkout invokes emit ``arch``?

    SM100 / SM103 compile with any CUDA 12.8+ toolkit; ``sm_107a`` needs an nvcc
    that lists ``compute_107`` (public CUDA 13.x toolkits do not), so it is
    probed once.
    """
    if arch not in ARCH_NVCC_FLAGS:
        return False
    if arch == "sm_107a":
        from ...compilation_context import _nvcc_supports_sm107

        return _nvcc_supports_sm107()
    return True


def registered_stages() -> tuple[str, ...]:
    """Stages the record registers, in launch order."""
    rec = record()
    present = tuple(stage for stage in STAGES if stage in rec)
    declared = tuple(rec.get("stages", present))
    if tuple(s for s in STAGES if s in declared) != present:
        raise ValueError(
            f"registry record declares stages {declared} but carries {present}"
        )
    return present


def _header_dirs():
    installed = [jit_env.FLASHINFER_CSRC_DIR, jit_env.FLASHINFER_INCLUDE_DIR]
    if (installed[0] / "tvm_ffi_utils.h").is_file() and (
        installed[1] / "flashinfer/layout.cuh"
    ).is_file():
        return installed
    checkout = Path(__file__).resolve().parents[3]
    source = [checkout / "csrc", checkout / "include"]
    if (source[0] / "tvm_ffi_utils.h").is_file() and (
        source[1] / "flashinfer/layout.cuh"
    ).is_file():
        return source
    raise FileNotFoundError("FlashInfer binding headers were not found")


@functools.cache
def gen_cake_dsa_train_module(stage: str, arch: str):
    """JIT spec of ``stage`` compiled with the exact flag set of ``arch``."""
    rec = record()
    if arch not in rec["arches"] or not toolchain_supports(arch):
        raise RuntimeError(
            f"the generated DSA training program is registered for {rec['arches']}; "
            f"{arch!r} is not served by this checkout"
        )
    physical = rec[stage]
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in physical["sources"]]
    return gen_jit_spec(
        name=f"{PROGRAM}_{stage}_{arch}_" + physical["closure_sha256"][:20],
        sources=sources,
        extra_cuda_cflags=[*ARCH_NVCC_FLAGS[arch], *physical["compile_flags"]],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_dsa_train_module(stage: str, arch: str):
    return gen_cake_dsa_train_module(stage, arch).build_and_load()
