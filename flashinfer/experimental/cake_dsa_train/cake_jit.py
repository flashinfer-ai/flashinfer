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
# policies that select the key-range-pass form of the backward
# (``key_pass_policy``, see ``cake_backend.KeyPassPolicy``) and the direct
# accumulation into the caller's packed rows through the natural-layout main
# stages (``dkv_direct``, see ``cake_backend.DkvDirectPolicy``), and one physical
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
            "bwd_main_natural",
            "bwd_compact",
            "bwd_main_pass",
            "bwd_main_pass_natural",
            "bwd_cast",
        ],
        "key_pass_policy": {
            "l2_budget_bytes": 104857600,
            "key_bytes": 2304,
            "workspace_budget_bytes": 671088640,
            "token_chunk_multiple": 128,
            "max_passes": 4,
            "rules": {
                "sm_100a": {
                    "tail_rows": 0,
                    "fixed_passes": 0,
                    "min_kv": 0,
                    "tail_min_kv": 0,
                    "tail_cap": 0,
                },
                "sm_103a": {
                    "tail_rows": 1,
                    "fixed_passes": 0,
                    "min_kv": 0,
                    "tail_min_kv": 131072,
                    "tail_cap": 3,
                },
                "sm_107a": {
                    "tail_rows": 1,
                    "fixed_passes": 2,
                    "min_kv": 131072,
                    "tail_min_kv": 131072,
                    "tail_cap": 0,
                },
            },
        },
        "dkv_direct": {"min_keys_per_query": 4},
        "fwd": {
            "module": "cake_dsa_h64_train_257d4674f88939dbe09d",
            "sources": [
                "cake_dsa_h64_train/cake_dsa_h64_train_257d4674f88939dbe09d_kernel.cu",
                "cake_dsa_h64_train/cake_dsa_h64_train_257d4674f88939dbe09d_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "closure_sha256": "cc4fd5a2a936efb2240c74c71c193b26564d3d821b41bc3111d50950491ad231",
            "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_delta": {
            "module": "cake_dsa_h64_train_bbf4638c65499139a5b5",
            "sources": [
                "cake_dsa_h64_train/cake_dsa_h64_train_bbf4638c65499139a5b5_kernel.cu",
                "cake_dsa_h64_train/cake_dsa_h64_train_bbf4638c65499139a5b5_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "closure_sha256": "9133aeeada149e76a90cbece84dfb58fed837f27f8e752df404ba87729da5028",
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_main": {
            "module": "cake_dsa_h64_train_71e04df5a9416cb7a05c",
            "sources": [
                "cake_dsa_h64_train/cake_dsa_h64_train_71e04df5a9416cb7a05c_kernel.cu",
                "cake_dsa_h64_train/cake_dsa_h64_train_71e04df5a9416cb7a05c_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "closure_sha256": "08f452bd2f165b5365f1e471b5491913fb5ba083ef11c2a350ffadacdb592ae0",
            "launch": {"block": [640, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_main_natural": {
            "module": "cake_dsa_h64_train_9aee0d770903bfa74820",
            "sources": [
                "cake_dsa_h64_train/cake_dsa_h64_train_9aee0d770903bfa74820_kernel.cu",
                "cake_dsa_h64_train/cake_dsa_h64_train_9aee0d770903bfa74820_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "closure_sha256": "1885a4ad70b709fe0cc4a0f3bcdc736a90e4aaf5051f4b3b08882418294bada9",
            "launch": {"block": [640, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_compact": {
            "module": "cake_dsa_h64_train_d3d9be180aa3af037978",
            "sources": [
                "cake_dsa_h64_train/cake_dsa_h64_train_d3d9be180aa3af037978_kernel.cu",
                "cake_dsa_h64_train/cake_dsa_h64_train_d3d9be180aa3af037978_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "closure_sha256": "f1f58a3c00a931ee8bde83d87044b19ecaed8f022f66fd6985bf6ac08e94c1ca",
            "launch": {"block": [128, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_main_pass": {
            "module": "cake_dsa_h64_train_4888876874aff07ea952",
            "sources": [
                "cake_dsa_h64_train/cake_dsa_h64_train_4888876874aff07ea952_kernel.cu",
                "cake_dsa_h64_train/cake_dsa_h64_train_4888876874aff07ea952_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "closure_sha256": "b90185422b6d1807e6335210cae1bb7f56ebb5e45117515884a09c1d44a58bee",
            "launch": {"block": [640, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_main_pass_natural": {
            "module": "cake_dsa_h64_train_0e1d6ec89184ab897d76",
            "sources": [
                "cake_dsa_h64_train/cake_dsa_h64_train_0e1d6ec89184ab897d76_kernel.cu",
                "cake_dsa_h64_train/cake_dsa_h64_train_0e1d6ec89184ab897d76_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "closure_sha256": "13655f2d662c2e9b003ffcd5bf81625c95bafb87e6242f49b0cb04709f5e1b8d",
            "launch": {"block": [640, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_cast": {
            "module": "cake_dsa_h64_train_2a461db82be6f8b6558d",
            "sources": [
                "cake_dsa_h64_train/cake_dsa_h64_train_2a461db82be6f8b6558d_kernel.cu",
                "cake_dsa_h64_train/cake_dsa_h64_train_2a461db82be6f8b6558d_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "closure_sha256": "88bffd050d5504a43830d82802c114b1bbbfe700756ec90c7e9e6707a478517a",
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
    },
}

PROGRAM = "cake_dsa_h64_train"
STAGES = (
    "fwd",
    "bwd_delta",
    "bwd_main",
    "bwd_main_natural",
    "bwd_compact",
    "bwd_main_pass",
    "bwd_main_pass_natural",
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
