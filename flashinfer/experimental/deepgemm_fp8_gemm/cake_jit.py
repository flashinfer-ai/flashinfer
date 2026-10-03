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
from typing import Any

from ...jit import env as jit_env
from ...jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

# Explicit target-owned registration of the generated FP8 E4M3 1D1D GEMM
# programs.  Each program is one CUDA source pair (kernel + host binding) that
# compiles for every listed architecture with that architecture's exact flag
# set; ``MODULES`` holds one record per program (sources relative to
# ``csrc/experimental/deepgemm_fp8_gemm``, compile flags, FFI entry, argument
# plan, closure identity, supported architectures) and ``KERNELS`` maps the
# route (``forward``: BF16 output, ``wgrad``: FP32 accumulate in place) to its
# program.  Both literals are populated verbatim by the generated-program
# export; do not edit them by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_deepgemm_fp8_gemm_18fcb3341bab99af1b01": {
        "role": "kernel",
        "sources": [
            "cake_deepgemm_fp8_gemm/cake_deepgemm_fp8_gemm_18fcb3341bab99af1b01_kernel.cu",
            "cake_deepgemm_fp8_gemm/cake_deepgemm_fp8_gemm_18fcb3341bab99af1b01_binding.cu",
        ],
        "compile_flags": ["--ptxas-options=--register-usage-level=10"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "C_tma"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "K_tiles"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
        "closure_sha256": {
            "sm_100a": "f0c9eb48f8ad1492f1a4f79faaa85c94e897348040061de1a1e1904cf1286b9b",
            "sm_103a": "292d08758477730808ecdd2724ba0c30a5a1a25511cbf4fffdbef56606506e84",
        },
    },
    "cake_deepgemm_fp8_gemm_3b61ba0203e4fccf63a1": {
        "role": "kernel",
        "sources": [
            "cake_deepgemm_fp8_gemm/cake_deepgemm_fp8_gemm_3b61ba0203e4fccf63a1_kernel.cu",
            "cake_deepgemm_fp8_gemm/cake_deepgemm_fp8_gemm_3b61ba0203e4fccf63a1_binding.cu",
        ],
        "compile_flags": ["--ptxas-options=--register-usage-level=10"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "C_tma"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "K_tiles"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
        "closure_sha256": {
            "sm_100a": "0d13b108e4ea5e138b30fe6205f98842dc6b769cdf8742058f444a41b75a51ba",
            "sm_103a": "7bd988d2eec77b991ef29282d1bf6ee391b162886091a335030f952ed41bbe85",
        },
    },
}
KERNELS: dict[str, str] = {
    "forward": "cake_deepgemm_fp8_gemm_3b61ba0203e4fccf63a1",
    "wgrad": "cake_deepgemm_fp8_gemm_18fcb3341bab99af1b01",
}

ARCH_NVCC_FLAGS = {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}
CSRC_SUBDIR = "experimental/deepgemm_fp8_gemm"


def supported_arches() -> tuple[str, ...]:
    """Architectures every registered program compiles for."""
    arches: set[str] | None = None
    for record in MODULES.values():
        current = set(record["arches"])
        arches = current if arches is None else arches & current
    return tuple(sorted(arches or ()))


def jit_spec(name: str, arch: str):
    """JIT build specification of program ``name`` for the exact ``arch``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(f"program {name} is not registered for {arch}")
    csrc = jit_env.FLASHINFER_CSRC_DIR / CSRC_SUBDIR
    return gen_jit_spec(
        name=f"{name}_{arch}",
        sources=[csrc / path for path in record["sources"]],
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[arch],
            *record["compile_flags"],
            "--device-entity-has-hidden-visibility=false",
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[
            csrc,
            jit_env.FLASHINFER_CSRC_DIR,
            jit_env.FLASHINFER_INCLUDE_DIR,
        ],
        use_fast_math=False,  # only the registered compile flags select math modes
    )


@functools.cache
def load_module(name: str, arch: str):
    """Build (or load from the JIT cache) program ``name`` for the exact ``arch``."""
    return jit_spec(name, arch).build_and_load()
