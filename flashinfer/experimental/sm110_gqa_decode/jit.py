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

# Program registry and JIT loader of the exact-SM110 GQA decode kernels.

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from ...jit.core import JitSpec, gen_jit_spec, logger, sm110a_nvcc_flags

# Every generated program the package compiles, keyed by module name: the
# short (capacity <= 64, 256 threads) and long (384 threads) decode programs
# and the three exact-shape specializations of the prepared API. Paths are
# relative to this package.
MODULES: dict[str, dict[str, Any]] = {
    "original": {
        "sources": [
            "csrc/sm110_gqa_decode/sm_110a/sm110_gqa_decode_short_kernel.cu",
            "csrc/sm110_gqa_decode/sm_110a/sm110_gqa_decode_short_binding.cu",
            "csrc/sm110_gqa_decode/sm_110a/sm110_gqa_decode_long_kernel.cu",
            "csrc/sm110_gqa_decode/sm_110a/sm110_gqa_decode_long_binding.cu",
        ],
        "compile_flags": [],
    },
    "n32_b4_direct": {
        "sources": [
            "csrc/prepared/sm110_gqa_decode_n32_b4_direct.cu",
            "csrc/prepared/sm110_gqa_decode_n32_b4_direct_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
    },
    "n32_disjoint_s10": {
        "sources": [
            "csrc/prepared/sm110_gqa_decode_n32_disjoint_s10.cu",
            "csrc/prepared/sm110_gqa_decode_n32_disjoint_s10_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
    },
    "n64_kvlast_s10": {
        "sources": [
            "csrc/prepared/sm110_gqa_decode_n64_kvlast_s10.cu",
            "csrc/prepared/sm110_gqa_decode_n64_kvlast_s10_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
    },
}

# Launch routes: the module and FFI entry serving each route, its kernel symbol
# and split count. Routes with ``num_splits`` above one write FP32 partials and
# merge them in the last CTA, so they need the caller-retained workspace that
# the prepared API allocates once.
ROUTES: dict[str, dict[str, Any]] = {
    "short": {
        "module": "original",
        "ffi_entry": "run_short",
        "kernel_symbol": "kernel_sm110_gqa_decode_short",
        "num_splits": 1,
    },
    "long": {
        "module": "original",
        "ffi_entry": "run_long",
        "kernel_symbol": "kernel_sm110_gqa_decode_long",
        "num_splits": 1,
    },
    "n32_b4_direct": {
        "module": "n32_b4_direct",
        "ffi_entry": "run",
        "kernel_symbol": "kernel_sm110_gqa_decode_n32_b4_direct",
        "num_splits": 1,
    },
    "n32_disjoint_s10": {
        "module": "n32_disjoint_s10",
        "ffi_entry": "run",
        "kernel_symbol": "kernel_sm110_gqa_decode_n32_disjoint_s10",
        "num_splits": 10,
    },
    "n64_kvlast_s10": {
        "module": "n64_kvlast_s10",
        "ffi_entry": "run",
        "kernel_symbol": "kernel_sm110_gqa_decode_n64_kvlast_s10",
        "num_splits": 10,
    },
}

# Capacities through this value use the short kernel.
SHORT_CAPACITY_MAX = 64
# Prepared default routes of the exact ``batch:capacity`` keys they were
# qualified for; every other shape above SHORT_CAPACITY_MAX uses ``long``.
PREPARED_ROUTES: dict[str, str] = {
    "4:256": "n32_b4_direct",
    "1:1024": "n64_kvlast_s10",
    "1:4096": "n32_disjoint_s10",
}


def gen_sm110_gqa_decode_module(module: str) -> JitSpec:
    """Create the exact-SM110a JIT specification of one registered module."""

    record = MODULES[module]
    root = Path(__file__).resolve().parent
    spec = gen_jit_spec(
        name=f"sm110_gqa_decode_{module}",
        sources=[root / source for source in record["sources"]],
        extra_cuda_cflags=[*sm110a_nvcc_flags, *record["compile_flags"]],
        extra_ldflags=["-lcuda"],
    )
    logger.info(f"Generated SM110 GQA decode JIT spec: {spec.name}")
    return spec


@functools.cache
def _check_exact_sm110a(device: Any = None) -> None:
    import torch

    from ...utils import get_compute_capability, is_sm110a_supported

    resolved = torch.device("cuda") if device is None else torch.device(device)
    capability = get_compute_capability(resolved)
    if capability != (11, 0) or not is_sm110a_supported(resolved):
        raise RuntimeError(
            "SM110 GQA decode requires compute capability 11.0 and CUDA 13.0 or newer; "
            f"got compute capability {capability[0]}.{capability[1]} with CUDA {torch.version.cuda}"
        )


@functools.cache
def _build_and_load(module: str) -> Any:
    loaded = gen_sm110_gqa_decode_module(module).build_and_load()
    logger.info(f"Loaded SM110 GQA decode module {module}")
    return loaded


def load_sm110_gqa_decode_module(*, device: Any = None, module: str) -> Any:
    """Build or load one registered module for an exact SM110a device."""

    _check_exact_sm110a(device)
    return _build_and_load(module)


__all__ = [
    "MODULES",
    "PREPARED_ROUTES",
    "ROUTES",
    "SHORT_CAPACITY_MAX",
    "gen_sm110_gqa_decode_module",
    "load_sm110_gqa_decode_module",
]
