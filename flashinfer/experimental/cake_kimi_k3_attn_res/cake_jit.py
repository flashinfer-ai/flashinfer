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
from ...jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

# Explicit target-owned registration of the generated Kimi-K3 AttnRes programs.
# ``MODULES``: one record per generated program (role, translation units
# relative to this package's ``csrc``, compile flags, FFI entry, argument plan,
# closure identity, CTA size, and the architectures the one source is compiled
# for).  ``KERNELS``: per architecture, the logical kernel key the host planner
# resolves (``cake_backend.plan_route(...).kernel_key``) -> module name.  Both
# literals are populated verbatim by the generated-program export; do not edit
# by hand.
MODULES: dict[str, dict[str, Any]] = {}
KERNELS: dict[str, dict[str, str]] = {}

# The programs use tcgen05 TMEM (the persistent and native ports) and are
# exact-architecture payloads: one source, compiled once per listed target.
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


def _nvcc_flags(arches: list[str]) -> list[str]:
    """The target code-generation flags of ``arches`` followed by the common flags, each once."""
    flags: list[str] = []
    for arch in arches:
        for flag in ARCH_NVCC_FLAGS[arch]:
            if flag not in flags:
                flags.append(flag)
    return flags


def registered_kernel_keys(arch: str) -> tuple[str, ...]:
    """Every logical kernel key registered for ``arch`` (empty without programs)."""
    return tuple(sorted(KERNELS.get(arch, {})))


def route_available(arch: str, required_keys: tuple[str, ...] = ()) -> bool:
    table = KERNELS.get(arch)
    if not table:
        return False
    return all(key in table for key in required_keys)


def kernel_module_name(arch: str, key: str) -> str:
    """The registered module of logical kernel ``key`` on ``arch``."""
    table = KERNELS.get(arch)
    if not table:
        raise NotImplementedError(
            f"No generated Kimi-K3 AttnRes programs are registered for {arch} in this checkout"
        )
    name = table.get(key)
    if name is None:
        raise NotImplementedError(
            f"The generated Kimi-K3 AttnRes kernel {key!r} for {arch} is not registered in this "
            "checkout (the export covers the contract's token counts / block counts; see the package README)"
        )
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"registered module {name!r} is not built for {arch} (arches: {record['arches']})"
        )
    return name


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
def gen_cake_kimi_k3_attn_res_module(name: str):
    record = MODULES[name]
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    return gen_jit_spec(
        name=f"{name}_" + record["closure_sha256"][:20],
        sources=sources,
        extra_cuda_cflags=[*_nvcc_flags(record["arches"]), *record["compile_flags"]],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_kimi_k3_attn_res_module(name: str):
    return gen_cake_kimi_k3_attn_res_module(name).build_and_load()
