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

# Explicit target-owned registration of the generated GEMM programs.  One
# record per physical module = one traced kernel instance (``template``) per
# architecture.  A record carries ``arch``, the instance ``template`` (the
# symbol the host planner derives from the operand layouts, the tile width,
# the output kind, the epilogue path and the TMA L2 eviction hints -- see
# ``cake_backend.instance_symbol`` and ``cake_backend.router_symbol``; the
# prefetch / L2-promotion / stage-count knobs are part of the symbol too but
# sit at their defaults in every registered instance), the kernel symbol, the JIT cache name,
# the translation units, compile flags, the FFI entry, the argument plan the
# host binds by keyword, the caller-owned descriptor workspace size, the
# closure identity and the launch geometry baked into the module (block,
# cluster).  Populated verbatim by the generated-program export; do not edit
# by hand.
MODULES: dict[str, dict[str, Any]] = {}

# ``arch -> template -> module name``: the instance table the host planner
# resolves a launch through.  Populated by the generated-program export next to
# ``MODULES``; do not edit by hand.
KERNELS: dict[str, dict[str, str]] = {}

# PLACEHOLDER (generated file list): the device / binding translation units of
# every registered module live under ``csrc/cake_dense_projection_gemm/<arch>/``
# next to this file and are named in each record's ``sources``.  They are
# written by the Cake exporter together with the two registries above; this
# checkout registers no program until that delivery lands.

ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
    "sm_107a": sm107a_nvcc_flags,
}
TRACKING_ISSUE = "flashinfer-ai/flashinfer#5677"


def toolchain_supports(arch: str) -> bool:
    """Can the nvcc this checkout invokes emit ``arch``?  (SM100 / SM103 / SM107 only.)"""
    return arch in ARCH_NVCC_FLAGS


def registered_templates(arch: str) -> tuple[str, ...]:
    """Instance templates registered for ``arch``, sorted."""
    return tuple(sorted(KERNELS.get(arch, {})))


def select_module(arch: str, template: str) -> str:
    """Return the registered module name serving ``template`` on ``arch``."""
    name = KERNELS.get(arch, {}).get(template)
    if name is None:
        registered = registered_templates(arch)
        raise NotImplementedError(
            f"The generated dense projection GEMM instance {template!r} for {arch} is "
            f"not registered in this checkout (registered: {list(registered) or 'none'}; "
            f"see {TRACKING_ISSUE})"
        )
    record = MODULES.get(name)
    if record is None or record.get("arch") != arch or record.get("template") != template:
        raise ValueError(
            f"registry record {name!r} does not serve template {template!r} on {arch}"
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
def gen_cake_dense_projection_gemm_module(name: str):
    record = MODULES[name]
    if not toolchain_supports(record["arch"]):
        raise RuntimeError(
            f"generated dense projection GEMM module {name!r} targets {record['arch']}, "
            "which this checkout cannot compile"
        )
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    return gen_jit_spec(
        name=f"{record['cache_name']}_" + record["closure_sha256"][:20],
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[record["arch"]],
            *record["compile_flags"],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_dense_projection_gemm_module(name: str):
    return gen_cake_dense_projection_gemm_module(name).build_and_load()
