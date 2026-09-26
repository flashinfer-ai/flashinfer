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

# Explicit target-owned registration of the generated programs.
#
# ``MODULES`` holds one record per physical generated module (a kernel plus
# its host binding): architecture, role, translation units, compile flags,
# FFI entry, argument plan and closure identity.
#
# ``ROUTES`` maps ``"<dtype>_p<page_size>_g<item_groups>_pm<partition_mode>__<arch>"``
# to the physical *main* (persistent decode) module of that variant.  The
# main kernel is traced per (KV dtype, page size, scheduler item-group
# capacity, partition mode); the host plan (``cake_backend.plan_varq_dcp_decode``)
# selects the variant from host-known scalars only.
#
# ``MERGE_MODULES`` maps an architecture to its split-KV merge kernel
# (``mla_varq_dcp_merge``), launched programmatically behind the main kernel
# by the ticket-scheduler variants whose plan can split an item.
#
# All three literals are populated verbatim by the generated-program export;
# do not edit them by hand.
MODULES: dict[str, dict[str, Any]] = {}
ROUTES: dict[str, dict[str, Any]] = {}
MERGE_MODULES: dict[str, str] = {}

STAGES = ("main", "merge")
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}
TRACKING_ISSUE = "flashinfer-ai/flashinfer#4658"


def route_name(
    arch: str, dtype: str, page_size: int, item_groups: int, partition_mode: int
) -> str:
    return f"{dtype}_p{int(page_size)}_g{int(item_groups)}_pm{int(bool(partition_mode))}__{arch}"


def main_route_available(
    arch: str, dtype: str, page_size: int, item_groups: int, partition_mode: int
) -> bool:
    return route_name(arch, dtype, page_size, item_groups, partition_mode) in ROUTES


def select_main_module(
    arch: str, dtype: str, page_size: int, item_groups: int, partition_mode: int
) -> str:
    """Return the registered main-kernel module of one physical variant."""
    name = route_name(arch, dtype, page_size, item_groups, partition_mode)
    record = ROUTES.get(name)
    if record is None:
        raise NotImplementedError(
            "The generated Cake MLA var-Q DCP decode program "
            f"{name!r} (dtype={dtype}, page_size={page_size}, "
            f"item_groups={item_groups}, partition_mode={int(bool(partition_mode))}) "
            f"is not registered in this checkout (see {TRACKING_ISSUE})"
        )
    module = record["main"]
    if module not in MODULES or MODULES[module]["arch"] != arch:
        raise RuntimeError(
            f"route {name!r} names an unregistered main module {module!r}"
        )
    return module


def select_merge_module(arch: str) -> str:
    """Return the registered split-KV merge kernel module for ``arch``."""
    module = MERGE_MODULES.get(arch)
    if module is None:
        raise NotImplementedError(
            "The generated Cake MLA var-Q DCP merge kernel for "
            f"{arch} is not registered in this checkout (see {TRACKING_ISSUE})"
        )
    if module not in MODULES or MODULES[module]["arch"] != arch:
        raise RuntimeError(f"merge module {module!r} for {arch} is not registered")
    return module


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
def gen_cake_mla_varq_dcp_decode_module(name: str):
    record = MODULES[name]
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    return gen_jit_spec(
        name=f"{name}_" + record["closure_sha256"][:20],
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
def load_cake_mla_varq_dcp_decode_module(name: str):
    return gen_cake_mla_varq_dcp_decode_module(name).build_and_load()
