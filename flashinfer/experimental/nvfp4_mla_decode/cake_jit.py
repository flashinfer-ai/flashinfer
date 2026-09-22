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

from ...jit import env as jit_env
from ...jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

# Explicit target-owned registration; each closure has separate translation
# units. One record per architecture; every record carries both physical
# stages of the generated program (the persistent decode kernel and the
# split-KV combine kernel) as separate FFI entries on the same module.
#
# TODO(cake_nvfp4_mla_decode ABI freeze, flashinfer-ai/flashinfer#5403):
# populate MODULES from the generated-program export once the kernel ABI is
# frozen. Each record needs, verbatim from the export:
#   "arch":           "sm_100a" | "sm_103a"
#   "sources":        [".../<arch>/cake_nvfp4_mla_decode_<hash>_kernel.cu",
#                      ".../<arch>/cake_nvfp4_mla_decode_<hash>_binding.cu"]
#                     (both stages live in the same translation units)
#   "compile_flags":  per-record nvcc flags emitted by the export
#   "arg_plan":       ordered [kind, name] pairs for the decode stage, kinds
#                     "tma_buffer" | "buffer" | "parameter" | "grid"; the
#                     names must match ``cake_backend.MAIN_KWARGS``
#   "reduce_arg_plan": ordered [kind, name] pairs for the combine stage; the
#                     names must match ``cake_backend.REDUCE_KWARGS``
#   "ffi_entry":      FFI symbol of the decode stage
#                     (kernel symbol kernel_cake_nvfp4_mla_decode)
#   "reduce_ffi_entry": FFI symbol of the combine stage
#                     (kernel symbol kernel_cake_nvfp4_mla_decode_reduce)
#   "closure_sha256": export closure digest (used in the JIT module name)
# Until then ``select_module`` raises and the public API reports the route as
# not yet available on this checkout.
MODULES: dict[str, dict] = {}

ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


def select_module(arch: str) -> str:
    """Return the registered module name for ``arch`` (``sm_100a`` / ``sm_103a``)."""
    for name, record in MODULES.items():
        if record["arch"] == arch:
            return name
    raise NotImplementedError(
        "The generated NVFP4 MLA decode program for "
        f"{arch} is not registered in this checkout yet "
        "(see flashinfer-ai/flashinfer#5403)"
    )


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
def gen_cake_nvfp4_mla_decode_module(name):
    record = MODULES[name]
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    return gen_jit_spec(
        name=name + "_" + record["closure_sha256"][:20],
        sources=sources,
        extra_cuda_cflags=[*ARCH_NVCC_FLAGS[record["arch"]], *record["compile_flags"]],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_nvfp4_mla_decode_module(name):
    return gen_cake_nvfp4_mla_decode_module(name).build_and_load()
