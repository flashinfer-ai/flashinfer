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
from ...jit.blackwell_msa import BlackwellMSATarget
from ...jit.core import (
    gen_jit_spec,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)

_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}


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
def gen_msa_decode_metadata_module(target: BlackwellMSATarget):
    """Prepare independent sparse rows for packed-FP8 MSA decode."""
    root = Path(__file__).resolve().parent / "csrc"
    return gen_jit_spec(
        f"msa_fp8_decode_metadata_{target}",
        [root / "msa_decode_metadata.cu"],
        extra_cuda_cflags=_NVCC_FLAGS[target],
        extra_include_paths=[root, *_header_dirs()],
    )


@functools.cache
def load_msa_decode_metadata_module(target: BlackwellMSATarget):
    return gen_msa_decode_metadata_module(target).build_and_load()
