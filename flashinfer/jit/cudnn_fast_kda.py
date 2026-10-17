# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the host-only cuDNN KDA helper with FlashInfer's JIT toolchain."""

import functools
import importlib.util
import sysconfig
from pathlib import Path

import torch
import tvm_ffi

from . import env as jit_env
from .core import gen_jit_spec


def gen_cudnn_fast_kda_module():
    # The module contains Python objects; keep incompatible interpreter ABIs apart.
    name = "cudnn_fast_kda_" + sysconfig.get_config_var("SOABI")
    library = Path(tvm_ffi.libinfo.find_libtvm_ffi())

    def load_python_module(_ffi_module):
        path = jit_env.FLASHINFER_JIT_DIR / name / f"{name}.so"
        spec = importlib.util.spec_from_file_location("_cudnn_fast_kda", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return gen_jit_spec(
        name,
        [jit_env.FLASHINFER_CSRC_DIR / "cudnn_fast_kda.cpp"],
        # Pybind11 requires the full CPython API, unlike ordinary TVM-FFI modules.
        extra_cflags=["-UPy_LIMITED_API", "-fvisibility=hidden"],
        extra_include_paths=[Path(torch.__file__).parent / "include"],
        extra_ldflags=[str(library), "-ldl", f"-Wl,-rpath,{library.parent}"],
        post_load_adapter=load_python_module,
    )


@functools.cache
def get_cudnn_fast_kda_module():
    return gen_cudnn_fast_kda_module().build_and_load()
