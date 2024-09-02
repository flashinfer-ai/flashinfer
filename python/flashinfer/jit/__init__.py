"""
Copyright (c) 2024 by FlashInfer team.

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

import os
import re
import torch.utils.cpp_extension as torch_cpp_ext
from typing import List
from .env import (
    FLASHINFER_JIT_DIR,
    FLASHINFER_GEN_SRC_DIR,
    FLASHINFER_INCLUDE_DIR,
    FLASHINFER_CSRC_DIR,
    CUTLASS_INCLUDE_DIR,
)
from .activation import get_act_and_mul_cu, gen_act_and_mul_cu


def check_cuda_arch():
    # cuda arch check for fp8 at the moment.
    for cuda_arch_flags in torch_cpp_ext._get_cuda_arch_flags():
        arch = int(re.search("compute_\d+", cuda_arch_flags).group()[-2:])
        if arch < 75:
            raise RuntimeError("FlashInfer requires sm75+")


def clear_cache_dir():
    if os.path.exists(FLASHINFER_JIT_DIR):
        for file in os.listdir(FLASHINFER_JIT_DIR):
            os.remove(os.path.join(FLASHINFER_JIT_DIR, file))


def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass


remove_unwanted_pytorch_nvcc_flags()


def load_cuda_ops(
    name: str,
    sources: List[str],
    extra_cflags: List[str] = ["-O3", "-Wno-switch-bool"],
    extra_cuda_cflags: List[str] = [
        "-O3",
        "-std=c++17",
        "--threads",
        "1",
        "-Xfatbin",
        "-compress-all",
        "-use_fast_math",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8",
    ],
    extra_ldflags=None,
    extra_include_paths=None,
    verbose=False,
):
    check_cuda_arch()
    build_directory = FLASHINFER_JIT_DIR / name
    if not os.path.exists(build_directory):
        os.makedirs(build_directory)
    if extra_include_paths is None:
        extra_include_paths = [
            FLASHINFER_INCLUDE_DIR,
            CUTLASS_INCLUDE_DIR,
            FLASHINFER_CSRC_DIR,
        ]
    return torch_cpp_ext.load(
        name,
        list(map(lambda _: str(_), sources)),
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=list(map(lambda _: str(_), extra_include_paths)),
        build_directory=build_directory,
        verbose=verbose,
        with_cuda=True,
    )
