"""
Copyright (c) 2023 by FlashInfer team.

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

from typing import List, Tuple

import pathlib
import os
import re
import itertools
import subprocess
import platform

import setuptools
import argparse
import torch
import torch.utils.cpp_extension as torch_cpp_ext

import generate_single_decode_inst, generate_single_prefill_inst, generate_batch_paged_decode_inst, generate_batch_padded_decode_inst, generate_batch_paged_prefill_inst, generate_batch_ragged_prefill_inst, generate_dispatch_inc

root = pathlib.Path(__name__).parent

enable_bf16 = True
# NOTE(Zihao): we haven't utilized fp8 tensor cores yet, so there is no
# cuda arch check for fp8 at the moment.
enable_fp8 = True
for cuda_arch_flags in torch_cpp_ext._get_cuda_arch_flags():
    arch = int(re.search("compute_\d+", cuda_arch_flags).group()[-2:])
    if arch < 75:
        raise RuntimeError("FlashInfer requires sm75+")
    elif arch == 75:
        # disable bf16 for sm75
        enable_bf16 = False

if enable_bf16:
    torch_cpp_ext.COMMON_NVCC_FLAGS.append("-DFLASHINFER_ENABLE_BF16")
if enable_fp8:
    torch_cpp_ext.COMMON_NVCC_FLAGS.append("-DFLASHINFER_ENABLE_FP8")


def write_if_different(path: pathlib.Path, content: str) -> None:
    if path.exists():
        with open(path, "r") as f:
            if f.read() == content:
                return
    with open(path, "w") as f:
        f.write(content)


def get_instantiation_cu() -> List[str]:
    prefix = "csrc/generated"
    (root / prefix).mkdir(parents=True, exist_ok=True)

    group_sizes = os.environ.get("FLASHINFER_GROUP_SIZES", "1,4,6,8").split(",")
    head_dims = os.environ.get("FLASHINFER_HEAD_DIMS", "64,128,256").split(",")
    kv_layouts = os.environ.get("FLASHINFER_KV_LAYOUTS", "0,1").split(",")
    pos_encoding_modes = os.environ.get("FLASHINFER_POS_ENCODING_MODES", "0,1,2").split(
        ","
    )
    allow_fp16_qk_reduction_options = os.environ.get(
        "FLASHINFER_ALLOW_FP16_QK_REDUCTION_OPTIONS", "0,1"
    ).split(",")
    causal_options = os.environ.get("FLASHINFER_CAUSAL_OPTIONS", "0,1").split(",")
    # dispatch.inc
    path = root / prefix / "dispatch.inc"
    write_if_different(
        path,
        generate_dispatch_inc.get_dispatch_inc_str(
            argparse.Namespace(
                group_sizes=map(int, group_sizes),
                head_dims=map(int, head_dims),
                kv_layouts=map(int, kv_layouts),
                pos_encoding_modes=map(int, pos_encoding_modes),
                allow_fp16_qk_reductions=map(int, allow_fp16_qk_reduction_options),
                causals=map(int, causal_options),
            )
        ),
    )

    idtypes = ["i32"]
    prefill_dtypes = ["f16"]
    decode_dtypes = ["f16"]
    if enable_bf16:
        prefill_dtypes.append("bf16")
        decode_dtypes.append("bf16")
    fp8_dtypes = []
    if enable_fp8:
        fp8_dtypes = ["e4m3", "e5m2"]

    files = []
    # single decode files
    for (
        group_size,
        head_dim,
        kv_layout,
        pos_encoding_mode,
    ) in itertools.product(
        group_sizes,
        head_dims,
        kv_layouts,
        pos_encoding_modes,
    ):
        for dtype in decode_dtypes:
            fname = f"single_decode_group_{group_size}_head_{head_dim}_layout_{kv_layout}_posenc_{pos_encoding_mode}_dtypein_{dtype}_dtypeout_{dtype}.cu"
            files.append(prefix + "/" + fname)
            content = generate_single_decode_inst.get_cu_file_str(
                group_size,
                head_dim,
                kv_layout,
                pos_encoding_mode,
                dtype,
                dtype,
            )
            write_if_different(root / prefix / fname, content)

        for dtype_in in fp8_dtypes:
            dtype_out = "f16"
            fname = f"single_decode_group_{group_size}_head_{head_dim}_layout_{kv_layout}_posenc_{pos_encoding_mode}_dtypein_{dtype_in}_dtypeout_{dtype_out}.cu"
            files.append(prefix + "/" + fname)
            content = generate_single_decode_inst.get_cu_file_str(
                group_size,
                head_dim,
                kv_layout,
                pos_encoding_mode,
                dtype_in,
                dtype_out,
            )
            write_if_different(root / prefix / fname, content)

    # batch decode files
    for (
        group_size,
        head_dim,
        kv_layout,
        pos_encoding_mode,
    ) in itertools.product(
        group_sizes,
        head_dims,
        kv_layouts,
        pos_encoding_modes,
    ):
        for idtype in idtypes:
            for dtype in decode_dtypes:
                fname = f"batch_paged_decode_group_{group_size}_head_{head_dim}_layout_{kv_layout}_posenc_{pos_encoding_mode}_dtypein_{dtype}_dtypeout_{dtype}_idtype_{idtype}.cu"
                files.append(prefix + "/" + fname)
                content = generate_batch_paged_decode_inst.get_cu_file_str(
                    group_size,
                    head_dim,
                    kv_layout,
                    pos_encoding_mode,
                    dtype,
                    dtype,
                    idtype,
                )
                write_if_different(root / prefix / fname, content)

            for dtype_in in fp8_dtypes:
                dtype_out = "f16"
                fname = f"batch_paged_decode_group_{group_size}_head_{head_dim}_layout_{kv_layout}_posenc_{pos_encoding_mode}_dtypein_{dtype_in}_dtypeout_{dtype_out}_idtype_{idtype}.cu"
                files.append(prefix + "/" + fname)
                content = generate_batch_paged_decode_inst.get_cu_file_str(
                    group_size,
                    head_dim,
                    kv_layout,
                    pos_encoding_mode,
                    dtype_in,
                    dtype_out,
                    idtype,
                )
                write_if_different(root / prefix / fname, content)

        for dtype in decode_dtypes:
            fname = f"batch_padded_decode_group_{group_size}_head_{head_dim}_layout_{kv_layout}_posenc_{pos_encoding_mode}_dtypein_{dtype}_dtypeout_{dtype}.cu"
            files.append(prefix + "/" + fname)
            content = generate_batch_padded_decode_inst.get_cu_file_str(
                group_size,
                head_dim,
                kv_layout,
                pos_encoding_mode,
                dtype,
                dtype,
            )
            write_if_different(root / prefix / fname, content)

        for dtype_in in fp8_dtypes:
            dtype_out = "f16"
            fname = f"batch_padded_decode_group_{group_size}_head_{head_dim}_layout_{kv_layout}_posenc_{pos_encoding_mode}_dtypein_{dtype_in}_dtypeout_{dtype_out}.cu"
            files.append(prefix + "/" + fname)
            content = generate_batch_padded_decode_inst.get_cu_file_str(
                group_size,
                head_dim,
                kv_layout,
                pos_encoding_mode,
                dtype_in,
                dtype_out,
            )
            write_if_different(root / prefix / fname, content)

    # single prefill files
    for (
        group_size,
        head_dim,
        kv_layout,
        pos_encoding_mode,
        allow_fp16_qk_reduction,
        causal,
    ) in itertools.product(
        group_sizes,
        head_dims,
        kv_layouts,
        pos_encoding_modes,
        allow_fp16_qk_reduction_options,
        causal_options,
    ):
        for dtype in prefill_dtypes:
            fname = f"single_prefill_group_{group_size}_head_{head_dim}_layout_{kv_layout}_posenc_{pos_encoding_mode}_fp16qkred_{allow_fp16_qk_reduction}_causal_{causal}_dtypein_{dtype}_dtypeout_{dtype}.cu"
            files.append(prefix + "/" + fname)
            content = generate_single_prefill_inst.get_cu_file_str(
                group_size,
                head_dim,
                kv_layout,
                pos_encoding_mode,
                allow_fp16_qk_reduction,
                causal,
                dtype,
                dtype,
            )
            write_if_different(root / prefix / fname, content)

    # batch paged prefill files
    for (
        group_size,
        head_dim,
        kv_layout,
        pos_encoding_mode,
        allow_fp16_qk_reduction,
        causal,
        idtype,
    ) in itertools.product(
        group_sizes,
        head_dims,
        kv_layouts,
        pos_encoding_modes,
        allow_fp16_qk_reduction_options,
        causal_options,
        idtypes,
    ):
        for dtype in prefill_dtypes:
            fname = f"batch_paged_prefill_group_{group_size}_head_{head_dim}_layout_{kv_layout}_posenc_{pos_encoding_mode}_fp16qkred_{allow_fp16_qk_reduction}_causal_{causal}_dtypein_{dtype}_dtypeout_{dtype}_idtype_{idtype}.cu"
            files.append(prefix + "/" + fname)
            content = generate_batch_paged_prefill_inst.get_cu_file_str(
                group_size,
                head_dim,
                kv_layout,
                pos_encoding_mode,
                allow_fp16_qk_reduction,
                causal,
                dtype,
                dtype,
                idtype,
                page_size_choices=[1, 16, 32],
            )
            write_if_different(root / prefix / fname, content)

    # batch ragged prefill files
    for (
        group_size,
        head_dim,
        kv_layout,
        pos_encoding_mode,
        allow_fp16_qk_reduction,
        causal,
        idtype,
    ) in itertools.product(
        group_sizes,
        head_dims,
        kv_layouts,
        pos_encoding_modes,
        allow_fp16_qk_reduction_options,
        causal_options,
        idtypes,
    ):
        for dtype in prefill_dtypes:
            fname = f"batch_ragged_prefill_group_{group_size}_head_{head_dim}_layout_{kv_layout}_posenc_{pos_encoding_mode}_fp16qkred_{allow_fp16_qk_reduction}_causal_{causal}_dtypein_{dtype}_dtypeout_{dtype}_idtype_{idtype}.cu"
            files.append(prefix + "/" + fname)
            content = generate_batch_ragged_prefill_inst.get_cu_file_str(
                group_size,
                head_dim,
                kv_layout,
                pos_encoding_mode,
                allow_fp16_qk_reduction,
                causal,
                dtype,
                dtype,
                idtype,
            )
            write_if_different(root / prefix / fname, content)

    return files


def get_version():
    version = os.getenv("FLASHINFER_BUILD_VERSION")
    if version is None:
        with open(root / "version.txt") as f:
            version = f.read().strip()
    return version


def get_cuda_version() -> Tuple[int, int]:
    if torch_cpp_ext.CUDA_HOME is None:
        nvcc = "nvcc"
    else:
        nvcc = os.path.join(torch_cpp_ext.CUDA_HOME, "bin/nvcc")
    txt = subprocess.check_output([nvcc, "--version"], text=True)
    major, minor = map(int, re.findall(r"release (\d+)\.(\d+),", txt)[0])
    return major, minor


def generate_build_meta() -> None:
    d = {}
    version = get_version()
    d["cuda_major"], d["cuda_minor"] = get_cuda_version()
    d["torch"] = torch.__version__
    d["python"] = platform.python_version()
    d["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    with open(root / "flashinfer/_build_meta.py", "w") as f:
        f.write(f"__version__ = {version!r}\n")
        f.write(f"build_meta = {d!r}")


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


class NinjaBuildExtension(torch_cpp_ext.BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            max_num_jobs_cores = max(1, os.cpu_count() // 2)
            os.environ["MAX_JOBS"] = str(max_num_jobs_cores)

        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    remove_unwanted_pytorch_nvcc_flags()
    generate_build_meta()
    ext_modules = []
    ext_modules.append(
        torch_cpp_ext.CUDAExtension(
            name="flashinfer._kernels",
            sources=[
                "csrc/single_decode.cu",
                "csrc/single_prefill.cu",
                "csrc/cascade.cu",
                "csrc/page.cu",
                "csrc/batch_decode.cu",
                "csrc/flashinfer_ops.cu",
                "csrc/batch_prefill.cu",
            ]
            + get_instantiation_cu(),
            include_dirs=[
                str(root.resolve() / "include"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--threads",
                    "8",
                    "-Xfatbin",
                    "-compress-all",
                ],
            },
        )
    )

    setuptools.setup(
        name="flashinfer",
        version=get_version(),
        packages=setuptools.find_packages(),
        author="FlashInfer team",
        license="Apache License 2.0",
        description="FlashInfer: Kernel Library for LLM Serving",
        url="https://github.com/flashinfer-ai/flashinfer",
        python_requires=">=3.8",
        ext_modules=ext_modules,
        cmdclass={"build_ext": NinjaBuildExtension},
    )
