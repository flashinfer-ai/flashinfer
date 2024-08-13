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

import generate_single_decode_inst, generate_single_prefill_inst, generate_batch_paged_decode_inst, generate_batch_paged_prefill_inst, generate_batch_ragged_prefill_inst, generate_dispatch_inc

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


def get_instantiation_cu() -> Tuple[List[str], List[str]]:
    prefix = "csrc/generated"
    (root / prefix).mkdir(parents=True, exist_ok=True)

    logits_hooks = os.environ.get("FLASHINFER_LOGITS_POST_HOOKS", "0,1").split(",")
    head_dims = os.environ.get("FLASHINFER_HEAD_DIMS", "64,128,256").split(",")
    pos_encoding_modes = os.environ.get("FLASHINFER_POS_ENCODING_MODES", "0,1,2").split(
        ","
    )
    allow_fp16_qk_reduction_options = os.environ.get(
        "FLASHINFER_ALLOW_FP16_QK_REDUCTION_OPTIONS", "0,1"
    ).split(",")
    mask_modes = os.environ.get("FLASHINFER_MASK_MODES", "0,1,2").split(",")
    # dispatch.inc
    path = root / prefix / "dispatch.inc"
    write_if_different(
        path,
        generate_dispatch_inc.get_dispatch_inc_str(
            argparse.Namespace(
                head_dims=map(int, head_dims),
                logits_post_hooks=map(int, logits_hooks),
                pos_encoding_modes=map(int, pos_encoding_modes),
                allow_fp16_qk_reductions=map(int, allow_fp16_qk_reduction_options),
                mask_modes=map(int, mask_modes),
            )
        ),
    )

    idtypes = ["i32"]
    prefill_dtypes = ["f16"]
    decode_dtypes = ["f16"]
    fp16_dtypes = ["f16"]
    fp8_dtypes = ["e4m3", "e5m2"]
    if enable_bf16:
        prefill_dtypes.append("bf16")
        decode_dtypes.append("bf16")
        fp16_dtypes.append("bf16")
    if enable_fp8:
        decode_dtypes.extend(fp8_dtypes)

    files_decode = []
    files_prefill = []
    # single decode files
    for (
        head_dim,
        logits_hook,
        pos_encoding_mode,
    ) in itertools.product(
        head_dims,
        logits_hooks,
        pos_encoding_modes,
    ):
        for dtype_q, dtype_kv in list(zip(decode_dtypes, decode_dtypes)) + list(
            itertools.product(fp16_dtypes, fp8_dtypes)
        ):
            dtype_out = dtype_q
            fname = f"single_decode_head_{head_dim}_logitshook_{logits_hook}_posenc_{pos_encoding_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_out}.cu"
            files_decode.append(prefix + "/" + fname)
            content = generate_single_decode_inst.get_cu_file_str(
                head_dim,
                logits_hook,
                pos_encoding_mode,
                dtype_q,
                dtype_kv,
                dtype_out,
            )
            write_if_different(root / prefix / fname, content)

    # batch decode files
    for (
        head_dim,
        logits_hook,
        pos_encoding_mode,
    ) in itertools.product(
        head_dims,
        logits_hooks,
        pos_encoding_modes,
    ):
        for idtype in idtypes:
            for dtype_q, dtype_kv in list(zip(decode_dtypes, decode_dtypes)) + list(
                itertools.product(fp16_dtypes, fp8_dtypes)
            ):
                dtype_out = dtype_q
                fname = f"batch_paged_decode_head_{head_dim}_logitshook_{logits_hook}_posenc_{pos_encoding_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_out}_idtype_{idtype}.cu"
                files_decode.append(prefix + "/" + fname)
                content = generate_batch_paged_decode_inst.get_cu_file_str(
                    head_dim,
                    logits_hook,
                    pos_encoding_mode,
                    dtype_q,
                    dtype_kv,
                    dtype_out,
                    idtype,
                )
                write_if_different(root / prefix / fname, content)

    # single prefill files
    for (
        head_dim,
        logits_hook,
        pos_encoding_mode,
        allow_fp16_qk_reduction,
        mask_mode,
    ) in itertools.product(
        head_dims,
        logits_hooks,
        pos_encoding_modes,
        allow_fp16_qk_reduction_options,
        mask_modes,
    ):
        for dtype_q, dtype_kv in list(zip(prefill_dtypes, prefill_dtypes)):
            fname = f"single_prefill_head_{head_dim}_logitshook_{logits_hook}_posenc_{pos_encoding_mode}_fp16qkred_{allow_fp16_qk_reduction}_mask_{mask_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_q}.cu"
            files_prefill.append(prefix + "/" + fname)
            content = generate_single_prefill_inst.get_cu_file_str(
                head_dim,
                logits_hook,
                pos_encoding_mode,
                allow_fp16_qk_reduction,
                mask_mode,
                dtype_q,  # dtype_q
                dtype_kv,  # dtype_kv
                dtype_q,  # dtype_out
            )
            write_if_different(root / prefix / fname, content)

    # batch paged prefill files
    for (
        head_dim,
        logits_hook,
        pos_encoding_mode,
        allow_fp16_qk_reduction,
        mask_mode,
        idtype,
    ) in itertools.product(
        head_dims,
        logits_hooks,
        pos_encoding_modes,
        allow_fp16_qk_reduction_options,
        mask_modes,
        idtypes,
    ):
        for dtype_q, dtype_kv in list(zip(prefill_dtypes, prefill_dtypes)) + list(
            itertools.product(prefill_dtypes, fp8_dtypes)
        ):
            fname = f"batch_paged_prefill_head_{head_dim}_logitshook_{logits_hook}_posenc_{pos_encoding_mode}_fp16qkred_{allow_fp16_qk_reduction}_mask_{mask_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_q}_idtype_{idtype}.cu"
            files_prefill.append(prefix + "/" + fname)
            content = generate_batch_paged_prefill_inst.get_cu_file_str(
                head_dim,
                logits_hook,
                pos_encoding_mode,
                allow_fp16_qk_reduction,
                mask_mode,
                dtype_q,  # dtype_q
                dtype_kv,  # dtype_kv
                dtype_q,  # dtype_out
                idtype,
            )
            write_if_different(root / prefix / fname, content)

    # batch ragged prefill files
    for (
        head_dim,
        logits_hook,
        pos_encoding_mode,
        allow_fp16_qk_reduction,
        mask_mode,
        idtype,
    ) in itertools.product(
        head_dims,
        logits_hooks,
        pos_encoding_modes,
        allow_fp16_qk_reduction_options,
        mask_modes,
        idtypes,
    ):
        for dtype_q, dtype_kv in list(zip(prefill_dtypes, prefill_dtypes)):
            fname = f"batch_ragged_prefill_head_{head_dim}_logitshook_{logits_hook}_posenc_{pos_encoding_mode}_fp16qkred_{allow_fp16_qk_reduction}_mask_{mask_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_q}_idtype_{idtype}.cu"
            files_prefill.append(prefix + "/" + fname)
            content = generate_batch_ragged_prefill_inst.get_cu_file_str(
                head_dim,
                logits_hook,
                pos_encoding_mode,
                allow_fp16_qk_reduction,
                mask_mode,
                dtype_q,  # dtype_q
                dtype_kv,  # dtype_kv
                dtype_q,  # dtype_out
                idtype,
            )
            write_if_different(root / prefix / fname, content)

    return files_prefill, files_decode


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
            max_num_jobs_cores = max(1, os.cpu_count())
            os.environ["MAX_JOBS"] = str(max_num_jobs_cores)

        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    remove_unwanted_pytorch_nvcc_flags()
    generate_build_meta()
    files_prefill, files_decode = get_instantiation_cu()
    include_dirs = [
        str(root.resolve() / "include"),
        str(root.resolve() / "3rdparty" / "cutlass" / "include"),  # for group gemm
    ]
    extra_compile_args = {
        "cxx": [
            "-O3",
            "-Wno-switch-bool",
        ],
        "nvcc": [
            "-O3",
            "-std=c++17",
            "--threads",
            "1",
            "-Xfatbin",
            "-compress-all",
            "-use_fast_math",
        ],
    }
    ext_modules = []
    ext_modules.append(
        torch_cpp_ext.CUDAExtension(
            name="flashinfer._kernels",
            sources=[
                "csrc/cascade.cu",
                "csrc/page.cu",
                "csrc/flashinfer_ops.cu",
                "csrc/sampling.cu",
                "csrc/norm.cu",
                "csrc/activation.cu",
                "csrc/rope.cu",
                "csrc/group_gemm.cu",
                "csrc/quantization.cu",
            ],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        )
    )
    ext_modules.append(
        torch_cpp_ext.CUDAExtension(
            name="flashinfer._decode",
            sources=[
                "csrc/single_decode.cu",
                "csrc/flashinfer_ops_decode.cu",
                "csrc/batch_decode.cu",
            ]
            + files_decode,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        )
    )
    ext_modules.append(
        torch_cpp_ext.CUDAExtension(
            name="flashinfer._prefill",
            sources=[
                "csrc/single_prefill.cu",
                "csrc/flashinfer_ops_prefill.cu",
                "csrc/batch_prefill.cu",
            ]
            + files_prefill,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
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
