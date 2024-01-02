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
import pathlib
import os
import re
import itertools
import subprocess
import platform

import setuptools
import torch
import torch.utils.cpp_extension as torch_cpp_ext

root = pathlib.Path(__name__).parent


def get_instantiation_cu() -> list[str]:
    prefix = "csrc/generated"
    (root / prefix).mkdir(parents=True, exist_ok=True)
    dtypes = {"fp16": "nv_half", "bf16": "nv_bfloat16"}
    group_sizes = os.environ.get("FLASHINFER_GROUP_SIZES", "1,4,8").split(",")
    head_dims = os.environ.get("FLASHINFER_HEAD_DIMS", "64, 128").split(",")
    group_sizes = [int(x) for x in group_sizes]
    head_dims = [int(x) for x in head_dims]
    causal_options = [False, True]
    allow_fp16_qk_reduction_options = [False, True]
    layout_options = ["HND", "NHD"]
    rotary_mode_options = ["None", "Llama"]

    # dispatch.inc
    path = root / prefix / "dispatch.inc"
    if not path.exists():
        with open(root / prefix / "dispatch.inc", "w") as f:
            f.write("#define _DISPATCH_CASES_group_size(...)      \\\n")
            for x in group_sizes:
                f.write(f"  _DISPATCH_CASE({x}, GROUP_SIZE, __VA_ARGS__) \\\n")
            f.write("// EOL\n")

            f.write("#define _DISPATCH_CASES_head_dim(...)        \\\n")
            for x in head_dims:
                f.write(f"  _DISPATCH_CASE({x}, HEAD_DIM, __VA_ARGS__) \\\n")
            f.write("// EOL\n")
            f.write("\n")

    files = []
    for (
        group_size,
        head_dim,
        dtype,
        causal,
        allow_fp16_qk_reduction,
        layout,
        rotary_mode,
    ) in itertools.product(
        group_sizes,
        head_dims,
        dtypes,
        causal_options,
        allow_fp16_qk_reduction_options,
        layout_options,
        rotary_mode_options,
    ):
        # paged batch prefill
        fname = f"paged_batch_prefill_group{group_size}_head{head_dim}_causal{causal}_fp16qk{allow_fp16_qk_reduction}_layout{layout}_rotary{rotary_mode}_{dtype}.cu"
        files.append(prefix + "/" + fname)
        if not (root / prefix / fname).exists():
            with open(root / prefix / fname, "w") as f:
                f.write('#include "../flashinfer_decl.h"\n\n')
                f.write(f"#include <flashinfer.cuh>\n\n")
                f.write(f"using namespace flashinfer;\n\n")
                f.write(
                    "INST_BatchPrefillPagedWrapper({}, {}, {}, {}, {}, {}, {})\n".format(
                        dtypes[dtype],
                        group_size,
                        head_dim,
                        str(causal).lower(),
                        str(allow_fp16_qk_reduction).lower(),
                        "QKVLayout::k" + layout,
                        "RotaryMode::k" + rotary_mode,
                    )
                )

        # ragged batch prefill
        fname = f"ragged_batch_prefill_group{group_size}_head{head_dim}_causal{causal}_fp16qk{allow_fp16_qk_reduction}_layout{layout}_rotary{rotary_mode}_{dtype}.cu"
        files.append(prefix + "/" + fname)
        if not (root / prefix / fname).exists():
            with open(root / prefix / fname, "w") as f:
                f.write('#include "../flashinfer_decl.h"\n\n')
                f.write(f"#include <flashinfer.cuh>\n\n")
                f.write(f"using namespace flashinfer;\n\n")
                f.write(
                    "INST_BatchPrefillRaggedWrapper({}, {}, {}, {}, {}, {}, {})\n".format(
                        dtypes[dtype],
                        group_size,
                        head_dim,
                        str(causal).lower(),
                        str(allow_fp16_qk_reduction).lower(),
                        "QKVLayout::k" + layout,
                        "RotaryMode::k" + rotary_mode,
                    )
                )

        # single prefill
        fname = f"single_prefill_group{group_size}_head{head_dim}_causal{causal}_fp16qk{allow_fp16_qk_reduction}_layout{layout}_rotary{rotary_mode}_{dtype}.cu"
        files.append(prefix + "/" + fname)
        if not (root / prefix / fname).exists():
            with open(root / prefix / fname, "w") as f:
                f.write('#include "../flashinfer_decl.h"\n\n')
                f.write(f"#include <flashinfer.cuh>\n\n")
                f.write(f"using namespace flashinfer;\n\n")
                f.write(
                    "INST_SinglePrefill({}, {}, {}, {}, {}, {}, {})\n".format(
                        dtypes[dtype],
                        group_size,
                        head_dim,
                        str(causal).lower(),
                        str(allow_fp16_qk_reduction).lower(),
                        "QKVLayout::k" + layout,
                        "RotaryMode::k" + rotary_mode,
                    )
                )

    return files


def get_version():
    version = os.getenv("FLASHINFER_BUILD_VERSION")
    if version is None:
        with open(root / "version.txt") as f:
            version = f.read().strip()
    return version


def get_cuda_version() -> tuple[int, int]:
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


if __name__ == "__main__":
    enable_bf16 = True
    for cuda_arch_flags in torch_cpp_ext._get_cuda_arch_flags():
        arch = int(re.search("compute_\d+", cuda_arch_flags).group()[-2:])
        if arch < 75:
            raise RuntimeError("FlashInfer requires sm75+")
        elif arch == 75:
            # disable bf16 for sm75
            enable_bf16 = False

    if enable_bf16:
        torch_cpp_ext.COMMON_NVCC_FLAGS.append("-DFLASHINFER_ENABLE_BF16")

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
                "nvcc": ["-O3"],
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
        python_requires=">=3.9",
        ext_modules=ext_modules,
        cmdclass={"build_ext": torch_cpp_ext.BuildExtension},
    )
