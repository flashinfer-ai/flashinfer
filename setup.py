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

import argparse
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import setuptools

root = Path(__file__).parent.resolve()
gen_dir = root / "csrc" / "generated"
build_meta = root / "flashinfer" / "_build_meta.py"

head_dims = os.environ.get("FLASHINFER_HEAD_DIMS", "64,128,256").split(",")
pos_encoding_modes = os.environ.get("FLASHINFER_POS_ENCODING_MODES", "0").split(",")
allow_fp16_qk_reductions = os.environ.get(
    "FLASHINFER_ALLOW_FP16_QK_REDUCTION_OPTIONS", "0"
).split(",")
mask_modes = os.environ.get("FLASHINFER_MASK_MODES", "0,1,2").split(",")

head_dims = list(map(int, head_dims))
pos_encoding_modes = list(map(int, pos_encoding_modes))
allow_fp16_qk_reductions = list(map(int, allow_fp16_qk_reductions))
mask_modes = list(map(int, mask_modes))

enable_aot = os.environ.get("FLASHINFER_ENABLE_AOT", "0") == "1"
enable_bf16 = os.environ.get("FLASHINFER_ENABLE_BF16", "1") == "1"
enable_fp8 = os.environ.get("FLASHINFER_ENABLE_FP8", "1") == "1"


def generate_cuda() -> None:
    try:  # no aot_build_utils in sdist
        sys.path.append(str(root))
        from aot_build_utils.generate import get_instantiation_cu
    except ImportError:
        return

    aot_kernel_uris = get_instantiation_cu(
        argparse.Namespace(
            path=gen_dir,
            head_dims=head_dims,
            pos_encoding_modes=pos_encoding_modes,
            allow_fp16_qk_reductions=allow_fp16_qk_reductions,
            mask_modes=mask_modes,
            enable_bf16=enable_bf16,
            enable_fp8=enable_fp8,
        )
    )
    aot_config_str = f"""prebuilt_ops_uri = set({aot_kernel_uris})"""
    (root / "flashinfer" / "jit" / "aot_config.py").write_text(aot_config_str)


ext_modules = []
cmdclass = {}
use_scm_version = {}
install_requires = ["torch", "ninja"]
build_meta.write_text("\n")
generate_cuda()

if enable_aot:
    import torch
    import torch.utils.cpp_extension as torch_cpp_ext
    from packaging.version import Version
    from setuptools_scm.version import get_local_node_and_date as default

    def get_cuda_version() -> Version:
        if torch_cpp_ext.CUDA_HOME is None:
            nvcc = "nvcc"
        else:
            nvcc = os.path.join(torch_cpp_ext.CUDA_HOME, "bin/nvcc")
        txt = subprocess.check_output([nvcc, "--version"], text=True)
        return Version(re.findall(r"release (\d+\.\d+),", txt)[0])

    class NinjaBuildExtension(torch_cpp_ext.BuildExtension):
        def __init__(self, *args, **kwargs) -> None:
            # do not override env MAX_JOBS if already exists
            if not os.environ.get("MAX_JOBS"):
                max_num_jobs_cores = max(1, os.cpu_count())
                os.environ["MAX_JOBS"] = str(max_num_jobs_cores)

            super().__init__(*args, **kwargs)

    # cuda arch check for fp8 at the moment.
    for cuda_arch_flags in torch_cpp_ext._get_cuda_arch_flags():
        arch = int(re.search(r"compute_(\d+)", cuda_arch_flags).group(1))
        if arch < 75:
            raise RuntimeError("FlashInfer requires sm75+")

    cuda_version = get_cuda_version()
    torch_full_version = Version(torch.__version__)
    torch_version = f"{torch_full_version.major}.{torch_full_version.minor}"
    local_version = f"cu{cuda_version.major}{cuda_version.minor}torch{torch_version}"

    aot_build_meta = {}
    aot_build_meta["cuda_major"] = cuda_version.major
    aot_build_meta["cuda_minor"] = cuda_version.minor
    aot_build_meta["torch"] = torch_version
    aot_build_meta["python"] = platform.python_version()
    aot_build_meta["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST")
    build_meta.write_text(f"build_meta = {aot_build_meta!r}\n")

    cmdclass["build_ext"] = NinjaBuildExtension
    use_scm_version["local_scheme"] = lambda x: f"{default(x)}.{local_version}"
    install_requires = [f"torch == {torch_version}"]

    if enable_bf16:
        torch_cpp_ext.COMMON_NVCC_FLAGS.append("-DFLASHINFER_ENABLE_BF16")
    if enable_fp8:
        torch_cpp_ext.COMMON_NVCC_FLAGS.append("-DFLASHINFER_ENABLE_FP8")

    for flag in [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]:
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass

    cutlass = root / "3rdparty" / "cutlass"
    include_dirs = [
        root.resolve() / "include",
        cutlass.resolve() / "include",  # for group gemm
        cutlass.resolve() / "tools" / "util" / "include",
    ]
    cxx_flags = [
        "-O3",
        "-Wno-switch-bool",
    ]
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "--threads=1",
        "-Xfatbin",
        "-compress-all",
        "-use_fast_math",
    ]
    sm90a_flags = "-gencode arch=compute_90a,code=sm_90a".split()
    kernel_sources = [
        "csrc/bmm_fp8.cu",
        "csrc/cascade.cu",
        "csrc/group_gemm.cu",
        "csrc/norm.cu",
        "csrc/page.cu",
        "csrc/quantization.cu",
        "csrc/rope.cu",
        "csrc/sampling.cu",
        "csrc/renorm.cu",
        "csrc/activation.cu",
        "csrc/batch_decode.cu",
        "csrc/batch_prefill.cu",
        "csrc/single_decode.cu",
        "csrc/single_prefill.cu",
        "csrc/flashinfer_ops.cu",
    ]
    kernel_sm90_sources = [
        "csrc/group_gemm_sm90.cu",
        "csrc/flashinfer_gemm_sm90_ops.cu",
    ]
    decode_sources = list(gen_dir.glob("*decode_head*.cu"))
    prefill_sources = list(gen_dir.glob("*prefill_head*.cu"))
    ext_modules = [
        torch_cpp_ext.CUDAExtension(
            name="flashinfer._kernels",
            sources=kernel_sources + decode_sources + prefill_sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
            py_limited_api=True,
        ),
        torch_cpp_ext.CUDAExtension(
            name="flashinfer._kernels_sm90",
            sources=kernel_sm90_sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags + sm90a_flags,
            },
            py_limited_api=True,
        ),
    ]

setuptools.setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    options={"bdist_wheel": {"py_limited_api": "cp38"}},
    install_requires=install_requires,
    use_scm_version=use_scm_version,
)
