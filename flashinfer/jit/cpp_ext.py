# SPDX - FileCopyrightText : 2023 - 2025 Flashinfer team
# SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
#
# SPDX - License - Identifier : Apache 2.0

# Adapted from https://github.com/pytorch/pytorch/blob/v2.7.0/torch/utils/cpp_extension.py

import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.cpp_extension import (
    _TORCH_PATH,
    CUDA_HOME,
    ROCM_HOME,
    _get_cuda_arch_flags,
    _get_num_workers,
    _get_pybind11_abi_build_flags,
    _get_rocm_arch_flags,
)

from . import env as jit_env


def _get_glibcxx_abi_build_flags() -> List[str]:
    glibcxx_abi_cflags = [
        "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))
    ]
    return glibcxx_abi_cflags


def join_multiline(vs: List[str]) -> str:
    return " $\n    ".join(vs)


def check_hip_availability() -> bool:
    hip_avail = (
        hasattr(torch, "cuda") and torch.cuda.is_available() and torch.version.hip
    )
    return hip_avail


def generate_ninja_build_for_op(
    name: str,
    sources: List[Path],
    extra_cflags: Optional[List[str]],
    extra_cuda_cflags: Optional[List[str]],
    extra_ldflags: Optional[List[str]],
    extra_include_dirs: Optional[List[Path]],
) -> str:
    system_includes = [
        sysconfig.get_path("include"),
        "$torch_home/include",
        "$torch_home/include/torch/csrc/api/include",
        "$cuda_home/include",
        jit_env.FLASHINFER_INCLUDE_DIR.resolve(),
        jit_env.FLASHINFER_CSRC_DIR.resolve(),
    ]

    if not check_hip_availability():
        system_includes += [p.resolve() for p in jit_env.CUTLASS_INCLUDE_DIRS]

    common_cflags = [
        "-DTORCH_EXTENSION_NAME=$name",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-DPy_LIMITED_API=0x03090000",
        "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1",
        "-DFLASHINFER_ENABLE_F16",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
    ]
    common_cflags += _get_pybind11_abi_build_flags()
    common_cflags += _get_glibcxx_abi_build_flags()
    if extra_include_dirs is not None:
        for dir in extra_include_dirs:
            common_cflags.append(f"-I{dir.resolve()}")
    for dir in system_includes:
        common_cflags.append(f"-isystem {dir}")

    cflags = [
        "--offload-arch=gfx942",
        "-fPIC",
    ]
    cflags += common_cflags
    if extra_cflags is not None:
        cflags += extra_cflags

    cuda_cflags: List[str] = []
    cc_env = os.environ.get("CC")
    if cc_env is not None:
        cuda_cflags += ["-ccbin", cc_env]
    cuda_cflags += [
        "$cflags",
        "-fPIC",
        "-DFLASHINFER_ENABLE_HIP",
    ]

    if extra_cuda_cflags is not None:
        cuda_cflags += extra_cuda_cflags

    ldflags = []
    if check_hip_availability():
        ldflags += [
            "-shared",
            "-L$torch_home/lib",
            "-lc10",
            "-lc10_hip",
            "-ltorch_cpu",
            "-ltorch_hip",
            "-ltorch",
            "-L$rocm_home/lib",
            "-lamdhip64",
        ]
    else:
        ldflags += [
            "-shared",
            "-L$torch_home/lib",
            "-lc10",
            "-lc10_cuda",
            "-ltorch_cpu",
            "-ltorch_cuda",
            "-ltorch",
            "-L$cuda_home/lib64",
            "-lcudart",
        ]
    if extra_ldflags is not None:
        ldflags += extra_ldflags

    cxx = os.environ.get("CXX", "c++")
    cuda_home = CUDA_HOME or "/usr/local/cuda"
    rocm_home = ROCM_HOME or "/opt/rocm"
    nvcc = os.environ.get("PYTORCH_NVCC", "$cuda_home/bin/nvcc")
    amdclang = os.environ.get("PYTORCH_AMDCLANG", "$rocm_home/bin/amdclang++")

    lines = []

    if check_hip_availability():
        lines += [
            "ninja_required_version = 1.3",
            f"name = {name}",
            f"rocm_home = {rocm_home}",
            f"torch_home = {_TORCH_PATH}",
            f"cxx = {cxx}",
            f"amdclang = {amdclang}",
            "",
            "common_cflags = " + join_multiline(common_cflags),
            "cflags = " + join_multiline(cflags),
            "post_cflags =",
            "cuda_cflags = " + join_multiline(cuda_cflags),
            "cuda_post_cflags =",
            "ldflags = " + join_multiline(ldflags),
            "",
            "rule compile",
            "  command = $cxx -MF $out.d $cflags -c $in -o $out $post_cflags $common_cflags",
            "  depfile = $out.d",
            "  deps = gcc",
            "",
            "rule hip_compile",
            "  command = $amdclang -xhip -MD -MF $out.d $cuda_cflags -c $in -o $out $cuda_post_cflags $cflags",
            "  depfile = $out.d",
            "  deps = gcc",
            "",
            "rule link",
            "  command = $cxx $in $ldflags -o $out",
            "",
        ]
    else:
        lines += [
            "ninja_required_version = 1.3",
            f"name = {name}",
            f"cuda_home = {cuda_home}",
            f"torch_home = {_TORCH_PATH}",
            f"cxx = {cxx}",
            f"nvcc = {nvcc}",
            "",
            "common_cflags = " + join_multiline(common_cflags),
            "cflags = " + join_multiline(cflags),
            "post_cflags =",
            "cuda_cflags = " + join_multiline(cuda_cflags),
            "cuda_post_cflags =",
            "ldflags = " + join_multiline(ldflags),
            "",
            "rule compile",
            "  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags",
            "  depfile = $out.d",
            "  deps = gcc",
            "",
            "rule cuda_compile",
            "  command = $nvcc --generate-dependencies-with-compile --dependency-output $out.d $cuda_cflags -c $in -o $out $cuda_post_cflags",
            "  depfile = $out.d",
            "  deps = gcc",
            "",
            "rule link",
            "  command = $cxx $in $ldflags -o $out",
            "",
        ]

    objects = []
    for source in sources:
        is_cuda = source.suffix == ".cu"
        object_suffix = ".o"
        cmd = ""
        if is_cuda and check_hip_availability():
            cmd = "hip_compile"
        elif is_cuda and not check_hip_availability():
            cmd = "cuda_compile"
        else:
            cmd = "compile"
        obj_name = source.with_suffix(object_suffix).name
        obj = f"$name/{obj_name}"
        objects.append(obj)
        lines.append(f"build {obj}: {cmd} {source.resolve()}")

    lines.append("")
    lines.append("build $name/$name.so: link " + " ".join(objects))
    lines.append("default $name/$name.so")
    lines.append("")

    return "\n".join(lines)


def run_ninja(workdir: Path, ninja_file: Path, verbose: bool) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    command = [
        "ninja",
        "-v",
        "-C",
        str(workdir.resolve()),
        "-f",
        str(ninja_file.resolve()),
    ]
    num_workers = _get_num_workers(verbose)
    if num_workers is not None:
        command += ["-j", str(num_workers)]

    sys.stdout.flush()
    sys.stderr.flush()
    try:
        subprocess.run(
            command,
            stdout=None if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(workdir.resolve()),
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        msg = "Ninja build failed."
        if e.output:
            msg += " Ninja output:\n" + e.output
        raise RuntimeError(msg) from e
