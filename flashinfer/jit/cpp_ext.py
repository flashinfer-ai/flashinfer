# Adapted from https://github.com/pytorch/pytorch/blob/v2.7.0/torch/utils/cpp_extension.py

import functools
import os
import re
import subprocess
import sys
import sysconfig
from packaging.version import Version
from pathlib import Path
from typing import List, Optional

import tvm_ffi
import torch

from . import env as jit_env
from ..compilation_context import CompilationContext


def parse_env_flags(env_var_name) -> List[str]:
    env_flags = os.environ.get(env_var_name)
    if env_flags:
        try:
            import shlex

            return shlex.split(env_flags)
        except ValueError as e:
            print(
                f"Warning: Could not parse {env_var_name} with shlex: {e}. Falling back to simple split.",
                file=sys.stderr,
            )
            return env_flags.split()
    return []


def _get_glibcxx_abi_build_flags() -> List[str]:
    glibcxx_abi_cflags = [
        "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))
    ]
    return glibcxx_abi_cflags


@functools.cache
def get_cuda_path() -> str:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is not None:
        return cuda_home
    # get output of "which nvcc"
    nvcc_path = subprocess.run(["which", "nvcc"], capture_output=True)
    if nvcc_path.returncode == 0:
        cuda_home = os.path.dirname(
            os.path.dirname(nvcc_path.stdout.decode("utf-8").strip())
        )
    else:
        cuda_home = "/usr/local/cuda"  # This default value is from: https://github.com/pytorch/pytorch/blob/ceb11a584d6b3fdc600358577d9bf2644f88def9/torch/utils/cpp_extension.py#L115
        if not os.path.exists(cuda_home):
            raise RuntimeError(
                f"Could not find nvcc and default {cuda_home=} doesn't exist"
            )
    return cuda_home


@functools.cache
def get_cuda_version() -> Version:
    # Try to query nvcc for CUDA version; if nvcc is unavailable, fall back to torch.version.cuda
    try:
        cuda_home = get_cuda_path()
        nvcc = os.path.join(cuda_home, "bin/nvcc")
        txt = subprocess.check_output([nvcc, "--version"], text=True)
        matches = re.findall(r"release (\d+\.\d+),", txt)
        if not matches:
            raise RuntimeError(
                f"Could not parse CUDA version from nvcc --version output: {txt}"
            )
        return Version(matches[0])
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as e:
        # NOTE(Zihao): when nvcc is unavailable, fall back to torch.version.cuda
        if torch.version.cuda is None:
            raise RuntimeError(
                "nvcc not found and PyTorch is not built with CUDA support. "
                "Could not determine CUDA version."
            ) from e
        return Version(torch.version.cuda)


def is_cuda_version_at_least(version_str: str) -> bool:
    return get_cuda_version() >= Version(version_str)


def join_multiline(vs: List[str]) -> str:
    return " $\n    ".join(vs)


def generate_ninja_build_for_op(
    name: str,
    sources: List[Path],
    extra_cflags: Optional[List[str]],
    extra_cuda_cflags: Optional[List[str]],
    extra_ldflags: Optional[List[str]],
    extra_include_dirs: Optional[List[Path]],
    needs_device_linking: bool = False,
) -> str:
    system_includes = [
        sysconfig.get_path("include"),
        "$cuda_home/include",
        "$cuda_home/include/cccl",
        tvm_ffi.libinfo.find_include_path(),
        tvm_ffi.libinfo.find_dlpack_include_path(),
        jit_env.FLASHINFER_INCLUDE_DIR.resolve(),
        jit_env.FLASHINFER_CSRC_DIR.resolve(),
    ]
    system_includes += [p.resolve() for p in jit_env.CUTLASS_INCLUDE_DIRS]
    system_includes.append(jit_env.SPDLOG_INCLUDE_DIR.resolve())

    cuda_home = get_cuda_path()
    if cuda_home == "/usr":
        # NOTE: this will resolve to /usr/include, which will mess up includes. See #1793
        system_includes.remove("$cuda_home/include")

    common_cflags = []
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        common_cflags.append("-DPy_LIMITED_API=0x03090000")
    common_cflags += _get_glibcxx_abi_build_flags()
    if extra_include_dirs is not None:
        for extra_dir in extra_include_dirs:
            common_cflags.append(f"-I{extra_dir.resolve()}")
    for sys_dir in system_includes:
        common_cflags.append(f"-isystem {sys_dir}")

    cflags = [
        "$common_cflags",
        "-fPIC",
    ]
    if extra_cflags is not None:
        cflags += extra_cflags

    cuda_cflags: List[str] = []
    cc_env = os.environ.get("CC")
    if cc_env is not None:
        cuda_cflags += ["-ccbin", cc_env]
    cuda_cflags += [
        "$common_cflags",
        "--compiler-options=-fPIC",
        "--expt-relaxed-constexpr",
    ]
    cuda_version = get_cuda_version()
    # enable -static-global-template-stub when cuda version >= 12.8
    if cuda_version >= Version("12.8"):
        cuda_cflags += [
            "-static-global-template-stub=false",
        ]

    cpp_ext_initial_compilation_context = CompilationContext()
    global_flags = cpp_ext_initial_compilation_context.get_nvcc_flags_list()
    if extra_cuda_cflags is not None:
        # Check if module provides architecture flags
        module_has_gencode = any(
            flag.startswith("-gencode=") for flag in extra_cuda_cflags
        )

        if module_has_gencode:
            # Use module's architecture flags, but keep global non-architecture flags
            global_non_arch_flags = [
                flag for flag in global_flags if not flag.startswith("-gencode=")
            ]
            cuda_cflags += global_non_arch_flags + extra_cuda_cflags
        else:
            # No module architecture flags, use both global and module flags
            cuda_cflags += global_flags + extra_cuda_cflags
    else:
        # No module flags, use global flags
        cuda_cflags += global_flags

    ldflags = [
        "-shared",
        "-L$cuda_home/lib64",
        "-L$cuda_home/lib64/stubs",
        "-lcudart",
        "-lcuda",
    ]

    env_extra_ldflags = parse_env_flags("FLASHINFER_EXTRA_LDFLAGS")
    if env_extra_ldflags is not None:
        ldflags += env_extra_ldflags

    if extra_ldflags is not None:
        ldflags += extra_ldflags

    extra_cflags = parse_env_flags("FLASHINFER_EXTRA_CFLAGS")
    if extra_cflags is not None:
        cflags += extra_cflags

    extra_cuda_cflags = parse_env_flags("FLASHINFER_EXTRA_CUDAFLAGS")
    if extra_cuda_cflags is not None:
        cuda_cflags += extra_cuda_cflags

    cxx = os.environ.get("CXX", "c++")
    nvcc = os.environ.get("FLASHINFER_NVCC", "$cuda_home/bin/nvcc")

    lines = [
        "ninja_required_version = 1.3",
        f"name = {name}",
        f"cuda_home = {cuda_home}",
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
    ]

    # Add nvcc linking rule for device code
    if needs_device_linking:
        lines.extend(
            [
                "rule nvcc_link",
                "  command = $nvcc -shared $in $ldflags -o $out",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "rule link",
                "  command = $cxx $in $ldflags -o $out",
                "",
            ]
        )

    objects = []
    for source in sources:
        is_cuda = source.suffix == ".cu"
        object_suffix = ".cuda.o" if is_cuda else ".o"
        cmd = "cuda_compile" if is_cuda else "compile"
        obj_name = source.with_suffix(object_suffix).name
        obj = f"$name/{obj_name}"
        objects.append(obj)
        lines.append(f"build {obj}: {cmd} {source.resolve()}")

    lines.append("")
    link_rule = "nvcc_link" if needs_device_linking else "link"
    lines.append(f"build $name/$name.so: {link_rule} " + " ".join(objects))
    lines.append("default $name/$name.so")
    lines.append("")

    return "\n".join(lines)


def _get_num_workers() -> Optional[int]:
    max_jobs = os.environ.get("MAX_JOBS")
    if max_jobs is not None and max_jobs.isdigit():
        return int(max_jobs)
    return None


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
    num_workers = _get_num_workers()
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
