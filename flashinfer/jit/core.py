import dataclasses
import logging
import os
import re
import warnings
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.utils.cpp_extension as torch_cpp_ext
from filelock import FileLock

from . import env as jit_env
from .cpp_ext import generate_ninja_build_for_op, run_ninja
from .utils import write_if_different

os.makedirs(jit_env.FLASHINFER_WORKSPACE_DIR, exist_ok=True)
os.makedirs(jit_env.FLASHINFER_CSRC_DIR, exist_ok=True)


class FlashInferJITLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        self.setLevel(logging.INFO)
        self.addHandler(logging.StreamHandler())
        log_path = jit_env.FLASHINFER_WORKSPACE_DIR / "flashinfer_jit.log"
        if not os.path.exists(log_path):
            # create an empty file
            with open(log_path, "w") as f:  # noqa: F841
                pass
        self.addHandler(logging.FileHandler(log_path))
        # set the format of the log
        self.handlers[0].setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.handlers[1].setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

    def info(self, msg):
        super().info("flashinfer.jit: " + msg)


logger = FlashInferJITLogger("flashinfer.jit")


def check_cuda_arch():
    # cuda arch check for fp8 at the moment.
    for cuda_arch_flags in torch_cpp_ext._get_cuda_arch_flags():
        arch = int(re.search(r"compute_(\d+)", cuda_arch_flags).group(1))
        if arch < 75:
            raise RuntimeError("FlashInfer requires sm75+")


def clear_cache_dir():
    if os.path.exists(jit_env.FLASHINFER_JIT_DIR):
        import shutil

        shutil.rmtree(jit_env.FLASHINFER_JIT_DIR)


sm90a_nvcc_flags = ["-gencode=arch=compute_90a,code=sm_90a"]
sm100a_nvcc_flags = ["-gencode=arch=compute_100a,code=sm_100a"]


@dataclasses.dataclass
class JitSpec:
    name: str
    sources: List[Path]
    extra_cflags: Optional[List[str]]
    extra_cuda_cflags: Optional[List[str]]
    extra_ldflags: Optional[List[str]]
    extra_include_dirs: Optional[List[Path]]
    is_class: bool = False

    @property
    def ninja_path(self) -> Path:
        return jit_env.FLASHINFER_JIT_DIR / self.name / "build.ninja"

    @property
    def jit_library_path(self) -> Path:
        return jit_env.FLASHINFER_JIT_DIR / self.name / f"{self.name}.so"

    def get_library_path(self) -> Path:
        if self.aot_path.exists():
            return self.aot_path
        return self.jit_library_path

    @property
    def aot_path(self) -> Path:
        return jit_env.FLASHINFER_AOT_DIR / self.name / f"{self.name}.so"

    def write_ninja(self) -> None:
        ninja_path = self.ninja_path
        ninja_path.parent.mkdir(parents=True, exist_ok=True)
        content = generate_ninja_build_for_op(
            name=self.name,
            sources=self.sources,
            extra_cflags=self.extra_cflags,
            extra_cuda_cflags=self.extra_cuda_cflags,
            extra_ldflags=self.extra_ldflags,
            extra_include_dirs=self.extra_include_dirs,
        )
        write_if_different(ninja_path, content)

    def build(self, verbose: bool) -> None:
        tmpdir = get_tmpdir()
        with FileLock(tmpdir / f"{self.name}.lock", thread_local=False):
            run_ninja(jit_env.FLASHINFER_JIT_DIR, self.ninja_path, verbose)

    def build_and_load(self, class_name: str = None):
        if self.aot_path.exists():
            so_path = self.aot_path
        else:
            so_path = self.jit_library_path
            verbose = os.environ.get("FLASHINFER_JIT_VERBOSE", "0") == "1"
            self.build(verbose)
        load_class = class_name is not None
        loader = torch.classes if load_class else torch.ops
        loader.load_library(so_path)
        if load_class:
            cls = torch._C._get_custom_class_python_wrapper(self.name, class_name)
            return cls
        return getattr(loader, self.name)


def gen_jit_spec(
    name: str,
    sources: List[Union[str, Path]],
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[List[Union[str, Path]]] = None,
) -> JitSpec:
    check_cuda_arch()
    verbose = os.environ.get("FLASHINFER_JIT_VERBOSE", "0") == "1"

    cflags = ["-O3", "-std=c++17", "-Wno-switch-bool"]
    cuda_cflags = [
        "-O3",
        "-std=c++17",
        "--threads=4",
        "-use_fast_math",
        "-DFLASHINFER_ENABLE_F16",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
    ]
    if verbose:
        cuda_cflags += [
            "-g",
            "-lineinfo",
            "--ptxas-options=-v",
            "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage",
            "-DCUTLASS_DEBUG_TRACE_LEVEL=2",
        ]
    else:
        # non debug mode
        cuda_cflags += ["-DNDEBUG"]

    if extra_cflags is not None:
        cflags += extra_cflags
    if extra_cuda_cflags is not None:
        cuda_cflags += extra_cuda_cflags
    if extra_include_paths is not None:
        extra_include_paths = [Path(x) for x in extra_include_paths]
    sources = [Path(x) for x in sources]

    spec = JitSpec(
        name=name,
        sources=sources,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_dirs=extra_include_paths,
    )
    spec.write_ninja()
    return spec


def get_tmpdir() -> Path:
    # TODO(lequn): Try /dev/shm first. This should help Lock on NFS.
    tmpdir = jit_env.FLASHINFER_JIT_DIR / "tmp"
    if not tmpdir.exists():
        tmpdir.mkdir(parents=True, exist_ok=True)
    return tmpdir


def build_jit_specs(
    specs: List[JitSpec],
    verbose: bool = False,
    skip_prebuilt: bool = True,
) -> None:
    lines: List[str] = []
    for spec in specs:
        if skip_prebuilt and spec.aot_path.exists():
            continue
        lines.append(f"subninja {spec.ninja_path}")
    if not lines:
        return

    lines = ["ninja_required_version = 1.3"] + lines + [""]

    tmpdir = get_tmpdir()
    with FileLock(tmpdir / "flashinfer_jit.lock", thread_local=False):
        ninja_path = tmpdir / "flashinfer_jit.ninja"
        write_if_different(ninja_path, "\n".join(lines))
        run_ninja(jit_env.FLASHINFER_JIT_DIR, ninja_path, verbose)


def load_cuda_ops(
    name: str,
    sources: List[Union[str, Path]],
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags=None,
    extra_include_paths=None,
):
    # TODO(lequn): Remove this function and use JitSpec directly.
    warnings.warn(
        "load_cuda_ops is deprecated. Use JitSpec directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    spec = gen_jit_spec(
        name=name,
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
    )
    return spec.build_and_load()
