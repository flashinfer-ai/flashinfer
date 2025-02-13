import logging
import os
import re
from pathlib import Path
from typing import List, Union, Optional
from contextlib import suppress
import torch.utils.cpp_extension as torch_cpp_ext
from filelock import FileLock

from .env import CUTLASS_INCLUDE_DIRS as CUTLASS_INCLUDE_DIRS
from .env import FLASHINFER_CSRC_DIR as FLASHINFER_CSRC_DIR
from .env import FLASHINFER_GEN_SRC_DIR as FLASHINFER_GEN_SRC_DIR
from .env import FLASHINFER_INCLUDE_DIR as FLASHINFER_INCLUDE_DIR
from .env import FLASHINFER_JIT_DIR as FLASHINFER_JIT_DIR
from .env import FLASHINFER_WORKSPACE_DIR as FLASHINFER_WORKSPACE_DIR

os.makedirs(FLASHINFER_WORKSPACE_DIR, exist_ok=True)
os.makedirs(FLASHINFER_CSRC_DIR, exist_ok=True)


class FlashInferJITLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        self.setLevel(logging.INFO)
        self.addHandler(logging.StreamHandler())
        log_path = FLASHINFER_WORKSPACE_DIR / "flashinfer_jit.log"
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
            suppress(ValueError)


remove_unwanted_pytorch_nvcc_flags()

sm90a_nvcc_flags = ["-gencode", "arch=compute_90a,code=sm_90a"]


def load_cuda_ops(
    name: str,
    sources: List[Union[str, Path]],
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags=None,
    extra_include_paths=None,
    verbose=False,
):
    if extra_cflags is None:
        extra_cflags = []
    if extra_cuda_cflags is None:
        extra_cuda_cflags = []

    cflags = ["-O3", "-Wno-switch-bool"]
    cuda_cflags = [
        "-O3",
        "-std=c++17",
        "--threads",
        "4",
        "-use_fast_math",
        "-DFLASHINFER_ENABLE_F16",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
    ]
    cflags += extra_cflags
    cuda_cflags += extra_cuda_cflags
    logger.info(f"Loading JIT ops: {name}")
    check_cuda_arch()
    build_directory = FLASHINFER_JIT_DIR / name
    os.makedirs(build_directory, exist_ok=True)
    if extra_include_paths is None:
        extra_include_paths = []
    extra_include_paths += [
        FLASHINFER_INCLUDE_DIR,
        FLASHINFER_CSRC_DIR,
    ] + CUTLASS_INCLUDE_DIRS
    lock = FileLock(FLASHINFER_JIT_DIR / f"{name}.lock", thread_local=False)
    with lock:
        module = torch_cpp_ext.load(
            name,
            list(map(lambda _: str(_), sources)),
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_cflags,
            extra_ldflags=extra_ldflags,
            extra_include_paths=list(map(lambda _: str(_), extra_include_paths)),
            build_directory=build_directory,
            verbose=verbose,
            with_cuda=True,
        )
    logger.info(f"Finished loading JIT ops: {name}")
    return module
