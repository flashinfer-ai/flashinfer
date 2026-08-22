#!/usr/bin/env python3
"""Smoke-test a candidate FlashInfer CI image."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Sequence
from typing import NoReturn


DOCKER_ARCH_TO_MACHINE = {
    "amd64": "x86_64",
    "arm64": "aarch64",
}


def _fail(message: str) -> NoReturn:
    raise SystemExit(f"ERROR: {message}")


def _require_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        _fail(f"required executable not found: {name}")
    return path


def _nvcc_cuda_version(nvcc: str) -> str:
    try:
        result = subprocess.run(
            [nvcc, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        _fail(f"could not run nvcc --version: {error}")

    match = re.search(r"\brelease\s+(\d+\.\d+)", result.stdout)
    if match is None:
        _fail("could not determine the CUDA version from nvcc --version")
    return match.group(1)


def _expected_cudnn_backend(version: str) -> int:
    try:
        major, minor, patch = (int(part) for part in version.split(".")[:3])
    except (TypeError, ValueError):
        _fail(f"invalid cuDNN package version: {version}")
    return major * 10000 + minor * 100 + patch


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected_cuda_version", help="expected CUDA major.minor")
    parser.add_argument(
        "expected_docker_arch",
        choices=sorted(DOCKER_ARCH_TO_MACHINE),
        help="expected Docker architecture",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    expected_cuda = args.expected_cuda_version
    expected_machine = DOCKER_ARCH_TO_MACHINE[args.expected_docker_arch]

    actual_machine = platform.machine()
    if actual_machine != expected_machine:
        _fail(f"image architecture is {actual_machine}; expected {expected_machine}")

    nvcc_cuda = _nvcc_cuda_version(_require_executable("nvcc"))
    if nvcc_cuda != expected_cuda:
        _fail(f"nvcc reports CUDA {nvcc_cuda}; expected {expected_cuda}")

    _require_executable("ptxas")
    _require_executable("mpirun")

    expected_cudnn = os.environ.get("FLASHINFER_CUDNN_VERSION")
    if not expected_cudnn:
        _fail("FLASHINFER_CUDNN_VERSION is not set")

    cuda_major = expected_cuda.split(".")[0]
    cudnn_package = f"nvidia-cudnn-cu{cuda_major}"

    cudnn = importlib.import_module("cudnn")
    importlib.import_module("cutlass")
    torch = importlib.import_module("torch")
    importlib.import_module("tvm_ffi")
    importlib.import_module("cuda.bindings.runtime")

    if torch.version.cuda != expected_cuda:
        _fail(f"PyTorch targets CUDA {torch.version.cuda}; expected {expected_cuda}")

    actual_cuda_python = importlib.metadata.version("cuda-python")
    if actual_cuda_python.split(".")[:2] != expected_cuda.split(".")[:2]:
        _fail(
            f"cuda-python targets CUDA {actual_cuda_python}; expected {expected_cuda}"
        )

    actual_cudnn = importlib.metadata.version(cudnn_package)
    if actual_cudnn != expected_cudnn:
        _fail(f"{cudnn_package} is {actual_cudnn}; expected {expected_cudnn}")

    expected_backend = _expected_cudnn_backend(expected_cudnn)
    actual_backend = cudnn.backend_version()
    if actual_backend != expected_backend:
        _fail(f"cuDNN backend is {actual_backend}; expected {expected_backend}")

    distributions = (
        "apache-tvm-ffi",
        "cuda-python",
        cudnn_package,
        "nvidia-cudnn-frontend",
        "nvidia-cutlass-dsl",
        "torch",
    )
    for distribution in distributions:
        print(f"{distribution}=={importlib.metadata.version(distribution)}")
    print(f"architecture={actual_machine}")
    print(f"cuda={torch.version.cuda}")
    print(f"Candidate CI image passed: CUDA {expected_cuda}, {expected_machine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
