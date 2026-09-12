#!/usr/bin/env python3
"""Smoke-test a candidate FlashInfer CI image."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn


DOCKER_ARCH_TO_MACHINE = {
    "amd64": "x86_64",
    "arm64": "aarch64",
}

DEPENDENCY_POLICY_PATH = Path("/install/ci/cuda-versions.json")


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


def _validate_cuda_tile_compiler() -> tuple[str, str, str]:
    try:
        importlib.import_module("cuda.tile.tune")
    except ImportError as error:
        _fail(f"could not import cuda.tile.tune: {error}")

    try:
        cuda_tile_version = importlib.metadata.version("cuda-tile")
        tileiras_version = importlib.metadata.version("nvidia-cuda-tileiras")
    except importlib.metadata.PackageNotFoundError as error:
        _fail(f"required cuda-tile package metadata not found: {error}")

    try:
        compile_module = importlib.import_module("cuda.tile._compile")
        compiler_path = os.fspath(compile_module._find_compiler_bin().path)
    except Exception as error:
        _fail(f"could not discover cuda-tile compiler: {error}")

    try:
        subprocess.run(
            [compiler_path, "--help"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        _fail("cuda-tile compiler --help timed out after 30 seconds")
    except subprocess.CalledProcessError as error:
        _fail(f"cuda-tile compiler --help exited with status {error.returncode}")
    except OSError as error:
        _fail(f"could not run cuda-tile compiler --help: {error}")

    return cuda_tile_version, tileiras_version, compiler_path


def _validate_cuda_tile_for_runtime(
    expected_cuda_version: str,
) -> tuple[str, str, str] | None:
    if expected_cuda_version.split(".", 1)[0] != "13":
        return None
    return _validate_cuda_tile_compiler()


def _validate_cuda_runtime_distributions(
    expected_cuda_version: str,
) -> list[tuple[str, str]]:
    expected_major = expected_cuda_version.split(".", 1)[0]
    runtime_distributions = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not isinstance(name, str):
            continue
        normalized_name = re.sub(r"[-_.]+", "-", name).lower()
        if normalized_name.startswith("nvidia-cuda-runtime"):
            runtime_distributions.append((name, distribution.version))

    runtime_distributions.sort(
        key=lambda item: (re.sub(r"[-_.]+", "-", item[0]).lower(), item[1])
    )
    if not runtime_distributions:
        _fail("no installed nvidia-cuda-runtime distribution found")

    for name, version in runtime_distributions:
        actual_major = version.split(".", 1)[0]
        if actual_major != expected_major:
            _fail(
                f"{name}=={version} targets CUDA {actual_major}; "
                f"expected CUDA {expected_major}"
            )
    return runtime_distributions


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

    cuda_runtime_distributions = _validate_cuda_runtime_distributions(expected_cuda)
    cuda_tile_compiler = _validate_cuda_tile_for_runtime(expected_cuda)

    actual_cudnn = importlib.metadata.version(cudnn_package)
    if actual_cudnn != expected_cudnn:
        _fail(f"{cudnn_package} is {actual_cudnn}; expected {expected_cudnn}")

    try:
        dependency_policy = json.loads(DEPENDENCY_POLICY_PATH.read_text())[
            "dependency_policy"
        ]
    except (KeyError, OSError, json.JSONDecodeError) as error:
        _fail(f"could not read CI dependency policy: {error}")
    for distribution, policy in dependency_policy.items():
        specifier = policy["ci_image_specifier"]
        if not isinstance(specifier, str) or not specifier.startswith("=="):
            _fail(f"{distribution} CI image specifier is not exact: {specifier!r}")
        expected_version = specifier.removeprefix("==")
        actual_version = importlib.metadata.version(distribution)
        if actual_version != expected_version:
            _fail(f"{distribution} is {actual_version}; expected {expected_version}")

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
    if cuda_tile_compiler is not None:
        cuda_tile_version, tileiras_version, compiler_path = cuda_tile_compiler
        print(f"cuda-tile=={cuda_tile_version}")
        print(f"nvidia-cuda-tileiras=={tileiras_version}")
        print(f"cuda-tile-compiler={compiler_path}")
    for distribution, version in cuda_runtime_distributions:
        print(f"cuda-runtime-distribution={distribution}=={version}")
    print(f"architecture={actual_machine}")
    print(f"cuda={torch.version.cuda}")
    print(f"Candidate CI image passed: CUDA {expected_cuda}, {expected_machine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
