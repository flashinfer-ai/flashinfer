#!/usr/bin/env python3
"""Smoke-test a candidate FlashInfer AOT builder image.

Run inside the image with its Python on stdin:
    python - <cuda-version> <pytorch-index> <cudnn-version> < test_aot_builder_image.py
"""

from __future__ import annotations

import importlib.metadata as m
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

SCCACHE_VERSION = "0.18.0"
MANIFEST = Path("/opt/flashinfer-aot-builder/manifest.json")
BUILD_REQUIREMENTS = (
    "build",
    "setuptools",
    "wheel",
    "packaging",
    "ninja",
    "numpy",
    "tqdm",
    "requests",
    "nvidia-ml-py",
    "apache-tvm-ffi",
    "torch",
)


def _fail(message: str) -> NoReturn:
    raise SystemExit(f"ERROR: {message}")


def _version(name: str) -> str:
    try:
        return m.version(name)
    except m.PackageNotFoundError:
        _fail(f"required distribution not installed: {name}")


def main() -> None:
    if len(sys.argv) != 4:
        _fail(
            "usage: test_aot_builder_image.py CUDA_VERSION PYTORCH_INDEX CUDNN_VERSION"
        )
    cuda_version, pytorch_index, cudnn_version = sys.argv[1:]

    for name in BUILD_REQUIREMENTS:
        _version(name)
    pip_version = tuple(int(p) for p in re.findall(r"\d+", _version("pip"))[:3])
    if pip_version < (26, 2, 1):
        _fail(f"pip {_version('pip')} is older than 26.2.1")

    import torch

    if torch.version.cuda != cuda_version:
        _fail(f"torch targets CUDA {torch.version.cuda}, expected {cuda_version}")

    cudnn_package = (
        "nvidia-cudnn-cu13" if cuda_version.startswith("13.") else "nvidia-cudnn-cu12"
    )
    if _version(cudnn_package) != cudnn_version:
        _fail(f"{cudnn_package} is {_version(cudnn_package)}, expected {cudnn_version}")

    # The provider's build-system requires must resolve offline: pip would
    # otherwise reach for the network at job time.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--no-index",
            "setuptools>=77",
            "packaging>=24",
            "wheel",
            "tqdm",
            "torch",
            "ninja",
            "requests",
            "numpy",
            "nvidia-ml-py",
            "apache-tvm-ffi>=0.1,<0.2",
        ],
        check=True,
    )

    sccache = shutil.which("sccache")
    if sccache is None:
        _fail("sccache not found")
    version = subprocess.run(
        [sccache, "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if version != f"sccache {SCCACHE_VERSION}":
        _fail(f"unexpected sccache version: {version}")

    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    out = subprocess.run(
        [nvcc, "--version"], check=True, capture_output=True, text=True
    ).stdout
    match = re.search(r"release (\d+\.\d+)", out)
    if match is None or match.group(1) != cuda_version:
        _fail(f"nvcc reports CUDA {match and match.group(1)}, expected {cuda_version}")

    manifest = json.loads(MANIFEST.read_text())
    if manifest["pytorch_index"] != pytorch_index:
        _fail(f"manifest PyTorch index {manifest['pytorch_index']} != {pytorch_index}")
    if os.environ.get("FLASHINFER_AOT_BUILDER_IMAGE") != "1":
        _fail("FLASHINFER_AOT_BUILDER_IMAGE is not set")
    if os.environ.get("FLASHINFER_AOT_BUILDER_PYTORCH_INDEX") != pytorch_index:
        _fail("FLASHINFER_AOT_BUILDER_PYTORCH_INDEX does not match")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("AOT builder image checks passed")


if __name__ == "__main__":
    main()
