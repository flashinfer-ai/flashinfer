# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Force-build pinned FlashKDA in a Slurm GPU allocation and write a manifest."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from kda_h12_evidence import (
    FLASH_KDA_BUILD_MANIFEST_SCHEMA_VERSION,
    SUPPORTED_ARCHITECTURES,
    _sha256,
    load_flash_kda_build_manifest,
    validate_flash_kda_build_manifest_schema,
    verify_flash_kda_checkout,
    write_json_atomic,
)


_RECORDED_BUILD_ENVIRONMENT = (
    "CC",
    "CXX",
    "CUDA_HOME",
    "FLASH_KDA_CUDA_ARCHS",
    "MAX_JOBS",
    "NVCC_PREPEND_FLAGS",
    "NVCC_THREADS",
    "TORCH_CUDA_ARCH_LIST",
)


def _command_version(executable: str, *args: str) -> str:
    completed = subprocess.run(
        [executable, *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    value = completed.stdout.strip()
    if not value:
        raise RuntimeError(f"toolchain command {executable!r} returned no version")
    return value


def _required_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"required build tool {name!r} is not on PATH")
    return str(Path(path).resolve())


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Phase-A FlashKDA build requires resolved Slurm environment {name}"
        )
    return value


def _validate_only_payload(manifest_path: Path | None) -> dict:
    if manifest_path is None:
        return {
            "schema_version": FLASH_KDA_BUILD_MANIFEST_SCHEMA_VERSION,
            "kind": "phase_a_flash_kda_in_allocation_build",
            "validation": "schema_only_no_cuda_import",
            "required_build_command_suffix": [
                "setup.py",
                "build_ext",
                "--inplace",
                "--force",
            ],
            "effective_flash_kda_cuda_archs": "auto",
            "effective_nvcc_threads_default": "32",
            "recorded_build_environment": list(_RECORDED_BUILD_ENVIRONMENT),
            "requires_slurm_gpu_allocation": True,
            "supported_architectures": sorted(SUPPORTED_ARCHITECTURES.values()),
        }
    payload, sha256 = load_flash_kda_build_manifest(manifest_path)
    return {
        "schema_version": payload["schema_version"],
        "kind": payload["kind"],
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256,
        "validation": "schema_only_no_cuda_import",
    }


def _build_manifest(*, source_dir: Path, manifest_path: Path) -> dict:
    allocation = {
        "slurm_job_id": _required_environment("SLURM_JOB_ID"),
        "slurm_cluster_name": _required_environment("SLURM_CLUSTER_NAME"),
        "slurm_partition": _required_environment("SLURM_JOB_PARTITION"),
        "slurm_node_list": _required_environment("SLURM_JOB_NODELIST"),
    }
    source_dir = source_dir.resolve(strict=True)
    manifest_path = manifest_path.resolve()
    if manifest_path.is_relative_to(source_dir):
        raise RuntimeError("build manifest must be written outside FlashKDA checkout")
    source = verify_flash_kda_checkout(source_dir=source_dir)

    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
    except ImportError as error:
        raise RuntimeError("FlashKDA allocation build requires torch") from error
    if not torch.cuda.is_available():
        raise RuntimeError("FlashKDA allocation build requires an allocated CUDA GPU")
    capability = torch.cuda.get_device_capability()
    if capability not in SUPPORTED_ARCHITECTURES:
        raise RuntimeError(
            "FlashKDA Phase-A build requires CC 10.0 or 10.3, got "
            f"CC {capability[0]}.{capability[1]}"
        )
    if CUDA_HOME is None:
        raise RuntimeError("torch did not resolve CUDA_HOME")

    nvcc_path = _required_executable("nvcc")
    cxx_path = _required_executable(os.environ.get("CXX", "c++"))
    python_executable = str(Path(sys.executable).resolve())
    build_command = [
        python_executable,
        "setup.py",
        "build_ext",
        "--inplace",
        "--force",
    ]
    subprocess.run(build_command, cwd=source_dir, check=True)
    after_build = verify_flash_kda_checkout(source_dir=source_dir)
    if after_build != source:
        raise RuntimeError("FlashKDA source identity changed during build")

    preimported = sorted(
        name for name in sys.modules if name == "flash_kda" or name == "flash_kda_C"
    )
    if preimported:
        raise RuntimeError(
            f"FlashKDA modules were imported before artifact binding: {preimported!r}"
        )
    sys.path.insert(0, str(source_dir))
    package = importlib.import_module("flash_kda")
    extension = importlib.import_module("flash_kda_C")
    package_path = Path(package.__file__).resolve(strict=True)
    extension_path = Path(extension.__file__).resolve(strict=True)
    for label, path in (
        ("package", package_path),
        ("extension", extension_path),
    ):
        if not path.is_relative_to(source_dir):
            raise RuntimeError(
                f"built FlashKDA {label} resolved outside checkout: {path}"
            )

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    device_uuid = getattr(properties, "uuid", None)
    if device_uuid is None or not str(device_uuid):
        raise RuntimeError("FlashKDA allocation build requires a CUDA device UUID")
    cuda_home = str(Path(CUDA_HOME).resolve())
    manifest = {
        "schema_version": FLASH_KDA_BUILD_MANIFEST_SCHEMA_VERSION,
        "kind": "phase_a_flash_kda_in_allocation_build",
        "source": source,
        "build": {
            "command": build_command,
            "cwd": str(source_dir),
            "environment": {
                key: (
                    os.environ.get(key, "auto")
                    if key == "FLASH_KDA_CUDA_ARCHS"
                    else os.environ.get(key, "32")
                    if key == "NVCC_THREADS"
                    else os.environ.get(key)
                )
                for key in _RECORDED_BUILD_ENVIRONMENT
            },
        },
        "toolchain": {
            "python_executable": python_executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "torch_cuda_version": str(torch.version.cuda),
            "cuda_home": cuda_home,
            "nvcc_path": nvcc_path,
            "nvcc_version": _command_version(nvcc_path, "--version"),
            "cxx_path": cxx_path,
            "cxx_version": _command_version(cxx_path, "--version"),
        },
        "allocation": allocation,
        "hardware": {
            "cuda_available": True,
            "cuda_arch": SUPPORTED_ARCHITECTURES[capability],
            "compute_capability": list(capability),
            "device_name": properties.name,
            "device_uuid": str(device_uuid),
        },
        "artifacts": {
            "package_path": str(package_path),
            "package_sha256": _sha256(package_path),
            "extension_path": str(extension_path),
            "extension_sha256": _sha256(extension_path),
        },
    }
    validate_flash_kda_build_manifest_schema(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flash-kda-source-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(_validate_only_payload(args.manifest), indent=2))
        return
    if args.flash_kda_source_dir is None or args.manifest is None:
        parser.error("--flash-kda-source-dir and --manifest are required for a build")
    manifest = _build_manifest(
        source_dir=args.flash_kda_source_dir,
        manifest_path=args.manifest,
    )
    write_json_atomic(args.manifest, manifest)
    print(
        json.dumps(
            {
                "manifest": str(args.manifest.resolve()),
                "extension_sha256": manifest["artifacts"]["extension_sha256"],
                "cuda_arch": manifest["hardware"]["cuda_arch"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
