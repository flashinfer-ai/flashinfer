"""
Copyright (c) 2025 by FlashInfer team.

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

"""Shared build utilities for flashinfer packages."""

import json
import os
import subprocess
from pathlib import Path
from typing import Optional


CI_CONFIG_FILE = Path(__file__).parent / "ci" / "cuda-versions.json"

_DEPENDENCY_SCOPE_FIELDS = {
    "provider_build": "provider_build_specifier",
    "cuda_extra": "cuda_extra_specifier",
    "ci_image": "ci_image_specifier",
}

CUDA_TILE_COMPILE_DEPENDENCY_REQUIREMENTS = (
    "nvidia-cuda-nvcc<13.4,>=13.2",
    "nvidia-cuda-tileiras<13.4,>=13.2",
    "nvidia-nvvm<13.4,>=13.2",
    "nvidia-nvjitlink<14,>=13.3",
    "nvidia-cuda-crt<13.4,>=13.2",
)


def get_dependency_requirements(
    scope: str,
    cuda_major: Optional[str] = None,
) -> list[str]:
    """Return dependency requirements for a configured installation scope."""
    try:
        specifier_field = _DEPENDENCY_SCOPE_FIELDS[scope]
    except KeyError as error:
        raise ValueError(f"unknown dependency scope: {scope}") from error

    if cuda_major is None:
        cuda_major = os.environ.get("CUDA_MAJOR")

    with CI_CONFIG_FILE.open() as config_file:
        config = json.load(config_file)

    requirements = []
    for package, dependency in config["dependency_policy"].items():
        extras = dependency.get("cuda_major_extras", {}).get(cuda_major, [])
        package_spec = package
        if extras:
            package_spec += f"[{','.join(extras)}]"
        requirements.append(f"{package_spec}{dependency[specifier_field]}")
    return requirements


def get_build_dependency_requirements(
    cuda_major: Optional[str] = None,
) -> list[str]:
    """Return minimum dependencies needed by the provider-wheel backends."""
    return get_dependency_requirements("provider_build", cuda_major)


def get_cuda_extra_dependency_requirements(
    cuda_major: Optional[str] = None,
) -> list[str]:
    """Return dependencies for the project's CUDA optional extras."""
    return get_dependency_requirements("cuda_extra", cuda_major)


def get_ci_image_dependency_requirements(
    cuda_major: Optional[str] = None,
) -> list[str]:
    """Return exact dependency selections for reproducible CI images."""
    return get_dependency_requirements("ci_image", cuda_major)


def get_cuda_tile_compile_dependency_requirements() -> list[str]:
    """Return the compiler-only dependency chain needed by cuda-tile."""
    return list(CUDA_TILE_COMPILE_DEPENDENCY_REQUIREMENTS)


def apply_cutlass_patches(cutlass_dir: Path) -> None:
    """Patch populated CUTLASS sources before packaging or native compilation."""
    header = (
        cutlass_dir
        / "include"
        / "cutlass"
        / "gemm"
        / "kernel"
        / "sm103_blockscaled_gemm_tma_warpspecialized.hpp"
    )
    original = header.read_bytes()
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    before = b"    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);"
    # NoSmem has no epilogue-load producers; only real load pipelines need init.
    after = (
        b"    EpiLoadPipeline epi_load_pipeline(\n"
        b"        shared_storage.pipelines.epi_load,\n"
        b"        epi_load_pipeline_params,\n"
        b"        cute::bool_constant<!IsNoSmemEpilogue>{});"
    ).replace(b"\n", newline)
    if before in original:
        header.write_bytes(original.replace(before, after))
        print(f"Applied CUTLASS SM103 NoSmem pipeline fix: {header}")
    elif after not in original:
        raise RuntimeError(
            f"Expected original or patched SM103 epilogue-load constructor in {header}"
        )


def get_git_version(cwd: Optional[Path] = None) -> str:
    """
    Get git commit hash.

    Args:
        cwd: Working directory for git command. If None, uses current directory.

    Returns:
        Git commit hash or "unknown" if git is not available.
    """
    try:
        git_version = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
        return git_version
    except Exception:
        return "unknown"


if __name__ == "__main__":
    apply_cutlass_patches(Path(__file__).resolve().parent / "3rdparty" / "cutlass")
