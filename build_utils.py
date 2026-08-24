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


def get_build_dependency_requirements(
    cuda_major: Optional[str] = None,
) -> list[str]:
    """Return exact build dependencies selected by the shared CI policy."""
    if cuda_major is None:
        cuda_major = os.environ.get("CUDA_MAJOR")

    with CI_CONFIG_FILE.open() as config_file:
        config = json.load(config_file)

    requirements = []
    for package, dependency in config["build_dependencies"].items():
        extras = dependency.get("cuda_major_extras", {}).get(cuda_major, [])
        package_spec = package
        if extras:
            package_spec += f"[{','.join(extras)}]"
        requirements.append(f"{package_spec}=={dependency['version']}")
    return requirements


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
