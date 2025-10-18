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

import subprocess
from pathlib import Path
from typing import Optional


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
