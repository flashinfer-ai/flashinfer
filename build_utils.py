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


SM_FAMILY_ORDER = ("sm9x", "sm10x", "sm12x")


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


def sm_family_for_capability(major: int, minor: int) -> str:
    if major < 10:
        return "sm9x"
    if major < 12:
        return "sm10x"
    return "sm12x"


def parse_cuda_arch_entry(entry: str) -> tuple[int, int]:
    """Parse arch entries like '9.0a', '90', or '120f' into (major, minor)."""
    entry = entry.strip()
    if "." in entry:
        major_str, minor_str = entry.split(".", 1)
        major_digits = "".join(c for c in major_str if c.isdigit())
        minor_digits = "".join(c for c in minor_str if c.isdigit())
        if not major_digits:
            raise ValueError(f"Invalid CUDA arch entry: {entry!r}")
        return int(major_digits), int(minor_digits) if minor_digits else 0

    digits = "".join(c for c in entry if c.isdigit())
    if not digits:
        raise ValueError(f"Invalid CUDA arch entry: {entry!r}")
    if len(digits) >= 2:
        return int(digits[:-1]), int(digits[-1])
    return int(digits), 0


def filter_arch_list_for_sm_family(arch_list: str, family: str) -> str:
    """Filter a space-separated FLASHINFER_CUDA_ARCH_LIST for one SM family."""
    if family not in SM_FAMILY_ORDER:
        raise ValueError(
            f"Invalid SM family {family!r}; expected one of {SM_FAMILY_ORDER}"
        )
    kept = []
    for entry in arch_list.split():
        major, minor = parse_cuda_arch_entry(entry)
        if sm_family_for_capability(major, minor) == family:
            kept.append(entry)
    return " ".join(kept)
