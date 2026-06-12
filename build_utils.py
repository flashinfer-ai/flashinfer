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


SM_FAMILY_ORDER = ("sm9x", "sm10x", "sm110", "sm12x")
SM_FAMILY_BASE_ARCHS = {
    "sm10x": ("8.0",),
    "sm110": ("8.0",),
    "sm12x": ("8.0",),
}


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
    if major == 10:
        return "sm10x"
    if major == 11:
        return "sm110"
    return "sm12x"


def jit_cache_sm_family_for_capabilities(capabilities: list[tuple[int, int]]) -> str:
    """Return the jit-cache wheel family covering the visible GPU capabilities."""
    if not capabilities:
        raise ValueError("No CUDA device capabilities were provided.")

    native_families = {
        sm_family_for_capability(major, minor) for major, minor in capabilities
    }
    if len(native_families) == 1:
        return next(iter(native_families))

    for family, base_archs in SM_FAMILY_BASE_ARCHS.items():
        base_caps = {parse_cuda_arch_entry(entry) for entry in base_archs}
        has_native_arch = any(
            sm_family_for_capability(major, minor) == family
            for major, minor in capabilities
        )
        if has_native_arch and all(
            sm_family_for_capability(major, minor) == family
            or (major, minor) in base_caps
            for major, minor in capabilities
        ):
            return family

    details = ", ".join(
        f"sm{major}{minor} ({sm_family_for_capability(major, minor)})"
        for major, minor in capabilities
    )
    raise ValueError(
        "Visible CUDA devices are not covered by a single "
        f"flashinfer-jit-cache SM-family wheel: {details}."
    )


def jit_cache_sm_family_covers_capabilities(
    family: str, capabilities: list[tuple[int, int]]
) -> bool:
    """Return whether a jit-cache wheel family contains all requested capabilities."""
    if family not in SM_FAMILY_ORDER:
        raise ValueError(
            f"Invalid SM family {family!r}; expected one of {SM_FAMILY_ORDER}"
        )

    base_caps = {
        parse_cuda_arch_entry(entry) for entry in SM_FAMILY_BASE_ARCHS.get(family, ())
    }
    return all(
        sm_family_for_capability(major, minor) == family or (major, minor) in base_caps
        for major, minor in capabilities
    )


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
    """Filter FLASHINFER_CUDA_ARCH_LIST for one jit-cache wheel family."""
    if family not in SM_FAMILY_ORDER:
        raise ValueError(
            f"Invalid SM family {family!r}; expected one of {SM_FAMILY_ORDER}"
        )

    parsed_entries = []
    for entry in arch_list.split():
        major, minor = parse_cuda_arch_entry(entry)
        parsed_entries.append((entry, major, minor))

    native_entries = []
    for entry, major, minor in parsed_entries:
        if sm_family_for_capability(major, minor) == family:
            native_entries.append(entry)

    if not native_entries:
        return ""

    kept = []
    existing_by_capability = {
        (major, minor): entry for entry, major, minor in parsed_entries
    }
    for base_arch in SM_FAMILY_BASE_ARCHS.get(family, ()):
        base_capability = parse_cuda_arch_entry(base_arch)
        kept.append(existing_by_capability.get(base_capability, base_arch))
    kept.extend(native_entries)

    seen = set()
    kept_unique = []
    for entry in kept:
        if entry not in seen:
            seen.add(entry)
            kept_unique.append(entry)
    return " ".join(kept_unique)
