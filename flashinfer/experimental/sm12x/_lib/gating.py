# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Capability gating for flashinfer.experimental.sm12x.

Deliberately vendored (not imported from flashinfer core): the experimental
tree has a zero-outbound-imports rule so that core refactors can never break
it, enforced by tests/experimental/lint/test_isolation.py.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import re

MIN_CUTLASS_DSL = "4.6.0"


def get_compute_capability(device=None) -> tuple[int, int] | None:
    import torch

    if not torch.cuda.is_available():
        return None
    dev = torch.device(device) if device is not None else torch.device("cuda")
    if dev.type != "cuda":
        return None
    return torch.cuda.get_device_capability(dev)


def is_sm12x(device=None) -> bool:
    """True when the target device is consumer Blackwell (SM120/SM121)."""
    cap = get_compute_capability(device)
    return cap is not None and cap[0] == 12 and cap[1] in (0, 1)


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


def cutlass_dsl_version() -> str | None:
    for dist in ("nvidia-cutlass-dsl", "cutlass"):
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def has_cutlass_dsl(minimum: str = MIN_CUTLASS_DSL) -> bool:
    version = cutlass_dsl_version()
    return version is not None and _version_tuple(version) >= _version_tuple(minimum)


def require_cutlass_dsl(minimum: str = MIN_CUTLASS_DSL) -> None:
    version = cutlass_dsl_version()
    if version is None:
        raise ImportError(
            "flashinfer.experimental.sm12x requires nvidia-cutlass-dsl "
            f">= {minimum} (not installed). Install with: "
            f"pip install 'nvidia-cutlass-dsl>={minimum}'"
        )
    if _version_tuple(version) < _version_tuple(minimum):
        raise ImportError(
            f"flashinfer.experimental.sm12x requires nvidia-cutlass-dsl >= {minimum}, "
            f"found {version}. Upgrade with: pip install 'nvidia-cutlass-dsl>={minimum}'"
        )


def has_triton() -> bool:
    return importlib.util.find_spec("triton") is not None


def require_triton(op: str) -> None:
    if not has_triton():
        raise ImportError(
            f"flashinfer.experimental.sm12x.{op} requires triton (bundled with "
            "torch on Linux x86-64; install it explicitly on other platforms)."
        )


def _requirement_met(requirement: str) -> bool:
    if requirement == "triton":
        return has_triton()
    if requirement == "multi_gpu":
        import torch

        return torch.cuda.is_available() and torch.cuda.device_count() >= 2
    raise ValueError(f"unknown op requirement {requirement!r}")


def default_is_supported(device=None, *, requires: tuple[str, ...] = ()) -> bool:
    """The standard ``is_supported`` used by ops without extra constraints."""
    if not is_sm12x(device):
        return False
    if not has_cutlass_dsl():
        return False
    return all(_requirement_met(req) for req in requires)


__all__ = [
    "MIN_CUTLASS_DSL",
    "get_compute_capability",
    "is_sm12x",
    "cutlass_dsl_version",
    "has_cutlass_dsl",
    "require_cutlass_dsl",
    "has_triton",
    "require_triton",
    "default_is_supported",
]
