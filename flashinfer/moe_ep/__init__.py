"""flashinfer.moe_ep — MoE Expert-Parallel dispatch/combine over NCCL-EP and NIXL-EP.

This package is a thin Python wrapper over two transport backends:

- ``flashinfer.moe_ep.nccl_ep``  — primary backend, wraps NVIDIA's ``nccl_ep``
  (built in-tree from ``3rdparty/nccl/contrib/nccl_ep``).
- ``flashinfer.moe_ep.nixl_ep``  — alternate backend, wraps ai-dynamo's
  ``nixl_ep`` (built in-tree from ``3rdparty/nixl/examples/device/ep``).

The shared libraries that back these wrappers (``libnccl_ep.so``,
``nixl_ep_cpp*.so``, etc.) are produced by the FlashInfer build only when
``BUILD_NVEP=1`` is set in the env at install time:

    BUILD_NVEP=1 pip install -e ".[nvep]"

Without ``BUILD_NVEP=1`` the package imports succeed but calling
:func:`create_fleet` raises :class:`MoEEpNotBuiltError` with rebuild
instructions. This file lays down only the import-time probe and the
``Fleet`` / ``Handle`` factory plumbing; the actual abstract classes and
backend implementations land in Part B of the integration plan.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "MoEEpNotBuiltError",
    "have_nccl_ep",
    "have_nixl_ep",
    "available_backends",
]


_pkg_dir = Path(__file__).parent
_REBUILD_HINT = (
    "flashinfer.moe_ep is not built. Rebuild with:\n"
    '    BUILD_NVEP=1 pip install -e ".[nvep]"\n'
    "from the FlashInfer source tree. See "
    "flashinfer/moe_ep/README.md for required system dependencies."
)


class MoEEpNotBuiltError(RuntimeError):
    """Raised when an EP backend is invoked but its native libs are missing."""


def _probe_nccl_ep() -> bool:
    """True if the NCCL-EP plugin .so was staged by the build.

    The base libnccl.so.2 is NOT staged into this package — it comes from the
    pip-installed nvidia-nccl-cu13 wheel. The runtime loader in
    flashinfer.moe_ep.nccl_ep loads it explicitly before opening libnccl_ep.so.
    """
    libs = _pkg_dir / "nccl_ep" / "_libs"
    return (libs / "libnccl_ep.so").exists()


def _probe_nixl_ep() -> bool:
    """True if the NIXL-EP plugin .so was staged by the build.

    The base libnixl.so + plugins are NOT staged into this package — they
    come from the pip-installed nixl-cu13 wheel. The runtime loader in
    flashinfer.moe_ep.nixl_ep loads them explicitly before opening
    nixl_ep_cpp.so.
    """
    libs = _pkg_dir / "nixl_ep" / "_libs"
    if not libs.is_dir():
        return False
    return any(libs.glob("nixl_ep_cpp*.so"))


def have_nccl_ep() -> bool:
    """Return True if the NCCL-EP backend native libs are present."""
    return _probe_nccl_ep()


def have_nixl_ep() -> bool:
    """Return True if the NIXL-EP backend native libs are present."""
    return _probe_nixl_ep()


def available_backends() -> list[str]:
    """Names of EP backends with both native libs and python wrappers present."""
    out: list[str] = []
    if have_nccl_ep():
        out.append("nccl_ep")
    if have_nixl_ep():
        out.append("nixl_ep")
    return out


def _require_built(backend: str) -> None:
    """Raise MoEEpNotBuiltError if `backend` is missing its native libs."""
    probe = {"nccl_ep": _probe_nccl_ep, "nixl_ep": _probe_nixl_ep}.get(backend)
    if probe is None:
        raise ValueError(
            f"unknown moe_ep backend {backend!r}; expected one of nccl_ep, nixl_ep"
        )
    if not probe():
        raise MoEEpNotBuiltError(
            f"moe_ep backend {backend!r} is not built.\n\n{_REBUILD_HINT}"
        )


# Quiet diagnostic at import time when a build flag was set but the libs
# are absent — most likely cause is a partial build (probe failure
# swallowed in BUILD_NVEP=1 best-effort mode). Helpful for first-time
# users. Covers all three opt-in flags: the legacy BUILD_NVEP alias plus
# the per-backend BUILD_NCCL_EP / BUILD_NIXL_EP.
_set_build_flags = [
    name
    for name in ("BUILD_NVEP", "BUILD_NCCL_EP", "BUILD_NIXL_EP")
    if os.environ.get(name, "").lower() in ("1", "true", "yes", "on")
]
if _set_build_flags and not available_backends():
    import warnings

    warnings.warn(
        f"{'/'.join(_set_build_flags)} was set, but no moe_ep backend "
        f"libraries were found under {_pkg_dir}. Check the build log "
        "for pre-flight probe misses (meson/make/nvcc/git on PATH, "
        "ucx/libibverbs via pkg-config, nixl-cu13 / nvidia-nccl-cu13 "
        "wheels importable) or meson/make compile failures.",
        RuntimeWarning,
        stacklevel=2,
    )
