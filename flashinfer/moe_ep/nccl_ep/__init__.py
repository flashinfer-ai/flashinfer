"""NCCL-EP backend.

Two pieces matter for import-time success:

1. The base NCCL runtime library, ``libnccl.so.2`` — *not* shipped inside this
   package. It's expected to come from the ``nvidia-nccl-cu13`` pip wheel,
   installed automatically when the user runs ``BUILD_NVEP=1 pip install ...``
   (see ``build_backend._install_nvep_runtime_wheels``).

2. The EP plugin, ``libnccl_ep.so`` — built in-tree from
   ``3rdparty/nccl/contrib/nccl_ep`` and staged into ``_libs/`` here.

The two ``_preload_*`` helpers below are intentionally module-level but
**not invoked at import time** — calling them eagerly would force libnccl
to load whenever Python touches this package (e.g. on ``from flashinfer
import *``), which we don't want. The Fleet/Handle wrapper code (Part B,
not yet landed) will call ``_load_libnccl_ep()`` lazily on first use.

Until that lands, importing this module always succeeds. Attempting to
actually exercise the backend will raise ``MoEEpNotBuiltError`` (see
``flashinfer.moe_ep``) if the lib staging is incomplete.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

from .. import MoEEpNotBuiltError

_pkg_dir = Path(__file__).resolve().parent
_libs_dir = _pkg_dir / "_libs"


def _find_libnccl() -> Path | None:
    """Locate libnccl.so.2 via the pip-installed nvidia-nccl-cu13 wheel.

    Returns the resolved path or None if it can't be found via the wheel.
    Falls back to letting the dynamic linker's default search find it via
    LD_LIBRARY_PATH, ldconfig, etc.
    """
    # The wheel installs the lib at <site-packages>/nvidia/nccl/lib/libnccl.so.2.
    try:
        import nvidia.nccl  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        # Newer wheels expose .lib as a subpackage; older ones place files
        # directly under nvidia/nccl/. Probe both.
        candidates = [
            Path(nvidia.nccl.__path__[0]) / "lib" / "libnccl.so.2",
            Path(nvidia.nccl.__path__[0]) / "libnccl.so.2",
        ]
    except Exception:
        return None
    for c in candidates:
        if c.exists():
            return c
    return None


def _preload_libnccl() -> None:
    """ctypes-load libnccl.so.2 with RTLD_GLOBAL before opening libnccl_ep.so.

    libnccl_ep.so links against ``libnccl.so.2`` by SONAME but doesn't have
    the wheel's site-packages location in its RPATH. Loading it explicitly
    here exports the symbols globally so the subsequent dlopen of
    libnccl_ep.so resolves them.
    """
    nccl_so = _find_libnccl()
    if nccl_so is not None:
        try:
            ctypes.CDLL(str(nccl_so), mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            raise MoEEpNotBuiltError(
                f"dlopen({nccl_so}) failed: {e}. The nvidia-nccl-cu13 wheel "
                "may be corrupted or built against an incompatible glibc/CUDA. "
                "Reinstall with: "
                "uv pip install --force-reinstall --no-deps 'nvidia-nccl-cu13>=2.30.4'"
            ) from e
        return
    # No wheel found; try the dynamic linker's default search.
    try:
        ctypes.CDLL("libnccl.so.2", mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        raise MoEEpNotBuiltError(
            "Could not locate libnccl.so.2. Install it with one of:\n"
            "    uv pip install --no-deps 'nvidia-nccl-cu13>=2.30.4'\n"
            "    pip install --no-deps 'nvidia-nccl-cu13>=2.30.4'\n"
            "or set LD_LIBRARY_PATH to a directory containing libnccl.so.2."
        ) from e


def _load_libnccl_ep() -> ctypes.CDLL:
    """Load the EP plugin .so, preloading its libnccl.so.2 dep first.

    Returns the opened CDLL handle. Caller is responsible for keeping a
    reference (the dynamic linker won't unload while the handle is alive).
    """
    so = _libs_dir / "libnccl_ep.so"
    if not so.exists():
        raise MoEEpNotBuiltError(
            f"libnccl_ep.so is not staged at {so}. Rebuild with:\n"
            '    BUILD_NVEP=1 pip install -e ".[nvep]"\n'
            "or BUILD_NCCL_EP=1 for an NCCL-EP-only build."
        )
    _preload_libnccl()
    try:
        return ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        raise MoEEpNotBuiltError(
            f"dlopen({so}) failed: {e}. Most likely the wheel's NCCL "
            "version doesn't match the one FlashInfer was built against "
            "(check NCCL_VERSION_CODE). Reinstall with "
            "BUILD_NCCL_EP_HERMETIC=1 to build libnccl from the pinned "
            "submodule instead."
        ) from e
