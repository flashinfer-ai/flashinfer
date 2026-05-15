"""NIXL-EP backend.

Two pieces matter for import-time success:

1. The base NIXL runtime libraries (``libnixl.so``, ``libnixl_capi.so``,
   ``libnixl_common.so``, ``libserdes.so``, etc.) — *not* shipped inside this
   package. They're expected to come from the ``nixl-cu13`` pip wheel,
   installed automatically when the user runs ``BUILD_NVEP=1 pip install ...``
   (see ``build_backend._install_nvep_runtime_wheels``).

2. The EP torch extension, ``nixl_ep_cpp*.so`` — built in-tree from
   ``3rdparty/nixl/examples/device/ep`` and staged into ``_libs/`` here.

The two ``_preload_*`` helpers below are intentionally module-level but
**not invoked at import time** — calling them eagerly would force libnixl
to load whenever Python touches this package. The Fleet/Handle wrapper
code (Part B, not yet landed) will call ``_load_nixl_ep_cpp()`` lazily on
first use.

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


# Order matters: libnixl_common must be available to libnixl, libnixl_capi
# depends on libnixl, etc. We load with RTLD_GLOBAL so each preceding lib
# exports symbols visible to subsequent dlopens.
_NIXL_BASE_LIBS = (
    "libnixl_common.so",
    "libserdes.so",
    "libnixl_build.so",
    "libnixl.so",
    "libnixl_capi.so",
)


def _find_nixl_lib_dir() -> Path | None:
    """Locate the NIXL base-lib directory via the pip-installed nixl-cu* wheel.

    Layout of the meson-python-packaged wheel:
        <site-packages>/nixl_cu13/                  — importable python module
        <site-packages>/.nixl_cu13.mesonpy.libs/    — libnixl.so + sibling libs

    We probe known package names (`nixl_cu13`, `nixl_cu12`, legacy `nixl`),
    look for the meson-python `.{name}.mesonpy.libs/` sidecar, and fall back
    to a glob over site-packages.
    """
    for pkg_name in ("nixl_cu13", "nixl_cu12", "nixl"):
        try:
            mod = __import__(pkg_name)
        except ImportError:
            continue
        try:
            pkg_root = Path(mod.__path__[0])
        except Exception:
            continue
        site_packages = pkg_root.parent
        for candidate in (
            site_packages / f".{pkg_name}.mesonpy.libs",
            pkg_root / "lib" / "x86_64-linux-gnu",
            pkg_root / "lib",
            pkg_root,
        ):
            if candidate.is_dir() and (candidate / "libnixl.so").exists():
                return candidate
    # Last resort: glob for any `.nixl_*.mesonpy.libs/` under site-packages.
    import site as _site

    for sp_str in _site.getsitepackages() + [_site.getusersitepackages()]:
        sp = Path(sp_str)
        if not sp.is_dir():
            continue
        for candidate in sp.glob(".nixl*.mesonpy.libs"):
            if (candidate / "libnixl.so").exists():
                return candidate
    return None


def _preload_libnixl() -> None:
    """ctypes-load the NIXL base libs with RTLD_GLOBAL before opening nixl_ep_cpp.so."""
    nixl_lib_dir = _find_nixl_lib_dir()
    if nixl_lib_dir is None:
        # Try the dynamic linker's default search for the minimum lib.
        try:
            ctypes.CDLL("libnixl.so", mode=ctypes.RTLD_GLOBAL)
            return
        except OSError as e:
            raise MoEEpNotBuiltError(
                "Could not locate the NIXL runtime libraries. Install with "
                "one of:\n"
                "    uv pip install --no-deps 'nixl-cu13>=1.0.1'\n"
                "    pip install --no-deps 'nixl-cu13>=1.0.1'\n"
                "or set LD_LIBRARY_PATH to a directory containing libnixl.so."
            ) from e

    for libname in _NIXL_BASE_LIBS:
        libpath = nixl_lib_dir / libname
        if libpath.exists():
            ctypes.CDLL(str(libpath), mode=ctypes.RTLD_GLOBAL)


def _load_nixl_ep_cpp() -> ctypes.CDLL:
    """Load the EP torch extension, preloading its NIXL base-lib deps first.

    Returns the opened CDLL handle. Caller is responsible for keeping a
    reference.
    """
    so_files = list(_libs_dir.glob("nixl_ep_cpp*.so"))
    if not so_files:
        raise MoEEpNotBuiltError(
            f"nixl_ep_cpp*.so is not staged under {_libs_dir}. Rebuild with:\n"
            '    BUILD_NVEP=1 pip install -e ".[nvep]"\n'
            "or BUILD_NIXL_EP=1 for a NIXL-EP-only build."
        )
    _preload_libnixl()
    return ctypes.CDLL(str(so_files[0]), mode=ctypes.RTLD_GLOBAL)
