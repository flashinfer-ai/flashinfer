# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Prefer a sibling Attention-TS DKG build for local FlashInfer development.

Python imports ``sitecustomize`` after processing site-package ``.pth`` files.
That ordering matters here: a developer may already have an editable
``nvctm-dsl`` installation whose meta-path finder points at another DKG
checkout.  When the selected local DKG build exists, this bootstrap removes
that redirect, puts the matching build/source paths first, and selects its
CuTe DSL runtime library.

Put the FlashInfer repository root on ``PYTHONPATH`` so Python selects this
module ahead of any system ``sitecustomize`` module.

Set ``FLASHINFER_ATTENTION_TS_DKG_ROOT`` to override the default sibling
``dkg-ts-fmha-flashinfer`` checkout.  Set
``FLASHINFER_ATTENTION_TS_DISABLE_DKG_BOOTSTRAP=1`` to leave Python's normal
environment unchanged.  The bootstrap is a no-op when the selected DKG build
is absent.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


_DISABLE_ENV = "FLASHINFER_ATTENTION_TS_DISABLE_DKG_BOOTSTRAP"
_DKG_ROOT_ENV = "FLASHINFER_ATTENTION_TS_DKG_ROOT"


def _is_enabled() -> bool:
    return os.environ.get(_DISABLE_ENV, "").lower() not in {"1", "true", "yes"}


def _resolve_dkg_root() -> Path:
    configured = os.environ.get(_DKG_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(__file__).resolve().parent.parent / "dkg-ts-fmha-flashinfer").resolve()


def _prepend_paths(paths: list[Path]) -> None:
    resolved = [str(path.resolve()) for path in paths if path.is_dir()]
    if not resolved:
        return
    sys.path[:] = [path for path in sys.path if path not in resolved]
    sys.path[:0] = resolved


def _remove_nvctm_redirects() -> None:
    """Remove editable nvctm finders installed before this startup hook."""
    sys.meta_path[:] = [
        finder for finder in sys.meta_path if type(finder).__name__ != "NvctmDslFinder"
    ]


def _ensure_mlir_type_compat() -> None:
    try:
        from cutlass._mlir import ir as mlir_ir
    except Exception:
        return

    for name in dir(mlir_ir):
        if not name.endswith("Type"):
            continue
        cls = getattr(mlir_ir, name)
        if not isinstance(cls, type) or hasattr(cls, "isinstance"):
            continue
        try:
            cls.isinstance = staticmethod(
                lambda value, expected_cls=cls: isinstance(value, expected_cls)
            )
        except Exception:
            continue


def _bootstrap_local_dkg() -> None:
    if not _is_enabled():
        return

    dkg_root = _resolve_dkg_root()
    python_packages = dkg_root / "build" / "cutlass_ir" / "python_packages"
    runtime_library = dkg_root / "build" / "lib" / "libcute_dsl_runtime.so"
    if not python_packages.is_dir() or not runtime_library.is_file():
        return

    _remove_nvctm_redirects()
    _prepend_paths(
        [
            python_packages,
            dkg_root / "cutlass_ir" / "runtime" / "python",
            dkg_root / "DkgDSL",
            dkg_root / "DkgDSL" / "tools",
        ]
    )
    os.environ["CUTE_DSL_LIBS"] = str(runtime_library.resolve())
    _ensure_mlir_type_compat()


_bootstrap_local_dkg()
