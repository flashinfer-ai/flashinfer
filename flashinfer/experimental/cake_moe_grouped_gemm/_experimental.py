"""
Copyright (c) 2026 by FlashInfer team.

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

# Experimental-API helpers of this package: FlashInfer's own when FlashInfer is
# importable, stand-ins otherwise, so that a copy of the package loaded on its
# own -- under another import path, in a process without a FlashInfer
# installation, or with ``sys.modules["flashinfer"] = None`` as a vendoring
# gate sets it -- still imports.
#
# This module is stdlib-only apart from the guarded block in ``_core_helpers``.
# That block and the JIT fallback of ``cake_build.py`` are the only places in
# the package that may name ``flashinfer``, and both sit inside
# ``try: ... except ImportError``.  Do not import ``flashinfer`` anywhere else
# here: ``tests/experimental/test_cake_moe_grouped_gemm_standalone.py`` parses
# the package and fails if you do.
#
# The guard is ``except ImportError`` only and runs when the package is
# imported.  A copy loaded next to an importable FlashInfer therefore binds
# that installation's helpers and imports ``flashinfer`` at package import; a
# FlashInfer without these helpers (an ImportError) and the vendoring gate
# (``sys.modules["flashinfer"] = None``) take the stand-ins.  The stand-ins emit
# the same warning text with the same attributes, ``stacklevel`` and once-only
# rule as FlashInfer's helpers; they do no API logging, read no environment
# variable, and their category is the ``ExperimentalWarning`` below (a
# ``UserWarning``), not FlashInfer's class.

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, Optional

__all__ = [
    "STANDALONE",
    "ExperimentalWarning",
    "flashinfer_experimental_api",
    "warn_experimental_backend_once",
]

# (api_name, backend) pairs the stand-in has already warned about.
_WARNED_BACKENDS: set[tuple[str, str]] = set()


class ExperimentalWarning(UserWarning):
    """Warning of the stand-in helpers (the package running without FlashInfer).

    Emitted once per experimental function on first use and once per
    (stable API, backend) pair.  Inside FlashInfer the warnings carry
    ``flashinfer.api_logging.ExperimentalWarning`` instead.
    """


def _core_helpers() -> Optional[tuple[Any, Any]]:
    """FlashInfer's two helpers as a pair, or ``None`` when they cannot be imported."""
    try:
        from flashinfer.api_logging import (
            flashinfer_experimental_api,
            warn_experimental_backend_once,
        )
    except ImportError:  # no importable FlashInfer, or one without these helpers
        return None
    return flashinfer_experimental_api, warn_experimental_backend_once


def _standalone_experimental_api(
    func: Optional[Callable] = None, *, feature: Optional[str] = None
) -> Callable:
    """Stand-in for ``flashinfer.api_logging.flashinfer_experimental_api``.

    Marks ``f`` and the returned wrapper ``is_experimental``, records
    ``experimental_feature``, prepends the docstring banner and emits one
    :class:`ExperimentalWarning` on the first call.  No API logging; the
    ``trace`` keyword of FlashInfer's decorator is not taken (no call site in
    this package passes it).
    """

    def decorator(f: Callable) -> Callable:
        f.is_experimental = True  # type: ignore[attr-defined]
        feature_name = feature if feature is not None else f.__qualname__
        warned = False

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal warned
            if not warned:
                warned = True
                warnings.warn(
                    f"'{feature_name}' is an experimental FlashInfer API: it "
                    f"provides no compatibility guarantees and may change or be "
                    f"removed without deprecation.",
                    ExperimentalWarning,
                    stacklevel=2,
                )
            return f(*args, **kwargs)

        wrapper.is_experimental = True  # type: ignore[attr-defined]
        wrapper.experimental_feature = feature_name  # type: ignore[attr-defined]
        banner = (
            ".. warning::\n"
            f"    ``{feature_name}`` is experimental: it provides no "
            f"compatibility guarantees and may change or be removed without "
            f"deprecation."
        )
        wrapper.__doc__ = banner if f.__doc__ is None else banner + "\n\n" + f.__doc__
        return wrapper

    return decorator if func is None else decorator(func)


def _standalone_warn_backend_once(
    api_name: str, backend: str, *, automatic: bool = False
) -> None:
    """Stand-in for ``flashinfer.api_logging.warn_experimental_backend_once``.

    Emits :class:`ExperimentalWarning` once per ``(api_name, backend)`` pair,
    attributed to the caller of the stable API (``stacklevel=3``).  The set of
    warned pairs is this module's, so a second copy of the package in the same
    process keeps its own.
    """
    key = (api_name, backend)
    if key in _WARNED_BACKENDS:
        return
    _WARNED_BACKENDS.add(key)
    how = "may be selected automatically for" if automatic else "selected for"
    warnings.warn(
        f"experimental backend '{backend}' {how} '{api_name}': it provides no "
        f"compatibility guarantees and may change or be removed without "
        f"deprecation.",
        ExperimentalWarning,
        stacklevel=3,
    )


_CORE = _core_helpers()
# True when the stand-ins are bound (FlashInfer's helpers were not importable).
STANDALONE = _CORE is None
flashinfer_experimental_api = (
    _CORE[0] if _CORE is not None else _standalone_experimental_api
)
warn_experimental_backend_once = (
    _CORE[1] if _CORE is not None else _standalone_warn_backend_once
)
