# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Op metadata contract for flashinfer.experimental.sm12x.

Every op directory's ``__init__.py`` is exactly: a docstring, an ``OpMeta``
named ``META``, a ``TYPE_CHECKING`` import block, and one call to
:func:`install_lazy_api`.  The registry test enforces the shape.

This module must stay stdlib-only: op ``__init__`` files import it at
namespace-import time, which must not pull torch/cutlass/triton.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, kw_only=True)
class Provenance:
    """Where an op's code came from (auditable against upstream forever)."""

    repo: str
    commit: str
    paths: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class OpMeta:
    """Machine-readable header describing one experimental op.

    ``api_style`` declares the op's lifecycle contract:

    - ``planned``:  ``Caps -> plan() -> bind() -> run*()``; ``plan`` is
      host-side and may allocate, ``bind`` only creates views (allocation
      free), ``run*`` is CUDA-graph-capture safe.
    - ``oneshot``:  plain functions with domain verbs (``mm``, ``quantize``).
    - ``stateful``: long-lived class instances (comm collectives).
    """

    name: str
    group: str
    api_style: Literal["planned", "oneshot", "stateful"]
    entry_points: tuple[str, ...]
    archs: tuple[str, ...] = ("sm120a", "sm121a")
    dtypes: tuple[str, ...] = ()
    recipes: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    status: Literal["experimental"] = "experimental"
    provenance: Provenance = field(
        default_factory=lambda: Provenance(repo="", commit="", paths=())
    )
    test_path: str = ""
    since: str = ""
    notes: str = ""

    @property
    def qualname(self) -> str:
        return f"{self.group}.{self.name}"


def install_lazy_api(module_globals: dict[str, Any], meta: OpMeta) -> None:
    """Wire an op module's lazy public surface from its ``META``.

    Installs PEP 562 ``__getattr__``/``__dir__`` plus ``__all__`` so that
    every entry point resolves from the op's ``api`` module on first access.
    Importing the op module itself therefore stays side-effect free (no
    cutlass/triton import, no torch custom-op registration).
    """

    module_name = module_globals["__name__"]

    def __getattr__(name: str) -> Any:
        if name in meta.entry_points:
            api = importlib.import_module(".api", module_name)
            value = getattr(api, name)
            module_globals[name] = value
            return value
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    def __dir__() -> list[str]:
        return sorted([*meta.entry_points, "META"])

    module_globals["__getattr__"] = __getattr__
    module_globals["__dir__"] = __dir__
    module_globals["__all__"] = [*meta.entry_points, "META"]


__all__ = ["OpMeta", "Provenance", "install_lazy_api"]
