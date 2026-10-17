"""The one place the CAKE backend test suites name their package.

Every CAKE backend suite (``tests/experimental/test_cake_<package>.py``, ``tests/norm/test_cake_rmsnorm_train.py``,
``tests/experimental/test_mok_*.py``) imports its package through :func:`cake_package` and imports nothing else from
flashinfer; the flashinfer public-API tests of a package live in its ``test_cake_<package>_api.py`` sibling.  A build
that vendors a package under another import path runs the backend suite unchanged after pointing the entry of
:data:`PACKAGES` at its own package (flashinfer issue #6553, ask 9).
"""

from __future__ import annotations

import importlib
from types import ModuleType

# suite / package name -> import path of the package
PACKAGES: dict[str, str] = {
    "cake_rmsnorm_train": "flashinfer.cake_rmsnorm_train_backend",
    "cake_dsa_train": "flashinfer.experimental.cake_dsa_train",
    "cake_dsa_indexer": "flashinfer.experimental.cake_dsa_indexer",
    "cake_lm_head_loss": "flashinfer.experimental.cake_lm_head_loss",
    "dense_projection_gemm": "flashinfer.experimental.dense_projection_gemm",
    "cake_moe_grouped_gemm": "flashinfer.experimental.cake_moe_grouped_gemm",
    "cake_mok_bf16": "flashinfer.experimental.cake_mok_bf16",
    "cake_stepfun_moe": "flashinfer.experimental.cake_stepfun_moe",
}


def cake_package(name: str, submodule: str | None = None) -> ModuleType:
    """The package module of ``name`` (a key of :data:`PACKAGES`), or its ``submodule`` (``"cake_backend"``, ...).

    ``cake_package("cake_lm_head_loss")`` is the package itself; ``cake_package("cake_lm_head_loss", "cake_backend")``
    is ``<package>.cake_backend``.  Raises ``KeyError`` for a name the table does not list.
    """
    path = PACKAGES[name]
    if submodule:
        path = f"{path}.{submodule}"
    return importlib.import_module(path)
