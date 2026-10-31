# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Compatibility import for flashinfer.cake_rmsnorm_silu_gate_det_train_backend.

``__all__`` is the det package's and an attribute of this package resolves on the det
package.  The submodules ``cake_build``, ``cake_jit`` are the det package's module objects,
registered in ``sys.modules`` under this package's name when it is imported.  This
package holds no sources and no build manifest.
"""

from importlib import import_module
import sys

_det = import_module("flashinfer.cake_rmsnorm_silu_gate_det_train_backend")

__all__ = list(_det.__all__)

for _submodule in ("cake_build", "cake_jit"):
    sys.modules[f"{__name__}.{_submodule}"] = import_module(
        f"{_det.__name__}.{_submodule}"
    )
del _submodule


def __getattr__(name):
    return getattr(_det, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_det)))
