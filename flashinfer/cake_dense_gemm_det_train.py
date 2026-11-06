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

Cake dense GEMMs for training, deterministic (SM100 / SM103 / SM107): the GLM-5.2
BF16 dense projection GEMM (K1) and the FP32 router-gate GEMM through split-BF16x3
emulation (K2).  The same inputs, hardware and plan give bitwise the same outputs
on every run.  The module selects the mode: no entry point takes a mode argument,
and no entry point raises or warns under ``torch.use_deterministic_algorithms``.

The backend lives in ``flashinfer.experimental.cake.cake_dense_gemm_det_train``
(``cake_backend.py``, the generated sources and ``build.json``).  This module
imports it on the first attribute access and returns the backend's own objects,
so a call adds no Python layer.  ``CONVENIENCE_CACHE_SIZE`` is a setting of the
backend module (``cake_backend.CONVENIENCE_CACHE_SIZE``), not of this module.

Usage::

    from flashinfer.cake_dense_gemm_det_train import (
        cake_dense_gemm_det_train_projection_forward,
        cake_dense_gemm_det_train_router_wgrad,
    )

    y = cake_dense_gemm_det_train_projection_forward(X, W)  # y[T, N] = X[T, K] @ W[N, K].T
    dW = cake_dense_gemm_det_train_router_wgrad(G, X)       # FP32 dW[N, K] = G[T, N].T @ X[T, K]
"""

from __future__ import annotations

import importlib
from typing import Any

_BACKEND = "flashinfer.experimental.cake.cake_dense_gemm_det_train"

# public name -> backend submodule that defines it
_EXPORTS: dict[str, str] = {
    "ARCHES": "cake_jit",
    "SUPPORTED_COMPUTE_CAPABILITIES": "cake_backend",
    "is_cake_dense_gemm_det_train_supported": "cake_backend",
    # K1: BF16 dense projection GEMM
    "cake_dense_gemm_det_train_projection": "cake_backend",
    "cake_dense_gemm_det_train_projection_forward": "cake_backend",
    "cake_dense_gemm_det_train_projection_dgrad": "cake_backend",
    "cake_dense_gemm_det_train_projection_wgrad": "cake_backend",
    "cake_dense_gemm_det_train_projection_plan": "cake_backend",
    "cake_dense_gemm_det_train_projection_prepare": "cake_backend",
    "cake_dense_gemm_det_train_projection_wgrad_prepare": "cake_backend",
    "CakeDenseGemmDetProjectionPlan": "cake_backend",
    "CakeDenseGemmDetProjectionLaunch": "cake_backend",
    # K2: FP32 router GEMM
    "cake_dense_gemm_det_train_router": "cake_backend",
    "cake_dense_gemm_det_train_router_forward": "cake_backend",
    "cake_dense_gemm_det_train_router_dgrad": "cake_backend",
    "cake_dense_gemm_det_train_router_wgrad": "cake_backend",
    "cake_dense_gemm_det_train_router_plan": "cake_backend",
    "cake_dense_gemm_det_train_router_prepare": "cake_backend",
    "CakeDenseGemmDetRouterPlan": "cake_backend",
    "CakeDenseGemmDetRouterLaunch": "cake_backend",
    # the convenience path's cached plans and workspace arenas
    "cake_dense_gemm_det_train_convenience_cache_clear": "cake_backend",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{_BACKEND}.{submodule}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return list(__all__)
