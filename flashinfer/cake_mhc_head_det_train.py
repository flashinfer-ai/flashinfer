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

Cake mHC training head, deterministic: the stream mean
``Y = mean_i X[:, i, :]`` of the last layer's four residual streams of the
four-stream mHC training layers (:mod:`flashinfer.cake_mhc_det_train`) and its
gradient, for hidden sizes 4096, 5120 and 7168 on SM100 / SM103 and 4096 on
SM107, bit for bit torch's BF16 ``X.mean(dim=1)`` and its gradient::

    Y = cake_mhc_head_det_train_forward(X)                    # BF16 [T, H] from [T, 4, H]
    dX = cake_mhc_head_det_train_backward(dY)                 # dX[:, i] = BF16(dY / 4)
    cake_mhc_head_det_train_backward(dY, dX=dX, accumulate=True)   # dX += BF16(dY / 4)
    Y = cake_mhc_head_det(X)                                  # autograd

Names and mode: this is the deterministic module of the head.  Its backend
package ``flashinfer.experimental.cake.cake_mhc_head_det_train`` is imported on
the first call or attribute access (importing this module imports no backend).
Every name this module exports carries the mode word ``det``
(``cake_mhc_head_det_train_*``, ``cake_mhc_head_det``,
``CakeMHCHeadDetFunction``) except the constants ``ARCHES`` and
``SUPPORTED_COMPUTE_CAPABILITIES``; the former names have no alias
(``flashinfer.cake_mhc_det_train``'s ``cake_mhc_det_train_head*``,
``CakeMHCDetTrainHeadFunction`` and ``is_cake_mhc_det_train_head_supported``,
and FlashInfer 0.7.3's ``cake_mhc_train_head*`` fail at import).  The entry
points take no ``deterministic`` keyword, give the same results under
``torch.use_deterministic_algorithms(True)`` and raise no alert.  Every output
(``Y``, the written ``dX`` and the accumulated ``dX``) repeats bit for bit for
identical inputs, also in the recompute of an activation checkpoint.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch

from .api_logging import flashinfer_api


# The backend package is imported on first use and cached: a call does not repeat the import.
@functools.cache
def _pkg():
    from .experimental.cake import cake_mhc_head_det_train  # noqa: PLC0415 - the backend package, imported on first use

    return cake_mhc_head_det_train


# The stable entry points are real ``def``s here (public-API tooling reads the
# decorated signatures of this module); their bodies live in the self-contained
# backend package next to the generated CUDA sources, imported on the first call.


@flashinfer_api
def cake_mhc_head_det_train_forward(
    X: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""The model's head: the mean of the four residual streams, ``Y = mean_i X[:, i, :]``.

    ``Y[t, d] = BF16(((X[t, 0, d] + X[t, 1, d]) + X[t, 2, d] + X[t, 3, d]) * 0.25)``
    with the sum in FP32 from zero in stream order and one rounding: bit for
    bit torch's ``X.mean(dim=1)`` of a BF16 ``X``, signed zeros and the NaN
    encoding included.

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous residual streams, ``H`` in 4096, 5120 or
        7168 (:func:`is_cake_mhc_head_det_train_supported`), 32-byte aligned.  ``T``
        is a runtime scalar; ``T == 0`` returns an empty ``Y`` without
        launching.
    out : torch.Tensor, optional
        Preallocated BF16 ``[T, H]`` contiguous ``Y`` (default ``None``: a new
        tensor); it must not overlap ``X``.

    Returns
    -------
    BF16 ``Y`` of shape ``[T, H]``.

    The launch runs on the current torch stream of ``X``'s device and
    allocates nothing beyond the default ``out``, so it can be captured in a
    CUDA graph once its program is loaded (one eager call first).
    """

    return _pkg().cake_mhc_head_det_train_forward(X, out=out)


@flashinfer_api
def cake_mhc_head_det_train_backward(
    dY: torch.Tensor,
    *,
    dX: Optional[torch.Tensor] = None,
    accumulate: bool = False,
) -> torch.Tensor:
    r"""Gradient of the head: ``dX[t, i, d] = BF16(dY[t, d] * 0.25)`` for each of the four streams.

    Bit for bit torch's ``MeanBackward`` of ``X.mean(dim=1)`` (the quarter is
    exact unless it is subnormal, and then rounds once).  With
    ``accumulate=True`` the kernel adds instead,
    ``dX[t, i, d] = BF16(dX[t, i, d] + BF16(dY[t, d] * 0.25))``, bit for bit
    torch's BF16 ``dX.add_(...)`` of that gradient (autograd's accumulation
    into an existing ``X.grad``).

    Parameters
    ----------
    dY : torch.Tensor
        BF16 ``[T, H]`` contiguous gradient of ``Y``, ``H`` in 4096, 5120 or
        7168, 32-byte aligned; ``T == 0`` launches nothing.
    dX : torch.Tensor, optional
        BF16 ``[T, 4, H]`` contiguous gradient of ``X`` to write (default
        ``None``: a new tensor) or, with ``accumulate=True``, to add into (then
        required); it must not overlap ``dY``.
    accumulate : bool
        Add into ``dX`` instead of overwriting it (default ``False``).

    Returns
    -------
    BF16 ``dX`` of shape ``[T, 4, H]`` (the caller's tensor when given).

    The launch runs on the current torch stream of ``dY``'s device.
    """

    return _pkg().cake_mhc_head_det_train_backward(dY, dX=dX, accumulate=accumulate)


@flashinfer_api
def cake_mhc_head_det(X: torch.Tensor) -> torch.Tensor:
    """Autograd-enabled :func:`cake_mhc_head_det_train_forward`: ``Y = mean_i X[:, i, :]``.

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous residual streams (may require
        gradients), ``H`` in 4096, 5120 or 7168.

    Returns
    -------
    BF16 ``Y`` of shape ``[T, H]``.  Its backward is
    :func:`cake_mhc_head_det_train_backward` (bit for bit torch's gradient of
    ``X.mean(dim=1)``); the node saves no tensor, so an activation checkpoint
    around it keeps nothing alive and its backward needs no recomputation.
    """

    return _pkg().cake_mhc_head_det(X)


# The backend objects this module re-exports, resolved on first access so that
# importing this module imports no backend.
_LAZY = (
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeMHCHeadDetFunction",
    "is_cake_mhc_head_det_train_supported",
)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(_pkg(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "ARCHES",  # noqa: F822 - resolved by __getattr__
    "SUPPORTED_COMPUTE_CAPABILITIES",  # noqa: F822
    "CakeMHCHeadDetFunction",  # noqa: F822
    "cake_mhc_head_det",
    "cake_mhc_head_det_train_backward",
    "cake_mhc_head_det_train_forward",
    "is_cake_mhc_head_det_train_supported",  # noqa: F822
]
