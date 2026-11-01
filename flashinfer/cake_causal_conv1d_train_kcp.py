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

Context-parallel (KCP) packed width-4 depthwise causal convolution + SiLU training (SM100 / SM103 / SM107) with
a selectable numerical mode: thin experimental entry points.  ``deterministic=True`` (the default) runs the
deterministic KCP entry points of :mod:`flashinfer.cake_causal_conv1d_det_train_kcp` unchanged;
``deterministic=False`` runs the non-deterministic KCP backend
``flashinfer.experimental.cake.cake_causal_conv1d_train_backend.kcp`` (imported on first use).  With
``torch.use_deterministic_algorithms(True)`` a call with ``deterministic=False`` runs the deterministic entry point
and warns once (``UserWarning``).

Both modes share the context contract (the ``KcpContext`` of the KDA step; the numerical mode is never stored in
it), FP32 taps and no bias, the halo exchanges (one ``all_to_all_single`` per direction, with recompute handles
for communication-free checkpoint recomputes, #7818), the recompute-handle rules and the multi-tensor calls.
``deterministic=False`` changes the main backward kernel only: ONE ordinary launch (no cooperative attribute, no
grid barrier) whose CTAs add their FP32 ``dweight`` rows into zeroed accumulators with unordered global
reductions.  ``y`` and ``dx`` are bitwise the deterministic route's; the rank-local FP32 ``dweight`` partial
stays within the deterministic route's error budget but is not bitwise reproducible run to run (its cross-rank
reduction remains the framework's).  Every rank of ``ctx.group`` must pass the same ``deterministic`` value (and
run under the same torch deterministic setting) for a call: the two modes run the same collectives, but mixing
them mixes reproducibility contracts across ranks.

Name map: ``cake_causal_conv1d_det_kcp`` -> :func:`cake_causal_conv1d_kcp`, ``cake_causal_conv1d_det_multi_kcp``
-> :func:`cake_causal_conv1d_multi_kcp`, ``RecomputeHandleDetKcp`` -> ``RecomputeHandleKcp`` (any object with an
``entries`` dict serves either mode).  The deterministic names and their previous aliases
(``flashinfer.cake_causal_conv1d_det_train.cake_causal_conv1d_kcp`` / ``cake_causal_conv1d_kcp_multi`` /
``KcpRecomputeHandle``) are unchanged and always deterministic::

    from flashinfer.cake_causal_conv1d_train_kcp import cake_causal_conv1d_kcp, cake_causal_conv1d_multi_kcp

    y_local = cake_causal_conv1d_kcp(x_local, weight, ctx)                       # deterministic (default)
    y_local = cake_causal_conv1d_kcp(x_local, weight, ctx, deterministic=False)  # one-launch backward
    q, k, v = cake_causal_conv1d_multi_kcp((xq, xk, xv), (wq, wk, wv), ctx, deterministic=False)
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence

import torch

from .api_logging import flashinfer_experimental_api

_FEATURE = "Cake context-parallel causal conv1d + SiLU training with a selectable numerical mode (SM100/SM103/SM107)"
# The entry points that already warned about the deterministic fallback under torch's deterministic mode.
_FALLBACK_WARNED: set[str] = set()


def _deterministic(deterministic: object, entry: str) -> bool:
    """Whether a call runs the deterministic entry point: ``deterministic`` itself, or torch's deterministic mode
    overriding ``deterministic=False`` with a one-time warning per entry point."""

    if not isinstance(deterministic, bool):
        raise TypeError(
            f"{entry}: deterministic must be a bool, got {type(deterministic).__name__}"
        )
    if deterministic:
        return True
    if torch.are_deterministic_algorithms_enabled():
        if entry not in _FALLBACK_WARNED:
            _FALLBACK_WARNED.add(entry)
            warnings.warn(
                f"{entry}(deterministic=False) runs the deterministic implementation because "
                "torch.use_deterministic_algorithms(True) is in effect",
                UserWarning,
                stacklevel=3,
            )
        return True
    return False


@flashinfer_experimental_api(feature=_FEATURE)
def cake_causal_conv1d_kcp(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    *,
    recompute_handle: Optional[Any] = None,
    deterministic: bool = True,
) -> torch.Tensor:
    """Autograd-enabled context-parallel packed width-4 depthwise causal convolution + SiLU of this rank's fragments.

    The first tokens of a fragment that continues a document read the history rows of the preceding fragment
    (exchanged through the context's halo collective) instead of zeros.  FP32 taps, no bias.

    Parameters
    ----------
    x_local : torch.Tensor
        BF16 ``[S_local, C]`` or ``[1, S_local, C]`` contiguous packed activations of this rank's fragments, in the
        partition's order; ``C`` is a multiple of 8.
    weight : torch.Tensor
        FP32 ``[C, 4]`` contiguous taps.
    ctx : ContextDetKcp
        From ``flashinfer.cake_kda_det_train_kcp.build_context_det_kcp`` (the same context the KDA step of the rank
        uses), whatever the numerical mode.
    recompute_handle : RecomputeHandleKcp, optional
        Keyword-only, default ``None``.  A handle (any object with an ``entries`` dict) makes the checkpoint
        recompute of this call run without the forward halo exchange, under the rules of
        :func:`flashinfer.cake_causal_conv1d_det_train_kcp.cake_causal_conv1d_det_kcp` (one handle per call and
        micro-step, created outside the outermost checkpoint region; every rank passes one or none).
    deterministic : bool
        ``True`` (default): ``cake_causal_conv1d_det_kcp`` (fixed-order, bitwise reproducible ``dweight`` partial).
        ``False``: the one-launch backward with an unordered ``dweight`` partial (``y`` / ``dx`` bitwise the
        deterministic ones).  torch's deterministic mode turns ``False`` into ``True`` with a one-time warning.

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x_local``'s shape.  The backward returns BF16 ``dx`` and the rank-local FP32 ``dweight``
        partial.
    """

    if _deterministic(deterministic, "cake_causal_conv1d_kcp"):
        from .cake_causal_conv1d_det_train_kcp import (  # noqa: PLC0415
            cake_causal_conv1d_det_kcp,
        )

        return cake_causal_conv1d_det_kcp(
            x_local, weight, ctx, recompute_handle=recompute_handle
        )
    from .experimental.cake.cake_causal_conv1d_train_backend import (  # noqa: PLC0415
        kcp as _kcp,
    )

    return _kcp.cake_causal_conv1d_kcp(
        x_local,
        weight,
        ctx,
        recompute_handle=recompute_handle,
        deterministic=False,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def cake_causal_conv1d_multi_kcp(
    xs: Sequence[torch.Tensor],
    weights: Sequence[torch.Tensor],
    ctx: Any,
    *,
    recompute_handle: Optional[Any] = None,
    deterministic: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Autograd-enabled context-parallel convolutions of several tensors of one layer (for example its q, k and v)
    with one halo exchange per direction.

    Outputs and ``dx`` are bitwise ``tuple(cake_causal_conv1d_kcp(x, w, ctx, deterministic=d) for x, w in
    zip(xs, weights))``; with ``deterministic=True`` every ``dweight`` is bitwise as well, with ``False`` each is
    that tensor's own unordered FP32 partial.

    Parameters
    ----------
    xs : sequence of torch.Tensor
        ``n >= 1`` BF16 ``[S_local, C_i]`` or ``[1, S_local, C_i]`` contiguous packed activations of this rank's
        fragments; each ``C_i`` is a multiple of 8 (they may differ).
    weights : sequence of torch.Tensor
        FP32 ``[C_i, 4]`` contiguous taps, one per tensor.
    ctx : ContextDetKcp
        As in :func:`cake_causal_conv1d_kcp`.  Every rank of the group calls this function once per layer, in the
        same order, with the same tensor count, channel counts and ``deterministic`` value.
    recompute_handle : RecomputeHandleKcp, optional
        One handle for the whole call (its ``"conv"`` entry holds the ``n`` history pools), under the rules of
        :func:`cake_causal_conv1d_kcp`.
    deterministic : bool
        As in :func:`cake_causal_conv1d_kcp`.

    Returns
    -------
    tuple of torch.Tensor
        The ``n`` BF16 outputs, each of its input's shape.
    """

    if _deterministic(deterministic, "cake_causal_conv1d_multi_kcp"):
        from .cake_causal_conv1d_det_train_kcp import (  # noqa: PLC0415
            cake_causal_conv1d_det_multi_kcp,
        )

        return cake_causal_conv1d_det_multi_kcp(
            xs, weights, ctx, recompute_handle=recompute_handle
        )
    from .experimental.cake.cake_causal_conv1d_train_backend import (  # noqa: PLC0415
        kcp as _kcp,
    )

    return _kcp.cake_causal_conv1d_multi_kcp(
        xs,
        weights,
        ctx,
        recompute_handle=recompute_handle,
        deterministic=False,
    )


# ``RecomputeHandleKcp`` (the non-deterministic KCP backend's recompute handle class) resolves on first access, so
# importing this module imports neither the experimental package nor a KCP host module (torch.distributed, the KCP
# loader).
def __getattr__(name: str) -> Any:
    if name == "RecomputeHandleKcp":
        from .experimental.cake.cake_causal_conv1d_train_backend.kcp import (  # noqa: PLC0415
            RecomputeHandleKcp,
        )

        return RecomputeHandleKcp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"RecomputeHandleKcp"})


__all__ = [
    "RecomputeHandleKcp",  # noqa: F822 - resolved by __getattr__
    "cake_causal_conv1d_kcp",
    "cake_causal_conv1d_multi_kcp",
]
