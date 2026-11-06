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
``entries`` dict serves either mode).  The deterministic names are unchanged and always deterministic; their
names before the det rename (``flashinfer.cake_causal_conv1d_det_train.cake_causal_conv1d_kcp`` /
``cake_causal_conv1d_kcp_multi`` / ``KcpRecomputeHandle``) are removed::

    from flashinfer.cake_causal_conv1d_train_kcp import cake_causal_conv1d_kcp, cake_causal_conv1d_multi_kcp

    y_local = cake_causal_conv1d_kcp(x_local, weight, ctx)                       # deterministic (default)
    y_local = cake_causal_conv1d_kcp(x_local, weight, ctx, deterministic=False)  # one-launch backward
    q, k, v = cake_causal_conv1d_multi_kcp((xq, xk, xv), (wq, wk, wv), ctx, deterministic=False)

The non-deterministic backward beside a kernel of another stream (FlashInfer issue #8365): ``bwd_reserve_sms=K``
(``deterministic=False`` only, SM100 / SM103; ``ValueError`` on SM107) launches the main backward ``K`` SMs short of
its single wave::

    y_local = cake_causal_conv1d_kcp(x_local, weight, ctx, deterministic=False, bwd_reserve_sms=1)
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence

import torch

from .api_logging import flashinfer_experimental_api

_FEATURE = "Cake context-parallel causal conv1d + SiLU training with a selectable numerical mode (SM100/SM103/SM107)"
# The entry points that already warned about the deterministic fallback under torch's deterministic mode.
_FALLBACK_WARNED: set[str] = set()


def _bwd_reserve_sms(
    bwd_reserve_sms: object, deterministic: object, entry: str
) -> Optional[int]:
    """The validated ``bwd_reserve_sms`` keyword (``None`` when not given: no validation)."""

    if bwd_reserve_sms is None:
        return None
    if isinstance(bwd_reserve_sms, bool) or not isinstance(bwd_reserve_sms, int):
        raise TypeError(
            f"{entry}: bwd_reserve_sms must be an int or None, got {type(bwd_reserve_sms).__name__}"
        )
    if bwd_reserve_sms < 0:
        raise ValueError(
            f"{entry}: bwd_reserve_sms must be >= 0, got {bwd_reserve_sms}"
        )
    if deterministic is True and bwd_reserve_sms >= 1:
        raise ValueError(
            f"{entry}: bwd_reserve_sms={bwd_reserve_sms} applies to deterministic=False only; with "
            "deterministic=True pass cooperative=False (the deterministic backward's form beside a kernel of "
            "another stream)"
        )
    return int(bwd_reserve_sms)


def _given(**keywords: Any) -> dict[str, Any]:
    """The keywords a call gave (not ``None``): forwarded so a call without them reaches the entry point with its
    own defaults."""

    return {name: value for name, value in keywords.items() if value is not None}


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
    cooperative: Optional[bool] = None,
    deterministic: bool = True,
    bwd_reserve_sms: Optional[int] = None,
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
    cooperative : bool, optional
        Keyword-only, default ``None``.  For the deterministic entry point, the launch form of its main backward,
        forwarded when given (``False`` starts beside a small kernel of another stream; see
        :func:`flashinfer.cake_causal_conv1d_det_train_kcp.cake_causal_conv1d_det_kcp`).  The non-deterministic
        main backward is always one ordinary launch: there ``None`` and ``False`` mean the same and ``True`` raises
        ``ValueError``.
    deterministic : bool
        ``True`` (default): ``cake_causal_conv1d_det_kcp`` (fixed-order, bitwise reproducible ``dweight`` partial).
        ``False``: the one-launch backward with an unordered ``dweight`` partial (``y`` / ``dx`` bitwise the
        deterministic ones).  torch's deterministic mode turns ``False`` into ``True`` with a one-time warning.
    bwd_reserve_sms : int, optional
        Keyword-only, default ``None`` (the non-deterministic package's ``RESERVE_DEFAULT``, 0).  The SMs the
        non-deterministic main backward leaves to kernels of other streams (issue #8365): ``K >= 1`` launches its
        single-wave grid ``K x ctas_per_sm`` CTAs short, so a kernel of another stream that holds up to ``K`` SMs
        (for example a communication kernel) does not push the top CTAs into a second wave; the unloaded backward
        gives up the work of those ``K`` SMs.  ``y``, ``dx`` and the exchanges are unchanged and ``dweight`` stays
        within the same budget.  With ``K = 0`` (or ``None``) the backward already launches one SM's CTAs short
        wherever the last channel-tile rank of the full grid holds at least ``ctas_per_sm`` CTAs, so ``K = 1``
        leaves the grid unchanged at those widths (for example ``C`` = 10240, 6144 and 5120 on a 152-SM GB200 /
        GB300, ``C`` = 10240, 6144 and 3072 on a 148-SM B200) and launches one SM's CTAs fewer at every other width.
        Validated only when given: a ``bool`` or a non-``int`` raises ``TypeError``, a negative value or one not
        below the device's SM count ``ValueError``.  On sm_107a ``K >= 1`` raises ``ValueError``: there the reserved
        grid costs up to 1.055x the shipped grid unloaded (``C`` = 1024, 62585 tokens), above this opt-in's 1.040
        limit.

        ==============  ===============  ================================================================
        deterministic   bwd_reserve_sms  backward
        ==============  ===============  ================================================================
        ``True``        ``None`` / 0     the deterministic entry point (the keyword is not forwarded)
        ``True``        ``>= 1``         ``ValueError``: pass ``cooperative=False`` instead (the
                                         deterministic backward's form beside a kernel of another stream)
        ``False``       ``None`` / 0     the non-deterministic backward's shipped grid
        ``False``       ``K >= 1``       the non-deterministic backward ``K`` SMs short (``ValueError`` on
                                         sm_107a)
        ==============  ===============  ================================================================

        Portable spelling beside a one-SM kernel of another stream: ``cooperative=False`` for
        ``deterministic=True``, ``bwd_reserve_sms=1`` for ``deterministic=False``.  Under
        ``torch.use_deterministic_algorithms(True)``, ``deterministic=False`` runs the deterministic entry point and
        the keyword (validated) is not forwarded.  Each rank may choose its own value (no collective depends on it).

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x_local``'s shape.  The backward returns BF16 ``dx`` and the rank-local FP32 ``dweight``
        partial.
    """

    if cooperative is not None or bwd_reserve_sms is not None:
        return _kcp_with_keywords(
            x_local,
            weight,
            ctx,
            recompute_handle=recompute_handle,
            cooperative=cooperative,
            deterministic=deterministic,
            bwd_reserve_sms=bwd_reserve_sms,
        )
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


def _kcp_with_keywords(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    *,
    recompute_handle: Optional[Any],
    cooperative: Optional[bool],
    deterministic: bool,
    bwd_reserve_sms: Optional[int],
) -> torch.Tensor:
    """:func:`cake_causal_conv1d_kcp` with ``cooperative`` or ``bwd_reserve_sms`` given: validated, then forwarded
    to the selected entry point (each keyword only when given)."""

    reserve = _bwd_reserve_sms(bwd_reserve_sms, deterministic, "cake_causal_conv1d_kcp")
    if _deterministic(deterministic, "cake_causal_conv1d_kcp"):
        from .cake_causal_conv1d_det_train_kcp import (  # noqa: PLC0415
            cake_causal_conv1d_det_kcp,
        )

        return cake_causal_conv1d_det_kcp(
            x_local,
            weight,
            ctx,
            recompute_handle=recompute_handle,
            **_given(cooperative=cooperative),
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
        **_given(cooperative=cooperative, bwd_reserve_sms=reserve),
    )


@flashinfer_experimental_api(feature=_FEATURE)
def cake_causal_conv1d_multi_kcp(
    xs: Sequence[torch.Tensor],
    weights: Sequence[torch.Tensor],
    ctx: Any,
    *,
    recompute_handle: Optional[Any] = None,
    cooperative: Optional[bool] = None,
    deterministic: bool = True,
    bwd_reserve_sms: Optional[int] = None,
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
    cooperative : bool, optional
        As in :func:`cake_causal_conv1d_kcp`, one form for every tensor of the call.
    deterministic : bool
        As in :func:`cake_causal_conv1d_kcp`.
    bwd_reserve_sms : int, optional
        As in :func:`cake_causal_conv1d_kcp`, one value for every tensor of the call.

    Returns
    -------
    tuple of torch.Tensor
        The ``n`` BF16 outputs, each of its input's shape.
    """

    if cooperative is not None or bwd_reserve_sms is not None:
        return _multi_kcp_with_keywords(
            xs,
            weights,
            ctx,
            recompute_handle=recompute_handle,
            cooperative=cooperative,
            deterministic=deterministic,
            bwd_reserve_sms=bwd_reserve_sms,
        )
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


def _multi_kcp_with_keywords(
    xs: Sequence[torch.Tensor],
    weights: Sequence[torch.Tensor],
    ctx: Any,
    *,
    recompute_handle: Optional[Any],
    cooperative: Optional[bool],
    deterministic: bool,
    bwd_reserve_sms: Optional[int],
) -> tuple[torch.Tensor, ...]:
    """:func:`cake_causal_conv1d_multi_kcp` with ``cooperative`` or ``bwd_reserve_sms`` given: validated, then
    forwarded to the selected entry point (each keyword only when given)."""

    reserve = _bwd_reserve_sms(
        bwd_reserve_sms, deterministic, "cake_causal_conv1d_multi_kcp"
    )
    if _deterministic(deterministic, "cake_causal_conv1d_multi_kcp"):
        from .cake_causal_conv1d_det_train_kcp import (  # noqa: PLC0415
            cake_causal_conv1d_det_multi_kcp,
        )

        return cake_causal_conv1d_det_multi_kcp(
            xs,
            weights,
            ctx,
            recompute_handle=recompute_handle,
            **_given(cooperative=cooperative),
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
        **_given(cooperative=cooperative, bwd_reserve_sms=reserve),
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
