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

Context-parallel (KCP) packed width-4 depthwise causal convolution + SiLU training (SM100 / SM103 / SM107), the
non-deterministic package: thin experimental entry points of
``flashinfer.experimental.cake.cake_causal_conv1d_train_backend.kcp`` (imported on first use).  The deterministic
KCP entry points are :mod:`flashinfer.cake_causal_conv1d_det_train_kcp` (``cake_causal_conv1d_det_kcp``,
``cake_causal_conv1d_det_multi_kcp``, ``RecomputeHandleDetKcp``); this module never runs them.

Numerical mode (the same rule as every non-deterministic training package; "torch deterministic mode" is
``torch.use_deterministic_algorithms(True)``, with or without ``warn_only=True``)::

    call                                   torch deterministic mode off   torch deterministic mode on
    this module, deterministic=False       non-deterministic              the forward runs; the backward raises
                                                                            RuntimeError before any exchange
                                                                            (warn_only=True: a UserWarning, then
                                                                            it runs); it never falls back
    this module, any other deterministic   ValueError naming the          ValueError
      value (the default True included)      deterministic entry point
    the deterministic names (*_det_*)      deterministic; no              deterministic; no warning
                                             deterministic keyword
                                             (TypeError)

Every entry point takes keyword-only ``deterministic: bool = True`` and runs only with ``deterministic=False``, so a
call states the non-deterministic mode explicitly and a call that relied on a deterministic default fails with an
error that names the deterministic entry point instead of silently changing mode.  Every rank of ``ctx.group`` runs
the same entry points (and the same torch deterministic setting) for a call.

Contract: the context of the KDA step (``ContextDetKcp`` from
``flashinfer.cake_kda_det_train_kcp.build_context_det_kcp``), FP32 taps and no bias, the halo exchanges (one
``all_to_all_single`` per direction, with recompute handles for communication-free checkpoint recomputes, #7818),
the recompute-handle rules and the multi-tensor calls are the deterministic KCP package's.  The main backward is ONE
ordinary launch (no cooperative attribute, no grid barrier) whose CTAs add their FP32 ``dweight`` rows into zeroed
accumulators with unordered global reductions.  ``y`` and ``dx`` are bitwise the deterministic KCP entry point's;
the rank-local FP32 ``dweight`` partial stays within the deterministic error budget but is not bitwise reproducible
run to run (its cross-rank reduction remains the framework's).  The package is chosen by name at each call, so a
checkpoint recompute (``torch.utils.checkpoint``, reentrant or not) runs the same package; keep torch's
deterministic setting the same from a forward to its backward.

Name map (deterministic -> this module): ``cake_causal_conv1d_det_kcp`` -> :func:`cake_causal_conv1d_kcp`,
``cake_causal_conv1d_det_multi_kcp`` -> :func:`cake_causal_conv1d_multi_kcp`, ``RecomputeHandleDetKcp`` ->
``RecomputeHandleKcp`` (any object with an ``entries`` dict serves either package)::

    from flashinfer.experimental.cake.cake_causal_conv1d_train_kcp import cake_causal_conv1d_kcp, cake_causal_conv1d_multi_kcp

    y_local = cake_causal_conv1d_kcp(x_local, weight, ctx, deterministic=False)  # one-launch backward
    q, k, v = cake_causal_conv1d_multi_kcp((xq, xk, xv), (wq, wk, wv), ctx, deterministic=False)

The backward beside a kernel of another stream (FlashInfer issue #8365): ``bwd_reserve_sms=K`` (SM100 / SM103;
``ValueError`` on SM107) launches the main backward ``K`` SMs short of its single wave::

    y_local = cake_causal_conv1d_kcp(x_local, weight, ctx, deterministic=False, bwd_reserve_sms=1)
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from ...api_logging import flashinfer_experimental_api

_FEATURE = "Cake context-parallel causal conv1d + SiLU training, non-deterministic package (SM100/SM103/SM107)"


def _nondeterministic_only(deterministic: object, entry: str, det_entry: str) -> None:
    """Refuse every ``deterministic`` value but ``False`` (the default ``True`` included) with a ``ValueError`` that
    names the deterministic entry point: this module runs the non-deterministic package only."""

    if deterministic is not False:
        raise ValueError(
            f"{entry} runs the non-deterministic package (unordered dweight reductions) and needs "
            f"deterministic=False, got deterministic={deterministic!r}; the deterministic entry point is "
            f"flashinfer.cake_causal_conv1d_det_train_kcp.{det_entry} (pass deterministic=False here for the "
            "non-deterministic path)"
        )


def _given(**keywords: Any) -> dict[str, Any]:
    """The keywords a call gave (not ``None``): forwarded so a call without them reaches the package with its own
    defaults."""

    return {name: value for name, value in keywords.items() if value is not None}


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
    """Autograd-enabled context-parallel packed width-4 depthwise causal convolution + SiLU of this rank's fragments,
    non-deterministic backward.

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
        uses).
    recompute_handle : RecomputeHandleKcp, optional
        Keyword-only, default ``None``.  A handle (any object with an ``entries`` dict) makes the checkpoint
        recompute of this call run without the forward halo exchange, under the rules of
        :func:`flashinfer.cake_causal_conv1d_det_train_kcp.cake_causal_conv1d_det_kcp` (one handle per call and
        micro-step, created outside the outermost checkpoint region; every rank passes one or none).
    cooperative : bool, optional
        Keyword-only, default ``None``.  The main backward is always one ordinary launch: ``None`` and ``False``
        mean the same and ``True`` raises ``ValueError`` (the deterministic entry point's cooperative form is
        ``cake_causal_conv1d_det_kcp(..., cooperative=True)``).
    deterministic : bool
        Keyword-only, default ``True``, which raises: pass ``False``.  Any value but ``False`` raises
        ``ValueError`` naming ``cake_causal_conv1d_det_kcp``.  With ``False``: the one-launch backward with an
        unordered ``dweight`` partial (``y`` / ``dx`` bitwise the deterministic ones).
    bwd_reserve_sms : int, optional
        Keyword-only, default ``None`` (the package's ``RESERVE_DEFAULT``, 0).  The SMs the main backward leaves to
        kernels of other streams (issue #8365): ``K >= 1`` launches its single-wave grid ``K x ctas_per_sm`` CTAs
        short, so a kernel of another stream that holds up to ``K`` SMs (for example a communication kernel) does
        not push the top CTAs into a second wave; the unloaded backward gives up the work of those ``K`` SMs.
        ``y``, ``dx`` and the exchanges are unchanged and ``dweight`` stays within the same budget.  With ``K = 0``
        (or ``None``) the backward already launches one SM's CTAs short wherever the last channel-tile rank of the
        full grid holds at least ``ctas_per_sm`` CTAs, so ``K = 1`` leaves the grid unchanged at those widths (for
        example ``C`` = 10240, 6144 and 5120 on a 152-SM GB200 / GB300, ``C`` = 10240, 6144 and 3072 on a 148-SM
        B200) and launches one SM's CTAs fewer at every other width.  Validated by the package when given: a
        ``bool`` or a non-``int`` raises ``TypeError``, a negative value or one not below the device's SM count
        ``ValueError``.  On sm_107a ``K >= 1`` raises ``ValueError``: there the reserved grid costs up to 1.055x the
        shipped grid unloaded (``C`` = 1024, 62585 tokens), above this opt-in's 1.040 limit.  The deterministic
        entry point's form beside a kernel of another stream is ``cooperative=False``.  Each rank may choose its own
        value (no collective depends on it).

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x_local``'s shape.  The backward returns BF16 ``dx`` and the rank-local FP32 ``dweight``
        partial.
    """

    _nondeterministic_only(
        deterministic, "cake_causal_conv1d_kcp", "cake_causal_conv1d_det_kcp"
    )
    from .cake_causal_conv1d_train_backend import (  # noqa: PLC0415
        kcp as _kcp,
    )

    return _kcp.cake_causal_conv1d_kcp(
        x_local,
        weight,
        ctx,
        recompute_handle=recompute_handle,
        deterministic=False,
        **_given(cooperative=cooperative, bwd_reserve_sms=bwd_reserve_sms),
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
    with one halo exchange per direction, non-deterministic backward.

    Outputs and ``dx`` are bitwise ``tuple(cake_causal_conv1d_kcp(x, w, ctx, deterministic=False) for x, w in
    zip(xs, weights))``; each ``dweight`` is that tensor's own unordered FP32 partial.

    Parameters
    ----------
    xs : sequence of torch.Tensor
        ``n >= 1`` BF16 ``[S_local, C_i]`` or ``[1, S_local, C_i]`` contiguous packed activations of this rank's
        fragments; each ``C_i`` is a multiple of 8 (they may differ).
    weights : sequence of torch.Tensor
        FP32 ``[C_i, 4]`` contiguous taps, one per tensor.
    ctx : ContextDetKcp
        As in :func:`cake_causal_conv1d_kcp`.  Every rank of the group calls this function once per layer, in the
        same order, with the same tensor count and channel counts.
    recompute_handle : RecomputeHandleKcp, optional
        One handle for the whole call (its ``"conv"`` entry holds the ``n`` history pools), under the rules of
        :func:`cake_causal_conv1d_kcp`.
    cooperative : bool, optional
        As in :func:`cake_causal_conv1d_kcp`.
    deterministic : bool
        As in :func:`cake_causal_conv1d_kcp`: only ``False`` runs; any other value raises ``ValueError`` naming
        ``cake_causal_conv1d_det_multi_kcp``.
    bwd_reserve_sms : int, optional
        As in :func:`cake_causal_conv1d_kcp`, one value for every tensor of the call.

    Returns
    -------
    tuple of torch.Tensor
        The ``n`` BF16 outputs, each of its input's shape.
    """

    _nondeterministic_only(
        deterministic,
        "cake_causal_conv1d_multi_kcp",
        "cake_causal_conv1d_det_multi_kcp",
    )
    from .cake_causal_conv1d_train_backend import (  # noqa: PLC0415
        kcp as _kcp,
    )

    return _kcp.cake_causal_conv1d_multi_kcp(
        xs,
        weights,
        ctx,
        recompute_handle=recompute_handle,
        deterministic=False,
        **_given(cooperative=cooperative, bwd_reserve_sms=bwd_reserve_sms),
    )


# ``RecomputeHandleKcp`` (the non-deterministic KCP package's recompute handle class) resolves on first access, so
# importing this module imports neither the non-deterministic package
# (``flashinfer.experimental.cake.cake_causal_conv1d_train_backend``) nor a KCP host module (torch.distributed, the
# KCP loader).
def __getattr__(name: str) -> Any:
    if name == "RecomputeHandleKcp":
        from .cake_causal_conv1d_train_backend.kcp import (  # noqa: PLC0415
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
