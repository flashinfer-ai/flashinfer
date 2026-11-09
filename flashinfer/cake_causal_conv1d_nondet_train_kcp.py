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
non-deterministic package: thin stable entry points over the experimental backend module
``flashinfer.experimental.cake.cake_causal_conv1d_nondet_train.kcp`` (imported on first use).  The deterministic
KCP package is :mod:`flashinfer.cake_causal_conv1d_det_train_kcp` (``cake_causal_conv1d_det_kcp``,
``cake_causal_conv1d_det_kcp_multi``, ``CakeCausalConv1dDetKcpRecomputeHandle``); this module never runs it.

Numerical mode (the same rule as every non-deterministic training package): the module name chooses the package,
and no entry point takes a mode keyword.  "torch deterministic mode" is ``torch.use_deterministic_algorithms(True)``,
with or without ``warn_only=True``::

    package                                        torch deterministic mode off   torch deterministic mode on
    the non-deterministic package                  non-deterministic              the forward runs; the backward raises
      (flashinfer.cake_causal_conv1d_nondet_                                        RuntimeError before any exchange
      train_kcp, this module)                                                       (warn_only=True: a UserWarning, then
                                                                                    it runs); it never falls back
    the deterministic package                      deterministic                  deterministic; no warning
      (flashinfer.cake_causal_conv1d_det_
      train_kcp)

The backward reads torch's deterministic setting when it runs (the autograd backward included).  Every rank of
``ctx.group`` runs the same entry points (and the same torch deterministic setting) for a call.

Contract: the context of the KDA step (``CakeKDADetKcpContext`` from
``flashinfer.cake_kda_det_train_kcp.cake_kda_det_train_kcp_build_context``), FP32 taps and no bias, the halo exchanges (one
``all_to_all_single`` per direction, with recompute handles for communication-free checkpoint recomputes, #7818),
the recompute-handle rules and the multi-tensor calls are the deterministic KCP package's.  The backward overlaps
the halo exchange: its main program runs before the exchange is waited (no cooperative attribute, no grid barrier;
its CTAs claim their tokens at run time, so a kernel resident on another stream slows it about in proportion to the
SMs it holds) and adds the FP32 ``dweight`` runs into zeroed accumulators with unordered global reductions; a tail-row
program then writes ``dx`` of the fragments' last rows from the received rows.  ``y`` and ``dx`` are bitwise the
deterministic KCP entry point's;
the rank-local FP32 ``dweight`` partial stays within the deterministic error budget but is not bitwise reproducible
run to run (its cross-rank reduction remains the framework's).  The package is chosen by name, so a checkpoint
recompute (``torch.utils.checkpoint``, reentrant or not) runs the same package.

Name map (the deterministic package -> this module): ``cake_causal_conv1d_det_kcp`` ->
:func:`cake_causal_conv1d_nondet_kcp`, ``cake_causal_conv1d_det_kcp_multi`` ->
:func:`cake_causal_conv1d_nondet_kcp_multi`, ``CakeCausalConv1dDetKcpRecomputeHandle`` -> ``CakeCausalConv1dNondetKcpRecomputeHandle``
(any object with an ``entries`` dict serves either package)::

    from flashinfer.cake_causal_conv1d_nondet_train_kcp import (
        cake_causal_conv1d_nondet_kcp,
        cake_causal_conv1d_nondet_kcp_multi,
    )

    y_local = cake_causal_conv1d_nondet_kcp(x_local, weight, ctx)  # overlapped backward
    q, k, v = cake_causal_conv1d_nondet_kcp_multi((xq, xk, xv), (wq, wk, wv), ctx)

The backward beside a kernel of another stream (FlashInfer issue #8365): the overlapped backward needs no setting
(its CTAs claim their tokens at run time); ``bwd_reserve_sms=K`` (SM100 / SM103; ``ValueError`` on SM107) runs the
one-program backward after the halo exchange instead, ``K`` SMs short of its single wave::

    y_local = cake_causal_conv1d_nondet_kcp(x_local, weight, ctx, bwd_reserve_sms=1)
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from .api_logging import flashinfer_api


def _given(**keywords: Any) -> dict[str, Any]:
    """The keywords a call gave (not ``None``): forwarded so a call without them reaches the package with its own
    defaults."""

    return {name: value for name, value in keywords.items() if value is not None}


@flashinfer_api
def cake_causal_conv1d_nondet_kcp(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    *,
    recompute_handle: Optional[Any] = None,
    cooperative: Optional[bool] = None,
    bwd_reserve_sms: Optional[int] = None,
) -> torch.Tensor:
    """Autograd-enabled context-parallel packed width-4 depthwise causal convolution + SiLU of this rank's fragments,
    non-deterministic backward.

    The first tokens of a fragment that continues a document read the history rows of the preceding fragment
    (exchanged through the context's halo collective) instead of zeros.  FP32 taps, no bias.  The backward overlaps
    the halo exchange and returns an unordered ``dweight`` partial (``y`` / ``dx`` bitwise the deterministic ones).  When torch's
    deterministic mode is on as the backward runs, the backward raises a ``RuntimeError`` before any exchange (a
    ``UserWarning``, then it runs, with ``warn_only=True``).

    Parameters
    ----------
    x_local : torch.Tensor
        BF16 ``[S_local, C]`` or ``[1, S_local, C]`` contiguous packed activations of this rank's fragments, in the
        partition's order; ``C`` is a multiple of 8.
    weight : torch.Tensor
        FP32 ``[C, 4]`` contiguous taps.
    ctx : CakeKDADetKcpContext
        From ``flashinfer.cake_kda_det_train_kcp.cake_kda_det_train_kcp_build_context`` (the same context the KDA step of the rank
        uses).
    recompute_handle : CakeCausalConv1dNondetKcpRecomputeHandle, optional
        Keyword-only, default ``None``.  A handle (any object with an ``entries`` dict) makes the checkpoint
        recompute of this call run without the forward halo exchange, under the rules of
        :func:`flashinfer.cake_causal_conv1d_det_train_kcp.cake_causal_conv1d_det_kcp` (one handle per call and
        micro-step, created outside the outermost checkpoint region; every rank passes one or none).
    cooperative : bool, optional
        Keyword-only, default ``None``.  The backward launches no cooperative grid: ``None`` and ``False``
        mean the same and ``True`` raises ``ValueError`` (the cooperative form belongs to the deterministic KCP
        package: ``cake_causal_conv1d_det_kcp(..., cooperative=True)``).
    bwd_reserve_sms : int, optional
        Keyword-only, default ``None`` (the package's ``RESERVE_DEFAULT``, 0).  The SMs the one-program backward
        leaves to kernels of other streams (issue #8365): ``K >= 1`` runs that backward after the halo exchange
        instead of the overlapped one and launches its single-wave grid ``K x ctas_per_sm`` CTAs short, so a kernel of another stream that holds up to ``K`` SMs (for example a communication kernel) does
        not push the top CTAs into a second wave; the unloaded backward gives up the work of those ``K`` SMs.
        ``y``, ``dx`` and the exchanges are unchanged and ``dweight`` stays within the same budget.  With ``K = 0``
        (or ``None``) the backward already launches one SM's CTAs short wherever the last channel-tile rank of the
        full grid holds at least ``ctas_per_sm`` CTAs, so ``K = 1`` leaves the grid unchanged at those widths (for
        example ``C`` = 10240, 6144 and 5120 on a 152-SM GB200 / GB300, ``C`` = 10240, 6144 and 3072 on a 148-SM
        B200) and launches one SM's CTAs fewer at every other width.  Validated by the package when given: a
        ``bool`` or a non-``int`` raises ``TypeError``, a negative value or one not below the device's SM count
        ``ValueError``.  On sm_107a ``K >= 1`` raises ``ValueError``: there the reserved grid costs up to 1.055x the
        shipped grid unloaded (``C`` = 1024, 62585 tokens), above this opt-in's 1.040 limit.  The deterministic
        KCP package's form beside a kernel of another stream is ``cooperative=False``.  Each rank may choose its
        own value (no collective depends on it).

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x_local``'s shape.  The backward returns BF16 ``dx`` and the rank-local FP32 ``dweight``
        partial.
    """

    from .experimental.cake.cake_causal_conv1d_nondet_train import (  # noqa: PLC0415
        kcp as _kcp,
    )

    return _kcp.cake_causal_conv1d_nondet_kcp(
        x_local,
        weight,
        ctx,
        recompute_handle=recompute_handle,
        **_given(cooperative=cooperative, bwd_reserve_sms=bwd_reserve_sms),
    )


@flashinfer_api
def cake_causal_conv1d_nondet_kcp_multi(
    xs: Sequence[torch.Tensor],
    weights: Sequence[torch.Tensor],
    ctx: Any,
    *,
    recompute_handle: Optional[Any] = None,
    cooperative: Optional[bool] = None,
    bwd_reserve_sms: Optional[int] = None,
) -> tuple[torch.Tensor, ...]:
    """Autograd-enabled context-parallel convolutions of several tensors of one layer (for example its q, k and v)
    with one halo exchange per direction, non-deterministic backward.

    Outputs and ``dx`` are bitwise ``tuple(cake_causal_conv1d_nondet_kcp(x, w, ctx) for x, w in zip(xs,
    weights))``; each ``dweight`` is that tensor's own unordered FP32 partial.  Under torch's deterministic mode the
    backward alerts like the single call's.

    Parameters
    ----------
    xs : sequence of torch.Tensor
        ``n >= 1`` BF16 ``[S_local, C_i]`` or ``[1, S_local, C_i]`` contiguous packed activations of this rank's
        fragments; each ``C_i`` is a multiple of 8 (they may differ).
    weights : sequence of torch.Tensor
        FP32 ``[C_i, 4]`` contiguous taps, one per tensor.
    ctx : CakeKDADetKcpContext
        As in :func:`cake_causal_conv1d_nondet_kcp`.  Every rank of the group calls this function once per layer, in
        the same order, with the same tensor count and channel counts.
    recompute_handle : CakeCausalConv1dNondetKcpRecomputeHandle, optional
        One handle for the whole call (its ``"conv"`` entry holds the ``n`` history pools), under the rules of
        :func:`cake_causal_conv1d_nondet_kcp`.
    cooperative : bool, optional
        As in :func:`cake_causal_conv1d_nondet_kcp`.
    bwd_reserve_sms : int, optional
        As in :func:`cake_causal_conv1d_nondet_kcp`, one value for every tensor of the call.

    Returns
    -------
    tuple of torch.Tensor
        The ``n`` BF16 outputs, each of its input's shape.
    """

    from .experimental.cake.cake_causal_conv1d_nondet_train import (  # noqa: PLC0415
        kcp as _kcp,
    )

    return _kcp.cake_causal_conv1d_nondet_kcp_multi(
        xs,
        weights,
        ctx,
        recompute_handle=recompute_handle,
        **_given(cooperative=cooperative, bwd_reserve_sms=bwd_reserve_sms),
    )


# ``CakeCausalConv1dNondetKcpRecomputeHandle`` (the non-deterministic KCP package's recompute handle class) resolves
# on first access, so importing this module imports neither the non-deterministic backend package
# (``flashinfer.experimental.cake.cake_causal_conv1d_nondet_train``) nor a KCP host module (torch.distributed, the KCP
# loader).
def __getattr__(name: str) -> Any:
    if name == "CakeCausalConv1dNondetKcpRecomputeHandle":
        from .experimental.cake.cake_causal_conv1d_nondet_train.kcp import (  # noqa: PLC0415
            CakeCausalConv1dNondetKcpRecomputeHandle,
        )

        return CakeCausalConv1dNondetKcpRecomputeHandle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"CakeCausalConv1dNondetKcpRecomputeHandle"})


__all__ = [
    "CakeCausalConv1dNondetKcpRecomputeHandle",  # noqa: F822 - resolved by __getattr__
    "cake_causal_conv1d_nondet_kcp",
    "cake_causal_conv1d_nondet_kcp_multi",
]
