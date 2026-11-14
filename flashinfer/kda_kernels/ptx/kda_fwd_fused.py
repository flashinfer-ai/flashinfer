# Copyright 2026 Co-developed by KDA (Kernel Design Agents)
# Copyright (c) 2026 KDA Team
# SPDX-License-Identifier: MIT
# See LICENSE.kda-for-kda.txt for the full license.

"""Independent static-PTX KDA-forward entry point."""

from __future__ import annotations

import torch

from .scheduler import prepare_fwd_pieces


class KDAForwardLaunch:
    def __init__(
        self,
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        *,
        out,
        lower_bound=-5.0,
        initial_state=None,
        final_state=None,
        cu_seqlens=None,
        allow_approximate_split=False,
        use_expected_norm=False,
        offsets=None,
        _prepare=prepare_fwd_pieces,
    ):
        """Normalize gate-bias shape and prepare the selected exact scheduler."""
        if dt_bias.ndim == 1:
            dt_bias = dt_bias.view(q.shape[2], q.shape[3])
        self.out = out
        self.final_state = final_state
        self._fused = _prepare(
            q,
            k,
            v,
            g,
            beta,
            scale,
            out,
            A_log,
            dt_bias,
            lower_bound,
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
            allow_approximate_split=allow_approximate_split,
            use_expected_norm=use_expected_norm,
            offsets=offsets,
        )

    @property
    def schedule(self):
        """Describe the recurrent schedule selected for the prepared launch."""
        return self._fused.schedule

    @property
    def state_input_mode(self):
        """Describe the prepared plan's recurrent-state precision and exchange."""
        return self._fused.state_input_mode

    @property
    def normalization_mode(self):
        """Describe the query/key normalization used by the prepared plan."""
        return self._fused.normalization_mode

    def launch(self, g=None):
        """Run the prepared plan and return its bound output and final state."""
        self._fused.launch(g=g)
        return self.out, self.final_state


def prepare_kda_forward(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    scale,
    *,
    out=None,
    lower_bound=-5.0,
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
    allow_approximate_split=False,
    use_expected_norm=False,
    offsets=None,
):
    """Plan one launch; ``offsets`` are the host-known packed boundaries."""
    if dt_bias.ndim == 1:
        dt_bias = dt_bias.view(q.shape[2], q.shape[3])
    if out is None:
        out = torch.empty_like(q)
    return KDAForwardLaunch(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        out=out,
        lower_bound=lower_bound,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        allow_approximate_split=allow_approximate_split,
        use_expected_norm=use_expected_norm,
        offsets=offsets,
    )


def kda_forward(*args, **kwargs):
    """Prepare and immediately execute one KDA forward invocation."""
    return prepare_kda_forward(*args, **kwargs).launch()


__all__ = ["KDAForwardLaunch", "kda_forward", "prepare_kda_forward"]
