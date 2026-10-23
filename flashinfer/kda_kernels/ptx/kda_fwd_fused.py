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
        offsets=None,
    ):
        self.out = out
        self.final_state = final_state
        self._fused = prepare_fwd_pieces(
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
            offsets=offsets,
        )

    @property
    def schedule(self):
        return self._fused.schedule

    @property
    def state_input_mode(self):
        return self._fused.state_input_mode

    @property
    def normalization_mode(self):
        return self._fused.normalization_mode

    def launch(self, g=None):
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
        offsets=offsets,
    )


__all__ = ["KDAForwardLaunch", "prepare_kda_forward"]
