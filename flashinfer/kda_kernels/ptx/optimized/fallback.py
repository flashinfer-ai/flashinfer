# Copyright 2026 Co-developed by KDA (Kernel Design Agents)
"""Independent exact recurrence for shapes without a retained PTX route."""

from __future__ import annotations

import torch
from .plan_cache import const_table


class RecurrentLaunch:
    def __init__(
        self,
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
        initial_state,
        final_state,
        cu_seqlens,
        ephemeral=False,
    ):
        heads = q.shape[2]
        tokens = q.numel() // (heads * q.shape[-1])

        if cu_seqlens is not None:
            cu = cu_seqlens.to(dtype=torch.int64)
        elif ephemeral:
            cu = const_table(
                ("cu_seqlens", tokens), lambda: [0, tokens], q.device, torch.int64
            )
        else:
            cu = torch.tensor([0, tokens], dtype=torch.int64, device=q.device)
        n_seqs = cu.numel() - 1
        q_flat = q.reshape(tokens, heads, 128)
        k_flat = k.reshape(tokens, heads, 128)
        v_flat = v.reshape(tokens, heads, 128)
        g_flat = g.reshape(tokens, heads, 128)
        beta_flat = beta.reshape(tokens, heads)
        self.args = {
            "q": q_flat,
            "q_tma": q_flat,
            "k": k_flat,
            "k_tma": k_flat,
            "v": v_flat,
            "v_tma": v_flat,
            "g": g_flat,
            "g_tma": g_flat,
            "beta": beta_flat,
            "beta_tma": beta_flat,
            "A_log": A_log,
            "dt_bias": dt_bias,
            "cu_seqlens": cu,
            "seq_order": (
                const_table(
                    ("seq_order", n_seqs), lambda: list(range(n_seqs)), q.device
                )
                if ephemeral
                else torch.arange(n_seqs, dtype=torch.int32, device=q.device)
            ),
            "initial_state": initial_state,
            "out": out.reshape(tokens, heads, 128),
            "out_tma": out.reshape(tokens, heads, 128),
            "final_state": final_state,
            "num_heads": heads,
            "use_initial_state": int(initial_state is not None),
            "store_final_state": int(final_state is not None),
            "scale": float(scale),
            "lower_bound": float(lower_bound),
        }
        self.schedule = "exact_recurrent_triton"

    def launch(self):
        from .recurrent_kernel import launch_recurrent

        args = self.args
        launch_recurrent(
            q=args["q"],
            k=args["k"],
            v=args["v"],
            g=args["g"],
            beta=args["beta"],
            a_log=args["A_log"],
            dt_bias=args["dt_bias"],
            out=args["out"],
            initial_state=args["initial_state"],
            final_state=args["final_state"],
            cu_seqlens=args["cu_seqlens"],
            scale=args["scale"],
            lower_bound=args["lower_bound"],
        )


def prepare_fwd(
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
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
    ephemeral=False,
):
    return RecurrentLaunch(
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
        initial_state,
        final_state,
        cu_seqlens,
        ephemeral,
    )


__all__ = ["prepare_fwd"]
