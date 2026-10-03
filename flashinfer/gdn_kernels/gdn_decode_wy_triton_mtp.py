"""
Triton GDN WY MTP — dispatching public entry point.

================================================================================
PROBLEM IT SOLVES
================================================================================
Full MTP speculation cycle for GDN in one public call. Given:
  * K accepted draft tokens from the previous cycle (per-batch count via
    `num_accepted[B]`) — these are replayed to advance state h0 → h_K
  * T new draft tokens to generate outputs for, starting from h_K
...compute: T outputs (no outputs for the K accepted tokens — they were
already verified) and write h_K back to the state pool (NOT h_{K+T} —
draft-token updates aren't committed until accepted next cycle).

Public API: `gated_delta_rule_mtp_auto(k_accepted, v_accepted, ..., q_new, ...)`.

K=0 (all drafts rejected) is handled by `num_accepted` passing through to
the underlying kernel — no separate fast path needed.

================================================================================
WHAT THIS FILE DOES
================================================================================
Thin Python-level router. No Triton code. Dispatches based on batch size
to one of two implementations:
  * fused — single kernel (`_mtp_fused.py`)
  * split — two kernels + PDL (`_mtp_split.py`)
"""

from typing import Optional

import torch

from .gdn_decode_wy_triton_mtp_fused import gated_delta_rule_mtp_fused
from .gdn_decode_wy_triton_mtp_split import gated_delta_rule_mtp_split


# Below this batch size we prefer the fused kernel (single launch beats the
# state-step/output-split).  Empirically (qwen3.5, T=8, HV=64, B200) the
# crossover is ~BS=8.
_FUSED_BS_THRESHOLD = 8


def gated_delta_rule_mtp_auto(
    k_accepted: torch.Tensor,
    v_accepted: torch.Tensor,
    a_accepted: torch.Tensor,
    b_accepted: torch.Tensor,
    num_accepted: torch.Tensor,
    q_new: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    a_new: torch.Tensor,
    b_new: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: Optional[torch.Tensor] = None,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    output: Optional[torch.Tensor] = None,
    force_impl: Optional[str] = None,  # "fused" | "split" | None (auto)
    launch_with_pdl: bool = True,
) -> torch.Tensor:
    """Auto-dispatch between fused and split-kernel MTP.

    Args:
        force_impl: if "fused" or "split", bypass the auto heuristic.
        launch_with_pdl: only applies to the split implementation.
    """
    B = q_new.shape[0]

    kw = dict(
        k_accepted=k_accepted,
        v_accepted=v_accepted,
        a_accepted=a_accepted,
        b_accepted=b_accepted,
        num_accepted=num_accepted,
        q_new=q_new,
        k_new=k_new,
        v_new=v_new,
        a_new=a_new,
        b_new=b_new,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        output=output,
    )

    impl = force_impl
    if impl is None:
        impl = "fused" if B <= _FUSED_BS_THRESHOLD else "split"

    if impl == "fused":
        return gated_delta_rule_mtp_fused(**kw)
    elif impl == "split":
        return gated_delta_rule_mtp_split(launch_with_pdl=launch_with_pdl, **kw)
    else:
        raise ValueError(f"force_impl must be 'fused' or 'split', got {impl!r}")
