"""GDN raw-v-cache spec-decode VERIFY kernel (no state flush) — #4081-style sibling.

Verify-only entry point over the raw-v-cache ring: computes the T draft-token
outputs from S0 evolved through the P = hist_len[b] committed ring rows + the T
drafts, appends the drafts to the ring at [P, P+T), and NEVER touches the state
pool (the serving loop rewinds speculation by setting hist_len = P + accepted).

Implemented as the fused kernel (gdn_decode_bf16_wy_vcache_flush) launched with
the never-flush flush_min sentinel: every request takes the verify path, the
Phase-5 fold stores are predicated off for all CTAs, and hist_len is never
reset. Ring/argument contract is identical to the flush wrapper (minus the
flush knobs). NOTE: the fused kernel still executes the Phase-5 compute tail
(stores predicated off) — a dedicated verify compile that elides Phase 5 is a
follow-up optimization; use the flush kernel for flush iterations either way.
"""

from typing import Optional

import torch

from .gdn_decode_bf16_wy_vcache_flush import (  # noqa: F401
    K_DIM,
    RING_MASK,
    RING_SLOTS,
    V_DIM_C,
    W_RING_C,
    gated_delta_rule_mtp_vcache_flush as _flush_call,
)


def gated_delta_rule_mtp_vcache(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    initial_state_source: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    a_cache: Optional[torch.Tensor] = None,
    b_cache: Optional[torch.Tensor] = None,
    hist_len: Optional[torch.Tensor] = None,
    cache_base: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GDN decode output + raw-v-ring append (verify path, no state write).

    Same contract as ``gated_delta_rule_mtp_vcache_flush`` minus the flush
    knobs. Returns ``output`` [B, T, HV, V] bf16.
    """
    assert q is not None
    T = q.shape[1]
    W_RING = W_RING_C
    return _flush_call(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        k_cache=k_cache,
        v_cache=v_cache,
        a_cache=a_cache,
        b_cache=b_cache,
        hist_len=hist_len,
        cache_base=cache_base,
        flush_min=W_RING - T + 1,  # never-flush sentinel: pure verify + append
        restart_hist_on_flush=False,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        scale=scale,
        output=output,
    )


__all__ = ["gated_delta_rule_mtp_vcache", "K_DIM", "V_DIM_C", "W_RING_C", "RING_SLOTS", "RING_MASK"]
