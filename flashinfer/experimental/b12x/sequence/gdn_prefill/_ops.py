"""Compiler-visible mutation boundary for prepared GDN prefill."""

import torch

from b12x.preparation.types import plan_from_handle, require_prepared


@torch.library.custom_op(
    "b12x::gdn_prefill",
    mutates_args=("scratch", "recurrent_state", "output"),
)
def gdn_prefill_out(
    scratch: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state_indices: torch.Tensor,
    final_state_indices: torch.Tensor,
    checkpoint_state_indices: torch.Tensor,
    checkpoint_offsets: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    output: torch.Tensor,
    plan_handle: int,
    scale: float,
    eps: float,
    max_live_tokens: int | None,
    max_live_seqs: int | None,
) -> None:
    plan = plan_from_handle(plan_handle)
    state = require_prepared(plan, "sequence.gdn_prefill", q.device)
    binding = state.bind(
        _plan=plan,
        scratch=scratch,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        recurrent_state=recurrent_state,
        cu_seqlens=cu_seqlens,
        initial_state_indices=initial_state_indices,
        final_state_indices=final_state_indices,
        checkpoint_state_indices=checkpoint_state_indices,
        checkpoint_offsets=checkpoint_offsets,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        output=output,
    )
    state.run(
        binding,
        scale=scale,
        eps=eps,
        max_live_tokens=max_live_tokens,
        max_live_seqs=max_live_seqs,
    )


@gdn_prefill_out.register_fake
def _fake(
    scratch,
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    recurrent_state,
    cu_seqlens,
    initial_state_indices,
    final_state_indices,
    checkpoint_state_indices,
    checkpoint_offsets,
    num_seqs,
    num_tokens,
    output,
    plan_handle,
    scale,
    eps,
    max_live_tokens,
    max_live_seqs,
) -> None:
    pass
