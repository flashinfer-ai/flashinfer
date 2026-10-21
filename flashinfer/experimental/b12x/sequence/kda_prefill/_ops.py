"""Opaque mutation boundary for the prepared CuTe KDA prefill pipeline."""

import torch

from b12x.preparation.types import plan_from_handle, require_prepared

_INPUTS = (
    "q",
    "k",
    "v",
    "raw_g",
    "raw_beta",
    "A_log",
    "dt_bias",
    "cu_seqlens",
    "initial_state_indices",
    "final_state_indices",
    "checkpoint_state_indices",
    "checkpoint_offsets",
    "num_seqs",
    "num_tokens",
)


@torch.library.custom_op(
    "b12x::kda_prefill",
    mutates_args=("scratch", "recurrent_state", "output"),
)
def _run(
    inputs: list[torch.Tensor],
    scratch: torch.Tensor,
    recurrent_state: torch.Tensor,
    output: torch.Tensor,
    plan_handle: int,
    lower_bound: float,
    scale: float | None,
    eps: float,
    max_live_tokens: int | None,
    max_live_seqs: int | None,
) -> None:
    plan = plan_from_handle(plan_handle)
    state = require_prepared(plan, "sequence.kda_prefill", output.device)
    binding = state.bind(
        _plan=plan,
        scratch=scratch,
        recurrent_state=recurrent_state,
        output=output,
        **dict(zip(_INPUTS, inputs, strict=True)),
    )
    state.run(
        binding,
        lower_bound=lower_bound,
        scale=scale,
        eps=eps,
        max_live_tokens=max_live_tokens,
        max_live_seqs=max_live_seqs,
    )


@_run.register_fake
def _fake_run(
    inputs,
    scratch,
    recurrent_state,
    output,
    plan_handle,
    lower_bound,
    scale,
    eps,
    max_live_tokens,
    max_live_seqs,
):
    pass


def run(binding, *, lower_bound, scale, eps, max_live_tokens, max_live_seqs):
    _run(
        [getattr(binding, name) for name in _INPUTS],
        binding.scratch,
        binding.recurrent_state,
        binding.output,
        binding.plan.handle,
        lower_bound,
        scale,
        eps,
        max_live_tokens,
        max_live_seqs,
    )
    return binding.output
