"""Prepared raw-input short convolution public API."""

from dataclasses import dataclass

import torch

from b12x._lib.gating import default_is_supported
from b12x.preparation import Plan
from b12x.preparation.types import plan_from_handle, require_prepared

from ._tuning import OPERANDS


@dataclass(frozen=True, kw_only=True)
class Caps:
    """Convolution geometry and fixed capacities for packed request buffers."""

    device: torch.device | str
    mode: str
    channels: int
    kernel_size: int
    max_tokens: int
    max_seqs: int
    max_state_slots: int
    dilation: int = 1
    max_speculative_tokens: int = 0


@dataclass(frozen=True)
class Binding:
    """Caller-owned inputs, persistent history, output, and scratch views."""

    plan: Plan
    scratch: torch.Tensor
    x: torch.Tensor
    conv_weight: torch.Tensor
    query_start_loc: torch.Tensor
    state_slot_ids: torch.Tensor
    state_is_fresh: torch.Tensor
    num_accepted_tokens: torch.Tensor
    request_is_prefill: torch.Tensor
    num_seqs: torch.Tensor
    num_tokens: torch.Tensor
    conv_state: torch.Tensor
    out: torch.Tensor
    gathered_state: torch.Tensor
    request_ids: torch.Tensor


def plan(caps: Caps, *, invocation=None, override=None) -> Plan:
    from ._preparation import make_plan

    return make_plan(caps, invocation=invocation, override=override)


def invocation_from_tensors(*, conv_state, scratch=None, **tensors):
    """Describe actual history strides and pointer alignments for preparation."""
    from ._preparation import invocation_from_tensors as describe

    return describe(conv_state=conv_state, scratch=scratch, **tensors)


def bind(
    plan: Plan,
    *,
    scratch,
    x,
    conv_weight,
    query_start_loc,
    state_slot_ids,
    state_is_fresh,
    num_accepted_tokens,
    num_seqs,
    num_tokens,
    conv_state,
    out,
    request_is_prefill=None,
) -> Binding:
    """Bind BF16 ``x/out [max_tokens, channels]`` and caller-owned history.

    Weights are BF16 ``[channels, kernel_size]``. Request metadata consists of
    int32 ``query_start_loc [max_seqs + 1]``, int64 ``state_slot_ids [max_seqs]``,
    bool ``state_is_fresh [max_seqs]``, int32 ``num_accepted_tokens [max_seqs]``,
    and int32 ``num_seqs/num_tokens [1]``. Mixed batches additionally require
    bool ``request_is_prefill [max_seqs]``. Only history may have a padded slot
    stride; every other operand must be contiguous. Mutated buffers must not
    overlap other operands. Binding does not allocate or read device values.
    """
    return require_prepared(plan, "sequence.short_conv", x.device).bind(
        plan=plan,
        scratch=scratch,
        x=x,
        conv_weight=conv_weight,
        query_start_loc=query_start_loc,
        state_slot_ids=state_slot_ids,
        state_is_fresh=state_is_fresh,
        num_accepted_tokens=num_accepted_tokens,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        conv_state=conv_state,
        out=out,
        request_is_prefill=request_is_prefill,
    )


@torch.library.custom_op(
    "b12x::short_conv",
    mutates_args=("conv_state", "out", "gathered_state", "request_ids"),
)
def _pipeline(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_slot_ids: torch.Tensor,
    state_is_fresh: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    request_is_prefill: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    conv_state: torch.Tensor,
    out: torch.Tensor,
    gathered_state: torch.Tensor,
    request_ids: torch.Tensor,
    plan_handle: int,
) -> None:
    require_prepared(
        plan_from_handle(plan_handle), "sequence.short_conv", x.device
    ).run_tensors(
        x,
        conv_weight,
        query_start_loc,
        state_slot_ids,
        state_is_fresh,
        num_accepted_tokens,
        request_is_prefill,
        num_seqs,
        num_tokens,
        conv_state,
        out,
        gathered_state,
        request_ids,
    )


@_pipeline.register_fake
def _pipeline_fake(
    x,
    conv_weight,
    query_start_loc,
    state_slot_ids,
    state_is_fresh,
    num_accepted_tokens,
    request_is_prefill,
    num_seqs,
    num_tokens,
    conv_state,
    out,
    gathered_state,
    request_ids,
    plan_handle,
) -> None:
    pass


def run(binding: Binding) -> torch.Tensor:
    """Execute retained launchers; write zero for padded rows and return ``out``."""
    if not isinstance(binding, Binding):
        raise TypeError("binding must be short_conv.Binding")
    _pipeline(*(getattr(binding, name) for name in OPERANDS), binding.plan.handle)
    return binding.out


def is_supported(device=None):
    return default_is_supported(device, requires=("triton",))


def clear_caches():
    """Compiled programs are owned by preparation sessions."""


__all__ = [
    "Caps",
    "Binding",
    "Plan",
    "plan",
    "invocation_from_tensors",
    "bind",
    "run",
    "is_supported",
    "clear_caches",
]
