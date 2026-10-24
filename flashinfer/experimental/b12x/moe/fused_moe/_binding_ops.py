"""Compiler-visible prepared MoE execution over caller-owned storage."""

import torch

from b12x.preparation.types import plan_from_handle, require_prepared


@torch.library.custom_op(
    "b12x::prepared_moe_out",
    mutates_args=("output", "scratch", "activation_amax"),
)
def prepared_moe_out(
    a: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    scratch: torch.Tensor,
    plan_handle: int,
    input_scales_static: bool,
    fast_math: bool | None,
    activation_amax: torch.Tensor | None,
    layer_idx: int | None,
    route_expert_map: torch.Tensor | None,
    output_expert_map: torch.Tensor | None,
    scales_expanded: bool,
    a4_prefill: bool | None,
) -> None:
    state = require_prepared(plan_from_handle(plan_handle), "moe.decode", a.device)
    binding = state.bind(
        a=a,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        scratch=scratch,
        input_scales_static=input_scales_static,
        fast_math=fast_math,
        activation_amax=activation_amax,
        layer_idx=layer_idx,
        route_expert_map=route_expert_map,
        output_expert_map=output_expert_map,
        scales_expanded=scales_expanded,
        a4_prefill=a4_prefill,
    )
    state.run(binding)


@prepared_moe_out.register_fake
def _fake(
    a,
    topk_weights,
    topk_ids,
    output,
    scratch,
    plan_handle,
    input_scales_static,
    fast_math,
    activation_amax,
    layer_idx,
    route_expert_map,
    output_expert_map,
    scales_expanded,
    a4_prefill,
) -> None:
    pass
