"""Compiler-visible mutations for prepared expert routing."""

import torch

from b12x.preparation.types import plan_from_handle, require_prepared


@torch.library.custom_op(
    "b12x::route_topk",
    mutates_args=("topk_logits", "topk_ids", "topk_weights"),
)
def route_topk_out(
    router_logits: torch.Tensor,
    topk_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    plan_handle: int,
    correction_bias: torch.Tensor | None,
    image_correction_bias: torch.Tensor | None,
    image_mask: torch.Tensor | None,
    num_valid_tokens: torch.Tensor | None,
) -> None:
    state = require_prepared(
        plan_from_handle(plan_handle), "moe.route_topk", router_logits.device
    )
    state.run(
        router_logits,
        topk_logits,
        topk_ids,
        topk_weights,
        correction_bias=correction_bias,
        image_correction_bias=image_correction_bias,
        image_mask=image_mask,
        num_valid_tokens=num_valid_tokens,
    )


@route_topk_out.register_fake
def _fake(
    router_logits,
    topk_logits,
    topk_ids,
    topk_weights,
    plan_handle,
    correction_bias,
    image_correction_bias,
    image_mask,
    num_valid_tokens,
) -> None:
    pass
