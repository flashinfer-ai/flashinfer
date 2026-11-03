"""Heuristic planning for the tensor-based fused-MoE API."""

from __future__ import annotations

from dataclasses import dataclass

from b12x.preparation import detect_device
from . import _impl
from ._preparation import _codec_scalar, _control_snapshot
from ._tuning import MoeDecodeConfig, MoeDecodeQuery, TUNING


@dataclass(frozen=True, kw_only=True)
class Caps(_impl.TPMoEScratchCaps):
    decode_config: MoeDecodeConfig | None = None

    def __post_init__(self):
        if self.decode_config is None:
            weight = self.weight_plan
            if not isinstance(weight, _impl.MoEWeightPreparationPlan):
                raise TypeError("weight_plan must be a MoEWeightPreparationPlan")
            quant_mode = _impl._normalize_quant_mode_for_source(
                self.quant_mode,
                weight.source_format,
            )
            limit, alpha, beta = _impl._normalize_swiglu_params(
                weight.activation,
                self.swiglu_limit,
                self.swiglu_alpha,
                self.swiglu_beta,
            )
            query = MoeDecodeQuery(
                quant_mode=quant_mode,
                quant_modes=tuple(sorted(weight.quant_modes)),
                source_format=weight.source_format,
                activation=weight.activation,
                io_dtype=weight.io_dtype,
                num_experts=weight.num_experts,
                hidden_size=weight.hidden_size,
                intermediate_size=weight.intermediate_size,
                top_k=self.num_topk,
                num_tokens=self.max_tokens,
                routed_rows=self.max_tokens * self.num_topk,
                route_num_experts=self.route_num_experts,
                route_logits_dtype=None
                if self.route_logits_dtype is None
                else str(self.route_logits_dtype).removeprefix("torch."),
                apply_router_weight_on_input=self.apply_router_weight_on_input,
                collect_activation_amax=self.collect_activation_amax,
                deterministic_output=self.deterministic_output,
                swiglu_limit=_codec_scalar(limit),
                swiglu_alpha=_codec_scalar(alpha),
                swiglu_beta=_codec_scalar(beta),
                w13_layout=weight.w13_layout,
                weight_layouts=tuple(
                    sorted(layout.value for layout in weight.weight_layouts)
                ),
                w4a16_weight_layout=weight.w4a16_weight_layout,
                w4a16_scale_format=weight.w4a16_scale_format,
                w4a16_block_size_m=self.w4a16_block_size_m,
                fast_math=self.w4a16_fast_math,
                numerical_recipe=None,
                controls=_control_snapshot(),
            )
            configured = TUNING.configure(
                query, device=detect_device(self.device).identity, search=False
            )
            object.__setattr__(self, "decode_config", configured.default)
        super().__post_init__()


def plan_execution(
    *,
    num_tokens,
    num_topk,
    device,
    weight_plan,
    quant_mode,
    decode_config=None,
    **kwargs,
):
    caps = Caps(
        max_tokens=num_tokens,
        num_topk=num_topk,
        device=device,
        weight_plan=weight_plan,
        quant_mode=quant_mode,
        decode_config=decode_config,
        **kwargs,
    )
    return _impl.plan_tp_moe_execution(
        num_tokens=num_tokens,
        num_topk=num_topk,
        device=device,
        weight_plan=weight_plan,
        quant_mode=quant_mode,
        decode_config=caps.decode_config,
        **kwargs,
    )
