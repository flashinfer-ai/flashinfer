"""Validation for native BTX weight owners supplied by checkpoint adapters."""

from __future__ import annotations

import torch

from b12x.moe._shared.execution import (
    MoEWeightPreparationPlan,
    PreparedWeightLayout,
)
from b12x.moe._shared.kernels.activations import is_gated_moe_activation
from b12x.moe._shared.kernels.w4a16.prepare import PreparedW4A16MoeWeights

from ._impl import (
    B12XFP4ExpertWeights,
    _PreparedWeightRepresentation,
)


def adopt_prepared_btx_weights(
    *,
    plan: MoEWeightPreparationPlan,
    prepared: PreparedW4A16MoeWeights,
    a1_gscale: torch.Tensor | None = None,
    a2_gscale: torch.Tensor | None = None,
) -> B12XFP4ExpertWeights:
    """Validate and adopt an existing native BTX owner without copying it.

    Checkpoint adapters use this operation when source tensors already occupy
    the native Trellis layout. The supplied plan remains the runtime authority;
    every storage, geometry, codebook, bitrate, and transform property must
    match it. The returned expert object retains ``prepared`` as its sole
    weight owner.

    Only uniform-rate BTX extents are supported. Mixed-rate BTX checkpoints
    require the atom-container preparation path exposed by ``prepare_weights``.
    """

    if not isinstance(plan, MoEWeightPreparationPlan):
        raise TypeError("plan must be a MoEWeightPreparationPlan")
    if plan.source_format != "btx":
        raise ValueError("native BTX adoption requires source_format='btx'")
    if len(plan.quant_modes) != 1 or not plan.quant_modes <= frozenset(
        {"w4a16", "w4a8_mx"}
    ):
        raise ValueError(
            "native BTX adoption supports exactly one of quant_mode='w4a16' "
            "or quant_mode='w4a8_mx'"
        )
    if not isinstance(prepared, PreparedW4A16MoeWeights):
        raise TypeError(
            "prepared must be PreparedW4A16MoeWeights from the native BTX "
            "Trellis preparation API"
        )

    trellis = prepared.trellis
    if trellis is None:
        raise ValueError("prepared BTX weights must carry Trellis state")
    expected_geometry = (
        plan.hidden_size,
        plan.intermediate_size,
        plan.num_experts,
    )
    actual_geometry = (
        prepared.hidden_size,
        prepared.intermediate_size,
        prepared.num_experts,
    )
    if actual_geometry != expected_geometry:
        raise ValueError(
            "prepared BTX geometry does not match the weight plan: "
            f"prepared={actual_geometry}, plan={expected_geometry}"
        )
    if prepared.source_format != "btx" or prepared.weight_layout != "trellis_t256":
        raise ValueError(
            "prepared weights must use source_format='btx' and "
            "weight_layout='trellis_t256'"
        )
    if prepared.params_dtype != torch.float16:
        raise TypeError(
            "native BTX weights must use torch.float16 parameters, got "
            f"{prepared.params_dtype}"
        )
    if prepared.is_gated != is_gated_moe_activation(plan.activation):
        raise ValueError("prepared BTX gated layout does not match the plan activation")
    if (
        trellis.codebook != plan.trellis_codebook
        or trellis.bits != plan.trellis_bits
        or trellis.tile_config != plan.trellis_tile_config
    ):
        raise ValueError(
            "prepared BTX codebook, bitrate, or tile configuration does not "
            "match the weight plan"
        )
    if (
        (plan.trellis_rate_granularity or "uniform") != "uniform"
        or plan.trellis_pair_kinds is not None
        or trellis.fc1_pair_kind is not None
        or trellis.fc2_pair_kind is not None
        or trellis.fc1_pair_modes is not None
        or trellis.fc2_pair_modes is not None
    ):
        raise ValueError("native BTX adoption supports uniform-rate extents only")
    if trellis.coupled_hadamard != plan.coupled_hadamard:
        raise ValueError(
            "prepared BTX coupled-Hadamard state does not match the weight plan"
        )

    quant_mode = next(iter(plan.quant_modes))
    representation = _PreparedWeightRepresentation(
        quant_mode=quant_mode,
        layout=PreparedWeightLayout.TRELLIS_NATIVE,
        value=prepared,
    )
    input_scale = torch.ones((), dtype=torch.float32, device=prepared.w13.device)
    return B12XFP4ExpertWeights(
        plan=plan,
        a1_gscale=a1_gscale if a1_gscale is not None else input_scale,
        w1_fp4=prepared.w13,
        w1_blockscale=prepared.w13_scale,
        w1_alphas=prepared.w13_global_scale,
        a2_gscale=a2_gscale if a2_gscale is not None else input_scale,
        w2_fp4=prepared.w2,
        w2_blockscale=prepared.w2_scale,
        w2_alphas=prepared.w2_global_scale,
        representation=representation,
    )
