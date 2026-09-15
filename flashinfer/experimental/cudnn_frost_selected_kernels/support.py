"""Cheap auto-admission policy; no artifact loading, JIT, or device work."""

import torch

from ...fused_moe.api import (
    GELU,
    GeGLU,
    GeGLUTanh,
    Identity,
    QuantFormat,
    ReLU,
    ReLU2,
    RoutingInputMode,
    SiLU,
    SiTU,
    SwiGLU,
    SwiGLUStep,
)

_SHORTLIST_TOP_K = {
    (12, 7168, 3072): (1, 2, 4),
    (8, 4096, 14336): (2,),
    (64, 2048, 1408): (6,),
}


def large_bf16_moe(config, act, arch):
    x = act.hidden_states_q
    compatible = (
        arch == 107
        and config.quant.pair == (QuantFormat.BF16, QuantFormat.BF16)
        and config.quant.output == QuantFormat.BF16
        and act.routing_input_mode == RoutingInputMode.PackedPrecomputed
        and x.ndim == 2
        and x.dtype == torch.bfloat16
    )
    if not compatible:
        return False
    geometry = (
        config.routing.num_experts,
        x.shape[1],
        config.experts.intermediate_size,
    )
    if isinstance(
        config.activation,
        (SwiGLU, GeGLU, GeGLUTanh, SwiGLUStep, SiTU, GELU, Identity, ReLU, ReLU2, SiLU),
    ):
        return (
            config.routing.top_k in _SHORTLIST_TOP_K.get(geometry, ())
            and 0 < act.num_tokens <= 12288
        )
    return False


is_eligible = large_bf16_moe


def create_runner(config, device):
    """Load the implementation only after cheap automatic-admission checks."""
    from .moe import automatic_candidate

    return automatic_candidate(config, device)
