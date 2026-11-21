# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Cheap BF16 input geometry checks for ordinary MoELayer candidates."""

import torch

from .....fused_moe.api import (
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

from ..support import SUPPORTED_ARCH_NUMBERS, moe_geometry
from ..activations import activation_name


def large_bf16_moe(config, act, arch):
    x = act.hidden_states_q
    compatible = (
        arch in SUPPORTED_ARCH_NUMBERS
        and config.quant.pair == (QuantFormat.BF16, QuantFormat.BF16)
        and config.quant.output == QuantFormat.BF16
        and act.routing_input_mode == RoutingInputMode.PackedPrecomputed
        and x.ndim == 2
        and x.dtype == torch.bfloat16
    )
    if not compatible:
        return False
    if isinstance(
        config.activation,
        (SwiGLU, GeGLU, GeGLUTanh, SwiGLUStep, SiTU, GELU, Identity, ReLU, ReLU2, SiLU),
    ):
        try:
            activation_name(config.activation)
        except NotImplementedError:
            return False
        # SM120 BF16 has no scale-factor layout; its kernels take whole
        # 32-element blocks (catalog contracts) and routing moves 8-element
        # packs, so e.g. gpt-oss's H = I = 2880 is admitted there.
        return moe_geometry(config, act, multiple=32 if arch == 120 else 128)
    return False


is_eligible = large_bf16_moe
