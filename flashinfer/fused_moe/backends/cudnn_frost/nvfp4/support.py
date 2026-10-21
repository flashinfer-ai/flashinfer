# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Cheap NVFP4 input geometry checks for ordinary MoELayer candidates."""

import torch

from .....fused_moe.api import QuantFormat, RoutingInputMode
from ..activations import activation_name
from ..support import SUPPORTED_ARCH_NUMBERS, moe_geometry


def is_eligible(config, act, arch):
    x = act.hidden_states_q
    if not (
        arch in SUPPORTED_ARCH_NUMBERS
        and config.quant.pair == (QuantFormat.NVFP4, QuantFormat.NVFP4)
        and config.quant.output == QuantFormat.BF16
        and act.routing_input_mode == RoutingInputMode.PackedPrecomputed
        and x.ndim == 2
        and x.dtype == torch.uint8
    ):
        return False
    try:
        activation_name(config.activation)
    except NotImplementedError:
        return False
    return moe_geometry(config, act, hidden_size=2 * x.shape[1])
