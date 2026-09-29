# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Cheap MXFP8 × MXFP4 auto-admission; source and shortlist checks live in the runner."""

import torch

from ....fused_moe.api import QuantFormat, RoutingInputMode
from ..activations import activation_name
from ..support import shortlisted_moe_geometry


def is_eligible(config, act, arch):
    x = act.hidden_states_q
    if not (
        arch == 107
        and config.quant.pair == (QuantFormat.MXFP4, QuantFormat.MXFP8)
        and config.quant.output == QuantFormat.BF16
        and act.routing_input_mode == RoutingInputMode.PackedPrecomputed
        and x.ndim == 2
        and x.dtype == torch.float8_e4m3fn
    ):
        return False
    try:
        activation_name(config.activation)
    except NotImplementedError:
        return False
    return shortlisted_moe_geometry(config, act)


def create_runner(config, device):
    from .moe import automatic_candidate

    return automatic_candidate(config, device)
