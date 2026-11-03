# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static per-tensor E4M3 scales with the shared routed FMA schedule."""

import torch


def pack_inputs(act, weights, config, device):
    from ...api import QuantFormat, RoutingInputMode

    x, ids, scores = act.hidden_states_q, act.topk_ids, act.topk_weights
    weight_only = config.quant.activation in (QuantFormat.BF16, QuantFormat.FP16)
    activation_dtype = (
        (
            torch.float16
            if config.quant.activation is QuantFormat.FP16
            else torch.bfloat16
        )
        if weight_only
        else torch.float8_e4m3fn
    )
    output_dtype = (
        torch.float16 if config.quant.output is QuantFormat.FP16 else torch.bfloat16
    )
    t, h = x.shape
    e, k = config.routing.num_experts, config.routing.top_k
    p = (config.experts.intermediate_size + 127) // 128 * 128
    branches = 2 if config.activation.is_gated else 1
    if act.hidden_states_scale is not None:
        raise ValueError(
            "Frost per-tensor FP8 folds the input scale into prepared weights"
        )
    if (
        x.dtype is not activation_dtype
        or x.device != device
        or not x.is_contiguous()
        or x.data_ptr() % 16
    ):
        raise ValueError(
            f"Frost per-tensor input must be aligned contiguous {activation_dtype} on the runner device"
        )
    view = weights.get_view("frost_low_latency")
    for name, shape, dtype in (
        ("fc1_expert_weights", (e, branches * p, h), torch.float8_e4m3fn),
        ("fc2_expert_weights", (e, h, p), torch.float8_e4m3fn),
        ("fc1_dequant_scale", (e,), torch.float32),
        ("fc2_act_quant_scale", (), torch.float32),
        ("fc2_dequant_scale", (e,), torch.float32),
    ):
        value = view[name]
        if (
            value.dtype != dtype
            or value.shape != shape
            or value.device != device
            or not value.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous {dtype} {shape} on {device}")
        if name.endswith("weights") and value.data_ptr() % 16:
            raise ValueError(f"{name} must be 16-byte aligned")
    packed = act.routing_input_mode == RoutingInputMode.PackedPrecomputed
    narrow = packed and config.finalize.do_finalize
    if packed and not narrow:
        scores = scores.to(torch.bfloat16)
    return [
        x,
        ids,
        scores,
        view["fc1_expert_weights"],
        view["fc2_expert_weights"],
        torch.empty((t * k, p), device=device, dtype=activation_dtype),
        torch.empty((t * k, h), device=device, dtype=output_dtype),
        torch.empty((t, h), device=device, dtype=output_dtype),
        narrow,
        view["fc1_dequant_scale"],
        view["fc2_act_quant_scale"].reshape(1),
        view["fc2_dequant_scale"],
    ]


def execute(inputs, config, tactic):
    from .fma.fp8_per_tensor import execute as run

    return run(inputs, config, tactic)
