# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical packed four-bit weights with live scales and typed 16-bit bindings."""

import torch


def prepare_weights(
    w1, s1, a1, w2, s2, a2, activation, *, scale_block_size=16, integer_weights=False
):
    """Preserve FP4/INT4 payloads and their linear scales; pad I to 128."""
    from .fma.activation_config import lower_activation

    lower_activation(activation)
    scale_dtype = (
        torch.bfloat16
        if integer_weights
        else (torch.float8_e8m0fnu if scale_block_size == 32 else torch.float8_e4m3fn)
    )
    group = scale_block_size
    if w1.dtype is not torch.uint8 or w2.dtype is not torch.uint8:
        raise TypeError("W4A16 requires packed uint8 weights")
    if w1.ndim != 3 or w2.ndim != 3:
        raise ValueError("weights must be 3-D")
    e, h, packed_i = w2.shape
    i = packed_i * 2
    branches = 2 if activation.is_gated else 1
    if e < 1 or h < 128 or h % 128 or i < 32 or i % 32:
        raise ValueError("require positive E, H divisible by 128 and I divisible by 32")
    for value, shape, dtype in (
        (w1, (e, branches * i, h // 2), torch.uint8),
        (w2, (e, h, i // 2), torch.uint8),
        (s1, (e, branches * i, h // group), scale_dtype),
        (s2, (e, h, i // group), scale_dtype),
        (a2, (e,), torch.float32),
    ):
        if (
            value.dtype != dtype
            or value.shape != shape
            or not value.is_cuda
            or value.device != w1.device
        ):
            raise ValueError(f"expected CUDA {dtype} tensor {shape} on {w1.device}")
    if (
        a1.dtype is not torch.float32
        or a1.shape not in ((e,), (e, branches))
        or a1.device != w1.device
    ):
        raise ValueError(
            "FC1 global scales must be FP32 [E] or [E,branches] on the weights' device"
        )
    if a1.ndim == 1:
        a1 = a1[:, None].expand(e, branches)
    physical = (i + 127) // 128 * 128
    first = torch.zeros(
        (e, branches, physical, h // 2), dtype=torch.uint8, device=w1.device
    )
    first[:, :, :i].copy_(w1.reshape(e, branches, i, h // 2))
    storage_dtype = torch.bfloat16 if integer_weights else torch.uint8
    first_scale = torch.zeros(
        (e, branches, physical, h // group), dtype=storage_dtype, device=w1.device
    )
    first_scale[:, :, :i].copy_(
        s1.view(storage_dtype).reshape(e, branches, i, h // group)
    )
    second = torch.zeros((e, h, physical // 2), dtype=torch.uint8, device=w1.device)
    second[:, :, : i // 2].copy_(w2)
    second_scale = torch.zeros(
        (e, h, physical // group), dtype=storage_dtype, device=w1.device
    )
    second_scale[:, :, : i // group].copy_(s2.view(storage_dtype))
    return dict(
        fc1_expert_weights=first.reshape(e, branches * physical, h // 2),
        fc1_block_scales=first_scale.reshape(e, branches * physical, h // group),
        fc1_dequant_scale=a1.contiguous(),
        fc2_expert_weights=second,
        fc2_block_scales=second_scale,
        fc2_dequant_scale=a2.contiguous(),
    )


def pack_inputs(act, weights, config, device):
    from ...api import RoutingInputMode, QuantFormat

    integer_weights = config.quant.weight is QuantFormat.MXINT4
    group = 16 if config.quant.weight is QuantFormat.NVFP4 else 32
    scale_dtype = torch.bfloat16 if integer_weights else torch.uint8

    activation_dtype = (
        torch.float16 if config.quant.activation is QuantFormat.FP16 else torch.bfloat16
    )
    output_dtype = (
        torch.float16 if config.quant.output is QuantFormat.FP16 else torch.bfloat16
    )
    x, ids, scores = act.hidden_states_q, act.topk_ids, act.topk_weights
    t, h = x.shape
    e, k = config.routing.num_experts, config.routing.top_k
    p = (config.experts.intermediate_size + 127) // 128 * 128
    branches = 2 if config.activation.is_gated else 1
    if act.hidden_states_scale is not None:
        raise ValueError("W4A16 input must not supply an activation scale")
    if (
        x.dtype is not activation_dtype
        or x.device != device
        or not x.is_contiguous()
        or x.data_ptr() % 16
    ):
        raise ValueError(
            f"W4A16 requires aligned contiguous {activation_dtype} input on the runner device"
        )
    view = weights.get_view("frost_low_latency")
    for name, shape, dtype in (
        ("fc1_expert_weights", (e, branches * p, h // 2), torch.uint8),
        ("fc1_block_scales", (e, branches * p, h // group), scale_dtype),
        ("fc1_dequant_scale", (e, branches), torch.float32),
        ("fc2_expert_weights", (e, h, p // 2), torch.uint8),
        ("fc2_block_scales", (e, h, p // group), scale_dtype),
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
    packed = act.routing_input_mode is RoutingInputMode.PackedPrecomputed
    narrow = packed and config.finalize.do_finalize
    if packed and not narrow:
        scores = scores.to(torch.bfloat16)
    return [
        x,
        view["fc1_expert_weights"],
        view["fc1_block_scales"],
        view["fc1_dequant_scale"],
        view["fc2_expert_weights"],
        view["fc2_block_scales"],
        view["fc2_dequant_scale"],
        ids,
        scores,
        torch.empty((t * k, p), device=device, dtype=activation_dtype),
        torch.empty((t * k, h), device=device, dtype=output_dtype),
        torch.empty((t, h), device=device, dtype=output_dtype),
        narrow,
    ]


def execute(inputs, config, tactic):
    from .fma.fp4_weight_only import execute as run

    return run(inputs, config, tactic)
