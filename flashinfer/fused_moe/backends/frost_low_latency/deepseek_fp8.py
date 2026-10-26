# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical E4M3 block weights and live FP32 scale bindings."""

import torch


def pack_inputs(act, weights, config, device):
    from ...api import RoutingInputMode

    x, ids, scores = act.hidden_states_q, act.topk_ids, act.topk_weights
    t, h = x.shape
    e, k, i = (
        config.routing.num_experts,
        config.routing.top_k,
        config.experts.intermediate_size,
    )
    if (
        x.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
        or x.device != device
        or not x.is_contiguous()
        or x.data_ptr() % 16
    ):
        raise ValueError(
            "DeepSeek Frost requires aligned contiguous BF16 or E4M3 input on the runner device"
        )
    sf = act.hidden_states_scale
    if x.dtype is torch.bfloat16:
        if sf is not None:
            raise ValueError("BF16 DeepSeek input must not supply activation scales")
        qx = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        sf = torch.empty((h // 128, t), device=device, dtype=torch.float32)
    else:
        qx = x
        if (
            sf is None
            or sf.dtype is not torch.float32
            or sf.shape != (h // 128, t)
            or sf.device != device
            or not sf.is_contiguous()
        ):
            raise ValueError(
                "E4M3 DeepSeek input requires contiguous FP32 [H/128,T] scales on the runner device"
            )
    view = weights.get_view("frost_low_latency")
    branches = 2 if config.activation.is_gated else 1
    for name, shape, dtype in (
        ("fc1_expert_weights", (e, branches * i, h), torch.float8_e4m3fn),
        ("fc1_block_scales", (e, branches * i // 128, h // 128), torch.float32),
        ("fc2_expert_weights", (e, h, i), torch.float8_e4m3fn),
        ("fc2_block_scales", (e, h // 128, i // 128), torch.float32),
    ):
        value = view[name]
        if (
            value.dtype != dtype
            or value.shape != shape
            or value.device != device
            or not value.is_contiguous()
            or (name.endswith("weights") and value.data_ptr() % 16)
        ):
            raise ValueError(
                f"{name} must be aligned contiguous {dtype} {shape} on {device}"
            )
    packed = act.routing_input_mode is RoutingInputMode.PackedPrecomputed
    narrow = packed and config.finalize.do_finalize
    if packed and not config.finalize.do_finalize:
        scores = scores.to(torch.bfloat16)
    return [
        x,
        qx,
        sf,
        view["fc1_expert_weights"],
        view["fc1_block_scales"],
        view["fc2_expert_weights"],
        view["fc2_block_scales"],
        ids,
        scores,
        torch.empty((t * k, i), device=device, dtype=torch.float32),
        torch.empty((t * k, i), device=device, dtype=torch.float8_e4m3fn),
        torch.empty((i // 128, t * k), device=device, dtype=torch.float32),
        torch.empty((t * k, h), device=device, dtype=torch.bfloat16),
        torch.empty((t, h), device=device, dtype=torch.bfloat16),
        narrow,
    ]


def execute(inputs, config, tactic):
    from .fma.deepseek_fp8 import execute as run

    return run(inputs, config, tactic)
