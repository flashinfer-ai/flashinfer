# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical NVFP4 bindings for shared low-token FMA device kernels."""

import torch

from ...api import RoutingInputMode


def pack_inputs(act, view, config, device):
    q, sf, ids, scores = (
        act.hidden_states_q,
        act.hidden_states_scale,
        act.topk_ids,
        act.topk_weights,
    )
    t, packed_h = q.shape
    h, i, e, k = (
        packed_h * 2,
        config.experts.intermediate_size,
        config.routing.num_experts,
        config.routing.top_k,
    )
    branches = 2 if config.activation.is_gated else 1
    specs = (
        ("fc1_expert_weights", torch.uint8, (e, branches * i, h // 2)),
        ("fc2_expert_weights", torch.uint8, (e, h, i // 2)),
        ("fc1_weight_block_scale", torch.uint8, (e, branches * i, h // 16)),
        ("fc2_weight_block_scale", torch.uint8, (e, h, i // 16)),
        ("fc1_dequant_scale", torch.float32, (e,)),
        ("fc2_dequant_scale", torch.float32, (e,)),
    )
    for name, dtype, shape in specs:
        value = view[name]
        alignment = 4 if dtype is torch.float32 else 16
        if (
            value.shape != shape
            or value.dtype != dtype
            or value.device != device
            or not value.is_contiguous()
            or value.data_ptr() % alignment
        ):
            raise ValueError(
                f"{name} must be aligned contiguous {dtype} {shape} on {device}"
            )
    for name in ("fc1_act_global_scale", "fc2_act_global_scale"):
        value = view[name]
        if (
            value.dtype != torch.float32
            or value.shape not in ((), (1,))
            or value.device != device
            or not value.is_contiguous()
            or value.data_ptr() % 4
        ):
            raise ValueError(f"{name} must be an aligned CUDA float32 scalar")
    if (
        q.dtype != torch.uint8
        or q.device != device
        or not q.is_contiguous()
        or q.data_ptr() % 16
    ):
        raise ValueError("NVFP4 input must be aligned contiguous CUDA uint8 [T,H/2]")
    if (
        sf is None
        or sf.shape != (t, h // 16)
        or sf.dtype not in (torch.uint8, torch.float8_e4m3fn)
        or sf.device != device
        or not sf.is_contiguous()
        or sf.data_ptr() % 16
    ):
        raise ValueError("NVFP4 input scales must be aligned linear E4M3 [T,H/16]")
    packed = act.routing_input_mode == RoutingInputMode.PackedPrecomputed
    narrow = packed and config.finalize.do_finalize
    if packed and not narrow:
        scores = scores.to(torch.bfloat16)
    first_scale = (
        view["fc1_weight_block_scale"].view(e, branches, i, h // 16).permute(1, 0, 2, 3)
    )
    return [
        q,
        sf.view(torch.uint8),
        ids,
        scores,
        view["fc1_expert_weights"],
        first_scale,
        view["fc1_dequant_scale"],
        view["fc2_expert_weights"],
        view["fc2_weight_block_scale"],
        view["fc2_dequant_scale"],
        view["fc2_act_global_scale"].reshape(1),
        torch.empty((t * k, i // 2), device=device, dtype=torch.uint8),
        torch.empty(t * k * 128 * (i // 16), device=device, dtype=torch.uint8),
        torch.empty(
            (t, h) if config.finalize.do_finalize else (t * k, h),
            device=device,
            dtype=torch.bfloat16,
        ),
        narrow,
    ]


def execute(inputs, config, tactic):
    from .fma.nvfp4.runtime import execute as run

    return run(inputs, config, tactic)
