# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capability router for MiniMax-H3 NVFP4 pre-attention CUDA stages."""

from __future__ import annotations

import importlib
from typing import Literal

import torch

MiniMaxH3Nvfp4Target = Literal["sm100a", "sm103a"]
MiniMaxH3Nvfp4Stage = Literal[
    "norm_adaln_nvfp4_quantize",
    "qkv_nvfp4_gemm_fused_pack",
]


def minimax_h3_nvfp4_target(device: torch.device) -> MiniMaxH3Nvfp4Target:
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    if capability == (10, 0):
        return "sm100a"
    if capability == (10, 3):
        return "sm103a"
    raise RuntimeError(
        "MiniMax-H3 NVFP4 pre-attention requires exact compute capability "
        f"10.0 or 10.3, got {capability[0]}.{capability[1]}"
    )


def minimax_h3_nvfp4_physical_module(device: torch.device):
    target = minimax_h3_nvfp4_target(device)
    return importlib.import_module(
        f".cake_minimax_h3_nvfp4_pre_attention_{target}",
        __package__,
    )


def minimax_h3_nvfp4_route_record(device: torch.device, P: int) -> dict:
    return minimax_h3_nvfp4_physical_module(device).minimax_h3_nvfp4_route_record(P)


def gen_minimax_h3_nvfp4_stage_module(
    device: torch.device,
    P: int,
    stage: MiniMaxH3Nvfp4Stage,
):
    return minimax_h3_nvfp4_physical_module(device).gen_minimax_h3_nvfp4_stage_module(
        P, stage
    )


def load_minimax_h3_nvfp4_stage_module(
    device: torch.device,
    P: int,
    stage: MiniMaxH3Nvfp4Stage,
):
    return minimax_h3_nvfp4_physical_module(device).load_minimax_h3_nvfp4_stage_module(
        P, stage
    )


def load_minimax_h3_nvfp4_route(device: torch.device, P: int):
    module = minimax_h3_nvfp4_physical_module(device)
    return (
        module.load_minimax_h3_nvfp4_stage_module(P, "norm_adaln_nvfp4_quantize"),
        module.load_minimax_h3_nvfp4_stage_module(P, "qkv_nvfp4_gemm_fused_pack"),
    )


__all__ = [
    "MiniMaxH3Nvfp4Stage",
    "MiniMaxH3Nvfp4Target",
    "gen_minimax_h3_nvfp4_stage_module",
    "load_minimax_h3_nvfp4_route",
    "load_minimax_h3_nvfp4_stage_module",
    "minimax_h3_nvfp4_physical_module",
    "minimax_h3_nvfp4_route_record",
    "minimax_h3_nvfp4_target",
]
