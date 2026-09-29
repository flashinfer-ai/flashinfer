# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capability router for MiniMax-H3 MXFP8 pre-attention CUDA stages."""

from __future__ import annotations

import importlib
from typing import Literal

import torch

MiniMaxH3Mxfp8Target = Literal["sm100a", "sm103a"]
MiniMaxH3Mxfp8Stage = Literal[
    "norm_adaln_mxfp8_quantize",
    "qk_rope_destination_mxfp8_pack",
]


def minimax_h3_mxfp8_target(device: torch.device) -> MiniMaxH3Mxfp8Target:
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    if capability == (10, 0):
        return "sm100a"
    if capability == (10, 3):
        return "sm103a"
    raise RuntimeError(
        "MiniMax-H3 MXFP8 pre-attention requires exact compute capability "
        f"10.0 or 10.3, got {capability[0]}.{capability[1]}"
    )


def minimax_h3_mxfp8_physical_module(device: torch.device):
    target = minimax_h3_mxfp8_target(device)
    return importlib.import_module(
        f".cake_minimax_h3_mxfp8_pre_attention_{target}",
        __package__,
    )


def minimax_h3_mxfp8_route_record(device: torch.device, M: int, P: int) -> dict:
    return minimax_h3_mxfp8_physical_module(device).minimax_h3_mxfp8_route_record(M, P)


def gen_minimax_h3_mxfp8_stage_module(
    device: torch.device,
    M: int,
    P: int,
    stage: MiniMaxH3Mxfp8Stage,
):
    return minimax_h3_mxfp8_physical_module(device).gen_minimax_h3_mxfp8_stage_module(
        M, P, stage
    )


def load_minimax_h3_mxfp8_stage_module(
    device: torch.device,
    M: int,
    P: int,
    stage: MiniMaxH3Mxfp8Stage,
):
    return minimax_h3_mxfp8_physical_module(device).load_minimax_h3_mxfp8_stage_module(
        M, P, stage
    )


def load_minimax_h3_mxfp8_route(device: torch.device, M: int, P: int):
    module = minimax_h3_mxfp8_physical_module(device)
    return (
        module.load_minimax_h3_mxfp8_stage_module(M, P, "norm_adaln_mxfp8_quantize"),
        module.load_minimax_h3_mxfp8_stage_module(
            M, P, "qk_rope_destination_mxfp8_pack"
        ),
    )


__all__ = [
    "MiniMaxH3Mxfp8Stage",
    "MiniMaxH3Mxfp8Target",
    "gen_minimax_h3_mxfp8_stage_module",
    "load_minimax_h3_mxfp8_route",
    "load_minimax_h3_mxfp8_stage_module",
    "minimax_h3_mxfp8_physical_module",
    "minimax_h3_mxfp8_route_record",
    "minimax_h3_mxfp8_target",
]
