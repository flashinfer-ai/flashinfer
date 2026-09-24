# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capability router for the MiniMax-H3 QKV quantize-and-pack CUDA programs."""

from __future__ import annotations

import importlib
from typing import Literal

import torch

MiniMaxH3QkvPackTarget = Literal["sm100a", "sm103a"]
MiniMaxH3QkvPackFormat = Literal["nvfp4", "mxfp8"]


def minimax_h3_qkv_pack_target(device: torch.device) -> MiniMaxH3QkvPackTarget:
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    if capability == (10, 0):
        return "sm100a"
    if capability == (10, 3):
        return "sm103a"
    raise RuntimeError(
        "MiniMax-H3 QKV quantize-and-pack requires exact compute capability "
        f"10.0 or 10.3, got {capability[0]}.{capability[1]}"
    )


def minimax_h3_qkv_pack_physical_module(device: torch.device):
    target = minimax_h3_qkv_pack_target(device)
    return importlib.import_module(
        f".cake_minimax_h3_qkv_quantize_pack_{target}",
        __package__,
    )


def minimax_h3_qkv_pack_route_record(
    device: torch.device, P: int, fmt: MiniMaxH3QkvPackFormat
) -> dict:
    return minimax_h3_qkv_pack_physical_module(device).minimax_h3_qkv_pack_route_record(
        P, fmt
    )


def gen_minimax_h3_qkv_pack_module(
    device: torch.device, P: int, fmt: MiniMaxH3QkvPackFormat
):
    return minimax_h3_qkv_pack_physical_module(device).gen_minimax_h3_qkv_pack_module(
        P, fmt
    )


def load_minimax_h3_qkv_pack_module(
    device: torch.device, P: int, fmt: MiniMaxH3QkvPackFormat
):
    return minimax_h3_qkv_pack_physical_module(device).load_minimax_h3_qkv_pack_module(
        P, fmt
    )


__all__ = [
    "MiniMaxH3QkvPackFormat",
    "MiniMaxH3QkvPackTarget",
    "gen_minimax_h3_qkv_pack_module",
    "load_minimax_h3_qkv_pack_module",
    "minimax_h3_qkv_pack_physical_module",
    "minimax_h3_qkv_pack_route_record",
    "minimax_h3_qkv_pack_target",
]
