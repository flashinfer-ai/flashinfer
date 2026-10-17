# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused gate/up activation followed by a dependent low-latency Down GEMM."""

from dataclasses import dataclass
from typing import Optional, Tuple

import cutlass
from cutlass import cute
from .gemm import RoutedBlockscaledGemm
from .quantization import (
    _mul,
    _sf_index,
    _inverse_scale,
    float_to_ue8m0_fast,
    cvt_f32_to_e4m3,
)


@dataclass(frozen=True)
class Geometry:
    tokens: int
    hidden: int
    intermediate: int
    logical_intermediate: int
    experts: int
    top_k: int
    clamp_limit: Optional[float] = None
    alpha: float = 1.0
    beta: float = 0.0

    def __post_init__(self):
        if not 1 <= self.tokens <= 8:
            raise ValueError("frost_low_latency requires 1 to 8 token rows")
        if self.hidden < 128 or self.hidden % 128:
            raise ValueError("hidden size must be a positive multiple of 128")
        if self.intermediate < 128 or self.intermediate % 128:
            raise ValueError(
                "physical intermediate size must be a positive multiple of 128"
            )
        if not 0 < self.logical_intermediate <= self.intermediate:
            raise ValueError(
                "logical intermediate size must fit in the physical tensor"
            )
        if not 1 <= self.top_k <= self.experts:
            raise ValueError("top_k must be between 1 and the number of experts")
        if self.clamp_limit is not None and self.clamp_limit <= 0:
            raise ValueError("clamp_limit must be positive or None")

    @property
    def routes(self):
        return self.tokens * self.top_k


@cute.jit
def activate_tile(
    partials,
    quantized,
    sf,
    logical: cutlass.Constexpr,
    physical: cutlass.Constexpr,
    limit: cutlass.Constexpr,
    alpha: cutlass.Constexpr,
    beta: cutlass.Constexpr,
    tile: cutlass.Int32,
    route: cutlass.Int32,
):
    tid, _, _ = cute.arch.thread_idx()
    lane = cute.arch.lane_idx()
    # The producing CTA owns adjacent 64-channel gate and up tiles. The
    # GEMM body ends with a CTA barrier before these FP32 values are read.
    if tid < 64:
        col = tile * 64 + tid
        if col < physical:
            gate = partials[route, tile * 128 + tid]
            up = partials[route, tile * 128 + tid + 64]
            value = cutlass.Float32(0)
            if col < logical:
                if cutlass.const_expr(limit is not None):
                    gate = cutlass.min(gate, cutlass.Float32(limit))
                    up = cutlass.max(
                        cutlass.min(up, cutlass.Float32(limit)), cutlass.Float32(-limit)
                    )
                gate_exp = gate
                if cutlass.const_expr(alpha != 1.0):
                    gate_exp = _mul(gate, cutlass.Float32(alpha))
                if cutlass.const_expr(beta != 0.0):
                    up = up + cutlass.Float32(beta)
                value = _mul(
                    gate / (1.0 + cute.math.exp(-gate_exp, fastmath=False)), up
                )
            maximum = cute.arch.warp_reduction_max(abs(value))
            scale = float_to_ue8m0_fast(_mul(maximum, cutlass.Float32(1.0 / 448.0)))
            code = cvt_f32_to_e4m3(_mul(value, _inverse_scale(scale)))
            quantized[route, col] = code.to(cutlass.Uint8).bitcast(cutlass.Float8E4M3FN)
            if lane == 0:
                group, groups = col // 32, physical // 32
                sf[route * 128 * groups + _sf_index(0, group, groups)] = scale.to(
                    cutlass.Uint8
                )
    cute.arch.barrier()


class Pipeline:
    """Compiled functions accept all pointers per call; no bound workspaces."""

    def __init__(self, geometry, gu_k=128, enable_pdl=True):
        if gu_k not in (128, 256) or geometry.hidden % gu_k:
            raise ValueError("GU tile K must be 128 or 256 and divide hidden size")
        self.geometry = geometry
        self.enable_pdl = enable_pdl
        self.gu = RoutedBlockscaledGemm(
            tile_k=gu_k,
            allocation_columns=64 if gu_k == 128 else 128,
            input_route_divisor=geometry.top_k,
            use_pdl=enable_pdl,
        )
        self.down = RoutedBlockscaledGemm(
            tile_k=128, allocation_columns=64, input_route_divisor=1, use_pdl=enable_pdl
        )

    @cute.jit
    def gate_up(self, pointers: Tuple, activation: Tuple, stream):
        args, grid = self.gu.prepare(*pointers, stream)
        self.gate_up_kernel(args, activation).launch(
            grid=grid,
            block=(256, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def gate_up_kernel(self, args: Tuple, activation: Tuple):
        if cutlass.const_expr(self.enable_pdl):
            # Expert IDs can be produced by an upstream router. Acquire them
            # before the weight producer uses them to select a TMA address.
            cute.arch.griddepcontrol_wait()
        tile, _, route = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        arena = smem.allocate(
            self.gu.shared_storage.size_in_bytes(), byte_alignment=1024
        )
        self.gu.device_body(*args, (tile, cutlass.Int32(0), route), arena)
        activate_tile(
            *activation,
            self.geometry.logical_intermediate,
            self.geometry.intermediate,
            self.geometry.clamp_limit,
            self.geometry.alpha,
            self.geometry.beta,
            tile,
            route,
        )
        # The upstream weight producer can trigger the Down launch early.
        # Down's activation producer waits for this entire kernel to finish,
        # including the quantized activation and scale-factor stores above.

    @cute.jit
    def down_project(self, pointers: Tuple, stream):
        args, grid = self.down.prepare(*pointers, stream)
        self.down_kernel(args).launch(
            grid=grid,
            block=(256, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def down_kernel(self, args: Tuple):
        tile, _, route = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        arena = smem.allocate(
            self.down.shared_storage.size_in_bytes(), byte_alignment=1024
        )
        self.down.device_body(*args, (tile, cutlass.Int32(0), route), arena)
