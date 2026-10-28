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
from .activation import activate
from .io import pack_scale, combine


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
    activation: str = "swiglu"
    activation_parameters: Optional[Tuple] = None
    gated: bool = True

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
    name: cutlass.Constexpr,
    parameters: cutlass.Constexpr,
    gated: cutlass.Constexpr,
    tile: cutlass.Int32,
    route: cutlass.Int32,
):
    tid, _, _ = cute.arch.thread_idx()
    lane = cute.arch.lane_idx()
    # The producing CTA owns adjacent 64-channel gate and up tiles. The
    # GEMM body ends with a CTA barrier before these FP32 values are read.
    width = 64 if cutlass.const_expr(gated) else 128
    if tid < width:
        col = tile * width + tid
        if col < physical:
            gate = cutlass.Float32(0)
            if cutlass.const_expr(gated):
                gate = partials[route, tile * 128 + tid]
                up = partials[route, tile * 128 + tid + 64]
            else:
                up = partials[route, col]
            value = cutlass.Float32(0)
            if col < logical:
                value = activate(up, gate, name, limit, alpha, beta, parameters)
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
    def launch(
        self,
        q,
        sf,
        ids,
        weights,
        w1,
        s1,
        w2,
        s2,
        packed_sf,
        partials,
        midq,
        midsf,
        routes,
        output,
        problem1: Tuple[
            cutlass.Int32,
            cutlass.Int32,
            cutlass.Int32,
            cutlass.Int32,
            cutlass.Int32,
            cutlass.Int32,
        ],
        problem2: Tuple[
            cutlass.Int32,
            cutlass.Int32,
            cutlass.Int32,
            cutlass.Int32,
            cutlass.Int32,
            cutlass.Int32,
        ],
        weight_dtype: cutlass.Constexpr,
        narrow_weights: cutlass.Constexpr,
        finalize: cutlass.Constexpr,
        stream,
    ):
        route_ids = cute.make_tensor(ids.iterator, cute.make_layout(cute.size(ids)))
        first = (
            cute.recast_ptr(w1.iterator, dtype=weight_dtype),
            cute.recast_ptr(s1.iterator, dtype=cutlass.Float8E8M0FNU),
            q.iterator,
            cute.recast_ptr(packed_sf.iterator, dtype=cutlass.Float8E8M0FNU),
            partials.iterator,
            None,
            None,
            problem1,
            route_ids,
        )
        second = (
            cute.recast_ptr(w2.iterator, dtype=weight_dtype),
            cute.recast_ptr(s2.iterator, dtype=cutlass.Float8E8M0FNU),
            midq.iterator,
            cute.recast_ptr(midsf.iterator, dtype=cutlass.Float8E8M0FNU),
            routes.iterator,
            None,
            None,
            problem2,
            route_ids,
        )
        pack_scale(sf, packed_sf, midsf, stream)
        self.gate_up(first, (partials, midq, midsf), stream)
        self.down_project(second, stream)
        if cutlass.const_expr(finalize):
            combine(routes, weights, output, narrow_weights, stream)

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
            self.geometry.activation,
            self.geometry.activation_parameters,
            self.geometry.gated,
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
