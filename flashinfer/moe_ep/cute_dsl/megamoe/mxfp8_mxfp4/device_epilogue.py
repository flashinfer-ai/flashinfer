# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Autonomous persistent epilogue for MXFP8 x MXFP4 MegaMoE.

The swap-AB MMA leaves feature rows and token columns in TMEM. FC1's
16dp256b loads match the gate8/up8 weight interleave, so each lane computes
SwiGLU directly without transposing or exchanging gate/up values. Only K32
block maxima cross lanes and paired warps. FC2 uses the same native loads
with a shared-memory BF16 transpose and coalesced route stores.

This module owns accumulator wait/release, tile traversal, output stores,
and FC1 readiness publication after the output data and scales are visible.

Routing scores are absent by construction.  FC2 publishes an unweighted BF16
term for every ``(source token, source top-k slot)``; the deterministic reducer
in :mod:`fc2_epilogue` owns score multiplication and ordered accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import Int32, Int64, dsl_user_op

from .device_scheduler import DevicePhase
from .fc1_epilogue import (
    Fc1Epilogue,
    Fc1EpilogueConfig,
    _clamp_and_swiglu,
    _quantize_with_amax,
)
from .fc2_store import stage_native_bf16, store_staged_vector
from .wait_stats import Slot, globaltimer_lo, stat_add, stats_slot_ptr
from .iket_markers import iket_enabled, iket_fine_ctas, is_fine_cta, load_iket
from .fc2_epilogue import Fc2RouteStore


WARP_SIZE = 32
EPILOGUE_WARPS = 4
# IDs 0/1/2 are CTA/tile/TMEM barriers; dispatch owns 8/9/10.
EXCHANGE_BARRIER_IDS = (3, 4)
TOKEN_METADATA_BYTES = 8
TMEM_ROW_STRIDE = 1 << 16
# Only the two warps' half-block maxima cross shared memory. Each row
# contains one amax per token; the final barrier protects reuse by the next group.
EXCHANGE_LANE_STRIDE = WARP_SIZE
EXCHANGE_ROWS = 2
EXCHANGE_WORDS = (EPILOGUE_WARPS // 2) * EXCHANGE_ROWS * EXCHANGE_LANE_STRIDE


@dataclass(frozen=True)
class DeviceEpilogueConfig:
    """Compile-time geometry and policy for the persistent epilogue.

    ``cta_tile_features`` is the swap-AB MMA M extent (weight/output rows),
    while ``cta_tile_tokens`` is its N extent.  FC1 rows are interleaved as
    eight gate rows followed by eight up rows, so a 128-row feature tile produces 64
    post-SwiGLU values. Each warpgroup covers all feature rows; with eight
    epilogue warps, the two warpgroups process alternating token groups.
    """

    cta_tile_features: int = 128
    cta_tile_tokens: int = 128
    accumulator_stages: int = 2
    epilogue_warps: int = EPILOGUE_WARPS
    barrier_id: int = 1
    gate_up_clamp: Optional[float] = None
    fast_math: bool = True

    def __post_init__(self) -> None:
        if self.cta_tile_features != 128:
            raise ValueError(
                "the correctness epilogue currently requires a 128-row "
                "swap-AB feature tile"
            )
        if self.cta_tile_tokens not in (2 * WARP_SIZE, 4 * WARP_SIZE, 6 * WARP_SIZE):
            # Every warp covers each 32-token group of its feature rows; N64,
            # N128, and N192 are the token tiles validated end to end.
            raise ValueError("the epilogue requires N64, N128, or N192")
        if self.accumulator_stages != 2:
            raise ValueError(
                "the correctness epilogue uses exactly two non-overlapping "
                "accumulator stages"
            )
        if self.epilogue_warps not in (4, 8):
            raise ValueError("the epilogue requires four or eight warps")
        if self.barrier_id < 0:
            raise ValueError("barrier_id must be non-negative")
        if self.epilogue_warps == 8 and self.barrier_id in EXCHANGE_BARRIER_IDS:
            raise ValueError("tile and exchange barriers must differ")
        if self.gate_up_clamp is not None and self.gate_up_clamp < 0.0:
            raise ValueError("gate_up_clamp must be None or non-negative")

    @property
    def token_groups(self) -> int:
        return self.cta_tile_tokens // WARP_SIZE

    @property
    def fc1_output_features_per_tile(self) -> int:
        return self.cta_tile_features // 2

    @property
    def fc1_active_warps(self) -> int:
        return self.epilogue_warps

    @property
    def fc2_active_warps(self) -> int:
        return self.epilogue_warps

    @property
    def accumulator_tmem_columns(self) -> int:
        return self.cta_tile_tokens * self.accumulator_stages


def route_metadata_host(packed: int) -> tuple[int, int, int]:
    """Decode the stable eight-byte token-source record on the host.

    Low 32 bits are ``source token``.  The high word contains
    ``source rank`` in bits 16..31 and ``source top-k slot`` in bits 0..15.
    This helper is also the host-side test oracle for the device decoder.
    """

    value = packed & ((1 << 64) - 1)
    high = value >> 32
    return ((high >> 16) & 0xFFFF, value & 0xFFFFFFFF, high & 0xFFFF)


@dsl_user_op
def _store_predicated_u8(
    ptr,
    value,
    valid,
    offset: cutlass.Constexpr[int] = 0,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Keep the byte offset immediate and predicate only the global store."""

    llvm.inline_asm(
        None,
        [
            ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(value).ir_value(loc=loc, ip=ip),
            Int32(valid).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; "
        f"@p st.global.u8 [$0 + {offset}], $1; }}",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _red_add_release_gpu_s32(
    counter_ptr,
    value,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Publish a task-tile completion after the caller's device fence."""

    llvm.inline_asm(
        None,
        [
            counter_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _load_route_metadata(token_metadata: cute.Tensor, pool_token):
    """Load ``(source rank, source token, source top-k)`` from one i64."""

    byte_address = cute.recast_tensor(
        token_metadata, cutlass.Uint8
    ).iterator.toint() + Int64(pool_token) * Int64(TOKEN_METADATA_BYTES)
    pointer = cute.make_ptr(
        Int64,
        byte_address,
        cute.AddressSpace.gmem,
        assumed_align=TOKEN_METADATA_BYTES,
    )
    packed = Int64(cute.arch.load(pointer, Int64, scope="gpu"))
    high = packed >> Int64(32)
    source_rank = Int32((high >> Int64(16)) & Int64(0xFFFF))
    source_token = Int32(packed & Int64(0xFFFFFFFF))
    source_topk = Int32(high & Int64(0xFFFF))
    return source_rank, source_token, source_topk


class PersistentMxfp8Mxfp4Epilogue:
    """Own the complete epilogue side of the persistent FC1/FC2 loop.

    The integrating kernel supplies the accumulator TMEM allocation, pipeline,
    scheduler consumer, output tensors, and optional source metadata/peer
    mapper.  It does not traverse accumulator tiles or publish readiness
    counters itself.
    """

    def __init__(self, config: Optional[DeviceEpilogueConfig] = None) -> None:
        if config is None:
            config = DeviceEpilogueConfig()
        self.config = config
        # IKET ranges (CUTE_DSL_COMPILER_OPT=iket only; see iket_markers).
        self.iket = iket_enabled()
        self._iket: Any = load_iket() if self.iket else None
        # Per-step FC1 tile ranges on the first CTAs only (see iket_markers).
        self.iket_fine_ctas = iket_fine_ctas()
        self.fc1 = Fc1Epilogue(
            Fc1EpilogueConfig(
                gate_up_clamp=config.gate_up_clamp,
                fast_math=config.fast_math,
            )
        )

    @property
    def num_acc_pipeline_stages(self) -> int:
        return self.config.accumulator_stages

    @property
    def num_accumulator_tmem_cols(self) -> int:
        return self.config.accumulator_tmem_columns

    @staticmethod
    def _tmem_layout() -> cute.Layout:
        return cute.make_layout(
            (((WARP_SIZE, WARP_SIZE), 1),),
            stride=(((TMEM_ROW_STRIDE, 1), 0),),
        )

    @staticmethod
    def _rmem_layout() -> cute.Layout:
        return cute.make_layout((((WARP_SIZE,), 1),), stride=(((1,), 0),))

    @cute.jit
    def _load_fc1_native_half(
        self,
        tmem_acc_tensor,
        warp_idx,
        token_group,
        acc_stage,
        half: cutlass.Constexpr[int],
    ):
        """Load 16 feature rows x 32 tokens without a register transpose.

        For register r and lane l, 16dp256b maps to row l//4 + 8*((r//2)%2)
        and column 2*(l%4) + r%2 + 8*(r//4). Gate8/up8 interleaving puts
        matching gate/up in registers (4*t,4*t+1)/(4*t+2,4*t+3).
        """
        offset = (
            (warp_idx % Int32(4)) * Int32(32 * TMEM_ROW_STRIDE)
            + Int32(half * 16 * TMEM_ROW_STRIDE)
            + token_group * Int32(32)
            + acc_stage * Int32(self.config.cta_tile_tokens)
        )
        source = cute.make_tensor(
            tmem_acc_tensor.iterator + cute.assume(offset, divby=32),
            cute.make_layout((((16, 32), 1),), stride=(((TMEM_ROW_STRIDE, 1), 0),)),
        )
        registers = cute.make_rmem_tensor(
            cute.make_layout((((16,), 1),), stride=(((1,), 0),)).shape,
            cutlass.Float32,
        )
        cute.copy(
            cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition.x4), cutlass.Float32
            ),
            source,
            registers,
        )
        return registers

    @cute.jit
    def _run_fc1_task_tile(
        self,
        *,
        work,
        tmem_acc_tensor: cute.Tensor,
        acc_stage,
        fc1_output: cute.Tensor,
        fc1_output_scales: cute.Tensor,
        epilogue_exchange: cute.Tensor,
        boundary,
        warp_idx,
        lane_idx,
        warpgroup_idx,
        fine_cta,
    ) -> None:
        """Traverse gate/up TMEM and publish E4M3+E8M0/K32 handoff rows."""

        # Swap-AB TMEM rows are interleaved gate/up features and columns are
        # tokens. Each warp owns 32 feature rows. Warpgroup-local TMEM rows
        # repeat in the second warpgroup, but exchange pairs must be disjoint.
        pair_idx = warp_idx // Int32(2)
        # Skip FC1 groups outside the tile's valid token range. The count
        # is uniform across a warpgroup, so all four warps skip together
        # and the FC1 barriers stay balanced.
        for token_group in cutlass.range_constexpr(self.config.token_groups):
            if cutlass.const_expr(self.config.epilogue_warps == 4) or (
                Int32(token_group % 2) == warpgroup_idx
            ):
                if Int32(token_group * WARP_SIZE) < work.valid_tokens_in_cta_tile:
                    self._run_fc1_token_group(
                        work=work,
                        token_group=token_group,
                        tmem_acc_tensor=tmem_acc_tensor,
                        acc_stage=acc_stage,
                        fc1_output=fc1_output,
                        fc1_output_scales=fc1_output_scales,
                        epilogue_exchange=epilogue_exchange,
                        boundary=boundary,
                        warp_idx=warp_idx,
                        lane_idx=lane_idx,
                        pair_idx=pair_idx,
                        fine_cta=fine_cta,
                    )

    @cute.jit
    def _run_fc1_token_group(
        self,
        *,
        work,
        token_group: cutlass.Constexpr[int],
        tmem_acc_tensor: cute.Tensor,
        acc_stage,
        fc1_output: cute.Tensor,
        fc1_output_scales: cute.Tensor,
        epilogue_exchange: cute.Tensor,
        boundary,
        warp_idx,
        lane_idx,
        pair_idx,
        fine_cta,
    ) -> None:
        """One 32-token FC1 group: native load, SwiGLU, amax exchange, store."""

        if cutlass.const_expr(self.iket_fine_ctas > 0):
            if fine_cta:
                self._iket.range_push("fc1_ld_math")
        first = self._load_fc1_native_half(
            tmem_acc_tensor, warp_idx, Int32(token_group), acc_stage, 0
        )
        second = self._load_fc1_native_half(
            tmem_acc_tensor, warp_idx, Int32(token_group), acc_stage, 1
        )
        gate = cute.make_rmem_tensor((16,), cutlass.Float32)
        up = cute.make_rmem_tensor((16,), cutlass.Float32)
        for token in cutlass.range_constexpr(8):
            register = (token // 2) * 4 + token % 2
            gate[token * 2] = first[register]
            up[token * 2] = first[register + 2]
            gate[token * 2 + 1] = second[register]
            up[token * 2 + 1] = second[register + 2]
        values = cute.make_rmem_tensor((16,), cutlass.Float32)
        _clamp_and_swiglu(values, gate, up, self.fc1._clamp, self.config.fast_math, 16)
        maxima = cute.make_rmem_tensor((8,), cutlass.Float32)
        for token in cutlass.range_constexpr(8):
            v0, v1 = values[token * 2], values[token * 2 + 1]
            # Seed amax with zero, including all-NaN fragments.
            amax = cute.arch.fmax(
                cutlass.Float32(0.0),
                cute.arch.fmax(cute.arch.fmax(v0, -v0), cute.arch.fmax(v1, -v1)),
            )
            # Eight lanes, spaced four apart, cover 16 features for
            # this token. The partner warp covers the other 16.
            for shift in cutlass.range_constexpr(3):
                amax = cute.arch.fmax(
                    amax, cute.arch.shuffle_sync_bfly(amax, offset=4 << shift)
                )
            maxima[token] = amax
            token_in_group = (lane_idx % Int32(4)) * Int32(2) + Int32(
                (token // 2) * 8 + token % 2
            )
            if lane_idx < Int32(4):
                epilogue_exchange[pair_idx, warp_idx % Int32(2), token_in_group] = amax
        if cutlass.const_expr(self.iket_fine_ctas > 0):
            if fine_cta:
                self._iket.range_pop()
                self._iket.range_push("fc1_amax_bar")
        boundary.arrive_and_wait()
        if cutlass.const_expr(self.iket_fine_ctas > 0):
            if fine_cta:
                self._iket.range_pop()
                self._iket.range_push("fc1_quant_store")
        # One address per lane and group. The native fragment owns rows
        # {0, 1, 8, 9, 16, 17, 24, 25} and features {0, 8}. The launcher
        # supplies a static row stride, so all remaining offsets are
        # store immediates rather than repeated 64-bit row multiplies.
        lane_row = (lane_idx % Int32(4)) * Int32(2)
        tile_row = work.tile_n_idx * Int32(self.config.cta_tile_tokens)
        group_row = tile_row + Int32(token_group * 32)
        block = work.tile_m_idx * Int32(2) + pair_idx % Int32(2)
        feature = (
            block * Int32(32) + (warp_idx % Int32(2)) * Int32(16) + lane_idx // Int32(4)
        )
        data_row = work.cumulative_data_physical_row + group_row + lane_row
        data_ptr = fc1_output.iterator + (
            Int64(data_row) * Int64(fc1_output.stride[0]) + Int64(feature)
        )
        scales = cute.make_rmem_tensor((8,), cutlass.Uint8)
        for token in cutlass.range_constexpr(8):
            token_in_group = (lane_idx % Int32(4)) * Int32(2) + Int32(
                (token // 2) * 8 + token % 2
            )
            amax = cute.arch.fmax(
                maxima[token],
                epilogue_exchange[
                    pair_idx, Int32(1) - warp_idx % Int32(2), token_in_group
                ],
            )
            pair_values = cute.make_rmem_tensor((2,), cutlass.Float32)
            pair_values[0], pair_values[1] = (
                values[token * 2],
                values[token * 2 + 1],
            )
            quantized = cute.make_rmem_tensor((2,), cutlass.Float8E4M3FN)
            raw_scale = _quantize_with_amax(pair_values, quantized, amax, 2)
            scales[token] = raw_scale.to(cutlass.Uint8)
            token_offset = Int32(token_group * 32) + token_in_group
            valid = token_offset < work.valid_tokens_in_cta_tile
            raw = cute.recast_tensor(quantized, cutlass.Uint8)
            # Use one predicated path for full and partial groups to avoid
            # duplicating the unrolled quantization and its register state.
            row_offset = (token // 2) * 8 + token % 2
            _store_predicated_u8(
                data_ptr, raw[0], valid, row_offset * fc1_output.stride[0]
            )
            _store_predicated_u8(
                data_ptr, raw[1], valid, row_offset * fc1_output.stride[0] + 8
            )
        if lane_idx < Int32(4) and warp_idx % Int32(2) == Int32(0):
            # Expert SF rows are padded to 128, and tile/group starts are
            # multiples of 32. Within a group only the atom's row%32 mode
            # varies (stride 16); derive the base from the canonical layout
            # once so K-block and outer-atom strides need no duplication.
            scale_row = work.cumulative_sf_physical_row + group_row + lane_row
            scale_ptr = cute.recast_ptr(
                fc1_output_scales.iterator, dtype=cutlass.Uint8
            ) + cute.crd2idx(
                (scale_row, block * Int32(32), Int32(0)), fc1_output_scales.layout
            )
            for token in cutlass.range_constexpr(8):
                row_offset = (token // 2) * 8 + token % 2
                valid = (
                    Int32(token_group * 32 + row_offset) + lane_row
                    < work.valid_tokens_in_cta_tile
                )
                _store_predicated_u8(scale_ptr, scales[token], valid, row_offset * 16)
        if cutlass.const_expr(self.iket_fine_ctas > 0):
            if fine_cta:
                self._iket.range_pop()
                self._iket.range_push("fc1_group_bar")
        boundary.arrive_and_wait()
        if cutlass.const_expr(self.iket_fine_ctas > 0):
            if fine_cta:
                self._iket.range_pop()

    @cute.jit
    def _run_fc2_task_tile(
        self,
        *,
        work,
        tmem_acc_tensor: cute.Tensor,
        acc_stage,
        route_store: Fc2RouteStore,
        token_metadata,
        epilogue_exchange,
        boundary,
        warp_idx,
        lane_idx,
        warpgroup_idx,
    ) -> None:
        """Traverse FC2 TMEM and store unweighted BF16 route terms."""

        for token_group in cutlass.range_constexpr(self.config.token_groups):
            if cutlass.const_expr(self.config.epilogue_warps == 4) or (
                Int32(token_group % 2) == warpgroup_idx
            ):
                self._run_fc2_token_group(
                    work=work,
                    token_group=token_group,
                    tmem_acc_tensor=tmem_acc_tensor,
                    acc_stage=acc_stage,
                    route_store=route_store,
                    token_metadata=token_metadata,
                    epilogue_exchange=epilogue_exchange,
                    boundary=boundary,
                    warp_idx=warp_idx,
                    lane_idx=lane_idx,
                )

    @cute.jit
    def _run_fc2_token_group(
        self,
        *,
        work,
        token_group: cutlass.Constexpr[int],
        tmem_acc_tensor: cute.Tensor,
        acc_stage,
        route_store: Fc2RouteStore,
        token_metadata,
        epilogue_exchange,
        boundary,
        warp_idx,
        lane_idx,
    ) -> None:
        """One 32-token FC2 group: native loads, BF16 STSM transpose, stores.

        Each warpgroup stages 16 tokens x 128 features in the exchange buffer
        and emits 256-byte contiguous row segments.
        """

        staging = epilogue_exchange.iterator
        first = self._load_fc1_native_half(
            tmem_acc_tensor, warp_idx, Int32(token_group), acc_stage, 0
        )
        second = self._load_fc1_native_half(
            tmem_acc_tensor, warp_idx, Int32(token_group), acc_stage, 1
        )
        # Resolve each token's destination once in its emitting warp. Broadcast
        # the final address, avoiding repeated peer lookup and row arithmetic
        # in every lane that stores a 16-byte piece of the same row.
        token_base = Int32(token_group * WARP_SIZE)
        pool_base = work.cumulative_data_physical_row + work.tile_n_idx * Int32(
            self.config.cta_tile_tokens
        )
        metadata_token = (warp_idx % 4) * 4 + lane_idx % 4 + (lane_idx // 4) * 16
        metadata_lane = lane_idx < 8
        row_address = Int64(0)
        if (
            metadata_lane
            and token_base + metadata_token < work.valid_tokens_in_cta_tile
        ):
            source_rank = Int32(0)
            source_token = pool_base + token_base + metadata_token
            source_topk = Int32(0)
            if cutlass.const_expr(token_metadata is not None):
                source_rank, source_token, source_topk = _load_route_metadata(
                    token_metadata, source_token
                )
            row_address = route_store.resolve_route_row(
                source_rank, source_token, source_topk
            ).iterator.toint()
        address_lo = row_address.to(cutlass.Uint32)
        address_hi = (row_address >> Int64(32)).to(cutlass.Uint32)

        for chunk in cutlass.range_constexpr(2):
            stage_native_bf16(first, staging, warp_idx, lane_idx, 0, chunk)
            stage_native_bf16(second, staging, warp_idx, lane_idx, 1, chunk)
            boundary.arrive_and_wait()
            for stripe in cutlass.range_constexpr(2):
                token = (warp_idx % 4) * 4 + lane_idx // 16 + stripe * 2 + chunk * 16
                feature = (lane_idx % 16) * 8
                offset = (warp_idx // 4) * 4096 + (token % 16) * 256 + feature * 2
                metadata_source = lane_idx // 16 + stripe * 2 + chunk * 4
                # All lanes shuffle, even when the final token group is partial.
                address = Int64(cute.arch.shuffle_sync(address_lo, metadata_source)) | (
                    Int64(cute.arch.shuffle_sync(address_hi, metadata_source))
                    << Int64(32)
                )
                if token_base + token < work.valid_tokens_in_cta_tile:
                    store_staged_vector(
                        staging,
                        offset,
                        0x70,
                        address,
                        work.tile_m_idx * Int32(self.config.cta_tile_features)
                        + feature,
                    )
            # Protect the shared tile from the next token group (or FC1 tile).
            boundary.arrive_and_wait()

    @cute.jit
    def run(
        self,
        *,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        scheduler_consumer,
        fc1_output: cute.Tensor,
        fc1_output_scales: cute.Tensor,
        route_store: Fc2RouteStore,
        fc1_done_counter: cute.Tensor,
        epilogue_exchange: cute.Tensor,
        warp_idx,
        tidx,
        token_metadata=None,
        stats=None,
        stats_word_offset=None,
    ) -> None:
        """Consume and retire all persistent FC1/FC2 work records.

        Publication happens after an all-epilogue barrier and a device-scope
        fence.  The FC1 release-add unlocks FC2 mainloop work.  FC2 needs no
        completion counter: the reducer runs after the collective kernel tail.
        """

        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.config.accumulator_stages,
        )
        boundary = pipeline.NamedBarrier(
            barrier_id=self.config.barrier_id,
            num_threads=self.config.epilogue_warps * WARP_SIZE,
        )
        # Only the tile boundary spans both warpgroups. E1 can leave one
        # warpgroup with fewer groups (or none), so exchange barriers cannot.
        warpgroup_idx = warp_idx // Int32(4)
        exchange_boundary = boundary
        if cutlass.const_expr(self.config.epilogue_warps == 8):
            exchange_boundary = pipeline.NamedBarrier(
                barrier_id=Int32(EXCHANGE_BARRIER_IDS[0]) + warpgroup_idx,
                num_threads=4 * WARP_SIZE,
            )
        lane_idx = tidx % Int32(WARP_SIZE)
        fine_cta = cutlass.Boolean(False)
        if cutlass.const_expr(self.iket_fine_ctas > 0):
            fine_cta = is_fine_cta(self.iket_fine_ctas)
        if cutlass.const_expr(stats is not None):
            loop_start = globaltimer_lo()
        work = scheduler_consumer.consume_work()
        while work.is_valid_tile:
            # The scheduler's peek concerns input/readiness and never permits
            # the epilogue to consume an accumulator before its pipeline stage.
            if cutlass.const_expr(stats is not None):
                waited = globaltimer_lo()
            if cutlass.const_expr(self.iket):
                self._iket.range_push("epi_wait_acc")
            acc_pipeline.consumer_wait(consumer_state)
            if cutlass.const_expr(self.iket):
                self._iket.range_pop()
                if work.phase == Int32(DevicePhase.FC1):
                    self._iket.range_push("epi_fc1")
                else:
                    self._iket.range_push("epi_fc2")
            if cutlass.const_expr(stats is not None):
                tile_start = globaltimer_lo()
                if tidx == Int32(0):
                    stat_add(
                        stats_slot_ptr(stats, Slot.EPI_ACC_FULL_NS, stats_word_offset),
                        tile_start - waited,
                    )
            acc_stage = Int32(consumer_state.index)

            if work.phase == Int32(DevicePhase.FC1):
                self._run_fc1_task_tile(
                    work=work,
                    tmem_acc_tensor=tmem_acc_tensor,
                    acc_stage=acc_stage,
                    fc1_output=fc1_output,
                    fc1_output_scales=fc1_output_scales,
                    epilogue_exchange=epilogue_exchange,
                    boundary=exchange_boundary,
                    warp_idx=warp_idx,
                    lane_idx=lane_idx,
                    warpgroup_idx=warpgroup_idx,
                    fine_cta=fine_cta,
                )
            else:
                self._run_fc2_task_tile(
                    work=work,
                    tmem_acc_tensor=tmem_acc_tensor,
                    acc_stage=acc_stage,
                    route_store=route_store,
                    token_metadata=token_metadata,
                    epilogue_exchange=epilogue_exchange,
                    boundary=exchange_boundary,
                    warp_idx=warp_idx,
                    lane_idx=lane_idx,
                    warpgroup_idx=warpgroup_idx,
                )

            # Every LDTM has completed before the accumulator slot is handed
            # back.  Register math and global stores no longer depend on TMEM.
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(consumer_state)
            consumer_state.advance()

            # Tile end, then the FC1-done publication (warp 0 only: tid 0
            # issues the GPU-scope fence and release-add).
            if cutlass.const_expr(self.iket_fine_ctas > 0):
                if fine_cta and work.phase == Int32(DevicePhase.FC1):
                    self._iket.range_push("fc1_tile_bar")
            boundary.arrive_and_wait()
            if cutlass.const_expr(self.iket_fine_ctas > 0):
                if fine_cta and work.phase == Int32(DevicePhase.FC1):
                    self._iket.range_pop()
            if cutlass.const_expr(self.iket):
                self._iket.range_pop()
            if cutlass.const_expr(self.iket_fine_ctas > 0):
                if (
                    fine_cta
                    and work.phase == Int32(DevicePhase.FC1)
                    and warp_idx == Int32(0)
                ):
                    self._iket.range_push("fc1_publish")
            if tidx == Int32(0):
                cute.arch.fence_acq_rel_gpu()
                if work.phase == Int32(DevicePhase.FC1):
                    _red_add_release_gpu_s32(
                        fc1_done_counter.iterator + work.counter_slot,
                        Int32(1),
                    )
                if cutlass.const_expr(stats is not None):
                    # Accumulator ready -> publication: the FC2 readiness latency.
                    elapsed = globaltimer_lo() - tile_start
                    if work.phase == Int32(DevicePhase.FC1):
                        stat_add(
                            stats_slot_ptr(
                                stats, Slot.EPI_FC1_TILE_NS, stats_word_offset
                            ),
                            elapsed,
                        )
                        stat_add(
                            stats_slot_ptr(
                                stats, Slot.EPI_FC1_TILE_COUNT, stats_word_offset
                            ),
                            Int32(1),
                        )
                    else:
                        stat_add(
                            stats_slot_ptr(
                                stats, Slot.EPI_FC2_TILE_NS, stats_word_offset
                            ),
                            elapsed,
                        )
                        stat_add(
                            stats_slot_ptr(
                                stats, Slot.EPI_FC2_TILE_COUNT, stats_word_offset
                            ),
                            Int32(1),
                        )
            if cutlass.const_expr(self.iket_fine_ctas > 0):
                if (
                    fine_cta
                    and work.phase == Int32(DevicePhase.FC1)
                    and warp_idx == Int32(0)
                ):
                    self._iket.range_pop()

            work = scheduler_consumer.consume_work()
        if cutlass.const_expr(stats is not None):
            if tidx == Int32(0):
                stat_add(
                    stats_slot_ptr(stats, Slot.EPI_LOOP_NS, stats_word_offset),
                    globaltimer_lo() - loop_start,
                )


__all__ = [
    "DeviceEpilogueConfig",
    "EPILOGUE_WARPS",
    "PersistentMxfp8Mxfp4Epilogue",
    "TOKEN_METADATA_BYTES",
    "TMEM_ROW_STRIDE",
    "EXCHANGE_LANE_STRIDE",
    "EXCHANGE_ROWS",
    "EXCHANGE_WORDS",
    "EXCHANGE_BARRIER_IDS",
    "WARP_SIZE",
    "route_metadata_host",
]
