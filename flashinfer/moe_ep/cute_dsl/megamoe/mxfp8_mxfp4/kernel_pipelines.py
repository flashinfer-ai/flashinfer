# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared storage and pipeline assembly for the persistent SM100 kernel.

This module is the single owner of per-CTA control storage and pipeline
construction for the native MXFP8 x MXFP4 MegaMoE kernel.  It deliberately
does not issue TMA or MMA instructions and it does not choose work records;
those responsibilities remain in :mod:`scale_tma_staging`,
:mod:`dynamic_mainloop`, and :mod:`device_scheduler` respectively.

The supported bring-up geometry is the native one-CTA M128 instruction.  A
cluster may still contain multiple one-CTA instructions for multicast and
persistent scheduling; ``one-CTA`` here means that no two CTAs jointly own a
single tcgen05 instruction or TMEM allocation.

``PersistentMxfp8Mxfp4Epilogue`` is autonomous.  Consequently it is the
authority for accumulator stage count, TMEM accumulator extent, pipeline
consumer population, and accumulator retirement.  The driver must call
:meth:`run_epilogue` rather than constructing another accumulator consumer.

Do not add ``from __future__ import annotations`` to this file.  ``@cute.struct``
reads live annotation objects when the storage type is created.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import tcgen05

from .device_epilogue import (
    EPILOGUE_WARPS,
    EXCHANGE_BARRIER_IDS,
    PersistentMxfp8Mxfp4Epilogue,
)
from .device_scheduler import DeviceSchedulerParams, PersistentDeviceScheduler
from .dispatch import Mxfp8DispatchPhase
from .fc2_store import fc2_exchange_words
from .scale_tma_staging import Mxfp8Mxfp4TmaStaging
from .stage_fit import MX_BLOCK_SIZE, OperandStageBytes, StageFit


WARP_SIZE = 32


@dataclass(frozen=True)
class KernelWarpIds:
    """Stable warp ownership used by the persistent driver.

    The numbering matches the proven W4A4 persistent orchestration.  Keeping
    it here prevents the scheduler, dispatch phase, and TMEM allocator from
    silently using different populations.
    """

    epilogue: Tuple[int, ...] = (0, 1, 2, 3)
    mma: int = 4
    tma_a: int = 5
    tma_b: int = 6
    scheduler: int = 7
    dispatch: Tuple[int, ...] = (8, 9, 10, 11)

    def __post_init__(self) -> None:
        all_ids = (
            *self.epilogue,
            self.mma,
            self.tma_a,
            self.tma_b,
            self.scheduler,
            *self.dispatch,
        )
        if len(self.epilogue) not in (4, 8):
            raise ValueError("the autonomous epilogue requires four or eight warps")
        if len(self.dispatch) not in (0, 4):
            raise ValueError("lean uses zero dispatch warps; mega uses four")
        if min(all_ids) < 0 or len(set(all_ids)) != len(all_ids):
            raise ValueError("kernel warp IDs must be distinct non-negative values")
        if tuple(sorted(all_ids)) != tuple(range(max(all_ids) + 1)):
            raise ValueError("kernel warp IDs must form a dense zero-based range")
        expected_epilogue: Tuple[int, ...] = (0, 1, 2, 3)
        if len(self.epilogue) == 8:
            start = 8 + len(self.dispatch)
            expected_epilogue += tuple(range(start, start + 4))
        if (
            self.epilogue != expected_epilogue
            or (self.mma, self.tma_a, self.tma_b, self.scheduler) != (4, 5, 6, 7)
            or self.dispatch not in ((), (8, 9, 10, 11))
        ):
            raise ValueError("warp roles must preserve the role-pure warpgroup layout")

    @classmethod
    def for_epilogue(cls, epilogue_warps: int, *, mega: bool = True) -> "KernelWarpIds":
        """Append a role-pure epilogue warpgroup after the existing roles."""
        if epilogue_warps not in (4, 8):
            raise ValueError("epilogue_warps must be 4 or 8")
        start = 12 if mega else 8
        epilogue: Tuple[int, ...] = (0, 1, 2, 3)
        if epilogue_warps == 8:
            epilogue += tuple(range(start, start + 4))
        return cls(epilogue=epilogue, dispatch=(8, 9, 10, 11) if mega else ())

    @property
    def threads_per_cta(self) -> int:
        return WARP_SIZE * (
            max(
                (
                    *self.epilogue,
                    self.mma,
                    self.tma_a,
                    self.tma_b,
                    self.scheduler,
                    *self.dispatch,
                )
            )
            + 1
        )


@dataclass(frozen=True)
class OperandSmem:
    """The four staged operands allocated after the control struct."""

    weight: Any
    activation: Any
    weight_scale: Any
    activation_scale: Any


@dataclass(frozen=True)
class MainloopPipelines:
    """Shared A/B TMA-UMMA stage ring plus the accumulator pipeline."""

    ab_pipeline: Any
    ab_producer: Any
    ab_consumer: Any
    accumulator: Any


@dataclass(frozen=True)
class SchedulerBundle:
    """The scheduler-warp producer and a fresh mainloop consumer."""

    scheduler: Any
    consumer: Any


class KernelPipelines:
    """Assemble shared storage, mainloop pipelines, scheduler, and TMEM.

    Parameters are compile-time objects.  ``scheduler_params`` may contain
    device values, but its storage shape (mode and number of stages) must be
    fixed before tracing the kernel.
    """

    def __init__(
        self,
        *,
        staging: Mxfp8Mxfp4TmaStaging,
        scheduler_params: DeviceSchedulerParams,
        dispatch_phase: Optional[Mxfp8DispatchPhase],
        epilogue: PersistentMxfp8Mxfp4Epilogue,
        warp_ids: Optional[KernelWarpIds] = None,
        tmem_alloc_barrier_id: int = 2,
    ) -> None:
        if warp_ids is None:
            warp_ids = KernelWarpIds.for_epilogue(
                epilogue.config.epilogue_warps, mega=dispatch_phase is not None
            )
        if len(warp_ids.epilogue) != epilogue.config.epilogue_warps:
            raise ValueError("epilogue and KernelWarpIds disagree on warp count")
        if (dispatch_phase is None) != (not warp_ids.dispatch):
            raise ValueError("dispatch warp population must match communication mode")
        self.validate_mma_tiler_mn(staging.mma_tiler_mn)
        self.two_cta = staging.cta_group == tcgen05.CtaGroup.TWO
        if self.two_cta and tuple(staging.cluster_shape_mn) != (2, 1):
            raise ValueError("2-CTA MMA requires a (2, 1) cluster")
        if staging.sf_vec_size != 32:
            raise ValueError("MXFP8 x MXFP4 pipelines require block-32 scaling")
        if epilogue.config.cta_tile_features != staging.cta_tile_shape_mnk[0]:
            raise ValueError("epilogue feature tile must match the staging CTA tile")
        if epilogue.config.cta_tile_tokens != staging.cta_tile_shape_mnk[1]:
            raise ValueError("epilogue token tile must match the staging CTA tile")
        if dispatch_phase is not None and tuple(
            dispatch_phase.cluster_shape_mn
        ) != tuple(staging.cluster_shape_mn):
            raise ValueError("dispatch and TMA staging must use the same cluster shape")
        if (
            dispatch_phase is not None
            and dispatch_phase.dispatch_warp_start != warp_ids.dispatch[0]
        ):
            raise ValueError("dispatch_warp_start must match KernelWarpIds.dispatch")
        if dispatch_phase is not None and dispatch_phase.num_dispatch_warps != len(
            warp_ids.dispatch
        ):
            raise ValueError("dispatch phase and KernelWarpIds disagree on warp count")
        # MMA, TMA-A, TMA-B, and scheduler join the epilogue warps.
        expected_other_warps = len(warp_ids.epilogue) + 4
        if (
            dispatch_phase is not None
            and dispatch_phase.num_other_warps != expected_other_warps
        ):
            raise ValueError(
                "dispatch num_other_warps must count epilogue, MMA, both TMA, "
                "and scheduler warps"
            )
        if tmem_alloc_barrier_id < 0:
            raise ValueError("tmem_alloc_barrier_id must be non-negative")
        if tmem_alloc_barrier_id == epilogue.config.barrier_id:
            raise ValueError(
                "TMEM allocation and epilogue boundary barriers must differ"
            )
        if (
            len(warp_ids.epilogue) == 8
            and tmem_alloc_barrier_id in EXCHANGE_BARRIER_IDS
        ):
            raise ValueError("TMEM allocation and exchange barriers must differ")

        self.staging = staging
        self.scheduler_params = scheduler_params
        self.dispatch_phase = dispatch_phase
        self.epilogue = epilogue
        self.warp_ids = warp_ids
        self.tmem_alloc_barrier_id = tmem_alloc_barrier_id

        self.num_ab_stages = staging.num_stages
        self.num_acc_stages = epilogue.num_acc_pipeline_stages
        self.num_accumulator_tmem_cols = epilogue.num_accumulator_tmem_cols
        _, _, self.num_scale_tmem_cols = staging.tmem_column_counts()
        required_tmem_cols = self.num_accumulator_tmem_cols + self.num_scale_tmem_cols
        # tcgen05.alloc accepts only a power-of-two multiple of 32 columns.
        # Scale tensors still start immediately after the accumulator; the
        # remaining tail is intentional allocation padding.
        self.num_tmem_cols = max(32, 1 << (required_tmem_cols - 1).bit_length())
        if self.num_tmem_cols > 512:
            raise ValueError(
                f"persistent kernel requires {required_tmem_cols} TMEM columns; "
                "the rounded allocation exceeds the 512-column limit"
            )

        self._shared_storage_type: Optional[type] = None

    @staticmethod
    def validate_mma_tiler_mn(mma_tiler_mn: Tuple[int, int]) -> None:
        """Reject unsupported geometry before constructing CuTe layouts.

        ``Mxfp8Mxfp4TmaStaging`` requires an active MLIR context.  Launchers
        use this host-only guard first so an M256 tactic fails clearly rather
        than entering CuTe tracing and constructing a two-CTA MMA.
        """

        if len(mma_tiler_mn) != 2 or mma_tiler_mn[0] not in (128, 256):
            raise ValueError(
                "persistent kernel requires an M128 (1-CTA) or M256 (2-CTA) MMA tile"
            )

    @property
    def threads_per_cta(self) -> int:
        return self.warp_ids.threads_per_cta

    def shared_storage_type(self) -> type:
        """Return the sole control-storage struct allocated by the driver.

        Operand stage tensors are intentionally not fields of this struct:
        their swizzled layouts are allocated by :meth:`allocate_operand_smem`
        immediately afterwards.  All barrier and publication storage is here.
        """

        if self._shared_storage_type is None:
            self._shared_storage_type = self.build_shared_storage_type(
                num_ab_stages=self.num_ab_stages,
                num_acc_stages=self.num_acc_stages,
                scheduler_params=self.scheduler_params,
                dispatch_phase=self.dispatch_phase,
                epilogue_warps=len(self.warp_ids.epilogue),
            )
        return self._shared_storage_type

    @staticmethod
    def build_shared_storage_type(
        *,
        num_ab_stages: int,
        num_acc_stages: int,
        scheduler_params: Any,
        dispatch_phase: Optional[Mxfp8DispatchPhase],
        epilogue_warps: int = EPILOGUE_WARPS,
    ) -> type:
        """Compose the control-storage struct for a given operand stage count.

        Shared by the kernel and :func:`fit_operand_stages`, so the stage fit
        sizes exactly the struct the kernel allocates.  Only
        ``scheduler_params.num_stages`` and ``load_balance_mode`` shape the
        scheduler storage, and no MLIR context is required.
        """

        SchedulerStorage = PersistentDeviceScheduler.make_storage_struct(
            scheduler_params
        )
        num_operand_barriers = num_ab_stages * 2
        num_acc_barriers = num_acc_stages * 2
        if epilogue_warps not in (4, 8):
            raise ValueError("epilogue_warps must be 4 or 8")
        exchange_words = fc2_exchange_words(epilogue_warps)
        exchange_alignment = 16

        if dispatch_phase is None:

            @cute.struct
            class LeanSharedStorage:
                ab_full_mbar: cute.struct.MemRange[cutlass.Int64, num_operand_barriers]
                acc_full_mbar: cute.struct.MemRange[cutlass.Int64, num_acc_barriers]
                scheduler: SchedulerStorage  # type: ignore[valid-type]
                epilogue_exchange: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, exchange_words],
                    exchange_alignment,
                ]
                tmem_dealloc_mbar: cutlass.Int64
                tmem_holding_buf: cutlass.Int32

            return LeanSharedStorage

        DispatchStorage = dispatch_phase.shared_storage_type()

        @cute.struct
        class SharedStorage:
            ab_full_mbar: cute.struct.MemRange[cutlass.Int64, num_operand_barriers]
            acc_full_mbar: cute.struct.MemRange[cutlass.Int64, num_acc_barriers]
            scheduler: SchedulerStorage  # type: ignore[valid-type]
            dispatch: DispatchStorage  # type: ignore[valid-type]
            epilogue_exchange: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, exchange_words],
                exchange_alignment,
            ]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        return SharedStorage

    def allocate_operand_smem(self, smem) -> OperandSmem:
        """Allocate A/B/SFA/SFB stage tensors with their authoritative layouts."""

        weight = smem.allocate_tensor(
            element_type=self.staging.weight_smem_dtype,
            layout=self.staging.weight_smem_layout.outer,
            byte_alignment=128,
            swizzle=self.staging.weight_smem_layout.inner,
        )
        activation = smem.allocate_tensor(
            element_type=self.staging.activation_dtype,
            layout=self.staging.activation_smem_layout.outer,
            byte_alignment=128,
            swizzle=self.staging.activation_smem_layout.inner,
        )
        weight_scale = smem.allocate_tensor(
            element_type=self.staging.scale_dtype,
            layout=self.staging.sfa_smem_layout,
            byte_alignment=128,
        )
        activation_scale = smem.allocate_tensor(
            element_type=self.staging.scale_dtype,
            layout=self.staging.sfb_smem_layout,
            byte_alignment=128,
        )
        return OperandSmem(weight, activation, weight_scale, activation_scale)

    def create_mainloop_pipelines(self, storage) -> MainloopPipelines:
        """Create the shared AB TMA/UMMA and accumulator pipelines."""

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 2)
        num_mcast_a = cute.size(self.staging.cluster_layout_vmnk.shape[2])
        num_mcast_b = cute.size(self.staging.cluster_layout_vmnk.shape[1])
        num_tma_consumers = num_mcast_a + num_mcast_b - 1
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar.data_ptr(),
            num_stages=self.num_ab_stages,
            producer_group=producer_group,
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_tma_consumers
            ),
            tx_count=0,
            cta_layout_vmnk=self.staging.cluster_layout_vmnk,
            defer_sync=True,
        )

        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar.data_ptr(),
            num_stages=self.num_acc_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            # With a 2-CTA MMA both CTAs' epilogue warps release the leader's
            # accumulator stage.
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.warp_ids.epilogue) * WARP_SIZE * (2 if self.two_cta else 1),
            ),
            cta_layout_vmnk=self.staging.cluster_layout_vmnk,
            defer_sync=True,
        )
        # Participant state is staged SSA.  Construct both participants once,
        # before warp-role divergence, and carry that state through the entire
        # persistent loop exactly as the proven MegaMoE kernel does.
        ab_producer, ab_consumer = ab_pipeline.make_participants()
        return MainloopPipelines(ab_pipeline, ab_producer, ab_consumer, acc_pipeline)

    def create_scheduler(self, storage, *, block_idx, grid_dim) -> SchedulerBundle:
        """Create scheduler publication and one consumer for this warp role."""

        consumer_warps = (
            self.warp_ids.tma_a,
            self.warp_ids.tma_b,
            self.warp_ids.mma,
            *self.warp_ids.epilogue,
        )
        scheduler = PersistentDeviceScheduler.create(
            self.scheduler_params,
            block_idx,
            grid_dim,
            storage.scheduler,
            num_consumer_threads=len(consumer_warps) * WARP_SIZE,
        )
        return SchedulerBundle(scheduler, scheduler.make_consumer())

    def make_tmem_allocator(self, storage):
        """Build the TMEM allocator shared by MMA and epilogue warps."""

        alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_barrier_id,
            num_threads=(1 + len(self.warp_ids.epilogue)) * WARP_SIZE,
        )
        return cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=alloc_barrier,
            allocator_warp_id=self.warp_ids.epilogue[0],
            is_two_cta=self.two_cta,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

    def make_accumulator_tensor(self, tmem_base_ptr):
        """Return the staged FP32 accumulator tensor rooted at ``tmem_base_ptr``."""

        acc_shape = self.staging.tiled_mma.partition_shape_C(self.staging.mma_tiler[:2])
        prototype = self.staging.tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stages)
        )
        return cute.make_tensor(tmem_base_ptr, prototype.layout)

    def make_scale_tmem_tensors(self, tmem_base_ptr):
        """Place block-32 SFA/SFB immediately after accumulator columns."""

        return self.staging.make_tmem_scale_tensors(
            tmem_base_ptr, self.num_accumulator_tmem_cols
        )

    def make_mma_tmem_sfb(self, tmem_base_ptr, tile_n_idx, tmem_sfb):
        """SFB TMEM view shifted to ``tile_n_idx``'s rows within its atom."""

        return self.staging.make_mma_tmem_sfb(
            tmem_base_ptr, self.num_accumulator_tmem_cols, tile_n_idx, tmem_sfb
        )

    def run_epilogue(
        self,
        *,
        acc_pipeline,
        scheduler_consumer,
        **kwargs,
    ) -> None:
        """Delegate the complete accumulator consumer lifecycle.

        The explicit pipeline/consumer names keep the driver from passing an
        AB participant or scheduler producer by accident.  Remaining keyword
        arguments are the numerical/output tensors owned by the epilogue API.
        """

        self.epilogue.run(
            acc_pipeline=acc_pipeline,
            scheduler_consumer=scheduler_consumer,
            **kwargs,
        )


# Every operand tensor in :meth:`KernelPipelines.allocate_operand_smem` is
# 128-byte aligned and follows the control struct.
_OPERAND_SMEM_ALIGNMENT = 128


def mma_operand_stage_bytes(mma_tiler_mnk: Tuple[int, int, int]) -> OperandStageBytes:
    """Per-CTA shared-memory bytes of one A/B/SFA/SFB stage.

    FP4 weights occupy 8-bit shared-memory containers for the mixed-kind MMA,
    activations are FP8, and each operand carries one E8M0 byte per 32 K
    elements.  ``tests/moe_ep/test_mxfp8_mxfp4_kernel_pipelines.py`` checks these figures against
    :meth:`Mxfp8Mxfp4TmaStaging.operand_stage_bytes` inside a trace.
    """

    m, n, k = mma_tiler_mnk
    # A 2-CTA (M256) MMA splits both operands across the pair: each CTA
    # stages its own 128 weight rows and half of the token tile.  SFB is
    # staged whole in each CTA (1-CTA SFB view).
    cta_m, cta_n = (m // 2, n // 2) if m == 256 else (m, n)
    return OperandStageBytes(
        weight_a=cta_m * k,
        activation_b=cta_n * k,
        weight_sfa=cta_m * (k // MX_BLOCK_SIZE),
        # SFB is staged in whole 128-row atoms even for N64 tiles.
        activation_sfb=(n + 127) // 128 * 128 * (k // MX_BLOCK_SIZE),
    )


def fit_operand_stages(
    *,
    mma_tiler_mnk: Tuple[int, int, int],
    scheduler_params: Any,
    dispatch_phase: Optional[Mxfp8DispatchPhase],
    smem_capacity_bytes: int,
    num_acc_stages: int = 2,
    minimum_stages: int = 2,
    maximum_stages: int = 8,
    epilogue_warps: int = EPILOGUE_WARPS,
) -> StageFit:
    """Deepest operand pipeline whose dynamic shared memory fits one CTA.

    Models the kernel's allocation exactly: the control struct from
    :meth:`KernelPipelines.build_shared_storage_type` (its barrier array grows
    with the stage count), rounded up to the operand alignment, plus one
    operand stage per pipeline stage. Boundary tests compare this model
    with the actual launch limit.
    """

    operand = mma_operand_stage_bytes(mma_tiler_mnk)
    for stages in range(maximum_stages, minimum_stages - 1, -1):
        storage = KernelPipelines.build_shared_storage_type(
            num_ab_stages=stages,
            num_acc_stages=num_acc_stages,
            scheduler_params=scheduler_params,
            dispatch_phase=dispatch_phase,
            epilogue_warps=epilogue_warps,
        )
        fixed = -(-storage.__sizeof__() // _OPERAND_SMEM_ALIGNMENT) * (
            _OPERAND_SMEM_ALIGNMENT
        )
        if fixed + stages * operand.total <= smem_capacity_bytes:
            return StageFit(
                stages=stages,
                bytes_per_stage=operand.total,
                available_bytes=smem_capacity_bytes,
                fixed_bytes=fixed,
            )
    raise ValueError(
        "MXFP8 x MXFP4 mainloop does not fit: even "
        f"{minimum_stages} operand stages exceed {smem_capacity_bytes} bytes"
    )


__all__ = [
    "KernelPipelines",
    "KernelWarpIds",
    "MainloopPipelines",
    "OperandSmem",
    "SchedulerBundle",
    "WARP_SIZE",
    "fit_operand_stages",
    "mma_operand_stage_bytes",
]
