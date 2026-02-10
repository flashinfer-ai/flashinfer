# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This file contains shared utilities for CuTe DSL dense block-scaled GEMM kernels.
# The pipeline classes are derived from:
#   https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL/cutlass/pipeline
# The PDL helpers and pointer utilities are derived from:
#   https://github.com/NVIDIA/TensorRT-LLM (tensorrt_llm/_torch/cute_dsl_kernels/blackwell/utils.py)

import ctypes
from dataclasses import dataclass
from typing import Optional, Union

import cutlass
import cutlass._mlir.dialects.cute as _cute_ir
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import AddressSpace, Numeric, Pointer, Type
from cutlass.cutlass_dsl import Boolean, dsl_user_op, if_generate
from cutlass.pipeline import (
    CooperativeGroup,
    PipelineAsync,
    PipelineOp,
    PipelineState,
)


##############################################################################
# PDL (Programmatic Dependent Launch) helpers
##############################################################################


@dsl_user_op
def griddepcontrol_wait(*, loc=None, ip=None) -> None:
    """Wait for the previous kernel's grid to finish execution.

    This instruction ensures that the instruction following it will not be
    issued until the previous grid has finished and memory has been flushed.
    Used as the entry-point bookend for Programmatic Dependent Launch (PDL).
    """
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="griddepcontrol.wait;",
        constraints="",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def griddepcontrol_launch_dependents(*, loc=None, ip=None) -> None:
    """Hint to the hardware that a dependent kernel can start launching.

    Issuing this instruction allows a dependent kernel to launch earlier,
    overlapping the tail of this kernel with the start of the next.
    Used as the exit-point bookend for Programmatic Dependent Launch (PDL).
    """
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="griddepcontrol.launch_dependents;",
        constraints="",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


##############################################################################
# Pointer utility (WAR for CuTeDSL make_ptr implementation)
##############################################################################


class _Pointer(Pointer):
    """Runtime pointer that interoperates with CuTe DSL's JIT compilation.

    This is a workaround for the standard CuTeDSL ``make_ptr`` implementation,
    allowing creation of pointers from raw integer addresses (e.g., from
    ``torch.Tensor.data_ptr()``) with specified dtype and alignment.

    Args:
        pointer: Integer memory address.
        dtype: Data type of elements pointed to.
        mem_space: Memory address space (default: generic).
        assumed_align: Alignment in bytes (default: inferred from dtype).
    """

    def __init__(
        self,
        pointer,
        dtype,
        mem_space: _cute_ir.AddressSpace = _cute_ir.AddressSpace.generic,
        assumed_align=None,
    ):
        self._pointer = pointer
        self._dtype = dtype
        self._addr_space = mem_space

        if assumed_align is None:
            self._assumed_align = dtype.width // 8
        else:
            self._assumed_align = assumed_align

        self._desc = None
        self._c_pointer = None
        assert int(self._pointer) % self._assumed_align == 0, (
            f"pointer must be {self._assumed_align} bytes aligned"
        )

    def size_in_bytes(self) -> int:
        return ctypes.sizeof(ctypes.c_void_p(int(self._pointer)))

    def __get_mlir_types__(self):
        return [self.mlir_type]

    def __c_pointers__(self):
        if self._c_pointer is None:
            self._desc = ctypes.c_void_p(int(self._pointer))
            self._c_pointer = ctypes.addressof(self._desc)
        return [self._c_pointer]

    def __new_from_mlir_values__(self, values):
        assert len(values) == 1
        return values[0]

    @property
    def mlir_type(self) -> ir.Type:
        return _cute_ir.PtrType.get(
            self._dtype.mlir_type, self._addr_space, self._assumed_align
        )

    @property
    def dtype(self) -> Type[Numeric]:
        return self._dtype

    @property
    def memspace(self):
        return self._addr_space

    def align(self, min_align: int, *, loc=None, ip=None) -> Pointer:
        raise NotImplementedError("align is not supported in runtime")

    def verify(self, expected_py_type):
        if expected_py_type is Pointer or (
            isinstance(expected_py_type, ir.Value) and expected_py_type.ty is Pointer
        ):
            return True
        return False

    def __str__(self) -> str:
        return f"Ptr<0x{int(self._pointer):016x}@{self._addr_space}>"

    def __repr__(self):
        return self.__str__()


def make_ptr(
    dtype: Type[Numeric],
    value: Union[int, ctypes._Pointer],
    mem_space: AddressSpace = AddressSpace.generic,
    assumed_align=None,
) -> Pointer:
    """Create a CuTe pointer from a raw memory address.

    Args:
        dtype: Data type of the pointer elements.
        value: Memory address as an integer or ctypes pointer.
        mem_space: Memory address space (default: generic).
        assumed_align: Alignment in bytes (default: inferred from dtype).

    Returns:
        A CuTe-compatible pointer object.
    """
    if isinstance(value, int):
        address_value = value
    elif isinstance(value, ctypes._Pointer):
        address_value = ctypes.cast(value, ctypes.c_void_p).value
        assert address_value is not None, "Pointer address is None"
    else:
        raise TypeError(
            f"Expect int or ctypes.POINTER for value but got {type(value)=}"
        )

    return _Pointer(address_value, dtype, mem_space, assumed_align=assumed_align)


##############################################################################
# General utilities
##############################################################################


def is_power_of_2(x: int) -> bool:
    """Check if an integer is a positive power of 2."""
    return x > 0 and (x & (x - 1)) == 0


def ceil_div(a: int, b: int) -> int:
    """Integer ceiling division."""
    return (a + b - 1) // b


##############################################################################
# Scale factor layout conversion (for test/validation use)
##############################################################################


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """Convert a scale factor tensor from MKL layout to MMA-compatible layout.

    This function reorders a scale factor tensor from its natural (M, K, L)
    layout into the ``BlockScaledBasicChunk`` layout expected by tcgen05
    block-scaled MMA instructions: M(32x4xrest_m) x K(4xrest_k) x L.

    This is primarily used for test/validation purposes. During inference,
    scale factors should already be in the 128x4 layout produced by
    ``nvfp4_quantize`` with ``sfLayout=SfLayout.layout_128x4``.

    Args:
        sf_ref_tensor: Source tensor in MKL layout.
        sf_mma_tensor: Destination tensor in M(32x4xrest_m) x K(4xrest_k) x L layout.
    """
    # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]


##############################################################################
# Pipeline: PipelineTmaUmma
##############################################################################


def pipeline_init_wait(cta_layout_vmnk: Optional[cute.Layout] = None):
    """Fence mbarrier initialization to ensure proper synchronization.

    Places a fence on mbarrier initialization. Unlike the CUTLASS version
    which also handles cluster synchronization, this simplified version
    only performs the fence -- cluster sync is handled separately by the
    caller to support the single-cluster fallback path.

    Args:
        cta_layout_vmnk: The CTA layout (unused, kept for API compatibility).
    """
    cute.arch.mbarrier_init_fence()


@dataclass(frozen=True)
class PipelineTmaUmma(PipelineAsync):
    """Pipeline for TMA producers and UMMA consumers.

    Used in Blackwell mainloops where TMA (Tensor Memory Access) producers
    load data from global memory to shared memory, and UMMA (Universal Matrix
    Multiply Accumulate) consumers process the data via tcgen05.mma.

    Attributes:
        is_leader_cta: Whether the current CTA is the leader in a 2-CTA group.
        cta_group: The CTA group configuration (ONE or TWO).
    """

    is_leader_cta: bool
    cta_group: cute.nvgpu.tcgen05.CtaGroup

    @staticmethod
    def _compute_mcast_arrival_mask(
        cta_layout_vmnk: cute.Layout, mcast_mode_mn: tuple[int, int]
    ):
        """Compute a mask for signaling arrivals to multicasting threadblocks.

        Args:
            cta_layout_vmnk: Layout of the cluster shape.
            mcast_mode_mn: Tuple specifying multicast modes for M and N dimensions.
                At least one must be 1.

        Returns:
            Computed multicast mask for barrier arrivals.
        """
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        tma_mcast_mask_a = cute.nvgpu.cpasync.create_tma_multicast_mask(
            cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=2
        )
        tma_mcast_mask_b = cute.nvgpu.cpasync.create_tma_multicast_mask(
            cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=1
        )

        block_in_cluster_coord_vmnk_peer = (
            cta_in_cluster_coord_vmnk[0] ^ 1,
            *cta_in_cluster_coord_vmnk[1:],
        )
        tma_mcast_mask_a_peer = cute.nvgpu.cpasync.create_tma_multicast_mask(
            cta_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=2
        )
        tma_mcast_mask_b_peer = cute.nvgpu.cpasync.create_tma_multicast_mask(
            cta_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=1
        )

        assert not (mcast_mode_mn[0] == 0 and mcast_mode_mn[1] == 0)
        if mcast_mode_mn[0] == 1 and mcast_mode_mn[1] == 1:
            return (
                tma_mcast_mask_a
                | tma_mcast_mask_b
                | tma_mcast_mask_a_peer
                | tma_mcast_mask_b_peer
            )
        elif mcast_mode_mn[1] == 1:
            return tma_mcast_mask_b | tma_mcast_mask_b_peer
        assert mcast_mode_mn[0] == 1
        return tma_mcast_mask_a | tma_mcast_mask_a_peer

    @staticmethod
    def _compute_is_leader_cta(cta_layout_vmnk: cute.Layout):
        """Compute whether the current CTA is the leader in a 2-CTA group.

        For 1-CTA kernels, all threadblocks are leaders.

        Args:
            cta_layout_vmnk: Layout of the cluster shape.

        Returns:
            True if the current threadblock is a leader.
        """
        bidx, bidy, _ = cute.arch.block_idx()

        mma_coord_vmnk = (
            bidx % cute.size(cta_layout_vmnk, mode=[0]),
            bidx // cute.size(cta_layout_vmnk, mode=[0]),
            bidy,
            None,
        )
        return mma_coord_vmnk[0] == 0

    @staticmethod
    def create(
        *,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        tx_count: int,
        barrier_storage: cute.Pointer = None,
        cta_layout_vmnk: Optional[cute.Layout] = None,
        mcast_mode_mn: tuple[int, int] = (1, 1),
    ):
        """Create and initialize a PipelineTmaUmma instance.

        Args:
            num_stages: Number of pipeline buffer stages.
            producer_group: CooperativeGroup for the TMA producer.
            consumer_group: CooperativeGroup for the UMMA consumer.
            tx_count: Bytes expected per transaction barrier for one stage.
            barrier_storage: Pointer to shared memory for this pipeline's mbarriers.
            cta_layout_vmnk: Layout of the cluster shape (None for single-CTA).
            mcast_mode_mn: Multicast modes for M and N dimensions.

        Returns:
            An initialized PipelineTmaUmma instance.
        """
        if not isinstance(barrier_storage, cute.Pointer):
            raise ValueError(
                f"Expected barrier_storage to be a cute.Pointer, but got {type(barrier_storage)}"
            )

        producer_type = PipelineOp.TmaLoad
        consumer_type = PipelineOp.TCGen05Mma

        producer = (producer_type, producer_group)
        consumer = (consumer_type, consumer_group)

        sync_object_full = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer, tx_count
        )
        sync_object_empty = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages, num_stages, consumer
        )

        if cta_layout_vmnk is None or cute.size(cta_layout_vmnk) == 1:
            # No mcast mask if not using clusters
            producer_mask = None
            # All threadblocks are leaders if not using clusters
            is_leader_cta = True
        else:
            producer_mask = PipelineTmaUmma._compute_mcast_arrival_mask(
                cta_layout_vmnk, mcast_mode_mn
            )
            is_leader_cta = PipelineTmaUmma._compute_is_leader_cta(cta_layout_vmnk)

        cta_group = (
            cute.nvgpu.tcgen05.CtaGroup.ONE
            if cta_layout_vmnk is None or cute.size(cta_layout_vmnk, mode=[0]) == 1
            else cute.nvgpu.tcgen05.CtaGroup.TWO
        )

        consumer_mask = producer_mask

        pipeline_init_wait(cta_layout_vmnk)

        return PipelineTmaUmma(
            sync_object_full,
            sync_object_empty,
            num_stages,
            producer_mask,
            consumer_mask,
            is_leader_cta,
            cta_group,
        )

    def consumer_release(self, state: PipelineState):
        """UMMA consumer release: signal that a buffer is empty.

        Args:
            state: Current pipeline state.
        """
        self.sync_object_empty.arrive(state.index, self.consumer_mask, self.cta_group)

    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
    ):
        """TMA producer acquire: conditionally wait for buffer empty, then set transaction barrier.

        For leader threadblocks, also sets the transaction barrier for the next
        TMA load operation.

        Args:
            state: Current pipeline state.
            try_acquire_token: Optional token from a prior ``producer_try_acquire``.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase),
        )
        if_generate(
            self.is_leader_cta,
            lambda: self.sync_object_full.arrive(state.index, self.producer_mask),
        )

    def producer_commit(self, state: PipelineState):
        """TMA producer commit: no-op since TMA automatically updates the transaction count.

        Args:
            state: Current pipeline state.
        """


##############################################################################
# Pipeline: PipelineUmmaAsync
##############################################################################


@dataclass(frozen=True)
class PipelineUmmaAsync(PipelineAsync):
    """Pipeline for UMMA producers and async thread consumers.

    Used for the accumulator pipeline where the MMA warp (UMMA producer)
    signals that the accumulator is ready, and epilogue warps (async thread
    consumers) drain the accumulator to global memory.

    Attributes:
        cta_group: The CTA group configuration (ONE or TWO).
    """

    cta_group: cute.nvgpu.tcgen05.CtaGroup

    @staticmethod
    def _compute_tmem_sync_mask(cta_layout_vmnk: cute.Layout):
        """Compute TMEM synchronization mask for cluster coordination.

        Args:
            cta_layout_vmnk: Layout of the cluster shape.

        Returns:
            TMEM sync mask.
        """
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        return cute.make_layout_image_mask(
            cta_layout_vmnk, cta_in_cluster_coord_vmnk, mode=0
        )

    @staticmethod
    def _compute_peer_cta_rank():
        """Compute the peer CTA rank for 2-CTA synchronization.

        Returns:
            The rank of the peer CTA in the cluster.
        """
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        return cta_rank_in_cluster // 2 * 2

    @staticmethod
    def create(
        *,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        barrier_storage: cute.Pointer = None,
        cta_layout_vmnk: Optional[cute.Layout] = None,
    ):
        """Create and initialize a PipelineUmmaAsync instance.

        Args:
            num_stages: Number of pipeline buffer stages.
            producer_group: CooperativeGroup for the UMMA producer (MMA warp).
            consumer_group: CooperativeGroup for the async consumer (epilogue warps).
            barrier_storage: Pointer to shared memory for this pipeline's mbarriers.
            cta_layout_vmnk: Layout of the cluster shape (None for single-CTA).

        Returns:
            An initialized PipelineUmmaAsync instance.
        """
        if not isinstance(barrier_storage, cute.Pointer):
            raise ValueError(
                f"Expected barrier_storage to be a cute.Pointer, but got {type(barrier_storage)}"
            )

        producer_type = PipelineOp.TCGen05Mma
        consumer_type = PipelineOp.AsyncThread

        producer = (producer_type, producer_group)
        consumer = (consumer_type, consumer_group)

        sync_object_full = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer
        )
        sync_object_empty = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages, num_stages, consumer
        )

        if cta_layout_vmnk is None or cute.size(cta_layout_vmnk) == 1:
            # Set mask to None if not using clusters (i.e. 1CTA kernels)
            producer_mask = None
        else:
            producer_mask = PipelineUmmaAsync._compute_tmem_sync_mask(cta_layout_vmnk)

        if cta_layout_vmnk is None or cute.size(cta_layout_vmnk, mode=[0]) == 1:
            # Set mask to None if not using 2CTA instructions
            consumer_mask = None
        else:
            consumer_mask = PipelineUmmaAsync._compute_peer_cta_rank()

        cta_group = (
            cute.nvgpu.tcgen05.CtaGroup.ONE
            if cta_layout_vmnk is None or cute.size(cta_layout_vmnk, mode=[0]) == 1
            else cute.nvgpu.tcgen05.CtaGroup.TWO
        )

        pipeline_init_wait(cta_layout_vmnk)

        return PipelineUmmaAsync(
            sync_object_full,
            sync_object_empty,
            num_stages,
            producer_mask,
            consumer_mask,
            cta_group,
        )

    def producer_commit(self, state: PipelineState):
        """UMMA producer commit: signal that the accumulator buffer is full.

        Args:
            state: Current pipeline state.
        """
        self.sync_object_full.arrive(state.index, self.producer_mask, self.cta_group)

    def producer_tail(self, state: PipelineState):
        """UMMA producer tail: drain remaining pipeline stages at the end.

        Only the leader CTA performs this operation in 2-CTA mode.

        Args:
            state: Current pipeline state.
        """
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        is_leader_cta = cta_rank_in_cluster % 2 == 0

        def then_body():
            # Advance to the last used buffer (num_stages - 1 times)
            for _i in range(self.num_stages - 1):
                state.advance()
            self.producer_acquire(state)

        if_generate(is_leader_cta, then_body)
