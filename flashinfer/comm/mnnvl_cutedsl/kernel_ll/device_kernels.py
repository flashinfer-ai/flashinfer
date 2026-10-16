# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Split low-latency BF16 MoE finalize, TP reduction, and RMSNorm."""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from ..cute_dsl_primitives import (
    NEGATIVE_ZERO_BF16_BITS,
    QUAD_BF16,
    VEC_BF16,
    WARP_SIZE,
    bf16x4_to_packed_u32x2,
    bf16x8_to_packed_u32x4,
    bulk_store_shared_to_global,
    bulk_wait_group_read,
    f32_to_bf16_bits,
    fence_proxy_async_shared_cta,
    fragment_has_negative_zero,
    load_global_bf16_as_f32,
    load_global_bf16_as_f32_predicated,
    load_global_u32x2,
    load_global_u32x4,
    load_volatile_u32,
    map_shared_to_peer,
    packed_u32x2_to_bf16x4,
    packed_u32x4_to_bf16x8,
    sanitize_negative_zero_u32x2,
    sanitize_negative_zero_u32x4,
    shuffle_sync_idx_u32,
    stmc_bf16x2,
    stmc_bf16x4,
    stmc_bf16x8,
    store_global_u32,
    store_global_u32x4,
    store_lamport_sentinel_u32x4,
    store_shared_cluster_f32,
    store_shared_u32x4,
)
from ..dynamic_fp8_epilogue import store_dynamic_fp8_cluster_row
from ..static_fp8_epilogue import store_rmsnorm_output

LAMPORT_GENERATIONS = 3
NEXT_STAGE = 0
ACTIVE_STAGE = 1


@cute.jit
def _group_leader_block_sum(
    value: Float32,
    warp_sums: cute.Tensor,
    warps: cutlass.Constexpr[int],
    leader_stride: cutlass.Constexpr[int],
) -> Float32:
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    for offset in cutlass.range_constexpr(1, WARP_SIZE):
        if cutlass.const_expr(offset >= leader_stride and (offset & (offset - 1)) == 0):
            value = value + cute.arch.shuffle_sync_bfly(
                value,
                offset=offset,
                mask=-1,
                mask_and_clamp=31,
            )
    if lane == 0:
        cute.arch.store((warp_sums + warp).llvm_ptr, value)
    cute.arch.barrier()

    result = Float32(0.0)
    if warp == 0:
        if lane < Int32(warps):
            result = cute.arch.load(
                (warp_sums + lane).llvm_ptr,
                Float32,
            )
        result = cute.arch.warp_reduction_sum(result)
        if lane == 0:
            cute.arch.store(warp_sums.llvm_ptr, result)
    cute.arch.barrier()
    return cute.arch.load(warp_sums.llvm_ptr, Float32)


class _ScalarFinalizePublishDeviceKernel:
    def __init__(
        self,
        *,
        hidden: int,
        top_k: int,
        tp: int,
        rank: int,
        capacity_m: int,
        threads: int,
        routed_scaling_factor: float,
        include_shared_expert: bool,
        load_shared_expert_before_pdl: bool,
        enable_pdl: bool,
        prefetch_group: int,
    ) -> None:
        if hidden <= 0 or hidden % 2:
            raise ValueError("hidden must be a positive multiple of 2")
        self.hidden = hidden
        self.top_k = top_k
        self.tp = tp
        self.rank = rank
        self.capacity_m = capacity_m
        self.threads = threads
        self.routed_scaling_factor = routed_scaling_factor
        self.ctas_per_token = math.ceil(hidden / threads)
        self.include_shared_expert = include_shared_expert
        self.load_shared_expert_before_pdl = load_shared_expert_before_pdl
        self.enable_pdl = enable_pdl
        self.prefetch_group = prefetch_group
        self.prefetch_groups = (top_k + prefetch_group - 1) // prefetch_group

    def smem_size_in_bytes(self) -> int:
        return self.top_k * 8

    @cute.jit
    def __call__(
        self,
        routed_output: cute.Tensor,
        expert_weights: cute.Tensor,
        permuted_indices: cute.Tensor,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_multicast_address: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            routed_output,
            expert_weights,
            permuted_indices,
            shared_output,
            stage_state,
            contribution_mailbox_multicast_address,
        ).launch(
            grid=(m * self.ctas_per_token, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.smem_size_in_bytes(),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        routed_output: cute.Tensor,
        expert_weights: cute.Tensor,
        permuted_indices: cute.Tensor,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_multicast_address: Int64,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        token = block // self.ctas_per_token
        cta_in_token = block % self.ctas_per_token
        hidden_index = cta_in_token * self.threads + tidx

        smem = cutlass.utils.SmemAllocator()
        staged_indices = smem.allocate_array(Int32, self.top_k)
        staged_weights = smem.allocate_array(Float32, self.top_k)
        metadata_index = Int32(tidx)
        while metadata_index < Int32(self.top_k):
            element = Int64(token) * self.top_k + Int64(metadata_index)
            row = cute.arch.load(
                (permuted_indices.iterator + element).llvm_ptr,
                Int32,
            )
            weight = load_global_bf16_as_f32(
                Int64((expert_weights.iterator + element).toint())
            )
            if cutlass.const_expr(self.routed_scaling_factor != 1.0):
                weight = weight * Float32(self.routed_scaling_factor)
            if row == Int32(-1):
                weight = Float32(0.0)
            cute.arch.store(
                (staged_indices + metadata_index).llvm_ptr,
                row,
            )
            cute.arch.store(
                (staged_weights + metadata_index).llvm_ptr,
                weight,
            )
            metadata_index = metadata_index + self.threads
        cute.arch.barrier()

        shared_value = Float32(0.0)
        if cutlass.const_expr(
            self.include_shared_expert and self.load_shared_expert_before_pdl
        ):
            if hidden_index < Int32(self.hidden):
                shared_element = Int64(token) * self.hidden + Int64(hidden_index)
                shared_value = load_global_bf16_as_f32(
                    Int64((shared_output.iterator + shared_element).toint())
                )

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        if cutlass.const_expr(
            self.include_shared_expert and not self.load_shared_expert_before_pdl
        ):
            if hidden_index < Int32(self.hidden):
                shared_element = Int64(token) * self.hidden + Int64(hidden_index)
                shared_value = load_global_bf16_as_f32(
                    Int64((shared_output.iterator + shared_element).toint())
                )

        stage = load_volatile_u32(stage_state.iterator + NEXT_STAGE)
        bits = Uint32(0)
        if hidden_index < Int32(self.hidden):
            accumulator = Float32(0.0)
            if cutlass.const_expr(self.prefetch_group == 1):
                for k in cutlass.range_constexpr(self.top_k):
                    row = cute.arch.load((staged_indices + k).llvm_ptr, Int32)
                    weight = cute.arch.load(
                        (staged_weights + k).llvm_ptr,
                        Float32,
                    )
                    source_element = Int64(row) * self.hidden + Int64(hidden_index)
                    accumulator = (
                        accumulator
                        + load_global_bf16_as_f32_predicated(
                            Int64((routed_output.iterator + source_element).toint()),
                            Int32(row != Int32(-1)),
                        )
                        * weight
                    )
            else:
                inputs = cute.make_rmem_tensor(
                    cute.make_layout((self.prefetch_group,)),
                    Float32,
                )
                inputs.fill(Float32(0.0))
                for group in cutlass.range_constexpr(self.prefetch_groups):
                    for item in cutlass.range_constexpr(self.prefetch_group):
                        k = group * self.prefetch_group + item
                        if cutlass.const_expr(k < self.top_k):
                            row = cute.arch.load(
                                (staged_indices + k).llvm_ptr,
                                Int32,
                            )
                            source_element = Int64(row) * self.hidden + Int64(
                                hidden_index
                            )
                            inputs[item] = load_global_bf16_as_f32_predicated(
                                Int64(
                                    (routed_output.iterator + source_element).toint()
                                ),
                                Int32(row != Int32(-1)),
                            )
                    for item in cutlass.range_constexpr(self.prefetch_group):
                        k = group * self.prefetch_group + item
                        if cutlass.const_expr(k < self.top_k):
                            accumulator = accumulator + inputs[item] * cute.arch.load(
                                (staged_weights + k).llvm_ptr,
                                Float32,
                            )
            if cutlass.const_expr(self.include_shared_expert):
                accumulator = accumulator + shared_value
            bits = f32_to_bf16_bits(accumulator)
            if bits == Uint32(NEGATIVE_ZERO_BF16_BITS):
                bits = Uint32(0)

        # Unlike the quad path, this publishes first; Lamport sentinels gate consumers.
        if block == 0 and tidx == 0:
            store_global_u32(
                stage_state.iterator + ACTIVE_STAGE,
                stage,
            )
        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        lane = cute.arch.lane_idx()
        partner_bits = shuffle_sync_idx_u32(
            bits,
            Int32(lane | Int32(1)),
        )
        if (lane & Int32(1)) == Int32(0) and hidden_index < Int32(self.hidden):
            packed = bits | (partner_bits << Uint32(16))
            mailbox_element = (
                (Int64(stage) * self.tp + self.rank) * self.capacity_m + Int64(token)
            ) * self.hidden + Int64(hidden_index)
            stmc_bf16x2(
                contribution_mailbox_multicast_address + mailbox_element * 2,
                packed,
            )


class _QuadFinalizePublishDeviceKernel:
    def __init__(
        self,
        *,
        hidden: int,
        top_k: int,
        tp: int,
        rank: int,
        capacity_m: int,
        threads: int,
        routed_scaling_factor: float,
        include_shared_expert: bool,
        load_shared_expert_before_pdl: bool,
        enable_pdl: bool,
        prefetch_group: int,
    ) -> None:
        if hidden <= 0 or hidden % QUAD_BF16:
            raise ValueError("hidden must be a positive multiple of 4")
        self.hidden = hidden
        self.top_k = top_k
        self.tp = tp
        self.rank = rank
        self.capacity_m = capacity_m
        self.threads = threads
        self.routed_scaling_factor = routed_scaling_factor
        self.fragments = hidden // QUAD_BF16
        self.ctas_per_token = math.ceil(self.fragments / threads)
        self.include_shared_expert = include_shared_expert
        self.load_shared_expert_before_pdl = load_shared_expert_before_pdl
        self.enable_pdl = enable_pdl
        self.prefetch_group = prefetch_group
        self.prefetch_groups = (top_k + prefetch_group - 1) // prefetch_group

    def smem_size_in_bytes(self) -> int:
        return self.top_k * 8

    @cute.jit
    def __call__(
        self,
        routed_output: cute.Tensor,
        expert_weights: cute.Tensor,
        permuted_indices: cute.Tensor,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_multicast_address: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            routed_output,
            expert_weights,
            permuted_indices,
            shared_output,
            stage_state,
            contribution_mailbox_multicast_address,
        ).launch(
            grid=(m * self.ctas_per_token, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.smem_size_in_bytes(),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        routed_output: cute.Tensor,
        expert_weights: cute.Tensor,
        permuted_indices: cute.Tensor,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_multicast_address: Int64,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        token = block // self.ctas_per_token
        cta_in_token = block % self.ctas_per_token
        fragment = cta_in_token * self.threads + tidx

        smem = cutlass.utils.SmemAllocator()
        staged_indices = smem.allocate_array(Int32, self.top_k)
        staged_weights = smem.allocate_array(Float32, self.top_k)
        metadata_index = Int32(tidx)
        while metadata_index < Int32(self.top_k):
            element = Int64(token) * self.top_k + Int64(metadata_index)
            row = cute.arch.load(
                (permuted_indices.iterator + element).llvm_ptr,
                Int32,
            )
            weight = load_global_bf16_as_f32(
                Int64((expert_weights.iterator + element).toint())
            )
            if cutlass.const_expr(self.routed_scaling_factor != 1.0):
                weight = weight * Float32(self.routed_scaling_factor)
            if row == Int32(-1):
                weight = Float32(0.0)
            cute.arch.store(
                (staged_indices + metadata_index).llvm_ptr,
                row,
            )
            cute.arch.store(
                (staged_weights + metadata_index).llvm_ptr,
                weight,
            )
            metadata_index = metadata_index + self.threads
        cute.arch.barrier()

        if cutlass.const_expr(self.include_shared_expert):
            shared_values = cute.make_rmem_tensor(
                cute.make_layout((QUAD_BF16,)), BFloat16
            )
            shared_values.fill(BFloat16(0.0))
            if cutlass.const_expr(self.load_shared_expert_before_pdl):
                if fragment < self.fragments:
                    shared_element = (
                        Int64(token) * self.hidden + Int64(fragment) * QUAD_BF16
                    )
                    shared_pointer = cute.make_ptr(
                        BFloat16,
                        (shared_output.iterator + shared_element).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=8,
                    )
                    shared_values.store(
                        packed_u32x2_to_bf16x4(load_global_u32x2(shared_pointer))
                    )

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        if cutlass.const_expr(
            self.include_shared_expert and not self.load_shared_expert_before_pdl
        ):
            if fragment < self.fragments:
                shared_element = (
                    Int64(token) * self.hidden + Int64(fragment) * QUAD_BF16
                )
                shared_pointer = cute.make_ptr(
                    BFloat16,
                    (shared_output.iterator + shared_element).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=8,
                )
                shared_values.store(
                    packed_u32x2_to_bf16x4(load_global_u32x2(shared_pointer))
                )

        stage = load_volatile_u32(stage_state.iterator + NEXT_STAGE)
        if fragment < self.fragments:
            accumulator = cute.make_rmem_tensor(cute.make_layout((QUAD_BF16,)), Float32)
            accumulator.fill(Float32(0.0))
            if cutlass.const_expr(self.prefetch_group == 1):
                for k in cutlass.range_constexpr(self.top_k):
                    row = cute.arch.load((staged_indices + k).llvm_ptr, Int32)
                    weight = cute.arch.load(
                        (staged_weights + k).llvm_ptr,
                        Float32,
                    )
                    if row != Int32(-1):
                        source_element = (
                            Int64(row) * self.hidden + Int64(fragment) * QUAD_BF16
                        )
                        source_pointer = cute.make_ptr(
                            BFloat16,
                            (routed_output.iterator + source_element).llvm_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=8,
                        )
                        accumulator.store(
                            accumulator.load()
                            + packed_u32x2_to_bf16x4(
                                load_global_u32x2(source_pointer)
                            ).to(Float32)
                            * weight
                        )
            else:
                inputs = cute.make_rmem_tensor(
                    cute.make_layout((self.prefetch_group, 2)),
                    Uint32,
                )
                inputs.fill(Uint32(0))
                for group in cutlass.range_constexpr(self.prefetch_groups):
                    for item in cutlass.range_constexpr(self.prefetch_group):
                        k = group * self.prefetch_group + item
                        if cutlass.const_expr(k < self.top_k):
                            row = cute.arch.load(
                                (staged_indices + k).llvm_ptr,
                                Int32,
                            )
                            for word in cutlass.range_constexpr(2):
                                inputs[item, word] = Uint32(0)
                            if row != Int32(-1):
                                source_element = (
                                    Int64(row) * self.hidden
                                    + Int64(fragment) * QUAD_BF16
                                )
                                source_pointer = cute.make_ptr(
                                    BFloat16,
                                    (routed_output.iterator + source_element).llvm_ptr,
                                    cute.AddressSpace.gmem,
                                    assumed_align=8,
                                )
                                source = load_global_u32x2(source_pointer)
                                for word in cutlass.range_constexpr(2):
                                    inputs[item, word] = source[word]
                    for item in cutlass.range_constexpr(self.prefetch_group):
                        k = group * self.prefetch_group + item
                        if cutlass.const_expr(k < self.top_k):
                            source = cute.make_rmem_tensor(
                                cute.make_layout((2,)),
                                Uint32,
                            )
                            for word in cutlass.range_constexpr(2):
                                source[word] = inputs[item, word]
                            accumulator.store(
                                accumulator.load()
                                + packed_u32x2_to_bf16x4(source.load()).to(Float32)
                                * cute.arch.load(
                                    (staged_weights + k).llvm_ptr,
                                    Float32,
                                )
                            )
            result = accumulator.load()
            if cutlass.const_expr(self.include_shared_expert):
                result = result + shared_values.load().to(Float32)
            result_packed = bf16x4_to_packed_u32x2(result.to(BFloat16))
            packed = sanitize_negative_zero_u32x2(result_packed)
            mailbox_element = (
                (Int64(stage) * self.tp + self.rank) * self.capacity_m + Int64(token)
            ) * self.hidden + Int64(fragment) * QUAD_BF16
            stmc_bf16x4(
                contribution_mailbox_multicast_address + mailbox_element * 2,
                packed,
            )

        if block == 0 and tidx == 0:
            store_global_u32(
                stage_state.iterator + ACTIVE_STAGE,
                stage,
            )
        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


class _SharedOnlyPublishDeviceKernel:
    def __init__(
        self,
        *,
        hidden: int,
        tp: int,
        rank: int,
        capacity_m: int,
        elements_per_thread: int,
        threads: int,
        release_before_store: bool,
        enable_pdl: bool,
        peer_stores: bool = False,
        multicast_fragments: int = 0,
        bulk_peer_stores: bool = False,
    ) -> None:
        if elements_per_thread not in (1, QUAD_BF16, VEC_BF16):
            raise ValueError("elements_per_thread must be 1, 4, or 8")
        if peer_stores and elements_per_thread != VEC_BF16:
            raise ValueError("peer_stores requires elements_per_thread == 8")
        if hidden <= 0 or hidden % elements_per_thread:
            raise ValueError("hidden must divide evenly across thread fragments")
        self.hidden = hidden
        self.tp = tp
        self.rank = rank
        self.capacity_m = capacity_m
        self.elements_per_thread = elements_per_thread
        self.threads = threads
        self.fragments = hidden // elements_per_thread
        self.ctas_per_token = math.ceil(self.fragments / threads)
        self.release_before_store = release_before_store
        self.enable_pdl = enable_pdl
        # Store to every rank's mailbox through its unicast mapping instead of
        # one multicast store; the local copy goes through this rank's own
        # entry of peer_mailboxes, which maps local memory, so it does not
        # cross NVLink. Each rank walks its peers starting from rank + 1 so
        # that the ranks' first stores land on different GPUs. The first multicast_fragments
        # fragments of a token keep the multicast store, which splits the
        # publish between the switch's multicast replication and the unicast
        # paths.
        self.peer_stores = peer_stores
        self.multicast_fragments = multicast_fragments if peer_stores else 0
        # Stage the CTA's unicast fragments in shared memory and send each
        # peer one bulk copy instead of one 16-byte store per thread.
        self.bulk_peer_stores = bulk_peer_stores and peer_stores
        if self.bulk_peer_stores and multicast_fragments % threads:
            raise ValueError("multicast_fragments must be a multiple of threads")

    def smem_size_in_bytes(self) -> int:
        return self.threads * 16 + 128 if self.bulk_peer_stores else 0

    @cute.jit
    def __call__(
        self,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_multicast_address: Int64,
        peer_mailboxes: cute.Tensor | None,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            shared_output,
            stage_state,
            contribution_mailbox_multicast_address,
            peer_mailboxes,
        ).launch(
            grid=(m * self.ctas_per_token, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.smem_size_in_bytes(),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_multicast_address: Int64,
        peer_mailboxes: cute.Tensor | None,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        token = block // self.ctas_per_token
        cta_in_token = block % self.ctas_per_token
        fragment = cta_in_token * self.threads + tidx
        if cutlass.const_expr(self.bulk_peer_stores):
            smem = cutlass.utils.SmemAllocator()
            staging = smem.allocate_array(Uint32, self.threads * 4, byte_alignment=128)

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        if cutlass.const_expr(self.elements_per_thread == 1):
            bits = Uint32(0)
            if fragment < self.fragments:
                element = Int64(token) * self.hidden + Int64(fragment)
                bits = f32_to_bf16_bits(
                    load_global_bf16_as_f32(
                        Int64((shared_output.iterator + element).toint())
                    )
                )
                if bits == Uint32(NEGATIVE_ZERO_BF16_BITS):
                    bits = Uint32(0)
            partner_bits = shuffle_sync_idx_u32(
                bits,
                Int32(cute.arch.lane_idx() | Int32(1)),
            )
            packed_word = bits | (partner_bits << Uint32(16))
        elif cutlass.const_expr(self.elements_per_thread == QUAD_BF16):
            packed_words = cute.make_rmem_tensor(cute.make_layout((2,)), Uint32)
            packed_words.fill(Uint32(0))
            if fragment < self.fragments:
                element = Int64(token) * self.hidden + Int64(fragment) * QUAD_BF16
                pointer = cute.make_ptr(
                    BFloat16,
                    (shared_output.iterator + element).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=8,
                )
                packed_words.store(
                    sanitize_negative_zero_u32x2(load_global_u32x2(pointer))
                )
        else:
            packed_words = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
            packed_words.fill(Uint32(0))
            if fragment < self.fragments:
                element = Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
                pointer = cute.make_ptr(
                    BFloat16,
                    (shared_output.iterator + element).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                packed_words.store(
                    sanitize_negative_zero_u32x4(load_global_u32x4(pointer))
                )

        stage = load_volatile_u32(stage_state.iterator + NEXT_STAGE)
        if cutlass.const_expr(self.release_before_store):
            if block == 0 and tidx == 0:
                store_global_u32(
                    stage_state.iterator + ACTIVE_STAGE,
                    stage,
                )
            cute.arch.barrier()
            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_launch_dependents()

        if cutlass.const_expr(self.elements_per_thread == 1):
            if (cute.arch.lane_idx() & Int32(1)) == Int32(
                0
            ) and fragment < self.fragments:
                mailbox_element = (
                    (Int64(stage) * self.tp + self.rank) * self.capacity_m
                    + Int64(token)
                ) * self.hidden + Int64(fragment)
                stmc_bf16x2(
                    contribution_mailbox_multicast_address + mailbox_element * 2,
                    packed_word,
                )
        elif cutlass.const_expr(self.elements_per_thread == QUAD_BF16):
            if fragment < self.fragments:
                mailbox_element = (
                    (Int64(stage) * self.tp + self.rank) * self.capacity_m
                    + Int64(token)
                ) * self.hidden + Int64(fragment) * QUAD_BF16
                stmc_bf16x4(
                    contribution_mailbox_multicast_address + mailbox_element * 2,
                    packed_words,
                )
        else:
            if fragment < self.fragments:
                mailbox_element = (
                    (Int64(stage) * self.tp + self.rank) * self.capacity_m
                    + Int64(token)
                ) * self.hidden + Int64(fragment) * VEC_BF16
                if cutlass.const_expr(self.bulk_peer_stores):
                    if cta_in_token * self.threads < self.multicast_fragments:
                        stmc_bf16x8(
                            contribution_mailbox_multicast_address
                            + mailbox_element * 2,
                            packed_words,
                        )
                elif cutlass.const_expr(self.peer_stores):
                    unicast = True
                    if cutlass.const_expr(self.multicast_fragments > 0):
                        unicast = fragment >= self.multicast_fragments
                    if unicast:
                        for step in cutlass.range_constexpr(1, self.tp + 1):
                            destination = (self.rank + step) % self.tp
                            peer_base = cute.arch.load(
                                (peer_mailboxes.iterator + destination).llvm_ptr,
                                Int64,
                            )
                            store_global_u32x4(
                                peer_base + mailbox_element * 2, packed_words
                            )
                    else:
                        stmc_bf16x8(
                            contribution_mailbox_multicast_address
                            + mailbox_element * 2,
                            packed_words,
                        )
                else:
                    stmc_bf16x8(
                        contribution_mailbox_multicast_address + mailbox_element * 2,
                        packed_words,
                    )

        if cutlass.const_expr(self.bulk_peer_stores):
            first_fragment = cta_in_token * self.threads
            if first_fragment >= self.multicast_fragments:
                store_shared_u32x4(staging + tidx * 4, packed_words)
                fence_proxy_async_shared_cta()
                cute.arch.barrier()
                if tidx < self.tp:
                    destination = (tidx + (self.rank + 1)) % self.tp
                    peer_base = cute.arch.load(
                        (peer_mailboxes.iterator + destination).llvm_ptr,
                        Int64,
                    )
                    chunk_element = (
                        (Int64(stage) * self.tp + self.rank) * self.capacity_m
                        + Int64(token)
                    ) * self.hidden + Int64(first_fragment) * VEC_BF16
                    valid = self.fragments - first_fragment
                    if valid > self.threads:
                        valid = Int32(self.threads)
                    bulk_store_shared_to_global(
                        peer_base + chunk_element * 2,
                        Int32(staging.toint()),
                        valid * 16,
                    )
                    bulk_wait_group_read(0)

        if cutlass.const_expr(not self.release_before_store):
            if block == 0 and tidx == 0:
                store_global_u32(
                    stage_state.iterator + ACTIVE_STAGE,
                    stage,
                )
            cute.arch.barrier()
            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_launch_dependents()


class _LamportResidualRMSNormDeviceKernel:
    def __init__(
        self,
        *,
        hidden: int,
        tp: int,
        capacity_m: int,
        cluster_size: int,
        rank_lanes: int,
        threads: int,
        rms_epsilon: float,
        weight_bias: float,
        add_residual: bool,
        write_residual_output: bool,
        apply_rms_norm: bool = True,
        enable_pdl: bool,
        dynamic_fp8: bool = False,
    ) -> None:
        if rank_lanes not in (1, 2, 4, 8):
            raise ValueError("rank_lanes must be 1, 2, 4, or 8")
        if tp % rank_lanes:
            raise ValueError("tp must be divisible by rank_lanes")
        if threads <= 0 or threads % WARP_SIZE or threads % rank_lanes:
            raise ValueError("threads must be a positive warp and rank-lane multiple")
        if hidden <= 0 or hidden % VEC_BF16:
            raise ValueError("hidden must be a positive multiple of 8")
        self.hidden = hidden
        self.tp = tp
        self.capacity_m = capacity_m
        self.cluster_size = cluster_size
        self.rank_lanes = rank_lanes
        self.threads = threads
        self.rms_epsilon = rms_epsilon
        self.weight_bias = weight_bias
        self.add_residual = add_residual
        self.write_residual_output = write_residual_output
        self.apply_rms_norm = apply_rms_norm
        # FP8 output with a per-token scale written to output_scale[token].
        self.dynamic_fp8 = dynamic_fp8
        self.enable_pdl = enable_pdl
        self.fragments = hidden // VEC_BF16
        self.groups_per_cta = threads // rank_lanes
        self.fragment_stride = cluster_size * self.groups_per_cta
        self.trips = math.ceil(self.fragments / self.fragment_stride)
        self.warps = threads // WARP_SIZE
        self.rank_waves = tp // rank_lanes

    def smem_size_in_bytes(self) -> int:
        return (self.warps + self.cluster_size) * 4 * (2 if self.dynamic_fp8 else 1)

    @cute.jit
    def __call__(
        self,
        contribution_mailbox: cute.Tensor,
        residual_source: cute.Tensor,
        gamma: cute.Tensor,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        norm_output_bf16: cute.Tensor | None,
        output_scale: cute.Tensor | None,
        stage_state: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            contribution_mailbox,
            residual_source,
            gamma,
            residual_output,
            norm_output,
            norm_output_bf16,
            output_scale,
            stage_state,
        ).launch(
            grid=(m, self.cluster_size, 1),
            block=(self.threads, 1, 1),
            cluster=(1, self.cluster_size, 1),
            smem=self.smem_size_in_bytes(),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        contribution_mailbox: cute.Tensor,
        residual_source: cute.Tensor,
        gamma: cute.Tensor,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        norm_output_bf16: cute.Tensor | None,
        output_scale: cute.Tensor | None,
        stage_state: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        cluster_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        rank_lane = tidx % self.rank_lanes
        group = tidx // self.rank_lanes
        base_fragment = cluster_rank * self.groups_per_cta + group
        if cutlass.const_expr(self.apply_rms_norm):
            # Every peer must have started before the RMS partial sums are
            # stored into its shared memory; the matching wait sits just
            # before that first remote store.
            cute.arch.cluster_arrive_relaxed()

        prenorm_fragments = cute.make_rmem_tensor(
            cute.make_layout(
                (self.trips, VEC_BF16),
                stride=(VEC_BF16, 1),
            ),
            BFloat16,
        )
        prenorm_fragments.fill(BFloat16(0.0))
        gamma_fragments = cute.make_rmem_tensor(
            cute.make_layout(
                (self.trips, VEC_BF16),
                stride=(VEC_BF16, 1),
            ),
            BFloat16,
        )
        gamma_fragments.fill(BFloat16(0.0))
        if cutlass.const_expr(self.add_residual):
            residual_fragments = cute.make_rmem_tensor(
                cute.make_layout(
                    (self.trips, VEC_BF16),
                    stride=(VEC_BF16, 1),
                ),
                BFloat16,
            )
            residual_fragments.fill(BFloat16(0.0))

        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            if fragment < self.fragments and rank_lane == 0:
                gamma_element = Int64(fragment) * VEC_BF16
                if cutlass.const_expr(self.apply_rms_norm):
                    gamma_pointer = cute.make_ptr(
                        BFloat16,
                        (gamma.iterator + gamma_element).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    gamma_fragments[trip, None].store(
                        packed_u32x4_to_bf16x8(load_global_u32x4(gamma_pointer))
                    )
                if cutlass.const_expr(self.add_residual):
                    residual_element = Int64(token) * self.hidden + gamma_element
                    residual_pointer = cute.make_ptr(
                        BFloat16,
                        (residual_source.iterator + residual_element).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    residual_fragments[trip, None].store(
                        packed_u32x4_to_bf16x8(load_global_u32x4(residual_pointer))
                    )

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        active_stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            lane_packed = cute.make_rmem_tensor(
                cute.make_layout((self.rank_waves, 4)),
                Uint32,
            )
            lane_packed.fill(Uint32(0))
            dirty = fragment < self.fragments
            while dirty:
                dirty = False
                for wave in cutlass.range_constexpr(self.rank_waves):
                    source_rank = wave * self.rank_lanes + rank_lane
                    if fragment < self.fragments:
                        source_element = (
                            (Int64(active_stage) * self.tp + Int64(source_rank))
                            * self.capacity_m
                            + Int64(token)
                        ) * self.hidden + Int64(fragment) * VEC_BF16
                        source_pointer = cute.make_ptr(
                            BFloat16,
                            (contribution_mailbox.iterator + source_element).llvm_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        packed = load_global_u32x4(
                            source_pointer,
                            volatile=True,
                        )
                        dirty = dirty | fragment_has_negative_zero(packed)
                        for word in cutlass.range_constexpr(4):
                            lane_packed[wave, word] = packed[word]

            lane_sum = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
            lane_sum.fill(Float32(0.0))
            for wave in cutlass.range_constexpr(self.rank_waves):
                packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                for word in cutlass.range_constexpr(4):
                    packed[word] = lane_packed[wave, word]
                lane_sum.store(
                    lane_sum.load() + packed_u32x4_to_bf16x8(packed.load()).to(Float32)
                )
            for offset in cutlass.range_constexpr(1, 5):
                if cutlass.const_expr(offset < self.rank_lanes and offset in (1, 2, 4)):
                    for element in cutlass.range_constexpr(VEC_BF16):
                        lane_sum[element] = lane_sum[
                            element
                        ] + cute.arch.shuffle_sync_bfly(
                            lane_sum[element],
                            offset=offset,
                            mask=-1,
                            mask_and_clamp=31,
                        )

            if fragment < self.fragments and rank_lane == 0:
                prenorm = lane_sum.load()
                if cutlass.const_expr(self.add_residual):
                    prenorm = prenorm + residual_fragments[trip, None].load().to(
                        Float32
                    )
                prenorm_bf16 = prenorm.to(BFloat16)
                prenorm_fragments[trip, None].store(prenorm_bf16)
                if cutlass.const_expr(self.write_residual_output):
                    output_element = (
                        Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
                    )
                    store_global_u32x4(
                        Int64((residual_output.iterator + output_element).toint()),
                        bf16x8_to_packed_u32x4(prenorm_bf16),
                    )

        if token == 0 and cluster_rank == 0 and tidx == 0:
            store_global_u32(
                stage_state.iterator + NEXT_STAGE,
                (active_stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS),
            )
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            for wave in cutlass.range_constexpr(self.rank_waves):
                source_rank = wave * self.rank_lanes + rank_lane
                if fragment < self.fragments:
                    source_element = (
                        (Int64(active_stage) * self.tp + Int64(source_rank))
                        * self.capacity_m
                        + Int64(token)
                    ) * self.hidden + Int64(fragment) * VEC_BF16
                    store_lamport_sentinel_u32x4(
                        Int64((contribution_mailbox.iterator + source_element).toint())
                    )

        # Without the norm, the reduced sum in residual_output is the result.
        if cutlass.const_expr(self.apply_rms_norm):
            thread_sum = Float32(0.0)
            for trip in cutlass.range_constexpr(self.trips):
                fragment = base_fragment + trip * self.fragment_stride
                if fragment < self.fragments and rank_lane == 0:
                    values = prenorm_fragments[trip, None].load().to(Float32)
                    thread_sum = thread_sum + (values * values).reduce(
                        cute.ReductionOp.ADD,
                        init_val=Float32(0.0),
                        reduction_profile=0,
                    )

            smem = cutlass.utils.SmemAllocator()
            warp_sums = smem.allocate_array(Float32, self.warps)
            cluster_sums = smem.allocate_array(Float32, self.cluster_size)
            cta_sum = _group_leader_block_sum(
                thread_sum,
                warp_sums,
                self.warps,
                self.rank_lanes,
            )
            cute.arch.cluster_wait()
            if tidx < self.cluster_size:
                local_slot = cluster_sums + cluster_rank
                remote_slot = map_shared_to_peer(local_slot, Int32(tidx))
                store_shared_cluster_f32(remote_slot, cta_sum)
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            full_sum = Float32(0.0)
            for peer in cutlass.range_constexpr(self.cluster_size):
                full_sum = full_sum + cute.arch.load(
                    (cluster_sums + peer).llvm_ptr,
                    Float32,
                )
            inv_rms = cute.math.rsqrt(
                full_sum / Float32(self.hidden) + Float32(self.rms_epsilon),
                fastmath=True,
            )
            if cutlass.const_expr(self.dynamic_fp8):
                store_dynamic_fp8_cluster_row(
                    prenorm_fragments,
                    gamma_fragments,
                    inv_rms,
                    norm_output,
                    norm_output_bf16,
                    output_scale,
                    smem.allocate_array(Float32, self.warps),
                    smem.allocate_array(Float32, self.cluster_size),
                    tidx,
                    token,
                    base_fragment,
                    rank_lane,
                    cluster_rank,
                    self.hidden,
                    self.fragment_stride,
                    self.threads,
                    self.cluster_size,
                    self.weight_bias,
                )
            # The dynamic FP8 epilogue above replaces this static/BF16 store.
            for trip in cutlass.range_constexpr(0 if self.dynamic_fp8 else self.trips):
                fragment = base_fragment + trip * self.fragment_stride
                if fragment < self.fragments and rank_lane == 0:
                    gamma_values = gamma_fragments[trip, None].load().to(Float32)
                    if cutlass.const_expr(self.weight_bias != 0.0):
                        gamma_values = gamma_values + Float32(self.weight_bias)
                    result = (
                        prenorm_fragments[trip, None].load().to(Float32)
                        * inv_rms
                        * gamma_values
                    ).to(BFloat16)
                    output_element = (
                        Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
                    )
                    store_rmsnorm_output(
                        bf16x8_to_packed_u32x4(result),
                        norm_output,
                        Int64(output_element),
                        output_scale,
                        norm_output_bf16,
                    )


class _LamportTwoShotDeviceKernel:
    """One-launch Lamport reduce-scatter plus multicast all-gather.

    The hidden dimension is split into ``tp`` contiguous shards and rank ``o``
    owns shard ``o``. Every rank stores each fragment of its contribution to
    the owner's reduce-scatter mailbox through the owner's unicast mapping; the
    owner keeps its own share in registers. The owner spins on the Lamport
    sentinels of the ``tp - 1`` peer copies, reduces in FP32 in rank order,
    adds its own residual shard, and multicast-stores the BF16 pre-norm fragment to every rank's
    all-gather mailbox. Every rank then spins on its all-gather copy and
    finishes the optional RMSNorm the way the one-shot collective does.

    Only the owner's residual enters a shard, so the residual must be the same
    on every rank, as tensor parallelism replicates it; the one-shot
    collective adds each rank's own residual instead.

    Per GPU this receives ``(tp - 1) / tp`` of an activation for the
    reduce-scatter and one activation of all-gather traffic, instead of the
    ``tp`` activations a one-shot multicast publish delivers, at the price of a
    second NVLink hop. It therefore pays off once the transfer, not the hop
    latency, dominates.

    The CTA that publishes a fragment is the CTA that reduces and gathers it on
    every rank, so no CTA waits on another CTA of its own rank and no grid-wide
    stage update is needed. Each CTA instead keeps its own Lamport generation
    in ``generations[cta]``: every rank launches the same CTA sequence, so the
    per-CTA counters agree across ranks, also when a CTA sits out a call with
    fewer tokens. A reduce-scatter slot written in call ``n`` is rewritten in
    call ``n + 3`` at the earliest, which needs the owner's all-gather of call
    ``n + 2``, issued after the owner finished call ``n``; the all-gather slots
    follow the same argument with the roles swapped.
    """

    def __init__(
        self,
        *,
        hidden: int,
        tp: int,
        rank: int,
        capacity_m: int,
        cluster_size: int,
        threads: int,
        rms_epsilon: float,
        weight_bias: float,
        add_residual: bool,
        write_residual_output: bool,
        apply_rms_norm: bool,
        enable_pdl: bool,
    ) -> None:
        if threads <= 0 or threads % WARP_SIZE:
            raise ValueError("threads must be a positive warp multiple")
        if hidden <= 0 or hidden % (VEC_BF16 * tp):
            raise ValueError("hidden must split into tp shards of 8-element fragments")
        self.hidden = hidden
        self.tp = tp
        self.rank = rank
        self.capacity_m = capacity_m
        self.cluster_size = cluster_size
        self.threads = threads
        self.rms_epsilon = rms_epsilon
        self.weight_bias = weight_bias
        self.add_residual = add_residual
        self.write_residual_output = write_residual_output
        self.apply_rms_norm = apply_rms_norm
        self.enable_pdl = enable_pdl
        self.fragments = hidden // VEC_BF16
        self.shard_fragments = self.fragments // tp
        self.fragment_stride = cluster_size * threads
        self.trips = math.ceil(self.fragments / self.fragment_stride)
        self.warps = threads // WARP_SIZE

    def smem_size_in_bytes(self) -> int:
        return (self.warps + self.cluster_size) * 4

    @cute.jit
    def __call__(
        self,
        local_contribution: cute.Tensor,
        scatter_mailbox: cute.Tensor,
        scatter_peer_mailboxes: cute.Tensor,
        gather_mailbox: cute.Tensor,
        gather_multicast_address: Int64,
        residual_source: cute.Tensor,
        gamma: cute.Tensor,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        norm_output_bf16: cute.Tensor | None,
        output_scale: cute.Tensor | None,
        generations: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        kernel = self.kernel(
            local_contribution,
            scatter_mailbox,
            scatter_peer_mailboxes,
            gather_mailbox,
            gather_multicast_address,
            residual_source,
            gamma,
            residual_output,
            norm_output,
            norm_output_bf16,
            output_scale,
            generations,
        )
        if cutlass.const_expr(self.apply_rms_norm):
            kernel.launch(
                grid=(m, self.cluster_size, 1),
                block=(self.threads, 1, 1),
                cluster=(1, self.cluster_size, 1),
                smem=self.smem_size_in_bytes(),
                stream=stream,
                use_pdl=self.enable_pdl,
            )
        else:
            # Without the norm no CTA reads another CTA's shared memory, so the
            # launch needs no cluster co-scheduling.
            kernel.launch(
                grid=(m, self.cluster_size, 1),
                block=(self.threads, 1, 1),
                smem=self.smem_size_in_bytes(),
                stream=stream,
                use_pdl=self.enable_pdl,
            )

    @cute.kernel
    def kernel(
        self,
        local_contribution: cute.Tensor,
        scatter_mailbox: cute.Tensor,
        scatter_peer_mailboxes: cute.Tensor,
        gather_mailbox: cute.Tensor,
        gather_multicast_address: Int64,
        residual_source: cute.Tensor,
        gamma: cute.Tensor,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        norm_output_bf16: cute.Tensor | None,
        output_scale: cute.Tensor | None,
        generations: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        token, chunk, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.apply_rms_norm):
            cluster_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        else:
            cluster_rank = cute.arch.make_warp_uniform(chunk)
        base_fragment = cluster_rank * self.threads + tidx
        cta_index = Int64(token) * self.cluster_size + Int64(cluster_rank)

        gamma_fragments = cute.make_rmem_tensor(
            cute.make_layout((self.trips, VEC_BF16), stride=(VEC_BF16, 1)),
            BFloat16,
        )
        gamma_fragments.fill(BFloat16(0.0))
        if cutlass.const_expr(self.apply_rms_norm):
            for trip in cutlass.range_constexpr(self.trips):
                fragment = base_fragment + trip * self.fragment_stride
                if fragment < self.fragments:
                    gamma_pointer = cute.make_ptr(
                        BFloat16,
                        (gamma.iterator + Int64(fragment) * VEC_BF16).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    gamma_fragments[trip, None].store(
                        packed_u32x4_to_bf16x8(load_global_u32x4(gamma_pointer))
                    )

        # The contribution and the residual may come from the kernel this launch
        # depends on, so unlike the static gamma they are read after the wait.
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        stage = load_volatile_u32(generations.iterator + cta_index)
        slot_stride = Int64(self.capacity_m) * self.hidden
        scatter_stage = Int64(stage) * self.tp
        gather_stage = Int64(stage) * slot_stride

        # Reduce-scatter publish: one unicast store to the fragment's owner.
        own_packed = cute.make_rmem_tensor(
            cute.make_layout((self.trips, 4), stride=(4, 1)),
            Uint32,
        )
        own_packed.fill(Uint32(0))
        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            if fragment < self.fragments:
                owner = fragment // self.shard_fragments
                token_element = Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
                source_pointer = cute.make_ptr(
                    BFloat16,
                    (local_contribution.iterator + token_element).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                packed = sanitize_negative_zero_u32x4(load_global_u32x4(source_pointer))
                for word in cutlass.range_constexpr(4):
                    own_packed[trip, word] = packed[word]
                if owner != Int32(self.rank):
                    peer_base = cute.arch.load(
                        (scatter_peer_mailboxes.iterator + owner).llvm_ptr,
                        Int64,
                    )
                    own_element = (
                        scatter_stage + self.rank
                    ) * slot_stride + token_element
                    store_global_u32x4(peer_base + own_element * 2, packed)

        if cutlass.const_expr(self.add_residual):
            residual_fragments = cute.make_rmem_tensor(
                cute.make_layout((self.trips, VEC_BF16), stride=(VEC_BF16, 1)),
                BFloat16,
            )
            residual_fragments.fill(BFloat16(0.0))
            for trip in cutlass.range_constexpr(self.trips):
                fragment = base_fragment + trip * self.fragment_stride
                if fragment < self.fragments:
                    if fragment // self.shard_fragments == Int32(self.rank):
                        residual_pointer = cute.make_ptr(
                            BFloat16,
                            (
                                residual_source.iterator
                                + (
                                    Int64(token) * self.hidden
                                    + Int64(fragment) * VEC_BF16
                                )
                            ).llvm_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        residual_fragments[trip, None].store(
                            packed_u32x4_to_bf16x8(load_global_u32x4(residual_pointer))
                        )

        # Owner: reduce its shard in rank order and multicast the BF16 pre-norm.
        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            token_element = Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
            owned = fragment < self.fragments
            if owned:
                owned = fragment // self.shard_fragments == Int32(self.rank)
            if owned:
                peer_packed = cute.make_rmem_tensor(
                    cute.make_layout((self.tp, 4)),
                    Uint32,
                )
                peer_packed.fill(Uint32(0))
                dirty = fragment < self.fragments
                while dirty:
                    dirty = False
                    for source_rank in cutlass.range_constexpr(self.tp):
                        if cutlass.const_expr(source_rank != self.rank):
                            source_element = (
                                scatter_stage + source_rank
                            ) * slot_stride + token_element
                            source_pointer = cute.make_ptr(
                                BFloat16,
                                (scatter_mailbox.iterator + source_element).llvm_ptr,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            packed = load_global_u32x4(source_pointer, volatile=True)
                            dirty = dirty | fragment_has_negative_zero(packed)
                            for word in cutlass.range_constexpr(4):
                                peer_packed[source_rank, word] = packed[word]
                lane_sum = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
                lane_sum.fill(Float32(0.0))
                for source_rank in cutlass.range_constexpr(self.tp):
                    packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                    for word in cutlass.range_constexpr(4):
                        if cutlass.const_expr(source_rank == self.rank):
                            packed[word] = own_packed[trip, word]
                        else:
                            packed[word] = peer_packed[source_rank, word]
                    lane_sum.store(
                        lane_sum.load()
                        + packed_u32x4_to_bf16x8(packed.load()).to(Float32)
                    )
                prenorm = lane_sum.load()
                if cutlass.const_expr(self.add_residual):
                    prenorm = prenorm + residual_fragments[trip, None].load().to(
                        Float32
                    )
                # A -0.0 result would read as an unwritten sentinel on every rank.
                reduced = sanitize_negative_zero_u32x4(
                    bf16x8_to_packed_u32x4(prenorm.to(BFloat16))
                )
                stmc_bf16x8(
                    gather_multicast_address + (gather_stage + token_element) * 2,
                    reduced,
                )
                for source_rank in cutlass.range_constexpr(self.tp):
                    if cutlass.const_expr(source_rank != self.rank):
                        source_element = (
                            scatter_stage + source_rank
                        ) * slot_stride + token_element
                        store_lamport_sentinel_u32x4(
                            Int64((scatter_mailbox.iterator + source_element).toint())
                        )

        # Every rank, the owner included: wait for the gathered pre-norm.
        prenorm_fragments = cute.make_rmem_tensor(
            cute.make_layout((self.trips, VEC_BF16), stride=(VEC_BF16, 1)),
            BFloat16,
        )
        prenorm_fragments.fill(BFloat16(0.0))
        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            token_element = Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
            if fragment < self.fragments:
                gather_pointer = cute.make_ptr(
                    BFloat16,
                    (gather_mailbox.iterator + gather_stage + token_element).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                gathered = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                gathered.fill(Uint32(0))
                dirty = fragment < self.fragments
                while dirty:
                    packed = load_global_u32x4(gather_pointer, volatile=True)
                    dirty = fragment_has_negative_zero(packed)
                    for word in cutlass.range_constexpr(4):
                        gathered[word] = packed[word]
                prenorm_fragments[trip, None].store(
                    packed_u32x4_to_bf16x8(gathered.load())
                )
                if cutlass.const_expr(self.write_residual_output):
                    store_global_u32x4(
                        Int64((residual_output.iterator + token_element).toint()),
                        gathered.load(),
                    )

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            token_element = Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
            if fragment < self.fragments:
                store_lamport_sentinel_u32x4(
                    Int64(
                        (gather_mailbox.iterator + gather_stage + token_element).toint()
                    )
                )

        smem = cutlass.utils.SmemAllocator()
        warp_sums = smem.allocate_array(Float32, self.warps)
        cluster_sums = smem.allocate_array(Float32, self.cluster_size)
        if cutlass.const_expr(self.apply_rms_norm):
            thread_sum = Float32(0.0)
            for trip in cutlass.range_constexpr(self.trips):
                fragment = base_fragment + trip * self.fragment_stride
                if fragment < self.fragments:
                    values = prenorm_fragments[trip, None].load().to(Float32)
                    thread_sum = thread_sum + (values * values).reduce(
                        cute.ReductionOp.ADD,
                        init_val=Float32(0.0),
                        reduction_profile=0,
                    )
            cta_sum = _group_leader_block_sum(thread_sum, warp_sums, self.warps, 1)
            # Peer CTAs must exist before accessing their distributed shared memory.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            if tidx < self.cluster_size:
                local_slot = cluster_sums + cluster_rank
                remote_slot = map_shared_to_peer(local_slot, Int32(tidx))
                store_shared_cluster_f32(remote_slot, cta_sum)
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            full_sum = Float32(0.0)
            for peer in cutlass.range_constexpr(self.cluster_size):
                full_sum = full_sum + cute.arch.load(
                    (cluster_sums + peer).llvm_ptr,
                    Float32,
                )
            inv_rms = cute.math.rsqrt(
                full_sum / Float32(self.hidden) + Float32(self.rms_epsilon),
                fastmath=True,
            )
            for trip in cutlass.range_constexpr(self.trips):
                fragment = base_fragment + trip * self.fragment_stride
                if fragment < self.fragments:
                    gamma_values = gamma_fragments[trip, None].load().to(Float32)
                    if cutlass.const_expr(self.weight_bias != 0.0):
                        gamma_values = gamma_values + Float32(self.weight_bias)
                    result = (
                        prenorm_fragments[trip, None].load().to(Float32)
                        * inv_rms
                        * gamma_values
                    ).to(BFloat16)
                    output_element = (
                        Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
                    )
                    store_rmsnorm_output(
                        bf16x8_to_packed_u32x4(result),
                        norm_output,
                        Int64(output_element),
                        output_scale,
                        norm_output_bf16,
                    )
        else:
            # Every thread must have read this CTA's generation before it advances.
            cute.arch.barrier()

        if tidx == 0:
            store_global_u32(
                generations.iterator + cta_index,
                (stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS),
            )
