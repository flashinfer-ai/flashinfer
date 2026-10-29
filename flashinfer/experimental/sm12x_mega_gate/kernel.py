# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0

"""CuTe DSL fused BF16 routing GEMM and sqrt-softplus top-k gate for SM120.

Port of DeepGEMM's ``sm100_bf16_mega_gate`` (the SM100a/SM103a backend lives in
``flashinfer/experimental/deepgemm_mega_gate``) to consumer Blackwell, which has
no tcgen05/TMEM: the GEMM runs on warp-level ``mma.sync.m16n8k16`` fed by TMA
and ldmatrix, and accumulates in registers.

Schedule (as DeepGEMM): a token block of ``block_tokens`` tokens is computed by
``L = num_expert_groups * num_split_k`` logical CTAs, each owning 128 experts of
one K slice and writing its FP32 partial logits to ``scratch``
(``[token blocks, split-K, block tokens, aligned experts]``). The weight is the
MMA A operand and the tokens the B operand, so decode-sized token counts only pad
up to the 8-wide MMA N granularity. The grid is persistent: ``num_worker_groups``
groups of L CTAs, at most one CTA per SM so that every group is co-resident, and
a group strides over the token blocks.

Warp roles (DeepGEMM's TMA / MMA / gate split):
- warps 0-3, MMA: per token block, the mainloop, the partial-logit stores and the
  arrival at the block's score barrier; then straight on to the next block.
- warp 4, TMA producer.
- warps 5.., gate (4 or 8, per schedule): once all L logical CTAs of a block have arrived, every logical
  CTA ranks a share of its tokens (DeepGEMM's order: token ``gate_warp * L +
  rank``, stride ``gate_warps * L``), one warp per token, overlapping the MMA of
  the next block.

Score barrier: per token block, an int32 arrival counter released with
``red.release.gpu`` by the MMA warps and acquired with ``ld.acquire.gpu`` by the
gate warps until it reads L, plus a waiter counter: every CTA that waited counts
itself after seeing L, and the last one zeroes both words. DeepGEMM tags its
barrier with ``%gridid``, which a CUDA graph kernel node keeps across replays
(DeepGEMM 5081f85 moved mega_mhc to an epoch for that reason); the self-rearming
lines need neither tags nor a teardown pass at exit, so every launch or graph
replay starts from zero.

Gate semantics follow DeepGEMM: ``score = sqrt(softplus(logit))`` (softplus is
the identity above 20), experts are ranked on ``score + bias`` (the image bias
for image tokens), exact ties resolve to the smaller expert index, and the
emitted weights are the unbiased scores normalized over the top-k sum
(``+ 1e-20``) times ``routed_scale``. Without a bias the ranking runs on the raw
logits (sqrt-softplus is monotonic) and only the selected ones are activated.
Shared experts take the slots after the routed top-k with weight 1. A physical
map spreads the tokens of a logical expert over its replicas with
``(ep_rank + token * 23333) % logical_count`` in uint32 arithmetic. The warp
top-k is DeepGEMM's: each lane sorts its 4 experts of every 128-expert wave and
every round offers the best unconsumed candidate of each wave.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as cutlass_utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Float32, Int32, Int64, Uint8, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import LayoutEnum

from .heuristics import (
    BLOCK_K,
    EXPERTS_PER_CTA,
    MegaGateConfig,
    align,
)

MMA_WARPS_M = 4
"""MMA warps along the experts: 32 of the CTA's 128 experts each."""
MMA_WARPS_N = 1
"""MMA warps along the tokens. 2 (two MMA warps per SM sub-partition) measured 0.93x-1.02x of
1 on a 110-SM board: the mainloop already keeps the tensor pipe ~80% busy (ncu, M=16384)."""
NUM_MMA_WARPS = MMA_WARPS_M * MMA_WARPS_N
PRODUCER_WARP = NUM_MMA_WARPS
FIRST_GATE_WARP = PRODUCER_WARP + 1
MMA_BARRIER_ID = 1
GATE_BARRIER_ID = 2
EXPERTS_PER_LANE = 4
"""Each lane holds 4 consecutive experts of every 128-expert wave."""
EXPERTS_PER_WAVE = 32 * EXPERTS_PER_LANE
COUNTER_STRIDE = 32
"""Int32 words per score-barrier line (128 B), one line per token block: word 0 counts the
logical CTAs that stored their partials, word ``WAITERS_WORD`` the ones that waited for them."""
WAITERS_WORD = 1
SPIN_BACKOFF_NS = 64
"""Pause between two polls of a score barrier (keeps the polling lane off the L2 and the
issue slots of the MMA warp sharing its SM sub-partition)."""
SOFTPLUS_THRESHOLD = 20.0
TOPK_SUM_EPSILON = 1e-20
DUPLICATE_HASH = 23333
NO_EXPERT = 0x7FFFFFFF
"""Sentinel larger than every expert index; loses every tie of the warp argmax."""


@dsl_user_op
def ld_acquire_gpu(address: Int64, *, loc=None, ip=None) -> Int32:
    """``ld.acquire.gpu`` of an int32 (also a compiler memory barrier)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.acquire.gpu.global.b32 $0, [$1];",
            "=r,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def red_release_gpu_add(address: Int64, value: Int32, *, loc=None, ip=None) -> None:
    """``red.release.gpu.add``: cumulatively publishes the writes ordered before it."""
    llvm.inline_asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def nanosleep(nanoseconds: int, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [],
        f"nanosleep.u32 {nanoseconds};",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ordered_key(value: Float32, *, loc=None, ip=None) -> Int32:
    """A signed int32 ordered like the (non-NaN) float ``value``, for the integer warp redux
    (SM120 has ``redux.sync.max`` for integers but not for floats). -0 is folded into +0."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [(Float32(value) + Float32(0.0)).ir_value(loc=loc, ip=ip)],
            "{ .reg .b32 b, m; mov.b32 b, $1; shr.s32 m, b, 31; "
            "and.b32 m, m, 0x7FFFFFFF; xor.b32 $0, b, m; }",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def load_vec4(tensor, offset):
    """One 16 B vector load of ``tensor[offset : offset + 4]`` (FP32 or int32); ``offset`` is a multiple of 4."""
    view = cute.make_tensor(
        tensor.iterator + cute.assume(offset, divby=4), cute.make_layout(4)
    )
    return view.load()


def sqrt_softplus(value):
    """``sqrt(softplus(value))`` with the identity above the softplus threshold."""
    softplus = cute.math.log1p(cute.math.exp(value))
    softplus = cutlass.select_(value > Float32(SOFTPLUS_THRESHOLD), value, softplus)
    return cute.math.sqrt(softplus)


def compare_swap_best_first(lhs, lhs_idx, rhs, rhs_idx):
    """Order two candidates: larger rank first, the smaller index first on a tie."""
    swap = cutlass.or_(rhs > lhs, cutlass.and_(rhs == lhs, rhs_idx < lhs_idx))
    return (
        cutlass.select_(swap, rhs, lhs),
        cutlass.select_(swap, rhs_idx, lhs_idx),
        cutlass.select_(swap, lhs, rhs),
        cutlass.select_(swap, lhs_idx, rhs_idx),
    )


class Sm12xMegaGateKernel:
    """One compiled schedule of the fused routing gate.

    Compile-time: the schedule (``MegaGateConfig``), the problem constants
    (hidden size, expert count, top-k, shared slots) and which optional inputs
    are present. Runtime: the token count, the worker-group count, the EP rank,
    the routed scale and the physical map width.
    """

    def __init__(
        self,
        config: MegaGateConfig,
        *,
        hidden: int,
        num_experts: int,
        num_topk: int,
        num_shared: int,
        has_bias: bool,
        has_image_bias: bool,
        has_mask: bool,
        has_physical_map: bool,
        has_unmapped: bool,
    ):
        self.block_tokens = config.block_tokens
        self.num_split_k = config.num_split_k
        self.num_stages = config.num_stages
        self.num_gate_warps = config.num_gate_warps
        self.num_threads = (FIRST_GATE_WARP + self.num_gate_warps) * 32
        self.hidden = hidden
        self.num_experts = num_experts
        self.num_aligned_experts = align(num_experts, EXPERTS_PER_CTA)
        self.num_expert_groups = self.num_aligned_experts // EXPERTS_PER_CTA
        if config.num_expert_groups != self.num_expert_groups:
            raise ValueError("the schedule does not match the expert count")
        if num_experts % EXPERTS_PER_LANE:
            raise ValueError(
                f"the expert count must be a multiple of {EXPERTS_PER_LANE}"
            )
        self.num_logical_ctas = self.num_expert_groups * self.num_split_k
        self.k_blocks_per_split = hidden // BLOCK_K // self.num_split_k
        self.num_waves = self.num_aligned_experts // EXPERTS_PER_WAVE
        self.num_lane_values = self.num_waves * EXPERTS_PER_LANE
        self.has_padded_experts = self.num_aligned_experts != num_experts
        self.num_topk = num_topk
        self.num_shared = num_shared
        self.num_slots = num_topk + num_shared
        self.has_bias = has_bias
        self.has_image_bias = has_image_bias
        self.has_scoring_bias = has_bias or has_image_bias
        self.has_mask = has_mask
        self.has_physical_map = has_physical_map
        self.has_unmapped = has_unmapped
        # The smallest butterfly covering the routed lanes; the other lanes add zero.
        self.sum_width = 8 if num_topk <= 8 else (16 if num_topk <= 16 else 32)

    # ------------------------------------------------------------------ layouts

    def _tiled_mma(self) -> cute.TiledMma:
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.BFloat16, Float32, (16, 8, 16)),
            (MMA_WARPS_M, MMA_WARPS_N, 1),
            permutation_mnk=(MMA_WARPS_M * 16, self.block_tokens, 16),
        )

    def _smem_layouts(self):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, cutlass.BFloat16, BLOCK_K
            ),
            cutlass.BFloat16,
        )
        sW_layout = cute.tile_to_shape(
            atom, (EXPERTS_PER_CTA, BLOCK_K, self.num_stages), order=(0, 1, 2)
        )
        sX_layout = cute.tile_to_shape(
            atom, (self.block_tokens, BLOCK_K, self.num_stages), order=(0, 1, 2)
        )
        return sW_layout, sX_layout

    def _shared_storage(self, sW_layout, sX_layout):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "mbar_ptr": cute.struct.MemRange[cutlass.Int64, self.num_stages * 2],
            "sW": cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(sW_layout)], 1024
            ],
            "sX": cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(sX_layout)], 1024
            ],
        }
        return cute.struct(SharedStorage)

    # --------------------------------------------------------------------- host

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        bias: cute.Tensor,
        image_bias: cute.Tensor,
        image_mask: cute.Tensor,
        mask: cute.Tensor,
        physical_map: cute.Tensor,
        logical_count: cute.Tensor,
        topk_idx: cute.Tensor,
        topk_weights: cute.Tensor,
        unmapped_idx: cute.Tensor,
        scratch: cute.Tensor,
        counters: cute.Tensor,
        num_tokens: Int32,
        ep_rank: Int32,
        routed_scale: Float32,
        map_width: Int32,
        num_worker_groups: Int32,
        stream: cuda.CUstream,
    ):
        sW_layout, sX_layout = self._smem_layouts()
        tiled_mma = self._tiled_mma()
        SharedStorage = self._shared_storage(sW_layout, sX_layout)
        tma_atom_w, tma_w = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            weight,
            cute.slice_(sW_layout, (None, None, 0)),
            (EXPERTS_PER_CTA, BLOCK_K),
        )
        tma_atom_x, tma_x = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            x,
            cute.slice_(sX_layout, (None, None, 0)),
            (self.block_tokens, BLOCK_K),
        )
        self.kernel(
            tma_atom_w,
            tma_w,
            tma_atom_x,
            tma_x,
            sW_layout,
            sX_layout,
            tiled_mma,
            SharedStorage,
            bias,
            image_bias,
            image_mask,
            mask,
            physical_map,
            logical_count,
            topk_idx,
            topk_weights,
            unmapped_idx,
            scratch,
            counters,
            num_tokens,
            ep_rank,
            routed_scale,
            map_width,
            num_worker_groups,
        ).launch(
            grid=(num_worker_groups * Int32(self.num_logical_ctas), 1, 1),
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    # ------------------------------------------------------------------- device

    @cute.kernel
    def kernel(
        self,
        tma_atom_w: cute.CopyAtom,
        tma_w: cute.Tensor,
        tma_atom_x: cute.CopyAtom,
        tma_x: cute.Tensor,
        sW_layout: cute.ComposedLayout,
        sX_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        bias: cute.Tensor,
        image_bias: cute.Tensor,
        image_mask: cute.Tensor,
        mask: cute.Tensor,
        physical_map: cute.Tensor,
        logical_count: cute.Tensor,
        topk_idx: cute.Tensor,
        topk_weights: cute.Tensor,
        unmapped_idx: cute.Tensor,
        scratch: cute.Tensor,
        counters: cute.Tensor,
        num_tokens: Int32,
        ep_rank: Int32,
        routed_scale: Float32,
        map_width: Int32,
        num_worker_groups: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # A worker group of L consecutive CTAs strides over the token blocks together.
        worker_group = bidx // Int32(self.num_logical_ctas)
        logical_rank = bidx % Int32(self.num_logical_ctas)
        expert_group = logical_rank % Int32(self.num_expert_groups)
        split_idx = logical_rank // Int32(self.num_expert_groups)
        num_token_blocks = (num_tokens + Int32(self.block_tokens - 1)) // Int32(
            self.block_tokens
        )

        if warp_idx == Int32(PRODUCER_WARP):
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_x)

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sW = storage.sW.get_tensor(sW_layout.outer, swizzle=sW_layout.inner)
        sX = storage.sX.get_tensor(sX_layout.outer, swizzle=sX_layout.inner)

        tx_bytes = (EXPERTS_PER_CTA + self.block_tokens) * BLOCK_K * 2
        load_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, NUM_MMA_WARPS
            ),
            tx_count=tx_bytes,
            barrier_storage=storage.mbar_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        cute.arch.sync_threads()

        # (128 experts, BLOCK_K, expert tiles, K blocks) and (block tokens, BLOCK_K, token blocks, K blocks)
        gW = cute.local_tile(tma_w, (EXPERTS_PER_CTA, BLOCK_K), (None, None))
        gX = cute.local_tile(tma_x, (self.block_tokens, BLOCK_K), (None, None))
        cta_layout = cute.make_layout(1)
        tWsW, tWgW = cpasync.tma_partition(
            tma_atom_w,
            0,
            cta_layout,
            cute.group_modes(sW, 0, 2),
            cute.group_modes(gW, 0, 2),
        )
        tXsX, tXgX = cpasync.tma_partition(
            tma_atom_x,
            0,
            cta_layout,
            cute.group_modes(sX, 0, 2),
            cute.group_modes(gX, 0, 2),
        )
        k_block_base = split_idx * Int32(self.k_blocks_per_split)
        counters_address = Int64(counters.iterator.toint())

        if warp_idx < Int32(NUM_MMA_WARPS):
            # ---------------------------------------------------- MMA warps
            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_stages
            )
            thr_mma = tiled_mma.get_slice(tidx)
            tCsW = thr_mma.partition_A(sW)
            tCsX = thr_mma.partition_B(sX)
            tCrW = thr_mma.make_fragment_A(tCsW[None, None, None, 0])
            tCrX = thr_mma.make_fragment_B(tCsX[None, None, None, 0])
            acc = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((EXPERTS_PER_CTA, self.block_tokens)),
                Float32,
            )
            # A warp's B slice per k16 is (block_tokens / MMA_WARPS_N) x 16: four 8x8
            # matrices per ldmatrix when that is a multiple of 16 tokens, two otherwise.
            tokens_per_warp = self.block_tokens // MMA_WARPS_N
            smem_copy_w = cute.make_tiled_copy_A(
                cute.make_copy_atom(
                    warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                    cutlass.BFloat16,
                ),
                tiled_mma,
            ).get_slice(tidx)
            smem_copy_x = cute.make_tiled_copy_B(
                cute.make_copy_atom(
                    warp.LdMatrix8x8x16bOp(
                        transpose=False,
                        num_matrices=4 if tokens_per_warp % 16 == 0 else 2,
                    ),
                    cutlass.BFloat16,
                ),
                tiled_mma,
            ).get_slice(tidx)
            tSsW = smem_copy_w.partition_S(sW)
            tSsX = smem_copy_x.partition_S(sX)
            tSrW = smem_copy_w.retile(tCrW)
            tSrX = smem_copy_x.retile(tCrX)
            num_k_phases = cute.size(tCsW, mode=[2])
            coords = thr_mma.partition_C(
                cute.make_identity_tensor((EXPERTS_PER_CTA, self.block_tokens))
            )
            row_stride = Int64(self.num_aligned_experts)

            for token_block in cutlass.range(
                worker_group, num_token_blocks, num_worker_groups, unroll=1
            ):
                acc.fill(0.0)
                for _k_block in cutlass.range(self.k_blocks_per_split, unroll=1):
                    load_pipeline.consumer_wait(consumer_state)
                    stage = consumer_state.index
                    for k_phase in cutlass.range_constexpr(num_k_phases):
                        cute.copy(
                            smem_copy_w,
                            tSsW[None, None, k_phase, stage],
                            tSrW[None, None, k_phase],
                        )
                        cute.copy(
                            smem_copy_x,
                            tSsX[None, None, k_phase, stage],
                            tSrX[None, None, k_phase],
                        )
                        cute.gemm(
                            tiled_mma,
                            acc,
                            tCrW[None, None, k_phase],
                            tCrX[None, None, k_phase],
                            acc,
                        )
                    load_pipeline.consumer_release(consumer_state)
                    consumer_state.advance()

                # Full token blocks are always stored: scratch is sized for them and the
                # rows past the last token are never read.
                scratch_base = (
                    Int64(token_block) * Int64(self.num_split_k) + Int64(split_idx)
                ) * Int64(self.block_tokens) * row_stride + Int64(expert_group) * Int64(
                    EXPERTS_PER_CTA
                )
                for i in cutlass.range_constexpr(cute.size(acc)):
                    coord = coords[i]
                    scratch[
                        scratch_base + Int64(coord[1]) * row_stride + Int64(coord[0])
                    ] = acc[i]

                # Arrive at the block's score barrier once all four warps have stored.
                cute.arch.barrier(
                    barrier_id=MMA_BARRIER_ID, number_of_threads=NUM_MMA_WARPS * 32
                )
                if tidx == 0:
                    cute.arch.fence_acq_rel_gpu()
                    red_release_gpu_add(
                        counters_address
                        + Int64(token_block) * Int64(COUNTER_STRIDE * 4),
                        Int32(1),
                    )
                # Reconverge warp 0 before the next block's .aligned ldmatrix / mma.sync.
                cute.arch.sync_warp()

        elif warp_idx == Int32(PRODUCER_WARP):
            # --------------------------------------------------- TMA producer
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )
            for token_block in cutlass.range(
                worker_group, num_token_blocks, num_worker_groups, unroll=1
            ):
                for k_block in cutlass.range(self.k_blocks_per_split, unroll=1):
                    load_pipeline.producer_acquire(producer_state)
                    # The MMA warps read this stage with ldmatrix (generic proxy): order
                    # those reads before the TMA's async-proxy refill, or the refill can
                    # land under a still-pending read (seen as corrupted partials once the
                    # gate warps load the memory system).
                    cute.arch.fence_view_async_shared()
                    k_idx = k_block_base + k_block
                    cute.copy(
                        tma_atom_w,
                        tWgW[(None, expert_group, k_idx)],
                        tWsW[(None, producer_state.index)],
                        tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
                    )
                    cute.copy(
                        tma_atom_x,
                        tXgX[(None, token_block, k_idx)],
                        tXsX[(None, producer_state.index)],
                        tma_bar_ptr=load_pipeline.producer_get_barrier(producer_state),
                    )
                    load_pipeline.producer_commit(producer_state)
                    producer_state.advance()
            load_pipeline.producer_tail(producer_state)

        else:
            # -------------------------------------------------------- gate warps
            self.gate_warps(
                warp_idx - Int32(FIRST_GATE_WARP),
                worker_group,
                logical_rank,
                num_worker_groups,
                num_token_blocks,
                counters,
                counters_address,
                bias,
                image_bias,
                image_mask,
                mask,
                physical_map,
                logical_count,
                topk_idx,
                topk_weights,
                unmapped_idx,
                scratch,
                num_tokens,
                ep_rank,
                routed_scale,
                map_width,
            )

    # --------------------------------------------------------------- gate warps

    @cute.jit
    def gate_warps(
        self,
        gate_warp: Int32,
        worker_group: Int32,
        logical_rank: Int32,
        num_worker_groups: Int32,
        num_token_blocks: Int32,
        counters: cute.Tensor,
        counters_address: Int64,
        bias: cute.Tensor,
        image_bias: cute.Tensor,
        image_mask: cute.Tensor,
        mask: cute.Tensor,
        physical_map: cute.Tensor,
        logical_count: cute.Tensor,
        topk_idx: cute.Tensor,
        topk_weights: cute.Tensor,
        unmapped_idx: cute.Tensor,
        scratch: cute.Tensor,
        num_tokens: Int32,
        ep_rank: Int32,
        routed_scale: Float32,
        map_width: Int32,
    ):
        """Every logical CTA of a block ranks a share of its tokens, one warp per token."""
        lane = cute.arch.lane_idx()
        # The per-expert constants of this lane's experts are loaded once, off the
        # critical path: biases for the ranking and the replica counts of the mapping
        # (exchanged with the selected value by shuffles instead of a dependent load).
        lane_bias = cute.make_rmem_tensor((self.num_lane_values,), Float32)
        lane_image_bias = cute.make_rmem_tensor((self.num_lane_values,), Float32)
        lane_counts = cute.make_rmem_tensor((self.num_lane_values,), Int32)
        for wave in cutlass.range_constexpr(self.num_waves):
            expert = lane * Int32(EXPERTS_PER_LANE) + Int32(wave * EXPERTS_PER_WAVE)
            if const_expr(self.has_padded_experts):
                expert = cutlass.min(expert, Int32(self.num_experts - EXPERTS_PER_LANE))
            if const_expr(self.has_bias):
                values = load_vec4(bias, Int64(expert))
                for j in cutlass.range_constexpr(EXPERTS_PER_LANE):
                    lane_bias[wave * EXPERTS_PER_LANE + j] = values[j]
            if const_expr(self.has_image_bias):
                values = load_vec4(image_bias, Int64(expert))
                for j in cutlass.range_constexpr(EXPERTS_PER_LANE):
                    lane_image_bias[wave * EXPERTS_PER_LANE + j] = values[j]
            if const_expr(self.has_physical_map):
                counts = load_vec4(logical_count, Int64(expert))
                for j in cutlass.range_constexpr(EXPERTS_PER_LANE):
                    lane_counts[wave * EXPERTS_PER_LANE + j] = counts[j]
        # Shared slot lanes map a fixed logical expert: E + (lane - top-k).
        shared_count = Int32(1)
        if const_expr(self.has_physical_map and self.num_shared > 0):
            shared_slot = cutlass.min(
                cutlass.max(lane - Int32(self.num_topk), Int32(0)),
                Int32(self.num_shared - 1),
            )
            shared_count = logical_count[Int32(self.num_experts) + shared_slot]
        for token_block in cutlass.range(
            worker_group, num_token_blocks, num_worker_groups, unroll=1
        ):
            block_token_base = token_block * Int32(self.block_tokens)
            tokens_in_block = cutlass.min(
                Int32(self.block_tokens), num_tokens - block_token_base
            )
            # Only CTAs with a share of the block wait for it.
            if logical_rank < tokens_in_block:
                if gate_warp == Int32(0):
                    if lane == Int32(0):
                        address = counters_address + Int64(token_block) * Int64(
                            COUNTER_STRIDE * 4
                        )
                        arrived = ld_acquire_gpu(address)
                        while arrived != Int32(self.num_logical_ctas):
                            nanosleep(SPIN_BACKOFF_NS)
                            arrived = ld_acquire_gpu(address)
                    cute.arch.sync_warp()
                cute.arch.barrier(
                    barrier_id=GATE_BARRIER_ID,
                    number_of_threads=self.num_gate_warps * 32,
                )
                # Count this CTA's pass through the wait. The result is only consumed after
                # the CTA's share of the block is ranked, so the round trip overlaps it.
                passed = Int32(0)
                line = token_block * Int32(COUNTER_STRIDE)
                if gate_warp == Int32(0):
                    if lane == Int32(0):
                        passed = cute.arch.atomic_add(
                            counters.iterator + line + Int32(WAITERS_WORD),
                            Int32(1),
                            sem="relaxed",
                            scope="gpu",
                        )
                for t_local in cutlass.range(
                    gate_warp * Int32(self.num_logical_ctas) + logical_rank,
                    tokens_in_block,
                    self.num_gate_warps * self.num_logical_ctas,
                    unroll=1,
                ):
                    token = block_token_base + t_local
                    if const_expr(self.has_mask):
                        if mask[token] != Uint8(0):
                            self.route_token(
                                token,
                                t_local,
                                token_block,
                                lane,
                                lane_bias,
                                lane_image_bias,
                                lane_counts,
                                shared_count,
                                image_mask,
                                physical_map,
                                topk_idx,
                                topk_weights,
                                unmapped_idx,
                                scratch,
                                ep_rank,
                                routed_scale,
                                map_width,
                            )
                        else:
                            self.store_masked_token(
                                token, lane, topk_idx, topk_weights, unmapped_idx
                            )
                    else:
                        self.route_token(
                            token,
                            t_local,
                            token_block,
                            lane,
                            lane_bias,
                            lane_image_bias,
                            lane_counts,
                            shared_count,
                            image_mask,
                            physical_map,
                            topk_idx,
                            topk_weights,
                            unmapped_idx,
                            scratch,
                            ep_rank,
                            routed_scale,
                            map_width,
                        )
                # Every waiter read the full count before it counted itself, so once the
                # last one has counted nobody reads this line again in this launch: rearm
                # it for the next launch (or CUDA graph replay).
                if gate_warp == Int32(0):
                    if lane == Int32(0):
                        num_waiters = cutlass.min(
                            Int32(self.num_logical_ctas), tokens_in_block
                        )
                        if passed == num_waiters - Int32(1):
                            counters[line] = Int32(0)
                            counters[line + Int32(WAITERS_WORD)] = Int32(0)

    @cute.jit
    def store_masked_token(
        self,
        token: Int32,
        lane: Int32,
        topk_idx: cute.Tensor,
        topk_weights: cute.Tensor,
        unmapped_idx: cute.Tensor,
    ):
        """A masked token routes nowhere: index -1 and weight 0 in every slot."""
        if lane < Int32(self.num_slots):
            topk_idx[Int64(token), Int64(lane)] = Int64(-1)
            topk_weights[Int64(token), Int64(lane)] = Float32(0.0)
        if const_expr(self.has_unmapped):
            if lane < Int32(self.num_topk):
                unmapped_idx[Int64(token), Int64(lane)] = Int64(-1)

    @cute.jit
    def route_token(
        self,
        token: Int32,
        t_local: Int32,
        token_block: Int32,
        lane: Int32,
        lane_bias: cute.Tensor,
        lane_image_bias: cute.Tensor,
        lane_counts: cute.Tensor,
        shared_count: Int32,
        image_mask: cute.Tensor,
        physical_map: cute.Tensor,
        topk_idx: cute.Tensor,
        topk_weights: cute.Tensor,
        unmapped_idx: cute.Tensor,
        scratch: cute.Tensor,
        ep_rank: Int32,
        routed_scale: Float32,
        map_width: Int32,
    ):
        """Rank one token with one warp and store its routing slots."""
        neg_inf = Float32(-float("inf"))
        lane_expert_base = lane * Int32(EXPERTS_PER_LANE)

        # 1. Reduce the split-K partials in a fixed order: one float4 per wave and
        #    split, every load of a wave issued before the sum consumes them.
        row_stride = Int64(self.num_aligned_experts)
        split_stride = Int64(self.block_tokens) * row_stride
        row_base = (
            Int64(token_block) * Int64(self.num_split_k) * split_stride
            + Int64(t_local) * row_stride
        )
        logits = cute.make_rmem_tensor((self.num_lane_values,), Float32)
        for wave in cutlass.range_constexpr(self.num_waves):
            offset = row_base + Int64(lane_expert_base + Int32(wave * EXPERTS_PER_WAVE))
            partials = [
                load_vec4(scratch, offset + Int64(split) * split_stride)
                for split in range(self.num_split_k)
            ]
            total = partials[0]
            for split in cutlass.range_constexpr(1, self.num_split_k):
                total = total + partials[split]
            for j in cutlass.range_constexpr(EXPERTS_PER_LANE):
                logits[wave * EXPERTS_PER_LANE + j] = total[j]

        # 2. Rank on score + bias, or on the raw logits without a bias; NaN ranks last.
        ranks = cute.make_rmem_tensor((self.num_lane_values,), Float32)
        values = cute.make_rmem_tensor((self.num_lane_values,), Float32)
        is_image_token = False
        if const_expr(self.has_image_bias):
            is_image_token = image_mask[token] != Uint8(0)
        for wave in cutlass.range_constexpr(self.num_waves):
            # Padded experts (a whole lane vector, E being a multiple of 4) are ranked out.
            expert = lane_expert_base + Int32(wave * EXPERTS_PER_WAVE)
            for j in cutlass.range_constexpr(EXPERTS_PER_LANE):
                v = wave * EXPERTS_PER_LANE + j
                rank = logits[v]
                if const_expr(self.has_scoring_bias):
                    score = sqrt_softplus(logits[v])
                    values[v] = score
                    expert_bias = Float32(0.0)
                    if const_expr(self.has_bias):
                        expert_bias = lane_bias[v]
                    if const_expr(self.has_image_bias):
                        expert_bias = cutlass.select_(
                            is_image_token, lane_image_bias[v], expert_bias
                        )
                    rank = score + expert_bias
                else:
                    values[v] = logits[v]
                rank = cutlass.select_(rank == rank, rank, neg_inf)
                if const_expr(self.has_padded_experts):
                    rank = cutlass.select_(
                        expert < Int32(self.num_experts), rank, neg_inf
                    )
                ranks[v] = rank

        # 3. Sort the 4 candidates of every wave, best first (DeepGEMM's sort4 network).
        sorted_ranks = cute.make_rmem_tensor((self.num_lane_values,), Float32)
        sorted_offsets = cute.make_rmem_tensor((self.num_lane_values,), Int32)
        for wave in cutlass.range_constexpr(self.num_waves):
            b = wave * EXPERTS_PER_LANE
            r0, i0, r1, i1 = compare_swap_best_first(
                ranks[b], Int32(0), ranks[b + 1], Int32(1)
            )
            r2, i2, r3, i3 = compare_swap_best_first(
                ranks[b + 2], Int32(2), ranks[b + 3], Int32(3)
            )
            r0, i0, r2, i2 = compare_swap_best_first(r0, i0, r2, i2)
            r1, i1, r3, i3 = compare_swap_best_first(r1, i1, r3, i3)
            r1, i1, r2, i2 = compare_swap_best_first(r1, i1, r2, i2)
            sorted_ranks[b], sorted_offsets[b] = r0, i0
            sorted_ranks[b + 1], sorted_offsets[b + 1] = r1, i1
            sorted_ranks[b + 2], sorted_offsets[b + 2] = r2, i2
            sorted_ranks[b + 3], sorted_offsets[b + 3] = r3, i3

        # 4. Warp top-k: every round each lane offers the best unconsumed candidate of
        #    each wave, the warp keeps the largest rank (ties: the smaller expert),
        #    the winning lane advances that wave's cursor and lane r keeps round r.
        cursors = cute.make_rmem_tensor((self.num_waves,), Int32)
        for wave in cutlass.range_constexpr(self.num_waves):
            cursors[wave] = Int32(0)
        selected_expert = Int32(NO_EXPERT)
        for r in cutlass.range_constexpr(self.num_topk):
            best_rank = neg_inf
            best_expert = Int32(NO_EXPERT)
            best_wave = Int32(0)
            for wave in cutlass.range_constexpr(self.num_waves):
                b = wave * EXPERTS_PER_LANE
                cursor = cursors[wave]
                candidate = cutlass.select_(
                    cursor == Int32(0),
                    sorted_ranks[b],
                    cutlass.select_(
                        cursor == Int32(1),
                        sorted_ranks[b + 1],
                        cutlass.select_(
                            cursor == Int32(2),
                            sorted_ranks[b + 2],
                            cutlass.select_(
                                cursor == Int32(3), sorted_ranks[b + 3], neg_inf
                            ),
                        ),
                    ),
                )
                offset = cutlass.select_(
                    cursor == Int32(0),
                    sorted_offsets[b],
                    cutlass.select_(
                        cursor == Int32(1),
                        sorted_offsets[b + 1],
                        cutlass.select_(
                            cursor == Int32(2),
                            sorted_offsets[b + 2],
                            sorted_offsets[b + 3],
                        ),
                    ),
                )
                # Strict: a tie keeps the earlier wave, whose experts are smaller.
                take = candidate > best_rank
                best_rank = cutlass.select_(take, candidate, best_rank)
                best_expert = cutlass.select_(
                    take,
                    lane_expert_base + Int32(wave * EXPERTS_PER_WAVE) + offset,
                    best_expert,
                )
                best_wave = cutlass.select_(take, Int32(wave), best_wave)
            # One integer redux on order-preserving keys instead of a 5-step float
            # shuffle butterfly; ties resolve to the smaller expert.
            best_key = ordered_key(best_rank)
            top_key = cute.arch.warp_redux_sync(best_key, kind="max")
            winner = cute.arch.warp_redux_sync(
                cutlass.select_(best_key == top_key, best_expert, Int32(NO_EXPERT)),
                kind="min",
            )
            is_winner = best_expert == winner
            for wave in cutlass.range_constexpr(self.num_waves):
                cursors[wave] = cutlass.select_(
                    cutlass.and_(is_winner, best_wave == Int32(wave)),
                    cursors[wave] + Int32(1),
                    cursors[wave],
                )
            selected_expert = cutlass.select_(lane == Int32(r), winner, selected_expert)

        # 5. The selected expert's unbiased value, fetched from the lane that holds it.
        owner_lane = (selected_expert >> 2) & Int32(31)
        owner_value = ((selected_expert >> 7) << 2) | (selected_expert & Int32(3))
        selected_value = Float32(0.0)
        selected_count = Int32(1)
        for v in cutlass.range_constexpr(self.num_lane_values):
            exchanged = cute.arch.shuffle_sync(values[v], offset=owner_lane)
            selected_value = cutlass.select_(
                owner_value == Int32(v), exchanged, selected_value
            )
            if const_expr(self.has_physical_map):
                exchanged_count = cute.arch.shuffle_sync(
                    lane_counts[v], offset=owner_lane
                )
                selected_count = cutlass.select_(
                    owner_value == Int32(v), exchanged_count, selected_count
                )
        # Keep the slot in bounds should no finite candidate remain.
        is_routed = lane < Int32(self.num_topk)
        selected_expert = cutlass.min(
            cutlass.max(selected_expert, Int32(0)), Int32(self.num_experts - 1)
        )

        # 6. Normalized unbiased weights of the routed slots.
        if const_expr(not self.has_scoring_bias):
            selected_value = sqrt_softplus(selected_value)
        topk_sum = cutlass.select_(is_routed, selected_value, Float32(0.0))
        for step in cutlass.range_constexpr(self.sum_width.bit_length() - 1):
            topk_sum = topk_sum + cute.arch.shuffle_sync_bfly(
                topk_sum, offset=(self.sum_width // 2) >> step
            )
        topk_sum = topk_sum + Float32(TOPK_SUM_EPSILON)
        weight = cutlass.select_(
            is_routed, selected_value / topk_sum * routed_scale, Float32(1.0)
        )

        # 7. Shared experts follow the routed slots; map logical to physical experts.
        logical_expert = selected_expert
        if const_expr(self.num_shared > 0):
            logical_expert = cutlass.select_(
                is_routed,
                selected_expert,
                lane + Int32(self.num_experts - self.num_topk),
            )
        physical_expert = logical_expert
        is_slot = lane < Int32(self.num_slots)
        if const_expr(self.has_physical_map):
            if is_slot:
                count = selected_count
                if const_expr(self.num_shared > 0):
                    count = cutlass.select_(is_routed, selected_count, shared_count)
                # A count must be positive; clamp so a bad table cannot index out of bounds.
                num_duplicates = Uint32(cutlass.max(count, Int32(1)))
                duplicate = (
                    Uint32(ep_rank) + Uint32(token) * Uint32(DUPLICATE_HASH)
                ) % num_duplicates
                physical_expert = physical_map[
                    Int64(logical_expert) * Int64(map_width) + Int64(duplicate)
                ]
        if is_slot:
            topk_idx[Int64(token), Int64(lane)] = Int64(physical_expert)
            topk_weights[Int64(token), Int64(lane)] = weight
        if const_expr(self.has_unmapped):
            if is_routed:
                unmapped_idx[Int64(token), Int64(lane)] = Int64(selected_expert)
