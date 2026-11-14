# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0

"""CuTe DSL fused BF16 routing GEMM and sqrt-softplus top-k gate for SM120.

Port of DeepGEMM's ``sm100_bf16_mega_gate`` (the SM100a/SM103a/SM107a backend lives in
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
- 4 MMA warps along the experts, times 1 or 2 along the tokens (2 for 96- and
  128-token blocks, a 32x64 accumulator tile per warp): per token block, the
  mainloop, the partial-logit stores and the arrival at the block's score barrier;
  then straight on to the next block. When a CTA ranks more tokens of a block than
  it has gate warps (wide blocks, or smaller blocks over few CTAs), they rank tokens
  of the CTA's last block with the gate warps once their own work is done (nothing
  overlaps that block's gate work).
- 1 TMA producer warp.
- gate warps (4 or 8, per schedule): once all L logical CTAs of a block have arrived, every logical
  CTA ranks a share of its tokens (DeepGEMM's order: token ``gate_warp * L +
  rank``, stride ``gate_warps * L``), one warp per token, overlapping the MMA of
  the next block. With at most 3 tokens per CTA and block (on the 110-SM board up to
  256 tokens with 384 experts, 288 with 256) the gate is cooperative instead: every (token, 128-expert wave) item
  goes to one of gate warps 1.. (its split-K loads and transcendentals), is posted
  sorted to shared memory, and a selector warp per token runs the top-k rounds over
  all waves; gate warp 0 polls the barrier and selects the first token.

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
    mma_warps_rank,
)

MMA_WARPS_M = 4
"""MMA warps along the experts: 32 of the CTA's 128 experts each. The schedule sets the MMA
warps along the tokens (``mma_warps_n``); the TMA producer and the gate warps follow them."""
MMA_BARRIER_ID = 1
GATE_BARRIER_ID = 2
LAST_BLOCK_BARRIER_ID = 3
"""The gate warps and, when they help (``mma_warps_rank``), the MMA warps on the CTA's last block."""
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
KEY_NONE = 0x807FFFFF - (1 << 32)
"""``ordered_key(-inf)``: the key of an unrankable or exhausted candidate, never selected."""


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


@dsl_user_op
def sqrt_rn(value: Float32, *, loc=None, ip=None) -> Float32:
    """``sqrt.rn.f32`` without its slow-path branch, bitwise equal to it for every input
    but NaN payloads (checked exhaustively on SM120). ptxas expands ``sqrt.rn`` into
    rsqrt + one correction behind a range check that calls a slow path, so every square
    root is its own branch region and a lane's 12 of them run one after the other. Here
    the same fast path runs on inputs below 2^-101 scaled by 2^64 (exact, and back by
    2^-32), and zeros and +inf pass through: branch-free, so they interleave."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "{ .reg .f32 x, y, s, h, e, r; .reg .pred tiny, zero, inf; "
            "setp.lt.f32 tiny, $1, 0f0D000000; "
            "mul.f32 x, $1, 0f5F800000; "
            "selp.f32 x, x, $1, tiny; "
            "rsqrt.approx.ftz.f32 y, x; "
            "mul.ftz.f32 s, x, y; "
            "mul.ftz.f32 h, y, 0f3F000000; "
            "neg.f32 e, s; "
            "fma.rn.f32 e, e, s, x; "
            "fma.rn.f32 r, e, h, s; "
            "mul.f32 s, r, 0f2F800000; "
            "selp.f32 r, s, r, tiny; "
            "setp.eq.f32 zero, $1, 0f00000000; "
            "setp.eq.f32 inf, $1, 0f7F800000; "
            "or.pred zero, zero, inf; "
            "selp.f32 $0, $1, r, zero; }",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def mul_hi_u32(lhs: Uint32, rhs: Uint32, *, loc=None, ip=None) -> Uint32:
    """The high 32 bits of the 64-bit product of two uint32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(lhs).ir_value(loc=loc, ip=ip),
                Uint32(rhs).ir_value(loc=loc, ip=ip),
            ],
            "mul.hi.u32 $0, $1, $2;",
            "=r,r,r",
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


def store_vec4(tensor, offset, vector):
    """One 16 B vector store to ``tensor[offset : offset + 4]``; ``offset`` is a multiple of 4."""
    view = cute.make_tensor(
        tensor.iterator + cute.assume(offset, divby=4), cute.make_layout(4)
    )
    view.store(vector)


@cute.jit
def replica_hash(token: Int32, ep_rank: Int32) -> Uint32:
    """DeepGEMM's token hash, with uint32 wraparound before replica selection."""
    return Uint32(ep_rank) + Uint32(token) * Uint32(DUPLICATE_HASH)


def sqrt_softplus(value):
    """``sqrt(softplus(value))`` with the identity above the softplus threshold."""
    softplus = cute.math.log1p(cute.math.exp(value))
    softplus = cutlass.select_(value > Float32(SOFTPLUS_THRESHOLD), value, softplus)
    return sqrt_rn(softplus)


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
        enable_pdl: bool = False,
        has_fixed_routing: bool = False,
    ):
        self.enable_pdl = enable_pdl
        self.block_tokens = config.block_tokens
        self.num_split_k = config.num_split_k
        self.num_stages = config.num_stages
        self.num_gate_warps = config.num_gate_warps
        self.cooperative = config.cooperative_gate
        self.mma_warps_n = config.mma_warps_n
        self.num_mma_warps = MMA_WARPS_M * self.mma_warps_n
        self.producer_warp = self.num_mma_warps
        self.first_gate_warp = self.producer_warp + 1
        self.num_threads = (self.first_gate_warp + self.num_gate_warps) * 32
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
        # A cooperative gate ranks the CTA's (at most a few) tokens of a block with every
        # gate warp: (token slot, wave) items go to gate warps 1.. in turn.
        self.cooperative_slots = (
            -(-self.block_tokens // self.num_logical_ctas) if self.cooperative else 0
        )
        if self.cooperative and self.num_gate_warps < 2:
            raise ValueError("a cooperative gate needs at least 2 gate warps")
        # Without a cooperative gate, when a CTA ranks more tokens of a block than it has
        # gate warps (wide blocks, or smaller blocks over few CTAs), the MMA warps rank
        # tokens of the CTA's last block too:
        # nothing overlaps that block's gate work, and they have nothing left to do.
        self.mma_warps_rank = mma_warps_rank(
            self.block_tokens,
            self.num_logical_ctas,
            self.cooperative,
            self.num_gate_warps,
        )
        self.num_last_block_rank_warps = self.num_gate_warps + (
            self.num_mma_warps if self.mma_warps_rank else 0
        )
        self.k_blocks_per_split = hidden // BLOCK_K // self.num_split_k
        self.num_waves = self.num_aligned_experts // EXPERTS_PER_WAVE
        self.num_lane_values = self.num_waves * EXPERTS_PER_LANE
        self.cooperative_items = self.cooperative_slots * self.num_waves
        self.items_per_warp = -(
            -self.cooperative_items // max(self.num_gate_warps - 1, 1)
        )
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
        self.has_fixed_routing = has_fixed_routing
        # The smallest butterfly covering the routed lanes; the other lanes add zero.
        self.sum_width = 8 if num_topk <= 8 else (16 if num_topk <= 16 else 32)

    # ------------------------------------------------------------------ layouts

    def _tiled_mma(self) -> cute.TiledMma:
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.BFloat16, Float32, (16, 8, 16)),
            (MMA_WARPS_M, self.mma_warps_n, 1),
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
        # The cooperative gate's exchange (sorted keys and expert ids per lane quad, values
        # and physical experts by expert); a placeholder otherwise.
        exchange = (
            self.cooperative_slots * self.num_aligned_experts
            if self.cooperative
            else EXPERTS_PER_LANE
        )
        for name, dtype in (
            ("exchange_keys", Int32),
            ("exchange_experts", Int32),
            ("exchange_values", Float32),
            ("exchange_physical", Int32),
        ):
            SharedStorage.__annotations__[name] = cute.struct.Align[
                cute.struct.MemRange[dtype, exchange], 16
            ]
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
        fix_mask: cute.Tensor,
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
            fix_mask,
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
            use_pdl=self.enable_pdl,
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
        fix_mask: cute.Tensor,
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

        if warp_idx == Int32(self.producer_warp):
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
                pipeline.Agent.Thread, self.num_mma_warps
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

        # Programmatic dependent launch: whatever a prerequisite grid may write (the
        # tokens, the maps, this plan's workspace) is touched only after griddepcontrol.wait
        # (a no-op without PDL), so only the launch and the prologue overlap that grid.
        if warp_idx < Int32(self.num_mma_warps):
            # ---------------------------------------------------- MMA warps
            cute.arch.griddepcontrol_wait()
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
            # A warp's B slice per k16 is (block_tokens / mma_warps_n) x 16: four 8x8
            # matrices per ldmatrix when that is a multiple of 16 tokens, two otherwise.
            tokens_per_warp = self.block_tokens // self.mma_warps_n
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

                # Only the block's tokens are stored (the rows past the last token are never
                # read): fewer stores for the arrival's release to wait for.
                tokens_in_block = cutlass.min(
                    Int32(self.block_tokens),
                    num_tokens - token_block * Int32(self.block_tokens),
                )
                scratch_base = (
                    Int64(token_block) * Int64(self.num_split_k) + Int64(split_idx)
                ) * Int64(self.block_tokens) * row_stride + Int64(expert_group) * Int64(
                    EXPERTS_PER_CTA
                )
                for i in cutlass.range_constexpr(cute.size(acc)):
                    coord = coords[i]
                    if coord[1] < tokens_in_block:
                        scratch[
                            scratch_base
                            + Int64(coord[1]) * row_stride
                            + Int64(coord[0])
                        ] = acc[i]

                # Arrive at the block's score barrier once all four warps have stored. The
                # barrier orders their stores before thread 0's release, which publishes them
                # cumulatively (as CUTLASS's GenericBarrier::arrive_inc): no separate fence.
                cute.arch.barrier(
                    barrier_id=MMA_BARRIER_ID,
                    number_of_threads=self.num_mma_warps * 32,
                )
                if tidx == 0:
                    red_release_gpu_add(
                        counters_address
                        + Int64(token_block) * Int64(COUNTER_STRIDE * 4),
                        Int32(1),
                    )
                # Reconverge warp 0 before the next block's .aligned ldmatrix / mma.sync.
                cute.arch.sync_warp()

            if const_expr(self.mma_warps_rank):
                # The CTA's last block: only its gate work is left, so rank tokens of it
                # with the gate warps (after their poll: the shared barrier below).
                if worker_group < num_token_blocks:
                    last_block = (
                        worker_group
                        + (num_token_blocks - Int32(1) - worker_group)
                        // num_worker_groups
                        * num_worker_groups
                    )
                    block_token_base = last_block * Int32(self.block_tokens)
                    tokens_in_block = cutlass.min(
                        Int32(self.block_tokens), num_tokens - block_token_base
                    )
                    if logical_rank < tokens_in_block:
                        lane = cute.arch.lane_idx()
                        (
                            lane_bias,
                            lane_image_bias,
                            lane_counts,
                            lane_magic,
                            shared_count,
                            shared_magic,
                        ) = self.load_gate_constants(
                            lane, bias, image_bias, logical_count, map_width
                        )
                        cute.arch.barrier(
                            barrier_id=LAST_BLOCK_BARRIER_ID,
                            number_of_threads=self.num_last_block_rank_warps * 32,
                        )
                        # The last rank warp ranks a token only past
                        # (rank warps - 1) * L: it counts the CTA's wait.
                        if warp_idx == Int32(self.num_mma_warps - 1):
                            if lane == Int32(0):
                                self.count_waiter(counters, last_block, tokens_in_block)
                        self.rank_block_tokens(
                            Int32(self.num_gate_warps) + warp_idx,
                            Int32(self.num_last_block_rank_warps),
                            logical_rank,
                            last_block,
                            block_token_base,
                            tokens_in_block,
                            lane,
                            lane_bias,
                            lane_image_bias,
                            lane_counts,
                            lane_magic,
                            shared_count,
                            shared_magic,
                            image_mask,
                            mask,
                            fix_mask,
                            physical_map,
                            topk_idx,
                            topk_weights,
                            unmapped_idx,
                            scratch,
                            ep_rank,
                            routed_scale,
                            map_width,
                        )

        elif warp_idx == Int32(self.producer_warp):
            # --------------------------------------------------- TMA producer
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )
            # Under PDL the tokens (a prerequisite grid's output) are loaded only after
            # griddepcontrol.wait. Loading the first weight tiles before it was measured
            # slower whenever the wait returns at once: the token tiles then queue behind
            # them in the TMA unit and the first stage, so every refill, starts later.
            cute.arch.griddepcontrol_wait()
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
            cute.arch.griddepcontrol_wait()
            exchange_layout = cute.make_layout(
                self.cooperative_slots * self.num_aligned_experts
                if self.cooperative
                else EXPERTS_PER_LANE
            )
            self.gate_warps(
                warp_idx - Int32(self.first_gate_warp),
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
                fix_mask,
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
                storage.exchange_keys.get_tensor(exchange_layout),
                storage.exchange_experts.get_tensor(exchange_layout),
                storage.exchange_values.get_tensor(exchange_layout),
                storage.exchange_physical.get_tensor(exchange_layout),
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
        fix_mask: cute.Tensor,
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
        exchange_keys: cute.Tensor,
        exchange_experts: cute.Tensor,
        exchange_values: cute.Tensor,
        exchange_physical: cute.Tensor,
    ):
        """Every logical CTA of a block ranks a share of its tokens: one warp per token, or,
        with a cooperative gate (at most 3 tokens per CTA), every gate warp on its tokens."""
        lane = cute.arch.lane_idx()
        (
            lane_bias,
            lane_image_bias,
            lane_counts,
            lane_magic,
            shared_count,
            shared_magic,
        ) = self.load_gate_constants(lane, bias, image_bias, logical_count, map_width)
        for token_block in cutlass.range(
            worker_group, num_token_blocks, num_worker_groups, unroll=1
        ):
            block_token_base = token_block * Int32(self.block_tokens)
            tokens_in_block = cutlass.min(
                Int32(self.block_tokens), num_tokens - block_token_base
            )
            # Only CTAs with a share of the block wait for it.
            if logical_rank < tokens_in_block:
                selector_shared_physical = Int32(0)
                selector_fixed = Int32(0)
                selector_given = Int32(NO_EXPERT)
                if const_expr(self.cooperative):
                    # The CTA's tokens are known before their logits are: look up their
                    # replicas (and any fixed routing) while waiting.
                    selector_shared_physical = self.cooperative_lookups(
                        gate_warp,
                        lane,
                        logical_rank,
                        block_token_base,
                        tokens_in_block,
                        physical_map,
                        lane_counts,
                        lane_magic,
                        shared_count,
                        shared_magic,
                        ep_rank,
                        map_width,
                        exchange_physical,
                    )
                    if const_expr(self.has_fixed_routing):
                        selector_fixed, selector_given = self.cooperative_fixed(
                            gate_warp,
                            lane,
                            logical_rank,
                            block_token_base,
                            tokens_in_block,
                            fix_mask,
                            unmapped_idx,
                        )
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
                last_block = token_block + num_worker_groups >= num_token_blocks
                if const_expr(self.mma_warps_rank):
                    if last_block:
                        cute.arch.barrier(
                            barrier_id=LAST_BLOCK_BARRIER_ID,
                            number_of_threads=self.num_last_block_rank_warps * 32,
                        )
                    else:
                        cute.arch.barrier(
                            barrier_id=GATE_BARRIER_ID,
                            number_of_threads=self.num_gate_warps * 32,
                        )
                else:
                    cute.arch.barrier(
                        barrier_id=GATE_BARRIER_ID,
                        number_of_threads=self.num_gate_warps * 32,
                    )
                # In the CTA's last block only the gate's share is left (the block's MMA
                # is complete everywhere): let a PDL dependent launch now.
                if last_block:
                    cute.arch.griddepcontrol_launch_dependents()
                # Count this CTA's pass through the wait (and rearm the line once every
                # waiter has), from a warp that ranks nothing first: gate warp 0 of a
                # cooperative gate, which waits for the waves anyway, else the last gate
                # warp, which ranks tokens of a block only past (gate warps - 1) * L (the
                # last MMA warp counts a last block the MMA warps help with).
                if gate_warp == Int32(
                    0 if self.cooperative else self.num_gate_warps - 1
                ):
                    if lane == Int32(0):
                        if const_expr(self.mma_warps_rank):
                            if cutlass.not_(last_block):
                                self.count_waiter(
                                    counters, token_block, tokens_in_block
                                )
                        else:
                            self.count_waiter(counters, token_block, tokens_in_block)
                if const_expr(self.cooperative):
                    self.cooperative_block(
                        gate_warp,
                        lane,
                        logical_rank,
                        token_block,
                        block_token_base,
                        tokens_in_block,
                        lane_bias,
                        lane_image_bias,
                        image_mask,
                        mask,
                        fix_mask,
                        physical_map,
                        topk_idx,
                        topk_weights,
                        unmapped_idx,
                        scratch,
                        routed_scale,
                        map_width,
                        selector_shared_physical,
                        selector_fixed,
                        selector_given,
                        exchange_keys,
                        exchange_experts,
                        exchange_values,
                        exchange_physical,
                    )
                    # The CTA's next block reuses the exchange (its replicas are posted
                    # before its wait).
                    if token_block + num_worker_groups < num_token_blocks:
                        cute.arch.barrier(
                            barrier_id=GATE_BARRIER_ID,
                            number_of_threads=self.num_gate_warps * 32,
                        )
                else:
                    num_rank_warps = Int32(self.num_gate_warps)
                    if const_expr(self.mma_warps_rank):
                        num_rank_warps = cutlass.select_(
                            last_block,
                            Int32(self.num_last_block_rank_warps),
                            num_rank_warps,
                        )
                    self.rank_block_tokens(
                        gate_warp,
                        num_rank_warps,
                        logical_rank,
                        token_block,
                        block_token_base,
                        tokens_in_block,
                        lane,
                        lane_bias,
                        lane_image_bias,
                        lane_counts,
                        lane_magic,
                        shared_count,
                        shared_magic,
                        image_mask,
                        mask,
                        fix_mask,
                        physical_map,
                        topk_idx,
                        topk_weights,
                        unmapped_idx,
                        scratch,
                        ep_rank,
                        routed_scale,
                        map_width,
                    )

    def load_gate_constants(self, lane, bias, image_bias, logical_count, map_width):
        """The per-expert constants of this lane's experts, prepared once off the critical
        path: biases for the ranking, and the replica counts of the mapping with their
        reciprocals (a token's replica of every lane expert is looked up early). Shared
        slot lanes map a fixed logical expert: E + (lane - top-k)."""
        lane_bias = cute.make_rmem_tensor((self.num_lane_values,), Float32)
        lane_image_bias = cute.make_rmem_tensor((self.num_lane_values,), Float32)
        lane_counts = cute.make_rmem_tensor((self.num_lane_values,), Int32)
        lane_magic = cute.make_rmem_tensor((self.num_lane_values,), Uint32)
        for wave in range(self.num_waves):
            expert = lane * Int32(EXPERTS_PER_LANE) + Int32(wave * EXPERTS_PER_WAVE)
            if const_expr(self.has_padded_experts):
                expert = cutlass.min(expert, Int32(self.num_experts - EXPERTS_PER_LANE))
            if const_expr(self.has_bias):
                values = load_vec4(bias, Int64(expert))
                for j in range(EXPERTS_PER_LANE):
                    lane_bias[wave * EXPERTS_PER_LANE + j] = values[j]
            if const_expr(self.has_image_bias):
                values = load_vec4(image_bias, Int64(expert))
                for j in range(EXPERTS_PER_LANE):
                    lane_image_bias[wave * EXPERTS_PER_LANE + j] = values[j]
            if const_expr(self.has_physical_map):
                counts = load_vec4(logical_count, Int64(expert))
                for j in range(EXPERTS_PER_LANE):
                    v = wave * EXPERTS_PER_LANE + j
                    lane_counts[v] = self.clamp_count(counts[j], map_width)
                    lane_magic[v] = self.count_magic(lane_counts[v])
        shared_count = Int32(1)
        shared_magic = Uint32(0)
        if const_expr(self.has_physical_map and self.num_shared > 0):
            shared_slot = cutlass.min(
                cutlass.max(lane - Int32(self.num_topk), Int32(0)),
                Int32(self.num_shared - 1),
            )
            shared_count = self.clamp_count(
                logical_count[Int32(self.num_experts) + shared_slot], map_width
            )
            shared_magic = self.count_magic(shared_count)
        return (
            lane_bias,
            lane_image_bias,
            lane_counts,
            lane_magic,
            shared_count,
            shared_magic,
        )

    @cute.jit
    def rank_block_tokens(
        self,
        rank_warp: Int32,
        num_rank_warps: Int32,
        logical_rank: Int32,
        token_block: Int32,
        block_token_base: Int32,
        tokens_in_block: Int32,
        lane: Int32,
        lane_bias: cute.Tensor,
        lane_image_bias: cute.Tensor,
        lane_counts: cute.Tensor,
        lane_magic: cute.Tensor,
        shared_count: Int32,
        shared_magic: Uint32,
        image_mask: cute.Tensor,
        mask: cute.Tensor,
        fix_mask: cute.Tensor,
        physical_map: cute.Tensor,
        topk_idx: cute.Tensor,
        topk_weights: cute.Tensor,
        unmapped_idx: cute.Tensor,
        scratch: cute.Tensor,
        ep_rank: Int32,
        routed_scale: Float32,
        map_width: Int32,
    ):
        """This warp's tokens of a block, one at a time, in DeepGEMM's order: token
        ``rank_warp * L + rank``, stride ``num_rank_warps * L``."""
        for t_local in cutlass.range(
            rank_warp * Int32(self.num_logical_ctas) + logical_rank,
            tokens_in_block,
            num_rank_warps * Int32(self.num_logical_ctas),
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
                        lane_magic,
                        shared_count,
                        shared_magic,
                        image_mask,
                        fix_mask,
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
                    lane_magic,
                    shared_count,
                    shared_magic,
                    image_mask,
                    fix_mask,
                    physical_map,
                    topk_idx,
                    topk_weights,
                    unmapped_idx,
                    scratch,
                    ep_rank,
                    routed_scale,
                    map_width,
                )

    def wave_value(self, lane_values, wave, j):
        """``lane_values[4 * wave + j]`` for a runtime ``wave``: a select over the waves
        (an indexed register array would go to local memory)."""
        value = lane_values[j]
        for w in range(1, self.num_waves):
            value = cutlass.select_(
                wave == Int32(w), lane_values[w * EXPERTS_PER_LANE + j], value
            )
        return value

    def slot_selector(self, slot: int) -> int:
        """The gate warp that runs a cooperative token slot's top-k rounds: warp 0 (which
        ranks no wave: it polls and counts the CTA's wait), then the last gate warps."""
        return 0 if slot == 0 else self.num_gate_warps - slot

    def cooperative_item(self, gate_warp, k, logical_rank, tokens_in_block):
        """This warp's ``k``-th (token slot, wave) item of a cooperative gate: items go to
        gate warps 1, 2, ... in turn. Returns the slot, the wave, the token in the block and
        whether the item exists."""
        item = gate_warp - Int32(1) + Int32(k * (self.num_gate_warps - 1))
        slot = item // Int32(self.num_waves)
        wave = item - slot * Int32(self.num_waves)
        t_local = logical_rank + slot * Int32(self.num_logical_ctas)
        valid = cutlass.and_(
            cutlass.and_(gate_warp > Int32(0), item < Int32(self.cooperative_items)),
            t_local < tokens_in_block,
        )
        return slot, wave, t_local, valid

    @cute.jit
    def count_waiter(
        self, counters: cute.Tensor, token_block: Int32, tokens_in_block: Int32
    ):
        """Count this CTA's pass through a block's wait. Every waiter read the full count
        before it counted itself, so once the last one has counted nobody reads the line
        again in this launch: rearm it for the next launch (or CUDA graph replay)."""
        line = token_block * Int32(COUNTER_STRIDE)
        passed = cute.arch.atomic_add(
            counters.iterator + line + Int32(WAITERS_WORD),
            Int32(1),
            sem="relaxed",
            scope="gpu",
        )
        num_waiters = cutlass.min(Int32(self.num_logical_ctas), tokens_in_block)
        if passed == num_waiters - Int32(1):
            counters[line] = Int32(0)
            counters[line + Int32(WAITERS_WORD)] = Int32(0)

    @cute.jit
    def clamp_count(self, count: Int32, map_width: Int32) -> Int32:
        """A replica count must be positive and at most the map width: clamp so that a bad
        table cannot index out of bounds (every lane looks up all of its experts)."""
        return cutlass.min(cutlass.max(count, Int32(1)), map_width)

    @cute.jit
    def count_magic(self, count: Int32) -> Uint32:
        """Reciprocal of a replica count for ``replica``: ``0xFFFFFFFF // count + 1``
        (0 for a count of 1)."""
        return Uint32(0xFFFFFFFF) // Uint32(count) + Uint32(1)

    @cute.jit
    def replica(self, token_hash: Uint32, count: Int32, magic: Uint32) -> Int32:
        """DeepGEMM's replica of a logical expert for a token, ``token_hash % count``, as a
        multiply by the count's reciprocal (a dozen integer ops, against a division's
        float conversions and reciprocal per lookup): the quotient estimate is exact or
        one too large, so the remainder is exact or wrapped below zero by the count."""
        divisor = Uint32(count)
        remainder = token_hash - mul_hi_u32(token_hash, magic) * divisor
        remainder = cutlass.select_(
            remainder >= divisor, remainder + divisor, remainder
        )
        return Int32(cutlass.select_(divisor == Uint32(1), Uint32(0), remainder))

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

    def load_partials(self, scratch, token_block, t_local, lane, wave):
        """One token's split-K partials of the lane's 4 experts of a wave: a float4 per split."""
        row_stride = Int64(self.num_aligned_experts)
        split_stride = Int64(self.block_tokens) * row_stride
        base = (
            Int64(token_block) * Int64(self.num_split_k) * split_stride
            + Int64(t_local) * row_stride
            + Int64(lane * Int32(EXPERTS_PER_LANE) + Int32(wave * EXPERTS_PER_WAVE))
        )
        return [
            load_vec4(scratch, base + Int64(split) * split_stride)
            for split in range(self.num_split_k)
        ]

    def lookup_physical(
        self,
        physical_map,
        token_hash,
        lane,
        wave,
        lane_counts,
        lane_magic,
        map_width,
        out,
    ):
        """A token's replica of the lane's 4 experts of a wave, into ``out[4 * wave:]``."""
        expert = lane * Int32(EXPERTS_PER_LANE) + Int32(wave * EXPERTS_PER_WAVE)
        if const_expr(self.has_padded_experts):
            expert = cutlass.min(expert, Int32(self.num_experts - EXPERTS_PER_LANE))
        for j in range(EXPERTS_PER_LANE):
            v = wave * EXPERTS_PER_LANE + j
            out[v] = physical_map[
                Int64(expert + Int32(j)) * Int64(map_width)
                + Int64(self.replica(token_hash, lane_counts[v], lane_magic[v]))
            ]

    def lookup_shared_physical(
        self, physical_map, token_hash, lane, shared_count, shared_magic, map_width
    ):
        """A token's replica of the shared expert of the lane's slot (shared slot lanes)."""
        shared_slot = cutlass.min(
            cutlass.max(lane - Int32(self.num_topk), Int32(0)),
            Int32(self.num_shared - 1),
        )
        return physical_map[
            Int64(Int32(self.num_experts) + shared_slot) * Int64(map_width)
            + Int64(self.replica(token_hash, shared_count, shared_magic))
        ]

    def rank_wave(
        self,
        partials,
        lane,
        wave,
        wave_bias,
        wave_image_bias,
        is_image_token,
        keys,
        experts,
        values,
        slot,
    ):
        """Sum a wave's partials in a fixed order, rank the lane's 4 experts on score + bias
        (or on the raw logits without a bias) and sort them best first (DeepGEMM's sort4).
        From ``slot``: ``keys`` and ``experts`` get the sorted order keys and expert ids,
        ``values`` the unbiased values by expert. ``wave_bias`` / ``wave_image_bias``: the
        4 experts' biases."""
        neg_inf = Float32(-float("inf"))
        total = partials[0]
        for split in range(1, self.num_split_k):
            total = total + partials[split]
        expert_base = lane * Int32(EXPERTS_PER_LANE) + Int32(wave * EXPERTS_PER_WAVE)
        ranks = []
        for j in range(EXPERTS_PER_LANE):
            rank = total[j]
            if const_expr(self.has_scoring_bias):
                score = sqrt_softplus(total[j])
                values[slot + j] = score
                expert_bias = Float32(0.0)
                if const_expr(self.has_bias):
                    expert_bias = wave_bias[j]
                if const_expr(self.has_image_bias):
                    expert_bias = cutlass.select_(
                        is_image_token, wave_image_bias[j], expert_bias
                    )
                rank = score + expert_bias
            else:
                values[slot + j] = total[j]
            # NaN ranks last; padded experts (a whole lane vector, E being a multiple
            # of 4) are ranked out.
            rank = cutlass.select_(rank == rank, rank, neg_inf)
            if const_expr(self.has_padded_experts):
                rank = cutlass.select_(
                    expert_base < Int32(self.num_experts), rank, neg_inf
                )
            ranks.append(rank)
        r0, i0, r1, i1 = compare_swap_best_first(ranks[0], Int32(0), ranks[1], Int32(1))
        r2, i2, r3, i3 = compare_swap_best_first(ranks[2], Int32(2), ranks[3], Int32(3))
        r0, i0, r2, i2 = compare_swap_best_first(r0, i0, r2, i2)
        r1, i1, r3, i3 = compare_swap_best_first(r1, i1, r3, i3)
        r1, i1, r2, i2 = compare_swap_best_first(r1, i1, r2, i2)
        for position, (rank, offset) in enumerate(
            ((r0, i0), (r1, i1), (r2, i2), (r3, i3))
        ):
            keys[slot + position] = ordered_key(rank)
            experts[slot + position] = expert_base + offset

    def topk_rounds(self, lane, keys, experts):
        """Warp top-k over the lanes' sorted wave lists (``keys``/``experts`` at 4 * wave):
        every round each lane offers the best unconsumed candidate of each wave, the warp
        keeps the largest key (ties: the smaller expert), the winning lane advances that
        wave's cursor and lane r keeps round r's expert (NO_EXPERT if none was left)."""
        cursors = [Int32(0)] * self.num_waves
        selected_expert = Int32(NO_EXPERT)
        for r in range(self.num_topk):
            best_key = Int32(KEY_NONE)
            best_expert = Int32(NO_EXPERT)
            best_wave = Int32(0)
            for wave in range(self.num_waves):
                b = wave * EXPERTS_PER_LANE
                cursor = cursors[wave]
                candidate = cutlass.select_(
                    cursor == Int32(0),
                    keys[b],
                    cutlass.select_(
                        cursor == Int32(1),
                        keys[b + 1],
                        cutlass.select_(
                            cursor == Int32(2),
                            keys[b + 2],
                            cutlass.select_(
                                cursor == Int32(3), keys[b + 3], Int32(KEY_NONE)
                            ),
                        ),
                    ),
                )
                expert = cutlass.select_(
                    cursor == Int32(0),
                    experts[b],
                    cutlass.select_(
                        cursor == Int32(1),
                        experts[b + 1],
                        cutlass.select_(
                            cursor == Int32(2), experts[b + 2], experts[b + 3]
                        ),
                    ),
                )
                # Strict: a tie keeps the earlier wave, whose experts are smaller.
                take = candidate > best_key
                best_key = cutlass.select_(take, candidate, best_key)
                best_expert = cutlass.select_(take, expert, best_expert)
                best_wave = cutlass.select_(take, Int32(wave), best_wave)
            # One integer redux on order-preserving keys (SM120 has no float redux) instead
            # of a 5-step shuffle butterfly; ties resolve to the smaller expert.
            top_key = cute.arch.warp_redux_sync(best_key, kind="max")
            winner = cute.arch.warp_redux_sync(
                cutlass.select_(best_key == top_key, best_expert, Int32(NO_EXPERT)),
                kind="min",
            )
            is_winner = best_expert == winner
            for wave in range(self.num_waves):
                cursors[wave] = cutlass.select_(
                    cutlass.and_(is_winner, best_wave == Int32(wave)),
                    cursors[wave] + Int32(1),
                    cursors[wave],
                )
            selected_expert = cutlass.select_(lane == Int32(r), winner, selected_expert)
        return selected_expert

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
        lane_magic: cute.Tensor,
        shared_count: Int32,
        shared_magic: Uint32,
        image_mask: cute.Tensor,
        fix_mask: cute.Tensor,
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
        # 1. Load every split-K partial of the lane's experts before anything consumes them
        #    (and a fixed-routing token's given experts).
        partials = [
            self.load_partials(scratch, token_block, t_local, lane, wave)
            for wave in range(self.num_waves)
        ]
        fixed = Int32(0)
        given_expert = Int32(NO_EXPERT)
        if const_expr(self.has_fixed_routing):
            fixed, given_expert = self.load_fixed_experts(
                token, lane, fix_mask, unmapped_idx
            )
        # While they are in flight, look up this token's replica of every lane expert (and
        # of the shared slot): no dependent lookup is left after the top-k.
        lane_physical = cute.make_rmem_tensor((self.num_lane_values,), Int32)
        shared_physical = Int32(0)
        if const_expr(self.has_physical_map):
            token_hash = replica_hash(token, ep_rank)
            for wave in cutlass.range_constexpr(self.num_waves):
                self.lookup_physical(
                    physical_map,
                    token_hash,
                    lane,
                    wave,
                    lane_counts,
                    lane_magic,
                    map_width,
                    lane_physical,
                )
            if const_expr(self.num_shared > 0):
                shared_physical = self.lookup_shared_physical(
                    physical_map,
                    token_hash,
                    lane,
                    shared_count,
                    shared_magic,
                    map_width,
                )

        # 2. Sum, rank and sort the lane's experts of every wave.
        is_image_token = False
        if const_expr(self.has_image_bias):
            is_image_token = image_mask[token] != Uint8(0)
        keys = cute.make_rmem_tensor((self.num_lane_values,), Int32)
        experts = cute.make_rmem_tensor((self.num_lane_values,), Int32)
        values = cute.make_rmem_tensor((self.num_lane_values,), Float32)
        for wave in cutlass.range_constexpr(self.num_waves):
            self.rank_wave(
                partials[wave],
                lane,
                wave,
                [
                    lane_bias[wave * EXPERTS_PER_LANE + j]
                    for j in range(EXPERTS_PER_LANE)
                ],
                [
                    lane_image_bias[wave * EXPERTS_PER_LANE + j]
                    for j in range(EXPERTS_PER_LANE)
                ],
                is_image_token,
                keys,
                experts,
                values,
                wave * EXPERTS_PER_LANE,
            )

        # 3. Warp top-k (a fixed-routing token keeps its given experts instead).
        selected_expert = given_expert
        if const_expr(self.has_fixed_routing):
            if fixed == Int32(0):
                selected_expert = self.topk_rounds(lane, keys, experts)
        else:
            selected_expert = self.topk_rounds(lane, keys, experts)

        # 4. The selected expert's unbiased value and physical expert, fetched from the
        #    lane that holds them.
        owner_lane = (selected_expert >> 2) & Int32(31)
        owner_value = ((selected_expert >> 7) << 2) | (selected_expert & Int32(3))
        selected_value = Float32(0.0)
        selected_physical = Int32(0)
        for v in cutlass.range_constexpr(self.num_lane_values):
            exchanged = cute.arch.shuffle_sync(values[v], offset=owner_lane)
            selected_value = cutlass.select_(
                owner_value == Int32(v), exchanged, selected_value
            )
            if const_expr(self.has_physical_map):
                exchanged_physical = cute.arch.shuffle_sync(
                    lane_physical[v], offset=owner_lane
                )
                selected_physical = cutlass.select_(
                    owner_value == Int32(v), exchanged_physical, selected_physical
                )
        self.finish_token(
            token,
            lane,
            selected_expert,
            selected_value,
            selected_physical,
            shared_physical,
            physical_map,
            topk_idx,
            topk_weights,
            unmapped_idx,
            routed_scale,
            map_width,
        )

    @cute.jit
    def cooperative_lookups(
        self,
        gate_warp: Int32,
        lane: Int32,
        logical_rank: Int32,
        block_token_base: Int32,
        tokens_in_block: Int32,
        physical_map: cute.Tensor,
        lane_counts: cute.Tensor,
        lane_magic: cute.Tensor,
        shared_count: Int32,
        shared_magic: Uint32,
        ep_rank: Int32,
        map_width: Int32,
        exchange_physical: cute.Tensor,
    ) -> Int32:
        """Before a block's score wait (its tokens are known, their logits not yet): post
        the replicas of the waves this warp ranks to shared memory, and return the shared
        slot's replica of the token this warp selects."""
        shared_physical = Int32(0)
        if const_expr(self.has_physical_map):
            for k in cutlass.range_constexpr(self.items_per_warp):
                slot, wave, t_local, valid = self.cooperative_item(
                    gate_warp, k, logical_rank, tokens_in_block
                )
                if valid:
                    token_hash = replica_hash(block_token_base + t_local, ep_rank)
                    quad = lane * Int32(EXPERTS_PER_LANE) + wave * Int32(
                        EXPERTS_PER_WAVE
                    )
                    expert = quad
                    if const_expr(self.has_padded_experts):
                        expert = cutlass.min(
                            expert, Int32(self.num_experts - EXPERTS_PER_LANE)
                        )
                    physical = cute.make_rmem_tensor((EXPERTS_PER_LANE,), Int32)
                    for j in cutlass.range_constexpr(EXPERTS_PER_LANE):
                        replica = self.replica(
                            token_hash,
                            self.wave_value(lane_counts, wave, j),
                            self.wave_value(lane_magic, wave, j),
                        )
                        physical[j] = physical_map[
                            Int64(expert + Int32(j)) * Int64(map_width) + Int64(replica)
                        ]
                    store_vec4(
                        exchange_physical,
                        slot * Int32(self.num_aligned_experts) + quad,
                        physical.load(),
                    )
            if const_expr(self.num_shared > 0):
                for slot in cutlass.range_constexpr(self.cooperative_slots):
                    if gate_warp == Int32(self.slot_selector(slot)):
                        t_local = logical_rank + Int32(slot * self.num_logical_ctas)
                        if t_local < tokens_in_block:
                            token_hash = replica_hash(
                                block_token_base + t_local, ep_rank
                            )
                            shared_physical = self.lookup_shared_physical(
                                physical_map,
                                token_hash,
                                lane,
                                shared_count,
                                shared_magic,
                                map_width,
                            )
        return shared_physical

    @cute.jit
    def cooperative_block(
        self,
        gate_warp: Int32,
        lane: Int32,
        logical_rank: Int32,
        token_block: Int32,
        block_token_base: Int32,
        tokens_in_block: Int32,
        lane_bias: cute.Tensor,
        lane_image_bias: cute.Tensor,
        image_mask: cute.Tensor,
        mask: cute.Tensor,
        fix_mask: cute.Tensor,
        physical_map: cute.Tensor,
        topk_idx: cute.Tensor,
        topk_weights: cute.Tensor,
        unmapped_idx: cute.Tensor,
        scratch: cute.Tensor,
        routed_scale: Float32,
        map_width: Int32,
        selector_shared_physical: Int32,
        selector_fixed: Int32,
        selector_given: Int32,
        exchange_keys: cute.Tensor,
        exchange_experts: cute.Tensor,
        exchange_values: cute.Tensor,
        exchange_physical: cute.Tensor,
    ):
        """Rank this CTA's tokens of a block (at most ``cooperative_slots``) with every gate
        warp. Each (token, 128-expert wave) item is ranked by one warp: the split-K loads
        and transcendentals of a wave per lane, every item's loads in flight at once. The
        items are posted to shared memory as sorted keys and expert ids plus values by
        expert; then each token's selector warp runs the top-k rounds over its waves."""
        # 1. Load the partials of every item this warp ranks (a missing item loads the
        #    CTA's first token of wave 0, which exists, and drops it).
        partials = []
        for k in cutlass.range_constexpr(self.items_per_warp):
            slot, wave, t_local, valid = self.cooperative_item(
                gate_warp, k, logical_rank, tokens_in_block
            )
            partials.append(
                self.load_partials(
                    scratch,
                    token_block,
                    cutlass.select_(valid, t_local, logical_rank),
                    lane,
                    cutlass.select_(valid, wave, Int32(0)),
                )
            )
        # 2. Rank, sort and post them.
        for k in cutlass.range_constexpr(self.items_per_warp):
            slot, wave, t_local, valid = self.cooperative_item(
                gate_warp, k, logical_rank, tokens_in_block
            )
            if valid:
                token = block_token_base + t_local
                if const_expr(self.has_mask):
                    if mask[token] != Uint8(0):
                        self.post_item(
                            partials[k],
                            token,
                            slot,
                            wave,
                            lane,
                            lane_bias,
                            lane_image_bias,
                            image_mask,
                            exchange_keys,
                            exchange_experts,
                            exchange_values,
                        )
                else:
                    self.post_item(
                        partials[k],
                        token,
                        slot,
                        wave,
                        lane,
                        lane_bias,
                        lane_image_bias,
                        image_mask,
                        exchange_keys,
                        exchange_experts,
                        exchange_values,
                    )
        cute.arch.barrier(
            barrier_id=GATE_BARRIER_ID, number_of_threads=self.num_gate_warps * 32
        )
        # 3. Each token's selector runs the top-k rounds and stores the token.
        for slot in cutlass.range_constexpr(self.cooperative_slots):
            if gate_warp == Int32(self.slot_selector(slot)):
                t_local = logical_rank + Int32(slot * self.num_logical_ctas)
                if t_local < tokens_in_block:
                    token = block_token_base + t_local
                    if const_expr(self.has_mask):
                        if mask[token] != Uint8(0):
                            self.select_slot(
                                slot,
                                token,
                                lane,
                                selector_shared_physical,
                                physical_map,
                                topk_idx,
                                topk_weights,
                                unmapped_idx,
                                routed_scale,
                                map_width,
                                exchange_keys,
                                exchange_experts,
                                exchange_values,
                                exchange_physical,
                                selector_fixed,
                                selector_given,
                            )
                        else:
                            self.store_masked_token(
                                token, lane, topk_idx, topk_weights, unmapped_idx
                            )
                    else:
                        self.select_slot(
                            slot,
                            token,
                            lane,
                            selector_shared_physical,
                            physical_map,
                            topk_idx,
                            topk_weights,
                            unmapped_idx,
                            routed_scale,
                            map_width,
                            exchange_keys,
                            exchange_experts,
                            exchange_values,
                            exchange_physical,
                            selector_fixed,
                            selector_given,
                        )

    @cute.jit
    def post_item(
        self,
        partials,
        token: Int32,
        slot: Int32,
        wave: Int32,
        lane: Int32,
        lane_bias: cute.Tensor,
        lane_image_bias: cute.Tensor,
        image_mask: cute.Tensor,
        exchange_keys: cute.Tensor,
        exchange_experts: cute.Tensor,
        exchange_values: cute.Tensor,
    ):
        """Rank and sort the lane's 4 experts of a wave of a token, posted at the token
        slot's exchange."""
        quad = lane * Int32(EXPERTS_PER_LANE) + wave * Int32(EXPERTS_PER_WAVE)
        wave_bias = [Float32(0.0)] * EXPERTS_PER_LANE
        wave_image_bias = [Float32(0.0)] * EXPERTS_PER_LANE
        is_image_token = False
        if const_expr(self.has_bias):
            wave_bias = [
                self.wave_value(lane_bias, wave, j) for j in range(EXPERTS_PER_LANE)
            ]
        if const_expr(self.has_image_bias):
            wave_image_bias = [
                self.wave_value(lane_image_bias, wave, j)
                for j in range(EXPERTS_PER_LANE)
            ]
            is_image_token = image_mask[token] != Uint8(0)
        keys = cute.make_rmem_tensor((EXPERTS_PER_LANE,), Int32)
        experts = cute.make_rmem_tensor((EXPERTS_PER_LANE,), Int32)
        values = cute.make_rmem_tensor((EXPERTS_PER_LANE,), Float32)
        self.rank_wave(
            partials,
            lane,
            wave,
            wave_bias,
            wave_image_bias,
            is_image_token,
            keys,
            experts,
            values,
            0,
        )
        offset = slot * Int32(self.num_aligned_experts) + quad
        store_vec4(exchange_keys, offset, keys.load())
        store_vec4(exchange_experts, offset, experts.load())
        store_vec4(exchange_values, offset, values.load())

    @cute.jit
    def select_slot(
        self,
        slot: int,
        token: Int32,
        lane: Int32,
        shared_physical: Int32,
        physical_map: cute.Tensor,
        topk_idx: cute.Tensor,
        topk_weights: cute.Tensor,
        unmapped_idx: cute.Tensor,
        routed_scale: Float32,
        map_width: Int32,
        exchange_keys: cute.Tensor,
        exchange_experts: cute.Tensor,
        exchange_values: cute.Tensor,
        exchange_physical: cute.Tensor,
        fixed: Int32,
        given_expert: Int32,
    ):
        """Top-k rounds over a cooperative token slot's posted waves, then its slots (a
        fixed-routing token keeps its given experts: ``cooperative_fixed``)."""
        base = Int32(slot * self.num_aligned_experts)
        keys = cute.make_rmem_tensor((self.num_lane_values,), Int32)
        experts = cute.make_rmem_tensor((self.num_lane_values,), Int32)
        for wave in cutlass.range_constexpr(self.num_waves):
            quad = (
                base + lane * Int32(EXPERTS_PER_LANE) + Int32(wave * EXPERTS_PER_WAVE)
            )
            wave_keys = load_vec4(exchange_keys, quad)
            wave_experts = load_vec4(exchange_experts, quad)
            for position in cutlass.range_constexpr(EXPERTS_PER_LANE):
                keys[wave * EXPERTS_PER_LANE + position] = wave_keys[position]
                experts[wave * EXPERTS_PER_LANE + position] = wave_experts[position]
        selected_expert = given_expert
        if const_expr(self.has_fixed_routing):
            if fixed == Int32(0):
                selected_expert = self.topk_rounds(lane, keys, experts)
        else:
            selected_expert = self.topk_rounds(lane, keys, experts)
        # Values and physical experts are posted by expert: one read each.
        expert = cutlass.min(
            cutlass.max(selected_expert, Int32(0)), Int32(self.num_experts - 1)
        )
        selected_value = cutlass.select_(
            selected_expert == Int32(NO_EXPERT),
            Float32(0.0),
            exchange_values[base + expert],
        )
        selected_physical = Int32(0)
        if const_expr(self.has_physical_map):
            selected_physical = exchange_physical[base + expert]
        self.finish_token(
            token,
            lane,
            selected_expert,
            selected_value,
            selected_physical,
            shared_physical,
            physical_map,
            topk_idx,
            topk_weights,
            unmapped_idx,
            routed_scale,
            map_width,
        )

    @cute.jit
    def load_fixed_experts(
        self,
        token: Int32,
        lane: Int32,
        fix_mask: cute.Tensor,
        unmapped_idx: cute.Tensor,
    ):
        """Fixed routing (DeepGEMM's ``fix_routing_mask``): whether a token keeps the
        logical experts of its ``unmapped_idx`` row instead of the top-k (1 or 0), and the
        expert of this lane's slot (NO_EXPERT past the top-k), in the row's order with
        duplicates kept and ids clamped into [0, E) so no lookup leaves the tables. Read
        before the token's logits are needed, so neither load is on the critical path;
        the weights come from the token's unbiased scores as for a ranked token."""
        given = unmapped_idx[
            Int64(token), Int64(cutlass.min(lane, Int32(self.num_topk - 1)))
        ]
        # Clamped as int64: narrowing first would wrap ids of 2^31 and above.
        expert = Int32(
            cutlass.min(cutlass.max(given, Int64(0)), Int64(self.num_experts - 1))
        )
        return (
            Int32(fix_mask[token]),
            cutlass.select_(lane < Int32(self.num_topk), expert, Int32(NO_EXPERT)),
        )

    @cute.jit
    def cooperative_fixed(
        self,
        gate_warp: Int32,
        lane: Int32,
        logical_rank: Int32,
        block_token_base: Int32,
        tokens_in_block: Int32,
        fix_mask: cute.Tensor,
        unmapped_idx: cute.Tensor,
    ):
        """Before a block's score wait: the fixed-routing flag and given experts of the
        token this warp selects (``load_fixed_experts``), or (0, NO_EXPERT)."""
        fixed = Int32(0)
        given = Int32(NO_EXPERT)
        for slot in cutlass.range_constexpr(self.cooperative_slots):
            if gate_warp == Int32(self.slot_selector(slot)):
                t_local = logical_rank + Int32(slot * self.num_logical_ctas)
                if t_local < tokens_in_block:
                    fixed, given = self.load_fixed_experts(
                        block_token_base + t_local, lane, fix_mask, unmapped_idx
                    )
        return fixed, given

    @cute.jit
    def finish_token(
        self,
        token: Int32,
        lane: Int32,
        selected_expert: Int32,
        selected_value: Float32,
        selected_physical: Int32,
        shared_physical: Int32,
        physical_map: cute.Tensor,
        topk_idx: cute.Tensor,
        topk_weights: cute.Tensor,
        unmapped_idx: cute.Tensor,
        routed_scale: Float32,
        map_width: Int32,
    ):
        """Normalize the routed weights and store the token's slots (lane r: round r's
        expert, then the shared experts) and its unmapped experts."""
        # Keep the slot in bounds should no finite candidate remain.
        is_routed = lane < Int32(self.num_topk)
        no_candidate = selected_expert == Int32(NO_EXPERT)
        selected_expert = cutlass.min(
            cutlass.max(selected_expert, Int32(0)), Int32(self.num_experts - 1)
        )

        # Normalized unbiased weights of the routed slots.
        if const_expr(not self.has_scoring_bias):
            # A slot left without a finite candidate keeps weight 0, as with a bias.
            selected_value = cutlass.select_(
                no_candidate, Float32(0.0), sqrt_softplus(selected_value)
            )
        topk_sum = cutlass.select_(is_routed, selected_value, Float32(0.0))
        for step in cutlass.range_constexpr(self.sum_width.bit_length() - 1):
            topk_sum = topk_sum + cute.arch.shuffle_sync_bfly(
                topk_sum, offset=(self.sum_width // 2) >> step
            )
        topk_sum = topk_sum + Float32(TOPK_SUM_EPSILON)
        weight = cutlass.select_(
            is_routed, selected_value / topk_sum * routed_scale, Float32(1.0)
        )

        # Shared experts follow the routed slots; map logical to physical experts.
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
            physical_expert = selected_physical
            if const_expr(self.num_shared > 0):
                physical_expert = cutlass.select_(
                    is_routed, selected_physical, shared_physical
                )
            if cutlass.and_(is_routed, no_candidate):
                # No finite candidate was left for the slot: the clamped expert's
                # first replica (a count of 1, as for an unowned expert).
                physical_expert = physical_map[
                    Int64(selected_expert) * Int64(map_width)
                ]
        if is_slot:
            topk_idx[Int64(token), Int64(lane)] = Int64(physical_expert)
            topk_weights[Int64(token), Int64(lane)] = weight
        if const_expr(self.has_unmapped):
            if is_routed:
                unmapped_idx[Int64(token), Int64(lane)] = Int64(selected_expert)
