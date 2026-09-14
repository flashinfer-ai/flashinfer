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

"""``SmemPResource`` — P operand staging for BMM2.

Producer (Softmax): converts S in registers to P and publishes per-lane local
sums back through ``TmemSResource``. Keeps Q64/Q128 overlays P on consumed S
columns in TMEM; Swaps retains the SMEM operand layout. Consumer (MmaTask)
publishes the corresponding TMEM address or SMEM descriptor.
"""

from dataclasses import dataclass
from typing import ClassVar

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.experimental import primitives as prims

from cutlass.experimental.task_scheduling.memory import (
    ResourceContext,
    SmemAllocation,
    TmemAllocation,
)
from cutlass.experimental.task_scheduling.resources import (
    WorkAttr,
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from ..fmha_decode_config import FmhaDecodeConfig
from ...placeholder_helpers import _placeholder_smem_array
from .helpers_common import (
    Constexpr,
    DecodeGenResourceBase,
    ResourceVars,
    fadd2,
    ffma2,
    fmul2,
    _TASK_CACHE_LANE_IDX,
    _TASK_CACHE_WARP_GRP_THREAD_IDX,
    _TASK_CACHE_WARP_IDX,
    _decode_gen_task_cache,
    _fp8_log2_quant_scale,
    _is_last_loop_iteration,
    _keeps_col_base,
    _keeps_row_idx,
    _keeps_tcgen05_ld,
    _keeps_tcgen05_st,
    _named_barrier_arrive,
    _neg_max_f32,
    _pack_float2_to_bf16,
    _pack_float2_to_fp16,
    _wait_for_mbarrier_phase,
)
from .helpers_output import (
    _keeps_p_smem_block_offset_bytes,
    _p_stsm_smem_offset_bytes,
    _store_transposed_smem8b,
    _store_transposed_smem8b_x2,
    _store_transposed_smem8b_x4,
)
from .helpers_softmax import (
    _compute_fp8_p_regs_and_local_sums,
    _compute_fp8_p_regs_and_local_sums_dense,
    _compute_p_values_and_local_sums_dense,
    _ex2_emulation_packed_f32x2,
    _pack_float4_to_fp8_e4m3,
    _pack_float4_to_fp8_e4m3_inline,
)
from .sage_scales import sage_scale_arr_size
from .smem_block_sparse_metadata import (
    _SOFTMAX_ROUTE_IS_PROXY_FLAG,
    _proxy_log2_block_mass,
)
from .tmem_s import TmemSResource


@cute.jit
def _route_is_proxy(route_flags: Int32) -> Int32:
    """Return one when the Keeps route flags mark a proxy route."""
    return Int32((route_flags & Int32(_SOFTMAX_ROUTE_IS_PROXY_FLAG)) != Int32(0))


# Tunable: number of score pairs per streamed fragment whose exponentials run
# as FMA polynomials instead of MUFU. The MUFU issue rate bounds the fragment
# otherwise, while the FMA pipe is nearly idle in the softmax warps. Larger
# shares grow the fragment body and the softmax warps become instruction-fetch
# bound again, so one quarter of the 16 pairs is the measured optimum.
KV_TILE_256_EX2_EMULATED_PAIRS = 4


def _pair_uses_ex2_emulation(pair_idx: int, pairs_per_fragment: int) -> bool:
    """Spread the emulated pairs evenly across a fragment's score pairs."""
    count = KV_TILE_256_EX2_EMULATED_PAIRS
    pairs = pairs_per_fragment
    return ((pair_idx + 1) * count) // pairs != (pair_idx * count) // pairs


@dataclass(kw_only=True)
class SmemPResource(DecodeGenResourceBase):
    """P operand resource consumed by BMM2.

    Softmax producers convert S to P, store it in the profile's TMEM or SMEM
    layout, and publish local sums back to TmemS. Most profiles use the generic
    full/empty P pipeline. Streamed profiles instead publish four independently ready
    K32 TMEM fragments; BMM2 consumes those fragments in order, while the
    matching TmemO full barrier prevents the next QK from overwriting aliased P.
    """

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "p_desc_0_slot",
            prims.Tcgen05SmemDesc,
            prims.Tcgen05SmemDesc(0),
            "P descriptor for VP MMA call 0.",
        ),
        (
            "p_desc_1_slot",
            prims.Tcgen05SmemDesc,
            prims.Tcgen05SmemDesc(0),
            "P descriptor for VP MMA call 1.",
        ),
        (
            "p_tmem_addr_0_slot",
            Int32,
            Int32(0),
            "TMEM P address for VP MMA call 0.",
        ),
        (
            "p_tmem_addr_1_slot",
            Int32,
            Int32(0),
            "TMEM P address for VP MMA call 1.",
        ),
    )
    inst_id: Constexpr[int] = 0
    cfg: Constexpr[FmhaDecodeConfig] = None
    scale_softmax_log2: Float32 = None
    use_variable_seqlens_kv: Constexpr[bool] = False
    tmem_s_ref: Constexpr[TmemSResource] = None
    tmem_o_ref: Constexpr[object] = None
    _alloc: Constexpr[SmemAllocation | None] = None
    _fragment_ready_alloc: Constexpr[SmemAllocation | None] = None
    _tmem_alloc: Constexpr[TmemAllocation | None] = None
    _tmem_base_addr: Int32 = None
    _smem_base_p: cutlass.Array = None
    _smem_base_p_i32: cutlass.Array = None
    _fragment_ready: cutlass.Array = None
    p_desc_0_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    p_desc_1_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    p_tmem_addr_0_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    p_tmem_addr_1_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def _init_placeholder_state(self) -> None:
        """Create placeholder P storage state."""
        self._tmem_base_addr = Int32(0)
        self._smem_base_p = _placeholder_smem_array(
            self.cfg.q_dtype,
            self.cfg.smem_p_tile_bytes // self.cfg.q_dtype_bytes,
        )
        self._smem_base_p_i32 = _placeholder_smem_array(
            Int32, self.cfg.smem_p_tile_bytes // 4
        )
        self._fragment_ready = _placeholder_smem_array(
            Int64, self.cfg.num_softmax_score_fragments
        )

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate P storage or the streamed fragment-ready barriers."""
        if self.cfg.streams_tmem_p_fragments:
            if self._fragment_ready_alloc is None:
                self._fragment_ready_alloc = SmemAllocation(
                    name=f"{self.name}_fragmentReady",
                    size_bytes=self.cfg.num_softmax_score_fragments * 8,
                    alignment=16,
                )
            return [self._fragment_ready_alloc]
        if self.cfg.uses_tmem_p:
            return []
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=f"{self.name}",
                size_bytes=self.cfg.smem_p_tile_bytes,
                alignment=self.cfg.stensor_align,
            )
        return [self._alloc]

    @cute.jit
    def _bind_fragment_ready(self, context: ResourceContext | None = None) -> None:
        """Bind the one-way streamed P-ready barriers from the SMEM context."""
        if cutlass.const_expr(
            self.cfg.streams_tmem_p_fragments
            and context is not None
            and context.smem_base is not None
            and self._fragment_ready_alloc is not None
        ):
            self._fragment_ready = cutlass.Array(
                context.smem_base.data_ptr() + self._fragment_ready_alloc.offset,
                dtype=Int64,
                shape=(self.cfg.num_softmax_score_fragments,),
                addrspace=3,
            )

    @cute.jit
    def create_function_variables(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Bind and initialize the streamed per-fragment ready barriers."""
        self._bind_fragment_ready(context)
        if cutlass.const_expr(self.cfg.streams_tmem_p_fragments):
            tidx, _, _ = cute.arch.thread_idx()
            producer_warps = (
                self.cfg.softmax0_num_warps
                if self.inst_id == 0
                else self.cfg.softmax1_num_warps
            )
            if tidx == Int32(0):
                for fragment_idx in cutlass.range_constexpr(
                    self.cfg.num_softmax_score_fragments
                ):
                    prims.mbarrier_init(
                        self._fragment_ready.data_ptr() + fragment_idx,
                        producer_warps,
                    )
        return {}

    @cute.jit
    def initialize_runtime_state_internal(
        self,
        context: ResourceContext | None = None,
        captured_schedule: bool = False,
    ) -> None:
        """Initialize generic resource state and bind fragment barriers."""
        super().initialize_runtime_state_internal(context, captured_schedule)
        self._bind_fragment_ready(context)

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Allocate the TMEM columns occupied by the Keeps P operand."""
        if not self.cfg.uses_tmem_p:
            return []
        if self._tmem_alloc is None:
            self._tmem_alloc = TmemAllocation(
                name=f"{self.name}_tmem",
                num_columns=self.cfg.tmem_p_cols_per_inst,
            )
        return [self._tmem_alloc]

    @cute.jit
    def _create_initial_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Bind P storage and initialize operand task locals."""
        self._bind_fragment_ready(context)
        if cutlass.const_expr(
            not self.cfg.uses_tmem_p
            and context is not None
            and context.smem_base is not None
        ):
            # P is materialized in SMEM because BMM2 consumes it as a tcgen05
            # SMEM operand.
            smem_base_ptr = context.smem_base.data_ptr() + self._alloc.offset
            self._smem_base_p = cutlass.Array(
                smem_base_ptr,
                dtype=self.cfg.q_dtype,
                shape=(self.cfg.smem_p_tile_bytes // self.cfg.q_dtype_bytes,),
                addrspace=3,
            )
            self._smem_base_p_i32 = cutlass.Array(
                smem_base_ptr,
                dtype=Int32,
                shape=(self.cfg.smem_p_tile_bytes // 4,),
                addrspace=3,
            )
        if cutlass.const_expr(
            self.cfg.uses_tmem_p
            and context is not None
            and context.tmem_ptr_i32 is not None
        ):
            self._tmem_base_addr = context.tmem_ptr_i32.load()
        return {
            "p_desc_0": cutlass.Int64(0),
            "p_desc_1": cutlass.Int64(0),
            "p_tmem_addr_0": Int32(0),
            "p_tmem_addr_1": Int32(0),
        }

    @cute.jit
    def _create_work_tile_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Provide P operand slots for one work tile."""
        _ = context
        return {
            "p_desc_0": cutlass.Int64(0),
            "p_desc_1": cutlass.Int64(0),
            "p_tmem_addr_0": Int32(0),
            "p_tmem_addr_1": Int32(0),
        }

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_compute_state(self, stage_info: StageInfo) -> None:
        """Initialize producer-side P registers and local sums."""
        # ProdAuxWork: bind the SMEM P tile and reset producer-local P/sum
        # state before the softmax producer starts writing this work tile.
        self._create_initial_task_locals(stage_info.context)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptor_state(self, stage_info: StageInfo) -> None:
        """Initialize consumer-side P operand placeholders."""
        # ConsAuxWork: mirror the P storage binding on the BMM2 side so operand
        # work can publish a valid descriptor or TMEM address for this tile.
        self._create_initial_task_locals(stage_info.context)

    @cute.jit
    def _proxy_route_exponent_addend(
        self, minus_max_scale: Float32, route_is_proxy: Int32
    ) -> Float32:
        """Add a proxy route's log2 block mass to the exponent addend.

        The max pass has already raised the row maximum by the same mass and
        shifted the ragged final summary's score by its shortfall, so every
        summary probability carries its token mass with denominator weight
        one. Exact routes and dense tiles leave the addend unchanged.
        """

        if cutlass.const_expr(self.cfg.use_block_sparse_proxy_routes):
            if route_is_proxy != Int32(0):
                minus_max_scale += Float32(_proxy_log2_block_mass(self.cfg))
        return minus_max_scale

    @producer_work
    @cute.jit
    def compute_p_fragments(
        self,
        stage_info: StageInfo,
        *,
        new_max_arr: cutlass.Array,
    ) -> None:
        """Stream every ordinary K32 fragment from one rolled loop."""
        self._compute_p_fragments_impl(
            stage_info,
            new_max_arr=new_max_arr,
            route_is_proxy=Int32(0),
        )

    @producer_work
    @cute.jit
    def compute_sage_p_fragments(
        self,
        stage_info: StageInfo,
        *,
        new_max_arr: cutlass.Array,
        sage_scale_arr: cutlass.Array,
    ) -> None:
        """Stream every K32 fragment with per-group Sage dequantization scales."""
        assert self.cfg.use_sage_attention
        self._compute_p_fragments_impl(
            stage_info,
            new_max_arr=new_max_arr,
            route_is_proxy=Int32(0),
            sage_scale_arr=sage_scale_arr,
        )

    @producer_work
    @cute.jit
    def compute_proxy_route_p_fragments(
        self,
        stage_info: StageInfo,
        *,
        new_max_arr: cutlass.Array,
        route_flags: Int32,
    ) -> None:
        """Stream every exact or proxy K32 fragment from one rolled loop."""
        assert self.cfg.use_block_sparse_proxy_routes
        self._compute_p_fragments_impl(
            stage_info,
            new_max_arr=new_max_arr,
            route_is_proxy=_route_is_proxy(route_flags),
        )

    @producer_work
    @cute.jit
    def compute_sage_proxy_route_p_fragments(
        self,
        stage_info: StageInfo,
        *,
        new_max_arr: cutlass.Array,
        route_flags: Int32,
        sage_scale_arr: cutlass.Array,
    ) -> None:
        """Stream exact or proxy K32 fragments with per-group Sage scales."""
        assert self.cfg.use_block_sparse_proxy_routes and self.cfg.use_sage_attention
        self._compute_p_fragments_impl(
            stage_info,
            new_max_arr=new_max_arr,
            route_is_proxy=_route_is_proxy(route_flags),
            sage_scale_arr=sage_scale_arr,
        )

    @cute.jit
    def _compute_p_fragments_impl(
        self,
        stage_info: StageInfo,
        *,
        new_max_arr: cutlass.Array,
        route_is_proxy: Int32,
        sage_scale_arr: cutlass.Array | None = None,
    ) -> None:
        """Reload, exponentiate, and publish all K32 fragments in a rolled loop.

        The fragment index is a runtime loop variable, so the exponentiation
        body exists once in the instruction stream and only the TMEM column
        offset, the fragment barrier, and the proxy tail bookkeeping depend on
        it. Unrolling the fragments would replicate that body for every
        fragment and both softmax instances and leave the softmax warps
        instruction-fetch bound. The max pass has already written masked (and
        mass-shifted) scores back to TMEM, so the reload needs no mask or route
        logic of its own beyond the route-uniform proxy addend. Sage attention
        folds each scale group's ``sfQ * sfK`` into the exponent multiplier,
        precomputed for all fragments before the loop and rotated by one
        fragment per iteration; the addend already uses the dequantized
        maximum.
        """
        _ = stage_info
        cfg = self.cfg
        assert cfg.streams_tmem_p_fragments
        assert self._tmem_alloc.offset == self.tmem_s_ref._alloc.offset
        # One FP32 score per column; two 16-bit or four FP8 probabilities per
        # packed column.
        fragment_regs = cfg.softmax_score_fragment_regs
        fragment_cols = cfg.fragment_p_packed_cols

        new_max = new_max_arr[0]
        safe_new_max = new_max
        if safe_new_max == _neg_max_f32():
            safe_new_max = Float32(0.0)
        minus_max_scale = Float32(-self.scale_softmax_log2 * safe_new_max)
        if cutlass.const_expr(cfg.use_fp8_qkv):
            # FP8 P carries the static 448 quantization scale; the row sums
            # follow it and the epilogue divides it back out.
            minus_max_scale += _fp8_log2_quant_scale()
        minus_max_scale = self._proxy_route_exponent_addend(
            minus_max_scale, route_is_proxy
        )
        tmem_base = self._tmem_base_addr + Int32(self._tmem_alloc.offset)
        tidx, _, _ = cute.arch.thread_idx()
        publishes_fragment = (tidx & Int32(31)) == Int32(0)

        groups = cfg.sage_k_groups_per_fragment
        if cutlass.const_expr(cfg.use_sage_attention):
            # Exponent multipliers ``c * sfQ * sfK`` in fragment order. The
            # rolled loop consumes the leading ``groups`` entries and shifts
            # the rest down each iteration, so no runtime selection is needed.
            exponent_multipliers = cutlass.Array(
                Float32, sage_scale_arr_size(cfg), space=cutlass.AddressSpace.rmem
            )
            for entry in cutlass.range_constexpr(sage_scale_arr_size(cfg)):
                exponent_multipliers[entry] = self.scale_softmax_log2 * Float32(
                    sage_scale_arr[entry]
                )

        total_sum = Float32(0.0)
        for fragment_idx in cutlass.range(cfg.num_softmax_score_fragments, unroll=1):
            fragment = Int32(fragment_idx)
            loaded = _keeps_tcgen05_ld(
                cfg,
                prims.make_tmem_ptr(
                    tmem_base + fragment * Int32(fragment_regs), Float32
                ),
                num=fragment_regs,
                offset=cfg.tile_size_kv // 2,
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            s_arr = cutlass.Array(
                Float32, fragment_regs, space=cutlass.AddressSpace.rmem
            )
            for score_idx in cutlass.range_constexpr(fragment_regs):
                s_arr[score_idx] = loaded[score_idx]

            group_multipliers = cutlass.Array(
                Float32, groups, space=cutlass.AddressSpace.rmem
            )
            if cutlass.const_expr(cfg.use_sage_attention):
                for group_idx in cutlass.range_constexpr(groups):
                    group_multipliers[group_idx] = Float32(
                        exponent_multipliers[group_idx]
                    )
                for entry in cutlass.range_constexpr(sage_scale_arr_size(cfg) - groups):
                    exponent_multipliers[entry] = Float32(
                        exponent_multipliers[entry + groups]
                    )
            else:
                group_multipliers[0] = self.scale_softmax_log2
            local_sum = self._exponentiate_fragment_pairs(
                s_arr, minus_max_scale, group_multipliers
            )

            if cutlass.const_expr(cfg.use_fp8_qkv):
                packed_regs = cutlass.Array(
                    Int32, fragment_cols, space=cutlass.AddressSpace.rmem
                )
                for col_idx in cutlass.range_constexpr(fragment_cols):
                    value_base = col_idx * 4
                    packed_regs[col_idx] = _pack_float4_to_fp8_e4m3_inline(
                        Float32(s_arr[value_base]),
                        Float32(s_arr[value_base + 1]),
                        Float32(s_arr[value_base + 2]),
                        Float32(s_arr[value_base + 3]),
                    )
                packed_p = packed_regs.data_ptr().load(count=fragment_cols, alignment=4)
            else:
                packed_p = (
                    s_arr.data_ptr()
                    .load(count=fragment_regs, alignment=4)
                    .to(cfg.q_dtype)
                    .bitcast(Int32)
                )
            _keeps_tcgen05_st(
                cfg,
                prims.make_tmem_ptr(tmem_base + fragment * Int32(fragment_cols), Int32),
                packed_p,
                offset=cfg.tmem_p_cols_per_inst,
            )
            cute.arch.fence_view_async_tmem_store()
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            if publishes_fragment:
                prims.mbarrier_arrive(self._fragment_ready.data_ptr() + fragment)
            total_sum += local_sum
        self.tmem_s_ref.store_p_local_sum(0, total_sum)

    @cute.jit
    def _exponentiate_fragment_pairs(
        self,
        s_arr: cutlass.Array,
        minus_max_scale: Float32,
        group_multipliers: cutlass.Array,
    ) -> Float32:
        """Turn one fragment of scaled scores into probabilities in place.

        Returns the fragment's probability sum. Eight independent chains keep
        the denominator update off one long dependency chain, and a configurable
        subset of pairs runs its exponentials on the FMA pipe. Each score pair
        uses the exponent multiplier of its compile-time scale group; without
        Sage attention there is one group holding the softmax scale.
        """
        pairs_per_fragment = self.cfg.softmax_score_fragment_regs // 2
        pairs_per_group = pairs_per_fragment // self.cfg.sage_k_groups_per_fragment
        sum_chains = cutlass.Array(Float32, 8, space=cutlass.AddressSpace.rmem)
        for chain_idx in cutlass.range_constexpr(8):
            sum_chains[chain_idx] = Float32(0.0)
        for pair_idx in cutlass.range_constexpr(pairs_per_fragment):
            value_idx = pair_idx * 2
            multiplier = Float32(group_multipliers[pair_idx // pairs_per_group])
            p0, p1 = cute.arch.fma_packed_f32x2(
                (Float32(s_arr[value_idx]), Float32(s_arr[value_idx + 1])),
                (multiplier, multiplier),
                (minus_max_scale, minus_max_scale),
            )
            if cutlass.const_expr(
                _pair_uses_ex2_emulation(pair_idx, pairs_per_fragment)
            ):
                p0, p1 = _ex2_emulation_packed_f32x2(p0, p1)
            else:
                p0 = Float32(cute.math.exp2(p0, fastmath=True))
                p1 = Float32(cute.math.exp2(p1, fastmath=True))
            s_arr[value_idx] = p0
            s_arr[value_idx + 1] = p1
            chain_idx = (pair_idx & 3) * 2
            sum_chains[chain_idx], sum_chains[chain_idx + 1] = (
                cute.arch.add_packed_f32x2(
                    (sum_chains[chain_idx], sum_chains[chain_idx + 1]),
                    (p0, p1),
                )
            )
        sum01 = cute.arch.add_packed_f32x2(
            (sum_chains[0], sum_chains[1]),
            (sum_chains[2], sum_chains[3]),
        )
        sum23 = cute.arch.add_packed_f32x2(
            (sum_chains[4], sum_chains[5]),
            (sum_chains[6], sum_chains[7]),
        )
        total_pair = cute.arch.add_packed_f32x2(sum01, sum23)
        return Float32(total_pair[0] + total_pair[1])

    @cute.jit
    def _compute_keeps_p(
        self,
        stage_info: StageInfo,
        *,
        new_max_arr: cutlass.Array,
        s_arr: cutlass.Array,
    ) -> None:
        """Materialize one complete-row Keeps probability tile.

        TQ128 gives each warp-group thread a complete 128-column row. TQ64
        gives paired lanes the low/high 64-column halves of one row. Each lane
        writes disjoint packed blocks into the TMEM or SMEM layout consumed by
        BMM2. Streamed profiles, including every block-sparse Keeps profile,
        produce P through the rolled fragment loop instead.
        """
        cfg = self.cfg
        # Every block-sparse Keeps profile streams P; only dense complete rows
        # reach this path.
        assert not cfg.streams_tmem_p_fragments and not cfg.use_block_sparse
        task_cache = _decode_gen_task_cache(stage_info)
        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        num_s_regs = cfg.num_s_regs_per_thread
        vector_elements = cfg.keeps_p_smem_vector_elements
        num_vector_blocks = num_s_regs // vector_elements
        row_idx = _keeps_row_idx(cfg, warp_grp_thread_idx)
        col_base = _keeps_col_base(cfg, lane_idx, num_s_regs)
        p_tmem_stage_base = Int32(0)
        if cutlass.const_expr(cfg.uses_tmem_p):
            p_tmem_stage_base = (
                self._tmem_base_addr
                + Int32(self._tmem_alloc.offset)
                + stage_info.stage_idx * cfg.tmem_s_cols
            )

        new_max = new_max_arr[0]
        safe_new_max = new_max
        if safe_new_max == _neg_max_f32():
            safe_new_max = Float32(0.0)
        neg_scaled_max = -self.scale_softmax_log2 * safe_new_max
        if cutlass.const_expr(cfg.use_fp8_qkv):
            neg_scaled_max += _fp8_log2_quant_scale()

        # Preserve four independent modulo-4 sum chains as two packed pairs,
        # without keeping a second 16-value P array live beside the S row.
        local_sum_pair_01 = (Float32(0.0), Float32(0.0))
        local_sum_pair_23 = (Float32(0.0), Float32(0.0))
        # Each vector block is exactly 16 bytes after conversion. Compute and
        # pack adjacent pairs directly into their final register payload.
        packed_p_regs = cfg.num_packed_p_regs if cfg.uses_two_inst_tmem_p else 4
        packed_p = cutlass.Array(Int32, packed_p_regs, space=cutlass.AddressSpace.rmem)
        for block_idx in cutlass.range_constexpr(num_vector_blocks):
            s_base = block_idx * vector_elements
            packed_base = block_idx * 4 if cfg.uses_two_inst_tmem_p else 0
            if cutlass.const_expr(cfg.use_fp8_qkv):
                for packed_idx in cutlass.range_constexpr(4):
                    val_base = packed_idx * 4
                    scaled_pair_01 = ffma2(
                        (
                            s_arr[s_base + val_base],
                            s_arr[s_base + val_base + 1],
                        ),
                        (self.scale_softmax_log2, self.scale_softmax_log2),
                        (neg_scaled_max, neg_scaled_max),
                    )
                    p_pair_01 = (
                        cute.math.exp2(scaled_pair_01[0], fastmath=True),
                        cute.math.exp2(scaled_pair_01[1], fastmath=True),
                    )
                    local_sum_pair_01 = fadd2(local_sum_pair_01, p_pair_01)
                    scaled_pair_23 = ffma2(
                        (
                            s_arr[s_base + val_base + 2],
                            s_arr[s_base + val_base + 3],
                        ),
                        (self.scale_softmax_log2, self.scale_softmax_log2),
                        (neg_scaled_max, neg_scaled_max),
                    )
                    p_pair_23 = (
                        cute.math.exp2(scaled_pair_23[0], fastmath=True),
                        cute.math.exp2(scaled_pair_23[1], fastmath=True),
                    )
                    local_sum_pair_23 = fadd2(local_sum_pair_23, p_pair_23)
                    packed_p[packed_base + packed_idx] = (
                        _pack_float4_to_fp8_e4m3_inline(
                            p_pair_01[0],
                            p_pair_01[1],
                            p_pair_23[0],
                            p_pair_23[1],
                        )
                    )
            else:
                for packed_idx in cutlass.range_constexpr(4):
                    val_base = packed_idx * 2
                    scaled_pair = ffma2(
                        (
                            s_arr[s_base + val_base],
                            s_arr[s_base + val_base + 1],
                        ),
                        (self.scale_softmax_log2, self.scale_softmax_log2),
                        (neg_scaled_max, neg_scaled_max),
                    )
                    p_pair = (
                        cute.math.exp2(scaled_pair[0], fastmath=True),
                        cute.math.exp2(scaled_pair[1], fastmath=True),
                    )
                    if cutlass.const_expr(packed_idx % 2 == 0):
                        local_sum_pair_01 = fadd2(local_sum_pair_01, p_pair)
                    else:
                        local_sum_pair_23 = fadd2(local_sum_pair_23, p_pair)
                    if cutlass.const_expr(cfg.use_bf16_qkv):
                        packed_p[packed_base + packed_idx] = _pack_float2_to_bf16(
                            p_pair[0], p_pair[1]
                        )
                    else:
                        packed_p[packed_base + packed_idx] = _pack_float2_to_fp16(
                            p_pair[0], p_pair[1]
                        )

            if cutlass.const_expr(cfg.uses_two_inst_tmem_p):
                # Retain the complete packed row and publish it once below.
                pass
            elif cutlass.const_expr(cfg.uses_tmem_p):
                # Each register packs two 16-bit P values. The q64 TMEM store shape
                # maps paired half-warps onto the low/high 32-column halves of
                # the 64-column UInt32 P tile.
                p_tmem_addr = p_tmem_stage_base + Int32(block_idx * 4)
                _keeps_tcgen05_st(
                    cfg,
                    prims.make_tmem_ptr(p_tmem_addr, Int32),
                    packed_p.data_ptr().load(count=4, alignment=4),
                    offset=cfg.num_packed_p_regs,
                )
            else:
                logical_col = col_base + Int32(s_base)
                smem_offset_bytes = _keeps_p_smem_block_offset_bytes(
                    cfg, row_idx, logical_col
                )
                smem_dst = self._smem_base_p_i32.subview(
                    smem_offset_bytes >> Int32(2)
                ).data_ptr()
                smem_dst.store(
                    packed_p.data_ptr().load(count=4, alignment=4), alignment=16
                )
        if cutlass.const_expr(cfg.uses_two_inst_tmem_p):
            # FP8 publishes the complete row with one x32 STTM. Dense 16-bit
            # Q128/KV128 uses x16 slices to limit Softmax register pressure.
            # Block-sparse two-instance profiles stream K32 fragments instead.
            assert cfg.num_packed_p_regs in (32, 64)
            regs_per_store = cfg.num_packed_p_regs if cfg.use_fp8_qkv else 16
            assert cfg.num_packed_p_regs % regs_per_store == 0
            for store_idx in cutlass.range_constexpr(
                cfg.num_packed_p_regs // regs_per_store
            ):
                packed_offset = store_idx * regs_per_store
                _keeps_tcgen05_st(
                    cfg,
                    prims.make_tmem_ptr(
                        p_tmem_stage_base + Int32(packed_offset), Int32
                    ),
                    (packed_p.data_ptr() + packed_offset).load(
                        count=regs_per_store, alignment=4
                    ),
                    # Separate the paired Softmax destinations by one packed
                    # row (the half-row split for x16/x32 TMEM layouts).
                    offset=cfg.num_packed_p_regs,
                )
            if cutlass.const_expr(cfg.ordered_softmax_early_release):
                # Hand the baton over as soon as this group's TMEM store has
                # issued: the partner's exp2/pack/TMEM store touch only its own
                # registers and TMEM region, so it need not wait for this
                # store to drain, the async fence, or the pipeline commit.
                # The exp2 phases stay serialized (shared MUFU), but the
                # store-drain + commit tail overlaps the partner's wakeup.
                _named_barrier_arrive(
                    cfg.resolved_softmax_order_barrier_threads,
                    barrier_id=cfg.softmax_order_barrier_id + 1 - self.inst_id,
                )
        local_sum_pair0 = fadd2(local_sum_pair_01, local_sum_pair_23)
        local_sum = local_sum_pair0[0] + local_sum_pair0[1]
        self.tmem_s_ref.store_p_local_sum(0, local_sum)

        # Publish the selected memory view before the task-level P pipeline
        # exposes this stage to BMM2.
        if cutlass.const_expr(cfg.uses_tmem_p):
            cute.arch.fence_view_async_tmem_store()
            if cutlass.const_expr(cfg.uses_staged_one_inst_tmem_p):
                # Synchronize the D256 producer warp group after its TMEM stores
                # are visible.
                prims.barrier_cta_sync(4 + self.inst_id, thread_count=128)
        else:
            # Each producer thread orders its own SMEM P stores with an
            # async-proxy fence before its own AsyncUmma producer-commit
            # mbarrier arrive. The full barrier counts all 128 softmax
            # threads, so the commit itself is the warp-group visibility
            # point and no extra named barrier is needed here.
            cute.arch.fence_view_async_shared()

    @cute.jit
    def _compute_p_impl(
        self,
        stage_info: StageInfo,
        *,
        new_max_arr: cutlass.Array,
        s_arr: cutlass.Array,
        route_is_proxy: Int32,
    ) -> None:
        """Compute P from S, stage its BMM2 operand, and publish local sums."""
        cfg = self.cfg
        if cutlass.const_expr(cfg.use_keeps_mma_ab):
            self._compute_keeps_p(
                stage_info,
                new_max_arr=new_max_arr,
                s_arr=s_arr,
            )
            return
        # ProdWork: transform the softmax S registers into the P operand layout
        # expected by BMM2, while recording the per-scale local sums consumed by
        # the denominator update.
        # Decode the scheduler cache once so every store path uses the same
        # warp/lane ownership for SMEM offsets and STSM swizzles.
        task_cache = _decode_gen_task_cache(stage_info)
        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        if cutlass.const_expr(cfg.tile_size_q == 32 and cfg.use_fp8_qkv):
            # Tile-Q=32 FP8 fast path: compute E4M3 P registers in the
            # same order consumed by the STSM helper, while also capturing
            # one local denominator sum per softmax scale group.
            packed_p = cutlass.Array(
                Int32, cfg.num_packed_p_regs, space=cutlass.AddressSpace.rmem
            )
            local_sum = cutlass.Array(
                Float32,
                cfg.num_softmax_scale_groups,
                space=cutlass.AddressSpace.rmem,
            )
            if cutlass.const_expr(
                not self.use_variable_seqlens_kv
                and cfg.total_kv_tiles > 0
                and (cfg.total_kv_tiles % cfg.num_insts_kv) == 0
                and cfg.q_tiles_are_full
            ):
                # Dense full tiles have no tail/window masking. Use the
                # straight-line helper to keep this path compact.
                for scale_pair_idx in cutlass.range_constexpr(4):
                    scale_base = scale_pair_idx * 2
                    q32_s_base = scale_pair_idx * 4
                    p_lo, p_hi, sum_lo, sum_hi = (
                        _compute_fp8_p_regs_and_local_sums_dense(
                            self.scale_softmax_log2,
                            new_max_arr[scale_base],
                            new_max_arr[scale_base + 1],
                            s_arr[q32_s_base],
                            s_arr[q32_s_base + 1],
                            s_arr[q32_s_base + 2],
                            s_arr[q32_s_base + 3],
                            s_arr[q32_s_base + 16],
                            s_arr[q32_s_base + 17],
                            s_arr[q32_s_base + 18],
                            s_arr[q32_s_base + 19],
                        )
                    )
                    packed_p[scale_pair_idx] = p_lo
                    packed_p[scale_pair_idx + 4] = p_hi
                    local_sum[scale_base] = sum_lo
                    local_sum[scale_base + 1] = sum_hi
            else:
                # General path handles masked S values from variable
                # seqlens, odd tail waves, sliding window, or split-KV.
                for scale_pair_idx in cutlass.range_constexpr(4):
                    scale_base = scale_pair_idx * 2
                    q32_s_base = scale_pair_idx * 4
                    p_lo, p_hi, sum_lo, sum_hi = _compute_fp8_p_regs_and_local_sums(
                        self.scale_softmax_log2,
                        new_max_arr[scale_base],
                        new_max_arr[scale_base + 1],
                        s_arr[q32_s_base],
                        s_arr[q32_s_base + 1],
                        s_arr[q32_s_base + 2],
                        s_arr[q32_s_base + 3],
                        s_arr[q32_s_base + 16],
                        s_arr[q32_s_base + 17],
                        s_arr[q32_s_base + 18],
                        s_arr[q32_s_base + 19],
                    )
                    packed_p[scale_pair_idx] = p_lo
                    packed_p[scale_pair_idx + 4] = p_hi
                    local_sum[scale_base] = sum_lo
                    local_sum[scale_base + 1] = sum_hi
            # Publish the local denominator contribution before committing P so
            # TmemS can update the online-softmax sum after P materialization.
            for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                self.tmem_s_ref.store_p_local_sum(scale_idx, local_sum[scale_idx])
            # Store the low and high K halves separately. The x4 helper writes
            # transposed E4M3 bytes into the SMEM layout expected by BMM2.
            _store_transposed_smem8b_x4(
                self._smem_base_p_i32,
                packed_p[0],
                packed_p[1],
                packed_p[2],
                packed_p[3],
                warp_grp_thread_idx,
                cfg.tile_size_q,
                cfg.tile_size_kv,
            )
            _store_transposed_smem8b_x4(
                self._smem_base_p_i32,
                packed_p[4],
                packed_p[5],
                packed_p[6],
                packed_p[7],
                warp_grp_thread_idx,
                cfg.tile_size_q,
                cfg.tile_size_kv,
                1,
            )
            # The P producer writes SMEM directly with STSM helpers. Fence
            # and synchronize the warpgroup before the TS pipeline commits
            # the stage to the BMM2 consumer.
            cute.arch.fence_view_async_shared()
            prims.barrier_cta_sync(4 + self.inst_id, thread_count=128)
            return

        if cutlass.const_expr(cfg.tile_size_q == 16 and cfg.use_fp8_qkv):
            # Tile-Q=16 FP8 fast path: each helper call handles two softmax
            # scale groups and returns the low/high K halves already packed for
            # STSM. This avoids keeping all 16 FP32 P values live and packing
            # them in a separate pass.
            packed_p = cutlass.Array(
                Int32, cfg.num_packed_p_regs, space=cutlass.AddressSpace.rmem
            )
            local_sum = cutlass.Array(
                Float32,
                cfg.num_softmax_scale_groups,
                space=cutlass.AddressSpace.rmem,
            )
            for scale_pair_idx in cutlass.range_constexpr(2):
                scale_base = scale_pair_idx * 2
                s_base = scale_pair_idx * 4
                if cutlass.const_expr(
                    not self.use_variable_seqlens_kv
                    and cfg.total_kv_tiles > 0
                    and (cfg.total_kv_tiles % cfg.num_insts_kv) == 0
                    and cfg.q_tiles_are_full
                ):
                    p_lo, p_hi, sum_lo, sum_hi = (
                        _compute_fp8_p_regs_and_local_sums_dense(
                            self.scale_softmax_log2,
                            new_max_arr[scale_base],
                            new_max_arr[scale_base + 1],
                            s_arr[s_base],
                            s_arr[s_base + 1],
                            s_arr[s_base + 2],
                            s_arr[s_base + 3],
                            s_arr[s_base + 8],
                            s_arr[s_base + 9],
                            s_arr[s_base + 10],
                            s_arr[s_base + 11],
                        )
                    )
                else:
                    p_lo, p_hi, sum_lo, sum_hi = _compute_fp8_p_regs_and_local_sums(
                        self.scale_softmax_log2,
                        new_max_arr[scale_base],
                        new_max_arr[scale_base + 1],
                        s_arr[s_base],
                        s_arr[s_base + 1],
                        s_arr[s_base + 2],
                        s_arr[s_base + 3],
                        s_arr[s_base + 8],
                        s_arr[s_base + 9],
                        s_arr[s_base + 10],
                        s_arr[s_base + 11],
                    )
                packed_p[scale_pair_idx] = p_lo
                packed_p[scale_pair_idx + 2] = p_hi
                local_sum[scale_base] = sum_lo
                local_sum[scale_base + 1] = sum_hi

            for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                self.tmem_s_ref.store_p_local_sum(scale_idx, local_sum[scale_idx])
            _store_transposed_smem8b(
                self._smem_base_p_i32,
                packed_p.data_ptr().load(count=cfg.num_packed_p_regs, alignment=4),
                warp_grp_thread_idx,
                cfg.tile_size_q,
                cfg.tile_size_kv,
                cfg.num_packed_p_regs,
            )
            cute.arch.fence_view_async_shared()
            prims.barrier_cta_sync(4 + self.inst_id, thread_count=128)
            return

        if cutlass.const_expr(cfg.tile_size_q in (16, 32)):
            # Generic tile-Q 16/32 path: compute P scalars for each
            # softmax scale group, accumulate local sums, then pack/store
            # to the BMM2 SMEM layout.
            q_repeats = max(cfg.tile_size_q // 8, 1)
            num_s_regs = cfg.num_s_regs_per_thread
            num_scale_groups = cfg.num_softmax_scale_groups
            p_vals = cutlass.Array(Float32, num_s_regs, space=cutlass.AddressSpace.rmem)
            local_sums = cutlass.Array(
                Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
            )
            for idx in cutlass.range_constexpr(num_s_regs):
                p_vals[idx] = Float32(0.0)
            for idx in cutlass.range_constexpr(num_scale_groups):
                local_sums[idx] = Float32(0.0)

            for scale_idx in cutlass.range_constexpr(num_scale_groups):
                # Convert each softmax scale group from S to P. Masked rows have
                # new_max == -inf and keep their initialized zero P/local_sum.
                new_max = new_max_arr[scale_idx]
                safe_new_max = new_max
                if safe_new_max == _neg_max_f32():
                    safe_new_max = Float32(0.0)
                neg_scaled_max = -self.scale_softmax_log2 * safe_new_max
                if cutlass.const_expr(cfg.use_fp8_qkv):
                    neg_scaled_max += _fp8_log2_quant_scale()
                neg_scaled_max = self._proxy_route_exponent_addend(
                    neg_scaled_max, route_is_proxy
                )
                if new_max != _neg_max_f32():
                    repeat_idx = scale_idx // 2
                    pair_idx = scale_idx % 2
                    generic_s_base = repeat_idx * 4 + pair_idx
                    for k_pair_idx in cutlass.range_constexpr(4):
                        if cutlass.const_expr(k_pair_idx < 2):
                            s_idx = generic_s_base + k_pair_idx * 2
                        else:
                            s_idx = (
                                generic_s_base + q_repeats * 4 + (k_pair_idx - 2) * 2
                            )
                        p_val = cute.math.exp2(
                            s_arr[s_idx] * self.scale_softmax_log2 + neg_scaled_max,
                            fastmath=True,
                        )
                        p_vals[s_idx] = p_val
                        local_sums[scale_idx] += p_val
            # Hand off denominator contributions through TmemS. P remains a pure
            # MMA operand in SMEM; sums are not reloaded from the P tile.
            for scale_idx in cutlass.range_constexpr(num_scale_groups):
                self.tmem_s_ref.store_p_local_sum(scale_idx, local_sums[scale_idx])

            if cutlass.const_expr(cfg.use_fp8_qkv):
                # FP8 P is packed four values per register and stored with
                # transposed 8-bit helpers so BMM2 sees the tcgen05 layout.
                packed_p = cutlass.Array(
                    Int32, cfg.num_packed_p_regs, space=cutlass.AddressSpace.rmem
                )
                for packed_idx in cutlass.range_constexpr(cfg.num_packed_p_regs):
                    val_base = packed_idx * 4
                    packed_p[packed_idx] = _pack_float4_to_fp8_e4m3(
                        p_vals[val_base],
                        p_vals[val_base + 1],
                        p_vals[val_base + 2],
                        p_vals[val_base + 3],
                    )
                _store_transposed_smem8b(
                    self._smem_base_p_i32,
                    packed_p.data_ptr().load(count=cfg.num_packed_p_regs, alignment=4),
                    warp_grp_thread_idx,
                    cfg.tile_size_q,
                    cfg.tile_size_kv,
                    cfg.num_packed_p_regs,
                )
                # Inline byte stores need an explicit CTA barrier before the
                # pipeline stage can be observed by the BMM2 consumer.
                cute.arch.fence_view_async_shared()
                prims.barrier_cta_sync(4 + self.inst_id, thread_count=128)
                return

            # FP16/BF16 P uses stmatrix stores. Each stmatrix group writes
            # one 8x8 fragment to the swizzled SMEM tile consumed by UMMA.
            regs_p = cutlass.Array(
                Int32,
                cfg.num_packed_p_regs,
                space=cutlass.AddressSpace.rmem,
            )
            for pair_idx in cutlass.range_constexpr(cfg.num_packed_p_regs):
                val_base = pair_idx * 2
                if cutlass.const_expr(cfg.use_bf16_qkv):
                    regs_p[pair_idx] = _pack_float2_to_bf16(
                        p_vals[val_base], p_vals[val_base + 1]
                    )
                else:
                    regs_p[pair_idx] = _pack_float2_to_fp16(
                        p_vals[val_base], p_vals[val_base + 1]
                    )
            warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
            lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
            for stsm_group_idx in cutlass.range_constexpr(cfg.num_packed_p_regs // 4):
                smem_offset_bytes = _p_stsm_smem_offset_bytes(
                    warp_idx, lane_idx, stsm_group_idx, cfg.tile_size_q
                )
                smem_dst = (
                    self._smem_base_p_i32.subview((smem_offset_bytes >> 2))
                ).data_ptr()
                prims.stmatrix(
                    smem_dst,
                    (regs_p.data_ptr() + stsm_group_idx * 4).load(count=4, alignment=4),
                    prims.MMALayout.COL,
                    shape=prims.StoreShape.M8N8,
                )
            cute.arch.fence_view_async_shared()
            return

        if cutlass.const_expr(cfg.use_fp8_qkv):
            # Tile-Q=8 FP8 path: compute packed P and local sums directly
            # from the eight S registers owned by this lane.
            packed_p = cutlass.Array(
                Int32, cfg.num_packed_p_regs, space=cutlass.AddressSpace.rmem
            )
            local_sum = cutlass.Array(
                Float32,
                cfg.num_softmax_scale_groups,
                space=cutlass.AddressSpace.rmem,
            )
            if cutlass.const_expr(
                not self.use_variable_seqlens_kv
                and not cfg.uses_runtime_q_kv_union
                and not cfg.use_split_kv
                and cfg.has_odd_kv_tail
                and self.inst_id == 1
            ):
                # In the static nonsplit profile the final inst1 wave is only
                # structural padding. Publish an exact zero contribution so
                # the paired instance cannot perturb final normalization.
                if _is_last_loop_iteration(stage_info):
                    packed_p[0] = Int32(0)
                    packed_p[1] = Int32(0)
                    local_sum[0] = Float32(0.0)
                    local_sum[1] = Float32(0.0)
                else:
                    packed_p[0], packed_p[1], local_sum[0], local_sum[1] = (
                        _compute_fp8_p_regs_and_local_sums(
                            self.scale_softmax_log2,
                            new_max_arr[0],
                            new_max_arr[1],
                            s_arr[0],
                            s_arr[1],
                            s_arr[2],
                            s_arr[3],
                            s_arr[4],
                            s_arr[5],
                            s_arr[6],
                            s_arr[7],
                        )
                    )
            elif cutlass.const_expr(
                not self.use_variable_seqlens_kv
                and cfg.total_kv_tiles > 0
                and (cfg.total_kv_tiles % cfg.num_insts_kv) == 0
            ):
                # Dense full-tile FP8 can produce packed P and sums in one
                # straight-line helper because no S entry is masked.
                packed_p[0], packed_p[1], local_sum[0], local_sum[1] = (
                    _compute_fp8_p_regs_and_local_sums_dense(
                        self.scale_softmax_log2,
                        new_max_arr[0],
                        new_max_arr[1],
                        s_arr[0],
                        s_arr[1],
                        s_arr[2],
                        s_arr[3],
                        s_arr[4],
                        s_arr[5],
                        s_arr[6],
                        s_arr[7],
                    )
                )
            else:
                # General FP8 path preserves masks from softmax by letting the
                # helper suppress entries whose S value is -inf.
                packed_p[0], packed_p[1], local_sum[0], local_sum[1] = (
                    _compute_fp8_p_regs_and_local_sums(
                        self.scale_softmax_log2,
                        new_max_arr[0],
                        new_max_arr[1],
                        s_arr[0],
                        s_arr[1],
                        s_arr[2],
                        s_arr[3],
                        s_arr[4],
                        s_arr[5],
                        s_arr[6],
                        s_arr[7],
                    )
                )
            # Publish sums through TmemS; packed E4M3 P bytes use the
            # transposed SMEM tile consumed by BMM2.
            for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                self.tmem_s_ref.store_p_local_sum(scale_idx, local_sum[scale_idx])
            _store_transposed_smem8b_x2(
                self._smem_base_p_i32,
                packed_p[0],
                packed_p[1],
                warp_grp_thread_idx,
                cfg.tile_size_q,
                cfg.tile_size_kv,
            )
        else:
            # Tile-Q=8 16-bit path: compute P scalars, accumulate local
            # sums, pack to 16-bit, and store a matrix tile into SMEM.
            local_sum = cutlass.Array(
                Float32,
                cfg.num_softmax_scale_groups,
                space=cutlass.AddressSpace.rmem,
            )
            p_vals = cutlass.Array(Float32, 8, space=cutlass.AddressSpace.rmem)
            for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                local_sum[scale_idx] = Float32(0.0)
            for p_idx in cutlass.range_constexpr(8):
                p_vals[p_idx] = Float32(0.0)
            # This path exponentiates ``c * (s - max)`` directly, so a proxy
            # route's log2 block mass enters as a lower exponent anchor
            # (``max - log2(mass) / c``); the sentinel of a fully masked group
            # is unchanged in fp32.
            exponent_anchor = new_max_arr
            if cutlass.const_expr(cfg.use_block_sparse_proxy_routes):
                exponent_anchor = cutlass.Array(
                    Float32,
                    cfg.num_softmax_scale_groups,
                    space=cutlass.AddressSpace.rmem,
                )
                for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                    exponent_anchor[scale_idx] = new_max_arr[scale_idx]
                if route_is_proxy != Int32(0):
                    anchor_shift = Float32(_proxy_log2_block_mass(cfg)) * cute.math.rcp(
                        self.scale_softmax_log2
                    )
                    for scale_idx in cutlass.range_constexpr(
                        cfg.num_softmax_scale_groups
                    ):
                        exponent_anchor[scale_idx] = (
                            exponent_anchor[scale_idx] - anchor_shift
                        )
            if cutlass.const_expr(
                not self.use_variable_seqlens_kv
                and not cfg.use_split_kv
                and cfg.total_kv_tiles > 0
                and (cfg.total_kv_tiles % cfg.num_insts_kv) == 0
            ):
                # The straight-line helper may synthesize P for an entirely
                # masked sparse instance whose maximum stayed at -inf. This is
                # intentionally safe: reduce_sums' guarded rescale and the
                # correction path's uses_instN gate both key on that sentinel
                # and discard the instance before its P/O contribution is visible.
                p_result = _compute_p_values_and_local_sums_dense(
                    self.scale_softmax_log2,
                    exponent_anchor[0],
                    exponent_anchor[1],
                    s_arr[0],
                    s_arr[1],
                    s_arr[2],
                    s_arr[3],
                    s_arr[4],
                    s_arr[5],
                    s_arr[6],
                    s_arr[7],
                )
                for p_idx in cutlass.range_constexpr(8):
                    p_vals[p_idx] = p_result[p_idx]
                for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                    local_sum[scale_idx] = p_result[8 + scale_idx]
            else:
                # General tile-Q=8 path preserves masked S entries as zero P
                # contribution by skipping groups whose new max stayed -inf.
                for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                    new_max = exponent_anchor[scale_idx]
                    if new_max != _neg_max_f32():
                        for pair_idx in cutlass.range_constexpr(2):
                            p_base = scale_idx + pair_idx * 2
                            scaled_pair = fmul2(
                                (
                                    self.scale_softmax_log2,
                                    self.scale_softmax_log2,
                                ),
                                fadd2(
                                    (s_arr[p_base], s_arr[p_base + 4]),
                                    (-new_max, -new_max),
                                ),
                            )
                            p_pair = (
                                cute.math.exp2(scaled_pair[0], fastmath=True),
                                cute.math.exp2(scaled_pair[1], fastmath=True),
                            )
                            p_vals[p_base] = p_pair[0]
                            p_vals[p_base + 4] = p_pair[1]
                            local_sum[scale_idx] += p_pair[0]
                            local_sum[scale_idx] += p_pair[1]
            # Pack the P scalars to match the dtype consumed by BMM2.
            regs_p = cutlass.Array(
                Int32, cfg.num_packed_p_regs, space=cutlass.AddressSpace.rmem
            )
            if cutlass.const_expr(cfg.use_bf16_qkv):
                for reg_idx in cutlass.range_constexpr(cfg.num_packed_p_regs):
                    val_base = reg_idx * 2
                    regs_p[reg_idx] = _pack_float2_to_bf16(
                        p_vals[val_base], p_vals[val_base + 1]
                    )
            else:
                for reg_idx in cutlass.range_constexpr(cfg.num_packed_p_regs):
                    val_base = reg_idx * 2
                    regs_p[reg_idx] = _pack_float2_to_fp16(
                        p_vals[val_base], p_vals[val_base + 1]
                    )
            # Publish the denominator contribution after P has been computed,
            # before the SMEM fence exposes the P tile to the downstream MMA.
            for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                self.tmem_s_ref.store_p_local_sum(scale_idx, local_sum[scale_idx])
            warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
            lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
            # Compute the stmatrix destination matching the P descriptor
            # swizzle and store the register fragment.
            slice_idx = warp_idx // Int32(2)
            warp_idx_in_slice = warp_idx % Int32(2)
            mtx_idx = lane_idx // Int32(8)
            thr_row_idx = lane_idx % Int32(8)
            mtx_col_idx = warp_idx_in_slice * Int32(4) + (mtx_idx % Int32(4))
            smem_offset_bytes = (
                slice_idx * Int32(8 * 128)
                + thr_row_idx * Int32(128)
                + ((mtx_col_idx ^ thr_row_idx) * Int32(16))
            )
            smem_dst = (
                self._smem_base_p_i32.subview((smem_offset_bytes >> 2))
            ).data_ptr()
            prims.stmatrix(
                smem_dst,
                regs_p.data_ptr().load(count=4, alignment=4),
                prims.MMALayout.COL,
                shape=prims.StoreShape.M8N8,
            )
        cute.arch.fence_view_async_shared()
        if cutlass.const_expr(cfg.use_fp8_qkv):
            # FP8 P uses inline STSM stores. Synchronize the producer
            # warpgroup before the UMMA-consumer pipeline is committed so
            # BMM2 cannot observe a partially written P tile.
            prims.barrier_cta_sync(4 + self.inst_id, thread_count=128)

    @producer_work
    @cute.jit
    def compute_p(
        self,
        stage_info: StageInfo,
        *,
        new_max_arr: cutlass.Array,
        s_arr: cutlass.Array,
    ) -> None:
        """Compute an exact/dense P tile without typed-route metadata."""

        self._compute_p_impl(
            stage_info,
            new_max_arr=new_max_arr,
            s_arr=s_arr,
            route_is_proxy=Int32(0),
        )

    @producer_work
    @cute.jit
    def compute_proxy_route_p(
        self,
        stage_info: StageInfo,
        *,
        new_max_arr: cutlass.Array,
        s_arr: cutlass.Array,
        keeps_route_flags: Int32,
        swaps_route_flags: Uint32,
    ) -> None:
        """Read the route kind from the active Keeps/SWAP metadata view and compute P.

        Keeps carries the kind in its Int32 route flags; SWAP keeps it in the
        bit-preserving Uint32 flags word of its seven-slot ABI. The max pass
        has already folded the mass of proxy summaries into the scores and the
        maximum, so only the route-uniform exponent addend remains here.
        """

        assert self.cfg.use_block_sparse_proxy_routes
        if cutlass.const_expr(self.cfg.use_keeps_mma_ab):
            route_is_proxy = _route_is_proxy(keeps_route_flags)
        else:
            route_is_proxy = Int32(
                (swaps_route_flags & Uint32(_SOFTMAX_ROUTE_IS_PROXY_FLAG)) != Uint32(0)
            )
        self._compute_p_impl(
            stage_info,
            new_max_arr=new_max_arr,
            s_arr=s_arr,
            route_is_proxy=route_is_proxy,
        )

    @consumer_work(
        returns=(
            p_desc_0_slot,
            p_desc_1_slot,
            p_tmem_addr_0_slot,
            p_tmem_addr_1_slot,
        )
    )
    @cute.jit
    def p_operands(
        self, stage_info: StageInfo
    ) -> tuple[
        prims.Tcgen05SmemDesc,
        prims.Tcgen05SmemDesc,
        Int32,
        Int32,
    ]:
        """Publish the stage-specific P operand consumed by BMM2."""
        cfg = self.cfg
        p_desc_0 = prims.Tcgen05SmemDesc(0)
        p_desc_1 = prims.Tcgen05SmemDesc(0)
        p_tmem_addr_0 = Int32(0)
        p_tmem_addr_1 = Int32(0)
        if cutlass.const_expr(cfg.uses_tmem_p):
            # ConsWork: select the physical TMEM stage paired with the P
            # pipeline token that was just waited. The allocation aliases the
            # stats-free columns of the corresponding S stage.
            p_stage_cols = cfg.tmem_s_cols
            if cutlass.const_expr(cfg.streams_tmem_p_fragments):
                # A streamed profile's four pipeline stages are K32 fragments of one P
                # operand, not four independent full S/P stages.
                p_stage_cols = cfg.fragment_p_packed_cols
            p_tmem_addr = self._tmem_base_addr + Int32(
                self._tmem_alloc.offset + stage_info.stage_idx * p_stage_cols
            )
            if cutlass.const_expr(self.inst_id == 0):
                p_tmem_addr_0 = p_tmem_addr
            else:
                p_tmem_addr_1 = p_tmem_addr
        else:
            # ConsWork: build the SMEM descriptor for P. Only the descriptor
            # slot corresponding to this resource instance is populated; the
            # MmaTask receives both slots and selects the active one for BMM2.
            p_desc = prims.Tcgen05SmemDesc.build(
                self._smem_base_p,
                leading_byte_offset=Int32(cfg.tile_size_q * 128),
                stride_byte_offset=1024,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            )
            if cutlass.const_expr(self.inst_id == 0):
                p_desc_0 = p_desc
            else:
                p_desc_1 = p_desc
        return p_desc_0, p_desc_1, p_tmem_addr_0, p_tmem_addr_1

    @consumer_work(returns=p_tmem_addr_0_slot)
    @cute.jit
    def wait_p_fragment(
        self,
        stage_info: StageInfo,
        *,
        fragment_idx: Constexpr[int],
    ) -> Int32:
        """Wait for and return the next streamed P-fragment TMEM address."""
        cfg = self.cfg
        _ = stage_info
        assert cfg.streams_tmem_p_fragments
        _wait_for_mbarrier_phase(
            self._fragment_ready.data_ptr() + Int32(fragment_idx),
            self.tmem_s_ref.producer_state.phase,
        )
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)

        p_tmem_addr = self._tmem_base_addr + Int32(
            self._tmem_alloc.offset + fragment_idx * cfg.fragment_p_packed_cols
        )
        return p_tmem_addr
