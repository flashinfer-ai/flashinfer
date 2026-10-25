# Copyright (c) 2025 by FlashInfer team.
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
#
# Adapted from TensorRT-LLM (Apache 2.0):
# tensorrt_llm/_torch/cute_dsl_kernels/blackwell/paged_mqa_logits/fp8_paged_mqa_logits.py
# Original copyright: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
CuTe DSL FP8 paged MQA logits kernel for Blackwell (SM100/SM103) and Rubin (SM107).

Architecture:
  - 384 threads: 256 math (2 WGs) + 128 specialized (2 TMA + 2 UMMA)
  - 1 TMA per KV block [128, 128], UMMA iterates 4x K=32
  - 2 warp groups process 2 KV blocks per iteration (kNumMathWarpGroups=2)
  - Q reloaded via TMA pipeline when q_idx (batch) changes
  - Persistent kernel: CTAs iterate through assigned (q_idx, kv_idx) pairs
  - Weights cached in registers: preloaded once per q_idx change (not per KV block)
  - KV Scales loaded via TMA to SMEM (separate pipeline per group, Math consumes)

Merged KV+Scale pipeline:
  - KV data and scales share a single TMA barrier per group
  - TMA loads both KV and Scale under one barrier (combined tx_count)
  - UMMA waits on merged barrier (for KV GEMM), does NOT release
  - Math waits on merged barrier (for scale read), Math releases

Fused KV layout:
  - KV data and scales stored contiguously per physical block:
    [num_phys_blocks, block_kv * (head_dim + 4)] bytes
  - Per block: [KV_all_tokens (block_kv * head_dim bytes)] [Scales (block_kv * 4 bytes)]
  - KV and Scale views are derived inside __call__ using CuTE pointer arithmetic

Scheduler:
  - schedule_meta[sm_idx] = (start_q_idx, start_kv_idx / kNumMathWarpGroups)
  - schedule_meta[sm_idx+1] = end boundary for this CTA
  - fetch_next_task pattern: each warp role independently advances (q_idx, kv_idx)
  - kv_idx in units of KV blocks, advances by kNumMathWarpGroups=2 per step

Dynamic shape support:
  - Model-constant dims (block_kv, head_dim, N, per_token) remain static for codegen
  - Runtime-varying dims (batch_size, num_phys_blocks, max_ctx, max_blocks_per_seq,
    num_ctas) are marked dynamic via mark_compact_shape_dynamic
  - Allows JIT cache reuse across different batch sizes / sequence lengths

Epilogue dtype flows (--acc_dtype / --epi_dtype):

  Flow 1: --acc_dtype fp32 --epi_dtype fp16
    Q(FP8) x K(FP8) -> MMA acc(FP32) -> TMEM(FP32)
      -> LDTM -> Reg(FP32) -> cvt FP16 -> ReLU(FP16)
      -> FMA(fma.rn.f16x2) with weights(FP16 from SMEM) -> partial sum(FP16)
      -> x scale(FP32->FP16) -> cvt output_dtype -> store logits(output_dtype)

  Flow 2: --acc_dtype fp16 --epi_dtype fp16
    Q(FP8) x K(FP8) -> MMA acc(FP16) -> TMEM(FP16, pack_16b)
      -> LDTM -> Reg(FP16) -> ReLU(FP16)
      -> FMA(fma.rn.f16x2) with weights(FP16 from SMEM) -> partial sum(FP16)
      -> x scale(FP32->FP16) -> cvt output_dtype -> store logits(output_dtype)

  --output_dtype: fp32 (default), fp16, bf16. Controls logits tensor dtype and final store conversion.
  Default: --acc_dtype fp32 --epi_dtype fp32 --output_dtype fp32
"""

from typing import Any, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float16, Int32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.arch import get_max_tmem_alloc_cols
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .._math_policy import _MathMode, _validate_math_mode

# CuTe DSL CUDA 13 validates rounding modes as string literals. The string
# form is also accepted by older wrappers, so keep it version-independent.
_RND_RN = "rn"


@dsl_user_op
def abs_f32(a: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    """Keep the packed fast-ReLU absolute operand visible to code generation."""
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip)],
            "abs.f32 $0, $1;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def relu_f32(a: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    """relu(x) = max(x, 0) as one FMNMX (PTX `max.NaN.f32`): exact and range-safe for
    every finite x (no intermediate can overflow), NaN-propagating like the
    cutlass.max it stands in for, one instruction per head where cutlass.max
    lowers to a compare + select pair."""
    f32_ty = cutlass.Float32.mlir_type
    return cutlass.Float32(
        llvm.inline_asm(
            f32_ty,
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip)],
            "max.NaN.f32 $0, $1, 0f00000000;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


# Epilogue FMA unroll. NUM_W_IN_REG is the split point between the register
# and SMEM weight paths, both unrolled this wide, so it must stay a multiple
# of it -- otherwise the two paths stop tiling the subtile and the epilogue
# reads past it. Mirrors _EPI_SUBTILE_UNROLL in attn_scores.py, which applies
# the same granularity to num_heads // num_epi_subtiles.
_EPI_UNROLL = 4


@dsl_user_op
def pack_f16x2(
    a: Float16,
    b: Float16,
    *,
    loc=None,
    ip=None,
) -> Int32:
    f16_ty = Float16.mlir_type
    i32_ty = Int32.mlir_type
    vec2_f16 = ir.VectorType.get([2], f16_ty, loc=loc)
    v = vector.from_elements(
        vec2_f16,
        (Float16(a).ir_value(loc=loc, ip=ip), Float16(b).ir_value(loc=loc, ip=ip)),
        loc=loc,
        ip=ip,
    )
    return Int32(llvm.bitcast(i32_ty, v, loc=loc, ip=ip))


@dsl_user_op
def unpack_f16x2(
    packed: Int32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float16, Float16]:
    f16_ty = Float16.mlir_type
    vec2_f16 = ir.VectorType.get([2], f16_ty, loc=loc)
    v = llvm.bitcast(vec2_f16, Int32(packed).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    r0 = Float16(
        vector.extract(v, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    )
    r1 = Float16(
        vector.extract(v, dynamic_position=[], static_position=[1], loc=loc, ip=ip)
    )
    return r0, r1


@dsl_user_op
def fma_f16x2(
    a: Int32,
    b: Int32,
    c: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    i32_ty = Int32.mlir_type
    return Int32(
        llvm.inline_asm(
            i32_ty,
            [
                Int32(a).ir_value(loc=loc, ip=ip),
                Int32(b).ir_value(loc=loc, ip=ip),
                Int32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f16x2 $0, $1, $2, $3;",
            "=r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def max_f16x2(
    a: Int32,
    b: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    i32_ty = Int32.mlir_type
    return Int32(
        llvm.inline_asm(
            i32_ty,
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "max.f16x2 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def add_f16x2(
    a: Int32,
    b: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    i32_ty = Int32.mlir_type
    return Int32(
        llvm.inline_asm(
            i32_ty,
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "add.f16x2 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


# Default register budget per math thread for the epilogue weight cache. 160
# is the largest footprint the per-slot policy in the engine requests at the shape
# the kernel was tuned for (num_heads=64, next_n=4), so it leaves every
# num_heads=64 configuration unchanged. The constructor's w_cache_regs
# overrides it so the API's internal per-shape policy can lower it where the
# default spills: at num_heads=32, next_n=6 the default caches 24 weights per
# slot (144 registers) and Nsight Compute shows ~12k local loads / ~8k local
# stores per launch; 96 registers (16 per slot) removes every spill and runs
# 8-15% (B200) / 11-35% (Rubin) faster; next_n=5, 7 and 8 spill the same way
# under the default and gain up to 24%, 9% and 29% on B200 (37% on Rubin at
# 8).  See _fp8_epilogue_policy in attn_scores.py for the measured table.
_MAX_W_CACHE_REGS = 160


def _arch_major_minor(arch: str) -> "tuple[int, int]":
    """Parse a codegen arch string (``"sm_107a"``, ``"sm_100f"``, ``"sm_100"``)
    into its compute-capability ``(major, minor)`` pair, e.g. ``(10, 7)``."""
    digits = arch.removeprefix("sm_").rstrip("af")
    return int(digits[:-1]), int(digits[-1])


def _is_rubin_arch(arch: str) -> bool:
    """Whether ``arch`` is a Rubin part (sm_107 / sm_109).

    Takes the compile-target arch as an explicit string (the same value the
    caller passes to ``cute.compile`` as ``--gpu-arch``) instead of querying
    the DSL: this is consulted at kernel *construction* time, before
    ``cute.compile`` installs the target, when a DSL arch query would return
    the ambient arch (``CUTE_DSL_ARCH`` or a device-ordinal-0 probe) -- the
    wrong device on a heterogeneous node.

    NOTE: there is no canonical "is Rubin" Arch API -- Arch.BlackwellArchs()
    lumps sm_107/sm_109 in with Blackwell, and Arch.is_family_of() only compares
    the major version (can't split Rubin from Blackwell) and is suffix-sensitive
    (False for a base sm_107). Within major==10 the archs are sm_100/103
    (Blackwell, minor 0/3) and sm_107/109 (Rubin, minor 7/9), so `minor >= 7` is
    the robust discriminator (sm_110/sm_101 are major 11, excluded).
    """
    major, minor = _arch_major_minor(arch)
    return major == 10 and minor >= 7


class _FP8MQALayoutMixin:
    head_dim: int
    mma_tiler_mn: tuple[int, int]
    cluster_shape_mn: tuple[int, int]
    num_kv_stages: int
    num_q_stages: int
    num_groups: int
    num_umma_stages: int

    def _setup_mma_layout(self, tiled_mma, a_dtype, b_dtype):
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])  # 32

        # Full-K: tile K = head_dim (128), 1 TMA per block
        mma_inst_tile_k = self.head_dim // mma_inst_shape_k  # 4
        full_k = mma_inst_shape_k * mma_inst_tile_k  # 128
        self.mma_tiler = (
            self.mma_tiler_mn[0],
            self.mma_tiler_mn[1],
            full_k,
        )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.epi_tile = self.cta_tile_shape_mnk[:2]

        # Each KV stage per group holds the full compute tile.
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            a_dtype,
            self.num_kv_stages,
        )
        # Each Q stage holds full [N, head_dim].
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            b_dtype,
            self.num_q_stages,
        )

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        self.num_tmem_alloc_cols_total = (
            self.num_tmem_alloc_cols * self.num_groups * self.num_umma_stages
        )
        return tiled_mma


class _FP8MQAEngineMixin(_FP8MQALayoutMixin):
    """Shared TMA, UMMA, math/TMEM and selective store-role lifecycles.

    Ordinary and selective kernels retain their own construction, storage and
    launch interfaces. Compile-time policies preserve their distinct geometry,
    warp roles, numerical reductions and output publication.
    State declarations are annotation-only; each kernel populates configuration,
    MMA layout and TMA transfer sizes before launch.
    """

    # Ordinary-only metadata flag; selective subclasses retain their own ABI.
    explicit_query_bounds: bool = False
    a_dtype: Any
    _math_mode: _MathMode
    acc_dtype: Any
    b_dtype: Any
    N: int
    block_kv: int
    early_tmem_copy: bool
    epi_dtype: Any
    next_n: int
    num_blocks_per_mma: int
    num_epi_subtiles: int
    num_heads: int
    num_kv_scale_tma_bytes: int
    num_math_threads: int
    num_math_warps: int
    num_q_tma_bytes: int
    output_dtype: Any
    phys_block_kv: int
    remove_kv_wait_in_epilogue: bool
    smem_pad_bytes: int
    use_flat_logits_view: bool
    output_index_bits: int
    use_paired_pipeline_polls: bool
    use_two_level_task_loop: bool
    w_cache_regs: int

    # Request-tiled consumers supply these pure policies. Ordinary roles
    # compile out their calls and retain their own reduction/view policy.
    _request_math_fragment: Any
    _make_math_tmem_copy_and_partition: Any

    _engine_relu_f32 = staticmethod(relu_f32)

    @property
    def math_mode(self) -> _MathMode:
        """Constructor-fixed FP32 epilogue policy; FP16 arithmetic is unchanged."""
        return self._math_mode

    @cute.jit
    def _release_kv_after_scale_use(self, kv_pipeline, kv_cons_state):
        """End a KV generation at the consumer's explicit final scale-use site.

        Ordinary calls at its existing early epilogue boundary; SAMPLE calls
        only after every flattened subtile has consumed the scale. Reduction
        and ingress adapters must not call this or advance states themselves.
        """
        cute.arch.fence_view_async_shared()
        kv_pipeline.consumer_release(kv_cons_state)
        kv_cons_state.advance()
        return kv_cons_state

    @cute.jit
    def _lookup_fp8_page(
        self,
        mBlockTable,
        mSamplePages,
        request,
        page,
        internal_sample: cutlass.Constexpr,
    ):
        """Pure guarded-ingress adapter; callers own page-tail eligibility.

        An internal sample page is a logical index into the original table,
        never a physical ID to clamp. Keep the existing missing-slot fallback
        and leave invalid live metadata to the existing host validation.
        """
        if cutlass.const_expr(internal_sample):
            logical_page = cutlass.Int32(0)
            if page < mSamplePages.shape[1]:
                logical_page = cutlass.Int32(mSamplePages[(request, page)])
            page = logical_page
        return mBlockTable[(request, page)]

    @cute.jit
    def _accumulate_fp32_heads(self, acc, acc_base, weights, weight_coords, sums):
        """Pure four-head reduction step; no pipeline or stage transitions.

        The consumer binds its existing ReLU implementation statically. The
        caller retains row/head mapping and final summation association.
        """
        if cutlass.const_expr(self.math_mode == "relu_fast"):
            # Compensated once in _math_scale_value, never in a publisher.
            # Explicit FP32 conversion also preserves FP16-acc/FP32-epi support.
            v0 = cutlass.Float32(acc[acc_base])
            v1 = cutlass.Float32(acc[acc_base + 1])
            v2 = cutlass.Float32(acc[acc_base + 2])
            v3 = cutlass.Float32(acc[acc_base + 3])
            a0, a1 = cute.arch.add_packed_f32x2((v0, v1), (abs_f32(v0), abs_f32(v1)))
            a2, a3 = cute.arch.add_packed_f32x2((v2, v3), (abs_f32(v2), abs_f32(v3)))
        else:
            # relu as one FMNMX per head (`max.NaN.f32`): exact and range-safe for
            # every finite accumulator (the (x + |x|) / 2 form it replaces overflowed
            # for x >= 2^127, reachable through unbounded scales / weights), NaN-
            # propagating like the scalar max, one instruction per head. Weights are
            # applied unscaled. Ported from DKG MR !27837, range-safe form.
            a0 = self._engine_relu_f32(acc[acc_base])
            a1 = self._engine_relu_f32(acc[acc_base + 1])
            a2 = self._engine_relu_f32(acc[acc_base + 2])
            a3 = self._engine_relu_f32(acc[acc_base + 3])
        w0 = weights[weight_coords[0]]
        w1 = weights[weight_coords[1]]
        w2 = weights[weight_coords[2]]
        w3 = weights[weight_coords[3]]
        s0x, s0y = cute.arch.fma_packed_f32x2(
            (a0, a1), (w0, w1), (sums[0], sums[1]), rnd=_RND_RN
        )
        s1x, s1y = cute.arch.fma_packed_f32x2(
            (a2, a3), (w2, w3), (sums[2], sums[3]), rnd=_RND_RN
        )
        return s0x, s0y, s1x, s1y

    @cute.jit
    def _issue_paged_kv_tile(
        self,
        kv_pipeline,
        kv_prod_state,
        phys_blks,
        tma_atom_a,
        tAgA,
        tAsA,
        tma_atom_s,
        tSgS,
        tSsS,
        a_mcast_mask,
    ):
        """Own one merged KV/scale producer generation after guarded ingress.

        Physical IDs are resolved by the consumer's ingress policy before this
        call. Acquiring and advancing the ring remain execution-engine work;
        the page adapter cannot change stage ownership.
        """
        kv_pipeline.producer_acquire(kv_prod_state)
        bar = kv_pipeline.producer_get_barrier(kv_prod_state)
        stage = kv_prod_state.index
        for i in cutlass.range_constexpr(self.num_blocks_per_mma):
            cute.copy(
                tma_atom_a,
                tAgA[(None, 0, 0, phys_blks[i])],
                tAsA[(None, i, stage)],
                tma_bar_ptr=bar,
                mcast_mask=a_mcast_mask,
            )
            cute.copy(
                tma_atom_s,
                tSgS[(None, 0, phys_blks[i])],
                tSsS[(None, i, stage)],
                tma_bar_ptr=bar,
            )
        kv_prod_state.advance()
        return kv_prod_state

    @cute.jit
    def _issue_umma_group(
        self,
        tiled_mma,
        tCrA,
        tCrB,
        tCtAcc_base,
        q_stage,
        kv_pipeline,
        kv_cons_state,
        umma_pipeline,
        umma_prod_state,
        paired_poll: cutlass.Constexpr,
    ):
        """Issue one group; math, not UMMA, owns the KV-stage release.

        The caller owns Q readiness and task traversal. In the existing single
        producer SAMPLE policy it calls this twice, group0 then group1, before
        advancing the task. Ordinary retains independent producer roles and
        plain waits. Keep all KV/UMMA generation transitions explicit here.
        """
        if cutlass.const_expr(paired_poll):
            kv_ready = kv_pipeline.consumer_try_wait(kv_cons_state)
            umma_empty = umma_pipeline.producer_try_acquire(umma_prod_state)
            kv_pipeline.consumer_wait(kv_cons_state, kv_ready)
            umma_pipeline.producer_acquire(umma_prod_state, umma_empty)
        else:
            kv_pipeline.consumer_wait(kv_cons_state)
            umma_pipeline.producer_acquire(umma_prod_state)
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        kv_stage = kv_cons_state.index
        tCtAcc = tCtAcc_base[(None, None, None, umma_prod_state.index)]
        for k_block in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
            cute.gemm(
                tiled_mma,
                tCtAcc,
                tCrA[None, None, k_block, kv_stage],
                tCrB[None, None, k_block, q_stage],
                tCtAcc,
            )
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        # No consumer_release here: the math role still needs shared scales.
        kv_cons_state.advance()
        umma_pipeline.producer_commit(umma_prod_state)
        umma_prod_state.advance()
        # Atom.set mutates trait IR too. Return/rebind the MMA as a plain local
        # so staged loops/branches carry it explicitly, just like ring states.
        return tiled_mma, kv_cons_state, umma_prod_state

    @cute.jit
    def _run_umma_role(
        self,
        mContextLens,
        batch_size,
        block_kv_val,
        end_q_idx,
        end_kv_idx,
        next_q_idx,
        next_kv_idx,
        next_num_kv,
        q_idx,
        has_work,
        is_leader_cta,
        q_pipeline,
        q_cons_state,
        kv_pipeline_0,
        kv_cons_state_0,
        umma_pipeline_0,
        umma_prod_state_0,
        tCrA_0,
        tCrB,
        tiled_mma,
        tmem,
        tCtAcc_fake_staged,
        group_0_offset,
        paired_poll: cutlass.Constexpr,
        serial_pair: cutlass.Constexpr,
        tCrA_1=None,
        kv_pipeline_1=None,
        kv_cons_state_1=None,
        umma_pipeline_1=None,
        umma_prod_state_1=None,
        group_1_offset=0,
    ):
        """Own one producer role's Q generations, task cursor and UMMA rings.

        Ordinary calls once per independent producer warp. SAMPLE calls once
        with serial_pair, committing group 0 before group 1 on the same Q
        stage. Register budgeting and initial state placement remain in the
        entry wrapper; all states die at the end of this complete role.
        """
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tCtAcc_base_0 = cute.make_tensor(
            tmem_ptr + group_0_offset, tCtAcc_fake_staged.layout
        )
        if cutlass.const_expr(serial_pair):
            tCtAcc_base_1 = cute.make_tensor(
                tmem_ptr + group_1_offset, tCtAcc_fake_staged.layout
            )

        if is_leader_cta:
            q_stage = cutlass.Int32(0)
            while has_work:
                q_idx_old = q_idx
                q_idx = next_q_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv

                # Q visibility is independent of KV TMA completion. Math owns
                # the Q release; each producer only observes its generation.
                # In particular, ordinary warp 11 receives only KV1 from TMA
                # warp 9: KV1 readiness cannot certify Q SMEM visibility.
                if q_idx != q_idx_old:
                    if q_idx_old < batch_size:
                        q_cons_state.advance()
                    q_pipeline.consumer_wait(q_cons_state)
                    q_stage = q_cons_state.index

                # Explicit MMA SSA propagation also crosses this role helper:
                # a mutated trait must not escape into a sibling DSL region.
                # UMMA remains unconditional for the aligned KV pair: an OOB
                # half computes padding data, whose output eligibility is
                # still controlled by the unchanged math epilogue.
                tiled_mma, kv_cons_state_0, umma_prod_state_0 = self._issue_umma_group(
                    tiled_mma,
                    tCrA_0,
                    tCrB,
                    tCtAcc_base_0,
                    q_stage,
                    kv_pipeline_0,
                    kv_cons_state_0,
                    umma_pipeline_0,
                    umma_prod_state_0,
                    paired_poll,
                )
                if cutlass.const_expr(serial_pair):
                    tiled_mma, kv_cons_state_1, umma_prod_state_1 = (
                        self._issue_umma_group(
                            tiled_mma,
                            tCrA_1,
                            tCrB,
                            tCtAcc_base_1,
                            q_stage,
                            kv_pipeline_1,
                            kv_cons_state_1,
                            umma_pipeline_1,
                            umma_prod_state_1,
                            paired_poll,
                        )
                    )

                next_kv_idx = kv_idx + 2
                if next_kv_idx >= num_kv:
                    next_q_idx = q_idx + 1
                    next_kv_idx = 0
                    if next_q_idx < batch_size:
                        next_num_kv = (
                            mContextLens[next_q_idx] + block_kv_val - 1
                        ) // block_kv_val
                    # Keep the exact CTA boundary reachable while skipping
                    # empty rows; SAMPLE lengths count sampled slots here.
                    while (
                        (next_num_kv == 0)
                        & (next_q_idx < batch_size)
                        & ((next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx))
                    ):
                        next_q_idx = next_q_idx + 1
                        if next_q_idx < batch_size:
                            next_num_kv = (
                                mContextLens[next_q_idx] + block_kv_val - 1
                            ) // block_kv_val
                has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)
        if cutlass.const_expr(serial_pair):
            return (
                tiled_mma,
                q_cons_state,
                kv_cons_state_0,
                umma_prod_state_0,
                kv_cons_state_1,
                umma_prod_state_1,
            )
        else:
            return tiled_mma, q_cons_state, kv_cons_state_0, umma_prod_state_0

    @cute.jit
    def _tma_q_source_views(
        self,
        q_idx,
        request_tiled: cutlass.Constexpr,
        mTileMeta,
        mB_nkl,
        mW_tma,
        thr_mma,
        sQ,
        sW,
        tma_atom_b,
        tma_atom_w,
        tBgB,
        tWgW,
    ):
        """Select ingress views only; the role owns Q/W barrier transitions."""
        if cutlass.const_expr(request_tiled):
            query_start = mTileMeta[(q_idx, 0)]
            shifted_b = cute.domain_offset((query_start * self.num_heads, 0, 0), mB_nkl)
            shifted_gB = cute.local_tile(
                shifted_b,
                cute.slice_(self.mma_tiler, (0, None, None)),
                (None, None, None),
            )
            shifted_tCgB = thr_mma.partition_B(shifted_gB)
            _, shifted_tBgB = cpasync.tma_partition(
                tma_atom_b,
                0,
                cute.make_layout(1),
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(shifted_tCgB, 0, 3),
            )
            shifted_w = cute.domain_offset((query_start, 0), mW_tma)
            shifted_gW = cute.local_tile(
                shifted_w, (self.next_n, self.num_heads), (None, None)
            )
            _, shifted_tWgW = cpasync.tma_partition(
                tma_atom_w,
                0,
                cute.make_layout(1),
                cute.group_modes(sW, 0, 2),
                cute.group_modes(shifted_gW, 0, 2),
            )
            return shifted_tBgB[(None, 0, 0, 0)], shifted_tWgW[(None, 0, 0)]
        else:
            return tBgB[(None, 0, q_idx)], tWgW[(None, q_idx)]

    @cute.jit
    def _run_tma_role(
        self,
        mContextLens,
        batch_size,
        block_kv_val,
        end_q_idx,
        end_kv_idx,
        next_q_idx,
        next_kv_idx,
        next_num_kv,
        q_idx,
        has_work,
        tidx,
        kv_pipeline,
        kv_prod_state,
        mBlockTable,
        mSamplePages,
        mTileMeta,
        tma_atom_a,
        tAgA,
        tAsA,
        tma_atom_s,
        tSgS,
        tSsS,
        a_mcast_mask,
        group: cutlass.Constexpr,
        owns_q: cutlass.Constexpr,
        request_tiled: cutlass.Constexpr,
        guarded_first_q: cutlass.Constexpr,
        guarded_prefetch: cutlass.Constexpr,
        sample_internal: cutlass.Constexpr,
        q_pipeline=None,
        q_prod_state=None,
        mB_nkl=None,
        mW_tma=None,
        thr_mma=None,
        sQ=None,
        sW=None,
        tma_atom_b=None,
        tma_atom_w=None,
        tBgB=None,
        tWgW=None,
        tBsB=None,
        tWsW=None,
        b_mcast_mask=None,
    ):
        """Own a paged TMA role's task/Q lookahead, cache and producer rings.

        Entry wrappers retain register budgeting and initial state placement.
        Group 1 statically has no Q/view access. Ordinary keeps unconditional
        first-Q and column-only page guards; SAMPLE guards idle Q and tiles.
        """
        lane_idx = tidx % 32
        cached_blks = [cutlass.Int32(0) for _ in range(self.num_blocks_per_mma)]
        kv_blk_ptr = cutlass.Int32(32)

        if cutlass.const_expr(owns_q):
            initial_q = True
            if cutlass.const_expr(guarded_first_q):
                initial_q = has_work
            # Idle SAMPLE CTAs have no valid packed query tile: construct
            # neither its metadata-derived view nor its transaction there.
            if initial_q:
                q_pipeline.producer_acquire(q_prod_state)
                q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                q_source, w_source = self._tma_q_source_views(
                    next_q_idx,
                    request_tiled,
                    mTileMeta,
                    mB_nkl,
                    mW_tma,
                    thr_mma,
                    sQ,
                    sW,
                    tma_atom_b,
                    tma_atom_w,
                    tBgB,
                    tWgW,
                )
                cute.copy(
                    tma_atom_b,
                    q_source,
                    tBsB[(None, q_prod_state.index)],
                    tma_bar_ptr=q_bar,
                    mcast_mask=b_mcast_mask,
                )
                cute.copy(
                    tma_atom_w,
                    w_source,
                    tWsW[(None, q_prod_state.index)],
                    tma_bar_ptr=q_bar,
                )
                q_prod_state.advance()

        while has_work:
            q_idx_old = q_idx
            q_idx = next_q_idx
            kv_idx = next_kv_idx
            num_kv = next_num_kv
            if q_idx != q_idx_old:
                kv_blk_ptr = cutlass.Int32(32)
                if cutlass.const_expr(owns_q):
                    prefetch_next = q_idx + 1
                    # Task traversal skips zero rows. Lookahead must stage
                    # the SAME next live Q/W; staging a skipped row silently
                    # gives its successor wrong logits without a barrier hang.
                    peek_ctx = cutlass.Int32(1)
                    if prefetch_next < end_q_idx:
                        peek_ctx = mContextLens[prefetch_next]
                    while (peek_ctx == 0) & (prefetch_next < end_q_idx):
                        prefetch_next = prefetch_next + 1
                        if prefetch_next < end_q_idx:
                            peek_ctx = mContextLens[prefetch_next]
                        else:
                            peek_ctx = cutlass.Int32(1)
                    if (prefetch_next < end_q_idx) | (
                        (prefetch_next == end_q_idx) & (end_kv_idx > 0)
                    ):
                        q_pipeline.producer_acquire(q_prod_state)
                        q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                        q_source, w_source = self._tma_q_source_views(
                            prefetch_next,
                            request_tiled,
                            mTileMeta,
                            mB_nkl,
                            mW_tma,
                            thr_mma,
                            sQ,
                            sW,
                            tma_atom_b,
                            tma_atom_w,
                            tBgB,
                            tWgW,
                        )
                        # Q and weights share the original transaction barrier
                        # and stage; there is no separate producer commit/tail.
                        cute.copy(
                            tma_atom_b,
                            q_source,
                            tBsB[(None, q_prod_state.index)],
                            tma_bar_ptr=q_bar,
                            mcast_mask=b_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_w,
                            w_source,
                            tWsW[(None, q_prod_state.index)],
                            tma_bar_ptr=q_bar,
                        )
                        q_prod_state.advance()

            # Each role owns a 32-lane page cache, reset on Q change or after
            # 32 compute tiles. Shuffle the current column before issuing TMA.
            if kv_blk_ptr == 32:
                kv_blk_ptr = cutlass.Int32(0)
                prefetch_kv = kv_idx + group + lane_idx * 2
                if cutlass.const_expr(guarded_prefetch):
                    if prefetch_kv < num_kv:
                        page_table_row = q_idx
                        if cutlass.const_expr(request_tiled):
                            page_table_row = mTileMeta[(q_idx, 1)]
                        num_phys_row = (
                            mContextLens[q_idx] + self.phys_block_kv - 1
                        ) // self.phys_block_kv
                        for i in cutlass.range_constexpr(self.num_blocks_per_mma):
                            cached_blks[i] = cutlass.Int32(0)
                            col = prefetch_kv * self.num_blocks_per_mma + i
                            # Guard BEFORE sample indirection/table lookup.
                            if col < num_phys_row:
                                cached_blks[i] = self._lookup_fp8_page(
                                    mBlockTable,
                                    mSamplePages,
                                    page_table_row,
                                    col,
                                    sample_internal,
                                )
                    else:
                        for i in cutlass.range_constexpr(self.num_blocks_per_mma):
                            cached_blks[i] = cutlass.Int32(0)
                else:
                    # Natural-width tables guarantee only ceil(ctx/page)
                    # columns, not a whole compute tile. Tail-predicate each
                    # column before lookup; padding uses safe physical page0
                    # and the unchanged epilogue excludes positions >=ctx.
                    num_phys_row = (
                        mContextLens[q_idx] + self.phys_block_kv - 1
                    ) // self.phys_block_kv
                    for i in cutlass.range_constexpr(self.num_blocks_per_mma):
                        col = prefetch_kv * self.num_blocks_per_mma + i
                        blk = cutlass.Int32(0)
                        if col < num_phys_row:
                            blk = self._lookup_fp8_page(
                                mBlockTable, mSamplePages, q_idx, col, sample_internal
                            )
                        cached_blks[i] = blk
            phys_blks = [cutlass.Int32(0)] * self.num_blocks_per_mma
            for i in cutlass.range_constexpr(self.num_blocks_per_mma):
                phys_blks[i] = cute.arch.shuffle_sync(cached_blks[i], kv_blk_ptr)
            kv_blk_ptr = kv_blk_ptr + 1
            kv_prod_state = self._issue_paged_kv_tile(
                kv_pipeline,
                kv_prod_state,
                phys_blks,
                tma_atom_a,
                tAgA,
                tAsA,
                tma_atom_s,
                tSgS,
                tSsS,
                a_mcast_mask,
            )

            next_kv_idx = kv_idx + 2
            if next_kv_idx >= num_kv:
                next_q_idx = q_idx + 1
                next_kv_idx = 0
                if next_q_idx < batch_size:
                    next_num_kv = (
                        mContextLens[next_q_idx] + block_kv_val - 1
                    ) // block_kv_val
                # Stop at the exact CTA end pair when skipping empty rows;
                # otherwise the end coordinate could become unreachable.
                while (
                    (next_num_kv == 0)
                    & (next_q_idx < batch_size)
                    & ((next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx))
                ):
                    next_q_idx = next_q_idx + 1
                    if next_q_idx < batch_size:
                        next_num_kv = (
                            mContextLens[next_q_idx] + block_kv_val - 1
                        ) // block_kv_val
            has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)
        if cutlass.const_expr(owns_q):
            return q_prod_state, kv_prod_state
        else:
            return kv_prod_state

    @cute.jit
    def _ordinary_math_fragment(
        self,
        acc_vec,
        sums,
        t: cutlass.Constexpr,
        i: cutlass.Constexpr,
        subtile_n: cutlass.Constexpr,
        NUM_W_IN_REG: cutlass.Constexpr,
        num_heads: cutlass.Constexpr,
        w_cache,
        sW,
        q_stage_local,
    ):
        """Pure per-row arithmetic; the consumer owns every stage lifetime."""
        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
            ps0, ps1 = sums
            packed_zero = pack_f16x2(Float16(0.0), Float16(0.0))
        else:
            s0x, s0y, s1x, s1y = sums
        # Reg-path: weights from registers
        reg_h_end = min(subtile_n, max(0, NUM_W_IN_REG - i * subtile_n))
        for h in cutlass.range_constexpr(0, reg_h_end, 4):
            n0 = h
            h_g = i * subtile_n + h
            if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                pa01 = pack_f16x2(
                    Float16(acc_vec[n0]),
                    Float16(acc_vec[n0 + 1]),
                )
                pa23 = pack_f16x2(
                    Float16(acc_vec[n0 + 2]),
                    Float16(acc_vec[n0 + 3]),
                )
                pa01 = max_f16x2(pa01, packed_zero)
                pa23 = max_f16x2(pa23, packed_zero)
                r0 = t * NUM_W_IN_REG + h_g
                pw01 = pack_f16x2(w_cache[r0], w_cache[r0 + 1])
                pw23 = pack_f16x2(w_cache[r0 + 2], w_cache[r0 + 3])
                ps0 = fma_f16x2(pa01, pw01, ps0)
                ps1 = fma_f16x2(pa23, pw23, ps1)
            else:
                # Strict uses range-safe NaN-aware max.
                # Fast opts into packed x+abs(x), compensated once in the
                # common scale helper; its doubled intermediates may overflow.
                # Neither mode rescales weights or changes FMA association.
                r0 = t * NUM_W_IN_REG + h_g
                s0x, s0y, s1x, s1y = self._accumulate_fp32_heads(
                    acc_vec,
                    n0,
                    w_cache,
                    (r0, r0 + 1, r0 + 2, r0 + 3),
                    (s0x, s0y, s1x, s1y),
                )
        # SMEM-path: weights from shared mem
        smem_h_start = max(0, NUM_W_IN_REG - i * subtile_n)
        for h in cutlass.range_constexpr(smem_h_start, subtile_n, 4):
            n0 = h
            h_g = i * subtile_n + h
            if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                pa01 = pack_f16x2(
                    Float16(acc_vec[n0]),
                    Float16(acc_vec[n0 + 1]),
                )
                pa23 = pack_f16x2(
                    Float16(acc_vec[n0 + 2]),
                    Float16(acc_vec[n0 + 3]),
                )
                pa01 = max_f16x2(pa01, packed_zero)
                pa23 = max_f16x2(pa23, packed_zero)
                pw01 = pack_f16x2(
                    sW[(t * num_heads + h_g, q_stage_local)],
                    sW[(t * num_heads + h_g + 1, q_stage_local)],
                )
                pw23 = pack_f16x2(
                    sW[(t * num_heads + h_g + 2, q_stage_local)],
                    sW[(t * num_heads + h_g + 3, q_stage_local)],
                )
                ps0 = fma_f16x2(pa01, pw01, ps0)
                ps1 = fma_f16x2(pa23, pw23, ps1)
            else:
                # Keep the same selected FP32 policy on the shared-weight
                # path as the register-weight path above. The common scale
                # helper owns fast-mode compensation, never this reduction.
                s0x, s0y, s1x, s1y = self._accumulate_fp32_heads(
                    acc_vec,
                    n0,
                    sW,
                    (
                        (
                            t * num_heads + h_g,
                            q_stage_local,
                        ),
                        (
                            t * num_heads + h_g + 1,
                            q_stage_local,
                        ),
                        (
                            t * num_heads + h_g + 2,
                            q_stage_local,
                        ),
                        (
                            t * num_heads + h_g + 3,
                            q_stage_local,
                        ),
                    ),
                    (s0x, s0y, s1x, s1y),
                )
        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
            return ps0, ps1
        else:
            return s0x, s0y, s1x, s1y

    @cute.jit
    def _store_ordinary_math_result(
        self,
        sums,
        scale_val,
        output_view,
        out_row_off,
        logits_stride0,
        q_idx,
        t: cutlass.Constexpr,
        next_n: cutlass.Constexpr,
        kv_pos,
        query_metadata=None,
    ):
        """Keep the ordinary final association, scale conversion and store."""
        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
            ps0, ps1 = sums
        else:
            s0x, s0y, s1x, s1y = sums
        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
            ps_sum = add_f16x2(ps0, ps1)
            sum_lo, sum_hi = unpack_f16x2(ps_sum)
            result_t = sum_lo + sum_hi
        else:
            result_t = s0x + s0y + s1x + s1y
        if cutlass.const_expr(self.explicit_query_bounds):
            # Metadata row: real output base, valid count, exact physical ends.
            # Padded Q/weights are real allocated lanes; they never publish rows.
            if t < query_metadata[(q_idx, 1)]:
                out_row = query_metadata[(q_idx, 0)] + t
                value = self.output_dtype(result_t * scale_val)
                if kv_pos >= query_metadata[(q_idx, 2 + t)]:
                    value = self.output_dtype(-float("inf"))
                if cutlass.const_expr(self.use_flat_logits_view):
                    offset_dtype = (
                        cutlass.Int64 if self.output_index_bits == 64 else cutlass.Int32
                    )
                    out_off = offset_dtype(out_row) * logits_stride0 + kv_pos
                    output_view[out_off] = value
                else:
                    output_view[(out_row, kv_pos)] = value
        else:
            if cutlass.const_expr(self.use_flat_logits_view):
                out_off_t = out_row_off + t * logits_stride0 + kv_pos
                if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                    output_view[out_off_t] = self.output_dtype(
                        result_t * Float16(scale_val)
                    )
                else:
                    output_view[out_off_t] = self.output_dtype(result_t * scale_val)
            else:
                out_row = q_idx * next_n + t
                if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                    output_view[(out_row, kv_pos)] = self.output_dtype(
                        result_t * Float16(scale_val)
                    )
                else:
                    output_view[(out_row, kv_pos)] = self.output_dtype(
                        result_t * scale_val
                    )

    @cute.jit
    def _math_scale_value(
        self, sScales, m_coord, stage, group, request_tiled: cutlass.Constexpr
    ):
        if cutlass.const_expr(request_tiled):
            scale = sScales[(m_coord, stage, group)]
        else:
            scale = sScales[(m_coord, stage)]
        # Fast FP32 ReLU doubles each value before weighted accumulation.
        # Compensate exactly once here across ordinary and all selective modes.
        # FP16 epilogues keep their original max/FMA/scale-cast arithmetic.
        if cutlass.const_expr(
            self.epi_dtype == cutlass.Float32 and self.math_mode == "relu_fast"
        ):
            scale = scale * cutlass.Float32(0.5)
        return scale

    @cute.jit
    def _load_math_fragment(
        self,
        tiled_copy,
        partition,
        fragment,
        index: cutlass.Constexpr,
        last: cutlass.Constexpr,
        umma_pipeline,
        umma_cons_state,
        prefetched: cutlass.Constexpr = False,
    ):
        if cutlass.const_expr(index > 0):
            if cutlass.const_expr(not prefetched):
                cute.copy(tiled_copy, partition[(None, None, None, 0, index)], fragment)
            cute.arch.fence_view_async_tmem_load()
        # Release only after the last LDTM + fence, before loading its
        # register fragment. The producer may reuse TMEM after that fence.
        if cutlass.const_expr(last):
            umma_pipeline.consumer_release(umma_cons_state)
            umma_cons_state.advance()
        acc_vec = fragment.load()
        if cutlass.const_expr(prefetched and not last):
            # Candidate score-plane policy overlaps the next LDTM with this
            # fragment's arithmetic, after materializing its current SSA value.
            cute.copy(tiled_copy, partition[(None, None, None, 0, index + 1)], fragment)
        return acc_vec, umma_cons_state

    @cute.jit
    def _consume_math_task(
        self,
        q_idx,
        kv_idx,
        q_stage_local,
        group,
        block_kv_val,
        m_coord,
        tCtAcc_base,
        tiled_copy,
        thr_copy,
        tTR_tAcc_base,
        tTR_rAcc,
        epi_sub_mn,
        kv_pipeline,
        kv_cons_state,
        umma_pipeline,
        umma_cons_state,
        sScales,
        w_cache,
        NUM_W_IN_REG: cutlass.Constexpr,
        sW,
        output_view,
        logits_stride0,
        out_row_off,
        mIndexCounts,
        mTileMeta,
        mPolicyValues,
        store_out_row_base,
        store_q_valid,
        store_causal_base,
        store_row_starts,
        store_row_ends,
        request_tiled: cutlass.Constexpr,
        paired_polls: cutlass.Constexpr,
        score_plane: cutlass.Constexpr = False,
        sStoreScores=None,
        store_stage=None,
    ):
        """Own one generation; request geometry is not sample-page indirection."""
        next_n = cutlass.const_expr(self.next_n)
        num_heads = cutlass.const_expr(self.num_heads)
        num_epi_subtiles = cutlass.const_expr(self.num_epi_subtiles)
        kv_pos = (kv_idx + group) * block_kv_val + m_coord
        if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
            # UMMA completion gives transitive visibility of matching KV.
            umma_pipeline.consumer_wait(umma_cons_state)
        elif cutlass.const_expr(paired_polls):
            # Keep both probes adjacent; their original tokens reach waits.
            kv_ready = kv_pipeline.consumer_try_wait(kv_cons_state)
            umma_ready = umma_pipeline.consumer_try_wait(umma_cons_state)
            kv_pipeline.consumer_wait(kv_cons_state, kv_ready)
            scale_val = self._math_scale_value(
                sScales, m_coord, kv_cons_state.index, group, request_tiled
            )
            umma_pipeline.consumer_wait(umma_cons_state, umma_ready)
        else:
            kv_pipeline.consumer_wait(kv_cons_state)
            scale_val = self._math_scale_value(
                sScales, m_coord, kv_cons_state.index, group, request_tiled
            )
            umma_pipeline.consumer_wait(umma_cons_state)
        if cutlass.const_expr(request_tiled):
            partition = tTR_tAcc_base[
                (None, None, None, None, None, umma_cons_state.index)
            ]
        else:
            tCtAcc = tCtAcc_base[(None, None, None, umma_cons_state.index)]
            tAcc = tCtAcc[((None, None), 0, 0)]
            partition = thr_copy.partition_S(cute.flat_divide(tAcc, epi_sub_mn))
        if cutlass.const_expr(self.early_tmem_copy):
            cute.copy(tiled_copy, partition[(None, None, None, 0, 0)], tTR_rAcc)
            if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                scale_val = self._math_scale_value(
                    sScales, m_coord, kv_cons_state.index, group, request_tiled
                )
            if cutlass.const_expr(not request_tiled):
                # Order scale LDS before producer reuse; early-copy policy
                # keeps first LDTM ahead of this shared fence/release.
                kv_cons_state = self._release_kv_after_scale_use(
                    kv_pipeline, kv_cons_state
                )
            cute.arch.fence_view_async_tmem_load()
        else:
            if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                scale_val = self._math_scale_value(
                    sScales, m_coord, kv_cons_state.index, group, request_tiled
                )
            if cutlass.const_expr(not request_tiled):
                # Default ordinary policy releases after scale LDS BEFORE
                # the first LDTM; request-tiled modes retain the late boundary.
                kv_cons_state = self._release_kv_after_scale_use(
                    kv_pipeline, kv_cons_state
                )
            cute.copy(tiled_copy, partition[(None, None, None, 0, 0)], tTR_rAcc)
            cute.arch.fence_view_async_tmem_load()
        sums: Any
        if cutlass.const_expr(request_tiled):
            if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                packed_zero = pack_f16x2(Float16(0.0), Float16(0.0))
                sums = (packed_zero, packed_zero)
            else:
                sums = (
                    cutlass.Float32(0.0),
                    cutlass.Float32(0.0),
                    cutlass.Float32(0.0),
                    cutlass.Float32(0.0),
                )
            subtile_n = cutlass.const_expr(self.N // num_epi_subtiles)
            for epi_subtile in cutlass.range_constexpr(num_epi_subtiles):
                acc_vec, umma_cons_state = self._load_math_fragment(
                    tiled_copy,
                    partition,
                    tTR_rAcc,
                    epi_subtile,
                    epi_subtile == num_epi_subtiles - 1,
                    umma_pipeline,
                    umma_cons_state,
                    score_plane,
                )
                sums = self._request_math_fragment(
                    acc_vec,
                    sums,
                    epi_subtile * subtile_n,
                    subtile_n,
                    next_n,
                    num_epi_subtiles,
                    num_heads,
                    NUM_W_IN_REG,
                    w_cache,
                    sW,
                    q_stage_local,
                    output_view,
                    mIndexCounts,
                    mTileMeta,
                    mPolicyValues,
                    q_idx,
                    store_out_row_base,
                    store_q_valid,
                    store_causal_base,
                    store_row_starts,
                    store_row_ends,
                    kv_pos,
                    cutlass.Boolean(True),
                    m_coord,
                    kv_pos,
                    scale_val,
                    score_plane,
                    sStoreScores,
                    store_stage,
                    group,
                )
            # All flattened subtiles/output stores still consume scale_val.
            # Releasing at the first load lets TMA overwrite a live generation.
            kv_cons_state = self._release_kv_after_scale_use(kv_pipeline, kv_cons_state)
        else:
            subtile_n = cutlass.const_expr(num_heads // num_epi_subtiles)
            if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                packed_zero = pack_f16x2(Float16(0.0), Float16(0.0))
            for t in cutlass.range_constexpr(next_n):
                if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                    sums = (packed_zero, packed_zero)
                else:
                    sums = (
                        cutlass.Float32(0.0),
                        cutlass.Float32(0.0),
                        cutlass.Float32(0.0),
                        cutlass.Float32(0.0),
                    )
                for i in cutlass.range_constexpr(num_epi_subtiles):
                    acc_vec, umma_cons_state = self._load_math_fragment(
                        tiled_copy,
                        partition,
                        tTR_rAcc,
                        t * num_epi_subtiles + i,
                        t == next_n - 1 and i == num_epi_subtiles - 1,
                        umma_pipeline,
                        umma_cons_state,
                    )
                    sums = self._ordinary_math_fragment(
                        acc_vec,
                        sums,
                        t,
                        i,
                        subtile_n,
                        NUM_W_IN_REG,
                        num_heads,
                        w_cache,
                        sW,
                        q_stage_local,
                    )
                self._store_ordinary_math_result(
                    sums,
                    scale_val,
                    output_view,
                    out_row_off,
                    logits_stride0,
                    q_idx,
                    t,
                    next_n,
                    kv_pos,
                    mTileMeta,
                )
        return kv_cons_state, umma_cons_state

    @cute.jit
    def _run_math_consumer(
        self,
        mContextLens,
        batch_size,
        block_kv_val,
        end_q_idx,
        end_kv_idx,
        next_q_idx,
        next_kv_idx,
        next_num_kv,
        q_idx,
        has_work,
        group,
        tCtAcc_base,
        copy_atom_t2r,
        epi_sub_mn,
        local_tidx,
        cC,
        q_pipeline,
        q_cons_state,
        kv_pipeline,
        kv_cons_state,
        umma_pipeline,
        umma_cons_state,
        sScales,
        sW,
        mLogits,
        request_tiled: cutlass.Constexpr,
        cache_regs: cutlass.Constexpr,
        two_level: cutlass.Constexpr,
        paired_polls: cutlass.Constexpr,
        mIndexCounts=None,
        mTileMeta=None,
        mPolicyValues=None,
        score_plane: cutlass.Constexpr = False,
        sStoreScores=None,
        store_pipeline=None,
    ):
        """Common Q/task owner; request-tiled stays flat, ordinary gated."""
        next_n = cutlass.const_expr(self.next_n)
        num_heads = cutlass.const_expr(self.num_heads)
        if cutlass.const_expr(score_plane):
            store_prod_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 2
            )
        if cutlass.const_expr(request_tiled):
            tiled_copy, tTR_tAcc_base, tTR_cC, tTR_rAcc = (
                self._make_math_tmem_copy_and_partition(
                    tCtAcc_base, epi_sub_mn, copy_atom_t2r, local_tidx, cC
                )
            )
            thr_copy = None
        else:
            tAcc_ref = tCtAcc_base[(None, None, None, 0)][((None, None), 0, 0)]
            tAcc_ref_epi = cute.flat_divide(tAcc_ref, epi_sub_mn)
            tiled_copy = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_ref_epi[(None, None, 0, 0)]
            )
            thr_copy = tiled_copy.get_slice(local_tidx)
            tTR_cC = thr_copy.partition_D(cC)
            tTR_rAcc = cute.make_fragment_like(tTR_cC, self.acc_dtype)
            tTR_tAcc_base = None
        m_coord = tTR_cC[0][0]
        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
            MAX_NUM_W_IN_REG = 64 if next_n <= 3 else 48
            MAX_W_CACHE_ELEMS = 2 * cache_regs
        else:
            MAX_NUM_W_IN_REG = 64 if next_n == 1 else 40 if next_n >= 4 else 52
            MAX_W_CACHE_ELEMS = cache_regs
        # Bound total per-row cache footprint as well as per-slot head count.
        # One constexpr expression keeps range_constexpr bounds static.
        NUM_W_IN_REG = max(
            _EPI_UNROLL,
            min(
                MAX_NUM_W_IN_REG,
                num_heads,
                ((MAX_W_CACHE_ELEMS // next_n) // _EPI_UNROLL) * _EPI_UNROLL,
            ),
        )
        if cutlass.const_expr(score_plane and self.N <= 128):
            # Preserve the candidate role-local full cache for tuned N<=128;
            # wider legal tiles keep the same bounded per-row cache above.
            NUM_W_IN_REG = num_heads
        assert NUM_W_IN_REG % _EPI_UNROLL == 0
        if cutlass.const_expr(request_tiled):
            w_cache = cute.make_rmem_tensor(
                cute.make_layout((next_n, NUM_W_IN_REG), stride=(NUM_W_IN_REG, 1)),
                self.epi_dtype,
            )
        else:
            w_cache = cute.make_rmem_tensor(NUM_W_IN_REG * next_n, self.epi_dtype)
        q_stage_local = cutlass.Int32(0)
        if cutlass.const_expr(request_tiled):
            store_out_row_base = cutlass.Int32(0)
            store_q_valid = cutlass.Int32(0)
            store_causal_base = cutlass.Int32(0)
            if cutlass.const_expr(not score_plane):
                store_row_starts = cute.make_rmem_tensor(next_n, cutlass.Int32)
                store_row_ends = cute.make_rmem_tensor(next_n, cutlass.Int32)
            else:
                store_row_starts = None
                store_row_ends = None
            output_view = mLogits
            logits_stride0 = 0
            out_row_off = 0
        else:
            store_out_row_base = None
            store_q_valid = None
            store_causal_base = None
            store_row_starts = None
            store_row_ends = None
            if cutlass.const_expr(self.use_flat_logits_view):
                # Flat logits view + carried row offset: the store address
                # is base + out_row*stride0 + kv_pos, but out_row only
                # changes on q-change. Recomputing the row term there (cold
                # path) leaves the hot path a single add -- the wide-multiply
                # address chain otherwise re-materializes every task because
                # the while-loop control flow hides the invariance.
                # Gated off on Rubin (sm_107): there the backend schedules the
                # UMMA consumer-release late in the math loop, throttling the
                # MMA pipe per KV tile -- regresses, growing with ctx.
                # Pass only the selected output_view to the store helper:
                # referencing mLogits from the untraced arm changes loop-region
                # captures (and the SASS schedule), even without changing the
                # selected store. Keep that tensor out of the gated store body.
                # Size by pitched span (rows * stride0), not visible width:
                # stores cover aligned padding past max_context_len, and the
                # host allocates the full pitch.
                output_offset_dtype = cutlass.Int32
                if cutlass.const_expr(self.output_index_bits == 64):
                    output_offset_dtype = cutlass.Int64
                logits_stride0 = output_offset_dtype(mLogits.layout.stride[0])
                output_view = cute.make_tensor(
                    mLogits.iterator,
                    cute.make_layout(
                        output_offset_dtype(cute.size(mLogits, mode=[0]))
                        * logits_stride0
                    ),
                )
                out_row_off = output_offset_dtype(0)
            else:
                output_view = mLogits
                logits_stride0 = 0
                out_row_off = 0
        while has_work:
            q_idx_old = q_idx
            q_idx = next_q_idx
            kv_idx = next_kv_idx
            num_kv = next_num_kv
            if cutlass.const_expr(score_plane):
                # Acquire before retiring/waiting Q, matching the existing
                # score-plane generation and avoiding a hidden policy release.
                store_pipeline.producer_acquire(store_prod_state)
            if q_idx != q_idx_old:
                if q_idx_old < batch_size:
                    q_pipeline.consumer_release(q_cons_state)
                    q_cons_state.advance()
                q_pipeline.consumer_wait(q_cons_state)
                q_stage_local = q_cons_state.index
                if cutlass.const_expr(request_tiled):
                    store_out_row_base = mTileMeta[(q_idx, 0)]
                    store_q_valid = mTileMeta[(q_idx, 2)]
                    store_causal_base = mTileMeta[(q_idx, 3)]
                    if cutlass.const_expr(not score_plane):
                        for row in cutlass.range_constexpr(next_n):
                            store_row_starts[row] = mTileMeta[(q_idx, 4 + row)]
                            store_row_ends[row] = mTileMeta[(q_idx, 4 + next_n + row)]
                elif cutlass.const_expr(self.use_flat_logits_view):
                    # Promote before either product; widening an already
                    # wrapped Int32 row term cannot recover its address.
                    out_row_off = (output_offset_dtype(q_idx) * next_n) * logits_stride0
                for t_i in cutlass.range_constexpr(next_n):
                    for w_j in cutlass.range_constexpr(NUM_W_IN_REG):
                        if cutlass.const_expr(request_tiled):
                            w_cache[(t_i, w_j)] = sW[(t_i, w_j, q_stage_local)]
                        else:
                            w_cache[t_i * NUM_W_IN_REG + w_j] = sW[
                                (t_i * num_heads + w_j, q_stage_local)
                            ]
            if cutlass.const_expr(request_tiled):
                store_stage = None
                if cutlass.const_expr(score_plane):
                    store_stage = store_prod_state.index
                kv_cons_state, umma_cons_state = self._consume_math_task(
                    q_idx,
                    kv_idx,
                    q_stage_local,
                    group,
                    block_kv_val,
                    m_coord,
                    tCtAcc_base,
                    tiled_copy,
                    thr_copy,
                    tTR_tAcc_base,
                    tTR_rAcc,
                    epi_sub_mn,
                    kv_pipeline,
                    kv_cons_state,
                    umma_pipeline,
                    umma_cons_state,
                    sScales,
                    w_cache,
                    NUM_W_IN_REG,
                    sW,
                    output_view,
                    logits_stride0,
                    out_row_off,
                    mIndexCounts,
                    mTileMeta,
                    mPolicyValues,
                    store_out_row_base,
                    store_q_valid,
                    store_causal_base,
                    store_row_starts,
                    store_row_ends,
                    request_tiled,
                    paired_polls,
                    score_plane,
                    sStoreScores,
                    store_stage,
                )
                if cutlass.const_expr(score_plane):
                    # The task's final shared fence covers its scale uses and
                    # every score-plane store. No shared data is written
                    # between that fence and this generation's publication.
                    store_pipeline.producer_commit(store_prod_state)
                    store_prod_state.advance()
                kv_idx = kv_idx + 2
            else:
                # Inner hot KV loop within this Q: the cold Q-change weight
                # preload and Q-rollover context reload previously sat inline
                # in the flat loop, costing taken-branch fetch bubbles on each
                # hot iteration. Replay exactly the original task sequence:
                # process (q, kv); advance kv by W; roll over to (q+1, 0) when
                # kv >= num_kv; stop at the exact (end_q, end_kv) pair.
                # Truth table after kv += W (W = NUM_MATH_WG):
                #   kv < num_kv, (q, kv) != end -> True: more KV in this Q
                #   kv < num_kv, (q, kv) == end -> False: global end mid-Q
                #   kv >= num_kv, any          -> False: outer rollover next
                # The first iteration is unconditional: the outer loop starts
                # only when the committed (q, kv) differs from the end pair.
                # Gate use_two_level_task_loop=False (Rubin next_n >= 4 auto):
                # forcing kv_go=False after the body makes this single-trip,
                # restoring the original per-task Q-change check without a
                # duplicate skeleton. SAMPLE above stays a flat task loop.
                kv_go = cutlass.Boolean(True)
                while kv_go:
                    kv_cons_state, umma_cons_state = self._consume_math_task(
                        q_idx,
                        kv_idx,
                        q_stage_local,
                        group,
                        block_kv_val,
                        m_coord,
                        tCtAcc_base,
                        tiled_copy,
                        thr_copy,
                        tTR_tAcc_base,
                        tTR_rAcc,
                        epi_sub_mn,
                        kv_pipeline,
                        kv_cons_state,
                        umma_pipeline,
                        umma_cons_state,
                        sScales,
                        w_cache,
                        NUM_W_IN_REG,
                        sW,
                        output_view,
                        logits_stride0,
                        out_row_off,
                        mIndexCounts,
                        mTileMeta,
                        mPolicyValues,
                        store_out_row_base,
                        store_q_valid,
                        store_causal_base,
                        store_row_starts,
                        store_row_ends,
                        request_tiled,
                        paired_polls,
                    )
                    kv_idx = kv_idx + 2
                    kv_go = (kv_idx < num_kv) & (
                        (q_idx != end_q_idx) | (kv_idx != end_kv_idx)
                    )
                    if cutlass.const_expr(not two_level):
                        kv_go = cutlass.Boolean(False)
            if kv_idx >= num_kv:
                next_q_idx = q_idx + 1
                next_kv_idx = 0
                if next_q_idx < batch_size:
                    next_num_kv = (
                        mContextLens[next_q_idx] + block_kv_val - 1
                    ) // block_kv_val
                # Every role skips the same zero rows without crossing its
                # exact CTA end pair or reading context past batch_size.
                while (
                    (next_num_kv == 0)
                    & (next_q_idx < batch_size)
                    & ((next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx))
                ):
                    next_q_idx = next_q_idx + 1
                    if next_q_idx < batch_size:
                        next_num_kv = (
                            mContextLens[next_q_idx] + block_kv_val - 1
                        ) // block_kv_val
            else:
                next_q_idx = q_idx
                next_kv_idx = kv_idx
            has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)
        if q_idx < batch_size:
            q_pipeline.consumer_release(q_cons_state)
            q_cons_state.advance()
        if cutlass.const_expr(score_plane):
            store_pipeline.producer_tail(store_prod_state)
            # Carry the new mutable state across the outer helper boundary,
            # like the existing Q/KV/UMMA states, rather than leaking an alias.
            return q_cons_state, kv_cons_state, umma_cons_state, store_prod_state
        return q_cons_state, kv_cons_state, umma_cons_state

    @cute.jit
    def _run_math_role(
        self,
        mContextLens,
        batch_size,
        block_kv_val,
        end_q_idx,
        end_kv_idx,
        next_q_idx,
        next_kv_idx,
        next_num_kv,
        q_idx,
        has_work,
        tidx,
        warpgroup_idx,
        tmem,
        num_tmem_alloc_cols_total,
        tCtAcc_fake_staged,
        group_offset,
        epi_sub_mn,
        copy_atom_t2r,
        mLogits,
        sW,
        q_pipeline,
        q_cons_state,
        kv_pipeline_0,
        umma_pipeline_0,
        sScales_0,
        request_tiled: cutlass.Constexpr,
        cache_regs: cutlass.Constexpr,
        kv_cons_state_0=None,
        umma_cons_state_0=None,
        kv_pipeline_1=None,
        umma_pipeline_1=None,
        sScales_1=None,
        kv_cons_state_1=None,
        umma_cons_state_1=None,
        mIndexCounts=None,
        mTileMeta=None,
        mPolicyValues=None,
        score_plane: cutlass.Constexpr = False,
        sStoreScores=None,
        store_pipeline=None,
    ):
        """One TMEM allocation/consumer/permit/free owner with static sinks."""
        tmem.allocate(num_tmem_alloc_cols_total)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        if cutlass.const_expr(request_tiled):
            math_tmem_ptr = tmem_ptr + warpgroup_idx * group_offset
            if cutlass.const_expr(self.acc_dtype == Float16):
                # Dynamic group selection retains whole-panel align8 for
                # packed-half LDTM; ordinary retains its static plane views.
                math_tmem_ptr = cute.make_ptr(
                    self.acc_dtype,
                    math_tmem_ptr.toint(),
                    cute.AddressSpace.tmem,
                    assumed_align=8,
                )
            tCtAcc_base = cute.make_tensor(math_tmem_ptr, tCtAcc_fake_staged.layout)
            local_tidx = tidx % 128
            cC = cute.make_identity_tensor(epi_sub_mn)
            kv_cons_state_0 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_kv_stages
            )
            umma_cons_state_0 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_umma_stages
            )
            math_states = self._run_math_consumer(
                mContextLens,
                batch_size,
                block_kv_val,
                end_q_idx,
                end_kv_idx,
                next_q_idx,
                next_kv_idx,
                next_num_kv,
                q_idx,
                has_work,
                warpgroup_idx,
                tCtAcc_base,
                copy_atom_t2r,
                epi_sub_mn,
                local_tidx,
                cC,
                q_pipeline,
                q_cons_state,
                kv_pipeline_0,
                kv_cons_state_0,
                umma_pipeline_0,
                umma_cons_state_0,
                sScales_0,
                sW,
                mLogits,
                True,
                cache_regs,
                False,
                True,
                mIndexCounts,
                mTileMeta,
                mPolicyValues,
                score_plane,
                sStoreScores,
                store_pipeline,
            )
            if cutlass.const_expr(score_plane):
                (
                    q_cons_state,
                    kv_cons_state_0,
                    umma_cons_state_0,
                    store_prod_state,
                ) = math_states
            else:
                q_cons_state, kv_cons_state_0, umma_cons_state_0 = math_states
        else:
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            tCtAcc_base_1 = cute.make_tensor(
                tmem_ptr + group_offset, tCtAcc_fake_staged.layout
            )
            local_tidx = tidx % 128
            cC = cute.make_identity_tensor(epi_sub_mn)
            if warpgroup_idx == 0:
                q_cons_state, kv_cons_state_0, umma_cons_state_0 = (
                    self._run_math_consumer(
                        mContextLens,
                        batch_size,
                        block_kv_val,
                        end_q_idx,
                        end_kv_idx,
                        next_q_idx,
                        next_kv_idx,
                        next_num_kv,
                        q_idx,
                        has_work,
                        0,
                        tCtAcc_base_0,
                        copy_atom_t2r,
                        epi_sub_mn,
                        local_tidx,
                        cC,
                        q_pipeline,
                        q_cons_state,
                        kv_pipeline_0,
                        kv_cons_state_0,
                        umma_pipeline_0,
                        umma_cons_state_0,
                        sScales_0,
                        sW,
                        mLogits,
                        False,
                        cache_regs,
                        self.use_two_level_task_loop,
                        self.use_paired_pipeline_polls,
                        mTileMeta=mTileMeta,
                    )
                )
            else:
                q_cons_state, kv_cons_state_1, umma_cons_state_1 = (
                    self._run_math_consumer(
                        mContextLens,
                        batch_size,
                        block_kv_val,
                        end_q_idx,
                        end_kv_idx,
                        next_q_idx,
                        next_kv_idx,
                        next_num_kv,
                        q_idx,
                        has_work,
                        1,
                        tCtAcc_base_1,
                        copy_atom_t2r,
                        epi_sub_mn,
                        local_tidx,
                        cC,
                        q_pipeline,
                        q_cons_state,
                        kv_pipeline_1,
                        kv_cons_state_1,
                        umma_pipeline_1,
                        umma_cons_state_1,
                        sScales_1,
                        sW,
                        mLogits,
                        False,
                        cache_regs,
                        self.use_two_level_task_loop,
                        self.use_paired_pipeline_polls,
                        mTileMeta=mTileMeta,
                    )
                )
        tmem.relinquish_alloc_permit()
        if cutlass.const_expr(request_tiled):
            tmem.free(tmem_ptr)
            if cutlass.const_expr(score_plane):
                return (
                    q_cons_state,
                    kv_cons_state_0,
                    umma_cons_state_0,
                    store_prod_state,
                )
            return q_cons_state, kv_cons_state_0, umma_cons_state_0
        else:
            tmem.free(tmem_ptr, num_tmem_alloc_cols_total)
            return (
                q_cons_state,
                kv_cons_state_0,
                umma_cons_state_0,
                kv_cons_state_1,
                umma_cons_state_1,
            )

    @cute.jit
    def _run_store_role(
        self,
        NUM_MATH_WG,
        NUM_STORE_WARPS,
        batch_size,
        block_kv_val,
        end_kv_idx,
        end_q_idx,
        has_work,
        mContextLens,
        mIndexCounts,
        mLogits,
        mPolicyValues,
        mTileMeta,
        next_kv_idx,
        next_n,
        next_num_kv,
        next_q_idx,
        q_idx,
        sStoreScores_0,
        sStoreScores_1,
        store_pipeline_0,
        store_pipeline_1,
    ):
        """Run one segmented candidate publisher warp."""
        if cutlass.const_expr(self.has_store_warp):
            cute.arch.setmaxregister_increase(self.store_wg_registers)
            store_cons_state_0 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 2
            )
            store_cons_state_1 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 2
            )
            store_logits_addr = self._store_uniform_i64(mLogits.iterator.toint())
            store_logits_stride = self._store_uniform_i64(mLogits.layout.stride[0])
            store_warp = cute.arch.make_warp_uniform(
                cute.arch.warp_idx() - cutlass.Int32(12)
            )
            store_lane = cute.arch.lane_idx()
            lane_private_segment_base = store_lane * cutlass.Int32(
                self.num_count_segments // 32
            )
            num_q_groups = (next_n + NUM_STORE_WARPS - 1) // NUM_STORE_WARPS
            num_lane_cursors = (
                self.split_kv
                if cutlass.const_expr(self.split_kv > 1)
                else self.num_count_segments // 32
            )
            num_private_cursors = (
                num_lane_cursors
                if cutlass.const_expr(self.count_segment_span == 1)
                else 1
                if cutlass.const_expr(self.count_segment_span == 128)
                else 4
            )
            private_cursors = [
                [cutlass.Int32(0) for _ in range(num_private_cursors)]
                for _ in range(num_q_groups)
            ]
            cursor_q_idx = cutlass.Int32(-1)
            cached_out_row_base = cutlass.Int32(0)
            cached_q_valid = cutlass.Int32(0)
            cached_causal_base = cutlass.Int32(0)
            cached_row_logits_addrs = [cutlass.Int64(0) for _ in range(num_q_groups)]
            cached_row_logical_ends = [cutlass.Int32(0) for _ in range(num_q_groups)]
            if cutlass.const_expr(not self.candidate_uses_causal_bounds):
                cached_row_logical_starts = [
                    cutlass.Int32(0) for _ in range(num_q_groups)
                ]
            cached_thresholds = [cutlass.Float32(0.0) for _ in range(num_q_groups)]

            while has_work:
                q_idx = next_q_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv

                if q_idx != cursor_q_idx:
                    cached_out_row_base = mTileMeta[(q_idx, 0)]
                    cached_q_valid = mTileMeta[(q_idx, 2)]
                    cached_causal_base = mTileMeta[(q_idx, 3)]
                    for q_group in cutlass.range_constexpr(num_q_groups):
                        row = store_warp + q_group * NUM_STORE_WARPS
                        if row < cached_q_valid:
                            out_row = cached_out_row_base + row
                            cached_row_logits_addrs[q_group] = (
                                store_logits_addr
                                + cutlass.Int64(
                                    cutlass.Uint64(cutlass.Uint32(out_row))
                                    * cutlass.Uint64(store_logits_stride)
                                )
                                * cutlass.Int64(8)
                            )
                            if cutlass.const_expr(self.candidate_uses_causal_bounds):
                                cached_row_logical_ends[q_group] = (
                                    cached_causal_base + row + 1
                                )
                            else:
                                # Explicit metadata has no causal-base origin;
                                # each row owns its own inclusive/exclusive bounds.
                                cached_row_logical_starts[q_group] = mTileMeta[
                                    (q_idx, 4 + row)
                                ]
                                cached_row_logical_ends[q_group] = mTileMeta[
                                    (q_idx, 4 + next_n + row)
                                ]
                            cached_thresholds[q_group] = self._store_policy_f32(
                                mPolicyValues[out_row]
                            )
                        for cursor_idx in cutlass.range_constexpr(num_private_cursors):
                            private_cursors[q_group][cursor_idx] = cutlass.Int32(0)
                    cursor_q_idx = q_idx

                out_row_base = cached_out_row_base
                q_valid = cached_q_valid
                kv_base = store_lane * 4
                logical_base_0 = kv_idx * block_kv_val + kv_base
                logical_base_1 = (kv_idx + 1) * block_kv_val + kv_base
                split_idx = cutlass.Int32(0)
                split_end_kv = num_kv
                if cutlass.const_expr(self.split_kv > 1):
                    num_kv_pairs = (num_kv + NUM_MATH_WG - 1) // NUM_MATH_WG
                    active_splits = min(num_kv_pairs, cutlass.Int32(self.split_kv))
                    kv_pair_idx = kv_idx // NUM_MATH_WG
                    split_idx = (kv_pair_idx * active_splits) // num_kv_pairs
                    split_end_pair = cute.ceil_div(
                        (split_idx + 1) * num_kv_pairs, active_splits
                    )
                    split_end_kv = min(split_end_pair * NUM_MATH_WG, num_kv)
                store_ready_0 = store_pipeline_0.consumer_try_wait(store_cons_state_0)
                store_ready_1 = store_pipeline_1.consumer_try_wait(store_cons_state_1)
                store_pipeline_0.consumer_wait(store_cons_state_0, store_ready_0)
                if store_warp >= q_valid:
                    store_pipeline_1.consumer_wait(store_cons_state_1, store_ready_1)
                needs_bounds = (q_valid < num_q_groups * NUM_STORE_WARPS) | (
                    kv_idx + NUM_MATH_WG >= num_kv
                )
                if cutlass.const_expr(not self.candidate_uses_causal_bounds):
                    needs_bounds = cutlass.Boolean(True)
                if needs_bounds:
                    # Peel partial-Q tiles and the final KV pair so the
                    # causal steady-state path carries no row/logical predicates.
                    # Explicit bounds can cut any interior tile, not just tails.
                    for q_group in cutlass.range_constexpr(num_q_groups):
                        row = store_warp + q_group * NUM_STORE_WARPS
                        if (
                            cutlass.const_expr(next_n >= NUM_STORE_WARPS)
                            or row < q_valid
                        ):
                            row_logits_addr = cached_row_logits_addrs[q_group]
                            row_logical_end = cached_row_logical_ends[q_group]
                            row_logical_start = cutlass.Int32(0)
                            if cutlass.const_expr(
                                not self.candidate_uses_causal_bounds
                            ):
                                row_logical_start = cached_row_logical_starts[q_group]
                            threshold = cached_thresholds[q_group]
                            # Tail warps still consume/release both score stages,
                            # but their rounded row slots have no score storage.
                            # Keep the logical row for publication predicates.
                            score_row = min(row, max(q_valid - 1, cutlass.Int32(0)))
                            scores_0 = self._store_load_v4(
                                sStoreScores_0,
                                score_row,
                                kv_base,
                                store_cons_state_0.index,
                            )
                            if cutlass.const_expr(q_group == 0):
                                if store_warp < q_valid:
                                    store_pipeline_1.consumer_wait(
                                        store_cons_state_1, store_ready_1
                                    )
                            scores_1 = self._store_load_v4(
                                sStoreScores_1,
                                score_row,
                                kv_base,
                                store_cons_state_1.index,
                            )
                            if cutlass.const_expr(
                                next_n >= NUM_STORE_WARPS
                                and q_group == num_q_groups - 1
                            ):
                                # Both score planes are now warp-local registers;
                                # let the math warpgroups reuse their SMEM stages
                                # while this final row group is compacted.
                                store_pipeline_0.consumer_release(store_cons_state_0)
                                store_cons_state_0.advance()
                                store_pipeline_1.consumer_release(store_cons_state_1)
                                store_cons_state_1.advance()
                            if cutlass.const_expr(self.count_segment_span == 1):
                                if cutlass.const_expr(self.split_kv > 1):
                                    split_cursor = private_cursors[q_group][0]
                                    for candidate_split in cutlass.range_constexpr(
                                        1, self.split_kv
                                    ):
                                        if split_idx == candidate_split:
                                            split_cursor = private_cursors[q_group][
                                                candidate_split
                                            ]
                                    split_segment = split_idx * 32 + store_lane
                                    for vec_idx in cutlass.range_constexpr(4):
                                        split_cursor = (
                                            self._store_candidate_lane_row_pair(  # type: ignore[attr-defined]
                                                row_logits_addr,
                                                q_valid,
                                                row,
                                                row_logical_end,
                                                row_logical_start,
                                                logical_base_0 + vec_idx,
                                                logical_base_1 + vec_idx,
                                                threshold,
                                                split_cursor,
                                                split_segment,
                                                scores_0[vec_idx],
                                                scores_1[vec_idx],
                                            )
                                        )
                                    for candidate_split in cutlass.range_constexpr(
                                        self.split_kv
                                    ):
                                        if split_idx == candidate_split:
                                            private_cursors[q_group][
                                                candidate_split
                                            ] = split_cursor
                                else:
                                    for vec_idx in cutlass.range_constexpr(4):
                                        cursor_idx = (
                                            0
                                            if cutlass.const_expr(
                                                self.num_count_segments == 32
                                            )
                                            else vec_idx // 2
                                            if cutlass.const_expr(
                                                self.num_count_segments == 64
                                            )
                                            else vec_idx
                                        )
                                        private_cursors[q_group][cursor_idx] = (
                                            self._store_candidate_lane_row_pair(  # type: ignore[attr-defined]
                                                row_logits_addr,
                                                q_valid,
                                                row,
                                                row_logical_end,
                                                row_logical_start,
                                                logical_base_0 + vec_idx,
                                                logical_base_1 + vec_idx,
                                                threshold,
                                                private_cursors[q_group][cursor_idx],
                                                lane_private_segment_base + cursor_idx,
                                                scores_0[vec_idx],
                                                scores_1[vec_idx],
                                            )
                                        )
                            elif cutlass.const_expr(self.count_segment_span == 128):
                                pooled_cursor = private_cursors[q_group][0]
                                for vec_idx in cutlass.range_constexpr(4):
                                    pooled_cursor = self._store_candidate_warp_row_pair(  # type: ignore[attr-defined]
                                        row_logits_addr,
                                        q_valid,
                                        row,
                                        row_logical_end,
                                        row_logical_start,
                                        logical_base_0 + vec_idx,
                                        logical_base_1 + vec_idx,
                                        threshold,
                                        pooled_cursor,
                                        split_idx,
                                        scores_0[vec_idx],
                                        scores_1[vec_idx],
                                    )
                                private_cursors[q_group][0] = pooled_cursor
                            else:
                                for vec_idx in cutlass.range_constexpr(4):
                                    private_cursors[q_group][vec_idx] = (
                                        self._store_candidate_warp_row_pair(  # type: ignore[attr-defined]
                                            row_logits_addr,
                                            q_valid,
                                            row,
                                            row_logical_end,
                                            row_logical_start,
                                            logical_base_0 + vec_idx,
                                            logical_base_1 + vec_idx,
                                            threshold,
                                            private_cursors[q_group][vec_idx],
                                            split_idx * 4 + vec_idx,
                                            scores_0[vec_idx],
                                            scores_1[vec_idx],
                                        )
                                    )
                else:
                    for q_group in cutlass.range_constexpr(num_q_groups):
                        row = store_warp + q_group * NUM_STORE_WARPS
                        row_logits_addr = cached_row_logits_addrs[q_group]
                        threshold = cached_thresholds[q_group]
                        scores_0 = self._store_load_v4(
                            sStoreScores_0,
                            row,
                            kv_base,
                            store_cons_state_0.index,
                        )
                        if cutlass.const_expr(q_group == 0):
                            store_pipeline_1.consumer_wait(
                                store_cons_state_1, store_ready_1
                            )
                        scores_1 = self._store_load_v4(
                            sStoreScores_1,
                            row,
                            kv_base,
                            store_cons_state_1.index,
                        )
                        if cutlass.const_expr(
                            next_n >= NUM_STORE_WARPS and q_group == num_q_groups - 1
                        ):
                            # Both score planes are now warp-local registers;
                            # let the math warpgroups reuse their SMEM stages
                            # while this final row group is compacted.
                            store_pipeline_0.consumer_release(store_cons_state_0)
                            store_cons_state_0.advance()
                            store_pipeline_1.consumer_release(store_cons_state_1)
                            store_cons_state_1.advance()
                        if cutlass.const_expr(self.count_segment_span == 1):
                            if cutlass.const_expr(self.split_kv > 1):
                                split_cursor = private_cursors[q_group][0]
                                for candidate_split in cutlass.range_constexpr(
                                    1, self.split_kv
                                ):
                                    if split_idx == candidate_split:
                                        split_cursor = private_cursors[q_group][
                                            candidate_split
                                        ]
                                split_segment = split_idx * 32 + store_lane
                                for vec_idx in cutlass.range_constexpr(4):
                                    split_cursor = (
                                        self._store_candidate_lane_row_pair_full(  # type: ignore[attr-defined]
                                            row_logits_addr,
                                            logical_base_0 + vec_idx,
                                            logical_base_1 + vec_idx,
                                            threshold,
                                            split_cursor,
                                            split_segment,
                                            scores_0[vec_idx],
                                            scores_1[vec_idx],
                                        )
                                    )
                                for candidate_split in cutlass.range_constexpr(
                                    self.split_kv
                                ):
                                    if split_idx == candidate_split:
                                        private_cursors[q_group][candidate_split] = (
                                            split_cursor
                                        )
                            else:
                                for vec_idx in cutlass.range_constexpr(4):
                                    cursor_idx = (
                                        0
                                        if cutlass.const_expr(
                                            self.num_count_segments == 32
                                        )
                                        else vec_idx // 2
                                        if cutlass.const_expr(
                                            self.num_count_segments == 64
                                        )
                                        else vec_idx
                                    )
                                    private_cursors[q_group][cursor_idx] = (
                                        self._store_candidate_lane_row_pair_full(  # type: ignore[attr-defined]
                                            row_logits_addr,
                                            logical_base_0 + vec_idx,
                                            logical_base_1 + vec_idx,
                                            threshold,
                                            private_cursors[q_group][cursor_idx],
                                            lane_private_segment_base + cursor_idx,
                                            scores_0[vec_idx],
                                            scores_1[vec_idx],
                                        )
                                    )
                        elif cutlass.const_expr(self.count_segment_span == 128):
                            pooled_cursor = private_cursors[q_group][0]
                            for vec_idx in cutlass.range_constexpr(4):
                                pooled_cursor = (
                                    self._store_candidate_warp_row_pair_full(  # type: ignore[attr-defined]
                                        row_logits_addr,
                                        logical_base_0 + vec_idx,
                                        logical_base_1 + vec_idx,
                                        threshold,
                                        pooled_cursor,
                                        split_idx,
                                        scores_0[vec_idx],
                                        scores_1[vec_idx],
                                    )
                                )
                            private_cursors[q_group][0] = pooled_cursor
                        else:
                            for vec_idx in cutlass.range_constexpr(4):
                                private_cursors[q_group][vec_idx] = (
                                    self._store_candidate_warp_row_pair_full(  # type: ignore[attr-defined]
                                        row_logits_addr,
                                        logical_base_0 + vec_idx,
                                        logical_base_1 + vec_idx,
                                        threshold,
                                        private_cursors[q_group][vec_idx],
                                        split_idx * 4 + vec_idx,
                                        scores_0[vec_idx],
                                        scores_1[vec_idx],
                                    )
                                )

                publish_split = kv_idx + NUM_MATH_WG >= num_kv
                if cutlass.const_expr(self.split_kv > 1):
                    publish_split = kv_idx + NUM_MATH_WG >= split_end_kv
                if publish_split:
                    for q_group in cutlass.range_constexpr(num_q_groups):
                        row = store_warp + q_group * NUM_STORE_WARPS
                        if cutlass.const_expr(self.count_segment_span == 1):
                            publish_count = row < q_valid
                        else:
                            publish_count = (store_lane == 0) & (row < q_valid)
                        if publish_count:
                            out_row = out_row_base + row
                            if cutlass.const_expr(self.count_segment_span == 1):
                                if cutlass.const_expr(self.split_kv > 1):
                                    for candidate_split in cutlass.range_constexpr(
                                        self.split_kv
                                    ):
                                        if split_idx == candidate_split:
                                            mIndexCounts[
                                                (
                                                    out_row,
                                                    candidate_split * 32
                                                    + store_lane
                                                    + 1,
                                                )
                                            ] = private_cursors[q_group][
                                                candidate_split
                                            ]
                                else:
                                    num_lane_segments = (
                                        1
                                        if cutlass.const_expr(
                                            self.num_count_segments == 32
                                        )
                                        else 2
                                        if cutlass.const_expr(
                                            self.num_count_segments == 64
                                        )
                                        else 4
                                    )
                                    for vec_idx in cutlass.range_constexpr(
                                        num_lane_segments
                                    ):
                                        mIndexCounts[
                                            (
                                                out_row,
                                                lane_private_segment_base + vec_idx + 1,
                                            )
                                        ] = private_cursors[q_group][vec_idx]
                            elif cutlass.const_expr(self.count_segment_span == 128):
                                mIndexCounts[(out_row, split_idx + 1)] = (
                                    private_cursors[q_group][0]
                                )
                            else:
                                for vec_idx in cutlass.range_constexpr(4):
                                    mIndexCounts[
                                        (out_row, split_idx * 4 + vec_idx + 1)
                                    ] = private_cursors[q_group][vec_idx]
                        if cutlass.const_expr(self.count_segment_span == 128):
                            private_cursors[q_group][0] = cutlass.Int32(0)
                        elif cutlass.const_expr(self.count_segment_span != 1):
                            # The cursor is warp-uniform state. Every lane must
                            # reset it before this persistent CTA accepts a new
                            # split work item; lane 0 alone publishes the count.
                            for vec_idx in cutlass.range_constexpr(4):
                                private_cursors[q_group][vec_idx] = cutlass.Int32(0)
                if cutlass.const_expr(next_n < NUM_STORE_WARPS):
                    store_pipeline_0.consumer_release(store_cons_state_0)
                    store_cons_state_0.advance()
                    store_pipeline_1.consumer_release(store_cons_state_1)
                    store_cons_state_1.advance()

                next_kv_idx = kv_idx + NUM_MATH_WG
                if next_kv_idx >= num_kv:
                    next_q_idx = q_idx + 1
                    next_kv_idx = 0
                    if next_q_idx < batch_size:
                        next_num_kv = (
                            mContextLens[next_q_idx] + block_kv_val - 1
                        ) // block_kv_val
                    while (
                        (next_num_kv == 0)
                        & (next_q_idx < batch_size)
                        & ((next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx))
                    ):
                        next_q_idx = next_q_idx + 1
                        if next_q_idx < batch_size:
                            next_num_kv = (
                                mContextLens[next_q_idx] + block_kv_val - 1
                            ) // block_kv_val
                has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)


class FP8MQALogitsKernel(_FP8MQAEngineMixin):
    """FP8 paged MQA logits kernel for Blackwell (SM100/SM103) and Rubin (SM107).

    Each CTA processes a range of (q_idx, kv_split) pairs.
    A split = 2 consecutive KV blocks within a sequence (one per warp group).
    Q is shared between warp groups and reloaded when q_idx changes.
    """

    def __init__(
        self,
        block_kv: int = 128,
        phys_block_kv: int = 128,
        num_heads: int = 64,
        head_dim: int = 128,
        next_n: int = 1,
        num_sms: int = 148,
        remove_kv_wait_in_epilogue: bool = False,
        early_tmem_copy: bool = False,
        smem_subpartition_opt: bool = False,
        max_kv_pipeline: bool = False,
        max_umma_pipeline: bool = False,
        num_epi_subtiles: int = 1,
        epi_dtype=cutlass.Float32,
        acc_dtype=cutlass.Float32,
        output_dtype=cutlass.Float32,
        use_flat_logits_view=None,
        use_two_level_task_loop=None,
        use_paired_pipeline_polls=None,
        w_cache_regs=None,
        *,
        arch: str,
        output_index_bits: int = 32,
        math_mode: str = "relu_strict",
        explicit_query_bounds: bool = False,
        num_q_stages: int | None = None,
        num_kv_stages: int | None = None,
    ):
        for stages in (num_q_stages, num_kv_stages):
            if stages is not None and (type(stages) is not int or stages not in (2, 3)):
                raise ValueError("explicit Q/KV stages must be 2 or 3")
        if (num_q_stages is not None or num_kv_stages is not None) and (
            max_kv_pipeline or max_umma_pipeline
        ):
            raise ValueError("explicit SMEM stages prohibit automatic pipeline sizing")
        if type(explicit_query_bounds) is not bool:
            raise TypeError("explicit_query_bounds must be bool")
        if explicit_query_bounds and (
            block_kv != 128
            or not 8 <= num_heads * next_n <= 256
            or (num_heads * next_n) % 8
        ):
            raise ValueError("explicit query geometry requires CG1 M128 and legal N")
        self.explicit_query_bounds = explicit_query_bounds
        if explicit_query_bounds and (
            epi_dtype != cutlass.Float32
            or acc_dtype != cutlass.Float32
            or output_dtype != cutlass.Float32
            or math_mode != "relu_strict"
        ):
            raise ValueError("explicit query bounds require strict all-FP32 math")
        self._math_mode = _validate_math_mode(math_mode)
        # arch is the cute.compile --gpu-arch target (e.g. "sm_100a"), passed
        # explicitly and required: __init__ runs before compilation installs
        # the target, so a DSL arch query here would silently read the ambient
        # (device-0 / CUTE_DSL_ARCH) arch instead -- see _is_rubin_arch.
        #
        # w_cache_regs: register budget for the epilogue weight cache (see
        # _MAX_W_CACHE_REGS); None keeps the default. Any positive value works:
        # the math WGs floor it to the FMA unroll per slot and never go below
        # one unroll, so it only decides how many weights per slot live in
        # registers versus the SMEM path.
        if w_cache_regs is not None and w_cache_regs <= 0:
            raise ValueError(
                f"w_cache_regs must be a positive register count; got {w_cache_regs}"
            )
        self.w_cache_regs = _MAX_W_CACHE_REGS if w_cache_regs is None else w_cache_regs
        if output_index_bits not in (32, 64):
            raise ValueError("output_index_bits must be 32 or 64")
        self.output_index_bits = output_index_bits
        self.block_kv = block_kv
        self.phys_block_kv = phys_block_kv
        self.num_blocks_per_mma = block_kv // phys_block_kv
        assert block_kv % phys_block_kv == 0, (
            f"block_kv={block_kv} must be divisible by phys_block_kv={phys_block_kv}"
        )
        assert self.num_blocks_per_mma <= 4, (
            f"num_blocks_per_mma={self.num_blocks_per_mma} exceeds max 4"
        )
        self.remove_kv_wait_in_epilogue = remove_kv_wait_in_epilogue
        self.early_tmem_copy = early_tmem_copy
        self.smem_subpartition_opt = smem_subpartition_opt
        # Per-arch optimization gates (None = auto). All consumed via
        # cutlass.const_expr in-kernel.
        # - use_flat_logits_view (flat logits view + carried row offset):
        #   regresses on Rubin (sm_107) — the backend schedules the UMMA
        #   consumer-release late in the math loop, throttling the MMA
        #   pipe per KV tile, growing with ctx. Pure win on Blackwell.
        #   Auto: off on Rubin.
        # - use_two_level_task_loop / use_paired_pipeline_polls: with the
        #   flat view off, both on wins on Rubin at nn<4; at nn=4 the
        #   baseline loop is the best achievable there. Auto: on except
        #   Rubin nn=4 (both fall back together so nn=4 compiles to the
        #   baseline task-loop form).
        is_rubin = _is_rubin_arch(arch)
        if use_flat_logits_view is None:
            use_flat_logits_view = not is_rubin
        if use_two_level_task_loop is None:
            use_two_level_task_loop = not (is_rubin and next_n >= 4)
        if use_paired_pipeline_polls is None:
            use_paired_pipeline_polls = not (is_rubin and next_n >= 4)
        self.use_flat_logits_view = use_flat_logits_view
        self.use_two_level_task_loop = use_two_level_task_loop
        self.use_paired_pipeline_polls = use_paired_pipeline_polls
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.next_n = next_n
        self.N = next_n * num_heads
        self.num_sms = num_sms
        self.num_epi_subtiles = num_epi_subtiles
        self.epi_dtype = epi_dtype
        self.epi_bytes = 2 if epi_dtype == cutlass.Float16 else 4
        # sW stage stride padded to 128-byte SMEM alignment for TMA bulk copy.
        # Without padding, e.g. fp16 + N=32 gives 64B per stage, so stage 1
        # at +64 would be misaligned (TMA requires 128-byte aligned SMEM dest).
        w_stage_bytes = self.N * self.epi_bytes
        self.w_stage_stride = ((w_stage_bytes + 127) // 128 * 128) // self.epi_bytes
        self.output_dtype = output_dtype
        if num_epi_subtiles > 1 and num_heads % num_epi_subtiles != 0:
            raise ValueError("num_heads must be divisible by num_epi_subtiles")
        if (num_heads // num_epi_subtiles) % 4 != 0:
            raise ValueError(
                "num_heads // num_epi_subtiles must be divisible by 4 (FMA unroll granularity)"
            )
        self.num_groups = 2

        self.num_math_threads = 256
        self.num_specialized_threads = 128
        self.threads_per_cta = 384
        self.num_math_warps = 8
        self.tma_warp_base = 8
        self.umma_warp_base = 10

        self.num_q_stages = 3 if num_q_stages is None else num_q_stages

        # TMEM columns available on the compile target: each group needs N
        # columns per UMMA stage, so max_umma_stages = TMEM_COLS // (2 * N).
        #
        # Queried per-arch rather than hardcoded to SM100's 512 because Rubin
        # (sm_107) exposes 576. Note this changes nothing through the public
        # API: max_umma_pipeline is never enabled there, so num_umma_stages is
        # pinned to 1 and TMEM_COLS goes unused. Even with the knob forced on
        # in a vendored-code experiment, 512 and 576 disagree only for N in
        # (128, 144]; the deployed num_heads=64 shapes (N in {64, 128, 192,
        # 256}) never land in that band, though checker-legal off-shapes
        # (e.g. num_heads=68, next_n=2 -> N=136) could. It is written this way
        # so the constant stops being wrong on a 576-column part, not because
        # it buys a stage today.
        _major, _minor = _arch_major_minor(arch)
        TMEM_COLS = get_max_tmem_alloc_cols(f"sm_{_major}{_minor}")
        if max_umma_pipeline:
            self.num_umma_stages = min(2, TMEM_COLS // (2 * self.N))
        else:
            self.num_umma_stages = 1

        if max_kv_pipeline:
            smem_capacity = utils.get_smem_capacity_in_bytes()
            # Reserve ~1 KB for barriers and misc
            SMEM_BUDGET = smem_capacity - 1024
            # KV+Scale per stage (×2 groups):
            #   2 * (block_kv * head_dim * 1B + block_kv * 4B)
            kv_scale_per_stage = 2 * (block_kv * head_dim + block_kv * 4)
            # Q+W per stage: Q is N * head_dim * 1B, W uses padded stride
            qw_per_stage = self.N * head_dim + self.w_stage_stride * self.epi_bytes
            qw_total = qw_per_stage * self.num_q_stages
            self.num_kv_stages = (SMEM_BUDGET - qw_total) // kv_scale_per_stage
        else:
            self.num_kv_stages = 3 if num_kv_stages is None else num_kv_stages

        # Pad SMEM to push sW/sScales into sub-partition 1 (>= 128KB),
        # avoiding sub-bank conflicts with UMMA reading sKV.
        # Layout: barriers(~256B) | sKV_0 | sKV_1 | sQ | [pad] | sW | sScales
        if self.smem_subpartition_opt:
            BOUNDARY = 128 * 1024
            used = (
                256
                + 2 * (block_kv * head_dim * self.num_kv_stages)
                + self.N * head_dim * self.num_q_stages
            )
            used = ((used + 127) // 128) * 128
            if used < BOUNDARY:
                self.smem_pad_bytes = ((BOUNDARY - used + 1023) // 1024) * 1024
            else:
                self.smem_pad_bytes = 0
        else:
            self.smem_pad_bytes = 0

        self.acc_dtype = acc_dtype
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.mma_tiler_mn = (block_kv, self.N)

    def _setup_mma(self, a_dtype, b_dtype, a_major, b_major):
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.a_major_mode = a_major
        self.b_major_mode = b_major

        self.mma_tiler = (*self.mma_tiler_mn, 1)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            a_dtype,
            a_major,
            b_major,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )
        return self._setup_mma_layout(tiled_mma, a_dtype, b_dtype)

    @cute.jit
    def __call__(
        self,
        kv_fused: cute.Tensor,  # Fused KV: [num_phys_blocks, block_bytes] FP8
        b: cute.Tensor,  # Q: [N, head_dim, batch_size]
        weights: cute.Tensor,  # [N, batch_size] (transposed for TMA)
        logits: cute.Tensor,  # [batch_size * next_n, max_context_len]
        block_table: cute.Tensor,  # [batch_size, max_blocks_per_seq]
        context_lens: cute.Tensor,  # [batch_size]
        schedule_meta: cute.Tensor,  # [num_sms+1, 2] int32
        num_phys_blocks: cutlass.Int32,
        batch_size: cutlass.Int32,
        stream: cuda.CUstream,
        query_metadata=None,
    ):
        if cutlass.const_expr(
            self.explicit_query_bounds != (query_metadata is not None)
        ):
            raise ValueError("explicit query metadata ABI mismatch")
        # Derive KV and Scale views from fused buffer using CuTE ops.
        # Fused layout per physical block: [KV data (phys_block_kv*head_dim)] [Scales (phys_block_kv*4)]
        phys_block_kv = self.phys_block_kv
        scale_offset_elems = phys_block_kv * self.head_dim  # in FP8 elements

        # Recast fused buffer to FP8 (same 1-byte elements, needed for MMA type inference)
        kv_fp8 = cute.recast_tensor(kv_fused, cutlass.Float8E4M3FN)

        # Q (b) was passed as uint8 to work around DLPack's lack of float8 support;
        # recast back to FP8 so MMA type inference and TMA descriptors are correct.
        b = cute.recast_tensor(b, cutlass.Float8E4M3FN)

        # Read the real per-block stride (bytes = FP8 elements) from the input.
        # When KV is the indexer K-cache pool view, the pool is laid out as
        # [num_blocks, num_layers, kvFactor, blockSize], so dim-0 stride =
        # num_layers * kvFactor * phys_block_bytes (not phys_block_bytes).
        # Using the input stride keeps both contiguous test path and the
        # strided prod path correct. FP8 = 1 byte per element, so the byte
        # stride is the same as the element stride after recast.
        kv_block_stride = kv_fused.layout.stride[0]

        # KV view: [phys_block_kv, head_dim, num_phys_blocks] FP8
        # Each TMA loads one physical block; multiple TMAs fill a compute tile.
        kv_layout = cute.make_layout(
            (phys_block_kv, self.head_dim, num_phys_blocks),
            stride=(self.head_dim, 1, kv_block_stride),
        )
        a = cute.make_tensor(kv_fp8.iterator, kv_layout)

        # Scale view: offset pointer to scale region, recast FP8 → Float32
        # [phys_block_kv, num_phys_blocks] float32 (after recast)
        scale_fp8_layout = cute.make_layout(
            (phys_block_kv * 4, num_phys_blocks),
            stride=(1, kv_block_stride),
        )
        scale_fp8 = cute.make_tensor(
            kv_fp8.iterator + scale_offset_elems, scale_fp8_layout
        )
        scales = cute.recast_tensor(scale_fp8, cutlass.Float32)

        a_dtype = a.element_type
        b_dtype = b.element_type
        a_major = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        b_major = utils.LayoutEnum.ROW_MAJOR.mma_major_mode()

        tiled_mma = self._setup_mma(a_dtype, b_dtype, a_major, b_major)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # TMA for KV (A) — fmha_decode_paged pattern.
        # Build a TMA SMEM layout via tiled_divide on the full compute-tile
        # layout, then select to drop trivial K dim. Atom uses mode [0] as
        # single-tile SMEM layout and (phys, head) as cta_tiler.
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        self.a_tma_view_layout = sm100_utils.make_smem_layout(
            tcgen05.OperandMajorMode.K,
            (self.block_kv, self.head_dim),
            a_dtype,
            self.num_kv_stages,
        )
        self.a_tma_view_layout = cute.tiled_divide(
            self.a_tma_view_layout, (self.phys_block_kv, self.head_dim)
        )
        # ((tile_M, tile_K), rest_M, rest_K, stages) → drop trivial rest_K
        self.a_tma_view_layout = cute.select(self.a_tma_view_layout, mode=[0, 1, 3])
        # ((tile_M, tile_K), rest_M=num_sub_blocks, stages)
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_load_op,
            a,
            self.a_tma_view_layout[0],  # atom SMEM = single-tile (mode 0)
            (self.phys_block_kv, self.head_dim),
        )

        # TMA for Q (B) — full K=128, L dim = batch_size (unchanged)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA for Weights — [N, batch_size], tile [N], L=batch_size
        self.w_smem_layout_staged = cute.make_layout(
            (self.N, self.num_q_stages),
            stride=(1, self.w_stage_stride),
        )
        w_smem_per_stage = cute.select(self.w_smem_layout_staged, mode=[0])
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_op,
            weights,
            w_smem_per_stage,
            self.w_smem_layout_staged.shape[:1],
        )

        # TMA for Scales — [phys_block_kv, num_phys_blocks], tile [phys_block_kv]
        # SMEM holds compute_block_kv scales per stage; filled by
        # num_blocks_per_mma sub-block TMAs at consecutive offsets.
        self.s_smem_layout_staged = cute.make_layout(
            (self.block_kv, self.num_kv_stages)
        )
        s_smem_per_subblock = cute.make_layout((phys_block_kv,))
        tma_atom_s, tma_tensor_s = cpasync.make_tiled_tma_atom(
            tma_load_op,
            scales,
            s_smem_per_subblock,
            (phys_block_kv,),
        )

        b_copy_size = cute.size_in_bytes(b_dtype, b_smem_layout)
        w_copy_size = self.N * self.epi_bytes
        # Per sub-block: phys_block_kv * head_dim (KV) + phys_block_kv * 4 (scales)
        kv_tma_bytes_per_subblock = phys_block_kv * self.head_dim
        scale_tma_bytes_per_subblock = phys_block_kv * 4
        # Total per compute tile = num_blocks_per_mma sub-blocks
        self.num_kv_scale_tma_bytes = self.num_blocks_per_mma * (
            kv_tma_bytes_per_subblock + scale_tma_bytes_per_subblock
        )
        # Q + Weights share barrier (like DeepGEMM)
        self.num_q_tma_bytes = b_copy_size * atom_thr_size + w_copy_size

        num_ctas = self.num_sms

        @cute.struct
        class SharedStorage:
            kv_mbar_0: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            kv_mbar_1: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.num_q_stages * 2]
            umma_mbar_0: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 2]
            umma_mbar_1: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 2]
            tmem_holding_buf: cutlass.Int32

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_w,
            tma_tensor_w,
            tma_atom_s,
            tma_tensor_s,
            logits,
            block_table,
            context_lens,
            schedule_meta,
            query_metadata,
            batch_size,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.w_smem_layout_staged,
            self.s_smem_layout_staged,
            self.a_tma_view_layout,
            self.epi_tile,
            SharedStorage,
        ).launch(
            grid=(1, 1, num_ctas),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,  # KV pool
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,  # Q (L dim = batch_size)
        tma_atom_w: cute.CopyAtom,
        mW_tma: cute.Tensor,  # Weights TMA coord tensor [N, batch_size]
        tma_atom_s: cute.CopyAtom,
        mS_tma: cute.Tensor,  # Scales TMA coord tensor [block_kv, num_phys_blocks]
        mLogits: cute.Tensor,  # [batch_size * next_n, max_context_len]
        mBlockTable: cute.Tensor,  # [batch_size, max_blocks_per_seq]
        mContextLens: cute.Tensor,  # [batch_size]
        mScheduleMeta: cute.Tensor,  # [num_sms+1, 2] int32
        mQueryMetadata,
        batch_size: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        w_smem_layout_staged: cute.Layout,
        s_smem_layout_staged: cute.Layout,
        a_tma_view_layout: cute.ComposedLayout,
        epi_tile: cute.Tile,
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        # Warp roles (matches DeepGEMM SM100)
        warpgroup_idx = warp_idx // 4
        is_math_warp = warp_idx < 8
        is_tma_warp_0 = warp_idx == 8
        is_tma_warp_1 = warp_idx == 9
        is_tma_warp = is_tma_warp_0 | is_tma_warp_1
        is_umma_warp_0 = warp_idx == 10
        is_umma_warp_1 = warp_idx == 11

        # Early schedule metadata load: issue global loads ASAP so their
        # ~200-cycle L2 latency overlaps with subsequent prologue setup
        # (SMEM alloc, TMA partition, MMA fragment creation, etc.)
        NUM_MATH_WG = 2  # kNumMathWarpGroups
        sm_idx = bidz
        start_q = mScheduleMeta[(sm_idx, 0)]
        start_kv_half = mScheduleMeta[(sm_idx, 1)]
        end_q_idx = mScheduleMeta[(sm_idx + 1, 0)]
        end_kv_half = mScheduleMeta[(sm_idx + 1, 1)]

        if is_tma_warp:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_s)

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        block_kv_val = self.block_kv
        num_heads = self.num_heads
        num_epi_subtiles = self.num_epi_subtiles

        # === Pipelines ===
        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)

        # Q pipeline: TMA producer -> Math consumer (8 math warps)
        # PipelineTmaAsync: consumer_release uses is_signalling_thread
        # (lane 0 per warp). 8 math warps * 1 lane-0 = 8 arrives.
        q_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 8)
        q_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_mbar.data_ptr(),
            num_stages=self.num_q_stages,
            producer_group=prod_group,
            consumer_group=q_cons_group,
            tx_count=self.num_q_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        q_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_q_stages
        )
        # Both Math WGs share the same pipeline state (advance in lockstep)
        q_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        # UMMA warps observe Q pipeline (wait only, no release)
        # to ensure Q is in SMEM before GEMM. Critical for UMMA warp 1
        # since TMA warp 1 only loads KV1 (not Q).
        q_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        q_cons_state_umma_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )

        # Merged KV+Scale pipelines use the configured per-group ring size.
        # Like DeepGEMM: KV data and scales share one barrier.
        # TMA loads both under one barrier. Math is consumer (releases).
        # UMMA also waits on this barrier (for KV GEMM) but does NOT release.
        math_warps_per_group = self.num_math_warps // 2  # 4 warps
        kv_cons_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, math_warps_per_group
        )
        kv_pipeline_0 = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar_0.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_scale_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        kv_pipeline_1 = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar_1.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_scale_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )

        kv_prod_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )
        kv_prod_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )
        # UMMA consumer states (wait only, no release)
        kv_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        kv_cons_state_umma_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        # Math consumer states (wait + release)
        kv_cons_state_math_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        kv_cons_state_math_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )

        # UMMA pipelines (per-group)
        math_threads_per_group = self.num_math_threads // 2
        umma_pipeline_0 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar_0.data_ptr(),
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, math_threads_per_group
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        umma_pipeline_1 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar_1.data_ptr(),
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, math_threads_per_group
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        umma_prod_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_umma_stages
        )
        umma_prod_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_umma_stages
        )
        umma_cons_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_umma_stages
        )
        umma_cons_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_umma_stages
        )

        # TMEM -- only Math warps (8*32=256) + UMMA warps (2*32=64) = 320 threads
        # TMA warps do NOT participate, so they can start TMA loads earlier.
        # Math warp 0 is the allocator (like fp16_gemm_3's epilogue warp 0),
        # because math warps are the last TMEM consumers (epilogue reads).
        tmem_alloc_num_threads = 320  # 10 warps: warp 0-7 (math) + warp 10-11 (umma)
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=tmem_alloc_num_threads
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,  # math warp 0 does alloc+free (last TMEM consumer)
            is_two_cta=False,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # SMEM allocation: per-group KV + shared Q
        sKV_0 = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sKV_1 = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sQ = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        # Pad SMEM to push sW/sScales into sub-partition 1 (>= 128KB)
        # to avoid sub-bank conflicts with UMMA reading sKV from
        # sub-partition 0.
        if cutlass.const_expr(self.smem_pad_bytes > 0):
            _ = smem.allocate(self.smem_pad_bytes)
        # Weights SMEM: [N, num_q_stages], shared Q barrier
        sW = smem.allocate_tensor(
            element_type=self.epi_dtype,
            layout=w_smem_layout_staged,
            byte_alignment=128,
        )
        # Scales SMEM: [block_kv, num_kv_stages] float32, per group
        sScales_0 = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=s_smem_layout_staged,
            byte_alignment=128,
        )
        sScales_1 = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=s_smem_layout_staged,
            byte_alignment=128,
        )

        a_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )

        # Partition KV (A): fmha_decode_paged pattern.
        # SMEM view is ((tile), num_sub_blocks, stages) -- built in __call__.
        # Use .outer (plain layout); swizzle is captured by sKV_0's iterator.
        # GMEM: local_tile by (phys, head), then group first 2 modes into tile.
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        sKV_0_for_tma = cute.make_tensor(sKV_0.iterator, a_tma_view_layout.outer)
        sKV_1_for_tma = cute.make_tensor(sKV_1.iterator, a_tma_view_layout.outer)
        gA = cute.local_tile(
            mA_mkl,
            (self.phys_block_kv, self.head_dim),
            coord=(None, None, None),
        )
        tAsA_0, tAgA_0 = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            sKV_0_for_tma,
            cute.group_modes(gA, 0, 2),
        )
        tAsA_1, tAgA_1 = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            sKV_1_for_tma,
            cute.group_modes(gA, 0, 2),
        )

        # Partition Q (B): shared SMEM, L dim = batch_size
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
        )
        tCgB = thr_mma.partition_B(gB_nkl)
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tBgB = tBgB[(None, 0, None, None)]  # [tma, K, L]

        # Partition Weights: standalone TMA, [N, batch_size] -> [N] per stage
        w_cta_layout = cute.make_layout((1,))
        tWsW, tWgW = cpasync.tma_partition(
            tma_atom_w,
            0,
            w_cta_layout,
            cute.group_modes(sW, 0, 1),
            cute.group_modes(mW_tma, 0, 1),
        )

        # Partition Scales: explicit sub_blocks + stages dims.
        # Layout (phys_block_kv, num_sub, stages) K-major with custom strides.
        s_tma_view_layout = cute.make_layout(
            (self.phys_block_kv, self.num_blocks_per_mma, self.num_kv_stages),
            stride=(1, self.phys_block_kv, self.block_kv),
        )
        sScales_0_for_tma = cute.make_tensor(sScales_0.iterator, s_tma_view_layout)
        sScales_1_for_tma = cute.make_tensor(sScales_1.iterator, s_tma_view_layout)
        # GMEM: local_tile by phys to match atom's tile size
        gS = cute.local_tile(mS_tma, (self.phys_block_kv,), coord=(None, None))
        tSsS_0, tSgS_0 = cpasync.tma_partition(
            tma_atom_s,
            0,
            cute.make_layout(1),
            sScales_0_for_tma,
            gS,
        )
        tSsS_1, tSgS_1 = cpasync.tma_partition(
            tma_atom_s,
            0,
            cute.make_layout(1),
            sScales_1_for_tma,
            gS,
        )

        # MMA fragments
        tCrA_0 = tiled_mma.make_fragment_A(sKV_0)
        tCrA_1 = tiled_mma.make_fragment_A(sKV_1)
        tCrB = tiled_mma.make_fragment_B(sQ)  # shared
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])

        # Staged acc (fp16_gemm_3 pattern): append UMMA stage dim
        # shape: (*acc_shape, STAGE) -- dynamic index on last dim reduces rank
        us = self.num_umma_stages
        cols = self.num_tmem_alloc_cols
        acc_shape_staged = cute.append(acc_shape, us)
        tCtAcc_fake_staged = tiled_mma.make_fragment_C(acc_shape_staged)

        # TMEM layout info (allocation deferred to UMMA/Math warp branches)
        cols_per_group = cols * us * (32 // self.acc_dtype.width)
        num_tmem_alloc_cols_total = self.num_tmem_alloc_cols_total

        # Epilogue setup
        c_layout = utils.LayoutEnum.ROW_MAJOR
        epi_sub_mn = (epi_tile[0], num_heads // num_epi_subtiles)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            c_layout,
            self.acc_dtype,
            self.acc_dtype,
            epi_sub_mn,
            use_2cta_instrs,
        )

        # ===== SCHEDULER: derive values from early-loaded schedule metadata =====
        # Consumers of the early schedule-metadata loads, placed at their
        # real use site. Known codegen limitation (not worked around):
        # the compiler does not keep these global loads and their
        # consumers apart -- regardless of source
        # placement, the clamp+add+shift chain is hoisted right after its
        # load, exposing the two L2 round-trips instead of hiding them
        # under the independent mbarrier-init work. ptxas trades exposed
        # load latency for shorter register lifetimes and re-solves to that
        # optimum no matter the IR order; getting the sink from source needs
        # a scheduling barrier (cfence) or a latency-aware cost model fix.
        # Clamp to avoid OOB when start_q == batch_size (zero-work CTA
        # sentinel). Zero-work CTAs get a stale current_num_kv, never used
        # because has_work will be False.
        start_q_clamped = min(start_q, batch_size - 1)
        current_num_kv = (
            mContextLens[start_q_clamped] + self.block_kv - 1
        ) // self.block_kv

        # ===== SCHEDULER STATE: sunk into each warp-role branch =====
        # Each warp role derives its own scheduler-state copy (end_kv_idx,
        # next_q_idx/next_kv_idx/next_num_kv, q_idx sentinel, has_work) from
        # the early-loaded schedule metadata AFTER its warpgroup_reg_alloc/
        # dealloc call. Deriving it here (before the branches) placed the
        # loop-carried values under the shared prologue register budget,
        # where the register pressure before the setmaxnreg
        # reallocation forced ptxas to spill the kv cursor + bound pair
        # to local memory; deriving per role eliminates
        # the spill. The block is pure arithmetic on already-loaded values,
        # so per-branch duplication is semantics-preserving.

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # ===== WARP-SPECIALIZED EXECUTION =====

        if is_tma_warp_0:
            # TMA warp 0: loads Q (prefetch) + KV for group 0
            # Matches DeepGEMM's TMA warp with kv_group_idx == 0
            cute.arch.warpgroup_reg_dealloc(24)

            # Per-role scheduler state (see comment above the branch chain)
            end_kv_idx = end_kv_half * NUM_MATH_WG
            next_q_idx = start_q
            next_kv_idx = start_kv_half * NUM_MATH_WG
            next_num_kv = current_num_kv
            q_idx = batch_size  # sentinel: no previous batch
            has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

            (q_prod_state, kv_prod_state_0) = self._run_tma_role(
                mContextLens,
                batch_size,
                block_kv_val,
                end_q_idx,
                end_kv_idx,
                next_q_idx,
                next_kv_idx,
                next_num_kv,
                q_idx,
                has_work,
                tidx,
                kv_pipeline_0,
                kv_prod_state_0,
                mBlockTable,
                mBlockTable,
                mBlockTable,
                tma_atom_a,
                tAgA_0,
                tAsA_0,
                tma_atom_s,
                tSgS_0,
                tSsS_0,
                a_mcast_mask,
                0,
                True,
                False,
                False,
                False,
                False,
                q_pipeline,
                q_prod_state,
                mB_nkl,
                mW_tma,
                thr_mma,
                sQ,
                sW,
                tma_atom_b,
                tma_atom_w,
                tBgB,
                tWgW,
                tBsB,
                tWsW,
                b_mcast_mask,
            )

        elif is_tma_warp_1:
            # TMA warp 1: loads KV + Scale for group 1 only
            # Matches DeepGEMM's TMA warp with kv_group_idx == 1
            cute.arch.warpgroup_reg_dealloc(24)

            # Per-role scheduler state (see comment above the branch chain)
            end_kv_idx = end_kv_half * NUM_MATH_WG
            next_q_idx = start_q
            next_kv_idx = start_kv_half * NUM_MATH_WG
            next_num_kv = current_num_kv
            q_idx = batch_size  # sentinel: no previous batch
            has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

            kv_prod_state_1 = self._run_tma_role(
                mContextLens,
                batch_size,
                block_kv_val,
                end_q_idx,
                end_kv_idx,
                next_q_idx,
                next_kv_idx,
                next_num_kv,
                q_idx,
                has_work,
                tidx,
                kv_pipeline_1,
                kv_prod_state_1,
                mBlockTable,
                mBlockTable,
                mBlockTable,
                tma_atom_a,
                tAgA_1,
                tAsA_1,
                tma_atom_s,
                tSgS_1,
                tSsS_1,
                a_mcast_mask,
                1,
                False,
                False,
                False,
                False,
                False,
            )

        elif is_umma_warp_0:
            # UMMA warp for group 0
            # Must wait on Q pipeline: TMA operations with different
            # barriers are NOT visibility-ordered even within the same
            # warp. KV0 barrier arriving does not guarantee Q SMEM
            # writes are visible.
            cute.arch.warpgroup_reg_dealloc(24)

            # Per-role scheduler state (see comment above the branch chain)
            end_kv_idx = end_kv_half * NUM_MATH_WG
            next_q_idx = start_q
            next_kv_idx = start_kv_half * NUM_MATH_WG
            next_num_kv = current_num_kv
            q_idx = batch_size  # sentinel: no previous batch
            has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

            # Each ordinary producer independently observes Q visibility.
            tiled_mma, q_cons_state_umma_0, kv_cons_state_umma_0, umma_prod_state_0 = (
                self._run_umma_role(
                    mContextLens,
                    batch_size,
                    block_kv_val,
                    end_q_idx,
                    end_kv_idx,
                    next_q_idx,
                    next_kv_idx,
                    next_num_kv,
                    q_idx,
                    has_work,
                    is_leader_cta,
                    q_pipeline,
                    q_cons_state_umma_0,
                    kv_pipeline_0,
                    kv_cons_state_umma_0,
                    umma_pipeline_0,
                    umma_prod_state_0,
                    tCrA_0,
                    tCrB,
                    tiled_mma,
                    tmem,
                    tCtAcc_fake_staged,
                    0,
                    False,
                    False,
                )
            )

        elif is_umma_warp_1:
            # UMMA warp for group 1
            # Explicitly waits on Q pipeline -- critical because TMA warp 1
            # only loads KV1, not Q. Without this wait, UMMA warp 1 can
            # start GEMM before TMA warp 0 finishes loading Q into SMEM.
            cute.arch.warpgroup_reg_dealloc(24)

            # Per-role scheduler state (see comment above the branch chain)
            end_kv_idx = end_kv_half * NUM_MATH_WG
            next_q_idx = start_q
            next_kv_idx = start_kv_half * NUM_MATH_WG
            next_num_kv = current_num_kv
            q_idx = batch_size  # sentinel: no previous batch
            has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

            # Each ordinary producer independently observes Q visibility.
            tiled_mma, q_cons_state_umma_1, kv_cons_state_umma_1, umma_prod_state_1 = (
                self._run_umma_role(
                    mContextLens,
                    batch_size,
                    block_kv_val,
                    end_q_idx,
                    end_kv_idx,
                    next_q_idx,
                    next_kv_idx,
                    next_num_kv,
                    q_idx,
                    has_work,
                    is_leader_cta,
                    q_pipeline,
                    q_cons_state_umma_1,
                    kv_pipeline_1,
                    kv_cons_state_umma_1,
                    umma_pipeline_1,
                    umma_prod_state_1,
                    tCrA_1,
                    tCrB,
                    tiled_mma,
                    tmem,
                    tCtAcc_fake_staged,
                    cols_per_group,
                    False,
                    False,
                )
            )

        elif is_math_warp:
            cute.arch.warpgroup_reg_alloc(240)

            # Per-role scheduler state (see comment above the branch chain).
            # Derived AFTER warpgroup_reg_alloc(240) so the loop-carried
            # scheduler words live under the 240-register math budget.
            end_kv_idx = end_kv_half * NUM_MATH_WG
            next_q_idx = start_q
            next_kv_idx = start_kv_half * NUM_MATH_WG
            next_num_kv = current_num_kv
            q_idx = batch_size  # sentinel: no previous batch
            has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

            (
                q_cons_state,
                kv_cons_state_math_0,
                umma_cons_state_0,
                kv_cons_state_math_1,
                umma_cons_state_1,
            ) = self._run_math_role(
                mContextLens,
                batch_size,
                block_kv_val,
                end_q_idx,
                end_kv_idx,
                next_q_idx,
                next_kv_idx,
                next_num_kv,
                q_idx,
                has_work,
                tidx,
                warpgroup_idx,
                tmem,
                num_tmem_alloc_cols_total,
                tCtAcc_fake_staged,
                cols_per_group,
                epi_sub_mn,
                copy_atom_t2r,
                mLogits,
                sW,
                q_pipeline,
                q_cons_state,
                kv_pipeline_0,
                umma_pipeline_0,
                sScales_0,
                False,
                self.w_cache_regs,
                kv_cons_state_math_0,
                umma_cons_state_0,
                kv_pipeline_1,
                umma_pipeline_1,
                sScales_1,
                kv_cons_state_math_1,
                umma_cons_state_1,
                mTileMeta=mQueryMetadata,
            )

        else:
            cute.arch.warpgroup_reg_dealloc(24)
