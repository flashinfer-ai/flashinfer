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
CuTe DSL FP8 paged MQA logits kernel for Blackwell (SM100).

Architecture:
  - Selective request-tiled storage/ABI and terminal publication live here.
  - The shared engine owns TMA, UMMA, math and optional score-plane store roles.
  - Two KV groups share Q; one UMMA warp issues both groups in order.
  - Segmented CANDIDATE uses 512 threads, including 128 store threads;
    the other modes retain their 384-thread role/register policy.
  - Persistent CTAs traverse assigned request/query-tile and KV-pair tasks.
  - Weights remain cached across KV blocks within each query generation.

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

The selective constructor specializes accumulation, reduction and output to FP32.
The ordinary facade owns its separate dtype and architecture contract.

"""

import inspect
from dataclasses import dataclass, replace

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .._fp8_mqa_constants import _COMPUTE_BLOCK_KV
from .._math_policy import _validate_math_mode
from .fp8_paged_mqa_logits import _FP8MQAEngineMixin

from .._selective_contract import (
    _COUNT_SEGMENTS,
    _MAX_STRIPED_SPLIT_KV,
    _SPLIT_KV_CHOICES,
    _SelectiveLogitsMode,
    _h64_single_pool_topology,
    _q1_owner_pool_topology,
)

# Max registers per thread for the epilogue weight cache. 160 is the largest
# footprint the per-slot policy below already requests at the shape it was
# tuned for (num_heads=64, next_n=4), so bounding here leaves every
# num_heads=64 configuration unchanged.
_MAX_W_CACHE_REGS = 160


@dsl_user_op
def store_global_u64_if(predicate, address, value, *, loc=None, ip=None) -> None:
    """Issue one predicated 64-bit store through a raw byte address."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(predicate).ir_value(loc=loc, ip=ip),
            cutlass.Int64(address).ir_value(loc=loc, ip=ip),
            cutlass.Uint64(value).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred p; setp.ne.b32 p, $0, 0; @p st.global.u64 [$1], $2; }",
        "r,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_shared_v4_f32(tensor, row, kv_base, stage, *, loc=None, ip=None):
    """Load four contiguous FP32 scores with one 16-byte shared-memory load."""
    elem_offset = cute.crd2idx((row, kv_base, stage), tensor.layout)
    smem_addr = (tensor.iterator + elem_offset).toint(loc=loc, ip=ip)
    result = llvm.inline_asm(
        llvm.StructType.get_literal([Int32.mlir_type] * 4),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        i32_bits_to_fp32(
            Int32(llvm.extractvalue(Int32.mlir_type, result, [idx], loc=loc, ip=ip))
        )
        for idx in range(4)
    )


@dataclass(frozen=True)
class FP8MQALogitsTiling:
    """Trace-time tiling and pipeline policy for one compiled specialization."""

    block_kv: int
    next_n: int
    num_q_stages: int
    num_kv_stages: int
    num_umma_stages: int
    num_epi_subtiles: int = 1
    remove_kv_wait_in_epilogue: bool = False
    early_tmem_copy: bool = False
    smem_subpartition_opt: bool = False


SELECTIVE_LOGITS_TILING = FP8MQALogitsTiling(
    block_kv=128,
    next_n=16,
    num_q_stages=3,
    num_kv_stages=3,
    num_umma_stages=2,
    num_epi_subtiles=8,
    remove_kv_wait_in_epilogue=True,
)


@dsl_user_op
def fp32_to_i32_bits(value, *, loc=None, ip=None) -> Int32:
    """Return the IEEE-754 payload without changing any score bits."""
    return Int32(
        llvm.bitcast(
            Int32.mlir_type,
            cutlass.Float32(value).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def i32_bits_to_fp32(value, *, loc=None, ip=None) -> cutlass.Float32:
    """Interpret an Int32 metadata word as an IEEE-754 float."""
    return cutlass.Float32(
        llvm.bitcast(
            cutlass.Float32.mlir_type,
            Int32(value).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


# Older CuTe DSL fmax wrappers lack NaN propagation. Select the intrinsic only
# when its signature supports it; otherwise preserve the same PTX semantics.
try:
    _FMAX_HAS_NAN = "nan" in inspect.signature(cute.arch.fmax).parameters
except (AttributeError, TypeError, ValueError):
    _FMAX_HAS_NAN = False


# Do not use x + abs(x) with a compensating half-scale: doubling x or its
# weighted contribution can overflow before the final scale makes it finite.
# For x=1, weight=FLT_MAX and KV scale=0.5, direct ReLU yields FLT_MAX/2.
@dsl_user_op
def relu_f32(a: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    """Apply range-safe, NaN-propagating ReLU before signed head weights."""
    if _FMAX_HAS_NAN:
        return cute.arch.fmax(a, 0.0, nan=True, loc=loc, ip=ip)
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip)],
            "max.NaN.f32 $0, $1, 0f00000000;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def make_warp_uniform_i64(value) -> cutlass.Int64:
    """Materialize one warp-uniform 64-bit value as two uniform halves."""
    bits = cutlass.Uint64(value)
    lo = cute.arch.make_warp_uniform(cutlass.Int32(bits & 0xFFFFFFFF))
    hi = cute.arch.make_warp_uniform(cutlass.Int32(bits >> 32))
    return cutlass.Int64(
        cutlass.Uint64(cutlass.Uint32(lo)) | (cutlass.Uint64(cutlass.Uint32(hi)) << 32)
    )


# Selective publication partitions flattened Q x H tiles; ordinary scoring
# owns a separate per-row epilogue and architecture policy.
class FP8MQASelectiveLogitsKernel(_FP8MQAEngineMixin):
    """Selective-logits kernel with bounded publication.

    ``SAMPLE`` writes a compact sampled-logit plane. Candidate/repair modes
    write the packed uint64 ``(float_bits, logical_index)`` ABI. They can use
    one global logical count or fixed store-warp segment counters consumed by
    the in-place finalizer. Other modes are
    histogram-only or exact-index-only; none materializes dense logits.
    Per-row exact starts/ends live in request-tile ``tile_meta`` columns, while
    threshold/radix-prefix words stay in caller-owned row storage. Publication
    performs no host readback or metadata-copy launch.
    """

    is_selective_logits_kernel = True
    sample_internal = False
    has_store_warp = False
    uses_single_umma_producer = True
    uses_request_tiled_q = True
    uses_contiguous_kv = False
    split_kv = 1
    num_count_segments: int
    store_wg_registers: int
    _run_store_warp = _FP8MQAEngineMixin._run_store_role
    _store_uniform_i64 = staticmethod(make_warp_uniform_i64)
    _store_load_v4 = staticmethod(load_shared_v4_f32)
    _store_policy_f32 = staticmethod(i32_bits_to_fp32)
    _engine_relu_f32 = staticmethod(relu_f32)

    @cute.jit
    def _release_kv_after_scale_use(self, kv_pipeline, kv_cons_state):
        """Defensively order late scale reads before selective KV-stage reuse."""
        cute.arch.fence_view_async_shared()
        kv_pipeline.consumer_release(kv_cons_state)
        kv_cons_state.advance()
        return kv_cons_state

    def _initialize_geometry(
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
        tiling: FP8MQALogitsTiling | None = None,
    ):
        if tiling is not None:
            block_kv = tiling.block_kv
            next_n = tiling.next_n
            num_epi_subtiles = tiling.num_epi_subtiles
        self.block_kv = block_kv
        self.phys_block_kv = phys_block_kv
        self.scale_tma_tile = phys_block_kv
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
        if self.N % num_epi_subtiles != 0:
            raise ValueError("N must be divisible by num_epi_subtiles")
        if num_epi_subtiles % next_n != 0 and next_n % num_epi_subtiles != 0:
            raise ValueError(
                "num_epi_subtiles must either partition each logical row "
                "or group a whole number of logical rows"
            )
        if (self.N // num_epi_subtiles) % 4 != 0:
            raise ValueError(
                "N // num_epi_subtiles must be divisible by 4 (FMA unroll granularity)"
            )
        self.num_groups = 2

        self.num_math_threads = 256
        self.num_specialized_threads = 128
        self.threads_per_cta = 384
        self.num_math_warps = 8
        self.tma_warp_base = 8
        self.umma_warp_base = 10
        self.math_wg_registers = 240
        self.specialized_wg_registers = 24
        # 3 stages for Q pipelining across batch sequences
        self.num_q_stages = tiling.num_q_stages if tiling is not None else 3

        # TMEM: 512 columns total, each group needs N columns per UMMA stage
        # max_umma_stages = 512 // (2 * N)
        TMEM_COLS = 512
        if tiling is not None:
            self.num_umma_stages = tiling.num_umma_stages
        elif max_umma_pipeline:
            self.num_umma_stages = min(2, TMEM_COLS // (2 * self.N))
        else:
            self.num_umma_stages = 1

        if tiling is not None:
            self.num_kv_stages = tiling.num_kv_stages
        elif max_kv_pipeline:
            smem_capacity = utils.get_smem_capacity_in_bytes()
            # Reserve ~1 KB for barriers and misc
            SMEM_BUDGET = smem_capacity - 1024
            # KV+Scale per stage (*2 groups):
            #   2 * (block_kv * head_dim * 1B + block_kv * 4B)
            kv_scale_per_stage = 2 * (block_kv * head_dim + block_kv * 4)
            # Q+W per stage: Q is N * head_dim * 1B, W uses padded stride
            qw_per_stage = self.N * head_dim + self.w_stage_stride * self.epi_bytes
            qw_total = qw_per_stage * self.num_q_stages
            self.num_kv_stages = (SMEM_BUDGET - qw_total) // kv_scale_per_stage
        else:
            self.num_kv_stages = 3

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

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        phys_block_kv: int,
        num_sms: int,
        mode: _SelectiveLogitsMode,
        capacity: int,
        radix_shift: int = 0,
        candidate_uses_causal_bounds: bool = True,
        split_kv: int = 1,
        num_count_segments: int = 0,
        count_segment_span: int = 0,
        sample_internal: bool = False,
        tiling: FP8MQALogitsTiling = SELECTIVE_LOGITS_TILING,
        math_mode: str = "relu_strict",
    ):
        self._math_mode = _validate_math_mode(math_mode)
        if not isinstance(mode, _SelectiveLogitsMode):
            raise ValueError("mode must be a _SelectiveLogitsMode")
        if capacity <= 0:
            raise ValueError("selective-logits capacity must be positive")
        if mode == _SelectiveLogitsMode.RADIX_HISTOGRAM and radix_shift not in (
            0,
            8,
            16,
            24,
        ):
            raise ValueError("radix_shift must be one of {0, 8, 16, 24}")
        self._initialize_geometry(
            block_kv=tiling.block_kv,
            phys_block_kv=phys_block_kv,
            num_heads=num_heads,
            head_dim=head_dim,
            next_n=tiling.next_n,
            num_sms=num_sms,
            remove_kv_wait_in_epilogue=tiling.remove_kv_wait_in_epilogue,
            early_tmem_copy=tiling.early_tmem_copy,
            smem_subpartition_opt=tiling.smem_subpartition_opt,
            num_epi_subtiles=tiling.num_epi_subtiles,
            epi_dtype=cutlass.Float32,
            acc_dtype=cutlass.Float32,
            output_dtype=cutlass.Float32,
            tiling=tiling,
        )
        self.mode = mode
        self.sample_internal = sample_internal
        self.capacity = capacity
        self.radix_shift = radix_shift
        self.candidate_uses_causal_bounds = candidate_uses_causal_bounds
        self.split_kv = split_kv
        self.num_count_segments = num_count_segments
        self.count_segment_span = count_segment_span
        self.has_store_warp = (
            mode == _SelectiveLogitsMode.CANDIDATE and num_count_segments > 0
        )
        # Wide segmented H64 candidates use the same math-register budget.
        # Cache fewer FP32 weights so epilogue/control values have register
        # headroom; uncached weights use the same held Q-stage SMEM path.
        self.math_weight_cache_regs = (
            64
            if self.has_store_warp
            and num_heads == 64
            and head_dim == 128
            and self.N == 256
            else _MAX_W_CACHE_REGS
        )
        if self.has_store_warp:
            single_pool = count_segment_span == 128 and _h64_single_pool_topology(
                num_heads,
                head_dim,
                tiling.next_n,
                tiling.num_umma_stages,
                split_kv,
                num_count_segments,
                capacity,
            )
            q1_pool = count_segment_span == 128 and _q1_owner_pool_topology(
                num_heads,
                head_dim,
                tiling.next_n,
                tiling.num_umma_stages,
                split_kv,
                num_count_segments,
                capacity,
            )
            if tiling.block_kv != 128:
                raise ValueError(
                    "dedicated candidate store warp requires KV128 compute tiles"
                )
            if (
                num_count_segments not in _COUNT_SEGMENTS
                and not single_pool
                and not q1_pool
            ):
                raise ValueError(
                    "dedicated candidate store warp requires "
                    "4, 8, 12, 16, 32, 64, or 128 segments, "
                    "or a supported per-split pooled topology"
                )
            if split_kv not in _SPLIT_KV_CHOICES:
                raise ValueError("candidate split_kv must be 1, 2, 4, 8, 16, 32, or 64")
            striped_split_layout = num_count_segments == 4 * split_kv
            pooled_split_layout = (
                single_pool
                or q1_pool
                or (split_kv > _MAX_STRIPED_SPLIT_KV and num_count_segments == split_kv)
            )
            lane_local_layout = split_kv <= 4 and num_count_segments == 32 * split_kv
            valid_split_layout = (
                striped_split_layout or pooled_split_layout or lane_local_layout
            )
            if split_kv > 1 and not valid_split_layout:
                required_layout = (
                    f"{split_kv}"
                    if split_kv > _MAX_STRIPED_SPLIT_KV
                    else f"{4 * split_kv}"
                )
                if split_kv <= 4:
                    required_layout += f" or {32 * split_kv}"
                raise ValueError(
                    f"split_kv={split_kv} requires {required_layout} candidate segments"
                )
            # Store coordinates are request-local shared-score coordinates.
            # KV128 tiles and KV32 segments keep every vector origin
            # four-aligned. Store warps without a mapped query row still
            # participate in every pipeline wait and release.
            assert tiling.block_kv % 4 == 0
            if num_count_segments == 4:
                assert count_segment_span % 4 == 0
            self.threads_per_cta = 512
            self.math_wg_registers = 160 if num_heads == 64 else 168
            self.specialized_wg_registers = 24
            self.store_wg_registers = 168 if num_heads == 64 else 152

    def _setup_mma(self, a_dtype, b_dtype, a_major, b_major):
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.a_major_mode = a_major
        self.b_major_mode = b_major

        self.mma_tiler = (*self.mma_tiler_mn, 1)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            a_dtype,
            b_dtype,
            a_major,
            b_major,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )
        return self._setup_mma_layout(tiled_mma, a_dtype, b_dtype)

    @cute.jit
    def _make_math_tmem_copy_and_partition(
        self,
        tCtAcc_base,
        epi_sub_mn,
        copy_atom_t2r,
        epi_tidx,
        cC,
    ):
        """Build the group-local TMEM copy view shared by both math WGs."""
        # (MMA_M, MMA_N, STAGE) -> (EPI_M, EPI_N, M, N, STAGE)
        tAcc = tCtAcc_base[((None, None), 0, 0, None)]
        tAcc_epi = cute.flat_divide(tAcc, epi_sub_mn)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
        tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)
        tTR_cC = thr_copy_t2r.partition_D(cC)
        tTR_rAcc = cute.make_rmem_tensor_like(tTR_cC, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc_base, tTR_cC, tTR_rAcc

    @cute.jit
    def _launch(
        self,
        kv_fused: cute.Tensor,
        b: cute.Tensor,
        weights: cute.Tensor,
        logits: cute.Tensor,
        block_table: cute.Tensor,
        index_counts: cute.Tensor,
        context_lens: cute.Tensor,
        schedule_meta: cute.Tensor,
        kv_scale: cute.Tensor,
        tile_meta: cute.Tensor,
        policy_values: cute.Tensor,
        num_phys_blocks: cutlass.Int32,
        batch_size: cutlass.Int32,
        stream: cuda.CUstream,
        sample_page_indices: cute.Tensor = None,
    ):
        # The common engine carries both the physical page table and selective
        # output counters. The public dense-logits path supplies its block table
        # in both slots; compile-time selective hooks are the only users of the
        # separate counter tensor.
        # Derive KV and Scale views from fused buffer using CuTE ops.
        # Fused layout per physical block: [KV data (phys_block_kv*head_dim)] [Scales (phys_block_kv*4)]
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

        if cutlass.const_expr(self.uses_contiguous_kv):
            kv_layout = cute.make_layout(
                (num_phys_blocks, self.head_dim, 1),
                stride=(kv_block_stride, 1, 0),
            )
            a = cute.make_tensor(kv_fp8.iterator, kv_layout)
            scales = kv_scale
        else:
            # KV view: [phys_block_kv, head_dim, num_phys_blocks] FP8
            # Each TMA loads one physical block; multiple TMAs fill a compute tile.
            kv_layout = cute.make_layout(
                (self.phys_block_kv, self.head_dim, num_phys_blocks),
                stride=(self.head_dim, 1, kv_block_stride),
            )
            a = cute.make_tensor(kv_fp8.iterator, kv_layout)
            # Scale view: offset pointer to scale region, recast FP8 -> Float32
            # [phys_block_kv, num_phys_blocks] float32 (after recast)
            scale_fp8_layout = cute.make_layout(
                (self.phys_block_kv * 4, num_phys_blocks),
                stride=(1, kv_block_stride),
            )
            fused_block_layout = cute.make_layout(
                (
                    self.phys_block_kv * (self.head_dim + 4),
                    num_phys_blocks,
                ),
                stride=(1, kv_block_stride),
            )
            fused_blocks = cute.make_tensor(kv_fp8.iterator, fused_block_layout)
            scale_storage = cute.domain_offset(
                (self.phys_block_kv * self.head_dim, 0), fused_blocks
            )
            scale_fp8 = cute.make_tensor(
                scale_storage.iterator,
                scale_fp8_layout,
            )
            scales = cute.recast_tensor(scale_fp8, cutlass.Float32)

        a_dtype = a.element_type
        b_dtype = b.element_type
        a_major = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        b_major = utils.LayoutEnum.ROW_MAJOR.mma_major_mode()

        tiled_mma = self._setup_mma(a_dtype, b_dtype, a_major, b_major)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # TMA for KV (A); fmha_decode_paged pattern.
        # Build a TMA SMEM layout via tiled_divide on the full compute-tile
        # layout, then select to drop trivial K dim. Atom uses mode [0] as
        # single-tile SMEM layout and (phys, head) as cta_tiler.
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        if cutlass.const_expr(self.uses_contiguous_kv):
            a_op = sm100_utils.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, tiled_mma.thr_id
            )
            tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
                a_op,
                a,
                self.a_smem_layout_staged,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            self.a_tma_view_layout = self.a_smem_layout_staged
        else:
            self.a_tma_view_layout = sm100_utils.make_smem_layout(
                cute.nvgpu.OperandMajorMode.K,
                (self.block_kv, self.head_dim),
                a_dtype,
                self.num_kv_stages,
            )
            # ((tile_M, tile_K), rest_M, rest_K, stages) -> drop trivial rest_K
            self.a_tma_view_layout = cute.tiled_divide(
                self.a_tma_view_layout, (self.phys_block_kv, self.head_dim)
            )
            # ((tile_M, tile_K), rest_M=num_sub_blocks, stages)
            self.a_tma_view_layout = cute.select(self.a_tma_view_layout, mode=[0, 1, 3])
            # atom SMEM = single-tile (mode 0)
            tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
                tma_load_op,
                a,
                self.a_tma_view_layout[0],
                (self.phys_block_kv, self.head_dim),
            )

        # TMA for Q (B); full K=128, L dim = batch_size (unchanged)
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

        # TMA for Weights; [N, batch_size], tile [N], L=batch_size
        # The selective-logits path keeps [query, head] weights and rebases each
        # request-owned query tile directly; paged decode retains [N, batch].
        self.w_smem_layout_staged = cute.make_layout(
            (self.N, self.num_q_stages),
            stride=(1, self.w_stage_stride),
        )
        if cutlass.const_expr(self.uses_request_tiled_q):
            w_smem_per_stage = cute.make_layout(
                (self.next_n, self.num_heads), stride=(self.num_heads, 1)
            )
            w_tma_tiler = (self.next_n, self.num_heads)
        else:
            w_smem_per_stage = cute.select(self.w_smem_layout_staged, mode=[0])
            w_tma_tiler = self.w_smem_layout_staged.shape[:1]
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_op,
            weights,
            w_smem_per_stage,
            w_tma_tiler,
        )

        # TMA for Scales; [phys_block_kv, num_phys_blocks], tile [phys_block_kv]
        # SMEM holds compute_block_kv scales per stage; filled by
        # num_blocks_per_mma sub-block TMAs at consecutive offsets.
        self.s_smem_layout_staged = cute.make_layout(
            (self.block_kv, self.num_kv_stages)
        )
        scale_tma_tile = self.scale_tma_tile
        s_smem_per_subblock = cute.make_layout((scale_tma_tile,))
        tma_atom_s, tma_tensor_s = cpasync.make_tiled_tma_atom(
            tma_load_op,
            scales,
            s_smem_per_subblock,
            (scale_tma_tile,),
        )

        b_copy_size = cute.size_in_bytes(b_dtype, b_smem_layout)
        w_copy_size = self.N * self.epi_bytes
        # Per sub-block: phys_block_kv * head_dim (KV) + phys_block_kv * 4 (scales)
        kv_tma_bytes_per_subblock = self.scale_tma_tile * self.head_dim
        scale_tma_bytes_per_subblock = self.scale_tma_tile * 4
        # Total per compute tile = num_blocks_per_mma sub-blocks
        self.num_kv_scale_tma_bytes = self.num_blocks_per_mma * (
            kv_tma_bytes_per_subblock + scale_tma_bytes_per_subblock
        )
        # Q + Weights share barrier (like DeepGEMM)
        self.num_q_tma_bytes = b_copy_size * atom_thr_size + w_copy_size

        num_ctas = self.num_sms

        @cute.struct
        class SharedStorage:
            kv_mbar: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 4]
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.num_q_stages * 2]
            umma_mbar: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 4]
            store_mbar: cute.struct.MemRange[cutlass.Int64, 8]
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
            index_counts,
            context_lens,
            schedule_meta,
            tile_meta,
            policy_values,
            sample_page_indices,
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
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.jit
    def _run_tma_warp_0(
        self,
        NUM_BLOCKS_PER_MMA,
        NUM_MATH_WG,
        a_mcast_mask,
        b_mcast_mask,
        batch_size,
        block_kv_val,
        end_kv_idx,
        end_q_idx,
        has_work,
        kv_pipeline_0,
        mA_mkl,
        mB_nkl,
        mBlockTable,
        mSamplePages,
        mContextLens,
        mS_tma,
        mTileMeta,
        mW_tma,
        next_kv_idx,
        next_num_kv,
        next_q_idx,
        q_idx,
        q_pipeline,
        sKV_0,
        sQ,
        sScales_0,
        sW,
        tAgA_0,
        tAsA_0,
        tBgB,
        tBsB,
        tSgS_0,
        tSsS_0,
        tWgW,
        tWsW,
        thr_mma,
        tidx,
        tma_atom_a,
        tma_atom_b,
        tma_atom_s,
        tma_atom_w,
    ):
        """Run the Q and first KV/scale TMA producer role."""
        cute.arch.setmaxregister_decrease(self.specialized_wg_registers)
        q_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_q_stages
        )
        kv_prod_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )

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
            mSamplePages,
            mTileMeta,
            tma_atom_a,
            tAgA_0,
            tAsA_0,
            tma_atom_s,
            tSgS_0,
            tSsS_0,
            a_mcast_mask,
            0,
            True,
            self.uses_request_tiled_q,
            True,
            True,
            self.sample_internal,
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

    @cute.jit
    def _run_tma_warp_1(
        self,
        NUM_BLOCKS_PER_MMA,
        NUM_MATH_WG,
        a_mcast_mask,
        batch_size,
        block_kv_val,
        end_kv_idx,
        end_q_idx,
        has_work,
        kv_pipeline_1,
        mA_mkl,
        mBlockTable,
        mSamplePages,
        mContextLens,
        mS_tma,
        mTileMeta,
        next_kv_idx,
        next_num_kv,
        next_q_idx,
        q_idx,
        sKV_1,
        sScales_1,
        tAgA_1,
        tAsA_1,
        tSgS_1,
        tSsS_1,
        thr_mma,
        tidx,
        tma_atom_a,
        tma_atom_s,
    ):
        """Run the second KV/scale TMA producer role."""
        cute.arch.setmaxregister_decrease(self.specialized_wg_registers)
        kv_prod_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )

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
            mSamplePages,
            mTileMeta,
            tma_atom_a,
            tAgA_1,
            tAsA_1,
            tma_atom_s,
            tSgS_1,
            tSsS_1,
            a_mcast_mask,
            1,
            False,
            self.uses_request_tiled_q,
            True,
            True,
            self.sample_internal,
        )

    @cute.jit
    def _run_umma_warp_0(
        self,
        NUM_MATH_WG,
        batch_size,
        block_kv_val,
        end_kv_idx,
        end_q_idx,
        has_work,
        is_leader_cta,
        kv_pipeline_0,
        kv_pipeline_1,
        mContextLens,
        next_kv_idx,
        next_num_kv,
        next_q_idx,
        q_idx,
        q_pipeline,
        tCrA_0,
        tCrA_1,
        tCrB,
        tCtAcc_fake_staged,
        tiled_mma,
        tmem,
        tmem_group_1_offset,
        umma_pipeline_0,
        umma_pipeline_1,
    ):
        """Run the first UMMA producer role and optional second ring."""
        cute.arch.setmaxregister_decrease(self.specialized_wg_registers)
        q_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        kv_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        umma_prod_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_umma_stages
        )
        if cutlass.const_expr(self.uses_single_umma_producer):
            kv_cons_state_umma_1 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_kv_stages
            )
            umma_prod_state_1 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_umma_stages
            )

        # These modes own one producer warp and two independent UMMA rings.
        # The engine advances its shared task only after both commits.
        (
            tiled_mma,
            q_cons_state_umma_0,
            kv_cons_state_umma_0,
            umma_prod_state_0,
            kv_cons_state_umma_1,
            umma_prod_state_1,
        ) = self._run_umma_role(
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
            True,
            True,
            tCrA_1,
            kv_pipeline_1,
            kv_cons_state_umma_1,
            umma_pipeline_1,
            umma_prod_state_1,
            tmem_group_1_offset,
        )

    @cute.jit
    def _request_math_fragment(
        self,
        acc_vec,
        sums,
        linear_n: cutlass.Constexpr,
        subtile_n: cutlass.Constexpr,
        next_n: cutlass.Constexpr,
        num_epi_subtiles: cutlass.Constexpr,
        num_heads: cutlass.Constexpr,
        NUM_W_IN_REG: cutlass.Constexpr,
        w_cache,
        sW,
        q_stage_local,
        mLogits,
        mIndexCounts,
        mTileMeta,
        mPolicyValues,
        q_idx,
        store_out_row_base,
        store_q_valid,
        store_causal_base,
        store_row_starts,
        store_row_ends,
        store_kv_pos,
        store_kv_valid,
        m_coord,
        kv_pos,
        scale_val,
        score_plane: cutlass.Constexpr = False,
        sStoreScores=None,
        store_stage=None,
        store_group=None,
    ):
        """Pure request-tiled reduction/output; no pipeline mutations."""
        s0x, s0y, s1x, s1y = sums
        # A wider TMEM load may contain multiple complete
        # logical rows. Keep each row's reduction state and
        # accumulator slice independent within the fragment.
        rows_per_subtile = max(1, next_n // num_epi_subtiles)
        row_n = subtile_n // rows_per_subtile
        h_base, subtile_base_t = cute.idx2crd(linear_n, (num_heads, next_n))
        for subtile_row in cutlass.range_constexpr(rows_per_subtile):
            t = subtile_base_t + subtile_row
            row_n_base = subtile_row * row_n
            if cutlass.const_expr(h_base == 0):
                s0x = cutlass.Float32(0.0)
                s0y = cutlass.Float32(0.0)
                s1x = cutlass.Float32(0.0)
                s1y = cutlass.Float32(0.0)
            # Reg-path: weights from registers
            reg_h_end = min(row_n, max(0, NUM_W_IN_REG - h_base))
            for h in cutlass.range_constexpr(0, reg_h_end, 4):
                n0 = row_n_base + h
                h_g = h_base + h
                s0x, s0y, s1x, s1y = self._accumulate_fp32_heads(
                    acc_vec,
                    n0,
                    w_cache,
                    ((t, h_g), (t, h_g + 1), (t, h_g + 2), (t, h_g + 3)),
                    (s0x, s0y, s1x, s1y),
                )
            # SMEM-path: weights from shared mem
            smem_h_start = max(0, NUM_W_IN_REG - h_base)
            for h in cutlass.range_constexpr(smem_h_start, row_n, 4):
                n0 = row_n_base + h
                h_g = h_base + h
                s0x, s0y, s1x, s1y = self._accumulate_fp32_heads(
                    acc_vec,
                    n0,
                    sW,
                    (
                        (t, h_g, q_stage_local),
                        (t, h_g + 1, q_stage_local),
                        (t, h_g + 2, q_stage_local),
                        (t, h_g + 3, q_stage_local),
                    ),
                    (s0x, s0y, s1x, s1y),
                )
            if cutlass.const_expr(h_base + row_n == num_heads):
                sum_x, sum_y = cute.arch.add_packed_f32x2(
                    (s0x, s0y),
                    (s1x, s1y),
                )
                result_t = sum_x + sum_y
                if cutlass.const_expr(score_plane):
                    # The common lifecycle owns stage acquire/commit/release;
                    # this pure sink writes only the final scaled score.
                    sStoreScores[(t, m_coord, store_stage, store_group)] = (
                        cutlass.Float32(result_t) * cutlass.Float32(scale_val)
                    )
                else:
                    self._store_result_packed_causal_meta(
                        mLogits,
                        mIndexCounts,
                        mTileMeta,
                        mPolicyValues,
                        q_idx,
                        store_out_row_base,
                        store_q_valid,
                        store_causal_base,
                        store_row_starts[t]
                        if cutlass.const_expr(self.is_selective_logits_kernel)
                        else cutlass.Int32(0),
                        store_row_ends[t]
                        if cutlass.const_expr(self.is_selective_logits_kernel)
                        else cutlass.Int32(0),
                        t,
                        store_kv_pos,
                        store_kv_valid,
                        result_t,
                        scale_val,
                        next_n,
                    )
        return s0x, s0y, s1x, s1y

    @cute.jit
    def _run_math_warp(
        self,
        NUM_MATH_WG,
        batch_size,
        block_kv_val,
        copy_atom_t2r,
        end_kv_idx,
        end_q_idx,
        epi_sub_mn,
        has_work,
        kv_pipeline_math,
        mContextLens,
        mIndexCounts,
        mLogits,
        mPolicyValues,
        mTileMeta,
        next_kv_idx,
        next_n,
        next_num_kv,
        next_q_idx,
        num_epi_subtiles,
        num_heads,
        num_tmem_alloc_cols_total,
        q_idx,
        q_pipeline,
        sScales,
        sW,
        tCtAcc_fake_staged,
        tidx,
        tmem,
        tmem_group_1_offset,
        umma_pipeline_math,
        warpgroup_idx,
        sStoreScores=None,
        store_pipeline_math=None,
    ):
        """Run one math warpgroup epilogue and score-plane producer."""
        cute.arch.setmaxregister_increase(self.math_wg_registers)
        # Both math WGs advance their own Q state in lockstep.
        q_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )

        math_states = self._run_math_role(
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
            tmem_group_1_offset,
            epi_sub_mn,
            copy_atom_t2r,
            mLogits,
            sW,
            q_pipeline,
            q_cons_state,
            kv_pipeline_math,
            umma_pipeline_math,
            sScales,
            True,
            self.math_weight_cache_regs,
            mIndexCounts=mIndexCounts,
            mTileMeta=mTileMeta,
            mPolicyValues=mPolicyValues,
            score_plane=self.has_store_warp,
            sStoreScores=sStoreScores,
            store_pipeline=store_pipeline_math,
        )
        if cutlass.const_expr(self.has_store_warp):
            (
                q_cons_state,
                kv_cons_state_math,
                umma_cons_state,
                _store_prod_state,
            ) = math_states
        else:
            q_cons_state, kv_cons_state_math, umma_cons_state = math_states

    @cute.jit
    def _run_reserved_warp(self):
        """Apply the specialized register budget to the reserved warp."""
        cute.arch.setmaxregister_decrease(self.specialized_wg_registers)

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
        mIndexCounts: cute.Tensor,  # selective counters or paged placeholder
        mContextLens: cute.Tensor,  # [batch_size]
        mScheduleMeta: cute.Tensor,  # [num_sms+1, 2] int32
        mTileMeta: cute.Tensor,  # selective request metadata or paged placeholder
        mPolicyValues: cute.Tensor,  # selective row policy or paged placeholder
        mSamplePages: cute.Tensor,
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
        # Warp 10 owns both UMMA rings; warp 11 remains reserved.
        is_store_warp = warp_idx >= 12

        # Early schedule metadata load: issue global loads ASAP so their
        # ~200-cycle L2 latency overlaps with subsequent prologue setup
        # (SMEM alloc, TMA partition, MMA fragment creation, etc.)
        NUM_MATH_WG = 2  # kNumMathWarpGroups
        NUM_STORE_WARPS = 4
        NUM_BLOCKS_PER_MMA = self.num_blocks_per_mma
        sm_idx = bidz
        start_q = mScheduleMeta[(sm_idx, 0)]
        start_kv_half = mScheduleMeta[(sm_idx, 1)]
        end_q_idx = mScheduleMeta[(sm_idx + 1, 0)]
        end_kv_half = mScheduleMeta[(sm_idx + 1, 1)]
        # Early mContextLens load: overlap ~200-cycle L2 latency with the
        # entire prologue setup (pipelines, SMEM alloc, TMA partition, etc.)
        # Clamp to avoid OOB when start_q == batch_size (zero-work CTA sentinel).
        # Note: zero-work CTAs get a stale current_num_kv (from the last batch
        # element), but it is never used because has_work will be False.
        start_q_clamped = min(start_q, batch_size - 1)
        current_num_kv = (
            mContextLens[start_q_clamped] + self.block_kv - 1
        ) // self.block_kv

        if is_tma_warp:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_s)

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        block_kv_val = self.block_kv
        num_heads = self.num_heads
        next_n = self.next_n
        num_epi_subtiles = self.num_epi_subtiles

        # === Pipelines ===
        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)

        # Q pipeline: TMA producer to Math consumer (8 math warps).
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
        # Merged KV+Scale pipelines (per-group, 3 stages each)
        # Like DeepGEMM: KV data and scales share one barrier.
        # TMA loads both under one barrier. Math is consumer (releases).
        # UMMA also waits on this barrier (for KV GEMM) but does NOT release.
        math_warps_per_group = self.num_math_warps // 2  # 4 warps
        kv_cons_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, math_warps_per_group
        )
        kv_pipeline_0 = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_scale_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        kv_pipeline_1 = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar.data_ptr() + self.num_kv_stages * 2,
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_scale_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        math_pipeline_ring = min(warpgroup_idx, cutlass.Int32(1))
        kv_pipeline_math = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar.data_ptr()
            + math_pipeline_ring * self.num_kv_stages * 2,
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_scale_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )

        # UMMA pipelines (per-group)
        math_threads_per_group = self.num_math_threads // 2
        umma_pipeline_0 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar.data_ptr(),
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, math_threads_per_group
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        umma_pipeline_1 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar.data_ptr() + self.num_umma_stages * 2,
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, math_threads_per_group
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        umma_pipeline_math = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar.data_ptr()
            + math_pipeline_ring * self.num_umma_stages * 2,
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, math_threads_per_group
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Candidate-only score handoff.  Each math warpgroup produces one
        # query_tile x KV128 FP32 plane; the dedicated store warpgroup consumes it.
        # Two stages per math group provide the requested direct 32-KiB
        # ping-pong without involving the reserved warp 11.
        if cutlass.const_expr(self.has_store_warp):
            store_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
            store_pipeline_0 = pipeline.PipelineAsync.create(
                barrier_storage=storage.store_mbar.data_ptr(),
                num_stages=2,
                producer_group=store_group,
                consumer_group=store_group,
            )
            store_pipeline_1 = pipeline.PipelineAsync.create(
                barrier_storage=storage.store_mbar.data_ptr() + 4,
                num_stages=2,
                producer_group=store_group,
                consumer_group=store_group,
            )
            # The math body selects its existing group-local ring once through a
            # uniform pointer offset. The two store-facing wrappers above
            # still own initialization and expose both rings independently.
            store_pipeline_math = pipeline.PipelineAsync.create(
                barrier_storage=storage.store_mbar.data_ptr() + math_pipeline_ring * 4,
                num_stages=2,
                producer_group=store_group,
                consumer_group=store_group,
                defer_sync=True,
            )
        # TMEM: only Math warps (8*32=256) + UMMA warps (2*32=64) = 320 threads
        # TMA warps do NOT participate, so they can start TMA loads earlier.
        # Math warp 0 is the allocator (like fp16_gemm_3's epilogue warp 0),
        # because math warps are the last TMEM consumers (epilogue reads).
        # The TMEM allocation barrier includes only roles that retrieve the
        # allocation.  A single UMMA producer reserves warp 11, reducing the
        # participant set from ten to nine warps.
        # 10 warps: warp 0-7 (math) + warp 10-11 (umma)
        tmem_alloc_num_threads = 288 if self.uses_single_umma_producer else 320
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
        # Preserve the logical (query row, head, stage) coordinates instead of
        # flattening row/head addresses throughout the producer and epilogue.
        sW = cute.make_tensor(
            sW.iterator,
            cute.make_layout(
                (next_n, num_heads, self.num_q_stages),
                stride=(num_heads, 1, self.w_stage_stride),
            ),
        )
        # Scales SMEM: [block_kv, num_kv_stages] float32, per group
        scales_layout = cute.make_layout(
            (block_kv_val, self.num_kv_stages, 2),
            stride=(1, block_kv_val, block_kv_val * self.num_kv_stages),
        )
        sScales = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=scales_layout,
            byte_alignment=128,
        )
        sScales_0 = sScales[(None, None, 0)]
        sScales_1 = sScales[(None, None, 1)]
        if cutlass.const_expr(self.has_store_warp):
            store_score_layout = cute.make_layout(
                (next_n, block_kv_val, 2, 2),
                stride=(
                    block_kv_val,
                    1,
                    next_n * block_kv_val,
                    next_n * block_kv_val * 2,
                ),
            )
            sStoreScores = smem.allocate_tensor(
                element_type=cutlass.Float32,
                layout=store_score_layout,
                byte_alignment=128,
            )
            sStoreScores_0 = sStoreScores[(None, None, None, 0)]
            sStoreScores_1 = sStoreScores[(None, None, None, 1)]

        a_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )

        # Partition KV (A): fmha_decode_paged pattern.
        # Partition KV (A): paged decode uses physical sub-blocks; causal
        # Request-tiled ingress keeps the upstream SMEM/MMA topology while
        # rebasing Q and weights through per-tile request metadata.
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        if cutlass.const_expr(self.uses_contiguous_kv):
            tCgA = thr_mma.partition_A(
                cute.local_tile(
                    mA_mkl,
                    cute.slice_(self.mma_tiler, (None, 0, None)),
                    (None, None, None),
                )
            )
            tAsA_0, tAgA_0 = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                cute.group_modes(sKV_0, 0, 3),
                cute.group_modes(tCgA, 0, 3),
            )
            tAsA_1, tAgA_1 = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                cute.group_modes(sKV_1, 0, 3),
                cute.group_modes(tCgA, 0, 3),
            )
        else:
            # SMEM view is ((tile), num_sub_blocks, stages) - built in __call__.
            # Use .outer (plain layout); swizzle is captured by sKV_0's iterator.
            sKV_0_for_tma = cute.make_tensor(sKV_0.iterator, a_tma_view_layout.outer)
            sKV_1_for_tma = cute.make_tensor(sKV_1.iterator, a_tma_view_layout.outer)
            # GMEM: local_tile by (phys, head), then group first 2 modes into tile.
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
        # Partition Q (B): request-tiled ingress rebases query_tile x H; paged
        # decode indexes the original L=batch dimension.
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

        # Partition Weights: standalone TMA, [N, batch_size] to [N] per stage
        # Partition Weights: retain the same linear SMEM stage while causal
        # selective logits views it as [query_tile, num_heads].
        if cutlass.const_expr(self.uses_request_tiled_q):
            gW = cute.local_tile(
                mW_tma,
                (self.next_n, self.num_heads),
                (None, None),
            )
            tWsW, tWgW = cpasync.tma_partition(
                tma_atom_w,
                0,
                cute.make_layout((1,)),
                cute.group_modes(sW, 0, 2),
                cute.group_modes(gW, 0, 2),
            )
        else:
            tWsW, tWgW = cpasync.tma_partition(
                tma_atom_w,
                0,
                cute.make_layout((1,)),
                cute.group_modes(sW, 0, 2),
                cute.group_modes(mW_tma, 0, 1),
            )

        # Partition Scales: explicit sub_blocks + stages dims.
        # Layout (phys_block_kv, num_sub, stages) K-major with custom strides.
        if cutlass.const_expr(self.uses_contiguous_kv):
            gS = cute.local_tile(mS_tma, (self.block_kv,), coord=(None,))
            sScales_0_for_tma = cute.group_modes(sScales_0, 0, 1)
            sScales_1_for_tma = cute.group_modes(sScales_1, 0, 1)
        else:
            # GMEM: local_tile by phys to match atom's tile size
            gS = cute.local_tile(mS_tma, (self.scale_tma_tile,), coord=(None, None))
            s_tma_view_layout = cute.make_layout(
                (
                    self.phys_block_kv,
                    self.num_blocks_per_mma,
                    self.num_kv_stages,
                ),
                stride=(1, self.phys_block_kv, self.block_kv),
            )
            sScales_0_for_tma = cute.make_tensor(sScales_0.iterator, s_tma_view_layout)
            sScales_1_for_tma = cute.make_tensor(sScales_1.iterator, s_tma_view_layout)
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
        # shape: (*acc_shape, STAGE); dynamic index on last dim reduces rank
        us = self.num_umma_stages
        cols = self.num_tmem_alloc_cols
        acc_shape_staged = cute.append(acc_shape, us)
        tCtAcc_fake_staged = tiled_mma.make_fragment_C(acc_shape_staged)

        # TMEM layout info (allocation deferred to UMMA/Math warp branches)
        cols_per_group = cols * us * (32 // self.acc_dtype.width)
        tmem_group_1_offset = cols_per_group
        num_tmem_alloc_cols_total = self.num_tmem_alloc_cols_total

        # Epilogue setup
        c_layout = utils.LayoutEnum.ROW_MAJOR
        epi_sub_mn = (epi_tile[0], self.N // num_epi_subtiles)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            c_layout,
            self.acc_dtype,
            self.acc_dtype,
            epi_sub_mn,
            use_2cta_instrs,
        )

        # ===== SCHEDULER: derive values from early-loaded schedule metadata =====
        end_kv_idx = end_kv_half * NUM_MATH_WG

        # Convert start to KV block units (matching DeepGEMM)
        current_q_idx = start_q
        current_kv_idx = start_kv_half * NUM_MATH_WG

        # ===== COMMON SCHEDULER STATE (before warp branches) =====
        # Each warp role independently maintains its own copy of these
        # variables (like DeepGEMM where each role creates its own scheduler).
        # Pre-fetch first task (current_num_kv loaded early above for latency hiding)
        next_q_idx = current_q_idx
        next_kv_idx = current_kv_idx
        next_num_kv = current_num_kv
        # Sentinel: no previous batch (matches DeepGEMM's q_idx = batch_size)
        q_idx = batch_size
        # While-loop termination flag (matches DeepGEMM's fetch_next_task pattern).
        # True if this CTA has work assigned (start != end in schedule_meta).
        has_work = (current_q_idx != end_q_idx) | (current_kv_idx != end_kv_idx)

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # ===== WARP-SPECIALIZED EXECUTION =====

        if is_tma_warp_0:
            self._run_tma_warp_0(
                NUM_BLOCKS_PER_MMA,
                NUM_MATH_WG,
                a_mcast_mask,
                b_mcast_mask,
                batch_size,
                block_kv_val,
                end_kv_idx,
                end_q_idx,
                has_work,
                kv_pipeline_0,
                mA_mkl,
                mB_nkl,
                mBlockTable,
                mSamplePages,
                mContextLens,
                mS_tma,
                mTileMeta,
                mW_tma,
                next_kv_idx,
                next_num_kv,
                next_q_idx,
                q_idx,
                q_pipeline,
                sKV_0,
                sQ,
                sScales_0,
                sW,
                tAgA_0,
                tAsA_0,
                tBgB,
                tBsB,
                tSgS_0,
                tSsS_0,
                tWgW,
                tWsW,
                thr_mma,
                tidx,
                tma_atom_a,
                tma_atom_b,
                tma_atom_s,
                tma_atom_w,
            )
        elif is_tma_warp_1:
            self._run_tma_warp_1(
                NUM_BLOCKS_PER_MMA,
                NUM_MATH_WG,
                a_mcast_mask,
                batch_size,
                block_kv_val,
                end_kv_idx,
                end_q_idx,
                has_work,
                kv_pipeline_1,
                mA_mkl,
                mBlockTable,
                mSamplePages,
                mContextLens,
                mS_tma,
                mTileMeta,
                next_kv_idx,
                next_num_kv,
                next_q_idx,
                q_idx,
                sKV_1,
                sScales_1,
                tAgA_1,
                tAsA_1,
                tSgS_1,
                tSsS_1,
                thr_mma,
                tidx,
                tma_atom_a,
                tma_atom_s,
            )
        elif is_umma_warp_0:
            self._run_umma_warp_0(
                NUM_MATH_WG,
                batch_size,
                block_kv_val,
                end_kv_idx,
                end_q_idx,
                has_work,
                is_leader_cta,
                kv_pipeline_0,
                kv_pipeline_1,
                mContextLens,
                next_kv_idx,
                next_num_kv,
                next_q_idx,
                q_idx,
                q_pipeline,
                tCrA_0,
                tCrA_1,
                tCrB,
                tCtAcc_fake_staged,
                tiled_mma,
                tmem,
                tmem_group_1_offset,
                umma_pipeline_0,
                umma_pipeline_1,
            )
        elif is_math_warp:
            if cutlass.const_expr(self.has_store_warp):
                self._run_math_warp(
                    NUM_MATH_WG,
                    batch_size,
                    block_kv_val,
                    copy_atom_t2r,
                    end_kv_idx,
                    end_q_idx,
                    epi_sub_mn,
                    has_work,
                    kv_pipeline_math,
                    mContextLens,
                    mIndexCounts,
                    mLogits,
                    mPolicyValues,
                    mTileMeta,
                    next_kv_idx,
                    next_n,
                    next_num_kv,
                    next_q_idx,
                    num_epi_subtiles,
                    num_heads,
                    num_tmem_alloc_cols_total,
                    q_idx,
                    q_pipeline,
                    sScales,
                    sW,
                    tCtAcc_fake_staged,
                    tidx,
                    tmem,
                    tmem_group_1_offset,
                    umma_pipeline_math,
                    warpgroup_idx,
                    sStoreScores,
                    store_pipeline_math,
                )
            else:
                self._run_math_warp(
                    NUM_MATH_WG,
                    batch_size,
                    block_kv_val,
                    copy_atom_t2r,
                    end_kv_idx,
                    end_q_idx,
                    epi_sub_mn,
                    has_work,
                    kv_pipeline_math,
                    mContextLens,
                    mIndexCounts,
                    mLogits,
                    mPolicyValues,
                    mTileMeta,
                    next_kv_idx,
                    next_n,
                    next_num_kv,
                    next_q_idx,
                    num_epi_subtiles,
                    num_heads,
                    num_tmem_alloc_cols_total,
                    q_idx,
                    q_pipeline,
                    sScales,
                    sW,
                    tCtAcc_fake_staged,
                    tidx,
                    tmem,
                    tmem_group_1_offset,
                    umma_pipeline_math,
                    warpgroup_idx,
                )
        elif is_store_warp:
            if cutlass.const_expr(self.has_store_warp):
                self._run_store_warp(
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
                )
            else:
                self._run_reserved_warp()
        else:
            self._run_reserved_warp()

    @cute.jit
    def __call__(
        self,
        kv_fused: cute.Tensor,
        q: cute.Tensor,
        weights: cute.Tensor,
        values: cute.Tensor,
        block_table: cute.Tensor,
        index_counts: cute.Tensor,
        context_lens: cute.Tensor,
        schedule_meta: cute.Tensor,
        tile_meta: cute.Tensor,
        policy_values: cute.Tensor,
        sample_page_indices: cute.Tensor,
        num_phys_blocks: cutlass.Int32,
        num_tiles: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._launch(
            kv_fused,
            q,
            weights,
            values,
            block_table,
            index_counts,
            context_lens,
            schedule_meta,
            block_table,
            tile_meta,
            policy_values,
            num_phys_blocks,
            num_tiles,
            stream,
            sample_page_indices,
        )

    @cute.jit
    def _store_candidate_warp_row_pair(
        self,
        row_logits_addr,
        q_valid,
        row,
        row_logical_end,
        row_logical_start,
        logical_base_0,
        logical_base_1,
        threshold,
        cursor,
        segment,
        score_0,
        score_1,
    ):
        """Append both ordered math streams for one store segment."""
        row_valid = row < q_valid
        selected_0 = (
            row_valid & (logical_base_0 < row_logical_end) & (score_0 >= threshold)
        )
        selected_1 = (
            row_valid & (logical_base_1 < row_logical_end) & (score_1 >= threshold)
        )
        if cutlass.const_expr(not self.candidate_uses_causal_bounds):
            selected_0 = selected_0 & (logical_base_0 >= row_logical_start)
            selected_1 = selected_1 & (logical_base_1 >= row_logical_start)
        selected_mask_0 = cute.arch.vote_ballot_sync(selected_0)
        selected_mask_1 = cute.arch.vote_ballot_sync(selected_1)
        lane = cute.arch.lane_idx()
        lane_mask = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
        lane_rank_0 = cutlass.Int32(cute.arch.popc(selected_mask_0 & lane_mask))
        lane_rank_1 = cutlass.Int32(cute.arch.popc(selected_mask_1 & lane_mask))
        warp_count_0 = cutlass.Int32(cute.arch.popc(selected_mask_0))
        warp_count_1 = cutlass.Int32(cute.arch.popc(selected_mask_1))
        segment_capacity = cutlass.Int32(self.capacity // self.num_count_segments)
        segment_begin = segment * segment_capacity
        segment_end = (segment + 1) * segment_capacity
        slot_0 = segment_begin + cursor + lane_rank_0
        cursor_1 = cursor + warp_count_0
        slot_1 = segment_begin + cursor_1 + lane_rank_1
        packed_0 = (
            cutlass.Uint64(cutlass.Uint32(fp32_to_i32_bits(score_0))) << 32
        ) | cutlass.Uint64(cutlass.Uint32(logical_base_0))
        packed_1 = (
            cutlass.Uint64(cutlass.Uint32(fp32_to_i32_bits(score_1))) << 32
        ) | cutlass.Uint64(cutlass.Uint32(logical_base_1))
        store_global_u64_if(
            selected_0 & (slot_0 < segment_end),
            row_logits_addr + cutlass.Int64(slot_0) * cutlass.Int64(8),
            packed_0,
        )
        store_global_u64_if(
            selected_1 & (slot_1 < segment_end),
            row_logits_addr + cutlass.Int64(slot_1) * cutlass.Int64(8),
            packed_1,
        )
        return cursor_1 + warp_count_1

    @cute.jit
    def _store_candidate_warp_row_pair_full(
        self,
        row_logits_addr,
        logical_base_0,
        logical_base_1,
        threshold,
        cursor,
        segment,
        score_0,
        score_1,
    ):
        """Append one full KV pair without row or logical-bound predicates."""
        selected_0 = score_0 >= threshold
        selected_1 = score_1 >= threshold
        selected_mask_0 = cute.arch.vote_ballot_sync(selected_0)
        selected_mask_1 = cute.arch.vote_ballot_sync(selected_1)
        lane = cute.arch.lane_idx()
        lane_mask = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
        lane_rank_0 = cutlass.Int32(cute.arch.popc(selected_mask_0 & lane_mask))
        lane_rank_1 = cutlass.Int32(cute.arch.popc(selected_mask_1 & lane_mask))
        warp_count_0 = cutlass.Int32(cute.arch.popc(selected_mask_0))
        warp_count_1 = cutlass.Int32(cute.arch.popc(selected_mask_1))
        segment_capacity = cutlass.Int32(self.capacity // self.num_count_segments)
        segment_begin = segment * segment_capacity
        segment_end = (segment + 1) * segment_capacity
        slot_0 = segment_begin + cursor + lane_rank_0
        cursor_1 = cursor + warp_count_0
        slot_1 = segment_begin + cursor_1 + lane_rank_1
        packed_0 = (
            cutlass.Uint64(cutlass.Uint32(fp32_to_i32_bits(score_0))) << 32
        ) | cutlass.Uint64(cutlass.Uint32(logical_base_0))
        packed_1 = (
            cutlass.Uint64(cutlass.Uint32(fp32_to_i32_bits(score_1))) << 32
        ) | cutlass.Uint64(cutlass.Uint32(logical_base_1))
        store_global_u64_if(
            selected_0 & (slot_0 < segment_end),
            row_logits_addr + cutlass.Int64(slot_0) * cutlass.Int64(8),
            packed_0,
        )
        store_global_u64_if(
            selected_1 & (slot_1 < segment_end),
            row_logits_addr + cutlass.Int64(slot_1) * cutlass.Int64(8),
            packed_1,
        )
        return cursor_1 + warp_count_1

    @cute.jit
    def _store_candidate_lane_row_pair(
        self,
        row_logits_addr,
        q_valid,
        row,
        row_logical_end,
        row_logical_start,
        logical_base_0,
        logical_base_1,
        threshold,
        cursor,
        segment,
        score_0,
        score_1,
    ):
        """Append one exact KV pair into a fixed store-lane slice."""
        row_valid = row < q_valid
        selected_0 = (
            row_valid & (logical_base_0 < row_logical_end) & (score_0 >= threshold)
        )
        selected_1 = (
            row_valid & (logical_base_1 < row_logical_end) & (score_1 >= threshold)
        )
        if cutlass.const_expr(not self.candidate_uses_causal_bounds):
            selected_0 = selected_0 & (logical_base_0 >= row_logical_start)
            selected_1 = selected_1 & (logical_base_1 >= row_logical_start)
        segment_capacity = cutlass.Int32(self.capacity // self.num_count_segments)
        segment_begin = segment * segment_capacity
        segment_end = segment_begin + segment_capacity
        slot_0 = segment_begin + cursor
        cursor_1 = cursor + cutlass.Int32(selected_0)
        slot_1 = segment_begin + cursor_1
        cursor_2 = cursor_1 + cutlass.Int32(selected_1)
        packed_0 = (
            cutlass.Uint64(cutlass.Uint32(fp32_to_i32_bits(score_0))) << 32
        ) | cutlass.Uint64(cutlass.Uint32(logical_base_0))
        packed_1 = (
            cutlass.Uint64(cutlass.Uint32(fp32_to_i32_bits(score_1))) << 32
        ) | cutlass.Uint64(cutlass.Uint32(logical_base_1))
        store_global_u64_if(
            selected_0 & (slot_0 < segment_end),
            row_logits_addr + cutlass.Int64(slot_0) * cutlass.Int64(8),
            packed_0,
        )
        store_global_u64_if(
            selected_1 & (slot_1 < segment_end),
            row_logits_addr + cutlass.Int64(slot_1) * cutlass.Int64(8),
            packed_1,
        )
        return cursor_2

    @cute.jit
    def _store_candidate_lane_row_pair_full(
        self,
        row_logits_addr,
        logical_base_0,
        logical_base_1,
        threshold,
        cursor,
        segment,
        score_0,
        score_1,
    ):
        """Append one full KV pair into a fixed store-lane slice."""
        selected_0 = score_0 >= threshold
        selected_1 = score_1 >= threshold
        segment_capacity = cutlass.Int32(self.capacity // self.num_count_segments)
        segment_begin = segment * segment_capacity
        segment_end = segment_begin + segment_capacity
        slot_0 = segment_begin + cursor
        cursor_1 = cursor + cutlass.Int32(selected_0)
        slot_1 = segment_begin + cursor_1
        cursor_2 = cursor_1 + cutlass.Int32(selected_1)
        packed_0 = (
            cutlass.Uint64(cutlass.Uint32(fp32_to_i32_bits(score_0))) << 32
        ) | cutlass.Uint64(cutlass.Uint32(logical_base_0))
        packed_1 = (
            cutlass.Uint64(cutlass.Uint32(fp32_to_i32_bits(score_1))) << 32
        ) | cutlass.Uint64(cutlass.Uint32(logical_base_1))
        store_global_u64_if(
            selected_0 & (slot_0 < segment_end),
            row_logits_addr + cutlass.Int64(slot_0) * cutlass.Int64(8),
            packed_0,
        )
        store_global_u64_if(
            selected_1 & (slot_1 < segment_end),
            row_logits_addr + cutlass.Int64(slot_1) * cutlass.Int64(8),
            packed_1,
        )
        return cursor_2

    @cute.jit
    def _ordered_key(self, score):
        bits = cutlass.Uint32(fp32_to_i32_bits(score))
        # Keep the sign dependency explicit across guarded policy-word loads:
        # negative scores invert every bit; nonnegative scores flip the sign bit.
        # Arithmetic sign extension avoids a select in the ordering transform.
        sign_mask = cutlass.Uint32(cutlass.Int32(bits) >> 31) | cutlass.Uint32(
            0x80000000
        )
        return bits ^ sign_mask

    @cute.jit
    def _store_result_packed_causal_meta(
        self,
        mLogits,
        mIndexCounts,
        mTileMeta,
        mPolicyValues,
        q_idx,
        out_row_base,
        q_valid,
        causal_base,
        row_start,
        row_end,
        row,
        store_kv_pos,
        store_kv_valid,
        result,
        scale_val,
        next_n: cutlass.Constexpr,
    ):
        out_row = out_row_base + row
        if cutlass.const_expr(
            self.mode == _SelectiveLogitsMode.CANDIDATE
            and self.candidate_uses_causal_bounds
        ):
            # Candidate rows are request-owned causal tiles; retain the upstream
            # hoisted base and avoid two bound loads in the hot path.
            valid = (
                (row < q_valid)
                & store_kv_valid
                & (store_kv_pos < causal_base + row + 1)
            )
        else:
            valid = (
                (row < q_valid)
                & store_kv_valid
                & (store_kv_pos >= row_start)
                & (store_kv_pos < row_end)
            )
        score = cutlass.Float32(result) * cutlass.Float32(scale_val)

        policy_word = cutlass.Int32(0)
        if cutlass.const_expr(self.mode != _SelectiveLogitsMode.SAMPLE):
            # Policy storage has logical rows, unlike the padded score arena.
            # Guard only the load: inactive lanes still join warp collectives.
            if row < q_valid:
                policy_word = mPolicyValues[out_row]

        if cutlass.const_expr(self.mode == _SelectiveLogitsMode.SAMPLE):
            if valid & (store_kv_pos < self.capacity):
                mLogits[(out_row, store_kv_pos)] = score
        elif cutlass.const_expr(self.mode == _SelectiveLogitsMode.RADIX_HISTOGRAM):
            ordered = self._ordered_key(score)
            prefix = cutlass.Uint32(policy_word)
            prefix_matches = cutlass.Boolean(True)
            if cutlass.const_expr(self.radix_shift != 24):
                prefix_matches = (ordered >> (self.radix_shift + 8)) == prefix
            if valid & prefix_matches:
                radix_bin = cutlass.Int32((ordered >> self.radix_shift) & 0xFF)
                cute.arch.atomic_add(
                    cute.domain_offset((out_row, radix_bin), mIndexCounts).iterator,
                    cutlass.Int32(1),
                    sem="relaxed",
                    scope="gpu",
                )
        elif cutlass.const_expr(self.mode == _SelectiveLogitsMode.EXACT):
            score_key = self._ordered_key(score)
            threshold_key = self._ordered_key(i32_bits_to_fp32(policy_word))
            lane = cute.arch.lane_idx()
            lane_mask = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
            bank_rows = mLogits.shape[0] // 2
            for bank in cutlass.range_constexpr(2):
                selected = valid & (score_key > threshold_key)
                if cutlass.const_expr(bank == 1):
                    selected = valid & (score_key == threshold_key)
                mask = cute.arch.vote_ballot_sync(selected)
                count = cutlass.Int32(cute.arch.popc(mask))
                rank = cutlass.Int32(cute.arch.popc(mask & lane_mask))
                base = cutlass.Int32(0)
                bank_row = out_row + bank * bank_rows
                if (lane == 0) & (count != 0):
                    base = cute.arch.atomic_add(
                        cute.domain_offset((bank_row, 0), mIndexCounts).iterator,
                        count,
                        sem="relaxed",
                        scope="gpu",
                    )
                slot = cute.arch.shuffle_sync(base, 0) + rank
                if selected & (slot < self.capacity):
                    mLogits[(bank_row, slot)] = store_kv_pos
        else:
            selected = valid & (score >= i32_bits_to_fp32(policy_word))

            segment_begin = cutlass.Int32(0)
            segment_end = cutlass.Int32(self.capacity)
            slot = cutlass.Int32(0)
            if cutlass.const_expr(
                self.num_count_segments > 0
                and (
                    self.mode == _SelectiveLogitsMode.CANDIDATE
                    or self.mode == _SelectiveLogitsMode.REPAIR
                )
            ):
                # Match the segmented producer's fixed store-warp
                # segments: a 384-token split is striped across 12 warps in
                # consecutive 32-token lanes.  The retained set at overflow is
                # part of the packed consumer ABI, so contiguous KV ranges are
                # not interchangeable here.
                segment = (store_kv_pos // self.count_segment_span) % cutlass.Int32(
                    self.num_count_segments
                )
                segment_begin = segment * self.capacity // self.num_count_segments
                segment_end = (segment + 1) * self.capacity // self.num_count_segments
                if selected:
                    slot = segment_begin + cute.arch.atomic_add(
                        cute.domain_offset(
                            (out_row, segment + 1), mIndexCounts
                        ).iterator,
                        cutlass.Int32(1),
                        sem="relaxed",
                        scope="gpu",
                    )
            else:
                selected_mask = cute.arch.vote_ballot_sync(selected)
                warp_count = cutlass.Int32(cute.arch.popc(selected_mask))
                lane = cute.arch.lane_idx()
                lane_mask = (
                    cutlass.Uint32(1) << cutlass.Uint32(lane)
                ) - cutlass.Uint32(1)
                lane_rank = cutlass.Int32(cute.arch.popc(selected_mask & lane_mask))
                warp_base = cutlass.Int32(0)
                if lane == cutlass.Int32(0) and warp_count != cutlass.Int32(0):
                    warp_base = cute.arch.atomic_add(
                        cute.domain_offset((out_row, 0), mIndexCounts).iterator,
                        warp_count,
                        sem="relaxed",
                        scope="gpu",
                    )
                warp_base = cute.arch.shuffle_sync(warp_base, cutlass.Int32(0))
                slot = warp_base + lane_rank
            if selected:
                if slot < segment_end:
                    # The packed consumer uses request-local logical indices.
                    logical_index = store_kv_pos
                    if cutlass.const_expr(
                        self.mode == _SelectiveLogitsMode.CANDIDATE
                        or self.mode == _SelectiveLogitsMode.REPAIR
                    ):
                        packed = (
                            cutlass.Uint64(cutlass.Uint32(fp32_to_i32_bits(score)))
                            << 32
                        ) | cutlass.Uint64(cutlass.Uint32(logical_index))
                        mLogits[(out_row, slot)] = cutlass.Int64(packed)
                    else:
                        mLogits[(out_row, slot)] = logical_index


# Warp w of a row warpgroup publishes the KV lanes it reads, one warp-wide
# span per compute tile, into candidate segment w.
_H64_CANDIDATE_SEGMENTS = 4
_H64_CANDIDATE_SEGMENT_SPAN = 32
# One page-table load per lane covers this many KV tiles, which the math role
# also unrolls statically so scale loads and TMEM reads of later tiles overlap.
_H64_CANDIDATE_PAGE_CACHE = 32
# Q4/N256 with one KV stream and two UMMA stages (see the class docstring).
H64_CANDIDATE_TILING = replace(
    SELECTIVE_LOGITS_TILING,
    next_n=4,
    num_q_stages=2,
    num_kv_stages=4,
    num_umma_stages=2,
    num_epi_subtiles=2,
)


class FP8MQASelectiveH64CandidateKernel(FP8MQASelectiveLogitsKernel):
    """Specialized H64 CANDIDATE variant: one query row per math warpgroup.

    :class:`FP8MQASelectiveLogitsKernel` scores an H64 Q4 tile (N = 4 rows x 64
    heads) with two KV streams of eight math warps. That fits one UMMA stage, and
    each math thread owns all four query rows of its KV lane, so only 16 of a row's
    64 head weights stay in registers; the other 48 are reread from shared memory
    for every key, and a larger register cache spills.
    For single-pool H64/D128 long-query prefill this variant keeps the same
    request-tiled launch, metadata, arithmetic and publication ABI, and changes
    only how the CTA's warps split the tile:

      - warps 0-15: four math warpgroups. Warpgroup ``r`` owns query row ``r`` of
        the tile and keeps that row's 64 weights in registers; its 128 threads
        read one TMEM lane (KV position) each.
      - warp 16: TMA producer for the request-tiled Q/weights and paged KV.
      - warp 18: UMMA producer. One KV stream leaves TMEM for two accumulator
        stages (2 x N256 = 512 columns), so math overlaps the next tile's MMA.

    Publication is warp-ranked: warp ``w`` of a row appends the keys of TMEM lanes
    ``32w..32w+31`` into segment ``w`` of the 4-segment candidate ABI
    (``values[row, w * span + i]``, count in ``index_counts[row, 1 + w]``), so the
    segmented PackedTopK, sample ladder and both repair stages are unchanged. A
    segment that fills before the row's total does is an overflow and is repaired
    like any other.

    Scores use the engine's head reduction (:meth:`_accumulate_fp32_heads`) and
    KV-scale policy, so they match the general layout bit for bit in both math
    modes. Tiles, page-table rows and exact row bounds come from the prepared
    ``tile_meta``/``context_lens``. KV and its FP32 scales are read in place from
    the fused paged pool, one page per KV compute tile.

    Construction, validation and the call ABI are the general kernel's
    four-segment warp-striped CANDIDATE topology (``split_kv`` 1,
    ``count_segment_span`` of one warp); only the launch and warp roles differ.
    Requires pages of exactly one compute tile.
    """

    # 640 threads leave at most 102 registers per thread.
    ptxas_max_registers = 96

    def __init__(
        self,
        *,
        num_sms: int,
        capacity: int,
        math_mode: str = "relu_strict",
    ):
        super().__init__(
            num_heads=64,
            head_dim=128,
            phys_block_kv=_COMPUTE_BLOCK_KV,
            num_sms=num_sms,
            mode=_SelectiveLogitsMode.CANDIDATE,
            capacity=capacity,
            split_kv=1,
            num_count_segments=_H64_CANDIDATE_SEGMENTS,
            count_segment_span=_H64_CANDIDATE_SEGMENT_SPAN,
            tiling=H64_CANDIDATE_TILING,
            math_mode=math_mode,
        )
        # One KV stream; no dedicated store warps: math warps publish directly.
        self.num_groups = 1
        self.has_store_warp = False
        self.threads_per_cta = 640

    @cute.jit
    def _launch(
        self,
        kv_fused: cute.Tensor,
        b: cute.Tensor,
        weights: cute.Tensor,
        logits: cute.Tensor,
        block_table: cute.Tensor,
        index_counts: cute.Tensor,
        context_lens: cute.Tensor,
        schedule_meta: cute.Tensor,
        kv_scale: cute.Tensor,
        tile_meta: cute.Tensor,
        policy_values: cute.Tensor,
        num_phys_blocks: cutlass.Int32,
        batch_size: cutlass.Int32,
        stream: cuda.CUstream,
        sample_page_indices: cute.Tensor = None,
    ):
        # Persistent CTAs walk query tiles directly: schedule_meta is the
        # general layout's KV-pair plan, and scales are read from the fused
        # pages, so neither it nor kv_scale or sample pages are used here.
        kv_fp8 = cute.recast_tensor(kv_fused, cutlass.Float8E4M3FN)
        b = cute.recast_tensor(b, cutlass.Float8E4M3FN)
        page_stride = kv_fused.layout.stride[0]
        # Each fused page holds one compute tile of K rows, then their FP32 scales.
        a = cute.make_tensor(
            kv_fp8.iterator,
            cute.make_layout(
                (self.block_kv, self.head_dim, num_phys_blocks),
                stride=(self.head_dim, 1, page_stride),
            ),
        )
        fused_pages = cute.make_tensor(
            kv_fp8.iterator,
            cute.make_layout(
                (self.block_kv * (self.head_dim + 4), num_phys_blocks),
                stride=(1, page_stride),
            ),
        )
        scale_bytes = cute.domain_offset(
            (self.block_kv * self.head_dim, 0), fused_pages
        )
        scales = cute.recast_tensor(
            cute.make_tensor(
                scale_bytes.iterator,
                cute.make_layout(
                    (self.block_kv * 4, num_phys_blocks), stride=(1, page_stride)
                ),
            ),
            cutlass.Float32,
        )

        a_major = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        b_major = utils.LayoutEnum.ROW_MAJOR.mma_major_mode()
        tiled_mma = self._setup_mma(a.element_type, b.element_type, a_major, b_major)
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        a_tma_view = cute.select(
            cute.tiled_divide(
                sm100_utils.make_smem_layout(
                    cute.nvgpu.OperandMajorMode.K,
                    (self.block_kv, self.head_dim),
                    a.element_type,
                    self.num_kv_stages,
                ),
                (self.block_kv, self.head_dim),
            ),
            mode=[0, 1, 3],
        )
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_load_op, a, a_tma_view[0], (self.block_kv, self.head_dim)
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mn, tiled_mma.thr_id
            ),
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        w_smem_layout = cute.make_layout(
            (self.N, self.num_q_stages), stride=(1, self.w_stage_stride)
        )
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_op,
            weights,
            cute.make_layout((self.next_n, self.num_heads), stride=(self.num_heads, 1)),
            (self.next_n, self.num_heads),
        )
        self.num_q_tma_bytes = cute.size_in_bytes(b.element_type, b_smem_layout) + (
            self.N * 4
        )
        self.num_kv_tma_bytes = self.block_kv * self.head_dim

        @cute.struct
        class SharedStorage:
            kv_mbar: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.num_q_stages * 2]
            umma_mbar: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 2]
            tmem_holding_buf: cutlass.Int32

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_w,
            tma_tensor_w,
            scales,
            block_table,
            logits,
            index_counts,
            context_lens,
            tile_meta,
            policy_values,
            batch_size,
            a_tma_view,
            w_smem_layout,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            SharedStorage,
        ).launch(
            grid=(1, 1, min(batch_size, self.num_sms)),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=0,
        )

    @cute.jit
    def _score_kv_tile(
        self,
        kv_idx,
        page,
        mScales,
        m_coord,
        query_slot,
        row_start,
        row_end,
        threshold,
        cursor,
        segment_rows,
        span,
        w_cache,
        tCtAcc_base,
        thr_copy,
        tiled_copy,
        tTR_rAcc,
        epi_sub_mn,
        umma_pipeline,
        umma_state,
    ):
        """Score one KV tile for this warpgroup's row and append its hits."""
        kv_pos = kv_idx * self.block_kv + m_coord
        ready = umma_pipeline.consumer_try_wait(umma_state)
        # The scale load overlaps the remaining MMA latency.
        scale = mScales[(m_coord, page)]
        if cutlass.const_expr(self.math_mode == "relu_fast"):
            scale = scale * cutlass.Float32(0.5)
        umma_pipeline.consumer_wait(umma_state, ready)
        tTR = thr_copy.partition_S(
            cute.flat_divide(
                tCtAcc_base[(None, None, None, umma_state.index)][((None, None), 0, 0)],
                epi_sub_mn,
            )
        )
        sums = (
            cutlass.Float32(0.0),
            cutlass.Float32(0.0),
            cutlass.Float32(0.0),
            cutlass.Float32(0.0),
        )
        heads_per_subtile = self.num_heads // self.num_epi_subtiles
        for i in cutlass.range_constexpr(self.num_epi_subtiles):
            cute.copy(
                tiled_copy,
                tTR[(None, None, None, 0, query_slot * self.num_epi_subtiles + i)],
                tTR_rAcc,
            )
            cute.arch.fence_view_async_tmem_load()
            if cutlass.const_expr(i == self.num_epi_subtiles - 1):
                # Every row's slice of this stage is in registers.
                umma_pipeline.consumer_release(umma_state)
                umma_state.advance()
            acc = tTR_rAcc.load()
            for h in cutlass.range_constexpr(0, heads_per_subtile, 4):
                head = i * heads_per_subtile + h
                sums = self._accumulate_fp32_heads(
                    acc, h, w_cache, (head, head + 1, head + 2, head + 3), sums
                )
        s0x, s0y, s1x, s1y = sums
        sum_x, sum_y = cute.arch.add_packed_f32x2((s0x, s0y), (s1x, s1y))
        score = cutlass.Float32(sum_x + sum_y) * cutlass.Float32(scale)
        hit = (kv_pos >= row_start) & (kv_pos < row_end) & (score >= threshold)
        mask = cute.arch.vote_ballot_sync(hit)
        if mask != 0:
            lane_mask = (cutlass.Uint32(1) << cute.arch.lane_idx()) - 1
            slot = cursor + cutlass.Int32(cute.arch.popc(mask & lane_mask))
            if hit & (slot < span):
                packed = (
                    cutlass.Uint64(cutlass.Uint32(fp32_to_i32_bits(score))) << 32
                ) | cutlass.Uint64(cutlass.Uint32(kv_pos))
                segment_rows[slot] = cutlass.Int64(packed)
            cursor = cursor + cutlass.Int32(cute.arch.popc(mask))
        return cursor, umma_state

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_w: cute.CopyAtom,
        mW_tma: cute.Tensor,
        mScales: cute.Tensor,
        mBlockTable: cute.Tensor,
        mLogits: cute.Tensor,
        mIndexCounts: cute.Tensor,
        mContextLens: cute.Tensor,
        mTileMeta: cute.Tensor,
        mPolicyValues: cute.Tensor,
        num_tiles: cutlass.Int32,
        a_tma_view: cute.ComposedLayout,
        w_smem_layout: cute.Layout,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        _, _, bidz = cute.arch.block_idx()
        grid = cute.arch.grid_dim()[2]
        warpgroup_idx = warp_idx // 4
        is_math_warp = warp_idx < 16
        is_tma_warp = warp_idx == 16
        is_umma_warp = warp_idx == 18
        last_column = mBlockTable.shape[1] - 1

        if is_tma_warp:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_w)

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        producer = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        # The 16 math warps release a Q/weight stage after the tile's last KV.
        q_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_mbar.data_ptr(),
            num_stages=self.num_q_stages,
            producer_group=producer,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 16),
            tx_count=self.num_q_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        # Math reads scales from the pool, so the MMA commit alone frees a stage.
        kv_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.kv_mbar.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=producer,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.num_kv_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        umma_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar.data_ptr(),
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 512),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # Math warp 0 allocates; all 16 math warps and the UMMA warp retrieve.
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=544),
            allocator_warp_id=0,
            is_two_cta=False,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sKV = smem.allocate_tensor(
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
        sW = smem.allocate_tensor(
            element_type=cutlass.Float32, layout=w_smem_layout, byte_alignment=128
        )
        # The same linear stage viewed as [query row, head, stage].
        sW = cute.make_tensor(
            sW.iterator,
            cute.make_layout(
                (self.next_n, self.num_heads, self.num_q_stages),
                stride=(self.num_heads, 1, self.w_stage_stride),
            ),
        )

        thr_mma = tiled_mma.get_slice(0)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.make_tensor(sKV.iterator, a_tma_view.outer),
            cute.group_modes(
                cute.local_tile(
                    mA, (self.block_kv, self.head_dim), coord=(None, None, None)
                ),
                0,
                2,
            ),
        )
        tBsB, _ = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(
                thr_mma.partition_B(
                    cute.local_tile(
                        mB_nkl,
                        cute.slice_(self.mma_tiler, (0, None, None)),
                        (None, None, None),
                    )
                ),
                0,
                3,
            ),
        )
        tWsW, _ = cpasync.tma_partition(
            tma_atom_w,
            0,
            cute.make_layout((1,)),
            cute.group_modes(sW, 0, 2),
            cute.group_modes(
                cute.local_tile(mW_tma, (self.next_n, self.num_heads), (None, None)),
                0,
                2,
            ),
        )
        tCrA = tiled_mma.make_fragment_A(sKV)
        tCrB = tiled_mma.make_fragment_B(sQ)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(
                tiled_mma.partition_shape_C(self.mma_tiler[:2]), self.num_umma_stages
            )
        )
        epi_sub_mn = (self.block_kv, self.num_heads // self.num_epi_subtiles)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            utils.LayoutEnum.ROW_MAJOR,
            self.acc_dtype,
            self.acc_dtype,
            epi_sub_mn,
            False,
        )
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # Descending tiles reach a long causal request's longest rows first.
        # Every role walks the same tile sequence and per-tile KV extent.
        first_tile = num_tiles - 1 - bidz
        if is_tma_warp:
            q_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_q_stages
            )
            kv_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_kv_stages
            )
            tile = first_tile
            while tile >= 0:
                q_pipeline.producer_acquire(q_state)
                q_bar = q_pipeline.producer_get_barrier(q_state)
                q_source, w_source = self._tma_q_source_views(
                    tile,
                    True,
                    mTileMeta,
                    mB_nkl,
                    mW_tma,
                    thr_mma,
                    sQ,
                    sW,
                    tma_atom_b,
                    tma_atom_w,
                    None,
                    None,
                )
                cute.copy(
                    tma_atom_b, q_source, tBsB[(None, q_state.index)], tma_bar_ptr=q_bar
                )
                cute.copy(
                    tma_atom_w, w_source, tWsW[(None, q_state.index)], tma_bar_ptr=q_bar
                )
                q_state.advance()
                page_row = mTileMeta[(tile, 1)]
                num_kv = (mContextLens[tile] + self.block_kv - 1) // self.block_kv
                chunk_base = cutlass.Int32(0)
                while chunk_base < num_kv:
                    # Lanes past the table read its last column, which no live
                    # tile uses.
                    page_cache = mBlockTable[
                        (
                            page_row,
                            min(chunk_base + cute.arch.lane_idx(), last_column),
                        )
                    ]
                    for chunk_slot in cutlass.range(
                        min(num_kv - chunk_base, _H64_CANDIDATE_PAGE_CACHE), unroll=1
                    ):
                        page = cute.arch.shuffle_sync(page_cache, chunk_slot)
                        kv_pipeline.producer_acquire(kv_state)
                        cute.copy(
                            tma_atom_a,
                            tAgA[(None, 0, 0, page)],
                            tAsA[(None, 0, kv_state.index)],
                            tma_bar_ptr=kv_pipeline.producer_get_barrier(kv_state),
                        )
                        kv_state.advance()
                    chunk_base = chunk_base + _H64_CANDIDATE_PAGE_CACHE
                tile = tile - grid
        elif is_umma_warp:
            tmem.wait_for_alloc()
            tCtAcc_base = cute.make_tensor(
                tmem.retrieve_ptr(self.acc_dtype), tCtAcc_fake.layout
            )
            q_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_q_stages
            )
            kv_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_kv_stages
            )
            umma_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_umma_stages
            )
            tile = first_tile
            while tile >= 0:
                q_pipeline.consumer_wait(q_state)
                num_kv = (mContextLens[tile] + self.block_kv - 1) // self.block_kv
                for _ in cutlass.range(num_kv, unroll=1):
                    kv_pipeline.consumer_wait(kv_state)
                    umma_pipeline.producer_acquire(umma_state)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    tCtAcc = tCtAcc_base[(None, None, None, umma_state.index)]
                    for k_block in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[None, None, k_block, kv_state.index],
                            tCrB[None, None, k_block, q_state.index],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    kv_pipeline.consumer_release(kv_state)
                    kv_state.advance()
                    umma_pipeline.producer_commit(umma_state)
                    umma_state.advance()
                # The math warps release this Q stage after their last tile.
                q_state.advance()
                tile = tile - grid
        elif is_math_warp:
            tmem.allocate(self.num_tmem_alloc_cols_total)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            local_tidx = tidx % 128
            segment = local_tidx // 32
            query_slot = warpgroup_idx
            tAcc_ref = tCtAcc_base[(None, None, None, 0)][((None, None), 0, 0)]
            tiled_copy = tcgen05.make_tmem_copy(
                copy_atom_t2r,
                cute.flat_divide(tAcc_ref, epi_sub_mn)[(None, None, 0, 0)],
            )
            thr_copy = tiled_copy.get_slice(local_tidx)
            tTR_cC = thr_copy.partition_D(cute.make_identity_tensor(epi_sub_mn))
            m_coord = tTR_cC[0][0]
            tTR_rAcc = cute.make_rmem_tensor_like(tTR_cC, self.acc_dtype)
            w_cache = cute.make_rmem_tensor((self.num_heads,), cutlass.Float32)
            span = self.capacity // _H64_CANDIDATE_SEGMENTS
            q_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_q_stages
            )
            umma_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_umma_stages
            )
            tile = first_tile
            while tile >= 0:
                q_pipeline.consumer_wait(q_state)
                for h in cutlass.range_constexpr(self.num_heads):
                    w_cache[h] = sW[(query_slot, h, q_state.index)]
                active = query_slot < mTileMeta[(tile, 2)]
                row = mTileMeta[(tile, 0)] + query_slot
                # Inactive slots keep an empty [0, 0) range and never publish.
                row_start = mTileMeta[(tile, 4 + query_slot)]
                row_end = mTileMeta[(tile, 4 + self.next_n + query_slot)]
                threshold = cutlass.Float32(float("inf"))
                if active:
                    threshold = i32_bits_to_fp32(mPolicyValues[row])
                segment_rows = cute.make_tensor(
                    mLogits.iterator
                    + cutlass.Int64(row) * mLogits.layout.stride[0]
                    + segment * span,
                    cute.make_layout(span),
                )
                page_row = mTileMeta[(tile, 1)]
                num_kv = (mContextLens[tile] + self.block_kv - 1) // self.block_kv
                cursor = cutlass.Int32(0)
                chunk_base = cutlass.Int32(0)
                while chunk_base < num_kv:
                    page_cache = mBlockTable[
                        (
                            page_row,
                            min(chunk_base + cute.arch.lane_idx(), last_column),
                        )
                    ]
                    chunk = min(num_kv - chunk_base, _H64_CANDIDATE_PAGE_CACHE)
                    if chunk == _H64_CANDIDATE_PAGE_CACHE:
                        # A static unroll overlaps later tiles' scale loads and
                        # TMEM reads.
                        for chunk_slot in cutlass.range_constexpr(
                            _H64_CANDIDATE_PAGE_CACHE
                        ):
                            cursor, umma_state = self._score_kv_tile(
                                chunk_base + chunk_slot,
                                cute.arch.shuffle_sync(page_cache, chunk_slot),
                                mScales,
                                m_coord,
                                query_slot,
                                row_start,
                                row_end,
                                threshold,
                                cursor,
                                segment_rows,
                                span,
                                w_cache,
                                tCtAcc_base,
                                thr_copy,
                                tiled_copy,
                                tTR_rAcc,
                                epi_sub_mn,
                                umma_pipeline,
                                umma_state,
                            )
                    else:
                        for chunk_slot in cutlass.range(chunk, unroll=1):
                            cursor, umma_state = self._score_kv_tile(
                                chunk_base + chunk_slot,
                                cute.arch.shuffle_sync(page_cache, chunk_slot),
                                mScales,
                                m_coord,
                                query_slot,
                                row_start,
                                row_end,
                                threshold,
                                cursor,
                                segment_rows,
                                span,
                                w_cache,
                                tCtAcc_base,
                                thr_copy,
                                tiled_copy,
                                tTR_rAcc,
                                epi_sub_mn,
                                umma_pipeline,
                                umma_state,
                            )
                    chunk_base = chunk_base + _H64_CANDIDATE_PAGE_CACHE
                if active & (cute.arch.lane_idx() == 0):
                    # A raw count above span is an overflow; PackedTopK flags it.
                    mIndexCounts[(row, segment + 1)] = cursor
                q_pipeline.consumer_release(q_state)
                q_state.advance()
                tile = tile - grid
            # Every math warp has finished its TMEM loads before the free.
            pipeline.NamedBarrier(barrier_id=2, num_threads=512).arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
