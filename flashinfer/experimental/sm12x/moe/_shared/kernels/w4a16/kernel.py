# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/moe/fused/w4a16/kernel.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""CuTeDSL W4A16 NVFP4/BF16 W4A16 MoE kernels."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from functools import partial
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cuda.bindings.runtime as cuda_runtime
import cutlass
import cutlass.cute as cute
import torch
from cutlass.base_dsl.compiler import OptLevel
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, T, Uint32, dsl_user_op

from flashinfer.experimental.sm12x._lib.compiler import (
    KernelCompileSpec,
    compile as sm12x_compile,
)
from flashinfer.experimental.sm12x._lib.intrinsics import (
    atomic_add_global_i32,
    bf16_mma_m16n8k16_f32,
    bf16_mma_rhs_fragments_as_mma_a_m16n8k16_f32,
    bfloat2_broadcast_lane,
    bfloat2_mul,
    bfloat2_to_float2_scaled,
    broadcast_f32_to_half2,
    broadcast_f32_to_bfloat2,
    cp_async4_shared_global,
    cp_async4_shared_global_pred,
    fabs_f32,
    fmax_f32,
    f16_mma_m16n8k16_f32,
    f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32,
    half2_to_float2_scaled,
    get_ptr_as_int64,
    half2_mul,
    ld_global_acquire_i32,
    ld_global_nc_u32,
    ld_global_v4_f32,
    ld_shared_f32,
    ld_shared_i32_relaxed,
    ld_shared_u32,
    ld_shared_v2_u32,
    ld_shared_v4_f32,
    ld_shared_v4_u32,
    ldmatrix_m8n8x4_b16,
    ldmatrix_m8n8x2_b16,
    packed_dequant_e2m1x4_to_bfloat2x2,
    packed_dequant_e2m1x4_to_half2x2,
    packed_dequant_e4m3x4_to_bfloat2x2,
    packed_dequant_e4m3x4_to_half2x2,
    packed_dequant_e8m0x4_to_bfloat2x2,
    packed_dequant_e8m0x4_to_half2x2,
    packed_dequant_nf3x8_to_bfloat2x4,
    nf3_codebook_pools,
    pack_f32x2_to_bfloat2,
    pack_f32x2_to_f16x2,
    red_add_global_bf16x2,
    red_add_global_release_i32,
    red_max_global_f32_nonnegative,
    shared_ptr_to_u32,
    st_global_v4_u32,
    st_global_i32,
    st_global_v4_f32,
    st_shared_bf16_from_f32,
    st_shared_f16_from_f32,
    st_shared_f32,
    st_shared_i32,
    st_shared_u32,
    st_shared_v4_f32,
    st_shared_v4_u32,
    threadfence,
    warp_reduce,
)
from flashinfer.experimental.sm12x._lib.utils import current_cuda_stream, make_ptr
from flashinfer.experimental.sm12x.moe._shared.kernels.w4a16.route_pack import (
    pack_topk_routes_by_expert as _pack_topk_routes_by_expert,
)
from flashinfer.experimental.sm12x.moe._shared.kernels.w4a16.host import (
    _W4A16_ALLOWED_ROUTED_SIZES,
    max_packed_route_slots,
    packed_gemm_scratch_elements,
    plan_w4a16_buffers,
    select_route_block_size_m,
    validate_activation,
)
from flashinfer.experimental.sm12x._lib.runtime_control import (
    raise_if_kernel_resolution_frozen,
)
from flashinfer.experimental.sm12x.moe._shared.kernels.activations import (
    SWIGLUOAI_UNINTERLEAVE,
    is_gated_moe_activation,
    normalize_moe_activation,
    normalize_swiglu_alpha_for_activation,
    normalize_swiglu_beta_for_activation,
    normalize_swiglu_limit_for_activation,
)
from flashinfer.experimental.sm12x.moe._shared.kernels.micro import (
    MoEMicroKernelBackend,
)


_ALLOWED_ROUTED_SIZES = _W4A16_ALLOWED_ROUTED_SIZES
_PACK_FACTOR = 8
_STAGES = 4
_E8M0_LOGICAL_TAIL_SCALE_N_ALIGNMENT = 64
_DEVICE_MAX_REG_BYTES = 255 * 1024
_DEFAULT_MAX_SHARED_MEM = 101_376
_SCALAR_ACC_FRAGMENT_WIDTH = 1
_WEIGHT_LAYOUTS = {"packed", "modelopt", "nf3_2p1"}
_MODEL_OPT_W13_LAYOUTS = {"w13", "w31"}
# NF3 codebook: 8 bf16 codepoints for the 3-bit ("nf3_2p1") weight layout. The
# device dequant reads these as prmt pools (see nf3_codebook_pools); the host
# injects the 4 pool words as compile-time constants into the GEMM.
_NF3_CODEBOOK = (-1.0, -0.6047, -0.3563, -0.1275, 0.1275, 0.3563, 0.6047, 1.0)
_SCALE_FORMATS = {
    "e4m3_k16": "e4m3_k16",
    "e8m0_k32": "e8m0_k32",
    "e4m3_k32": "e4m3_k32",
}
_E8M0_K32_FP16_GLOBAL_COMPENSATION = float(2.0**7)
_E8M0_K32_BF16_GLOBAL_COMPENSATION = float(2.0**119)
_MAX_DIRECT_TOPK_ROUTE_M = 6
_W4A16_SMALL_M_DIRECT_MAX_M = 8

# TC-decode: a small-M decode specialization that runs on the PACKED W4A16
# object (the same weights/scales the prefill GEMM uses). It reuses the packed
# tensor-core MMA inner loop but folds the top-k sum into the FC2 store
# epilogue (dropping the separate top-k-sum launch). It never regresses vs the
# packed GEMM within its supported M range, so it is ALWAYS used for the small-M
# direct-topk decode sizes — there is no opt-in/opt-out switch.
# TC-decode is available for the whole small-M direct-topk range. Its only hard
# ceiling is the direct-topk route cap (above it, expert route-packing wins).
# _TC_DECODE_M is retained for callers/tests that enumerate the supported sizes.
_TC_DECODE_MAX_M = _W4A16_SMALL_M_DIRECT_MAX_M
_TC_DECODE_M = tuple(range(1, _TC_DECODE_MAX_M + 1))


@dsl_user_op
def _materialize_w4a16_topk_route_f32(value, *, loc=None, ip=None):
    """Keep the BF16-to-F32 conversion outside the unrolled reduction add.

    NVVM 23 otherwise folds the conversion into ``add.rn.f32.bf16``.  On
    SM120, ptxas assigns the six live BF16 sources to a sparse register span
    for that instruction sequence, increasing this kernel from 15 to 18 GPRs.
    The identity ``mov.b32`` is an opaque precision/lifetime boundary in NVVM;
    ptxas removes it and emits the spill-free 15-GPR CVT/FADD sequence.
    """
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _m_specialization_key(size_m: int) -> int:
    """M bucket for branches that genuinely specialize on token count."""
    return 1 if int(size_m) == 1 else 0


def _fake_m_for_specialization(size_m: int) -> int:
    return 1 if _m_specialization_key(size_m) == 1 else 2


# The W4A16 launch model chooses blocks/SM from static resource usage
# for each specialization.  These register counts were measured from the local
# SM121 JIT output and keep launch occupancy stable across refactors.
_W4A16_REGS_SM121 = {
    (256, 1, 8, 8, True): 118,
    (256, 1, 16, 4, True): 118,
    (256, 1, 16, 8, True): 118,
    (256, 1, 32, 2, True): 118,
    (128, 1, 4, 8, True): 118,
    (128, 1, 8, 4, True): 120,
    (256, 1, 8, 8, False): 158,
    (128, 1, 4, 8, False): 154,
    (128, 1, 8, 4, False): 143,
    (256, 2, 16, 4, False): 212,
    (128, 2, 4, 8, False): 215,
    (128, 2, 8, 4, False): 214,
    (256, 3, 16, 4, False): 249,
    (128, 3, 4, 8, False): 249,
    (128, 3, 8, 4, False): 250,
    (256, 4, 16, 4, False): 255,
    (128, 4, 4, 8, False): 255,
    (128, 4, 8, 4, False): 255,
}
# Layout overlay: the nf3_2p1 dequant inner loop (codebook prmt pools, 12B
# staged triples) allocates fewer registers than the packed e2m1 loop.
# Measured from ncu on SM120 JIT output at the production hybrid tile config;
# unmeasured specializations fall back to the packed table (a conservative
# upper bound).
_W4A16_REGS_SM121_NF3 = {
    (256, 4, 16, 4, False): 238,
}
_SMALL_BATCH_TILE_CONFIGS = (
    (128, 128, 256),
    (64, 128, 128),
    (128, 64, 128),
)
_LARGE_BATCH_TILE_CONFIGS = (
    (64, 256, 256),
    (64, 128, 128),
    (128, 64, 128),
)


def _covering_count(total: int, quantum: int) -> int:
    return (total + quantum - 1) // quantum


def _e8m0_logical_tail_scale_n(size_n: int) -> int:
    return (
        (int(size_n) + _E8M0_LOGICAL_TAIL_SCALE_N_ALIGNMENT - 1)
        // _E8M0_LOGICAL_TAIL_SCALE_N_ALIGNMENT
    ) * _E8M0_LOGICAL_TAIL_SCALE_N_ALIGNMENT


def _normalize_swiglu_limit(swiglu_limit: float | None) -> float | None:
    return normalize_swiglu_limit_for_activation("silu", swiglu_limit)


def _normalize_activation_swiglu_params(
    activation: str,
    swiglu_limit: float | None,
    swiglu_alpha: float | None,
    swiglu_beta: float | None,
) -> tuple[float | None, float, float]:
    activation = normalize_moe_activation(activation)
    return (
        normalize_swiglu_limit_for_activation(activation, swiglu_limit),
        normalize_swiglu_alpha_for_activation(activation, swiglu_alpha),
        normalize_swiglu_beta_for_activation(activation, swiglu_beta),
    )


def _w4a16_num_regs(
    *,
    cta_threads: int,
    cta_m_blocks: int,
    cta_n_blocks: int,
    cta_k_blocks: int,
    uses_m_block_8: bool,
    weight_layout: str = "packed",
) -> int:
    key = (
        int(cta_threads),
        int(cta_m_blocks),
        int(cta_n_blocks),
        int(cta_k_blocks),
        bool(uses_m_block_8),
    )
    if weight_layout == "nf3_2p1":
        hit = _W4A16_REGS_SM121_NF3.get(key)
        if hit is not None:
            return hit
    try:
        return _W4A16_REGS_SM121[key]
    except KeyError as exc:
        raise ValueError(
            f"missing W4A16 register count for NVFP4 BF16 specialization {key}"
        ) from exc


def _shared_memory_footprint(
    *,
    cta_m_blocks: int,
    tile_n: int,
    tile_k: int,
    scale_format: str = "e4m3_k16",
    weight_layout: str = "packed",
) -> int:
    cta_m = int(cta_m_blocks) * 16
    cta_n = int(tile_n)
    cta_k = int(tile_k)
    sh_block_meta_size = cta_m * 16
    sh_a_size = _STAGES * (cta_m * cta_k) * 2
    sh_b_size = _STAGES * (cta_k * cta_n // _PACK_FACTOR) * 4
    if weight_layout == "nf3_2p1":
        # NF3 stages 12-byte triples per 32-code unit instead of 16-byte int4
        # units (see b_unit_bytes) -- 3/4 of the packed B stage footprint.
        sh_b_size = sh_b_size * 3 // 4
    sh_red_size = cta_m * (cta_n + 8) * 2
    sh_bias_size = cta_n * 2
    tmp_size = min(sh_b_size, sh_red_size) + sh_bias_size
    tmp_size = max(max(sh_b_size, sh_red_size), tmp_size)
    sh_s_size = (
        _covering_count(cta_k, _scale_group_size(scale_format)) * cta_n * 2 * _STAGES
    )
    return tmp_size + sh_a_size + sh_s_size + sh_block_meta_size


def _determine_blocks_per_sm(
    *,
    problem_m: int,
    problem_n: int,
    top_k: int,
    cta_threads: int,
    cta_m_blocks: int,
    tile_n: int,
    tile_k: int,
    uses_m_block_8: bool,
    sms: int,
    max_shared_mem: int,
    scale_format: str = "e4m3_k16",
    weight_layout: str = "packed",
) -> int:
    num_regs = _w4a16_num_regs(
        cta_threads=cta_threads,
        cta_m_blocks=cta_m_blocks,
        cta_n_blocks=tile_n // 16,
        cta_k_blocks=tile_k // 16,
        uses_m_block_8=uses_m_block_8,
        weight_layout=weight_layout,
    )
    register_bytes = max(num_regs, 1) * int(cta_threads) * 4
    smem_bytes = _shared_memory_footprint(
        cta_m_blocks=cta_m_blocks,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_format=scale_format,
        weight_layout=weight_layout,
    )
    blocks_per_sm_limit = min(
        _DEVICE_MAX_REG_BYTES // register_bytes,
        int(max_shared_mem) // (smem_bytes + 1536),
    )
    if uses_m_block_8:
        # Small-M (moe_block_size==8) TC-decode is weight-bandwidth/overhead
        # bound, not parallelism bound (only m*top_k route-blocks of GEMM work).
        # The fused FC1->activation->FC2 path crosses several grid barriers whose
        # tid==0 atomic-counter increment serializes across all grid_x CTAs, so an
        # oversized grid pays barrier-atomic latency proportional to grid_x for no
        # extra GEMM throughput. Pin one persistent CTA per SM to minimize the
        # barrier participant count while still covering the machine for the
        # I_tp=1024 GEMMs. The split-K persistent loop is grid_x-agnostic, so this
        # is numerically identical.
        blocks_per_sm_limit = 1
    elif cta_m_blocks == 1:
        blocks_per_sm_limit = max(min(blocks_per_sm_limit, 4), 1)
    else:
        blocks_per_sm_limit = max(min(blocks_per_sm_limit, 2), 1)

    work_cta_count = (int(problem_n) // int(tile_n)) * int(problem_m) * int(top_k) * 4
    if work_cta_count < int(sms) * blocks_per_sm_limit:
        blocks_per_sm_limit = max(work_cta_count // int(sms), 1)
    return int(blocks_per_sm_limit)


def _candidate_tile_fits(
    *,
    problem_n: int,
    problem_k: int,
    cta_m_blocks: int,
    tile_n: int,
    tile_k: int,
    cta_threads: int,
    max_shared_mem: int,
    scale_format: str = "e4m3_k16",
    weight_layout: str = "packed",
    allow_logical_tail: bool = False,
) -> bool:
    if int(tile_k) == -1 or int(tile_n) == -1 or int(cta_threads) == -1:
        return False
    scale_group_size = _scale_group_size(scale_format)
    exact_n = int(problem_n) % int(tile_n) == 0
    exact_k = int(problem_k) % int(tile_k) == 0
    exact_scale_k = int(problem_k) % scale_group_size == 0
    if not allow_logical_tail:
        if not exact_n or not exact_k:
            return False
        if not exact_scale_k or int(tile_k) % scale_group_size != 0:
            return False
    elif (
        _normalize_scale_format(scale_format) != "e8m0_k32"
        or int(problem_n) % 16 != 0
        or int(problem_k) % 8 != 0
        or int(tile_n) % 16 != 0
        or int(tile_k) % scale_group_size != 0
    ):
        return False
    if int(tile_n) < 64 or int(tile_k) < 64 or int(cta_threads) < 128:
        return False
    smem_bytes = _shared_memory_footprint(
        cta_m_blocks=cta_m_blocks,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_format=scale_format,
        weight_layout=weight_layout,
    )
    return smem_bytes <= int(max_shared_mem)


def _select_tile_config(
    *,
    problem_m: int,
    problem_n: int,
    problem_k: int,
    top_k: int,
    moe_block_size: int,
    sms: int,
    max_shared_mem: int,
    required_cta_threads: int | None = None,
    scale_format: str = "e4m3_k16",
    weight_layout: str = "packed",
    allow_logical_tail: bool = False,
) -> tuple[int, int, int, int]:
    cta_m_blocks = _covering_count(moe_block_size, 16)
    uses_m_block_8 = moe_block_size == 8
    configs = (
        _LARGE_BATCH_TILE_CONFIGS if cta_m_blocks > 1 else _SMALL_BATCH_TILE_CONFIGS
    )
    best_blocks_per_sm = 0
    best_tile_config: tuple[int, int, int, int] | None = None
    for tile_k, tile_n, cta_threads in configs:
        if required_cta_threads is not None and int(cta_threads) != int(
            required_cta_threads
        ):
            continue
        if not _candidate_tile_fits(
            problem_n=problem_n,
            problem_k=problem_k,
            cta_m_blocks=cta_m_blocks,
            tile_n=tile_n,
            tile_k=tile_k,
            cta_threads=cta_threads,
            max_shared_mem=int(max_shared_mem) - 512,
            scale_format=scale_format,
            weight_layout=weight_layout,
            allow_logical_tail=allow_logical_tail,
        ):
            continue
        occupancy_problem_n = (
            _covering_count(int(problem_n), int(tile_n)) * int(tile_n)
            if allow_logical_tail
            else int(problem_n)
        )
        blocks_per_sm_limit = _determine_blocks_per_sm(
            problem_m=problem_m,
            problem_n=occupancy_problem_n,
            top_k=top_k,
            cta_threads=cta_threads,
            cta_m_blocks=cta_m_blocks,
            tile_n=tile_n,
            tile_k=tile_k,
            uses_m_block_8=uses_m_block_8,
            sms=sms,
            max_shared_mem=max_shared_mem,
            scale_format=scale_format,
            weight_layout=weight_layout,
        )
        if blocks_per_sm_limit > best_blocks_per_sm:
            best_blocks_per_sm = blocks_per_sm_limit
            best_tile_config = (tile_k, tile_n, cta_threads, blocks_per_sm_limit)
    if best_tile_config is None:
        cta_thread_msg = (
            ""
            if required_cta_threads is None
            else f", required_cta_threads={required_cta_threads}"
        )
        raise ValueError(
            "no valid W4A16 tile config for "
            f"M/N/K={problem_m}/{problem_n}/{problem_k}, moe_block_size={moe_block_size}"
            f"{cta_thread_msg}"
        )
    return best_tile_config


@dataclass(frozen=True)
class W4A16GemmCompileResult:
    compiled: object
    tile_n: int
    tile_k: int
    moe_block_size: int
    max_m_blocks: int
    blocks_per_sm: int
    weight_layout: str = "packed"
    scale_format: str = "e4m3_k16"
    w13_layout: str = "w13"


@dataclass(frozen=True)
class W4A16ActivationCompileResult:
    compiled: object
    rows: int
    intermediate_size: int
    activation: str
    swiglu_limit: float | None
    swiglu_alpha: float
    swiglu_beta: float


@dataclass(frozen=True)
class W4A16TopKSumCompileResult:
    compiled: object
    m: int
    topk: int
    hidden_size: int


@dataclass(frozen=True)
class W4A16FusedMoeCompileResult:
    compiled: object
    size_m: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    activation: str
    apply_router_weight_on_input: bool
    zero_fc2_output: bool
    element_dtype: str
    fast_math: bool
    swiglu_limit: float | None
    swiglu_alpha: float
    swiglu_beta: float
    fc1_tile_n: int
    fc1_tile_k: int
    fc2_tile_n: int
    fc2_tile_k: int
    moe_block_size: int
    max_m_blocks: int
    blocks_per_sm: int
    weight_layout: str = "packed"
    w13_layout: str = "w13"
    direct_topk_routes: bool = False
    scale_format: str = "e4m3_k16"
    tc_decode_fused_sum: bool = False
    collect_activation_amax: bool = False


@dataclass(frozen=True)
class _W4A16GemmLaunch:
    kernel: W4A16GemmCompileResult
    c_tmp: torch.Tensor


class _W4A16SmallMDirectLaunch(NamedTuple):
    compiled: object
    grid_x: int
    m: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    topk: int
    activation: str
    fast_math: bool
    topk_ids_dtype: torch.dtype


class MoEMicroKernelW4A16SmallMDirect(MoEMicroKernelBackend):
    """Decode-sized W4A16 specialization using the native ModelOpt layout."""

    _SUPPORTED_M = tuple(range(1, _W4A16_SMALL_M_DIRECT_MAX_M + 1))

    @classmethod
    def is_supported(
        cls,
        *,
        m: int,
        hidden_size: int,
        intermediate_size: int,
        topk: int,
        num_experts: int,
        scale_format: str = "e4m3_k16",
    ) -> bool:
        # E8M0 reads the shared packed scale grid. Both block axes must be /32.
        # The FC2 chunking (256 intermediate values per chunk, fc2_n_chunks)
        # masks a tail inside a single chunk correctly (I=128 oracle-covered),
        # but multi-chunk shards with a partial last chunk (352, 384) index
        # scale columns past the e8m0 grid and corrupt decode outputs — the
        # e4m3_k16 path masks this fine; e8m0 must cover multi-chunk exactly.
        if scale_format == "e8m0_k32":
            fc2_n_chunks = ((int(intermediate_size) // 2) + 127) // 128
            scale_block_ok = (
                int(hidden_size) % 32 == 0
                and int(intermediate_size) % 32 == 0
                and (fc2_n_chunks == 1 or int(intermediate_size) % 256 == 0)
            )
        else:
            scale_block_ok = True
        return (
            int(m) in cls._SUPPORTED_M
            and int(m) <= _W4A16_SMALL_M_DIRECT_MAX_M
            and int(hidden_size) > 0
            and int(hidden_size) % 128 == 0
            and int(intermediate_size) > 0
            and int(intermediate_size) % 16 == 0
            and scale_block_ok
            and 0 < int(topk) <= 32
            and int(num_experts) > 0
            and MoEMicroKernelBackend.is_supported(
                int(m),
                int(hidden_size),
                int(intermediate_size),
                int(topk),
                int(num_experts),
            )
        )

    def __init__(
        self,
        *,
        activation: str,
        fast_math: bool,
        share_input_across_experts: bool,
        share_expert_scales: bool,
        single_token: bool,
        scale_format: str = "e4m3_k16",
        swiglu_limit: float | None = None,
        swiglu_alpha: float | None = None,
        swiglu_beta: float | None = None,
        w13_layout: str = "w13",
    ):
        super().__init__(
            sf_vec_size=16,
            mma_tiler_mn=(64, 128),
            output_tile_count_n=1,
            fast_math=fast_math,
            activation=activation,
            share_input_across_experts=share_input_across_experts,
            share_expert_scales=share_expert_scales,
            single_token=single_token,
            dynamic_down_scale=False,
            w4a16_mode=True,
            scale_format=scale_format,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            w13_layout=w13_layout,
        )


class W4A16GemmKernel:
    def __init__(
        self,
        *,
        size_m: int,
        size_n: int,
        size_k: int,
        num_experts: int,
        top_k: int,
        mul_topk_weights: bool,
        tile_n: int,
        tile_k: int,
        moe_block_size: int,
        max_m_blocks: int,
        element_dtype: str = "bf16",
        epilogue_activation: str | None = None,
        weight_layout: str = "packed",
        scale_format: str = "e4m3_k16",
        w13_layout: str = "w13",
        source_n_rotation: int = 0,
        single_token_route_fast_path: bool = False,
        direct_topk_routes: bool = False,
        fused_topk_sum: bool = False,
        fused_sum_topk: int = 1,
        schedule_whole_tiles: bool = False,
    ):
        if element_dtype not in {"bf16", "fp16"}:
            raise ValueError(f"unsupported element_dtype {element_dtype!r}")
        if weight_layout not in _WEIGHT_LAYOUTS:
            raise ValueError(f"unsupported W4A16 weight_layout {weight_layout!r}")
        scale_format = _normalize_scale_format(scale_format)
        if weight_layout == "modelopt":
            if w13_layout not in _MODEL_OPT_W13_LAYOUTS:
                raise ValueError(f"unsupported W4A16 w13_layout {w13_layout!r}")
        else:
            w13_layout = "packed"
            source_n_rotation = 0
        if weight_layout == "nf3_2p1":
            if scale_format != "e4m3_k32":
                raise ValueError(
                    "nf3_2p1 W4A16 weights require scale_format='e4m3_k32'"
                )
            if element_dtype != "bf16":
                raise ValueError("nf3_2p1 W4A16 weights are bf16-activation only (v1)")
        if epilogue_activation not in (None, "relu2"):
            raise ValueError(
                "W4A16 GEMM epilogue activation currently supports only relu2"
            )
        if tile_n % 16 != 0 or tile_k % 16 != 0:
            raise ValueError("tile_n/tile_k must be multiples of 16")
        scale_group_size = _scale_group_size(scale_format)
        has_n_tile_tail = int(size_n) % int(tile_n) != 0
        has_k_tile_tail = int(size_k) % int(tile_k) != 0
        has_scale_k_tail = int(size_k) % scale_group_size != 0
        has_logical_tail = has_n_tile_tail or has_k_tile_tail or has_scale_k_tail
        if has_logical_tail:
            if weight_layout != "modelopt" or scale_format != "e8m0_k32":
                raise ValueError(
                    "W4A16 logical tail tiles are only supported for native "
                    "E8M0 K/32 weights"
                )
            if int(size_n) % 16 != 0:
                raise ValueError(
                    "native E8M0 W4A16 tail tiles require size_n % 16 == 0"
                )
            if int(size_k) % 8 != 0:
                raise ValueError("native E8M0 W4A16 tail tiles require size_k % 8 == 0")
            if int(tile_k) % scale_group_size != 0:
                raise ValueError(
                    "native E8M0 W4A16 tail tiles require tile_k multiples of 32"
                )
        else:
            if size_n % tile_n != 0:
                raise ValueError("size_n must be divisible by tile_n")
            if size_k % tile_k != 0:
                raise ValueError("size_k must be divisible by tile_k")
            if scale_format == "e8m0_k32" and (size_k % 32 != 0 or tile_k % 32 != 0):
                raise ValueError(
                    "E8M0 K/32 W4A16 scales require size_k/tile_k multiples of 32"
                )
            if scale_format == "e4m3_k32" and (size_k % 32 != 0 or tile_k % 32 != 0):
                raise ValueError(
                    "E4M3 K/32 W4A16 scales require size_k/tile_k multiples of 32"
                )
        if moe_block_size not in _ALLOWED_ROUTED_SIZES:
            raise ValueError(f"unsupported moe_block_size {moe_block_size}")
        if moe_block_size != 8 and moe_block_size % 16 != 0:
            raise ValueError("moe_block_size must be 8 or a multiple of 16")
        cta_threads = tile_n * tile_k // 64
        if cta_threads not in (128, 256):
            raise ValueError("W4A16 GEMM expects 128 or 256 CTA threads")
        self.size_m = int(size_m)
        self.size_n = int(size_n)
        self.size_k = int(size_k)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.mul_topk_weights = bool(mul_topk_weights)
        self.tile_n = int(tile_n)
        self.tile_k = int(tile_k)
        self.cta_n_blocks = int(tile_n // 16)
        self.cta_k_blocks = int(tile_k // 16)
        self.cta_threads = int(cta_threads)
        self.moe_block_size = int(moe_block_size)
        self.element_dtype = element_dtype
        self.is_fp16 = element_dtype == "fp16"
        self.epilogue_relu2 = epilogue_activation == "relu2"
        self.weight_layout = weight_layout
        self.weight_layout_nf3 = weight_layout == "nf3_2p1"
        self.scale_format = scale_format
        self.scale_format_e8m0_k32 = scale_format == "e8m0_k32"
        # k32 scale cadence (two K16 rows share one scale group). Shared by the
        # native E8M0 K/32 path and the NF3 e4m3-style K/32 path; the scale
        # DECODE stays keyed on scale_format_e8m0_k32 (e4m3_k32 uses the e4m3
        # decode arm, its 2**116 compensation lives in the global_scale tensor).
        self.scale_k32 = scale_format in ("e8m0_k32", "e4m3_k32")
        # NF3 codebook prmt pools (host-computed compile-time constants injected
        # into the device dequant). Only consumed by the nf3_2p1 path; cheap to
        # compute unconditionally.
        (
            self._nf3_cb0,
            self._nf3_cb1,
            self._nf3_cb2,
            self._nf3_cb3,
        ) = nf3_codebook_pools(_NF3_CODEBOOK)
        self.scale_group_size = int(scale_group_size)
        self.scale_k_groups = _covering_count(self.size_k, self.scale_group_size)
        self.n_tiles = _covering_count(self.size_n, self.tile_n)
        self.k_tiles = _covering_count(self.size_k, self.tile_k)
        self.covered_size_n = self.n_tiles * self.tile_n
        self.covered_size_k = self.k_tiles * self.tile_k
        self.has_n_tile_tail = bool(has_n_tile_tail)
        self.has_k_tile_tail = bool(has_k_tile_tail)
        self.has_scale_k_tail = bool(has_scale_k_tail)
        self.has_logical_tail = bool(has_logical_tail)
        self.scale_size_n = (
            _e8m0_logical_tail_scale_n(self.size_n)
            if self.scale_format_e8m0_k32 and self.has_n_tile_tail
            else self.size_n
        )
        self.scale_n_groups = self.scale_size_n // 16
        self.w13_layout = w13_layout
        self.source_n_rotation = int(source_n_rotation)
        self.single_token_route_fast_path = bool(single_token_route_fast_path)
        self.direct_topk_routes = bool(direct_topk_routes)
        self.fused_topk_sum = bool(fused_topk_sum)
        self.fused_sum_topk = int(fused_sum_topk)
        # Whole-tile persistent scheduling: every mn-tile is computed by one
        # CTA over the full K (grid-strided waves, ragged last wave), skipping
        # the split-K tail machinery entirely. Requires the host to bound the
        # wave count; used by the exact-geometry hybrid decode schedule.
        self.schedule_whole_tiles = bool(schedule_whole_tiles)
        if self.schedule_whole_tiles and not self.direct_topk_routes:
            raise ValueError("schedule_whole_tiles requires direct_topk_routes")
        if self.fused_topk_sum and not self.direct_topk_routes:
            raise ValueError("fused_topk_sum requires direct_topk_routes")
        if self.fused_topk_sum and self.fused_sum_topk < 1:
            raise ValueError("fused_sum_topk must be >= 1")
        self.cta_m_blocks = int(_covering_count(moe_block_size, 16))
        self.uses_m_block_8 = moe_block_size == 8
        self.max_m_blocks = int(max_m_blocks)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            self.sms = int(props.multi_processor_count)
            max_shared_mem = int(
                getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
            )
        else:
            self.sms = 120
            max_shared_mem = _DEFAULT_MAX_SHARED_MEM
        self.blocks_per_sm = _determine_blocks_per_sm(
            problem_m=self.size_m,
            problem_n=self.covered_size_n,
            top_k=self.top_k,
            cta_threads=self.cta_threads,
            cta_m_blocks=self.cta_m_blocks,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            uses_m_block_8=self.uses_m_block_8,
            sms=self.sms,
            max_shared_mem=max_shared_mem,
            scale_format=scale_format,
            weight_layout=weight_layout,
        )

        # W4A16 shared-memory geometry, in int4 units unless noted.
        self.a_sh_stride = 16 * self.cta_k_blocks // 8
        self.a_sh_stage = self.a_sh_stride * (16 * self.cta_m_blocks)
        self.a_gl_rd_delta_o = 16 * self.cta_k_blocks // 8
        self.a_sh_wr_delta = self.a_sh_stride * (
            self.cta_threads // self.a_gl_rd_delta_o
        )
        self.a_sh_wr_iters = _covering_count(self.a_sh_stage, self.a_sh_wr_delta)
        self.a_sh_rd_delta_i = self.a_sh_stride * 16

        self.b_sh_stride = ((self.cta_n_blocks * 16) * 16 // _PACK_FACTOR) // 4
        self.b_thread_vecs = 1
        self.b_sh_stride_threads = self.b_sh_stride
        self.b_sh_stage = self.b_sh_stride * self.cta_k_blocks
        self.b_sh_wr_iters = self.b_sh_stage // self.cta_threads
        # NF3 ("nf3_2p1") stages 12-byte triples per 32-code unit instead of the
        # 16-byte int4 unit; the per-thread unit count is unchanged, only the
        # bytes-per-unit and the 16-byte cp.async chunk count differ.
        self.b_unit_bytes = 12 if self.weight_layout_nf3 else 16
        self.b_sh_stage_bytes = self.b_sh_stage * self.b_unit_bytes
        if self.weight_layout_nf3:
            if self.b_sh_stage_bytes % 16 != 0:
                raise ValueError(
                    "nf3_2p1 B stage bytes must be a multiple of 16; "
                    f"got {self.b_sh_stage_bytes}"
                )
            self.b_sh_chunks = self.b_sh_stage_bytes // 16
            self.b_sh_wr_iters_nf3 = _covering_count(self.b_sh_chunks, self.cta_threads)
        else:
            self.b_sh_chunks = self.b_sh_stage
            self.b_sh_wr_iters_nf3 = self.b_sh_wr_iters

        self.s_sh_stride = 16 * self.cta_n_blocks // 16
        self.s_tb_groups = (
            self.cta_k_blocks // 2 if self.scale_k32 else self.cta_k_blocks
        )
        self.s_sh_stage = self.s_tb_groups * self.s_sh_stride
        self.tb_n_warps = self.cta_n_blocks // 4

        sh_block_route_indices = self.moe_block_size // 4
        sh_rd_block_route_indices = self.moe_block_size // 4
        sh_block_topk_weights = self.moe_block_size // 2
        self.sh_valid_count_off = (
            sh_block_route_indices + sh_rd_block_route_indices + sh_block_topk_weights
        )
        self.sh_route_off = 0
        self.sh_rd_route_off = sh_block_route_indices
        self.sh_topk_off = sh_block_route_indices + sh_rd_block_route_indices

        sh_red_size = (2 * self.cta_n_blocks + 1) * 16 * self.cta_m_blocks
        # B region size in int4 (16-byte) units. NF3 units are 12 bytes, so the
        # region is 0.75x; round the total up to a 16-byte multiple so the
        # following SMEM regions keep their 16-byte alignment (exact for all
        # supported tiles since b_sh_stage is a multiple of 4).
        sh_b_size = _covering_count(_STAGES * self.b_sh_stage_bytes, 16)
        sh_size_min = min(sh_red_size, sh_b_size)
        sh_size_max = max(sh_red_size, sh_b_size)
        sh_bias_size = self.cta_n_blocks * 16 // 8
        sh_b_red_bias_size = max(sh_size_max, sh_size_min + sh_bias_size)
        self.sh_b_off = self.sh_valid_count_off
        self.sh_red_off = self.sh_valid_count_off
        self.sh_s_off = self.sh_valid_count_off + sh_b_red_bias_size
        self.sh_a_off = self.sh_s_off + _STAGES * self.s_sh_stage
        self.shared_int4 = self.sh_a_off + _STAGES * self.a_sh_stage
        self.shared_words = self.shared_int4 * 4

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (
            self.size_n,
            self.size_k,
            self.covered_size_n,
            self.covered_size_k,
            self.scale_k_groups,
            self.has_logical_tail,
            self.num_experts,
            self.top_k,
            self.mul_topk_weights,
            self.tile_n,
            self.tile_k,
            self.cta_threads,
            self.moe_block_size,
            self.element_dtype,
            self.epilogue_relu2,
            self.weight_layout,
            self.scale_format,
            self.w13_layout,
            self.source_n_rotation,
            self.single_token_route_fast_path,
            self.direct_topk_routes,
            self.fused_topk_sum,
            self.fused_sum_topk,
            self.cta_m_blocks,
            self.uses_m_block_8,
            self.shared_words,
            # Launch bounds are part of the compiled kernel.  Keep binaries
            # planned for different residency targets out of the same cache
            # entry even when their arithmetic geometry otherwise matches.
            self.blocks_per_sm,
            self.schedule_whole_tiles,
        )

    @cute.jit
    def _activation_smem_permuted_offset(self, i: Int32) -> Int32:
        row = i // Int32(self.a_gl_rd_delta_o)
        return Int32(self.a_gl_rd_delta_o) * row + (
            (i - row * Int32(self.a_gl_rd_delta_o)) ^ (row & Int32(7))
        )

    @cute.jit
    def _int4_addr(self, smem_base: Int32, int4_off: Int32) -> Int32:
        return smem_base + int4_off * Int32(16)

    @cute.jit
    def _dequant_e2m1x4_to_elem2x2(self, packed: Uint32):
        if cutlass.const_expr(self.is_fp16):
            return packed_dequant_e2m1x4_to_half2x2(packed)
        return packed_dequant_e2m1x4_to_bfloat2x2(packed)

    @cute.jit
    def _dequant_e4m3x4_to_elem2x2(self, packed: Uint32):
        if cutlass.const_expr(self.is_fp16):
            return packed_dequant_e4m3x4_to_half2x2(packed)
        return packed_dequant_e4m3x4_to_bfloat2x2(packed)

    @cute.jit
    def _dequant_scale_x4_to_elem2x2(self, packed: Uint32):
        if cutlass.const_expr(self.scale_format_e8m0_k32):
            if cutlass.const_expr(self.is_fp16):
                return packed_dequant_e8m0x4_to_half2x2(packed)
            return packed_dequant_e8m0x4_to_bfloat2x2(packed)
        if cutlass.const_expr(self.is_fp16):
            return packed_dequant_e4m3x4_to_half2x2(packed)
        return packed_dequant_e4m3x4_to_bfloat2x2(packed)

    @cute.jit
    def _elem2_mul(self, a: Uint32, b: Uint32) -> Uint32:
        if cutlass.const_expr(self.is_fp16):
            return half2_mul(a, b)
        return bfloat2_mul(a, b)

    @cute.jit
    def _broadcast_f32_to_elem2(self, x: cutlass.Float32) -> Uint32:
        if cutlass.const_expr(self.is_fp16):
            return broadcast_f32_to_half2(x)
        return broadcast_f32_to_bfloat2(x)

    @cute.jit
    def _pack_f32x2_to_elem2(self, x0: cutlass.Float32, x1: cutlass.Float32) -> Uint32:
        if cutlass.const_expr(self.is_fp16):
            return pack_f32x2_to_f16x2(x0, x1)
        return pack_f32x2_to_bfloat2(x0, x1)

    @cute.jit
    def _elem2_to_f32x2(self, packed: Uint32):
        if cutlass.const_expr(self.is_fp16):
            return half2_to_float2_scaled(packed, cutlass.Float32(1.0))
        return bfloat2_to_float2_scaled(packed, cutlass.Float32(1.0))

    @cute.jit
    def _relu2_elem2(self, packed: Uint32) -> Uint32:
        x0, x1 = self._elem2_to_f32x2(packed)
        if x0 < cutlass.Float32(0.0):
            x0 = cutlass.Float32(0.0)
        if x1 < cutlass.Float32(0.0):
            x1 = cutlass.Float32(0.0)
        return self._pack_f32x2_to_elem2(x0 * x0, x1 * x1)

    @cute.jit
    def _st_shared_elem_from_f32(self, addr: Int32, val: cutlass.Float32):
        if cutlass.const_expr(self.is_fp16):
            st_shared_f16_from_f32(addr, val)
        else:
            st_shared_bf16_from_f32(addr, val)

    @cute.jit
    def _mma_m16n8k16_f32(
        self,
        d0: cutlass.Float32,
        d1: cutlass.Float32,
        d2: cutlass.Float32,
        d3: cutlass.Float32,
        a0: Uint32,
        a1: Uint32,
        a2: Uint32,
        a3: Uint32,
        b0: Uint32,
        b1: Uint32,
    ):
        if cutlass.const_expr(self.is_fp16):
            return f16_mma_m16n8k16_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)
        return bf16_mma_m16n8k16_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)

    @cute.jit
    def _mma_rhs_fragments_as_mma_a_m16n8k16_f32(
        self,
        d0: cutlass.Float32,
        d1: cutlass.Float32,
        d2: cutlass.Float32,
        d3: cutlass.Float32,
        b0_0: Uint32,
        b1_0: Uint32,
        b0_1: Uint32,
        b1_1: Uint32,
        a0: Uint32,
        a1: Uint32,
    ):
        if cutlass.const_expr(self.is_fp16):
            return f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                d0, d1, d2, d3, b0_0, b1_0, b0_1, b1_1, a0, a1
            )
        return bf16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
            d0, d1, d2, d3, b0_0, b1_0, b0_1, b1_1, a0, a1
        )

    @cute.jit
    def __call__(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        active_m: cutlass.Int32,
        grid_x: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        grid = (grid_x, 1, 1)
        self.kernel(
            a_bf16_flat,
            b_i32_flat,
            c_bf16_flat,
            scales_i32_flat,
            global_scale,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            topk_weights_flat,
            c_tmp_f32_flat,
            locks_i32_flat,
            active_m,
        ).launch(
            grid=grid,
            block=[self.cta_threads, 1, 1],
            min_blocks_per_mp=self.blocks_per_sm,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        active_m: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        cta = Int32(bidx)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words],
                1024,
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())

        grid_x, _, _ = cute.arch.grid_dim()
        self._run_persistent_gemm(
            a_bf16_flat,
            b_i32_flat,
            c_bf16_flat,
            scales_i32_flat,
            global_scale,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            topk_weights_flat,
            c_tmp_f32_flat,
            locks_i32_flat,
            smem_base,
            tid,
            cta,
            Int32(grid_x),
            Int32(active_m),
        )

    @cute.jit
    def _run_persistent_gemm(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_size_m: Int32,
        emit_tile: cutlass.Constexpr = None,
    ):
        n_tiles = Int32(self.n_tiles)
        route_blocks = active_size_m * Int32(self.top_k)
        if cutlass.const_expr(not self.direct_topk_routes):
            route_blocks = packed_route_count[Int32(0)].to(Int32) // Int32(
                self.moe_block_size
            )
        k_tiles = Int32(self.k_tiles)
        global_mn_tiles = route_blocks * n_tiles

        tail_mn_tiles = global_mn_tiles
        full_grid_mn_iters = Int32(0)
        force_one_tile_per_cta = Int32(0)
        if cutlass.const_expr(self.schedule_whole_tiles):
            # Whole-tile waves: one CTA computes each mn-tile over the full K,
            # task = cta + wave * grid_x, ragged last wave skipped through the
            # route_block_idx bound below. No split-K tail, no lock traffic.
            tail_mn_tiles = Int32(0)
            full_grid_mn_iters = (global_mn_tiles + grid_x - Int32(1)) // grid_x
        if cutlass.const_expr(not self.schedule_whole_tiles):
            if cutlass.const_expr(self.uses_m_block_8):
                # TC-decode small-M: when every mn-tile fits inside the launched grid
                # (FC1 has only route_blocks*n_tiles tiles, far fewer than grid_x),
                # the default tail path fans each mn-tile across multiple CTAs along
                # K and pays a lock-serialized cross-CTA split-K finalize plus the
                # reduction-turn handshake. Instead give the first global_mn_tiles
                # CTAs exactly one full mn-tile (all k_tiles, reduce_slice_count==1,
                # no finalize, no lock traffic) and idle the rest. grid_x and the
                # grid-barrier participant count are unchanged (so FC2 coverage is
                # untouched); only FC1's intra-GEMM work partition changes. Numerically
                # identical: a single CTA computes the whole K-reduction per tile.
                if global_mn_tiles <= grid_x:
                    force_one_tile_per_cta = Int32(1)
            if force_one_tile_per_cta != Int32(0):
                tail_mn_tiles = Int32(0)
                full_grid_mn_iters = Int32(1)
            elif global_mn_tiles > grid_x:
                tail_mn_tiles = global_mn_tiles - (global_mn_tiles // grid_x) * grid_x
                if tail_mn_tiles * Int32(3) <= grid_x:
                    tail_mn_tiles += grid_x
                full_grid_mn_iters = (global_mn_tiles - tail_mn_tiles) // grid_x

        iters = (k_tiles * tail_mn_tiles + grid_x - Int32(1)) // grid_x

        lock_slot = Int32(0)
        if tail_mn_tiles >= grid_x:
            lock_slot = cta
        else:
            lock_slot = (iters * cta) // k_tiles - Int32(1)

        in_tail_region = Int32(0)
        has_work = Int32(1)
        work_mn_tile = cta
        reduce_k_tile = Int32(0)
        route_block_idx = Int32(0)
        output_n_tile = Int32(0)
        if iters == Int32(0) and full_grid_mn_iters == Int32(0):
            has_work = Int32(0)

        while has_work != Int32(0):
            reduce_tile_count = Int32(0)
            reduce_slice_count = Int32(1)
            reduce_slice_idx = Int32(0)

            if in_tail_region == Int32(0) and full_grid_mn_iters > Int32(0):
                route_block_idx = work_mn_tile // n_tiles
                output_n_tile = work_mn_tile - route_block_idx * n_tiles
                reduce_k_tile = Int32(0)
                reduce_tile_count = k_tiles
                full_grid_mn_iters -= Int32(1)
            else:
                if in_tail_region == Int32(0):
                    in_tail_region = Int32(1)
                    tail_mn_base = global_mn_tiles - tail_mn_tiles
                    cta_iter_start = iters * cta
                    work_mn_tile = cta_iter_start // k_tiles
                    reduce_k_tile = cta_iter_start - work_mn_tile * k_tiles
                    global_mn_tile = work_mn_tile + tail_mn_base
                    route_block_idx = global_mn_tile // n_tiles
                    output_n_tile = global_mn_tile - route_block_idx * n_tiles

                if work_mn_tile < tail_mn_tiles and iters > Int32(0):
                    reduce_tile_count = iters * (cta + Int32(1)) - (
                        k_tiles * work_mn_tile + reduce_k_tile
                    )
                    if reduce_tile_count < Int32(0):
                        reduce_tile_count = Int32(0)
                    if reduce_k_tile + reduce_tile_count > k_tiles:
                        reduce_tile_count = k_tiles - reduce_k_tile

                    if reduce_tile_count > Int32(0):
                        first_reduce_boundary = iters * (
                            (k_tiles * work_mn_tile + iters - Int32(1)) // iters
                        )
                        if first_reduce_boundary <= k_tiles * (work_mn_tile + Int32(1)):
                            reduce_boundary_offset = (
                                first_reduce_boundary - k_tiles * work_mn_tile
                            )
                            reduce_slice_count = (
                                k_tiles - reduce_boundary_offset + iters - Int32(1)
                            ) // iters
                            if reduce_boundary_offset > Int32(0):
                                reduce_slice_count += Int32(1)
                            reduce_boundary_delta = iters * cta - first_reduce_boundary
                            if reduce_boundary_delta < Int32(0):
                                reduce_slice_idx = reduce_slice_count - Int32(1)
                            else:
                                if reduce_boundary_offset == Int32(
                                    0
                                ) and reduce_boundary_delta == Int32(0):
                                    reduce_slice_idx = reduce_slice_count - Int32(1)
                                else:
                                    reduce_slice_idx = (
                                        reduce_slice_count
                                        - Int32(1)
                                        - reduce_boundary_delta // iters
                                    )
                                    if reduce_boundary_offset > Int32(0):
                                        reduce_slice_idx -= Int32(1)

                        if tail_mn_tiles >= grid_x:
                            if reduce_slice_count > Int32(
                                1
                            ) and reduce_slice_idx == reduce_slice_count - Int32(1):
                                lock_slot += Int32(1)
                        else:
                            lock_slot += Int32(1)
                    else:
                        has_work = Int32(0)
                else:
                    has_work = Int32(0)

            if (
                has_work != Int32(0)
                and reduce_tile_count > Int32(0)
                and route_block_idx < route_blocks
            ):
                if cutlass.const_expr(emit_tile is not None):
                    # Trace-time tile emission hook: the caller owns expert
                    # resolution and the _run_tile dispatch (e.g. the hybrid
                    # multi-tier route map). The scheduling state machine above
                    # is unchanged; only tile emission is delegated.
                    emit_tile(
                        route_block_idx,
                        output_n_tile,
                        reduce_k_tile,
                        reduce_tile_count,
                        reduce_slice_count,
                        reduce_slice_idx,
                        lock_slot,
                    )
                else:
                    if cutlass.const_expr(self.direct_topk_routes):
                        expert_idx = packed_route_indices[route_block_idx].to(Int32)
                    else:
                        expert_idx = block_expert_ids[route_block_idx].to(Int32)
                    if expert_idx >= Int32(0):
                        self._run_tile(
                            a_bf16_flat,
                            b_i32_flat,
                            c_bf16_flat,
                            scales_i32_flat,
                            global_scale,
                            packed_route_indices,
                            topk_weights_flat,
                            c_tmp_f32_flat,
                            locks_i32_flat,
                            smem_base,
                            tid,
                            route_block_idx,
                            expert_idx,
                            output_n_tile,
                            reduce_k_tile,
                            reduce_tile_count,
                            reduce_slice_count,
                            reduce_slice_idx,
                            lock_slot,
                            active_size_m,
                        )

            if has_work != Int32(0):
                if in_tail_region == Int32(0):
                    work_mn_tile += grid_x
                else:
                    reduce_k_tile = Int32(0)
                    work_mn_tile += Int32(1)
                    output_n_tile += Int32(1)
                    if output_n_tile == n_tiles:
                        output_n_tile = Int32(0)
                        route_block_idx += Int32(1)

    @cute.jit
    def _read_moe_block_data(
        self,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        global_scale_f32: cutlass.Float32,
        active_size_m: Int32,
    ) -> Int32:
        if cutlass.const_expr(self.direct_topk_routes):
            if tid == Int32(0):
                idx = route_block_idx
                st_shared_i32(smem_base + Int32(self.sh_route_off * 16), idx)
                rd_row = idx // Int32(self.top_k)
                st_shared_i32(smem_base + Int32(self.sh_rd_route_off * 16), rd_row)
                if cutlass.const_expr(self.mul_topk_weights):
                    topk = topk_weights_flat[idx].to(cutlass.Float32) * global_scale_f32
                    st_shared_u32(
                        smem_base + Int32(self.sh_topk_off * 16),
                        self._broadcast_f32_to_elem2(topk),
                    )
            cute.arch.sync_threads()
            return Int32(1)

        if cutlass.const_expr(self.single_token_route_fast_path):
            if tid == Int32(0):
                idx = packed_route_indices[
                    route_block_idx * Int32(self.moe_block_size)
                ].to(Int32)
                st_shared_i32(smem_base + Int32(self.sh_route_off * 16), idx)
                rd_row = idx // Int32(self.top_k)
                st_shared_i32(smem_base + Int32(self.sh_rd_route_off * 16), rd_row)
                if cutlass.const_expr(self.mul_topk_weights):
                    safe_idx = idx
                    if idx >= active_size_m * Int32(self.top_k):
                        safe_idx = Int32(0)
                    topk = (
                        topk_weights_flat[safe_idx].to(cutlass.Float32)
                        * global_scale_f32
                    )
                    st_shared_u32(
                        smem_base + Int32(self.sh_topk_off * 16),
                        self._broadcast_f32_to_elem2(topk),
                    )
            cute.arch.sync_threads()
            return Int32(1)

        route_indices_int4_addr = self._int4_addr(
            smem_base, Int32(self.sh_route_off) + tid
        )
        route_indices_gmem = get_ptr_as_int64(
            packed_route_indices,
            route_block_idx * Int32(self.moe_block_size) + tid * Int32(4),
        )
        cp_async4_shared_global_pred(
            route_indices_int4_addr,
            route_indices_gmem,
            (tid < Int32(self.moe_block_size // 4)).to(Int32),
        )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        if tid >= Int32(self.cta_threads - 32):
            size_per_thread = _covering_count(self.moe_block_size, 32)
            lane = tid - Int32(self.cta_threads - 32)
            local_count = Int32(0)
            for i in cutlass.range_constexpr(size_per_thread):
                j = lane * Int32(size_per_thread) + Int32(i)
                if j < Int32(self.moe_block_size):
                    idx = ld_shared_i32_relaxed(
                        smem_base + Int32(self.sh_route_off * 16) + j * Int32(4)
                    )
                    if idx < active_size_m * Int32(self.top_k):
                        local_count += Int32(1)
            valid = cute.arch.warp_redux_sync(local_count, "add")
            if lane == Int32(0):
                st_shared_i32(smem_base + Int32(self.sh_valid_count_off * 16), valid)

        if tid < Int32(self.moe_block_size):
            idx = ld_shared_i32_relaxed(
                smem_base + Int32(self.sh_route_off * 16) + tid * Int32(4)
            )
            rd_row = idx // Int32(self.top_k)
            st_shared_i32(
                smem_base + Int32(self.sh_rd_route_off * 16) + tid * Int32(4),
                rd_row,
            )
            if cutlass.const_expr(self.mul_topk_weights):
                safe_idx = idx
                if idx >= active_size_m * Int32(self.top_k):
                    safe_idx = Int32(0)
                topk = (
                    topk_weights_flat[safe_idx].to(cutlass.Float32) * global_scale_f32
                )
                packed_topk = self._broadcast_f32_to_elem2(topk)
                # top-k weights are cached as packed element pairs.
                topk_word_addr = (
                    smem_base + Int32(self.sh_topk_off * 16) + tid * Int32(4)
                )
                st_shared_u32(topk_word_addr, packed_topk)

        cute.arch.sync_threads()
        valid_count = ld_shared_i32_relaxed(
            smem_base + Int32(self.sh_valid_count_off * 16)
        )
        cute.arch.sync_threads()
        return valid_count

    @cute.jit
    def _run_tile(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        expert_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
        active_size_m: Int32,
    ):
        if cutlass.const_expr(self.uses_m_block_8):
            self._run_tile_m8(
                a_bf16_flat,
                b_i32_flat,
                c_bf16_flat,
                scales_i32_flat,
                global_scale,
                packed_route_indices,
                topk_weights_flat,
                c_tmp_f32_flat,
                locks_i32_flat,
                smem_base,
                tid,
                route_block_idx,
                expert_idx,
                output_n_tile,
                reduce_k_tile,
                reduce_tile_count,
                reduce_slice_count,
                reduce_slice_idx,
                lock_slot,
                active_size_m,
            )
        else:
            self._run_tile_large_m(
                a_bf16_flat,
                b_i32_flat,
                c_bf16_flat,
                scales_i32_flat,
                global_scale,
                packed_route_indices,
                topk_weights_flat,
                c_tmp_f32_flat,
                locks_i32_flat,
                smem_base,
                tid,
                route_block_idx,
                expert_idx,
                output_n_tile,
                reduce_k_tile,
                reduce_tile_count,
                reduce_slice_count,
                reduce_slice_idx,
                lock_slot,
                active_size_m,
            )

    @cute.jit
    def _tile_common_prologue(
        self,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        expert_idx: Int32,
        output_n_tile: Int32,
        active_size_m: Int32,
    ):
        global_scale_f32 = global_scale[expert_idx].to(cutlass.Float32)
        if cutlass.const_expr(self.scale_format_e8m0_k32):
            if cutlass.const_expr(self.is_fp16):
                global_scale_f32 *= cutlass.Float32(_E8M0_K32_FP16_GLOBAL_COMPENSATION)
            else:
                global_scale_f32 *= cutlass.Float32(_E8M0_K32_BF16_GLOBAL_COMPENSATION)
        block_valid_rows = self._read_moe_block_data(
            packed_route_indices,
            topk_weights_flat,
            smem_base,
            tid,
            route_block_idx,
            global_scale_f32,
            active_size_m,
        )
        (
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            s_sh_rd,
        ) = self._tile_stream_offsets(tid, expert_idx, output_n_tile)
        return (
            global_scale_f32,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            s_sh_rd,
        )

    @cute.jit
    def _tile_stream_offsets(self, tid: Int32, expert_idx: Int32, output_n_tile: Int32):
        a_gl_stride = Int32(self.size_k // 8)
        b_gl_stride = Int32(16 * self.size_n // (_PACK_FACTOR * 4))
        s_gl_stride = Int32(self.scale_n_groups)
        if cutlass.const_expr(self.scale_k32):
            if cutlass.const_expr(self.scale_format_e8m0_k32 and self.has_logical_tail):
                scales_expert_stride = Int32(self.scale_k_groups * self.scale_n_groups)
            else:
                scales_expert_stride = Int32((self.size_n * self.size_k) // (32 * 16))
        else:
            scales_expert_stride = Int32((self.size_n * self.size_k) // (16 * 16))
        b_expert_off = (
            Int32((self.size_n * self.size_k) // (_PACK_FACTOR * 4)) * expert_idx
        )
        scales_expert_off = scales_expert_stride * expert_idx

        a_gl_rd_row = tid // Int32(self.a_gl_rd_delta_o)
        a_gl_rd_col0 = tid - a_gl_rd_row * Int32(self.a_gl_rd_delta_o)
        a_sh_wr = Int32(self.a_sh_stride) * (tid // Int32(self.a_gl_rd_delta_o)) + (
            tid - (tid // Int32(self.a_gl_rd_delta_o)) * Int32(self.a_gl_rd_delta_o)
        )
        a_rows_per_iter = Int32(self.cta_threads // self.a_gl_rd_delta_o)

        if cutlass.const_expr(self.cta_threads <= self.b_sh_stride):
            b_gl_rd_base = tid
        else:
            b_gl_rd_base = b_gl_stride * (tid // Int32(self.b_sh_stride)) + (
                tid % Int32(self.b_sh_stride)
            )
        b_gl_rd_base += b_expert_off + Int32(self.b_sh_stride) * output_n_tile
        b_sh_rd = tid
        b_sh_rd += (b_sh_rd // Int32(self.b_sh_stride)) * Int32(
            self.b_sh_stride * (self.b_sh_wr_iters - 1)
        )

        s_sh_rd = Int32(8) * ((tid // Int32(32)) % Int32(self.tb_n_warps)) + (
            tid & Int32(31)
        ) // Int32(4)
        return (
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            s_sh_rd,
        )

    @cute.jit
    def _a_shared_read_offset(self, tid: Int32, lanes_per_row: cutlass.Constexpr[int]):
        a_sh_rd = Int32(self.a_sh_stride) * (
            (tid & Int32(31)) % Int32(lanes_per_row)
        ) + (tid & Int32(31)) // Int32(lanes_per_row)
        a_sh_rd += (
            Int32(2)
            * ((tid // Int32(32)) // Int32(self.tb_n_warps))
            * Int32(self.b_sh_wr_iters)
        )
        return a_sh_rd

    @cute.jit
    def _run_tile_m8(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        expert_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
        active_size_m: Int32,
    ):
        (
            global_scale_f32,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            s_sh_rd,
        ) = self._tile_common_prologue(
            global_scale,
            packed_route_indices,
            topk_weights_flat,
            smem_base,
            tid,
            route_block_idx,
            expert_idx,
            output_n_tile,
            active_size_m,
        )
        a_sh_rd = self._a_shared_read_offset(tid, 8)

        # LLVM 23 also promotes this 16-f32 accumulator across the pipelined
        # control-flow joins, repeatedly packing adjacent values through i64
        # temporaries.  Keep the values in independent scalar fragments, as in
        # the large-M path below, so those PHIs remain scalar.
        acc = [
            cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH,), cutlass.Float32)
            for _ in range(16 // _SCALAR_ACC_FRAGMENT_WIDTH)
        ]
        for frag in cutlass.range_constexpr(16 // _SCALAR_ACC_FRAGMENT_WIDTH):
            acc[frag].fill(0.0)

        k_tiles = reduce_tile_count
        self._prefetch_initial_tiles(
            a_bf16_flat,
            b_i32_flat,
            scales_i32_flat,
            smem_base,
            tid,
            k_tiles,
            reduce_k_tile,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
            expert_idx,
        )

        b_scale_cur = cute.make_rmem_tensor((2, 4), Uint32)
        b_scale_next = cute.make_rmem_tensor((2, 4), Uint32)
        self._load_b_scale_register_bundle(
            b_scale_cur,
            smem_base,
            tid,
            b_sh_rd,
            s_sh_rd,
            Int32(0),
            Int32(0),
        )
        a_regs_cur = cute.make_rmem_tensor((2,), Uint32)
        a_regs_next = cute.make_rmem_tensor((2,), Uint32)
        self._load_a_register_bundle(
            a_regs_cur,
            smem_base,
            a_sh_rd,
            Int32(0),
            Int32(0),
            True,
        )
        self._run_mma_pipeline(
            a_bf16_flat,
            b_i32_flat,
            scales_i32_flat,
            smem_base,
            tid,
            acc,
            acc,
            acc,
            acc,
            b_scale_cur,
            b_scale_next,
            a_regs_cur,
            a_regs_next,
            b_sh_rd,
            s_sh_rd,
            a_sh_rd,
            k_tiles,
            reduce_k_tile,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
            expert_idx,
            True,
        )

        self._finish_tile(
            acc,
            acc,
            acc,
            acc,
            c_bf16_flat,
            c_tmp_f32_flat,
            locks_i32_flat,
            smem_base,
            tid,
            output_n_tile,
            block_valid_rows,
            global_scale_f32,
            reduce_slice_count,
            reduce_slice_idx,
            lock_slot,
            True,
        )

    @cute.jit
    def _run_tile_large_m(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        expert_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
        active_size_m: Int32,
    ):
        (
            global_scale_f32,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            s_sh_rd,
        ) = self._tile_common_prologue(
            global_scale,
            packed_route_indices,
            topk_weights_flat,
            smem_base,
            tid,
            route_block_idx,
            expert_idx,
            output_n_tile,
            active_size_m,
        )
        a_sh_rd = self._a_shared_read_offset(tid, 16)

        # Keep each accumulator element in its own scalar rmem tensor. LLVM 23
        # otherwise promotes the 128-f32 accumulator to wide vector PHIs and
        # repeatedly packs/unpacks adjacent f32 values through i64 temporaries.
        acc0 = [
            cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH,), cutlass.Float32)
            for _ in range(32 // _SCALAR_ACC_FRAGMENT_WIDTH)
        ]
        for frag in cutlass.range_constexpr(32 // _SCALAR_ACC_FRAGMENT_WIDTH):
            acc0[frag].fill(0.0)
        acc1 = acc0
        acc2 = acc0
        acc3 = acc0
        if cutlass.const_expr(self.cta_m_blocks > 1):
            acc1 = [
                cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH,), cutlass.Float32)
                for _ in range(32 // _SCALAR_ACC_FRAGMENT_WIDTH)
            ]
            for frag in cutlass.range_constexpr(32 // _SCALAR_ACC_FRAGMENT_WIDTH):
                acc1[frag].fill(0.0)
        if cutlass.const_expr(self.cta_m_blocks > 2):
            acc2 = [
                cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH,), cutlass.Float32)
                for _ in range(32 // _SCALAR_ACC_FRAGMENT_WIDTH)
            ]
            for frag in cutlass.range_constexpr(32 // _SCALAR_ACC_FRAGMENT_WIDTH):
                acc2[frag].fill(0.0)
        if cutlass.const_expr(self.cta_m_blocks > 3):
            acc3 = [
                cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH,), cutlass.Float32)
                for _ in range(32 // _SCALAR_ACC_FRAGMENT_WIDTH)
            ]
            for frag in cutlass.range_constexpr(32 // _SCALAR_ACC_FRAGMENT_WIDTH):
                acc3[frag].fill(0.0)

        k_tiles = reduce_tile_count
        self._prefetch_initial_tiles(
            a_bf16_flat,
            b_i32_flat,
            scales_i32_flat,
            smem_base,
            tid,
            k_tiles,
            reduce_k_tile,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
            expert_idx,
        )

        b_scale_cur = cute.make_rmem_tensor((2, 4), Uint32)
        b_scale_next = cute.make_rmem_tensor((2, 4), Uint32)
        self._load_b_scale_register_bundle(
            b_scale_cur,
            smem_base,
            tid,
            b_sh_rd,
            s_sh_rd,
            Int32(0),
            Int32(0),
        )
        a_regs = cute.make_rmem_tensor((self.cta_m_blocks, 4), Uint32)
        a_regs_next = cute.make_rmem_tensor((self.cta_m_blocks, 4), Uint32)
        self._load_a_register_bundle(
            a_regs,
            smem_base,
            a_sh_rd,
            Int32(0),
            Int32(0),
            False,
        )
        self._run_mma_pipeline(
            a_bf16_flat,
            b_i32_flat,
            scales_i32_flat,
            smem_base,
            tid,
            acc0,
            acc1,
            acc2,
            acc3,
            b_scale_cur,
            b_scale_next,
            a_regs,
            a_regs_next,
            b_sh_rd,
            s_sh_rd,
            a_sh_rd,
            k_tiles,
            reduce_k_tile,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
            expert_idx,
            False,
        )

        self._finish_tile(
            acc0,
            acc1,
            acc2,
            acc3,
            c_bf16_flat,
            c_tmp_f32_flat,
            locks_i32_flat,
            smem_base,
            tid,
            output_n_tile,
            block_valid_rows,
            global_scale_f32,
            reduce_slice_count,
            reduce_slice_idx,
            lock_slot,
            False,
        )

    @cute.jit
    def _run_mma_pipeline(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        acc0,
        acc1,
        acc2,
        acc3,
        b_scale_cur: cute.Tensor,
        b_scale_next: cute.Tensor,
        a_regs_cur: cute.Tensor,
        a_regs_next: cute.Tensor,
        b_sh_rd: Int32,
        s_sh_rd: Int32,
        a_sh_rd: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
        expert_idx: Int32,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        b_frag = cute.make_rmem_tensor((2, 2), Uint32)
        tile_idx = Int32(0)
        while tile_idx < k_tiles:
            for pipe in cutlass.range_constexpr(_STAGES):
                if tile_idx < k_tiles:
                    for kk in cutlass.range_constexpr(self.b_sh_wr_iters):
                        self._load_next_fragment_bundle(
                            b_scale_next,
                            a_regs_next,
                            smem_base,
                            tid,
                            b_sh_rd,
                            s_sh_rd,
                            a_sh_rd,
                            pipe,
                            kk,
                            tile_idx,
                            k_tiles,
                            uses_m_block_8,
                        )

                        self._prefetch_pipeline_step(
                            a_bf16_flat,
                            b_i32_flat,
                            scales_i32_flat,
                            smem_base,
                            tid,
                            pipe,
                            kk,
                            tile_idx,
                            k_tiles,
                            reduce_k_tile,
                            block_valid_rows,
                            a_gl_stride,
                            b_gl_stride,
                            s_gl_stride,
                            scales_expert_off,
                            b_gl_rd_base,
                            a_gl_rd_row,
                            a_gl_rd_col0,
                            a_sh_wr,
                            a_rows_per_iter,
                            output_n_tile,
                            expert_idx,
                        )

                        for jj in cutlass.range_constexpr(4):
                            if cutlass.const_expr(self.weight_layout_nf3):
                                # NF3 triple: lo16 = half (jj % 2) of word
                                # (jj // 2); hi8 = byte jj of the hi word; scale
                                # register is the same per-jj lane as packed.
                                lo_w = b_scale_cur[0, jj // 2]
                                lo16 = (lo_w >> Uint32(16 * (jj % 2))) & Uint32(0xFFFF)
                                hi8 = (b_scale_cur[0, 2] >> Uint32(8 * jj)) & Uint32(
                                    0xFF
                                )
                                s = b_scale_cur[1, jj]
                                self._scaled_dequant_b_fragment_nf3(
                                    b_frag, lo16, hi8, s
                                )
                            else:
                                q, s = self._select_b_scale_register(jj, b_scale_cur)
                                self._scaled_dequant_b_fragment(b_frag, q, s)
                            if cutlass.const_expr(uses_m_block_8):
                                self._mma_accumulate_m8(
                                    acc0,
                                    jj,
                                    a_regs_cur,
                                    b_frag,
                                )
                            else:
                                for mb in cutlass.range_constexpr(self.cta_m_blocks):
                                    if cutlass.const_expr(mb == 0):
                                        self._mma_accumulate_large_m(
                                            acc0, a_regs_cur, mb, jj, b_frag
                                        )
                                    elif cutlass.const_expr(mb == 1):
                                        self._mma_accumulate_large_m(
                                            acc1, a_regs_cur, mb, jj, b_frag
                                        )
                                    elif cutlass.const_expr(mb == 2):
                                        self._mma_accumulate_large_m(
                                            acc2, a_regs_cur, mb, jj, b_frag
                                        )
                                    else:
                                        self._mma_accumulate_large_m(
                                            acc3, a_regs_cur, mb, jj, b_frag
                                        )

                        if cutlass.const_expr(uses_m_block_8):
                            self._copy_a_register_bundle(
                                a_regs_cur,
                                a_regs_next,
                                uses_m_block_8,
                            )
                            self._copy_b_scale_register_bundle(
                                b_scale_cur, b_scale_next
                            )
                        else:
                            self._copy_b_scale_register_bundle(
                                b_scale_cur, b_scale_next
                            )
                            self._copy_a_register_bundle(
                                a_regs_cur,
                                a_regs_next,
                                uses_m_block_8,
                            )
                    tile_idx += Int32(1)
            cute.arch.sync_threads()
            if tile_idx < k_tiles:
                self._load_b_scale_register_bundle(
                    b_scale_cur,
                    smem_base,
                    tid,
                    b_sh_rd,
                    s_sh_rd,
                    Int32(0),
                    Int32(0),
                )
                self._load_a_register_bundle(
                    a_regs_cur,
                    smem_base,
                    a_sh_rd,
                    Int32(0),
                    Int32(0),
                    uses_m_block_8,
                )

    @cute.jit
    def _finish_tile(
        self,
        acc0,
        acc1,
        acc2,
        acc3,
        c_bf16_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        output_n_tile: Int32,
        block_valid_rows: Int32,
        global_scale_f32: cutlass.Float32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        if cutlass.const_expr(uses_m_block_8):
            self._fold_cta_partials_m8(acc0, smem_base, tid)
        else:
            self._fold_cta_partials_large_m(acc0, acc1, acc2, acc3, smem_base, tid)

        if reduce_slice_count > Int32(1):
            self._wait_for_reduction_turn(
                locks_i32_flat, lock_slot, reduce_slice_idx, tid
            )
            self._combine_splitk_accumulators(
                acc0,
                acc1,
                acc2,
                acc3,
                c_tmp_f32_flat,
                block_valid_rows,
                lock_slot,
                reduce_slice_idx,
                reduce_slice_count,
                tid,
                uses_m_block_8,
            )
            self._publish_reduction_turn(
                locks_i32_flat,
                lock_slot,
                reduce_slice_idx == reduce_slice_count - Int32(1),
                tid,
            )

        if reduce_slice_idx == reduce_slice_count - Int32(1):
            if cutlass.const_expr(uses_m_block_8):
                self._store_tile_m8(
                    acc0,
                    c_bf16_flat,
                    smem_base,
                    tid,
                    output_n_tile,
                    block_valid_rows,
                    global_scale_f32,
                )
            else:
                self._store_tile_large_m(
                    acc0,
                    acc1,
                    acc2,
                    acc3,
                    c_bf16_flat,
                    smem_base,
                    tid,
                    output_n_tile,
                    block_valid_rows,
                    global_scale_f32,
                )

    @cute.jit
    def _wait_for_reduction_turn(
        self,
        locks_i32_flat: cute.Tensor,
        lock_slot: Int32,
        count: Int32,
        tid: Int32,
    ):
        lock_addr = get_ptr_as_int64(locks_i32_flat, lock_slot)
        if tid == Int32(0):
            state = Int32(-1)
            while state != count:
                state = ld_global_acquire_i32(lock_addr)
        cute.arch.sync_threads()

    @cute.jit
    def _publish_reduction_turn(
        self,
        locks_i32_flat: cute.Tensor,
        lock_slot: Int32,
        reset,
        tid: Int32,
    ):
        lock_addr = get_ptr_as_int64(locks_i32_flat, lock_slot)
        cute.arch.sync_threads()
        if tid == Int32(0):
            if reset:
                st_global_i32(lock_addr, Int32(0))
            else:
                red_add_global_release_i32(lock_addr, Int32(1))

    @cute.jit
    def _merge_splitk_vec4(
        self,
        c_tmp_f32_flat: cute.Tensor,
        f32_off: Int32,
        reduce_slice_idx: Int32,
        reduce_slice_count: Int32,
        c0: cutlass.Float32,
        c1: cutlass.Float32,
        c2: cutlass.Float32,
        c3: cutlass.Float32,
    ):
        if reduce_slice_idx != Int32(0):
            r0, r1, r2, r3 = ld_global_v4_f32(get_ptr_as_int64(c_tmp_f32_flat, f32_off))
            c0 = c0 + r0
            c1 = c1 + r1
            c2 = c2 + r2
            c3 = c3 + r3
        if reduce_slice_idx != reduce_slice_count - Int32(1):
            st_global_v4_f32(
                get_ptr_as_int64(c_tmp_f32_flat, f32_off),
                c0,
                c1,
                c2,
                c3,
            )
        return c0, c1, c2, c3

    @cute.jit
    def _combine_splitk_accumulators(
        self,
        acc0,
        acc1,
        acc2,
        acc3,
        c_tmp_f32_flat: cute.Tensor,
        block_valid_rows: Int32,
        lock_slot: Int32,
        reduce_slice_idx: Int32,
        reduce_slice_count: Int32,
        tid: Int32,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        active_threads = Int32(32 * self.tb_n_warps)
        c_size_int4 = Int32((self.cta_m_blocks * 16 * self.cta_n_blocks * 16) // 4)
        c_cur_offset = lock_slot * c_size_int4
        if cutlass.const_expr(uses_m_block_8):
            if tid < active_threads:
                for jj in cutlass.range_constexpr(4):
                    k = jj * 2
                    (
                        acc0[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc0[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc0[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc0[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                    ) = self._merge_splitk_slot(
                        c_tmp_f32_flat,
                        c_cur_offset,
                        active_threads,
                        Int32(k),
                        tid,
                        reduce_slice_idx,
                        reduce_slice_count,
                        acc0[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc0[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc0[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc0[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                    )
        else:
            lane_row = (tid & Int32(31)) // Int32(4)
            if tid < active_threads:
                for mb in cutlass.range_constexpr(self.cta_m_blocks):
                    if cutlass.const_expr(mb == 0):
                        self._combine_splitk_accumulator_block(
                            acc0,
                            mb,
                            c_tmp_f32_flat,
                            c_cur_offset,
                            active_threads,
                            block_valid_rows,
                            lane_row,
                            tid,
                            reduce_slice_idx,
                            reduce_slice_count,
                        )
                    elif cutlass.const_expr(mb == 1):
                        self._combine_splitk_accumulator_block(
                            acc1,
                            mb,
                            c_tmp_f32_flat,
                            c_cur_offset,
                            active_threads,
                            block_valid_rows,
                            lane_row,
                            tid,
                            reduce_slice_idx,
                            reduce_slice_count,
                        )
                    elif cutlass.const_expr(mb == 2):
                        self._combine_splitk_accumulator_block(
                            acc2,
                            mb,
                            c_tmp_f32_flat,
                            c_cur_offset,
                            active_threads,
                            block_valid_rows,
                            lane_row,
                            tid,
                            reduce_slice_idx,
                            reduce_slice_count,
                        )
                    else:
                        self._combine_splitk_accumulator_block(
                            acc3,
                            mb,
                            c_tmp_f32_flat,
                            c_cur_offset,
                            active_threads,
                            block_valid_rows,
                            lane_row,
                            tid,
                            reduce_slice_idx,
                            reduce_slice_count,
                        )

    @cute.jit
    def _combine_splitk_accumulator_block(
        self,
        acc,
        mb: cutlass.Constexpr[int],
        c_tmp_f32_flat: cute.Tensor,
        c_cur_offset: Int32,
        active_threads: Int32,
        block_valid_rows: Int32,
        lane_row: Int32,
        tid: Int32,
        reduce_slice_idx: Int32,
        reduce_slice_count: Int32,
    ):
        for flat_j in cutlass.range_constexpr(8):
            row_valid = Int32(mb * 16) + lane_row < block_valid_rows
            if row_valid:
                (
                    acc[(flat_j * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                    acc[(flat_j * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                    acc[(flat_j * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                    acc[(flat_j * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                ) = self._merge_splitk_slot(
                    c_tmp_f32_flat,
                    c_cur_offset,
                    active_threads,
                    Int32(mb * 8 + flat_j),
                    tid,
                    reduce_slice_idx,
                    reduce_slice_count,
                    acc[(flat_j * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                    acc[(flat_j * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                    acc[(flat_j * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                    acc[(flat_j * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                )

    @cute.jit
    def _merge_splitk_slot(
        self,
        c_tmp_f32_flat: cute.Tensor,
        c_cur_offset: Int32,
        active_threads: Int32,
        slot: Int32,
        tid: Int32,
        reduce_slice_idx: Int32,
        reduce_slice_count: Int32,
        c0: cutlass.Float32,
        c1: cutlass.Float32,
        c2: cutlass.Float32,
        c3: cutlass.Float32,
    ):
        int4_off = c_cur_offset + active_threads * slot + tid
        return self._merge_splitk_vec4(
            c_tmp_f32_flat,
            int4_off * Int32(4),
            reduce_slice_idx,
            reduce_slice_count,
            c0,
            c1,
            c2,
            c3,
        )

    @cute.jit
    def _load_a_registers_large_m(
        self,
        smem_base: Int32,
        a_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
        m_block: Int32,
    ):
        a_addr = self._int4_addr(
            smem_base,
            Int32(self.sh_a_off)
            + pipe * Int32(self.a_sh_stage)
            + self._activation_smem_permuted_offset(
                Int32(2) * kk + m_block * Int32(self.a_sh_rd_delta_i) + a_sh_rd
            ),
        )
        return ldmatrix_m8n8x4_b16(a_addr)

    @cute.jit
    def _load_a_registers_m8(
        self,
        smem_base: Int32,
        a_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        a_addr = self._int4_addr(
            smem_base,
            Int32(self.sh_a_off)
            + pipe * Int32(self.a_sh_stage)
            + self._activation_smem_permuted_offset(Int32(2) * kk + a_sh_rd),
        )
        return ldmatrix_m8n8x2_b16(a_addr)

    @cute.jit
    def _load_a_registers_large_m_bundle(
        self,
        regs: cute.Tensor,
        smem_base: Int32,
        a_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        for mb in cutlass.range_constexpr(self.cta_m_blocks):
            a0, a1, a2, a3 = self._load_a_registers_large_m(
                smem_base,
                a_sh_rd,
                pipe,
                kk,
                Int32(mb),
            )
            regs[mb, 0] = a0
            regs[mb, 1] = a1
            regs[mb, 2] = a2
            regs[mb, 3] = a3

    @cute.jit
    def _load_a_registers_m8_bundle(
        self,
        regs: cute.Tensor,
        smem_base: Int32,
        a_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        a0, a1 = self._load_a_registers_m8(smem_base, a_sh_rd, pipe, kk)
        regs[0] = a0
        regs[1] = a1

    @cute.jit
    def _clear_a_register_bundle_large_m(self, regs: cute.Tensor):
        for mb in cutlass.range_constexpr(self.cta_m_blocks):
            for reg in cutlass.range_constexpr(4):
                regs[mb, reg] = Uint32(0)

    @cute.jit
    def _clear_a_register_bundle_m8(self, regs: cute.Tensor):
        for reg in cutlass.range_constexpr(2):
            regs[reg] = Uint32(0)

    @cute.jit
    def _copy_a_register_bundle_large_m(self, dst: cute.Tensor, src: cute.Tensor):
        for mb in cutlass.range_constexpr(self.cta_m_blocks):
            for reg in cutlass.range_constexpr(4):
                dst[mb, reg] = src[mb, reg]

    @cute.jit
    def _copy_a_register_bundle_m8(self, dst: cute.Tensor, src: cute.Tensor):
        for reg in cutlass.range_constexpr(2):
            dst[reg] = src[reg]

    @cute.jit
    def _load_a_register_bundle(
        self,
        regs: cute.Tensor,
        smem_base: Int32,
        a_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        if cutlass.const_expr(uses_m_block_8):
            self._load_a_registers_m8_bundle(regs, smem_base, a_sh_rd, pipe, kk)
        else:
            self._load_a_registers_large_m_bundle(regs, smem_base, a_sh_rd, pipe, kk)

    @cute.jit
    def _clear_a_register_bundle(
        self,
        regs: cute.Tensor,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        if cutlass.const_expr(uses_m_block_8):
            self._clear_a_register_bundle_m8(regs)
        else:
            self._clear_a_register_bundle_large_m(regs)

    @cute.jit
    def _copy_a_register_bundle(
        self,
        dst: cute.Tensor,
        src: cute.Tensor,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        if cutlass.const_expr(uses_m_block_8):
            self._copy_a_register_bundle_m8(dst, src)
        else:
            self._copy_a_register_bundle_large_m(dst, src)

    @cute.jit
    def _load_b_scale_registers(
        self,
        smem_base: Int32,
        tid: Int32,
        b_sh_rd: Int32,
        s_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        if cutlass.const_expr(self.weight_layout == "modelopt"):
            q0, q1, q2, q3 = self._load_b_registers_modelopt_shared(
                smem_base,
                b_sh_rd,
                pipe,
                kk,
            )
        elif cutlass.const_expr(self.weight_layout_nf3):
            # NF3-MAPPING-V1: the thread's 32-code unit index within the pipe's
            # B region equals the stock staged-unit index (algebraically
            # identical to b_sh_stride*kk + b_sh_rd): cur_group_id is the K16 row
            # within the tile, (tid % b_sh_stride) is the N-pair. Each unit is a
            # 12-byte (lo0, lo1, hi) triple; three scalar u32 loads over stride-3
            # words are bank-conflict free (gcd(3, 32) == 1). q3 is unused.
            nf3_warp_id = tid // Int32(32)
            nf3_warp_row = nf3_warp_id // Int32(self.tb_n_warps)
            nf3_group = Int32(self.b_sh_wr_iters) * nf3_warp_row + kk
            nf3_unit = nf3_group * Int32(self.b_sh_stride) + (
                tid % Int32(self.b_sh_stride)
            )
            b_addr = (
                smem_base
                + Int32(self.sh_b_off * 16)
                + pipe * Int32(self.b_sh_stage_bytes)
                + nf3_unit * Int32(12)
            )
            q0 = ld_shared_u32(b_addr)
            q1 = ld_shared_u32(b_addr + Int32(4))
            q2 = ld_shared_u32(b_addr + Int32(8))
            q3 = Uint32(0)
        else:
            b_addr = self._int4_addr(
                smem_base,
                Int32(self.sh_b_off)
                + pipe * Int32(self.b_sh_stage)
                + Int32(self.b_sh_stride) * kk
                + b_sh_rd,
            )
            q0, q1, q2, q3 = ld_shared_v4_u32(b_addr)

        warp_id = tid // Int32(32)
        warp_row = warp_id // Int32(self.tb_n_warps)
        cur_group_id = Int32(self.b_sh_wr_iters) * warp_row + kk
        if cutlass.const_expr(self.scale_k32):
            scale_group_id = cur_group_id // Int32(2)
        else:
            scale_group_id = cur_group_id
        s_addr = (
            smem_base
            + Int32(self.sh_s_off * 16)
            + pipe * Int32(self.s_sh_stage * 16)
            + (s_sh_rd + scale_group_id * Int32(2 * self.s_sh_stride)) * Int32(8)
        )
        s_pack0, s_pack1 = ld_shared_v2_u32(s_addr)
        s0, s1 = self._dequant_scale_x4_to_elem2x2(s_pack0)
        s2, s3 = self._dequant_scale_x4_to_elem2x2(s_pack1)
        return q0, q1, q2, q3, s0, s1, s2, s3

    @cute.jit
    def _load_b_scale_register_bundle(
        self,
        regs: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        b_sh_rd: Int32,
        s_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        q0, q1, q2, q3, s0, s1, s2, s3 = self._load_b_scale_registers(
            smem_base,
            tid,
            b_sh_rd,
            s_sh_rd,
            pipe,
            kk,
        )
        regs[0, 0] = q0
        regs[0, 1] = q1
        regs[0, 2] = q2
        regs[0, 3] = q3
        regs[1, 0] = s0
        regs[1, 1] = s1
        regs[1, 2] = s2
        regs[1, 3] = s3

    @cute.jit
    def _clear_b_scale_register_bundle(self, regs: cute.Tensor):
        for row in cutlass.range_constexpr(2):
            for col in cutlass.range_constexpr(4):
                regs[row, col] = Uint32(0)

    @cute.jit
    def _copy_b_scale_register_bundle(self, dst: cute.Tensor, src: cute.Tensor):
        for row in cutlass.range_constexpr(2):
            for col in cutlass.range_constexpr(4):
                dst[row, col] = src[row, col]

    @cute.jit
    def _select_b_scale_register(self, jj: cutlass.Constexpr[int], regs: cute.Tensor):
        return regs[0, jj], regs[1, jj]

    @cute.jit
    def _load_next_fragment_bundle(
        self,
        b_scale_next: cute.Tensor,
        a_regs_next: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        b_sh_rd: Int32,
        s_sh_rd: Int32,
        a_sh_rd: Int32,
        pipe: cutlass.Constexpr[int],
        kk: cutlass.Constexpr[int],
        tile_idx: Int32,
        k_tiles: Int32,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        self._clear_b_scale_register_bundle(b_scale_next)
        self._clear_a_register_bundle(a_regs_next, uses_m_block_8)

        if cutlass.const_expr(kk + 1 < self.b_sh_wr_iters):
            if tile_idx < k_tiles:
                self._load_b_scale_register_bundle(
                    b_scale_next,
                    smem_base,
                    tid,
                    b_sh_rd,
                    s_sh_rd,
                    Int32(pipe),
                    Int32(kk + 1),
                )
                self._load_a_register_bundle(
                    a_regs_next,
                    smem_base,
                    a_sh_rd,
                    Int32(pipe),
                    Int32(kk + 1),
                    uses_m_block_8,
                )
        else:
            next_tile = tile_idx + Int32(1)
            if next_tile < k_tiles:
                self._load_b_scale_register_bundle(
                    b_scale_next,
                    smem_base,
                    tid,
                    b_sh_rd,
                    s_sh_rd,
                    Int32((pipe + 1) % _STAGES),
                    Int32(0),
                )
                self._load_a_register_bundle(
                    a_regs_next,
                    smem_base,
                    a_sh_rd,
                    Int32((pipe + 1) % _STAGES),
                    Int32(0),
                    uses_m_block_8,
                )

    @cute.jit
    def _scaled_dequant_b_fragment(self, frag: cute.Tensor, q: Uint32, s: Uint32):
        bq1 = q
        bq0 = bq1 << Uint32(8)
        b0_0, b0_1 = self._dequant_e2m1x4_to_elem2x2(bq0)
        b1_0, b1_1 = self._dequant_e2m1x4_to_elem2x2(bq1)
        s_lane0 = bfloat2_broadcast_lane(s, Int32(0))
        s_lane1 = bfloat2_broadcast_lane(s, Int32(1))
        b0_0 = self._elem2_mul(b0_0, s_lane0)
        b0_1 = self._elem2_mul(b0_1, s_lane0)
        b1_0 = self._elem2_mul(b1_0, s_lane1)
        b1_1 = self._elem2_mul(b1_1, s_lane1)
        frag[0, 0] = b0_0
        frag[0, 1] = b0_1
        frag[1, 0] = b1_0
        frag[1, 1] = b1_1

    @cute.jit
    def _scaled_dequant_b_fragment_nf3(
        self, frag: cute.Tensor, lo16: Uint32, hi8: Uint32, s: Uint32
    ):
        # NF3: 8 3-bit codes (2 lo-bits in lo16, 1 hi-bit in hi8) -> 4 bf16x2
        # codebook fragments, in the SAME element order the packed path uses:
        # o0=(cb[c0],cb[c1]) -> frag[0,0], o1 -> frag[0,1], o2 -> frag[1,0],
        # o3 -> frag[1,1]. frag[0,*] takes the N-col-0 scale lane, frag[1,*] the
        # N-col-1 lane (identical broadcast to _scaled_dequant_b_fragment).
        o0, o1, o2, o3 = packed_dequant_nf3x8_to_bfloat2x4(
            lo16,
            hi8,
            Uint32(self._nf3_cb0),
            Uint32(self._nf3_cb1),
            Uint32(self._nf3_cb2),
            Uint32(self._nf3_cb3),
        )
        s_lane0 = bfloat2_broadcast_lane(s, Int32(0))
        s_lane1 = bfloat2_broadcast_lane(s, Int32(1))
        frag[0, 0] = self._elem2_mul(o0, s_lane0)
        frag[0, 1] = self._elem2_mul(o1, s_lane0)
        frag[1, 0] = self._elem2_mul(o2, s_lane1)
        frag[1, 1] = self._elem2_mul(o3, s_lane1)

    @cute.jit
    def _mma_accumulate_m8(
        self,
        acc,
        jj: cutlass.Constexpr[int],
        a_regs: cute.Tensor,
        b_frag: cute.Tensor,
    ):
        d0, d1, d2, d3 = self._mma_rhs_fragments_as_mma_a_m16n8k16_f32(
            acc[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            b_frag[0, 0],
            b_frag[1, 0],
            b_frag[0, 1],
            b_frag[1, 1],
            a_regs[0],
            a_regs[1],
        )
        acc[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d0
        acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d1
        acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d2
        acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d3

    @cute.jit
    def _mma_accumulate_large_m(
        self,
        acc,
        a_regs: cute.Tensor,
        mb: cutlass.Constexpr[int],
        jj: cutlass.Constexpr[int],
        b_frag: cute.Tensor,
    ):
        d0, d1, d2, d3 = self._mma_m16n8k16_f32(
            acc[(jj * 8) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 8) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            acc[(jj * 8 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 8 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            acc[(jj * 8 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 8 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            acc[(jj * 8 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 8 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            a_regs[mb, 0],
            a_regs[mb, 1],
            a_regs[mb, 2],
            a_regs[mb, 3],
            b_frag[0, 0],
            b_frag[0, 1],
        )
        acc[(jj * 8) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 8) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d0
        acc[(jj * 8 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 8 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d1
        acc[(jj * 8 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 8 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d2
        acc[(jj * 8 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 8 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d3
        d0, d1, d2, d3 = self._mma_m16n8k16_f32(
            acc[(jj * 8 + 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 8 + 4) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            acc[(jj * 8 + 5) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 8 + 5) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            acc[(jj * 8 + 6) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 8 + 6) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            acc[(jj * 8 + 7) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 8 + 7) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            a_regs[mb, 0],
            a_regs[mb, 1],
            a_regs[mb, 2],
            a_regs[mb, 3],
            b_frag[1, 0],
            b_frag[1, 1],
        )
        acc[(jj * 8 + 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 8 + 4) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d0
        acc[(jj * 8 + 5) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 8 + 5) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d1
        acc[(jj * 8 + 6) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 8 + 6) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d2
        acc[(jj * 8 + 7) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 8 + 7) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d3

    @cute.jit
    def _source_n_from_logical(self, logical_n: Int32) -> Int32:
        source_n = logical_n
        if cutlass.const_expr(self.source_n_rotation != 0):
            source_n += Int32(self.source_n_rotation)
            if source_n >= Int32(self.size_n):
                source_n -= Int32(self.size_n)
        return source_n

    @cute.jit
    def _stage_b_tile_modelopt_native(
        self,
        b_u8_flat: cute.Tensor,
        smem_addr: Int32,
        expert_idx: Int32,
        output_n_tile: Int32,
        tile_idx: Int32,
        local_int4: Int32,
    ):
        chunks_per_row = Int32(self.tile_k // 32)
        local_n = local_int4 // chunks_per_row
        local_k_vec = local_int4 - local_n * chunks_per_row
        logical_n = output_n_tile * Int32(self.tile_n) + local_n
        source_n = self._source_n_from_logical(logical_n)
        packed_cols = Int32(self.size_k // 2)
        tile_byte = tile_idx * Int32(self.tile_k // 2) + local_k_vec * Int32(16)
        byte_offset = (
            Int64(expert_idx) * Int64(self.size_n * (self.size_k // 2))
            + Int64(source_n) * Int64(packed_cols)
            + Int64(tile_byte)
        )
        if cutlass.const_expr(self.has_logical_tail):
            valid_n = logical_n < Int32(self.size_n)
            valid_bytes = packed_cols - tile_byte
            if valid_bytes > Int32(16):
                valid_bytes = Int32(16)
            if (
                valid_n
                and valid_bytes >= Int32(16)
                and cutlass.const_expr(not self.has_scale_k_tail)
            ):
                cp_async4_shared_global(
                    smem_addr,
                    get_ptr_as_int64(b_u8_flat, byte_offset),
                )
            else:
                v0 = Uint32(0)
                v1 = Uint32(0)
                v2 = Uint32(0)
                v3 = Uint32(0)
                if valid_n and valid_bytes > Int32(0):
                    src = get_ptr_as_int64(b_u8_flat, byte_offset)
                    if valid_bytes >= Int32(4):
                        v0 = ld_global_nc_u32(src)
                    if valid_bytes >= Int32(8):
                        v1 = ld_global_nc_u32(src + Int64(4))
                    if valid_bytes >= Int32(12):
                        v2 = ld_global_nc_u32(src + Int64(8))
                    if valid_bytes >= Int32(16):
                        v3 = ld_global_nc_u32(src + Int64(12))
                st_shared_v4_u32(smem_addr, v0, v1, v2, v3)
        else:
            cp_async4_shared_global(
                smem_addr,
                get_ptr_as_int64(b_u8_flat, byte_offset),
            )

    @cute.jit
    def _load_modelopt_shared_byte(
        self,
        smem_base: Int32,
        pipe: Int32,
        n_tile: Int32,
        k_tile: Int32,
        warp_id: Int32,
        tc_col: Int32,
        tc_row: Int32,
        n_delta: cutlass.Constexpr[int],
        k_delta: cutlass.Constexpr[int],
    ) -> Uint32:
        local_n = n_tile * Int32(64) + warp_id * Int32(16) + tc_col + Int32(n_delta)
        local_k = k_tile * Int32(16) + tc_row + Int32(k_delta)
        byte_offset = local_n * Int32(self.tile_k // 2) + local_k // Int32(2)
        word_byte_offset = byte_offset - (byte_offset & Int32(3))
        word = ld_shared_u32(
            smem_base
            + Int32(self.sh_b_off * 16)
            + pipe * Int32(self.b_sh_stage * 16)
            + word_byte_offset
        )
        shift = Uint32((byte_offset - word_byte_offset) * Int32(8))
        return (word >> shift) & Uint32(0xFF)

    @cute.jit
    def _pack_modelopt_byte_pair(
        self,
        word: Uint32,
        q: Uint32,
        low_shift: cutlass.Constexpr[int],
        high_shift: cutlass.Constexpr[int],
    ) -> Uint32:
        low = q & Uint32(0xF)
        high = (q >> Uint32(4)) & Uint32(0xF)
        return word | (low << Uint32(low_shift)) | (high << Uint32(high_shift))

    @cute.jit
    def _load_modelopt_shared_packed_word_for_lane(
        self,
        smem_base: Int32,
        pipe: Int32,
        n_tile: Int32,
        k_tile: Int32,
        warp_id: Int32,
        tc_col: Int32,
        tc_row: Int32,
    ) -> Uint32:
        q0 = self._load_modelopt_shared_byte(
            smem_base, pipe, n_tile, k_tile, warp_id, tc_col, tc_row, 0, 0
        )
        q1 = self._load_modelopt_shared_byte(
            smem_base, pipe, n_tile, k_tile, warp_id, tc_col, tc_row, 0, 8
        )
        q2 = self._load_modelopt_shared_byte(
            smem_base, pipe, n_tile, k_tile, warp_id, tc_col, tc_row, 8, 0
        )
        q3 = self._load_modelopt_shared_byte(
            smem_base, pipe, n_tile, k_tile, warp_id, tc_col, tc_row, 8, 8
        )
        word = Uint32(0)
        word = self._pack_modelopt_byte_pair(word, q0, 0, 16)
        word = self._pack_modelopt_byte_pair(word, q1, 4, 20)
        word = self._pack_modelopt_byte_pair(word, q2, 8, 24)
        word = self._pack_modelopt_byte_pair(word, q3, 12, 28)
        return word

    @cute.jit
    def _load_b_registers_modelopt_shared(
        self,
        smem_base: Int32,
        b_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        packed_word_index = (Int32(self.b_sh_stride) * kk + b_sh_rd) * Int32(4)
        words_per_k_tile = Int32((self.tile_n // 64) * 128)
        k_tile = packed_word_index // words_per_k_tile
        pos_in_k_tile = packed_word_index - k_tile * words_per_k_tile
        n_tile = pos_in_k_tile // Int32(128)
        pos = pos_in_k_tile - n_tile * Int32(128)
        th_id = pos // Int32(4)
        tc_col = th_id // Int32(4)
        tc_row = (th_id - tc_col * Int32(4)) * Int32(2)
        q0 = self._load_modelopt_shared_packed_word_for_lane(
            smem_base, pipe, n_tile, k_tile, Int32(0), tc_col, tc_row
        )
        q1 = self._load_modelopt_shared_packed_word_for_lane(
            smem_base, pipe, n_tile, k_tile, Int32(1), tc_col, tc_row
        )
        q2 = self._load_modelopt_shared_packed_word_for_lane(
            smem_base, pipe, n_tile, k_tile, Int32(2), tc_col, tc_row
        )
        q3 = self._load_modelopt_shared_packed_word_for_lane(
            smem_base, pipe, n_tile, k_tile, Int32(3), tc_col, tc_row
        )
        return q0, q1, q2, q3

    @cute.jit
    def _stage_k_tile_async(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pipe: Int32,
        tile_idx: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
        expert_idx: Int32,
    ):
        for i in cutlass.range_constexpr(self.a_sh_wr_iters):
            row = a_rows_per_iter * Int32(i) + a_gl_rd_row
            route_index = Int32(0)
            if row < Int32(self.moe_block_size):
                route_index = ld_shared_i32_relaxed(
                    smem_base + Int32(self.sh_rd_route_off * 16) + row * Int32(4)
                )
            a_int4 = (
                route_index * a_gl_stride
                + tile_idx * Int32(self.a_gl_rd_delta_o)
                + a_gl_rd_col0
            )
            a_dst = self._int4_addr(
                smem_base,
                Int32(self.sh_a_off)
                + pipe * Int32(self.a_sh_stage)
                + self._activation_smem_permuted_offset(
                    Int32(i * self.a_sh_wr_delta) + a_sh_wr
                ),
            )
            if cutlass.const_expr(self.has_k_tile_tail):
                a_k_int4 = tile_idx * Int32(self.a_gl_rd_delta_o) + a_gl_rd_col0
                if row < block_valid_rows and a_k_int4 < a_gl_stride:
                    cp_async4_shared_global(
                        a_dst,
                        get_ptr_as_int64(a_bf16_flat, a_int4 * Int32(8)),
                    )
                else:
                    st_shared_v4_u32(a_dst, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
            else:
                cp_async4_shared_global_pred(
                    a_dst,
                    get_ptr_as_int64(a_bf16_flat, a_int4 * Int32(8)),
                    (row < block_valid_rows).to(Int32),
                )

        if cutlass.const_expr(self.weight_layout_nf3):
            # NF3 flat-span staging: the packer lays the per-(expert,
            # output_n_tile, k-stage) B block out as ONE contiguous 12-byte-triple
            # span (n_tile-major, then K16-row, then N-pair), so we copy it
            # verbatim as 16-byte cp.async chunks. SMEM byte X of the pipe's B
            # region == global byte X of the span, so the register read (unit*12)
            # indexes it directly. b_gl_rd_base is unused on this path.
            nf3_units_per_expert = (self.size_n * self.size_k) // 32
            nf3_ntile_stride = (self.size_k // 16) * (self.tile_n // 2)
            nf3_span_base_unit = (
                Int32(nf3_units_per_expert) * expert_idx
                + Int32(nf3_ntile_stride) * output_n_tile
                + tile_idx * Int32(self.cta_k_blocks * self.b_sh_stride)
            )
            for i in cutlass.range_constexpr(self.b_sh_wr_iters_nf3):
                nf3_chunk = Int32(i * self.cta_threads) + tid
                b_dst = (
                    smem_base
                    + Int32(self.sh_b_off * 16)
                    + pipe * Int32(self.b_sh_stage_bytes)
                    + nf3_chunk * Int32(16)
                )
                # int32-element index of the chunk's first word:
                # (span_base_unit*12 + chunk*16) / 4 = span_base_unit*3 + chunk*4.
                b_src_i32 = nf3_span_base_unit * Int32(3) + nf3_chunk * Int32(4)
                cp_async4_shared_global_pred(
                    b_dst,
                    get_ptr_as_int64(b_i32_flat, b_src_i32),
                    (nf3_chunk < Int32(self.b_sh_chunks)).to(Int32),
                )

        for i in cutlass.range_constexpr(
            0 if self.weight_layout_nf3 else self.b_sh_wr_iters
        ):
            b_src_int4 = (
                b_gl_rd_base
                + tile_idx * Int32(self.cta_k_blocks) * b_gl_stride
                + Int32(i * (self.cta_threads // self.b_sh_stride)) * b_gl_stride
            )
            b_dst = self._int4_addr(
                smem_base,
                Int32(self.sh_b_off)
                + pipe * Int32(self.b_sh_stage)
                + Int32(i * self.cta_threads)
                + tid,
            )
            if cutlass.const_expr(self.weight_layout == "packed"):
                cp_async4_shared_global(
                    b_dst,
                    get_ptr_as_int64(b_i32_flat, b_src_int4 * Int32(4)),
                )
            else:
                self._stage_b_tile_modelopt_native(
                    b_i32_flat,
                    b_dst,
                    expert_idx,
                    output_n_tile,
                    tile_idx,
                    Int32(i * self.cta_threads) + tid,
                )

        if tid < Int32(self.s_sh_stage):
            s_k_group = tile_idx * Int32(self.s_tb_groups) + tid // Int32(
                self.s_sh_stride
            )
            s_n_group = Int32(self.s_sh_stride) * output_n_tile + (
                tid % Int32(self.s_sh_stride)
            )
            s_src_int4 = scales_expert_off + s_gl_stride * s_k_group + s_n_group
            s_dst = self._int4_addr(
                smem_base,
                Int32(self.sh_s_off) + pipe * Int32(self.s_sh_stage) + tid,
            )
            if cutlass.const_expr(self.has_logical_tail):
                if s_k_group < Int32(self.scale_k_groups) and s_n_group < Int32(
                    self.scale_n_groups
                ):
                    cp_async4_shared_global(
                        s_dst,
                        get_ptr_as_int64(scales_i32_flat, s_src_int4 * Int32(4)),
                    )
                else:
                    st_shared_v4_u32(s_dst, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
            else:
                cp_async4_shared_global(
                    s_dst,
                    get_ptr_as_int64(scales_i32_flat, s_src_int4 * Int32(4)),
                )

        cute.arch.cp_async_commit_group()

    @cute.jit
    def _prefetch_pipeline_step(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pipe: cutlass.Constexpr[int],
        kk: cutlass.Constexpr[int],
        tile_idx: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
        expert_idx: Int32,
    ):
        if cutlass.const_expr(kk == self.b_sh_wr_iters - 2):
            self._prefetch_lookahead_tile(
                a_bf16_flat,
                b_i32_flat,
                scales_i32_flat,
                smem_base,
                tid,
                pipe,
                tile_idx,
                k_tiles,
                reduce_k_tile,
                block_valid_rows,
                a_gl_stride,
                b_gl_stride,
                s_gl_stride,
                scales_expert_off,
                b_gl_rd_base,
                a_gl_rd_row,
                a_gl_rd_col0,
                a_sh_wr,
                a_rows_per_iter,
                output_n_tile,
                expert_idx,
            )

    @cute.jit
    def _prefetch_initial_tiles(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
        expert_idx: Int32,
    ):
        for pipe in cutlass.range_constexpr(_STAGES - 1):
            if Int32(pipe) < k_tiles:
                self._stage_k_tile_async(
                    a_bf16_flat,
                    b_i32_flat,
                    scales_i32_flat,
                    smem_base,
                    tid,
                    Int32(pipe),
                    reduce_k_tile + Int32(pipe),
                    block_valid_rows,
                    a_gl_stride,
                    b_gl_stride,
                    s_gl_stride,
                    scales_expert_off,
                    b_gl_rd_base,
                    a_gl_rd_row,
                    a_gl_rd_col0,
                    a_sh_wr,
                    a_rows_per_iter,
                    output_n_tile,
                    expert_idx,
                )
            else:
                cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(_STAGES - 2)
        cute.arch.sync_threads()

    @cute.jit
    def _prefetch_lookahead_tile(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pipe: cutlass.Constexpr[int],
        tile_idx: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
        expert_idx: Int32,
    ):
        fetch_tile = tile_idx + Int32(_STAGES - 1)
        if fetch_tile < k_tiles:
            self._stage_k_tile_async(
                a_bf16_flat,
                b_i32_flat,
                scales_i32_flat,
                smem_base,
                tid,
                Int32((pipe + _STAGES - 1) % _STAGES),
                reduce_k_tile + fetch_tile,
                block_valid_rows,
                a_gl_stride,
                b_gl_stride,
                s_gl_stride,
                scales_expert_off,
                b_gl_rd_base,
                a_gl_rd_row,
                a_gl_rd_col0,
                a_sh_wr,
                a_rows_per_iter,
                output_n_tile,
                expert_idx,
            )
        else:
            cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(_STAGES - 2)
        cute.arch.sync_threads()

    @cute.jit
    def _reduction_offsets(self, tid: Int32):
        red_idx = tid // Int32(self.b_sh_stride_threads)
        red_sh_stride = Int32(self.b_sh_stride_threads * 4 * 2)
        red_sh_delta = Int32(self.b_sh_stride_threads)
        red_sh_rd = red_sh_stride * (tid // Int32(self.b_sh_stride_threads)) + (
            tid % Int32(self.b_sh_stride_threads)
        )
        return red_idx, red_sh_stride, red_sh_delta, red_sh_rd

    @cute.jit
    def _fold_cta_partials_m8(self, acc, smem_base: Int32, tid: Int32):
        red_off = self.cta_threads // self.b_sh_stride_threads // 2
        if cutlass.const_expr(red_off >= 1):
            red_idx, red_sh_stride, red_sh_delta, red_sh_rd = self._reduction_offsets(
                tid
            )
            if cutlass.const_expr(red_off == 2):
                if Int32(2) <= red_idx and red_idx < Int32(4):
                    for jj in cutlass.range_constexpr(4):
                        red_sh_wr = red_sh_delta * Int32(jj * 2) + (
                            red_sh_rd - red_sh_stride * Int32(2)
                        )
                        st_shared_v4_f32(
                            self._int4_addr(
                                smem_base, Int32(self.sh_red_off) + red_sh_wr
                            ),
                            acc[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                            ],
                            acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                            ],
                            acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                            ],
                            acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                            ],
                        )
                cute.arch.sync_threads()

            if Int32(1) <= red_idx and red_idx < Int32(2):
                for jj in cutlass.range_constexpr(4):
                    red_sh_wr = red_sh_delta * Int32(jj * 2) + (
                        red_sh_rd - red_sh_stride
                    )
                    if cutlass.const_expr(red_off > 1):
                        rd_addr = self._int4_addr(
                            smem_base,
                            Int32(self.sh_red_off)
                            + red_sh_delta * Int32(jj * 2)
                            + red_sh_rd,
                        )
                        wr_addr = self._int4_addr(
                            smem_base, Int32(self.sh_red_off) + red_sh_wr
                        )
                        r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                        w0, w1, w2, w3 = ld_shared_v4_f32(wr_addr)
                        acc[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ] = (
                            acc[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                            ]
                            + r0
                            + w0
                        )
                        acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ] = (
                            acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                            ]
                            + r1
                            + w1
                        )
                        acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ] = (
                            acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                            ]
                            + r2
                            + w2
                        )
                        acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ] = (
                            acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                            ]
                            + r3
                            + w3
                        )
                    st_shared_v4_f32(
                        self._int4_addr(smem_base, Int32(self.sh_red_off) + red_sh_wr),
                        acc[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                    )
            cute.arch.sync_threads()

            if red_idx == Int32(0):
                for jj in cutlass.range_constexpr(4):
                    rd_addr = self._int4_addr(
                        smem_base,
                        Int32(self.sh_red_off)
                        + red_sh_delta * Int32(jj * 2)
                        + red_sh_rd,
                    )
                    r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                    acc[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] = (
                        acc[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ]
                        + r0
                    )
                    acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] = (
                        acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ]
                        + r1
                    )
                    acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] = (
                        acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ]
                        + r2
                    )
                    acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] = (
                        acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ]
                        + r3
                    )
            cute.arch.sync_threads()

    @cute.jit
    def _output_store_cursor(self, tid: Int32, output_n_tile: Int32):
        c_gl_stride = Int32(self.size_n // 8)
        c_sh_stride = Int32(2 * self.cta_n_blocks + 1)
        c_gl_wr_delta = c_gl_stride * Int32(self.cta_threads // (2 * self.cta_n_blocks))
        c_sh_rd_delta = c_sh_stride * Int32(self.cta_threads // (2 * self.cta_n_blocks))
        c_gl_wr = (
            c_gl_stride * (tid // Int32(2 * self.cta_n_blocks))
            + (tid % Int32(2 * self.cta_n_blocks))
            + Int32(2 * self.cta_n_blocks) * output_n_tile
        )
        c_sh_rd = c_sh_stride * (tid // Int32(2 * self.cta_n_blocks)) + (
            tid % Int32(2 * self.cta_n_blocks)
        )
        return c_gl_stride, c_sh_stride, c_gl_wr_delta, c_sh_rd_delta, c_gl_wr, c_sh_rd

    @cute.jit
    def _output_store_cursor_tail(self, tid: Int32, output_n_tile: Int32):
        c_gl_stride = Int32(self.size_n // 8)
        c_gl_stride_covered = Int32(self.covered_size_n // 8)
        c_sh_stride = Int32(2 * self.cta_n_blocks + 1)
        c_gl_wr_delta = c_gl_stride_covered * Int32(
            self.cta_threads // (2 * self.cta_n_blocks)
        )
        c_sh_rd_delta = c_sh_stride * Int32(self.cta_threads // (2 * self.cta_n_blocks))
        c_gl_wr = (
            c_gl_stride_covered * (tid // Int32(2 * self.cta_n_blocks))
            + (tid % Int32(2 * self.cta_n_blocks))
            + Int32(2 * self.cta_n_blocks) * output_n_tile
        )
        c_sh_rd = c_sh_stride * (tid // Int32(2 * self.cta_n_blocks)) + (
            tid % Int32(2 * self.cta_n_blocks)
        )
        return (
            c_gl_stride,
            c_gl_stride_covered,
            c_sh_stride,
            c_gl_wr_delta,
            c_sh_rd_delta,
            c_gl_wr,
            c_sh_rd,
        )

    @cute.jit
    def _drain_output_smem(
        self,
        c_bf16_flat: cute.Tensor,
        smem_base: Int32,
        c_gl_stride: Int32,
        c_gl_wr: Int32,
        c_gl_wr_delta: Int32,
        c_sh_rd: Int32,
        c_sh_rd_delta: Int32,
        block_valid_rows: Int32,
        store_iters: cutlass.Constexpr[int],
    ):
        for _ in cutlass.range_constexpr(store_iters):
            row = c_gl_wr // c_gl_stride
            if row < block_valid_rows:
                route_index = ld_shared_i32_relaxed(
                    smem_base + Int32(self.sh_route_off * 16) + row * Int32(4)
                )
                true_idx = route_index * c_gl_stride + (c_gl_wr % c_gl_stride)
                q0, q1, q2, q3 = ld_shared_v4_u32(
                    self._int4_addr(smem_base, Int32(self.sh_red_off) + c_sh_rd)
                )
                if cutlass.const_expr(self.mul_topk_weights):
                    scale_bf2 = ld_shared_u32(
                        smem_base + Int32(self.sh_topk_off * 16) + row * Int32(4)
                    )
                    q0 = self._elem2_mul(q0, scale_bf2)
                    q1 = self._elem2_mul(q1, scale_bf2)
                    q2 = self._elem2_mul(q2, scale_bf2)
                    q3 = self._elem2_mul(q3, scale_bf2)
                if cutlass.const_expr(self.epilogue_relu2):
                    q0 = self._relu2_elem2(q0)
                    q1 = self._relu2_elem2(q1)
                    q2 = self._relu2_elem2(q2)
                    q3 = self._relu2_elem2(q3)
                if cutlass.const_expr(self.fused_topk_sum):
                    # Fold the per-route partials into the per-token output in
                    # the epilogue (drops the separate top-k-sum launch). The
                    # output must be zeroed before launch; each route slot maps
                    # to token = route_index // top_k.  bf16x2 add lands two
                    # consecutive hidden lanes per word.
                    token_idx = route_index // Int32(self.fused_sum_topk)
                    out_idx = token_idx * c_gl_stride + (c_gl_wr % c_gl_stride)
                    out_addr = get_ptr_as_int64(c_bf16_flat, out_idx * Int32(8))
                    red_add_global_bf16x2(out_addr, q0)
                    red_add_global_bf16x2(out_addr + Int64(4), q1)
                    red_add_global_bf16x2(out_addr + Int64(8), q2)
                    red_add_global_bf16x2(out_addr + Int64(12), q3)
                else:
                    st_global_v4_u32(
                        get_ptr_as_int64(c_bf16_flat, true_idx * Int32(8)),
                        q0,
                        q1,
                        q2,
                        q3,
                    )
            c_gl_wr += c_gl_wr_delta
            c_sh_rd += c_sh_rd_delta
        cute.arch.sync_threads()

    @cute.jit
    def _drain_output_smem_tail(
        self,
        c_bf16_flat: cute.Tensor,
        smem_base: Int32,
        c_gl_stride: Int32,
        c_gl_stride_covered: Int32,
        c_gl_wr: Int32,
        c_gl_wr_delta: Int32,
        c_sh_rd: Int32,
        c_sh_rd_delta: Int32,
        block_valid_rows: Int32,
        store_iters: cutlass.Constexpr[int],
    ):
        for _ in cutlass.range_constexpr(store_iters):
            row = c_gl_wr // c_gl_stride_covered
            col_word = c_gl_wr - row * c_gl_stride_covered
            if row < block_valid_rows and col_word < c_gl_stride:
                route_index = ld_shared_i32_relaxed(
                    smem_base + Int32(self.sh_route_off * 16) + row * Int32(4)
                )
                true_idx = route_index * c_gl_stride + col_word
                q0, q1, q2, q3 = ld_shared_v4_u32(
                    self._int4_addr(smem_base, Int32(self.sh_red_off) + c_sh_rd)
                )
                if cutlass.const_expr(self.mul_topk_weights):
                    scale_bf2 = ld_shared_u32(
                        smem_base + Int32(self.sh_topk_off * 16) + row * Int32(4)
                    )
                    q0 = self._elem2_mul(q0, scale_bf2)
                    q1 = self._elem2_mul(q1, scale_bf2)
                    q2 = self._elem2_mul(q2, scale_bf2)
                    q3 = self._elem2_mul(q3, scale_bf2)
                if cutlass.const_expr(self.epilogue_relu2):
                    q0 = self._relu2_elem2(q0)
                    q1 = self._relu2_elem2(q1)
                    q2 = self._relu2_elem2(q2)
                    q3 = self._relu2_elem2(q3)
                if cutlass.const_expr(self.fused_topk_sum):
                    token_idx = route_index // Int32(self.fused_sum_topk)
                    out_idx = token_idx * c_gl_stride + col_word
                    out_addr = get_ptr_as_int64(c_bf16_flat, out_idx * Int32(8))
                    red_add_global_bf16x2(out_addr, q0)
                    red_add_global_bf16x2(out_addr + Int64(4), q1)
                    red_add_global_bf16x2(out_addr + Int64(8), q2)
                    red_add_global_bf16x2(out_addr + Int64(12), q3)
                else:
                    st_global_v4_u32(
                        get_ptr_as_int64(c_bf16_flat, true_idx * Int32(8)),
                        q0,
                        q1,
                        q2,
                        q3,
                    )
            c_gl_wr += c_gl_wr_delta
            c_sh_rd += c_sh_rd_delta
        cute.arch.sync_threads()

    @cute.jit
    def _store_tile_m8(
        self,
        acc,
        c_bf16_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        output_n_tile: Int32,
        block_valid_rows: Int32,
        global_scale_f32: cutlass.Float32,
    ):
        if cutlass.const_expr(self.has_n_tile_tail):
            (
                c_gl_stride,
                c_gl_stride_covered,
                c_sh_stride,
                c_gl_wr_delta,
                c_sh_rd_delta,
                c_gl_wr,
                c_sh_rd,
            ) = self._output_store_cursor_tail(tid, output_n_tile)
        else:
            (
                c_gl_stride,
                c_sh_stride,
                c_gl_wr_delta,
                c_sh_rd_delta,
                c_gl_wr,
                c_sh_rd,
            ) = self._output_store_cursor(tid, output_n_tile)
            c_gl_stride_covered = c_gl_stride
        c_sh_wr = (
            Int32(8) * c_sh_stride * (((tid & Int32(31)) % Int32(4)) * Int32(2))
            + (tid & Int32(31)) // Int32(4)
            + Int32(64) * (tid // Int32(32))
        )

        if tid // Int32(32) < Int32(self.tb_n_warps):
            write_scale = cutlass.Float32(1.0)
            if cutlass.const_expr(not self.mul_topk_weights):
                write_scale = global_scale_f32
            for jj in cutlass.range_constexpr(4):
                wr = c_sh_wr + Int32(16 * jj)
                self._st_shared_elem_from_f32(
                    smem_base + Int32(self.sh_red_off * 16) + (wr * Int32(2)),
                    acc[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ]
                    * write_scale,
                )
                self._st_shared_elem_from_f32(
                    smem_base
                    + Int32(self.sh_red_off * 16)
                    + ((wr + Int32(8) * c_sh_stride) * Int32(2)),
                    acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ]
                    * write_scale,
                )
                self._st_shared_elem_from_f32(
                    smem_base
                    + Int32(self.sh_red_off * 16)
                    + ((wr + Int32(8)) * Int32(2)),
                    acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ]
                    * write_scale,
                )
                self._st_shared_elem_from_f32(
                    smem_base
                    + Int32(self.sh_red_off * 16)
                    + ((wr + Int32(8) + Int32(8) * c_sh_stride) * Int32(2)),
                    acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ]
                    * write_scale,
                )
        cute.arch.sync_threads()

        store_iters = _covering_count(16, self.cta_threads // (2 * self.cta_n_blocks))
        if cutlass.const_expr(self.has_n_tile_tail):
            self._drain_output_smem_tail(
                c_bf16_flat,
                smem_base,
                c_gl_stride,
                c_gl_stride_covered,
                c_gl_wr,
                c_gl_wr_delta,
                c_sh_rd,
                c_sh_rd_delta,
                block_valid_rows,
                store_iters,
            )
        else:
            self._drain_output_smem(
                c_bf16_flat,
                smem_base,
                c_gl_stride,
                c_gl_wr,
                c_gl_wr_delta,
                c_sh_rd,
                c_sh_rd_delta,
                block_valid_rows,
                store_iters,
            )

    @cute.jit
    def _fold_cta_partials_large_m(
        self,
        acc0,
        acc1,
        acc2,
        acc3,
        smem_base: Int32,
        tid: Int32,
    ):
        red_off = self.cta_threads // self.b_sh_stride_threads // 2
        if cutlass.const_expr(red_off >= 1):
            red_idx, red_sh_stride, red_sh_delta, red_sh_rd = self._reduction_offsets(
                tid
            )

            for mb in cutlass.range_constexpr(self.cta_m_blocks):
                if cutlass.const_expr(mb == 0):
                    self._fold_cta_partials_large_m_block(
                        acc0,
                        smem_base,
                        red_off,
                        red_idx,
                        red_sh_stride,
                        red_sh_delta,
                        red_sh_rd,
                    )
                elif cutlass.const_expr(mb == 1):
                    self._fold_cta_partials_large_m_block(
                        acc1,
                        smem_base,
                        red_off,
                        red_idx,
                        red_sh_stride,
                        red_sh_delta,
                        red_sh_rd,
                    )
                elif cutlass.const_expr(mb == 2):
                    self._fold_cta_partials_large_m_block(
                        acc2,
                        smem_base,
                        red_off,
                        red_idx,
                        red_sh_stride,
                        red_sh_delta,
                        red_sh_rd,
                    )
                else:
                    self._fold_cta_partials_large_m_block(
                        acc3,
                        smem_base,
                        red_off,
                        red_idx,
                        red_sh_stride,
                        red_sh_delta,
                        red_sh_rd,
                    )

    @cute.jit
    def _fold_cta_partials_large_m_block(
        self,
        acc,
        smem_base: Int32,
        red_off: cutlass.Constexpr[int],
        red_idx: Int32,
        red_sh_stride: Int32,
        red_sh_delta: Int32,
        red_sh_rd: Int32,
    ):
        if cutlass.const_expr(red_off == 2):
            if Int32(2) <= red_idx and red_idx < Int32(4):
                for flat_j in cutlass.range_constexpr(8):
                    red_sh_wr = red_sh_delta * Int32(flat_j) + (
                        red_sh_rd - red_sh_stride * Int32(2)
                    )
                    st_shared_v4_f32(
                        self._int4_addr(smem_base, Int32(self.sh_red_off) + red_sh_wr),
                        acc[(flat_j * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc[(flat_j * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc[(flat_j * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc[(flat_j * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                    )
            cute.arch.sync_threads()

        if Int32(1) <= red_idx and red_idx < Int32(2):
            for flat_j in cutlass.range_constexpr(8):
                red_sh_wr = red_sh_delta * Int32(flat_j) + (red_sh_rd - red_sh_stride)
                if cutlass.const_expr(red_off > 1):
                    rd_addr = self._int4_addr(
                        smem_base,
                        Int32(self.sh_red_off)
                        + red_sh_delta * Int32(flat_j)
                        + red_sh_rd,
                    )
                    wr_addr = self._int4_addr(
                        smem_base,
                        Int32(self.sh_red_off) + red_sh_wr,
                    )
                    r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                    w0, w1, w2, w3 = ld_shared_v4_f32(wr_addr)
                    acc[(flat_j * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] = (
                        acc[(flat_j * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ]
                        + r0
                        + w0
                    )
                    acc[(flat_j * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] = (
                        acc[(flat_j * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ]
                        + r1
                        + w1
                    )
                    acc[(flat_j * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] = (
                        acc[(flat_j * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ]
                        + r2
                        + w2
                    )
                    acc[(flat_j * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] = (
                        acc[(flat_j * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ]
                        + r3
                        + w3
                    )
                st_shared_v4_f32(
                    self._int4_addr(smem_base, Int32(self.sh_red_off) + red_sh_wr),
                    acc[(flat_j * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                    acc[(flat_j * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                    acc[(flat_j * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                    acc[(flat_j * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                )
        cute.arch.sync_threads()

        if red_idx == Int32(0):
            for flat_j in cutlass.range_constexpr(8):
                rd_addr = self._int4_addr(
                    smem_base,
                    Int32(self.sh_red_off) + red_sh_delta * Int32(flat_j) + red_sh_rd,
                )
                r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                acc[(flat_j * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (flat_j * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                ] = (
                    acc[(flat_j * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ]
                    + r0
                )
                acc[(flat_j * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (flat_j * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                ] = (
                    acc[(flat_j * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ]
                    + r1
                )
                acc[(flat_j * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (flat_j * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                ] = (
                    acc[(flat_j * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ]
                    + r2
                )
                acc[(flat_j * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (flat_j * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                ] = (
                    acc[(flat_j * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ]
                    + r3
                )
        cute.arch.sync_threads()

    @cute.jit
    def _write_bf16x2_shared(
        self,
        smem_base: Int32,
        half2_idx: Int32,
        c0: cutlass.Float32,
        c1: cutlass.Float32,
        write_scale: cutlass.Float32,
    ):
        packed = self._pack_f32x2_to_elem2(c0 * write_scale, c1 * write_scale)
        st_shared_u32(
            smem_base + Int32(self.sh_red_off * 16) + half2_idx * Int32(4),
            packed,
        )

    @cute.jit
    def _store_tile_large_m(
        self,
        acc0,
        acc1,
        acc2,
        acc3,
        c_bf16_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        output_n_tile: Int32,
        block_valid_rows: Int32,
        global_scale_f32: cutlass.Float32,
    ):
        if cutlass.const_expr(self.has_n_tile_tail):
            (
                c_gl_stride,
                c_gl_stride_covered,
                c_sh_stride,
                c_gl_wr_delta,
                c_sh_rd_delta,
                c_gl_wr,
                c_sh_rd,
            ) = self._output_store_cursor_tail(tid, output_n_tile)
        else:
            (
                c_gl_stride,
                c_sh_stride,
                c_gl_wr_delta,
                c_sh_rd_delta,
                c_gl_wr,
                c_sh_rd,
            ) = self._output_store_cursor(tid, output_n_tile)
            c_gl_stride_covered = c_gl_stride
        c_sh_wr = (
            Int32(4) * c_sh_stride * ((tid & Int32(31)) // Int32(4))
            + (tid & Int32(31)) % Int32(4)
            + Int32(32) * (tid // Int32(32))
        )

        if tid // Int32(32) < Int32(self.tb_n_warps):
            write_scale = cutlass.Float32(1.0)
            if cutlass.const_expr(not self.mul_topk_weights):
                write_scale = global_scale_f32
            for mb in cutlass.range_constexpr(self.cta_m_blocks):
                if cutlass.const_expr(mb == 0):
                    self._store_tile_large_m_block(
                        acc0, smem_base, c_sh_wr, c_sh_stride, write_scale
                    )
                elif cutlass.const_expr(mb == 1):
                    self._store_tile_large_m_block(
                        acc1, smem_base, c_sh_wr, c_sh_stride, write_scale
                    )
                elif cutlass.const_expr(mb == 2):
                    self._store_tile_large_m_block(
                        acc2, smem_base, c_sh_wr, c_sh_stride, write_scale
                    )
                else:
                    self._store_tile_large_m_block(
                        acc3, smem_base, c_sh_wr, c_sh_stride, write_scale
                    )
                c_sh_wr += Int32(16 * (4 * (2 * self.cta_n_blocks + 1)))
        cute.arch.sync_threads()

        store_iters = _covering_count(
            16 * self.cta_m_blocks,
            self.cta_threads // (2 * self.cta_n_blocks),
        )
        if cutlass.const_expr(self.has_n_tile_tail):
            self._drain_output_smem_tail(
                c_bf16_flat,
                smem_base,
                c_gl_stride,
                c_gl_stride_covered,
                c_gl_wr,
                c_gl_wr_delta,
                c_sh_rd,
                c_sh_rd_delta,
                block_valid_rows,
                store_iters,
            )
        else:
            self._drain_output_smem(
                c_bf16_flat,
                smem_base,
                c_gl_stride,
                c_gl_wr,
                c_gl_wr_delta,
                c_sh_rd,
                c_sh_rd_delta,
                block_valid_rows,
                store_iters,
            )

    @cute.jit
    def _store_tile_large_m_block(
        self,
        acc,
        smem_base: Int32,
        c_sh_wr: Int32,
        c_sh_stride: Int32,
        write_scale: cutlass.Float32,
    ):
        for jj in cutlass.range_constexpr(4):
            wr = c_sh_wr + Int32(8 * jj)
            self._write_bf16x2_shared(
                smem_base,
                wr,
                acc[(jj * 8) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (jj * 8) % _SCALAR_ACC_FRAGMENT_WIDTH
                ],
                acc[(jj * 8 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (jj * 8 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                ],
                write_scale,
            )
            self._write_bf16x2_shared(
                smem_base,
                wr + (Int32(4) * c_sh_stride) * Int32(8) + Int32(0),
                acc[(jj * 8 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (jj * 8 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                ],
                acc[(jj * 8 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (jj * 8 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                ],
                write_scale,
            )
            self._write_bf16x2_shared(
                smem_base,
                wr + Int32(4),
                acc[(jj * 8 + 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (jj * 8 + 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                ],
                acc[(jj * 8 + 5) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (jj * 8 + 5) % _SCALAR_ACC_FRAGMENT_WIDTH
                ],
                write_scale,
            )
            self._write_bf16x2_shared(
                smem_base,
                wr + (Int32(4) * c_sh_stride) * Int32(8) + Int32(4),
                acc[(jj * 8 + 6) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (jj * 8 + 6) % _SCALAR_ACC_FRAGMENT_WIDTH
                ],
                acc[(jj * 8 + 7) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (jj * 8 + 7) % _SCALAR_ACC_FRAGMENT_WIDTH
                ],
                write_scale,
            )


class W4A16FusedMoeKernel:
    def __init__(
        self,
        *,
        size_m: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        activation: str,
        apply_router_weight_on_input: bool,
        zero_fc2_output: bool,
        fc1_tile_n: int,
        fc1_tile_k: int,
        fc2_tile_n: int,
        fc2_tile_k: int,
        moe_block_size: int,
        max_m_blocks: int,
        element_dtype: str = "bf16",
        fast_math: bool = True,
        swiglu_limit: float | None = None,
        swiglu_alpha: float | None = None,
        swiglu_beta: float | None = None,
        weight_layout: str = "packed",
        scale_format: str = "e4m3_k16",
        w13_layout: str = "w13",
        direct_topk_routes: bool = False,
        tc_decode_fused_sum: bool = False,
        tc_zero_output: bool = True,
        collect_activation_amax: bool = False,
        schedule_whole_tiles: bool = False,
    ):
        activation = normalize_moe_activation(activation)
        is_gated = validate_activation(activation)
        swiglu_limit, swiglu_alpha, swiglu_beta = _normalize_activation_swiglu_params(
            activation,
            swiglu_limit,
            swiglu_alpha,
            swiglu_beta,
        )
        if weight_layout not in _WEIGHT_LAYOUTS:
            raise ValueError(f"unsupported W4A16 weight_layout {weight_layout!r}")
        scale_format = _normalize_scale_format(scale_format)
        if weight_layout == "modelopt":
            if w13_layout not in _MODEL_OPT_W13_LAYOUTS:
                raise ValueError(f"unsupported W4A16 w13_layout {w13_layout!r}")
        else:
            w13_layout = "packed"
        self.tc_decode_fused_sum = bool(tc_decode_fused_sum)
        # When two TC-decode launches share one pre-zeroed output (the NF3 hybrid
        # runs an NVFP4 launch then an NF3 launch into the same tensor), only the
        # first must zero it. Default True preserves single-launch behavior.
        self.tc_zero_output = bool(tc_zero_output)
        self.collect_activation_amax = bool(collect_activation_amax)
        if self.collect_activation_amax and bool(direct_topk_routes):
            raise ValueError("activation amax collection requires route-packed W4A16")
        if self.collect_activation_amax and self.tc_decode_fused_sum:
            raise ValueError(
                "activation amax collection is incompatible with TC-decode"
            )
        if self.tc_decode_fused_sum and not bool(direct_topk_routes):
            raise ValueError("tc_decode_fused_sum requires direct_topk_routes")
        if self.tc_decode_fused_sum and element_dtype != "bf16":
            raise ValueError("tc_decode_fused_sum currently requires bf16 activations")
        fc1_cols = int(intermediate_size) * (2 if is_gated else 1)
        routed_rows = int(size_m) * int(top_k)
        self.size_m = int(size_m)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.fc1_cols = int(fc1_cols)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.moe_block_size = int(moe_block_size)
        self.activation = activation
        self.activation_is_gated = is_gated
        self.activation_is_swigluoai = activation == SWIGLUOAI_UNINTERLEAVE
        self.has_swiglu_limit = swiglu_limit is not None
        self.swiglu_limit = 0.0 if swiglu_limit is None else swiglu_limit
        self.swiglu_alpha = float(swiglu_alpha)
        self.swiglu_beta = float(swiglu_beta)
        self.weight_layout = weight_layout
        self.scale_format = scale_format
        self.w13_layout = w13_layout
        self.apply_router_weight_on_input = bool(apply_router_weight_on_input)
        self.zero_fc2_output = bool(zero_fc2_output)
        self.element_dtype = element_dtype
        self.is_fp16 = element_dtype == "fp16"
        self.fast_math = bool(fast_math)
        self.direct_topk_routes = bool(direct_topk_routes)
        self.schedule_whole_tiles = bool(schedule_whole_tiles)
        fc1_source_n_rotation = (
            int(intermediate_size)
            if (weight_layout == "modelopt" and w13_layout == "w13" and is_gated)
            else 0
        )
        self.fc1 = W4A16GemmKernel(
            size_m=size_m,
            size_n=fc1_cols,
            size_k=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            mul_topk_weights=bool(apply_router_weight_on_input),
            tile_n=fc1_tile_n,
            tile_k=fc1_tile_k,
            moe_block_size=moe_block_size,
            max_m_blocks=max_m_blocks,
            element_dtype=element_dtype,
            epilogue_activation=None if is_gated else "relu2",
            weight_layout=weight_layout,
            scale_format=scale_format,
            w13_layout=w13_layout,
            source_n_rotation=fc1_source_n_rotation,
            single_token_route_fast_path=size_m == 1 and not self.direct_topk_routes,
            direct_topk_routes=self.direct_topk_routes,
            schedule_whole_tiles=self.schedule_whole_tiles,
        )
        self.fc2 = W4A16GemmKernel(
            size_m=routed_rows,
            size_n=hidden_size,
            size_k=intermediate_size,
            num_experts=num_experts,
            top_k=1,
            mul_topk_weights=not bool(apply_router_weight_on_input),
            tile_n=fc2_tile_n,
            tile_k=fc2_tile_k,
            moe_block_size=moe_block_size,
            max_m_blocks=max_m_blocks,
            element_dtype=element_dtype,
            weight_layout=weight_layout,
            scale_format=scale_format,
            w13_layout=w13_layout,
            single_token_route_fast_path=size_m == 1 and not self.direct_topk_routes,
            direct_topk_routes=self.direct_topk_routes,
            fused_topk_sum=self.tc_decode_fused_sum,
            fused_sum_topk=int(top_k),
            schedule_whole_tiles=self.schedule_whole_tiles,
        )
        self.cta_threads = max(self.fc1.cta_threads, self.fc2.cta_threads)
        if self.fc1.cta_threads != self.fc2.cta_threads:
            raise ValueError(
                "fused W4A16 kernel expects matching FC1/FC2 thread counts"
            )
        self.sms = self.fc1.sms
        self.blocks_per_sm = min(self.fc1.blocks_per_sm, self.fc2.blocks_per_sm)
        self.shared_words = max(self.fc1.shared_words, self.fc2.shared_words)
        self.barrier_count_off = self.sms * 4
        self.barrier_sense_off = self.sms * 4 + 1

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (
            self.hidden_size,
            self.intermediate_size,
            self.fc1_cols,
            self.num_experts,
            self.top_k,
            self.activation,
            self.activation_is_gated,
            self.activation_is_swigluoai,
            self.has_swiglu_limit,
            self.swiglu_limit,
            self.swiglu_alpha,
            self.swiglu_beta,
            self.weight_layout,
            self.scale_format,
            self.apply_router_weight_on_input,
            self.zero_fc2_output,
            self.element_dtype,
            self.fast_math,
            self.direct_topk_routes,
            self.tc_zero_output,
            self.collect_activation_amax,
            self.fc1.__cache_key__,
            self.fc2.__cache_key__,
            self.cta_threads,
            self.sms,
            self.shared_words,
            self.blocks_per_sm,
        )

    @cute.jit
    def _cast_elem(self, x: cutlass.Float32):
        if cutlass.const_expr(self.is_fp16):
            return cutlass.Float16(x)
        return cutlass.BFloat16(x)

    @cute.jit
    def _clamp_swiglu_inputs(
        self,
        gate: cutlass.Float32,
        up: cutlass.Float32,
    ):
        if cutlass.const_expr(self.has_swiglu_limit):
            limit = cutlass.Float32(self.swiglu_limit)
            neg_limit = cutlass.Float32(-self.swiglu_limit)
            if gate > limit:
                gate = limit
            if up > limit:
                up = limit
            if up < neg_limit:
                up = neg_limit
        return gate, up

    @cute.jit
    def __call__(
        self,
        a_bf16_ptr: cute.Pointer,
        w13_i32_flat: cute.Tensor,
        w2_i32_flat: cute.Tensor,
        fc1_bf16_flat: cute.Tensor,
        activated_bf16_flat: cute.Tensor,
        fc2_bf16_flat: cute.Tensor,
        w13_scales_i32_flat: cute.Tensor,
        w2_scales_i32_flat: cute.Tensor,
        w13_global_scale: cute.Tensor,
        w2_global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        activation_amax_flat: cute.Tensor,
        layer_idx: cutlass.Int32,
        topk_weights_ptr: cute.Pointer,
        fc1_c_tmp_f32_flat: cute.Tensor,
        fc2_c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        active_m: cutlass.Int32,
        grid_x: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        a_bf16_flat = cute.make_tensor(
            a_bf16_ptr,
            layout=cute.make_layout((active_m * Int32(self.hidden_size),), stride=(1,)),
        )
        topk_weights_flat = cute.make_tensor(
            topk_weights_ptr,
            layout=cute.make_layout((active_m * Int32(self.top_k),), stride=(1,)),
        )
        grid = (grid_x, 1, 1)
        self.kernel(
            a_bf16_flat,
            w13_i32_flat,
            w2_i32_flat,
            fc1_bf16_flat,
            activated_bf16_flat,
            fc2_bf16_flat,
            w13_scales_i32_flat,
            w2_scales_i32_flat,
            w13_global_scale,
            w2_global_scale,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            activation_amax_flat,
            layer_idx,
            topk_weights_flat,
            fc1_c_tmp_f32_flat,
            fc2_c_tmp_f32_flat,
            locks_i32_flat,
            active_m,
        ).launch(
            grid=grid,
            block=[self.cta_threads, 1, 1],
            min_blocks_per_mp=self.blocks_per_sm,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        a_bf16_flat: cute.Tensor,
        w13_i32_flat: cute.Tensor,
        w2_i32_flat: cute.Tensor,
        fc1_bf16_flat: cute.Tensor,
        activated_bf16_flat: cute.Tensor,
        fc2_bf16_flat: cute.Tensor,
        w13_scales_i32_flat: cute.Tensor,
        w2_scales_i32_flat: cute.Tensor,
        w13_global_scale: cute.Tensor,
        w2_global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        activation_amax_flat: cute.Tensor,
        layer_idx: cutlass.Int32,
        topk_weights_flat: cute.Tensor,
        fc1_c_tmp_f32_flat: cute.Tensor,
        fc2_c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        active_m: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_x_raw, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        cta = Int32(bidx)
        grid_x = Int32(grid_x_raw)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words],
                1024,
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())

        self._moe_body(
            a_bf16_flat,
            w13_i32_flat,
            w2_i32_flat,
            fc1_bf16_flat,
            activated_bf16_flat,
            fc2_bf16_flat,
            w13_scales_i32_flat,
            w2_scales_i32_flat,
            w13_global_scale,
            w2_global_scale,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            activation_amax_flat,
            layer_idx,
            topk_weights_flat,
            fc1_c_tmp_f32_flat,
            fc2_c_tmp_f32_flat,
            locks_i32_flat,
            smem_base,
            tid,
            cta,
            grid_x,
            active_m,
        )

    @cute.jit
    def _moe_body(
        self,
        a_bf16_flat: cute.Tensor,
        w13_i32_flat: cute.Tensor,
        w2_i32_flat: cute.Tensor,
        fc1_bf16_flat: cute.Tensor,
        activated_bf16_flat: cute.Tensor,
        fc2_bf16_flat: cute.Tensor,
        w13_scales_i32_flat: cute.Tensor,
        w2_scales_i32_flat: cute.Tensor,
        w13_global_scale: cute.Tensor,
        w2_global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        activation_amax_flat: cute.Tensor,
        layer_idx: cutlass.Int32,
        topk_weights_flat: cute.Tensor,
        fc1_c_tmp_f32_flat: cute.Tensor,
        fc2_c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: cutlass.Int32,
        fc1_emit_tile: cutlass.Constexpr = None,
        fc2_emit_tile: cutlass.Constexpr = None,
    ):
        # Phase assembly shared by the single-tier fused kernel and the hybrid
        # multi-tier entry: zero prologue, FC1, grid barrier, activation, grid
        # barrier, FC2. The emit hooks delegate per-tile expert resolution and
        # dispatch (used by the hybrid route map); None keeps the single-tier
        # resolution inside _run_persistent_gemm.
        if cutlass.const_expr(self.tc_decode_fused_sum):
            # The TC-decode FC2 epilogue atomically accumulates per-route
            # partials directly into the per-token output, so the output must be
            # pre-zeroed. Previously this was a SEPARATE host-side output.zero_()
            # kernel launch on the latency-bound decode critical path (an extra
            # launch + its grid-fill memset before the fused kernel even starts).
            # Fold it into the fused kernel prologue here: every CTA zeroes a
            # grid-strided slice of the output BEFORE FC1, and the EXISTING
            # post-FC1 grid barrier (already required to order FC1 writes before
            # the activation/FC2 read) makes all zero stores globally visible
            # before the first FC2 atomic -- so no extra barrier is added. The
            # tiny m*hidden bf16 memset (decode: <=4*4096 elems) is dwarfed by
            # FC1's whole-K FP4-weight stream, but we delete one whole kernel
            # launch from the per-decode chain. The TC-decode output is per-token
            # (top_k routes atomically summed into the SAME token row), so the
            # zero span is active_m*hidden_size -- NOT the per-route
            # active_m*top_k*hidden_size of _zero_fc2_output.
            # tc_zero_output=False skips the zero (a paired earlier launch has
            # already zeroed the shared output); the grid barrier below is
            # unconditional so ordering is preserved either way.
            if cutlass.const_expr(self.tc_zero_output):
                zidx = cta * Int32(self.cta_threads) + tid
                zstride = grid_x * Int32(self.cta_threads)
                ztotal = active_m * Int32(self.hidden_size)
                zzero = self._cast_elem(cutlass.Float32(0.0))
                while zidx < ztotal:
                    fc2_bf16_flat[zidx] = zzero
                    zidx += zstride

        if cutlass.const_expr(self.activation_is_gated):
            self.fc1._run_persistent_gemm(
                a_bf16_flat,
                w13_i32_flat,
                fc1_bf16_flat,
                w13_scales_i32_flat,
                w13_global_scale,
                packed_route_indices,
                block_expert_ids,
                packed_route_count,
                topk_weights_flat,
                fc1_c_tmp_f32_flat,
                locks_i32_flat,
                smem_base,
                tid,
                cta,
                grid_x,
                active_m,
                fc1_emit_tile,
            )
            self._grid_barrier(locks_i32_flat, tid, grid_x)
            self._run_activation(
                fc1_bf16_flat,
                activated_bf16_flat,
                tid,
                cta,
                grid_x,
                active_m,
            )
        else:
            self.fc1._run_persistent_gemm(
                a_bf16_flat,
                w13_i32_flat,
                activated_bf16_flat,
                w13_scales_i32_flat,
                w13_global_scale,
                packed_route_indices,
                block_expert_ids,
                packed_route_count,
                topk_weights_flat,
                fc1_c_tmp_f32_flat,
                locks_i32_flat,
                smem_base,
                tid,
                cta,
                grid_x,
                active_m,
                fc1_emit_tile,
            )
        self._grid_barrier(locks_i32_flat, tid, grid_x)
        if cutlass.const_expr(self.collect_activation_amax):
            self._collect_activation_amax_epilogue(
                a_bf16_flat,
                activated_bf16_flat,
                packed_route_indices,
                block_expert_ids,
                packed_route_count,
                activation_amax_flat,
                layer_idx,
                smem_base,
                tid,
                cta,
                grid_x,
                active_m,
            )
        if cutlass.const_expr(self.zero_fc2_output):
            self._zero_fc2_output(fc2_bf16_flat, tid, cta, grid_x, active_m)
            self._grid_barrier(locks_i32_flat, tid, grid_x)
        self.fc2._run_persistent_gemm(
            activated_bf16_flat,
            w2_i32_flat,
            fc2_bf16_flat,
            w2_scales_i32_flat,
            w2_global_scale,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            topk_weights_flat,
            fc2_c_tmp_f32_flat,
            locks_i32_flat,
            smem_base,
            tid,
            cta,
            grid_x,
            active_m * Int32(self.top_k),
            fc2_emit_tile,
        )

    @cute.jit
    def _grid_barrier(
        self,
        locks_i32_flat: cute.Tensor,
        tid: Int32,
        grid_x: Int32,
    ):
        cute.arch.sync_threads()
        if tid == Int32(0):
            count_addr = get_ptr_as_int64(locks_i32_flat, Int32(self.barrier_count_off))
            sense_addr = get_ptr_as_int64(locks_i32_flat, Int32(self.barrier_sense_off))
            old_sense = ld_global_acquire_i32(sense_addr)
            old_count = atomic_add_global_i32(count_addr, Int32(1))
            if old_count == grid_x - Int32(1):
                st_global_i32(count_addr, Int32(0))
                threadfence()
                red_add_global_release_i32(sense_addr, Int32(1))
            else:
                sense = old_sense
                while sense == old_sense:
                    sense = ld_global_acquire_i32(sense_addr)
        cute.arch.sync_threads()

    @cute.jit
    def _zero_fc2_output(
        self,
        fc2_bf16_flat: cute.Tensor,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: cutlass.Int32,
    ):
        idx = cta * Int32(self.cta_threads) + tid
        stride = grid_x * Int32(self.cta_threads)
        total = active_m * Int32(self.top_k * self.hidden_size)
        zero = self._cast_elem(cutlass.Float32(0.0))
        while idx < total:
            fc2_bf16_flat[idx] = zero
            idx += stride

    @cute.jit
    def _reduce_and_red_activation_amax(
        self,
        local_max: cutlass.Float32,
        activation_amax_flat: cute.Tensor,
        layer_idx: cutlass.Int32,
        expert_idx: Int32,
        slot: Int32,
        smem_base: Int32,
        tid: Int32,
    ):
        lane = tid & Int32(31)
        warp_idx = tid // Int32(32)
        warp_amax = warp_reduce(local_max, fmax_f32)
        if lane == Int32(0):
            st_shared_f32(smem_base + warp_idx * Int32(4), warp_amax)
        cute.arch.sync_threads()

        if warp_idx == Int32(0):
            block_amax = cutlass.Float32(0.0)
            if lane < Int32(self.cta_threads // 32):
                block_amax = ld_shared_f32(smem_base + lane * Int32(4))
            block_amax = warp_reduce(block_amax, fmax_f32)
            if lane == Int32(0) and block_amax > cutlass.Float32(0.0):
                out_idx = (layer_idx * Int32(self.num_experts) + expert_idx) * Int32(
                    2
                ) + slot
                red_max_global_f32_nonnegative(
                    get_ptr_as_int64(activation_amax_flat, out_idx),
                    block_amax,
                )
        cute.arch.sync_threads()

    @cute.jit
    def _collect_activation_amax_epilogue(
        self,
        a_bf16_flat: cute.Tensor,
        activated_bf16_flat: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        activation_amax_flat: cute.Tensor,
        layer_idx: cutlass.Int32,
        smem_base: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: cutlass.Int32,
    ):
        live_routes = active_m * Int32(self.top_k)
        route_count = packed_route_count[Int32(0)].to(Int32)
        route_blocks = (route_count + Int32(self.moe_block_size) - Int32(1)) // Int32(
            self.moe_block_size
        )
        expert_idx = cta
        while expert_idx < Int32(self.num_experts):
            local_fc1 = cutlass.Float32(0.0)
            local_fc2 = cutlass.Float32(0.0)
            has_route_block = Int32(0)
            block_idx = Int32(0)
            while block_idx < route_blocks:
                block_expert = block_expert_ids[block_idx].to(Int32)
                if block_expert == expert_idx:
                    has_route_block = Int32(1)
                    route_pos = block_idx * Int32(self.moe_block_size)
                    route_stop = route_pos + Int32(self.moe_block_size)
                    if route_stop > route_count:
                        route_stop = route_count
                    while route_pos < route_stop:
                        route_idx = packed_route_indices[route_pos].to(Int32)
                        if route_idx >= Int32(0) and route_idx < live_routes:
                            token_idx = route_idx // Int32(self.top_k)
                            fc1_col = tid
                            while fc1_col < Int32(self.hidden_size):
                                v1 = a_bf16_flat[
                                    token_idx * Int32(self.hidden_size) + fc1_col
                                ].to(cutlass.Float32)
                                local_fc1 = fmax_f32(local_fc1, fabs_f32(v1))
                                fc1_col += Int32(self.cta_threads)

                            fc2_col = tid
                            while fc2_col < Int32(self.intermediate_size):
                                v2 = activated_bf16_flat[
                                    route_idx * Int32(self.intermediate_size) + fc2_col
                                ].to(cutlass.Float32)
                                local_fc2 = fmax_f32(local_fc2, fabs_f32(v2))
                                fc2_col += Int32(self.cta_threads)
                        route_pos += Int32(1)
                block_idx += Int32(1)

            if has_route_block != Int32(0):
                self._reduce_and_red_activation_amax(
                    local_fc1,
                    activation_amax_flat,
                    layer_idx,
                    expert_idx,
                    Int32(0),
                    smem_base,
                    tid,
                )
                self._reduce_and_red_activation_amax(
                    local_fc2,
                    activation_amax_flat,
                    layer_idx,
                    expert_idx,
                    Int32(1),
                    smem_base,
                    tid,
                )
            expert_idx += grid_x

    @cute.jit
    def _run_activation(
        self,
        fc1_bf16_flat: cute.Tensor,
        activated_bf16_flat: cute.Tensor,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: cutlass.Int32,
    ):
        idx = cta * Int32(self.cta_threads) + tid
        stride = grid_x * Int32(self.cta_threads)
        total = active_m * Int32(self.top_k * self.intermediate_size)
        while idx < total:
            if cutlass.const_expr(self.activation_is_gated):
                row = idx // Int32(self.intermediate_size)
                col = idx - row * Int32(self.intermediate_size)
                base = row * Int32(self.fc1_cols)
                gate = fc1_bf16_flat[base + col].to(cutlass.Float32)
                up = fc1_bf16_flat[base + Int32(self.intermediate_size) + col].to(
                    cutlass.Float32
                )
                gate, up = self._clamp_swiglu_inputs(gate, up)
                sigmoid_arg = gate
                up_term = up
                if cutlass.const_expr(self.activation_is_swigluoai):
                    sigmoid_arg = cutlass.Float32(self.swiglu_alpha) * gate
                    up_term = up + cutlass.Float32(self.swiglu_beta)
                if cutlass.const_expr(self.fast_math):
                    exp_neg_gate = cute.math.exp(-sigmoid_arg, fastmath=True)
                else:
                    exp_neg_gate = cute.math.exp(-sigmoid_arg, fastmath=False)
                silu = gate / (cutlass.Float32(1.0) + exp_neg_gate)
                if cutlass.const_expr(self.activation_is_swigluoai):
                    activated_bf16_flat[idx] = self._cast_elem(silu * up_term)
                else:
                    activated_bf16_flat[idx] = self._cast_elem(
                        self._cast_elem(silu) * self._cast_elem(up_term)
                    )
            else:
                x = fc1_bf16_flat[idx].to(cutlass.Float32)
                if x < cutlass.Float32(0.0):
                    x = cutlass.Float32(0.0)
                activated_bf16_flat[idx] = self._cast_elem(x * x)
            idx += stride


class W4A16FusedMoeHybridKernel:
    """Two-tier heterogeneous-quantization fused MoE entry.

    Composition over W4A16FusedMoeKernel: each tier's fused kernel supplies
    its FC1/FC2 children and the shared phase machinery (_moe_body, grid
    barrier, activation). This entry only widens the launch ABI to carry both
    tiers' weights plus a global-expert descriptor map, and installs the
    route-map emit hooks that resolve tier + local expert per mn-tile. Tier 0
    drives scheduling; both tiers are validated to identical geometry, so the
    schedule constants agree by construction.

    Route contract: the routes tensor carries GLOBAL expert ids (one per
    size_m*top_k slot, -1 for inactive). tier_local_map holds one int32
    descriptor per global expert id: (tier << 8) | local_expert_id, negative
    for unmapped. Unmapped or out-of-range ids skip the tile, matching the
    -1-route behavior of the single-tier direct path.
    """

    ABI_VERSION = 1

    def __init__(
        self,
        *,
        tier0: W4A16FusedMoeKernel,
        tier1: W4A16FusedMoeKernel,
        map_slots: int,
    ):
        for name, moe in (("tier0", tier0), ("tier1", tier1)):
            if not moe.direct_topk_routes:
                raise ValueError(f"hybrid W4A16 {name} requires direct_topk_routes")
            if not moe.tc_decode_fused_sum:
                raise ValueError(f"hybrid W4A16 {name} requires tc_decode_fused_sum")
            if not moe.activation_is_gated:
                raise ValueError(f"hybrid W4A16 {name} requires a gated activation")
            if moe.collect_activation_amax:
                raise ValueError(
                    f"hybrid W4A16 {name} is incompatible with activation amax"
                )
            if moe.zero_fc2_output:
                raise ValueError(f"hybrid W4A16 {name} forbids zero_fc2_output")
            if not moe.tc_zero_output:
                raise ValueError(f"hybrid W4A16 {name} requires tc_zero_output")
        for attr in (
            "size_m",
            "hidden_size",
            "intermediate_size",
            "fc1_cols",
            "top_k",
            "moe_block_size",
            "activation",
            "activation_is_swigluoai",
            "has_swiglu_limit",
            "swiglu_limit",
            "swiglu_alpha",
            "swiglu_beta",
            "element_dtype",
            "is_fp16",
            "fast_math",
            "apply_router_weight_on_input",
            "cta_threads",
            "sms",
            "blocks_per_sm",
            "barrier_count_off",
            "barrier_sense_off",
            "schedule_whole_tiles",
        ):
            if getattr(tier0, attr) != getattr(tier1, attr):
                raise ValueError(
                    f"hybrid W4A16 tiers disagree on {attr}: "
                    f"{getattr(tier0, attr)!r} != {getattr(tier1, attr)!r}"
                )
        for phase in ("fc1", "fc2"):
            gemm0 = getattr(tier0, phase)
            gemm1 = getattr(tier1, phase)
            if (gemm0.n_tiles, gemm0.k_tiles, gemm0.tile_n, gemm0.tile_k) != (
                gemm1.n_tiles,
                gemm1.k_tiles,
                gemm1.tile_n,
                gemm1.tile_k,
            ):
                raise ValueError(f"hybrid W4A16 tiers disagree on {phase} tiling")
            if (
                gemm0.top_k,
                gemm0.mul_topk_weights,
                gemm0.fused_topk_sum,
                gemm0.fused_sum_topk,
                gemm0.moe_block_size,
                gemm0.cta_threads,
                gemm0.schedule_whole_tiles,
            ) != (
                gemm1.top_k,
                gemm1.mul_topk_weights,
                gemm1.fused_topk_sum,
                gemm1.fused_sum_topk,
                gemm1.moe_block_size,
                gemm1.cta_threads,
                gemm1.schedule_whole_tiles,
            ):
                raise ValueError(f"hybrid W4A16 tiers disagree on {phase} semantics")
        if int(map_slots) < tier0.num_experts + tier1.num_experts:
            raise ValueError(
                "hybrid W4A16 map_slots must cover both tiers' expert counts"
            )
        if tier0.num_experts > 256 or tier1.num_experts > 256:
            raise ValueError(
                "hybrid W4A16 local expert ids must fit the 8-bit descriptor field"
            )
        self.tier0 = tier0
        self.tier1 = tier1
        self.map_slots = int(map_slots)
        self.size_m = tier0.size_m
        self.hidden_size = tier0.hidden_size
        self.intermediate_size = tier0.intermediate_size
        self.top_k = tier0.top_k
        self.element_dtype = tier0.element_dtype
        self.cta_threads = tier0.cta_threads
        self.sms = tier0.sms
        self.blocks_per_sm = tier0.blocks_per_sm
        self.shared_words = max(tier0.shared_words, tier1.shared_words)

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (
            "w4a16_fused_moe_hybrid",
            self.ABI_VERSION,
            self.map_slots,
            self.tier0.__cache_key__,
            self.tier1.__cache_key__,
            self.shared_words,
        )

    @cute.jit
    def _emit_route_map_tile(
        self,
        is_fc1: cutlass.Constexpr,
        a_bf16_flat: cute.Tensor,
        t0_b_i32_flat: cute.Tensor,
        t0_scales_i32_flat: cute.Tensor,
        t0_global_scale: cute.Tensor,
        t1_b_i32_flat: cute.Tensor,
        t1_scales_i32_flat: cute.Tensor,
        t1_global_scale: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        global_topk_ids_i32_flat: cute.Tensor,
        tier_local_map_i32_flat: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        active_size_m: Int32,
        route_block_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
    ):
        gid = global_topk_ids_i32_flat[route_block_idx].to(Int32)
        if gid >= Int32(0) and gid < Int32(self.map_slots):
            descriptor = tier_local_map_i32_flat[gid].to(Int32)
            if descriptor >= Int32(0):
                tier = descriptor >> Int32(8)
                local_expert = descriptor & Int32(0xFF)
                if tier == Int32(0):
                    if local_expert < Int32(self.tier0.num_experts):
                        if cutlass.const_expr(is_fc1):
                            self.tier0.fc1._run_tile(
                                a_bf16_flat,
                                t0_b_i32_flat,
                                c_bf16_flat,
                                t0_scales_i32_flat,
                                t0_global_scale,
                                global_topk_ids_i32_flat,
                                topk_weights_flat,
                                c_tmp_f32_flat,
                                locks_i32_flat,
                                smem_base,
                                tid,
                                route_block_idx,
                                local_expert,
                                output_n_tile,
                                reduce_k_tile,
                                reduce_tile_count,
                                reduce_slice_count,
                                reduce_slice_idx,
                                lock_slot,
                                active_size_m,
                            )
                        else:
                            self.tier0.fc2._run_tile(
                                a_bf16_flat,
                                t0_b_i32_flat,
                                c_bf16_flat,
                                t0_scales_i32_flat,
                                t0_global_scale,
                                global_topk_ids_i32_flat,
                                topk_weights_flat,
                                c_tmp_f32_flat,
                                locks_i32_flat,
                                smem_base,
                                tid,
                                route_block_idx,
                                local_expert,
                                output_n_tile,
                                reduce_k_tile,
                                reduce_tile_count,
                                reduce_slice_count,
                                reduce_slice_idx,
                                lock_slot,
                                active_size_m,
                            )
                else:
                    if tier == Int32(1):
                        if local_expert < Int32(self.tier1.num_experts):
                            if cutlass.const_expr(is_fc1):
                                self.tier1.fc1._run_tile(
                                    a_bf16_flat,
                                    t1_b_i32_flat,
                                    c_bf16_flat,
                                    t1_scales_i32_flat,
                                    t1_global_scale,
                                    global_topk_ids_i32_flat,
                                    topk_weights_flat,
                                    c_tmp_f32_flat,
                                    locks_i32_flat,
                                    smem_base,
                                    tid,
                                    route_block_idx,
                                    local_expert,
                                    output_n_tile,
                                    reduce_k_tile,
                                    reduce_tile_count,
                                    reduce_slice_count,
                                    reduce_slice_idx,
                                    lock_slot,
                                    active_size_m,
                                )
                            else:
                                self.tier1.fc2._run_tile(
                                    a_bf16_flat,
                                    t1_b_i32_flat,
                                    c_bf16_flat,
                                    t1_scales_i32_flat,
                                    t1_global_scale,
                                    global_topk_ids_i32_flat,
                                    topk_weights_flat,
                                    c_tmp_f32_flat,
                                    locks_i32_flat,
                                    smem_base,
                                    tid,
                                    route_block_idx,
                                    local_expert,
                                    output_n_tile,
                                    reduce_k_tile,
                                    reduce_tile_count,
                                    reduce_slice_count,
                                    reduce_slice_idx,
                                    lock_slot,
                                    active_size_m,
                                )

    @cute.jit
    def __call__(
        self,
        a_bf16_ptr: cute.Pointer,
        t0_w13_i32_flat: cute.Tensor,
        t0_w2_i32_flat: cute.Tensor,
        t0_w13_scales_i32_flat: cute.Tensor,
        t0_w2_scales_i32_flat: cute.Tensor,
        t0_w13_global_scale: cute.Tensor,
        t0_w2_global_scale: cute.Tensor,
        t1_w13_i32_flat: cute.Tensor,
        t1_w2_i32_flat: cute.Tensor,
        t1_w13_scales_i32_flat: cute.Tensor,
        t1_w2_scales_i32_flat: cute.Tensor,
        t1_w13_global_scale: cute.Tensor,
        t1_w2_global_scale: cute.Tensor,
        global_topk_ids_i32_flat: cute.Tensor,
        tier_local_map_i32_flat: cute.Tensor,
        fc1_bf16_flat: cute.Tensor,
        activated_bf16_flat: cute.Tensor,
        output_bf16_flat: cute.Tensor,
        topk_weights_ptr: cute.Pointer,
        fc1_c_tmp_f32_flat: cute.Tensor,
        fc2_c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        active_m: cutlass.Int32,
        grid_x: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        a_bf16_flat = cute.make_tensor(
            a_bf16_ptr,
            layout=cute.make_layout((active_m * Int32(self.hidden_size),), stride=(1,)),
        )
        topk_weights_flat = cute.make_tensor(
            topk_weights_ptr,
            layout=cute.make_layout((active_m * Int32(self.top_k),), stride=(1,)),
        )
        self.kernel(
            a_bf16_flat,
            t0_w13_i32_flat,
            t0_w2_i32_flat,
            t0_w13_scales_i32_flat,
            t0_w2_scales_i32_flat,
            t0_w13_global_scale,
            t0_w2_global_scale,
            t1_w13_i32_flat,
            t1_w2_i32_flat,
            t1_w13_scales_i32_flat,
            t1_w2_scales_i32_flat,
            t1_w13_global_scale,
            t1_w2_global_scale,
            global_topk_ids_i32_flat,
            tier_local_map_i32_flat,
            fc1_bf16_flat,
            activated_bf16_flat,
            output_bf16_flat,
            topk_weights_flat,
            fc1_c_tmp_f32_flat,
            fc2_c_tmp_f32_flat,
            locks_i32_flat,
            active_m,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self.cta_threads, 1, 1],
            min_blocks_per_mp=self.blocks_per_sm,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        a_bf16_flat: cute.Tensor,
        t0_w13_i32_flat: cute.Tensor,
        t0_w2_i32_flat: cute.Tensor,
        t0_w13_scales_i32_flat: cute.Tensor,
        t0_w2_scales_i32_flat: cute.Tensor,
        t0_w13_global_scale: cute.Tensor,
        t0_w2_global_scale: cute.Tensor,
        t1_w13_i32_flat: cute.Tensor,
        t1_w2_i32_flat: cute.Tensor,
        t1_w13_scales_i32_flat: cute.Tensor,
        t1_w2_scales_i32_flat: cute.Tensor,
        t1_w13_global_scale: cute.Tensor,
        t1_w2_global_scale: cute.Tensor,
        global_topk_ids_i32_flat: cute.Tensor,
        tier_local_map_i32_flat: cute.Tensor,
        fc1_bf16_flat: cute.Tensor,
        activated_bf16_flat: cute.Tensor,
        output_bf16_flat: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        fc1_c_tmp_f32_flat: cute.Tensor,
        fc2_c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        active_m: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_x_raw, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        cta = Int32(bidx)
        grid_x = Int32(grid_x_raw)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words],
                1024,
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())

        fc1_emit_tile = partial(
            self._emit_route_map_tile,
            True,
            a_bf16_flat,
            t0_w13_i32_flat,
            t0_w13_scales_i32_flat,
            t0_w13_global_scale,
            t1_w13_i32_flat,
            t1_w13_scales_i32_flat,
            t1_w13_global_scale,
            fc1_bf16_flat,
            global_topk_ids_i32_flat,
            tier_local_map_i32_flat,
            topk_weights_flat,
            fc1_c_tmp_f32_flat,
            locks_i32_flat,
            smem_base,
            tid,
            active_m,
        )
        fc2_emit_tile = partial(
            self._emit_route_map_tile,
            False,
            activated_bf16_flat,
            t0_w2_i32_flat,
            t0_w2_scales_i32_flat,
            t0_w2_global_scale,
            t1_w2_i32_flat,
            t1_w2_scales_i32_flat,
            t1_w2_global_scale,
            output_bf16_flat,
            global_topk_ids_i32_flat,
            tier_local_map_i32_flat,
            topk_weights_flat,
            fc2_c_tmp_f32_flat,
            locks_i32_flat,
            smem_base,
            tid,
            active_m * Int32(self.top_k),
        )
        # Tier 0 drives the shared phase assembly; the unused single-tier
        # route/amax parameters receive placeholder tensors that the direct
        # route + no-amax const_expr configuration never reads.
        self.tier0._moe_body(
            a_bf16_flat,
            t0_w13_i32_flat,
            t0_w2_i32_flat,
            fc1_bf16_flat,
            activated_bf16_flat,
            output_bf16_flat,
            t0_w13_scales_i32_flat,
            t0_w2_scales_i32_flat,
            t0_w13_global_scale,
            t0_w2_global_scale,
            global_topk_ids_i32_flat,
            global_topk_ids_i32_flat,
            global_topk_ids_i32_flat,
            global_topk_ids_i32_flat,
            Int32(0),
            topk_weights_flat,
            fc1_c_tmp_f32_flat,
            fc2_c_tmp_f32_flat,
            locks_i32_flat,
            smem_base,
            tid,
            cta,
            grid_x,
            active_m,
            fc1_emit_tile,
            fc2_emit_tile,
        )


class W4A16ActivationKernel:
    def __init__(
        self,
        *,
        rows: int,
        intermediate_size: int,
        activation: str,
        element_dtype: str = "bf16",
        fast_math: bool = True,
        swiglu_limit: float | None = None,
        swiglu_alpha: float | None = None,
        swiglu_beta: float | None = None,
    ):
        activation = normalize_moe_activation(activation)
        is_gated = validate_activation(activation)
        swiglu_limit, swiglu_alpha, swiglu_beta = _normalize_activation_swiglu_params(
            activation,
            swiglu_limit,
            swiglu_alpha,
            swiglu_beta,
        )
        if element_dtype not in {"bf16", "fp16"}:
            raise ValueError(f"unsupported element_dtype {element_dtype!r}")
        if rows <= 0 or intermediate_size <= 0:
            raise ValueError("rows and intermediate_size must be positive")
        self.rows = int(rows)
        self.intermediate_size = int(intermediate_size)
        self.activation = activation
        self.is_gated = is_gated
        self.is_swigluoai = activation == SWIGLUOAI_UNINTERLEAVE
        self.has_swiglu_limit = swiglu_limit is not None
        self.swiglu_limit = 0.0 if swiglu_limit is None else swiglu_limit
        self.swiglu_alpha = float(swiglu_alpha)
        self.swiglu_beta = float(swiglu_beta)
        self.element_dtype = element_dtype
        self.is_fp16 = element_dtype == "fp16"
        self.fast_math = bool(fast_math)
        self.cta_threads = 256

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (
            self.intermediate_size,
            self.activation,
            self.is_gated,
            self.is_swigluoai,
            self.has_swiglu_limit,
            self.swiglu_limit,
            self.swiglu_alpha,
            self.swiglu_beta,
            self.element_dtype,
            self.fast_math,
            self.cta_threads,
        )

    @cute.jit
    def _cast_elem(self, x: cutlass.Float32):
        if cutlass.const_expr(self.is_fp16):
            return cutlass.Float16(x)
        return cutlass.BFloat16(x)

    @cute.jit
    def _clamp_swiglu_inputs(
        self,
        gate: cutlass.Float32,
        up: cutlass.Float32,
    ):
        if cutlass.const_expr(self.has_swiglu_limit):
            limit = cutlass.Float32(self.swiglu_limit)
            neg_limit = cutlass.Float32(-self.swiglu_limit)
            if gate > limit:
                gate = limit
            if up > limit:
                up = limit
            if up < neg_limit:
                up = neg_limit
        return gate, up

    @cute.jit
    def __call__(
        self,
        fc1_flat: cute.Tensor,
        activated_flat: cute.Tensor,
        active_rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        total = active_rows * Int32(self.intermediate_size)
        grid = (_covering_count(total, self.cta_threads), 1, 1)
        self.kernel(fc1_flat, activated_flat, active_rows).launch(
            grid=grid,
            block=[self.cta_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        fc1_flat: cute.Tensor,
        activated_flat: cute.Tensor,
        active_rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        idx = Int32(bidx) * Int32(self.cta_threads) + Int32(tidx)
        total = Int32(active_rows) * Int32(self.intermediate_size)
        if idx < total:
            if cutlass.const_expr(self.is_gated):
                row = idx // Int32(self.intermediate_size)
                col = idx - row * Int32(self.intermediate_size)
                base = row * Int32(2 * self.intermediate_size)
                gate = fc1_flat[base + col].to(cutlass.Float32)
                up = fc1_flat[base + Int32(self.intermediate_size) + col].to(
                    cutlass.Float32
                )
                gate, up = self._clamp_swiglu_inputs(gate, up)
                sigmoid_arg = gate
                up_term = up
                if cutlass.const_expr(self.is_swigluoai):
                    sigmoid_arg = cutlass.Float32(self.swiglu_alpha) * gate
                    up_term = up + cutlass.Float32(self.swiglu_beta)
                if cutlass.const_expr(self.fast_math):
                    exp_neg_gate = cute.math.exp(-sigmoid_arg, fastmath=True)
                else:
                    exp_neg_gate = cute.math.exp(-sigmoid_arg, fastmath=False)
                silu = gate / (cutlass.Float32(1.0) + exp_neg_gate)
                if cutlass.const_expr(self.is_swigluoai):
                    activated_flat[idx] = self._cast_elem(silu * up_term)
                else:
                    activated_flat[idx] = self._cast_elem(
                        self._cast_elem(silu) * self._cast_elem(up_term)
                    )
            else:
                x = fc1_flat[idx].to(cutlass.Float32)
                if x < cutlass.Float32(0.0):
                    x = cutlass.Float32(0.0)
                activated_flat[idx] = self._cast_elem(x * x)


class W4A16TopKSumKernel:
    def __init__(self, *, topk: int, hidden_size: int, element_dtype: str = "bf16"):
        if element_dtype not in {"bf16", "fp16"}:
            raise ValueError(f"unsupported element_dtype {element_dtype!r}")
        if topk <= 0 or hidden_size <= 0:
            raise ValueError("topk and hidden_size must be positive")
        self.topk = int(topk)
        self.hidden_size = int(hidden_size)
        self.element_dtype = element_dtype
        self.is_fp16 = element_dtype == "fp16"
        self.cta_threads = 256

    @cute.jit
    def _cast_elem(self, x: cutlass.Float32):
        if cutlass.const_expr(self.is_fp16):
            return cutlass.Float16(x)
        return cutlass.BFloat16(x)

    @cute.jit
    def __call__(
        self,
        fc2_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        active_m: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        fc2_flat = cute.make_tensor(
            fc2_ptr,
            layout=cute.make_layout(
                (active_m * Int32(self.topk * self.hidden_size),), stride=(1,)
            ),
        )
        output_flat = cute.make_tensor(
            output_ptr,
            layout=cute.make_layout((active_m * Int32(self.hidden_size),), stride=(1,)),
        )
        total = active_m * Int32(self.hidden_size)
        grid = (_covering_count(total, self.cta_threads), 1, 1)
        self.kernel(fc2_flat, output_flat, active_m).launch(
            grid=grid,
            block=[self.cta_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        fc2_flat: cute.Tensor,
        output_flat: cute.Tensor,
        active_m: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        idx = Int32(bidx) * Int32(self.cta_threads) + Int32(tidx)
        total = active_m * Int32(self.hidden_size)
        if idx < total:
            token = idx // Int32(self.hidden_size)
            col = idx - token * Int32(self.hidden_size)
            acc = cutlass.Float32(0.0)
            for route in cutlass.range_constexpr(self.topk):
                row = token * Int32(self.topk) + Int32(route)
                route_value = fc2_flat[row * Int32(self.hidden_size) + col].to(
                    cutlass.Float32
                )
                acc += _materialize_w4a16_topk_route_f32(route_value)
            output_flat[idx] = self._cast_elem(acc)


_CACHE: dict[tuple, W4A16GemmCompileResult] = {}
_FUSED_CACHE: dict[tuple, W4A16FusedMoeCompileResult] = {}
_ACTIVATION_CACHE: dict[tuple, W4A16ActivationCompileResult] = {}
_SUM_CACHE: dict[tuple, W4A16TopKSumCompileResult] = {}
_SMALL_M_DIRECT_CACHE: dict[tuple, _W4A16SmallMDirectLaunch] = {}


def _normalize_element_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    raise TypeError(f"unsupported W4A16 activation dtype {dtype}")


def _normalize_scale_format(scale_format: str) -> str:
    try:
        return _SCALE_FORMATS[scale_format.lower()]
    except KeyError as exc:
        raise ValueError(
            "scale_format must be one of 'e4m3_k16', 'e8m0_k32', or 'e4m3_k32', "
            f"got {scale_format!r}"
        ) from exc


def _scale_group_size(scale_format: str) -> int:
    return (
        32 if _normalize_scale_format(scale_format) in ("e8m0_k32", "e4m3_k32") else 16
    )


def _scale_fake_int32_elements(
    *,
    num_experts: int,
    size_k: int,
    size_n: int,
    scale_format: str,
    allow_k_tail: bool = False,
    allow_n_tail: bool = False,
) -> int:
    group_size = _scale_group_size(scale_format)
    if int(size_n) % 16 != 0:
        raise ValueError(f"W4A16 {scale_format} scales require size_n divisible by 16")
    if int(size_k) % group_size != 0 and not allow_k_tail:
        raise ValueError(
            f"W4A16 {scale_format} scales require size_k divisible by {group_size}, "
            f"got {size_k}"
        )
    groups_k = (
        _covering_count(int(size_k), group_size)
        if allow_k_tail
        else int(size_k) // group_size
    )
    scale_size_n = (
        _e8m0_logical_tail_scale_n(int(size_n)) if allow_n_tail else int(size_n)
    )
    return int(num_experts) * groups_k * (scale_size_n // 4)


def _cutlass_element_dtype(element_dtype: str):
    if element_dtype == "bf16":
        return cutlass.BFloat16
    if element_dtype == "fp16":
        return cutlass.Float16
    raise ValueError(f"unsupported element_dtype {element_dtype!r}")


def _small_m_direct_supported(
    *,
    m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    activation: str,
    apply_router_weight_on_input: bool,
    swiglu_limit: float | None,
    swiglu_alpha: float,
    swiglu_beta: float,
    element_dtype: str,
    weight_layout: str,
    w13_layout: str,
    scale_format: str = "e4m3_k16",
    expert_map: torch.Tensor | None = None,
) -> bool:
    if os.environ.get("FLASHINFER_EXP_SM12X_W4A16_SMALL_M_DIRECT", "1") == "0":
        return False
    activation = normalize_moe_activation(activation)
    swiglu_limit, swiglu_alpha, swiglu_beta = _normalize_activation_swiglu_params(
        activation,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
    )
    if activation == SWIGLUOAI_UNINTERLEAVE and swiglu_limit is None:
        return False
    return (
        element_dtype == "bf16"
        # Packed serving weights use the W4A16 TC-decode path for small M.
        # The direct micro kernel is kept limited to native ModelOpt weights.
        and weight_layout == "modelopt"
        and w13_layout in ("w13", "w31")
        and not bool(apply_router_weight_on_input)
        and (swiglu_limit is None or activation in ("silu", SWIGLUOAI_UNINTERLEAVE))
        and expert_map is None
        and MoEMicroKernelW4A16SmallMDirect.is_supported(
            m=m,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            topk=topk,
            num_experts=num_experts,
            scale_format=scale_format,
        )
    )


def _compile_w4a16_small_m_direct(
    *,
    m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    activation: str,
    fast_math: bool,
    topk_ids_dtype: torch.dtype,
    device: torch.device | None,
    scale_format: str = "e4m3_k16",
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    w13_layout: str = "w13",
) -> _W4A16SmallMDirectLaunch:
    if topk_ids_dtype not in (torch.int32, torch.int64):
        raise TypeError("small-M W4A16 direct path requires int32/int64 topk_ids")
    activation = normalize_moe_activation(activation)
    swiglu_limit, swiglu_alpha, swiglu_beta = _normalize_activation_swiglu_params(
        activation,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
    )
    cache_key = (
        "w4a16_small_m_direct",
        None if device is None else int(device.index or 0),
        int(m),
        int(hidden_size),
        int(intermediate_size),
        int(num_experts),
        int(topk),
        activation,
        bool(fast_math),
        topk_ids_dtype,
        scale_format,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
        w13_layout,
    )
    cached = _SMALL_M_DIRECT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    kernel = MoEMicroKernelW4A16SmallMDirect(
        activation=activation,
        fast_math=bool(fast_math),
        share_input_across_experts=(int(m) == 1),
        share_expert_scales=True,
        single_token=(int(m) == 1),
        scale_format=scale_format,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        w13_layout=w13_layout,
    )
    kernel.configure(
        int(m),
        int(hidden_size),
        int(intermediate_size),
        int(topk),
        int(num_experts),
        device=device,
    )

    def dummy(dt):
        return make_ptr(dt, 16, cute.AddressSpace.gmem, assumed_align=16)

    ids_dtype = cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    barrier_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = sm12x_compile(
        kernel,
        dummy(cutlass.BFloat16),
        dummy(cutlass.Uint8),
        dummy(cutlass.Uint8),
        dummy(cutlass.Float32),
        dummy(cutlass.Float32),
        dummy(cutlass.Float32),
        dummy(cutlass.Uint32),
        dummy(cutlass.Uint8),
        dummy(cutlass.Uint8),
        dummy(cutlass.Float32),
        dummy(ids_dtype),
        dummy(cutlass.Float32),
        dummy(cutlass.BFloat16),
        barrier_fake,
        barrier_fake,
        Int32(m),
        Int32(kernel.grid_x),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_facts(
            "moe.w4a16.small_m_direct",
            1,
            ("device_index", None if device is None else int(device.index or 0)),
            ("m", int(m)),
            ("hidden_size", int(hidden_size)),
            ("intermediate_size", int(intermediate_size)),
            ("num_experts", int(num_experts)),
            ("topk", int(topk)),
            ("activation", activation),
            ("fast_math", bool(fast_math)),
            ("topk_ids_dtype", str(topk_ids_dtype)),
            ("scale_format", scale_format),
            ("swiglu_limit", swiglu_limit),
            ("swiglu_alpha", swiglu_alpha),
            ("swiglu_beta", swiglu_beta),
            ("w13_layout", w13_layout),
            ("grid_x", int(kernel.grid_x)),
        ),
    )
    launch = _W4A16SmallMDirectLaunch(
        compiled=compiled,
        grid_x=int(kernel.grid_x),
        m=int(m),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_experts=int(num_experts),
        topk=int(topk),
        activation=activation,
        fast_math=bool(fast_math),
        topk_ids_dtype=topk_ids_dtype,
    )
    _SMALL_M_DIRECT_CACHE[cache_key] = launch
    return launch


def compile_w4a16_gemm(
    *,
    size_m: int,
    size_n: int,
    size_k: int,
    num_experts: int,
    top_k: int,
    mul_topk_weights: bool,
    tile_n: int,
    tile_k: int,
    moe_block_size: int,
    max_m_blocks: int,
    element_dtype: str = "bf16",
    scale_format: str = "e4m3_k16",
) -> W4A16GemmCompileResult:
    scale_format = _normalize_scale_format(scale_format)
    cutlass_dtype = _cutlass_element_dtype(element_dtype)
    if torch.cuda.is_available():
        device = int(torch.cuda.current_device())
    else:
        device = None
    kernel = W4A16GemmKernel(
        size_m=size_m,
        size_n=size_n,
        size_k=size_k,
        num_experts=num_experts,
        top_k=top_k,
        mul_topk_weights=mul_topk_weights,
        tile_n=tile_n,
        tile_k=tile_k,
        moe_block_size=moe_block_size,
        max_m_blocks=max_m_blocks,
        element_dtype=element_dtype,
        scale_format=scale_format,
    )
    cache_key = (
        "w4a16_gemm",
        device,
        kernel.__cache_key__,
    )
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return replace(
            cached,
            max_m_blocks=max_m_blocks,
            blocks_per_sm=kernel.blocks_per_sm,
        )

    compile_size_m = _fake_m_for_specialization(size_m)
    compile_route_blocks = 1
    compile_route_slots = compile_route_blocks * int(moe_block_size)
    a_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (compile_size_m * size_k,),
        assumed_align=16,
    )
    b_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (num_experts * (size_k // 16) * (size_n // 16 * 32),),
        assumed_align=16,
    )
    c_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (compile_size_m * top_k * size_n,),
        assumed_align=16,
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (
            _scale_fake_int32_elements(
                num_experts=num_experts,
                size_k=size_k,
                size_n=size_n,
                scale_format=scale_format,
            ),
        ),
        assumed_align=16,
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (num_experts,),
        assumed_align=16,
    )
    packed_routes_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (compile_route_slots,),
        assumed_align=16,
    )
    block_experts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (compile_route_blocks,),
        assumed_align=16,
    )
    route_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    topk_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (compile_size_m * top_k,),
        assumed_align=4,
    )
    c_tmp_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (
            max(
                size_n * compile_route_slots,
                4 * 256 * moe_block_size * 256,
            ),
        ),
        assumed_align=16,
    )
    locks_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (4 * 256,),
        assumed_align=16,
    )

    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = sm12x_compile(
        kernel,
        a_fake,
        b_fake,
        c_fake,
        scales_fake,
        global_scale_fake,
        packed_routes_fake,
        block_experts_fake,
        route_count_fake,
        topk_fake,
        c_tmp_fake,
        locks_fake,
        Int32(compile_size_m),
        Int32(1),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "moe.w4a16.gemm",
            1,
            cache_key,
        ),
    )
    result = W4A16GemmCompileResult(
        compiled=compiled,
        tile_n=tile_n,
        tile_k=tile_k,
        moe_block_size=moe_block_size,
        max_m_blocks=max_m_blocks,
        blocks_per_sm=kernel.blocks_per_sm,
        scale_format=scale_format,
    )
    _CACHE[cache_key] = result
    return result


def compile_w4a16_fused_moe(
    *,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    activation: str,
    apply_router_weight_on_input: bool,
    zero_fc2_output: bool,
    moe_block_size: int,
    max_m_blocks: int,
    element_dtype: str = "bf16",
    fast_math: bool = True,
    sms: int,
    max_shared_mem: int,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    weight_layout: str = "packed",
    scale_format: str = "e4m3_k16",
    w13_layout: str = "w13",
    direct_topk_routes: bool = False,
    tc_decode_fused_sum: bool = False,
    collect_activation_amax: bool = False,
    force_tile_config: tuple[int, int, int, int] | None = None,
) -> W4A16FusedMoeCompileResult:
    scale_format = _normalize_scale_format(scale_format)
    cutlass_dtype = _cutlass_element_dtype(element_dtype)
    device = int(torch.cuda.current_device()) if torch.cuda.is_available() else None
    activation = normalize_moe_activation(activation)
    is_gated = validate_activation(activation)
    swiglu_limit, swiglu_alpha, swiglu_beta = _normalize_activation_swiglu_params(
        activation,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
    )
    if weight_layout not in _WEIGHT_LAYOUTS:
        raise ValueError(f"unsupported W4A16 weight_layout {weight_layout!r}")
    if weight_layout == "modelopt":
        if w13_layout not in _MODEL_OPT_W13_LAYOUTS:
            raise ValueError(f"unsupported W4A16 w13_layout {w13_layout!r}")
    else:
        w13_layout = "packed"
    direct_topk_routes = bool(direct_topk_routes)
    tc_decode_fused_sum = bool(tc_decode_fused_sum)
    collect_activation_amax = bool(collect_activation_amax)
    if collect_activation_amax and (direct_topk_routes or tc_decode_fused_sum):
        raise ValueError(
            "W4A16 activation amax collection requires the route-packed fused path"
        )
    # The TC-decode path validates M in {1,2,4,8} itself and uses direct-topk
    # routing for the whole {1,2,4,8} range, so it lifts the default decode cap.
    direct_topk_m_cap = (
        _W4A16_SMALL_M_DIRECT_MAX_M if tc_decode_fused_sum else _MAX_DIRECT_TOPK_ROUTE_M
    )
    if direct_topk_routes and (
        int(size_m) > direct_topk_m_cap
        or weight_layout not in ("packed", "nf3_2p1")
        or bool(zero_fc2_output)
    ):
        raise ValueError(
            "direct_topk_routes is only valid for small-M packed W4A16 without expert_map"
        )
    fc1_cols = int(intermediate_size) * (2 if is_gated else 1)
    routed_rows = int(size_m) * int(top_k)
    # Logical K/N tails are needed for every shard the tile table can't
    # divide, not just sub-32 ones: 2048/TP6 = 352 and 3072/TP16 = 192 are
    # 32-aligned yet have no dividing tile_k/tile_n. %32 shards are the
    # ceil-scale-grid subset of the same machinery (%32 != 0 implies
    # %128 != 0).
    allow_native_logical_tail = (
        weight_layout == "modelopt"
        and scale_format == "e8m0_k32"
        and int(intermediate_size) % 128 != 0
    )
    fc1_tile_k, fc1_tile_n, fc1_cta_threads, _ = _select_tile_config(
        problem_m=size_m,
        problem_n=fc1_cols,
        problem_k=hidden_size,
        top_k=top_k,
        moe_block_size=moe_block_size,
        sms=sms,
        max_shared_mem=max_shared_mem,
        scale_format=scale_format,
        weight_layout=weight_layout,
        allow_logical_tail=allow_native_logical_tail,
    )
    fc2_tile_k, fc2_tile_n, fc2_cta_threads, _ = _select_tile_config(
        problem_m=routed_rows,
        problem_n=hidden_size,
        problem_k=intermediate_size,
        top_k=1,
        moe_block_size=moe_block_size,
        sms=sms,
        max_shared_mem=max_shared_mem,
        scale_format=scale_format,
        weight_layout=weight_layout,
        allow_logical_tail=allow_native_logical_tail,
    )
    if fc1_cta_threads != fc2_cta_threads:
        common_cta_threads = min(fc1_cta_threads, fc2_cta_threads)
        fc1_tile_k, fc1_tile_n, fc1_cta_threads, _ = _select_tile_config(
            problem_m=size_m,
            problem_n=fc1_cols,
            problem_k=hidden_size,
            top_k=top_k,
            moe_block_size=moe_block_size,
            sms=sms,
            max_shared_mem=max_shared_mem,
            required_cta_threads=common_cta_threads,
            scale_format=scale_format,
            weight_layout=weight_layout,
            allow_logical_tail=allow_native_logical_tail,
        )
        fc2_tile_k, fc2_tile_n, fc2_cta_threads, _ = _select_tile_config(
            problem_m=routed_rows,
            problem_n=hidden_size,
            problem_k=intermediate_size,
            top_k=1,
            moe_block_size=moe_block_size,
            sms=sms,
            max_shared_mem=max_shared_mem,
            required_cta_threads=common_cta_threads,
            scale_format=scale_format,
            weight_layout=weight_layout,
            allow_logical_tail=allow_native_logical_tail,
        )
        if fc1_cta_threads != fc2_cta_threads:
            raise ValueError(
                "fused W4A16 FC1/FC2 selected different thread counts: "
                f"{fc1_cta_threads} vs {fc2_cta_threads}"
            )
    # TC-decode FC1 wide-N override (single-wave-collapse sizes only): in the m8
    # fused path the host right-sizes grid_x to the FC1 mn-tile count and forces
    # one whole mn-tile per CTA over the full K. With the default fc1_tile_n=128,
    # FC1 (N=fc1_cols=2*intermediate_size) produces size_m*top_k*(fc1_cols/128)
    # mn-tiles. For TP=2 I_tp=1024 (fc1_cols=2048 => 16 n-tiles/route) bs=1 is
    # 96 tiles (<= 188 SMs, one wave); bs=2 is 192 -- JUST over the single-wave
    # SM cap -- forcing a 2-wave launch of grid_x=96 (half the machine idle per
    # wave, a serialized second FC1 wave of pure tail latency on the bandwidth-
    # bound decode). Widening FC1 to tile_n=256 (256-wide N slab per CTA over
    # full K; tile_k=64 keeps cta_threads=256) HALVES FC1's mn-tile count, so
    # bs=2 collapses to 96 tiles = exactly ONE wave, removing that whole second
    # FC1 wave. The narrower tile_k=64 is slower per-tile, so we apply this ONLY
    # where it turns a 2-wave launch into a 1-wave launch: the default 128-wide
    # FC1 spans 2 waves (sms < default_mn_tiles <= 2*sms) AND the wide tile fits
    # in one (default/2 <= sms). bs=1 (one wave already) and bs>=4 (still multi-
    # wave after halving) keep the faster default 128x128 tile.
    # Guarded by fc1_cols%256==0, smem-fit, and 256-thread geometry so the fused
    # FC1/FC2 single-thread-geometry contract is preserved.
    default_fc1_mn_tiles = (
        int(size_m) * int(top_k) * (int(fc1_cols) // int(fc1_tile_n))
        if fc1_tile_n > 0
        else 0
    )
    if (
        bool(tc_decode_fused_sum)
        and int(fc1_cols) % 256 == 0
        and fc1_tile_n == 128
        and (fc1_tile_n * fc1_tile_k) // 64 == 256
        and int(sms) < default_fc1_mn_tiles <= 2 * int(sms)
        and (default_fc1_mn_tiles // 2) <= int(sms)
    ):
        wide_fc1_tile_k = 64
        if _candidate_tile_fits(
            problem_n=fc1_cols,
            problem_k=hidden_size,
            cta_m_blocks=_covering_count(moe_block_size, 16),
            tile_n=256,
            tile_k=wide_fc1_tile_k,
            cta_threads=256,
            max_shared_mem=int(max_shared_mem) - 512,
            scale_format=scale_format,
            weight_layout=weight_layout,
        ):
            fc1_tile_n = 256
            fc1_tile_k = wide_fc1_tile_k
            fc1_cta_threads = 256
    # TC-decode FC2 wide-N override: in the m8 fused path the host right-sizes
    # grid_x to the FC1 mn-tile count (one persistent wave, no split-K). FC2
    # (N=hidden_size, K=intermediate_size) with the default tile_n=128 produces
    # route_blocks*(hidden_size/128) mn-tiles -- roughly double FC1's count --
    # so FC2 would need ~2 persistent waves while FC1 fits in 1; that second
    # FC2 wave is pure serialized tail latency on the bandwidth-bound decode.
    # Selecting tile_n=256 for FC2 (a 256-wide N slab per CTA over full K)
    # halves FC2's mn-tile count so it also fits one wave. Guarded by smem-fit,
    # hidden_size%256==0, and matching cta_threads so the fused single
    # thread-geometry contract is preserved.
    if (
        bool(tc_decode_fused_sum)
        and int(hidden_size) % 256 == 0
        and fc2_tile_n == 128
        and fc1_cta_threads == 256
    ):
        wide_fc2_tile_k = 64
        if _candidate_tile_fits(
            problem_n=hidden_size,
            problem_k=intermediate_size,
            cta_m_blocks=_covering_count(moe_block_size, 16),
            tile_n=256,
            tile_k=wide_fc2_tile_k,
            cta_threads=256,
            max_shared_mem=int(max_shared_mem) - 512,
            scale_format=scale_format,
            weight_layout=weight_layout,
        ):
            fc2_tile_n = 256
            fc2_tile_k = wide_fc2_tile_k
            fc2_cta_threads = 256
    # TC-decode FC2 ultra-wide override (perfect wave-balance with FC1): the
    # persistent grid_x is right-sized to FC1's mn-tile count. After the FC1/FC2
    # wide-N (tile_n=256) overrides, bs=2 still has FC1=route_blocks*8 tiles vs
    # FC2=route_blocks*16 tiles -- FC2 is DOUBLE FC1 and thus needs a second
    # persistent wave at grid_x sized to FC1. That FC2 second wave is pure
    # serialized tail latency on the bandwidth-bound decode. Widening FC2 to
    # tile_n=512 (a 512-wide N slab per CTA over full K) halves FC2's mn-tile
    # count again so FC2 == FC1's tile count and fits the SAME single wave. A
    # 512-wide N tile needs tile_k=32 to keep cta_threads=256 (512*32/64); that
    # is below the generic tile_k>=64 fits-floor, so we validate the footprint
    # directly here. tile_k=32 == the e8m0_k32 scale group, so cta_k_blocks=2
    # with one e8m0 scale group per k-tile -- the existing scale layout is a
    # clean covering. Fire ONLY when it drops FC2 from >1 wave to FC1's wave
    # count (bs=2). Numerically identical: only the FC2 output-tile width changes.
    fc1_mn_after = (
        int(size_m) * int(top_k) * (int(fc1_cols) // int(fc1_tile_n))
        if fc1_tile_n > 0
        else 0
    )
    fc2_mn_after = (
        int(size_m) * int(top_k) * (int(hidden_size) // int(fc2_tile_n))
        if fc2_tile_n > 0
        else 0
    )
    if (
        bool(tc_decode_fused_sum)
        and int(hidden_size) % 512 == 0
        and fc2_tile_n == 256
        and fc1_cta_threads == 256
        and fc1_mn_after > 0
        and fc2_mn_after > fc1_mn_after
        and (fc2_mn_after // 2) <= fc1_mn_after
        and fc1_mn_after <= int(sms)
    ):
        ultra_fc2_tile_k = 32
        ultra_smem = _shared_memory_footprint(
            cta_m_blocks=_covering_count(moe_block_size, 16),
            tile_n=512,
            tile_k=ultra_fc2_tile_k,
            scale_format=scale_format,
            weight_layout=weight_layout,
        )
        if (
            int(intermediate_size) % ultra_fc2_tile_k == 0
            and ultra_smem <= int(max_shared_mem) - 512
        ):
            fc2_tile_n = 512
            fc2_tile_k = ultra_fc2_tile_k
            fc2_cta_threads = 256
    if force_tile_config is not None:
        # Explicit (fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n) pin. The
        # NF3 ("nf3_2p1") flat-span weight layout is packed for a specific CTA
        # N-tile, so hybrid deployments pin ONE tile config across every m
        # regime instead of the m-dependent auto selection (and TC-decode
        # wide-N overrides) above. Overrides whatever was selected; the tiles
        # land in the GEMM cache keys, so no cache collision is possible.
        fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n = (
            int(v) for v in force_tile_config
        )
        fc1_cta_threads = (fc1_tile_n * fc1_tile_k) // 64
        fc2_cta_threads = (fc2_tile_n * fc2_tile_k) // 64
        if fc1_cta_threads != fc2_cta_threads:
            raise ValueError(
                "force_tile_config FC1/FC2 thread counts must match, got "
                f"{fc1_cta_threads} vs {fc2_cta_threads}"
            )
        for name, forced_pn, forced_pk, forced_tn, forced_tk in (
            ("fc1", fc1_cols, hidden_size, fc1_tile_n, fc1_tile_k),
            ("fc2", hidden_size, intermediate_size, fc2_tile_n, fc2_tile_k),
        ):
            if not _candidate_tile_fits(
                problem_n=forced_pn,
                problem_k=forced_pk,
                cta_m_blocks=_covering_count(moe_block_size, 16),
                tile_n=forced_tn,
                tile_k=forced_tk,
                cta_threads=fc1_cta_threads,
                max_shared_mem=int(max_shared_mem) - 512,
                scale_format=scale_format,
                weight_layout=weight_layout,
                allow_logical_tail=allow_native_logical_tail,
            ):
                raise ValueError(
                    f"force_tile_config {name} tile "
                    f"(tile_k={forced_tk}, tile_n={forced_tn}) does not fit "
                    f"problem N/K={forced_pn}/{forced_pk} at "
                    f"moe_block_size={moe_block_size}"
                )
    kernel = W4A16FusedMoeKernel(
        size_m=size_m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        zero_fc2_output=zero_fc2_output,
        fc1_tile_n=fc1_tile_n,
        fc1_tile_k=fc1_tile_k,
        fc2_tile_n=fc2_tile_n,
        fc2_tile_k=fc2_tile_k,
        moe_block_size=moe_block_size,
        max_m_blocks=max_m_blocks,
        element_dtype=element_dtype,
        fast_math=fast_math,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        weight_layout=weight_layout,
        scale_format=scale_format,
        w13_layout=w13_layout,
        direct_topk_routes=direct_topk_routes,
        tc_decode_fused_sum=tc_decode_fused_sum,
        collect_activation_amax=collect_activation_amax,
    )
    cache_key = (
        "w4a16_fused_moe",
        device,
        kernel.__cache_key__,
    )
    cached = _FUSED_CACHE.get(cache_key)
    if cached is not None:
        return replace(
            cached,
            size_m=size_m,
            max_m_blocks=max_m_blocks,
            blocks_per_sm=kernel.blocks_per_sm,
        )

    if (not collect_activation_amax) and _small_m_direct_supported(
        m=size_m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=top_k,
        activation=activation,
        apply_router_weight_on_input=bool(apply_router_weight_on_input),
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        element_dtype=element_dtype,
        weight_layout=weight_layout,
        w13_layout=w13_layout,
        scale_format=scale_format,
    ):
        for ids_dtype in (torch.int32, torch.int64):
            _compile_w4a16_small_m_direct(
                m=size_m,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                topk=top_k,
                activation=activation,
                fast_math=fast_math,
                topk_ids_dtype=ids_dtype,
                scale_format=scale_format,
                swiglu_limit=swiglu_limit,
                swiglu_alpha=swiglu_alpha,
                swiglu_beta=swiglu_beta,
                w13_layout=w13_layout,
                device=torch.device("cuda", device) if device is not None else None,
            )

    compile_size_m = _fake_m_for_specialization(size_m)
    compile_routed_rows = int(compile_size_m) * int(top_k)
    compile_route_blocks = compile_routed_rows if direct_topk_routes else 1
    compile_route_slots = compile_route_blocks * int(moe_block_size)
    packed_route_fake_elements = (
        compile_routed_rows if direct_topk_routes else compile_route_slots
    )
    a_fake = make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    if weight_layout == "modelopt":
        w13_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8,
            (num_experts * fc1_cols * (hidden_size // 2),),
            assumed_align=16,
        )
        w2_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8,
            (num_experts * hidden_size * (intermediate_size // 2),),
            assumed_align=16,
        )
    elif weight_layout == "nf3_2p1":
        # NF3: int32, 3 words per 32-code unit; (size_n // 2) units per K16 row.
        w13_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (num_experts * (hidden_size // 16) * (fc1_cols // 2) * 3,),
            assumed_align=16,
        )
        w2_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (num_experts * (intermediate_size // 16) * (hidden_size // 2) * 3,),
            assumed_align=16,
        )
    else:
        w13_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (num_experts * (hidden_size // 16) * (fc1_cols // 16 * 32),),
            assumed_align=16,
        )
        w2_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (num_experts * (intermediate_size // 16) * (hidden_size // 16 * 32),),
            assumed_align=16,
        )
    fc1_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (compile_routed_rows * fc1_cols,),
        assumed_align=16,
    )
    activated_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (compile_routed_rows * intermediate_size,),
        assumed_align=16,
    )
    fc2_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (compile_routed_rows * hidden_size,),
        assumed_align=16,
    )
    w13_scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (
            _scale_fake_int32_elements(
                num_experts=num_experts,
                size_k=hidden_size,
                size_n=fc1_cols,
                scale_format=scale_format,
                allow_n_tail=allow_native_logical_tail,
            ),
        ),
        assumed_align=16,
    )
    w2_scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (
            _scale_fake_int32_elements(
                num_experts=num_experts,
                size_k=intermediate_size,
                size_n=hidden_size,
                scale_format=scale_format,
                allow_k_tail=allow_native_logical_tail,
            ),
        ),
        assumed_align=16,
    )
    w13_global_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (num_experts,),
        assumed_align=16,
    )
    w2_global_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (num_experts,),
        assumed_align=16,
    )
    packed_routes_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (packed_route_fake_elements,),
        assumed_align=16,
    )
    block_experts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (compile_route_blocks,),
        assumed_align=16,
    )
    route_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    activation_amax_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (num_experts * 2,),
        assumed_align=4,
    )
    topk_fake = make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4)
    fc1_c_tmp_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (
            max(
                fc1_cols * compile_route_slots,
                4 * 256 * moe_block_size * 256,
            ),
        ),
        assumed_align=16,
    )
    fc2_c_tmp_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (
            max(
                hidden_size * compile_route_slots,
                4 * 256 * moe_block_size * 256,
            ),
        ),
        assumed_align=16,
    )
    locks_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (4 * 256 + 2,),
        assumed_align=16,
    )

    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = sm12x_compile(
        kernel,
        a_fake,
        w13_fake,
        w2_fake,
        fc1_fake,
        activated_fake,
        fc2_fake,
        w13_scales_fake,
        w2_scales_fake,
        w13_global_fake,
        w2_global_fake,
        packed_routes_fake,
        block_experts_fake,
        route_count_fake,
        activation_amax_fake,
        0,
        topk_fake,
        fc1_c_tmp_fake,
        fc2_c_tmp_fake,
        locks_fake,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "moe.w4a16.fused_moe",
            1,
            cache_key,
        ),
        dsl_compile_options=OptLevel(2),
    )
    result = W4A16FusedMoeCompileResult(
        compiled=compiled,
        size_m=size_m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        activation=activation,
        apply_router_weight_on_input=bool(apply_router_weight_on_input),
        zero_fc2_output=bool(zero_fc2_output),
        element_dtype=element_dtype,
        fast_math=bool(fast_math),
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        fc1_tile_n=fc1_tile_n,
        fc1_tile_k=fc1_tile_k,
        fc2_tile_n=fc2_tile_n,
        fc2_tile_k=fc2_tile_k,
        moe_block_size=moe_block_size,
        max_m_blocks=max_m_blocks,
        blocks_per_sm=kernel.blocks_per_sm,
        weight_layout=weight_layout,
        w13_layout=w13_layout,
        direct_topk_routes=kernel.direct_topk_routes,
        scale_format=scale_format,
        tc_decode_fused_sum=bool(tc_decode_fused_sum),
        collect_activation_amax=collect_activation_amax,
    )
    _FUSED_CACHE[cache_key] = result
    return result


@dataclass(frozen=True)
class W4A16FusedMoeHybridCompileResult:
    compiled: object
    size_m: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    activation: str
    element_dtype: str
    fast_math: bool
    map_slots: int
    tier0_num_experts: int
    tier0_weight_layout: str
    tier0_scale_format: str
    tier0_w13_layout: str
    tier1_num_experts: int
    tier1_weight_layout: str
    tier1_scale_format: str
    tier1_w13_layout: str
    fc1_tile_n: int
    fc1_tile_k: int
    fc2_tile_n: int
    fc2_tile_k: int
    moe_block_size: int
    max_m_blocks: int
    cta_threads: int
    blocks_per_sm: int
    shared_memory_bytes: int
    schedule_whole_tiles: bool
    direct_topk_routes: bool
    tc_decode_fused_sum: bool
    # -1 when the compiled object came from the on-disk object cache, whose
    # loader does not expose CUDA-dialect introspection; a fresh compile of the
    # identical source/toolchain state ran the spill admission below at least
    # once before the entry could exist.
    registers_per_thread: int
    local_memory_bytes: int


def _query_w4a16_kernel_resources(compiled: object) -> tuple[str, int, int] | None:
    """Return (symbol, registers/thread, local bytes/thread) for a one-kernel
    CUDA-dialect compile result, or None when the object does not expose the
    introspection surface (e.g. an on-disk object-cache reload)."""

    kernel_info = getattr(compiled, "kernel_info", None)
    to_executor = getattr(compiled, "to", None)
    if not isinstance(kernel_info, dict) or not callable(to_executor):
        return None
    symbols = tuple(kernel_info)
    if len(symbols) != 1 or not isinstance(symbols[0], str) or not symbols[0]:
        return None
    executor = to_executor(int(torch.cuda.current_device()))
    libraries = tuple(
        getattr(getattr(executor, "jit_module", None), "cuda_library", None) or ()
    )
    if len(libraries) != 1:
        return None
    success = cuda_runtime.cudaError_t(0)
    kernel_status, kernel_handle = cuda_runtime.cudaLibraryGetKernel(
        libraries[0], symbols[0].encode("utf-8")
    )
    if kernel_status != success:
        raise RuntimeError(
            f"cudaLibraryGetKernel failed for {symbols[0]}: {kernel_status}"
        )
    attributes_status, attributes = cuda_runtime.cudaFuncGetAttributes(kernel_handle)
    if attributes_status != success:
        raise RuntimeError(
            f"cudaFuncGetAttributes failed for {symbols[0]}: {attributes_status}"
        )
    registers_per_thread = int(getattr(attributes, "numRegs", -1))
    local_memory_bytes = int(getattr(attributes, "localSizeBytes", -1))
    if registers_per_thread < 0 or local_memory_bytes < 0:
        raise RuntimeError(f"incomplete CUDA function attributes for {symbols[0]}")
    return symbols[0], registers_per_thread, local_memory_bytes


def _w4a16_weight_flat_elements(
    *,
    num_experts: int,
    size_n: int,
    size_k: int,
    weight_layout: str,
) -> int:
    """Flat element count of a packed W4A16 weight tensor (uint8 for modelopt,
    int32 otherwise), matching the compile-time fake construction."""

    if weight_layout == "modelopt":
        return int(num_experts) * int(size_n) * (int(size_k) // 2)
    if weight_layout == "nf3_2p1":
        return int(num_experts) * (int(size_k) // 16) * (int(size_n) // 2) * 3
    return int(num_experts) * (int(size_k) // 16) * (int(size_n) // 16 * 32)


def compile_w4a16_fused_moe_hybrid(
    *,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    tier0_num_experts: int,
    tier1_num_experts: int,
    top_k: int,
    activation: str,
    map_slots: int,
    moe_block_size: int = 8,
    element_dtype: str = "bf16",
    fast_math: bool = True,
    sms: int,
    max_shared_mem: int,
    tier0_weight_layout: str = "packed",
    tier0_scale_format: str = "e4m3_k16",
    tier0_w13_layout: str = "packed",
    tier1_weight_layout: str = "nf3_2p1",
    tier1_scale_format: str = "e4m3_k32",
    tier1_w13_layout: str = "w13",
    force_tile_config: tuple[int, int, int, int],
    schedule_whole_tiles: bool = True,
) -> W4A16FusedMoeHybridCompileResult:
    """Compile the two-tier heterogeneous-quantization fused MoE kernel.

    v1 contract: TC-decode direct-topk geometry (gated activation, fused FC2
    top-k sum) with an explicitly pinned tile config shared by both tiers.
    Spill admission is fail-closed on every fresh compile: nonzero local
    memory raises before the result can be cached or launched.
    """

    activation = normalize_moe_activation(activation)
    if not validate_activation(activation):
        raise ValueError("hybrid W4A16 requires a gated activation")
    cutlass_dtype = _cutlass_element_dtype(element_dtype)
    device = int(torch.cuda.current_device()) if torch.cuda.is_available() else None
    fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n = (int(v) for v in force_tile_config)
    max_m_blocks = int(size_m) * int(top_k)

    def _tier_kernel(
        num_experts: int, weight_layout: str, scale_format: str, w13_layout: str
    ) -> W4A16FusedMoeKernel:
        return W4A16FusedMoeKernel(
            size_m=size_m,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            activation=activation,
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            fc1_tile_n=fc1_tile_n,
            fc1_tile_k=fc1_tile_k,
            fc2_tile_n=fc2_tile_n,
            fc2_tile_k=fc2_tile_k,
            moe_block_size=moe_block_size,
            max_m_blocks=max_m_blocks,
            element_dtype=element_dtype,
            fast_math=fast_math,
            weight_layout=weight_layout,
            scale_format=scale_format,
            w13_layout=w13_layout,
            direct_topk_routes=True,
            tc_decode_fused_sum=True,
            schedule_whole_tiles=bool(schedule_whole_tiles),
        )

    kernel = W4A16FusedMoeHybridKernel(
        tier0=_tier_kernel(
            int(tier0_num_experts),
            tier0_weight_layout,
            _normalize_scale_format(tier0_scale_format),
            tier0_w13_layout,
        ),
        tier1=_tier_kernel(
            int(tier1_num_experts),
            tier1_weight_layout,
            _normalize_scale_format(tier1_scale_format),
            tier1_w13_layout,
        ),
        map_slots=int(map_slots),
    )
    if kernel.shared_words * 4 > int(max_shared_mem) - 512:
        raise ValueError(
            "hybrid W4A16 shared memory exceeds the device limit: "
            f"{kernel.shared_words * 4} > {int(max_shared_mem) - 512}"
        )

    cache_key = (
        "w4a16_fused_moe_hybrid",
        device,
        kernel.__cache_key__,
    )
    cached = _FUSED_CACHE.get(cache_key)
    if cached is not None:
        return replace(cached, size_m=size_m, max_m_blocks=max_m_blocks)

    fc1_cols = kernel.tier0.fc1_cols
    compile_size_m = _fake_m_for_specialization(size_m)
    compile_routed_rows = int(compile_size_m) * int(top_k)

    def _weight_fake(num_experts: int, size_n: int, size_k: int, weight_layout: str):
        elements = _w4a16_weight_flat_elements(
            num_experts=num_experts,
            size_n=size_n,
            size_k=size_k,
            weight_layout=weight_layout,
        )
        fake_dtype = cutlass.Uint8 if weight_layout == "modelopt" else cutlass.Int32
        return cute.runtime.make_fake_compact_tensor(
            fake_dtype, (elements,), assumed_align=16
        )

    def _scales_fake(num_experts: int, size_n: int, size_k: int, scale_format: str):
        return cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (
                _scale_fake_int32_elements(
                    num_experts=num_experts,
                    size_k=size_k,
                    size_n=size_n,
                    scale_format=scale_format,
                ),
            ),
            assumed_align=16,
        )

    def _global_fake(num_experts: int):
        return cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (num_experts,), assumed_align=16
        )

    a_fake = make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    topk_fake = make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4)
    tier0 = kernel.tier0
    tier1 = kernel.tier1
    scratch_elements = max(
        fc1_cols * compile_routed_rows,
        hidden_size * compile_routed_rows,
        4 * 256 * moe_block_size * 256,
    )
    compile_args = (
        a_fake,
        _weight_fake(tier0.num_experts, fc1_cols, hidden_size, tier0.weight_layout),
        _weight_fake(
            tier0.num_experts, hidden_size, intermediate_size, tier0.weight_layout
        ),
        _scales_fake(tier0.num_experts, fc1_cols, hidden_size, tier0.scale_format),
        _scales_fake(
            tier0.num_experts, hidden_size, intermediate_size, tier0.scale_format
        ),
        _global_fake(tier0.num_experts),
        _global_fake(tier0.num_experts),
        _weight_fake(tier1.num_experts, fc1_cols, hidden_size, tier1.weight_layout),
        _weight_fake(
            tier1.num_experts, hidden_size, intermediate_size, tier1.weight_layout
        ),
        _scales_fake(tier1.num_experts, fc1_cols, hidden_size, tier1.scale_format),
        _scales_fake(
            tier1.num_experts, hidden_size, intermediate_size, tier1.scale_format
        ),
        _global_fake(tier1.num_experts),
        _global_fake(tier1.num_experts),
        cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (compile_routed_rows,), assumed_align=16
        ),
        cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (int(map_slots),), assumed_align=16
        ),
        cute.runtime.make_fake_compact_tensor(
            cutlass_dtype, (compile_routed_rows * fc1_cols,), assumed_align=16
        ),
        cute.runtime.make_fake_compact_tensor(
            cutlass_dtype,
            (compile_routed_rows * intermediate_size,),
            assumed_align=16,
        ),
        cute.runtime.make_fake_compact_tensor(
            cutlass_dtype, (compile_size_m * hidden_size,), assumed_align=16
        ),
        topk_fake,
        cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (scratch_elements,), assumed_align=16
        ),
        cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (scratch_elements,), assumed_align=16
        ),
        cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (4 * 256 + 2,), assumed_align=16
        ),
        1,
        1,
        current_cuda_stream(),
    )

    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = sm12x_compile(
        kernel,
        *compile_args,
        compile_spec=KernelCompileSpec.from_key(
            "moe.w4a16.fused_moe_hybrid",
            W4A16FusedMoeHybridKernel.ABI_VERSION,
            cache_key,
        ),
    )
    registers_per_thread = -1
    local_memory_bytes = -1
    resources = _query_w4a16_kernel_resources(compiled)
    if resources is not None:
        _, registers_per_thread, local_memory_bytes = resources
        if local_memory_bytes != 0:
            raise RuntimeError(
                "hybrid W4A16 codegen spills to local memory "
                f"({local_memory_bytes} bytes/thread); refusing to admit the "
                "kernel for the latency-critical decode path"
            )

    result = W4A16FusedMoeHybridCompileResult(
        compiled=compiled,
        size_m=size_m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        activation=activation,
        element_dtype=element_dtype,
        fast_math=bool(fast_math),
        map_slots=int(map_slots),
        tier0_num_experts=tier0.num_experts,
        tier0_weight_layout=tier0.weight_layout,
        tier0_scale_format=tier0.scale_format,
        tier0_w13_layout=tier0.w13_layout,
        tier1_num_experts=tier1.num_experts,
        tier1_weight_layout=tier1.weight_layout,
        tier1_scale_format=tier1.scale_format,
        tier1_w13_layout=tier1.w13_layout,
        fc1_tile_n=fc1_tile_n,
        fc1_tile_k=fc1_tile_k,
        fc2_tile_n=fc2_tile_n,
        fc2_tile_k=fc2_tile_k,
        moe_block_size=moe_block_size,
        max_m_blocks=max_m_blocks,
        cta_threads=kernel.cta_threads,
        blocks_per_sm=kernel.blocks_per_sm,
        shared_memory_bytes=kernel.shared_words * 4,
        schedule_whole_tiles=bool(schedule_whole_tiles),
        direct_topk_routes=True,
        tc_decode_fused_sum=True,
        registers_per_thread=registers_per_thread,
        local_memory_bytes=local_memory_bytes,
    )
    _FUSED_CACHE[cache_key] = result
    return result


def clear_w4a16_kernel_cache() -> None:
    _CACHE.clear()
    _FUSED_CACHE.clear()
    _ACTIVATION_CACHE.clear()
    _SUM_CACHE.clear()
    _SMALL_M_DIRECT_CACHE.clear()


def compile_w4a16_activation(
    *,
    rows: int,
    intermediate_size: int,
    activation: str,
    element_dtype: str = "bf16",
    fast_math: bool = True,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
) -> W4A16ActivationCompileResult:
    cutlass_dtype = _cutlass_element_dtype(element_dtype)
    activation = normalize_moe_activation(activation)
    is_gated = validate_activation(activation)
    swiglu_limit, swiglu_alpha, swiglu_beta = _normalize_activation_swiglu_params(
        activation,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
    )
    kernel = W4A16ActivationKernel(
        rows=rows,
        intermediate_size=intermediate_size,
        activation=activation,
        element_dtype=element_dtype,
        fast_math=fast_math,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
    )
    cache_key = (
        "w4a16_activation",
        kernel.__cache_key__,
    )
    cached = _ACTIVATION_CACHE.get(cache_key)
    if cached is not None:
        return replace(cached, rows=rows)

    w13_shards = 2 if is_gated else 1
    compile_rows = _fake_m_for_specialization(rows)
    fc1_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (compile_rows * w13_shards * intermediate_size,),
        assumed_align=16,
    )
    activated_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (compile_rows * intermediate_size,),
        assumed_align=16,
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = sm12x_compile(
        kernel,
        fc1_fake,
        activated_fake,
        Int32(compile_rows),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "moe.w4a16.activation",
            1,
            cache_key,
        ),
    )
    result = W4A16ActivationCompileResult(
        compiled=compiled,
        rows=rows,
        intermediate_size=intermediate_size,
        activation=activation,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
    )
    _ACTIVATION_CACHE[cache_key] = result
    return result


def compile_w4a16_topk_sum(
    *,
    m: int,
    topk: int,
    hidden_size: int,
    element_dtype: str = "bf16",
) -> W4A16TopKSumCompileResult:
    cutlass_dtype = _cutlass_element_dtype(element_dtype)
    cache_key = ("w4a16_topk_sum", element_dtype, topk, hidden_size)
    cached = _SUM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    fc2_fake = make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    output_fake = make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    kernel = W4A16TopKSumKernel(
        topk=topk,
        hidden_size=hidden_size,
        element_dtype=element_dtype,
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = sm12x_compile(
        kernel,
        fc2_fake,
        output_fake,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "moe.w4a16.topk_sum",
            1,
            cache_key,
        ),
    )
    result = W4A16TopKSumCompileResult(
        compiled=compiled,
        m=0,
        topk=topk,
        hidden_size=hidden_size,
    )
    _SUM_CACHE[cache_key] = result
    return result


def _w4a16_small_m_direct_launch_flat(
    a_input: torch.Tensor,
    w13_u8: torch.Tensor,
    w13_scale_u8: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    inter_u32: torch.Tensor,
    w2_u8: torch.Tensor,
    w2_scale_u8: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    barrier_count: torch.Tensor,
    barrier_epoch: torch.Tensor,
    m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    activation: str,
    fast_math: bool,
    scale_format: str,
    has_swiglu_limit: bool,
    swiglu_limit_value: float,
    swiglu_alpha: float,
    swiglu_beta: float,
    w13_layout: str,
    stream_int: int,
) -> None:
    swiglu_limit = float(swiglu_limit_value) if has_swiglu_limit else None
    direct_launch = _compile_w4a16_small_m_direct(
        m=m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=topk,
        activation=activation,
        fast_math=bool(fast_math),
        topk_ids_dtype=topk_ids.dtype,
        scale_format=scale_format,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        w13_layout=w13_layout,
        device=a_input.device,
    )

    def ptr(dt, tensor: torch.Tensor):
        return make_ptr(dt, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    ids_dtype = cutlass.Int64 if topk_ids.dtype == torch.int64 else cutlass.Int32
    direct_launch.compiled(
        ptr(cutlass.BFloat16, a_input),
        ptr(cutlass.Uint8, w13_u8),
        ptr(cutlass.Uint8, w13_scale_u8.view(torch.uint8)),
        ptr(cutlass.Float32, w13_global_scale),
        ptr(cutlass.Float32, w13_global_scale),
        ptr(cutlass.Float32, w2_global_scale),
        ptr(cutlass.Uint32, inter_u32.view(torch.uint32)),
        ptr(cutlass.Uint8, w2_u8),
        ptr(cutlass.Uint8, w2_scale_u8.view(torch.uint8)),
        ptr(cutlass.Float32, w2_global_scale),
        ptr(ids_dtype, topk_ids),
        ptr(cutlass.Float32, topk_weights),
        ptr(cutlass.BFloat16, output),
        barrier_count,
        barrier_epoch,
        Int32(m),
        Int32(direct_launch.grid_x),
        cuda.CUstream(stream_int),
    )


@torch.library.custom_op(
    "flashinfer_sm12x::w4a16_small_m_direct_launch",
    mutates_args="unknown",
)
def _w4a16_small_m_direct_launch_op(
    a_input: torch.Tensor,
    w13_u8: torch.Tensor,
    w13_scale_u8: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    inter_u32: torch.Tensor,
    w2_u8: torch.Tensor,
    w2_scale_u8: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    barrier_count: torch.Tensor,
    barrier_epoch: torch.Tensor,
    m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    activation: str,
    fast_math: bool,
    scale_format: str,
    has_swiglu_limit: bool,
    swiglu_limit_value: float,
    swiglu_alpha: float,
    swiglu_beta: float,
    w13_layout: str,
    stream_int: int,
) -> None:
    _w4a16_small_m_direct_launch_flat(
        a_input=a_input,
        w13_u8=w13_u8,
        w13_scale_u8=w13_scale_u8,
        w13_global_scale=w13_global_scale,
        w2_global_scale=w2_global_scale,
        inter_u32=inter_u32,
        w2_u8=w2_u8,
        w2_scale_u8=w2_scale_u8,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        output=output,
        barrier_count=barrier_count,
        barrier_epoch=barrier_epoch,
        m=m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=topk,
        activation=activation,
        fast_math=fast_math,
        scale_format=scale_format,
        has_swiglu_limit=has_swiglu_limit,
        swiglu_limit_value=swiglu_limit_value,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        w13_layout=w13_layout,
        stream_int=stream_int,
    )


@_w4a16_small_m_direct_launch_op.register_fake
def _w4a16_small_m_direct_launch_fake(
    a_input: torch.Tensor,
    w13_u8: torch.Tensor,
    w13_scale_u8: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    inter_u32: torch.Tensor,
    w2_u8: torch.Tensor,
    w2_scale_u8: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    barrier_count: torch.Tensor,
    barrier_epoch: torch.Tensor,
    m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    activation: str,
    fast_math: bool,
    scale_format: str,
    has_swiglu_limit: bool,
    swiglu_limit_value: float,
    swiglu_alpha: float,
    swiglu_beta: float,
    w13_layout: str,
    stream_int: int,
) -> None:
    return None


def _w4a16_fused_moe_launch_flat(
    a_input: torch.Tensor,
    w13_arg: torch.Tensor,
    w2_arg: torch.Tensor,
    fc1_out: torch.Tensor,
    activated: torch.Tensor,
    fc2_out: torch.Tensor,
    w13_scale_i32: torch.Tensor,
    w2_scale_i32: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    packed_route_indices: torch.Tensor,
    block_expert_ids: torch.Tensor,
    packed_route_count: torch.Tensor,
    activation_amax: torch.Tensor | None,
    layer_idx: int,
    topk_weights: torch.Tensor,
    fc1_scratch: torch.Tensor,
    fc2_scratch: torch.Tensor,
    workspace: torch.Tensor,
    m: int,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    activation: str,
    apply_router_weight_on_input: bool,
    zero_fc2_output: bool,
    moe_block_size: int,
    max_m_blocks: int,
    element_dtype: str,
    fast_math: bool,
    sms: int,
    max_shared_mem: int,
    has_swiglu_limit: bool,
    swiglu_limit_value: float,
    swiglu_alpha: float,
    swiglu_beta: float,
    weight_layout: str,
    scale_format: str,
    w13_layout: str,
    fc1_tile_k: int,
    fc1_tile_n: int,
    fc2_tile_k: int,
    fc2_tile_n: int,
    direct_topk_routes: bool,
    tc_decode_fused_sum: bool,
    collect_activation_amax: bool,
    stream_int: int,
) -> None:
    swiglu_limit = float(swiglu_limit_value) if has_swiglu_limit else None
    collect_activation_amax = bool(collect_activation_amax)
    if collect_activation_amax and activation_amax is None:
        raise ValueError("activation_amax is required for calibrated W4A16 launch")
    activation_amax_arg = (
        activation_amax.view(-1) if activation_amax is not None else w13_global_scale
    )
    fused = compile_w4a16_fused_moe(
        size_m=size_m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=topk,
        activation=activation,
        apply_router_weight_on_input=bool(apply_router_weight_on_input),
        zero_fc2_output=bool(zero_fc2_output),
        moe_block_size=moe_block_size,
        max_m_blocks=max_m_blocks,
        element_dtype=element_dtype,
        fast_math=bool(fast_math),
        sms=sms,
        max_shared_mem=max_shared_mem,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        weight_layout=weight_layout,
        scale_format=scale_format,
        w13_layout=w13_layout,
        direct_topk_routes=bool(direct_topk_routes),
        tc_decode_fused_sum=bool(tc_decode_fused_sum),
        collect_activation_amax=collect_activation_amax,
        # The custom-op boundary cannot carry the compiled launch object. Re-pin
        # its selected geometry so tile-specific packs (notably NF3) resolve the
        # identical cache entry instead of silently recompiling with auto tiles.
        force_tile_config=(fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n),
    )
    fused.compiled(
        make_ptr(
            _cutlass_element_dtype(element_dtype),
            a_input.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        w13_arg,
        w2_arg,
        fc1_out,
        activated,
        fc2_out,
        w13_scale_i32,
        w2_scale_i32,
        w13_global_scale,
        w2_global_scale,
        packed_route_indices,
        block_expert_ids,
        packed_route_count,
        activation_amax_arg,
        int(layer_idx),
        make_ptr(
            cutlass.Float32,
            topk_weights.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        fc1_scratch,
        fc2_scratch,
        workspace,
        m,
        _w4a16_fused_persistent_grid_x(
            fused=fused,
            m=m,
            topk=topk,
            intermediate_size=intermediate_size,
            activation=activation,
            direct_topk_routes=bool(direct_topk_routes),
            sms=sms,
        ),
        cuda.CUstream(stream_int),
    )


def _w4a16_fused_persistent_grid_x(
    *,
    fused: W4A16FusedMoeCompileResult,
    m: int,
    topk: int,
    intermediate_size: int,
    activation: str,
    direct_topk_routes: bool,
    sms: int,
) -> int:
    """Right-size the persistent grid for the fused FC1+FC2 launch.

    The default over-subscribes the cooperative grid (sms*blocks_per_sm CTAs)
    and forces the larger GEMM into the cross-CTA split-K tail (lock-serialized
    finalize + c_tmp gmem round-trip) whenever its mn-tile count exceeds the
    grid. For the small-M direct-topk decode the host knows the exact FC1
    mn-tile count, so pick the fewest full persistent waves that fit the
    co-residency cap and set grid_x to that wave's tile count. Then every CTA
    owns whole FC1 (and FC2, whose tile count is an integer multiple) mn-tiles
    over the full K -- no split-K reduction, no lock traffic -- with fewer
    grid-barrier participants, while staying <= the cap so the cooperative
    barrier never deadlocks. Falls back to the default for the route-pack path
    where the host cannot know route_blocks ahead of the launch.
    """
    cap = int(sms) * int(fused.blocks_per_sm)
    if not direct_topk_routes or m <= 0:
        return max(cap, 1)
    is_gated = is_gated_moe_activation(activation)
    fc1_cols = int(intermediate_size) * (2 if is_gated else 1)
    fc1_tile_n = int(getattr(fused, "fc1_tile_n", 0))
    if fc1_tile_n <= 0 or fc1_cols % fc1_tile_n != 0:
        return max(cap, 1)
    n_tiles = fc1_cols // fc1_tile_n
    route_blocks = int(m) * int(topk)
    fc1_mn_tiles = route_blocks * n_tiles
    if fc1_mn_tiles <= 0 or cap <= 0:
        return max(cap, 1)
    if bool(getattr(fused, "schedule_whole_tiles", False)):
        # Whole-tile scheduling tolerates ragged waves, so the whole-cover
        # constraint below does not apply. Minimize the FC1+FC2 critical path
        # in whole-tile waves per CTA; ties go to the smaller grid (fewer
        # grid-barrier participants). Measured on the GLM-5.2 TP4 hybrid
        # shard (m=4: FC2 768 tiles), cap=188 CTAs runs FC2 in 5-deep waves
        # vs 6-deep at the FC1-right-sized 128 CTAs: 95.9us vs 107.1us.
        fc2_tile_n = int(getattr(fused, "fc2_tile_n", 0))
        hidden = int(getattr(fused, "hidden_size", 0))
        if fc2_tile_n <= 0 or hidden <= 0 or hidden % fc2_tile_n != 0:
            return max(cap, 1)
        fc2_mn_tiles = route_blocks * (hidden // fc2_tile_n)

        def whole_tile_critical_path(grid: int) -> int:
            return -(-fc1_mn_tiles // grid) + -(-fc2_mn_tiles // grid)

        candidates = sorted({int(cap), min(fc1_mn_tiles, int(cap))})
        return min(candidates, key=lambda g: (whole_tile_critical_path(g), g))
    waves = (fc1_mn_tiles + cap - 1) // cap
    if waves <= 0:
        return max(cap, 1)
    # Only commit to the right-sized grid when every wave is a whole tile cover
    # (no remainder => no split-K tail); otherwise keep the safe default.
    if fc1_mn_tiles % waves != 0:
        return max(cap, 1)
    grid_x = fc1_mn_tiles // waves
    if grid_x < 1 or grid_x > cap:
        return max(cap, 1)
    return grid_x


@torch.library.custom_op(
    "flashinfer_sm12x::w4a16_fused_moe_launch",
    mutates_args="unknown",
)
def _w4a16_fused_moe_launch_op(
    a_input: torch.Tensor,
    w13_arg: torch.Tensor,
    w2_arg: torch.Tensor,
    fc1_out: torch.Tensor,
    activated: torch.Tensor,
    fc2_out: torch.Tensor,
    w13_scale_i32: torch.Tensor,
    w2_scale_i32: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    packed_route_indices: torch.Tensor,
    block_expert_ids: torch.Tensor,
    packed_route_count: torch.Tensor,
    topk_weights: torch.Tensor,
    fc1_scratch: torch.Tensor,
    fc2_scratch: torch.Tensor,
    workspace: torch.Tensor,
    m: int,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    activation: str,
    apply_router_weight_on_input: bool,
    zero_fc2_output: bool,
    moe_block_size: int,
    max_m_blocks: int,
    element_dtype: str,
    fast_math: bool,
    sms: int,
    max_shared_mem: int,
    has_swiglu_limit: bool,
    swiglu_limit_value: float,
    swiglu_alpha: float,
    swiglu_beta: float,
    weight_layout: str,
    scale_format: str,
    w13_layout: str,
    fc1_tile_k: int,
    fc1_tile_n: int,
    fc2_tile_k: int,
    fc2_tile_n: int,
    direct_topk_routes: bool,
    tc_decode_fused_sum: bool,
    stream_int: int,
) -> None:
    _w4a16_fused_moe_launch_flat(
        a_input=a_input,
        w13_arg=w13_arg,
        w2_arg=w2_arg,
        fc1_out=fc1_out,
        activated=activated,
        fc2_out=fc2_out,
        w13_scale_i32=w13_scale_i32,
        w2_scale_i32=w2_scale_i32,
        w13_global_scale=w13_global_scale,
        w2_global_scale=w2_global_scale,
        packed_route_indices=packed_route_indices,
        block_expert_ids=block_expert_ids,
        packed_route_count=packed_route_count,
        activation_amax=None,
        layer_idx=0,
        topk_weights=topk_weights,
        fc1_scratch=fc1_scratch,
        fc2_scratch=fc2_scratch,
        workspace=workspace,
        m=m,
        size_m=size_m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=topk,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        zero_fc2_output=zero_fc2_output,
        moe_block_size=moe_block_size,
        max_m_blocks=max_m_blocks,
        element_dtype=element_dtype,
        fast_math=fast_math,
        sms=sms,
        max_shared_mem=max_shared_mem,
        has_swiglu_limit=has_swiglu_limit,
        swiglu_limit_value=swiglu_limit_value,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        weight_layout=weight_layout,
        scale_format=scale_format,
        w13_layout=w13_layout,
        fc1_tile_k=fc1_tile_k,
        fc1_tile_n=fc1_tile_n,
        fc2_tile_k=fc2_tile_k,
        fc2_tile_n=fc2_tile_n,
        direct_topk_routes=direct_topk_routes,
        tc_decode_fused_sum=tc_decode_fused_sum,
        collect_activation_amax=False,
        stream_int=stream_int,
    )


@_w4a16_fused_moe_launch_op.register_fake
def _w4a16_fused_moe_launch_fake(
    a_input: torch.Tensor,
    w13_arg: torch.Tensor,
    w2_arg: torch.Tensor,
    fc1_out: torch.Tensor,
    activated: torch.Tensor,
    fc2_out: torch.Tensor,
    w13_scale_i32: torch.Tensor,
    w2_scale_i32: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    packed_route_indices: torch.Tensor,
    block_expert_ids: torch.Tensor,
    packed_route_count: torch.Tensor,
    topk_weights: torch.Tensor,
    fc1_scratch: torch.Tensor,
    fc2_scratch: torch.Tensor,
    workspace: torch.Tensor,
    m: int,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    activation: str,
    apply_router_weight_on_input: bool,
    zero_fc2_output: bool,
    moe_block_size: int,
    max_m_blocks: int,
    element_dtype: str,
    fast_math: bool,
    sms: int,
    max_shared_mem: int,
    has_swiglu_limit: bool,
    swiglu_limit_value: float,
    swiglu_alpha: float,
    swiglu_beta: float,
    weight_layout: str,
    scale_format: str,
    w13_layout: str,
    fc1_tile_k: int,
    fc1_tile_n: int,
    fc2_tile_k: int,
    fc2_tile_n: int,
    direct_topk_routes: bool,
    tc_decode_fused_sum: bool,
    stream_int: int,
) -> None:
    return None


@torch.library.custom_op(
    "flashinfer_sm12x::w4a16_fused_moe_calibrated_launch",
    mutates_args="unknown",
)
def _w4a16_fused_moe_calibrated_launch_op(
    a_input: torch.Tensor,
    w13_arg: torch.Tensor,
    w2_arg: torch.Tensor,
    fc1_out: torch.Tensor,
    activated: torch.Tensor,
    fc2_out: torch.Tensor,
    w13_scale_i32: torch.Tensor,
    w2_scale_i32: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    packed_route_indices: torch.Tensor,
    block_expert_ids: torch.Tensor,
    packed_route_count: torch.Tensor,
    activation_amax: torch.Tensor,
    layer_idx: int,
    topk_weights: torch.Tensor,
    fc1_scratch: torch.Tensor,
    fc2_scratch: torch.Tensor,
    workspace: torch.Tensor,
    m: int,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    activation: str,
    apply_router_weight_on_input: bool,
    zero_fc2_output: bool,
    moe_block_size: int,
    max_m_blocks: int,
    element_dtype: str,
    fast_math: bool,
    sms: int,
    max_shared_mem: int,
    has_swiglu_limit: bool,
    swiglu_limit_value: float,
    swiglu_alpha: float,
    swiglu_beta: float,
    weight_layout: str,
    scale_format: str,
    w13_layout: str,
    fc1_tile_k: int,
    fc1_tile_n: int,
    fc2_tile_k: int,
    fc2_tile_n: int,
    stream_int: int,
) -> None:
    _w4a16_fused_moe_launch_flat(
        a_input=a_input,
        w13_arg=w13_arg,
        w2_arg=w2_arg,
        fc1_out=fc1_out,
        activated=activated,
        fc2_out=fc2_out,
        w13_scale_i32=w13_scale_i32,
        w2_scale_i32=w2_scale_i32,
        w13_global_scale=w13_global_scale,
        w2_global_scale=w2_global_scale,
        packed_route_indices=packed_route_indices,
        block_expert_ids=block_expert_ids,
        packed_route_count=packed_route_count,
        activation_amax=activation_amax,
        layer_idx=layer_idx,
        topk_weights=topk_weights,
        fc1_scratch=fc1_scratch,
        fc2_scratch=fc2_scratch,
        workspace=workspace,
        m=m,
        size_m=size_m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=topk,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        zero_fc2_output=zero_fc2_output,
        moe_block_size=moe_block_size,
        max_m_blocks=max_m_blocks,
        element_dtype=element_dtype,
        fast_math=fast_math,
        sms=sms,
        max_shared_mem=max_shared_mem,
        has_swiglu_limit=has_swiglu_limit,
        swiglu_limit_value=swiglu_limit_value,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        weight_layout=weight_layout,
        scale_format=scale_format,
        w13_layout=w13_layout,
        fc1_tile_k=fc1_tile_k,
        fc1_tile_n=fc1_tile_n,
        fc2_tile_k=fc2_tile_k,
        fc2_tile_n=fc2_tile_n,
        direct_topk_routes=False,
        tc_decode_fused_sum=False,
        collect_activation_amax=True,
        stream_int=stream_int,
    )


@_w4a16_fused_moe_calibrated_launch_op.register_fake
def _w4a16_fused_moe_calibrated_launch_fake(
    a_input: torch.Tensor,
    w13_arg: torch.Tensor,
    w2_arg: torch.Tensor,
    fc1_out: torch.Tensor,
    activated: torch.Tensor,
    fc2_out: torch.Tensor,
    w13_scale_i32: torch.Tensor,
    w2_scale_i32: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    packed_route_indices: torch.Tensor,
    block_expert_ids: torch.Tensor,
    packed_route_count: torch.Tensor,
    activation_amax: torch.Tensor,
    layer_idx: int,
    topk_weights: torch.Tensor,
    fc1_scratch: torch.Tensor,
    fc2_scratch: torch.Tensor,
    workspace: torch.Tensor,
    m: int,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    activation: str,
    apply_router_weight_on_input: bool,
    zero_fc2_output: bool,
    moe_block_size: int,
    max_m_blocks: int,
    element_dtype: str,
    fast_math: bool,
    sms: int,
    max_shared_mem: int,
    has_swiglu_limit: bool,
    swiglu_limit_value: float,
    swiglu_alpha: float,
    swiglu_beta: float,
    weight_layout: str,
    scale_format: str,
    w13_layout: str,
    fc1_tile_k: int,
    fc1_tile_n: int,
    fc2_tile_k: int,
    fc2_tile_n: int,
    stream_int: int,
) -> None:
    return None


def _w4a16_topk_sum_launch_flat(
    fc2_out: torch.Tensor,
    output: torch.Tensor,
    m: int,
    topk: int,
    hidden_size: int,
    element_dtype: str,
    stream_int: int,
) -> None:
    sum_kernel = compile_w4a16_topk_sum(
        m=m,
        topk=topk,
        hidden_size=hidden_size,
        element_dtype=element_dtype,
    )
    sum_kernel.compiled(
        make_ptr(
            _cutlass_element_dtype(element_dtype),
            fc2_out.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            _cutlass_element_dtype(element_dtype),
            output.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        m,
        cuda.CUstream(stream_int),
    )


@torch.library.custom_op(
    "flashinfer_sm12x::w4a16_topk_sum_launch",
    mutates_args=("output",),
)
def _w4a16_topk_sum_launch_op(
    fc2_out: torch.Tensor,
    output: torch.Tensor,
    m: int,
    topk: int,
    hidden_size: int,
    element_dtype: str,
    stream_int: int,
) -> None:
    _w4a16_topk_sum_launch_flat(
        fc2_out=fc2_out,
        output=output,
        m=m,
        topk=topk,
        hidden_size=hidden_size,
        element_dtype=element_dtype,
        stream_int=stream_int,
    )


@_w4a16_topk_sum_launch_op.register_fake
def _w4a16_topk_sum_launch_fake(
    fc2_out: torch.Tensor,
    output: torch.Tensor,
    m: int,
    topk: int,
    hidden_size: int,
    element_dtype: str,
    stream_int: int,
) -> None:
    return None


def _get_c_tmp(
    elements: int,
    *,
    device: torch.device,
    scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    if scratch is not None:
        if scratch.dtype != torch.float32:
            raise TypeError("W4A16 c_tmp scratch buffers must be torch.float32")
        if scratch.device != device:
            raise ValueError(f"W4A16 c_tmp scratch buffers must be on {device}")
        if not scratch.is_contiguous():
            raise ValueError("W4A16 c_tmp scratch buffers must be contiguous")
        if int(scratch.numel()) >= int(elements):
            return scratch[: int(elements)]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "W4A16 GEMM scratch is not initialized for CUDA graph capture; "
            "provide a preallocated fc*_c_tmp workspace with sufficient capacity"
        )
    return torch.empty((elements,), dtype=torch.float32, device=device)


def _validate_topk_ids(
    topk_ids: torch.Tensor,
    *,
    require_cuda: bool,
    require_contiguous: bool = True,
) -> None:
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must be torch.int32 or torch.int64")
    if require_cuda and not topk_ids.is_cuda:
        raise ValueError("topk_ids must be a CUDA tensor")
    if require_contiguous and not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")


def _validate_expert_map(
    expert_map: torch.Tensor | None,
    *,
    device: torch.device | None = None,
    exact_num_experts: int | None = None,
) -> None:
    if expert_map is None:
        return
    if expert_map.dtype != torch.int32:
        raise TypeError("expert_map must be torch.int32")
    if device is None:
        if not expert_map.is_cuda:
            raise ValueError("expert_map must be a CUDA tensor")
    elif expert_map.device != device:
        raise ValueError("expert_map must be on the same device as a_input")
    if exact_num_experts is None:
        if expert_map.ndim != 1 or not expert_map.is_contiguous():
            raise ValueError("expert_map must be a contiguous rank-1 tensor")
        return
    if not expert_map.is_contiguous():
        raise ValueError("expert_map must be contiguous")
    if expert_map.ndim != 1 or int(expert_map.numel()) != int(exact_num_experts):
        raise ValueError(
            f"expert_map must have shape {(int(exact_num_experts),)}, got {tuple(expert_map.shape)}"
        )


def _validate_activation_amax(
    activation_amax: torch.Tensor | None,
    *,
    layer_idx: int | None,
    num_experts: int,
    device: torch.device,
) -> int | None:
    if activation_amax is None:
        if layer_idx is not None:
            raise ValueError("layer_idx requires activation_amax")
        return None
    if layer_idx is None:
        raise ValueError("layer_idx is required when activation_amax is provided")
    if activation_amax.dtype != torch.float32:
        raise TypeError("activation_amax must be torch.float32")
    if activation_amax.device != device:
        raise ValueError("activation_amax must be on the same device as a_input")
    if not activation_amax.is_cuda:
        raise ValueError("activation_amax must be a CUDA tensor")
    if not activation_amax.is_contiguous():
        raise ValueError("activation_amax must be contiguous")
    if activation_amax.ndim != 3 or int(activation_amax.shape[2]) != 2:
        raise ValueError("activation_amax must have shape [num_layers, num_experts, 2]")
    if int(activation_amax.shape[1]) < int(num_experts):
        raise ValueError(
            "activation_amax expert dimension is smaller than the local expert count"
        )
    layer = int(layer_idx)
    if layer < 0 or layer >= int(activation_amax.shape[0]):
        raise ValueError(
            f"layer_idx {layer} is out of bounds for activation_amax with "
            f"{int(activation_amax.shape[0])} layers"
        )
    return layer


def _compile_w4a16_gemm_launch(
    *,
    size_m: int,
    size_n: int,
    size_k: int,
    num_experts: int,
    top_k: int,
    mul_topk_weights: bool,
    moe_block_size: int,
    max_m_blocks: int,
    element_dtype: str,
    packed_route_indices: torch.Tensor,
    sms: int,
    max_shared_mem: int,
    device: torch.device,
    c_tmp: torch.Tensor | None = None,
    scale_format: str = "e4m3_k16",
) -> _W4A16GemmLaunch:
    tile_k, tile_n, _, _ = _select_tile_config(
        problem_m=size_m,
        problem_n=size_n,
        problem_k=size_k,
        top_k=top_k,
        moe_block_size=moe_block_size,
        sms=sms,
        max_shared_mem=max_shared_mem,
        scale_format=scale_format,
    )
    kernel = compile_w4a16_gemm(
        size_m=size_m,
        size_n=size_n,
        size_k=size_k,
        num_experts=num_experts,
        top_k=top_k,
        mul_topk_weights=mul_topk_weights,
        tile_n=tile_n,
        tile_k=tile_k,
        moe_block_size=moe_block_size,
        max_m_blocks=max_m_blocks,
        element_dtype=element_dtype,
        scale_format=scale_format,
    )
    c_tmp = _get_c_tmp(
        packed_gemm_scratch_elements(
            size_n=size_n,
            route_slots=int(packed_route_indices.numel()),
            moe_block_size=moe_block_size,
            sms=sms,
        ),
        device=device,
        scratch=c_tmp,
    )
    return _W4A16GemmLaunch(kernel=kernel, c_tmp=c_tmp)


def pack_topk_routes_by_expert(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    *,
    expert_map: torch.Tensor | None = None,
    packed_route_indices: torch.Tensor | None = None,
    block_expert_ids: torch.Tensor | None = None,
    packed_route_count: torch.Tensor | None = None,
    expert_offsets: torch.Tensor | None = None,
    stream: cuda.CUstream | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group top-k routes by expert and pad each group to the GEMM M-block size."""
    _validate_topk_ids(topk_ids, require_cuda=True)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    _validate_expert_map(expert_map, exact_num_experts=int(num_experts))
    del stream
    return _pack_topk_routes_by_expert(
        topk_ids,
        int(block_size),
        int(num_experts),
        expert_map=expert_map,
        packed_route_indices=packed_route_indices,
        block_expert_ids=block_expert_ids,
        packed_route_count=packed_route_count,
        expert_offsets=expert_offsets,
    )


def run_w4a16_moe(
    a_input: torch.Tensor,
    prepared,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str,
    intermediate_cache13: torch.Tensor,
    intermediate_cache2: torch.Tensor,
    output: torch.Tensor,
    fc1_c_tmp: torch.Tensor | None = None,
    fc2_c_tmp: torch.Tensor | None = None,
    packed_route_indices: torch.Tensor | None = None,
    block_expert_ids: torch.Tensor | None = None,
    packed_route_count: torch.Tensor | None = None,
    expert_offsets: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
    activation_amax: torch.Tensor | None = None,
    layer_idx: int | None = None,
    apply_router_weight_on_input: bool = False,
    fast_math: bool = True,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    fused_launch: W4A16FusedMoeCompileResult | None = None,
    topk_sum_launch: W4A16TopKSumCompileResult | None = None,
    stream: cuda.CUstream | None = None,
) -> torch.Tensor:
    activation = normalize_moe_activation(activation)
    is_gated = validate_activation(activation)
    swiglu_limit, swiglu_alpha, swiglu_beta = _normalize_activation_swiglu_params(
        activation,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
    )
    element_dtype = _normalize_element_dtype(a_input.dtype)
    if output.dtype != a_input.dtype:
        raise TypeError(f"output must have dtype {a_input.dtype}, got {output.dtype}")
    prepared_dtype = getattr(prepared, "params_dtype", a_input.dtype)
    if prepared_dtype != a_input.dtype:
        raise TypeError(
            f"prepared weights were built for {prepared_dtype}, but a_input has dtype {a_input.dtype}"
        )
    weight_layout = getattr(prepared, "weight_layout", "packed")
    if weight_layout not in _WEIGHT_LAYOUTS:
        raise ValueError(f"unsupported W4A16 weight_layout {weight_layout!r}")
    scale_format = _normalize_scale_format(
        getattr(prepared, "scale_format", None)
        or (
            "e8m0_k32"
            if getattr(prepared, "source_format", "") == "fp4_e8m0_k32"
            else "e4m3_k16"
        )
    )
    w13_layout = getattr(
        prepared,
        "w13_layout",
        "w13" if weight_layout == "modelopt" else "packed",
    )
    if weight_layout == "modelopt":
        if w13_layout not in _MODEL_OPT_W13_LAYOUTS:
            raise ValueError(f"unsupported W4A16 w13_layout {w13_layout!r}")
    else:
        w13_layout = "packed"
    if topk_weights.dtype != torch.float32:
        raise TypeError("topk_weights must be torch.float32")
    _validate_topk_ids(topk_ids, require_cuda=False, require_contiguous=False)
    if (
        not a_input.is_contiguous()
        or not topk_weights.is_contiguous()
        or not topk_ids.is_contiguous()
    ):
        raise ValueError("a_input, topk_weights, and topk_ids must be contiguous")
    _validate_expert_map(expert_map, device=a_input.device)

    m, hidden_size = a_input.shape
    topk = int(topk_ids.shape[1])
    if tuple(topk_weights.shape) != (m, topk):
        raise ValueError(f"topk_weights must have shape {(m, topk)}")
    if int(prepared.hidden_size) != hidden_size:
        raise ValueError("prepared hidden_size does not match a_input")
    if bool(prepared.is_gated) != is_gated:
        raise ValueError("prepared weights do not match activation")
    if tuple(output.shape) != (m, hidden_size):
        raise ValueError(f"output must have shape {(m, hidden_size)}")
    if expert_map is not None and int(expert_map.numel()) < int(prepared.num_experts):
        raise ValueError("expert_map cannot be shorter than the local expert count")
    layer_idx_int = _validate_activation_amax(
        activation_amax,
        layer_idx=layer_idx,
        num_experts=int(prepared.num_experts),
        device=a_input.device,
    )
    collect_activation_amax = activation_amax is not None

    route_num_experts = (
        int(expert_map.numel()) if expert_map is not None else int(prepared.num_experts)
    )
    if fused_launch is None:
        block_size_m = select_route_block_size_m(m, topk, route_num_experts)
    else:
        block_size_m = int(fused_launch.moe_block_size)
    if block_size_m not in _ALLOWED_ROUTED_SIZES:
        raise ValueError(f"unsupported W4A16 moe_block_size={block_size_m}")

    stream = current_cuda_stream() if stream is None else stream
    if (not collect_activation_amax) and _small_m_direct_supported(
        m=m,
        hidden_size=hidden_size,
        intermediate_size=int(prepared.intermediate_size),
        num_experts=int(prepared.num_experts),
        topk=topk,
        activation=activation,
        apply_router_weight_on_input=bool(apply_router_weight_on_input),
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        element_dtype=element_dtype,
        weight_layout=weight_layout,
        w13_layout=w13_layout,
        scale_format=scale_format,
        expert_map=expert_map,
    ):
        if topk_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("W4A16 small-M direct path requires int32/int64 topk_ids")
        if not topk_ids.is_cuda:
            raise ValueError("W4A16 small-M direct path requires CUDA topk_ids")
        if not intermediate_cache2.is_contiguous() or not output.is_contiguous():
            raise ValueError(
                "W4A16 small-M direct path requires contiguous intermediate_cache2 and output"
            )
        if intermediate_cache2.dtype != a_input.dtype:
            raise TypeError(f"intermediate_cache2 must be {a_input.dtype}")
        if int(prepared.workspace.numel()) < 2:
            raise ValueError("prepared W4A16 workspace is too small for small-M direct")
        intermediate_size = int(prepared.intermediate_size)
        fc2_n_chunks = ((intermediate_size // 2) + 127) // 128
        inter_u32_per_m = fc2_n_chunks * 128 * topk
        inter_u32 = intermediate_cache2.view(-1).view(torch.uint32)
        if int(inter_u32.numel()) < m * inter_u32_per_m:
            raise ValueError(
                "intermediate_cache2 is smaller than the W4A16 small-M direct scratch "
                f"requirement: have_u32={int(inter_u32.numel())}, "
                f"need_u32={m * inter_u32_per_m}"
            )
        micro_w13_scale = getattr(prepared, "micro_w13_scale", None)
        micro_w2_scale = getattr(prepared, "micro_w2_scale", None)
        micro_w13_global = getattr(prepared, "micro_w13_global_scale", None)
        micro_w2_global = getattr(prepared, "micro_w2_global_scale", None)
        if (
            micro_w13_scale is None
            or micro_w2_scale is None
            or micro_w13_global is None
            or micro_w2_global is None
        ):
            raise RuntimeError(
                "W4A16 small-M direct path requires prepared micro scale metadata"
            )
        barrier_count = prepared.workspace[-2:-1]
        barrier_epoch = prepared.workspace[-1:]
        barrier_count.zero_()
        barrier_epoch.zero_()
        torch.ops.flashinfer_sm12x.w4a16_small_m_direct_launch(
            a_input,
            prepared.w13.view(torch.uint8),
            micro_w13_scale,
            micro_w13_global,
            micro_w2_global,
            inter_u32[: m * inter_u32_per_m],
            prepared.w2.view(torch.uint8),
            micro_w2_scale,
            topk_ids,
            topk_weights,
            output,
            barrier_count,
            barrier_epoch,
            m,
            hidden_size,
            intermediate_size,
            int(prepared.num_experts),
            topk,
            activation,
            bool(fast_math),
            scale_format,
            swiglu_limit is not None,
            float(swiglu_limit or 0.0),
            float(swiglu_alpha),
            float(swiglu_beta),
            w13_layout,
            int(stream),
        )
        return output

    # TC-decode: small-M packed decode that folds the top-k sum into the FC2
    # store epilogue. Reuses the packed tensor-core MMA path; only the launch
    # scheduling/epilogue changes. Requires the packed object, bf16 gated
    # activation, int32 routes, no expert_map, and a runtime-compiled launch.
    # A preplanned launch built with the TC-decode fused-sum epilogue carries
    # ``tc_decode_fused_sum``; accept it through the binding path. A runtime
    # ``fused_launch is None`` (e.g. the standalone benchmark) compiles its own.
    preplanned_tc_decode = bool(getattr(fused_launch, "tc_decode_fused_sum", False))
    use_tc_decode = bool(
        (not collect_activation_amax)
        and (fused_launch is None or preplanned_tc_decode)
        and weight_layout in ("packed", "nf3_2p1")
        and expert_map is None
        and is_gated
        and element_dtype == "bf16"
        and topk_ids.dtype in (torch.int32, torch.int64)
        and topk_ids.is_cuda
        and int(m) <= _TC_DECODE_MAX_M
    )
    if use_tc_decode and topk_ids.dtype != torch.int32:
        # The inline direct-topk route path needs int32 route indices.
        topk_ids = topk_ids.to(torch.int32)

    direct_topk_eligible = (
        (not collect_activation_amax)
        and (m <= _MAX_DIRECT_TOPK_ROUTE_M or use_tc_decode)
        and weight_layout in ("packed", "nf3_2p1")
        and expert_map is None
    )
    use_direct_topk_routes = bool(
        direct_topk_eligible
        and topk_ids.dtype == torch.int32
        and (
            fused_launch is None
            or bool(getattr(fused_launch, "direct_topk_routes", False))
        )
    )
    if (
        bool(getattr(fused_launch, "direct_topk_routes", False))
        and not use_direct_topk_routes
    ):
        raise RuntimeError(
            "preplanned W4A16 direct top-k routing requires small-M packed "
            "int32 topk_ids without expert_map"
        )

    # TC-decode requires the inline direct-topk route path (no route-pack).
    use_tc_decode = bool(use_tc_decode and use_direct_topk_routes)

    # A preplanned TC-decode launch atomically accumulates FC2 partials into the
    # (pre-zeroed) output and emits no separate top-k sum. If it was selected but
    # the decode preconditions don't hold, running it would corrupt the output,
    # so fail loudly instead.
    if preplanned_tc_decode and not use_tc_decode:
        raise RuntimeError(
            "preplanned TC-decode W4A16 launch requires small-M packed bf16 "
            f"decode (m <= {_TC_DECODE_MAX_M}, cuda int32/int64 topk_ids, "
            "no expert_map)"
        )

    route_slots_for_scratch = int(m) * int(topk) * int(block_size_m)
    required_m_blocks = int(m) * int(topk) if use_direct_topk_routes else 0
    if fused_launch is not None and not use_direct_topk_routes:
        route_slots_capacity = max_packed_route_slots(
            int(fused_launch.size_m) * int(topk),
            int(block_size_m),
            route_num_experts,
        )
        route_blocks_capacity = (route_slots_capacity + int(block_size_m) - 1) // int(
            block_size_m
        )
        if packed_route_indices is not None:
            if int(packed_route_indices.numel()) < route_slots_capacity:
                raise ValueError(
                    "packed_route_indices is smaller than the selected W4A16 launch capacity"
                )
            packed_route_indices = packed_route_indices[:route_slots_capacity]
        if block_expert_ids is not None:
            if int(block_expert_ids.numel()) < route_blocks_capacity:
                raise ValueError(
                    "block_expert_ids is smaller than the selected W4A16 launch capacity"
                )
            block_expert_ids = block_expert_ids[:route_blocks_capacity]
    if use_direct_topk_routes:
        packed_route_indices = topk_ids.view(-1)
        if block_expert_ids is None:
            block_expert_ids = packed_route_indices
        if packed_route_count is None:
            packed_route_count = packed_route_indices
    else:
        packed_route_indices, block_expert_ids, packed_route_count = (
            pack_topk_routes_by_expert(
                topk_ids,
                block_size_m,
                route_num_experts,
                expert_map=expert_map,
                packed_route_indices=packed_route_indices,
                block_expert_ids=block_expert_ids,
                packed_route_count=packed_route_count,
                expert_offsets=expert_offsets,
                stream=stream,
            )
        )
        route_slots_for_scratch = int(packed_route_indices.numel())
        required_m_blocks = int(block_expert_ids.numel())

    props = torch.cuda.get_device_properties(a_input.device)
    sms = int(props.multi_processor_count)
    max_shared_mem = int(
        getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
    )
    buffer_plan = plan_w4a16_buffers(
        prepared,
        m=m,
        topk=topk,
        route_num_experts=route_num_experts,
        sms=sms,
    )
    intermediate_size = int(prepared.intermediate_size)
    fc1_cols = buffer_plan.fc1_cols

    if intermediate_cache13.numel() < buffer_plan.intermediate_cache13_elements:
        raise ValueError(
            f"intermediate_cache13 has {intermediate_cache13.numel()} elements; "
            f"need at least {buffer_plan.intermediate_cache13_elements}"
        )
    if intermediate_cache2.numel() < buffer_plan.intermediate_cache2_elements:
        raise ValueError(
            f"intermediate_cache2 has {intermediate_cache2.numel()} elements; "
            f"need at least {buffer_plan.intermediate_cache2_elements}"
        )
    if (
        intermediate_cache13.dtype != a_input.dtype
        or intermediate_cache2.dtype != a_input.dtype
    ):
        raise TypeError(f"intermediate caches must be {a_input.dtype}")
    if (
        not intermediate_cache13.is_contiguous()
        or not intermediate_cache2.is_contiguous()
        or not output.is_contiguous()
    ):
        raise ValueError("intermediate caches and output must be contiguous")

    intermediate_cache13_flat = intermediate_cache13.view(-1)
    intermediate_cache2_flat = intermediate_cache2.view(-1)

    if int(prepared.workspace.numel()) < sms * 4 + 2:
        raise ValueError("prepared W4A16 workspace is too small for fused FC1+FC2")
    if fused_launch is None:
        fused = compile_w4a16_fused_moe(
            size_m=m,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=int(prepared.num_experts),
            top_k=topk,
            activation=activation,
            apply_router_weight_on_input=bool(apply_router_weight_on_input),
            zero_fc2_output=expert_map is not None,
            moe_block_size=block_size_m,
            max_m_blocks=int(required_m_blocks),
            element_dtype=element_dtype,
            fast_math=bool(fast_math),
            sms=sms,
            max_shared_mem=max_shared_mem,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            weight_layout=weight_layout,
            scale_format=scale_format,
            w13_layout=w13_layout,
            direct_topk_routes=use_direct_topk_routes,
            tc_decode_fused_sum=use_tc_decode,
            collect_activation_amax=collect_activation_amax,
        )
    else:
        if int(fused_launch.size_m) < m:
            raise RuntimeError(
                "preplanned W4A16 fused MoE launch capacity is smaller than requested rows: "
                f"requested={m}, planned={int(fused_launch.size_m)}"
            )
        expected_fused = (
            hidden_size,
            intermediate_size,
            int(prepared.num_experts),
            topk,
            activation,
            bool(apply_router_weight_on_input),
            expert_map is not None,
            element_dtype,
            bool(fast_math),
            swiglu_limit,
            float(swiglu_alpha),
            float(swiglu_beta),
            weight_layout,
            scale_format,
            w13_layout,
            bool(use_direct_topk_routes),
            bool(collect_activation_amax),
            block_size_m,
        )
        actual_fused = (
            int(fused_launch.hidden_size),
            int(fused_launch.intermediate_size),
            int(fused_launch.num_experts),
            int(fused_launch.top_k),
            fused_launch.activation,
            bool(fused_launch.apply_router_weight_on_input),
            bool(fused_launch.zero_fc2_output),
            fused_launch.element_dtype,
            bool(fused_launch.fast_math),
            fused_launch.swiglu_limit,
            float(fused_launch.swiglu_alpha),
            float(fused_launch.swiglu_beta),
            getattr(fused_launch, "weight_layout", "packed"),
            getattr(fused_launch, "scale_format", "e4m3_k16"),
            getattr(
                fused_launch,
                "w13_layout",
                "w13"
                if getattr(fused_launch, "weight_layout", "packed") == "modelopt"
                else "packed",
            ),
            bool(getattr(fused_launch, "direct_topk_routes", False)),
            bool(getattr(fused_launch, "collect_activation_amax", False)),
            int(fused_launch.moe_block_size),
        )
        if actual_fused != expected_fused or int(fused_launch.max_m_blocks) < int(
            required_m_blocks
        ):
            raise RuntimeError(
                "preplanned W4A16 fused MoE launch does not match requested contract: "
                f"requested={expected_fused + (int(required_m_blocks),)}, "
                f"planned={actual_fused + (int(fused_launch.max_m_blocks),)}"
            )
        fused = fused_launch
    if weight_layout == "nf3_2p1":
        prepared_tiles = (
            int(getattr(prepared, "fc1_tile_n", 0)),
            int(getattr(prepared, "fc2_tile_n", 0)),
        )
        compiled_tiles = (int(fused.fc1_tile_n), int(fused.fc2_tile_n))
        if prepared_tiles != compiled_tiles:
            raise RuntimeError(
                "prepared NF3 packing geometry does not match the compiled "
                "W4A16 launch: "
                f"prepared_fc1_fc2_tile_n={prepared_tiles}, "
                f"compiled_fc1_fc2_tile_n={compiled_tiles}"
            )
    capacity_m = int(fused.size_m)
    capacity_routed_rows = capacity_m * topk
    if intermediate_cache13_flat.numel() < capacity_routed_rows * max(
        fc1_cols, hidden_size
    ):
        raise ValueError(
            "intermediate_cache13 is smaller than the selected W4A16 launch capacity: "
            f"capacity_rows={capacity_m}, topk={topk}"
        )
    if intermediate_cache2_flat.numel() < capacity_routed_rows * intermediate_size:
        raise ValueError(
            "intermediate_cache2 is smaller than the selected W4A16 launch capacity: "
            f"capacity_rows={capacity_m}, topk={topk}"
        )
    fc1_out = intermediate_cache13_flat[: capacity_routed_rows * fc1_cols]
    activated = intermediate_cache2_flat[: capacity_routed_rows * intermediate_size]
    if use_tc_decode:
        # FC2 atomically accumulates per-route partials directly into the
        # per-token output, so the output is the FC2 store target and must be
        # pre-zeroed. The fused tc_decode kernel now zeroes the output in its
        # own prologue (before FC1, made visible by the existing post-FC1 grid
        # barrier), so the separate host-side output.zero_() launch is removed
        # from the decode critical path here. This drops the separate top-k-sum
        # launch as well.
        fc2_out = output.view(-1)
    else:
        fc2_out = intermediate_cache13_flat[: capacity_routed_rows * hidden_size]
    fc1_scratch = _get_c_tmp(
        packed_gemm_scratch_elements(
            size_n=fc1_cols,
            route_slots=int(route_slots_for_scratch),
            moe_block_size=block_size_m,
            sms=sms,
        ),
        device=a_input.device,
        scratch=fc1_c_tmp,
    )
    fc2_scratch = _get_c_tmp(
        packed_gemm_scratch_elements(
            size_n=hidden_size,
            route_slots=int(route_slots_for_scratch),
            moe_block_size=block_size_m,
            sms=sms,
        ),
        device=a_input.device,
        scratch=fc2_c_tmp,
    )
    if weight_layout == "modelopt":
        w13_arg = prepared.w13.view(torch.uint8).view(-1)
        w2_arg = prepared.w2.view(torch.uint8).view(-1)
    else:
        w13_arg = prepared.w13.view(torch.int32).view(-1)
        w2_arg = prepared.w2.view(torch.int32).view(-1)
    launch_common = (
        a_input,
        w13_arg,
        w2_arg,
        fc1_out,
        activated,
        fc2_out,
        prepared.w13_scale.view(torch.uint8).view(torch.int32).view(-1),
        prepared.w2_scale.view(torch.uint8).view(torch.int32).view(-1),
        prepared.w13_global_scale,
        prepared.w2_global_scale,
        packed_route_indices,
        block_expert_ids,
        packed_route_count,
    )
    launch_tail = (
        topk_weights,
        fc1_scratch,
        fc2_scratch,
        prepared.workspace,
        m,
        capacity_m,
        hidden_size,
        intermediate_size,
        int(prepared.num_experts),
        topk,
        activation,
        bool(apply_router_weight_on_input),
        expert_map is not None,
        block_size_m,
        int(fused.max_m_blocks),
        element_dtype,
        bool(fast_math),
        sms,
        max_shared_mem,
        swiglu_limit is not None,
        float(swiglu_limit or 0.0),
        float(swiglu_alpha),
        float(swiglu_beta),
        weight_layout,
        scale_format,
        w13_layout,
        int(fused.fc1_tile_k),
        int(fused.fc1_tile_n),
        int(fused.fc2_tile_k),
        int(fused.fc2_tile_n),
    )
    if collect_activation_amax:
        assert activation_amax is not None
        assert layer_idx_int is not None
        torch.ops.flashinfer_sm12x.w4a16_fused_moe_calibrated_launch(
            *launch_common,
            activation_amax,
            int(layer_idx_int),
            *launch_tail,
            int(stream),
        )
    else:
        torch.ops.flashinfer_sm12x.w4a16_fused_moe_launch(
            *launch_common,
            *launch_tail,
            bool(use_direct_topk_routes),
            bool(use_tc_decode),
            int(stream),
        )

    if use_tc_decode:
        # FC2 already wrote the top-k-summed result into `output`.
        return output

    if topk_sum_launch is not None:
        expected_sum = (topk, hidden_size)
        actual_sum = (
            int(topk_sum_launch.topk),
            int(topk_sum_launch.hidden_size),
        )
        if actual_sum != expected_sum:
            raise RuntimeError(
                "preplanned W4A16 top-k sum launch does not match requested contract: "
                f"requested={expected_sum}, planned={actual_sum}"
            )
    torch.ops.flashinfer_sm12x.w4a16_topk_sum_launch(
        fc2_out,
        output,
        m,
        topk,
        hidden_size,
        element_dtype,
        int(stream),
    )
    return output


def build_w4a16_tier_local_map(
    tier0_global_ids,
    tier1_global_ids,
    *,
    map_slots: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build the int32 [map_slots] global-expert descriptor table.

    Entry g = (tier << 8) | local_expert_id for a mapped global expert id g,
    -1 for unmapped. tierN_global_ids[i] is the global id of that tier's local
    expert i, i.e. the order the tier's weights were packed in.
    """

    map_slots = int(map_slots)
    table = torch.full((map_slots,), -1, dtype=torch.int32)
    seen: set[int] = set()
    for tier, ids in ((0, tier0_global_ids), (1, tier1_global_ids)):
        ids_list = [int(v) for v in ids]
        if len(ids_list) > 256:
            raise ValueError(
                f"tier {tier} has {len(ids_list)} experts; the descriptor "
                "local-id field is 8 bits"
            )
        for local, gid in enumerate(ids_list):
            if gid < 0 or gid >= map_slots:
                raise ValueError(
                    f"tier {tier} local expert {local} has global id {gid} "
                    f"outside [0, {map_slots})"
                )
            if gid in seen:
                raise ValueError(f"global expert id {gid} is mapped twice")
            seen.add(gid)
            table[gid] = (tier << 8) | local
    if device is not None:
        table = table.to(device)
    return table.contiguous()


def _w4a16_hybrid_validate_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    elements: int,
    assumed_align: int,
    exact: bool,
) -> None:
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    actual = int(tensor.numel())
    if (exact and actual != int(elements)) or (not exact and actual < int(elements)):
        relation = "exactly" if exact else "at least"
        raise ValueError(
            f"{name} must contain {relation} {int(elements)} elements, got {actual}"
        )
    address = int(tensor.data_ptr())
    if address == 0 or address % int(assumed_align) != 0:
        raise ValueError(
            f"{name} address {address:#x} violates assumed_align={int(assumed_align)}"
        )


def _w4a16_hybrid_validate_aliases(
    tensors: tuple[tuple[str, torch.Tensor, bool], ...],
) -> None:
    """Reject byte overlaps involving a mutable tensor or a route-map input."""

    protected = {"global_topk_ids", "tier_local_map"}
    ranges = tuple(
        (
            name,
            int(tensor.data_ptr()),
            int(tensor.data_ptr()) + int(tensor.numel()) * int(tensor.element_size()),
            bool(mutable),
        )
        for name, tensor, mutable in tensors
    )
    for left_idx in range(len(ranges)):
        left_name, left_start, left_stop, left_mutable = ranges[left_idx]
        for right_idx in range(left_idx + 1, len(ranges)):
            right_name, right_start, right_stop, right_mutable = ranges[right_idx]
            overlap_matters = (
                left_mutable
                or right_mutable
                or left_name in protected
                or right_name in protected
            )
            if not overlap_matters:
                continue
            if left_start < right_stop and right_start < left_stop:
                raise ValueError(
                    "hybrid W4A16 unsafe storage alias: "
                    f"{left_name} overlaps {right_name}"
                )


def _w4a16_fused_moe_hybrid_launch_flat(
    a_input: torch.Tensor,
    t0_w13: torch.Tensor,
    t0_w2: torch.Tensor,
    t0_w13_scale_i32: torch.Tensor,
    t0_w2_scale_i32: torch.Tensor,
    t0_w13_global: torch.Tensor,
    t0_w2_global: torch.Tensor,
    t1_w13: torch.Tensor,
    t1_w2: torch.Tensor,
    t1_w13_scale_i32: torch.Tensor,
    t1_w2_scale_i32: torch.Tensor,
    t1_w13_global: torch.Tensor,
    t1_w2_global: torch.Tensor,
    global_topk_ids: torch.Tensor,
    tier_local_map: torch.Tensor,
    fc1_out: torch.Tensor,
    activated: torch.Tensor,
    output_out: torch.Tensor,
    topk_weights: torch.Tensor,
    fc1_scratch: torch.Tensor,
    fc2_scratch: torch.Tensor,
    workspace: torch.Tensor,
    m: int,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    t0_num_experts: int,
    t1_num_experts: int,
    topk: int,
    activation: str,
    map_slots: int,
    moe_block_size: int,
    element_dtype: str,
    fast_math: bool,
    sms: int,
    max_shared_mem: int,
    t0_weight_layout: str,
    t0_scale_format: str,
    t0_w13_layout: str,
    t1_weight_layout: str,
    t1_scale_format: str,
    t1_w13_layout: str,
    fc1_tile_k: int,
    fc1_tile_n: int,
    fc2_tile_k: int,
    fc2_tile_n: int,
    schedule_whole_tiles: bool,
    stream_int: int,
) -> None:
    if not a_input.is_cuda:
        raise ValueError("hybrid W4A16 input must be a CUDA tensor")
    device_index = a_input.device.index
    current_device = int(torch.cuda.current_device())
    if device_index is None or int(device_index) != current_device:
        raise ValueError(
            "hybrid W4A16 input must be on the current CUDA device: "
            f"cuda:{device_index} != cuda:{current_device}"
        )
    current_stream_int = int(torch.cuda.current_stream(a_input.device).cuda_stream)
    if int(stream_int) != current_stream_int:
        raise ValueError(
            "hybrid W4A16 stream must be the current input-device stream: "
            f"{int(stream_int)} != {current_stream_int}"
        )
    if int(m) < 1 or int(m) > int(size_m):
        raise ValueError(f"hybrid W4A16 requires 1 <= m <= size_m, got m={m}")

    hybrid = compile_w4a16_fused_moe_hybrid(
        size_m=size_m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        tier0_num_experts=t0_num_experts,
        tier1_num_experts=t1_num_experts,
        top_k=topk,
        activation=activation,
        map_slots=map_slots,
        moe_block_size=moe_block_size,
        element_dtype=element_dtype,
        fast_math=bool(fast_math),
        sms=sms,
        max_shared_mem=max_shared_mem,
        tier0_weight_layout=t0_weight_layout,
        tier0_scale_format=t0_scale_format,
        tier0_w13_layout=t0_w13_layout,
        tier1_weight_layout=t1_weight_layout,
        tier1_scale_format=t1_scale_format,
        tier1_w13_layout=t1_w13_layout,
        force_tile_config=(fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n),
        schedule_whole_tiles=bool(schedule_whole_tiles),
    )

    element_torch_dtype = torch.float16 if element_dtype == "fp16" else torch.bfloat16
    fc1_cols = int(intermediate_size) * 2
    routed_rows = int(m) * int(topk)
    weight_dtypes = {
        "modelopt": torch.uint8,
    }
    tensor_contracts = (
        ("a_input", a_input, element_torch_dtype, m * hidden_size, 16, False, False),
        (
            "t0_w13",
            t0_w13,
            weight_dtypes.get(t0_weight_layout, torch.int32),
            _w4a16_weight_flat_elements(
                num_experts=t0_num_experts,
                size_n=fc1_cols,
                size_k=hidden_size,
                weight_layout=t0_weight_layout,
            ),
            16,
            True,
            False,
        ),
        (
            "t0_w2",
            t0_w2,
            weight_dtypes.get(t0_weight_layout, torch.int32),
            _w4a16_weight_flat_elements(
                num_experts=t0_num_experts,
                size_n=hidden_size,
                size_k=intermediate_size,
                weight_layout=t0_weight_layout,
            ),
            16,
            True,
            False,
        ),
        (
            "t1_w13",
            t1_w13,
            weight_dtypes.get(t1_weight_layout, torch.int32),
            _w4a16_weight_flat_elements(
                num_experts=t1_num_experts,
                size_n=fc1_cols,
                size_k=hidden_size,
                weight_layout=t1_weight_layout,
            ),
            16,
            True,
            False,
        ),
        (
            "t1_w2",
            t1_w2,
            weight_dtypes.get(t1_weight_layout, torch.int32),
            _w4a16_weight_flat_elements(
                num_experts=t1_num_experts,
                size_n=hidden_size,
                size_k=intermediate_size,
                weight_layout=t1_weight_layout,
            ),
            16,
            True,
            False,
        ),
        ("t0_w13_scale_i32", t0_w13_scale_i32, torch.int32, 1, 16, False, False),
        ("t0_w2_scale_i32", t0_w2_scale_i32, torch.int32, 1, 16, False, False),
        ("t1_w13_scale_i32", t1_w13_scale_i32, torch.int32, 1, 16, False, False),
        ("t1_w2_scale_i32", t1_w2_scale_i32, torch.int32, 1, 16, False, False),
        (
            "t0_w13_global",
            t0_w13_global,
            torch.float32,
            t0_num_experts,
            16,
            True,
            False,
        ),
        ("t0_w2_global", t0_w2_global, torch.float32, t0_num_experts, 16, True, False),
        (
            "t1_w13_global",
            t1_w13_global,
            torch.float32,
            t1_num_experts,
            16,
            True,
            False,
        ),
        ("t1_w2_global", t1_w2_global, torch.float32, t1_num_experts, 16, True, False),
        (
            "global_topk_ids",
            global_topk_ids,
            torch.int32,
            routed_rows,
            16,
            False,
            False,
        ),
        ("tier_local_map", tier_local_map, torch.int32, map_slots, 16, True, False),
        (
            "fc1_out",
            fc1_out,
            element_torch_dtype,
            routed_rows * fc1_cols,
            16,
            False,
            True,
        ),
        (
            "activated",
            activated,
            element_torch_dtype,
            routed_rows * intermediate_size,
            16,
            False,
            True,
        ),
        (
            "output_out",
            output_out,
            element_torch_dtype,
            m * hidden_size,
            16,
            False,
            True,
        ),
        ("topk_weights", topk_weights, torch.float32, routed_rows, 4, False, False),
        ("fc1_scratch", fc1_scratch, torch.float32, 1, 16, False, True),
        ("fc2_scratch", fc2_scratch, torch.float32, 1, 16, False, True),
        (
            "workspace",
            workspace,
            torch.int32,
            int(sms) * 4 + 2,
            16,
            False,
            True,
        ),
    )
    for name, tensor, dtype, elements, align, exact, _ in tensor_contracts:
        _w4a16_hybrid_validate_tensor(
            name,
            tensor,
            dtype=dtype,
            elements=elements,
            assumed_align=align,
            exact=exact,
        )
        if tensor.device != a_input.device:
            raise ValueError(
                f"{name} device {tensor.device} does not match input "
                f"device {a_input.device}"
            )
    _w4a16_hybrid_validate_aliases(
        tuple(
            (name, tensor, mutable)
            for name, tensor, _, _, _, _, mutable in tensor_contracts
        )
    )

    grid_x = _w4a16_fused_persistent_grid_x(
        fused=hybrid,
        m=m,
        topk=topk,
        intermediate_size=intermediate_size,
        activation=activation,
        direct_topk_routes=True,
        sms=sms,
    )
    hybrid.compiled(
        make_ptr(
            _cutlass_element_dtype(element_dtype),
            a_input.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        t0_w13,
        t0_w2,
        t0_w13_scale_i32,
        t0_w2_scale_i32,
        t0_w13_global,
        t0_w2_global,
        t1_w13,
        t1_w2,
        t1_w13_scale_i32,
        t1_w2_scale_i32,
        t1_w13_global,
        t1_w2_global,
        global_topk_ids,
        tier_local_map,
        fc1_out,
        activated,
        output_out,
        make_ptr(
            cutlass.Float32,
            topk_weights.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        fc1_scratch,
        fc2_scratch,
        workspace,
        m,
        grid_x,
        cuda.CUstream(stream_int),
    )


@torch.library.custom_op(
    "flashinfer_sm12x::w4a16_fused_moe_hybrid_launch",
    mutates_args="unknown",
    device_types="cuda",
)
def _w4a16_fused_moe_hybrid_launch_op(
    a_input: torch.Tensor,
    t0_w13: torch.Tensor,
    t0_w2: torch.Tensor,
    t0_w13_scale_i32: torch.Tensor,
    t0_w2_scale_i32: torch.Tensor,
    t0_w13_global: torch.Tensor,
    t0_w2_global: torch.Tensor,
    t1_w13: torch.Tensor,
    t1_w2: torch.Tensor,
    t1_w13_scale_i32: torch.Tensor,
    t1_w2_scale_i32: torch.Tensor,
    t1_w13_global: torch.Tensor,
    t1_w2_global: torch.Tensor,
    global_topk_ids: torch.Tensor,
    tier_local_map: torch.Tensor,
    fc1_out: torch.Tensor,
    activated: torch.Tensor,
    output_out: torch.Tensor,
    topk_weights: torch.Tensor,
    fc1_scratch: torch.Tensor,
    fc2_scratch: torch.Tensor,
    workspace: torch.Tensor,
    m: int,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    t0_num_experts: int,
    t1_num_experts: int,
    topk: int,
    activation: str,
    map_slots: int,
    moe_block_size: int,
    element_dtype: str,
    fast_math: bool,
    sms: int,
    max_shared_mem: int,
    t0_weight_layout: str,
    t0_scale_format: str,
    t0_w13_layout: str,
    t1_weight_layout: str,
    t1_scale_format: str,
    t1_w13_layout: str,
    fc1_tile_k: int,
    fc1_tile_n: int,
    fc2_tile_k: int,
    fc2_tile_n: int,
    schedule_whole_tiles: bool,
    stream_int: int,
) -> None:
    _w4a16_fused_moe_hybrid_launch_flat(
        a_input,
        t0_w13,
        t0_w2,
        t0_w13_scale_i32,
        t0_w2_scale_i32,
        t0_w13_global,
        t0_w2_global,
        t1_w13,
        t1_w2,
        t1_w13_scale_i32,
        t1_w2_scale_i32,
        t1_w13_global,
        t1_w2_global,
        global_topk_ids,
        tier_local_map,
        fc1_out,
        activated,
        output_out,
        topk_weights,
        fc1_scratch,
        fc2_scratch,
        workspace,
        m,
        size_m,
        hidden_size,
        intermediate_size,
        t0_num_experts,
        t1_num_experts,
        topk,
        activation,
        map_slots,
        moe_block_size,
        element_dtype,
        fast_math,
        sms,
        max_shared_mem,
        t0_weight_layout,
        t0_scale_format,
        t0_w13_layout,
        t1_weight_layout,
        t1_scale_format,
        t1_w13_layout,
        fc1_tile_k,
        fc1_tile_n,
        fc2_tile_k,
        fc2_tile_n,
        schedule_whole_tiles,
        stream_int,
    )


@_w4a16_fused_moe_hybrid_launch_op.register_fake
def _w4a16_fused_moe_hybrid_launch_fake(
    a_input: torch.Tensor,
    t0_w13: torch.Tensor,
    t0_w2: torch.Tensor,
    t0_w13_scale_i32: torch.Tensor,
    t0_w2_scale_i32: torch.Tensor,
    t0_w13_global: torch.Tensor,
    t0_w2_global: torch.Tensor,
    t1_w13: torch.Tensor,
    t1_w2: torch.Tensor,
    t1_w13_scale_i32: torch.Tensor,
    t1_w2_scale_i32: torch.Tensor,
    t1_w13_global: torch.Tensor,
    t1_w2_global: torch.Tensor,
    global_topk_ids: torch.Tensor,
    tier_local_map: torch.Tensor,
    fc1_out: torch.Tensor,
    activated: torch.Tensor,
    output_out: torch.Tensor,
    topk_weights: torch.Tensor,
    fc1_scratch: torch.Tensor,
    fc2_scratch: torch.Tensor,
    workspace: torch.Tensor,
    m: int,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    t0_num_experts: int,
    t1_num_experts: int,
    topk: int,
    activation: str,
    map_slots: int,
    moe_block_size: int,
    element_dtype: str,
    fast_math: bool,
    sms: int,
    max_shared_mem: int,
    t0_weight_layout: str,
    t0_scale_format: str,
    t0_w13_layout: str,
    t1_weight_layout: str,
    t1_scale_format: str,
    t1_w13_layout: str,
    fc1_tile_k: int,
    fc1_tile_n: int,
    fc2_tile_k: int,
    fc2_tile_n: int,
    schedule_whole_tiles: bool,
    stream_int: int,
) -> None:
    return None


def run_w4a16_moe_hybrid(
    a_input: torch.Tensor,
    prepared_tier0,
    prepared_tier1,
    topk_weights: torch.Tensor,
    global_topk_ids: torch.Tensor,
    tier_local_map: torch.Tensor,
    *,
    activation: str,
    intermediate_cache13: torch.Tensor,
    intermediate_cache2: torch.Tensor,
    output: torch.Tensor,
    force_tile_config: tuple[int, int, int, int],
    size_m: int | None = None,
    fc1_c_tmp: torch.Tensor | None = None,
    fc2_c_tmp: torch.Tensor | None = None,
    fast_math: bool = True,
    schedule_whole_tiles: bool = True,
    stream: cuda.CUstream | None = None,
) -> torch.Tensor:
    """Run one hybrid two-tier W4A16 fused MoE layer.

    global_topk_ids carries GLOBAL expert ids ([m, topk] int32, -1 inactive);
    tier_local_map is the descriptor table from build_w4a16_tier_local_map.
    prepared_tier0/prepared_tier1 are per-tier prepared weight bundles whose
    packing tile config must match force_tile_config.
    """

    activation = normalize_moe_activation(activation)
    if not validate_activation(activation):
        raise ValueError("hybrid W4A16 requires a gated activation")
    element_dtype = _normalize_element_dtype(a_input.dtype)
    m, hidden_size = a_input.shape
    capacity_m = int(size_m) if size_m is not None else int(m)
    topk = int(topk_weights.shape[-1])
    if topk_weights.dtype != torch.float32:
        raise TypeError("topk_weights must be torch.float32")
    if global_topk_ids.dtype != torch.int32:
        raise TypeError("global_topk_ids must be torch.int32")
    if global_topk_ids.shape != (m, topk) and global_topk_ids.numel() != m * topk:
        raise ValueError(
            f"global_topk_ids must have m*topk={m * topk} elements, got "
            f"{tuple(global_topk_ids.shape)}"
        )

    for name, prepared in (("tier0", prepared_tier0), ("tier1", prepared_tier1)):
        prepared_dtype = getattr(prepared, "params_dtype", a_input.dtype)
        if prepared_dtype != a_input.dtype:
            raise TypeError(
                f"{name} prepared weights were built for {prepared_dtype}, "
                f"but a_input has dtype {a_input.dtype}"
            )
        if int(prepared.hidden_size) != int(hidden_size):
            raise ValueError(f"{name} hidden_size mismatch")
    intermediate_size = int(prepared_tier0.intermediate_size)
    if int(prepared_tier1.intermediate_size) != intermediate_size:
        raise ValueError("hybrid tiers disagree on intermediate_size")
    fc1_cols = intermediate_size * 2

    def _tier_layouts(prepared) -> tuple[str, str, str]:
        weight_layout = getattr(prepared, "weight_layout", "packed")
        scale_format = _normalize_scale_format(
            getattr(prepared, "scale_format", None) or "e4m3_k16"
        )
        if weight_layout == "modelopt":
            w13_layout = getattr(prepared, "w13_layout", "w13")
        else:
            w13_layout = "packed"
        return weight_layout, scale_format, w13_layout

    t0_layouts = _tier_layouts(prepared_tier0)
    t1_layouts = _tier_layouts(prepared_tier1)

    def _weight_args(prepared) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(prepared, "weight_layout", "packed") == "modelopt":
            return (
                prepared.w13.view(torch.uint8).view(-1),
                prepared.w2.view(torch.uint8).view(-1),
            )
        return (
            prepared.w13.view(torch.int32).view(-1),
            prepared.w2.view(torch.int32).view(-1),
        )

    t0_w13, t0_w2 = _weight_args(prepared_tier0)
    t1_w13, t1_w2 = _weight_args(prepared_tier1)

    props = torch.cuda.get_device_properties(a_input.device)
    sms = int(props.multi_processor_count)
    max_shared_mem = int(
        getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
    )

    capacity_routed_rows = capacity_m * topk
    intermediate_cache13_flat = intermediate_cache13.view(-1)
    intermediate_cache2_flat = intermediate_cache2.view(-1)
    if intermediate_cache13_flat.numel() < capacity_routed_rows * fc1_cols:
        raise ValueError("intermediate_cache13 is smaller than launch capacity")
    if intermediate_cache2_flat.numel() < capacity_routed_rows * intermediate_size:
        raise ValueError("intermediate_cache2 is smaller than launch capacity")
    fc1_out = intermediate_cache13_flat[: capacity_routed_rows * fc1_cols]
    activated = intermediate_cache2_flat[: capacity_routed_rows * intermediate_size]

    scratch_elements = packed_gemm_scratch_elements(
        size_n=max(fc1_cols, hidden_size),
        route_slots=capacity_routed_rows,
        moe_block_size=8,
        sms=sms,
    )
    fc1_scratch = _get_c_tmp(scratch_elements, device=a_input.device, scratch=fc1_c_tmp)
    fc2_scratch = _get_c_tmp(scratch_elements, device=a_input.device, scratch=fc2_c_tmp)
    workspace = prepared_tier0.workspace
    stream = current_cuda_stream() if stream is None else stream

    fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n = (int(v) for v in force_tile_config)
    torch.ops.flashinfer_sm12x.w4a16_fused_moe_hybrid_launch(
        a_input,
        t0_w13,
        t0_w2,
        prepared_tier0.w13_scale.view(torch.uint8).view(torch.int32).view(-1),
        prepared_tier0.w2_scale.view(torch.uint8).view(torch.int32).view(-1),
        prepared_tier0.w13_global_scale,
        prepared_tier0.w2_global_scale,
        t1_w13,
        t1_w2,
        prepared_tier1.w13_scale.view(torch.uint8).view(torch.int32).view(-1),
        prepared_tier1.w2_scale.view(torch.uint8).view(torch.int32).view(-1),
        prepared_tier1.w13_global_scale,
        prepared_tier1.w2_global_scale,
        global_topk_ids.view(-1),
        tier_local_map,
        fc1_out,
        activated,
        output.view(-1),
        topk_weights,
        fc1_scratch,
        fc2_scratch,
        workspace,
        int(m),
        capacity_m,
        int(hidden_size),
        intermediate_size,
        int(prepared_tier0.num_experts),
        int(prepared_tier1.num_experts),
        topk,
        activation,
        int(tier_local_map.numel()),
        8,
        element_dtype,
        bool(fast_math),
        sms,
        max_shared_mem,
        t0_layouts[0],
        t0_layouts[1],
        t0_layouts[2],
        t1_layouts[0],
        t1_layouts[1],
        t1_layouts[2],
        fc1_tile_k,
        fc1_tile_n,
        fc2_tile_k,
        fc2_tile_n,
        bool(schedule_whole_tiles),
        int(stream),
    )
    return output


__all__ = [
    "W4A16ActivationCompileResult",
    "W4A16FusedMoeCompileResult",
    "W4A16FusedMoeHybridCompileResult",
    "W4A16GemmCompileResult",
    "W4A16TopKSumCompileResult",
    "W4A16FusedMoeKernel",
    "W4A16FusedMoeHybridKernel",
    "W4A16ActivationKernel",
    "W4A16GemmKernel",
    "W4A16TopKSumKernel",
    "build_w4a16_tier_local_map",
    "clear_w4a16_kernel_cache",
    "compile_w4a16_activation",
    "compile_w4a16_fused_moe",
    "compile_w4a16_fused_moe_hybrid",
    "compile_w4a16_gemm",
    "compile_w4a16_topk_sum",
    "pack_topk_routes_by_expert",
    "run_w4a16_moe",
    "run_w4a16_moe_hybrid",
]
