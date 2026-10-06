"""CuTeDSL W4A16 NVFP4/BF16 W4A16 MoE kernels."""

from __future__ import annotations

from b12x._lib.quant.block_codec import BLOCK_CODECS, IQ2_CODECS, block_codec

import os
from dataclasses import dataclass, replace
from functools import partial
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.base_dsl.compiler import OptLevel
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, T, Uint32, dsl_user_op

from b12x._lib.compiler import (
    KernelCompileSpec,
    compile as b12x_compile,
)
from b12x._lib.compile_plan import attach_programs
from b12x._lib.program_cache import register_program_cache
from b12x._lib.intrinsics import (
    atomic_add_global_i32,
    bf16_mma_m16n8k16_f32,
    bf16_mma_rhs_fragments_as_mma_a_m16n8k16_f32,
    bfloat2_broadcast_lane,
    bfloat2_mul,
    bfloat2_to_float2_scaled,
    broadcast_f32_to_half2,
    broadcast_f32_to_bfloat2,
    cp_async4_shared_global,
    cp_async_u64_shared_global,
    cp_async4_shared_global_pred,
    fabs_f32,
    fmax_f32,
    f16_mma_m16n8k16_f32,
    f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32,
    fp8x4_e4m3_to_bfloat2x2_native_sm120,
    fp8x4_e4m3_to_half2x2,
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
    packed_decode_iq2_xs_to_bfloat2x4,
    iq2_xxs_descriptor_pair,
    q8_0_pair_to_bf16x2,
    ld_shared_u16_offset,
    packed_dequant_trellis_to_bfloat2x4,
    packed_dequant_trellis_to_half2x4,
    packed_dequant_trellis_stream_to_bfloat2x4,
    packed_dequant_trellis_stream_to_half2x4,
    packed_decode_lut_fp16_to_bfloat2x4,
    packed_decode_lut_fp16_to_half2x4,
    packed_decode_lut_e4m3_to_e4m3x8,
    packed_decode_lut_e4m3_direct_to_e4m3x8,
    ld_global_nc_v4_u32,
    pack_f32x2_to_bfloat2,
    pack_f32x2_to_f16x2,
    red_add_global_bf16x2,
    red_add_global_v4_f32,
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
    trellis_align_stream_u32x2,
    warp_reduce,
)
from b12x._lib.quant.iq2_xs import iq2_xs_execution_lut
from b12x._lib.quant.lut_e4m3 import (
    lut_e4m3_direct_table,
    lut_e4m3_value_table,
)
from b12x.moe._shared.kernels.trellis_ring import (
    trellis256_lane_geom_bits as _trellis_ring_lane_geom_bits,
)
from b12x.moe._shared.trellis_codebooks import (
    MCG,
    LUT_E4M3,
    LUT_FP16,
    validate_codebook_bits,
)
from b12x._lib.quant.lut_fp16 import (
    lut_fp16_segment_table,
)
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.moe._shared.kernels.w4a16.route_pack import (
    pack_topk_routes_by_expert as _pack_topk_routes_by_expert,
)
from b12x.moe._shared.kernels.w4a16.host import (
    _W4A16_ALLOWED_ROUTED_SIZES,
    max_packed_route_slots,
    packed_gemm_scratch_elements,
    plan_w4a16_buffers,
    prefill_fused_sum_eligible,
    prefill_fused_sum_enabled,
    select_route_block_size_m,
    validate_activation,
)
from b12x._lib.runtime_control import (
    raise_if_kernel_resolution_frozen,
)
from b12x.moe._shared.kernels.activations import (
    SITU,
    SITU_DEFAULT_BETA,
    SITU_DEFAULT_LINEAR_BETA,
    SWIGLUOAI_UNINTERLEAVE,
    is_gated_moe_activation,
    normalize_moe_activation,
    normalize_swiglu_alpha_for_activation,
    normalize_swiglu_beta_for_activation,
    normalize_swiglu_limit_for_activation,
)
from b12x.moe._shared.kernels.micro import (
    MoEMicroKernelBackend,
)


_ALLOWED_ROUTED_SIZES = _W4A16_ALLOWED_ROUTED_SIZES
_PACK_FACTOR = 8
_STAGES = 4


def _w4a16_small_m_occupancy() -> int:
    """Resolve the packed NVFP4 small-M schedule: one CTA/four stages or two/three."""
    value = os.environ.get("B12X_W4A16_SMALL_M_OCCUPANCY", "1")
    if value not in ("1", "2"):
        raise ValueError("B12X_W4A16_SMALL_M_OCCUPANCY must be 1 or 2")
    return int(value)


def _w4a16_small_m_double_occupancy(
    *, weight_layout: str, scale_format: str, uses_m_block_8: bool,
    small_m_occupancy: int | None = None,
) -> bool:
    """Whether a small-M plan runs two CTAs per SM with three stages: opted in,
    and packed NVFP4 weights with E4M3 scales, the measured geometry."""
    return (
        uses_m_block_8
        and weight_layout == "packed"
        and scale_format == "e4m3_k16"
        and (_w4a16_small_m_occupancy() if small_m_occupancy is None else small_m_occupancy) == 2
    )


def _w4a16_small_m_splitk_enabled() -> bool:
    """Experimental small-M split-K schedule toggle.  NOT yet correct.

    The m8 TC-decode schedule normally assigns one whole-K mn-tile per CTA
    when the grid is larger than the tile count, idling the remaining CTAs.
    That is the right call for cheap decoders (MCG/MUL1), where the stripe
    split-K finalize costs more than it recovers, but it leaves ~5/6 of the
    GPU idle through the FC1 phase of expensive decoders: at TP12 decode
    the lookup-table codebook path drops from 210.5/263.7 us to 81.5/99.9 us
    (P33/P24) under this flag, and MCG from 54.8/58.9 us to 34.4/38.5 us.

    Pair-rate correctness under the stripe partition requires the decode
    and pair window loads to see ABSOLUTE k-tile indices
    (``reduce_k_tile + tile_idx``); the pipelines' ``tile_idx`` is
    slice-local and only the staging bases carry the slice offset.  The
    K-axis K2/K4 boundary (``logical_k16 < 8``) mis-selects rates on every
    mid-column slice otherwise (P33 is rate-uniform and immune).  Fixed and
    validated: the full pair reference suite passes with this flag on and
    off.  The flag participates in every affected kernel cache key.
    """

    return os.environ.get("B12X_W4A16_SMALL_M_SPLITK", "0") == "1"


def _lut_e4m3_smem_enabled() -> bool:
    """Stage the 4 KiB ``lut_e4m3`` value table once per fused CTA.

    The dynamic switch exists to compare the staged table against the direct
    L1/L2 path under an otherwise identical compiled split-K schedule.
    """

    return os.environ.get("B12X_LUT_E4M3_SMEM", "1") == "1"


_E8M0_LOGICAL_TAIL_SCALE_N_ALIGNMENT = 64
_DEVICE_MAX_REG_BYTES = 256 * 1024
_DEFAULT_MAX_SHARED_MEM = 101_376
_SCALAR_ACC_FRAGMENT_WIDTH = 1
_WEIGHT_LAYOUTS = {"packed", "modelopt", "trellis_t256", "iq2_xs", "iq2_xxs", "q8_0"}
_MODEL_OPT_W13_LAYOUTS = {"w13", "w31"}
_TRELLIS256_W13_LAYOUTS = {"packed", "trellis_t256_proj"}
# Native t256 tiles contain 256 tail-biting codes at one compile-time
# bitrate. Their exact storage is [16*bits] int16 == [8*bits] uint32 per tile.
_TRELLIS256_BITS = (2, 3, 4, 5, 6)
_TRELLIS256_CODEBOOKS = {MCG, LUT_E4M3, LUT_FP16}
_LUT_E4M3_VALUE_TABLE_ENTRIES = 1 << 12
_LUT_E4M3_SMEM_REGION_BYTES = _LUT_E4M3_VALUE_TABLE_ENTRIES
_SCALE_FORMATS = {
    "e4m3_k16": "e4m3_k16",
    "e8m0_k32": "e8m0_k32",
    "e4m3_k32": "e4m3_k32",
    "iq2_xs": "iq2_xs",
    "iq2_xxs": "iq2_xxs",
    "q8_0": "q8_0",
}
_E8M0_K32_FP16_GLOBAL_COMPENSATION = float(2.0**7)
_E8M0_K32_BF16_GLOBAL_COMPENSATION = float(2.0**119)
_MAX_DIRECT_TOPK_ROUTE_M = 6
_W4A16_SMALL_M_DIRECT_MAX_M = 8
_FC2_DIRECT_MIN_EXPERT_CAPACITY = 1024
_TC_DECODE_PACK_COLLIDING_PAIRS = 3
_TC_DECODE_PACK_SM_COVERAGE_CAP = 64
_TC_DECODE_PACK_SM_COVERAGE_NUMERATOR = 7
_TC_DECODE_PACK_SM_COVERAGE_DENOMINATOR = 8


@dsl_user_op
def _pack_modelopt_words(
    a0: Uint32, a1: Uint32, b0: Uint32, b1: Uint32, byte_index: Uint32,
    *, loc=None, ip=None,
) -> Uint32:
    return Uint32(llvm.inline_asm(
        T.i32(),
        [value.ir_value(loc=loc, ip=ip) for value in (a0, a1, b0, b1, byte_index)],
        """
        {
            .reg .b32 sel, a, b, q, t, u;
            mad.lo.u32 sel, $5, 17, 64;
            prmt.b32 a, $1, $2, sel;
            prmt.b32 b, $3, $4, sel;
            prmt.b32 q, a, b, 0x5410;
            shr.u32 t, q, 4;
            xor.b32 t, t, q;
            and.b32 t, t, 0x00f000f0;
            shl.b32 u, t, 4;
            xor.b32 q, q, t;
            xor.b32 q, q, u;
            prmt.b32 $0, q, q, 0x3120;
        }
        """,
        "=r,r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    ))


@dsl_user_op
def _dequant_native_fp4_word(
    packed: Uint32, is_fp16: cutlass.Constexpr[bool], *, loc=None, ip=None,
):
    dtype = "f16" if is_fp16 else "bf16"
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [packed.ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 h0, h1, h2, h3, tmp;
            .reg .b8 q0, q1, q2, q3;
            cvt.u8.u32 q0, $4;
            shr.u32 tmp, $4, 8;
            cvt.u8.u32 q1, tmp;
            shr.u32 tmp, $4, 16;
            cvt.u8.u32 q2, tmp;
            shr.u32 tmp, $4, 24;
            cvt.u8.u32 q3, tmp;
        """
        + "\n".join(f"cvt.rn.{dtype}x2.e2m1x2 h{i}, q{i};" for i in range(4))
        + """
            prmt.b32 $0, h0, h2, 0x5410;
            prmt.b32 $1, h0, h2, 0x7632;
            prmt.b32 $2, h1, h3, 0x5410;
            prmt.b32 $3, h1, h3, 0x7632;
        }
        """,
        "=r,=r,=r,=r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return tuple(Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _gather_native_scale_bytes(a: Uint32, b: Uint32, c: Uint32, d: Uint32, byte: Uint32, *, loc=None, ip=None):
    return Uint32(llvm.inline_asm(
        T.i32(), [v.ir_value(loc=loc, ip=ip) for v in (a, b, c, d, byte)],
        """
        {
            .reg .b32 selector, lo, hi;
            mad.lo.u32 selector, $5, 17, 64;
            prmt.b32 lo, $1, $2, selector;
            prmt.b32 hi, $3, $4, selector;
            prmt.b32 $0, lo, hi, 0x5410;
        }
        """,
        "=r,r,r,r,r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    ))


def _trellis256_execution_lut(
    device: torch.device | str, codebook: str, *, direct_lut: bool = False
) -> torch.Tensor:
    if codebook == LUT_FP16:
        return lut_fp16_segment_table(device)
    if direct_lut:
        return lut_e4m3_direct_table(device)
    return lut_e4m3_value_table(device)

# TC-decode runs on the packed W4A16 object and folds the top-k sum into the FC2
# store epilogue. It is available across the small-M direct-topk range; the
# planner switches to expert-packed routing when route reuse amortizes packing.
# _TC_DECODE_M is retained for callers and tests that enumerate the range.
_TC_DECODE_MAX_M = _W4A16_SMALL_M_DIRECT_MAX_M
_TC_DECODE_M = tuple(range(1, _TC_DECODE_MAX_M + 1))


def _w4a16_tc_decode_preferred(
    *,
    m: int,
    topk: int,
    num_experts: int,
    sms: int,
) -> bool:
    """Choose direct TC-decode until expert packing has useful reuse.

    With ``r`` uniformly distributed routes over ``E`` experts, the birthday
    approximation predicts ``r * (r - 1) / (2 * E)`` colliding route pairs.
    Route packing starts to amortize its histogram/sort and separate top-k sum
    once it predicts several reusable expert rows and enough routed rows to
    cover the machine. The coverage proxy saturates because every routed row
    fans out into multiple FC tiles.
    """

    m = int(m)
    topk = int(topk)
    num_experts = max(int(num_experts), 1)
    sms = max(int(sms), 1)
    if m < 1 or m > _TC_DECODE_MAX_M:
        return False
    routed_rows = m * topk
    coverage_sms = min(sms, _TC_DECODE_PACK_SM_COVERAGE_CAP)
    pack_has_reuse = (
        routed_rows * _TC_DECODE_PACK_SM_COVERAGE_DENOMINATOR
        >= coverage_sms * _TC_DECODE_PACK_SM_COVERAGE_NUMERATOR
        and routed_rows * (routed_rows - 1)
        >= 2 * _TC_DECODE_PACK_COLLIDING_PAIRS * num_experts
    )
    return not pack_has_reuse


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
# for each specialization. Most entries are measured from the local SM121 JIT
# output; conservative caps keep supported intermediate route tiles launchable
# when their exact count cannot change the resulting occupancy.
_W4A16_REGS_SM121 = {
    (256, 1, 8, 8, True): 118,
    (256, 1, 16, 4, True): 118,
    (256, 1, 16, 8, True): 118,
    (256, 1, 32, 2, True): 118,
    (128, 1, 4, 8, True): 118,
    (128, 1, 8, 4, True): 120,
    (256, 1, 8, 8, False): 158,
    # The wide-N single-route tile is already above the one-CTA/SM register
    # threshold in the narrower N=128 schedule. The architectural cap keeps
    # the same launch contract until this specialization is measured directly.
    (256, 1, 16, 4, False): 255,
    (128, 1, 4, 8, False): 154,
    (128, 1, 8, 4, False): 143,
    (256, 2, 16, 4, False): 212,
    (128, 2, 4, 8, False): 215,
    (128, 2, 8, 4, False): 214,
    # Measured from the spill-free mixed-Trellis block-32 prefill kernel.
    # FC2 keeps the paired-M8 schedule qualified by mixed_trellis.py.
    (256, 2, 8, 8, False): 175,
    (256, 3, 16, 4, False): 249,
    # Block-48 is already above the one-CTA/SM register threshold at block-32.
    # Use the architectural cap until it has its own measured resource entry;
    # both values produce the same one-CTA/SM launch contract.
    (256, 3, 8, 8, False): 255,
    (128, 3, 4, 8, False): 249,
    (128, 3, 8, 4, False): 250,
    (256, 4, 16, 4, False): 255,
    # Measured by the mixed-Trellis block-64 qualification gate. This is the
    # stock one-grid prefill tile geometry (N=128, K=128) with four 16-row
    # route blocks per CTA. Keeping the measured entry permits block-64
    # routing without changing the K3/K4 accumulation geometry.
    (256, 4, 8, 8, False): 255,
    (128, 4, 4, 8, False): 255,
    (128, 4, 8, 4, False): 255,
}
# Unmeasured specializations fall back to the packed table (a conservative
# upper bound).
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
    if weight_layout in BLOCK_CODECS:
        if uses_m_block_8 and (
            cta_threads == 256 or (cta_n_blocks == 4 and cta_k_blocks == 8)
        ):
            return 128
        return 255
    try:
        return _W4A16_REGS_SM121[key]
    except KeyError as exc:
        raise ValueError(
            f"missing W4A16 register count for NVFP4 BF16 specialization {key}"
        ) from exc


def _iq2_xs_stage_bytes(tile_k: int, tile_n: int, codec: str = "iq2_xs") -> int:
    return (
        tile_k * tile_n // block_codec(codec).pack_factor
        + _covering_count(tile_k, block_codec(codec).block_weights) * tile_n * 2
        + tile_k // 32 * tile_n * (block_codec(codec).subscale_bytes // 8)
    )


def _w4a16_pipeline_stages(
    *, weight_layout: str, tile_n: int, tile_k: int, uses_m_block_8: bool,
    pipeline_stages: int | None = None, scale_format: str = "e4m3_k16",
    small_m_occupancy: int | None = None,
) -> int:
    if pipeline_stages is not None:
        if type(pipeline_stages) is not int or pipeline_stages not in (2, 3, 4, 5):
            raise ValueError("W4A16 pipeline_stages must be 2, 3, 4 or 5")
        return pipeline_stages
    if weight_layout in BLOCK_CODECS and tile_n == 128 and tile_k == 128:
        return 2
    if weight_layout in BLOCK_CODECS and uses_m_block_8 and tile_n == 64 and tile_k == 128:
        return 3
    if _w4a16_small_m_double_occupancy(
        weight_layout=weight_layout, scale_format=scale_format,
        uses_m_block_8=uses_m_block_8,
        small_m_occupancy=small_m_occupancy,
    ):
        # Three stages leave room for two resident CTAs per SM.
        return 3
    return _STAGES


def _reduction_shared_bytes(tile_n: int, tile_k: int, uses_m_block_8: bool) -> int:
    """Extent of the FP32 stores in the CTA partial reduction.

    M8 uses four vec4 stores with a two-vector stride; larger M uses eight
    contiguous vec4 stores. The K128 first pass also uses a second row eight
    strides away. This is larger than the dense BF16 output tile and may
    extend into the dead activation staging region.
    It must never overlap a codec LUT appended after the GEMM's shared region.
    """
    reduction_rows = int(tile_k) // 64
    if reduction_rows == 0:
        return 0
    vectors = (7 if uses_m_block_8 else 8) + (8 if reduction_rows == 2 else 0)
    return (int(tile_n) // 2) * vectors * 16


def _shared_memory_footprint(
    *,
    cta_m_blocks: int,
    tile_n: int,
    tile_k: int,
    scale_format: str = "e4m3_k16",
    weight_layout: str = "packed",
    weight_bits: int = 4,
    uses_m_block_8: bool = False,
    pipeline_stages: int | None = None,
    small_m_occupancy: int | None = None,
) -> int:
    cta_m = int(cta_m_blocks) * 16
    cta_n = int(tile_n)
    cta_k = int(tile_k)
    stages = _w4a16_pipeline_stages(
        weight_layout=weight_layout, tile_n=cta_n, tile_k=cta_k,
        uses_m_block_8=uses_m_block_8,
        pipeline_stages=pipeline_stages, scale_format=scale_format,
        small_m_occupancy=small_m_occupancy,
    )
    activation_rows = 8 if uses_m_block_8 and stages == 3 else cta_m
    sh_block_meta_size = activation_rows * 16
    sh_a_size = stages * (activation_rows * cta_k) * 2
    staged_weight_bits = int(weight_bits)
    sh_b_size = stages * (cta_k * cta_n * staged_weight_bits // 8)
    if weight_layout in BLOCK_CODECS:
        sh_b_size = stages * _iq2_xs_stage_bytes(cta_k, cta_n, weight_layout)
    sh_red_size = cta_m * (cta_n + 8) * 2
    sh_bias_size = cta_n * 2
    tmp_size = min(sh_b_size, sh_red_size) + sh_bias_size
    tmp_size = max(max(sh_b_size, sh_red_size), tmp_size)
    sh_s_size = 0
    if weight_layout not in BLOCK_CODECS:
        sh_s_size = (
            _covering_count(cta_k, _scale_group_size(scale_format))
            * cta_n
            * 2
            * stages
        )
    lut_size = block_codec(weight_layout).lut_bytes(selectors=True) if weight_layout in BLOCK_CODECS else 0
    gemm_size = tmp_size + sh_a_size + sh_s_size + sh_block_meta_size
    route_rows = 8 if uses_m_block_8 else cta_m
    gemm_size = max(
        gemm_size,
        route_rows * 16 + _reduction_shared_bytes(cta_n, cta_k, uses_m_block_8),
    )
    return gemm_size + lut_size


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
    weight_bits: int = 4,
    pipeline_stages: int | None = None,
    small_m_occupancy: int | None = None,
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
        weight_bits=weight_bits,
        uses_m_block_8=uses_m_block_8,
        pipeline_stages=pipeline_stages,
        small_m_occupancy=small_m_occupancy,
    )
    blocks_per_sm_limit = min(
        _DEVICE_MAX_REG_BYTES // register_bytes,
        int(max_shared_mem) // (smem_bytes + 1536),
    )
    if uses_m_block_8 and weight_layout not in BLOCK_CODECS:
        # Small-M (moe_block_size==8) TC-decode is weight-bandwidth/overhead
        # bound, not parallelism bound (only m*top_k route-blocks of GEMM work).
        # The fused FC1->activation->FC2 path crosses several grid barriers whose
        # tid==0 atomic-counter increment serializes across all grid_x CTAs, so an
        # oversized grid pays barrier-atomic latency proportional to grid_x for no
        # extra GEMM throughput. Pin one persistent CTA per SM to minimize the
        # barrier participant count while still covering the machine for the
        # I_tp=1024 GEMMs. The split-K persistent loop is grid_x-agnostic, so this
        # is numerically identical. Narrow NVFP4 experts may opt into two
        # CTAs per SM (see _w4a16_small_m_occupancy).
        blocks_per_sm_limit = (
            min(blocks_per_sm_limit, 2)
            if _w4a16_small_m_double_occupancy(
                weight_layout=weight_layout, scale_format=scale_format,
                uses_m_block_8=uses_m_block_8,
                small_m_occupancy=small_m_occupancy,
            )
            else 1
        )
    elif uses_m_block_8:
        block_limit = 4 if tile_n == 64 and tile_k == 128 else 2
        blocks_per_sm_limit = max(min(blocks_per_sm_limit, block_limit), 1)
    elif cta_m_blocks == 1:
        blocks_per_sm_limit = max(min(blocks_per_sm_limit, 4), 1)
    else:
        blocks_per_sm_limit = max(min(blocks_per_sm_limit, 2), 1)

    work_cta_count = (int(problem_n) // int(tile_n)) * int(problem_m) * int(top_k) * 4
    if (
        not (weight_layout in BLOCK_CODECS and uses_m_block_8)
        and work_cta_count < int(sms) * blocks_per_sm_limit
    ):
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
    weight_bits: int = 4,
    allow_logical_tail: bool = False,
    allow_qualified_fc2_tile: bool = False,
    uses_m_block_8: bool = False,
    pipeline_stages: int | None = None,
    small_m_occupancy: int | None = None,
) -> bool:
    if int(tile_k) == -1 or int(tile_n) == -1 or int(cta_threads) == -1:
        return False
    if weight_layout in BLOCK_CODECS and int(tile_k) > 128:
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
    # The FC2 wave-balanced decode schedule uses one qualified 32x512 tile.
    # Keep the generic tile floor while allowing that exact geometry to pass
    # through the explicit-pin path used by the Torch custom-op boundary.
    wide_n_fc2_tile = (
        allow_qualified_fc2_tile
        and int(tile_k) == 32
        and int(tile_n) == 512
        and int(cta_threads) == 256
    )
    if (
        int(tile_n) < 64
        or int(cta_threads) < 128
        or (int(tile_k) < 64 and not wide_n_fc2_tile)
    ):
        return False
    smem_bytes = _shared_memory_footprint(
        cta_m_blocks=cta_m_blocks,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_format=scale_format,
        weight_layout=weight_layout,
        weight_bits=weight_bits,
        uses_m_block_8=uses_m_block_8,
        pipeline_stages=pipeline_stages,
        small_m_occupancy=small_m_occupancy,
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
    weight_bits: int = 4,
    allow_logical_tail: bool = False,
    small_m_occupancy: int | None = None,
) -> tuple[int, int, int, int]:
    cta_m_blocks = _covering_count(moe_block_size, 16)
    uses_m_block_8 = moe_block_size == 8
    configs = (
        _LARGE_BATCH_TILE_CONFIGS if cta_m_blocks > 1 else _SMALL_BATCH_TILE_CONFIGS
    )
    best_occupancy = 0
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
            weight_bits=weight_bits,
            allow_logical_tail=allow_logical_tail,
            small_m_occupancy=small_m_occupancy,
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
            weight_bits=weight_bits,
            small_m_occupancy=small_m_occupancy,
        )
        occupancy = blocks_per_sm_limit * (
            cta_threads if weight_layout in BLOCK_CODECS else 1
        )
        if occupancy > best_occupancy:
            best_occupancy = occupancy
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
    dense_route_fast_path: bool = False
    trellis_bits: int = 3
    trellis_codebook: str = LUT_E4M3
    trellis_pair_kind: str | None = None
    trellis_rate_axis: str | None = None


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
    full_rotation: bool = False
    intermediate_hadamard: bool = False
    num_experts: int = 0
    route_num_experts: int = 0
    route_ids_dtype: torch.dtype = torch.int32
    use_expert_map: bool = False
    broadcast_svh: bool = False


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
    use_expert_map: bool = False
    scale_format: str = "e4m3_k16"
    tc_decode_fused_sum: bool = False
    prefill_fused_sum_fp32: bool = False
    collect_activation_amax: bool = False
    schedule_whole_tiles: bool = False
    intermediate_rotation: bool = False
    dual_a: bool = False
    trellis_bits: int = 3
    trellis_codebook: str = LUT_E4M3
    fc1_trellis_pair_kind: str | None = None
    fc2_trellis_pair_kind: str | None = None
    full_rotation: bool = False
    intermediate_hadamard: bool = False
    rotation_input_dtype: str = "fp16"
    cta_threads: int = -1
    shared_memory_bytes: int = -1
    broadcast_suh: bool = False
    small_m_direct_launches: tuple[_W4A16SmallMDirectLaunch, ...] = ()
    trellis_direct_lut: bool = False


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


class _W4A16FC2DirectLaunch(NamedTuple):
    compiled: object
    grid_x: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
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
        compile_time_phase: int = 0,
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
            e4m3_scale_layout="modelopt",
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            w13_layout=w13_layout,
            compile_time_phase=compile_time_phase,
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
        trellis_bits: int = 3,
        trellis_codebook: str = LUT_E4M3,
        trellis_pair_kind: str | None = None,
        trellis_rate_axis: str | None = None,
        source_n_rotation: int = 0,
        single_token_route_fast_path: bool = False,
        direct_topk_routes: bool = False,
        dense_route_fast_path: bool = False,
        dual_a: bool = False,
        route_major_a: bool = False,
        fused_topk_sum: bool = False,
        fused_sum_fp32: bool = False,
        fused_sum_topk: int = 1,
        schedule_whole_tiles: bool = False,
        dynamic_num_experts: bool = False,
        schedule_route_block_factor: int = 1,
        pipeline_stages: int | None = None,
        skip_empty_m_blocks: bool | None = None,
        small_m_occupancy: int | None = None,
    ):
        if element_dtype not in {"bf16", "fp16"}:
            raise ValueError(f"unsupported element_dtype {element_dtype!r}")
        if weight_layout not in _WEIGHT_LAYOUTS:
            raise ValueError(f"unsupported W4A16 weight_layout {weight_layout!r}")
        trellis_bits = int(trellis_bits)
        trellis_codebook = str(trellis_codebook).lower()
        scale_format = _normalize_scale_format(scale_format)
        if weight_layout == "modelopt":
            if w13_layout not in _MODEL_OPT_W13_LAYOUTS:
                raise ValueError(f"unsupported W4A16 w13_layout {w13_layout!r}")
        elif weight_layout == "trellis_t256":
            if w13_layout not in _TRELLIS256_W13_LAYOUTS:
                raise ValueError(f"unsupported trellis_t256 w13_layout {w13_layout!r}")
        else:
            w13_layout = "packed"
            source_n_rotation = 0
        if weight_layout == "trellis_t256":
            if trellis_codebook not in _TRELLIS256_CODEBOOKS:
                raise ValueError(
                    "trellis_t256 codebook must be one of "
                    f"{sorted(_TRELLIS256_CODEBOOKS)}, got {trellis_codebook!r}"
                )
            if trellis_bits not in _TRELLIS256_BITS:
                raise ValueError(
                    "trellis_t256 bits must be one of "
                    f"{_TRELLIS256_BITS}, got {trellis_bits}"
                )
            validate_codebook_bits(trellis_codebook, trellis_bits)
            if scale_format != "e4m3_k32":
                raise ValueError(
                    "trellis_t256 W4A16 weights require scale_format='e4m3_k32'"
                )
        elif weight_layout in BLOCK_CODECS:
            if element_dtype != "bf16":
                raise ValueError("IQ2_XS W4A16 requires BF16 activations")
            if scale_format != weight_layout:
                raise ValueError("IQ2_XS W4A16 weights require scale_format='iq2_xs'")
            if size_k % max(128, block_codec(weight_layout).block_weights) != 0 or size_n % 16 != 0:
                raise ValueError(
                    "IQ2_XS W4A16 requires K % 256 == 0 and N % 16 == 0; "
                    f"got N={size_n} K={size_k}"
                )
        trellis_pair_kind = (
            None if trellis_pair_kind is None else str(trellis_pair_kind).upper()
        )
        trellis_rate_axis = (
            None if trellis_rate_axis is None else str(trellis_rate_axis).lower()
        )
        if (trellis_pair_kind is None) != (trellis_rate_axis is None):
            raise ValueError(
                "trellis_pair_kind and trellis_rate_axis must be supplied together"
            )
        if trellis_pair_kind is not None:
            if weight_layout != "trellis_t256":
                raise ValueError("trellis pairs require trellis_t256 weights")
            if trellis_pair_kind not in {
                "P24",
                "P33",
                "P43",
                "P44",
                "PDYNAMIC",
                "P33_P43",
            }:
                raise ValueError(
                    "trellis_pair_kind must be P24, P33, P43, P44, "
                    "PDYNAMIC, or P33_P43, got "
                    f"{trellis_pair_kind!r}"
                )
            if trellis_rate_axis not in {"k", "n"}:
                raise ValueError(
                    f"trellis_rate_axis must be 'k' or 'n', got {trellis_rate_axis!r}"
                )
            if trellis_bits != 3:
                raise ValueError(
                    "Trellis pair decoding requires the trellis_bits=3 base "
                    "specialization"
                )
        if epilogue_activation not in (None, "relu2"):
            raise ValueError(
                "W4A16 GEMM epilogue activation currently supports only relu2"
            )
        if tile_n % 16 != 0 or tile_k % 16 != 0:
            raise ValueError("tile_n/tile_k must be multiples of 16")
        if weight_layout in BLOCK_CODECS and tile_k > 128:
            raise ValueError("IQ2_XS warp reduction requires tile_k <= 128")
        if trellis_rate_axis == "n" and (size_n % 256 or tile_n != 256):
            raise ValueError(
                "N-axis trellis pairs require size_n % 256 == 0 and "
                "tile_n=256 so every CTA consumes one complete fixed-rate pair"
            )
        if trellis_rate_axis == "k" and (
            size_k != 256 or tile_k > 128 or 128 % tile_k
        ):
            raise ValueError(
                "K-axis trellis pairs require size_k=256 and a tile_k that "
                "divides one 128-channel record"
            )
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
        self.dynamic_num_experts = bool(dynamic_num_experts)
        if self.dynamic_num_experts and weight_layout not in {
            "packed",
            "trellis_t256",
            "iq2_xs",
            "iq2_xxs",
            "q8_0",
        }:
            raise ValueError(
                "dynamic_num_experts is only supported for packed and "
                "native compressed-codebook weights"
            )
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
        self.trellis_bits = trellis_bits
        self.trellis_codebook = trellis_codebook
        self.trellis_pair_kind = trellis_pair_kind
        self.trellis_rate_axis = trellis_rate_axis
        self.weight_layout_trellis256_pair = trellis_pair_kind is not None
        self.trellis_pair_dynamic = trellis_pair_kind in {"PDYNAMIC", "P33_P43"}
        self.trellis_pair_compact_offsets = trellis_pair_kind == "P33_P43"
        static_pair_rates = {
            "P24": (2, 4),
            "P33": (3, 3),
            "P43": (4, 3),
            "P44": (4, 4),
        }
        self.trellis_pair_low_bits, self.trellis_pair_high_bits = (
            static_pair_rates.get(trellis_pair_kind, (3, 3))
        )
        self.lut_e4m3_smem = False
        self.trellis_direct_lut = False
        # Large-M route blocks only issue the MMAs of 16-row blocks that hold
        # live routes; padded rows of a partial block are never read back.
        self.skip_empty_m_blocks = (
            os.environ.get("B12X_W4A16_SKIP_EMPTY_M_BLOCKS", "1") == "1"
            if skip_empty_m_blocks is None else skip_empty_m_blocks
        )
        # Small-M stripe split-K: opt out of the one-tile-per-CTA fast path
        # so decode-heavy small-M phases spread each mn-tile's K range across
        # multiple CTAs (existing tail scheduling plus cross-CTA finalize).
        self.small_m_splitk = (
            weight_layout in BLOCK_CODECS and moe_block_size == 8
        ) or _w4a16_small_m_splitk_enabled()
        self.weight_layout_trellis256 = weight_layout == "trellis_t256"
        self.weight_layout_block = weight_layout in BLOCK_CODECS
        self.iq2_xxs = weight_layout == "iq2_xxs"
        self.q8 = weight_layout == "q8_0"
        self.block_pack_factor = block_codec(weight_layout).pack_factor if self.weight_layout_block else 1
        self.block_k = block_codec(weight_layout).block_weights if self.weight_layout_block else 256
        self.block_metadata_bytes = block_codec(weight_layout).metadata_bytes if self.weight_layout_block else 0
        self.block_tile_bytes = 256 // self.block_pack_factor
        self.iq2_xs_smem_lut = False
        self.weight_layout_trellis256_proj = (
            self.weight_layout_trellis256 and w13_layout == "trellis_t256_proj"
        )
        if self.weight_layout_trellis256_proj and (
            self.size_n % 2 != 0 or (self.size_n // 2) % self.tile_n != 0
        ):
            raise ValueError(
                "trellis_t256_proj requires each FC1 projection to contain "
                "an integral number of CTA N tiles"
            )
        self.b_region_variable = (
            self.weight_layout_trellis256 or self.weight_layout_block
        )
        self.b_bundle_rows = 2
        self.scale_format = scale_format
        self.native_nvfp4_scales = (
            weight_layout == "modelopt" and scale_format == "e4m3_k16"
        )
        self.scale_format_e8m0_k32 = scale_format == "e8m0_k32"
        # k32 scale cadence (two K16 rows share one scale group); the scale
        # DECODE stays keyed on scale_format_e8m0_k32 (e4m3_k32 uses the e4m3
        # decode arm, its 2**116 compensation lives in the global_scale tensor).
        self.scale_k32 = scale_format in ("e8m0_k32", "e4m3_k32")
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
        self.dense_route_fast_path = bool(dense_route_fast_path)
        if self.dense_route_fast_path and (
            self.direct_topk_routes
            or self.single_token_route_fast_path
            or self.num_experts != 1
            or self.top_k != 1
            or self.mul_topk_weights
        ):
            raise ValueError(
                "dense_route_fast_path requires E=1, top_k=1, no top-k weights, "
                "and no other route fast path"
            )
        self.dual_a = bool(dual_a)
        if self.dual_a and not self.weight_layout_trellis256_proj:
            raise ValueError(
                "dual_a is only valid for projection-major trellis_t256 FC1"
            )
        self.route_major_a = bool(route_major_a)
        if self.route_major_a and not self.dual_a:
            raise ValueError("route_major_a requires the exact dual-A FC1 path")
        self.fused_topk_sum = bool(fused_topk_sum)
        self.fused_sum_fp32 = bool(fused_sum_fp32)
        self.fused_sum_topk = int(fused_sum_topk)
        # Whole-tile persistent scheduling: every mn-tile is computed by one
        # CTA over the full K (grid-strided waves, ragged last wave), skipping
        # the split-K tail machinery entirely. Requires the host to bound the
        # wave count; used by the exact-geometry hybrid decode schedule.
        self.schedule_whole_tiles = bool(schedule_whole_tiles)
        self.schedule_route_block_factor = int(schedule_route_block_factor)
        if self.schedule_route_block_factor < 1:
            raise ValueError("schedule_route_block_factor must be >= 1")
        if self.schedule_route_block_factor != 1 and (
            not self.schedule_whole_tiles
            or self.direct_topk_routes
            or self.dense_route_fast_path
        ):
            raise ValueError(
                "grouped route-block scheduling requires route-packed "
                "whole-tile execution"
            )
        if (
            self.schedule_whole_tiles
            and not self.direct_topk_routes
            and not self.weight_layout_trellis256
        ):
            raise ValueError(
                "schedule_whole_tiles requires direct_topk_routes or trellis_t256"
            )
        if self.fused_topk_sum and self.fused_sum_topk < 1:
            raise ValueError("fused_sum_topk must be >= 1")
        if self.fused_sum_fp32 and not self.fused_topk_sum:
            raise ValueError("fused_sum_fp32 requires fused_topk_sum")
        self.cta_m_blocks = int(_covering_count(moe_block_size, 16))
        self.uses_m_block_8 = moe_block_size == 8
        self.stages = _w4a16_pipeline_stages(
            weight_layout=weight_layout, tile_n=self.tile_n, tile_k=self.tile_k,
            uses_m_block_8=self.uses_m_block_8,
            pipeline_stages=pipeline_stages, scale_format=scale_format,
            small_m_occupancy=small_m_occupancy,
        )
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
        self.max_shared_mem = max_shared_mem
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
            weight_bits=(
                max(4, self.trellis_bits) if self.weight_layout_trellis256 else 4
            ),
            pipeline_stages=self.stages,
            small_m_occupancy=small_m_occupancy,
        )

        # W4A16 shared-memory geometry, in int4 units unless noted.
        self.a_sh_stride = 16 * self.cta_k_blocks // 8
        self.a_sh_stage = self.a_sh_stride * (
            8 if self.uses_m_block_8 and self.stages == 3 else 16 * self.cta_m_blocks
        )
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
        # Native t256 uses 4*bits bytes per 32 codes;
        # packed/modelopt use 16 bytes for the same logical unit.
        self.b_unit_bytes = 16
        if self.weight_layout_trellis256:
            self.b_unit_bytes = 4 * self.trellis_bits
            if self.weight_layout_trellis256_pair:
                if self.trellis_rate_axis == "k":
                    self.b_unit_bytes = 16
                elif self.trellis_pair_compact_offsets:
                    self.b_unit_bytes = 14
                elif not self.trellis_pair_dynamic:
                    self.b_unit_bytes = 2 * (
                        self.trellis_pair_low_bits + self.trellis_pair_high_bits
                    )
        self.b_sh_stage_bytes = self.b_sh_stage * self.b_unit_bytes
        if self.weight_layout_block:
            self.b_sh_stage_bytes = _iq2_xs_stage_bytes(self.tile_k, self.tile_n, weight_layout)
        if self.b_region_variable:
            if self.b_sh_stage_bytes % 16 != 0:
                raise ValueError(
                    "trellis B stage bytes must be a multiple of 16; "
                    f"got {self.b_sh_stage_bytes}"
                )
            self.b_sh_chunks = self.b_sh_stage_bytes // 16
            self.b_sh_wr_iters_var = _covering_count(self.b_sh_chunks, self.cta_threads)
        else:
            self.b_sh_chunks = self.b_sh_stage
            self.b_sh_wr_iters_var = self.b_sh_wr_iters

        self.s_sh_stride = 16 * self.cta_n_blocks // 16
        self.s_tb_groups = (
            self.cta_k_blocks // 2 if self.scale_k32 else self.cta_k_blocks
        )
        self.s_sh_stage = (
            0
            if self.weight_layout_block
            else self.s_tb_groups * self.s_sh_stride
        )
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
        # B region size in int4 (16-byte) units, rounded up to a 16-byte
        # multiple so the following SMEM regions keep their 16-byte alignment
        # (exact for all supported tiles since b_sh_stage is a multiple of 4).
        sh_b_size = _covering_count(self.stages * self.b_sh_stage_bytes, 16)
        sh_size_min = min(sh_red_size, sh_b_size)
        sh_size_max = max(sh_red_size, sh_b_size)
        sh_bias_size = self.cta_n_blocks * 16 // 8
        sh_b_red_bias_size = max(sh_size_max, sh_size_min + sh_bias_size)
        self.sh_b_off = self.sh_valid_count_off
        self.sh_red_off = self.sh_valid_count_off
        self.sh_s_off = self.sh_valid_count_off + sh_b_red_bias_size
        self.sh_a_off = self.sh_s_off + self.stages * self.s_sh_stage
        self.shared_int4 = self.sh_a_off + self.stages * self.a_sh_stage
        self.shared_int4 = max(
            self.shared_int4,
            self.sh_red_off + _reduction_shared_bytes(
                self.tile_n, self.tile_k, self.uses_m_block_8
            ) // 16,
        )
        self.shared_words = self.shared_int4 * 4
        if self.shared_words * 4 > int(max_shared_mem):
            raise ValueError(
                "W4A16 shared-memory footprint exceeds device opt-in limit: "
                f"{self.shared_words * 4} > {int(max_shared_mem)} bytes "
                f"(layout={self.weight_layout})"
            )

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (
            self.size_n,
            self.size_k,
            self.covered_size_n,
            self.covered_size_k,
            self.scale_k_groups,
            self.has_logical_tail,
            None if self.dynamic_num_experts else self.num_experts,
            self.dynamic_num_experts,
            self.top_k,
            self.mul_topk_weights,
            self.tile_n,
            self.tile_k,
            self.stages,
            self.cta_threads,
            self.moe_block_size,
            self.element_dtype,
            self.epilogue_relu2,
            self.weight_layout,
            self.trellis_bits,
            self.trellis_codebook,
            self.trellis_pair_kind,
            self.trellis_rate_axis,
            self.scale_format,
            self.native_nvfp4_scales,
            self.w13_layout,
            self.source_n_rotation,
            self.single_token_route_fast_path,
            self.direct_topk_routes,
            self.dense_route_fast_path,
            self.dual_a,
            self.route_major_a,
            self.fused_topk_sum,
            self.fused_sum_fp32,
            self.fused_sum_topk,
            self.size_m if self.fused_sum_fp32 else None,
            self.cta_m_blocks,
            self.uses_m_block_8,
            self.shared_words,
            # Launch bounds are part of the compiled kernel.  Keep binaries
            # planned for different residency targets out of the same cache
            # entry even when their arithmetic geometry otherwise matches.
            self.blocks_per_sm,
            self.schedule_whole_tiles,
            self.schedule_route_block_factor,
            self.lut_e4m3_smem,
            self.trellis_direct_lut,
            self.small_m_splitk,
            self.skip_empty_m_blocks,
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
    def _iq2_xs_descriptor_offset(
        self, expert_idx: Int32, global_k16: Int32, global_n16: Int32
    ):
        tile = (
            (Int64(expert_idx) * Int64(self.size_n // 64) + Int64(global_n16 // Int32(4)))
            * Int64(self.size_k // 128) + Int64(global_k16 // Int32(8))
        )
        return tile * Int64(32 * self.block_tile_bytes // 4) + (
            Int64(global_k16 % Int32(8)) * Int64(4) + Int64(global_n16 % Int32(4))
        ) * Int64(self.block_tile_bytes // 4)

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
        if cutlass.const_expr(self.native_nvfp4_scales):
            if cutlass.const_expr(self.is_fp16):
                return fp8x4_e4m3_to_half2x2(packed)
            s0, s1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(packed)
            # The bitwise FP4 decoder produces values scaled by 2**-126.
            # All finite E4M3 scales remain representable after a 2**119 lift;
            # the remaining factor of 128 is applied with the weight global.
            lift = Uint32(0x7B007B00)
            return bfloat2_mul(s0, lift), bfloat2_mul(s1, lift)
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
        a_bf16_ptr: cute.Pointer,
        a_alt_bf16_ptr: cute.Pointer,
        b_i32_flat: cute.Tensor,
        c_bf16_ptr: cute.Pointer,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        trellis_lut_flat: cute.Tensor,
        active_m: cutlass.Int32,
        grid_x: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        a_bf16_flat = cute.make_tensor(
            a_bf16_ptr,
            layout=cute.make_layout((active_m * Int32(self.size_k),), stride=(1,)),
        )
        a_alt_bf16_flat = cute.make_tensor(
            a_alt_bf16_ptr,
            layout=cute.make_layout((active_m * Int32(self.size_k),), stride=(1,)),
        )
        c_bf16_flat = cute.make_tensor(
            c_bf16_ptr,
            layout=cute.make_layout(
                (active_m * Int32(self.top_k) * Int32(self.size_n),), stride=(1,)
            ),
        )
        grid = (grid_x, 1, 1)
        self.kernel(
            a_bf16_flat,
            a_alt_bf16_flat,
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
            trellis_lut_flat,
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
        a_alt_bf16_flat: cute.Tensor,
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
        trellis_lut_flat: cute.Tensor,
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
        trellis_lut_addr = get_ptr_as_int64(trellis_lut_flat, Int32(0))
        self._run_persistent_gemm(
            a_bf16_flat,
            a_alt_bf16_flat,
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
            trellis_lut_addr,
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
        a_alt_bf16_flat: cute.Tensor,
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
        trellis_lut_addr: Int64,
        smem_base: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_size_m: Int32,
        emit_tile: cutlass.Constexpr = None,
    ):
        n_tiles = Int32(self.n_tiles)
        route_blocks = active_size_m * Int32(self.top_k)
        if cutlass.const_expr(self.dense_route_fast_path):
            route_blocks = (
                active_size_m + Int32(self.moe_block_size) - Int32(1)
            ) // Int32(self.moe_block_size)
        elif cutlass.const_expr(not self.direct_topk_routes):
            route_blocks = packed_route_count[Int32(0)].to(Int32) // Int32(
                self.moe_block_size * self.schedule_route_block_factor
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
            if cutlass.const_expr(self.uses_m_block_8 and not self.small_m_splitk):
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
                # Decode-heavy codecs opt back into the stripe split-K partition
                # via ``B12X_W4A16_SMALL_M_SPLITK`` (see
                # ``_w4a16_small_m_splitk_enabled``): idling 5/6 of the grid
                # through an expensive FC1 costs far more than the finalize.
                if global_mn_tiles <= grid_x:
                    force_one_tile_per_cta = Int32(1)
            whole_k_waves = Int32(0)
            if cutlass.const_expr(self.weight_layout_block and self.direct_topk_routes):
                if global_mn_tiles >= grid_x:
                    whole_k_waves = Int32(1)
            if whole_k_waves != Int32(0):
                tail_mn_tiles = Int32(0)
                full_grid_mn_iters = (global_mn_tiles + grid_x - Int32(1)) // grid_x
            elif force_one_tile_per_cta != Int32(0):
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
                    if cutlass.const_expr(self.dense_route_fast_path):
                        expert_idx = Int32(0)
                    elif cutlass.const_expr(self.direct_topk_routes):
                        expert_idx = packed_route_indices[route_block_idx].to(Int32)
                    else:
                        expert_idx = block_expert_ids[route_block_idx].to(Int32)
                    if expert_idx >= Int32(0):
                        self._run_tile(
                            a_bf16_flat,
                            a_alt_bf16_flat,
                            b_i32_flat,
                            c_bf16_flat,
                            scales_i32_flat,
                            global_scale,
                            packed_route_indices,
                            topk_weights_flat,
                            c_tmp_f32_flat,
                            locks_i32_flat,
                            trellis_lut_addr,
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
        if cutlass.const_expr(self.dense_route_fast_path):
            block_row = route_block_idx * Int32(self.moe_block_size)
            valid_rows = active_size_m - block_row
            if valid_rows > Int32(self.moe_block_size):
                valid_rows = Int32(self.moe_block_size)
            if valid_rows < Int32(0):
                valid_rows = Int32(0)
            if tid < Int32(self.moe_block_size):
                row = block_row + tid
                st_shared_i32(
                    smem_base + Int32(self.sh_route_off * 16) + tid * Int32(4),
                    row,
                )
                st_shared_i32(
                    smem_base + Int32(self.sh_rd_route_off * 16) + tid * Int32(4),
                    row,
                )
            cute.arch.sync_threads()
            return valid_rows

        if cutlass.const_expr(self.direct_topk_routes):
            if tid == Int32(0):
                idx = route_block_idx
                st_shared_i32(smem_base + Int32(self.sh_route_off * 16), idx)
                rd_row = idx // Int32(self.top_k)
                if cutlass.const_expr(self.route_major_a):
                    rd_row = idx
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
                if cutlass.const_expr(self.route_major_a):
                    rd_row = idx
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
            if cutlass.const_expr(self.route_major_a):
                rd_row = idx
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
        a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        trellis_lut_addr: Int64,
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
        if cutlass.const_expr(self.trellis_pair_dynamic):
            # One expert-static, CTA-uniform branch chooses a fully specialized
            # pair path.  The mode never changes during this tile, so carrying
            # it as a constexpr keeps all inner staging/decode loops branchless.
            pair_mode = scales_i32_flat[expert_idx].to(Int64)
            if cutlass.const_expr(self.trellis_pair_compact_offsets):
                pair_mode = pair_mode & Int64(1)
            if pair_mode != Int64(0):
                self._run_tile_with_pair_override(
                    a_bf16_flat,
                    a_alt_bf16_flat,
                    b_i32_flat,
                    c_bf16_flat,
                    scales_i32_flat,
                    global_scale,
                    packed_route_indices,
                    topk_weights_flat,
                    c_tmp_f32_flat,
                    locks_i32_flat,
                    trellis_lut_addr,
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
                    2 if cutlass.const_expr(self.trellis_pair_compact_offsets) else 1,
                )
            else:
                self._run_tile_with_pair_override(
                    a_bf16_flat,
                    a_alt_bf16_flat,
                    b_i32_flat,
                    c_bf16_flat,
                    scales_i32_flat,
                    global_scale,
                    packed_route_indices,
                    topk_weights_flat,
                    c_tmp_f32_flat,
                    locks_i32_flat,
                    trellis_lut_addr,
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
                    0,
                )
            return
        self._run_tile_with_pair_override(
            a_bf16_flat,
            a_alt_bf16_flat,
            b_i32_flat,
            c_bf16_flat,
            scales_i32_flat,
            global_scale,
            packed_route_indices,
            topk_weights_flat,
            c_tmp_f32_flat,
            locks_i32_flat,
            trellis_lut_addr,
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
            -1,
        )

    @cute.jit
    def _run_tile_with_pair_override(
        self,
        a_bf16_flat: cute.Tensor,
        a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        trellis_lut_addr: Int64,
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
        dynamic_pair_override: cutlass.Constexpr[int],
    ):
        if cutlass.const_expr(self.uses_m_block_8):
            self._run_tile_m8(
                a_bf16_flat,
                a_alt_bf16_flat,
                b_i32_flat,
                c_bf16_flat,
                scales_i32_flat,
                global_scale,
                packed_route_indices,
                topk_weights_flat,
                c_tmp_f32_flat,
                locks_i32_flat,
                trellis_lut_addr,
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
                dynamic_pair_override,
            )
        else:
            self._run_tile_large_m(
                a_bf16_flat,
                a_alt_bf16_flat,
                b_i32_flat,
                c_bf16_flat,
                scales_i32_flat,
                global_scale,
                packed_route_indices,
                topk_weights_flat,
                c_tmp_f32_flat,
                locks_i32_flat,
                trellis_lut_addr,
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
                dynamic_pair_override,
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
        if cutlass.const_expr(self.native_nvfp4_scales and not self.is_fp16):
            global_scale_f32 *= cutlass.Float32(128.0)
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
        a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        trellis_lut_addr: Int64,
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
        dynamic_pair_override: cutlass.Constexpr[int],
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
            a_alt_bf16_flat,
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
            dynamic_pair_override,
        )

        b_scale_cur = cute.make_rmem_tensor((self.b_bundle_rows, 4), Uint32)
        b_scale_next = cute.make_rmem_tensor((self.b_bundle_rows, 4), Uint32)
        self._load_b_scale_register_bundle(
            b_scale_cur,
            smem_base,
            tid,
            b_sh_rd,
            s_sh_rd,
            Int32(0),
            Int32(0),
            reduce_k_tile,
            dynamic_pair_override,
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
            a_alt_bf16_flat,
            b_i32_flat,
            scales_i32_flat,
            trellis_lut_addr,
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
            dynamic_pair_override,
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
        a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        trellis_lut_addr: Int64,
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
        dynamic_pair_override: cutlass.Constexpr[int],
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
            a_alt_bf16_flat,
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
            dynamic_pair_override,
        )

        b_scale_cur = cute.make_rmem_tensor((self.b_bundle_rows, 4), Uint32)
        b_scale_next = cute.make_rmem_tensor((self.b_bundle_rows, 4), Uint32)
        self._load_b_scale_register_bundle(
            b_scale_cur,
            smem_base,
            tid,
            b_sh_rd,
            s_sh_rd,
            Int32(0),
            Int32(0),
            reduce_k_tile,
            dynamic_pair_override,
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
            a_alt_bf16_flat,
            b_i32_flat,
            scales_i32_flat,
            trellis_lut_addr,
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
            dynamic_pair_override,
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
        a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        trellis_lut_addr: Int64,
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
        dynamic_pair_override: cutlass.Constexpr[int],
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        b_frag = cute.make_rmem_tensor((2, 2), Uint32)
        live_m_blocks = (block_valid_rows + Int32(15)) // Int32(16)
        tile_idx = Int32(0)
        while tile_idx < k_tiles:
            for pipe in cutlass.range(
                self.stages, unroll=1 if self.weight_layout_block else 0,
                unroll_full=not self.weight_layout_block
            ):
                if tile_idx < k_tiles:
                    for kk in cutlass.range_constexpr(self.b_sh_wr_iters):
                        if cutlass.const_expr(
                            not self.weight_layout_block
                            or kk + 1 < self.b_sh_wr_iters
                        ):
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
                                reduce_k_tile,
                                dynamic_pair_override,
                                uses_m_block_8,
                            )

                        self._prefetch_pipeline_step(
                            a_bf16_flat,
                            a_alt_bf16_flat,
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
                            dynamic_pair_override,
                        )

                        if cutlass.const_expr(self.trellis_pair_dynamic):
                            if cutlass.const_expr(
                                int(dynamic_pair_override) != 0
                            ):
                                self._dequant_and_accumulate_bundle(
                                    acc0,
                                    acc1,
                                    acc2,
                                    acc3,
                                    b_frag,
                                    b_scale_cur,
                                    a_regs_cur,
                                    tid,
                                    smem_base,
                                    Int32(pipe),
                                    reduce_k_tile + tile_idx,
                                    kk,
                                    trellis_lut_addr,
                                    uses_m_block_8,
                                    dynamic_pair_override,
                                    live_m_blocks,
                                )
                            else:
                                self._dequant_and_accumulate_bundle(
                                    acc0,
                                    acc1,
                                    acc2,
                                    acc3,
                                    b_frag,
                                    b_scale_cur,
                                    a_regs_cur,
                                    tid,
                                    smem_base,
                                    Int32(pipe),
                                    reduce_k_tile + tile_idx,
                                    kk,
                                    trellis_lut_addr,
                                    uses_m_block_8,
                                    0,
                                    live_m_blocks,
                                )
                        else:
                            self._dequant_and_accumulate_bundle(
                                acc0,
                                acc1,
                                acc2,
                                acc3,
                                b_frag,
                                b_scale_cur,
                                a_regs_cur,
                                tid,
                                smem_base,
                                Int32(pipe),
                                reduce_k_tile + tile_idx,
                                kk,
                                trellis_lut_addr,
                                uses_m_block_8,
                                -1,
                                live_m_blocks,
                            )

                        if cutlass.const_expr(
                            self.weight_layout_block
                            and kk + 1 == self.b_sh_wr_iters
                        ):
                            # All inline weight reads finish before a stage is reused.
                            cute.arch.cp_async_wait_group(self.stages - 2)
                            cute.arch.sync_threads()
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
                                reduce_k_tile,
                                dynamic_pair_override,
                                uses_m_block_8,
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
                    reduce_k_tile + tile_idx,
                    dynamic_pair_override,
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
    def _dequant_and_accumulate_bundle(
        self,
        acc0,
        acc1,
        acc2,
        acc3,
        b_frag: cute.Tensor,
        b_scale_cur: cute.Tensor,
        a_regs_cur: cute.Tensor,
        tid: Int32,
        smem_base: Int32,
        pipe: Int32,
        tile_idx: Int32,
        kk: cutlass.Constexpr[int],
        trellis_lut_addr: Int64,
        uses_m_block_8: cutlass.Constexpr[bool],
        dynamic_pair_override: cutlass.Constexpr[int],
        live_m_blocks: Int32,
    ):
        if cutlass.const_expr(
            uses_m_block_8
            and self.weight_layout_trellis256_pair
            and self.trellis_rate_axis == "k"
            and (
                self.trellis_pair_kind in {"P24", "P43", "P44"}
                or (
                    self.trellis_pair_dynamic
                    and int(dynamic_pair_override) in (1, 2)
                )
            )
        ):
            # FC2 assigns an entire warp/kk fragment to one side of an
            # asymmetric pair.  Select the record bitrate once, outside the
            # four unrolled N16 fragments.
            low_bits = self.trellis_pair_low_bits
            high_bits = self.trellis_pair_high_bits
            if cutlass.const_expr(
                self.trellis_pair_dynamic and int(dynamic_pair_override) == 1
            ):
                low_bits = 2
                high_bits = 4
            elif cutlass.const_expr(
                self.trellis_pair_dynamic and int(dynamic_pair_override) == 2
            ):
                low_bits = 4
                high_bits = 3
            warp_id = tid >> Int32(5)
            warp_row = warp_id // Int32(self.tb_n_warps)
            kt_local = Int32(self.b_sh_wr_iters) * warp_row + Int32(kk)
            logical_k16 = tile_idx * Int32(self.cta_k_blocks) + kt_local
            if logical_k16 < Int32(8):
                for jj in cutlass.range_constexpr(4):
                    self._scaled_dequant_b_fragment_trellis256_bits(
                        b_frag,
                        b_scale_cur[0, jj],
                        b_scale_cur[1, jj],
                        trellis_lut_addr,
                        low_bits,
                    )
                    self._mma_accumulate_m8(acc0, jj, a_regs_cur, b_frag)
            else:
                for jj in cutlass.range_constexpr(4):
                    self._scaled_dequant_b_fragment_trellis256_bits(
                        b_frag,
                        b_scale_cur[0, jj],
                        b_scale_cur[1, jj],
                        trellis_lut_addr,
                        high_bits,
                    )
                    self._mma_accumulate_m8(acc0, jj, a_regs_cur, b_frag)
            return

        for jj in cutlass.range_constexpr(4):
            if cutlass.const_expr(self.weight_layout_block):
                q0, q1, base_pair, subscale_pair = self._load_iq2_xs_fragment(
                    smem_base, tid, pipe, Int32(kk), jj
                )
                self._scaled_dequant_b_fragment_iq2_xs(
                    b_frag,
                    q0,
                    q1,
                    base_pair,
                    subscale_pair,
                    trellis_lut_addr,
                    tid,
                )
            elif cutlass.const_expr(self.weight_layout_trellis256):
                if cutlass.const_expr(self.weight_layout_trellis256_pair):
                    if cutlass.const_expr(int(dynamic_pair_override) == 0):
                        self._scaled_dequant_b_fragment_trellis256_bits(
                            b_frag,
                            b_scale_cur[0, jj],
                            b_scale_cur[1, jj],
                            trellis_lut_addr,
                            3,
                        )
                    elif cutlass.const_expr(int(dynamic_pair_override) == 1):
                        self._scaled_dequant_b_fragment_trellis256_p24(
                            b_frag,
                            b_scale_cur[0, jj],
                            b_scale_cur[1, jj],
                            trellis_lut_addr,
                            tid,
                            jj,
                            tile_idx,
                            kk,
                        )
                    elif cutlass.const_expr(int(dynamic_pair_override) == 2):
                        self._scaled_dequant_b_fragment_trellis256_p43(
                            b_frag,
                            b_scale_cur[0, jj],
                            b_scale_cur[1, jj],
                            trellis_lut_addr,
                            tid,
                            jj,
                            tile_idx,
                            kk,
                        )
                    else:
                        self._scaled_dequant_b_fragment_trellis256_pair(
                            b_frag,
                            b_scale_cur[0, jj],
                            b_scale_cur[1, jj],
                            trellis_lut_addr,
                            tid,
                            jj,
                            tile_idx,
                            kk,
                        )
                else:
                    self._scaled_dequant_b_fragment_trellis256(
                        b_frag,
                        b_scale_cur[0, jj],
                        b_scale_cur[1, jj],
                        trellis_lut_addr,
                    )
            else:
                q, s = self._select_b_scale_register(jj, b_scale_cur)
                self._scaled_dequant_b_fragment(b_frag, q, s)
            if cutlass.const_expr(uses_m_block_8):
                self._mma_accumulate_m8(acc0, jj, a_regs_cur, b_frag)
            elif cutlass.const_expr(self.skip_empty_m_blocks and self.cta_m_blocks > 1):
                self._mma_accumulate_large_m(acc0, a_regs_cur, 0, jj, b_frag)
                if live_m_blocks > Int32(1):
                    self._mma_accumulate_large_m(acc1, a_regs_cur, 1, jj, b_frag)
                if cutlass.const_expr(self.cta_m_blocks > 2):
                    if live_m_blocks > Int32(2):
                        self._mma_accumulate_large_m(acc2, a_regs_cur, 2, jj, b_frag)
                if cutlass.const_expr(self.cta_m_blocks > 3):
                    if live_m_blocks > Int32(3):
                        self._mma_accumulate_large_m(acc3, a_regs_cur, 3, jj, b_frag)
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
            self._fold_cta_partials_large_m(
                acc0,
                acc1,
                acc2,
                acc3,
                smem_base,
                tid,
            )

        if reduce_slice_count > Int32(1):
            if cutlass.const_expr(self.weight_layout_block and uses_m_block_8):
                self._combine_iq2_xs_splitk_partials(
                    acc0, c_tmp_f32_flat, locks_i32_flat,
                    lock_slot, reduce_slice_idx, reduce_slice_count, tid,
                )
            else:
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
    def _combine_iq2_xs_splitk_partials(
        self,
        acc,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        lock_slot: Int32,
        reduce_slice_idx: Int32,
        reduce_slice_count: Int32,
        tid: Int32,
    ):
        cta, _, _ = cute.arch.block_idx()
        active_threads = Int32(32 * self.tb_n_warps)
        tile_elements = Int64(8 * self.tile_n)
        lock_addr = get_ptr_as_int64(locks_i32_flat, Int64(lock_slot))
        # A stripe publishes at most its first partial tile. Its final partial
        # is the reducer, so one scratch slot per CTA remains live until read.
        if reduce_slice_idx != reduce_slice_count - Int32(1):
            if tid < active_threads:
                for jj in cutlass.range_constexpr(4):
                    offset = Int64(cta) * tile_elements + (
                        Int64(active_threads) * Int64(jj) + Int64(tid)
                    ) * Int64(4)
                    st_global_v4_f32(
                        get_ptr_as_int64(c_tmp_f32_flat, offset),
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
            if tid == Int32(0):
                red_add_global_release_i32(lock_addr, Int32(1))
        else:
            if tid == Int32(0):
                arrived = Int32(-1)
                while arrived != reduce_slice_count - Int32(1):
                    arrived = ld_global_acquire_i32(lock_addr)
            cute.arch.sync_threads()
            if tid < active_threads:
                for jj in cutlass.range_constexpr(4):
                    s0 = cutlass.Float32(0.0)
                    s1 = cutlass.Float32(0.0)
                    s2 = cutlass.Float32(0.0)
                    s3 = cutlass.Float32(0.0)
                    for part in cutlass.range(reduce_slice_count - Int32(1), unroll=1):
                        source_cta = Int64(cta) + Int64(
                            reduce_slice_count - Int32(1) - part
                        )
                        offset = source_cta * tile_elements + (
                            Int64(active_threads) * Int64(jj) + Int64(tid)
                        ) * Int64(4)
                        v0, v1, v2, v3 = ld_global_v4_f32(
                            get_ptr_as_int64(c_tmp_f32_flat, offset)
                        )
                        s0 = v0 + s0
                        s1 = v1 + s1
                        s2 = v2 + s2
                        s3 = v3 + s3
                    acc[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] += s0
                    acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] += s1
                    acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] += s2
                    acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] += s3
            cute.arch.sync_threads()
            if tid == Int32(0):
                st_global_i32(lock_addr, Int32(0))

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
    def _load_iq2_xs_fragment(
        self,
        smem_base: Int32,
        tid: Int32,
        pipe: Int32,
        kk: Int32,
        jj: cutlass.Constexpr[int],
    ):
        lane = tid & Int32(31)
        warp_id = tid >> Int32(5)
        warp_row = warp_id // Int32(self.tb_n_warps)
        warp_n = warp_id % Int32(self.tb_n_warps)
        kt_local = Int32(self.b_sh_wr_iters) * warp_row + kk
        tc_col = lane // Int32(4)
        b_region = (
            smem_base + Int32(self.sh_b_off * 16)
            + pipe * Int32(self.b_sh_stage_bytes)
        )
        base_region = b_region + Int32(self.cta_k_blocks * self.cta_n_blocks * self.block_tile_bytes)
        scale_region = base_region + Int32(
            _covering_count(self.cta_k_blocks * 16, self.block_k) * self.cta_n_blocks * 32
        )
        local_n16 = Int32(4) * warp_n + Int32(jj)
        base_addr = base_region + (
            (kt_local // Int32(self.block_k // 16) * Int32(self.cta_n_blocks) + local_n16)
            * Int32(8) + tc_col
        ) * Int32(4)
        base_pair = ld_shared_u32(base_addr)
        if cutlass.const_expr(self.q8):
            tile_base = (kt_local * Int32(self.cta_n_blocks) + local_n16) * Int32(16)
            addr = b_region + (tile_base + tc_col * Int32(2)) * Int32(16) + (tid & Int32(3)) * Int32(2)
            q0 = Uint32(ld_shared_u16_offset(addr, 0)) | (Uint32(ld_shared_u16_offset(addr, 8)) << 16)
            q1 = Uint32(ld_shared_u16_offset(addr, 16)) | (Uint32(ld_shared_u16_offset(addr, 24)) << 16)
            subscale_pair = Uint32(0)
        elif cutlass.const_expr(self.iq2_xxs):
            tile_base = ((kt_local // Int32(2) * Int32(2)) * Int32(self.cta_n_blocks) + local_n16) * Int32(16)
            record_addr = b_region + (tile_base + tc_col * Int32(2)) * Int32(4)
            grid0, grid1 = ld_shared_v2_u32(record_addr)
            signs0, signs1 = ld_shared_v2_u32(record_addr + Int32(self.cta_n_blocks * self.block_tile_bytes))
            q0 = iq2_xxs_descriptor_pair(grid0, signs0, kt_local % Int32(2))
            q1 = iq2_xxs_descriptor_pair(grid1, signs1, kt_local % Int32(2))
            subscale_pair = (signs0 >> 28) | ((signs1 >> 28) << 8)
        else:
            scale_addr = scale_region + (
                (kt_local // Int32(2) * Int32(self.cta_n_blocks) + local_n16)
                * Int32(16) + tc_col // Int32(2) * Int32(4)
            )
            shift = (tc_col % Int32(2)) * Int32(16) + (kt_local % Int32(2)) * Int32(4)
            scale_pair = ld_shared_u32(scale_addr)
            tile_base = (kt_local * Int32(self.cta_n_blocks) + local_n16) * Int32(16)
            q0, q1 = ld_shared_v2_u32(
                b_region + (tile_base + tc_col * Int32(2)) * Int32(4)
            )
            subscale_pair = scale_pair >> shift
        return q0, q1, base_pair, subscale_pair

    @cute.jit
    def _load_b_scale_registers(
        self,
        smem_base: Int32,
        tid: Int32,
        b_sh_rd: Int32,
        s_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
        tile_idx: Int32,
        dynamic_pair_override: cutlass.Constexpr[int],
    ):
        if cutlass.const_expr(self.weight_layout_trellis256):
            # TRELLIS-256: no per-weight scale; the (2,4) bundle carries the 4
            # jj-tiles' pre-funnelled windows -> regs[0,jj]=win_a, regs[1,jj]=win_b.
            return self._load_b_registers_trellis256(
                smem_base, tid, pipe, kk, tile_idx, dynamic_pair_override
            )
        if cutlass.const_expr(self.weight_layout == "modelopt"):
            q0, q1, q2, q3 = self._load_b_registers_modelopt_shared(
                smem_base,
                b_sh_rd,
                pipe,
                kk,
            )
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
        if cutlass.const_expr(self.native_nvfp4_scales):
            words = cute.make_rmem_tensor((2, 4), Uint32)
            local_n = (warp_id % Int32(self.tb_n_warps)) * Int32(64) + (
                (tid & Int32(31)) // Int32(4)
            )
            stage_base = (
                smem_base + Int32(self.sh_s_off * 16)
                + pipe * Int32(self.s_sh_stage * 16)
            )
            vector_n = 128 if self.tile_n % 128 == 0 and self.source_n_rotation % 128 == 0 else 64
            vector_words = vector_n // 32
            for i in cutlass.range_constexpr(4):
                n0 = local_n + Int32(i * 8)
                word0, word1 = ld_shared_v2_u32(stage_base + (
                    (scale_group_id // Int32(4)) * Int32(self.tile_n)
                    + (n0 // Int32(vector_n) * Int32(32) + n0 % Int32(32)) * Int32(vector_words)
                    + n0 % Int32(vector_n) // Int32(32)
                ) * Int32(4))
                words[0, i] = word0
                words[1, i] = word1
            byte = Uint32(scale_group_id % Int32(4))
            s_pack0 = _gather_native_scale_bytes(words[0, 0], words[0, 1], words[0, 2], words[0, 3], byte)
            s_pack1 = _gather_native_scale_bytes(words[1, 0], words[1, 1], words[1, 2], words[1, 3], byte)
        else:
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
        tile_idx: Int32,
        dynamic_pair_override: cutlass.Constexpr[int],
    ):
        if cutlass.const_expr(self.weight_layout_block):
            return
        q0, q1, q2, q3, s0, s1, s2, s3 = self._load_b_scale_registers(
            smem_base,
            tid,
            b_sh_rd,
            s_sh_rd,
            pipe,
            kk,
            tile_idx,
            dynamic_pair_override,
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
        for row in cutlass.range_constexpr(self.b_bundle_rows):
            for col in cutlass.range_constexpr(4):
                regs[row, col] = Uint32(0)

    @cute.jit
    def _copy_b_scale_register_bundle(self, dst: cute.Tensor, src: cute.Tensor):
        for row in cutlass.range_constexpr(self.b_bundle_rows):
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
        pipe: Int32,
        kk: cutlass.Constexpr[int],
        tile_idx: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        dynamic_pair_override: cutlass.Constexpr[int],
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
                    reduce_k_tile + tile_idx,
                    dynamic_pair_override,
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
                    Int32((pipe + 1) % self.stages),
                    Int32(0),
                    reduce_k_tile + next_tile,
                    dynamic_pair_override,
                )
                self._load_a_register_bundle(
                    a_regs_next,
                    smem_base,
                    a_sh_rd,
                    Int32((pipe + 1) % self.stages),
                    Int32(0),
                    uses_m_block_8,
                )

    @cute.jit
    def _scaled_dequant_b_fragment(self, frag: cute.Tensor, q: Uint32, s: Uint32):
        if cutlass.const_expr(self.native_nvfp4_scales and self.is_fp16):
            b0_0, b0_1, b1_0, b1_1 = _dequant_native_fp4_word(q, self.is_fp16)
        else:
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
    def _scaled_dequant_b_fragment_iq2_xs(
        self,
        frag: cute.Tensor,
        q_row0: Uint32,
        q_row1: Uint32,
        base_pair: Uint32,
        subscale_pair: Uint32,
        execution_lut_addr: Int64,
        tid: Int32,
    ):
        if cutlass.const_expr(self.q8):
            b0_0 = q8_0_pair_to_bf16x2(q_row0, base_pair)
            b0_1 = q8_0_pair_to_bf16x2(q_row0 >> 16, base_pair)
            b1_0 = q8_0_pair_to_bf16x2(q_row1, base_pair >> 16)
            b1_1 = q8_0_pair_to_bf16x2(q_row1 >> 16, base_pair >> 16)
        else:
            pair_byte_offset = (tid & Int32(3)) * Int32(2)
            b0_0, b0_1, b1_0, b1_1 = packed_decode_iq2_xs_to_bfloat2x4(
                q_row0,
                q_row1,
                base_pair,
                subscale_pair,
                execution_lut_addr,
                pair_byte_offset,
                shared_lut=self.iq2_xs_smem_lut,
                selector_lut=True,
            )
        frag[0, 0] = b0_0
        frag[0, 1] = b0_1
        frag[1, 0] = b1_0
        frag[1, 1] = b1_1

    @cute.jit
    def _trellis256_lane_geom_bits(
        self,
        lane: Int32,
        weight_offset: cutlass.Constexpr[int],
        weight_count: cutlass.Constexpr[int],
        bits: cutlass.Constexpr[int],
    ):
        return _trellis_ring_lane_geom_bits(
            lane, weight_offset, weight_count, bits
        )

    @cute.jit
    def _trellis256_lane_geom(
        self,
        lane: Int32,
        weight_offset: cutlass.Constexpr[int],
        weight_count: cutlass.Constexpr[int],
    ):
        return self._trellis256_lane_geom_bits(
            lane, weight_offset, weight_count, self.trellis_bits
        )

    @cute.jit
    def _trellis_funnel256(self, a: Uint32, b: Uint32, s2: Int32):
        merged = (Int64(a) << Int64(32)) | Int64(b)
        return Uint32(merged >> Int64(s2))

    @cute.jit
    def _scaled_dequant_b_fragment_trellis256_bits(
        self,
        frag: cute.Tensor,
        win_a: Uint32,
        win_b: Uint32,
        trellis_lut_addr: Int64,
        bits: cutlass.Constexpr[int],
    ):
        if cutlass.const_expr(self.trellis_codebook == "mcg"):
            if cutlass.const_expr(int(bits) == 6):
                if cutlass.const_expr(self.is_fp16):
                    o0, o1, o2, o3 = packed_dequant_trellis_stream_to_half2x4(
                        win_a, win_b, int(bits)
                    )
                else:
                    o0, o1, o2, o3 = packed_dequant_trellis_stream_to_bfloat2x4(
                        win_a, win_b, int(bits)
                    )
            elif cutlass.const_expr(self.is_fp16):
                o0, o1, o2, o3 = packed_dequant_trellis_to_half2x4(
                    win_a, win_b, int(bits)
                )
            else:
                o0, o1, o2, o3 = packed_dequant_trellis_to_bfloat2x4(
                    win_a, win_b, int(bits)
                )
        elif cutlass.const_expr(self.trellis_codebook == LUT_FP16):
            if cutlass.const_expr(self.is_fp16):
                o0, o1, o2, o3 = packed_decode_lut_fp16_to_half2x4(
                    win_a, win_b, trellis_lut_addr, int(bits)
                )
            else:
                o0, o1, o2, o3 = packed_decode_lut_fp16_to_bfloat2x4(
                    win_a, win_b, trellis_lut_addr, int(bits)
                )
        else:
            if cutlass.const_expr(self.trellis_direct_lut):
                e_lo, e_hi = packed_decode_lut_e4m3_direct_to_e4m3x8(
                    win_a, win_b, trellis_lut_addr, int(bits),
                    in_shared=self.lut_e4m3_smem,
                )
            else:
                e_lo, e_hi = packed_decode_lut_e4m3_to_e4m3x8(
                    win_a,
                    win_b,
                    trellis_lut_addr,
                    int(bits),
                    value_table_in_shared=self.lut_e4m3_smem,
                )
            if cutlass.const_expr(self.is_fp16):
                o0, o1 = fp8x4_e4m3_to_half2x2(e_lo)
                o2, o3 = fp8x4_e4m3_to_half2x2(e_hi)
            else:
                o0, o1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(e_lo)
                o2, o3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(e_hi)
        frag[0, 0] = o0
        frag[0, 1] = o1
        frag[1, 0] = o2
        frag[1, 1] = o3

    @cute.jit
    def _scaled_dequant_b_fragment_trellis256(
        self,
        frag: cute.Tensor,
        win_a: Uint32,
        win_b: Uint32,
        trellis_lut_addr: Int64,
    ):
        self._scaled_dequant_b_fragment_trellis256_bits(
            frag, win_a, win_b, trellis_lut_addr, self.trellis_bits
        )

    @cute.jit
    def _scaled_dequant_b_fragment_trellis256_pair(
        self,
        frag: cute.Tensor,
        win_a: Uint32,
        win_b: Uint32,
        trellis_lut_addr: Int64,
        tid: Int32,
        jj: cutlass.Constexpr[int],
        tile_idx: Int32,
        kk: cutlass.Constexpr[int],
    ):
        # Dynamic pair kernels dispatch to bitrate-specific arms before entering
        # the MMA loop. This fallback serves compile-time static pair rates.
        if cutlass.const_expr(self.trellis_pair_kind == "P33"):
            self._scaled_dequant_b_fragment_trellis256_bits(
                frag, win_a, win_b, trellis_lut_addr, 3
            )
            return
        self._scaled_dequant_b_fragment_trellis256_pair_rates(
            frag,
            win_a,
            win_b,
            trellis_lut_addr,
            tid,
            jj,
            tile_idx,
            kk,
            self.trellis_pair_low_bits,
            self.trellis_pair_high_bits,
        )

    @cute.jit
    def _scaled_dequant_b_fragment_trellis256_p24(
        self,
        frag: cute.Tensor,
        win_a: Uint32,
        win_b: Uint32,
        trellis_lut_addr: Int64,
        tid: Int32,
        jj: cutlass.Constexpr[int],
        tile_idx: Int32,
        kk: cutlass.Constexpr[int],
    ):
        self._scaled_dequant_b_fragment_trellis256_pair_rates(
            frag, win_a, win_b, trellis_lut_addr, tid, jj, tile_idx, kk, 2, 4
        )

    @cute.jit
    def _scaled_dequant_b_fragment_trellis256_p43(
        self,
        frag: cute.Tensor,
        win_a: Uint32,
        win_b: Uint32,
        trellis_lut_addr: Int64,
        tid: Int32,
        jj: cutlass.Constexpr[int],
        tile_idx: Int32,
        kk: cutlass.Constexpr[int],
    ):
        self._scaled_dequant_b_fragment_trellis256_pair_rates(
            frag, win_a, win_b, trellis_lut_addr, tid, jj, tile_idx, kk, 4, 3
        )

    @cute.jit
    def _scaled_dequant_b_fragment_trellis256_pair_rates(
        self,
        frag: cute.Tensor,
        win_a: Uint32,
        win_b: Uint32,
        trellis_lut_addr: Int64,
        tid: Int32,
        jj: cutlass.Constexpr[int],
        tile_idx: Int32,
        kk: cutlass.Constexpr[int],
        low_bits: cutlass.Constexpr[int],
        high_bits: cutlass.Constexpr[int],
    ):
        warp_id = tid >> Int32(5)
        if cutlass.const_expr(self.trellis_rate_axis == "n"):
            # The MMA work is striped LLHH inside each four-N16 warp slab so
            # every warp executes two K2 and two K4 fragments.  The epilogue
            # scatters these accumulator fragments back to the reference
            # record0 || record1 channel order before the outer Hadamard.
            if cutlass.const_expr(jj < 2):
                self._scaled_dequant_b_fragment_trellis256_bits(
                    frag, win_a, win_b, trellis_lut_addr, low_bits
                )
            else:
                self._scaled_dequant_b_fragment_trellis256_bits(
                    frag, win_a, win_b, trellis_lut_addr, high_bits
                )
        else:
            warp_row = warp_id // Int32(self.tb_n_warps)
            kt_local = Int32(self.b_sh_wr_iters) * warp_row + Int32(kk)
            logical_k16 = tile_idx * Int32(self.cta_k_blocks) + kt_local
            if logical_k16 < Int32(8):
                self._scaled_dequant_b_fragment_trellis256_bits(
                    frag, win_a, win_b, trellis_lut_addr, low_bits
                )
            else:
                self._scaled_dequant_b_fragment_trellis256_bits(
                    frag, win_a, win_b, trellis_lut_addr, high_bits
                )

    @cute.jit
    def _load_trellis256_pair_tile_windows(
        self,
        b_region: Int32,
        tile_base_u32: Int32,
        lane: Int32,
        bits: cutlass.Constexpr[int],
    ):
        ia, ib, s2, _ = self._trellis256_lane_geom_bits(lane, 0, 8, bits)
        a = ld_shared_u32(b_region + (tile_base_u32 + ia) * Int32(4))
        b = ld_shared_u32(b_region + (tile_base_u32 + ib) * Int32(4))
        return (
            self._trellis_funnel256(a, b, s2),
            self._trellis_funnel256(a, b, s2 + Int32(4 * int(bits))),
        )

    @cute.jit
    def _load_b_registers_trellis256_pair(
        self,
        smem_base: Int32,
        tid: Int32,
        pipe: Int32,
        kk: Int32,
        tile_idx: Int32,
        dynamic_pair_override: cutlass.Constexpr[int],
    ):
        regs = cute.make_rmem_tensor((2, 4), Uint32)
        if cutlass.const_expr(
            self.trellis_pair_kind == "P24"
            or (
                self.trellis_pair_dynamic
                and int(dynamic_pair_override) == 1
            )
        ):
            self._load_b_registers_trellis256_pair_bits(
                regs, smem_base, tid, pipe, kk, tile_idx, 2, 4
            )
        elif cutlass.const_expr(
            self.trellis_pair_kind == "P43"
            or (self.trellis_pair_dynamic and int(dynamic_pair_override) == 2)
        ):
            self._load_b_registers_trellis256_pair_bits(
                regs, smem_base, tid, pipe, kk, tile_idx, 4, 3
            )
        elif cutlass.const_expr(self.trellis_pair_kind == "P44"):
            self._load_b_registers_trellis256_pair_bits(
                regs, smem_base, tid, pipe, kk, tile_idx, 4, 4
            )
        else:
            self._load_b_registers_trellis256_pair_bits(
                regs, smem_base, tid, pipe, kk, tile_idx, 3, 3
            )
        return (
            regs[0, 0],
            regs[0, 1],
            regs[0, 2],
            regs[0, 3],
            regs[1, 0],
            regs[1, 1],
            regs[1, 2],
            regs[1, 3],
        )

    @cute.jit
    def _load_b_registers_trellis256_pair_bits(
        self,
        regs: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pipe: Int32,
        kk: Int32,
        tile_idx: Int32,
        low_bits: cutlass.Constexpr[int],
        high_bits: cutlass.Constexpr[int],
    ):
        lane = tid & Int32(31)
        warp_id = tid >> Int32(5)
        warp_row = warp_id // Int32(self.tb_n_warps)
        w_n = warp_id % Int32(self.tb_n_warps)
        kt_local = Int32(self.b_sh_wr_iters) * warp_row + kk
        b_region = (
            smem_base + Int32(self.sh_b_off * 16) + pipe * Int32(self.b_sh_stage_bytes)
        )
        wa = [Uint32(0), Uint32(0), Uint32(0), Uint32(0)]
        wb = [Uint32(0), Uint32(0), Uint32(0), Uint32(0)]
        if cutlass.const_expr(self.trellis_rate_axis == "n"):
            pair_span_u32 = 8 * 8 * (low_bits + high_bits)
            high_base_u32 = 8 * 8 * low_bits
            for jj in cutlass.range_constexpr(4):
                # Reference storage remains record0 || record1.  MMA register
                # assignment is LLHH within a warp; the epilogue restores the
                # original contiguous output-channel order.
                if cutlass.const_expr(jj < 2):
                    record_n16 = Int32(2) * w_n + Int32(jj)
                    tile_base = (
                        kt_local * Int32(pair_span_u32)
                        + record_n16 * Int32(8 * low_bits)
                    )
                    wa[jj], wb[jj] = self._load_trellis256_pair_tile_windows(
                        b_region, tile_base, lane, low_bits
                    )
                else:
                    record_n16 = Int32(2) * w_n + Int32(jj - 2)
                    tile_base = (
                        kt_local * Int32(pair_span_u32)
                        + Int32(high_base_u32)
                        + record_n16 * Int32(8 * high_bits)
                    )
                    wa[jj], wb[jj] = self._load_trellis256_pair_tile_windows(
                        b_region, tile_base, lane, high_bits
                    )
        else:
            # Each K16 row owns a max-K4 shared slot.  K2/K3 rows occupy only
            # the compact prefix; the remainder is intentionally untouched.
            logical_k16 = tile_idx * Int32(self.cta_k_blocks) + kt_local
            tile_slot_u32 = self.cta_n_blocks * 8 * 4
            kt_base_u32 = kt_local * Int32(tile_slot_u32)
            for jj in cutlass.range_constexpr(4):
                local_n16 = Int32(4) * w_n + Int32(jj)
                tile_base = Int32(0)
                # CuTe must specialize the payload width before evaluating the
                # device-side row predicate, so these equal bodies cannot be
                # combined into one Python boolean expression.
                if cutlass.const_expr(int(low_bits) == int(high_bits)):  # noqa: SIM114
                    tile_base = kt_base_u32 + local_n16 * Int32(8 * low_bits)
                    wa[jj], wb[jj] = self._load_trellis256_pair_tile_windows(
                        b_region, tile_base, lane, low_bits
                    )
                elif logical_k16 < Int32(8):
                    tile_base = kt_base_u32 + local_n16 * Int32(8 * low_bits)
                    wa[jj], wb[jj] = self._load_trellis256_pair_tile_windows(
                        b_region, tile_base, lane, low_bits
                    )
                else:
                    tile_base = kt_base_u32 + local_n16 * Int32(8 * high_bits)
                    wa[jj], wb[jj] = self._load_trellis256_pair_tile_windows(
                        b_region, tile_base, lane, high_bits
                    )
        for jj in cutlass.range_constexpr(4):
            regs[0, jj] = wa[jj]
            regs[1, jj] = wb[jj]

    @cute.jit
    def _load_b_registers_trellis256(
        self,
        smem_base: Int32,
        tid: Int32,
        pipe: Int32,
        kk: Int32,
        tile_idx: Int32,
        dynamic_pair_override: cutlass.Constexpr[int],
    ):
        if cutlass.const_expr(self.weight_layout_trellis256_pair):
            return self._load_b_registers_trellis256_pair(
                smem_base,
                tid,
                pipe,
                kk,
                tile_idx,
                dynamic_pair_override,
            )
        lane = tid & Int32(31)
        warp_id = tid >> Int32(5)
        warp_row = warp_id // Int32(self.tb_n_warps)
        w_n = warp_id % Int32(self.tb_n_warps)
        kt_local = Int32(self.b_sh_wr_iters) * warp_row + kk
        b_region = (
            smem_base + Int32(self.sh_b_off * 16) + pipe * Int32(self.b_sh_stage_bytes)
        )
        tile_u32 = 8 * self.trellis_bits
        base_u32 = (kt_local * Int32(self.cta_n_blocks) + Int32(4) * w_n) * Int32(
            tile_u32
        )
        wa = [Uint32(0), Uint32(0), Uint32(0), Uint32(0)]
        wb = [Uint32(0), Uint32(0), Uint32(0), Uint32(0)]
        if cutlass.const_expr(self.trellis_bits <= 4):
            ia, ib, s2, _ = self._trellis256_lane_geom(lane, 0, 8)
            for jj in cutlass.range_constexpr(4):
                tbase = base_u32 + Int32(jj * tile_u32)
                a = ld_shared_u32(b_region + (tbase + ia) * Int32(4))
                b = ld_shared_u32(b_region + (tbase + ib) * Int32(4))
                wa[jj] = self._trellis_funnel256(a, b, s2)
                wb[jj] = self._trellis_funnel256(
                    a, b, s2 + Int32(4 * self.trellis_bits)
                )
        elif cutlass.const_expr(self.trellis_bits == 5):
            ib0, ib1, sb, _ = self._trellis256_lane_geom(lane, 0, 4)
            ia0, ia1, sa, _ = self._trellis256_lane_geom(lane, 4, 4)
            for jj in cutlass.range_constexpr(4):
                tbase = base_u32 + Int32(jj * tile_u32)
                b0 = ld_shared_u32(b_region + (tbase + ib0) * Int32(4))
                b1 = ld_shared_u32(b_region + (tbase + ib1) * Int32(4))
                a0 = ld_shared_u32(b_region + (tbase + ia0) * Int32(4))
                a1 = ld_shared_u32(b_region + (tbase + ia1) * Int32(4))
                wb[jj] = self._trellis_funnel256(b0, b1, sb)
                wa[jj] = self._trellis_funnel256(a0, a1, sa)
        else:
            i0, i2, s2, delta = self._trellis256_lane_geom(lane, 0, 8)
            i1 = i0 + Int32(1)
            i1 = i1 - Int32(tile_u32) * (i1 >= Int32(tile_u32)).to(Int32)
            for jj in cutlass.range_constexpr(4):
                tbase = base_u32 + Int32(jj * tile_u32)
                z0 = ld_shared_u32(b_region + (tbase + i0) * Int32(4))
                z1 = ld_shared_u32(b_region + (tbase + i1) * Int32(4))
                z2 = ld_shared_u32(b_region + (tbase + i2) * Int32(4))
                wa[jj], wb[jj] = trellis_align_stream_u32x2(z0, z1, z2, s2, delta)
        return wa[0], wa[1], wa[2], wa[3], wb[0], wb[1], wb[2], wb[3]

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
    def _modelopt_smem_permuted_offset(self, byte_offset: Int32) -> Int32:
        # Distribute row gathers across banks while preserving 16-byte alignment.
        return byte_offset ^ ((byte_offset >> Int32(3)) & Int32(0x70))

    @cute.jit
    def _stage_modelopt_scales(
        self,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pipe: Int32,
        expert_idx: Int32,
        output_n_tile: Int32,
        tile_idx: Int32,
    ):
        words_per_row = self.s_tb_groups // 4
        vector_n = 128 if self.tile_n % 128 == 0 and self.source_n_rotation % 128 == 0 else 64
        vector_words = vector_n // 32
        vectors_per_k_word = self.tile_n // vector_words
        scale_cols = _covering_count(self.size_k // 16, 4) * 4
        expert_words = _covering_count(self.size_n, 128) * 128 * scale_cols // 4
        stage_base = (
            smem_base + Int32(self.sh_s_off * 16)
            + pipe * Int32(self.s_sh_stage * 16)
        )
        for i in cutlass.range_constexpr(_covering_count(self.s_sh_stage * 4 // vector_words, self.cta_threads)):
            vector_idx = tid + Int32(i * self.cta_threads)
            if vector_idx < Int32(self.s_sh_stage * 4 // vector_words):
                n_group = vector_idx % Int32(vectors_per_k_word)
                local_n = (n_group // Int32(32)) * Int32(vector_n) + n_group % Int32(32)
                local_k_word = vector_idx // Int32(vectors_per_k_word)
                k_word = tile_idx * Int32(words_per_row) + local_k_word
                logical_n = output_n_tile * Int32(self.tile_n) + local_n
                source_n = self._source_n_from_logical(logical_n)
                offset = (
                    Int64(expert_idx) * Int64(expert_words)
                    + Int64(source_n // Int32(128)) * Int64(scale_cols * 32)
                    + Int64(k_word) * Int64(128)
                    + Int64(source_n % Int32(32)) * Int64(4)
                    + Int64((source_n % Int32(128)) // Int32(32))
                )
                if logical_n < Int32(self.size_n) and k_word < Int32(scale_cols // 4):
                    if cutlass.const_expr(vector_words == 4):
                        cp_async4_shared_global(
                            stage_base + vector_idx * Int32(16),
                            get_ptr_as_int64(scales_i32_flat, offset),
                        )
                    else:
                        cp_async_u64_shared_global(
                            stage_base + vector_idx * Int32(8),
                            get_ptr_as_int64(scales_i32_flat, offset),
                        )
                else:
                    for word in cutlass.range_constexpr(vector_words):
                        st_shared_u32(stage_base + (vector_idx * Int32(vector_words) + Int32(word)) * Int32(4), Uint32(0))

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
        local_n = n_tile * Int32(64) + warp_id * Int32(16) + tc_col
        byte_offset = local_n * Int32(self.tile_k // 2) + k_tile * Int32(8)
        stage_base = (
            smem_base
            + Int32(self.sh_b_off * 16)
            + pipe * Int32(self.b_sh_stage * 16)
        )
        a0, a1 = ld_shared_v2_u32(
            stage_base + self._modelopt_smem_permuted_offset(byte_offset)
        )
        b0, b1 = ld_shared_v2_u32(
            stage_base + self._modelopt_smem_permuted_offset(
                byte_offset + Int32(8 * (self.tile_k // 2))
            )
        )
        return _pack_modelopt_words(a0, a1, b0, b1, Uint32(tc_row // Int32(2)))

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
        a_alt_bf16_flat: cute.Tensor,
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
        dynamic_pair_override: cutlass.Constexpr[int],
    ):
        for i in cutlass.range_constexpr(self.a_sh_wr_iters):
            row = a_rows_per_iter * Int32(i) + a_gl_rd_row
            route_index = Int32(0)
            if row < Int32(self.moe_block_size):
                route_index = ld_shared_i32_relaxed(
                    smem_base
                    + Int32(self.sh_rd_route_off * 16)
                    + row * Int32(4)
                )
            a_int4 = (
                Int64(route_index) * Int64(a_gl_stride)
                + Int64(tile_idx) * Int64(self.a_gl_rd_delta_o)
                + Int64(a_gl_rd_col0)
            )
            a_dst = self._int4_addr(
                smem_base,
                Int32(self.sh_a_off)
                + pipe * Int32(self.a_sh_stage)
                + self._activation_smem_permuted_offset(
                    Int32(i * self.a_sh_wr_delta) + a_sh_wr
                ),
            )
            a_src = get_ptr_as_int64(a_bf16_flat, a_int4 * Int64(8))
            if cutlass.const_expr(self.dual_a):
                # Projection-major FC1 validates that each CTA N tile is wholly
                # gate or wholly up.  Select the matching pre-rotated A operand
                # with the same logical projection boundary used by B staging.
                # The selected source still lands in the one existing A SMEM
                # stage, so this adds neither shared memory nor MMA work.
                if output_n_tile >= Int32(self.n_tiles // 2):
                    a_src = get_ptr_as_int64(a_alt_bf16_flat, a_int4 * Int32(8))
            if cutlass.const_expr(self.has_k_tile_tail):
                a_k_int4 = tile_idx * Int32(self.a_gl_rd_delta_o) + a_gl_rd_col0
                if row < block_valid_rows and a_k_int4 < a_gl_stride:
                    cp_async4_shared_global(
                        a_dst,
                        a_src,
                    )
                else:
                    st_shared_v4_u32(a_dst, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
            else:
                cp_async4_shared_global_pred(
                    a_dst,
                    a_src,
                    (row < block_valid_rows).to(Int32),
                )

        if cutlass.const_expr(self.weight_layout_block):
            n16_total = self.size_n // 16
            chunks_per_k16 = self.cta_n_blocks * (self.block_tile_bytes // 16)
            total_chunks = self.cta_k_blocks * chunks_per_k16
            b_region = smem_base + Int32(self.sh_b_off * 16) + pipe * Int32(self.b_sh_stage_bytes)
            packed_tile = Int64(0)
            if cutlass.const_expr(self.tile_n == 64 and self.tile_k == 128):
                packed_tile = (
                    (Int64(expert_idx) * Int64(self.size_n // 64) + Int64(output_n_tile))
                    * Int64(self.size_k // 128) + Int64(tile_idx)
                )
            for i in cutlass.range_constexpr(_covering_count(total_chunks, self.cta_threads)):
                chunk = Int32(i * self.cta_threads) + tid
                local_tile = chunk // Int32(self.block_tile_bytes // 16)
                local_k16 = local_tile // Int32(self.cta_n_blocks)
                local_n16 = local_tile % Int32(self.cta_n_blocks)
                tile_chunk = chunk % Int32(self.block_tile_bytes // 16)
                global_k16 = tile_idx * Int32(self.cta_k_blocks) + local_k16
                global_n16 = output_n_tile * Int32(self.cta_n_blocks) + local_n16
                if cutlass.const_expr(self.tile_n == 64 and self.tile_k == 128):
                    source_u32 = packed_tile * Int64(32 * self.block_tile_bytes // 4) + Int64(chunk) * Int64(4)
                else:
                    source_u32 = self._iq2_xs_descriptor_offset(
                        expert_idx, global_k16, global_n16
                    ) + Int64(tile_chunk) * Int64(4)
                cp_async4_shared_global_pred(
                    b_region + chunk * Int32(16),
                    get_ptr_as_int64(b_i32_flat, source_u32),
                    (chunk < Int32(total_chunks)).to(Int32),
                )
            metadata_addr = get_ptr_as_int64(scales_i32_flat, Int64(0))
            expert_count = Int64(cute.size(b_i32_flat)) // Int64(self.size_k * self.size_n // (4 * self.block_pack_factor))
            base_plane_bytes = expert_count * Int64(self.size_k // self.block_k * self.size_n * 2)
            base_chunks = _covering_count(self.cta_k_blocks * 16, self.block_k) * self.cta_n_blocks * 2
            base_region = b_region + Int32(self.cta_k_blocks * self.cta_n_blocks * self.block_tile_bytes)
            scale_region = base_region + Int32(base_chunks * 16)
            for i in cutlass.range_constexpr(_covering_count(base_chunks, self.cta_threads)):
                chunk = Int32(i * self.cta_threads) + tid
                if chunk < Int32(base_chunks):
                    if cutlass.const_expr(not self.q8 and self.tile_n == 64 and self.tile_k == 128):
                        base_tile = (
                            (Int64(expert_idx) * Int64(self.size_k // self.block_k)
                             + Int64(tile_idx // Int32(2)))
                            * Int64(self.size_n // 64) + Int64(output_n_tile)
                        )
                        base_offset = base_tile * Int64(128) + Int64(chunk) * Int64(16)
                    else:
                        local_tile = chunk // Int32(2)
                        global_k256 = tile_idx * Int32(self.cta_k_blocks) // Int32(self.block_k // 16) + local_tile // Int32(self.cta_n_blocks)
                        global_n16 = output_n_tile * Int32(self.cta_n_blocks) + local_tile % Int32(self.cta_n_blocks)
                        block = (
                            (Int64(expert_idx) * Int64(self.size_k // self.block_k) + Int64(global_k256))
                            * Int64(n16_total) + Int64(global_n16)
                        )
                        base_offset = block * Int64(32) + Int64(chunk % Int32(2)) * Int64(16)
                    cp_async4_shared_global(
                        base_region + chunk * Int32(16),
                        metadata_addr + base_offset,
                    )
            scale_chunks = 0 if self.iq2_xxs or self.q8 else self.cta_k_blocks // 2 * self.cta_n_blocks
            for i in cutlass.range_constexpr(_covering_count(scale_chunks, self.cta_threads)):
                chunk = Int32(i * self.cta_threads) + tid
                if chunk < Int32(scale_chunks):
                    if cutlass.const_expr(self.tile_n == 64 and self.tile_k == 128):
                        scale_chunk = packed_tile * Int64(16) + Int64(chunk)
                    else:
                        global_k32 = tile_idx * Int32(self.cta_k_blocks // 2) + chunk // Int32(self.cta_n_blocks)
                        global_n16 = output_n_tile * Int32(self.cta_n_blocks) + chunk % Int32(self.cta_n_blocks)
                        scale_tile = (
                            (Int64(expert_idx) * Int64(self.size_n // 64)
                             + Int64(global_n16 // Int32(4)))
                            * Int64(self.size_k // 128) + Int64(global_k32 // Int32(4))
                        )
                        scale_chunk = (
                            scale_tile * Int64(16)
                            + Int64(global_k32 % Int32(4)) * Int64(4)
                            + Int64(global_n16 % Int32(4))
                        )
                    cp_async4_shared_global(
                        scale_region + chunk * Int32(16),
                        metadata_addr + base_plane_bytes
                        + scale_chunk * Int64(16),
                    )

        if cutlass.const_expr(self.weight_layout_trellis256):
            t256_n16 = self.size_n // 16
            if cutlass.const_expr(self.weight_layout_trellis256_pair):
                low_bits = 3
                high_bits = 3
                if cutlass.const_expr(
                    self.trellis_pair_kind == "P24"
                    or (
                        self.trellis_pair_dynamic
                        and int(dynamic_pair_override) == 1
                    )
                ):
                    low_bits = 2
                    high_bits = 4
                elif cutlass.const_expr(
                    self.trellis_pair_kind == "P43"
                    or (
                        self.trellis_pair_dynamic
                        and int(dynamic_pair_override) == 2
                    )
                ):
                    low_bits = 4
                    high_bits = 3
                elif cutlass.const_expr(self.trellis_pair_kind == "P44"):
                    low_bits = 4
                    high_bits = 4
                if cutlass.const_expr(self.trellis_rate_axis == "n"):
                    # Preparation swizzles the reference record-major payload
                    # into one fixed-size compact pair span per K16 row.
                    pair_u32_per_k16 = Int32(8 * 8 * (low_bits + high_bits))
                    t256_chunks_per_kt = self.cta_n_blocks * (
                        low_bits + high_bits
                    )
                    t256_total_chunks = self.cta_k_blocks * t256_chunks_per_kt
                    t256_pair_u32 = (self.size_k // 16) * pair_u32_per_k16
                    for i in cutlass.range_constexpr(self.b_sh_wr_iters_var):
                        t256_chunk = Int32(i * self.cta_threads) + tid
                        t256_kt = t256_chunk // Int32(t256_chunks_per_kt)
                        t256_in_kt = (
                            t256_chunk - t256_kt * Int32(t256_chunks_per_kt)
                        )
                        b_dst = (
                            smem_base
                            + Int32(self.sh_b_off * 16)
                            + pipe * Int32(self.b_sh_stage_bytes)
                            + t256_chunk * Int32(16)
                        )
                        pair_base_i64 = Int64(0)
                        if cutlass.const_expr(self.weight_layout_trellis256_proj):
                            # Projection-major FC1 payload: [projection, E,
                            # K16, compact-pair-row].  TP12 has exactly one
                            # 256-channel pair per gate/up projection.
                            pair_plane_u32 = Int64(cute.size(b_i32_flat)) // Int64(2)
                            if cutlass.const_expr(self.trellis_pair_compact_offsets):
                                pair_descriptor = scales_i32_flat[expert_idx].to(Int64)
                                pair_base_i64 = (
                                    Int64(output_n_tile) * pair_plane_u32
                                    + (pair_descriptor >> Int64(1))
                                )
                            else:
                                pair_base_i64 = (
                                    Int64(output_n_tile) * pair_plane_u32
                                    + Int64(expert_idx) * Int64(t256_pair_u32)
                                )
                        else:
                            # Dense/expert-major payload: [E, pair, K16,
                            # compact-pair-row].
                            if cutlass.const_expr(self.trellis_pair_compact_offsets):
                                pair_descriptor = scales_i32_flat[expert_idx].to(Int64)
                                pair_base_i64 = (
                                    pair_descriptor >> Int64(1)
                                ) + Int64(output_n_tile) * Int64(t256_pair_u32)
                            else:
                                pair_base_i64 = (
                                    Int64(expert_idx) * Int64(self.size_n // 256)
                                    + Int64(output_n_tile)
                                ) * Int64(t256_pair_u32)
                        b_src_i64 = (
                            pair_base_i64
                            + (
                                Int64(tile_idx) * Int64(self.cta_k_blocks)
                                + Int64(t256_kt)
                            )
                            * Int64(pair_u32_per_k16)
                            + Int64(t256_in_kt) * Int64(4)
                        )
                        cp_async4_shared_global_pred(
                            b_dst,
                            get_ptr_as_int64(b_i32_flat, b_src_i64),
                            (t256_chunk < Int32(t256_total_chunks)).to(Int32),
                        )
                else:
                    # The two source records remain contiguous in both payload
                    # and physical K.  Reserve K4-sized shared slots, then copy
                    # each compact K2/K3/K4 source tile into its K16 slot's
                    # prefix.
                    max_chunks_per_kt = self.cta_n_blocks * 8
                    t256_expert_u32 = (
                        (self.size_k // 16)
                        * t256_n16
                        * 4
                        * (low_bits + high_bits)
                    )
                    low_record_u32 = Int32(8 * t256_n16 * 8 * low_bits)
                    for i in cutlass.range_constexpr(self.b_sh_wr_iters_var):
                        t256_chunk = Int32(i * self.cta_threads) + tid
                        t256_kt = t256_chunk // Int32(max_chunks_per_kt)
                        t256_in_kt = (
                            t256_chunk - t256_kt * Int32(max_chunks_per_kt)
                        )
                        logical_k16 = (
                            tile_idx * Int32(self.cta_k_blocks) + t256_kt
                        )
                        high_record = (logical_k16 >= Int32(8)).to(Int32)
                        local_k16 = logical_k16 - high_record * Int32(8)
                        record_bits = Int32(low_bits)
                        if high_record != Int32(0):
                            record_bits = Int32(high_bits)
                        record_base_u32 = high_record * low_record_u32
                        actual_chunks = Int32(self.cta_n_blocks * 2) * record_bits
                        b_dst = (
                            smem_base
                            + Int32(self.sh_b_off * 16)
                            + pipe * Int32(self.b_sh_stage_bytes)
                            + t256_chunk * Int32(16)
                        )
                        expert_base_u32 = Int64(t256_expert_u32) * Int64(expert_idx)
                        if cutlass.const_expr(self.trellis_pair_compact_offsets):
                            pair_descriptor = scales_i32_flat[expert_idx].to(Int64)
                            expert_base_u32 = pair_descriptor >> Int64(1)
                        b_src_i64 = (
                            expert_base_u32
                            + Int64(record_base_u32)
                            + Int64(local_k16)
                            * Int64(t256_n16 * 8)
                            * Int64(record_bits)
                            + Int64(output_n_tile)
                            * Int64(self.cta_n_blocks * 8)
                            * Int64(record_bits)
                            + Int64(t256_in_kt) * Int64(4)
                        )
                        cp_async4_shared_global_pred(
                            b_dst,
                            get_ptr_as_int64(b_i32_flat, b_src_i64),
                            (t256_in_kt < actual_chunks).to(Int32),
                        )
            else:
                t256_tile_u32 = 8 * self.trellis_bits
                t256_expert_u32 = (
                    (self.size_k // 16) * t256_n16 * t256_tile_u32
                )
                t256_chunks_per_kt = self.cta_n_blocks * (2 * self.trellis_bits)
                t256_total_chunks = self.cta_k_blocks * t256_chunks_per_kt
                for i in cutlass.range_constexpr(self.b_sh_wr_iters_var):
                    t256_chunk = Int32(i * self.cta_threads) + tid
                    t256_kt = t256_chunk // Int32(t256_chunks_per_kt)
                    t256_in_kt = (
                        t256_chunk - t256_kt * Int32(t256_chunks_per_kt)
                    )
                    b_dst = (
                        smem_base
                        + Int32(self.sh_b_off * 16)
                        + pipe * Int32(self.b_sh_stage_bytes)
                        + t256_chunk * Int32(16)
                    )
                    if cutlass.const_expr(self.weight_layout_trellis256_proj):
                        t256_half_n16 = t256_n16 // 2
                        t256_out_n16 = output_n_tile * Int32(self.cta_n_blocks)
                        t256_proj = (
                            t256_out_n16 >= Int32(t256_half_n16)
                        ).to(Int32)
                        t256_local_n16 = (
                            t256_out_n16 - t256_proj * Int32(t256_half_n16)
                        )
                        t256_proj_expert_u32 = (
                            (self.size_k // 16)
                            * t256_half_n16
                            * t256_tile_u32
                        )
                        # Projection-major W13 is physically [2, E, ...].
                        t256_plane_u32 = Int64(cute.size(b_i32_flat)) // Int64(2)
                        b_src_i64 = (
                            Int64(t256_proj) * t256_plane_u32
                            + Int64(expert_idx) * Int64(t256_proj_expert_u32)
                            + (
                                Int64(tile_idx) * Int64(self.cta_k_blocks)
                                + Int64(t256_kt)
                            )
                            * Int64(t256_half_n16 * t256_tile_u32)
                            + Int64(t256_local_n16) * Int64(t256_tile_u32)
                            + Int64(t256_in_kt) * Int64(4)
                        )
                    else:
                        b_src_i64 = (
                            Int64(t256_expert_u32) * Int64(expert_idx)
                            + (
                                Int64(tile_idx) * Int64(self.cta_k_blocks)
                                + Int64(t256_kt)
                            )
                            * Int64(t256_n16 * t256_tile_u32)
                            + Int64(output_n_tile)
                            * Int64(self.cta_n_blocks * t256_tile_u32)
                            + Int64(t256_in_kt) * Int64(4)
                        )
                    cp_async4_shared_global_pred(
                        b_dst,
                        get_ptr_as_int64(b_i32_flat, b_src_i64),
                        (t256_chunk < Int32(t256_total_chunks)).to(Int32),
                    )

        for i in cutlass.range_constexpr(
            0 if self.b_region_variable else self.b_sh_wr_iters
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
                b_dst = (
                    smem_base
                    + Int32(self.sh_b_off * 16)
                    + pipe * Int32(self.b_sh_stage * 16)
                    + self._modelopt_smem_permuted_offset(
                        (Int32(i * self.cta_threads) + tid) * Int32(16)
                    )
                )
                self._stage_b_tile_modelopt_native(
                    b_i32_flat,
                    b_dst,
                    expert_idx,
                    output_n_tile,
                    tile_idx,
                    Int32(i * self.cta_threads) + tid,
                )

        # trellis_t256 has no per-weight scale and its register-load arm returns
        # before touching scale SMEM.  Const-expr-elide the otherwise dead HBM
        # reads so every layer can share a four-byte aligned dummy scale tensor
        # instead of retaining 54 MiB of packed ones.
        if cutlass.const_expr(self.native_nvfp4_scales):
            self._stage_modelopt_scales(
                scales_i32_flat, smem_base, tid, pipe, expert_idx,
                output_n_tile, tile_idx,
            )
        elif cutlass.const_expr(not self.weight_layout_trellis256 and not self.weight_layout_block):
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
                        st_shared_v4_u32(
                            s_dst, Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                        )
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
        a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pipe: Int32,
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
        dynamic_pair_override: cutlass.Constexpr[int],
    ):
        if cutlass.const_expr(kk == self.b_sh_wr_iters - 2):
            self._prefetch_lookahead_tile(
                a_bf16_flat,
                a_alt_bf16_flat,
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
                dynamic_pair_override,
            )

    @cute.jit
    def _prefetch_initial_tiles(
        self,
        a_bf16_flat: cute.Tensor,
        a_alt_bf16_flat: cute.Tensor,
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
        dynamic_pair_override: cutlass.Constexpr[int],
    ):
        for pipe in cutlass.range_constexpr(self.stages - 1):
            if Int32(pipe) < k_tiles:
                self._stage_k_tile_async(
                    a_bf16_flat,
                    a_alt_bf16_flat,
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
                    dynamic_pair_override,
                )
            else:
                cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(self.stages - 2)
        cute.arch.sync_threads()

    @cute.jit
    def _prefetch_lookahead_tile(
        self,
        a_bf16_flat: cute.Tensor,
        a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pipe: Int32,
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
        dynamic_pair_override: cutlass.Constexpr[int],
    ):
        fetch_tile = tile_idx + Int32(self.stages - 1)
        if fetch_tile < k_tiles:
            self._stage_k_tile_async(
                a_bf16_flat,
                a_alt_bf16_flat,
                b_i32_flat,
                scales_i32_flat,
                smem_base,
                tid,
                Int32((pipe + self.stages - 1) % self.stages),
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
                dynamic_pair_override,
            )
        else:
            cute.arch.cp_async_commit_group()
        if cutlass.const_expr(not self.weight_layout_block):
            cute.arch.cp_async_wait_group(self.stages - 2)
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
                true_idx = Int64(route_index) * Int64(c_gl_stride) + Int64(
                    c_gl_wr % c_gl_stride
                )
                q0, q1, q2, q3 = ld_shared_v4_u32(
                    self._int4_addr(smem_base, Int32(self.sh_red_off) + c_sh_rd)
                )
                if cutlass.const_expr(self.mul_topk_weights):
                    scale_bf2 = ld_shared_u32(
                        smem_base
                        + Int32(self.sh_topk_off * 16)
                        + row * Int32(4)
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
                    col_word = c_gl_wr % c_gl_stride
                    if cutlass.const_expr(self.fused_sum_fp32):
                        out_elem = (
                            Int64(token_idx) * Int64(self.size_n)
                            + Int64(col_word) * Int64(8)
                        )
                        out_addr = get_ptr_as_int64(c_bf16_flat, out_elem)
                        q00, q01 = self._elem2_to_f32x2(q0)
                        q10, q11 = self._elem2_to_f32x2(q1)
                        q20, q21 = self._elem2_to_f32x2(q2)
                        q30, q31 = self._elem2_to_f32x2(q3)
                        red_add_global_v4_f32(out_addr, q00, q01, q10, q11)
                        red_add_global_v4_f32(
                            out_addr + Int64(16), q20, q21, q30, q31
                        )
                    else:
                        out_idx = Int64(token_idx) * Int64(c_gl_stride) + Int64(
                            col_word
                        )
                        out_addr = get_ptr_as_int64(
                            c_bf16_flat, out_idx * Int64(8)
                        )
                        red_add_global_bf16x2(out_addr, q0)
                        red_add_global_bf16x2(out_addr + Int64(4), q1)
                        red_add_global_bf16x2(out_addr + Int64(8), q2)
                        red_add_global_bf16x2(out_addr + Int64(12), q3)
                else:
                    st_global_v4_u32(
                        get_ptr_as_int64(c_bf16_flat, true_idx * Int64(8)),
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
                true_idx = Int64(route_index) * Int64(c_gl_stride) + Int64(col_word)
                q0, q1, q2, q3 = ld_shared_v4_u32(
                    self._int4_addr(smem_base, Int32(self.sh_red_off) + c_sh_rd)
                )
                if cutlass.const_expr(self.mul_topk_weights):
                    scale_bf2 = ld_shared_u32(
                        smem_base
                        + Int32(self.sh_topk_off * 16)
                        + row * Int32(4)
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
                    if cutlass.const_expr(self.fused_sum_fp32):
                        out_elem = (
                            Int64(token_idx) * Int64(self.size_n)
                            + Int64(col_word) * Int64(8)
                        )
                        out_addr = get_ptr_as_int64(c_bf16_flat, out_elem)
                        q00, q01 = self._elem2_to_f32x2(q0)
                        q10, q11 = self._elem2_to_f32x2(q1)
                        q20, q21 = self._elem2_to_f32x2(q2)
                        q30, q31 = self._elem2_to_f32x2(q3)
                        red_add_global_v4_f32(out_addr, q00, q01, q10, q11)
                        red_add_global_v4_f32(
                            out_addr + Int64(16), q20, q21, q30, q31
                        )
                    else:
                        out_idx = (
                            Int64(token_idx) * Int64(c_gl_stride)
                            + Int64(col_word)
                        )
                        out_addr = get_ptr_as_int64(
                            c_bf16_flat, out_idx * Int64(8)
                        )
                        red_add_global_bf16x2(out_addr, q0)
                        red_add_global_bf16x2(out_addr + Int64(4), q1)
                        red_add_global_bf16x2(out_addr + Int64(8), q2)
                        red_add_global_bf16x2(out_addr + Int64(12), q3)
                else:
                    st_global_v4_u32(
                        get_ptr_as_int64(c_bf16_flat, true_idx * Int64(8)),
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
                if cutlass.const_expr(
                    self.weight_layout_trellis256_pair
                    and self.trellis_rate_axis == "n"
                ):
                    # MMA register assignment is [L0,L1,H0,H1] per warp for
                    # balanced P24 work.  Scatter the four N16 fragments back
                    # to contiguous [L(0..7),H(0..7)] record order in output
                    # SMEM.  This is a bijection and occurs before the outer
                    # H128, so the public channel/transform semantics do not
                    # change.
                    warp_n = tid // Int32(32)
                    semantic_n16 = Int32(2) * warp_n + Int32(jj)
                    if cutlass.const_expr(jj >= 2):
                        semantic_n16 = (
                            Int32(8)
                            + Int32(2) * warp_n
                            + Int32(jj - 2)
                        )
                    compute_n16 = Int32(4) * warp_n + Int32(jj)
                    wr += Int32(16) * (semantic_n16 - compute_n16)
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
                        acc0,
                        smem_base,
                        c_sh_wr,
                        c_sh_stride,
                        tid // Int32(32),
                        write_scale,
                    )
                elif cutlass.const_expr(mb == 1):
                    self._store_tile_large_m_block(
                        acc1,
                        smem_base,
                        c_sh_wr,
                        c_sh_stride,
                        tid // Int32(32),
                        write_scale,
                    )
                elif cutlass.const_expr(mb == 2):
                    self._store_tile_large_m_block(
                        acc2,
                        smem_base,
                        c_sh_wr,
                        c_sh_stride,
                        tid // Int32(32),
                        write_scale,
                    )
                else:
                    self._store_tile_large_m_block(
                        acc3,
                        smem_base,
                        c_sh_wr,
                        c_sh_stride,
                        tid // Int32(32),
                        write_scale,
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
        warp_n: Int32,
        write_scale: cutlass.Float32,
    ):
        for jj in cutlass.range_constexpr(4):
            wr = c_sh_wr + Int32(8 * jj)
            if cutlass.const_expr(
                self.weight_layout_trellis256_pair
                and self.trellis_rate_axis == "n"
            ):
                # Match the M<=8 epilogue: restore the reference record order
                # from the balanced LLHH per-warp MMA assignment before H128.
                semantic_n16 = Int32(2) * warp_n + Int32(jj)
                if cutlass.const_expr(jj >= 2):
                    semantic_n16 = (
                        Int32(8)
                        + Int32(2) * warp_n
                        + Int32(jj - 2)
                    )
                compute_n16 = Int32(4) * warp_n + Int32(jj)
                wr += Int32(8) * (semantic_n16 - compute_n16)
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
        fc2_moe_block_size: int | None = None,
        fc2_schedule_route_block_factor: int = 1,
        element_dtype: str = "bf16",
        fast_math: bool = True,
        swiglu_limit: float | None = None,
        swiglu_alpha: float | None = None,
        swiglu_beta: float | None = None,
        weight_layout: str = "packed",
        scale_format: str = "e4m3_k16",
        w13_layout: str = "w13",
        trellis_bits: int = 3,
        trellis_codebook: str = LUT_E4M3,
        fc1_trellis_pair_kind: str | None = None,
        fc2_trellis_pair_kind: str | None = None,
        direct_topk_routes: bool = False,
        use_expert_map: bool = False,
        tc_decode_fused_sum: bool = False,
        prefill_fused_sum_fp32: bool = False,
        tc_zero_output: bool = True,
        collect_activation_amax: bool = False,
        schedule_whole_tiles: bool = False,
        intermediate_rotation: bool = False,
        full_rotation: bool = False,
        intermediate_hadamard: bool = False,
        rotation_input_dtype: str = "fp16",
        broadcast_suh: bool = False,
        pipeline_stages: int | None = None,
        skip_empty_m_blocks: bool | None = None,
        trellis_decode_table: str = "auto",
        small_m_occupancy: int | None = None,
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
        elif weight_layout == "trellis_t256":
            if w13_layout not in _TRELLIS256_W13_LAYOUTS:
                raise ValueError(f"unsupported trellis_t256 w13_layout {w13_layout!r}")
        else:
            w13_layout = "packed"
        self.tc_decode_fused_sum = bool(tc_decode_fused_sum)
        self.prefill_fused_sum_fp32 = bool(prefill_fused_sum_fp32)
        if self.tc_decode_fused_sum and self.prefill_fused_sum_fp32:
            raise ValueError(
                "TC-decode and large-M FP32 route reduction are mutually exclusive"
            )
        # When two TC-decode launches share one pre-zeroed output, only the
        # first must zero it. Default True preserves single-launch behavior.
        self.tc_zero_output = bool(tc_zero_output)
        self.collect_activation_amax = bool(collect_activation_amax)
        if self.collect_activation_amax and bool(direct_topk_routes):
            raise ValueError("activation amax collection requires route-packed W4A16")
        if self.collect_activation_amax and (
            self.tc_decode_fused_sum or self.prefill_fused_sum_fp32
        ):
            raise ValueError(
                "activation amax collection is incompatible with fused route reduction"
            )
        if self.tc_decode_fused_sum and not bool(direct_topk_routes):
            raise ValueError("tc_decode_fused_sum requires direct_topk_routes")
        if self.tc_decode_fused_sum and element_dtype != "bf16":
            raise ValueError("tc_decode_fused_sum currently requires bf16 activations")
        if self.prefill_fused_sum_fp32 and element_dtype != "bf16":
            raise ValueError("prefill_fused_sum_fp32 requires bf16 activations")
        if self.prefill_fused_sum_fp32 and int(size_m) <= _TC_DECODE_MAX_M:
            raise ValueError(
                "prefill_fused_sum_fp32 requires a token capacity above the decode range"
            )
        fc1_cols = int(intermediate_size) * (2 if is_gated else 1)
        routed_rows = int(size_m) * int(top_k)
        self.size_m = int(size_m)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.fc1_cols = int(fc1_cols)
        self.num_experts = int(num_experts)
        # Compact tier expert counts are artifact data, not launch geometry.
        # Keep the exact count in the plan/result metadata, but exclude it from
        # code specialization for the packed MXFP4 and EXL3 serving layouts.
        # All E-sized tensor views are reconstructed from a runtime scalar at
        # the launch boundary.
        self.dynamic_num_experts = weight_layout in {
            "packed",
            "trellis_t256",
            "iq2_xs",
            "iq2_xxs",
            "q8_0",
        }
        self.top_k = int(top_k)
        self.moe_block_size = int(moe_block_size)
        # Stripe split-K spreads each mn-tile's K range across many CTAs for
        # decode-heavy small-M phases. It is incompatible with whole-tile
        # scheduling and grouped FC2 route subtiles.
        self.small_m_splitk = (
            weight_layout in BLOCK_CODECS and moe_block_size == 8
        ) or _w4a16_small_m_splitk_enabled()
        if self.small_m_splitk:
            schedule_whole_tiles = False
            fc2_schedule_route_block_factor = 1
        self.fc2_moe_block_size = int(
            moe_block_size if fc2_moe_block_size is None else fc2_moe_block_size
        )
        self.fc2_schedule_route_block_factor = int(fc2_schedule_route_block_factor)
        if (
            self.fc2_moe_block_size not in _ALLOWED_ROUTED_SIZES
            or self.moe_block_size % self.fc2_moe_block_size != 0
        ):
            raise ValueError(
                "FC2 route subtile must be an allowed divisor of the packed "
                f"route block: packed={self.moe_block_size}, "
                f"fc2={self.fc2_moe_block_size}"
            )
        expected_fc2_schedule_factor = self.moe_block_size // self.fc2_moe_block_size
        if (
            self.fc2_schedule_route_block_factor < 1
            or expected_fc2_schedule_factor % self.fc2_schedule_route_block_factor != 0
        ):
            raise ValueError(
                "FC2 schedule factor must divide one packed route block: "
                f"factor={self.fc2_schedule_route_block_factor}, "
                f"maximum={expected_fc2_schedule_factor}"
            )
        self.activation = activation
        self.activation_is_gated = is_gated
        self.activation_is_situ = activation == SITU
        self.activation_is_swigluoai = activation == SWIGLUOAI_UNINTERLEAVE
        self.has_swiglu_limit = swiglu_limit is not None
        self.swiglu_limit = 0.0 if swiglu_limit is None else swiglu_limit
        self.swiglu_alpha = float(swiglu_alpha)
        self.swiglu_beta = float(swiglu_beta)
        self.weight_layout = weight_layout
        self.trellis_bits = int(trellis_bits)
        self.trellis_codebook = str(trellis_codebook).lower()
        if self.weight_layout == "trellis_t256":
            if self.trellis_codebook not in _TRELLIS256_CODEBOOKS:
                raise ValueError(
                    "trellis_t256 codebook must be one of "
                    f"{sorted(_TRELLIS256_CODEBOOKS)}, got {self.trellis_codebook!r}"
                )
        self.fc1_trellis_pair_kind = (
            None
            if fc1_trellis_pair_kind is None
            else str(fc1_trellis_pair_kind).upper()
        )
        self.fc2_trellis_pair_kind = (
            None
            if fc2_trellis_pair_kind is None
            else str(fc2_trellis_pair_kind).upper()
        )
        if (self.fc1_trellis_pair_kind is None) != (
            self.fc2_trellis_pair_kind is None
        ):
            raise ValueError(
                "fused trellis pair weights require both FC1 and FC2 pair kinds"
            )
        if self.fc1_trellis_pair_kind is not None:
            if weight_layout != "trellis_t256":
                raise ValueError("fused trellis pairs require trellis_t256 weights")
            if self.trellis_bits != 3:
                raise ValueError(
                    "fused trellis pairs require the trellis_bits=3 base "
                    "specialization"
                )
            dynamic_kinds = {"PDYNAMIC", "P33_P43"}
            static_kinds = {"P24", "P33", "P43", "P44"}
            if (
                self.fc1_trellis_pair_kind in dynamic_kinds
                or self.fc2_trellis_pair_kind in dynamic_kinds
            ):
                if self.fc1_trellis_pair_kind != self.fc2_trellis_pair_kind:
                    raise ValueError(
                        "dynamic fused trellis pair kinds must match"
                    )
            elif (
                self.fc1_trellis_pair_kind not in static_kinds
                or self.fc2_trellis_pair_kind not in static_kinds
            ):
                raise ValueError("unsupported static fused trellis pair kind")
        self.scale_format = scale_format
        self.w13_layout = w13_layout
        self.apply_router_weight_on_input = bool(apply_router_weight_on_input)
        self.zero_fc2_output = bool(zero_fc2_output)
        self.element_dtype = element_dtype
        self.is_fp16 = element_dtype == "fp16"
        self.fast_math = bool(fast_math)
        self.direct_topk_routes = bool(direct_topk_routes)
        self.use_expert_map = bool(use_expert_map)
        if self.use_expert_map and not self.direct_topk_routes:
            raise ValueError("use_expert_map requires direct_topk_routes")
        self.schedule_whole_tiles = bool(
            (schedule_whole_tiles or weight_layout == "trellis_t256")
            and not self.small_m_splitk
        )
        self.intermediate_rotation = bool(intermediate_rotation)
        if self.intermediate_rotation:
            if weight_layout != "trellis_t256":
                raise ValueError(
                    "intermediate_rotation is only supported for trellis_t256"
                )
            if not is_gated or self.activation_is_swigluoai or self.has_swiglu_limit:
                raise ValueError(
                    "intermediate_rotation requires unclamped gated silu or situ "
                    "(no swiglu limit/oai)"
                )
            if int(intermediate_size) % 128 != 0:
                raise ValueError(
                    "intermediate_rotation requires intermediate_size % 128 == 0"
                )
        self.full_rotation = bool(full_rotation)
        self.intermediate_hadamard = bool(intermediate_hadamard)
        # suh tables hold one row shared by every expert: index them with a
        # zero expert stride.
        self.broadcast_suh = bool(broadcast_suh)
        self.rotation_input_dtype = str(rotation_input_dtype)
        self.rotation_input_is_fp16 = self.rotation_input_dtype == "fp16"
        if self.full_rotation:
            if not self.intermediate_rotation:
                raise ValueError("full_rotation requires intermediate_rotation")
            if element_dtype != "fp16":
                raise ValueError("full_rotation requires fp16 GEMM operands")
            if self.rotation_input_dtype not in {"bf16", "fp16"}:
                raise ValueError("full_rotation input dtype must be 'bf16' or 'fp16'")
            if self.tc_decode_fused_sum or self.prefill_fused_sum_fp32:
                raise ValueError("full_rotation is incompatible with fused route reduction")
            if self.apply_router_weight_on_input:
                raise ValueError(
                    "full_rotation applies router weights only in the fp32 top-k sum"
                )
            if int(hidden_size) % 128 != 0:
                raise ValueError("full_rotation requires hidden_size % 128 == 0")
        if self.intermediate_hadamard:
            if not self.full_rotation:
                raise ValueError("intermediate Hadamard requires full rotation")
            if int(hidden_size) % 512 != 0:
                raise ValueError("intermediate Hadamard requires hidden_size % 512 == 0")
            if int(intermediate_size) % 128 != 0:
                raise ValueError(
                    "intermediate Hadamard requires intermediate_size % 128 == 0"
                )
        self.dual_a = bool(
            self.intermediate_rotation
            and weight_layout == "trellis_t256"
            and w13_layout == "trellis_t256_proj"
        )
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
            trellis_bits=self.trellis_bits,
            trellis_codebook=self.trellis_codebook,
            trellis_pair_kind=self.fc1_trellis_pair_kind,
            trellis_rate_axis=(
                "n" if self.fc1_trellis_pair_kind is not None else None
            ),
            source_n_rotation=fc1_source_n_rotation,
            single_token_route_fast_path=size_m == 1 and not self.direct_topk_routes,
            direct_topk_routes=self.direct_topk_routes,
            dual_a=self.dual_a,
            route_major_a=self.full_rotation,
            schedule_whole_tiles=self.schedule_whole_tiles,
            dynamic_num_experts=self.dynamic_num_experts,
            pipeline_stages=pipeline_stages,
            skip_empty_m_blocks=skip_empty_m_blocks,
            small_m_occupancy=small_m_occupancy,
        )
        self.fc2 = W4A16GemmKernel(
            size_m=routed_rows,
            size_n=hidden_size,
            size_k=intermediate_size,
            num_experts=num_experts,
            top_k=1,
            mul_topk_weights=(
                not bool(apply_router_weight_on_input) and not self.full_rotation
            ),
            tile_n=fc2_tile_n,
            tile_k=fc2_tile_k,
            moe_block_size=self.fc2_moe_block_size,
            max_m_blocks=(
                max_m_blocks * self.moe_block_size // self.fc2_moe_block_size
            ),
            element_dtype=element_dtype,
            weight_layout=weight_layout,
            scale_format=scale_format,
            w13_layout=("packed" if weight_layout == "trellis_t256" else w13_layout),
            trellis_bits=self.trellis_bits,
            trellis_codebook=self.trellis_codebook,
            trellis_pair_kind=self.fc2_trellis_pair_kind,
            trellis_rate_axis=(
                "k" if self.fc2_trellis_pair_kind is not None else None
            ),
            single_token_route_fast_path=size_m == 1 and not self.direct_topk_routes,
            direct_topk_routes=self.direct_topk_routes,
            fused_topk_sum=(
                (self.tc_decode_fused_sum and weight_layout not in BLOCK_CODECS)
                or self.prefill_fused_sum_fp32
            ),
            fused_sum_fp32=self.prefill_fused_sum_fp32,
            fused_sum_topk=int(top_k),
            schedule_whole_tiles=self.schedule_whole_tiles,
            dynamic_num_experts=self.dynamic_num_experts,
            schedule_route_block_factor=self.fc2_schedule_route_block_factor,
            pipeline_stages=pipeline_stages,
            skip_empty_m_blocks=skip_empty_m_blocks,
            small_m_occupancy=small_m_occupancy,
        )
        self.cta_threads = max(self.fc1.cta_threads, self.fc2.cta_threads)
        if self.fc1.cta_threads != self.fc2.cta_threads:
            raise ValueError(
                "fused W4A16 kernel expects matching FC1/FC2 thread counts"
            )
        self.sms = self.fc1.sms
        self.blocks_per_sm = min(self.fc1.blocks_per_sm, self.fc2.blocks_per_sm)
        self.shared_words = max(self.fc1.shared_words, self.fc2.shared_words)
        self.iq2_lut_bytes = block_codec(weight_layout).lut_bytes(selectors=True) if weight_layout in BLOCK_CODECS else 0
        self.iq2_xs_lut_off = self.shared_words * 4
        if self.weight_layout in IQ2_CODECS:
            self.shared_words += self.iq2_lut_bytes // 4
            self.fc1.iq2_xs_smem_lut = True
            self.fc2.iq2_xs_smem_lut = True
        if trellis_decode_table not in {"auto", "compact", "full"}:
            raise ValueError("trellis decode table must be auto, compact, or full")
        compact_smem = (
            self.trellis_codebook == LUT_E4M3
            and _lut_e4m3_smem_enabled()
        )
        lut_offset = (self.shared_words * 4 + 15) // 16 * 16
        # The cooperative grid must retain its occupancy contract. A direct
        # table is eligible only when it fits without reducing resident CTAs.
        direct_eligible = (
            self.weight_layout == "trellis_t256"
            and self.trellis_codebook == LUT_E4M3
            and self.intermediate_hadamard
            and self.trellis_bits == 2
            and self.fc1_trellis_pair_kind is None
            and self.fc2_trellis_pair_kind is None
            and self.blocks_per_sm == 1
            and lut_offset + 65536 <= self.fc1.max_shared_mem
        )
        if trellis_decode_table == "full" and not direct_eligible:
            raise ValueError(
                "full trellis decode table is incompatible with the planned kernel geometry"
            )
        self.trellis_direct_lut = direct_eligible and (
            trellis_decode_table == "full"
            or (trellis_decode_table == "auto" and compact_smem and size_m <= 16)
        )
        self.fc1.trellis_direct_lut = self.trellis_direct_lut
        self.fc2.trellis_direct_lut = self.trellis_direct_lut
        self.lut_e4m3_smem = compact_smem or self.trellis_direct_lut
        self.trellis_lut_smem_bytes = (
            65536 if self.trellis_direct_lut else _LUT_E4M3_SMEM_REGION_BYTES
        )
        self.lut_e4m3_smem_off = 0
        if self.lut_e4m3_smem:
            self.lut_e4m3_smem_off = (
                self.shared_words * 4 + 15
            ) // 16 * 16
            self.shared_words = (
                self.lut_e4m3_smem_off
                + self.trellis_lut_smem_bytes
            ) // 4
            if self.shared_words * 4 > self.fc1.max_shared_mem:
                raise ValueError(f"full trellis decode table requires {self.shared_words * 4} bytes")
            self.fc1.lut_e4m3_smem = True
            self.fc2.lut_e4m3_smem = True
        self.barrier_count_off = self.sms * 4
        self.barrier_sense_off = self.sms * 4 + 1

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (
            self.hidden_size,
            self.intermediate_size,
            self.fc1_cols,
            None if self.dynamic_num_experts else self.num_experts,
            self.dynamic_num_experts,
            self.top_k,
            self.activation,
            self.activation_is_gated,
            self.activation_is_situ,
            self.activation_is_swigluoai,
            self.has_swiglu_limit,
            self.swiglu_limit,
            self.swiglu_alpha,
            self.swiglu_beta,
            self.weight_layout,
            self.trellis_bits,
            self.trellis_codebook,
            self.fc1_trellis_pair_kind,
            self.fc2_trellis_pair_kind,
            self.scale_format,
            self.apply_router_weight_on_input,
            self.zero_fc2_output,
            self.element_dtype,
            self.fast_math,
            self.direct_topk_routes,
            self.use_expert_map,
            self.tc_zero_output,
            self.collect_activation_amax,
            self.intermediate_rotation,
            self.dual_a,
            self.full_rotation,
            self.intermediate_hadamard,
            self.broadcast_suh,
            self.rotation_input_dtype,
            self.lut_e4m3_smem,
            self.trellis_direct_lut,
            self.small_m_splitk,
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
    def _emit_expert_map_tile(
        self,
        is_fc1: cutlass.Constexpr,
        a_bf16_flat: cute.Tensor,
        a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        global_topk_ids_i32_flat: cute.Tensor,
        expert_map_i32_flat: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        trellis_lut_addr: Int64,
        smem_base: Int32,
        tid: Int32,
        active_size_m: Int32,
        weight_num_experts: Int32,
        route_num_experts: Int32,
        route_block_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
    ):
        # Direct decode keeps the router's global ids in route-major order.
        # Resolve the compact weight row at the point of use, and reject both
        # invalid global ids and unmapped (-1) entries before any weight access.
        global_expert = global_topk_ids_i32_flat[route_block_idx].to(Int32)
        local_expert = Int32(-1)
        if global_expert >= Int32(0) and global_expert < route_num_experts:
            local_expert = expert_map_i32_flat[global_expert].to(Int32)
        if local_expert >= Int32(0) and local_expert < weight_num_experts:
            if cutlass.const_expr(is_fc1):
                self.fc1._run_tile(
                    a_bf16_flat,
                    a_alt_bf16_flat,
                    b_i32_flat,
                    c_bf16_flat,
                    scales_i32_flat,
                    global_scale,
                    global_topk_ids_i32_flat,
                    topk_weights_flat,
                    c_tmp_f32_flat,
                    locks_i32_flat,
                    trellis_lut_addr,
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
                self.fc2._run_tile(
                    a_bf16_flat,
                    a_alt_bf16_flat,
                    b_i32_flat,
                    c_bf16_flat,
                    scales_i32_flat,
                    global_scale,
                    global_topk_ids_i32_flat,
                    topk_weights_flat,
                    c_tmp_f32_flat,
                    locks_i32_flat,
                    trellis_lut_addr,
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
        a_alt_bf16_ptr: cute.Pointer,
        rotation_input_ptr: cute.Pointer,
        w13_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        w13_elements: cutlass.Int64,
        w2_elements: cutlass.Int64,
        fc1_bf16_flat: cute.Tensor,
        activated_bf16_flat: cute.Tensor,
        fc2_bf16_flat: cute.Tensor,
        w13_scales_ptr: cute.Pointer,
        w2_scales_ptr: cute.Pointer,
        w13_global_scale_ptr: cute.Pointer,
        w2_global_scale_ptr: cute.Pointer,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        activation_amax_flat: cute.Tensor,
        layer_idx: cutlass.Int32,
        topk_weights_ptr: cute.Pointer,
        fc1_c_tmp_f32_flat: cute.Tensor,
        fc2_c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        rot_scales_ptr: cute.Pointer,
        suh_gate_ptr: cute.Pointer,
        suh_up_ptr: cute.Pointer,
        expert_map_ptr: cute.Pointer,
        fc1_trellis_lut_ptr: cute.Pointer,
        fc2_trellis_lut_ptr: cute.Pointer,
        weight_num_experts: cutlass.Int32,
        route_num_experts: cutlass.Int32,
        active_m: cutlass.Int32,
        grid_x: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        expert_count = Int64(weight_num_experts)
        w13_i32_flat = cute.make_tensor(
            w13_ptr,
            layout=cute.make_layout((Int64(w13_elements),), stride=(1,)),
        )
        w2_i32_flat = cute.make_tensor(
            w2_ptr,
            layout=cute.make_layout((Int64(w2_elements),), stride=(1,)),
        )
        w13_metadata_elements = expert_count * Int64(
            self.fc1.size_k // self.fc1.block_k * self.fc1.size_n
            * self.fc1.block_metadata_bytes // 4
        )
        w2_metadata_elements = expert_count * Int64(
            self.fc2.size_k // self.fc2.block_k * self.fc2.size_n
            * self.fc2.block_metadata_bytes // 4
        )
        if cutlass.const_expr(not self.fc1.weight_layout_block):
            w13_metadata_elements = (
                expert_count
                if cutlass.const_expr(self.fc1.trellis_pair_compact_offsets)
                else expert_count
                * Int64(self.fc1.scale_k_groups)
                * Int64(self.fc1.scale_size_n // 4)
            )
        if cutlass.const_expr(not self.fc2.weight_layout_block):
            w2_metadata_elements = (
                expert_count
                if cutlass.const_expr(self.fc2.trellis_pair_compact_offsets)
                else expert_count
                * Int64(self.fc2.scale_k_groups)
                * Int64(self.fc2.scale_size_n // 4)
            )
        if cutlass.const_expr(self.fc1.native_nvfp4_scales):
            w13_metadata_elements = expert_count * Int64(
                _covering_count(self.fc1.size_n, 128) * 128
                * _covering_count(self.fc1.size_k // 16, 4)
            )
            w2_metadata_elements = expert_count * Int64(
                _covering_count(self.fc2.size_n, 128) * 128
                * _covering_count(self.fc2.size_k // 16, 4)
            )
        w13_scales_i32_flat = cute.make_tensor(
            w13_scales_ptr,
            layout=cute.make_layout((w13_metadata_elements,), stride=(1,)),
        )
        w2_scales_i32_flat = cute.make_tensor(
            w2_scales_ptr,
            layout=cute.make_layout((w2_metadata_elements,), stride=(1,)),
        )
        w13_global_scale = cute.make_tensor(
            w13_global_scale_ptr,
            layout=cute.make_layout((expert_count,), stride=(1,)),
        )
        w2_global_scale = cute.make_tensor(
            w2_global_scale_ptr,
            layout=cute.make_layout((expert_count,), stride=(1,)),
        )
        rot_rows = Int64(active_m) * Int64(self.top_k)
        if cutlass.const_expr(self.full_rotation):
            rot_rows = expert_count
        rot_width = 3 * self.intermediate_size
        if cutlass.const_expr(self.intermediate_hadamard):
            rot_width = 6 * self.intermediate_size
        rot_scales_flat = cute.make_tensor(
            rot_scales_ptr,
            layout=cute.make_layout(
                (rot_rows * Int64(rot_width),),
                stride=(1,),
            ),
        )
        suh_rows = expert_count
        if cutlass.const_expr(self.broadcast_suh):
            suh_rows = Int64(1)
        suh_gate_flat = cute.make_tensor(
            suh_gate_ptr,
            layout=cute.make_layout(
                (suh_rows * Int64(self.hidden_size),),
                stride=(1,),
            ),
        )
        suh_up_flat = cute.make_tensor(
            suh_up_ptr,
            layout=cute.make_layout(
                (suh_rows * Int64(self.hidden_size),),
                stride=(1,),
            ),
        )
        a_rows = Int64(active_m)
        if cutlass.const_expr(self.full_rotation):
            a_rows = Int64(active_m) * Int64(self.top_k)
        a_bf16_flat = cute.make_tensor(
            a_bf16_ptr,
            layout=cute.make_layout((a_rows * Int64(self.hidden_size),), stride=(1,)),
        )
        a_alt_bf16_flat = cute.make_tensor(
            a_alt_bf16_ptr,
            layout=cute.make_layout((a_rows * Int64(self.hidden_size),), stride=(1,)),
        )
        rotation_input_flat = cute.make_tensor(
            rotation_input_ptr,
            layout=cute.make_layout(
                (Int64(active_m) * Int64(self.hidden_size),), stride=(1,)
            ),
        )
        topk_weights_flat = cute.make_tensor(
            topk_weights_ptr,
            layout=cute.make_layout(
                (Int64(active_m) * Int64(self.top_k),), stride=(1,)
            ),
        )
        expert_map_flat = cute.make_tensor(
            expert_map_ptr,
            layout=cute.make_layout(
                (Int64(route_num_experts) + Int64(1),), stride=(1,)
            ),
        )
        fc1_trellis_lut_flat = cute.make_tensor(
            fc1_trellis_lut_ptr,
            layout=cute.make_layout(
                (Int64(_LUT_E4M3_VALUE_TABLE_ENTRIES),), stride=(1,)
            ),
        )
        fc2_trellis_lut_flat = cute.make_tensor(
            fc2_trellis_lut_ptr,
            layout=cute.make_layout(
                (Int64(_LUT_E4M3_VALUE_TABLE_ENTRIES),), stride=(1,)
            ),
        )
        grid = (grid_x, 1, 1)
        self.kernel(
            a_bf16_flat,
            a_alt_bf16_flat,
            rotation_input_flat,
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
            rot_scales_flat,
            suh_gate_flat,
            suh_up_flat,
            expert_map_flat,
            fc1_trellis_lut_flat,
            fc2_trellis_lut_flat,
            weight_num_experts,
            route_num_experts,
            active_m,
        ).launch(
            grid=grid,
            block=[self.cta_threads, 1, 1],
            min_blocks_per_mp=self.blocks_per_sm,
            # The fused body crosses software all-CTA barriers between FC1,
            # activation, and FC2. Require whole-grid admission so unrelated
            # work cannot occupy an SM while resident CTAs wait for peers that
            # have not been scheduled yet.
            cooperative=True,
            # Occupancy uses the full SMEM budget. CUTLASS's automatic carveout
            # omits the per-CTA reservation: two 32 KiB Q8_0 CTAs request a
            # 64 KiB carveout but cannot both reside there, rejecting the
            # cooperative launch. Match the budget used by the launch planner.
            preferred_smem_carveout=100 if self.fc1.weight_layout_block else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        a_bf16_flat: cute.Tensor,
        a_alt_bf16_flat: cute.Tensor,
        rotation_input_flat: cute.Tensor,
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
        rot_scales_flat: cute.Tensor,
        suh_gate_flat: cute.Tensor,
        suh_up_flat: cute.Tensor,
        expert_map_flat: cute.Tensor,
        fc1_trellis_lut_flat: cute.Tensor,
        fc2_trellis_lut_flat: cute.Tensor,
        weight_num_experts: cutlass.Int32,
        route_num_experts: cutlass.Int32,
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
        fc1_trellis_lut_addr = get_ptr_as_int64(fc1_trellis_lut_flat, Int32(0))
        fc2_trellis_lut_addr = get_ptr_as_int64(fc2_trellis_lut_flat, Int32(0))

        # The LUT ABI slot carries the shared byte offset for either the
        # compact value table or the complete 2-bit direct table.
        fc1_phase_lut_addr = fc1_trellis_lut_addr
        fc2_phase_lut_addr = fc2_trellis_lut_addr
        if cutlass.const_expr(self.weight_layout in IQ2_CODECS):
            for i in cutlass.range_constexpr(_covering_count(self.iq2_lut_bytes // 16, self.cta_threads)):
                chunk = Int32(i * self.cta_threads) + tid
                if chunk < Int32(self.iq2_lut_bytes // 16):
                    cp_async4_shared_global(
                        smem_base + Int32(self.iq2_xs_lut_off) + chunk * Int32(16),
                        fc1_trellis_lut_addr + Int64(chunk) * Int64(16),
                    )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            table_addr = Int64(smem_base + Int32(self.iq2_xs_lut_off))
            fc1_phase_lut_addr = table_addr
            fc2_phase_lut_addr = table_addr
        if cutlass.const_expr(self.lut_e4m3_smem):
            self._lut_smem_copy(
                fc1_trellis_lut_addr,
                smem_base + Int32(self.lut_e4m3_smem_off),
                self.trellis_lut_smem_bytes,
                tid,
            )
            cute.arch.sync_threads()
            table_addr = Int64(
                smem_base + Int32(self.lut_e4m3_smem_off)
            )
            fc1_phase_lut_addr = table_addr
            fc2_phase_lut_addr = table_addr

        fc1_emit_tile = None
        fc2_emit_tile = None
        if cutlass.const_expr(self.use_expert_map):
            fc1_emit_tile = partial(
                self._emit_expert_map_tile,
                True,
                a_bf16_flat,
                a_alt_bf16_flat,
                w13_i32_flat,
                fc1_bf16_flat if cutlass.const_expr(self.activation_is_gated) else activated_bf16_flat,
                w13_scales_i32_flat,
                w13_global_scale,
                packed_route_indices,
                expert_map_flat,
                topk_weights_flat,
                fc1_c_tmp_f32_flat,
                locks_i32_flat,
                fc1_phase_lut_addr,
                smem_base,
                tid,
                active_m,
                weight_num_experts,
                route_num_experts,
            )
            fc2_emit_tile = partial(
                self._emit_expert_map_tile,
                False,
                activated_bf16_flat,
                activated_bf16_flat,
                w2_i32_flat,
                (
                    fc1_bf16_flat
                    if cutlass.const_expr(self.tc_decode_fused_sum and self.weight_layout in BLOCK_CODECS)
                    else fc2_bf16_flat
                ),
                w2_scales_i32_flat,
                w2_global_scale,
                packed_route_indices,
                expert_map_flat,
                topk_weights_flat,
                fc2_c_tmp_f32_flat,
                locks_i32_flat,
                fc2_phase_lut_addr,
                smem_base,
                tid,
                active_m * Int32(self.top_k),
                weight_num_experts,
                route_num_experts,
            )
        self._moe_body(
            a_bf16_flat,
            a_alt_bf16_flat,
            rotation_input_flat,
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
            rot_scales_flat,
            suh_gate_flat,
            suh_up_flat,
            expert_map_flat,
            fc1_trellis_lut_addr,
            fc2_trellis_lut_addr,
            weight_num_experts,
            route_num_experts,
            smem_base,
            tid,
            cta,
            grid_x,
            active_m,
            fc1_emit_tile,
            fc2_emit_tile,
        )

    @cute.jit
    def _moe_body(
        self,
        a_bf16_flat: cute.Tensor,
        a_alt_bf16_flat: cute.Tensor,
        rotation_input_flat: cute.Tensor,
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
        rot_scales_flat: cute.Tensor,
        suh_gate_flat: cute.Tensor,
        suh_up_flat: cute.Tensor,
        expert_map_flat: cute.Tensor,
        fc1_trellis_lut_addr: Int64,
        fc2_trellis_lut_addr: Int64,
        weight_num_experts: Int32,
        route_num_experts: Int32,
        smem_base: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: cutlass.Int32,
        fc1_emit_tile: cutlass.Constexpr = None,
        fc2_emit_tile: cutlass.Constexpr = None,
    ):
        # Phase assembly shared by the single-tier fused kernel and the hybrid
        # multi-tier entry: FC1, activation, FC2, and optional output reduction,
        # with grid barriers between dependent phases. The emit hooks delegate per-tile expert resolution and
        # dispatch (used by the hybrid route map); None keeps the single-tier
        # resolution inside _run_persistent_gemm.
        # The trellis LUT parameters carry the raw global address unless the
        # single-tier entry staged the value table in shared memory.
        fc1_phase_lut = fc1_trellis_lut_addr
        fc2_phase_lut = fc2_trellis_lut_addr
        if cutlass.const_expr(self.weight_layout in IQ2_CODECS):
            table_addr = Int64(smem_base + Int32(self.iq2_xs_lut_off))
            fc1_phase_lut = table_addr
            fc2_phase_lut = table_addr
        if cutlass.const_expr(self.lut_e4m3_smem):
            table_addr = Int64(
                smem_base + Int32(self.lut_e4m3_smem_off)
            )
            fc1_phase_lut = table_addr
            fc2_phase_lut = table_addr
        if cutlass.const_expr(self.full_rotation):
            if cutlass.const_expr(self.intermediate_hadamard):
                self._run_input_rotation_intermediate_hadamard(
                    rotation_input_flat,
                    a_bf16_flat,
                    suh_gate_flat,
                    packed_route_indices,
                    block_expert_ids,
                    packed_route_count,
                    expert_map_flat,
                    weight_num_experts,
                    route_num_experts,
                    tid,
                    cta,
                    grid_x,
                    active_m,
                )
            else:
                self._run_input_rotation(
                    rotation_input_flat,
                    a_bf16_flat,
                    a_alt_bf16_flat,
                    suh_gate_flat,
                    suh_up_flat,
                    packed_route_indices,
                    block_expert_ids,
                    packed_route_count,
                    expert_map_flat,
                    weight_num_experts,
                    route_num_experts,
                    tid,
                    cta,
                    grid_x,
                    active_m,
                )
            self._grid_barrier(locks_i32_flat, tid, grid_x)
        if cutlass.const_expr(
            (self.tc_decode_fused_sum and self.weight_layout not in BLOCK_CODECS)
            or self.prefill_fused_sum_fp32
        ):
            # FC2 route reduction atomically accumulates into one row per token.
            # Every CTA zeroes a grid-strided slice before FC1. The mandatory
            # post-FC1 grid barrier orders these stores before every FC2 atomic.
            # ``tc_zero_output=False`` is valid only when a paired launch has
            # already zeroed the same output and participates in that barrier.
            if cutlass.const_expr(self.tc_zero_output):
                zidx = cta * Int32(self.cta_threads) + tid
                zstride = grid_x * Int32(self.cta_threads)
                zzero = (
                    cutlass.Float32(0.0)
                    if cutlass.const_expr(self.prefill_fused_sum_fp32)
                    else self._cast_elem(cutlass.Float32(0.0))
                )
                if cutlass.const_expr(self.prefill_fused_sum_fp32):
                    zidx_i64 = Int64(zidx)
                    zstride_i64 = Int64(zstride)
                    ztotal_i64 = Int64(active_m) * Int64(self.hidden_size)
                    while zidx_i64 < ztotal_i64:
                        fc2_bf16_flat[zidx_i64] = zzero
                        zidx_i64 += zstride_i64
                else:
                    ztotal = active_m * Int32(self.hidden_size)
                    while zidx < ztotal:
                        fc2_bf16_flat[zidx] = zzero
                        zidx += zstride

        if cutlass.const_expr(self.activation_is_gated):
            self.fc1._run_persistent_gemm(
                a_bf16_flat,
                a_alt_bf16_flat,
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
                fc1_phase_lut,
                smem_base,
                tid,
                cta,
                grid_x,
                active_m,
                fc1_emit_tile,
            )
            self._grid_barrier(locks_i32_flat, tid, grid_x)
            if cutlass.const_expr(self.full_rotation):
                if cutlass.const_expr(self.intermediate_hadamard):
                    self._run_activation_intermediate_hadamard(
                        fc1_bf16_flat,
                        activated_bf16_flat,
                        rot_scales_flat,
                        packed_route_indices,
                        block_expert_ids,
                        packed_route_count,
                        expert_map_flat,
                        weight_num_experts,
                        route_num_experts,
                        tid,
                        cta,
                        grid_x,
                        active_m,
                    )
                else:
                    self._run_activation_compact(
                        fc1_bf16_flat,
                        activated_bf16_flat,
                        rot_scales_flat,
                        packed_route_indices,
                        block_expert_ids,
                        packed_route_count,
                        expert_map_flat,
                        weight_num_experts,
                        route_num_experts,
                        tid,
                        cta,
                        grid_x,
                        active_m,
                    )
            else:
                self._run_activation(
                    fc1_bf16_flat,
                    activated_bf16_flat,
                    rot_scales_flat,
                    tid,
                    cta,
                    grid_x,
                    active_m,
                )
        else:
            self.fc1._run_persistent_gemm(
                a_bf16_flat,
                a_alt_bf16_flat,
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
                fc1_phase_lut,
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
        if cutlass.const_expr(self.zero_fc2_output and not self.prefill_fused_sum_fp32):
            self._zero_fc2_output(fc2_bf16_flat, tid, cta, grid_x, active_m)
            self._grid_barrier(locks_i32_flat, tid, grid_x)
        fc2_store = fc2_bf16_flat
        if cutlass.const_expr(self.tc_decode_fused_sum and self.weight_layout in BLOCK_CODECS):
            # FC1 storage is reusable after the activation handoff.
            fc2_store = fc1_bf16_flat
        self.fc2._run_persistent_gemm(
            activated_bf16_flat,
            activated_bf16_flat,
            w2_i32_flat,
            fc2_store,
            w2_scales_i32_flat,
            w2_global_scale,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            topk_weights_flat,
            fc2_c_tmp_f32_flat,
            locks_i32_flat,
            fc2_phase_lut,
            smem_base,
            tid,
            cta,
            grid_x,
            active_m * Int32(self.top_k),
            fc2_emit_tile,
        )
        if cutlass.const_expr(self.tc_decode_fused_sum and self.weight_layout in BLOCK_CODECS):
            self._grid_barrier(locks_i32_flat, tid, grid_x)
            self._sum_iq2_xs_routes(
                fc1_bf16_flat,
                fc2_bf16_flat,
                packed_route_indices,
                expert_map_flat,
                weight_num_experts,
                route_num_experts,
                tid,
                cta,
                grid_x,
                active_m,
            )

    @cute.jit
    def _sum_iq2_xs_routes(
        self,
        route_output: cute.Tensor,
        output: cute.Tensor,
        topk_ids: cute.Tensor,
        expert_map: cute.Tensor,
        weight_num_experts: Int32,
        route_num_experts: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: Int32,
    ):
        idx = Int64(cta) * Int64(self.cta_threads) + Int64(tid)
        stride = Int64(grid_x) * Int64(self.cta_threads)
        total = Int64(active_m) * Int64(self.hidden_size)
        while idx < total:
            token = idx // Int64(self.hidden_size)
            col = idx % Int64(self.hidden_size)
            value = cutlass.Float32(0.0)
            partials = cute.make_rmem_tensor((4,), cutlass.Float32)
            for group in cutlass.range(self.top_k // 4, unroll=1):
                partials.fill(0.0)
                for j in cutlass.range_constexpr(4):
                    slot = group * Int32(4) + Int32(j)
                    route = token * Int64(self.top_k) + Int64(slot)
                    expert = topk_ids[route].to(Int32)
                    if cutlass.const_expr(self.use_expert_map):
                        global_expert = expert
                        expert = Int32(-1)
                        if global_expert >= Int32(0) and global_expert < route_num_experts:
                            expert = expert_map[global_expert].to(Int32)
                    if expert >= Int32(0) and expert < weight_num_experts:
                        offset = route * Int64(self.hidden_size) + col
                        partials[j] = route_output[offset].to(cutlass.Float32)
                for j in cutlass.range_constexpr(4):
                    value += partials[j]
            for slot in cutlass.range_constexpr(self.top_k // 4 * 4, self.top_k):
                route = token * Int64(self.top_k) + Int64(slot)
                expert = topk_ids[route].to(Int32)
                if cutlass.const_expr(self.use_expert_map):
                    global_expert = expert
                    expert = Int32(-1)
                    if global_expert >= Int32(0) and global_expert < route_num_experts:
                        expert = expert_map[global_expert].to(Int32)
                if expert >= Int32(0) and expert < weight_num_experts:
                    offset = route * Int64(self.hidden_size) + col
                    value += route_output[offset].to(cutlass.Float32)
            if cutlass.const_expr(not self.tc_zero_output):
                value += output[idx].to(cutlass.Float32)
            output[idx] = cutlass.BFloat16(value)
            idx += stride

    @cute.jit
    def _lut_smem_copy(
        self,
        src_addr: Int64,
        dst_off: Int32,
        nbytes: cutlass.Constexpr[int],
        tid: Int32,
    ):
        chunks = Int32(int(nbytes) // 16)
        i = tid
        while i < chunks:
            v0, v1, v2, v3 = ld_global_nc_v4_u32(src_addr + Int64(i) * Int64(16))
            st_shared_v4_u32(dst_off + i * Int32(16), v0, v1, v2, v3)
            i += Int32(self.cta_threads)

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
    def _run_input_rotation(
        self,
        x_input_flat: cute.Tensor,
        a_gate_flat: cute.Tensor,
        a_up_flat: cute.Tensor,
        suh_gate_flat: cute.Tensor,
        suh_up_flat: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        expert_map_flat: cute.Tensor,
        weight_num_experts: Int32,
        route_num_experts: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: cutlass.Int32,
    ):
        # One warp owns one (packed route, H128 block).  The packed route id is
        # the logical flattened top-k route, so route//top_k is the source token
        # and route is the row in each projection-specific A scratch.  Padding
        # slots carry the live-route sentinel and are skipped.  Both scale
        # multiplies are rounded to fp16 before the existing fp32-register
        # H128 butterfly, and each H128 result is rounded to fp16 on store.
        lane = tid & Int32(31)
        warp_in_cta = tid >> Int32(5)
        warps_per_cta = Int32(self.cta_threads // 32)
        nblk = Int32(self.hidden_size // 128)
        live_routes = active_m * Int32(self.top_k)
        route_count = packed_route_count[Int32(0)].to(Int32)
        if cutlass.const_expr(self.direct_topk_routes):
            route_count = live_routes
        gwarp = cta * warps_per_cta + warp_in_cta
        gw_stride = grid_x * warps_per_cta
        total_units = route_count * nblk
        elem = lane * Int32(4)
        unit = gwarp
        while unit < total_units:
            route_pos = unit // nblk
            blk = unit - route_pos * nblk
            route = packed_route_indices[route_pos].to(Int32)
            expert = block_expert_ids[route_pos // Int32(self.moe_block_size)].to(Int32)
            if cutlass.const_expr(self.direct_topk_routes):
                route = route_pos
                expert = packed_route_indices[route_pos].to(Int32)
                if cutlass.const_expr(self.use_expert_map):
                    global_expert = expert
                    expert = Int32(-1)
                    if global_expert >= Int32(0) and global_expert < route_num_experts:
                        expert = expert_map_flat[global_expert].to(Int32)
            if (
                route >= Int32(0)
                and route < live_routes
                and expert >= Int32(0)
                and expert < weight_num_experts
            ):
                token = route // Int32(self.top_k)
                col0 = blk * Int32(128) + elem
                x_base = token * Int32(self.hidden_size) + col0
                if cutlass.const_expr(self.broadcast_suh):
                    s_base = col0
                else:
                    s_base = expert * Int32(self.hidden_size) + col0
                out_base = route * Int32(self.hidden_size) + col0

                x0 = cutlass.Float16(
                    x_input_flat[x_base + Int32(0)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x1 = cutlass.Float16(
                    x_input_flat[x_base + Int32(1)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x2 = cutlass.Float16(
                    x_input_flat[x_base + Int32(2)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x3 = cutlass.Float16(
                    x_input_flat[x_base + Int32(3)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                sg0 = suh_gate_flat[s_base + Int32(0)].to(cutlass.Float32)
                sg1 = suh_gate_flat[s_base + Int32(1)].to(cutlass.Float32)
                sg2 = suh_gate_flat[s_base + Int32(2)].to(cutlass.Float32)
                sg3 = suh_gate_flat[s_base + Int32(3)].to(cutlass.Float32)
                su0 = suh_up_flat[s_base + Int32(0)].to(cutlass.Float32)
                su1 = suh_up_flat[s_base + Int32(1)].to(cutlass.Float32)
                su2 = suh_up_flat[s_base + Int32(2)].to(cutlass.Float32)
                su3 = suh_up_flat[s_base + Int32(3)].to(cutlass.Float32)

                g0 = cutlass.Float16(x0 * sg0).to(cutlass.Float32)
                g1 = cutlass.Float16(x1 * sg1).to(cutlass.Float32)
                g2 = cutlass.Float16(x2 * sg2).to(cutlass.Float32)
                g3 = cutlass.Float16(x3 * sg3).to(cutlass.Float32)
                u0 = cutlass.Float16(x0 * su0).to(cutlass.Float32)
                u1 = cutlass.Float16(x1 * su1).to(cutlass.Float32)
                u2 = cutlass.Float16(x2 * su2).to(cutlass.Float32)
                u3 = cutlass.Float16(x3 * su3).to(cutlass.Float32)
                gh0, gh1, gh2, gh3 = self._had128_quad(g0, g1, g2, g3, lane)
                uh0, uh1, uh2, uh3 = self._had128_quad(u0, u1, u2, u3, lane)
                a_gate_flat[out_base + Int32(0)] = cutlass.Float16(gh0)
                a_gate_flat[out_base + Int32(1)] = cutlass.Float16(gh1)
                a_gate_flat[out_base + Int32(2)] = cutlass.Float16(gh2)
                a_gate_flat[out_base + Int32(3)] = cutlass.Float16(gh3)
                a_up_flat[out_base + Int32(0)] = cutlass.Float16(uh0)
                a_up_flat[out_base + Int32(1)] = cutlass.Float16(uh1)
                a_up_flat[out_base + Int32(2)] = cutlass.Float16(uh2)
                a_up_flat[out_base + Int32(3)] = cutlass.Float16(uh3)
            unit += gw_stride

    @cute.jit
    def _run_input_rotation_intermediate_hadamard(
        self,
        x_input_flat: cute.Tensor,
        a_shared_flat: cute.Tensor,
        suh_flat: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        expert_map_flat: cute.Tensor,
        weight_num_experts: Int32,
        route_num_experts: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: cutlass.Int32,
    ):
        """Apply the shared H512 boundary followed by ordinary H128 inputs."""

        lane = tid & Int32(31)
        warp_in_cta = tid >> Int32(5)
        warps_per_cta = Int32(self.cta_threads // 32)
        nblk = Int32(self.hidden_size // 512)
        live_routes = active_m * Int32(self.top_k)
        route_count = packed_route_count[Int32(0)].to(Int32)
        if cutlass.const_expr(self.direct_topk_routes):
            route_count = live_routes
        gwarp = cta * warps_per_cta + warp_in_cta
        gw_stride = grid_x * warps_per_cta
        total_units = route_count * nblk
        elem = lane * Int32(4)
        unit = gwarp
        while unit < total_units:
            route_pos = unit // nblk
            blk = unit - route_pos * nblk
            route = packed_route_indices[route_pos].to(Int32)
            expert = block_expert_ids[route_pos // Int32(self.moe_block_size)].to(
                Int32
            )
            if cutlass.const_expr(self.direct_topk_routes):
                route = route_pos
                expert = packed_route_indices[route_pos].to(Int32)
                if cutlass.const_expr(self.use_expert_map):
                    global_expert = expert
                    expert = Int32(-1)
                    if global_expert >= Int32(0) and global_expert < route_num_experts:
                        expert = expert_map_flat[global_expert].to(Int32)
            if (
                route >= Int32(0)
                and route < live_routes
                and expert >= Int32(0)
                and expert < weight_num_experts
            ):
                token = route // Int32(self.top_k)
                col0 = blk * Int32(512) + elem
                x_base = token * Int32(self.hidden_size) + col0

                x00 = cutlass.Float16(
                    x_input_flat[x_base + Int32(0)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x01 = cutlass.Float16(
                    x_input_flat[x_base + Int32(1)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x02 = cutlass.Float16(
                    x_input_flat[x_base + Int32(2)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x03 = cutlass.Float16(
                    x_input_flat[x_base + Int32(3)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x10 = cutlass.Float16(
                    x_input_flat[x_base + Int32(128)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x11 = cutlass.Float16(
                    x_input_flat[x_base + Int32(129)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x12 = cutlass.Float16(
                    x_input_flat[x_base + Int32(130)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x13 = cutlass.Float16(
                    x_input_flat[x_base + Int32(131)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x20 = cutlass.Float16(
                    x_input_flat[x_base + Int32(256)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x21 = cutlass.Float16(
                    x_input_flat[x_base + Int32(257)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x22 = cutlass.Float16(
                    x_input_flat[x_base + Int32(258)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x23 = cutlass.Float16(
                    x_input_flat[x_base + Int32(259)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x30 = cutlass.Float16(
                    x_input_flat[x_base + Int32(384)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x31 = cutlass.Float16(
                    x_input_flat[x_base + Int32(385)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x32 = cutlass.Float16(
                    x_input_flat[x_base + Int32(386)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x33 = cutlass.Float16(
                    x_input_flat[x_base + Int32(387)].to(cutlass.Float32)
                ).to(cutlass.Float32)

                h00, h01, h02, h03 = self._had128_quad(
                    x00, x01, x02, x03, lane
                )
                h10, h11, h12, h13 = self._had128_quad(
                    x10, x11, x12, x13, lane
                )
                h20, h21, h22, h23 = self._had128_quad(
                    x20, x21, x22, x23, lane
                )
                h30, h31, h32, h33 = self._had128_quad(
                    x30, x31, x32, x33, lane
                )
                c00, c10, c20, c30 = self._had4_normalized(h00, h10, h20, h30)
                c01, c11, c21, c31 = self._had4_normalized(h01, h11, h21, h31)
                c02, c12, c22, c32 = self._had4_normalized(h02, h12, h22, h32)
                c03, c13, c23, c33 = self._had4_normalized(h03, h13, h23, h33)

                if cutlass.const_expr(self.broadcast_suh):
                    s_base = col0
                else:
                    s_base = expert * Int32(self.hidden_size) + col0
                out_base = route * Int32(self.hidden_size) + col0
                for group in cutlass.range_constexpr(4):
                    offset = Int32(group * 128)
                    if cutlass.const_expr(group == 0):
                        c0, c1, c2, c3 = c00, c01, c02, c03
                    elif cutlass.const_expr(group == 1):
                        c0, c1, c2, c3 = c10, c11, c12, c13
                    elif cutlass.const_expr(group == 2):
                        c0, c1, c2, c3 = c20, c21, c22, c23
                    else:
                        c0, c1, c2, c3 = c30, c31, c32, c33
                    s0 = suh_flat[s_base + offset + Int32(0)].to(cutlass.Float32)
                    s1 = suh_flat[s_base + offset + Int32(1)].to(cutlass.Float32)
                    s2 = suh_flat[s_base + offset + Int32(2)].to(cutlass.Float32)
                    s3 = suh_flat[s_base + offset + Int32(3)].to(cutlass.Float32)
                    c0 = cutlass.Float16(c0 * s0).to(cutlass.Float32)
                    c1 = cutlass.Float16(c1 * s1).to(cutlass.Float32)
                    c2 = cutlass.Float16(c2 * s2).to(cutlass.Float32)
                    c3 = cutlass.Float16(c3 * s3).to(cutlass.Float32)
                    o0, o1, o2, o3 = self._had128_quad(c0, c1, c2, c3, lane)
                    a_shared_flat[out_base + offset + Int32(0)] = cutlass.Float16(o0)
                    a_shared_flat[out_base + offset + Int32(1)] = cutlass.Float16(o1)
                    a_shared_flat[out_base + offset + Int32(2)] = cutlass.Float16(o2)
                    a_shared_flat[out_base + offset + Int32(3)] = cutlass.Float16(o3)
            unit += gw_stride

    @cute.jit
    def _load_intermediate_hadamard_pre_quad(
        self,
        fc1_flat: cute.Tensor,
        rotations_flat: cute.Tensor,
        row: Int32,
        expert: Int32,
        pre_block: Int32,
        lane: Int32,
    ):
        """Decode one logical 128-coordinate intermediate-Hadamard preactivation record."""

        if cutlass.const_expr(self.fc1_trellis_pair_kind is None):
            # The uniform-rate layout divides one interleaved intermediate-Hadamard window
            # across the two physical FC1 slots.
            chunk = lane >> Int32(3)
            chunk_lane = lane & Int32(7)
            channel_group = pre_block * Int32(2) + (chunk >> Int32(1))
            slot = chunk & Int32(1)
            coord = channel_group * Int32(32) + chunk_lane * Int32(4)
        else:
            # A fixed-rate pair slot contains two complete 128-coordinate
            # records.  Apply the record-local output Hadamard before the
            # intermediate Hadamard by selecting one physical slot and half.
            slot = pre_block & Int32(1)
            half = pre_block >> Int32(1)
            if cutlass.const_expr(self.fc1_trellis_pair_kind == "P43"):
                half = half ^ Int32(1)
            coord = half * Int32(128) + lane * Int32(4)
        isz = Int32(self.intermediate_size)
        fc1_base = row * Int32(2 * self.intermediate_size) + slot * isz + coord
        rot_base = expert * Int32(6 * self.intermediate_size)
        scale_base = rot_base + slot * isz + coord
        v0 = fc1_flat[fc1_base + Int32(0)].to(cutlass.Float32)
        v1 = fc1_flat[fc1_base + Int32(1)].to(cutlass.Float32)
        v2 = fc1_flat[fc1_base + Int32(2)].to(cutlass.Float32)
        v3 = fc1_flat[fc1_base + Int32(3)].to(cutlass.Float32)
        h0, h1, h2, h3 = self._had128_quad(v0, v1, v2, v3, lane)
        h0 *= rotations_flat[scale_base + Int32(0)].to(cutlass.Float32)
        h1 *= rotations_flat[scale_base + Int32(1)].to(cutlass.Float32)
        h2 *= rotations_flat[scale_base + Int32(2)].to(cutlass.Float32)
        h3 *= rotations_flat[scale_base + Int32(3)].to(cutlass.Float32)
        h0, h1, h2, h3 = self._had128_quad(h0, h1, h2, h3, lane)
        sign_base = (
            rot_base
            + Int32(3 * self.intermediate_size)
            + pre_block * Int32(128)
            + lane * Int32(4)
        )
        h0 *= rotations_flat[sign_base + Int32(0)].to(cutlass.Float32)
        h1 *= rotations_flat[sign_base + Int32(1)].to(cutlass.Float32)
        h2 *= rotations_flat[sign_base + Int32(2)].to(cutlass.Float32)
        h3 *= rotations_flat[sign_base + Int32(3)].to(cutlass.Float32)
        return h0, h1, h2, h3

    @cute.jit
    def _run_activation_intermediate_hadamard(
        self,
        fc1_flat: cute.Tensor,
        activated_flat: cute.Tensor,
        rotations_flat: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        expert_map_flat: cute.Tensor,
        weight_num_experts: Int32,
        route_num_experts: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: cutlass.Int32,
    ):
        """Recover interleaved gate/up, evaluate SiTU, and prepare FC2."""

        lane = tid & Int32(31)
        warp_in_cta = tid >> Int32(5)
        warps_per_cta = Int32(self.cta_threads // 32)
        nblk = Int32(self.intermediate_size // 128)
        live_routes = active_m * Int32(self.top_k)
        route_count = packed_route_count[Int32(0)].to(Int32)
        if cutlass.const_expr(self.direct_topk_routes):
            route_count = live_routes
        gwarp = cta * warps_per_cta + warp_in_cta
        gw_stride = grid_x * warps_per_cta
        total_units = route_count * nblk
        unit = gwarp
        while unit < total_units:
            route_pos = unit // nblk
            post_block = unit - route_pos * nblk
            row = packed_route_indices[route_pos].to(Int32)
            expert = block_expert_ids[route_pos // Int32(self.moe_block_size)].to(
                Int32
            )
            if cutlass.const_expr(self.direct_topk_routes):
                row = route_pos
                expert = packed_route_indices[route_pos].to(Int32)
                if cutlass.const_expr(self.use_expert_map):
                    global_expert = expert
                    expert = Int32(-1)
                    if global_expert >= Int32(0) and global_expert < route_num_experts:
                        expert = expert_map_flat[global_expert].to(Int32)
            if (
                row >= Int32(0)
                and row < live_routes
                and expert >= Int32(0)
                and expert < weight_num_experts
            ):
                p0 = post_block * Int32(2)
                a0, a1, a2, a3 = self._load_intermediate_hadamard_pre_quad(
                    fc1_flat, rotations_flat, row, expert, p0, lane
                )
                b0, b1, b2, b3 = self._load_intermediate_hadamard_pre_quad(
                    fc1_flat, rotations_flat, row, expert, p0 + Int32(1), lane
                )
                if cutlass.const_expr(self.activation_is_situ):
                    beta = cutlass.Float32(SITU_DEFAULT_BETA)
                    linear_beta = cutlass.Float32(SITU_DEFAULT_LINEAR_BETA)
                    aa0 = (
                        beta
                        * cute.math.tanh(a0 / beta, fastmath=self.fast_math)
                        * self._sigmoid_f32(a0)
                        * linear_beta
                        * cute.math.tanh(a1 / linear_beta, fastmath=self.fast_math)
                    )
                    aa1 = (
                        beta
                        * cute.math.tanh(a2 / beta, fastmath=self.fast_math)
                        * self._sigmoid_f32(a2)
                        * linear_beta
                        * cute.math.tanh(a3 / linear_beta, fastmath=self.fast_math)
                    )
                    bb0 = (
                        beta
                        * cute.math.tanh(b0 / beta, fastmath=self.fast_math)
                        * self._sigmoid_f32(b0)
                        * linear_beta
                        * cute.math.tanh(b1 / linear_beta, fastmath=self.fast_math)
                    )
                    bb1 = (
                        beta
                        * cute.math.tanh(b2 / beta, fastmath=self.fast_math)
                        * self._sigmoid_f32(b2)
                        * linear_beta
                        * cute.math.tanh(b3 / linear_beta, fastmath=self.fast_math)
                    )
                else:
                    aa0 = self._silu_f32(a0) * a1
                    aa1 = self._silu_f32(a2) * a3
                    bb0 = self._silu_f32(b0) * b1
                    bb1 = self._silu_f32(b2) * b3

                source0 = (lane & Int32(15)) << Int32(1)
                source1 = source0 + Int32(1)
                av0 = cute.arch.shuffle_sync(aa0, source0)
                av1 = cute.arch.shuffle_sync(aa1, source0)
                av2 = cute.arch.shuffle_sync(aa0, source1)
                av3 = cute.arch.shuffle_sync(aa1, source1)
                bv0 = cute.arch.shuffle_sync(bb0, source0)
                bv1 = cute.arch.shuffle_sync(bb1, source0)
                bv2 = cute.arch.shuffle_sync(bb0, source1)
                bv3 = cute.arch.shuffle_sync(bb1, source1)
                v0, v1, v2, v3 = av0, av1, av2, av3
                if lane >= Int32(16):
                    v0, v1, v2, v3 = bv0, bv1, bv2, bv3

                isz = Int32(self.intermediate_size)
                col0 = post_block * Int32(128) + lane * Int32(4)
                rot_base = expert * Int32(6 * self.intermediate_size)
                sign_base = rot_base + Int32(5 * self.intermediate_size) + col0
                v0 *= rotations_flat[sign_base + Int32(0)].to(cutlass.Float32)
                v1 *= rotations_flat[sign_base + Int32(1)].to(cutlass.Float32)
                v2 *= rotations_flat[sign_base + Int32(2)].to(cutlass.Float32)
                v3 *= rotations_flat[sign_base + Int32(3)].to(cutlass.Float32)
                v0, v1, v2, v3 = self._had128_quad(v0, v1, v2, v3, lane)

                down_base = rot_base + Int32(2 * self.intermediate_size) + col0
                v0 *= rotations_flat[down_base + Int32(0)].to(cutlass.Float32)
                v1 *= rotations_flat[down_base + Int32(1)].to(cutlass.Float32)
                v2 *= rotations_flat[down_base + Int32(2)].to(cutlass.Float32)
                v3 *= rotations_flat[down_base + Int32(3)].to(cutlass.Float32)
                v0, v1, v2, v3 = self._had128_quad(v0, v1, v2, v3, lane)
                out_base = row * isz + col0
                activated_flat[out_base + Int32(0)] = self._cast_elem(v0)
                activated_flat[out_base + Int32(1)] = self._cast_elem(v1)
                activated_flat[out_base + Int32(2)] = self._cast_elem(v2)
                activated_flat[out_base + Int32(3)] = self._cast_elem(v3)
            unit += gw_stride

    @cute.jit
    def _run_activation_compact(
        self,
        fc1_bf16_flat: cute.Tensor,
        activated_bf16_flat: cute.Tensor,
        rot_scales_flat: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        expert_map_flat: cute.Tensor,
        weight_num_experts: Int32,
        route_num_experts: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: cutlass.Int32,
    ):
        # A9: traverse the already-packed expert blocks and index the persistent
        # [E,3I] table directly.  FC1/activation rows stay at their logical
        # flattened route ids, so packing order cannot change arithmetic or the
        # per-row output location.
        lane = tid & Int32(31)
        warp_in_cta = tid >> Int32(5)
        warps_per_cta = Int32(self.cta_threads // 32)
        nblk = Int32(self.intermediate_size // 128)
        live_routes = active_m * Int32(self.top_k)
        route_count = packed_route_count[Int32(0)].to(Int32)
        if cutlass.const_expr(self.direct_topk_routes):
            route_count = live_routes
        fc1_cols = Int32(self.fc1_cols)
        isz = Int32(self.intermediate_size)
        rot_row = Int32(3 * self.intermediate_size)
        gwarp = cta * warps_per_cta + warp_in_cta
        gw_stride = grid_x * warps_per_cta
        total_units = route_count * nblk
        elem = lane * Int32(4)
        unit = gwarp
        while unit < total_units:
            route_pos = unit // nblk
            blk = unit - route_pos * nblk
            row = packed_route_indices[route_pos].to(Int32)
            expert = block_expert_ids[route_pos // Int32(self.moe_block_size)].to(Int32)
            if cutlass.const_expr(self.direct_topk_routes):
                row = route_pos
                expert = packed_route_indices[route_pos].to(Int32)
                if cutlass.const_expr(self.use_expert_map):
                    global_expert = expert
                    expert = Int32(-1)
                    if global_expert >= Int32(0) and global_expert < route_num_experts:
                        expert = expert_map_flat[global_expert].to(Int32)
            if (
                row >= Int32(0)
                and row < live_routes
                and expert >= Int32(0)
                and expert < weight_num_experts
            ):
                col0 = blk * Int32(128) + elem
                g_base = row * fc1_cols + col0
                u_base = g_base + isz
                s_base = expert * rot_row + col0
                g0 = fc1_bf16_flat[g_base + Int32(0)].to(cutlass.Float32)
                g1 = fc1_bf16_flat[g_base + Int32(1)].to(cutlass.Float32)
                g2 = fc1_bf16_flat[g_base + Int32(2)].to(cutlass.Float32)
                g3 = fc1_bf16_flat[g_base + Int32(3)].to(cutlass.Float32)
                u0 = fc1_bf16_flat[u_base + Int32(0)].to(cutlass.Float32)
                u1 = fc1_bf16_flat[u_base + Int32(1)].to(cutlass.Float32)
                u2 = fc1_bf16_flat[u_base + Int32(2)].to(cutlass.Float32)
                u3 = fc1_bf16_flat[u_base + Int32(3)].to(cutlass.Float32)
                gh0, gh1, gh2, gh3 = self._had128_quad(g0, g1, g2, g3, lane)
                uh0, uh1, uh2, uh3 = self._had128_quad(u0, u1, u2, u3, lane)
                svg0 = rot_scales_flat[s_base + Int32(0)].to(cutlass.Float32)
                svg1 = rot_scales_flat[s_base + Int32(1)].to(cutlass.Float32)
                svg2 = rot_scales_flat[s_base + Int32(2)].to(cutlass.Float32)
                svg3 = rot_scales_flat[s_base + Int32(3)].to(cutlass.Float32)
                svu0 = rot_scales_flat[s_base + isz + Int32(0)].to(cutlass.Float32)
                svu1 = rot_scales_flat[s_base + isz + Int32(1)].to(cutlass.Float32)
                svu2 = rot_scales_flat[s_base + isz + Int32(2)].to(cutlass.Float32)
                svu3 = rot_scales_flat[s_base + isz + Int32(3)].to(cutlass.Float32)
                ig0 = gh0 * svg0
                ig1 = gh1 * svg1
                ig2 = gh2 * svg2
                ig3 = gh3 * svg3
                iu0 = uh0 * svu0
                iu1 = uh1 * svu1
                iu2 = uh2 * svu2
                iu3 = uh3 * svu3
                down = isz + isz
                sd0 = rot_scales_flat[s_base + down + Int32(0)].to(cutlass.Float32)
                sd1 = rot_scales_flat[s_base + down + Int32(1)].to(cutlass.Float32)
                sd2 = rot_scales_flat[s_base + down + Int32(2)].to(cutlass.Float32)
                sd3 = rot_scales_flat[s_base + down + Int32(3)].to(cutlass.Float32)
                if cutlass.const_expr(self.activation_is_situ):
                    beta = cutlass.Float32(SITU_DEFAULT_BETA)
                    linear_beta = cutlass.Float32(SITU_DEFAULT_LINEAR_BETA)
                    a0 = (
                        beta
                        * cute.math.tanh(ig0 / beta, fastmath=self.fast_math)
                        * self._sigmoid_f32(ig0)
                        * linear_beta
                        * cute.math.tanh(iu0 / linear_beta, fastmath=self.fast_math)
                        * sd0
                    )
                    a1 = (
                        beta
                        * cute.math.tanh(ig1 / beta, fastmath=self.fast_math)
                        * self._sigmoid_f32(ig1)
                        * linear_beta
                        * cute.math.tanh(iu1 / linear_beta, fastmath=self.fast_math)
                        * sd1
                    )
                    a2 = (
                        beta
                        * cute.math.tanh(ig2 / beta, fastmath=self.fast_math)
                        * self._sigmoid_f32(ig2)
                        * linear_beta
                        * cute.math.tanh(iu2 / linear_beta, fastmath=self.fast_math)
                        * sd2
                    )
                    a3 = (
                        beta
                        * cute.math.tanh(ig3 / beta, fastmath=self.fast_math)
                        * self._sigmoid_f32(ig3)
                        * linear_beta
                        * cute.math.tanh(iu3 / linear_beta, fastmath=self.fast_math)
                        * sd3
                    )
                else:
                    a0 = self._silu_f32(ig0) * iu0 * sd0
                    a1 = self._silu_f32(ig1) * iu1 * sd1
                    a2 = self._silu_f32(ig2) * iu2 * sd2
                    a3 = self._silu_f32(ig3) * iu3 * sd3
                o0, o1, o2, o3 = self._had128_quad(a0, a1, a2, a3, lane)
                out_base = row * isz + col0
                activated_bf16_flat[out_base + Int32(0)] = self._cast_elem(o0)
                activated_bf16_flat[out_base + Int32(1)] = self._cast_elem(o1)
                activated_bf16_flat[out_base + Int32(2)] = self._cast_elem(o2)
                activated_bf16_flat[out_base + Int32(3)] = self._cast_elem(o3)
            unit += gw_stride

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
    def _had128_quad(
        self,
        v0: cutlass.Float32,
        v1: cutlass.Float32,
        v2: cutlass.Float32,
        v3: cutlass.Float32,
        lane: Int32,
    ):
        # Blockwise-128 Walsh-Hadamard (Sylvester/natural order) across ONE warp,
        # bit-identical in structure to exllamav3 had_hf_r_128_inner: lane `t`
        # owns the 4 consecutive elements [4t..4t+3], so element index within the
        # 128-block is 4*lane + reg.  H_128 = H_4(reg, in-register) (x) H_32(lane,
        # warp butterfly), then * r_scale = 1/sqrt(128).
        s0 = v0 + v1
        d0 = v0 - v1
        s1 = v2 + v3
        d1 = v2 - v3
        h0 = s0 + s1
        h1 = d0 + d1
        h2 = s0 - s1
        h3 = d0 - d1
        for i in cutlass.range_constexpr(5):
            st = 1 << i
            p0 = cute.arch.shuffle_sync_bfly(h0, offset=st)
            p1 = cute.arch.shuffle_sync_bfly(h1, offset=st)
            p2 = cute.arch.shuffle_sync_bfly(h2, offset=st)
            p3 = cute.arch.shuffle_sync_bfly(h3, offset=st)
            if (lane & Int32(st)) != Int32(0):
                h0 = p0 - h0
                h1 = p1 - h1
                h2 = p2 - h2
                h3 = p3 - h3
            else:
                h0 = p0 + h0
                h1 = p1 + h1
                h2 = p2 + h2
                h3 = p3 + h3
        rs = cutlass.Float32(0.088388347648)  # scale / sqrt(128), scale = 1
        return h0 * rs, h1 * rs, h2 * rs, h3 * rs

    @cute.jit
    def _had4_normalized(
        self,
        v0: cutlass.Float32,
        v1: cutlass.Float32,
        v2: cutlass.Float32,
        v3: cutlass.Float32,
    ):
        """Normalized natural-order H4 over four H128 subblocks."""
        s0 = v0 + v1
        d0 = v0 - v1
        s1 = v2 + v3
        d1 = v2 - v3
        rs = cutlass.Float32(0.5)
        return (
            (s0 + s1) * rs,
            (d0 + d1) * rs,
            (s0 - s1) * rs,
            (d0 - d1) * rs,
        )

    @cute.jit
    def _sigmoid_f32(self, x: cutlass.Float32) -> cutlass.Float32:
        if cutlass.const_expr(self.fast_math):
            e = cute.math.exp(-x, fastmath=True)
        else:
            e = cute.math.exp(-x, fastmath=False)
        return cutlass.Float32(1.0) / (cutlass.Float32(1.0) + e)

    @cute.jit
    def _silu_f32(self, x: cutlass.Float32) -> cutlass.Float32:
        return x * self._sigmoid_f32(x)

    @cute.jit
    def _run_activation(
        self,
        fc1_bf16_flat: cute.Tensor,
        activated_bf16_flat: cute.Tensor,
        rot_scales_flat: cute.Tensor,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: cutlass.Int32,
    ):
        if cutlass.const_expr(self.intermediate_rotation):
            # Rotation-aware epilogue (trellis_t256 tail).  Warp-cooperative over
            # (routed-row, 128-block) units; each warp owns one 128-wide block of
            # a row's intermediate.  Per row r the FC1 output is [gate(I) | up(I)]
            # (fc1_cols = 2I) and rot_scales_flat[r] = [svh_gate(I)|svh_up(I)|
            # suh_down(I)] (3I fp16).  Applies:
            #   ig2 = had128(gate) * svh_gate ;  iu2 = had128(up) * svh_up
            #   ia  = silu(ig2) * iu2
            #   ia2 = had128(ia * suh_down)           -> activated[r, block]
            lane = tid & Int32(31)
            warp_in_cta = tid >> Int32(5)
            warps_per_cta = Int32(self.cta_threads // 32)
            nblk = Int32(self.intermediate_size // 128)
            fc1_cols = Int32(self.fc1_cols)
            isz = Int32(self.intermediate_size)
            rot_row = Int32(3 * self.intermediate_size)
            gwarp = cta * warps_per_cta + warp_in_cta
            gw_stride = grid_x * warps_per_cta
            total_units = active_m * Int32(self.top_k) * nblk
            e = lane * Int32(4)
            unit = gwarp
            while unit < total_units:
                row = unit // nblk
                blk = unit - row * nblk
                col0 = blk * Int32(128) + e
                g_base = row * fc1_cols + col0
                u_base = g_base + isz
                s_base = row * rot_row + col0
                # load 4 gate + 4 up
                g0 = fc1_bf16_flat[g_base + Int32(0)].to(cutlass.Float32)
                g1 = fc1_bf16_flat[g_base + Int32(1)].to(cutlass.Float32)
                g2 = fc1_bf16_flat[g_base + Int32(2)].to(cutlass.Float32)
                g3 = fc1_bf16_flat[g_base + Int32(3)].to(cutlass.Float32)
                u0 = fc1_bf16_flat[u_base + Int32(0)].to(cutlass.Float32)
                u1 = fc1_bf16_flat[u_base + Int32(1)].to(cutlass.Float32)
                u2 = fc1_bf16_flat[u_base + Int32(2)].to(cutlass.Float32)
                u3 = fc1_bf16_flat[u_base + Int32(3)].to(cutlass.Float32)
                # blockwise-128 Hadamard on gate and up (pre-silu, output side)
                gh0, gh1, gh2, gh3 = self._had128_quad(g0, g1, g2, g3, lane)
                uh0, uh1, uh2, uh3 = self._had128_quad(u0, u1, u2, u3, lane)
                # post-scale svh_gate / svh_up
                svg0 = rot_scales_flat[s_base + Int32(0)].to(cutlass.Float32)
                svg1 = rot_scales_flat[s_base + Int32(1)].to(cutlass.Float32)
                svg2 = rot_scales_flat[s_base + Int32(2)].to(cutlass.Float32)
                svg3 = rot_scales_flat[s_base + Int32(3)].to(cutlass.Float32)
                svu0 = rot_scales_flat[s_base + isz + Int32(0)].to(cutlass.Float32)
                svu1 = rot_scales_flat[s_base + isz + Int32(1)].to(cutlass.Float32)
                svu2 = rot_scales_flat[s_base + isz + Int32(2)].to(cutlass.Float32)
                svu3 = rot_scales_flat[s_base + isz + Int32(3)].to(cutlass.Float32)
                ig2_0 = gh0 * svg0
                ig2_1 = gh1 * svg1
                ig2_2 = gh2 * svg2
                ig2_3 = gh3 * svg3
                iu2_0 = uh0 * svu0
                iu2_1 = uh1 * svu1
                iu2_2 = uh2 * svu2
                iu2_3 = uh3 * svu3
                # silu(gate) * up, then pre-scale suh_down
                d2 = isz + isz
                sd0 = rot_scales_flat[s_base + d2 + Int32(0)].to(cutlass.Float32)
                sd1 = rot_scales_flat[s_base + d2 + Int32(1)].to(cutlass.Float32)
                sd2 = rot_scales_flat[s_base + d2 + Int32(2)].to(cutlass.Float32)
                sd3 = rot_scales_flat[s_base + d2 + Int32(3)].to(cutlass.Float32)
                a0 = self._silu_f32(ig2_0) * iu2_0 * sd0
                a1 = self._silu_f32(ig2_1) * iu2_1 * sd1
                a2 = self._silu_f32(ig2_2) * iu2_2 * sd2
                a3 = self._silu_f32(ig2_3) * iu2_3 * sd3
                # blockwise-128 Hadamard on the activation (post-silu, input side)
                o0, o1, o2, o3 = self._had128_quad(a0, a1, a2, a3, lane)
                out_base = row * isz + col0
                activated_bf16_flat[out_base + Int32(0)] = self._cast_elem(o0)
                activated_bf16_flat[out_base + Int32(1)] = self._cast_elem(o1)
                activated_bf16_flat[out_base + Int32(2)] = self._cast_elem(o2)
                activated_bf16_flat[out_base + Int32(3)] = self._cast_elem(o3)
                unit += gw_stride
            return
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
                sigmoid = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + exp_neg_gate)
                silu = gate * sigmoid
                if cutlass.const_expr(self.activation_is_situ):
                    beta = cutlass.Float32(SITU_DEFAULT_BETA)
                    linear_beta = cutlass.Float32(SITU_DEFAULT_LINEAR_BETA)
                    situ_gate = (
                        beta
                        * cute.math.tanh(gate / beta, fastmath=self.fast_math)
                        * sigmoid
                    )
                    situ_up = linear_beta * cute.math.tanh(
                        up / linear_beta,
                        fastmath=self.fast_math,
                    )
                    activated_bf16_flat[idx] = self._cast_elem(situ_gate * situ_up)
                elif cutlass.const_expr(self.activation_is_swigluoai):
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
        self.is_situ = activation == SITU
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
            self.is_situ,
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
                sigmoid = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + exp_neg_gate)
                silu = gate * sigmoid
                if cutlass.const_expr(self.is_situ):
                    beta = cutlass.Float32(SITU_DEFAULT_BETA)
                    linear_beta = cutlass.Float32(SITU_DEFAULT_LINEAR_BETA)
                    situ_gate = (
                        beta
                        * cute.math.tanh(gate / beta, fastmath=self.fast_math)
                        * sigmoid
                    )
                    situ_up = linear_beta * cute.math.tanh(
                        up / linear_beta,
                        fastmath=self.fast_math,
                    )
                    activated_flat[idx] = self._cast_elem(situ_gate * situ_up)
                elif cutlass.const_expr(self.is_swigluoai):
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
    def __init__(
        self,
        *,
        topk: int,
        hidden_size: int,
        element_dtype: str = "bf16",
        full_rotation: bool = False,
        intermediate_hadamard: bool = False,
        num_experts: int = 0,
        route_num_experts: int = 0,
        use_expert_map: bool = False,
        broadcast_svh: bool = False,
        float32_output: bool = False,
    ):
        if element_dtype not in {"bf16", "fp16"}:
            raise ValueError(f"unsupported element_dtype {element_dtype!r}")
        if topk <= 0 or hidden_size <= 0:
            raise ValueError("topk and hidden_size must be positive")
        self.topk = int(topk)
        self.hidden_size = int(hidden_size)
        self.element_dtype = element_dtype
        self.is_fp16 = element_dtype == "fp16"
        self.full_rotation = bool(full_rotation)
        self.intermediate_hadamard = bool(intermediate_hadamard)
        self.num_experts = int(num_experts)
        self.route_num_experts = int(route_num_experts)
        self.use_expert_map = bool(use_expert_map)
        # svh_table holds a single row shared by every expert; index it with a
        # zero expert stride.
        self.broadcast_svh = bool(broadcast_svh)
        self.float32_output = bool(float32_output)
        if self.use_expert_map:
            if self.route_num_experts <= 0:
                raise ValueError("expert-map top-k sum requires route_num_experts > 0")
            if self.num_experts <= 0:
                raise ValueError("expert-map top-k sum requires num_experts > 0")
        if self.full_rotation:
            if self.element_dtype != "fp16":
                raise ValueError("full-rotation top-k sum requires fp16 route values")
            if self.hidden_size % 128 != 0:
                raise ValueError(
                    "full-rotation top-k sum requires hidden_size % 128 == 0"
                )
            if self.num_experts <= 0:
                raise ValueError("full-rotation top-k sum requires num_experts > 0")
        if self.intermediate_hadamard:
            if not self.full_rotation:
                raise ValueError("intermediate-Hadamard top-k sum requires full rotation")
            if self.hidden_size % 512 != 0:
                raise ValueError(
                    "intermediate-Hadamard top-k sum requires hidden_size % 512 == 0"
                )
        self.route_warps = 8
        self.cta_threads = 256

    @cute.jit
    def _cast_elem(self, x: cutlass.Float32):
        if cutlass.const_expr(self.float32_output):
            return x
        if cutlass.const_expr(self.is_fp16):
            return cutlass.Float16(x)
        return cutlass.BFloat16(x)

    @cute.jit
    def __call__(
        self,
        fc2_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        topk_weights_ptr: cute.Pointer,
        route_expert_ids_ptr: cute.Pointer,
        expert_map_ptr: cute.Pointer,
        svh_ptr: cute.Pointer,
        weight_num_experts: cutlass.Int32,
        route_num_experts: cutlass.Int32,
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
        topk_weights_flat = cute.make_tensor(
            topk_weights_ptr,
            layout=cute.make_layout((active_m * Int32(self.topk),), stride=(1,)),
        )
        route_expert_ids_flat = cute.make_tensor(
            route_expert_ids_ptr,
            layout=cute.make_layout((active_m * Int32(self.topk),), stride=(1,)),
        )
        expert_map_flat = cute.make_tensor(
            expert_map_ptr,
            layout=cute.make_layout(
                (Int64(route_num_experts) + Int64(1),), stride=(1,)
            ),
        )
        svh_rows = Int64(weight_num_experts)
        if cutlass.const_expr(self.broadcast_svh):
            svh_rows = Int64(1)
        svh_flat = cute.make_tensor(
            svh_ptr,
            layout=cute.make_layout((svh_rows * Int64(self.hidden_size),), stride=(1,)),
        )
        if cutlass.const_expr(self.full_rotation):
            if cutlass.const_expr(self.intermediate_hadamard):
                total = active_m * Int32(self.hidden_size // 512)
            else:
                total = active_m * Int32(self.hidden_size // 128)
            # One CTA owns one H128 output slab.  The full-rotation kernel
            # parallelizes the top-k routes across its eight warps, then has
            # warp zero accumulate the staged route values in router order.
            # The previous launch assigned one slab to one warp; K3 decode
            # consequently launched only 28 active warps and serialized all
            # 16 routes in each of them.
            grid = (total, 1, 1)
        else:
            total = active_m * Int32(self.hidden_size)
            grid = (_covering_count(total, self.cta_threads), 1, 1)
        self.kernel(
            fc2_flat,
            output_flat,
            topk_weights_flat,
            route_expert_ids_flat,
            expert_map_flat,
            svh_flat,
            weight_num_experts,
            route_num_experts,
            active_m,
        ).launch(
            grid=grid,
            block=[self.cta_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        fc2_flat: cute.Tensor,
        output_flat: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        route_expert_ids_flat: cute.Tensor,
        expert_map_flat: cute.Tensor,
        svh_flat: cute.Tensor,
        weight_num_experts: cutlass.Int32,
        route_num_experts: cutlass.Int32,
        active_m: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(
            self.intermediate_hadamard and self.broadcast_svh
        ):
            # The output scale is shared by every expert.  Linearity therefore
            # permits route reduction before the ordinary H128 cancellation:
            #
            #   sum_r w_r * (H128(v_r) * s)
            #     = H128(sum_r w_r * v_r) * s.
            #
            # This applies the two H128 transforms once per output subblock
            # instead of applying the first transform once per routed expert.
            tid = Int32(tidx)
            lane = tid & Int32(31)
            warp = tid >> Int32(5)
            unit = Int32(bidx)
            nblk = Int32(self.hidden_size // 512)
            total_units = active_m * nblk
            if unit < total_units:
                token = unit // nblk
                blk = unit - token * nblk
                block_col = blk * Int32(512)
                reduced_ptr = cute.arch.alloc_smem(cutlass.Float32, 512)
                reduced = cute.make_tensor(
                    reduced_ptr, cute.make_layout(512)
                )

                if warp < Int32(4):
                    sub = warp
                    col0 = block_col + sub * Int32(128) + lane * Int32(4)
                    acc0 = cutlass.Float32(0.0)
                    acc1 = cutlass.Float32(0.0)
                    acc2 = cutlass.Float32(0.0)
                    acc3 = cutlass.Float32(0.0)
                    for route in cutlass.range_constexpr(self.topk):
                        row = token * Int32(self.topk) + Int32(route)
                        raw_expert = route_expert_ids_flat[row].to(Int32)
                        expert = raw_expert
                        if cutlass.const_expr(self.use_expert_map):
                            expert = Int32(-1)
                            if raw_expert >= Int32(0) and raw_expert < Int32(
                                route_num_experts
                            ):
                                expert = expert_map_flat[raw_expert].to(Int32)
                        if expert >= Int32(0) and expert < Int32(weight_num_experts):
                            weight = topk_weights_flat[row].to(cutlass.Float32)
                            base = row * Int32(self.hidden_size) + col0
                            acc0 += (
                                fc2_flat[base + Int32(0)].to(cutlass.Float32)
                                * weight
                            )
                            acc1 += (
                                fc2_flat[base + Int32(1)].to(cutlass.Float32)
                                * weight
                            )
                            acc2 += (
                                fc2_flat[base + Int32(2)].to(cutlass.Float32)
                                * weight
                            )
                            acc3 += (
                                fc2_flat[base + Int32(3)].to(cutlass.Float32)
                                * weight
                            )
                    acc0, acc1, acc2, acc3 = self._had128_quad(
                        acc0, acc1, acc2, acc3, lane
                    )
                    acc0 *= svh_flat[col0 + Int32(0)].to(cutlass.Float32)
                    acc1 *= svh_flat[col0 + Int32(1)].to(cutlass.Float32)
                    acc2 *= svh_flat[col0 + Int32(2)].to(cutlass.Float32)
                    acc3 *= svh_flat[col0 + Int32(3)].to(cutlass.Float32)
                    acc0, acc1, acc2, acc3 = self._had128_quad(
                        acc0, acc1, acc2, acc3, lane
                    )
                    reduced_base = sub * Int32(128) + lane * Int32(4)
                    reduced[reduced_base + Int32(0)] = acc0
                    reduced[reduced_base + Int32(1)] = acc1
                    reduced[reduced_base + Int32(2)] = acc2
                    reduced[reduced_base + Int32(3)] = acc3
                cute.arch.sync_threads()

                if warp == Int32(0):
                    elem = lane * Int32(4)
                    for reg in cutlass.range_constexpr(4):
                        off = elem + Int32(reg)
                        o0, o1, o2, o3 = self._had4_normalized(
                            reduced[off + Int32(0)],
                            reduced[off + Int32(128)],
                            reduced[off + Int32(256)],
                            reduced[off + Int32(384)],
                        )
                        out_base = token * Int32(self.hidden_size) + block_col + off
                        output_flat[out_base + Int32(0)] = o0
                        output_flat[out_base + Int32(128)] = o1
                        output_flat[out_base + Int32(256)] = o2
                        output_flat[out_base + Int32(384)] = o3
            return
        if cutlass.const_expr(self.intermediate_hadamard):
            tid = Int32(tidx)
            lane = tid & Int32(31)
            warp = tid >> Int32(5)
            unit = Int32(bidx)
            nblk = Int32(self.hidden_size // 512)
            total_units = active_m * nblk
            if unit < total_units:
                token = unit // nblk
                blk = unit - token * nblk
                block_col = blk * Int32(512)
                route_values_ptr = cute.arch.alloc_smem(
                    cutlass.Float32, self.topk * 512
                )
                route_values = cute.make_tensor(
                    route_values_ptr, cute.make_layout(self.topk * 512)
                )
                route_weights_ptr = cute.arch.alloc_smem(cutlass.Float32, self.topk)
                route_weights = cute.make_tensor(
                    route_weights_ptr, cute.make_layout(self.topk)
                )

                # Each task is one (route, H128 subblock).  Eight warps cover
                # the 4*topk tasks while retaining the existing independent
                # warp-local H128 implementation.
                for task_group in cutlass.range_constexpr(
                    (self.topk * 4 + self.route_warps - 1) // self.route_warps
                ):
                    task = warp + Int32(task_group * self.route_warps)
                    route = task >> Int32(2)
                    sub = task & Int32(3)
                    valid_route = Int32(0)
                    expert = Int32(-1)
                    row = token * Int32(self.topk) + route
                    if route < Int32(self.topk):
                        raw_expert = route_expert_ids_flat[row].to(Int32)
                        expert = raw_expert
                        if cutlass.const_expr(self.use_expert_map):
                            expert = Int32(-1)
                            if raw_expert >= Int32(0) and raw_expert < Int32(
                                route_num_experts
                            ):
                                expert = expert_map_flat[raw_expert].to(Int32)
                        if expert >= Int32(0) and expert < Int32(weight_num_experts):
                            valid_route = Int32(1)

                    hs0 = cutlass.Float32(0.0)
                    hs1 = cutlass.Float32(0.0)
                    hs2 = cutlass.Float32(0.0)
                    hs3 = cutlass.Float32(0.0)
                    col0 = block_col + sub * Int32(128) + lane * Int32(4)
                    if valid_route != Int32(0):
                        base = row * Int32(self.hidden_size) + col0
                        v0 = fc2_flat[base + Int32(0)].to(cutlass.Float32)
                        v1 = fc2_flat[base + Int32(1)].to(cutlass.Float32)
                        v2 = fc2_flat[base + Int32(2)].to(cutlass.Float32)
                        v3 = fc2_flat[base + Int32(3)].to(cutlass.Float32)
                        h0, h1, h2, h3 = self._had128_quad(v0, v1, v2, v3, lane)
                        if cutlass.const_expr(self.broadcast_svh):
                            sbase = col0
                        else:
                            sbase = expert * Int32(self.hidden_size) + col0
                        hs0 = h0 * svh_flat[sbase + Int32(0)].to(cutlass.Float32)
                        hs1 = h1 * svh_flat[sbase + Int32(1)].to(cutlass.Float32)
                        hs2 = h2 * svh_flat[sbase + Int32(2)].to(cutlass.Float32)
                        hs3 = h3 * svh_flat[sbase + Int32(3)].to(cutlass.Float32)
                    value_base = route * Int32(512) + sub * Int32(128) + lane * Int32(4)
                    if route < Int32(self.topk):
                        route_values[value_base + Int32(0)] = hs0
                        route_values[value_base + Int32(1)] = hs1
                        route_values[value_base + Int32(2)] = hs2
                        route_values[value_base + Int32(3)] = hs3
                        if lane == Int32(0) and sub == Int32(0):
                            weight = cutlass.Float32(0.0)
                            if valid_route != Int32(0):
                                weight = topk_weights_flat[row].to(cutlass.Float32)
                            route_weights[route] = weight
                cute.arch.sync_threads()

                # Four warps reduce one H128 subblock each.  The first 512
                # shared values become the weighted, ordinary-unrotated H512
                # vector, ready for the exact intermediate-Hadamard residual transform.
                if warp < Int32(4):
                    sub = warp
                    acc0 = cutlass.Float32(0.0)
                    acc1 = cutlass.Float32(0.0)
                    acc2 = cutlass.Float32(0.0)
                    acc3 = cutlass.Float32(0.0)
                    for route in cutlass.range_constexpr(self.topk):
                        value_base = Int32(route * 512) + sub * Int32(128) + lane * Int32(4)
                        weight = route_weights[Int32(route)]
                        acc0 += route_values[value_base + Int32(0)] * weight
                        acc1 += route_values[value_base + Int32(1)] * weight
                        acc2 += route_values[value_base + Int32(2)] * weight
                        acc3 += route_values[value_base + Int32(3)] * weight
                    acc0, acc1, acc2, acc3 = self._had128_quad(
                        acc0, acc1, acc2, acc3, lane
                    )
                    reduced_base = sub * Int32(128) + lane * Int32(4)
                    route_values[reduced_base + Int32(0)] = acc0
                    route_values[reduced_base + Int32(1)] = acc1
                    route_values[reduced_base + Int32(2)] = acc2
                    route_values[reduced_base + Int32(3)] = acc3
                cute.arch.sync_threads()

                if warp == Int32(0):
                    elem = lane * Int32(4)
                    for reg in cutlass.range_constexpr(4):
                        off = elem + Int32(reg)
                        o0, o1, o2, o3 = self._had4_normalized(
                            route_values[off + Int32(0)],
                            route_values[off + Int32(128)],
                            route_values[off + Int32(256)],
                            route_values[off + Int32(384)],
                        )
                        out_base = token * Int32(self.hidden_size) + block_col + off
                        output_flat[out_base + Int32(0)] = o0
                        output_flat[out_base + Int32(128)] = o1
                        output_flat[out_base + Int32(256)] = o2
                        output_flat[out_base + Int32(384)] = o3
            return
        if cutlass.const_expr(self.full_rotation):
            tid = Int32(tidx)
            lane = tid & Int32(31)
            warp = tid >> Int32(5)
            unit = Int32(bidx)
            nblk = Int32(self.hidden_size // 128)
            total_units = active_m * nblk
            if unit < total_units:
                token = unit // nblk
                blk = unit - token * nblk
                col0 = blk * Int32(128) + lane * Int32(4)
                route_values_ptr = cute.arch.alloc_smem(
                    cutlass.Float32, self.topk * 128
                )
                route_values = cute.make_tensor(
                    route_values_ptr, cute.make_layout(self.topk * 128)
                )
                route_weights_ptr = cute.arch.alloc_smem(cutlass.Float32, self.topk)
                route_weights = cute.make_tensor(
                    route_weights_ptr, cute.make_layout(self.topk)
                )
                for route_group in cutlass.range_constexpr(
                    (self.topk + self.route_warps - 1) // self.route_warps
                ):
                    route = warp + Int32(route_group * self.route_warps)
                    valid_route = Int32(0)
                    expert = Int32(-1)
                    row = token * Int32(self.topk) + route
                    if route < Int32(self.topk):
                        raw_expert = route_expert_ids_flat[row].to(Int32)
                        expert = raw_expert
                        if cutlass.const_expr(self.use_expert_map):
                            expert = Int32(-1)
                            if raw_expert >= Int32(0) and raw_expert < Int32(
                                route_num_experts
                            ):
                                expert = expert_map_flat[raw_expert].to(Int32)
                        if expert >= Int32(0) and expert < Int32(weight_num_experts):
                            valid_route = Int32(1)

                    hs0 = cutlass.Float32(0.0)
                    hs1 = cutlass.Float32(0.0)
                    hs2 = cutlass.Float32(0.0)
                    hs3 = cutlass.Float32(0.0)
                    if valid_route != Int32(0):
                        base = row * Int32(self.hidden_size) + col0
                        v0 = fc2_flat[base + Int32(0)].to(cutlass.Float32)
                        v1 = fc2_flat[base + Int32(1)].to(cutlass.Float32)
                        v2 = fc2_flat[base + Int32(2)].to(cutlass.Float32)
                        v3 = fc2_flat[base + Int32(3)].to(cutlass.Float32)
                        h0, h1, h2, h3 = self._had128_quad(v0, v1, v2, v3, lane)
                        if cutlass.const_expr(self.broadcast_svh):
                            sbase = col0
                        else:
                            sbase = expert * Int32(self.hidden_size) + col0
                        s0 = svh_flat[sbase + Int32(0)].to(cutlass.Float32)
                        s1 = svh_flat[sbase + Int32(1)].to(cutlass.Float32)
                        s2 = svh_flat[sbase + Int32(2)].to(cutlass.Float32)
                        s3 = svh_flat[sbase + Int32(3)].to(cutlass.Float32)
                        hs0 = h0 * s0
                        hs1 = h1 * s1
                        hs2 = h2 * s2
                        hs3 = h3 * s3
                    value_base = route * Int32(128) + lane * Int32(4)
                    if route < Int32(self.topk):
                        route_values[value_base + Int32(0)] = hs0
                        route_values[value_base + Int32(1)] = hs1
                        route_values[value_base + Int32(2)] = hs2
                        route_values[value_base + Int32(3)] = hs3
                        if lane == Int32(0):
                            weight = cutlass.Float32(0.0)
                            if valid_route != Int32(0):
                                weight = topk_weights_flat[row].to(cutlass.Float32)
                            route_weights[route] = weight
                cute.arch.sync_threads()

                if warp == Int32(0):
                    acc0 = cutlass.Float32(0.0)
                    acc1 = cutlass.Float32(0.0)
                    acc2 = cutlass.Float32(0.0)
                    acc3 = cutlass.Float32(0.0)
                    for route in cutlass.range_constexpr(self.topk):
                        value_base = Int32(route * 128) + lane * Int32(4)
                        weight = route_weights[Int32(route)]
                        acc0 += route_values[value_base + Int32(0)] * weight
                        acc1 += route_values[value_base + Int32(1)] * weight
                        acc2 += route_values[value_base + Int32(2)] * weight
                        acc3 += route_values[value_base + Int32(3)] * weight
                    out_base = token * Int32(self.hidden_size) + col0
                    output_flat[out_base + Int32(0)] = acc0
                    output_flat[out_base + Int32(1)] = acc1
                    output_flat[out_base + Int32(2)] = acc2
                    output_flat[out_base + Int32(3)] = acc3
            return
        idx = Int32(bidx) * Int32(self.cta_threads) + Int32(tidx)
        total = active_m * Int32(self.hidden_size)
        if idx < total:
            token = idx // Int32(self.hidden_size)
            col = idx - token * Int32(self.hidden_size)
            acc = cutlass.Float32(0.0)
            for route in cutlass.range_constexpr(self.topk):
                row = token * Int32(self.topk) + Int32(route)
                valid_route = Int32(1)
                if cutlass.const_expr(self.use_expert_map):
                    raw_expert = route_expert_ids_flat[row].to(Int32)
                    expert = Int32(-1)
                    if raw_expert >= Int32(0) and raw_expert < route_num_experts:
                        expert = expert_map_flat[raw_expert].to(Int32)
                    if expert < Int32(0) or expert >= weight_num_experts:
                        valid_route = Int32(0)
                if valid_route != Int32(0):
                    route_value = fc2_flat[Int64(row) * Int64(self.hidden_size) + Int64(col)].to(
                        cutlass.Float32
                    )
                    acc += _materialize_w4a16_topk_route_f32(route_value)
            output_flat[idx] = self._cast_elem(acc)

    @cute.jit
    def _had128_quad(
        self,
        v0: cutlass.Float32,
        v1: cutlass.Float32,
        v2: cutlass.Float32,
        v3: cutlass.Float32,
        lane: Int32,
    ):
        s0 = v0 + v1
        d0 = v0 - v1
        s1 = v2 + v3
        d1 = v2 - v3
        h0 = s0 + s1
        h1 = d0 + d1
        h2 = s0 - s1
        h3 = d0 - d1
        for i in cutlass.range_constexpr(5):
            st = 1 << i
            p0 = cute.arch.shuffle_sync_bfly(h0, offset=st)
            p1 = cute.arch.shuffle_sync_bfly(h1, offset=st)
            p2 = cute.arch.shuffle_sync_bfly(h2, offset=st)
            p3 = cute.arch.shuffle_sync_bfly(h3, offset=st)
            if (lane & Int32(st)) != Int32(0):
                h0 = p0 - h0
                h1 = p1 - h1
                h2 = p2 - h2
                h3 = p3 - h3
            else:
                h0 = p0 + h0
                h1 = p1 + h1
                h2 = p2 + h2
                h3 = p3 + h3
        rs = cutlass.Float32(0.088388347648)
        return h0 * rs, h1 * rs, h2 * rs, h3 * rs

    @cute.jit
    def _had4_normalized(
        self,
        v0: cutlass.Float32,
        v1: cutlass.Float32,
        v2: cutlass.Float32,
        v3: cutlass.Float32,
    ):
        s0 = v0 + v1
        d0 = v0 - v1
        s1 = v2 + v3
        d1 = v2 - v3
        rs = cutlass.Float32(0.5)
        return (
            (s0 + s1) * rs,
            (d0 + d1) * rs,
            (s0 - s1) * rs,
            (d0 - d1) * rs,
        )



_CACHE: dict[tuple, W4A16GemmCompileResult] = {}
_FUSED_CACHE: dict[tuple, W4A16FusedMoeCompileResult] = {}
_ACTIVATION_CACHE: dict[tuple, W4A16ActivationCompileResult] = {}
_SUM_CACHE: dict[tuple, W4A16TopKSumCompileResult] = {}
_SMALL_M_DIRECT_CACHE: dict[tuple, _W4A16SmallMDirectLaunch] = {}
_FC2_DIRECT_CACHE: dict[tuple, _W4A16FC2DirectLaunch] = {}
register_program_cache(_CACHE)
register_program_cache(_FUSED_CACHE)
register_program_cache(_ACTIVATION_CACHE)
register_program_cache(_SUM_CACHE)
register_program_cache(_SMALL_M_DIRECT_CACHE)
register_program_cache(_FC2_DIRECT_CACHE)


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
            "scale_format must be one of 'e4m3_k16', 'e8m0_k32', "
            "'e4m3_k32', or 'iq2_xs', "
            f"got {scale_format!r}"
        ) from exc


def _scale_group_size(scale_format: str) -> int:
    return 32 if _normalize_scale_format(scale_format) in (
        "e8m0_k32",
        "e4m3_k32",
    ) else 16


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
    if os.environ.get("B12X_W4A16_SMALL_M_DIRECT", "1") == "0":
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


def _small_m_direct_host_barrier_reset_enabled() -> bool:
    """Whether to clear reusable small-M grid-barrier state before launch.

    The micro-kernel barrier resets its arrival counter and advances its epoch
    before releasing the grid, so completed launches can safely reuse both
    scalars.  Keep the historical host reset as the default while the
    persistent-epoch path is evaluated independently and end-to-end.
    """
    return os.environ.get("B12X_W4A16_SMALL_M_HOST_BARRIER_RESET", "1") != "0"


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
    compiled = b12x_compile(
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
        Int32(num_experts),
        Int32(m),
        Int32(kernel.grid_x),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_facts(
            "moe.w4a16.small_m_direct",
            3,
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


def _w4a16_fc2_direct_expert_capacity(num_experts: int) -> int:
    """Return the compile-time capacity bucket for FC2-only X4T endpoints.

    The resident endpoint count is runtime artifact data. Keeping a bounded
    power-of-two capacity in the CuTe tensor type avoids one compilation per
    layer without changing route-ID based address arithmetic.
    """

    num_experts = int(num_experts)
    if num_experts < 1:
        raise ValueError("W4A16 FC2-only execution requires at least one expert")
    return max(
        _FC2_DIRECT_MIN_EXPERT_CAPACITY,
        1 << (num_experts - 1).bit_length(),
    )


def _compile_w4a16_fc2_direct(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk_ids_dtype: torch.dtype,
    device: torch.device,
) -> _W4A16FC2DirectLaunch:
    """Compile the runtime-M native-MXFP4 FC2-only microkernel."""

    if topk_ids_dtype not in (torch.int32, torch.int64):
        raise TypeError("W4A16 FC2-only route IDs must be int32 or int64")
    # Expert count changes the fake tensor extent used by CuTe, but it does
    # not change the FC2 address arithmetic: every access is based on the
    # runtime route ID and fixed per-expert strides.  Compiling the exact
    # resident count made a model with matrix-granular endpoints JIT one
    # otherwise-identical kernel per layer.  Use a capacity bucket instead;
    # route validation still guarantees IDs are within the actual tensors.
    expert_capacity = _w4a16_fc2_direct_expert_capacity(num_experts)
    cache_key = (
        "w4a16_fc2_direct",
        int(device.index or 0),
        int(hidden_size),
        int(intermediate_size),
        int(expert_capacity),
        topk_ids_dtype,
    )
    cached = _FC2_DIRECT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    kernel = MoEMicroKernelW4A16SmallMDirect(
        activation="silu",
        fast_math=False,
        share_input_across_experts=False,
        share_expert_scales=True,
        single_token=False,
        scale_format="e8m0_k32",
        compile_time_phase=2,
    )
    # m=2 selects the runtime-M body. FC2 does not retain token activations in
    # registers, so the compiled loop remains valid above the fused kernel's
    # ordinary eight-token ceiling.
    kernel.configure(
        2,
        int(hidden_size),
        int(intermediate_size),
        1,
        int(expert_capacity),
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
    compiled = b12x_compile(
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
        Int32(expert_capacity),
        Int32(2),
        Int32(kernel.grid_x),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_facts(
            "moe.w4a16.fc2_direct",
            5,
            ("device_index", int(device.index or 0)),
            ("hidden_size", int(hidden_size)),
            ("intermediate_size", int(intermediate_size)),
            ("expert_capacity", int(expert_capacity)),
            ("topk_ids_dtype", str(topk_ids_dtype)),
            ("grid_x", int(kernel.grid_x)),
        ),
    )
    launch = _W4A16FC2DirectLaunch(
        compiled=compiled,
        grid_x=int(kernel.grid_x),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_experts=int(expert_capacity),
        topk_ids_dtype=topk_ids_dtype,
    )
    _FC2_DIRECT_CACHE[cache_key] = launch
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
    weight_layout: str = "packed",
    scale_format: str = "e4m3_k16",
    w13_layout: str = "packed",
    trellis_bits: int = 3,
    trellis_codebook: str = LUT_E4M3,
    trellis_pair_kind: str | None = None,
    trellis_rate_axis: str | None = None,
    dense_route_fast_path: bool = False,
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
        weight_layout=weight_layout,
        scale_format=scale_format,
        w13_layout=w13_layout,
        trellis_bits=trellis_bits,
        trellis_codebook=trellis_codebook,
        trellis_pair_kind=trellis_pair_kind,
        trellis_rate_axis=trellis_rate_axis,
        dense_route_fast_path=bool(dense_route_fast_path),
        schedule_whole_tiles=weight_layout == "trellis_t256",
    )
    cache_key = (
        "w4a16_gemm",
        device,
        kernel.__cache_key__,
    )
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return attach_programs(
            replace(cached, max_m_blocks=max_m_blocks, blocks_per_sm=kernel.blocks_per_sm),
            cached.compiled,
        )

    compile_size_m = _fake_m_for_specialization(size_m)
    compile_route_blocks = 1
    compile_route_slots = compile_route_blocks * int(moe_block_size)
    a_fake = make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    if weight_layout == "trellis_t256":
        b_fake_elements = (
            num_experts * (size_k // 16) * (size_n // 16) * (8 * int(trellis_bits))
        )
    else:
        b_fake_elements = num_experts * (size_k // 16) * (size_n // 16 * 32)
    b_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (b_fake_elements,),
        assumed_align=16,
    )
    c_fake = make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
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
    trellis_lut_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (_LUT_E4M3_VALUE_TABLE_ENTRIES,),
        assumed_align=16,
    )

    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = b12x_compile(
        kernel,
        a_fake,
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
        trellis_lut_fake,
        Int32(compile_size_m),
        Int32(1),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "moe.w4a16.gemm",
            4,
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
        weight_layout=weight_layout,
        scale_format=scale_format,
        w13_layout=w13_layout,
        dense_route_fast_path=bool(dense_route_fast_path),
        trellis_bits=int(trellis_bits),
        trellis_codebook=str(trellis_codebook).lower(),
        trellis_pair_kind=trellis_pair_kind,
        trellis_rate_axis=trellis_rate_axis,
    )
    attach_programs(result, compiled)
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
    direct_token_capacity: int | None = None,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    weight_layout: str = "packed",
    scale_format: str = "e4m3_k16",
    w13_layout: str = "w13",
    trellis_bits: int = 3,
    trellis_codebook: str = LUT_E4M3,
    fc1_trellis_pair_kind: str | None = None,
    fc2_trellis_pair_kind: str | None = None,
    direct_topk_routes: bool = False,
    use_expert_map: bool = False,
    tc_decode_fused_sum: bool = False,
    prefill_fused_sum_fp32: bool = False,
    collect_activation_amax: bool = False,
    force_tile_config: tuple[int, int, int, int] | None = None,
    pipeline_stages: int | None = None,
    skip_empty_m_blocks: bool | None = None,
    intermediate_rotation: bool = False,
    full_rotation: bool = False,
    intermediate_hadamard: bool = False,
    rotation_input_dtype: str | None = None,
    broadcast_suh: bool = False,
    trellis_decode_table: str = "auto",
    _require_cached: bool = False,
    small_m_occupancy: int | None = None,
) -> W4A16FusedMoeCompileResult:
    if small_m_occupancy is None:
        small_m_occupancy = _w4a16_small_m_occupancy()
    scale_format = _normalize_scale_format(scale_format)
    intermediate_rotation = bool(intermediate_rotation)
    full_rotation = bool(full_rotation)
    intermediate_hadamard = bool(intermediate_hadamard)
    rotation_input_dtype = (
        element_dtype if rotation_input_dtype is None else str(rotation_input_dtype)
    )
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
    trellis_bits = int(trellis_bits)
    if weight_layout == "trellis_t256":
        if trellis_bits not in _TRELLIS256_BITS:
            raise ValueError(
                f"trellis_t256 bits must be one of {_TRELLIS256_BITS}, got {trellis_bits}"
            )
    elif trellis_bits != 3:
        raise ValueError("trellis_bits is only valid for trellis_t256 weights")
    # Existing 3-bpw scheduling was conservatively planned as 4 bpw. Keep that
    # grid contract stable for D6; widen only the 5/6-bpw specializations.
    weight_bits = max(4, trellis_bits) if weight_layout == "trellis_t256" else 4
    # GATE 5: the PRODUCTION 256-weight-tile fused-megakernel B-staging is now
    # wired (per-warp native [K/16,N/16,8*bits u32] tile staging + the per-lane
    # bitrate-specialized read) and ADMITTED at 3 bpw against a full-GEMM
    # numeric oracle (megakernel FC1->silu->FC2 vs a torch reference over tiles;
    # tests/test_trellis256_fullgemm_oracle.py, max rel-err <= 2^-6).  The GATE-4
    # NotImplementedError gate is therefore lifted.  See GATE5_RESULT.md.
    if weight_layout == "modelopt":
        if w13_layout not in _MODEL_OPT_W13_LAYOUTS:
            raise ValueError(f"unsupported W4A16 w13_layout {w13_layout!r}")
    elif weight_layout == "trellis_t256":
        if w13_layout not in _TRELLIS256_W13_LAYOUTS:
            raise ValueError(f"unsupported trellis_t256 w13_layout {w13_layout!r}")
    else:
        w13_layout = "packed"
    direct_topk_routes = bool(direct_topk_routes)
    use_expert_map = bool(use_expert_map)
    tc_decode_fused_sum = bool(tc_decode_fused_sum)
    prefill_fused_sum_fp32 = bool(prefill_fused_sum_fp32)
    if tc_decode_fused_sum and prefill_fused_sum_fp32:
        raise ValueError(
            "TC-decode and large-M FP32 route reduction are mutually exclusive"
        )
    if use_expert_map and not direct_topk_routes:
        raise ValueError("use_expert_map requires direct_topk_routes")
    collect_activation_amax = bool(collect_activation_amax)
    if collect_activation_amax and (
        direct_topk_routes or tc_decode_fused_sum or prefill_fused_sum_fp32
    ):
        raise ValueError(
            "W4A16 activation amax collection requires the route-packed fused path"
        )
    if collect_activation_amax and intermediate_rotation:
        raise ValueError(
            "W4A16 activation amax collection is incompatible with intermediate rotation"
        )
    if full_rotation:
        if not intermediate_rotation:
            raise ValueError("full_rotation requires intermediate_rotation")
        if weight_layout != "trellis_t256":
            raise ValueError("full_rotation is only supported for trellis_t256")
        if element_dtype != "fp16":
            raise ValueError("full_rotation requires element_dtype='fp16'")
        if rotation_input_dtype not in {"bf16", "fp16"}:
            raise ValueError(
                "rotation_input_dtype must be 'bf16' or 'fp16' for full_rotation"
            )
        if tc_decode_fused_sum or prefill_fused_sum_fp32:
            raise ValueError("full_rotation is incompatible with fused route reduction")
        if apply_router_weight_on_input:
            raise ValueError(
                "full_rotation requires apply_router_weight_on_input=False"
            )
    if intermediate_hadamard and not full_rotation:
        raise ValueError("intermediate_hadamard requires full_rotation")
    if collect_activation_amax and weight_layout == "trellis_t256":
        raise NotImplementedError(
            "trellis_t256 activation-amax collection is not exposed through the "
            "registered launch ABI; refusing to compile a bitrate-ambiguous kernel"
        )
    # The TC-decode path validates M in {1,2,4,8} itself and uses direct-topk
    # routing for the whole {1,2,4,8} range, so it lifts the default decode cap.
    direct_topk_m_cap = (
        _W4A16_SMALL_M_DIRECT_MAX_M
        if tc_decode_fused_sum or use_expert_map
        else _MAX_DIRECT_TOPK_ROUTE_M
    )
    direct_weight_layout_ok = weight_layout in {"packed", "iq2_xs", "iq2_xxs", "q8_0"} or (
        full_rotation and weight_layout == "trellis_t256"
    )
    if direct_topk_routes and (
        int(size_m) > direct_topk_m_cap
        or not direct_weight_layout_ok
        or bool(zero_fc2_output)
    ):
        raise ValueError(
            "direct_topk_routes is only valid for supported small-M W4A16 decode"
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
        weight_bits=weight_bits,
        allow_logical_tail=allow_native_logical_tail,
        small_m_occupancy=small_m_occupancy,
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
        weight_bits=weight_bits,
        allow_logical_tail=allow_native_logical_tail,
        small_m_occupancy=small_m_occupancy,
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
            weight_bits=weight_bits,
            allow_logical_tail=allow_native_logical_tail,
            small_m_occupancy=small_m_occupancy,
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
            weight_bits=weight_bits,
            allow_logical_tail=allow_native_logical_tail,
            small_m_occupancy=small_m_occupancy,
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
        and weight_layout not in BLOCK_CODECS
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
            weight_bits=weight_bits,
            small_m_occupancy=small_m_occupancy,
        ):
            fc1_tile_n = 256
            fc1_tile_k = wide_fc1_tile_k
            fc1_cta_threads = 256
    # Packed decode FC2 wide-N override: the direct path right-sizes grid_x to
    # the FC1 mn-tile count, while the expert-packed path uses the same
    # persistent cap with a device-known live block count. FC2
    # (N=hidden_size, K=intermediate_size) with the default tile_n=128 produces
    # route_blocks*(hidden_size/128) mn-tiles -- roughly double FC1's count --
    # so FC2 would need ~2 persistent waves while FC1 fits in 1; that second
    # FC2 wave is pure serialized tail latency on the bandwidth-bound decode.
    # Selecting tile_n=256 for FC2 (a 256-wide N slab per CTA over full K)
    # halves FC2's mn-tile count so it also fits one wave. Guarded by smem-fit,
    # hidden_size%256==0, and matching cta_threads so the fused single
    # thread-geometry contract is preserved.
    if (
        weight_layout == "packed"
        and int(moe_block_size) == 8
        and int(size_m) <= _TC_DECODE_MAX_M
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
            weight_bits=weight_bits,
            small_m_occupancy=small_m_occupancy,
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
        if _candidate_tile_fits(
            problem_n=hidden_size,
            problem_k=intermediate_size,
            cta_m_blocks=_covering_count(moe_block_size, 16),
            tile_n=512,
            tile_k=ultra_fc2_tile_k,
            cta_threads=256,
            max_shared_mem=int(max_shared_mem) - 512,
            scale_format=scale_format,
            weight_layout=weight_layout,
            weight_bits=weight_bits,
            allow_qualified_fc2_tile=True,
            small_m_occupancy=small_m_occupancy,
        ):
            fc2_tile_n = 512
            fc2_tile_k = ultra_fc2_tile_k
            fc2_cta_threads = 256
    if force_tile_config is not None:
        # Some weight layouts are packed for a specific CTA N-tile. An explicit
        # (fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n) tuple therefore pins
        # one geometry across every M regime. Tile values are part of the GEMM
        # cache key, so distinct packed layouts cannot collide.
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
                weight_bits=weight_bits,
                allow_logical_tail=allow_native_logical_tail,
                allow_qualified_fc2_tile=name == "fc2",
                uses_m_block_8=moe_block_size == 8,
                pipeline_stages=pipeline_stages,
                small_m_occupancy=small_m_occupancy,
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
        trellis_bits=trellis_bits,
        trellis_codebook=trellis_codebook,
        fc1_trellis_pair_kind=fc1_trellis_pair_kind,
        fc2_trellis_pair_kind=fc2_trellis_pair_kind,
        direct_topk_routes=direct_topk_routes,
        use_expert_map=use_expert_map,
        tc_decode_fused_sum=tc_decode_fused_sum,
        prefill_fused_sum_fp32=prefill_fused_sum_fp32,
        collect_activation_amax=collect_activation_amax,
        intermediate_rotation=intermediate_rotation,
        full_rotation=full_rotation,
        intermediate_hadamard=intermediate_hadamard,
        rotation_input_dtype=rotation_input_dtype,
        broadcast_suh=broadcast_suh,
        pipeline_stages=pipeline_stages,
        skip_empty_m_blocks=skip_empty_m_blocks,
        trellis_decode_table=trellis_decode_table,
        small_m_occupancy=small_m_occupancy,
    )
    cache_key = (
        "w4a16_fused_moe",
        device,
        kernel.__cache_key__,
    )
    cached = _FUSED_CACHE.get(cache_key)
    if cached is None and _require_cached:
        raise RuntimeError(
            "W4A16 fused MoE launch is not resolved for CUDA graph capture "
            f"(m={size_m}, moe_block_size={moe_block_size}, "
            f"max_m_blocks={max_m_blocks}); run an eager warmup at this token "
            "count before capturing"
        )

    # Packed routes use rounded storage capacity; native direct launches use
    # the exact planned row count, which may be smaller than that bucket.
    direct_m = size_m if direct_token_capacity is None else int(direct_token_capacity)
    if not 0 < direct_m <= size_m:
        raise ValueError("direct token capacity must be within packed capacity")
    small_m_direct_launches = []
    if (not collect_activation_amax) and _small_m_direct_supported(
        m=direct_m,
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
            direct = _compile_w4a16_small_m_direct(
                m=direct_m,
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
            small_m_direct_launches.append(direct)

    if cached is not None:
        return attach_programs(
            replace(
                cached,
                size_m=size_m,
                num_experts=num_experts,
                max_m_blocks=max_m_blocks,
                blocks_per_sm=kernel.blocks_per_sm,
                small_m_direct_launches=tuple(small_m_direct_launches),
            ),
            cached.compiled,
            *(launch.compiled for launch in small_m_direct_launches),
        )

    compile_size_m = _fake_m_for_specialization(size_m)
    compile_routed_rows = int(compile_size_m) * int(top_k)
    compile_route_blocks = compile_routed_rows if direct_topk_routes else 1
    compile_route_slots = compile_route_blocks * int(moe_block_size)
    packed_route_fake_elements = (
        compile_routed_rows if direct_topk_routes else compile_route_slots
    )
    a_fake = make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    rotation_input_fake = make_ptr(
        _cutlass_element_dtype(rotation_input_dtype),
        16,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    weight_cutlass_dtype = (
        cutlass.Uint8 if weight_layout == "modelopt" else cutlass.Int32
    )
    # Weight and E-sized metadata tensors are pointer-only launch arguments.
    # Their runtime views are reconstructed inside W4A16FusedMoeKernel.__call__
    # from weight_num_experts, keeping the compiled ABI independent of a
    # layer's compact tier expert count.
    w13_fake = make_ptr(
        weight_cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    w2_fake = make_ptr(
        weight_cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
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
        cutlass.Float32 if kernel.prefill_fused_sum_fp32 else cutlass_dtype,
        ((compile_size_m if kernel.prefill_fused_sum_fp32 else compile_routed_rows)
         * hidden_size,),
        assumed_align=16,
    )
    pair_metadata_cutlass_dtype = (
        cutlass.Int64
        if fc1_trellis_pair_kind == "P33_P43"
        else cutlass.Int32
    )
    w13_scales_fake = make_ptr(
        pair_metadata_cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    w2_scales_fake = make_ptr(
        pair_metadata_cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    w13_global_fake = make_ptr(
        cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    w2_global_fake = make_ptr(
        cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16
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
    rot_scales_fake = make_ptr(
        cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    suh_gate_fake = make_ptr(
        cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    suh_up_fake = make_ptr(
        cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    expert_map_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    trellis_lut_fake = make_ptr(
        cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
    )

    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = b12x_compile(
        kernel,
        a_fake,
        a_fake,
        rotation_input_fake,
        w13_fake,
        w2_fake,
        Int64(1),
        Int64(1),
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
        rot_scales_fake,
        suh_gate_fake,
        suh_up_fake,
        expert_map_fake,
        trellis_lut_fake,
        trellis_lut_fake,
        Int32(num_experts),
        Int32(num_experts),
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "moe.w4a16.fused_moe",
            8,
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
        use_expert_map=kernel.use_expert_map,
        scale_format=scale_format,
        tc_decode_fused_sum=bool(tc_decode_fused_sum),
        prefill_fused_sum_fp32=bool(prefill_fused_sum_fp32),
        collect_activation_amax=collect_activation_amax,
        schedule_whole_tiles=kernel.schedule_whole_tiles,
        intermediate_rotation=intermediate_rotation,
        dual_a=kernel.dual_a,
        trellis_bits=trellis_bits,
        trellis_codebook=kernel.trellis_codebook,
        fc1_trellis_pair_kind=kernel.fc1_trellis_pair_kind,
        fc2_trellis_pair_kind=kernel.fc2_trellis_pair_kind,
        full_rotation=full_rotation,
        intermediate_hadamard=intermediate_hadamard,
        rotation_input_dtype=rotation_input_dtype,
        cta_threads=kernel.cta_threads,
        shared_memory_bytes=kernel.shared_words * 4,
        broadcast_suh=bool(broadcast_suh),
        small_m_direct_launches=tuple(small_m_direct_launches),
        trellis_direct_lut=kernel.trellis_direct_lut,
    )
    attach_programs(result, compiled, *(launch.compiled for launch in small_m_direct_launches))
    _FUSED_CACHE[cache_key] = result
    return result

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
    return int(num_experts) * (int(size_k) // 16) * (int(size_n) // 16 * 32)




def clear_w4a16_kernel_cache() -> None:
    _CACHE.clear()
    _FUSED_CACHE.clear()
    _ACTIVATION_CACHE.clear()
    _SUM_CACHE.clear()
    _SMALL_M_DIRECT_CACHE.clear()
    _FC2_DIRECT_CACHE.clear()


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
    compiled = b12x_compile(
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
    full_rotation: bool = False,
    intermediate_hadamard: bool = False,
    num_experts: int = 0,
    route_num_experts: int = 0,
    route_ids_dtype: torch.dtype = torch.int32,
    use_expert_map: bool = False,
    broadcast_svh: bool = False,
    float32_output: bool = False,
) -> W4A16TopKSumCompileResult:
    cutlass_dtype = _cutlass_element_dtype(element_dtype)
    if route_ids_dtype not in (torch.int32, torch.int64):
        raise TypeError("top-k route expert ids must be int32 or int64")
    route_cutlass_dtype = (
        cutlass.Int32 if route_ids_dtype == torch.int32 else cutlass.Int64
    )
    cache_key = (
        "w4a16_topk_sum",
        element_dtype,
        topk,
        hidden_size,
        bool(full_rotation),
        bool(intermediate_hadamard),
        None if full_rotation else int(num_experts),
        None if full_rotation else int(route_num_experts),
        str(route_ids_dtype),
        bool(use_expert_map),
        bool(broadcast_svh),
        bool(float32_output),
    )
    cached = _SUM_CACHE.get(cache_key)
    if cached is not None:
        return attach_programs(
            replace(
                cached,
                num_experts=int(num_experts),
                route_num_experts=int(route_num_experts),
            ),
            cached,
        )

    fc2_fake = make_ptr(cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    output_dtype = cutlass.Float32 if full_rotation or float32_output else cutlass_dtype
    output_fake = make_ptr(output_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    topk_weights_fake = make_ptr(
        cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    route_ids_align = 4 if route_ids_dtype == torch.int32 else 8
    route_ids_fake = make_ptr(
        route_cutlass_dtype,
        route_ids_align,
        cute.AddressSpace.gmem,
        assumed_align=route_ids_align,
    )
    expert_map_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    svh_fake = make_ptr(cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16)
    kernel = W4A16TopKSumKernel(
        topk=topk,
        hidden_size=hidden_size,
        element_dtype=element_dtype,
        full_rotation=full_rotation,
        intermediate_hadamard=intermediate_hadamard,
        num_experts=num_experts,
        route_num_experts=route_num_experts,
        use_expert_map=use_expert_map,
        broadcast_svh=broadcast_svh,
        float32_output=float32_output,
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = b12x_compile(
        kernel,
        fc2_fake,
        output_fake,
        topk_weights_fake,
        route_ids_fake,
        expert_map_fake,
        svh_fake,
        Int32(num_experts),
        Int32(route_num_experts),
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "moe.w4a16.topk_sum",
            4,
            cache_key,
        ),
    )
    result = W4A16TopKSumCompileResult(
        compiled=compiled,
        m=0,
        topk=topk,
        hidden_size=hidden_size,
        full_rotation=bool(full_rotation),
        intermediate_hadamard=bool(intermediate_hadamard),
        num_experts=int(num_experts),
        route_num_experts=int(route_num_experts),
        route_ids_dtype=route_ids_dtype,
        use_expert_map=bool(use_expert_map),
        broadcast_svh=bool(broadcast_svh),
    )
    attach_programs(result, compiled)
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
    *,
    launcher: _W4A16SmallMDirectLaunch | None = None,
) -> None:
    swiglu_limit = float(swiglu_limit_value) if has_swiglu_limit else None
    direct_launch = launcher if launcher is not None else _compile_w4a16_small_m_direct(
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
        Int32(num_experts),
        Int32(m),
        Int32(direct_launch.grid_x),
        cuda.CUstream(stream_int),
    )


@torch.library.custom_op(
    "b12x::w4a16_small_m_direct_launch",
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


def _w4a16_fc2_direct_launch_flat(
    intermediate: torch.Tensor,
    w2_u8: torch.Tensor,
    w2_scale_u8: torch.Tensor,
    w2_global_scale: torch.Tensor,
    route_ids: torch.Tensor,
    route_weights: torch.Tensor,
    output: torch.Tensor,
    barrier_count: torch.Tensor,
    barrier_epoch: torch.Tensor,
    m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    stream_int: int,
    *,
    launcher=None,
) -> None:
    launch = launcher if launcher is not None else _compile_w4a16_fc2_direct(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk_ids_dtype=route_ids.dtype,
        device=intermediate.device,
    )

    def ptr(dt, tensor: torch.Tensor):
        return make_ptr(dt, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    ids_dtype = cutlass.Int64 if route_ids.dtype == torch.int64 else cutlass.Int32
    launch.compiled(
        ptr(cutlass.BFloat16, intermediate),
        ptr(cutlass.Uint8, w2_u8),
        ptr(cutlass.Uint8, w2_scale_u8.view(torch.uint8)),
        ptr(cutlass.Float32, w2_global_scale),
        ptr(cutlass.Float32, w2_global_scale),
        ptr(cutlass.Float32, w2_global_scale),
        ptr(cutlass.Uint32, intermediate.view(torch.uint32)),
        ptr(cutlass.Uint8, w2_u8),
        ptr(cutlass.Uint8, w2_scale_u8.view(torch.uint8)),
        ptr(cutlass.Float32, w2_global_scale),
        ptr(ids_dtype, route_ids),
        ptr(cutlass.Float32, route_weights),
        ptr(cutlass.BFloat16, output),
        barrier_count,
        barrier_epoch,
        Int32(num_experts),
        Int32(m),
        Int32(launch.grid_x),
        cuda.CUstream(stream_int),
    )


@torch.library.custom_op(
    "b12x::w4a16_fc2_direct_launch",
    mutates_args="unknown",
)
def _w4a16_fc2_direct_launch_op(
    intermediate: torch.Tensor,
    w2_u8: torch.Tensor,
    w2_scale_u8: torch.Tensor,
    w2_global_scale: torch.Tensor,
    route_ids: torch.Tensor,
    route_weights: torch.Tensor,
    output: torch.Tensor,
    barrier_count: torch.Tensor,
    barrier_epoch: torch.Tensor,
    m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    stream_int: int,
) -> None:
    _w4a16_fc2_direct_launch_flat(
        intermediate=intermediate,
        w2_u8=w2_u8,
        w2_scale_u8=w2_scale_u8,
        w2_global_scale=w2_global_scale,
        route_ids=route_ids,
        route_weights=route_weights,
        output=output,
        barrier_count=barrier_count,
        barrier_epoch=barrier_epoch,
        m=m,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        stream_int=stream_int,
    )


@_w4a16_fc2_direct_launch_op.register_fake
def _w4a16_fc2_direct_launch_fake(
    intermediate: torch.Tensor,
    w2_u8: torch.Tensor,
    w2_scale_u8: torch.Tensor,
    w2_global_scale: torch.Tensor,
    route_ids: torch.Tensor,
    route_weights: torch.Tensor,
    output: torch.Tensor,
    barrier_count: torch.Tensor,
    barrier_epoch: torch.Tensor,
    m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    stream_int: int,
) -> None:
    return None


_ROT_SCALES_DUMMY: dict[object, torch.Tensor] = {}


def _rot_scales_dummy(device: torch.device) -> torch.Tensor:
    """Tiny fp16 placeholder for the rot_scales kernel slot when the
    intermediate-rotation epilogue is disabled (never dereferenced by the
    const_expr-gated kernel; keeps one compiled signature across all layouts)."""
    t = _ROT_SCALES_DUMMY.get(device)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "W4A16 rotation placeholder is not initialized for CUDA graph "
                "capture; prewarm the planned launch before capturing"
            )
        t = torch.zeros(1, dtype=torch.float16, device=device)
        _ROT_SCALES_DUMMY[device] = t
    return t


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
    prefill_fused_sum_fp32: bool,
    collect_activation_amax: bool,
    stream_int: int,
    expert_map: torch.Tensor | None = None,
    rot_scales: torch.Tensor | None = None,
    intermediate_rotation: bool = False,
    a_input_up: torch.Tensor | None = None,
    trellis_bits: int = 3,
    trellis_codebook: str = LUT_E4M3,
    fc1_trellis_pair_kind: str | None = None,
    fc2_trellis_pair_kind: str | None = None,
    full_rotation: bool = False,
    intermediate_hadamard: bool = False,
    rotation_input: torch.Tensor | None = None,
    suh_gate_table: torch.Tensor | None = None,
    suh_up_table: torch.Tensor | None = None,
    launcher: W4A16FusedMoeCompileResult | None = None,
) -> None:
    swiglu_limit = float(swiglu_limit_value) if has_swiglu_limit else None
    collect_activation_amax = bool(collect_activation_amax)
    intermediate_rotation = bool(intermediate_rotation)
    full_rotation = bool(full_rotation)
    intermediate_hadamard = bool(intermediate_hadamard)
    use_expert_map = expert_map is not None
    if use_expert_map:
        if not direct_topk_routes:
            raise ValueError("expert_map launch requires direct_topk_routes")
        assert expert_map is not None
        if (
            expert_map.dtype != torch.int32
            or expert_map.device != a_input.device
            or expert_map.ndim != 1
            or not expert_map.is_contiguous()
        ):
            raise ValueError("expert_map must be contiguous int32 on the input device")
    if collect_activation_amax and activation_amax is None:
        raise ValueError("activation_amax is required for calibrated W4A16 launch")
    activation_amax_arg = (
        activation_amax if activation_amax is not None else w13_global_scale
    ).view(-1)
    if intermediate_rotation:
        if rot_scales is None:
            raise ValueError("intermediate_rotation launch requires rot_scales")
        rot_scales_arg = rot_scales.view(-1)
    else:
        rot_scales_arg = _rot_scales_dummy(w13_global_scale.device)
    if full_rotation and a_input_up is None:
        raise ValueError("full_rotation launch requires a distinct up A scratch")
    if a_input_up is None:
        a_input_up = a_input
    if (
        full_rotation
        and not intermediate_hadamard
        and a_input_up.data_ptr() == a_input.data_ptr()
    ):
        raise ValueError("full_rotation gate/up A scratches must not alias")
    if full_rotation and rotation_input is None:
        raise ValueError("full_rotation launch requires the raw rotation input")
    if rotation_input is None:
        rotation_input = a_input
    broadcast_suh = False
    if full_rotation:
        if suh_gate_table is None or suh_up_table is None:
            raise ValueError(
                "full_rotation launch requires suh_gate_table and suh_up_table"
            )
        suh_gate_arg = suh_gate_table.reshape(-1)
        suh_up_arg = suh_up_table.reshape(-1)
        broadcast_suh = suh_gate_arg.numel() == hidden_size
        if broadcast_suh != (suh_up_arg.numel() == hidden_size):
            raise ValueError(
                "suh gate/up tables must both be per-expert or both broadcast"
            )
        rotation_input_dtype = _normalize_element_dtype(rotation_input.dtype)
    else:
        suh_gate_arg = _rot_scales_dummy(w13_global_scale.device)
        suh_up_arg = suh_gate_arg
        rotation_input_dtype = element_dtype
    if launcher is not None and launcher.broadcast_suh != broadcast_suh:
        raise ValueError("prepared W4A16 input rotation layout differs from the bound tables")
    fused = launcher or compile_w4a16_fused_moe(
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
        trellis_bits=trellis_bits,
        trellis_codebook=trellis_codebook,
        fc1_trellis_pair_kind=fc1_trellis_pair_kind,
        fc2_trellis_pair_kind=fc2_trellis_pair_kind,
        direct_topk_routes=bool(direct_topk_routes),
        use_expert_map=use_expert_map,
        tc_decode_fused_sum=bool(tc_decode_fused_sum),
        prefill_fused_sum_fp32=bool(prefill_fused_sum_fp32),
        collect_activation_amax=collect_activation_amax,
        # The custom-op boundary cannot carry the compiled launch object. Re-pin
        # its selected geometry so tile-specific packs resolve the
        # identical cache entry instead of silently recompiling with auto tiles.
        force_tile_config=(fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n),
        intermediate_rotation=intermediate_rotation,
        full_rotation=full_rotation,
        intermediate_hadamard=intermediate_hadamard,
        rotation_input_dtype=rotation_input_dtype,
        broadcast_suh=broadcast_suh,
    )
    weight_cutlass_dtype = (
        cutlass.Uint8 if weight_layout == "modelopt" else cutlass.Int32
    )
    expert_map_addr = (
        packed_route_indices.data_ptr() if expert_map is None else expert_map.data_ptr()
    )
    route_num_experts = 0 if expert_map is None else int(expert_map.numel())
    if weight_layout in IQ2_CODECS:
        iq2_lut = iq2_xs_execution_lut(a_input.device, selectors=True, codec=weight_layout)
        fc1_trellis_lut_addr = iq2_lut.data_ptr()
        fc2_trellis_lut_addr = iq2_lut.data_ptr()
    elif weight_layout == "trellis_t256" and trellis_codebook != "mcg":
        trellis_rank_lut = _trellis256_execution_lut(
            a_input.device, trellis_codebook, direct_lut=fused.trellis_direct_lut
        )
        fc1_trellis_lut_addr = trellis_rank_lut.data_ptr()
        fc2_trellis_lut_addr = trellis_rank_lut.data_ptr()
    else:
        # Non-trellis and MCG kernels never dereference this ABI slot.
        fc1_trellis_lut_addr = w13_scale_i32.data_ptr()
        fc2_trellis_lut_addr = w13_scale_i32.data_ptr()
    fused.compiled(
        make_ptr(
            _cutlass_element_dtype(element_dtype),
            a_input.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            _cutlass_element_dtype(element_dtype),
            a_input_up.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            _cutlass_element_dtype(rotation_input_dtype),
            rotation_input.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            weight_cutlass_dtype,
            w13_arg.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            weight_cutlass_dtype,
            w2_arg.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        Int64(w13_arg.numel()),
        Int64(w2_arg.numel()),
        fc1_out,
        activated,
        fc2_out,
        make_ptr(
            cutlass.Int64
            if fc1_trellis_pair_kind == "P33_P43"
            else cutlass.Int32,
            w13_scale_i32.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Int64
            if fc2_trellis_pair_kind == "P33_P43"
            else cutlass.Int32,
            w2_scale_i32.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            w13_global_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            w2_global_scale.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
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
        make_ptr(
            cutlass.Float16,
            rot_scales_arg.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float16,
            suh_gate_arg.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float16,
            suh_up_arg.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Int32,
            expert_map_addr,
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            cutlass.Uint8,
            fc1_trellis_lut_addr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Uint8,
            fc2_trellis_lut_addr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        Int32(num_experts),
        Int32(route_num_experts),
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
    if (
        fused.weight_layout in BLOCK_CODECS and fused.moe_block_size == 8
    ) or _w4a16_small_m_splitk_enabled():
        # Stripe split-K wants the full persistent grid: each small-M FC1
        # column's K range is fanned across many CTAs, so right-sizing the
        # grid to the mn-tile count would recreate the serialization the
        # flag exists to remove.
        return max(cap, 1)
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
    "b12x::w4a16_fused_moe_launch",
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
        prefill_fused_sum_fp32=False,
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
    "b12x::w4a16_fused_moe_calibrated_launch",
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
        prefill_fused_sum_fp32=False,
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
    *,
    full_rotation: bool = False,
    intermediate_hadamard: bool = False,
    num_experts: int = 0,
    topk_weights: torch.Tensor | None = None,
    route_expert_ids: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
    svh_table: torch.Tensor | None = None,
    launcher: W4A16TopKSumCompileResult | None = None,
) -> None:
    full_rotation = bool(full_rotation)
    intermediate_hadamard = bool(intermediate_hadamard)
    route_ids_dtype = (
        torch.int32 if route_expert_ids is None else route_expert_ids.dtype
    )
    route_num_experts = 0 if expert_map is None else int(expert_map.numel())
    broadcast_svh = (
        full_rotation
        and svh_table is not None
        and svh_table.numel() == hidden_size
    )
    if launcher is not None and launcher.broadcast_svh != broadcast_svh:
        raise ValueError("prepared W4A16 output rotation layout differs from the bound table")
    sum_kernel = launcher or compile_w4a16_topk_sum(
        m=m,
        topk=topk,
        hidden_size=hidden_size,
        element_dtype=element_dtype,
        full_rotation=full_rotation,
        intermediate_hadamard=intermediate_hadamard,
        num_experts=num_experts,
        route_num_experts=route_num_experts,
        route_ids_dtype=route_ids_dtype,
        use_expert_map=expert_map is not None,
        broadcast_svh=broadcast_svh,
        float32_output=output.dtype == torch.float32,
    )
    dummy_addr = output.data_ptr()
    weights_addr = dummy_addr if topk_weights is None else topk_weights.data_ptr()
    route_ids_addr = (
        dummy_addr if route_expert_ids is None else route_expert_ids.data_ptr()
    )
    expert_map_addr = dummy_addr if expert_map is None else expert_map.data_ptr()
    svh_addr = dummy_addr if svh_table is None else svh_table.data_ptr()
    route_cutlass_dtype = (
        cutlass.Int32 if route_ids_dtype == torch.int32 else cutlass.Int64
    )
    route_align = 4 if route_ids_dtype == torch.int32 else 8
    sum_kernel.compiled(
        make_ptr(
            _cutlass_element_dtype(element_dtype),
            fc2_out.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32 if full_rotation or output.dtype == torch.float32 else _cutlass_element_dtype(element_dtype),
            output.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass.Float32,
            weights_addr,
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            route_cutlass_dtype,
            route_ids_addr,
            cute.AddressSpace.gmem,
            assumed_align=route_align,
        ),
        make_ptr(
            cutlass.Int32,
            expert_map_addr,
            cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        make_ptr(
            cutlass.Float16,
            svh_addr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        Int32(num_experts),
        Int32(route_num_experts),
        m,
        cuda.CUstream(stream_int),
    )


@torch.library.custom_op(
    "b12x::w4a16_topk_sum_launch",
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
    packed_route_indices: torch.Tensor | None,
    sms: int,
    max_shared_mem: int,
    device: torch.device,
    c_tmp: torch.Tensor | None = None,
    weight_layout: str = "packed",
    scale_format: str = "e4m3_k16",
    w13_layout: str = "packed",
    trellis_bits: int = 3,
    trellis_codebook: str = LUT_E4M3,
    trellis_pair_kind: str | None = None,
    trellis_rate_axis: str | None = None,
    dense_route_fast_path: bool = False,
    route_slots: int | None = None,
    force_tile_config: tuple[int, int] | None = None,
) -> _W4A16GemmLaunch:
    planner_weight_bits = (
        max(4, int(trellis_bits)) if weight_layout == "trellis_t256" else 4
    )
    if force_tile_config is None:
        tile_k, tile_n, _, _ = _select_tile_config(
            problem_m=size_m,
            problem_n=size_n,
            problem_k=size_k,
            top_k=top_k,
            moe_block_size=moe_block_size,
            sms=sms,
            max_shared_mem=max_shared_mem,
            scale_format=scale_format,
            weight_layout=weight_layout,
            weight_bits=planner_weight_bits,
        )
    else:
        tile_k, tile_n = (int(v) for v in force_tile_config)
        cta_threads = tile_n * tile_k // 64
        if not _candidate_tile_fits(
            problem_n=size_n,
            problem_k=size_k,
            cta_m_blocks=_covering_count(moe_block_size, 16),
            tile_n=tile_n,
            tile_k=tile_k,
            cta_threads=cta_threads,
            max_shared_mem=max_shared_mem - 512,
            scale_format=scale_format,
            weight_layout=weight_layout,
            weight_bits=planner_weight_bits,
        ):
            raise ValueError(
                "forced standalone W4A16 tile does not fit: "
                f"tile_k/tile_n={tile_k}/{tile_n}, N/K={size_n}/{size_k}"
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
        weight_layout=weight_layout,
        scale_format=scale_format,
        w13_layout=w13_layout,
        trellis_bits=trellis_bits,
        trellis_codebook=trellis_codebook,
        trellis_pair_kind=trellis_pair_kind,
        trellis_rate_axis=trellis_rate_axis,
        dense_route_fast_path=bool(dense_route_fast_path),
    )
    if route_slots is None:
        if packed_route_indices is None:
            raise ValueError(
                "standalone W4A16 launch requires route_slots or packed_route_indices"
            )
        route_slots = int(packed_route_indices.numel())
    c_tmp = _get_c_tmp(
        packed_gemm_scratch_elements(
            size_n=size_n,
            route_slots=int(route_slots),
            moe_block_size=moe_block_size,
            sms=sms,
            weight_layout=weight_layout,
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
    expert_counts: torch.Tensor | None = None,
    launches=None,
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
        expert_counts=expert_counts,
        launches=launches,
    )


def _trellis256_dense_tile_config(size_k: int, size_n: int) -> tuple[int, int]:
    """Return an m-invariant t256 tile so dense row bits cannot drift with m."""
    if int(size_k) % 64 != 0:
        raise ValueError(f"trellis_t256 dense K must be divisible by 64, got {size_k}")
    if int(size_n) % 256 == 0:
        return (64, 256)
    if int(size_n) % 128 == 0:
        # With dense M=64, (tile_k=64,tile_n=128) maps to the measured
        # (cta_threads=128, cta_m/n/k_blocks=4/8/4) register specialization.
        # The superficially symmetric 128x128 tile would need a missing
        # 256-thread 4/8/8 register entry and fail before compilation.
        return (64, 128)
    raise ValueError(
        "trellis_t256 dense N must be divisible by 128 (or 256 for the wide tile), "
        f"got N={size_n}"
    )


def _trellis256_dense_launch_geometry(
    *,
    size_m: int,
    size_k: int,
    size_n: int,
    sms: int,
) -> tuple[int, tuple[int, int]]:
    """Avoid short spill waves in the persistent dense-Trellis schedule.

    A default M64/N256 launch can leave only a handful of CTAs in a second or
    third SM wave. Narrow projections benefit from a smaller M tile, while wide
    projections already expose enough N parallelism and only need N128 for a
    two-wave spill. Past three waves the smaller tile's scheduler overhead costs
    more than the remaining imbalance.
    """
    default = (64, (64, 256 if size_n % 256 == 0 else 128))
    if size_n % 256 != 0 or sms <= 0:
        return default
    tasks = _covering_count(int(size_m), 64) * (int(size_n) // 256)
    waves = _covering_count(tasks, int(sms))
    if waves not in (2, 3):
        return default
    spill = tasks - (waves - 1) * int(sms)
    if spill > max(16, int(sms) // 10):
        return default
    n_tiles_256 = int(size_n) // 256
    if n_tiles_256 <= 32:
        return (48, (64, 128 if waves == 2 else 256))
    if waves == 2:
        return (64, (64, 128))
    if int(size_k) <= 4096:
        return (48, (64, 256))
    return default



def _resolve_exl3_hadamard_128(hadamard_128):
    if hadamard_128 is None:
        try:
            import exllamav3_ext
        except ImportError as exc:
            raise RuntimeError(
                "run_trellis256_dense requires exllamav3_ext.had_r_128 for the "
                "EXL3 input/output rotations"
            ) from exc
        hadamard_128 = exllamav3_ext.had_r_128
    elif not callable(hadamard_128):
        hadamard_128 = getattr(hadamard_128, "had_r_128", None)
    if not callable(hadamard_128):
        raise TypeError("hadamard_128 must be callable or expose had_r_128")
    return hadamard_128


def _trellis_dense_buffer(
    name: str,
    buffer: torch.Tensor | None,
    *,
    shape: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Validate caller-owned dense scratch or allocate it before capture."""
    if buffer is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"Trellis dense {name} is not initialized for CUDA graph capture; "
                "provide caller-owned storage"
            )
        return torch.empty(shape, dtype=dtype, device=device)
    if (
        tuple(buffer.shape) != shape
        or buffer.dtype != dtype
        or buffer.device != device
        or not buffer.is_contiguous()
        or int(buffer.data_ptr()) % 16 != 0
    ):
        raise ValueError(
            f"{name} must be contiguous, 16-byte-aligned {dtype} with shape "
            f"{shape} on {device}"
        )
    return buffer


def _run_trellis256_dense_current_device(
    x: torch.Tensor,
    prepared_dense,
    *,
    launch: _W4A16GemmLaunch,
    execution_lut: torch.Tensor | None,
    grid_cap: int,
    output: torch.Tensor | None = None,
    gemm_output: torch.Tensor | None = None,
    input_f16: torch.Tensor | None = None,
    rotated_f16: torch.Tensor | None = None,
    rotated_compute: torch.Tensor | None = None,
    gemm_output_f16: torch.Tensor | None = None,
    output_f16: torch.Tensor | None = None,
    hadamard_128,
    stream: cuda.CUstream | None = None,
) -> torch.Tensor:
    """Run one native EXL3 linear on the already-selected CUDA device.

    This is a true dense entry: E=1 and contiguous row identities are synthesized
    inside the kernel, so no top-k tensors or route-packing kernels are created.
    Outer rotations follow EXL3 order exactly: fp16 ``suh`` multiply before the
    input H128, and fp16 ``svh`` multiply after the output H128.
    """
    if getattr(prepared_dense, "weight_layout", None) != "trellis_t256":
        raise ValueError("run_trellis256_dense requires prepared trellis_t256 weights")
    if int(getattr(prepared_dense, "num_experts", 0)) != 1:
        raise ValueError("run_trellis256_dense requires an honest E=1 prepared weight")
    trellis_codebook = str(
        getattr(prepared_dense, "trellis_codebook", "")
    ).lower()
    if trellis_codebook not in _TRELLIS256_CODEBOOKS:
        raise NotImplementedError(
            "run_trellis256_dense has no decoder for codebook "
            f"{trellis_codebook!r}"
        )
    trellis_bits = int(getattr(prepared_dense, "trellis_bits", 0))
    if trellis_bits not in _TRELLIS256_BITS:
        raise ValueError(
            f"prepared dense weight has invalid trellis_bits={trellis_bits}"
        )
    trellis_pair_kind = getattr(prepared_dense, "trellis_pair_kind", None)
    trellis_rate_axis = getattr(prepared_dense, "trellis_rate_axis", None)
    if (trellis_pair_kind is None) != (trellis_rate_axis is None):
        raise ValueError("prepared pair weight has incomplete pair metadata")
    compute_dtype = getattr(prepared_dense, "params_dtype", None)
    if compute_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            "prepared dense weight must select fp16 or bf16 compute, got "
            f"{compute_dtype}"
        )
    cutlass_dtype = (
        cutlass.Float16 if compute_dtype == torch.float16 else cutlass.BFloat16
    )
    if x.ndim != 2 or int(x.shape[0]) <= 0:
        raise ValueError(f"x must be a non-empty rank-2 tensor, got {tuple(x.shape)}")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"x must be fp16 or bf16, got {x.dtype}")
    if not x.is_cuda or not x.is_contiguous():
        raise ValueError("x must be a contiguous CUDA tensor")
    m, size_k = (int(v) for v in x.shape)
    size_n = int(prepared_dense.out_features)
    if size_k != int(prepared_dense.in_features):
        raise ValueError(
            f"x has K={size_k}, prepared dense weight expects {prepared_dense.in_features}"
        )
    if prepared_dense.trellis.device != x.device:
        raise ValueError("x and prepared dense weight must be on the same CUDA device")
    output = _trellis_dense_buffer(
        "output",
        output,
        shape=(m, size_n),
        dtype=x.dtype,
        device=x.device,
    )
    if not callable(hadamard_128):
        raise TypeError("prepared Trellis transform must be callable")

    gemm_output = _trellis_dense_buffer(
        "gemm_output",
        gemm_output,
        shape=(m, size_n),
        dtype=compute_dtype,
        device=x.device,
    )
    if x.dtype == torch.float16:
        x_f16 = x
    else:
        input_f16 = _trellis_dense_buffer(
            "input_f16",
            input_f16,
            shape=(m, size_k),
            dtype=torch.float16,
            device=x.device,
        )
        input_f16.copy_(x)
        x_f16 = input_f16
    rotated_f16 = _trellis_dense_buffer(
        "rotated_f16",
        rotated_f16,
        shape=(m, size_k),
        dtype=torch.float16,
        device=x.device,
    )
    hadamard_128(x_f16, rotated_f16, prepared_dense.suh, None, 1.0)
    if compute_dtype == torch.float16:
        rotated_compute = rotated_f16
    else:
        rotated_compute = _trellis_dense_buffer(
            "rotated_compute",
            rotated_compute,
            shape=(m, size_k),
            dtype=torch.bfloat16,
            device=x.device,
        )
        rotated_compute.copy_(rotated_f16)

    if int(launch.kernel.moe_block_size) <= 0:
        raise ValueError("prepared Trellis launch has invalid block-row geometry")
    moe_block_size = int(launch.kernel.moe_block_size)
    route_blocks = (m + moe_block_size - 1) // moe_block_size
    if int(launch.kernel.max_m_blocks) < route_blocks:
        raise ValueError("prepared Trellis launch cannot cover the requested rows")
    if (
        launch.kernel.trellis_bits,
        launch.kernel.trellis_codebook,
        launch.kernel.trellis_pair_kind,
        launch.kernel.trellis_rate_axis,
    ) != (
        trellis_bits,
        trellis_codebook,
        trellis_pair_kind,
        trellis_rate_axis,
    ):
        raise ValueError("prepared Trellis launcher metadata differs from the weight")
    tile_n = int(launch.kernel.tile_n)
    n_tiles = size_n // tile_n
    grid_x = min(int(grid_cap), max(route_blocks * n_tiles, 1))
    if trellis_codebook != "mcg" and execution_lut is None:
        raise ValueError("prepared Trellis decoder is missing its execution LUT")
    stream = current_cuda_stream() if stream is None else stream
    dummy_i32 = prepared_dense.workspace[:1]
    launch.kernel.compiled(
        make_ptr(
            cutlass_dtype,
            rotated_compute.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        make_ptr(
            cutlass_dtype,
            rotated_compute.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        prepared_dense.trellis,
        make_ptr(
            cutlass_dtype,
            gemm_output.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        prepared_dense.scale.view(torch.uint8).view(torch.int32).view(-1),
        prepared_dense.global_scale,
        dummy_i32,
        dummy_i32,
        dummy_i32,
        prepared_dense.global_scale,
        launch.c_tmp,
        prepared_dense.workspace,
        # MCG kernels never dereference the LUT ABI slot.
        dummy_i32 if trellis_codebook == "mcg" else execution_lut,
        m,
        grid_x,
        stream,
    )

    if compute_dtype == torch.float16:
        gemm_f16 = gemm_output
    else:
        gemm_output_f16 = _trellis_dense_buffer(
            "gemm_output_f16",
            gemm_output_f16,
            shape=(m, size_n),
            dtype=torch.float16,
            device=x.device,
        )
        gemm_output_f16.copy_(gemm_output)
        gemm_f16 = gemm_output_f16
    if output.dtype == torch.float16:
        hadamard_128(gemm_f16, output, None, prepared_dense.svh, 1.0)
    else:
        output_f16 = _trellis_dense_buffer(
            "output_f16",
            output_f16,
            shape=(m, size_n),
            dtype=torch.float16,
            device=x.device,
        )
        hadamard_128(gemm_f16, output_f16, None, prepared_dense.svh, 1.0)
        output.copy_(output_f16)
    return output


def run_trellis256_dense(
    x: torch.Tensor,
    prepared_dense,
    *,
    launch: _W4A16GemmLaunch,
    execution_lut: torch.Tensor | None,
    grid_cap: int,
    output: torch.Tensor | None = None,
    gemm_output: torch.Tensor | None = None,
    input_f16: torch.Tensor | None = None,
    rotated_f16: torch.Tensor | None = None,
    rotated_compute: torch.Tensor | None = None,
    gemm_output_f16: torch.Tensor | None = None,
    output_f16: torch.Tensor | None = None,
    hadamard_128,
) -> torch.Tensor:
    """Run a session-materialized native or compact P24/P33 EXL3 linear."""
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    with torch.cuda.device(x.device):
        return _run_trellis256_dense_current_device(
            x,
            prepared_dense,
            launch=launch,
            execution_lut=execution_lut,
            grid_cap=grid_cap,
            output=output,
            gemm_output=gemm_output,
            input_f16=input_f16,
            rotated_f16=rotated_f16,
            rotated_compute=rotated_compute,
            gemm_output_f16=gemm_output_f16,
            output_f16=output_f16,
            hadamard_128=hadamard_128,
            stream=None,
        )


def _resolve_route_block_size_m(
    *,
    m: int,
    topk: int,
    route_num_experts: int,
    planned_block_size_m: int | None,
    fused_launch: W4A16FusedMoeCompileResult | None,
) -> int:
    """Resolve one route geometry shared by scratch, packing, and GEMM."""
    planned = None if planned_block_size_m is None else int(planned_block_size_m)
    if planned is not None and planned not in _ALLOWED_ROUTED_SIZES:
        raise ValueError(f"unsupported planned W4A16 moe_block_size={planned}")
    if fused_launch is None:
        return planned or select_route_block_size_m(m, topk, route_num_experts)

    compiled = int(fused_launch.moe_block_size)
    if planned is not None and compiled != planned:
        raise RuntimeError(
            "preplanned W4A16 route geometry does not match the fused launch: "
            f"planned_block_size_m={planned}, launch_block_size_m={compiled}"
        )
    return compiled


def _w4a16_stream_is_capturing(
    stream: cuda.CUstream,
    *,
    current_stream: cuda.CUstream,
) -> bool:
    """Observe capture on either Torch's current stream or an explicit stream."""
    current_capturing = torch.cuda.is_current_stream_capturing()
    if current_capturing or int(stream) == int(current_stream):
        return current_capturing
    result, status = cuda.cuStreamIsCapturing(stream)
    if result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            f"cuStreamIsCapturing failed for the selected W4A16 stream: {result}"
        )
    return int(status) != 0


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
    prefill_sum_accum: torch.Tensor | None = None,
    fc1_c_tmp: torch.Tensor | None = None,
    fc2_c_tmp: torch.Tensor | None = None,
    packed_route_indices: torch.Tensor | None = None,
    block_expert_ids: torch.Tensor | None = None,
    packed_route_count: torch.Tensor | None = None,
    expert_offsets: torch.Tensor | None = None,
    expert_counts: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
    output_expert_map: torch.Tensor | None = None,
    activation_amax: torch.Tensor | None = None,
    layer_idx: int | None = None,
    apply_router_weight_on_input: bool = False,
    fast_math: bool = True,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    fused_launch: W4A16FusedMoeCompileResult | None = None,
    topk_sum_launch: W4A16TopKSumCompileResult | None = None,
    route_block_size_m: int | None = None,
    intermediate_rotation_scales: torch.Tensor | None = None,
    a_input_up: torch.Tensor | None = None,
    full_rotation: bool = False,
    suh_gate_table: torch.Tensor | None = None,
    suh_up_table: torch.Tensor | None = None,
    svh_table: torch.Tensor | None = None,
    rotation_a_gate: torch.Tensor | None = None,
    rotation_a_up: torch.Tensor | None = None,
    route_mode: str = "auto",
    route_pack_launches=None,
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
    full_rotation = bool(full_rotation)
    route_mode = str(route_mode).lower()
    if route_mode not in {"auto", "direct", "packed"}:
        raise ValueError("W4A16 route_mode must be 'auto', 'direct', or 'packed'")
    rotation_input_dtype = _normalize_element_dtype(a_input.dtype)
    prepared_dtype = getattr(prepared, "params_dtype", a_input.dtype)
    if full_rotation:
        element_dtype = _normalize_element_dtype(prepared_dtype)
        if element_dtype != "fp16":
            raise TypeError("full_rotation requires fp16 prepared weights/scratch")
        if output.dtype != torch.float32:
            raise TypeError(
                f"full_rotation output must be torch.float32, got {output.dtype}"
            )
    else:
        element_dtype = rotation_input_dtype
        if output.dtype != a_input.dtype:
            raise TypeError(
                f"output must have dtype {a_input.dtype}, got {output.dtype}"
            )
        if prepared_dtype != a_input.dtype:
            raise TypeError(
                f"prepared weights were built for {prepared_dtype}, but a_input has dtype {a_input.dtype}"
            )
    weight_layout = getattr(prepared, "weight_layout", "packed")
    if weight_layout not in _WEIGHT_LAYOUTS:
        raise ValueError(f"unsupported W4A16 weight_layout {weight_layout!r}")
    trellis_bits = int(getattr(prepared, "trellis_bits", 3))
    intermediate_hadamard = bool(getattr(prepared, "intermediate_hadamard", False))
    trellis_codebook = str(
        getattr(prepared, "trellis_codebook", None) or LUT_E4M3
    ).lower()
    fc1_trellis_pair_kind = getattr(prepared, "fc1_trellis_pair_kind", None)
    fc2_trellis_pair_kind = getattr(prepared, "fc2_trellis_pair_kind", None)
    fc1_trellis_pair_modes = getattr(prepared, "fc1_trellis_pair_modes", None)
    fc2_trellis_pair_modes = getattr(prepared, "fc2_trellis_pair_modes", None)
    prepared_tile_config = getattr(prepared, "tile_config", None)
    if (fc1_trellis_pair_kind is None) != (fc2_trellis_pair_kind is None):
        raise ValueError("prepared trellis pair weights have incomplete pair metadata")
    if weight_layout == "trellis_t256":
        if trellis_bits not in _TRELLIS256_BITS:
            raise ValueError(
                f"prepared trellis_t256 bitrate must be in {_TRELLIS256_BITS}, "
                f"got {trellis_bits}"
            )
        if trellis_codebook not in _TRELLIS256_CODEBOOKS:
            raise NotImplementedError(
                "trellis_t256 execution has no decoder for codebook "
                f"{trellis_codebook!r}"
            )
        if fc1_trellis_pair_kind is not None:
            fc1_trellis_pair_kind = str(fc1_trellis_pair_kind).upper()
            fc2_trellis_pair_kind = str(fc2_trellis_pair_kind).upper()
            dynamic_kinds = {"PDYNAMIC", "P33_P43"}
            static_kinds = {"P24", "P33", "P43", "P44"}
            dynamic_pairs = (
                fc1_trellis_pair_kind in dynamic_kinds
                or fc2_trellis_pair_kind in dynamic_kinds
            )
            if dynamic_pairs:
                if fc1_trellis_pair_kind != fc2_trellis_pair_kind:
                    raise ValueError(
                        "prepared dynamic fused trellis pair kinds must match"
                    )
            elif (
                fc1_trellis_pair_kind not in static_kinds
                or fc2_trellis_pair_kind not in static_kinds
            ):
                raise ValueError("unsupported prepared static trellis pair kind")
            pair_metadata_dtype = (
                torch.int64
                if fc1_trellis_pair_kind == "P33_P43"
                else torch.int32
            )
            if trellis_bits != 3:
                raise ValueError(
                    "prepared trellis pairs require the trellis_bits=3 base "
                    "specialization"
                )
            if dynamic_pairs:
                for name, modes in (
                    ("fc1", fc1_trellis_pair_modes),
                    ("fc2", fc2_trellis_pair_modes),
                ):
                    if (
                        modes is None
                        or modes.dtype != pair_metadata_dtype
                        or modes.device != a_input.device
                        or tuple(modes.shape) != (int(prepared.num_experts),)
                        or not modes.is_contiguous()
                    ):
                        raise ValueError(
                            f"prepared dynamic {name} trellis pair metadata must be "
                            f"contiguous {pair_metadata_dtype}[num_experts] on the "
                            "input device"
                        )
        if activation_amax is not None:
            raise NotImplementedError(
                "trellis_t256 activation-amax collection is not exposed through "
                "the registered launch ABI"
            )
    if intermediate_hadamard and not full_rotation:
        raise ValueError(
            "prepared intermediate-Hadamard metadata requires full rotation; "
            f"got intermediate_hadamard={intermediate_hadamard}, full_rotation={full_rotation}"
        )
    scale_format = _normalize_scale_format(
        getattr(prepared, "scale_format", None)
        or (
            "e8m0_k32"
            if getattr(prepared, "source_format", "") == "fp4_e8m0_k32"
            else "e4m3_k16"
        )
    )
    x4t_w13_scale = getattr(prepared, "x4t_w13_scale", None)
    x4t_w2_scale = getattr(prepared, "x4t_w2_scale", None)
    if (x4t_w13_scale is None) != (x4t_w2_scale is None):
        raise ValueError("prepared X4T weights have incomplete scale metadata")
    use_x4t_scale_predecode = x4t_w13_scale is not None
    if use_x4t_scale_predecode and (
        weight_layout not in ("packed", "modelopt") or scale_format != "e8m0_k32"
    ):
        raise ValueError(
            "X4T scale predecode requires native or packed FP4 with E8M0 K/32 scales"
        )
    w13_layout = getattr(
        prepared,
        "w13_layout",
        "w13" if weight_layout == "modelopt" else "packed",
    )
    if weight_layout == "modelopt":
        if w13_layout not in _MODEL_OPT_W13_LAYOUTS:
            raise ValueError(f"unsupported W4A16 w13_layout {w13_layout!r}")
    elif weight_layout == "trellis_t256":
        if w13_layout not in _TRELLIS256_W13_LAYOUTS:
            raise ValueError(f"unsupported trellis_t256 w13_layout {w13_layout!r}")
    else:
        w13_layout = "packed"
    dual_a_required = bool(
        intermediate_rotation_scales is not None
        and weight_layout == "trellis_t256"
        and w13_layout == "trellis_t256_proj"
    )
    if full_rotation and not dual_a_required:
        raise ValueError(
            "full_rotation requires projection-major trellis intermediate rotation"
        )
    if dual_a_required and a_input_up is None and not full_rotation:
        raise ValueError(
            "exact projection-major trellis_t256 rotation requires a_input_up"
        )
    if a_input_up is not None and (not dual_a_required or full_rotation):
        raise ValueError(
            "a_input_up is only valid for exact projection-major trellis_t256 rotation"
        )
    if a_input_up is not None:
        if (
            a_input_up.shape != a_input.shape
            or a_input_up.dtype != a_input.dtype
            or a_input_up.device != a_input.device
        ):
            raise ValueError(
                "a_input_up must match a_input shape, dtype, and device; got "
                f"{tuple(a_input_up.shape)}/{a_input_up.dtype}/{a_input_up.device} "
                f"vs {tuple(a_input.shape)}/{a_input.dtype}/{a_input.device}"
            )
        if not a_input_up.is_contiguous():
            raise ValueError("a_input_up must be contiguous")
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
    _validate_expert_map(output_expert_map, device=a_input.device)
    if getattr(prepared, "x4t_packed_pair_programs", None) is not None and expert_map is not None:
        raise NotImplementedError("DS4.1 packed X4T supports local TP expert IDs without expert mapping")
    if output_expert_map is not None and not full_rotation:
        raise ValueError("output_expert_map is only valid with full_rotation")

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
    if output_expert_map is not None and int(output_expert_map.numel()) < int(
        prepared.num_experts
    ):
        raise ValueError(
            "output_expert_map cannot be shorter than the local expert count"
        )
    if full_rotation:
        if apply_router_weight_on_input:
            raise ValueError(
                "full_rotation applies router weights only in the fp32 top-k sum"
            )
        num_local_experts = int(prepared.num_experts)
        intermediate_size_full = int(prepared.intermediate_size)
        # H-side tables may hold a single broadcast row; the kernels index them
        # with a zero expert stride.
        rotation_width = (6 if intermediate_hadamard else 3) * intermediate_size_full
        for name, table, shapes in (
            (
                "suh_gate_table",
                suh_gate_table,
                ((num_local_experts, hidden_size), (1, hidden_size)),
            ),
            (
                "suh_up_table",
                suh_up_table,
                ((num_local_experts, hidden_size), (1, hidden_size)),
            ),
            (
                "svh_table",
                svh_table,
                ((num_local_experts, hidden_size), (1, hidden_size)),
            ),
            (
                "intermediate_rotation_scales",
                intermediate_rotation_scales,
                ((num_local_experts, rotation_width),),
            ),
        ):
            if table is None:
                raise ValueError(f"full_rotation requires {name}")
            if (
                table.dtype != torch.float16
                or table.device != a_input.device
                or tuple(table.shape) not in shapes
                or not table.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous fp16 {shapes} on {a_input.device}; "
                    f"got {tuple(table.shape)}/{table.dtype}/{table.device}/"
                    f"contiguous={table.is_contiguous()}"
                )
        required_a = m * topk * hidden_size
        for name, scratch in (
            ("rotation_a_gate", rotation_a_gate),
            ("rotation_a_up", rotation_a_up),
        ):
            if scratch is None:
                raise ValueError(f"full_rotation requires preallocated {name}")
            if (
                scratch.dtype != torch.float16
                or scratch.device != a_input.device
                or not scratch.is_contiguous()
                or int(scratch.numel()) < required_a
            ):
                raise ValueError(
                    f"{name} must be contiguous fp16 on {a_input.device} with at least "
                    f"{required_a} elements"
                )
        assert rotation_a_gate is not None and rotation_a_up is not None
        if (
            not intermediate_hadamard
            and rotation_a_gate.data_ptr() == rotation_a_up.data_ptr()
        ):
            raise ValueError("full_rotation gate/up A scratches must not alias")
    elif any(
        value is not None
        for value in (
            suh_gate_table,
            suh_up_table,
            svh_table,
            rotation_a_gate,
            rotation_a_up,
        )
    ):
        raise ValueError("full-rotation tables/scratch require full_rotation=True")
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
    block_size_m = _resolve_route_block_size_m(
        m=m,
        topk=topk,
        route_num_experts=route_num_experts,
        planned_block_size_m=route_block_size_m,
        fused_launch=fused_launch,
    )
    if block_size_m not in _ALLOWED_ROUTED_SIZES:
        raise ValueError(f"unsupported W4A16 moe_block_size={block_size_m}")

    props = torch.cuda.get_device_properties(a_input.device)
    sms = int(props.multi_processor_count)
    current_stream = current_cuda_stream()
    stream = current_stream if stream is None else stream
    if (
        route_mode != "packed"
        and (not collect_activation_amax)
        and _small_m_direct_supported(
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
        )
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
        if use_x4t_scale_predecode:
            # Native and packed GEMMs consume the same expanded scale grid.
            # The early-return micro path must refresh it before every launch.
            from b12x._lib.quant.x4t_packed_scales import _launch_x4t_packed_scale_pair

            programs = prepared.x4t_packed_pair_programs
            if programs is None or w13_layout != "w31":
                raise ValueError("Native X4T requires prepared gate/up scale programs")
            _launch_x4t_packed_scale_pair(
                x4t_w13_scale, x4t_w2_scale, topk_ids.view(-1),
                micro_w13_scale, micro_w2_scale,
                program=programs[2 if topk_ids.dtype == torch.int64 else 0],
                stream=stream,
            )
        barrier_count = prepared.workspace[-2:-1]
        barrier_epoch = prepared.workspace[-1:]
        if _small_m_direct_host_barrier_reset_enabled():
            prepared.workspace[-2:].zero_()
        direct_launch = None
        if fused_launch is not None:
            direct_launch = next((
                launch for launch in fused_launch.small_m_direct_launches
                if launch.topk_ids_dtype == topk_ids.dtype
            ), None)
            if direct_launch is None or direct_launch.m != m:
                raise RuntimeError("native W4A16 direct launch was not prepared for this token count and route dtype")
        launch_args = (
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
        if direct_launch is None:
            torch.ops.b12x.w4a16_small_m_direct_launch(*launch_args)
        else:
            _w4a16_small_m_direct_launch_flat(*launch_args, launcher=direct_launch)
        return output

    # TC decode keeps the top-k reduction inside the fused tensor-core kernel.
    # Direct-route emit hooks resolve global expert IDs against compact tiers
    # without materializing remapped IDs or masked router weights.
    # Prepared launches retain this contract in tc_decode_fused_sum.
    preplanned_tc_decode = bool(getattr(fused_launch, "tc_decode_fused_sum", False))
    preplanned_prefill_fused_sum = bool(
        getattr(fused_launch, "prefill_fused_sum_fp32", False)
    )
    prefer_tc_decode = route_mode == "direct" or (
        route_mode == "auto"
        and _w4a16_tc_decode_preferred(
            m=m,
            topk=topk,
            num_experts=int(prepared.num_experts),
            sms=sms,
        )
    )
    use_tc_decode = bool(
        (not collect_activation_amax)
        and route_mode != "packed"
        and (fused_launch is None or preplanned_tc_decode)
        and weight_layout in {"packed", "iq2_xs", "iq2_xxs", "q8_0"}
        and (is_gated or (weight_layout in BLOCK_CODECS and activation == "relu2"))
        and element_dtype == "bf16"
        and topk_ids.dtype in (torch.int32, torch.int64)
        and topk_ids.is_cuda
        and prefer_tc_decode
    )
    if use_tc_decode and topk_ids.dtype != torch.int32:
        # The inline direct-topk route path needs int32 route indices.
        topk_ids = topk_ids.to(torch.int32)

    mapped_direct = expert_map is not None
    direct_m_cap = (
        _W4A16_SMALL_M_DIRECT_MAX_M if mapped_direct else _MAX_DIRECT_TOPK_ROUTE_M
    )
    direct_layout_ok = weight_layout in {"packed", "iq2_xs", "iq2_xxs", "q8_0"} or (
        mapped_direct and full_rotation and weight_layout == "trellis_t256"
    )
    direct_topk_eligible = (
        (not collect_activation_amax)
        and route_mode != "packed"
        and (m <= direct_m_cap or use_tc_decode)
        and direct_layout_ok
    )
    use_direct_topk_routes = bool(
        direct_topk_eligible
        and topk_ids.dtype == torch.int32
        and topk_ids.is_cuda
        and (
            fused_launch is None
            or bool(getattr(fused_launch, "direct_topk_routes", False))
        )
        and (
            fused_launch is None
            or bool(getattr(fused_launch, "use_expert_map", False)) == mapped_direct
        )
    )
    if (
        bool(getattr(fused_launch, "direct_topk_routes", False))
        and not use_direct_topk_routes
    ):
        raise RuntimeError(
            "preplanned W4A16 direct top-k routing requires a matching small-M "
            "CUDA int32 route/map contract"
        )

    # TC-decode requires the inline direct-topk route path (no route-pack).
    use_tc_decode = bool(use_tc_decode and use_direct_topk_routes)

    if route_mode == "direct" and not use_direct_topk_routes:
        raise RuntimeError(
            "planned W4A16 direct routing is unsupported for this launch shape"
        )

    prefill_fused_sum_requested = (
        preplanned_prefill_fused_sum
        if fused_launch is not None
        else prefill_fused_sum_enabled()
    )
    use_prefill_fused_sum = prefill_fused_sum_eligible(
        dtype=element_dtype,
        m=int(fused_launch.size_m) if fused_launch is not None else m,
        full_rotation=full_rotation,
        weight_layout=weight_layout,
        collect_activation_amax=collect_activation_amax,
        enabled=prefill_fused_sum_requested,
    )
    use_fused_topk_sum = bool(use_tc_decode or use_prefill_fused_sum)

    if use_prefill_fused_sum:
        required_accum_elements = (
            int(fused_launch.size_m) if fused_launch is not None else int(m)
        ) * hidden_size
        if (
            prefill_sum_accum is None
            or prefill_sum_accum.dtype != torch.float32
            or prefill_sum_accum.device != a_input.device
            or not prefill_sum_accum.is_contiguous()
            or prefill_sum_accum.numel() < required_accum_elements
        ):
            raise ValueError(
                "W4A16 prefill fused sum requires a contiguous FP32 accumulator "
                f"with at least {required_accum_elements} elements on "
                f"{a_input.device}"
            )

    # Prepared TC decode owns the per-token reduction and omits the separate
    # top-k sum launch. A mismatched routing contract would corrupt output.
    if preplanned_tc_decode and not use_tc_decode:
        raise RuntimeError(
            "preplanned TC-decode W4A16 launch requires small-M packed bf16 "
            f"decode (m <= {_TC_DECODE_MAX_M}, cuda int32/int64 topk_ids)"
        )
    if preplanned_prefill_fused_sum and not use_prefill_fused_sum:
        raise RuntimeError(
            "preplanned W4A16 prefill fused-sum launch requires the enabled "
            "large-M packed or modelopt BF16 route-reduction contract"
        )

    route_slots_for_scratch = int(m) * int(topk) * int(block_size_m)
    required_m_blocks = int(m) * int(topk) if use_direct_topk_routes else 0
    if fused_launch is not None and not use_direct_topk_routes:
        route_slots_capacity = max_packed_route_slots(
            int(fused_launch.size_m) * int(topk),
            int(block_size_m),
            route_num_experts,
        )
        if route_pack_launches is not None:
            # A retained small-M GEMM can use a route packer prepared for a
            # larger capacity. Its kernels initialize every planned route slot;
            # do not shrink those views to the selected GEMM's live-row bucket.
            route_slots_capacity = max(
                route_slots_capacity, route_pack_launches.max_packed_routes
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
        if fused_launch is not None and not full_rotation and route_pack_launches is None:
            raise RuntimeError(
                "prepared W4A16 packed route execution is missing route-pack programs"
            )
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
                expert_counts=expert_counts,
                launches=route_pack_launches,
                stream=stream,
            )
        )
        route_slots_for_scratch = int(packed_route_indices.numel())
        required_m_blocks = int(block_expert_ids.numel())

    if use_x4t_scale_predecode:
        # X4T keeps the exact nibble weights resident and expands only the
        # scale planes of experts touched by this routed call. The scratch is
        # caller-owned and may be shared across all layers on this stream.
        from b12x._lib.quant.x4t_scales import (
            decode_x4t_tp12_w4a16_scales,
        )

        assert x4t_w13_scale is not None and x4t_w2_scale is not None
        x4t_expert_ids = (
            packed_route_indices if use_direct_topk_routes else block_expert_ids
        )
        assert x4t_expert_ids is not None
        programs = getattr(prepared, "x4t_packed_pair_programs", None)
        if programs is not None:
            from b12x._lib.quant.x4t_packed_scales import _launch_x4t_packed_scale_pair

            counts = not use_direct_topk_routes
            active = expert_counts if counts else x4t_expert_ids
            sorted_ids = False
            block_bound = min(block_expert_ids.numel(), topk_ids.numel())
            if counts and (expert_counts is None or block_bound < int(prepared.num_experts)):
                # A nonempty packed block contains at least one routed row.
                # Its sorted expert list therefore needs no more entries than
                # the routed-row count; the packer fills unused entries with -1.
                # Bounding the grid avoids scheduling all experts for decode.
                active = block_expert_ids[:block_bound]
                counts = False
                sorted_ids = True
            if active is None:
                raise ValueError("Packed X4T routing requires caller-owned expert counts")
            if sorted_ids:
                program_index = 3
            elif counts:
                program_index = 1
            else:
                program_index = 2 if active.dtype == torch.int64 else 0
            _launch_x4t_packed_scale_pair(
                x4t_w13_scale, x4t_w2_scale, active,
                prepared.w13_scale, prepared.w2_scale,
                program=programs[program_index], stream=stream,
            )
        else:
            decode_x4t_tp12_w4a16_scales(
                x4t_w13_scale,
                x4t_w2_scale,
                x4t_expert_ids,
                prepared.w13_scale,
                prepared.w2_scale,
                expert_map=expert_map if use_direct_topk_routes else None,
                w13_row_rotation=int(getattr(prepared, "x4t_w13_row_rotation", 0)),
                expert_ids_unique=bool(use_direct_topk_routes and m == 1),
                stream=stream,
            )

    max_shared_mem = int(
        getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
    )
    buffer_plan = plan_w4a16_buffers(
        prepared,
        m=m,
        topk=topk,
        route_num_experts=route_num_experts,
        sms=sms,
        dtype=(prepared_dtype if full_rotation else a_input.dtype),
        full_rotation=full_rotation,
        block_size_m=block_size_m,
        weight_layout=weight_layout,
        collect_activation_amax=collect_activation_amax,
        prefill_fused_sum=use_prefill_fused_sum,
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
    cache_dtype = prepared_dtype if full_rotation else a_input.dtype
    if (
        intermediate_cache13.dtype != cache_dtype
        or intermediate_cache2.dtype != cache_dtype
    ):
        raise TypeError(f"intermediate caches must be {cache_dtype}")
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
            zero_fc2_output=(
                expert_map is not None
                and not full_rotation
                and not use_direct_topk_routes
            ),
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
            trellis_bits=trellis_bits,
            trellis_codebook=trellis_codebook,
            fc1_trellis_pair_kind=fc1_trellis_pair_kind,
            fc2_trellis_pair_kind=fc2_trellis_pair_kind,
            direct_topk_routes=use_direct_topk_routes,
            use_expert_map=mapped_direct and use_direct_topk_routes,
            tc_decode_fused_sum=use_tc_decode,
            prefill_fused_sum_fp32=use_prefill_fused_sum,
            collect_activation_amax=collect_activation_amax,
            intermediate_rotation=intermediate_rotation_scales is not None,
            full_rotation=full_rotation,
            intermediate_hadamard=intermediate_hadamard,
            rotation_input_dtype=rotation_input_dtype,
            broadcast_suh=full_rotation and suh_gate_table.numel() == hidden_size,
            force_tile_config=prepared_tile_config,
            _require_cached=_w4a16_stream_is_capturing(
                stream,
                current_stream=current_stream,
            ),
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
            (
                expert_map is not None
                and not full_rotation
                and not use_direct_topk_routes
            ),
            element_dtype,
            bool(fast_math),
            swiglu_limit,
            float(swiglu_alpha),
            float(swiglu_beta),
            weight_layout,
            scale_format,
            w13_layout,
            trellis_bits,
            trellis_codebook,
            fc1_trellis_pair_kind,
            fc2_trellis_pair_kind,
            bool(use_direct_topk_routes),
            mapped_direct and use_direct_topk_routes,
            bool(use_tc_decode),
            bool(use_prefill_fused_sum),
            bool(collect_activation_amax),
            block_size_m,
            bool(intermediate_rotation_scales is not None),
            dual_a_required,
            full_rotation,
            intermediate_hadamard,
            rotation_input_dtype,
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
            int(getattr(fused_launch, "trellis_bits", 3)),
            str(
                getattr(
                    fused_launch,
                    "trellis_codebook",
                    LUT_E4M3,
                )
            ).lower(),
            getattr(fused_launch, "fc1_trellis_pair_kind", None),
            getattr(fused_launch, "fc2_trellis_pair_kind", None),
            bool(getattr(fused_launch, "direct_topk_routes", False)),
            bool(getattr(fused_launch, "use_expert_map", False)),
            bool(getattr(fused_launch, "tc_decode_fused_sum", False)),
            bool(getattr(fused_launch, "prefill_fused_sum_fp32", False)),
            bool(getattr(fused_launch, "collect_activation_amax", False)),
            int(fused_launch.moe_block_size),
            bool(getattr(fused_launch, "intermediate_rotation", False)),
            bool(getattr(fused_launch, "dual_a", False)),
            bool(getattr(fused_launch, "full_rotation", False)),
            bool(getattr(fused_launch, "intermediate_hadamard", False)),
            getattr(fused_launch, "rotation_input_dtype", fused_launch.element_dtype),
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
    capacity_m = int(fused.size_m)
    capacity_routed_rows = capacity_m * topk
    required_cache13_elements = (
        capacity_routed_rows * fc1_cols
        if use_prefill_fused_sum
        else capacity_routed_rows * max(fc1_cols, hidden_size)
    )
    if intermediate_cache13_flat.numel() < required_cache13_elements:
        raise ValueError(
            "intermediate_cache13 is smaller than the selected W4A16 launch capacity: "
            f"capacity_rows={capacity_m}, topk={topk}, "
            f"available_elements={intermediate_cache13_flat.numel()}, "
            f"required_elements={required_cache13_elements}, "
            f"fused_topk_sum={use_fused_topk_sum}, "
            f"prefill_fused_sum={use_prefill_fused_sum}, "
            f"collect_activation_amax={collect_activation_amax}, "
            f"full_rotation={full_rotation}, weight_layout={weight_layout}, "
            f"element_dtype={element_dtype}"
        )
    if intermediate_cache2_flat.numel() < capacity_routed_rows * intermediate_size:
        raise ValueError(
            "intermediate_cache2 is smaller than the selected W4A16 launch capacity: "
            f"capacity_rows={capacity_m}, topk={topk}"
        )
    fc1_store_cols = (
        max(fc1_cols, hidden_size)
        if use_tc_decode and weight_layout in BLOCK_CODECS
        else fc1_cols
    )
    fc1_out = intermediate_cache13_flat[: capacity_routed_rows * fc1_store_cols]
    activated = intermediate_cache2_flat[: capacity_routed_rows * intermediate_size]
    if use_prefill_fused_sum:
        assert prefill_sum_accum is not None
        fc2_out = prefill_sum_accum[: capacity_m * hidden_size]
    elif use_tc_decode:
        # The fused kernel owns per-token reduction and output initialization.
        fc2_out = output.view(-1)
    else:
        fc2_out = intermediate_cache13_flat[: capacity_routed_rows * hidden_size]
    fc1_scratch = _get_c_tmp(
        packed_gemm_scratch_elements(
            size_n=fc1_cols,
            route_slots=int(route_slots_for_scratch),
            moe_block_size=block_size_m,
            sms=sms,
            weight_layout=weight_layout,
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
            weight_layout=weight_layout,
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
    w13_scale_or_pair_modes = (
        fc1_trellis_pair_modes
        if fc1_trellis_pair_kind in {"PDYNAMIC", "P33_P43"}
        else prepared.w13_scale.view(torch.uint8).view(torch.int32).view(-1)
    )
    w2_scale_or_pair_modes = (
        fc2_trellis_pair_modes
        if fc2_trellis_pair_kind in {"PDYNAMIC", "P33_P43"}
        else prepared.w2_scale.view(torch.uint8).view(torch.int32).view(-1)
    )
    assert w13_scale_or_pair_modes is not None
    assert w2_scale_or_pair_modes is not None
    launch_common = (
        a_input,
        w13_arg,
        w2_arg,
        fc1_out,
        activated,
        fc2_out,
        w13_scale_or_pair_modes,
        w2_scale_or_pair_modes,
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
        bool(fused.zero_fc2_output),
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
    _intermediate_rotation = intermediate_rotation_scales is not None
    if _intermediate_rotation and (
        collect_activation_amax
        or use_tc_decode
        or (use_direct_topk_routes and not full_rotation)
        or weight_layout != "trellis_t256"
    ):
        raise ValueError(
            "intermediate_rotation_scales requires the trellis_t256 fused path "
            "(no calibration / tc-decode; direct routing requires full_rotation)"
        )
    if _intermediate_rotation:
        need = (
            int(prepared.num_experts)
            * (6 if intermediate_hadamard else 3)
            * intermediate_size
            if full_rotation
            else int(m) * topk * 3 * intermediate_size
        )
        rot_arg = intermediate_rotation_scales.reshape(-1)
        if int(rot_arg.numel()) < need or rot_arg.dtype != torch.float16:
            raise ValueError(
                "intermediate_rotation_scales must be fp16 with >= "
                f"{need} elements ({'experts' if full_rotation else 'routes'}"
                f"*{6 if intermediate_hadamard and full_rotation else 3}*intermediate); got "
                f"numel={int(rot_arg.numel())} dtype={rot_arg.dtype}"
            )
        rot_arg = rot_arg[:need]
    else:
        rot_arg = None
    if collect_activation_amax:
        assert activation_amax is not None
        assert layer_idx_int is not None
        torch.ops.b12x.w4a16_fused_moe_calibrated_launch(
            *launch_common,
            activation_amax,
            int(layer_idx_int),
            *launch_tail,
            int(stream),
        )
    elif (
        fused_launch is not None
        or _intermediate_rotation
        or weight_layout in {"trellis_t256", "iq2_xs", "iq2_xxs", "q8_0"}
        or (mapped_direct and use_direct_topk_routes)
        or use_prefill_fused_sum
    ):
        # Native t256 bypasses the registered torch op so its shape-derived
        # bitrate reaches compilation without widening the stable public op ABI.
        # Other production layouts keep the registered path byte-identical.
        (
            _rc_a,
            _rc_w13,
            _rc_w2,
            _rc_fc1,
            _rc_act,
            _rc_fc2,
            _rc_w13s,
            _rc_w2s,
            _rc_w13g,
            _rc_w2g,
            _rc_pri,
            _rc_bei,
            _rc_prc,
        ) = launch_common
        (
            _lt_tw,
            _lt_fc1s,
            _lt_fc2s,
            _lt_ws,
            _lt_m,
            _lt_capm,
            _lt_h,
            _lt_i,
            _lt_ne,
            _lt_tk,
            _lt_act,
            _lt_arwi,
            _lt_zfo,
            _lt_bsm,
            _lt_mmb,
            _lt_edt,
            _lt_fm,
            _lt_sms,
            _lt_msm,
            _lt_hsl,
            _lt_slv,
            _lt_sa,
            _lt_sb,
            _lt_wl,
            _lt_sf,
            _lt_w13l,
            _lt_fc1tk,
            _lt_fc1tn,
            _lt_fc2tk,
            _lt_fc2tn,
        ) = launch_tail
        launch_a = rotation_a_gate if full_rotation else _rc_a
        launch_a_up = rotation_a_up if full_rotation else a_input_up
        # Intermediate-Hadamard gate/up matrices consume the same transformed input. Reuse
        # the gate buffer instead of materializing an identical second row.
        if full_rotation and intermediate_hadamard:
            launch_a_up = rotation_a_gate
        assert launch_a is not None
        _w4a16_fused_moe_launch_flat(
            a_input=launch_a,
            w13_arg=_rc_w13,
            w2_arg=_rc_w2,
            fc1_out=_rc_fc1,
            activated=_rc_act,
            fc2_out=_rc_fc2,
            w13_scale_i32=_rc_w13s,
            w2_scale_i32=_rc_w2s,
            w13_global_scale=_rc_w13g,
            w2_global_scale=_rc_w2g,
            packed_route_indices=_rc_pri,
            block_expert_ids=_rc_bei,
            packed_route_count=_rc_prc,
            activation_amax=None,
            layer_idx=0,
            topk_weights=_lt_tw,
            fc1_scratch=_lt_fc1s,
            fc2_scratch=_lt_fc2s,
            workspace=_lt_ws,
            m=_lt_m,
            size_m=_lt_capm,
            hidden_size=_lt_h,
            intermediate_size=_lt_i,
            num_experts=_lt_ne,
            topk=_lt_tk,
            activation=_lt_act,
            apply_router_weight_on_input=_lt_arwi,
            zero_fc2_output=_lt_zfo,
            moe_block_size=_lt_bsm,
            max_m_blocks=_lt_mmb,
            element_dtype=_lt_edt,
            fast_math=_lt_fm,
            sms=_lt_sms,
            max_shared_mem=_lt_msm,
            has_swiglu_limit=_lt_hsl,
            swiglu_limit_value=_lt_slv,
            swiglu_alpha=_lt_sa,
            swiglu_beta=_lt_sb,
            weight_layout=_lt_wl,
            scale_format=_lt_sf,
            w13_layout=_lt_w13l,
            fc1_tile_k=_lt_fc1tk,
            fc1_tile_n=_lt_fc1tn,
            fc2_tile_k=_lt_fc2tk,
            fc2_tile_n=_lt_fc2tn,
            direct_topk_routes=bool(use_direct_topk_routes),
            tc_decode_fused_sum=bool(use_tc_decode),
            prefill_fused_sum_fp32=bool(use_prefill_fused_sum),
            collect_activation_amax=False,
            stream_int=int(stream),
            expert_map=expert_map if use_direct_topk_routes else None,
            rot_scales=rot_arg,
            intermediate_rotation=_intermediate_rotation,
            a_input_up=launch_a_up,
            trellis_bits=trellis_bits,
            trellis_codebook=trellis_codebook,
            fc1_trellis_pair_kind=fc1_trellis_pair_kind,
            fc2_trellis_pair_kind=fc2_trellis_pair_kind,
            full_rotation=full_rotation,
            intermediate_hadamard=intermediate_hadamard,
            rotation_input=_rc_a,
            suh_gate_table=suh_gate_table,
            suh_up_table=suh_up_table,
            launcher=fused,
        )
    else:
        torch.ops.b12x.w4a16_fused_moe_launch(
            *launch_common,
            *launch_tail,
            bool(use_direct_topk_routes),
            bool(use_tc_decode),
            int(stream),
        )

    if use_prefill_fused_sum:
        assert prefill_sum_accum is not None
        output.copy_(prefill_sum_accum[: m * hidden_size].view(m, hidden_size))
        return output
    if use_tc_decode:
        # FC2 already wrote the top-k-summed result into `output`.
        return output

    sum_expert_map = output_expert_map if output_expert_map is not None else expert_map
    sum_uses_map = sum_expert_map is not None and (
        full_rotation or use_direct_topk_routes
    )
    if topk_sum_launch is not None:
        expected_sum = (
            topk,
            hidden_size,
            full_rotation,
            intermediate_hadamard,
            int(prepared.num_experts) if full_rotation or sum_uses_map else 0,
            0 if not sum_uses_map else int(sum_expert_map.numel()),
            topk_ids.dtype if full_rotation or sum_uses_map else torch.int32,
            sum_uses_map,
        )
        actual_sum = (
            int(topk_sum_launch.topk),
            int(topk_sum_launch.hidden_size),
            bool(getattr(topk_sum_launch, "full_rotation", False)),
            bool(getattr(topk_sum_launch, "intermediate_hadamard", False)),
            int(getattr(topk_sum_launch, "num_experts", 0)),
            int(getattr(topk_sum_launch, "route_num_experts", 0)),
            getattr(topk_sum_launch, "route_ids_dtype", torch.int32),
            bool(getattr(topk_sum_launch, "use_expert_map", False)),
        )
        if actual_sum != expected_sum:
            raise RuntimeError(
                "preplanned W4A16 top-k sum launch does not match requested contract: "
                f"requested={expected_sum}, planned={actual_sum}"
            )
    if topk_sum_launch is not None or full_rotation or sum_uses_map:
        if full_rotation:
            assert svh_table is not None
        _w4a16_topk_sum_launch_flat(
            fc2_out,
            output,
            m,
            topk,
            hidden_size,
            element_dtype,
            int(stream),
            full_rotation=full_rotation,
            intermediate_hadamard=intermediate_hadamard,
            num_experts=int(prepared.num_experts),
            topk_weights=topk_weights if full_rotation else None,
            route_expert_ids=topk_ids,
            expert_map=sum_expert_map if sum_uses_map else None,
            svh_table=svh_table if full_rotation else None,
            launcher=topk_sum_launch,
        )
    else:
        torch.ops.b12x.w4a16_topk_sum_launch(
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














__all__ = [
    "W4A16ActivationCompileResult",
    "W4A16FusedMoeCompileResult",
    "W4A16GemmCompileResult",
    "W4A16TopKSumCompileResult",
    "W4A16FusedMoeKernel",
    "W4A16ActivationKernel",
    "W4A16GemmKernel",
    "W4A16TopKSumKernel",
    "build_w4a16_tier_local_map",
    "clear_w4a16_kernel_cache",
    "compile_w4a16_activation",
    "compile_w4a16_fused_moe",
    "compile_w4a16_gemm",
    "compile_w4a16_topk_sum",
    "pack_topk_routes_by_expert",
    "run_trellis256_dense",
    "run_w4a16_moe",
]
