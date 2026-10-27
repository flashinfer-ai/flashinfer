"""W4A16 MoE configuration and device helpers."""

from __future__ import annotations

from b12x._lib.quant.block_codec import BLOCK_CODECS
from b12x.moe._shared.trellis_codebooks import LUT_E4M3
from b12x.moe._shared.trellis_codebooks import LUT_FP16
from b12x.moe._shared.trellis_codebooks import MCG
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import Uint32
from b12x.moe._shared.kernels.w4a16.host import _W4A16_ALLOWED_ROUTED_SIZES
from b12x._lib.quant.block_codec import block_codec
import cutlass
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
from b12x._lib.quant.lut_e4m3 import lut_e4m3_direct_table
from b12x._lib.quant.lut_e4m3 import lut_e4m3_value_table
from b12x._lib.quant.lut_fp16 import lut_fp16_segment_table
from b12x.moe._shared.kernels.activations import normalize_moe_activation
from b12x.moe._shared.kernels.activations import normalize_swiglu_alpha_for_activation
from b12x.moe._shared.kernels.activations import normalize_swiglu_beta_for_activation
from b12x.moe._shared.kernels.activations import normalize_swiglu_limit_for_activation
import os
import torch


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
    *,
    weight_layout: str,
    scale_format: str,
    uses_m_block_8: bool,
    small_m_occupancy: int | None = None,
) -> bool:
    """Whether a small-M plan runs two CTAs per SM with three stages: opted in,
    and packed NVFP4 weights with E4M3 scales, the measured
    geometry."""
    return (
        uses_m_block_8
        and weight_layout == "packed"
        and scale_format == "e4m3_k16"
        and (
            _w4a16_small_m_occupancy()
            if small_m_occupancy is None
            else small_m_occupancy
        )
        == 2
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

_DEFAULT_SHARED_MEM_PER_SM = 102_400

# Shared memory the driver reserves per resident CTA.
_CTA_RESERVED_SHARED_MEM = 1024


def _resident_cta_shared_bytes(shared_bytes: int) -> int:
    """Shared memory one resident CTA occupies: its 1 KiB-aligned allocation and the reservation."""
    return _covering_count(int(shared_bytes), 1024) * 1024 + _CTA_RESERVED_SHARED_MEM


def _device_shared_mem_per_sm() -> int:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return int(
            getattr(
                props, "shared_memory_per_multiprocessor", _DEFAULT_SHARED_MEM_PER_SM
            )
        )
    return _DEFAULT_SHARED_MEM_PER_SM


def _cooperative_smem_carveout(
    *, blocks_per_sm: int, shared_bytes: int, shared_mem_per_sm: int
) -> int:
    """Preferred shared-memory carveout (percent) that admits ``blocks_per_sm`` CTAs.

    CUTLASS derives the automatic carveout from the CTAs' dynamic shared
    memory alone, without the per-CTA reservation: two CTAs of just under
    32 KiB get a 64 KiB carveout they do not fit in, and the cooperative
    launch is rejected (CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE). Wherever
    the automatic carveout admits the CTAs this selects the same carveout
    size, since the driver rounds a preference up to a supported size.
    """
    needed = int(blocks_per_sm) * _resident_cta_shared_bytes(shared_bytes)
    return min(100, _covering_count(100 * needed, int(shared_mem_per_sm)))


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

# Router-weight precision of the top-k combine.  By default FC2 multiplies each
# routed row by its router weight in the epilogue *after* the BF16 store
# (weight rounded to BF16, product rounded to BF16).  With
# B12X_W4A16_FP32_TOPK_WEIGHTS=1, FC2 stores the unweighted BF16 expert output
# and the top-k sum applies the FP32 router weight in its FP32 accumulation:
# out = bf16(sum_k w_k * bf16(y_k)), the reference MoE combine (for example
# HF ``index_add_(expert_output * w)`` into an FP32 buffer, one final cast).
_FP32_TOPK_WEIGHTS = os.environ.get("B12X_W4A16_FP32_TOPK_WEIGHTS", "0") == "1"

_MAX_DIRECT_TOPK_ROUTE_M = 6

_W4A16_SMALL_M_DIRECT_MAX_M = 8

_FC2_DIRECT_MIN_EXPERT_CAPACITY = 1024

_TC_DECODE_PACK_COLLIDING_PAIRS = 3

_TC_DECODE_PACK_SM_COVERAGE_CAP = 64

_TC_DECODE_PACK_SM_COVERAGE_NUMERATOR = 7

_TC_DECODE_PACK_SM_COVERAGE_DENOMINATOR = 8


@dsl_user_op
def _pack_f32x2_to_bf16x2_rn(
    x0: cutlass.Float32, x1: cutlass.Float32, *, loc=None, ip=None
) -> Uint32:
    """Pack two float32 values as bf16x2 (x0 in the low half), RN, unsaturated."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Float32(x0).ir_value(loc=loc, ip=ip),
                cutlass.Float32(x1).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pack_modelopt_words(
    a0: Uint32,
    a1: Uint32,
    b0: Uint32,
    b1: Uint32,
    byte_index: Uint32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
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
        )
    )


@dsl_user_op
def _dequant_native_fp4_word(
    packed: Uint32,
    is_fp16: cutlass.Constexpr[bool],
    *,
    loc=None,
    ip=None,
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
        "=r,=r,=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@dsl_user_op
def _gather_native_scale_bytes(
    a: Uint32, b: Uint32, c: Uint32, d: Uint32, byte: Uint32, *, loc=None, ip=None
):
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [v.ir_value(loc=loc, ip=ip) for v in (a, b, c, d, byte)],
            """
        {
            .reg .b32 selector, lo, hi;
            mad.lo.u32 selector, $5, 17, 64;
            prmt.b32 lo, $1, $2, selector;
            prmt.b32 hi, $3, $4, selector;
            prmt.b32 $0, lo, hi, 0x5410;
        }
        """,
            "=r,r,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


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
    # K32 FC2 tile of intermediate sizes 32 mod 64 (2048/TP6 = 352): one warp
    # row of four 64-column warps, the per-warp work of the N64/K128 tile.
    (128, 1, 16, 2, True): 128,
    (256, 1, 8, 8, False): 158,
    # The wide-N single-route tile is already above the one-CTA/SM register
    # threshold in the narrower N=128 schedule. The architectural cap keeps
    # the same launch contract until this specialization is measured directly.
    (256, 1, 16, 4, False): 255,
    (128, 1, 4, 8, False): 154,
    (128, 1, 8, 4, False): 143,
    (128, 1, 16, 2, False): 160,
    (128, 2, 16, 2, False): 220,
    (128, 3, 16, 2, False): 255,
    (128, 4, 16, 2, False): 255,
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

# Packed NVFP4 shards whose K is 32 mod 64 (FC2 of 2048/TP6 = 352 channels)
# have no dividing K64/K128 tile. They fall back to whole K32 stages; the
# 128-thread tile pairs with the N64/K128 FC1 tile of the same shard.
_K32_FALLBACK_TILE_CONFIGS = ((32, 256, 128),)


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
    *,
    weight_layout: str,
    tile_n: int,
    tile_k: int,
    uses_m_block_8: bool,
    pipeline_stages: int | None = None,
    scale_format: str = "e4m3_k16",
    small_m_occupancy: int | None = None,
) -> int:
    if pipeline_stages is not None:
        if type(pipeline_stages) is not int or pipeline_stages not in (2, 3, 4, 5):
            raise ValueError("W4A16 pipeline_stages must be 2, 3, 4 or 5")
        return pipeline_stages
    if weight_layout in BLOCK_CODECS and tile_n == 128 and tile_k == 128:
        return 2
    if (
        weight_layout in BLOCK_CODECS
        and uses_m_block_8
        and tile_n == 64
        and tile_k == 128
    ):
        return 3
    if _w4a16_small_m_double_occupancy(
        weight_layout=weight_layout,
        scale_format=scale_format,
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
    problem_k: int = 0,
) -> int:
    cta_m = int(cta_m_blocks) * 16
    cta_n = int(tile_n)
    cta_k = int(tile_k)
    stages = _w4a16_pipeline_stages(
        weight_layout=weight_layout,
        tile_n=cta_n,
        tile_k=cta_k,
        uses_m_block_8=uses_m_block_8,
        pipeline_stages=pipeline_stages,
        scale_format=scale_format,
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
            _covering_count(cta_k, _scale_group_size(scale_format)) * cta_n * 2 * stages
        )
    lut_size = (
        block_codec(weight_layout).lut_bytes(selectors=True)
        if weight_layout in BLOCK_CODECS
        else 0
    )
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
    problem_k: int = 0,
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
        problem_k=problem_k,
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
                weight_layout=weight_layout,
                scale_format=scale_format,
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
    allow_k32_tile: bool = False,
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
    # Packed NVFP4 K that is 32 mod 64 has no dividing K64 tile; whole K32
    # stages with two K16 steps per warp (N = 2 x cta_threads) cover it.
    k32_tile = (
        allow_k32_tile
        and weight_layout == "packed"
        and _normalize_scale_format(scale_format) == "e4m3_k16"
        and int(tile_k) == 32
        and int(problem_k) % 64 == 32
        and int(tile_n) == 2 * int(cta_threads)
    )
    if (
        int(tile_n) < 64
        or int(cta_threads) < 128
        or (int(tile_k) < 64 and not (wide_n_fc2_tile or k32_tile))
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
        problem_k=problem_k,
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
    candidates = tuple((config, False) for config in configs)
    if (
        weight_layout == "packed"
        and not allow_logical_tail
        and int(problem_k) % 64 == 32
    ):
        candidates += tuple((config, True) for config in _K32_FALLBACK_TILE_CONFIGS)
    for (tile_k, tile_n, cta_threads), k32_fallback in candidates:
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
            allow_k32_tile=k32_fallback,
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
            problem_k=problem_k,
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
    return (
        32
        if _normalize_scale_format(scale_format)
        in (
            "e8m0_k32",
            "e4m3_k32",
        )
        else 16
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
