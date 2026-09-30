"""SM120 quantized MLA cache record writers.

``concat_and_cache_glm_next_mla`` writes the GLM-5.3-Flash absorbed-MLA
record consumed by the explicit ``ModelType.GLM_NEXT`` sparse-MLA recipe.
The cache view selects either the 528-byte FP8 record:

    [   0, 512)  512 x E4M3 latent values (four consecutive 128-dim groups)
    [ 512, 528)  4 x fp32 group scales (group amax / 448.0)

The record has no RoPE payload.  The cache may have a padded page stride so a
selector-only pooled-K tail can share each physical allocation; only the
semantic row is written. The 304-byte NVFP4 record uses the common packed
latent layout in bytes [0, 288), zeroes [288, 292), stores its per-token outer
scale at [292, 296), and zero-pads [296, 304). Slot-to-page arithmetic stays in
Int64, including the products by page and record stride.

``concat_and_cache_nvfp4_mla_fp8_rope`` quantizes the MLA compressed latent
(512 16-bit dims) plus the decoupled RoPE key (64 dims) into the compact
368 B/token ``nvfp4_ds_mla`` record that this package's fp8-RoPE sparse-MLA
readers consume (``traits.kv_gmem_stride == 368``; offsets anchored by
``prefill_mg._NVFP4_FP8_ROPE_SCALE_OFFSET == 288`` and
``prefill_mg._NVFP4_ROPE_GMEM_OFFSET == 304``):

    [   0, 256)  packed E2M1 NoPE (512 x 4-bit, 32 group-16 blocks)
    [ 256, 288)  32 x E4M3 group scale bytes (group amax / 6.0)
    [ 288, 292)  fp32 RoPE scale (rope amax / 448.0)
    [ 292, 304)  zero pad
    [ 304, 368)  64 x E4M3 RoPE

This is the KV_FP8_ROPE=1 layout: the stock 432 B record's [288, 304) pad
carries the fp32 RoPE scale and the RoPE bytes stay at their stock offset,
re-encoded E4M3 (128 B BF16 -> 64 B E4M3), so the record shrinks in place.

One CTA per token; ``slot_mapping`` entries < 0 are skipped (padded CUDA
graph slots). The quantization recipe is the standard NVFP4 one at an
implicit global scale of 1.0, spelled with the same PTX conversions the
rest of this package uses: ``amax * rcp.approx.ftz(6.0)`` ->
``cvt.rn.satfinite.e4m3x2`` scale byte -> hardware-exact E4M3 decode
(``cvt.rn.f16x2.e4m3x2`` -- denormal-correct, the same decode the read
path applies) -> ``rcp.approx.ftz`` inverse ->
``cvt.rn.satfinite.e2m1x2`` packing (``quantize_and_pack_16_fast``).

The RoPE lane stores ``scale = amax / 448.0`` as fp32 at [288, 292) and
``cvt.rn.satfinite.e4m3x2(v / scale)`` bytes at [304, 368); the readers
reconstruct ``e4m3_decode(byte) * scale``
(``prefill_mg._ld_global_nvfp4_fp8_rope_bfloat2``).

``per_token_scale=True`` selects the inline-scale two-level variant: the
NoPE lane derives its own per-token second-level scale instead of assuming
the implicit global 1.0 (which parks small-magnitude tokens' group scales in
E4M3 subnormals -- the defect the static per-layer
``VLLM_NVFP4_MLA_SCALES_FILE`` calibration papers over).  Warp 0 reduces the
32 group amaxes to the token amax (butterfly shuffle), stores

    s_t = token_amax / (6.0 * 448.0)

as fp32 at [292, 296) (the first 4 bytes of the pad; [296, 304) stays zero),
and every group scale byte is encoded relative to ``s_t`` so the largest
group's E4M3 scale lands at the top of the E4M3 range by construction.  The
readers reconstruct ``e2m1 * e4m3_decode(scale_byte) * s_t`` -- the same
expression as the static path with ``latent_scale := s_t`` sourced from the
record instead of the launch.  The record width, RoPE lane, and all offsets
are unchanged; legacy records (zero at [292, 296)) are NOT readable in this
mode, so the mode is server-static and joins the kernel compile identity.
"""

from __future__ import annotations

from functools import lru_cache
from threading import RLock

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compiler import (
    KernelCompileSpec,
    compile as compile_cute,
    run_compiled,
    tensor_compile_fact,
)
from b12x._lib.program_cache import register_program_cache
from b12x._lib.intrinsics import (
    cvt_e4m3_to_f32_via_f16,
    cvt_f32_to_e4m3,
    cvt_f32x4_to_e4m3x4,
    f16x2_to_f32x2,
    fabs_f32,
    fmax_f32,
    get_ptr_as_int64,
    max_abs_16,
    pow2_ceil_ue8m0,
    quantize_and_pack_16_fast,
    rcp_approx_ftz,
    st_global_f32,
    st_global_u32,
    st_global_u64,
    st_global_u8,
)
from b12x._lib.utils import current_cuda_stream

_KV_LORA_RANK = 512
_PE_DIM = 64
_GROUP_SIZE = 16
_NUM_GROUPS = _KV_LORA_RANK // _GROUP_SIZE  # 32
_NOPE_BYTES = _KV_LORA_RANK // 2  # 256
_SCALE_BYTES = _NUM_GROUPS  # 32
# KV_FP8_ROPE=1 record geometry (matches the shipped readers: fp32 scale in
# the stock record's pad, E4M3 RoPE at the stock RoPE offset).
_ROPE_SCALE_OFFSET = _NOPE_BYTES + _SCALE_BYTES  # 288
_PAD_OFFSET = _ROPE_SCALE_OFFSET + 4  # 292
_PAD_BYTES = 12
_ROPE_OFFSET = _PAD_OFFSET + _PAD_BYTES  # 304
_RECORD_BYTES = _ROPE_OFFSET + _PE_DIM  # 368
_THREADS = 128
# E4M3 rope scale: exact compile-time f32 constant (double 1/448 rounded to
# f32 once), NOT a runtime rcp.approx -- torch references must mirror this.
_E4M3_MAX_RCP = 1.0 / 448.0
# Per-token second-level (NVFP4 two-level) scale: fp32 at [292, 296), value
# token_amax / (E2M1_MAX * E4M3_MAX).  Same exact-constant contract as
# _E4M3_MAX_RCP -- torch references must mirror this.
_LATENT_SCALE_OFFSET = _PAD_OFFSET  # 292
_LATENT_SCALE_BYTES = 4
_TWO_LEVEL_RCP = 1.0 / (6.0 * 448.0)

# GLM-5.3-Flash absorbed-MLA cache geometry.  Keep these names distinct from
# the nvfp4_ds_mla constants above: both writers intentionally live here, but
# their records are unrelated ABIs.
_GLM_NEXT_LATENT_DIM = 512
_GLM_NEXT_GROUP_SIZE = 128
_GLM_NEXT_NUM_GROUPS = _GLM_NEXT_LATENT_DIM // _GLM_NEXT_GROUP_SIZE
_GLM_NEXT_SCALE_OFFSET = _GLM_NEXT_LATENT_DIM
_GLM_NEXT_RECORD_BYTES = _GLM_NEXT_LATENT_DIM + _GLM_NEXT_NUM_GROUPS * 4
_GLM_NEXT_NVFP4_RECORD_BYTES = 304
_GLM_NEXT_E4M3_MAX_RCP = 1.0 / 448.0
_GLM_NEXT_WRITER_LOCK = RLock()
_GLM_NEXT_WRITER_COMPILED: dict[tuple[int, int, torch.dtype], object] = {}
_NVFP4_WRITER_COMPILED: dict[
    tuple[int, int, torch.dtype, torch.dtype, bool, bool], object
] = {}
register_program_cache(_GLM_NEXT_WRITER_COMPILED, lock=_GLM_NEXT_WRITER_LOCK)
register_program_cache(_NVFP4_WRITER_COMPILED, lock=_GLM_NEXT_WRITER_LOCK)


@dsl_user_op
def _ld_global_u32(base_ptr: Int64, *, loc=None, ip=None) -> Uint32:
    """Plain (coherent) 32-bit global load."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
            "ld.global.b32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16x2_to_f32x2(bf2: Uint32, *, loc=None, ip=None):
    """Exact promotion of packed bfloat16x2 to two float32 (no scaling)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(bf2).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 lo, hi;
            and.b32 lo, $2, 0xFFFF;
            shr.b32 hi, $2, 16;
            shl.b32 lo, lo, 16;
            shl.b32 hi, hi, 16;
            mov.b32 $0, lo;
            mov.b32 $1, hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    f0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    f1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    return Float32(f0), Float32(f1)


@cute.jit
def _glm_next_cache_record_address(
    base: Int64,
    slot: Int64,
    block_stride: Int64,
    entry_stride: Int64,
    *,
    block_size: cutlass.Constexpr,
) -> Int64:
    """Resolve a flat slot without narrowing either scaled offset."""
    block_size64 = Int64(block_size)
    block_idx = slot // block_size64
    block_off = slot - block_idx * block_size64
    return base + block_idx * block_stride + block_off * entry_stride


class ConcatAndCacheNvfp4MlaFp8RopeKernel:
    """Per-token packed MLA record writer, including V4.1 indexed records.

    With ``has_rope=True``, each token uses the 368-byte KV_FP8_ROPE=1
    nvfp4_ds_mla record. With ``has_rope=False``, each token uses the 304-byte
    GLM_NEXT record and omits the RoPE lane. ``dsv41=True`` instead specializes
    the single-level group scales with their minimum floor and omits all bytes
    after the 288-byte E2M1-plus-E4M3 record.

    Thread mapping (128 threads/CTA): threads 0-31 quantize one 16-dim
    group each (eight coherent 32-bit loads -> exact f32 promote -> E2M1
    pack + E4M3 scale byte); threads 0-11 zero the [292, 304) pad; thread
    32 quantizes the RoPE lane to E4M3 with one fp32 per-token scale.
    """

    def __init__(
        self,
        block_size: int,
        is_bf16: bool,
        per_token_scale: bool = False,
        has_rope: bool = True,
        dsv41: bool = False,
    ):
        self.block_size = int(block_size)
        self.is_bf16 = bool(is_bf16)
        self.per_token_scale = bool(per_token_scale)
        self.has_rope = bool(has_rope)
        self.dsv41 = bool(dsv41)

    @cute.jit
    def __call__(
        self,
        kv_c: cute.Tensor,  # (num_tokens, 512) bf16/f16
        k_pe: cute.Tensor,  # (num_tokens, 64) bf16/f16
        kv_cache: cute.Tensor,  # (num_blocks, block_size, 368) u8
        slot_mapping: cute.Tensor,  # (num_tokens, 1) int64
        kv_c_stride: Int64,  # kv_c.stride(0), elements
        k_pe_stride: Int64,  # k_pe.stride(0), elements
        block_stride: Int64,  # kv_cache.stride(0), bytes
        entry_stride: Int64,  # kv_cache.stride(1), bytes
        slot_capacity: Int64,
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            kv_c_stride,
            k_pe_stride,
            block_stride,
            entry_stride,
            slot_capacity,
        ).launch(
            grid=(num_tokens, 1, 1),
            block=[_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        kv_c: cute.Tensor,
        k_pe: cute.Tensor,
        kv_cache: cute.Tensor,
        slot_mapping: cute.Tensor,
        kv_c_stride: Int64,
        k_pe_stride: Int64,
        block_stride: Int64,
        entry_stride: Int64,
        slot_capacity: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        token_idx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        token = Int32(token_idx)

        slot = Int64(slot_mapping[token])
        if (slot >= Int64(0)) & (slot < slot_capacity):
            dst = _glm_next_cache_record_address(
                get_ptr_as_int64(kv_cache, 0),
                slot,
                block_stride,
                entry_stride,
                block_size=self.block_size,
            )

            # --- NoPE: one 16-dim group per thread -> 8 B E2M1 + 1 scale byte.
            if tid < Int32(_NUM_GROUPS):
                src_elem = token.to(Int64) * kv_c_stride + tid.to(Int64) * Int64(
                    _GROUP_SIZE
                )
                vals = cute.make_rmem_tensor((_GROUP_SIZE,), Float32)
                for i in cutlass.range_constexpr(_GROUP_SIZE // 2):
                    pair = _ld_global_u32(
                        get_ptr_as_int64(kv_c, src_elem + Int64(2 * i))
                    )
                    if cutlass.const_expr(self.is_bf16):
                        f0, f1 = _bf16x2_to_f32x2(pair)
                    else:
                        f0, f1 = f16x2_to_f32x2(pair)
                    vals[2 * i] = f0
                    vals[2 * i + 1] = f1

                group_amax = max_abs_16(vals)
                if cutlass.const_expr(self.per_token_scale):
                    # Two-level NVFP4: warp-reduce the 32 group amaxes to the
                    # token amax (threads 0-31 are exactly warp 0), derive
                    # s_t = token_amax/(6*448) so the largest group's E4M3
                    # scale byte encodes 448 (top of range, no subnormals),
                    # and quantize every group relative to s_t.  Lane 0
                    # stores s_t fp32 at [292, 296) for the readers.
                    token_amax = group_amax
                    tmp = cute.arch.shuffle_sync_bfly(token_amax, offset=1)
                    token_amax = fmax_f32(token_amax, tmp)
                    tmp = cute.arch.shuffle_sync_bfly(token_amax, offset=2)
                    token_amax = fmax_f32(token_amax, tmp)
                    tmp = cute.arch.shuffle_sync_bfly(token_amax, offset=4)
                    token_amax = fmax_f32(token_amax, tmp)
                    tmp = cute.arch.shuffle_sync_bfly(token_amax, offset=8)
                    token_amax = fmax_f32(token_amax, tmp)
                    tmp = cute.arch.shuffle_sync_bfly(token_amax, offset=16)
                    token_amax = fmax_f32(token_amax, tmp)
                    latent_scale = token_amax * Float32(_TWO_LEVEL_RCP)
                    if tid == Int32(0):
                        st_global_f32(dst + Int64(_LATENT_SCALE_OFFSET), latent_scale)
                    scale_u32 = Uint32(0)
                    packed64 = Uint64(0)
                    if latent_scale != Float32(0.0):
                        inv_latent = rcp_approx_ftz(latent_scale)
                        scale_f32 = (group_amax * inv_latent) * rcp_approx_ftz(
                            Float32(6.0)
                        )
                        scale_u32 = cvt_f32_to_e4m3(scale_f32)
                        decoded_scale = cvt_e4m3_to_f32_via_f16(scale_u32)
                        if decoded_scale != Float32(0.0):
                            packed64 = quantize_and_pack_16_fast(
                                vals, rcp_approx_ftz(decoded_scale) * inv_latent
                            )
                    st_global_u64(dst + tid.to(Int64) * Int64(8), packed64)
                    st_global_u8(
                        dst + Int64(_NOPE_BYTES) + tid.to(Int64),
                        cutlass.Uint8(scale_u32 & Uint32(0xFF)),
                    )
                else:
                    # NVFP4 block quant at global scale 1.0: scale byte =
                    # e4m3(amax/6); values scaled by rcp.approx.ftz of the
                    # hardware-exact decode of that byte (what the reader
                    # multiplies back), then satfinite E2M1.
                    if cutlass.const_expr(self.dsv41):
                        group_amax = fmax_f32(group_amax, Float32(6.0 * 2.0**-9))
                        scale_f32 = group_amax / Float32(6.0)
                    else:
                        scale_f32 = group_amax * rcp_approx_ftz(Float32(6.0))
                    scale_u32 = cvt_f32_to_e4m3(scale_f32)
                    decoded_scale = cvt_e4m3_to_f32_via_f16(scale_u32)
                    packed64 = Uint64(0)
                    if decoded_scale != Float32(0.0):
                        if cutlass.const_expr(self.dsv41):
                            inv_scale = Float32(1.0) / decoded_scale
                        else:
                            inv_scale = rcp_approx_ftz(decoded_scale)
                        packed64 = quantize_and_pack_16_fast(vals, inv_scale)
                    st_global_u64(dst + tid.to(Int64) * Int64(8), packed64)
                    st_global_u8(
                        dst + Int64(_NOPE_BYTES) + tid.to(Int64),
                        cutlass.Uint8(scale_u32 & Uint32(0xFF)),
                    )

            # --- Zero pad: [292, 304) in static mode; [296, 304) when the
            # per-token latent scale occupies [292, 296).
            if cutlass.const_expr(self.per_token_scale and not self.dsv41):
                if tid < Int32(_PAD_BYTES - _LATENT_SCALE_BYTES):
                    st_global_u8(
                        dst + Int64(_PAD_OFFSET + _LATENT_SCALE_BYTES) + tid.to(Int64),
                        cutlass.Uint8(0),
                    )
            elif cutlass.const_expr(not self.dsv41):
                if tid < Int32(_PAD_BYTES):
                    st_global_u8(
                        dst + Int64(_PAD_OFFSET) + tid.to(Int64),
                        cutlass.Uint8(0),
                    )

            if cutlass.const_expr(not self.has_rope and not self.dsv41):
                if tid == Int32(_NUM_GROUPS):
                    st_global_u32(dst + Int64(_ROPE_SCALE_OFFSET), Uint32(0))

            # --- RoPE lane: amax -> fp32 scale at [288, 292) -> satfinite
            # E4M3 bytes at [304, 368).
            if cutlass.const_expr(self.has_rope) and tid == Int32(_NUM_GROUPS):
                rope_vals = cute.make_rmem_tensor((_PE_DIM,), Float32)
                for i in cutlass.range_constexpr(_PE_DIM // 2):
                    pair = _ld_global_u32(
                        get_ptr_as_int64(
                            k_pe,
                            token.to(Int64) * k_pe_stride + Int64(2 * i),
                        )
                    )
                    if cutlass.const_expr(self.is_bf16):
                        f0, f1 = _bf16x2_to_f32x2(pair)
                    else:
                        f0, f1 = f16x2_to_f32x2(pair)
                    rope_vals[2 * i] = f0
                    rope_vals[2 * i + 1] = f1
                rope_amax = Float32(0.0)
                for i in cutlass.range_constexpr(_PE_DIM):
                    rope_amax = fmax_f32(rope_amax, fabs_f32(rope_vals[i]))
                rope_scale = rope_amax * Float32(_E4M3_MAX_RCP)
                st_global_f32(dst + Int64(_ROPE_SCALE_OFFSET), rope_scale)
                for w in cutlass.range_constexpr(_PE_DIM // 8):
                    q8 = Uint64(0)
                    if rope_scale != Float32(0.0):
                        inv = rcp_approx_ftz(rope_scale)
                        lo = cvt_f32x4_to_e4m3x4(
                            rope_vals[8 * w + 0] * inv,
                            rope_vals[8 * w + 1] * inv,
                            rope_vals[8 * w + 2] * inv,
                            rope_vals[8 * w + 3] * inv,
                        )
                        hi = cvt_f32x4_to_e4m3x4(
                            rope_vals[8 * w + 4] * inv,
                            rope_vals[8 * w + 5] * inv,
                            rope_vals[8 * w + 6] * inv,
                            rope_vals[8 * w + 7] * inv,
                        )
                        q8 = lo.to(Uint64) | (hi.to(Uint64) << Uint64(32))
                    st_global_u64(dst + Int64(_ROPE_OFFSET + 8 * w), q8)


class ConcatAndCacheGlmNextMlaKernel:
    """BF16 to GLM_NEXT FP8 or V4.1 SWA E4M3-plus-UE8M0 cache writer.

    Each warp owns one consecutive 128-dim quantization group.  Its 32 lanes
    load and store four adjacent values, reduce amax within the warp, and lane
    zero writes the group's FP32 scale. V4.1 specializes this to 8-lane,
    32-value groups and one pow2-ceil UE8M0 byte per group. There is one CTA
    per source token.
    """

    def __init__(self, block_size: int, dsv41: bool = False):
        self.block_size = int(block_size)
        self.dsv41 = bool(dsv41)
        self.group_size = 32 if dsv41 else _GLM_NEXT_GROUP_SIZE
        self.lanes_per_group = self.group_size // 4

    @cute.jit
    def __call__(
        self,
        kv_c: cute.Tensor,  # (num_tokens, 512) bf16
        kv_cache: cute.Tensor,  # (num_blocks, block_size, 528) u8
        slot_mapping: cute.Tensor,  # (num_tokens,) int32/int64
        kv_c_stride: Int64,  # kv_c.stride(0), elements
        block_stride: Int64,  # kv_cache.stride(0), bytes
        entry_stride: Int64,  # kv_cache.stride(1), bytes
        slot_capacity: Int64,
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            kv_c,
            kv_cache,
            slot_mapping,
            kv_c_stride,
            block_stride,
            entry_stride,
            slot_capacity,
        ).launch(
            grid=(num_tokens, 1, 1),
            block=[_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        kv_c: cute.Tensor,
        kv_cache: cute.Tensor,
        slot_mapping: cute.Tensor,
        kv_c_stride: Int64,
        block_stride: Int64,
        entry_stride: Int64,
        slot_capacity: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        token_idx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        token = Int32(token_idx)
        token64 = token.to(Int64)

        slot = Int64(slot_mapping[token])
        if (slot >= Int64(0)) & (slot < slot_capacity):
            dst = _glm_next_cache_record_address(
                get_ptr_as_int64(kv_cache, 0),
                slot,
                block_stride,
                entry_stride,
                block_size=self.block_size,
            )

            group = tid // Int32(self.lanes_per_group)
            lane = tid - group * Int32(self.lanes_per_group)
            src_elem = (
                token64 * kv_c_stride
                + group.to(Int64) * Int64(self.group_size)
                + lane.to(Int64) * Int64(4)
            )
            pair01 = _ld_global_u32(get_ptr_as_int64(kv_c, src_elem))
            pair23 = _ld_global_u32(get_ptr_as_int64(kv_c, src_elem + Int64(2)))
            v0, v1 = _bf16x2_to_f32x2(pair01)
            v2, v3 = _bf16x2_to_f32x2(pair23)

            group_amax = fabs_f32(v0)
            group_amax = fmax_f32(group_amax, fabs_f32(v1))
            group_amax = fmax_f32(group_amax, fabs_f32(v2))
            group_amax = fmax_f32(group_amax, fabs_f32(v3))
            group_amax = fmax_f32(
                group_amax,
                cute.arch.shuffle_sync_bfly(group_amax, offset=1),
            )
            group_amax = fmax_f32(
                group_amax,
                cute.arch.shuffle_sync_bfly(group_amax, offset=2),
            )
            group_amax = fmax_f32(
                group_amax,
                cute.arch.shuffle_sync_bfly(group_amax, offset=4),
            )
            if cutlass.const_expr(not self.dsv41):
                group_amax = fmax_f32(
                    group_amax,
                    cute.arch.shuffle_sync_bfly(group_amax, offset=8),
                )
                group_amax = fmax_f32(
                    group_amax,
                    cute.arch.shuffle_sync_bfly(group_amax, offset=16),
                )

            if cutlass.const_expr(self.dsv41):
                scale, scale_byte = pow2_ceil_ue8m0(
                    fmax_f32(group_amax, Float32(1e-4))
                    * Float32(_E4M3_MAX_RCP)
                )
            else:
                # Preserve the GLM_NEXT zero-group unit scale.
                scale = Float32(1.0)
                if group_amax != Float32(0.0):
                    scale = group_amax * Float32(_GLM_NEXT_E4M3_MAX_RCP)
            inv_scale = rcp_approx_ftz(scale)
            packed = cvt_f32x4_to_e4m3x4(
                v0 * inv_scale,
                v1 * inv_scale,
                v2 * inv_scale,
                v3 * inv_scale,
            )
            st_global_u32(
                dst
                + group.to(Int64) * Int64(self.group_size)
                + lane.to(Int64) * Int64(4),
                packed,
            )
            if lane == Int32(0):
                if cutlass.const_expr(self.dsv41):
                    st_global_u8(
                        dst + Int64(_GLM_NEXT_SCALE_OFFSET) + group.to(Int64),
                        cutlass.Uint8(scale_byte & Uint32(0xFF)),
                    )
                else:
                    st_global_f32(
                        dst + Int64(_GLM_NEXT_SCALE_OFFSET) + group.to(Int64) * Int64(4),
                        scale,
                    )


@lru_cache(maxsize=None)
def _build_concat_and_cache_nvfp4_mla_fp8_rope_kernel(
    block_size: int,
    is_bf16: bool,
    per_token_scale: bool = False,
    has_rope: bool = True,
) -> ConcatAndCacheNvfp4MlaFp8RopeKernel:
    return ConcatAndCacheNvfp4MlaFp8RopeKernel(
        block_size, is_bf16, per_token_scale, has_rope
    )


def clear_nvfp4_mla_fp8_rope_kv_cache_kernel_cache() -> None:
    _build_concat_and_cache_nvfp4_mla_fp8_rope_kernel.cache_clear()
    _build_compressed_cache_writer.cache_clear()
    with _GLM_NEXT_WRITER_LOCK:
        _NVFP4_WRITER_COMPILED.clear()


@lru_cache(maxsize=None)
def _build_concat_and_cache_glm_next_mla_kernel(
    block_size: int,
) -> ConcatAndCacheGlmNextMlaKernel:
    return ConcatAndCacheGlmNextMlaKernel(block_size)


def clear_glm_next_mla_kv_cache_kernel_cache() -> None:
    _build_concat_and_cache_glm_next_mla_kernel.cache_clear()
    _build_concat_and_cache_nvfp4_mla_fp8_rope_kernel.cache_clear()
    _build_compressed_cache_writer.cache_clear()
    with _GLM_NEXT_WRITER_LOCK:
        _GLM_NEXT_WRITER_COMPILED.clear()
        _NVFP4_WRITER_COMPILED.clear()


def _torch_to_cutlass_dtype(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.uint8:
        return cutlass.Uint8
    if dtype == torch.int32:
        return cutlass.Int32
    if dtype == torch.int64:
        return cutlass.Int64
    raise TypeError(f"unsupported dtype {dtype}")


def _to_kernel_tensor(
    tensor: torch.Tensor,
    *,
    assumed_align: int,
    leading_dim: int,
) -> cute.Tensor:
    if hasattr(tensor, "fake_mode"):
        from cutlass.cute.runtime import make_fake_tensor
        return make_fake_tensor(
            _torch_to_cutlass_dtype(tensor.dtype),
            tuple(cute.sym_int(32) for _ in tensor.shape),
            tuple(1 if index == leading_dim else cute.sym_int(64)
                  for index in range(tensor.ndim)),
            assumed_align=assumed_align,
        )
    cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
    cute_tensor.element_type = _torch_to_cutlass_dtype(tensor.dtype)
    return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)


def _glm_next_cache_byte_offset(
    slot: int,
    *,
    block_size: int,
    block_stride: int,
    entry_stride: int = _GLM_NEXT_RECORD_BYTES,
) -> int:
    """Host mirror of the writer's Int64 slot-to-byte address calculation."""
    if slot < 0:
        raise ValueError("slot must be non-negative")
    if block_size <= 0 or block_stride <= 0 or entry_stride <= 0:
        raise ValueError("cache strides and block_size must be positive")
    block_idx, block_off = divmod(int(slot), int(block_size))
    return block_idx * int(block_stride) + block_off * int(entry_stride)


def _glm_next_cache_writer_signature(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[int, int, torch.dtype]:
    device_index = kv_c.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return int(device_index), int(kv_cache.shape[1]), slot_mapping.dtype


def _glm_next_cache_writer_launch(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[object, tuple[object, ...], KernelCompileSpec]:
    num_tokens = int(slot_mapping.shape[0])
    block_size = int(kv_cache.shape[1])
    slot_capacity = int(kv_cache.shape[0]) * block_size
    kernel = _build_concat_and_cache_glm_next_mla_kernel(block_size)
    args = (
        _to_kernel_tensor(kv_c, assumed_align=4, leading_dim=1),
        _to_kernel_tensor(kv_cache, assumed_align=16, leading_dim=2),
        _to_kernel_tensor(
            slot_mapping,
            assumed_align=8 if slot_mapping.dtype == torch.int64 else 4,
            leading_dim=0,
        ),
        Int64(int(kv_c.stride(0))),
        Int64(int(kv_cache.stride(0))),
        Int64(int(kv_cache.stride(1))),
        Int64(slot_capacity),
        Int32(num_tokens),
        current_cuda_stream(),
    )
    cache_key = (
        tensor_compile_fact("kv_c", kv_c, dynamic_dims=(0,), dynamic_strides=(0,)),
        tensor_compile_fact(
            "kv_cache",
            kv_cache,
            dynamic_dims=(0,),
            dynamic_strides=(0,),
        ),
        tensor_compile_fact("slot_mapping", slot_mapping, dynamic_dims=(0,)),
        block_size,
    )
    spec = KernelCompileSpec.from_key(
        "attention.mla.glm_next_kv_cache",
        1,
        cache_key,
        labels=("kv_c", "kv_cache", "slot_mapping", "block_size"),
    )
    return kernel, args, spec


def _compile_glm_next_mla_cache_writer(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> object:
    signature = _glm_next_cache_writer_signature(kv_c, kv_cache, slot_mapping)
    with _GLM_NEXT_WRITER_LOCK:
        compiled = _GLM_NEXT_WRITER_COMPILED.get(signature)
    if compiled is None:
        kernel, args, spec = _glm_next_cache_writer_launch(kv_c, kv_cache, slot_mapping)
        compiled = compile_cute(kernel, *args, compile_spec=spec)
        with _GLM_NEXT_WRITER_LOCK:
            _GLM_NEXT_WRITER_COMPILED[signature] = compiled
    return compiled


def _concat_and_cache_glm_next_mla_flat_launch(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    if int(slot_mapping.shape[0]) == 0:
        return
    signature = _glm_next_cache_writer_signature(kv_c, kv_cache, slot_mapping)
    with _GLM_NEXT_WRITER_LOCK:
        compiled = _GLM_NEXT_WRITER_COMPILED.get(signature)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "GLM_NEXT cache-writer compile miss during CUDA graph capture; "
                "prepare its exact writer specialization before capture"
            )
        compiled = _compile_glm_next_mla_cache_writer(kv_c, kv_cache, slot_mapping)
    _, args, _ = _glm_next_cache_writer_launch(kv_c, kv_cache, slot_mapping)
    run_compiled(compiled, args)


@torch.library.custom_op(
    "b12x::concat_and_cache_glm_next_mla",
    mutates_args=("kv_cache",),
)
def _concat_and_cache_glm_next_mla_op(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    _concat_and_cache_glm_next_mla_flat_launch(kv_c, kv_cache, slot_mapping)


@_concat_and_cache_glm_next_mla_op.register_fake
def _concat_and_cache_glm_next_mla_fake(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    return None


def _validate_glm_next_mla_cache_writer_args(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Validate the fixed-buffer GLM_NEXT cache-writer contract."""
    if kv_c.ndim != 2 or int(kv_c.shape[1]) != _GLM_NEXT_LATENT_DIM:
        raise ValueError(
            "kv_c must be (num_tokens, "
            f"{_GLM_NEXT_LATENT_DIM}), got {tuple(kv_c.shape)}"
        )
    if kv_c.dtype != torch.bfloat16:
        raise TypeError(f"kv_c must be BF16, got {kv_c.dtype}")
    record_bytes = int(kv_cache.shape[2]) if kv_cache.ndim == 3 else -1
    if kv_cache.ndim != 3 or record_bytes not in (
        _GLM_NEXT_RECORD_BYTES,
        _GLM_NEXT_NVFP4_RECORD_BYTES,
    ):
        raise ValueError(
            "kv_cache must be (num_pages, page_size, "
            f"{_GLM_NEXT_RECORD_BYTES}|{_GLM_NEXT_NVFP4_RECORD_BYTES}) uint8, "
            f"got {tuple(kv_cache.shape)}"
        )
    num_pages = int(kv_cache.shape[0])
    page_size = int(kv_cache.shape[1])
    if num_pages <= 0 or page_size <= 0:
        raise ValueError("kv_cache num_pages and page_size must be positive")
    if kv_cache.dtype != torch.uint8:
        raise TypeError(f"kv_cache must be uint8, got {kv_cache.dtype}")
    if slot_mapping.ndim != 1 or slot_mapping.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError(
            "slot_mapping must be a 1-D int32 or int64 tensor, got "
            f"{tuple(slot_mapping.shape)} {slot_mapping.dtype}"
        )
    if not slot_mapping.is_contiguous():
        raise ValueError("slot_mapping must be contiguous")
    num_tokens = int(slot_mapping.shape[0])
    if int(kv_c.shape[0]) < num_tokens:
        raise ValueError(
            f"kv_c must cover slot_mapping's {num_tokens} tokens, got "
            f"{int(kv_c.shape[0])} rows"
        )
    if kv_c.stride(1) != 1:
        raise ValueError("kv_c rows must be innermost-contiguous")
    if kv_c.stride(0) % 2 != 0 or kv_c.data_ptr() % 4 != 0:
        raise ValueError("kv_c rows must be 4-byte aligned (even row stride)")
    if kv_cache.stride(2) != 1 or kv_cache.stride(1) != record_bytes:
        raise ValueError("kv_cache must have packed semantic records")
    semantic_page_bytes = page_size * record_bytes
    page_stride = int(kv_cache.stride(0))
    if page_stride < semantic_page_bytes:
        raise ValueError(
            "kv_cache page stride must cover every semantic record in the page"
        )
    if (
        kv_cache.data_ptr() % 16 != 0
        or page_stride % 16 != 0
        or kv_cache.stride(1) % 16 != 0
    ):
        raise ValueError("kv_cache base, pages, and records must be 16-byte aligned")
    slot_capacity = num_pages * page_size
    if num_tokens >= 2**31:
        raise ValueError("slot_mapping token count must fit in int32 launch geometry")
    if slot_capacity >= 2**63:
        raise ValueError("kv_cache slot capacity must fit in int64")
    if not (kv_c.is_cuda and kv_cache.is_cuda and slot_mapping.is_cuda):
        raise ValueError("all tensors must be on CUDA")
    if len({kv_c.device, kv_cache.device, slot_mapping.device}) != 1:
        raise ValueError("all tensors must be on the same device")




def concat_and_cache_glm_next_mla(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    plan,
) -> None:
    """Write GLM_NEXT cache records through a prepared plan."""
    from b12x.attention.sparse_mla._preparation import writer_state
    writer_state(plan, device=kv_cache.device).run(kv_c, kv_cache, slot_mapping)


def concat_and_cache_glm_next_mla_fp8(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    plan,
) -> None:
    """Write the fixed 528-byte GLM_NEXT FP8 recipe through preparation."""
    if int(kv_cache.shape[-1]) != _GLM_NEXT_RECORD_BYTES:
        raise ValueError(
            "GLM_NEXT FP8 writer requires 528-byte records, got "
            f"{int(kv_cache.shape[-1])}"
        )
    concat_and_cache_glm_next_mla(kv_c, kv_cache, slot_mapping, plan=plan)


def concat_and_cache_glm_next_mla_nvfp4(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    plan,
) -> None:
    """Write the fixed 304-byte GLM_NEXT NVFP4 recipe through preparation."""
    if int(kv_cache.shape[-1]) != _GLM_NEXT_NVFP4_RECORD_BYTES:
        raise ValueError(
            "GLM_NEXT NVFP4 writer requires 304-byte records, got "
            f"{int(kv_cache.shape[-1])}"
        )
    concat_and_cache_glm_next_mla(kv_c, kv_cache, slot_mapping, plan=plan)


def _nvfp4_mla_writer_signature(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    per_token_scale: bool,
    has_rope: bool,
) -> tuple[int, int, torch.dtype, torch.dtype, bool, bool]:
    device_index = kv_c.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return (
        int(device_index),
        int(kv_cache.shape[1]),
        kv_c.dtype,
        slot_mapping.dtype,
        bool(per_token_scale),
        bool(has_rope),
    )


def _nvfp4_mla_writer_launch(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    per_token_scale: bool,
    has_rope: bool,
) -> tuple[object, tuple[object, ...], KernelCompileSpec]:
    num_tokens = int(slot_mapping.shape[0])
    block_size = int(kv_cache.shape[1])
    slot_capacity = int(kv_cache.shape[0]) * block_size
    is_bf16 = kv_c.dtype == torch.bfloat16
    kernel = _build_concat_and_cache_nvfp4_mla_fp8_rope_kernel(
        block_size, is_bf16, per_token_scale, has_rope
    )

    args = (
        _to_kernel_tensor(kv_c, assumed_align=4, leading_dim=1),
        _to_kernel_tensor(k_pe, assumed_align=4, leading_dim=1),
        _to_kernel_tensor(kv_cache, assumed_align=16, leading_dim=2),
        _to_kernel_tensor(
            slot_mapping,
            assumed_align=8 if slot_mapping.dtype == torch.int64 else 4,
            leading_dim=0,
        ),
        Int64(int(kv_c.stride(0))),
        Int64(int(k_pe.stride(0))),
        Int64(int(kv_cache.stride(0))),
        Int64(int(kv_cache.stride(1))),
        Int64(slot_capacity),
        Int32(num_tokens),
        current_cuda_stream(),
    )
    cache_key = (
        tensor_compile_fact("kv_c", kv_c, dynamic_dims=(0,), dynamic_strides=(0,)),
        tensor_compile_fact("k_pe", k_pe, dynamic_dims=(0,), dynamic_strides=(0,)),
        tensor_compile_fact(
            "kv_cache",
            kv_cache,
            dynamic_dims=(0,),
            dynamic_strides=(0, 1),
        ),
        tensor_compile_fact("slot_mapping", slot_mapping, dynamic_dims=(0,)),
        str(kv_c.dtype),
        block_size,
        bool(per_token_scale),
        bool(has_rope),
    )
    op_name = (
        "attention.mla.nvfp4_fp8_rope_kv_cache"
        if has_rope
        else "attention.mla.glm_next_nvfp4_kv_cache"
    )
    version = 3 if has_rope else 1
    spec = KernelCompileSpec.from_key(
        op_name,
        version,
        cache_key,
        labels=(
            "kv_c",
            "k_pe",
            "kv_cache",
            "slot_mapping",
            "kv_dtype",
            "block_size",
            "per_token_scale",
            "has_rope",
        ),
    )
    return kernel, args, spec


def _compile_nvfp4_mla_writer(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    per_token_scale: bool,
    has_rope: bool,
) -> object:
    signature = _nvfp4_mla_writer_signature(
        kv_c, kv_cache, slot_mapping, per_token_scale, has_rope
    )
    with _GLM_NEXT_WRITER_LOCK:
        compiled = _NVFP4_WRITER_COMPILED.get(signature)
    if compiled is None:
        kernel, args, spec = _nvfp4_mla_writer_launch(
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            per_token_scale,
            has_rope,
        )
        compiled = compile_cute(kernel, *args, compile_spec=spec)
        with _GLM_NEXT_WRITER_LOCK:
            _NVFP4_WRITER_COMPILED[signature] = compiled
    return compiled


def _concat_and_cache_nvfp4_mla_fp8_rope_flat_launch(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    per_token_scale: bool = False,
    has_rope: bool = True,
) -> None:
    if int(slot_mapping.shape[0]) == 0:
        return
    signature = _nvfp4_mla_writer_signature(
        kv_c, kv_cache, slot_mapping, per_token_scale, has_rope
    )
    with _GLM_NEXT_WRITER_LOCK:
        compiled = _NVFP4_WRITER_COMPILED.get(signature)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "NVFP4 MLA cache-writer compile miss during CUDA graph capture; "
                "warm the exact specialization before capture"
            )
        compiled = _compile_nvfp4_mla_writer(
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            per_token_scale,
            has_rope,
        )
    _, args, _ = _nvfp4_mla_writer_launch(
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        per_token_scale,
        has_rope,
    )
    run_compiled(compiled, args)


@torch.library.custom_op(
    "b12x::concat_and_cache_nvfp4_mla_fp8_rope",
    mutates_args=("kv_cache",),
)
def _concat_and_cache_nvfp4_mla_fp8_rope_op(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    per_token_scale: bool = False,
) -> None:
    _concat_and_cache_nvfp4_mla_fp8_rope_flat_launch(
        kv_c, k_pe, kv_cache, slot_mapping, per_token_scale
    )


@torch.library.custom_op(
    "b12x::concat_and_cache_glm_next_nvfp4_mla",
    mutates_args=("kv_cache",),
)
def _concat_and_cache_glm_next_nvfp4_mla_op(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    _concat_and_cache_nvfp4_mla_fp8_rope_flat_launch(
        kv_c,
        kv_c,
        kv_cache,
        slot_mapping,
        per_token_scale=True,
        has_rope=False,
    )


@_concat_and_cache_glm_next_nvfp4_mla_op.register_fake
def _concat_and_cache_glm_next_nvfp4_mla_fake(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    return None


@_concat_and_cache_nvfp4_mla_fp8_rope_op.register_fake
def _concat_and_cache_nvfp4_mla_fp8_rope_fake(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    per_token_scale: bool = False,
) -> None:
    return None


def concat_and_cache_nvfp4_mla_fp8_rope(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    scale: torch.Tensor | None = None,
    per_token_scale: bool = False,
) -> None:
    """Write ``num_tokens`` KV_FP8_ROPE=1 nvfp4_ds_mla records.

    :param kv_c: MLA compressed latent, ``(>= num_tokens, 512)`` bf16/f16.
    :param k_pe: decoupled RoPE key, ``(>= num_tokens, 64)``, same dtype.
    :param kv_cache: paged cache viewed ``(num_blocks, block_size, 368)``
        uint8; mutated in place.
    :param slot_mapping: ``(num_tokens,)`` int64 flat slot ids; entries outside
        ``[0, num_blocks * block_size)`` are skipped.
    :param scale: accepted for signature parity with the fp8 cache-op
        family; the nvfp4_ds_mla record has an implicit global scale of 1.0
        (group scales carry all magnitude), so it is unused.
    :param per_token_scale: write inline-scale two-level records: the
        per-token second-level scale ``token_amax/(6*448)`` is stored fp32 at
        record bytes [292, 296) and group scales are encoded relative to it.
        Readers must run in the matching ``latent_scale_per_token`` mode.
    """
    del scale
    if kv_c.ndim != 2 or int(kv_c.shape[1]) != _KV_LORA_RANK:
        raise ValueError(
            f"kv_c must be (num_tokens, {_KV_LORA_RANK}), got {tuple(kv_c.shape)}"
        )
    if k_pe.ndim != 2 or int(k_pe.shape[1]) != _PE_DIM:
        raise ValueError(
            f"k_pe must be (num_tokens, {_PE_DIM}), got {tuple(k_pe.shape)}"
        )
    if kv_c.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"kv_c must be bf16/f16, got {kv_c.dtype}")
    if k_pe.dtype != kv_c.dtype:
        raise TypeError(f"k_pe dtype {k_pe.dtype} must match kv_c dtype {kv_c.dtype}")
    if kv_cache.ndim != 3 or int(kv_cache.shape[2]) != _RECORD_BYTES:
        raise ValueError(
            "kv_cache must be (num_blocks, block_size, "
            f"{_RECORD_BYTES}) uint8, got {tuple(kv_cache.shape)}"
        )
    if int(kv_cache.shape[0]) <= 0 or int(kv_cache.shape[1]) <= 0:
        raise ValueError("kv_cache num_blocks and block_size must be positive")
    if kv_cache.dtype != torch.uint8:
        raise TypeError(f"kv_cache must be uint8, got {kv_cache.dtype}")
    if slot_mapping.ndim != 1 or slot_mapping.dtype != torch.int64:
        raise TypeError(
            "slot_mapping must be a 1-D int64 tensor, got "
            f"{tuple(slot_mapping.shape)} {slot_mapping.dtype}"
        )
    if not slot_mapping.is_contiguous():
        raise ValueError("slot_mapping must be contiguous")
    num_tokens = int(slot_mapping.shape[0])
    if int(kv_c.shape[0]) < num_tokens or int(k_pe.shape[0]) < num_tokens:
        raise ValueError(
            f"kv_c/k_pe must cover slot_mapping's {num_tokens} tokens, got "
            f"{int(kv_c.shape[0])}/{int(k_pe.shape[0])} rows"
        )
    if kv_c.stride(1) != 1 or k_pe.stride(1) != 1 or kv_cache.stride(2) != 1:
        raise ValueError(
            "kv_c/k_pe rows and kv_cache records must be innermost-contiguous"
        )
    if kv_c.stride(0) % 2 != 0 or kv_c.data_ptr() % 4 != 0:
        # The group loads are 32-bit (element pairs), same as the CUDA writer.
        raise ValueError("kv_c rows must be 4-byte aligned (even row stride)")
    if k_pe.stride(0) % 2 != 0 or k_pe.data_ptr() % 4 != 0:
        raise ValueError("k_pe rows must be 4-byte aligned (even row stride)")
    if (
        kv_cache.data_ptr() % 16 != 0
        or kv_cache.stride(0) % 16 != 0
        or kv_cache.stride(1) % 16 != 0
    ):
        raise ValueError("kv_cache records must be 16-byte aligned")
    if int(kv_cache.shape[0]) * int(kv_cache.shape[1]) >= 2**63:
        raise ValueError("kv_cache slot capacity must fit in int64")
    if not (
        kv_c.is_cuda and k_pe.is_cuda and kv_cache.is_cuda and slot_mapping.is_cuda
    ):
        raise ValueError("all tensors must be on CUDA")
    if len({kv_c.device, k_pe.device, kv_cache.device, slot_mapping.device}) != 1:
        raise ValueError("all tensors must be on the same device")

    torch.ops.b12x.concat_and_cache_nvfp4_mla_fp8_rope(
        kv_c, k_pe, kv_cache, slot_mapping, per_token_scale
    )

def page_nbytes(
    page_size: int,
    *,
    cache_format: str = "deepseek_v4",
    cache_kind: str = "swa",
) -> int:
    """Size a compressed MLA byte page (V4.1 has no page-local tail).

    V4 retains its grouped-footer and existing padded page ABI. V4.1 stores
    contiguous token records: SWA is 512 E4M3 bytes + 16 UE8M0 bytes, indexed
    is 256 packed E2M1 bytes + 32 E4M3 scale bytes.
    """
    if not isinstance(page_size, int) or isinstance(page_size, bool) or page_size <= 0:
        raise ValueError("page_size must be a positive integer")
    if cache_kind not in ("swa", "indexed"):
        raise ValueError("cache_kind must be 'swa' or 'indexed'")
    if cache_format == "deepseek_v4":
        from .compressed_reference import compressed_sparse_mla_page_nbytes

        return compressed_sparse_mla_page_nbytes(page_size)
    if cache_format != "deepseek_v41":
        raise ValueError("cache_format must be 'deepseek_v4' or 'deepseek_v41'")
    return page_size * (528 if cache_kind == "swa" else 288)


@lru_cache(maxsize=None)
def _build_compressed_cache_writer(page_size: int, cache_kind: str):
    if cache_kind == "swa":
        return ConcatAndCacheGlmNextMlaKernel(page_size, dsv41=True)
    return ConcatAndCacheNvfp4MlaFp8RopeKernel(
        page_size, is_bf16=True, has_rope=False, dsv41=True
    )


def _validate_compressed_cache_writer(
    kv: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_size: int,
    cache_kind: str,
    cache_format: str,
) -> None:
    payload_bytes = page_nbytes(
        page_size, cache_format=cache_format, cache_kind=cache_kind
    )
    if cache_format != "deepseek_v41":
        raise ValueError("write_cache supports only cache_format='deepseek_v41'")
    if kv.ndim != 2 or int(kv.shape[1]) != 512:
        raise ValueError("kv must have shape [rows, 512]")
    if kv.dtype != torch.bfloat16:
        raise TypeError("kv must be BF16")
    if cache.ndim != 2 or int(cache.shape[1]) < payload_bytes:
        raise ValueError(f"cache must have shape [pages, >= {payload_bytes}]")
    if cache.dtype != torch.uint8:
        raise TypeError("cache must be uint8")
    if slot_mapping.ndim != 1 or slot_mapping.dtype not in (torch.int32, torch.int64):
        raise TypeError("slot_mapping must be a 1-D int32 or int64 tensor")
    if not slot_mapping.is_contiguous():
        raise ValueError("slot_mapping must be contiguous")
    rows = int(slot_mapping.shape[0])
    if int(kv.shape[0]) < rows:
        raise ValueError("kv must cover every slot_mapping row")
    if kv.stride(1) != 1 or kv.stride(0) % 2 or kv.data_ptr() % 4:
        raise ValueError("kv rows must be contiguous and 4-byte aligned")
    if (
        cache.stride(1) != 1
        or cache.stride(0) < int(cache.shape[1])
        or cache.stride(0) % 16
        or cache.data_ptr() % 16
    ):
        raise ValueError("cache requires contiguous byte rows and 16-byte aligned pages")
    if int(cache.shape[0]) <= 0:
        raise ValueError("cache must contain at least one page")
    if rows >= 2**31:
        raise ValueError("slot_mapping row count must fit in int32 launch geometry")
    if (
        int(cache.shape[0]) * page_size >= 2**63
        or int(cache.shape[0]) * int(cache.stride(0)) >= 2**63
    ):
        raise ValueError("cache capacity and byte offsets must fit in int64")
    if not (kv.is_cuda and cache.is_cuda and slot_mapping.is_cuda):
        raise ValueError("all tensors must be on CUDA")
    if len({kv.device, cache.device, slot_mapping.device}) != 1:
        raise ValueError("all tensors must be on the same device")




def _compressed_cache_writer_args(kv, cache, slot_mapping, page_size, cache_kind):
    kv_arg = _to_kernel_tensor(kv, assumed_align=4, leading_dim=1)
    cache_arg = _to_kernel_tensor(cache, assumed_align=16, leading_dim=1)
    slots_arg = _to_kernel_tensor(
        slot_mapping,
        assumed_align=8 if slot_mapping.dtype == torch.int64 else 4,
        leading_dim=0,
    )
    record_bytes = 528 if cache_kind == "swa" else 288
    tail_args = (
        Int64(int(cache.stride(0))),
        Int64(record_bytes),
        Int64(int(cache.shape[0]) * page_size),
        Int32(int(slot_mapping.shape[0])),
        current_cuda_stream(),
    )
    if cache_kind == "swa":
        args = (kv_arg, cache_arg, slots_arg, Int64(int(kv.stride(0))), *tail_args)
    else:
        args = (
            kv_arg,
            kv_arg,
            cache_arg,
            slots_arg,
            Int64(int(kv.stride(0))),
            Int64(int(kv.stride(0))),
            *tail_args,
        )
    return args


def _compressed_cache_writer_launch(kv, cache, slot_mapping, page_size, cache_kind):
    kernel = _build_compressed_cache_writer(page_size, cache_kind)
    args = _compressed_cache_writer_args(kv, cache, slot_mapping, page_size, cache_kind)
    spec = KernelCompileSpec.from_key(
        "attention.compressed_sparse_mla.kv_cache",
        1,
        (
            tensor_compile_fact("kv", kv, dynamic_dims=(0,), dynamic_strides=(0,)),
            tensor_compile_fact(
                "cache", cache, dynamic_dims=(0, 1), dynamic_strides=(0,)
            ),
            tensor_compile_fact("slot_mapping", slot_mapping, dynamic_dims=(0,)),
            page_size,
            cache_kind,
            "deepseek_v41",
        ),
        labels=("kv", "cache", "slot_mapping", "page_size", "cache_kind", "cache_format"),
    )
    return kernel, args, spec
