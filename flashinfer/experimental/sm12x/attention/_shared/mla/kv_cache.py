# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/mla/kv_cache.py @ 345dd0a6 (2026-07-18) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""nvfp4_ds_mla KV_FP8_ROPE=1 record writer (SM120).

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
"""

from __future__ import annotations

from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from flashinfer.experimental.sm12x._lib.compiler import (
    KernelCompileSpec,
    launch as sm12x_launch,
    tensor_compile_fact,
)
from flashinfer.experimental.sm12x._lib.intrinsics import (
    cvt_e4m3_to_f32_via_f16,
    cvt_f32_to_e4m3,
    cvt_f32x4_to_e4m3x4,
    f16x2_to_f32x2,
    fabs_f32,
    fmax_f32,
    get_ptr_as_int64,
    max_abs_16,
    quantize_and_pack_16_fast,
    rcp_approx_ftz,
    st_global_f32,
    st_global_u64,
    st_global_u8,
)
from flashinfer.experimental.sm12x._lib.utils import current_cuda_stream

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


class ConcatAndCacheNvfp4MlaFp8RopeKernel:
    """Per-token KV_FP8_ROPE=1 nvfp4_ds_mla record writer.

    Thread mapping (128 threads/CTA): threads 0-31 quantize one 16-dim
    group each (eight coherent 32-bit loads -> exact f32 promote -> E2M1
    pack + E4M3 scale byte); threads 0-11 zero the [292, 304) pad; thread
    32 quantizes the RoPE lane to E4M3 with one fp32 per-token scale.
    """

    def __init__(self, block_size: int, is_bf16: bool):
        self.block_size = int(block_size)
        self.is_bf16 = bool(is_bf16)

    @cute.jit
    def __call__(
        self,
        kv_c: cute.Tensor,  # (num_tokens, 512) bf16/f16
        k_pe: cute.Tensor,  # (num_tokens, 64) bf16/f16
        kv_cache: cute.Tensor,  # (num_blocks, block_size, 368) u8
        slot_mapping: cute.Tensor,  # (num_tokens, 1) int64
        kv_c_stride: Int32,  # kv_c.stride(0), elements
        k_pe_stride: Int32,  # k_pe.stride(0), elements
        block_stride: Int64,  # kv_cache.stride(0), bytes
        entry_stride: Int32,  # kv_cache.stride(1), bytes
        slot_capacity: Int32,
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
        kv_c_stride: Int32,
        k_pe_stride: Int32,
        block_stride: Int64,
        entry_stride: Int32,
        slot_capacity: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        token_idx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        token = Int32(token_idx)

        slot = Int64(slot_mapping[token])
        if (slot >= Int64(0)) & (slot < slot_capacity.to(Int64)):
            # Capacity is host-asserted in (0, 2^31), so the block/offset
            # split is safe in Int32 after the Int64 bounds check.
            slot32 = slot.to(Int32)
            block_idx = slot32 // Int32(self.block_size)
            block_off = slot32 % Int32(self.block_size)
            dst = (
                get_ptr_as_int64(kv_cache, 0)
                + block_idx.to(Int64) * block_stride
                + (block_off * entry_stride).to(Int64)
            )

            # --- NoPE: one 16-dim group per thread -> 8 B E2M1 + 1 scale byte.
            if tid < Int32(_NUM_GROUPS):
                src_elem = token * kv_c_stride + tid * Int32(_GROUP_SIZE)
                vals = cute.make_rmem_tensor((_GROUP_SIZE,), Float32)
                for i in cutlass.range_constexpr(_GROUP_SIZE // 2):
                    pair = _ld_global_u32(
                        get_ptr_as_int64(kv_c, src_elem + Int32(2 * i))
                    )
                    if cutlass.const_expr(self.is_bf16):
                        f0, f1 = _bf16x2_to_f32x2(pair)
                    else:
                        f0, f1 = f16x2_to_f32x2(pair)
                    vals[2 * i] = f0
                    vals[2 * i + 1] = f1

                # NVFP4 block quant at global scale 1.0: scale byte =
                # e4m3(amax/6); values scaled by rcp.approx.ftz of the
                # hardware-exact decode of that byte (what the reader
                # multiplies back), then satfinite E2M1.
                group_amax = max_abs_16(vals)
                scale_f32 = group_amax * rcp_approx_ftz(Float32(6.0))
                scale_u32 = cvt_f32_to_e4m3(scale_f32)
                decoded_scale = cvt_e4m3_to_f32_via_f16(scale_u32)
                packed64 = Uint64(0)
                if decoded_scale != Float32(0.0):
                    packed64 = quantize_and_pack_16_fast(
                        vals, rcp_approx_ftz(decoded_scale)
                    )
                st_global_u64(dst + (tid * Int32(8)).to(Int64), packed64)
                st_global_u8(
                    dst + Int64(_NOPE_BYTES) + tid.to(Int64),
                    cutlass.Uint8(scale_u32 & Uint32(0xFF)),
                )

            # --- Zero pad [292, 304).
            if tid < Int32(_PAD_BYTES):
                st_global_u8(
                    dst + Int64(_PAD_OFFSET) + tid.to(Int64),
                    cutlass.Uint8(0),
                )

            # --- RoPE lane: amax -> fp32 scale at [288, 292) -> satfinite
            # E4M3 bytes at [304, 368).
            if tid == Int32(_NUM_GROUPS):
                rope_vals = cute.make_rmem_tensor((_PE_DIM,), Float32)
                for i in cutlass.range_constexpr(_PE_DIM // 2):
                    pair = _ld_global_u32(
                        get_ptr_as_int64(k_pe, token * k_pe_stride + Int32(2 * i))
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


@lru_cache(maxsize=None)
def _build_concat_and_cache_nvfp4_mla_fp8_rope_kernel(
    block_size: int, is_bf16: bool
) -> ConcatAndCacheNvfp4MlaFp8RopeKernel:
    return ConcatAndCacheNvfp4MlaFp8RopeKernel(block_size, is_bf16)


def clear_nvfp4_mla_fp8_rope_kv_cache_kernel_cache() -> None:
    _build_concat_and_cache_nvfp4_mla_fp8_rope_kernel.cache_clear()


def _torch_to_cutlass_dtype(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.uint8:
        return cutlass.Uint8
    if dtype == torch.int64:
        return cutlass.Int64
    raise TypeError(f"unsupported dtype {dtype}")


def _to_kernel_tensor(
    tensor: torch.Tensor,
    *,
    assumed_align: int,
    leading_dim: int,
) -> cute.Tensor:
    cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
    cute_tensor.element_type = _torch_to_cutlass_dtype(tensor.dtype)
    return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)


def _concat_and_cache_nvfp4_mla_fp8_rope_flat_launch(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    num_tokens = int(slot_mapping.shape[0])
    if num_tokens == 0:
        return
    block_size = int(kv_cache.shape[1])
    slot_capacity = int(kv_cache.shape[0]) * block_size
    is_bf16 = kv_c.dtype == torch.bfloat16
    kernel = _build_concat_and_cache_nvfp4_mla_fp8_rope_kernel(block_size, is_bf16)

    args = (
        _to_kernel_tensor(kv_c, assumed_align=4, leading_dim=1),
        _to_kernel_tensor(k_pe, assumed_align=4, leading_dim=1),
        _to_kernel_tensor(kv_cache, assumed_align=16, leading_dim=2),
        _to_kernel_tensor(slot_mapping, assumed_align=8, leading_dim=0),
        Int32(int(kv_c.stride(0))),
        Int32(int(k_pe.stride(0))),
        Int64(int(kv_cache.stride(0))),
        Int32(int(kv_cache.stride(1))),
        Int32(slot_capacity),
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
    )
    spec = KernelCompileSpec.from_key(
        "attention.mla.nvfp4_fp8_rope_kv_cache",
        2,
        cache_key,
        labels=(
            "kv_c",
            "k_pe",
            "kv_cache",
            "slot_mapping",
            "kv_dtype",
            "block_size",
        ),
    )
    sm12x_launch(
        kernel,
        compile_spec=spec,
        compile_args=args,
        runtime_args=args,
    )


@torch.library.custom_op(
    "flashinfer_sm12x::concat_and_cache_nvfp4_mla_fp8_rope",
    mutates_args=("kv_cache",),
)
def _concat_and_cache_nvfp4_mla_fp8_rope_op(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    _concat_and_cache_nvfp4_mla_fp8_rope_flat_launch(kv_c, k_pe, kv_cache, slot_mapping)


@_concat_and_cache_nvfp4_mla_fp8_rope_op.register_fake
def _concat_and_cache_nvfp4_mla_fp8_rope_fake(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    return None


def concat_and_cache_nvfp4_mla_fp8_rope(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    scale: torch.Tensor | None = None,
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
    if int(kv_cache.shape[0]) * int(kv_cache.shape[1]) >= 2**31:
        raise ValueError("kv_cache slot capacity must fit in int32")
    if not (
        kv_c.is_cuda and k_pe.is_cuda and kv_cache.is_cuda and slot_mapping.is_cuda
    ):
        raise ValueError("all tensors must be on CUDA")
    if len({kv_c.device, k_pe.device, kv_cache.device, slot_mapping.device}) != 1:
        raise ValueError("all tensors must be on the same device")

    torch.ops.flashinfer_sm12x.concat_and_cache_nvfp4_mla_fp8_rope(
        kv_c, k_pe, kv_cache, slot_mapping
    )
