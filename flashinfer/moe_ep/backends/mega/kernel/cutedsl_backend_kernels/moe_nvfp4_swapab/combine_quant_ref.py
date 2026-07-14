"""Device-consistent host quantizers for the fc2/combine low-precision wire.

These reproduce the fused kernel's fc2-epilogue quantization (combine_dtype =
mxfp8 / nvfp4-PACK6) BIT-FOR-BIT, so that a host reference fed the byte-exact
per-topk fc2 output produces the exact same quantized planes the kernel emits --
the prerequisite for an element-wise bit-exact end-to-end combine reference.

Why a new module (the existing `topk_reduce.make_*_input` /
`runner_common.nvfp4_quantize_per_block_16` are NOT bit-exact): an adversarially
verified op-spec (workflow combine-quant-spec, 20/21 claims confirmed) found every
existing host quantizer diverges from the hardware cvt path.  Each step below cites
its claim id.  SOLVED-and-reused: the reciprocal (`_rcp_approx_ftz_f32_cuda` ==
`rcp.approx.ftz.f32`, claim RC-1) and the reduction (`faithful_*_reference_sum`,
claims RD-1/2).  NEW here: the scale path (bit-tricks, mul-by-constant, rcp.approx)
and the two HW cvt instructions, replicated via Triton inline-asm that mirrors the
exact kernel asm in `common/moe_utils.py` (cvt.rn.satfinite.{e4m3x2,e2m1x2}.f32).

Validated by byte-equality against the kernel's own combine_output_q / combine_sf_q
/ combine_global_q planes on GB200 (see scripts megamoe_combine_quant_*).
"""

from __future__ import annotations

import functools

import torch

from .runner_common import _rcp_approx_ftz_f32_cuda
from .topk_reduce import (
    FP8_E4M3FN_MAX,
    MXFP8_SCALE_BLOCK_SIZE,
    NVFP4_E2M1_MAX,
    NVFP4_GLOBAL_SCALE_BLOCK_SIZE,
    NVFP4_SFC_SCALE_BLOCK_SIZE,
)

_FP32_MAX = torch.finfo(torch.float32).max


# ---------------------------------------------------------------------------
# Triton replicas of the two HW cvt instructions (claims MX-4, NV-6, NV-9).
# Per-element form: cvt.rn.satfinite.<fmt>x2.f32 d, 0.0, $1 puts cvt(0.0)=0 in
# the HIGH lane and cvt($1) in the LOW lane, so masking the low byte/nibble of
# the result yields exactly the HW code for $1 (round-to-nearest-even, FTZ-off,
# saturating).  Packing into the kernel plane layout is done in torch afterward.
# ---------------------------------------------------------------------------
@functools.lru_cache(None)
def _get_cvt_e4m3_triton_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _cvt_e4m3_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        code = tl.inline_asm_elementwise(
            "{ .reg .b16 t;\n"
            "  cvt.rn.satfinite.e4m3x2.f32 t, 0f00000000, $1;\n"
            "  cvt.u32.u16 $0, t; }",
            "=r,f",
            [x],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        tl.store(y_ptr + offsets, (code & 0xFF).to(tl.uint8), mask=mask)

    return triton, _cvt_e4m3_kernel


@functools.lru_cache(None)
def _get_cvt_e2m1_triton_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _cvt_e2m1_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        nib = tl.inline_asm_elementwise(
            "{ .reg .b8 d; .reg .b16 t;\n"
            "  cvt.rn.satfinite.e2m1x2.f32 d, 0f00000000, $1;\n"
            "  mov.b16 t, {d, d};\n"
            "  cvt.u32.u16 $0, t; }",
            "=r,f",
            [x],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        tl.store(y_ptr + offsets, (nib & 0xF).to(tl.uint8), mask=mask)

    return triton, _cvt_e2m1_kernel


def _cvt_e4m3_codes_cuda(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> uint8 e4m3 codes, bit-exact to cvt.rn.satfinite.e4m3x2.f32."""
    if not x.is_cuda or x.dtype != torch.float32:
        raise ValueError(f"expects CUDA float32; got device={x.device} dtype={x.dtype}")
    xc = x.contiguous()
    out = torch.empty(xc.shape, dtype=torch.uint8, device=xc.device)
    n = xc.numel()
    if n == 0:
        return out
    triton, kernel = _get_cvt_e4m3_triton_kernel()
    block = 1024
    kernel[(triton.cdiv(n, block),)](xc, out, n, BLOCK=block)
    return out


def _cvt_e2m1_nibbles_cuda(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> uint8 e2m1 nibble codes (low 4 bits), bit-exact to HW cvt."""
    if not x.is_cuda or x.dtype != torch.float32:
        raise ValueError(f"expects CUDA float32; got device={x.device} dtype={x.dtype}")
    xc = x.contiguous()
    out = torch.empty(xc.shape, dtype=torch.uint8, device=xc.device)
    n = xc.numel()
    if n == 0:
        return out
    triton, kernel = _get_cvt_e2m1_triton_kernel()
    block = 1024
    kernel[(triton.cdiv(n, block),)](xc, out, n, BLOCK=block)
    return out


def _pack_fp4_pairs(nibbles: torch.Tensor) -> torch.Tensor:
    """Pack e2m1 nibbles along the last dim into bytes: lo=even idx, hi=odd."""
    lo = nibbles[..., 0::2]
    hi = nibbles[..., 1::2]
    return ((hi << 4) | lo).contiguous()


# ---------------------------------------------------------------------------
# MXFP8 combine quantizer (per-32 e8m0).  Claims MX-1..MX-4.
# ---------------------------------------------------------------------------
def _e8m0_round_up_byte(scale_f32: torch.Tensor) -> torch.Tensor:
    """Round scale UP to a power of two and return its UE8M0 byte (claim MX-2).

    byte = ieee_exp_field + (mantissa != 0 ? 1 : 0); equivalent to the kernel's
    integer bit-trick (epilogue_refactor.py:2262-2267), NOT log2/ceil/pow.
    """
    bits = scale_f32.contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    carry = (bits & 0x7FFFFF) != 0
    return (exp + carry.to(torch.int32)).to(torch.uint8)


def mxfp8_quantize_combine(src_fp32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize fp32 per-topk fc2 ``(T,K,H)`` to the kernel's MXFP8 combine planes.

    Returns ``(q_codes_uint8 (T,K,H), e8m0_byte_uint8 (T,K,H//block))``.  View
    q_codes as float8_e4m3fn and e8m0 as float8_e8m0fnu to match the kernel's
    combine_output_q / combine_sf_q.
    """
    if src_fp32.dtype != torch.float32 or not src_fp32.is_cuda:
        raise ValueError("src must be CUDA float32 (T,K,H)")
    T, K, H = src_fp32.shape
    block = MXFP8_SCALE_BLOCK_SIZE
    if H % block != 0:
        raise ValueError(f"H={H} not divisible by {block}")
    cols = H // block

    absmax = src_fp32.abs().reshape(T, K, cols, block).amax(dim=-1)  # (T,K,cols)
    # MX-1: scale = fmax(absmax * f32(1/448), 2^-30)  -- mul by constant, not divide.
    scale = torch.clamp_min(absmax * (1.0 / FP8_E4M3FN_MAX), 2.0**-30)
    # MX-2: e8m0 byte via integer bit-trick (round-up-to-pow2).
    e8m0_byte = _e8m0_round_up_byte(scale)  # (T,K,cols)
    # MX-3: exact pow2 reciprocal inv = bitcast((254 - byte) << 23); scaled = src*inv.
    inv = ((254 - e8m0_byte.to(torch.int32)) << 23).view(torch.float32)  # (T,K,cols)
    inv_H = inv.repeat_interleave(block, dim=-1)[:, :, :H]
    scaled = src_fp32 * inv_H
    # MX-4: fp32 -> fp8 e4m3 via HW cvt replica.
    q_codes = _cvt_e4m3_codes_cuda(scaled)  # (T,K,H) uint8
    return q_codes, e8m0_byte


# ---------------------------------------------------------------------------
# NVFP4 PACK6 combine quantizer.  Claims NV-3..NV-9, NV-11.
# ---------------------------------------------------------------------------
def nvfp4_pack6_quantize_combine(
    src_fp32: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize fp32 per-topk fc2 ``(T,K,H)`` to the kernel's NVFP4 PACK6 planes.

    Returns ``(q_fp4_bytes_uint8 (T,K,H//2), pack6_plane_uint8 (T,K,(H//32)*8))``.
    View q as float4_e2m1fn_x2 to match combine_output_q; pack6_plane matches
    combine_global_q (consumed via nvfp4_pack6_views).
    """
    if src_fp32.dtype != torch.float32 or not src_fp32.is_cuda:
        raise ValueError("src must be CUDA float32 (T,K,H)")
    T, K, H = src_fp32.shape
    sfc_blk = NVFP4_SFC_SCALE_BLOCK_SIZE  # 16
    gl_blk = NVFP4_GLOBAL_SCALE_BLOCK_SIZE  # 32
    if H % gl_blk != 0:
        raise ValueError(f"H={H} not divisible by {gl_blk}")
    n_tiles = H // gl_blk  # per-32 global tiles
    cols16 = H // sfc_blk  # per-16 sfc sub-blocks (== 2*n_tiles)
    sub_per_tile = gl_blk // sfc_blk  # 2

    absx = src_fp32.abs()
    amax32 = absx.reshape(T, K, n_tiles, gl_blk).amax(dim=-1)  # (T,K,n_tiles)
    amax16 = absx.reshape(T, K, cols16, sfc_blk).amax(dim=-1)  # (T,K,cols16)

    # NV-3: global = fmax(amax32 * f32(1/(6*448)), 2^-16)  -- mul by constant.
    global_f32 = torch.clamp_min(
        amax32 * (1.0 / (NVFP4_E2M1_MAX * FP8_E4M3FN_MAX)), 2.0**-16
    )  # (T,K,n_tiles)
    # NV-4: inv_global = rcp.approx.ftz(global).
    inv_global = _rcp_approx_ftz_f32_cuda(global_f32)
    # broadcast per-32 -> per-16 (each global tile covers sub_per_tile sfc blocks)
    inv_global_16 = inv_global.repeat_interleave(sub_per_tile, dim=-1)[:, :, :cols16]
    global_16 = global_f32.repeat_interleave(sub_per_tile, dim=-1)[:, :, :cols16]
    # NV-5: sfc = fmax(fmin(amax16 * f32(1/6) * inv_global, 448), 2^-16).
    sfc_f32 = amax16 * (1.0 / NVFP4_E2M1_MAX) * inv_global_16
    sfc_f32 = torch.clamp_min(torch.clamp_max(sfc_f32, FP8_E4M3FN_MAX), 2.0**-16)
    # NV-6: sfc -> e4m3 via HW cvt; read back to f32.
    sfc_codes = _cvt_e4m3_codes_cuda(sfc_f32)  # (T,K,cols16) uint8
    sfc_rt = sfc_codes.view(torch.float8_e4m3fn).float()  # readback
    # NV-7: acc_scale = fmin(rcp_approx(sfc_rt*global), F32_MAX) * fmin(sfc_rt*1e30, 1.0).
    prod = sfc_rt * global_16
    inv_prod = _rcp_approx_ftz_f32_cuda(prod.contiguous())
    underflow_mask = torch.clamp_max(sfc_rt * 1e30, 1.0)
    acc_scale = torch.clamp_max(inv_prod, _FP32_MAX) * underflow_mask  # (T,K,cols16)
    # NV-8: scaled = src * acc_scale (per-16 broadcast to H).
    acc_scale_H = acc_scale.repeat_interleave(sfc_blk, dim=-1)[:, :, :H]
    scaled = src_fp32 * acc_scale_H
    # NV-9: fp32 -> fp4 e2m1 via HW cvt; pack pairs (e0 -> low nibble).
    nibbles = _cvt_e2m1_nibbles_cuda(scaled)  # (T,K,H) uint8 nibble codes
    q_fp4 = _pack_fp4_pairs(nibbles)  # (T,K,H//2) uint8

    # NV-11: PACK6 plane (T,K,n_tiles*8): per tile g bytes 0..3 = global fp32 LE,
    # byte 4 = sfc sub-block 0, byte 5 = sfc sub-block 1, bytes 6,7 = pad = 0.
    plane = torch.zeros((T, K, n_tiles, 8), dtype=torch.uint8, device=src_fp32.device)
    plane[..., 0:4] = (
        global_f32.reshape(T, K, n_tiles, 1).contiguous().view(torch.uint8)
    )
    sfc_codes_2 = sfc_codes.reshape(T, K, n_tiles, sub_per_tile)
    plane[..., 4:6] = sfc_codes_2
    plane = plane.reshape(T, K, n_tiles * 8).contiguous()
    return q_fp4, plane
