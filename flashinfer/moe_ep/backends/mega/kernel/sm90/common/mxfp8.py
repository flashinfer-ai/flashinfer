"""MXFP8 expert weights -> SM90 FP8 BlockScale, converted once at preprocess.

SM90 has no block-scaled tensor-core MMA, so the SM90 FP8 mega backends cannot
consume MXFP8 (E4M3 payload, one E8M0 scale per 1x32 K block) natively.
:func:`mxfp8_to_fp8_block128` instead rewrites the weights once, at layer init,
into the DeepGEMM-style 128x128 fp32 block-scale layout both SM90 FP8 backends
already run; nothing is converted per forward.

The rewrite is exact except for underflow: MXFP8 dequantizes exactly to fp32,
and each 128x128 block is re-quantized with a power-of-two scale, so every
element becomes its original E4M3 payload shifted by a whole number of
binades.  Only elements shifted below E4M3's normal range (2**-6) lose mantissa
bits, which takes a wide E8M0 spread inside one block;
:func:`count_inexact_fp8_block128` reports how many did.

Host-only: must not import a kernel tree (the SM90 pull tree is
process-exclusive with the SM100 tree, and both SM90 backends use this).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    import torch

    from ......weights import PrequantizedMoEWeights

MXFP8_BLOCK_K = 32
FP8_BLOCK = 128
# E8M0 byte 0xFF encodes NaN (OCP MX spec); every other byte b is 2**(b - 127).
_E8M0_NAN = 0xFF


def e8m0_bytes(scale: "torch.Tensor") -> "torch.Tensor":
    """Raw E8M0 bytes of a ``float8_e8m0fnu`` or ``uint8`` scale tensor."""
    import torch

    if scale.dtype == torch.float8_e8m0fnu:
        return scale.view(torch.uint8)
    if scale.dtype == torch.uint8:
        return scale
    raise ValueError(
        f"E8M0 scales must be torch.float8_e8m0fnu or torch.uint8, got {scale.dtype}"
    )


def _e8m0_to_f32(scale_bytes: "torch.Tensor") -> "torch.Tensor":
    import torch

    # Build 2**(b - 127) from its fp32 bit pattern (exact; no exp2/pow
    # rounding).  b == 0 is 2**-127, an fp32 subnormal: mantissa bit 22.
    b = scale_bytes.to(torch.int32)
    bits = torch.where(b > 0, b << 23, torch.full_like(b, 1 << 22))
    return bits.view(torch.float32)


def dequantize_mxfp8(weight: "torch.Tensor", scale: "torch.Tensor") -> "torch.Tensor":
    """Exact fp32 value of an MXFP8 ``(..., N, K)`` payload.

    ``scale`` is ``(..., N, K // 32)`` E8M0 (``float8_e8m0fnu`` or ``uint8``).
    Every E4M3 value times a power of two is representable in fp32, so this
    is exact.
    """
    import torch

    sf = _e8m0_to_f32(e8m0_bytes(scale)).repeat_interleave(MXFP8_BLOCK_K, dim=-1)
    return weight.to(torch.float32) * sf


def quantize_fp8_block128_pow2(
    weight_nk: "torch.Tensor",
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """``(N, K)`` -> E4M3 ``(N, K)`` + fp32 ``(N/128, K/128)`` power-of-two scales.

    Same ``fp32 ~= payload * scale`` convention as the SM90 FP8 backends'
    128x128 quantizers, but each block's scale is the smallest power of two
    ``2**s`` with ``amax / 2**s <= 448``, so ``payload = w * 2**-s`` is a
    binade shift (exact unless it underflows E4M3) instead of a re-rounding.
    All-zero blocks get scale 1.
    """
    import torch

    n, k = weight_nk.shape
    if n % FP8_BLOCK or k % FP8_BLOCK:
        raise ValueError(
            f"weight ({n}, {k}) must be {FP8_BLOCK}-aligned for 128x128 block scales"
        )
    blocks = weight_nk.to(torch.float32).reshape(
        n // FP8_BLOCK, FP8_BLOCK, k // FP8_BLOCK, FP8_BLOCK
    )
    amax = blocks.abs().amax(dim=(1, 3))
    if not bool(torch.isfinite(amax).all()):
        # Includes MXFP8 weights whose E8M0 scale overflows fp32 on dequant.
        raise ValueError("weight has non-finite values; cannot block-quantize")
    # amax = mant * 2**exp with mant in [0.5, 1); 448 = 0.875 * 2**9.  Working
    # from frexp keeps the bound exact (no amax / 448 rounding).
    mant, exp = torch.frexp(amax)
    s = torch.where(mant <= 0.875, exp - 9, exp - 8)
    s = torch.where(amax > 0, s, torch.zeros_like(s))
    if bool((s > 126).any()):
        raise ValueError("weight magnitude exceeds the fp32 block-scale range")
    # Keep 2**s and 2**-s fp32-normal; clamping up only flushes blocks whose
    # amax is below ~2**-117 (far under any real weight) toward zero.
    s = s.clamp_min(-126)
    scale = ((s + 127) << 23).view(torch.float32)
    inv_scale = ((127 - s) << 23).view(torch.float32)
    payload = (blocks * inv_scale[:, None, :, None]).reshape(n, k)
    return payload.to(torch.float8_e4m3fn), scale


def mxfp8_to_fp8_block128(
    weight: "torch.Tensor", scale: "torch.Tensor"
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """MXFP8 ``(E, N, K)`` + E8M0 ``(E, N, K/32)`` -> E4M3 ``(E, N, K)`` + fp32 ``(E, N/128, K/128)``.

    Converts one expert at a time to bound the fp32 working set.
    """
    import torch

    num_experts, n, k = weight.shape
    payload = torch.empty(
        num_experts, n, k, dtype=torch.float8_e4m3fn, device=weight.device
    )
    block_scale = torch.empty(
        num_experts,
        n // FP8_BLOCK,
        k // FP8_BLOCK,
        dtype=torch.float32,
        device=weight.device,
    )
    for expert in range(num_experts):
        q, sf = quantize_fp8_block128_pow2(
            dequantize_mxfp8(weight[expert], scale[expert])
        )
        payload[expert].copy_(q)
        block_scale[expert].copy_(sf)
    return payload, block_scale


def count_inexact_fp8_block128(
    reference: "torch.Tensor", payload: "torch.Tensor", scale: "torch.Tensor"
) -> int:
    """Elements of ``reference`` (fp32 ``(..., N, K)``) that ``payload * scale`` misses."""
    import torch

    *lead, n, k = payload.shape
    blocks = payload.to(torch.float32).reshape(
        *lead, n // FP8_BLOCK, FP8_BLOCK, k // FP8_BLOCK, FP8_BLOCK
    )
    dequant = (blocks * scale[..., :, None, :, None]).reshape(*lead, n, k)
    return int((dequant != reference).sum())


def validate_mxfp8_pack(
    weights: "PrequantizedMoEWeights",
    *,
    intermediate_size: int,
    hidden_size: int,
    num_local_experts: int,
    kernel_name: str,
) -> None:
    """Raise ``MoEEpConfigError`` unless ``weights`` is a well-formed MXFP8 E4M3 pack.

    Expected: ``float8_e4m3fn`` ``w13`` ``(E, 2I, H)`` / ``w2`` ``(E, H, I)``
    with E8M0 scales ``(E, 2I, H/32)`` / ``(E, H, I/32)``, all on one device,
    and no E8M0 NaN bytes.
    """
    import torch

    from ......core.validation.common import MoEEpConfigError

    recipe = (
        f"{kernel_name} accepts PrequantizedMoEWeights only as MXFP8: "
        "float8_e4m3fn w13/w2 with E8M0 scales (torch.float8_e8m0fnu or "
        "torch.uint8), one per 1x32 K block"
    )
    if hidden_size % MXFP8_BLOCK_K or intermediate_size % MXFP8_BLOCK_K:
        raise MoEEpConfigError(
            f"{recipe}; hidden_size ({hidden_size}) and intermediate_size "
            f"({intermediate_size}) must be multiples of {MXFP8_BLOCK_K}"
        )
    legs = (
        ("w13", (num_local_experts, 2 * intermediate_size, hidden_size)),
        ("w2", (num_local_experts, hidden_size, intermediate_size)),
    )
    device = weights.w13.device
    for name, shape in legs:
        data = getattr(weights, name)
        scale = getattr(weights, f"{name}_scale")
        if data.dtype == torch.float8_e5m2:
            raise MoEEpConfigError(
                f"{recipe}; E5M2 MXFP8 is not supported ({name} is float8_e5m2)"
            )
        if data.dtype != torch.float8_e4m3fn:
            raise MoEEpConfigError(f"{recipe}; got {name}.dtype={data.dtype}")
        if scale.dtype not in (torch.float8_e8m0fnu, torch.uint8):
            raise MoEEpConfigError(f"{recipe}; got {name}_scale.dtype={scale.dtype}")
        scale_shape = (*shape[:-1], shape[-1] // MXFP8_BLOCK_K)
        if tuple(data.shape) != shape:
            raise MoEEpConfigError(
                f"{kernel_name} MXFP8 {name} must have shape {shape}, "
                f"got {tuple(data.shape)}"
            )
        if tuple(scale.shape) != scale_shape:
            raise MoEEpConfigError(
                f"{kernel_name} MXFP8 {name}_scale must have shape {scale_shape}, "
                f"got {tuple(scale.shape)}"
            )
        if data.device != device or scale.device != device:
            raise MoEEpConfigError(
                f"{kernel_name} MXFP8 w13/w2 and their scales must be on one device"
            )
        if bool((e8m0_bytes(scale) == _E8M0_NAN).any()):
            raise MoEEpConfigError(
                f"{kernel_name} MXFP8 {name}_scale contains the E8M0 NaN byte 0xFF"
            )


__all__ = [
    "FP8_BLOCK",
    "MXFP8_BLOCK_K",
    "count_inexact_fp8_block128",
    "dequantize_mxfp8",
    "e8m0_bytes",
    "mxfp8_to_fp8_block128",
    "quantize_fp8_block128_pow2",
    "validate_mxfp8_pack",
]
