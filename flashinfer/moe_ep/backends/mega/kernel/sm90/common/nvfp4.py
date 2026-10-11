"""NVFP4 expert weights for the SM90 W4A16 mega backend (host-only).

Checkpoint contract (shared with the SM100 W4A16 backend): packed E2M1 bytes
``(E, N, K/2)`` (``uint8`` or ``float4_e2m1fn_x2``; the low nibble holds the
even-k element), linear E4M3 per-16 block scales ``(E, N, K/16)``, and an FP32
per-expert global scale applied in the epilogue (``fc*_alpha``):
``w = e2m1 * scale * alpha``.

The SM90 kernel decodes weights into its register-sourced WGMMA operand from
one TMA box per 128-K tile and row PAIR (:func:`augment_w4a16`): 144 bytes =
``[row 2p payload | row 2p+1 payload | row 2p scales | row 2p+1 scales]``
(64 + 64 + 8 + 8).  Each 64-byte payload is lane-permuted so the 16 bytes
one WGMMA fragment lane decodes per tile are contiguous (one 16-byte load);
payloads come first to keep those loads 16-byte aligned.  Inside each 32-bit
word the nibbles are regrouped so one shift + mask places both codes of a
BF16 pair.  Blocks are canonicalized first (exact): a zero scale zeroes its
codes and a negative scale moves its sign into the codes, which lets the
kernel convert scales without zero/sign handling.  Mirrors
``kernel_src/sm90/pull_style_cutedsl_megakernel/src/moe_hopper_fp8/
kernel_fp8_glu_fc12_swapab.py`` (``W4A16PairTileBytes``).

Host-only: must not import a kernel tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    import torch

NVFP4_BLOCK_K = 16
W4A16_TILE_K = 128
W4A16_PAYLOAD_BYTES = W4A16_TILE_K // 2
W4A16_SCALE_BYTES = W4A16_TILE_K // NVFP4_BLOCK_K
W4A16_PAIR_TILE_BYTES = 2 * (W4A16_PAYLOAD_BYTES + W4A16_SCALE_BYTES)
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Largest finite E4M3 x E2M1 product: the global scale maps amax onto it.
_NVFP4_RANGE = 448.0 * 6.0


def packed_bytes(weight: "torch.Tensor") -> "torch.Tensor":
    """``uint8`` view of packed E2M1 data (``uint8`` or ``float4_e2m1fn_x2``)."""
    import torch

    if weight.dtype == torch.uint8:
        return weight
    if weight.dtype == getattr(torch, "float4_e2m1fn_x2", None):
        return weight.view(torch.uint8)
    raise ValueError(
        f"packed NVFP4 weights must be uint8 or float4_e2m1fn_x2, got {weight.dtype}"
    )


def augment_w4a16(packed: "torch.Tensor", scales: "torch.Tensor") -> "torch.Tensor":
    """``(E, N, K/2)`` packed + ``(E, N, K/16)`` E4M3 bytes -> ``(E, N/2, K/128 * 144)`` uint8.

    Payload lane permutation: tile byte ``kb*8 + h*4 + l`` (k16 block ``kb``,
    k half ``h``, fragment lane ``l = k % 8 // 2``) moves to ``l*16 + kb*2 + h``.
    Then, per 32-bit word, the codes of byte ``b`` (even k, odd k) move to
    nibbles ``b`` and ``b + 4``.  Raises ``ValueError`` on a NaN block scale.
    """
    import torch

    num_experts, rows, half_k = packed.shape
    k_tiles = 2 * half_k // W4A16_TILE_K
    blocks = 2 * half_k // NVFP4_BLOCK_K
    scale_bytes = scales.view(torch.uint8)
    magnitude = scale_bytes & 0x7F
    if bool(torch.any(magnitude == 0x7F)):
        raise ValueError("NVFP4 block scales must not be NaN")
    # Canonicalize (value-preserving): zero-scale blocks get zero codes;
    # a negative scale flips every code's sign bit instead.
    sign_flip = torch.where(scale_bytes >= 0x80, 0x88, 0).to(torch.uint8)
    codes = packed.reshape(num_experts, rows, blocks, NVFP4_BLOCK_K // 2)
    codes = torch.where(
        (magnitude == 0)[..., None], 0, codes ^ sign_flip[..., None]
    ).to(torch.uint8)
    payload = (
        codes.reshape(num_experts, rows // 2, 2, k_tiles, 8, 2, 4)
        .permute(0, 1, 3, 2, 6, 4, 5)  # (E, pairs, kt, row, lane, kb, h)
        .reshape(-1, 4)
    )
    # Per word: nibbles (byte b, even/odd e) at 2b + e -> b + 4e.
    nibbles = torch.stack((payload & 0xF, payload >> 4), dim=-1)  # (.., b, e)
    nibbles = nibbles.transpose(-1, -2).reshape(-1, 4, 2)  # (.., e*4 + b) pairs
    payload = (nibbles[..., 0] | (nibbles[..., 1] << 4)).reshape(
        num_experts, rows // 2, k_tiles, 2 * W4A16_PAYLOAD_BYTES
    )
    scale = (
        magnitude.reshape(num_experts, rows // 2, 2, k_tiles, W4A16_SCALE_BYTES)
        .permute(0, 1, 3, 2, 4)
        .reshape(num_experts, rows // 2, k_tiles, 2 * W4A16_SCALE_BYTES)
    )
    return (
        torch.cat((payload, scale), dim=-1)
        .reshape(num_experts, rows // 2, k_tiles * W4A16_PAIR_TILE_BYTES)
        .contiguous()
    )


def dequantize_nvfp4(
    packed: "torch.Tensor",
    scales: "torch.Tensor",
    alpha: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Exact FP32 value of an NVFP4 ``(E, N, K/2)`` pack (``alpha``: ``(E,)``)."""
    import torch

    lut = torch.tensor(
        [*E2M1_VALUES, *(-v for v in E2M1_VALUES)],
        dtype=torch.float32,
        device=packed.device,
    )
    data = packed_bytes(packed)
    codes = torch.stack((data & 15, data >> 4), dim=-1).flatten(-2).long()
    values = lut[codes] * scales.view(torch.float8_e4m3fn).float().repeat_interleave(
        NVFP4_BLOCK_K, dim=-1
    )
    if alpha is not None:
        values = values * alpha.float().view(-1, 1, 1)
    return values


def quantize_nvfp4(
    weight: "torch.Tensor",
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """``(E, N, K)`` float -> (packed uint8, E4M3 scale bytes, FP32 alpha ``(E,)``).

    Per-expert global scale ``amax / (448 * 6)``; per-16 E4M3 scales of the
    globally scaled block amax / 6; nearest E2M1 code.  One expert at a time
    to bound the FP32 working set.
    """
    import torch

    num_experts, rows, k = weight.shape
    device = weight.device
    packed = torch.empty(num_experts, rows, k // 2, dtype=torch.uint8, device=device)
    scales = torch.empty(
        num_experts, rows, k // NVFP4_BLOCK_K, dtype=torch.uint8, device=device
    )
    alpha = torch.empty(num_experts, dtype=torch.float32, device=device)
    edges = torch.tensor(
        [(a + b) / 2 for a, b in zip(E2M1_VALUES, E2M1_VALUES[1:], strict=False)],
        device=device,
    )
    for expert in range(num_experts):
        w = weight[expert].float()
        expert_alpha = w.abs().amax().clamp_min(1e-30) / _NVFP4_RANGE
        blocks = (w / expert_alpha).reshape(rows, k // NVFP4_BLOCK_K, NVFP4_BLOCK_K)
        block_scale = (blocks.abs().amax(dim=-1) / 6.0).to(torch.float8_e4m3fn)
        mag = (blocks / block_scale.float().clamp_min(2.0**-9).unsqueeze(-1)).abs()
        codes = torch.bucketize(mag, edges).to(torch.uint8)
        codes = (codes | ((blocks < 0).to(torch.uint8) << 3)).reshape(rows, k)
        packed[expert] = codes[:, 0::2] | (codes[:, 1::2] << 4)
        scales[expert] = block_scale.view(torch.uint8)
        alpha[expert] = expert_alpha
    return packed, scales, alpha


__all__ = [
    "NVFP4_BLOCK_K",
    "W4A16_PAIR_TILE_BYTES",
    "W4A16_TILE_K",
    "augment_w4a16",
    "dequantize_nvfp4",
    "packed_bytes",
    "quantize_nvfp4",
]
