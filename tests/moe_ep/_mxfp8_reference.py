"""Pure-torch MXFP8 (OCP MX, E4M3 + E8M0 per 1x32) reference quantizer for tests.

Independent of every kernel tree and of ``flashinfer.quantization`` so the
SM90 MXFP8 tests can build checkpoint-style packs anywhere.
"""

from __future__ import annotations

import torch

MXFP8_BLOCK_K = 32
E4M3_MAX = 448.0


def _pow2_f64(biased: torch.Tensor) -> torch.Tensor:
    """Exact float64 ``2**(biased - 127)`` for E8M0 bytes (built from bits)."""
    return ((biased.to(torch.int64) + (1023 - 127)) << 52).view(torch.float64)


def mxfp8_quantize_ref(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(..., K)`` -> E4M3 ``(..., K)`` + uint8 E8M0 ``(..., K // 32)``.

    OCP MX recipe: ``shared_exp = floor(log2(amax)) - 8`` (8 = E4M3 emax),
    payload ``x / 2**shared_exp`` saturated to +-448.  All-zero blocks get
    the unit scale (byte 127).
    """
    *lead, k = weight.shape
    assert k % MXFP8_BLOCK_K == 0, k
    blocks = weight.to(torch.float32).reshape(*lead, k // MXFP8_BLOCK_K, MXFP8_BLOCK_K)
    amax = blocks.abs().amax(dim=-1)
    _mant, exp = torch.frexp(amax)  # floor(log2(amax)) == exp - 1
    biased = torch.where(amax > 0, exp - 1 - 8 + 127, torch.full_like(exp, 127))
    biased = biased.clamp(0, 254)
    scale = _pow2_f64(biased).to(torch.float32)
    payload = (blocks / scale[..., None]).clamp(-E4M3_MAX, E4M3_MAX)
    return (
        payload.reshape(*lead, k).to(torch.float8_e4m3fn),
        biased.to(torch.uint8),
    )


def mxfp8_dequantize_ref(
    payload: torch.Tensor, scale_bytes: torch.Tensor
) -> torch.Tensor:
    """float64 value of an MXFP8 payload (exact; independent of the code under test)."""
    sf = _pow2_f64(scale_bytes)
    return payload.to(torch.float64) * sf.repeat_interleave(MXFP8_BLOCK_K, dim=-1)
