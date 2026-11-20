"""DeepGEMM FP8 quantization-parity gate.

This is the byte-exact comparison-against-DeepGEMM that the SM120 dense-FP8 port
never had: the prior work *asserted* "activations are stored at 1x32 ue8m0,
identical to DeepGEMM's 1d1d" but never measured it, and the GEMM correctness
gate compared b12x against a reference that consumed b12x's *own* quantized FP8,
so any divergence from DeepGEMM was invisible. These tests pin the contract:

  1. ACTIVATION quant is byte-for-byte identical to DeepGEMM `per_token_cast_to_fp8`
     (gran_k=32): same FP8 values AND same UE8M0 scales.
  2. UE8M0 rounding via `ceil(log2)/exp2` equals DeepGEMM's bit-exact
     `ceil_to_ue8m0`, including at exact powers of two.
  3. WEIGHT packing re-quantizes arbitrary-fp32 checkpoint block scales
     (DeepSeek `weight_scale_inv`) onto exact UE8M0 -- matching DeepGEMM's
     `per_block_cast_to_fp8` accuracy -- instead of rounding the scale and
     leaving the FP8 values stale (~2.7x worse).
  4. Already-UE8M0 scales keep their FP8 values verbatim (no spurious re-quant).
"""

from __future__ import annotations

import pytest
import torch

from b12x.gemm._shared.block_fp8 import (
    pack_block_fp8_linear_weight_mxfp8,
    quantize_block_fp8_linear_input_mxfp8,
)
from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch

from b12x.testing.reference.helpers import require_b12x


# --- DeepGEMM reference (verbatim from deepgemm-other/deep_gemm/utils/math.py) ---
def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def _align(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def _per_token_cast_to_fp8(x: torch.Tensor, gran_k: int):
    m, n = x.shape
    padded_n = _align(n, gran_k)
    xp = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    xp[:, :n] = x
    xv = xp.view(m, padded_n // gran_k, gran_k)
    amax = xv.abs().float().amax(dim=2).view(m, padded_n // gran_k).clamp(1e-4)
    sf = _ceil_to_ue8m0(amax / 448.0)
    fp8 = (
        (xv * (1.0 / sf.unsqueeze(2)))
        .to(torch.float8_e4m3fn)
        .view(m, padded_n)[:, :n]
        .contiguous()
    )
    return fp8, sf  # sf: fp32 power-of-two, [m, n//gran_k]


def _sf_fp32_to_e8m0_u8(sf: torch.Tensor) -> torch.Tensor:
    return (torch.log2(sf).round().clamp(-127, 127) + 127).to(torch.uint8)


def _relfro(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).flatten().float().norm() / b.flatten().float().norm()).item()


def test_activation_quant_byte_exact_with_deepgemm() -> None:
    """b12x activation MXFP8 quant == DeepGEMM per_token_cast_to_fp8 (gran_k=32)."""
    require_b12x()
    torch.manual_seed(0)
    M, K = 64, 5376  # Nemotron down-proj K (= 42 * 128)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 3.0

    rows = quantize_block_fp8_linear_input_mxfp8(x)
    b_vals = rows.values.view(torch.uint8)
    b_sf = rows.scale_rows.view(torch.uint8)[0]  # [M, K//32]

    dg_vals, dg_sf_f = _per_token_cast_to_fp8(x, gran_k=32)
    dg_vals = dg_vals.view(torch.uint8)
    dg_sf = _sf_fp32_to_e8m0_u8(dg_sf_f)

    # b12x quantizes at 32-element K groups -> one UE8M0 scale per 32.
    assert b_sf.shape[1] == K // 32, f"expected gran_k=32 scale grid, got {b_sf.shape}"
    torch.testing.assert_close(b_vals, dg_vals, rtol=0, atol=0)
    torch.testing.assert_close(b_sf, dg_sf, rtol=0, atol=0)


def test_ue8m0_rounding_matches_bit_exact_ceil() -> None:
    """ceil(log2(x))/exp2 == bit-exact ceil_to_ue8m0, incl. exact powers of two."""
    require_b12x()
    dev = "cuda"
    ks = torch.arange(-30, 30, device=dev, dtype=torch.float32)
    pow2 = 448.0 * torch.exp2(ks)
    dense = torch.logspace(-12, 12, 100000, device=dev, dtype=torch.float32)
    amax = torch.cat([pow2, pow2 * (1 - 1e-6), pow2 * (1 + 1e-6), dense])

    bit_exact = _ceil_to_ue8m0(amax / 448.0)
    safe = amax / 448.0
    formula = torch.exp2(torch.clamp(torch.ceil(torch.log2(safe)), -127.0, 127.0))

    be_exp = torch.log2(bit_exact).round().clamp(-127, 127)
    fm_exp = torch.log2(formula).round().clamp(-127, 127)
    assert (be_exp == fm_exp).all(), (
        "UE8M0 rounding diverges from bit-exact ceil_to_ue8m0"
    )


def _deepseek_checkpoint(N: int, K: int, dev: str):
    """(w_fp8, s_fp32 [N/128,K/128], w_orig) — arbitrary fp32 128x128 block scale."""
    w_orig = (torch.randn(N, K, device=dev) * 0.2).to(torch.bfloat16).float()
    wv = w_orig.view(N // 128, 128, K // 128, 128)
    amax = wv.abs().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    s = amax / 448.0  # arbitrary fp32, NOT a power of two
    w_fp8 = (wv / s).to(torch.float8_e4m3fn).view(N, K)
    return w_fp8, s.view(N // 128, K // 128).to(torch.float32), w_orig


def test_weight_pack_requantizes_arbitrary_fp32_scales_to_parity() -> None:
    """Arbitrary fp32 checkpoint scales are re-quantized onto UE8M0 (DeepGEMM parity).

    The fix must (a) reach near the irreducible e4m3 floor and (b) be strictly
    better than the old "round the scale, keep stale FP8 values" behaviour.
    """
    require_b12x()
    torch.manual_seed(0)
    N, K = 4096, 4096
    w_fp8, s_fp32, w_orig = _deepseek_checkpoint(N, K, "cuda")

    packed = pack_block_fp8_linear_weight_mxfp8(w_fp8, s_fp32)
    deq = dequantize_mxfp8_rows_torch(packed.weight.values, packed.weight.scale_rows)

    # Old (buggy) approach: keep w_fp8, round the fp32 scale to nearest power of two.
    s_round = torch.exp2(torch.round(torch.log2(s_fp32)).clamp(-127, 127))
    s_round_e = s_round.repeat_interleave(128, 0).repeat_interleave(128, 1)
    deq_old = w_fp8.float() * s_round_e

    err_new = _relfro(deq, w_orig)
    err_old = _relfro(deq_old, w_orig)
    assert err_new < 0.05, (
        f"weight quant error {err_new:.4f} not at parity (expected < 0.05)"
    )
    assert err_new < err_old * 0.6, (
        f"re-quant ({err_new:.4f}) must be much better than round-keep ({err_old:.4f})"
    )


def test_weight_pack_keeps_ue8m0_values_verbatim() -> None:
    """Already-UE8M0 scales must NOT trigger re-quant: FP8 values stay byte-identical."""
    require_b12x()
    torch.manual_seed(0)
    N, K = 2048, 2048
    w_orig = (torch.randn(N, K, device="cuda") * 0.2).to(torch.bfloat16).float()
    wv = w_orig.view(N // 128, 128, K // 128, 128)
    amax = wv.abs().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    exp = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    w_fp8 = (wv / torch.exp2(exp)).to(torch.float8_e4m3fn).view(N, K)
    s_e8m0 = (
        (exp.view(N // 128, K // 128) + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    )

    packed = pack_block_fp8_linear_weight_mxfp8(w_fp8, s_e8m0)
    torch.testing.assert_close(
        packed.weight.values.view(torch.uint8), w_fp8.view(torch.uint8), rtol=0, atol=0
    )


@pytest.mark.parametrize("tokens", (1, 9))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_v41_activation_floor_matches_reference(tokens, dtype) -> None:
    """Zero/tiny K32 groups and exact UE8M0 boundaries use source quantization."""
    require_b12x()
    peaks = torch.tensor(
        [0.0, 1e-8, 5e-5, 1e-4, 2e-4, 0.875, 1.75, 3.5],
        device="cuda",
        dtype=dtype,
    )
    pattern = torch.linspace(-1, 1, 32, device="cuda", dtype=torch.float32)
    source = (peaks.float()[:, None] * pattern).to(dtype).reshape(1, 256)
    source = source.repeat(tokens, 1)
    expected_values, expected_scales = _per_token_cast_to_fp8(source, 32)
    actual = quantize_block_fp8_linear_input_mxfp8(
        source,
        block_size=(32, 32),
    )
    torch.testing.assert_close(
        actual.values.view(torch.uint8),
        expected_values.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual.scale_rows.view(torch.uint8)[0],
        _sf_fp32_to_e8m0_u8(expected_scales),
        rtol=0,
        atol=0,
    )
    # The older recipe deliberately retains its unit scale for a zero group.
    legacy = quantize_block_fp8_linear_input_mxfp8(source)
    assert torch.all(legacy.scale_rows.view(torch.uint8)[0, :, 0] == 127)
