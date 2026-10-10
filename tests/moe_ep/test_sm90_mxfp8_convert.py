"""Host-side checks for the SM90 MXFP8 -> FP8 BlockScale weight conversion.

Covers ``backends/mega/kernel/sm90/common/mxfp8.py`` only.  No kernel tree is
imported, so this runs in the ``unit`` target on any machine (CPU, plus CUDA
when available).  Kernel-level MXFP8 coverage lives in the per-backend SM90
tests (``test_sm90_pull_fp8_kernel_vs_reference.py``,
``test_sm90_push_fp8_backend.py``).
"""

from __future__ import annotations

import pytest
import torch

from flashinfer.moe_ep import MoEWeightPack
from flashinfer.moe_ep.backends.mega.kernel.sm90.common import mxfp8 as conv
from flashinfer.moe_ep.core.validation.common import MoEEpConfigError

from ._mxfp8_reference import mxfp8_dequantize_ref, mxfp8_quantize_ref

# Half of E4M3's subnormal step (2**-9): the worst-case rounding error, on the
# block-scaled grid, of an element shifted into the subnormal range.
_E4M3_HALF_SUBNORMAL_STEP = 2.0**-10

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _random_mxfp8(shape, *, device, seed, spread=0):
    """Checkpoint-style MXFP8 of randn weights; ``spread`` adds random binades per 1x32 block."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(*shape, generator=g)
    if spread:
        *lead, k = shape
        shift = torch.randint(
            0, spread + 1, (*lead, k // conv.MXFP8_BLOCK_K), generator=g
        )
        w = w * torch.exp2(-shift.float()).repeat_interleave(conv.MXFP8_BLOCK_K, -1)
    payload, scale = mxfp8_quantize_ref(w)
    return payload.to(device), scale.to(device)


@pytest.mark.parametrize("device", _DEVICES)
def test_dequantize_mxfp8_is_exact(device):
    payload, scale = _random_mxfp8((256, 512), device=device, seed=0)
    # Exercise the E8M0 extremes too: byte 0 is 2**-127 (an fp32 subnormal);
    # 2**119 is the largest scale whose 448 * 2**k product stays fp32-finite.
    scale[0, 0] = 0
    scale[1, 0] = 127 + 119
    got = conv.dequantize_mxfp8(payload, scale)
    assert got.dtype == torch.float32
    want = mxfp8_dequantize_ref(payload, scale)
    torch.testing.assert_close(got.to(torch.float64), want, atol=0, rtol=0)
    # float8_e8m0fnu scales read the same bytes.
    got_e8m0 = conv.dequantize_mxfp8(payload, scale.view(torch.float8_e8m0fnu))
    assert torch.equal(got, got_e8m0)


@pytest.mark.parametrize("device", _DEVICES)
def test_pow2_block_scales_are_minimal_powers_of_two(device):
    g = torch.Generator(device="cpu").manual_seed(1)
    w = (torch.randn(256, 384, generator=g) * 3.0).to(device)
    # Pin one block's amax exactly on the 448 * 2**k boundary.
    w[:128, :128] = 0.25
    w[0, 0] = 448.0 * 2.0**-3
    q, sf = conv.quantize_fp8_block128_pow2(w)
    assert q.dtype == torch.float8_e4m3fn and sf.dtype == torch.float32
    assert sf.shape == (2, 3)
    mant, _ = torch.frexp(sf)
    assert torch.all(mant == 0.5), "block scales must be powers of two"
    amax = w.reshape(2, 128, 3, 128).abs().amax(dim=(1, 3))
    assert torch.all(amax / sf <= 448.0)
    # Minimal: half the scale would overflow E4M3.
    assert torch.all(amax / (sf / 2) > 448.0)
    assert sf[0, 0].item() == 2.0**-3
    assert torch.isfinite(q.to(torch.float32)).all()


@pytest.mark.parametrize("device", _DEVICES)
def test_pow2_zero_block_gets_unit_scale(device):
    w = torch.zeros(128, 256, device=device)
    w[:, 128:] = 1.0
    q, sf = conv.quantize_fp8_block128_pow2(w)
    assert sf[0, 0].item() == 1.0
    assert torch.all(q[:, :128].to(torch.float32) == 0)


@pytest.mark.parametrize("device", _DEVICES)
def test_mxfp8_to_fp8_block128_is_lossless_for_checkpoint_weights(device):
    """randn MXFP8 weights convert bit-exactly (E8M0 spread stays tiny)."""
    payload, scale = _random_mxfp8((3, 256, 512), device=device, seed=2)
    q, sf = conv.mxfp8_to_fp8_block128(payload, scale)
    assert q.shape == (3, 256, 512) and sf.shape == (3, 2, 4)
    reference = conv.dequantize_mxfp8(payload, scale)
    inexact = conv.count_inexact_fp8_block128(reference, q, sf)
    # An element is only inexact if a binade shift pushes it below E4M3's
    # normal range; with randn weights that is a handful per million at most.
    assert inexact <= reference.numel() * 1e-4, inexact


@pytest.mark.parametrize("device", _DEVICES)
def test_mxfp8_to_fp8_block128_error_bound_with_wide_scale_spread(device):
    """Wide E8M0 spread inside a block: only underflow rounding, bounded by half a subnormal step."""
    payload, scale = _random_mxfp8((2, 128, 256), device=device, seed=3, spread=12)
    q, sf = conv.mxfp8_to_fp8_block128(payload, scale)
    reference = mxfp8_dequantize_ref(payload, scale)
    dequant = (
        q.to(torch.float64).reshape(2, 1, 128, 2, 128)
        * sf.to(torch.float64)[:, :, None, :, None]
    ).reshape(2, 128, 256)
    err = (dequant - reference).abs()
    bound = _E4M3_HALF_SUBNORMAL_STEP * sf.to(torch.float64).repeat_interleave(
        128, 1
    ).repeat_interleave(128, 2)
    assert torch.all(err <= bound)
    inexact = conv.count_inexact_fp8_block128(reference.to(torch.float32), q, sf)
    assert inexact == int((err > 0).sum())
    assert inexact > 0, "spread=12 should underflow some elements"


def test_scale_overflowing_fp32_raises_instead_of_nan():
    payload = torch.full((128, 128), 448.0).to(torch.float8_e4m3fn)
    scale = torch.full((128, 4), 254, dtype=torch.uint8)  # 2**127 * 448 -> inf
    with pytest.raises(ValueError, match="non-finite"):
        conv.mxfp8_to_fp8_block128(payload[None], scale[None])


def test_quantize_requires_128_alignment():
    with pytest.raises(ValueError, match="128-aligned"):
        conv.quantize_fp8_block128_pow2(torch.ones(128, 96))


def _pack(*, num_experts=2, inter=128, hidden=256, device="cpu"):
    w13, w13_sf = _random_mxfp8((num_experts, 2 * inter, hidden), device=device, seed=4)
    w2, w2_sf = _random_mxfp8((num_experts, hidden, inter), device=device, seed=5)
    return dict(w13=w13, w2=w2, w13_scale=w13_sf, w2_scale=w2_sf)


def _validate(fields, *, num_experts=2, inter=128, hidden=256):
    conv.validate_mxfp8_pack(
        MoEWeightPack(**fields),
        intermediate_size=inter,
        hidden_size=hidden,
        num_local_experts=num_experts,
        kernel_name="sm90_test",
    )


def test_validate_accepts_uint8_and_e8m0_scales():
    fields = _pack()
    _validate(fields)
    fields["w13_scale"] = fields["w13_scale"].view(torch.float8_e8m0fnu)
    _validate(fields)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda f: f.update(w13=f["w13"].float().to(torch.float8_e5m2)), "E5M2"),
        (lambda f: f.update(w2=f["w2"].to(torch.bfloat16)), "w2.dtype"),
        (lambda f: f.update(w2_scale=f["w2_scale"].float()), "w2_scale.dtype"),
        (lambda f: f.update(w13=f["w13"][:, :128]), "w13 must have shape"),
        (lambda f: f.update(w13_scale=f["w13_scale"][..., :4]), "w13_scale must"),
        (lambda f: f["w2_scale"].__setitem__((0, 0, 0), 0xFF), "NaN"),
    ],
    ids=["e5m2", "bad_dtype", "bad_scale_dtype", "bad_shape", "bad_scale_shape", "nan"],
)
def test_validate_rejects(mutate, match):
    fields = _pack()
    mutate(fields)
    with pytest.raises(MoEEpConfigError, match=match):
        _validate(fields)


def test_validate_rejects_unaligned_sizes():
    with pytest.raises(MoEEpConfigError, match="multiples of 32"):
        _validate(_pack(), hidden=240)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_validate_rejects_mixed_devices():
    fields = _pack()
    fields["w2"] = fields["w2"].cuda()
    with pytest.raises(MoEEpConfigError, match="one device"):
        _validate(fields)
