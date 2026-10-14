"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch

from flashinfer.experimental.nvfp4_attention.cake_backend import (
    pack_nvfp4_attention_inputs,
    quantize_nvfp4_rows,
)
from flashinfer.prefill import prepare_nvfp4_attention

SM103 = (10, 3)


def _skip_unless_sm103():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != SM103:
        pytest.skip("SM103 required")


def _devices():
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


# --- Reference packing (the eager implementation the backend replaced) -------
# Kept verbatim as the bit-exact oracle of the packed operand layouts.


def _reference_quantize_nvfp4(x):
    x_f32 = x.float()
    groups = x_f32.shape[-1] // 16
    blocks = x_f32.reshape(*x_f32.shape[:-1], groups, 16)
    raw_scale = (blocks.abs().amax(dim=-1) / 6.0).clamp(
        min=2.0**-9, max=torch.finfo(torch.float8_e4m3fn).max
    )
    scale_fp8 = raw_scale.to(torch.float8_e4m3fn)
    scale = scale_fp8.float()
    normalized = blocks / scale.unsqueeze(-1)
    ax = normalized.abs().clamp_max(6.0)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    code = torch.bucketize(ax, boundaries).to(torch.uint8)
    code |= ((normalized < 0) & (code != 0)).to(torch.uint8) << 3
    code = code.reshape(*x_f32.shape)
    pairs = code.reshape(*code.shape[:-1], code.shape[-1] // 2, 2)
    packed = (pairs[..., 0] & 0x0F) | ((pairs[..., 1] & 0x0F) << 4)
    return packed.contiguous(), scale_fp8.view(torch.uint8).contiguous()


def _reference_prepack_sf_tiles(scale_u8):
    tiles = scale_u8.shape[0]
    sf = scale_u8.reshape(tiles, 4, 32, 1, 8)
    sf = sf.reshape(tiles, 4, 4, 8, 1, 2, 4)
    sf = sf.permute(0, 4, 2, 5, 3, 1, 6).contiguous()
    return sf.reshape(tiles * 32, 32)


def _reference_prepack_sf_tiles_split_ksets(scale_u8):
    tiles = scale_u8.shape[0]

    def pack_half(values):
        sf = values.contiguous().reshape(tiles, 4, 32, 1, 4)
        sf = sf.reshape(tiles, 4, 4, 8, 1, 1, 4)
        sf = sf.permute(0, 4, 2, 5, 3, 1, 6).contiguous()
        return sf.reshape(tiles, 512)

    lo = pack_half(scale_u8[..., :4]).reshape(tiles * 16, 32)
    hi = pack_half(scale_u8[..., 4:]).reshape(tiles * 16, 32)
    return lo, hi


def _reference_pack(q, k, v):
    batch, heads, seqlen, dim = q.shape
    bh = batch * heads
    sources = [x.reshape(bh, seqlen, dim).contiguous() for x in (q, k, v)]
    packed_q, sq = _reference_quantize_nvfp4(sources[0])
    packed_k, sk = _reference_quantize_nvfp4(sources[1])
    blocks = seqlen // 128
    sq = _reference_prepack_sf_tiles(sq.reshape(bh, blocks, 128, 8).reshape(-1, 128, 8))
    sk = _reference_prepack_sf_tiles(sk.reshape(bh, blocks, 128, 8).reshape(-1, 128, 8))
    packed_v, sv = _reference_quantize_nvfp4(sources[2].transpose(-1, -2).contiguous())
    scales = sv.reshape(bh, 128, blocks, 8).permute(0, 2, 1, 3).contiguous()
    lo, hi = _reference_prepack_sf_tiles_split_ksets(scales.reshape(-1, 128, 8))
    return dict(
        Q=packed_q.reshape(bh * seqlen, 64),
        K=packed_k.reshape(bh * seqlen, 64),
        Vt=packed_v.reshape(bh * 128, seqlen // 2),
        SFQ=sq,
        SFK=sk,
        SFVtLo=lo,
        SFVtHi=hi,
    )


def _inputs_with_edge_values(batch, heads, seqlen, device, seed=7):
    """Random BF16 inputs seeded with the values that exercise every code path:
    zeros, exact E2M1 thresholds times a scale, saturated magnitudes and
    infinities."""
    generator = torch.Generator().manual_seed(seed)
    q, k, v = (
        torch.randn((batch, heads, seqlen, 128), generator=generator).to(
            dtype=torch.bfloat16, device=device
        )
        for _ in range(3)
    )
    largest = torch.finfo(torch.bfloat16).max
    specials = torch.tensor(
        [
            0.0,
            -0.0,
            0.25,
            -0.25,
            0.75,
            -0.75,
            5.0,
            -5.0,
            6.0,
            -6.0,
            2688.0,
            -2688.0,
            4096.0,
            largest,
            -largest,
            float("inf"),
            -float("inf"),
        ],
        dtype=torch.bfloat16,
        device=device,
    )
    for x in (q, k, v):
        x[0, 0, : specials.numel(), 0] = specials
        x[0, 0, 0, : specials.numel()] = specials
        x[0, 0, :16, 1] = 0.0
    return q, k, v


def _assert_same_pack(actual, expected):
    assert set(actual) == set(expected)
    for name in expected:
        assert actual[name].dtype == torch.uint8, name
        assert actual[name].shape == expected[name].shape, name
        assert actual[name].is_contiguous(), name
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)


def _nhd_view(x):
    """``[B,H,S,128]`` view of a tensor stored as ``[B,S,H,128]``."""
    storage = x.transpose(1, 2).contiguous()
    view = storage.transpose(1, 2)
    assert not view.is_contiguous()
    return view


@pytest.mark.parametrize(
    "batch,heads,seqlen", [(4, 8, 4096), (1, 8, 32768), (8, 32, 8192)]
)
def test_nvfp4_attention(batch, heads, seqlen):
    _skip_unless_sm103()
    torch.manual_seed(42)
    q = torch.randn((batch, heads, seqlen, 128), dtype=torch.bfloat16, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    out = torch.empty_like(q)
    attention = prepare_nvfp4_attention(q, k, v, out, backend="cake")
    assert attention() is out
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(out, expected, atol=1.0, rtol=0.1)
    snapshot = out.clone()
    out.zero_()
    assert attention() is out
    torch.testing.assert_close(out, snapshot, atol=0, rtol=0)


def test_nvfp4_attention_accepts_strided_qkv():
    """``[B,S,H,128]`` storage viewed as ``[B,H,S,128]`` is quantized in place and
    produces the output of the contiguous copies bit for bit."""
    _skip_unless_sm103()
    torch.manual_seed(3)
    q = torch.randn((2, 4, 2048, 128), dtype=torch.bfloat16, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    expected = torch.empty_like(q)
    prepare_nvfp4_attention(q, k, v, expected, backend="cake")()
    out = torch.empty_like(q)
    attention = prepare_nvfp4_attention(
        _nhd_view(q), _nhd_view(k), _nhd_view(v), out, backend="cake"
    )
    assert attention() is out
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_nvfp4_attention_requires_contiguous_output():
    _skip_unless_sm103()
    q = torch.randn((1, 2, 512, 128), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="contiguous"):
        prepare_nvfp4_attention(q, q, q, _nhd_view(torch.empty_like(q)), backend="cake")


def test_nvfp4_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        prepare_nvfp4_attention(None, None, None, None, backend="unknown")


@pytest.mark.parametrize("device", _devices())
def test_nvfp4_packing_matches_reference_bitwise(device):
    q, k, v = _inputs_with_edge_values(1, 2, 1024, device)
    _assert_same_pack(pack_nvfp4_attention_inputs(q, k, v), _reference_pack(q, k, v))


@pytest.mark.parametrize("device", _devices())
def test_nvfp4_packing_is_stride_independent(device):
    q, k, v = _inputs_with_edge_values(2, 3, 512, device, seed=11)
    expected = pack_nvfp4_attention_inputs(q, k, v)
    strided = pack_nvfp4_attention_inputs(_nhd_view(q), _nhd_view(k), _nhd_view(v))
    _assert_same_pack(strided, expected)


@pytest.mark.parametrize("device", _devices())
def test_nvfp4_quantization_saturates_finite_scales(device):
    largest = torch.finfo(torch.bfloat16).max
    values = torch.tensor(
        [0.0, 1.0, 2688.0, 4096.0, -4096.0, largest, -largest],
        dtype=torch.bfloat16,
        device=device,
    )
    x = values[:, None].expand(-1, 16).contiguous()
    packed, scale_bytes = quantize_nvfp4_rows(x)
    scales = scale_bytes.view(torch.float8_e4m3fn).float()
    expected_scales = torch.tensor(
        [2.0**-9, 0.171875, 448.0, 448.0, 448.0, 448.0, 448.0], device=device
    )[:, None]
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)
    expected_packed = torch.tensor(
        [0x00, 0x77, 0x77, 0x77, 0xFF, 0x77, 0xFF],
        dtype=torch.uint8,
        device=device,
    )[:, None].expand(-1, 8)
    torch.testing.assert_close(packed, expected_packed, atol=0, rtol=0)
