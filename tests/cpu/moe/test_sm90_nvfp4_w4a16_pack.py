# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU checks of the SM90 NVFP4 W4A16 weight packing (no GPU needed).

The packed format is checked by unpacking it again (the inverse permutation
and the GF(2) inverse of the in-register decoder) and comparing the recovered
FP32 weights bit for bit with a direct dequantization of the checkpoint.
"""

import pytest
import torch

import flashinfer.fused_moe.sm90_nvfp4_w4a16 as w4a16
from flashinfer.jit.sm90_nvfp4_w4a16_moe import (
    _wgmma_rs_specializations,
    check_sm90_nvfp4_w4a16_moe_shape,
)

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _deq(fp4, sf_bytes, gs):
    c = torch.stack((fp4 & 15, fp4 >> 4), -1).reshape(fp4.shape[0], -1).long()
    v = _E2M1[c & 7]
    v = torch.where((c & 8).bool(), -v, v)
    return (
        v * sf_bytes.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1) * gs
    )


def _checkpoint(H, I, E, lo=0x28, hi=0x40, seed=0):
    g = torch.Generator().manual_seed(seed)
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), generator=g, dtype=torch.uint8)
    w2 = torch.randint(0, 256, (E, H, I // 2), generator=g, dtype=torch.uint8)
    s13 = torch.randint(lo, hi, (E, 2 * I, H // 16), generator=g, dtype=torch.uint8)
    s2 = torch.randint(lo, hi, (E, H, I // 16), generator=g, dtype=torch.uint8)
    gs13 = torch.rand((E, 2), generator=g) + 0.1
    gs2 = torch.rand((E,), generator=g) + 0.1
    return w13, s13, gs13, w2, s2, gs2


def test_decoder_tables_invert():
    enc, dec = w4a16._code_tables()
    for i in range(8):
        for b in (0, 1, 0x5A, 0xFF):
            pair = int(enc[i, b]) & (2**64 - 1)
            assert w4a16._decoded(pair) == b << (8 * i)
            assert int(dec[i, b]) & (2**64 - 1) == w4a16._decoded(b << (8 * i))


@pytest.mark.parametrize(
    "H,I,E",
    [(1024, 512, 2), (1536, 768, 1), (2048, 1024, 1), (1024, 1280, 1)],
)
def test_pack_roundtrip_is_exact(H, I, E):
    w13, s13, gs13, w2, s2, gs2 = _checkpoint(H, I, E, seed=H + I)
    s13[0, 3, 5] = 0  # a zero-scale block
    packed = w4a16.prepare_sm90_nvfp4_w4a16_weights(w13, s13, gs13, w2, s2, gs2)
    assert packed.w13.shape == (E, H // 256, 2 * I // 256, w4a16.WEIGHT_BLOCK_BYTES)
    assert packed.w2.shape == (E, I // 256, H // 256, w4a16.WEIGHT_BLOCK_BYTES)
    assert (packed.hidden_size, packed.intermediate_size, packed.num_experts) == (
        H,
        I,
        E,
    )
    for e in range(E):
        r13, r2 = w4a16.dequantize_sm90_nvfp4_w4a16_weights(packed, e)
        ref13 = torch.cat(
            [
                _deq(w13[e, :I], s13[e, :I], gs13[e, 0]),
                _deq(w13[e, I:], s13[e, I:], gs13[e, 1]),
            ]
        )
        assert torch.equal(r13, ref13)
        assert torch.equal(r2, _deq(w2[e], s2[e], gs2[e]))


def test_small_scales_are_shifted_losslessly():
    """Scales below 2.0 are representable after a power-of-two shift into the global scale."""
    H, I, E = 1024, 512, 1
    w13, s13, gs13, w2, s2, gs2 = _checkpoint(H, I, E, lo=0x08, hi=0x30)
    packed = w4a16.prepare_sm90_nvfp4_w4a16_weights(w13, s13, gs13[:, 0], w2, s2, gs2)
    r13, _ = w4a16.dequantize_sm90_nvfp4_w4a16_weights(packed, 0)
    assert torch.equal(r13, _deq(w13[0], s13[0], gs13[0, 0]))
    assert torch.equal(packed.w13_global_scale[0], packed.w13_global_scale[1])


def test_wide_scale_range_is_rejected():
    H, I, E = 1024, 512, 1
    w13, s13, gs13, w2, s2, gs2 = _checkpoint(H, I, E)
    s2[0, 0, 0] = 0x01  # 2^-9: more than 224x below the largest scale of the matrix
    with pytest.raises(ValueError, match="at least 2"):
        w4a16.prepare_sm90_nvfp4_w4a16_weights(w13, s13, gs13, w2, s2, gs2)


def _e4m3(*values):
    return torch.tensor(values, dtype=torch.float32).to(torch.float8_e4m3fn)


@pytest.mark.parametrize(
    "hi,lo,ok",
    [
        (448.0, 2.0, True),  # span 224, largest already at the top
        (
            256.0,
            1.25,
            False,
        ),  # span 204.8 < 224, but 1.25 stays below 2 (256 cannot double)
        (256.0, 2.5, True),  # span 102.4
        (3.5, 0.03125, True),  # span 112 after a shift by 2^7
        (3.5, 0.015625, True),  # span 224 after a shift by 2^7 (448 and 2)
        (240.0, 1.75, False),  # span 137: 240 cannot double, 1.75 stays below 2
    ],
)
def test_scale_span_condition(hi, lo, ok):
    """The documented rule: after the power-of-two shift that brings the largest block
    scale into (224, 448], every nonzero block scale must be at least 2."""
    sf = _e4m3(hi, lo, 0.0, hi).reshape(1, 4).view(torch.uint8)
    _, factor, bad = w4a16._normalize_scales(sf, 1)
    assert bad == (0 if ok else 1)
    top = float(hi * factor[0])
    assert 224.0 < top <= 448.0
    H, I, E = 1024, 512, 1
    w13, s13, gs13, w2, s2, gs2 = _checkpoint(H, I, E)
    s2[0] = _e4m3(hi).view(torch.uint8)
    s2[0, 0, 1] = _e4m3(lo).view(torch.uint8)[0]
    if ok:
        packed = w4a16.prepare_sm90_nvfp4_w4a16_weights(w13, s13, gs13, w2, s2, gs2)
        _, r2 = w4a16.dequantize_sm90_nvfp4_w4a16_weights(packed, 0)
        assert torch.equal(r2, _deq(w2[0], s2[0], gs2[0]))
    else:
        with pytest.raises(ValueError, match="at least 2"):
            w4a16.prepare_sm90_nvfp4_w4a16_weights(w13, s13, gs13, w2, s2, gs2)


def test_non_finite_scale_is_rejected():
    H, I, E = 1024, 512, 1
    w13, s13, gs13, w2, s2, gs2 = _checkpoint(H, I, E)
    s13[0, 0, 0] = 0x7F  # NaN
    with pytest.raises(ValueError, match="finite"):
        w4a16.prepare_sm90_nvfp4_w4a16_weights(w13, s13, gs13, w2, s2, gs2)


@pytest.mark.parametrize(
    "H,I,ok",
    [
        (4096, 4096, True),
        (7168, 2048, True),
        (2048, 768, True),
        (1024, 512, True),
        (8192, 8192, True),
        (2880, 2880, False),  # hidden % 512
        (4096, 1408, False),  # intermediate % 256
        (512, 512, False),  # hidden < 1024
        (4096, 256, False),  # intermediate < 512
        (16384, 4096, False),  # hidden > 8192
    ],
)
def test_shape_contract(H, I, ok):
    if ok:
        check_sm90_nvfp4_w4a16_moe_shape(H, I)
    else:
        with pytest.raises(ValueError):
            check_sm90_nvfp4_w4a16_moe_shape(H, I)


def test_pack_rejects_mismatched_tensors():
    w13, s13, gs13, w2, s2, gs2 = _checkpoint(1024, 512, 2)
    with pytest.raises(ValueError):
        w4a16.prepare_sm90_nvfp4_w4a16_weights(w13, s13[:, :, :-1], gs13, w2, s2, gs2)
    with pytest.raises(ValueError):
        w4a16.prepare_sm90_nvfp4_w4a16_weights(w13, s13, gs13, w2[:1], s2, gs2)
    with pytest.raises(ValueError):
        w4a16.prepare_sm90_nvfp4_w4a16_weights(w13, s13.float(), gs13, w2, s2, gs2)


def test_wgmma_specialization_operands():
    src = _wgmma_rs_specializations((24, 256))
    assert "wgmma_rs<24>" in src and "m64n24k16" in src
    # 12 accumulators, then 4 A registers and the descriptor
    assert "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11}, {%12,%13,%14,%15}, %16," in src
    assert '"+f"(d[127])' in src and "%132, 1, 1, 1, 0;" in src


@pytest.mark.parametrize(
    "shape,fraction,expected",
    [
        ((4096, 4096, 32, 8), 1.0, 256),  # 64 routes per expert
        ((4096, 4096, 8, 8), 1.0, 64),
        ((2048, 768, 32, 8), 1.0, 683),  # small matrices: work bound
        ((3072, 1024, 8, 2), 1.0, 1366),
        ((1024, 512, 4, 1), 1.0, 2049),  # capped at the default
        ((4096, 4096, 32, 8), 32 / 256, 2048),  # EP: 32 of 256 experts local
    ],
)
def test_auto_prefill_min_tokens(shape, fraction, expected):
    H, I, E, k = shape
    got = w4a16.sm90_nvfp4_w4a16_moe_prefill_min_tokens(H, I, E, k, fraction)
    assert got == expected


def test_auto_prefill_min_tokens_rejects_bad_fraction():
    for fraction in (0.0, 1.5):
        with pytest.raises(ValueError):
            w4a16.sm90_nvfp4_w4a16_moe_prefill_min_tokens(4096, 4096, 32, 8, fraction)


def test_routing_inputs_keep_out_of_range_int64_ids_skipped():
    """int64 ids are narrowed to int32 for the kernels; ids that would wrap into
    [0, E) (e.g. 2**32 -> 0) must stay outside it."""
    big = [2**32, 2**32 + 3, -(2**32), -(2**32) + 1, 2**31, -(2**31) - 1, 2**63 - 1]
    ids = torch.tensor([big + [5], [0, 1, 2, 3, 31, 32, -1, 7]], dtype=torch.int64)
    w = torch.rand(ids.shape)
    out_ids, out_w = w4a16._routing_inputs(ids, w)
    assert out_ids.dtype == torch.int32 and out_ids.is_contiguous()
    assert torch.equal(out_w, w)
    valid = (ids >= 0) & (ids < w4a16.MAX_LOCAL_EXPERTS)
    assert torch.equal(out_ids[valid], ids[valid].int())
    assert bool(((out_ids[~valid] < 0) | (out_ids[~valid] >= 32)).all())
    # top-k < 8 is padded with skipped slots
    out_ids, out_w = w4a16._routing_inputs(ids[:, :3], w[:, :3])
    assert out_ids.shape == (2, 8) and bool((out_ids[:, 3:] == -1).all())
    assert bool(((out_ids[0, :3] < 0) | (out_ids[0, :3] >= 32)).all())
    assert bool((out_w[:, 3:] == 0).all())
