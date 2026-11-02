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

Codec tests of the compressed expert weights (WCX) of ``sm90_bf16_bf16_bf16_push_cake``.

CPU only.  The chunk code is checked against an independent scalar definition
(escape rule and decode), on special values (signed zeros, inf / NaN payloads,
subnormals, the largest finite values), on the exponent-offset window edges,
exhaustively over every bf16 pattern next to every partner exponent class,
and on the stage layout; the torch encoder must produce the bytes of the
numpy reference encoder.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flashinfer.moe_ep.kernel_src.sm90.cake_bf16_megamoe.shim import cake_wcx as wcx

RNG_SEED = 1360_27


def _bf(x) -> np.ndarray:
    """float32 -> bf16 bit patterns (round to nearest even); NaN -> 0x7FC0."""
    x = np.asarray(x, np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    r[np.isnan(x)] = 0x7FC0
    return r


def _scalar_escape(chunk) -> bool:
    es = [(int(v) >> 7) & 0xFF for v in chunk]
    top = max(es)
    if top == 255:
        return True
    return any(
        (int(v) & 0x7FFF) != 0 and top - e > wcx.MAX_OFF
        for v, e in zip(chunk, es, strict=True)
    )


def _scalar_decode(data, hdr, tail) -> list[int]:
    if hdr == wcx.ESC:
        raw = bytes(data[:48]) + bytes(tail)
        return [int.from_bytes(raw[2 * i : 2 * i + 2], "little") for i in range(32)]
    out = []
    for i in range(32):
        smb = int(data[i])
        nib = (int(data[32 + i // 2]) >> (4 * (i % 2))) & 0xF
        sign = (smb >> 7) & 1
        if nib == wcx.ZERO_CODE:
            out.append(sign << 15)
        else:
            out.append((sign << 15) | ((hdr - nib) << 7) | (smb & 0x7F))
    return out


def _no_escape(shape, seed: int) -> np.ndarray:
    """Random bf16 patterns whose exponents lie in [118, 122] (no zeros): no chunk can escape."""
    rng = np.random.default_rng(seed)
    sign = rng.integers(0, 2, shape).astype(np.uint16) << 15
    exp = rng.integers(118, 123, shape).astype(np.uint16) << 7
    return sign | exp | rng.integers(0, 128, shape).astype(np.uint16)


def _roundtrip(c: np.ndarray) -> np.ndarray:
    data, hdr, tails, esc = wcx._encode_chunks(c)
    assert np.array_equal(wcx._decode_chunks(data, hdr, tails), c)
    return esc


def test_gaussian_chunks_roundtrip():
    rng = np.random.default_rng(RNG_SEED)
    g = _bf(rng.standard_normal((4096, 32)).astype(np.float32) * 0.02)
    esc = _roundtrip(g)
    assert esc.mean() < 0.01


SPECIALS = {
    "+0": 0x0000,
    "-0": 0x8000,
    "+inf": 0x7F80,
    "-inf": 0xFF80,
    "qnan": 0x7FC0,
    "nan_payload": 0x7F81,
    "-nan": 0xFFC1,
    "max_finite": 0x7F7F,
    "-max_finite": 0xFF7F,
    "min_normal": 0x0080,
    "-min_normal": 0x8080,
    "min_sub": 0x0001,
    "max_sub": 0x007F,
    "-sub": 0x8055,
    "one": 0x3F80,
}


@pytest.mark.parametrize("partner_exp", [0, 1, 14, 15, 16, 127, 254])
@pytest.mark.parametrize("name", list(SPECIALS))
def test_special_values(name, partner_exp):
    rng = np.random.default_rng(RNG_SEED + partner_exp)
    c = np.zeros((1, 32), np.uint16)
    c[0, 0] = SPECIALS[name]
    c[0, 1] = (partner_exp << 7) | 0x2A
    c[0, 2] = 0x8000
    c[0, 3:] = (
        (rng.integers(0, 2, 29) << 15).astype(np.uint16)
        | (partner_exp << 7)
        | rng.integers(0, 128, 29).astype(np.uint16)
    )
    data, hdr, tails, esc = wcx._encode_chunks(c)
    assert np.array_equal(wcx._decode_chunks(data, hdr, tails), c)
    assert bool(esc[0]) == _scalar_escape(c[0])
    assert _scalar_decode(data[0], int(hdr[0]), tails[0] if esc[0] else b"") == [
        int(x) for x in c[0]
    ]


@pytest.mark.parametrize("name", ["+inf", "-inf", "qnan", "nan_payload", "-nan"])
def test_inf_nan_always_escape(name):
    c = np.full((1, 32), 0x3F80, np.uint16)
    c[0, 5] = SPECIALS[name]
    assert bool(wcx._encode_chunks(c)[3][0])
    _roundtrip(c)


@pytest.mark.parametrize(
    "off,want",
    [(0, False), (13, False), (14, False), (15, True), (16, True), (100, True)],
)
def test_offset_window_edges(off, want):
    c = np.full((1, 32), (120 << 7) | 1, np.uint16)
    c[0, 7] = ((120 - off) << 7) | 0x33 if 120 - off > 0 else 0x0033
    assert bool(wcx._encode_chunks(c)[3][0]) == want
    _roundtrip(c)


def test_zero_code_with_max_exponent():
    c = np.full((1, 32), (254 << 7) | 0x7F, np.uint16)
    c[0, ::2] = 0x8000
    assert not wcx._encode_chunks(c)[3][0]
    _roundtrip(c)


def test_all_zero_chunks():
    c = np.zeros((3, 32), np.uint16)
    c[1] = 0x8000
    c[2, ::3] = 0x8000
    assert not _roundtrip(c).any()


def test_min_max_exponent_chunks():
    rng = np.random.default_rng(RNG_SEED)
    c = np.empty((2, 32), np.uint16)
    c[0] = (1 << 7) | rng.integers(0, 128, 32).astype(np.uint16)
    c[1] = (254 << 7) | rng.integers(0, 128, 32).astype(np.uint16)
    c[:, ::2] |= 0x8000
    assert not _roundtrip(c).any()


def test_subnormal_only_chunks():
    rng = np.random.default_rng(RNG_SEED)
    c = rng.integers(0, 128, (4, 32)).astype(np.uint16)
    c[:, 1::2] |= 0x8000
    assert not _roundtrip(c).any()


@pytest.mark.parametrize("partner_exp", [0, 1, 7, 14, 15, 60, 127, 128, 200, 253, 254])
def test_exhaustive_patterns(partner_exp):
    allv = np.arange(65536, dtype=np.uint16)
    c = np.empty((65536, 32), np.uint16)
    c[:] = (partner_exp << 7) | 0x11
    c[:, 9] = allv
    esc = _roundtrip(c)
    ev = (allv.astype(np.int32) >> 7) & 0xFF
    top = np.maximum(ev, partner_exp)
    zero = (allv & 0x7FFF) == 0
    want = (
        (top == 255)
        | ((~zero) & (top - ev > wcx.MAX_OFF))
        | ((top - partner_exp) > wcx.MAX_OFF)
    )
    assert np.array_equal(esc, want)


def test_scalar_crosscheck():
    rng = np.random.default_rng(RNG_SEED)
    mix = rng.integers(0, 65536, (2000, 32)).astype(np.uint16)
    mix[:500] = _bf(
        rng.standard_normal((500, 32)).astype(np.float32)
        * 10.0 ** rng.integers(-30, 30, (500, 1))
    )
    mix[500:600, ::4] = 0
    data, hdr, tails, esc = wcx._encode_chunks(mix)
    ti = 0
    for i in range(mix.shape[0]):
        assert _scalar_escape(mix[i]) == bool(esc[i])
        tail = tails[ti] if esc[i] else b""
        ti += int(esc[i])
        assert _scalar_decode(data[i], int(hdr[i]), tail) == [int(x) for x in mix[i]]


def test_chunk_bit_layout():
    c = np.zeros((1, 32), np.uint16)
    c[0, 0] = 0x3F80  # 1.0
    c[0, 1] = 0xBFC0  # -1.5
    c[0, 2] = 0x8000  # -0
    c[0, 3] = 0x3E80  # 0.25
    data, hdr, _, _ = wcx._encode_chunks(c)
    assert int(hdr[0]) == 127
    assert list(data[0, :4]) == [0x00, 0xC0, 0x80, 0x00]
    assert int(data[0, 32]) == 0x00 and int(data[0, 33]) == (15 | (2 << 4))
    assert data.shape[1] == 48


def test_stage_constants():
    assert wcx.CHUNKS_PER_STAGE == 512
    assert wcx.SLOT_OFFSET == 25088
    assert wcx.wcx_stage_bytes(0) == 25088
    assert wcx.wcx_stage_bytes(0) / 32768 == 0.765625
    assert wcx.WCX_S_CAP == 192
    assert wcx.wcx_stage_bytes(wcx.WCX_S_CAP) == wcx.WCX_SLOT_STRIDE
    assert wcx.WCX_SLOT_STRIDE % 128 == 0
    assert wcx.wcx_stage_bytes(9) % 16 == 0
    with pytest.raises(ValueError):
        wcx.wcx_stage_bytes(-1)


@pytest.mark.parametrize(
    "shape", [(1, 512, 256), (2, 256, 7168 // 4), (3, 256, 64), (2, 768, 512)]
)
def test_stage_roundtrip(shape):
    rng = np.random.default_rng(RNG_SEED + sum(shape))
    e, n, k = shape
    w = _bf(rng.standard_normal(shape).astype(np.float32) * k**-0.5)
    buf, st = wcx.wcx_encode_reference(w)
    assert buf.shape == (e, n // 256, k // 64, wcx.wcx_stage_bytes(st.slots))
    assert np.array_equal(wcx.wcx_decode_reference(buf, n, k), w)
    assert st.compressed_bytes == buf.size and st.bf16_bytes == 2 * w.size


def test_stage_layout_positions():
    """The chunk of stage (ns, ks) holding w[n, 32 h : 32 h + 32] of the tile is chunk 2 n + h."""
    w = _no_escape((1, 512, 128), RNG_SEED)
    buf, st = wcx.wcx_encode_reference(w)
    assert st.escaped_chunks == 0
    for ns, ks, n, h in ((0, 0, 0, 0), (1, 1, 255, 1), (0, 1, 17, 0), (1, 0, 200, 1)):
        stage = buf[0, ns, ks]
        c = 2 * n + h
        chunk = w[0, ns * 256 + n, ks * 64 + 32 * h : ks * 64 + 32 * h + 32][None]
        data, hdr, _, _ = wcx._encode_chunks(chunk)
        assert np.array_equal(stage[48 * c : 48 * c + 48], data[0])
        assert stage[wcx.HEADER_OFFSET + c] == hdr[0]


def test_all_escape_stage():
    w = np.full((1, 256, 64), 0x7FC0, np.uint16)
    buf, st = wcx.wcx_encode_reference(w)
    assert st.slots == 512 and st.escaped_chunks == 512
    assert np.array_equal(wcx.wcx_decode_reference(buf, 256, 64), w)
    with pytest.raises(ValueError):
        wcx.wcx_encode_reference(w, slots=4)


def test_fixed_slots_and_escape_slot_order():
    rng = np.random.default_rng(RNG_SEED)
    w = _bf(rng.standard_normal((1, 512, 128)).astype(np.float32))
    w[0, 3, 5] = 0x7F80
    w[0, 300, 70] = 0x0001
    w[0, 3, 40] = 0xFFC1
    _, st0 = wcx.wcx_encode_reference(w)
    buf, st = wcx.wcx_encode_reference(w, slots=st0.slots + 2)
    assert st.slots == st0.slots + 2
    assert np.array_equal(wcx.wcx_decode_reference(buf, 512, 128), w)
    # stage (0, 0): chunks 6 (row 3, k 0..31) and 7 (row 3, k 32..63) escape -> slots 0 and 1 hold their tails
    stage = buf[0, 0, 0]
    assert (
        stage[wcx.HEADER_OFFSET + 6] == wcx.ESC
        and stage[wcx.HEADER_OFFSET + 7] == wcx.ESC
    )
    raw6 = np.ascontiguousarray(w[0, 3, 0:32].astype("<u2")).view(np.uint8)
    raw7 = np.ascontiguousarray(w[0, 3, 32:64].astype("<u2")).view(np.uint8)
    assert np.array_equal(stage[wcx.SLOT_OFFSET : wcx.SLOT_OFFSET + 16], raw6[48:])
    assert np.array_equal(stage[wcx.SLOT_OFFSET + 16 : wcx.SLOT_OFFSET + 32], raw7[48:])
    assert np.array_equal(stage[48 * 6 : 48 * 7], raw6[:48])


def test_slots_are_tensor_wide_max_over_experts():
    w = _no_escape((3, 256, 128), RNG_SEED)
    w[1, 0:7, 0] = 0x7F80  # 7 escapes in stage (0, 0) of expert 1
    w[2, 0:3, 64] = 0x0001  # 3 escapes in stage (0, 1) of expert 2
    buf, st = wcx.wcx_encode_reference(w)
    assert st.slots == 7
    assert buf.shape[-1] == wcx.wcx_stage_bytes(7)
    assert np.array_equal(wcx.wcx_decode_reference(buf, 256, 128), w)


def test_rejects_unaligned_shapes():
    with pytest.raises(ValueError):
        wcx.wcx_encode_reference(np.zeros((1, 300, 64), np.uint16))
    with pytest.raises(ValueError):
        wcx.wcx_encode(torch.zeros(1, 256, 100, dtype=torch.bfloat16))


def _torch_bf16(bits: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16)


TORCH_CASES = [
    "gauss",
    "specials",
    "escape_heavy",
    "all_escape",
    "zeros",
    "random_bits",
]


def _torch_case(name: str) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED + TORCH_CASES.index(name))
    if name == "gauss":
        return _bf(rng.standard_normal((2, 512, 256)).astype(np.float32) * 256**-0.5)
    if name == "specials":
        w = _bf(rng.standard_normal((2, 256, 128)).astype(np.float32))
        vals = np.array(list(SPECIALS.values()), np.uint16)
        idx = rng.integers(0, w.size, 200)
        w.reshape(-1)[idx] = vals[rng.integers(0, vals.size, idx.size)]
        return w
    if name == "escape_heavy":
        w = _bf(rng.standard_normal((2, 256, 256)).astype(np.float32))
        mask = rng.random(w.shape) < 0.01
        w[mask] = _bf(
            rng.standard_normal(int(mask.sum())).astype(np.float32) * 2.0**-30
        )
        return w
    if name == "all_escape":
        return np.full((1, 256, 64), 0xFFC1, np.uint16)
    if name == "zeros":
        w = np.zeros((1, 256, 128), np.uint16)
        w[0, ::3, ::5] = 0x8000
        return w
    return rng.integers(0, 65536, (1, 256, 128)).astype(np.uint16)


@pytest.mark.parametrize("name", TORCH_CASES)
def test_torch_encoder_matches_reference(name):
    w = _torch_case(name)
    ref, st_ref = wcx.wcx_encode_reference(w)
    got, st = wcx.wcx_encode(_torch_bf16(w))
    assert st == st_ref
    assert got.dtype == torch.uint8 and tuple(got.shape) == ref.shape
    assert np.array_equal(got.numpy(), ref)


def test_torch_encoder_fixed_slots_and_overflow():
    w = _torch_case("escape_heavy")
    _, st = wcx.wcx_encode_reference(w)
    ref, _ = wcx.wcx_encode_reference(w, slots=st.slots + 5)
    got, st2 = wcx.wcx_encode(_torch_bf16(w), slots=st.slots + 5)
    assert st2.slots == st.slots + 5 and np.array_equal(got.numpy(), ref)
    with pytest.raises(ValueError):
        wcx.wcx_encode(_torch_bf16(w), slots=st.slots - 1)


def test_escape_counts_per_stage():
    w = _torch_case("escape_heavy")
    _, st = wcx.wcx_encode_reference(w)
    cnt = torch.stack(
        [wcx.wcx_stage_escape_counts(_torch_bf16(w[e])) for e in range(w.shape[0])]
    )
    assert int(cnt.max()) == st.slots and int(cnt.sum()) == st.escaped_chunks


@pytest.mark.parametrize(
    "pos",
    [0, 33, 47, wcx.HEADER_OFFSET + 1, wcx.SLOT_OFFSET + 5],
    ids=["smb", "nibble", "nibble_last", "header", "escape_slot"],
)
def test_one_byte_flip_changes_decode(pos):
    w = _no_escape((1, 256, 64), RNG_SEED)
    w[0, 10, 3] = 0x7F80  # chunk 20 escapes -> escape slot 0
    buf, st = wcx.wcx_encode_reference(w)
    assert st.slots == 1
    bad = buf.copy()
    bad.reshape(-1)[pos] ^= 0x01
    assert not np.array_equal(wcx.wcx_decode_reference(bad, 256, 64), w)


# ================================================================ codec version 2 (swap-AB register decode)
def _v2_scalar_escape(chunk) -> bool:
    has_zero = any((int(v) & 0x7FFF) == 0 for v in chunk)
    return _scalar_escape(chunk) or has_zero


def test_v2_stage_constants():
    assert wcx.WCX2_SLOT_OFF == 25152
    assert wcx.wcx2_stage_bytes(0) == 25152
    assert wcx.wcx2_stage_bytes(wcx.WCX2_S_CAP) <= wcx.WCX_SLOT_STRIDE
    assert wcx.WCX2_SLOT_OFF + 16 * (wcx.WCX2_S_CAP + 1) > wcx.WCX_SLOT_STRIDE
    with pytest.raises(ValueError):
        wcx.wcx2_stage_bytes(wcx.WCX2_S_CAP + 1)
    assert wcx.WCX2_S_CAP == 188


def test_v2_chunk_map_is_a_permutation():
    pairs = set(zip(wcx._V2_ROW.tolist(), wcx._V2_HK.tolist(), strict=True))
    assert len(pairs) == 512 and {r for r, _ in pairs} == set(range(256))


@pytest.mark.parametrize("partner_exp", [0, 1, 7, 14, 15, 16, 60, 127, 254])
def test_v2_escape_rule_matches_scalar(partner_exp):
    vals = np.array(list(SPECIALS.values()), np.uint16)
    rows = []
    for v in vals:
        for z in (0, 1):
            c = np.full(32, partner_exp << 7, np.uint16)
            c[3] = v
            if z:
                c[9] = 0x8000
            rows.append(c)
    c = np.stack(rows)
    esc = wcx._analyze_v2(c)[3]
    assert esc.tolist() == [_v2_scalar_escape(x) for x in c]


def test_v2_signed_zero_chunks_escape_and_roundtrip():
    """v2 has no zero code: every chunk holding +0 or -0 (one, several, or all elements) escapes and decodes bit-exactly,
    including the sign of each zero; chunks without zeros next to them keep the compact code."""
    rng = np.random.default_rng(RNG_SEED + 31)
    # |w| in [0.025, 0.05): one exponent class, so no chunk escapes except the planted ones
    w = _bf(
        (
            rng.choice([-1.0, 1.0], (256, 64)) * rng.uniform(0.5, 1.0, (256, 64)) * 0.05
        ).astype(np.float32)
    )
    w[0, 5] = 0x0000  # one +0 (row 0, k half 0)
    w[1, 40] = 0x8000  # one -0 (row 1, k half 1)
    w[2, :32] = 0x0000  # all +0
    w[3, 32:] = 0x8000  # all -0
    w[4, :32:2] = 0x8000  # alternating -0 / +0
    w[4, 1:32:2] = 0x0000
    w[5, 33] = 0x8000  # -0 next to a large partner (E >= 15 chunk)
    w[5, 34] = 0x7F00
    zero_chunks = {(0, 0), (1, 1), (2, 0), (3, 1), (4, 0), (5, 1)}
    buf, st = wcx.wcx2_encode_reference(w[None])
    assert st.slots == len(zero_chunks) == st.escaped_chunks
    stage = buf[0, 0, 0]
    hdr = stage[wcx.WCX2_HDR_OFF : wcx.WCX2_PRE_OFF]
    for c2 in range(512):
        key = (int(wcx._V2_ROW[c2]), int(wcx._V2_HK[c2]))
        assert (int(hdr[c2]) == wcx.ESC) == (key in zero_chunks)
    dec = wcx.wcx2_decode_reference(buf, 256, 64)[0]
    assert np.array_equal(dec, w)  # bit-exact, signs of zero included
    assert np.array_equal(
        wcx.wcx2_fragments_emulated(stage), wcx.wcx2_expected_fragments(w)
    )
    got, st_t = wcx.wcx2_encode(_torch_bf16(w[None]))
    assert st_t == st and np.array_equal(got.numpy(), buf)


V2_CASES = [
    "gauss",
    "specials",
    "escape_heavy",
    "zeros",
    "tiny_zeros",
    "random_bits",
    "subnormal",
]


def _v2_case(name: str) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED + 100 + V2_CASES.index(name))
    if name == "tiny_zeros":
        # chunks with a zero (here E < 15): v2 escapes, v1 does not
        w = (
            rng.integers(0, 2, (1, 256, 128)) << 15
            | rng.integers(1, 14, (1, 256, 128)) << 7
            | rng.integers(0, 128, (1, 256, 128))
        ).astype(np.uint16)
        w[0, ::4, ::7] = (
            0x0000  # 64 rows x 2 k-halves = 128 escaped chunks per stage (<= WCX2_S_CAP)
        )
        return w
    if name == "subnormal":
        w = _bf(rng.standard_normal((1, 256, 128)).astype(np.float32) * 2.0**-125)
        return w
    if name in ("zeros", "random_bits"):
        # the v2 format bounds a stage at WCX2_S_CAP escaped chunks: rows 0..63 carry the escaping pattern (128 escaped
        # chunks per stage), the other rows are gauss
        w = _torch_case("gauss")[:1, :256, :128].copy()
        w[:, :64] = _torch_case(name)[:1, :64, :128]
        return w
    return _torch_case(name) if name in TORCH_CASES else None


@pytest.mark.parametrize("name", ["zeros", "random_bits"])
def test_v2_escape_count_over_cap_raises(name):
    w = _torch_case(name)
    assert (
        int(
            wcx._analyze_v2(wcx._v2_stage_tiles_np(w[0]).reshape(-1, 32))[3]
            .reshape(-1, 512)
            .sum(1)
            .max()
        )
        > wcx.WCX2_S_CAP
    )
    with pytest.raises(ValueError):
        wcx.wcx2_encode_reference(w)
    with pytest.raises(ValueError):
        wcx.wcx2_encode(_torch_bf16(w))
    with pytest.raises(ValueError):
        wcx.wcx2_stage_bytes(wcx.WCX2_S_CAP + 1)
    assert wcx.wcx2_stage_bytes(wcx.WCX2_S_CAP) <= wcx.WCX_SLOT_STRIDE


@pytest.mark.parametrize("name", V2_CASES)
def test_v2_roundtrip_through_register_emulator(name):
    w = _v2_case(name)[:, :256, :128]
    buf, st = wcx.wcx2_encode_reference(w)
    if name in ("zeros", "random_bits"):
        assert st.slots >= 100  # the escape path is exercised
    assert st.stage_bytes == wcx.wcx2_stage_bytes(st.slots)
    dec = wcx.wcx2_decode_reference(buf, w.shape[1], w.shape[2])
    assert np.array_equal(dec, w)


@pytest.mark.parametrize("name", V2_CASES)
def test_v2_emulated_fragments_match_expected(name):
    w = _v2_case(name)[0, :256, :64]
    buf, _ = wcx.wcx2_encode_reference(w[None])
    fr = wcx.wcx2_fragments_emulated(buf[0, 0, 0])
    assert np.array_equal(fr, wcx.wcx2_expected_fragments(w))


@pytest.mark.parametrize("name", V2_CASES)
def test_v2_torch_encoder_matches_reference(name):
    w = _v2_case(name)
    ref, st_ref = wcx.wcx2_encode_reference(w)
    got, st = wcx.wcx2_encode(_torch_bf16(w))
    assert st == st_ref
    assert np.array_equal(got.numpy(), ref)


def test_v2_escape_slots_equal_v1_without_tiny_zero_chunks():
    w = _torch_case("escape_heavy")
    _, s1 = wcx.wcx_encode_reference(w)
    _, s2 = wcx.wcx2_encode_reference(w)
    assert s2.slots == s1.slots and s2.escaped_chunks == s1.escaped_chunks
    assert s2.stage_bytes == s1.stage_bytes + 64


def test_v2_plane_positions():
    rng = np.random.default_rng(RNG_SEED + 7)
    w = _no_escape((256, 64), RNG_SEED + 8)
    buf, st = wcx.wcx2_encode_reference(w[None])
    assert st.slots == 0
    stage = buf[0, 0, 0]
    for ct, kk, b, rg in [
        (0, 0, 0, 0),
        (37, 1, 1, 0),
        (130, 2, 0, 1),
        (255, 3, 1, 1),
        (int(rng.integers(256)), 2, 1, 0),
    ]:
        G, t = ct >> 2, ct & 3
        h, wq, g = G >> 5, (G >> 3) & 3, G & 7
        c = 64 * h + 16 * wq + g + 8 * rg
        row = 64 * (c >> 5) + (c & 31) + 32 * b
        word = stage[
            kk * 4096 + ct * 16 + 4 * (2 * b + rg) : kk * 4096
            + ct * 16
            + 4 * (2 * b + rg)
            + 4
        ]
        for e, k in enumerate((2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9)):
            v = int(w[row, 16 * kk + k])
            assert int(word[e]) == (((v >> 8) & 0x80) | (v & 0x7F))
        hb = stage[wcx.WCX2_HDR_OFF + 8 * G + 4 * b + 2 * rg + (kk >> 1)]
        chunk = w[row, 32 * (kk >> 1) : 32 * (kk >> 1) + 32]
        assert int(hb) == max((int(x) >> 7) & 0xFF for x in chunk)


def test_v2_escape_prefix_and_slot_order():
    w = _torch_case("escape_heavy")[0, :256, :64]
    buf, st = wcx.wcx2_encode_reference(w[None])
    stage = buf[0, 0, 0]
    hdr = stage[wcx.WCX2_HDR_OFF : wcx.WCX2_PRE_OFF]
    esc = hdr == wcx.ESC
    pre = stage[wcx.WCX2_PRE_OFF : wcx.WCX2_SLOT_OFF]
    assert (
        pre.tolist()
        == (np.cumsum(esc.reshape(64, 8).sum(1)) - esc.reshape(64, 8).sum(1)).tolist()
    )
    assert int(esc.sum()) <= st.slots and esc.any()
