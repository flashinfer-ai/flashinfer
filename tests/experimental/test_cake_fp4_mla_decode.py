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

Tests of the experimental Cake MLA decode package over TensorRT-LLM's
``fp4_mla`` KV cache (``plan_fp4_mla_decode`` and its runner).  The suite reaches
its package through ``tests.cake.test_helpers.cake_package`` and imports nothing
else from flashinfer; ``test_cake_fp4_mla_decode_api.py`` tests the public
``flashinfer.mla.prepare_fp4_mla_decode`` route.  The cache and the query are
written here in TensorRT-LLM's layout (E2M1 codes, E4M3 block scales in the
128x4 swizzle, the NoPE scales shared by each 16-token x 16-channel tile, the
RoPE as a base and a residual) from random latents; the reference is float64
attention over the dequantized bytes.  The probabilities are NVFP4 in the
kernel, so the bound is on the per-row relative L2 error.
"""

import math

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package

pkg = cake_package("cake_fp4_mla_decode")

PAGE = 128
ROW_BYTES = 320
SF_COLS = 40
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
Q_GLOBAL_SCALE = 6.0 * 448.0 / 400.0
KV_GLOBAL_SCALE = 6.0 * 448.0 / 30.0
POISON = 0x7F  # bytes the writer leaves alone: NaN scales, stale codes
REL_L2_BOUND = 0.2


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if tuple(torch.cuda.get_device_capability()) not in ((10, 0), (10, 3)):
        pytest.skip("SM100 or SM103 required")
    return torch.device("cuda", torch.cuda.current_device())


# --------------------------------------------------------------------------- the writer


def _sf_offset(row, col):
    return (
        col % 4
        + (col // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + (row // 128) * 128 * SF_COLS
    )


def _e2m1_codes(x):
    """Nearest E2M1 code (sign bit 3) of ``x`` in units of its scale."""
    table = torch.tensor(E2M1, dtype=torch.float64, device=x.device)
    mag = (x.abs().unsqueeze(-1) - table).abs().argmin(-1)
    return (mag | ((x < 0) & (mag > 0)).long() << 3).to(torch.uint8)


def _e4m3(x):
    """E4M3 bytes (uint8) of positive scales."""
    return x.clamp(min=2.0**-9, max=448.0).to(torch.float8_e4m3fn).view(torch.uint8)


def _scale_values(scale_bytes):
    return scale_bytes.view(torch.float8_e4m3fn).double()


def _quantize(x, scale_bytes, global_scale):
    """Codes of ``x`` [..., 16 k] against per-16 E4M3 scale bytes [..., k]
    (stored as ``amax gs / 6``; a value is ``code scale / gs``)."""
    scale = _scale_values(scale_bytes).repeat_interleave(16, -1) / global_scale
    return _e2m1_codes(x / scale)


def _pack(codes):
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def _rope_split(rope, global_scale):
    """RoPE base (its per-16 quantization) and residual (the rest, quantized)."""
    groups = rope.unflatten(-1, (-1, 16))
    base_sf = _e4m3(groups.abs().amax(-1) * global_scale / 6.0)
    base = _quantize(rope, base_sf, global_scale)
    base_val = _decode(base, base_sf, global_scale)
    resid = rope - base_val
    resid_sf = _e4m3(resid.unflatten(-1, (-1, 16)).abs().amax(-1) * global_scale / 6.0)
    return base, base_sf, _quantize(resid, resid_sf, global_scale), resid_sf


def _decode(codes, scale_bytes, global_scale):
    table = torch.tensor(E2M1, dtype=torch.float64, device=codes.device)
    value = torch.cat((table, -table))[codes.long()]
    return value * _scale_values(scale_bytes).repeat_interleave(16, -1) / global_scale


def _write_case(seed, heads, q_len, kv_lens, width, device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    batch = len(kv_lens)
    used = [-(-n // PAGE) for n in kv_lens]
    pages = sum(used) + batch
    perm = torch.randperm(pages, generator=gen).tolist()
    page_table = torch.zeros(batch, width, dtype=torch.int32)
    kv = torch.full((pages, PAGE, ROW_BYTES), POISON, dtype=torch.uint8)
    sf = torch.full((pages, PAGE * SF_COLS), POISON, dtype=torch.uint8)
    offs = _sf_offset(
        torch.arange(PAGE)[:, None], torch.arange(SF_COLS)[None, :]
    )  # [128, 40]
    cursor = 0
    for b, n in enumerate(kv_lens):
        ids = perm[cursor : cursor + used[b]]
        cursor += used[b]
        page_table[b, : used[b]] = torch.tensor(ids, dtype=torch.int32)
        if n == 0:
            continue
        lat = (torch.randn(n, 576, generator=gen, dtype=torch.float64) * 0.25).clamp(
            -1, 1
        )
        nope, rope = lat[:, :512], lat[:, 512:]
        rb, rb_sf, rr, rr_sf = _rope_split(rope, KV_GLOBAL_SCALE)
        for i, page in enumerate(ids):
            t0, t1 = i * PAGE, min(n, (i + 1) * PAGE)
            tok = t1 - t0
            blocks = -(-tok // 16)
            padded = torch.zeros(blocks * 16, 512, dtype=torch.float64)
            padded[:tok] = nope[t0:t1]
            tile = padded.unflatten(0, (blocks, 16)).unflatten(-1, (32, 16))
            tile_sf = _e4m3(
                tile.abs().amax((1, 3)) * KV_GLOBAL_SCALE / 6.0
            )  # [blocks, 32]
            row_sf = tile_sf.repeat_interleave(16, 0)[:tok]  # [tok, 32]
            codes = _quantize(nope[t0:t1], row_sf, KV_GLOBAL_SCALE)
            kv[page, :tok, :256] = _pack(codes)
            kv[page, :tok, 256:288] = _pack(rb[t0:t1])
            kv[page, :tok, 288:320] = _pack(rr[t0:t1])
            cols = torch.cat((row_sf, rb_sf[t0:t1], rr_sf[t0:t1]), -1)
            # The NoPE scales of a block's every row, valid or not (the tile's).
            rows_sf = torch.cat(
                (
                    tile_sf.repeat_interleave(16, 0),
                    torch.full((blocks * 16, 8), POISON, dtype=torch.uint8),
                ),
                -1,
            )
            rows_sf[:tok] = cols
            n_rows = rows_sf.shape[0]
            sf[page, offs[:n_rows].reshape(-1)] = rows_sf.reshape(-1)
    rows = batch * q_len * heads
    q = (torch.randn(rows, 576, generator=gen, dtype=torch.float64) * 0.25).clamp(-1, 1)
    qn = q[:, :512]
    qn_sf = _e4m3(qn.unflatten(-1, (32, 16)).abs().amax(-1) * Q_GLOBAL_SCALE / 6.0)
    qn_codes = _quantize(qn, qn_sf, Q_GLOBAL_SCALE)
    qrb, qrb_sf, qrr, qrr_sf = _rope_split(q[:, 512:], Q_GLOBAL_SCALE)
    # [qN 512 | rb0 rr0 rb1 rr1 rb2 rr2 rb3 rr3], 16-channel groups.
    tail = torch.stack(
        (qrb.unflatten(-1, (4, 16)), qrr.unflatten(-1, (4, 16))), 2
    ).flatten(1)
    tail_sf = torch.stack(
        (qrb_sf.unflatten(-1, (4, 1)), qrr_sf.unflatten(-1, (4, 1))), 2
    ).flatten(1)
    q_fp4 = _pack(torch.cat((qn_codes, tail), -1))
    q_cols = torch.cat((qn_sf, tail_sf), -1)  # [rows, 40]
    q_sf = torch.zeros(-(-rows // 128) * 128 * SF_COLS, dtype=torch.uint8)
    q_sf[
        _sf_offset(torch.arange(rows)[:, None], torch.arange(SF_COLS)[None, :]).reshape(
            -1
        )
    ] = q_cols.reshape(-1)
    return {
        "kv": kv.to(device),
        "sf": sf.to(device),
        "q_fp4": q_fp4.to(device),
        "q_sf": q_sf.to(device),
        "page_table": page_table.to(device),
        "kv_lens": torch.tensor(kv_lens, dtype=torch.int32, device=device),
        "q_gs": torch.tensor([Q_GLOBAL_SCALE], dtype=torch.float32, device=device),
        "kv_gs": torch.tensor([KV_GLOBAL_SCALE], dtype=torch.float32, device=device),
    }


# --------------------------------------------------------------------------- the reference


def _dequant_rows(row_bytes, sf_bytes, rows, global_scale):
    """``[R, 320]`` code bytes with their scale bytes (row ``rows[i]`` of the
    swizzle) -> float64 ``[R, 640]``."""
    lo, hi = row_bytes & 0x0F, row_bytes >> 4
    codes = torch.stack((lo, hi), -1).flatten(-2)
    offs = _sf_offset(rows[:, None], torch.arange(SF_COLS, device=rows.device)[None, :])
    return _decode(codes, sf_bytes[offs], global_scale)


def _reference(case, heads, q_len, kv_lens, sm_scale):
    device = case["kv"].device
    batch = len(kv_lens)
    rows = batch * q_len * heads
    q = _dequant_rows(
        case["q_fp4"][:rows],
        case["q_sf"],
        torch.arange(rows, device=device),
        Q_GLOBAL_SCALE,
    )
    tail = q[:, 512:].unflatten(-1, (4, 2, 16))
    qn, qrb, qrr = q[:, :512], tail[:, :, 0].flatten(1), tail[:, :, 1].flatten(1)
    out = torch.zeros(rows, 512, dtype=torch.float64, device=device)
    tok = torch.arange(PAGE, device=device)
    for b, n in enumerate(kv_lens):
        if n == 0:
            continue
        ids = case["page_table"][b, : -(-n // PAGE)].long()
        k = torch.cat(
            [
                _dequant_rows(case["kv"][p], case["sf"][p], tok, KV_GLOBAL_SCALE)
                for p in ids.tolist()
            ]
        )[:n]
        kn, krb, krr = k[:, :512], k[:, 512:576], k[:, 576:]
        for t in range(q_len):
            seen = n - (q_len - 1 - t)
            if seen <= 0:
                continue
            r = slice((b * q_len + t) * heads, (b * q_len + t + 1) * heads)
            s = sm_scale * (
                qn[r] @ kn[:seen].T
                + (qrb[r] + qrr[r]) @ krb[:seen].T
                + qrb[r] @ krr[:seen].T
            )
            out[r] = torch.softmax(s, -1) @ kn[:seen]
    return out


def _row_rel(got, ref):
    got = got.double()
    rel = (got - ref).norm(dim=1) / ref.norm(dim=1).clamp(min=1e-300)
    return torch.where(torch.isfinite(got).all(1), rel, torch.full_like(rel, math.inf))


def _run(decode, case, out, sm_scale):
    return decode(
        case["kv"],
        case["sf"],
        case["q_fp4"],
        case["q_sf"],
        case["page_table"],
        case["kv_lens"],
        case["q_gs"],
        case["kv_gs"],
        out,
        sm_scale=sm_scale,
    )


# --------------------------------------------------------------------------- tests


CASES = [
    # heads, q_len, kv_lens, page-table width (None: just wide enough), min
    # splits, the reducer's warps per row when the plan splits
    (128, 4, (300, 2000, 77), None, 1, None),
    (96, 8, (1000, 2049), None, 1, None),
    (
        96,
        1,
        (700, 1301, 64),
        None,
        1,
        None,
    ),  # row tiles straddling 128-row scale blocks
    (128, 4, (3000, 1100), 40, 2, 2),  # several splits, some past the keys
    (128, 4, (2, 0, 300), None, 1, None),  # rows that see no key
    (16, 8, (5, 3000, 0), 32, 2, 2),
    (16, 4, (3000, 4000), 40, 2, 4),  # few rows: four warps per row
    (128, 4, (8000, 5000, 100), 64, 2, 1),  # many rows: one warp per row
    # Rows of at least 32,768 keys: the decode build with long rows.
    (16, 2, (33000, 500), None, 1, None),
    (128, 1, (40000,), None, 2, 4),
]


@pytest.mark.parametrize("heads,q_len,kv_lens,width,min_splits,warps", CASES)
def test_fp4_mla_decode_matches_the_fp64_reference(
    heads, q_len, kv_lens, width, min_splits, warps
):
    device = _device_or_skip()
    width = width or max(-(-n // PAGE) for n in kv_lens) + 1
    sm_scale = 1.0 / math.sqrt(192.0)
    case = _write_case(sum(kv_lens) + heads, heads, q_len, kv_lens, width, device)
    decode = pkg.plan_fp4_mla_decode(
        batch=len(kv_lens),
        q_len=q_len,
        num_heads=heads,
        page_table_width=width,
        device=device,
        backend="cake",
    )
    assert decode.plan.num_split >= min_splits
    assert decode.plan.long_rows == (width * PAGE >= 32768)
    if warps is not None:
        assert decode.plan.merge_warps_per_row == warps
    rows = len(kv_lens) * q_len * heads
    out = torch.full((rows, 512), float("nan"), dtype=torch.bfloat16, device=device)
    assert _run(decode, case, out, sm_scale) is out
    torch.cuda.synchronize()
    ref = _reference(case, heads, q_len, kv_lens, sm_scale)
    keyless = ref.norm(dim=1) == 0
    assert torch.equal(out[keyless].float(), torch.zeros_like(out[keyless].float()))
    rel = _row_rel(out[~keyless], ref[~keyless])
    assert rel.numel() > 0 and float(rel.max()) <= REL_L2_BOUND, float(rel.max())
    again = torch.empty_like(out)
    _run(decode, case, again, sm_scale)
    assert torch.equal(out.view(torch.int16), again.view(torch.int16))


def test_fp4_mla_decode_graph_replays_follow_kv_lens():
    device = _device_or_skip()
    heads, q_len, width = 128, 4, 24
    lens = [(2000, 300), (2900, 17), (5, 3000)]
    sm_scale = 1.0 / math.sqrt(192.0)
    case = _write_case(7, heads, q_len, (3000, 3000), width, device)
    decode = pkg.plan_fp4_mla_decode(
        batch=2, q_len=q_len, num_heads=heads, page_table_width=width, device=device
    )
    out = torch.empty((2 * q_len * heads, 512), dtype=torch.bfloat16, device=device)
    stream = torch.cuda.Stream(device)
    with torch.cuda.stream(stream):
        _run(decode, case, out, sm_scale)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _run(decode, case, out, sm_scale)
    for kv_lens in lens:
        case["kv_lens"].copy_(torch.tensor(kv_lens, dtype=torch.int32))
        out.fill_(float("nan"))
        graph.replay()
        replayed = out.clone()
        eager = torch.empty_like(out)
        _run(decode, case, eager, sm_scale)
        torch.cuda.synchronize()
        assert torch.equal(replayed.view(torch.int16), eager.view(torch.int16)), kv_lens
        ref = _reference(case, heads, q_len, kv_lens, sm_scale)
        keyless = ref.norm(dim=1) == 0
        assert float(_row_rel(replayed[~keyless], ref[~keyless]).max()) <= REL_L2_BOUND


def test_fp4_mla_decode_rejects_mismatched_tensors():
    device = _device_or_skip()
    case = _write_case(3, 128, 4, (300,), 4, device)
    decode = pkg.plan_fp4_mla_decode(
        batch=1, q_len=4, num_heads=128, page_table_width=4, device=device
    )
    out = torch.empty((512, 512), dtype=torch.bfloat16, device=device)
    bad = dict(case, page_table=case["page_table"][:, :3].contiguous())
    with pytest.raises(ValueError, match="page_table"):
        _run(decode, bad, out, 0.1)
    with pytest.raises(ValueError, match="out"):
        _run(decode, case, out[:10], 0.1)
    with pytest.raises(ValueError, match="kv_cache"):
        _run(decode, dict(case, kv=case["kv"][:, :64]), out, 0.1)


def test_short_build_gives_the_long_builds_bits_on_short_rows():
    """The decode build without long rows (picked by ``max_kv``) gives the
    long-row build's bits on rows that see fewer than 32,768 keys."""
    device = _device_or_skip()
    heads, q_len, kv_lens, width = 128, 2, (9000, 31000, 1), 300
    sm_scale = 1.0 / math.sqrt(192.0)
    case = _write_case(11, heads, q_len, kv_lens, width, device)
    kwargs = dict(
        batch=3, q_len=q_len, num_heads=heads, page_table_width=width, device=device
    )
    long_rows = pkg.plan_fp4_mla_decode(**kwargs)
    short_rows = pkg.plan_fp4_mla_decode(**kwargs, max_kv=31000)
    assert long_rows.plan.long_rows and not short_rows.plan.long_rows
    assert long_rows.decode != short_rows.decode
    assert long_rows.plan.num_split == short_rows.plan.num_split
    rows = 3 * q_len * heads
    a = torch.full((rows, 512), float("nan"), dtype=torch.bfloat16, device=device)
    b = torch.full_like(a, float("nan"))
    _run(long_rows, case, a, sm_scale)
    _run(short_rows, case, b, sm_scale)
    torch.cuda.synchronize()
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_plan_picks_the_builds_from_the_bucket():
    long_rows_for, merge_warps_per_row = pkg.long_rows_for, pkg.merge_warps_per_row
    assert not long_rows_for(page_table_width=255)
    assert long_rows_for(page_table_width=256)
    assert not long_rows_for(page_table_width=1024, max_kv=32767)
    assert long_rows_for(page_table_width=1024, max_kv=32768)
    assert [merge_warps_per_row(r) for r in (1, 256, 257, 1024, 1025)] == [
        4,
        4,
        2,
        2,
        1,
    ]


def test_split_rule_depends_on_the_bucket_only():
    num_split_for = pkg.num_split_for
    assert (
        num_split_for(pairs=74, batch=256, m_tiles=4, rows=131072, page_table_width=17)
        == 1
    )
    wide = num_split_for(pairs=74, batch=1, m_tiles=4, rows=512, page_table_width=1024)
    assert 1 < wide <= 16
    assert (
        num_split_for(pairs=74, batch=1, m_tiles=1, rows=128, page_table_width=3) == 1
    )
