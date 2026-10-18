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

import math
import re

import pytest
import torch
from flashinfer.experimental.kimi_k3_fp8_projection import cake_backend as cb
from flashinfer.experimental.kimi_k3_fp8_projection.cake_backend import (
    AMAX_FLOOR,
    ARCHES,
    BLOCK,
    DECODE_MAX_M,
    DECODE_TABLE_BUCKETS,
    E4M3_MAX,
    PROJECTION_FAMILIES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    ceil_to_ue8m0,
    decode_config,
    decode_module_stages,
    k_sets,
    n_padded,
    quant_units,
    required_kernel_keys,
    swizzle_sf_128x4,
    unswizzle_sf_128x4,
)
from flashinfer.experimental.kimi_k3_fp8_projection.cake_jit import (
    KERNELS,
    MODULES,
    decode_kernel_key,
    kernel_program,
    route_available,
)
from flashinfer.experimental.kimi_k3_fp8_projection.decode_table import (
    DECODE_TABLE,
    DECODE_TABLE_OVERRIDES,
    decode_table,
)
from flashinfer.gemm import (
    allocate_kimi_k3_fp8_projection_workspace,
    kimi_k3_fp8_projection,
    prepare_kimi_k3_fp8_projection,
    prepare_kimi_k3_fp8_projection_weights,
)

ATOL = 1e-2
RTOL = 1e-2
SM_COUNT = 148
# Measured SM counts of the dispatch tables (round 7): B200 148, GB300 152.
SM_COUNTS = {"sm_100a": 148, "sm_103a": 152}
# Representative rows (tp, module, M): every route of the dispatch (fused decode incl.
# resident token tiles, quantization launch + decode, quantization launch + GEMM) and
# the correctness-only row counts of the contract (partial tiles, > 256 rows, padded
# output stride).
GPU_ROWS = [
    ("tp8", "q_proj", 1, 0),
    ("tp8", "f_b", 8, 0),
    ("tp8", "f_b", 256, 0),
    ("tp1", "f_b", 256, 0),
    ("tp8", "kv_a", 64, 0),
    # Round 6 (lever L5): the N = 576 family at M > 256 takes the 192-wide GEMM N tile (TMA-store epilogue on the
    # aligned view; the 8-byte-aligned row stride 580 takes its staged register epilogue, padding untouched)
    ("tp8", "kv_a", 4096, 0),
    ("tp1", "kv_a", 4097, 4),
    ("tp8", "fused_qkvg", 256, 0),
    ("tp8", "b_proj", 3, 0),
    ("tp8", "kv_b", 129, 4),
    ("tp1", "kv_b", 256, 0),
    # Round 8: the tp1 o_proj M = 64 cell ships two TMEM accumulator sets on sm_103a (the ``_a2`` program); its M = 256
    # cell pins the quantization launch width (4 K blocks per half warp on sm_100a, 2 on sm_103a) where the rule gives 1
    ("tp1", "o_proj", 64, 0),
    ("tp1", "o_proj", 256, 0),
    # Round 8: the bucket-256 GEMM cells route 65 <= M <= 256 to the GEMM; below 128 rows the TMA-store ``OUT``
    # box (128 rows) is taller than the view and the TMA unit clips the rows >= M (``allow_oob_box`` on the row axis)
    ("tp1", "kv_b", 100, 0),
    ("tp1", "q_b", 65, 0),
    ("tp1", "q_proj", 127, 0),
    ("tp8", "q_proj", 1000, 0),
    # 16-byte row stride (6288) with a column edge (6284) that is not: register epilogue, padding untouched
    ("tp8", "in_proj_qkvgfab", 1000, 4),
    # round 11 (CAKE-985): rows whose table cells carry the round-10 / round-11 GEMM program tokens, so the composed
    # program key below is exercised on a GPU: ``_pdlb`` on the aligned (TMA-store) view of the sm_103a 50,28,1024 cell,
    # ``_st`` (sm_100a 12,1,2048) / ``_gs`` (sm_103a 12,1,2048 and 96,1,2048), ``_cs<S>`` (sm_103a 5,28,2048) and ``_g64``
    # (sm_103a 144,6,16384); on the other architecture each row resolves to that table's own form.
    ("tp8", "in_proj_qkvgfab", 1000, 0),
    ("tp8", "f_b", 2048, 0),
    ("tp1", "f_b", 2048, 0),
    ("tp8", "kv_a", 2048, 0),
    ("tp8", "q_b", 16384, 0),
    ("tp8", "q_b", 4096, 0),
    ("tp1", "o_proj", 4097, 4),
    (
        "tp8",
        "f_a",
        4096,
        0,
    ),  # narrow-N large-M row: tabulated fused decode route above DECODE_MAX_M
    ("tp1", "b_proj", 4097, 4),
    # Round 6 continuation 12 (lever SKO): tabulated ``gemm_sk`` rows inside the stream-K wave window -- the TMA-store
    # instance on the aligned view and the staged register instance on the 8-byte-aligned padded row stride
    ("tp8", "q_proj", 4096, 0),
    ("tp1", "o_proj", 4096, 4),
    # Round 6 continuation 17/18 (lever SKF): tabulated ``gemm_skf`` rows (fractional wave) -- the fix-up stream-K
    # TMA-store instance on the aligned view (256-wide and 192-wide N tiles)
    ("tp1", "fused_qkvg", 256, 0),
    ("tp1", "in_proj_qkvgfab", 256, 0),
    ("tp8", "kv_a", 16384, 0),
    # Round 8: the register-epilogue fallback of TMA-store cells on an 8-byte-aligned output
    # stride (sm_100a 12,1,512; 96,28,64 on both architectures; no padded-stride contract row
    # lies in these buckets, so the export covers the fallback programs with correctness-only
    # coverage shapes). Where the cell is not a TMA-store cell the row is a plain
    # unaligned-stride decode row.
    ("tp8", "f_b", 512, 4),
    ("tp1", "q_proj", 64, 4),
]


# ---------------------------------------------------------------------------
# Torch reference (exact quantized-operand emulation)
# ---------------------------------------------------------------------------


def per_token_cast_to_fp8(x):
    m, k = x.shape
    xv = x.view(m, k // BLOCK, BLOCK)
    amax = xv.abs().float().amax(dim=2).clamp(AMAX_FLOOR)
    sf = ceil_to_ue8m0(amax / E4M3_MAX)
    q = (xv.float() * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, k)
    return q, sf


def per_block_cast_to_fp8(w):
    n, k = w.shape
    wv = w.view(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    amax = wv.abs().float().amax(dim=(1, 3), keepdim=True).clamp(AMAX_FLOOR)
    sf = amax / E4M3_MAX
    q = (wv.float() * (1.0 / sf)).to(torch.float8_e4m3fn).view(n, k)
    return q, sf.view(n // BLOCK, k // BLOCK)


def make_weight(n_valid, K, device, seed):
    g = torch.Generator(device=device).manual_seed(seed + 31 * n_valid + K)
    n_pad = n_padded(n_valid)
    w = torch.randn((n_pad, K), device=device, generator=g, dtype=torch.float32) * 0.02
    w[n_valid:] = 0.0
    w_q, sf = per_block_cast_to_fp8(w.to(torch.bfloat16))
    n128 = -(-n_valid // BLOCK) * BLOCK
    return w_q[:n128].contiguous(), sf[: n128 // BLOCK].reshape(
        n128 // BLOCK, 1, K // BLOCK, 1
    ).contiguous()


def make_activation(M, K, device, seed):
    g = torch.Generator(device=device).manual_seed(seed + 7 * M + K)
    return torch.randn((M, K), device=device, generator=g, dtype=torch.float32).to(
        torch.bfloat16
    )


def reference(x, weight, scale, n_valid):
    n, k = weight.shape
    w2, s2 = cb.requant_weight_ue8m0(weight, scale)
    a_q, a_sf = per_token_cast_to_fp8(x)
    a = (
        a_q.float().view(x.shape[0], k // BLOCK, BLOCK)
        * a_sf.view(x.shape[0], k // BLOCK, 1)
    ).view(x.shape[0], k)
    w = (
        w2.float().view(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
        * s2.view(n // BLOCK, 1, k // BLOCK, 1)
    ).view(n, k)
    out = torch.empty((x.shape[0], n), dtype=torch.float32, device=x.device)
    for i in range(0, x.shape[0], 4096):
        out[i : i + 4096] = a[i : i + 4096] @ w.T
    return out[:, :n_valid].to(torch.bfloat16)


def bf16_ulp(mag):
    bits = mag.abs().view(torch.int32) & 0x7F800000
    return (bits.view(torch.float32) * (2.0**-7)).clamp_min(
        torch.finfo(torch.bfloat16).tiny
    )


def assert_matches(actual, expected):
    """Zero-budget rule of the source contract: ``|out - ref| <= atol + rtol |ref| + 2 bf16 ulp``."""
    o = actual.float()
    r = expected.float()
    assert torch.isfinite(o).all()
    bound = ATOL + RTOL * r.abs() + 2.0 * bf16_ulp(r)
    bad = (o - r).abs() > bound
    assert int(bad.sum()) == 0, f"{int(bad.sum())} elements exceed the tolerance"


# ---------------------------------------------------------------------------
# Host rules (CPU)
# ---------------------------------------------------------------------------


def test_ue8m0_rounding():
    values = torch.tensor([1.0, 1.5, 0.75, 3.0e-5, 1024.0, 0.1], dtype=torch.float32)
    rounded = ceil_to_ue8m0(values)
    expected = torch.tensor([2.0 ** math.ceil(math.log2(v)) for v in values.tolist()])
    assert torch.equal(rounded, expected)
    assert torch.equal(
        cb.ue8m0_byte(rounded), (torch.log2(rounded) + 127).to(torch.uint8)
    )


def test_scale_swizzle_roundtrip():
    sf = (
        torch.arange(300 * 6, dtype=torch.int32)
        .remainder(251)
        .to(torch.uint8)
        .reshape(300, 6)
    )
    swizzled = swizzle_sf_128x4(sf)
    assert swizzled.numel() == 3 * 128 * 8
    assert torch.equal(unswizzle_sf_128x4(swizzled, 300, 6), sf)


def test_weight_scale_tiles_bn_layout():
    # Round 6 (lever L5): the 192-wide GEMM instance reads B scales per 192-row tile in logical N order across the two
    # 128-lane TMEM blocks (rows 96..127 of a tile sit in block 0, lanes 96..127); 256 reproduces the CTA-pair layout.
    torch.manual_seed(622)
    K, n_pad = 7168, 768
    sf = torch.randint(100, 140, (n_pad // 128, K // 128), dtype=torch.uint8)
    assert torch.equal(
        cb.weight_scale_tiles_bn(sf, K, 256), cb.weight_scale_tiles(sf, K)
    )
    ks, n_tiles = cb.k_sets(K), -(-n_pad // 192)
    t192 = cb.weight_scale_tiles_bn(sf, K, 192)
    assert t192.numel() == n_tiles * ks * 2 * cb.SF_TILE_BYTES
    flat = t192.view(n_tiles, ks, 2, cb.SF_TILE_BYTES).permute(0, 2, 1, 3).reshape(-1)
    per_row = cb.unswizzle_sf_128x4(flat, n_tiles * 256, ks * 4)
    rows = sf.repeat_interleave(128, dim=0).repeat_interleave(4, dim=1)
    for t in range(n_tiles):
        expect = torch.zeros((256, ks * 4), dtype=torch.uint8)
        lo, hi = t * 192, min(t * 192 + 192, n_pad)
        expect[: hi - lo, : rows.shape[1]] = rows[lo:hi]
        assert torch.equal(per_row[t * 256 : (t + 1) * 256], expect), t
    # Round 10 (CAKE-985 S3): the 384-wide pair tile lays each tile's rows across three 128-lane blocks (the last tile of
    # the 768-row storage holds rows 768..767 -> zeros past N_pad); the SFB view is 12 rows of 128 B per (tile, K set).
    n_tiles384 = -(-n_pad // 384)
    t384 = cb.weight_scale_tiles_bn(sf, K, 384)
    assert t384.numel() == n_tiles384 * ks * 3 * cb.SF_TILE_BYTES
    assert (
        cb.weight_scale_sf_rows(384) == 12
        and cb.weight_scale_sf_rows(192) == cb.weight_scale_sf_rows(256) == 8
    )
    flat = (
        t384.view(n_tiles384, ks, 3, cb.SF_TILE_BYTES).permute(0, 2, 1, 3).reshape(-1)
    )
    per_row = cb.unswizzle_sf_128x4(flat, n_tiles384 * 384, ks * 4)
    for t in range(n_tiles384):
        expect = torch.zeros((384, ks * 4), dtype=torch.uint8)
        lo, hi = t * 384, min(t * 384 + 384, n_pad)
        expect[: hi - lo, : rows.shape[1]] = rows[lo:hi]
        assert torch.equal(per_row[t * 384 : (t + 1) * 384], expect), t
    with pytest.raises(ValueError):
        cb.weight_scale_tiles_bn(sf, K, 100)
    with pytest.raises(ValueError):
        cb.weight_scale_tiles_bn(sf, K, 512)


def test_n384_cell_rules_and_g64_stems():
    """Round 10 (CAKE-985 S3): ``gemm_n384_cell_conflicts`` names the keys a ``gemm_bn`` 384 cell cannot carry (the Cake
    launcher's rule), ``_g64_stem`` places the raster token after the bn token and refuses the register epilogue; round 11
    (CAKE-985 T1a): ``gemm_pdlb`` joins the 384-wide conflicts, ``_pdlb_stem`` closes the stem with the token and refuses
    the register epilogue and the 384-wide programs, ``_entry_gemm_pdlb`` reads the cell key as 0 / 1; round 11 (CAKE-985
    T2-e): ``gemm_static`` joins the 384-wide conflicts, ``_st_stem`` takes the _cs<S> slot of the plain _n192_k128 stems,
    ``gemm_static_cell_conflicts`` names the keys a static cell cannot carry."""
    assert cb.gemm_n384_cell_conflicts(None) == []
    assert (
        cb.gemm_n384_cell_conflicts(
            {
                "route": "gemm",
                "gemm_bn": 384,
                "gemm_bk": 128,
                "gemm_gm": 64,
                "quant_units": 4,
            }
        )
        == []
    )
    assert cb.gemm_n384_cell_conflicts({"gemm_bn": 384, "gemm_bk": 256}) == ["gemm_bk"]
    assert cb.gemm_n384_cell_conflicts(
        {"gemm_bn": 384, "gemm_pf": 2, "gemm_skf": 2, "gemm_ts_slots": 4}
    ) == ["gemm_pf", "gemm_skf", "gemm_ts_slots"]
    assert cb._g64_stem("gemm_tstore") == "gemm_tstore_g64"
    assert cb._g64_stem("gemm_rstaged_n192_k128") == "gemm_rstaged_n192_g64_k128"
    assert cb._g64_stem("gemm_tstore_n384_k128") == "gemm_tstore_n384_g64_k128"
    with pytest.raises(ValueError, match="register epilogue"):
        cb._g64_stem("gemm")
    # round 11 (CAKE-985 T1a, _pdlb): the last stem token, before ``_pf<D>``; no register-epilogue or 384-wide form
    assert "gemm_pdlb" in cb.GEMM_N384_CELL_CONFLICTS
    assert cb.gemm_n384_cell_conflicts({"gemm_bn": 384, "gemm_pdlb": 1}) == [
        "gemm_pdlb"
    ]
    assert cb.gemm_n384_cell_conflicts({"gemm_bn": 384, "gemm_pdlb": 0}) == []
    assert cb._pdlb_stem("gemm_tstore") == "gemm_tstore_pdlb"
    assert cb._pdlb_stem("gemm_rstaged_n192_k128") == "gemm_rstaged_n192_k128_pdlb"
    assert (
        cb._pdlb_stem("gemm_tstore_n192_k128_cs2") == "gemm_tstore_n192_k128_cs2_pdlb"
    )
    assert (
        cb._pdlb_stem("gemm_tstore_n192_k128_skf4") == "gemm_tstore_n192_k128_skf4_pdlb"
    )
    assert (
        cb._pdlb_stem("gemm_rstaged_n192_g64_k128") == "gemm_rstaged_n192_g64_k128_pdlb"
    )
    assert cb.gemm_kernel_key(cb._pdlb_stem("gemm_tstore"), 2) == "gemm_tstore_pdlb_pf2"
    for stem in ("gemm", "gemm_n192", "gemm_tstore_n384_k128"):
        with pytest.raises(ValueError, match="no _pdlb form"):
            cb._pdlb_stem(stem)
    assert cb._entry_gemm_pdlb({"route": "gemm", "gemm_pdlb": 1}) is True
    assert cb._entry_gemm_pdlb({"route": "gemm", "gemm_pdlb": 0}) is False
    assert cb._entry_gemm_pdlb({"route": "gemm"}) is False
    with pytest.raises(ValueError, match="gemm_pdlb"):
        cb._entry_gemm_pdlb({"route": "gemm", "gemm_pdlb": 2})
    # round 11 (CAKE-985 T2-e, _st): the static one-tile form takes the _cs<S> slot of the plain _n192_k128 TMA-store /
    # staged-register stems; no 384-wide, register-epilogue, stream-K, cluster split-K or _pdlb composition
    assert "gemm_static" in cb.GEMM_N384_CELL_CONFLICTS
    assert cb.gemm_n384_cell_conflicts({"gemm_bn": 384, "gemm_static": 1}) == [
        "gemm_static"
    ]
    assert cb._st_stem("gemm_tstore_n192_k128") == "gemm_tstore_n192_k128_st"
    assert cb._st_stem("gemm_rstaged_n192_g64_k128") == "gemm_rstaged_n192_g64_k128_st"
    for stem in (
        "gemm_n192_k128",
        "gemm_tstore",
        "gemm_tstore_n192",
        "gemm_tstore_k128",
        "gemm_tstore_n384_k128",
        "gemm_tstore_n192_k128_cs2",
        "gemm_tstore_n192_k128_skf",
    ):
        assert not cb._st_eligible(stem), stem
        with pytest.raises(ValueError, match="no _st form"):
            cb._st_stem(stem)
    with pytest.raises(ValueError, match="no _pdlb form"):
        cb._pdlb_stem("gemm_tstore_n192_k128_st")
    assert cb._entry_gemm_static({"route": "gemm", "gemm_static": 1}) is True
    assert cb._entry_gemm_static({"route": "gemm", "gemm_static": 0}) is False
    assert cb._entry_gemm_static({"route": "gemm"}) is False
    with pytest.raises(ValueError, match="gemm_static"):
        cb._entry_gemm_static({"route": "gemm", "gemm_static": 2})
    assert cb.gemm_static_cell_conflicts(None) == ["gemm_bn", "gemm_bk"]
    assert (
        cb.gemm_static_cell_conflicts(
            {
                "route": "gemm",
                "gemm_bn": 192,
                "gemm_bk": 128,
                "gemm_static": 1,
                "gemm_gm": 64,
                "quant_units": 4,
            }
        )
        == []
    )
    assert cb.gemm_static_cell_conflicts(
        {"gemm_bn": 192, "gemm_bk": 128, "gemm_cs": 2, "gemm_pdlb": 1}
    ) == [
        "gemm_cs",
        "gemm_pdlb",
    ]
    assert cb.gemm_static_cell_conflicts(
        {
            "gemm_bn": 192,
            "gemm_bk": 128,
            "gemm_pf": 2,
            "gemm_skf": 2,
            "gemm_ts_slots": 4,
        }
    ) == [
        "gemm_pf",
        "gemm_skf",
        "gemm_ts_slots",
    ]
    assert cb.gemm_static_cell_conflicts({"gemm_bn": 256, "gemm_bk": 128}) == [
        "gemm_bn"
    ]
    # round 11 (CAKE-985 T6, _gs): the static grid-stride form takes the same slot of the plain _k128 / _n192_k128 TMA-store stems;
    # exclusive with _st, no 384-wide, staged-register, register-epilogue, stream-K, cluster split-K, stage or _pdlb composition
    assert (
        "gemm_gs" in cb.GEMM_N384_CELL_CONFLICTS
        and "gemm_gs" in cb.GEMM_STATIC_CELL_CONFLICTS
    )
    assert "gemm_static" in cb.GEMM_GS_CELL_CONFLICTS
    assert cb.gemm_n384_cell_conflicts({"gemm_bn": 384, "gemm_gs": 1}) == ["gemm_gs"]
    assert cb.gemm_static_cell_conflicts(
        {"gemm_bn": 192, "gemm_bk": 128, "gemm_gs": 1}
    ) == ["gemm_gs"]
    for stem, expected in (
        ("gemm_tstore_k128", "gemm_tstore_k128_gs"),
        ("gemm_tstore_n192_k128", "gemm_tstore_n192_k128_gs"),
        ("gemm_tstore_g64_k128", "gemm_tstore_g64_k128_gs"),
        ("gemm_tstore_n192_g64_k128", "gemm_tstore_n192_g64_k128_gs"),
    ):
        assert cb._gs_eligible(stem) and cb._gs_stem(stem) == expected, stem
    for stem in (
        "gemm_k128",
        "gemm_rstaged_k128",
        "gemm_rstaged_n192_k128",
        "gemm_tstore",
        "gemm_tstore_n192",
        "gemm_tstore_n384_k128",
        "gemm_tstore_n192_k128_cs2",
        "gemm_tstore_n192_k128_st",
        "gemm_tstore_k128_s2_ts4",
        "gemm_tstore_k128_skf",
    ):
        assert not cb._gs_eligible(stem), stem
        with pytest.raises(ValueError, match="no _gs form"):
            cb._gs_stem(stem)
    with pytest.raises(ValueError, match="no _pdlb form"):
        cb._pdlb_stem("gemm_tstore_k128_gs")
    assert cb._entry_gemm_gs({"route": "gemm", "gemm_gs": 1}) is True
    assert cb._entry_gemm_gs({"route": "gemm", "gemm_gs": 0}) is False
    assert cb._entry_gemm_gs({"route": "gemm"}) is False
    with pytest.raises(ValueError, match="gemm_gs"):
        cb._entry_gemm_gs({"route": "gemm", "gemm_gs": 2})
    assert cb.gemm_gs_cell_conflicts(None) == ["gemm_bk"]
    assert (
        cb.gemm_gs_cell_conflicts(
            {
                "route": "gemm",
                "gemm_bk": 128,
                "gemm_gs": 1,
                "gemm_gm": 64,
                "quant_units": 4,
            }
        )
        == []
    )
    assert (
        cb.gemm_gs_cell_conflicts(
            {"route": "gemm", "gemm_bn": 192, "gemm_bk": 128, "gemm_gs": 1}
        )
        == []
    )
    assert cb.gemm_gs_cell_conflicts(
        {"gemm_bn": 192, "gemm_bk": 128, "gemm_static": 1, "gemm_pdlb": 1}
    ) == ["gemm_pdlb", "gemm_static"]
    assert cb.gemm_gs_cell_conflicts({"gemm_bn": 384, "gemm_bk": 128}) == ["gemm_bn"]
    assert cb.gemm_gs_cell_conflicts(
        {"gemm_bk": 128, "gemm_pf": 2, "gemm_skf": 2, "gemm_ts_slots": 4}
    ) == ["gemm_pf", "gemm_skf", "gemm_ts_slots"]
    # the launched pairs P' = min(T, sm_count // 2): the T6 prototype launches (kv_b 1024 / 16384 at bn 192 / 256) and a one-tile launch
    assert cb._gs_pairs(148, 8, 128) == 74 and cb._gs_pairs(152, 8, 128) == 76
    assert cb._gs_pairs(148, 128, 96) == 74 and cb._gs_pairs(152, 128, 96) == 76
    assert (
        cb._gs_pairs(148, 8, 12) == 48 and cb._gs_pairs(152, 8, 16) == 64
    )  # tp8:kv_b:1024 (m_tiles 8): bn 256 / 192, T <= resident pairs -> P' = T
    with pytest.raises(ValueError, match="at least one pair tile"):
        cb._gs_pairs(148, 0, 3)


def test_sk_bindings_carry_no_split_scalars(monkeypatch):
    """Round 11 (CAKE-985 T6, ``_gs``): the stream-K hand-off bindings of a GEMM launch are the two workspace tensors only;
    the split scalars (``sk_pairs`` / ``sk_rem`` / ``sk_ksplit`` / ``sk_dp``) come from the stage arguments
    (``_sk_scalar_kwargs``) alone.  The call-bound dictionary is applied over the stage arguments, so a scalar binding would
    shadow them: the grid-stride programs read their launched pairs P' from ``sk_pairs`` with no hand-off area, and a zero
    binding launched them with no tiles (the FIN tip-3 FI target test found the ``_gs`` rows' output unwritten)."""
    import types

    monkeypatch.setattr(
        cb,
        "_sk_dummy",
        lambda device: (
            torch.zeros((64,), dtype=torch.float32),
            torch.zeros((128,), dtype=torch.uint32),
        ),
    )
    # the largest hand-off area of the classes below: the stream-K form's ``rem`` tiles of GEMM_SK_TILE_BYTES behind the flags
    sf = torch.zeros(
        (cb.GEMM_SK_FLAG_BYTES + 8 * cb.GEMM_SK_TILE_BYTES,), dtype=torch.uint8
    )
    device = torch.device("cpu")
    tensors = {"sk_partials", "sk_flags"}
    assert tensors <= cb._CALL_ARGS
    plain = types.SimpleNamespace(
        gemm_sk=None,
        gemm_skf=None,
        gemm_skp=None,
        gemm_gs=False,
        gemm_gs_pairs=0,
        counters_offset=0,
    )
    gs = types.SimpleNamespace(
        gemm_sk=None,
        gemm_skf=None,
        gemm_skp=None,
        gemm_gs=True,
        gemm_gs_pairs=76,
        counters_offset=0,
    )
    skf = types.SimpleNamespace(
        gemm_sk=None,
        gemm_skf=types.SimpleNamespace(
            pairs=74, total=5, slots=2, first=3, partial_bytes=4096
        ),
        gemm_skp=None,
        gemm_gs=False,
        gemm_gs_pairs=0,
        counters_offset=0,
    )
    sk = types.SimpleNamespace(
        gemm_sk=types.SimpleNamespace(pairs=74, rem=6, ksplit=7, dp=1),
        gemm_skf=None,
        gemm_skp=None,
        gemm_gs=False,
        gemm_gs_pairs=0,
        counters_offset=0,
    )
    for plan in (plain, gs, skf, sk):
        bound = cb._sk_launch_kwargs(plan, sf, device)
        scalars = cb._sk_scalar_kwargs(plan)
        assert set(bound) == tensors, sorted(bound)
        assert not (set(bound) & set(scalars)), sorted(set(bound) & set(scalars))
        assert set(scalars) >= {"sk_pairs", "sk_rem", "sk_ksplit", "sk_dp"}
    assert cb._sk_scalar_kwargs(plain) == dict(
        sk_pairs=0, sk_rem=0, sk_ksplit=0, sk_dp=0
    )
    assert cb._sk_scalar_kwargs(gs) == dict(sk_pairs=76, sk_rem=0, sk_ksplit=0, sk_dp=0)
    assert cb._sk_scalar_kwargs(skf) == dict(
        sk_pairs=74, sk_rem=5, sk_ksplit=2, sk_dp=3
    )
    assert cb._sk_scalar_kwargs(sk) == dict(sk_pairs=74, sk_rem=6, sk_ksplit=7, sk_dp=1)
    with pytest.raises(ValueError, match="stream-K partial area"):
        cb._sk_launch_kwargs(skf, sf[:64], device)


def test_padding_and_k_sets():
    assert n_padded(12) == 256 and n_padded(6284) == 6400 and n_padded(1536) == 1536
    assert cb.n_padded_128(12) == 128 and cb.n_padded_128(6284) == 6400
    assert (
        k_sets(128) == 2
        and k_sets(512) == 4
        and k_sets(7168) == 56
        and k_sets(1536) == 12
    )
    with pytest.raises(ValueError):
        k_sets(100)


def test_quant_units_rule():
    for M in (1, 8, 64, 256):
        assert quant_units(M, 56, SM_COUNT) == 1
    assert quant_units(4096, 56, SM_COUNT) == 4  # K = 7168
    assert quant_units(4096, 1, SM_COUNT) == 1  # K = 128
    assert (
        quant_units(4096, 4, SM_COUNT) == 2
    )  # K = 512: four blocks per half warp would leave < 4 CTAs/SM
    assert quant_units(16384, 4, SM_COUNT) == 4  # K = 512
    assert quant_units(1000, 12, SM_COUNT) == 2  # K = 1536, M = 1000
    assert quant_units(16384, 96, SM_COUNT) == 4  # K = 12288
    # The rule scales with the device: a 64-SM part keeps four blocks per half warp at a quarter of the rows.
    assert quant_units(1024, 56, 64) == 4


def test_decode_table_overrides_are_per_architecture_cells():
    # One shared table holds the cells every architecture measured alike; a cell whose fastest route differs
    # per architecture lives only in the overrides, with one entry per architecture, and never adds a family.
    assert set(DECODE_TABLE_OVERRIDES) == set(ARCHES)
    override_keys = {
        frozenset(overrides) for overrides in DECODE_TABLE_OVERRIDES.values()
    }
    assert len(override_keys) == 1
    (keys,) = override_keys
    assert not (keys & set(DECODE_TABLE))
    for arch, overrides in DECODE_TABLE_OVERRIDES.items():
        merged = decode_table(arch)
        assert set(merged) == set(DECODE_TABLE) | keys
        for key, entry in overrides.items():
            assert merged[key] == entry
    for key in keys:
        entries = [DECODE_TABLE_OVERRIDES[arch][key] for arch in ARCHES]
        assert any(entry != entries[0] for entry in entries[1:])
    with pytest.raises(ValueError, match="no measured dispatch table"):
        decode_table("sm_90a")


@pytest.mark.parametrize("arch", ARCHES)
def test_decode_table_covers_every_family(arch):
    for tp, modules in PROJECTION_FAMILIES.items():
        for name, (n_valid, K) in modules.items():
            n_tiles128 = -(-n_valid // BLOCK)
            num_k_iters = -(-K // 256)
            for bucket in DECODE_TABLE_BUCKETS:
                entry = cb.decode_table_entry(bucket, n_tiles128, num_k_iters, arch)
                if bucket > DECODE_MAX_M:
                    # Large buckets hold the families measured faster on the decode kernel, the GEMM-routed rows whose
                    # cell pins an instance lever (round 6: the 192-wide N tile, the ordered / fix-up stream-K forms;
                    # round 7: the weight prefetch distance and the sliced stream-K form) and, since round 7, explicit
                    # plain-GEMM cells (every measured M <= 2048 cell is tabulated; an absent cell is plain GEMM too).
                    assert entry is None or entry["route"] in ("decode", "gemm"), (
                        f"{arch} {tp}:{name} bucket {bucket}: {entry}"
                    )
                    continue
                assert entry is not None, f"{arch} {tp}:{name} bucket {bucket}"
                assert entry["route"] in ("decode", "gemm")


@pytest.mark.parametrize("arch", ARCHES)
def test_decode_config_rules(arch):
    # Above DECODE_MAX_M the table decides: a tabulated bucket cell routes its whole bucket (round 7: M = 257 follows
    # the 512 cell of the 12-tile K = 7168 family), an absent cell is plain GEMM; the narrow-N families stay decode.
    cell_512 = cb.decode_table_entry(512, 12, 28, arch)
    assert cell_512 is not None
    assert (decode_config(257, 12, 28, arch, SM_COUNT) is not None) == (
        cell_512["route"] == "decode"
    )
    assert decode_config(4096, 12, 28, arch, SM_COUNT) is None
    assert decode_config(4096, 1, 28, arch, SM_COUNT) is not None
    for key, entry in decode_table(arch).items():
        n_tiles128, num_k_iters, bucket = (int(v) for v in key.split(","))
        cfg = decode_config(bucket, n_tiles128, num_k_iters, arch, SM_COUNT)
        if entry["route"] == "gemm":
            assert cfg is None
            continue
        assert cfg is not None
        assert cfg.tok == entry["tok"] and cfg.fused == entry["fused"]
        assert 1 <= cfg.split <= num_k_iters
        assert cfg.tiles == n_tiles128 * -(-bucket // cfg.tok)
        # Round 6 (lever M): at the bucket's M every N tile's m tiles fill whole clusters; one cluster item per N tile.
        assert cfg.mc == int(entry.get("mc", 1)) and cfg.pf == int(entry.get("pf", 0))
        # Round 8 (lever PW; adopted round 11 C20 on the sm_103a 192,2,64 cell): the first-item W-stream L2 prefetch depth is a
        # table key of any decode row (no constraint); it names the ``_pw<D>`` program and changes no launch argument.
        assert cfg.pw == int(entry.get("pw", 0))
        # Round-6 next loop (lever PX): the BF16 token-tile L2 prefetch applies to fused, non-resident rows only.
        assert cfg.pfx == (
            int(entry.get("pfx", 0)) if (cfg.fused and not cfg.resident) else 0
        )
        # Round-6 continuation 7 (lever PI-W): the next-item W/SFW L2 prefetch rides on the weight prefetch (pf > 0 rows only).
        assert cfg.pfi == (int(entry.get("pfi", 0)) if cfg.pf > 0 else 0)
        # Round-6 continuation 8 (lever QW16): the quantizing-warp count is a table key of fused rows (8 = default).
        assert cfg.qwarps == (
            int(entry.get("qwarps", cb.DEC_QUANT_WARPS))
            if cfg.fused
            else cb.DEC_QUANT_WARPS
        )
        assert (f"_w{cfg.qwarps}" in cfg.kernel_key) == (
            cfg.fused and cfg.qwarps != cb.DEC_QUANT_WARPS
        )
        if cfg.mc > 1:
            assert cfg.split == 1 and cfg.csplit == 1 and not cfg.resident
            assert cfg.m_tiles % cfg.mc == 0 and cfg.total_work == cfg.tiles // cfg.mc
        else:
            assert cfg.total_work == cfg.tiles * cfg.split
        expected_grid = (
            min(cfg.total_work, int(entry.get("grid") or SM_COUNT))
            if cfg.persist
            else cfg.total_work
        )
        if cfg.csplit > 1:
            expected_grid = cfg.csplit * max(
                1,
                min(
                    expected_grid // cfg.csplit,
                    cb.decode_cluster_capacity(arch, cfg.csplit),
                ),
            )
        if cfg.mc > 1:
            expected_grid = cfg.mc * max(
                1,
                min(
                    expected_grid,
                    SM_COUNT // cfg.mc,
                    cb.decode_cluster_capacity(arch, cfg.mc),
                ),
            )
        assert cfg.grid == expected_grid
        # Round 6 (lever D): a table row may pin the ring deeper than the host default (``12,28,256``: 5 stages).
        assert (
            1
            <= cfg.module_stages
            <= cfg.stages
            <= max(cb.DEC_MAX_STAGES, int(entry.get("stages") or 0))
        )
        if cfg.resident:
            assert cfg.fused and num_k_iters == 1 and cfg.split == 1 and cfg.tok <= 64
            assert -(-cfg.total_work // cfg.grid) <= cb.DEC_RES_SLOTS
        assert cfg.kernel_key.startswith(f"decode:t{cfg.tok}_p{cfg.module_stages}")
        # Round 6: ``_mc<C>`` / ``_pf<D>`` close the key (after the cluster split-K field); strip them for the older checks.
        assert (f"_mc{cfg.mc}" in cfg.kernel_key) == (cfg.mc > 1)
        assert (f"_pf{cfg.pf}" in cfg.kernel_key) == (cfg.pf > 0)
        assert (f"_pw{cfg.pw}" in cfg.kernel_key) == (
            cfg.pw > 0
        )  # round 8 (lever PW): ``_pw<D>`` after ``_pf<D>``, before ``_px<D>``
        assert (f"_px{cfg.pfx}" in cfg.kernel_key) == (cfg.pfx > 0)
        assert (f"_pi{cfg.pfi}" in cfg.kernel_key) == (cfg.pfi > 0)
        # Round 7 (lever XP): the parallel-issue DSM exchange closes the key; cluster split-K rows only.
        assert ("_xp" in cfg.kernel_key) == cfg.xp
        assert cfg.xp == (bool(entry.get("xp", False)) and cfg.csplit > 1)
        # Round 6 (lever Q1; adopted round 8): the tensor-map descriptor prefetch is a table key of any decode row;
        # ``_dpf`` follows the TMA-store token and precedes ``_pi<N>`` / ``_xp`` (the Cake symbol order).
        assert cfg.dpf == bool(entry.get("dpf", False))
        assert ("_dpf" in cfg.kernel_key) == cfg.dpf
        # Round 8: the epilogue-chunk and cluster split-K tokens (``_c<N>``, ``_cs<N>[a]``) follow ``_q`` as well (the
        # 14-CTA fused cells adopted narrow lanes), so they are stripped before the ``_q`` suffix check below.
        core_key = re.sub(
            r"(_c\d+)?(_cs\d+a?)?(_mc\d+)?(_pf\d+)?(_pw\d+)?(_px\d+)?(_dpf)?(_pi\d+)?(_xp)?$",
            "",
            cfg.kernel_key,
        )
        # Round-3 fused knobs: a decoupled ring only for fused, non-resident rows; narrow units divide evenly.
        if entry.get("xb_stages"):
            assert (
                cfg.fused
                and not cfg.resident
                and 1 <= cfg.xb_stages <= cb.DEC_XB_MAX_STAGES
            )
            assert f"_r{cfg.xb_stages}" in cfg.kernel_key
            # Round-6 continuation 9 (lever XBH): the half-slot ring is a table key of fused ring rows with narrow units.
            assert cfg.xbh == (bool(entry.get("xbh", False)) and cfg.qlanes != 16)
            assert (f"_r{cfg.xb_stages}_xh" in cfg.kernel_key) == cfg.xbh
            # Round-6 continuation 10 (lever QER): the early half-slot release is a table key of xbh rows only (``_qe`` after ``_xh``).
            assert cfg.qer == (bool(entry.get("qer", False)) and cfg.xbh)
            assert ("_xh_qe" in cfg.kernel_key) == cfg.qer
            # Round 4 lever F (adopted in round 7 on the t16 cs4 M = 256 cells): the FP8 token ring is a table key of
            # BF16-ring rows (``_x<N>`` after the ring fields, before ``_q``); the three rings + the epilogue chunk fit the pool.
            if entry.get("xq_stages"):
                assert 1 <= cfg.xq_stages <= cb.DEC_XQ_MAX_STAGES
                assert f"_x{cfg.xq_stages}" in cfg.kernel_key
                assert (
                    cfg.module_stages * (cb.DEC_W_BYTES + 1024)
                    + cfg.xq_stages * (cfg.tok_rows * cb.BLOCK_K + 1024)
                    + cfg.xb_stages * cfg.tok * 512
                    + min(cfg.epi_chunk, cfg.tok) * 128 * 4
                    <= cb.DEC_SMEM_CAP
                )
            else:
                assert cfg.xq_stages == 0 and not re.search(r"_x\d", cfg.kernel_key)
        else:
            assert cfg.xb_stages == 0 and not re.search(r"_r\d", cfg.kernel_key)
            assert not cfg.xbh and "_xh" not in cfg.kernel_key
            assert not cfg.qer and "_qe" not in cfg.kernel_key
            assert cfg.xq_stages == 0 and not re.search(r"_x\d", cfg.kernel_key)
        assert cfg.qlanes in (4, 8, 16)
        assert (2 * cfg.tok) % (cfg.qwarps * (32 // cfg.qlanes)) == 0
        if not (cfg.fused and not cfg.resident) or "qlanes" not in entry:
            assert cfg.qlanes == 16 and "_q" not in core_key
        else:
            assert cfg.qlanes >= entry["qlanes"]
        assert core_key.endswith(f"_q{cfg.qlanes}") == (cfg.qlanes != 16)


@pytest.mark.parametrize("arch", ARCHES)
def test_decode_config_round3_fused_rows(arch):
    # N = 128, K = 7168 (tp8 f_a / b_proj, tp1 f_a): the fused rows that carry the decoupled BF16 ring and narrow units.
    cfg = decode_config(4096, 1, 28, arch, SM_COUNT)
    assert (cfg.tok, cfg.stages, cfg.xb_stages, cfg.qlanes) == (32, 3, 5, 4)
    assert cfg.kernel_key == "decode:t32_p3_fused_r5_q4"
    cfg = decode_config(16384, 1, 28, arch, SM_COUNT)
    # Round 4: split 1 on 128 persistent CTAs (2 balanced work items each) replaces split 4 on 148 (6.9 items each).
    assert (cfg.tok, cfg.split, cfg.stages, cfg.xb_stages, cfg.qlanes, cfg.grid) == (
        64,
        1,
        2,
        3,
        4,
        128,
    )
    assert cfg.total_work == 256
    # Round 6 (lever P): the row prefetches its weight tiles four stages ahead into L2 (``_pf4``; 1.02x on both GPUs).
    assert cfg.pf == 4
    # Round 6 continuation 7 (lever PI-W): during the last ``pf`` stages of a work item the load warp also prefetches the
    # NEXT item's first W / SFW tiles into L2 (``_pi2``; 2 work items per CTA on this row).
    assert cfg.pfi == 2
    # Round 6 continuation 8 (lever QW16): 16 quantizing warps convert the 64-token BF16 tile of each stage (``_w16``, right
    # after ``_fused``): the same narrow (token, 128-K block) units split over twice the warps, bit-exact by construction.
    assert cfg.qwarps == 16
    # Round 6 continuation 9 (lever XBH): the BF16 ring is loaded and released per 128-K block (``_xh`` after ``_r3``); the
    # 16 warps split by K block, same units and arithmetic.
    assert cfg.xbh
    # Round 6 continuation 10 (lever QER): the quantizing warps release each half slot right after their register loads (``_qe``
    # after ``_xh``); same units, arithmetic and stores.
    assert cfg.qer
    # Round 8 pass 1 (lever PFX): ``pfx 1`` adopted on this cell on both GPUs (+2.6-3.2 %, bit-exact) -> ``_px1`` after ``_pf4``.
    assert cfg.kernel_key == "decode:t64_p2_fused_w16_r3_xh_qe_q4_pf4_px1_pi2"
    # 16-token tiles cannot keep eight 4-lane groups busy per stage: the table's 4 lanes widen to 8, coupled staging.
    # Round 5: the 24-tile M = 256 row moves to a 4-CTA cluster split-K route (each CTA owns a quarter of K, FP32 partials
    # are exchanged through distributed shared memory in one round); the small dedicated inbox is used (no aliasing).
    # Round 9 (K2, tier A' rule-5 packets k2x2_sm_100a / k2x5_sm_103a): the cluster widens to 6 CTAs on both GPUs (each
    # CTA owns a sixth of K; the same one-round DSM exchange, bit-exact; the table keeps ``split`` 4 and the config takes
    # the cluster width as its split), so ``split`` = ``csplit`` = 6 and the key carries ``_cs6``.
    cfg = decode_config(256, 1, 28, arch, SM_COUNT)
    assert (cfg.tok, cfg.split, cfg.csplit, cfg.cs_alias, cfg.fused) == (
        16,
        6,
        6,
        False,
        True,
    )
    # Round 7 (lever XR): the 4-slot BF16 ring and the 4-slot FP8 token ring run next to the 4-stage W ring of
    # this instance (4 x 33792 + 4 x 9216 + 4 x 8192 B + the 8 KB epilogue chunk; the 7-line cluster inbox fits beside them);
    # 1.015-1.022x on both GPUs in 7 own-process rounds, bit-exact.
    # Round 8 (convergence passes): the cell runs 8 quantizer lanes on both GPUs (``_q8``); on sm_103a the BF16 ring is a
    # two-stage half-slot ring with the early release (``_r2_xh_qe``) and the token tile is prefetched one stage ahead
    # (``_px1`` after ``_cs4``). See the Cake design record, round 8.  Round 9 (K2): with the 6-CTA inbox the SMEM clamp
    # keeps three W stages on sm_100a (4-slot rings; ``_p3``) and four on sm_103a (2-slot ring; ``_p4``).
    xb, module_stages, key = {
        "sm_100a": (4, 3, "decode:t16_p3_fused_r4_x4_q8_cs6"),
        "sm_103a": (2, 4, "decode:t16_p4_fused_r2_xh_qe_x4_q8_cs6_px1"),
    }[arch]
    assert (cfg.xb_stages, cfg.xq_stages, cfg.module_stages, cfg.qlanes) == (
        xb,
        4,
        module_stages,
        8,
    )
    assert cfg.kernel_key == key


@pytest.mark.parametrize("arch", ARCHES)
def test_decode_config_round6_rules(arch):
    # Round 6 (levers P + M): the 48-50-tile M = 256 rows pair the two 128-token m tiles of one N tile in a 2-CTA
    # cluster that shares the W stage through TMA multicast and prefetches W three stages ahead into L2.
    cfg = decode_config(256, 50, 28, arch, SM_COUNT)
    assert (cfg.tok, cfg.split, cfg.csplit, cfg.mc, cfg.pf) == (128, 1, 1, 2, 3)
    assert cfg.m_tiles == 2 and cfg.tiles == 100 and cfg.total_work == 50
    assert cfg.grid == 2 * min(50, SM_COUNT // 2, cb.decode_cluster_capacity(arch, 2))
    assert cfg.kernel_key == "decode:t128_p3_mc2_pf3"
    # Bucket edge: an M whose m-tile count does not fill whole clusters (65..128 rows -> one 128-token tile) falls back
    # to the plain instance -- multicast off and no prefetch (the fallback is not a tabulated route).
    cfg = decode_config(65, 50, 28, arch, SM_COUNT)
    assert (cfg.tok, cfg.m_tiles, cfg.mc, cfg.pf) == (128, 1, 1, 0)
    assert cfg.kernel_key == "decode:t128_p3"
    # Lever M64 (round-6 continuation 6): the 48- / 50-tile M = 64 buckets take the 32-token tile with the 5-stage ring,
    # the 16-row epilogue chunk and W prefetch two stages ahead (96 / 100 CTAs instead of 48 / 50; bit-exact with the
    # 64-token route).
    cfg = decode_config(64, 50, 28, arch, SM_COUNT)
    assert (cfg.tok, cfg.stages, cfg.module_stages, cfg.epi_chunk, cfg.mc, cfg.pf) == (
        32,
        5,
        5,
        16,
        1,
        2,
    )
    assert cfg.m_tiles == 2 and cfg.tiles == 100
    assert cfg.kernel_key == "decode:t32_p5_c16_pf2"
    cfg = decode_config(64, 48, 28, arch, SM_COUNT)
    assert (cfg.tok, cfg.stages, cfg.epi_chunk, cfg.pf) == (
        32,
        5,
        16,
        2,
    ) and cfg.tiles == 96
    assert cfg.kernel_key == "decode:t32_p5_c16_pf2"
    # Lever D: the 12-tile M = 256 rows pin a 5-stage ring with 16-row epilogue flushes.
    cfg = decode_config(256, 12, 28, arch, SM_COUNT)
    assert (cfg.tok, cfg.stages, cfg.module_stages, cfg.epi_chunk) == (32, 5, 5, 16)
    assert cfg.kernel_key == "decode:t32_p5_c16"
    # Lever GP: only the tabulated GEMM-routed row prefetches (tp1 q_proj M = 256); other GEMM shapes do not.
    assert cb.gemm_prefetch_distance(256, 96, 28, arch) == 2
    assert cb.gemm_prefetch_distance(4096, 96, 28, arch) == 0
    assert cb.gemm_prefetch_distance(257, 12, 28, arch) == 0


@pytest.mark.parametrize("arch", ARCHES)
def test_decode_config_round6_continuation_rules(arch):
    # Lever E1: ``tstore`` rows launch the TMA-store epilogue program (``_tso``) only on a 16-byte-aligned output view; the
    # plain key (register epilogue) is the same instance otherwise.  Split-K / cluster rows never carry the flag.
    n_tstore = 0
    for key, entry in decode_table(arch).items():
        if entry["route"] != "decode":
            continue
        n_tiles128, num_k_iters, bucket = (int(v) for v in key.split(","))
        cfg = decode_config(bucket, n_tiles128, num_k_iters, arch, SM_COUNT)
        assert cfg.tstore == bool(entry.get("tstore", False))
        assert not cfg.kernel_key.endswith("_tso")
        if cfg.tstore:
            n_tstore += 1
            assert cfg.split == 1 and cfg.csplit == 1 and cfg.tok >= 32
            # ``_tso`` precedes the ``_dpf`` / ``_pi<N>`` / ``_xp`` tail of the key (round 8: the 1,28,16384 cell carries
            # ``pfi``, the sm_103a 96,1,256 cell ``dpf``).
            head, tail = re.fullmatch(
                r"(.*?)((?:_dpf)?(?:_pi\d+)?(?:_xp)?)", cfg.kernel_key
            ).groups()
            assert cfg.kernel_key_for(True) == f"{head}_tso{tail}"
        assert cfg.kernel_key_for(False) == cfg.kernel_key
    # Round 8 (knob-catalog sweep, passes 1-5) adopted further bit-exact tstore cells on both GPUs, and pass 14 adopted
    # ``tstore`` on the 24,2,512 cell of both architectures (27 / 29); see the Cake design record, round 8 (R8-14 and
    # the pass records).  Round 9 (K2, rule-5 packets): the sm_103a 5,28,512 cell moved to a 3-CTA cluster split-K route,
    # which has no TMA-store epilogue, so its ``tstore`` flag is 0 (27 / 28).  Round 10 (S1): the sm_103a 5,28,2048 cell
    # moved to the cluster split-K GEMM route (``gemm_cs`` 2), leaving the decode table (27 / 27).
    assert n_tstore == {"sm_100a": 27, "sm_103a": 27}[arch]
    # Round-6 next loop (lever PX-S): eight small fused buckets per architecture prefetch their BF16 token tile one stage
    # ahead of its TMA load (``pfx: 1`` -> the ``_px1`` program); a prefetch changes no data path and no launch argument.
    n_pfx = 0
    for key, entry in decode_table(arch).items():
        if entry["route"] != "decode" or not entry.get("pfx"):
            continue
        n_tiles128, num_k_iters, bucket = (int(v) for v in key.split(","))
        cfg = decode_config(bucket, n_tiles128, num_k_iters, arch, SM_COUNT)
        # round 7 (lever XP): `_xp` closes the key after `_px1` on the 14-CTA cells that adopted the parallel-issue exchange
        assert (
            cfg.fused
            and not cfg.resident
            and cfg.pfx == 1
            and re.search(r"_px1(_tso)?(_dpf)?(_pi\d+)?(_xp)?$", cfg.kernel_key)
        )
        n_pfx += 1
    # Round 8 pass 1 added ``pfx 1`` on the 1,28,16384 cell (both GPUs) and on the sm_103a 1,28,64 / 1,28,256 cells.
    assert n_pfx == {"sm_100a": 9, "sm_103a": 11}[arch]
    assert decode_config(8, 24, 2, arch, SM_COUNT).kernel_key.endswith(
        "_px1"
    )  # tp8 kv_b M = 8: 1.076-1.083x B200 / 1.034-1.042x B300
    assert not decode_config(1, 24, 2, arch, SM_COUNT).kernel_key.endswith(
        "_px1"
    )  # tp8 kv_b M = 1 keeps the plain instance (B300-only win)
    cfg = decode_config(
        256, 56, 6, arch, SM_COUNT
    )  # tp8 o_proj M = 256: 1.07x on both GPUs
    # Round 8: the sm_100a cell also prefetches its weight stage (``pf: 1`` -> ``_pf1``; 1.03-1.06x B200).
    assert (
        cfg.tstore
        and cfg.kernel_key_for(True)
        == {
            "sm_100a": "decode:t128_p3_pf1_tso",
            "sm_103a": "decode:t128_p3_tso",
        }[arch]
    )
    # Round 8 (pass 14): the tp1 f_b M = 256 cell (96 N tiles, one K iteration: fused, resident, TMA store) adopted the
    # round-6 descriptor prefetch (lever Q1, table key ``dpf``) on sm_103a alone -- ``_dpf`` follows ``_tso`` in the
    # TMA-store form and closes the register form; sm_100a keeps the plain pair of the same instance.
    cfg = decode_config(256, 96, 1, arch, SM_COUNT)
    assert cfg.fused and cfg.resident and cfg.tstore and cfg.module_stages == 2
    assert cfg.dpf == (arch == "sm_103a")
    assert (
        cfg.kernel_key
        == {
            "sm_100a": "decode:t64_p2_fused_res",
            "sm_103a": "decode:t64_p2_fused_res_dpf",
        }[arch]
    )
    assert (
        cfg.kernel_key_for(True)
        == {
            "sm_100a": "decode:t64_p2_fused_res_tso",
            "sm_103a": "decode:t64_p2_fused_res_tso_dpf",
        }[arch]
    )
    # Lever C16: the M <= 64 rows that used a 12-28-way global split-K now run one cluster per output tile (14 CTAs = two
    # 256-K stages each; non-portable cluster), the 12-tile family a 7-CTA cluster (12 clusters <= capacity 15) and the
    # 17-tile family a 4-CTA cluster; the grid is whole clusters within the measured co-resident capacity.
    for M, n_tiles, c, clusters in (
        (1, 1, 14, 1),
        (64, 1, 14, 4),
        (1, 5, 14, 5),
        (1, 12, 7, 12),
        (1, 17, 4, 17),
    ):
        cfg = decode_config(M, n_tiles, 28, arch, SM_COUNT)
        assert (cfg.tok, cfg.split, cfg.csplit, cfg.fused) == (16, c, c, True), (
            M,
            n_tiles,
            cfg,
        )
        assert clusters <= cb.decode_cluster_capacity(arch, c)
        assert cfg.grid == c * clusters and cfg.total_work == cfg.grid
        assert f"_cs{c}" in cfg.kernel_key and not cfg.tstore
        # the small-inbox exchange of a 7..16-wide cluster gives up one t16 stage (2C - 1 inbox lines next to the ring)
        # Round 8: the BF16-ring / narrow-lane levers adopted on some of these cells (``_r<N>[_xh[_qe]]``, ``_q8``) sit
        # between ``_fused`` and ``_cs``; the 7-CTA cell runs four stages next to its 3-slot ring.
        assert cfg.module_stages in (3, 4), cfg
        ring = (
            f"_r{cfg.xb_stages}"
            + ("_xh" if cfg.xbh else "")
            + ("_qe" if cfg.qer else "")
            if cfg.xb_stages
            else ""
        )
        lanes = f"_q{cfg.qlanes}" if cfg.qlanes != 16 else ""
        # round-6 next loop (lever PX): the M = 8 buckets of this family carry the `_px1` suffix
        # round 7 (lever XP): the 14-CTA cells 1,28,1 / 1,28,64 / 5,28,1 close the key with `_xp` (parallel-issue DSM exchange)
        assert (
            cfg.kernel_key
            == f"decode:t16_p{cfg.module_stages}_fused{ring}{lanes}_cs{c}"
            + (f"_px{cfg.pfx}" if cfg.pfx else "")
            + ("_xp" if cfg.xp else "")
        ), cfg.kernel_key
        assert cfg.xp == ((n_tiles, M) in ((1, 1), (1, 64), (5, 1)))
    assert (
        cb.decode_cluster_capacity(arch, 14) == 7
        and cb.decode_cluster_capacity(arch, 9) == 15
    )
    # The plan carries both programs of every tstore row (the register program is the fallback of unaligned views).
    required = set(cb.required_kernel_keys(arch, SM_COUNT))
    assert (
        {
            "decode:t128_p3",
            "decode:t128_p3_tso",
            "decode:t16_p3_fused_cs14_xp",
            "decode:t16_p3_fused_cs14_px1",
            "decode:t16_p4_fused_r3_cs7",  # round 8: the 7-CTA M = 1 cell carries the 3-slot BF16 ring
        }
        <= required
    )
    assert not {
        k
        for k in required
        if k.endswith("_tso") and k.removesuffix("_tso") not in required
    }
    # ... including the TMA-store forms a lever token follows (``_tso_dpf``, ``_tso_pi<N>``): the register twin is the
    # key without its ``_tso`` token (the grammar-exact rule of the export's ``decode_register_twin``).
    assert not {
        k for k in required if "_tso" in k and k.replace("_tso", "", 1) not in required
    }
    assert "decode:t16_p4_fused_cs14" not in required
    # Round 8 (pass 14): both programs of the sm_103a dpf cell are in its plan; sm_100a ships the plain pair of that cell
    # and no ``_dpf`` program at all (the dpf cell is the only 64-token resident instance of sm_103a).
    dpf_pair = {"decode:t64_p2_fused_res_dpf", "decode:t64_p2_fused_res_tso_dpf"}
    plain_pair = {"decode:t64_p2_fused_res", "decode:t64_p2_fused_res_tso"}
    assert (dpf_pair <= required) == (arch == "sm_103a")
    assert (plain_pair <= required) == (arch == "sm_100a")
    assert any("_dpf" in k for k in required) == (arch == "sm_103a")
    # Lever L5: the N = 576 kv_a rows at M > 256 (buckets 4096 and 16384) take the 192-wide GEMM N tile: three 192-column
    # tiles stream and multiply no padded columns (1.04-1.09x on both GPUs, bit-exact with the 256-wide output); the
    # fused_qkv_a family (N = 2112) measured slower with it and stays 256-wide, as does every untabulated shape.
    narrow = {k: e for k, e in decode_table(arch).items() if "gemm_bn" in e}
    # round 7 adds measured 192-wide cells on the 512 / 1024 / 2048 buckets; the round-6 kv_a cells stay
    assert {"5,28,16384", "5,28,4096"} <= set(narrow)
    assert all(
        e["route"] == "gemm" and e["gemm_bn"] in (192, 384) for e in narrow.values()
    )
    # Round 10 (CAKE-985 S3, the 128 x 384 pair tile, ``gemm_bn`` 384): the exact ``gemm_bn`` cell sets of the table.
    # 96,28,512 (tp1 q_proj, N = 12288) adopted the 384-wide TMA-store form on both GPUs after R8-3 confirms on the
    # cell's contract row tp1:q_proj:512 (tree 7d77cb55, node alone: B200 step b0e67e6c quiet 1.0322, shared 1.0322 /
    # 1.0322; GB300 step beac25b1 quiet 1.0370, shared 1.0370 / 1.0362; bit-exact, Tier A); 48,28,1024 (tp8 fused_qkvg,
    # N = 6144) adopted it on sm_103a only (GB300 step 89f144ca quiet 1.0604, shared 1.0751 / 1.0734; Tier A' vs the
    # fix-up stream-K record, rule-5 packet PASS step a0095eec) while sm_100a keeps its 192 + pf1 record cell (the 384
    # arms read 0.966 there); 12,28,4096 (tp8 q_proj, N = 1536) adopted it on sm_103a only (GB300 step 2ceb0c33, node
    # alone: quiet 1.0357, shared 1.0349 / 1.0356; bit-exact vs the ordered stream-K record, Tier A) while sm_100a (the
    # 384 arms read 1.0065 there) replaced its ordered stream-K record with the prefetched stream-K form (round 10
    # S2, ``gemm_skp`` 4; the ``skp_rows`` block below).  A 384 cell carries none of
    # ``gemm_pf`` / ``gemm_sk`` / ``gemm_skf`` / ``gemm_sks`` / ``gemm_skp`` (the last: lever SKP, the FIN merge rule)
    # (``gemm_n384_cell_conflicts``: the instance implies ``gemm_bk`` 128, the plain CLC schedule, no weight prefetch).
    n192 = {k for k, e in narrow.items() if e["gemm_bn"] == 192}
    n384 = {k for k, e in narrow.items() if e["gemm_bn"] == 384}
    n192_common = {
        "12,1,16384",  # round 11 C25 (CAKE-985 T7, FIN tip 8): gemm_bn 192 on both GPUs after the C24 staging-slot fix
        "12,1,2048",
        "12,1,4096",
        "12,28,16384",
        "12,28,2048",
        "17,28,16384",
        "192,2,1024",
        "192,2,256",
        "24,2,1024",
        "24,2,2048",
        "48,28,4096",
        "48,28,512",
        "5,28,16384",
        "5,28,4096",
        "50,28,1024",
        "50,28,512",
        "96,1,2048",
        "96,1,512",
        "96,28,256",
    }
    # round 10 S1: sm_103a 5,28,2048 (tp1/tp8:kv_a:2048) left the decode table for the 192-wide cluster split-K
    # GEMM cell (``gemm_cs`` 2, ``gemm_bk`` 128), so it is a 192-wide cell there only.  Round 10 S6: the sm_103a
    # 50,28,512 cell (tp8:in_proj_qkvgfab:512) adopted ``gemm_bn`` 192 next to its ``gemm_bk`` 128 (GB300 node alone,
    # confirm step 88ce5919: quiet 1.053, shared 1.060 / 1.059, bit-exact, Tier A) -- the ``gemm_rstaged_n192_k128``
    # program the sm_100a cell has launched since round 8 (the 8-byte-aligned 6284-column edge keeps the staged-register
    # epilogue), so the cell is 192-wide on both architectures now.
    # Round 11 T7 pass 1 (CAKE-985): the sm_100a 17,28,16384 cell (tp1:fused_qkv_a:16384) adopted ``gemm_bn`` 192 +
    # ``gemm_bk`` 128 (B200 node alone, R8-3 confirm step 7da46b38: quiet 1.0153, shared 1.0168 / 1.0169, bit-exact, Tier A)
    # -- the ``gemm_tstore_n192_k128`` program of its sm_103a twin (there since round 9, with ``gemm_skf`` 1), so the cell is
    # 192-wide on both architectures now.
    # Round 11 T3 (CAKE-985 U-T3-1): the OOB-safe admissibility rule (``gemm_n192_admissible``: every 32-row B box of the partial
    # last tile stored or fully out of range) admits the 192-wide tile on the o_proj rows (7168 columns); the sm_103a cells
    # 56,48,512 / 56,6,512 / 56,48,1024 adopted ``gemm_bn`` 192 (GB300 node alone, R8-3 step b2c9faab: quiet 1.0826 / 1.0233,
    # shared 1.0907, 1.0921 / 1.0181, 1.0197; S5 step 91f753b8: 1.0703 x3; all bit-exact, Tier A).  Round 11 T4-U2: the sm_103a
    # 96,28,16384 cell (tp1:q_proj:16384) is a new 384-wide cell (R8-3 step 2eb3a4c0: quiet 1.0193, shared 1.0173 / 1.0182,
    # bit-exact) with a record-default sm_100a twin.
    # Round 11 C17 (CAKE-985 T3, Tier A'): the sm_100a 56,48,1024 cell (tp1:o_proj:1024) carries ``gemm_bn`` 192 next to its
    # ``gemm_bk`` 128 + ``gemm_skf`` 1 (B200 node alone, R8-3 step cb326b20 on tree 5d76afd2: quiet 1.0651, shared 1.0675 / 1.0675,
    # controls in band; rule-5 refined 5(a) packet PASS step acc7795e), program ``gemm_tstore_n192_k128_skf`` -- the program the
    # 5,28,16384 cell launches on both GPUs, so no new instance.
    assert (
        n192
        == n192_common
        | {
            "sm_100a": {"48,28,1024", "56,48,1024"},
            "sm_103a": {
                "24,2,4096",
                "5,28,2048",
                "50,28,4096",
                "56,48,512",
                "56,6,512",
                "56,48,1024",
                "12,28,4096",
            },
        }[arch]
    )
    # Round 11 T4-U2 (C11): the sm_100a 384,28,16384 (tp1:fused_qkvg:16384) and 386,28,16384 (tp1:in_proj_qkvgfab:16384) cells
    # are new 384-wide cells (B200 node alone, power-capped rows, R8-3 step fcb28e62: quiet 1.0208 / 1.0131, shared 1.0282, 1.0237 /
    # 1.0203, 1.0195, bit-exact, Tier A) with record-default sm_103a twins (the GB300 arms read below the bar there).  Round 11
    # T4-U2 (C12): the sm_103a 56,48,16384 cell (tp1:o_proj:16384) is a new 384-wide cell (GB300 node alone, R8-3 re-confirm step
    # 895173e9: quiet 1.0363, shared 1.0332 / 1.0349, controls in band, bit-exact, Tier A) with a record-default sm_100a twin.
    # Round 11 T3 screen2 (C13): the sm_103a 12,28,4096 cell (tp8:q_proj:4096, N = 1536 = 8 x 192) left the 384-wide tile for
    # ``gemm_bn`` 192 + ``gemm_bk`` 128 (GB300 node alone, R8-3 step 4ae5004a on tree 16a5ad81: quiet 1.0208, shared 1.0208 /
    # 1.0215, controls in band, bit-exact, Tier A), so it is a 192-wide cell on sm_103a now (sm_100a keeps ``gemm_skp`` 4).
    assert (
        n384
        == {
            "sm_100a": {"96,28,512", "384,28,16384", "386,28,16384"},
            "sm_103a": {"96,28,512", "48,28,1024", "96,28,16384", "56,48,16384"},
        }[arch]
    )
    for key in sorted(n384):
        cell = narrow[key]
        assert cell["route"] == "gemm" and cell["gemm_bn"] == 384
        assert not any(
            cell.get(k)
            for k in ("gemm_pf", "gemm_sk", "gemm_skf", "gemm_sks", "gemm_skp")
        )
        assert cb.gemm_n384_cell_conflicts(cell) == []
    assert cb.gemm_block_n(512, 96, 28, arch, 12288, 12288) == 384
    assert (
        cb.gemm_block_n(384, 96, 28, arch, 12288, 12288) == 384
    )  # the 512 bucket (M 257-512) is one cell
    assert (
        cb.gemm_block_n(1024, 48, 28, arch, 6144, 6144)
        == {"sm_100a": 192, "sm_103a": 384}[arch]
    )
    assert (
        cb.gemm_block_n(4096, 12, 28, arch, 1536, 1536)
        == {"sm_100a": 256, "sm_103a": 192}[arch]
    )  # round 11 C13: the sm_103a cell is 192-wide (was 384 since round 10 S3)
    # round 10 S6: the 192 storage-fit rule passes on the tp8 in_proj_qkvgfab edge (33 x 192 = 6336 <= 6400 stored
    # columns) on both GPUs, and the cell's staged-register 192-wide 128-K program is in both plans
    assert cb.gemm_block_n(512, 50, 28, arch, 6284, 6400) == 192
    assert "gemm_rstaged_n192_k128" in required
    # round 11 (CAKE-985 T3, U-T3-1): ``gemm_n192_admissible`` -- the storage-fit way in is untouched, and the OOB-safe rule
    # (``n_valid % 32 == 0 and n_pad == n_tiles128 x 128``: every 32-row B box of the partial last tile is stored or fully
    # out of range) admits the N = 7168 o_proj families (38 x 192 = 7296 > 7168 stored rows) and the N = 49376 tp1 in_proj
    # family (258 x 192 = 49536 > 49408); a valid edge off the 32-row boundary or an inconsistent padding stays refused.  No
    # cell of those families carries ``gemm_bn`` yet, so the rule alone selects no new program on either host.
    assert cb.gemm_n192_admissible(6284, 6400, 50)
    assert cb.gemm_n192_admissible(576, 768, 5)
    assert cb.gemm_n192_admissible(7168, 7168, 56) and cb.gemm_n192_admissible(
        49376, 49408, 386
    )
    assert not cb.gemm_n192_admissible(7160, 7168, 56)
    assert not cb.gemm_n192_admissible(250, 256, 2)
    assert not cb.gemm_n192_admissible(7168, 7232, 56)
    # Round 11 T3 adoption (C8 810285f1): the sm_103a cells 56,48,512 / 56,6,512 / 56,48,1024 carry ``gemm_bn`` 192 (GB300
    # node alone, R8-3 step b2c9faab / S5 step 91f753b8, bit-exact); every other cell of those families stays 256-wide -- the
    # B200 screen (step c55c988b) read 0.73 / 0.89 / not bit-exact there -- except the 384-wide T4-U2 cells of these families
    # (the ``n384`` inventory above): sm_100a 386,28,16384 (C11) and sm_103a 56,48,16384 (C12), neither of them 192-wide.
    # Round 11 C17: the sm_100a 56,48,1024 cell joined them (Tier A', R8-3 step cb326b20 + packet step acc7795e; see above).
    n192_o_proj = {
        "sm_100a": {"56,48,1024"},
        "sm_103a": {"56,48,512", "56,6,512", "56,48,1024"},
    }[arch]
    n384_in_proj = {"sm_100a": {"386,28,16384"}, "sm_103a": {"56,48,16384"}}[arch]
    assert {
        k
        for k, e in decode_table(arch).items()
        if k.startswith(("56,6,", "56,48,", "386,28,")) and e.get("gemm_bn") == 192
    } == n192_o_proj
    assert {
        k
        for k, e in decode_table(arch).items()
        if k.startswith(("56,6,", "56,48,", "386,28,")) and e.get("gemm_bn") == 384
    } == n384_in_proj
    assert (
        not {
            k
            for k, e in decode_table(arch).items()
            if k.startswith(("56,6,", "56,48,", "386,28,")) and "gemm_bn" in e
        }
        - n192_o_proj
        - n384_in_proj
    )
    for bucket, n_tiles128, num_k_iters in (
        (512, 56, 6),
        (512, 56, 48),
        (1024, 56, 48),
        (2048, 56, 48),
        (16384, 56, 48),
    ):
        key = f"{n_tiles128},{num_k_iters},{bucket}"
        assert cb.gemm_block_n(bucket, n_tiles128, num_k_iters, arch, 7168, 7168) == (
            192 if key in n192_o_proj else 384 if key in n384_in_proj else 256
        )
    # round 10 S6 / S5 v2 (CAKE-985): the GEMM-cell adoptions of the FIN confirms, all bit-exact (Tier A).  sm_103a (GB300
    # 964581 node alone): 192,2,2048 (tp1:kv_b:2048, gemm_bk 128; R8-3 confirm step 856eec52: quiet 1.0164, shared 1.0229 /
    # 1.0218; a new cell, so sm_100a reads the 256-K default for the key), 96,1,4096 (tp1:f_b:4096, gemm_bk 128; re-run
    # step 6bb7e8f4: quiet 1.0167, shared 1.0292 / 1.0278; K = 128 is one 128-K block, so the key's K index stays
    # ceil(128 / 256) = 1 and the cell's program is the ``gemm_tstore_k128_s2_ts4`` form next to its r9 G1 stage / slot keys
    # -- the program the sm_100a cell has launched since round 9) and 144,6,256 (tp1:q_b:256, gemm_bk 128; S5 v2
    # interleaved confirm step e752af34: 1.025 / 1.0273 / 1.0273).  sm_100a (B200 2226889 exclusive node): 144,6,256
    # (gemm_bk 128; S5 v2 step d89bcdb7: 1.0261 / 1.0240 / 1.0239, so the cell is 128-K on both GPUs now) and 24,2,16384
    # (tp8:kv_b:16384, gemm_stages 2 / gemm_ts_slots 4 -- the r9 K6 arm G1 pair; R8-3 confirm step 9cd4283a: quiet 1.0242,
    # shared 1.0271 / 1.0280 -- then gemm_bk 128 too, adoption 5: node-alone R8-3 confirm step 30e8940b on B200 2228368,
    # quiet 1.026, shared 1.029 / 1.028, so the cell launches the ``gemm_tstore_k128_s2_ts4`` program the sm_103a twin has
    # launched since round 9).  No new program: the ``_k128`` and
    # ``_s2_ts4`` TMA-store forms are in both plans already.
    # Round 11 C17 (CAKE-985 T7): the sm_100a 192,2,2048 cell adopted ``gemm_bk`` 128 too (B200 node alone, R8-3 step cb326b20 on
    # tree 5d76afd2: quiet 1.0157, shared 1.0168 / 1.0178, controls in band, bit-exact, Tier A), so both GPUs launch
    # ``gemm_tstore_k128`` / ``gemm_rstaged_k128`` there now.
    assert cb.gemm_block_k(2048, 192, 2, arch) == 128
    assert cb.gemm_block_k(4096, 96, 1, arch) == 128 and cb.gemm_stages_ts_slots(
        4096, 96, 1, arch
    ) == (2, 4)
    assert cb.gemm_block_k(256, 144, 6, arch) == 128
    assert cb.gemm_stages_ts_slots(16384, 24, 2, arch) == (2, 4)
    # round 11 C18 (CAKE-985 T7 pass 3): the sm_103a 96,1,512 cell (tp1:f_b:512, gemm_bn 192, 256-K) carries the stage / slot pair
    # too (GB300 S5 v2 step d6567748; program gemm_tstore_n192_s2_ts4); the sm_100a cell keeps gemm_bn 192 + gemm_bk 128
    assert (
        cb.gemm_stages_ts_slots(512, 96, 1, arch)
        == {"sm_100a": (3, 1), "sm_103a": (2, 4)}[arch]
    )
    assert cb.gemm_block_k(512, 96, 1, arch) == {"sm_100a": 128, "sm_103a": 256}[arch]
    # round 11 C25 (CAKE-985 T7, FIN tip 8): the 12,1,16384 cell (tp8:f_b:16384, 256-K, gemm_stages 2 / gemm_ts_slots 4 since
    # round 9) adopted gemm_bn 192 on BOTH GPUs once the FIN tip-7 fix C24 (TMA-store staging-slot wait covers the 192-wide
    # drain) made the n192 x s2_ts4 composition exact: S5 v2 on the fixed tree, node alone, bit-exact -- B200 step ee27c6e0
    # 1.02372 / 1.02367 / 1.02372, GB300 step fc1e9fd3 1.01575 / 1.01772 / 1.01575; program gemm_tstore_n192_s2_ts4
    assert cb.gemm_stages_ts_slots(16384, 12, 1, arch) == (2, 4)
    assert cb.gemm_block_n(16384, 12, 1, arch, 1536, 1536) == 192
    # round 11 C20 (CAKE-985 T7 pass 4; GB300 S5 v2 step 23316854 on tree 66fec20d, mode ab 3 x 1000, all bit-exact, Tier A): the sm_103a
    # 1,28,2048 cell (tp1/tp8 b_proj:2048 + f_a:2048) runs 16 quantizing warps (``qwarps`` 16 -> ``_w16`` right after ``_fused``: the same
    # 8-lane units and per-128-block reduction split over twice the warps; 1.04054 x3 / 1.04027 x3, aa 1.0) and the sm_103a 192,2,64 cell
    # (tp1:kv_b:64) prefetches its first work item's W stream two stages deep before the PDL wait (``pw`` 2 -> ``_pw2`` in the ``_pf<D>``
    # slot; 1.02326 / 1.01938 / 1.01938).  sm_100a keeps the 192,2,64 cell.  ``decode:t64_p2_fused_r2_xh_q8_cs4_xp`` leaves the sm_103a plan
    # (the 1,28,2048 cell was its only sm_103a producer); ``decode:t64_p4_tso`` stays (144,6,64 / 24,2,256 / 384,28,64) next to the new
    # ``_pw2`` pair (the register twin ``decode:t64_p4_pw2`` reaches the export plan through the coverage row ``tp1_kv_b_m64_fb``).
    # round 11 C21 (CAKE-985 T7 pass 4; B200 exclusive node, S5 v2 step 65837e54 on tree 66fec20d, mode ab 3 x 1000, 1965 MHz): the sm_100a
    # 1,28,2048 cell runs 16 quantizing warps too (tp1:b_proj 1.04015 / 1.03797 / 1.04017, tp1:f_a 1.04 / 1.03895 / 1.04206, tp8:b_proj 1.03797 /
    # 1.03797 / 1.03593, tp8:f_a 1.04 / 1.04007 / 1.04, aa 0.998-1.002, bit-exact, Tier A), so the ``_w16`` program is in both plans and the plain
    # ``decode:t64_p2_fused_r2_xh_q8_cs4_xp`` is in neither (each GPU's 1,28,2048 cell was its only producer).
    cfg_2048 = decode_config(2048, 1, 28, arch, SM_COUNT)
    assert (
        cfg_2048.qwarps,
        cfg_2048.qlanes,
        cfg_2048.xb_stages,
        cfg_2048.xbh,
        cfg_2048.csplit,
        cfg_2048.xp,
    ) == (16, 8, 2, True, 4, True)
    assert cfg_2048.kernel_key == "decode:t64_p2_fused_w16_r2_xh_q8_cs4_xp"
    assert "decode:t64_p2_fused_r2_xh_q8_cs4_xp" not in required
    assert "decode:t64_p2_fused_w16_r2_xh_q8_cs4_xp" in required
    cfg_kvb64 = decode_config(64, 192, 2, arch, SM_COUNT)
    assert (
        cfg_kvb64.tok,
        cfg_kvb64.split,
        cfg_kvb64.fused,
        cfg_kvb64.tstore,
        cfg_kvb64.pw,
    ) == (64, 1, False, True, {"sm_100a": 0, "sm_103a": 2}[arch])
    assert (
        cfg_kvb64.kernel_key
        == {"sm_100a": "decode:t64_p4", "sm_103a": "decode:t64_p4_pw2"}[arch]
    )
    assert (
        cfg_kvb64.kernel_key_for(True)
        == {"sm_100a": "decode:t64_p4_tso", "sm_103a": "decode:t64_p4_pw2_tso"}[arch]
    )
    assert {"decode:t64_p4", "decode:t64_p4_tso"} <= required
    assert ({"decode:t64_p4_pw2", "decode:t64_p4_pw2_tso"} <= required) == (
        arch == "sm_103a"
    )
    assert (
        cb.gemm_block_k(16384, 24, 2, arch) == 128
    )  # round 10 adoption 5: 128-K on both GPUs
    # Round 11 T6 (CAKE-985): the static grid-stride ``_gs`` cells (``gemm_gs`` 1 next to ``gemm_bk`` 128; TMA-store rows only, the
    # staged-register rows keep the ``_k128`` program).  C14 (sm_103a, GB300 S5 v2 step a5e13a03): 96,1,2048 / 12,1,4096 / 24,2,1024 /
    # 24,2,2048 on the 192-wide form.  C17 (sm_100a, B200 node alone, R8-3 step cb326b20 + S5 v2 step 76fc5ec4 on tree 5d76afd2, all
    # bit-exact, Tier A): the same four cells plus 17,28,2048 (tp1/tp8:fused_qkv_a:2048), 56,6,2048 (tp8:o_proj:2048), 24,2,4096
    # (tp8:kv_b:4096) and 56,6,512 (tp8:o_proj:512) on the 256-wide form -- ``gemm_tstore_k128_gs`` was an sm_100a program only until
    # C20.  C20 (sm_103a, T6-U3e; GB300 node alone, form-change screen step d20d3aa8 + S5 v2 step 5f04d5ef on tree 69baa8e0, bit-exact,
    # Tier A): 12,1,2048 (tp8:f_b:2048; gemm_bn 192 + gemm_bk 128 + gemm_gs 1, 1.02041 x3, the no-GS counterpart 1.0 x3) and 18,6,2048
    # (tp8:q_b:2048; gemm_bk 128 + gemm_gs 1 next to its quant_units 2, 1.01643 x3, the counterpart 1.00932 x3) left their round-7
    # ``gemm_pf`` 2 behind, so ``gemm_tstore_k128_gs`` is in both plans now and ``gemm_tstore_n192_pf2`` is an sm_100a program only
    # (12,28,2048 / 48,28,512); ``gemm_tstore_pf2`` stays on sm_103a through 144,6,512 / 144,6,1024 / 144,6,2048 / 17,28,2048 / 386,28,512 /
    # 56,6,2048 / 56,6,4096.
    # C19: the sm_100a 192,2,512 cell (tp1:kv_b:512) joined on the 256-wide form and left its round-7 ``gemm_pf`` 2 behind (the
    # ``_gs`` form has no prefetch composition; B200 node alone, T6 U3d screen step 5f247cc4 on tree 6f3dbb1f, S5 v2 step 4c54d65e
    # on tree 66fec20d: 1.01987 / 1.01762 / 1.01762, aa 1.0 x3, bit-exact, Tier A; gemm_tstore_pf2 -> gemm_tstore_k128_gs).
    # ``gemm_tstore_pf2`` stays in both plans: sm_100a 144,6,512 / 144,6,1024 / 56,48,512 and the sm_103a pf-2 256-wide cells.
    gs_rows = {k for k, e in decode_table(arch).items() if e.get("gemm_gs")}
    assert (
        gs_rows
        == {
            "sm_100a": {
                "17,28,2048",
                "56,6,2048",
                "96,1,2048",
                "12,1,4096",
                "24,2,1024",
                "24,2,2048",
                "24,2,4096",
                "56,6,512",
                "192,2,512",
            },
            "sm_103a": {
                "96,1,2048",
                "12,1,4096",
                "24,2,1024",
                "24,2,2048",
                "12,1,2048",
                "18,6,2048",
            },
        }[arch]
    )
    assert (
        cb.gemm_prefetch_distance(512, 192, 2, arch) == 0
        and cb.gemm_block_k(512, 192, 2, arch) == 128
    )  # both GPUs: 128-K, no prefetch
    assert "gemm_tstore_pf2" in required
    assert all(
        e["route"] == "gemm"
        and e.get("gemm_bk") == 128
        and cb.gemm_gs_cell_conflicts(e) == []
        for e in (decode_table(arch)[k] for k in gs_rows)
    )
    assert "gemm_tstore_n192_k128_gs" in required
    assert (
        "gemm_tstore_k128_gs" in required
    )  # C20: both GPUs (sm_103a through 18,6,2048)
    # C20 pins: both cells are 128-K without prefetch on sm_103a; sm_100a keeps 12,1,2048 on gemm_bk 128 + gemm_static 1 (no prefetch) and
    # 18,6,2048 on the record defaults (256-K, no prefetch)
    assert (
        cb.gemm_prefetch_distance(2048, 12, 1, arch) == 0
        and cb.gemm_block_k(2048, 12, 1, arch) == 128
    )
    assert (
        cb.gemm_prefetch_distance(2048, 18, 6, arch) == 0
        and cb.gemm_block_k(2048, 18, 6, arch) == {"sm_100a": 256, "sm_103a": 128}[arch]
    )
    assert ("gemm_tstore_n192_pf2" in required) == (arch == "sm_100a")
    # Round 11 C22 (CAKE-985 T7 pass 4, Tier A): the sm_100a 56,48,1024 cell (tp1:o_proj:1024) prefetches its W tiles one stage ahead
    # too (``gemm_pf`` 1 next to its ``gemm_bn`` 192 + ``gemm_bk`` 128 + ``gemm_skf`` 1; B200 node alone, R8-3 confirm step ab7a66fd on
    # tree 66fec20d: quiet 1.0195, shared 1.0188 / 1.0181, controls in band, bit-exact; 70.112 -> 68.816 us) -- program
    # ``gemm_tstore_n192_k128_skf_pf1`` (Cake symbol ``kimi_k3_fp8_proj_gemm_tstore_pf1_n192_k128_skf``), new to the sm_100a plan; the
    # sm_103a cell has prefetched there since round 11 T7 pass 2 (``gemm_tstore_n192_k128_pf1``); the staged-register twin is unchanged.
    assert (
        cb.gemm_prefetch_distance(1024, 56, 48, arch) == 1
        and cb.gemm_block_k(1024, 56, 48, arch) == 128
    )
    assert ("gemm_tstore_n192_k128_skf_pf1" in required) == (arch == "sm_100a")
    # Round 11 C23 (CAKE-985 T7 passes 3/4, Tier A by construction): the sm_100a 18,6,64 cell (tp8:q_b:64) prefetches its first work
    # item's W stream two stages deep before the PDL wait (``pw`` 2; B200 S5 v2 interleaved confirms t7p3 step c6c3991a 1.02336 / 1.02336 /
    # 1.02351 and t7p4 step 65837e54 1.02381 x3 on tree 66fec20d, aa 1.0, bit-exact, symbols stable; 7.14 -> 6.72 us) -- program
    # ``decode:t16_p4_pw2`` replaces ``decode:t16_p4`` in the sm_100a plan (this cell is its only producer); the sm_103a cell is unchanged.
    cfg_qb64 = decode_config(64, 18, 6, arch, SM_COUNT)
    assert (
        cfg_qb64.tok,
        cfg_qb64.split,
        cfg_qb64.fused,
        cfg_qb64.tstore,
        cfg_qb64.pw,
    ) == (16, 1, False, False, {"sm_100a": 2, "sm_103a": 0}[arch])
    assert (
        cfg_qb64.kernel_key
        == {"sm_100a": "decode:t16_p4_pw2", "sm_103a": "decode:t16_p4"}[arch]
    )
    assert ("decode:t16_p4_pw2" in required) == (arch == "sm_100a")
    assert ("decode:t16_p4" in required) == (arch == "sm_103a")
    assert cb.gemm_k128_cell_epilogues(192, 2, 2048) == (
        True,
        True,
    )  # the 2048 bucket holds an 8-byte-stride row
    assert cb.gemm_k128_cell_epilogues(96, 1, 4096) == (
        True,
        False,
    ) and cb.gemm_k128_cell_epilogues(144, 6, 256) == (True, False)
    assert cb.gemm_k128_cell_epilogues(24, 2, 16384) == (
        True,
        True,
    )  # the 16384 bucket too; its staged rows keep the untokened program
    assert {
        "gemm_tstore_k128",
        "gemm_tstore_k128_s2_ts4",
        "gemm_tstore_s2_ts4",
    } <= required
    # round 10 S5 v2, full3 (CAKE-985): the sm_103a 18,6,4096 cell (tp8:q_b:4096, gemm_bk 128 since round 9) prefetches its W
    # tiles one stage ahead (``gemm_pf`` 1; GB300 node alone, step 19f5ae7d: 1.0168 x3, aa 1.0, bit-exact; the full1 confirm
    # e752af34 read 1.0142 / 1.0168 / 1.0134, borderline).  Its program ``gemm_tstore_k128_pf1`` (Cake symbol
    # ``kimi_k3_fp8_proj_gemm_tstore_pf1_k128``) is a new instance of the sm_103a plan; sm_100a keeps its plain cell there.
    # Round 11 T7 pass 1 (CAKE-985): the sm_100a cell prefetches too (``gemm_pf`` 1 on its 256-K program; B200 node alone,
    # R8-3 confirm step 7da46b38: quiet 1.009, shared 1.0167 / 1.0167, bit-exact, Tier A) -- ``gemm_tstore_pf1``, the program
    # the sm_100a 144,6,4096 cell has launched since round 8; the ``gemm_bk`` 128 arm of the same pass failed its confirm.
    # Round 11 T7 pass 2 (C15): the sm_100a cell adopted ``gemm_bk`` 128 next to its ``gemm_pf`` 1 after all (B200 node alone,
    # R8-3 confirm step 3e09b4fa on tree 16a5ad81: quiet 1.0168, shared 1.0185 / 1.0185, controls in band, bit-exact, Tier A),
    # so both GPUs launch ``gemm_tstore_k128_pf1`` there now; ``gemm_tstore_pf1`` stays in the plan for the 144,6,4096 cell.
    q_b_4096 = decode_table(arch)["18,6,4096"]
    assert (q_b_4096.get("gemm_pf", 0), q_b_4096.get("gemm_bk", 256)) == (1, 128)
    assert cb.gemm_k128_cell_epilogues(18, 6, 4096) == (True, False)
    assert "gemm_tstore_k128_pf1" in required
    assert "gemm_tstore_pf1" in required
    # round 10 S5 v2 (CAKE-985), the sm_100a decode-cell adoptions (B200 2226889 exclusive node, step d89bcdb7, Tier A
    # bit-exact): 1,28,1024 (the fused 6-CTA cluster split-K cell of the tp*:{b_proj,f_a}:1024 rows) prefetches its W tiles
    # one stage ahead into L2 (table key ``pf`` 1 = the env knob LOOM_KIMI_DECODE_PF; ``_pf1`` after ``_cs6``; 1.0266 /
    # 1.0242-1.0266 / 1.0264 on tp1:b_proj / tp1:f_a / tp8:b_proj), and 5,28,512 (the 3-CTA cluster split-K kv_a cell) pins
    # the quantization launch width ``quant_units`` 2 (1.0185-1.0209 on tp1 / tp8:kv_a:512; the decode program is unchanged)
    # -- the width the sm_103a cell has carried since round 9 (its ``acc_bufs`` 2 companion was not confirmed on sm_100a and
    # is not adopted).
    cs6_1024 = decode_config(1024, 1, 28, arch, SM_COUNT)
    assert (cs6_1024.fused, cs6_1024.csplit, cs6_1024.pf, cs6_1024.pfi) == (
        True,
        6,
        {"sm_100a": 1, "sm_103a": 0}[arch],
        0,
    )
    assert ("_pf1" in cs6_1024.kernel_key) == (arch == "sm_100a")
    assert cb.table_quant_units(512, 5, 28, arch) == 2
    assert (
        cb.decode_table_entry(512, 5, 28, arch).get("acc_bufs")
        == {"sm_100a": None, "sm_103a": 2}[arch]
    )
    sm_100a_fused_qkvg_1024 = decode_table("sm_100a")["48,28,1024"]
    assert (
        sm_100a_fused_qkvg_1024["gemm_bn"] == 192
        and sm_100a_fused_qkvg_1024.get("gemm_pf") == 1
    )
    # the 384-wide programs the adopted cells launch: always the ``_k128`` TMA-store form on these 16-byte-aligned
    # column edges; the staged-register form only where the cell's bucket carries 8-byte-stride correctness rows
    assert "gemm_tstore_n384_k128" in required
    assert ("gemm_rstaged_n384_k128" in required) == any(
        cb.gemm_k128_cell_epilogues(*(int(v) for v in key.split(",")))[1]
        for key in n384
    )
    for M in (4096, 4097, 16384):
        assert cb.gemm_block_n(M, 5, 28, arch, 576, 768) == 192
    # round 7: M = 257 follows the kv_a 512 cell (the decode route, so the GEMM tile width is the default)
    cell_512 = cb.decode_table_entry(512, 5, 28, arch)
    assert cell_512 is not None and cell_512["route"] == "decode"
    assert cb.gemm_block_n(257, 5, 28, arch, 576, 768) == 256
    assert cb.gemm_block_n(256, 5, 28, arch, 576, 768) == 256  # tabulated decode row
    assert (
        cb.gemm_block_n(4096, 17, 28, arch, 2112, 2304) == 256
    )  # fused_qkv_a keeps 256
    assert cb.gemm_block_n(4096, 96, 28, arch, 12288, 12288) == 256
    # the narrow tile is refused when its padded N would read past the stored 256-padded rows
    assert cb.gemm_block_n(4096, 5, 28, arch, 250, 256) == 256
    # Round 11 C17 (CAKE-985): sm_100a keeps no 192-wide 256-K cell without prefetch (96,1,2048 / 24,2,1024 / 24,2,2048 took
    # ``gemm_bk`` 128 + ``gemm_gs`` 1; 12,28,2048 / 48,28,512 / 48,28,1024 prefetch, so their TMA-store rows launch the ``_pf<D>``
    # form), and the mirror lists the plain ``gemm_tstore_n192`` only for a cell without prefetch / stream-K lever (export-plan
    # parity: ``required_kernel_keys``).  Round 11 C18 (CAKE-985 T7 pass 3): sm_103a 96,1,512 (tp1:f_b:512, its last such cell)
    # adopted ``gemm_stages`` 2 / ``gemm_ts_slots`` 4 (GB300 node alone, S5 v2 step d6567748 on tree cea637bf: 1.02827 / 1.03191 /
    # 1.02827, aa 1.0 x3, bit-exact, Tier A) and launches ``gemm_tstore_n192_s2_ts4`` -- a new program of the sm_103a plan -- so the
    # plain 192-wide TMA-store program is in neither plan now (12,28,2048 / 56,6,512 prefetch; 12,1,2048 prefetched until C20 moved it
    # to the ``_gs`` form on gemm_bk 128).  The staged-register
    # 192-wide program stays on both GPUs (the 8-byte-stride rows of the 192-wide 256-K cells).
    assert "gemm_rstaged_n192" in required
    assert "gemm_tstore_n192" not in required
    # Round 11 C25 (CAKE-985 T7, FIN tip 8): 12,1,16384 (tp8:f_b:16384) adopted ``gemm_bn`` 192 on both GPUs after the C24
    # staging-slot fix (B200 S5 v2 step ee27c6e0 1.02372 / 1.02367 / 1.02372; GB300 step fc1e9fd3 1.01575 / 1.01772 / 1.01575;
    # bit-exact, Tier A), so ``gemm_tstore_n192_s2_ts4`` is in BOTH plans now (sm_103a since C18 through 96,1,512).
    assert "gemm_tstore_n192_s2_ts4" in required
    # Round 6 continuation 12 (lever SKO): the tabulated ``gemm_sk`` rows launch the ordered stream-K instance when the
    # launch has one full wave of CTA pairs plus at most half a wave of tail tiles: the head pair of each tail tile runs
    # the first K half and hands its FP32 partial to the tail pair, which continues the same accumulation (bit-exact).
    # A cell whose ordered form was replaced keeps the flag as ``gemm_sk: 0`` next to its new form (round 9: the sm_100a
    # 56,48,4096 cell moved to the fix-up form), so the flag is read for truth, as the host does.
    sk_rows = {k: e for k, e in decode_table(arch).items() if e.get("gemm_sk")}
    # round 7 adds measured stream-K cells on the 512 / 1024 / 2048 buckets; every cell's window opens at its bucket.
    # Round 8 (R8-19): the sm_103a 56,48,4096 selection is pruned -- on the 152-SM part the 448-tile launch leaves a
    # 68-tile fractional wave (more than half a wave), so the ordered window never opened there (a plain cell); the
    # sm_100a cell carries exactly one stream-K form (ordered until round 9's third-wave adoption, fix-up after it).
    # Round 10 (CAKE-985 S3): the 96,28,512 cell dropped its ordered stream-K flag on both GPUs when it adopted the
    # 384-wide pair tile (``gemm_n384_cell_conflicts``); the 12,28,4096 cell dropped it on sm_103a when that cell
    # adopted the 384-wide tile too (R8-3 step 2ceb0c33) and on sm_100a when it adopted the prefetched stream-K form
    # (S2, ``gemm_skp`` 4, exclusive-node confirm step 32b8cc76): the ordered record is in no architecture's sk_rows
    # now -- sm_100a carries the SKP cell, sm_103a the 384-wide cell.
    assert "12,28,4096" not in sk_rows and "96,28,512" not in sk_rows
    o_proj_4096 = decode_table(arch).get("56,48,4096", {})
    assert (bool(o_proj_4096.get("gemm_sk")) != bool(o_proj_4096.get("gemm_skf"))) == (
        arch == "sm_100a"
    )
    assert all(e["route"] == "gemm" and e["gemm_sk"] == 1 for e in sk_rows.values())
    sm_count = SM_COUNTS[arch]
    for key in sk_rows:
        n_tiles128, num_k_iters, bucket = (int(v) for v in key.split(","))
        plan = cb.gemm_stream_k_plan(
            bucket,
            n_tiles128,
            num_k_iters,
            arch,
            sm_count,
            cb._m_tiles(bucket),
            n_tiles128 // 2,
        )
        assert (
            plan is not None
            and plan.pairs == sm_count // 2
            and plan.grid == 2 * plan.dp
        )
        expected_ks = (
            int(sk_rows[key].get("gemm_sk_ksplit", 0)) or (num_k_iters + 1) // 2
        )
        assert 0 < plan.rem <= plan.pairs - plan.rem and plan.ksplit == expected_ks
        assert 2 <= plan.ksplit <= num_k_iters - 2
        assert plan.dp + plan.rem == (cb._m_tiles(bucket) // 2) * (n_tiles128 // 2)
    # 96 tiles over 74 pairs would plan 22 tail tiles with head 16 of 28 K iterations (table key gemm_sk_ksplit,
    # continuation 13), but since round 10 the 12,28,4096 cell carries ``gemm_sk`` on neither architecture (sm_100a:
    # ``gemm_skp`` 4, sm_103a: ``gemm_bn`` 384), so the ordered planner reads no flag there: plain schedule, as below
    # one wave or untabulated
    assert cb.gemm_stream_k_plan(4096, 12, 28, arch, 148, cb._m_tiles(4096), 6) is None
    # 448 tiles over 74 pairs: 6 full waves (888 CTAs, one pair per data-parallel tile) + 4 tail tiles, head 28 of 48 (table key gemm_sk_ksplit)
    assert cb.gemm_stream_k_plan(4096, 56, 48, arch, 148, cb._m_tiles(4096), 28) == (
        cb.StreamKPlan(74, 4, 28, 444, 888) if "56,48,4096" in sk_rows else None
    )
    assert cb.gemm_stream_k_plan(2048, 12, 28, arch, 148, cb._m_tiles(2048), 6) is None
    assert cb.gemm_stream_k_plan(4096, 96, 28, arch, 148, cb._m_tiles(4096), 48) is None
    # the 256-K ordered stream-K program is required exactly where a ``gemm_sk`` cell without ``gemm_bk`` 128 remains
    # (round 10 S3: the sm_103a 12,28,4096 cell moved to the 384-wide tile; S2: the sm_100a 12,28,4096 cell -- the last
    # 256-K ordered cell -- moved to the prefetched form, so no architecture requires ``gemm_tstore_sk`` now; sm_103a
    # 56,48,1024 launches the ``_k128_sk`` form); the staged-register ordered form is never required
    assert ("gemm_tstore_sk" in required) == any(
        int(e.get("gemm_bk", 256)) != 128 for e in sk_rows.values()
    )
    assert "gemm_rstaged_sk" not in required
    # Round 10 (CAKE-985 S2, lever SKP): the tabulated ``gemm_skp`` rows launch the prefetched stream-K instance
    # (``_skp`` / ``_skp4`` / ``_skp8`` by the cell's slot bound) when the geometry admits a plan.  The sm_100a 12,28,4096
    # cell (tp8 q_proj, N = 1536) replaced its ordered stream-K record (``gemm_sk`` 1 / ``gemm_sk_ksplit`` 16) with
    # ``gemm_skp`` 4 after the exclusive-node R8-3 confirm (B200 step 32b8cc76: quiet 1.031, shared 1.029 / 1.029; Tier
    # A-prime -- exact FP32 partial sums in a fixed cross-CTA order, rule-5 packet c942d29f PASS; synccheck 991d1ba3 /
    # racecheck c5bb1bff clean); the sm_103a cell carries the 384-wide tile instead (no ``_skp`` form of it).  The key
    # is exclusive with ``gemm_sk`` / ``gemm_skf`` / ``gemm_sks`` and with ``gemm_bn`` 384 on one cell
    # (``gemm_stream_k_prefetch`` refuses such a cell; the Cake loader refuses such a table).
    skp_rows = {k: e for k, e in decode_table(arch).items() if e.get("gemm_skp")}
    assert set(skp_rows) == {"sm_100a": {"12,28,4096"}, "sm_103a": set()}[arch]
    for key, cell in skp_rows.items():
        n_tiles128, num_k_iters, bucket = (int(v) for v in key.split(","))
        assert cell["route"] == "gemm" and cell["gemm_skp"] in cb.GEMM_SKP_SLOT_FORMS
        assert not any(
            cell.get(k) for k in ("gemm_sk", "gemm_sk_ksplit", "gemm_skf", "gemm_sks")
        )
        assert int(cell.get("gemm_bn", cb.BLOCK_N)) != 384
        assert (
            cb.gemm_stream_k_prefetch(bucket, n_tiles128, num_k_iters, arch)
            == cell["gemm_skp"]
        )
    # tp8:q_proj:4096 on B200 (148 SMs = 74 resident pairs): 16 m pairs x 6 n tiles = 96 CTA-pair tiles = one full wave
    # + 22 tail tiles -> regime F with S = min(74 // 22, cap + 1, 28 // 3) = 3 pieces per tail tile (two contributor
    # slots, cuts 0 / 9 / 18 / 28), the 74 data-parallel tiles through the CLC ring first (grid 148 CTAs, finishers
    # after dp_cap = 1 tile), 88 flags and 22 x 2 x 256 KiB of partials; the 1-slot form admits no plan there (S would
    # be 2, the ordered form's domain), so the cell's program is ``gemm_tstore_skp4``.  sm_103a reads no key: plain.
    q_proj_4096 = cb.gemm_stream_k_prefetch_plan(
        4096, 12, 28, arch, 148, cb._m_tiles(4096), 6, 256
    )
    if arch == "sm_100a":
        assert q_proj_4096 is not None and q_proj_4096.program_suffix == "_skp4"
        assert (
            q_proj_4096.regime,
            q_proj_4096.pairs,
            q_proj_4096.first,
            q_proj_4096.sk_tiles,
        ) == ("F", 74, 74, 22)
        assert (
            q_proj_4096.S,
            q_proj_4096.slots,
            q_proj_4096.inst_slots,
            q_proj_4096.dp_cap,
        ) == (3, 2, 4, 1)
        assert (q_proj_4096.grid, q_proj_4096.cuts, q_proj_4096.flag_count) == (
            148,
            (0, 9, 18, 28),
            88,
        )
        assert (
            q_proj_4096.partial_bytes == 22 * 2 * cb.gemm_skf_tile_bytes(256)
            and q_proj_4096.makespan == 38
        )
    else:
        assert q_proj_4096 is None
    assert {k for k in required if "_skp" in k} == (
        {"gemm_tstore_skp4"} if arch == "sm_100a" else set()
    )
    # Round 6 continuation 17/18 (lever SKF): the tabulated ``gemm_skf`` rows launch the fix-up stream-K instance when
    # the launch has a fractional wave: the last full wave plus the fraction (118 / 119 tiles x 28 K iterations) is cut
    # into 74 equal K ranges; the pair finishing a tile adds the other contributors' FP32 partials in ordinal order
    # (reduction order of those tiles differs from the plain schedule; accepted by the user, max_abs_err 0.03125).
    skf_rows = {k: e for k, e in decode_table(arch).items() if e.get("gemm_skf")}
    # round 7 adds measured fix-up stream-K cells on the 512 / 1024 / 2048 buckets; the round-6 cells stay
    assert {"384,28,256", "386,28,256", "5,28,16384"} <= set(skf_rows)
    assert all(
        e["route"] == "gemm" and e["gemm_skf"] == 1 and not e.get("gemm_sk")
        for e in skf_rows.values()
    )
    # 192 tiles (1 x 192 or 128 x 3 / 2) over 74 pairs: 2 full waves + 44 -> SK region = 118 tiles x 28 = 3304 K
    # iterations from tile 74 on, 44.6 iterations per pair, at most one contributor slot per tile, grid 148 CTAs
    assert cb.gemm_stream_k_fixup_plan(
        256, 384, 28, arch, 148, cb._m_tiles(256), 192, 256
    ) == (
        cb.StreamKFixupPlan(
            74, 3304, 1, 74, 118, 148, 118 * cb.gemm_skf_tile_bytes(256)
        )
    )
    assert cb.gemm_stream_k_fixup_plan(
        256, 386, 28, arch, 148, cb._m_tiles(256), 193, 256
    ) == (
        cb.StreamKFixupPlan(
            74, 3332, 1, 74, 119, 148, 119 * cb.gemm_skf_tile_bytes(256)
        )
    )
    assert cb.gemm_stream_k_fixup_plan(
        16384, 5, 28, arch, 148, cb._m_tiles(16384), 3, 192
    ) == (
        cb.StreamKFixupPlan(
            74, 3304, 1, 74, 118, 148, 118 * cb.gemm_skf_tile_bytes(192)
        )
    )
    assert cb.gemm_skf_tile_bytes(256) == cb.GEMM_SK_TILE_BYTES
    # untabulated shapes, the ordered-form rows and whole waves keep the plain / ordered schedule
    assert (
        cb.gemm_stream_k_fixup_plan(4096, 5, 28, arch, 148, cb._m_tiles(4096), 3, 192)
        is None
    )
    assert (
        cb.gemm_stream_k_fixup_plan(4096, 12, 28, arch, 148, cb._m_tiles(4096), 6, 256)
        is None
    )
    # another M of the same 256 bucket (2 M tiles) takes the same split; a whole-wave SM count keeps the plain schedule
    assert cb.gemm_stream_k_fixup_plan(
        200, 384, 28, arch, 148, cb._m_tiles(200), 192, 256
    ) == cb.gemm_stream_k_fixup_plan(
        256, 384, 28, arch, 148, cb._m_tiles(256), 192, 256
    )
    assert (
        cb.gemm_stream_k_fixup_plan(256, 384, 28, arch, 96, cb._m_tiles(256), 192, 256)
        is None
    )
    # The kv_a 16384 cell (5,28,16384: ``gemm_bn`` 192 + ``gemm_skf``) names the 192-wide fix-up program; round 9 (K6) adds
    # ``gemm_bk`` 128 to that cell on both GPUs, so its program carries ``_k128`` before ``_skf`` (round 8, lever L8).
    kv_a_16384 = decode_table(arch)["5,28,16384"]
    assert kv_a_16384.get("gemm_skf") and kv_a_16384.get("gemm_bn") == 192
    n192_skf = (
        "gemm_tstore_n192"
        + ("_k128" if kv_a_16384.get("gemm_bk") == 128 else "")
        + "_skf"
    )
    assert {"gemm_tstore_skf", n192_skf} <= required
    assert not any(
        k.startswith(("gemm_rstaged", "gemm_n192", "gemm_pf")) and k.endswith("_skf")
        for k in required
    )


def test_decode_module_stage_clamp():
    # 128-token unfused: 4 stages of 66 KB do not fit the pool -> 3.
    assert decode_module_stages(128, 4, False, False) == 3
    # Round 8 (table key ``epi_groups``): a second epilogue warp group doubles the 16 KB staging tile of the 64/128-token
    # tiles (t128: 2 stages; t64: 3 stages of 50 KB); below 64 tokens the dispatcher keeps one group.
    assert decode_module_stages(128, 4, False, False, epi_groups=2) == 2
    assert decode_module_stages(64, 4, False, False) == 4
    assert decode_module_stages(64, 4, False, False, epi_groups=2) == 3
    assert decode_module_stages(32, 4, False, False, epi_groups=2) == 4
    # 16-token fused: the smallest instance keeps every requested stage.
    assert decode_module_stages(16, 4, True, False) == 4
    # Resident 64-token tiles: the stages hold only W + scales, four resident slots still fit four.
    assert decode_module_stages(64, 4, True, True) == 4
    # Host rule for the same instance: the staged-BF16 stage size admits two stages -> a p2 module.
    assert decode_module_stages(64, 2, True, True) == 2
    # Decoupled ring: the ring bytes leave the pool before the stage clamp (t32: 3 x 42 KB + 5 x 16 KB; t64: 2 x 50 KB + 3 x 32 KB).
    assert decode_module_stages(32, 3, True, False, 5) == 3
    assert decode_module_stages(64, 2, True, False, 3) == 2
    assert decode_module_stages(64, 3, True, False, 3) == 2


def test_decode_kernel_key_round8_suffixes():
    # Round 8 (levers ACC_BUFS / EPI_GROUPS): ``_a<n>`` / ``_e<n>`` sit right after the ``_fused[_w<n>]`` block and before
    # ``_res`` / the ring fields (the Cake symbol order) and are omitted at their default of 1, so every earlier key is unchanged.
    assert (
        decode_kernel_key(64, 4, False, False, csplit=2, cs_alias=True)
        == "decode:t64_p4_cs2a"
    )
    assert (
        decode_kernel_key(64, 4, False, False, csplit=2, cs_alias=True, acc_bufs=2)
        == "decode:t64_p4_a2_cs2a"
    )
    assert (
        decode_kernel_key(64, 3, False, False, csplit=2, cs_alias=True, epi_groups=2)
        == "decode:t64_p3_e2_cs2a"
    )
    assert (
        decode_kernel_key(128, 2, False, False, acc_bufs=2, epi_groups=2, tstore=True)
        == "decode:t128_p2_a2_e2_tso"
    )
    assert (
        decode_kernel_key(64, 2, True, True, qwarps=16, acc_bufs=2)
        == "decode:t64_p2_fused_w16_a2_res"
    )
    assert (
        decode_kernel_key(
            64, 2, True, False, xb_stages=3, qlanes=4, acc_bufs=2, epi_groups=2
        )
        == "decode:t64_p2_fused_a2_e2_r3_q4"
    )
    assert (
        decode_kernel_key(64, 4, False, False, acc_bufs=1, epi_groups=1)
        == "decode:t64_p4"
    )
    # Round 6 (lever Q1; adopted round 8): ``_dpf`` follows ``_tso`` and precedes ``_pi<n>`` / ``_xp`` (the Cake symbol
    # order); it is omitted when the row does not pin ``dpf``, so every earlier key is unchanged.
    assert (
        decode_kernel_key(64, 2, True, True, dpf=True) == "decode:t64_p2_fused_res_dpf"
    )
    assert (
        decode_kernel_key(64, 2, True, True, tstore=True, dpf=True)
        == "decode:t64_p2_fused_res_tso_dpf"
    )
    assert (
        decode_kernel_key(
            128, 2, False, False, acc_bufs=2, pf=3, tstore=True, dpf=True, pfi=1
        )
        == "decode:t128_p2_a2_pf3_tso_dpf_pi1"
    )
    assert (
        decode_kernel_key(64, 3, False, False, csplit=2, dpf=True, xp=True)
        == "decode:t64_p3_cs2_dpf_xp"
    )
    # Round 8 (lever PW; adopted round 11 C20 on the sm_103a 192,2,64 cell): ``_pw<D>`` follows ``_pf<D>`` and precedes ``_px<D>`` /
    # ``_tso`` (the Cake symbol order); the register twin drops ``_tso`` alone; omitted at 0, so every earlier key is unchanged.
    assert (
        decode_kernel_key(64, 4, False, False, pw=2, tstore=True)
        == "decode:t64_p4_pw2_tso"
    )
    assert decode_kernel_key(64, 4, False, False, pw=2) == "decode:t64_p4_pw2"
    assert (
        decode_kernel_key(
            128, 3, False, False, mc=2, pf=1, pw=2, tstore=True, dpf=True, pfi=1
        )
        == "decode:t128_p3_mc2_pf1_pw2_tso_dpf_pi1"
    )
    assert (
        decode_kernel_key(64, 4, False, False, pw=0, tstore=True) == "decode:t64_p4_tso"
    )


@pytest.mark.parametrize("arch", ARCHES)
def test_decode_config_round8_rules(arch):
    # Round 8 (levers ACC_BUFS / EPI_GROUPS / QUANT_UNITS): every table cell's levers reach the resolved config and the
    # kernel key; two epilogue groups only on the 64/128-token tiles; the quantization width of a two-launch cell is read
    # from the cell (decode and GEMM routes) and must divide the K blocks.
    for key, entry in decode_table(arch).items():
        n_tiles128, num_k_iters, bucket = (int(v) for v in key.split(","))
        assert cb.table_quant_units(bucket, n_tiles128, num_k_iters, arch) == int(
            entry.get("quant_units", 0)
        )
        cfg = decode_config(bucket, n_tiles128, num_k_iters, arch, SM_COUNT)
        if cfg is None:
            assert entry["route"] == "gemm"
            continue
        assert cfg.acc_bufs == int(entry.get("acc_bufs", 1)) and cfg.acc_bufs in (1, 2)
        assert cfg.epi_groups == (
            int(entry.get("epi_groups", 1)) if cfg.tok >= 64 else 1
        )
        assert cfg.epi_groups in (1, 2)
        assert cfg.quant_units == int(entry.get("quant_units", 0))
        head = re.match(
            r"decode:t\d+_p\d+(?:_fused(?:_w\d+)?)?(_a\d+)?(_e\d+)?", cfg.kernel_key
        )
        assert head is not None
        assert head.group(1) == (f"_a{cfg.acc_bufs}" if cfg.acc_bufs != 1 else None), (
            cfg.kernel_key
        )
        assert head.group(2) == (
            f"_e{cfg.epi_groups}" if cfg.epi_groups != 1 else None
        ), cfg.kernel_key
        assert re.search(r"_[ae]\d", cfg.kernel_key[head.end() :]) is None
        assert cfg.kernel_key_for(True).startswith(head.group(0))
    # The tp1 o_proj M = 64 cell (56 N tiles, 48 K iterations): one cluster split-K instance on both GPUs, two TMEM
    # accumulator sets on sm_103a alone -- a program of its own that the plan of that architecture must carry.
    cfg = decode_config(64, 56, 48, arch, SM_COUNT)
    assert (cfg.tok, cfg.split, cfg.csplit, cfg.cs_alias, cfg.module_stages) == (
        64,
        2,
        2,
        True,
        4,
    )
    assert cfg.acc_bufs == {"sm_100a": 1, "sm_103a": 2}[arch]
    assert (
        cfg.kernel_key
        == {"sm_100a": "decode:t64_p4_cs2a", "sm_103a": "decode:t64_p4_a2_cs2a"}[arch]
    )
    assert (
        cfg.kernel_key_for(True) == cfg.kernel_key
    )  # cluster split-K rows have no TMA-store epilogue
    assert cfg.kernel_key in required_kernel_keys(arch, SM_COUNT)
    assert decode_config(9, 56, 48, arch, SM_COUNT).kernel_key == cfg.kernel_key
    assert "_a" not in decode_config(1, 56, 48, arch, SM_COUNT).kernel_key
    # Per-cell quantization width: the tp1 o_proj M = 256 cell (K = 12288: 96 K blocks) pins 4 (sm_100a) / 2 (sm_103a)
    # K blocks per half warp where the rule gives 1; a width that does not divide the K blocks is a table error.
    assert cb.table_quant_units(256, 56, 48, arch) == {"sm_100a": 4, "sm_103a": 2}[arch]
    assert quant_units(256, 96, SM_COUNT) == 1
    assert (
        cb.launch_quant_units(
            256, 96, cb.table_quant_units(256, 56, 48, arch), SM_COUNT
        )
        == {"sm_100a": 4, "sm_103a": 2}[arch]
    )
    assert cb.launch_quant_units(256, 96, 0, SM_COUNT) == quant_units(256, 96, SM_COUNT)
    assert cb.launch_quant_units(4096, 96, 0, SM_COUNT) == quant_units(
        4096, 96, SM_COUNT
    )
    with pytest.raises(ValueError, match="does not divide"):
        cb.launch_quant_units(256, 96, 5, SM_COUNT)
    # A GEMM-route cell carries the width too (tp8 o_proj M = 2048 on sm_103a: 56 N tiles, 6 K iterations, 12 K blocks).
    assert cb.table_quant_units(2048, 56, 6, arch) == {"sm_100a": 0, "sm_103a": 2}[arch]
    if arch == "sm_103a":
        assert cb.decode_table_entry(2048, 56, 6, arch)["route"] == "gemm"
        assert decode_config(2048, 56, 6, arch, SM_COUNT) is None
        assert cb.launch_quant_units(2048, 12, 2, SM_COUNT) == 2
    # Every width a cell pins is a registered quantization key of that architecture's plan.
    required = set(required_kernel_keys(arch, SM_COUNT))
    for entry in decode_table(arch).values():
        if entry.get("quant_units"):
            assert f"quant:u{int(entry['quant_units'])}" in required


@pytest.mark.parametrize("arch", ARCHES)
def test_required_kernel_keys_are_registered_when_programs_exist(arch):
    required = required_kernel_keys(arch, SM_COUNT)
    assert "gemm" in required and "quant:u1" in required and "quant:u4" in required
    assert any(key.startswith("decode:") for key in required)
    # Round 8 (pass 14): a cell that pins ``dpf`` puts both of its ``_dpf`` programs into the plan (the register form
    # and, on its TMA-store rows, the ``_tso_dpf`` form); three such cells on sm_103a (the pass-14 cell plus the
    # probe-confirmed round-8 adoptions 96,28,1 and 96,28,8 = tp1 q_proj M 1 / 8), none on sm_100a.
    dpf_cells = 0
    for key, entry in decode_table(arch).items():
        if entry.get("route") != "decode" or not entry.get("dpf"):
            continue
        n_tiles128, num_k_iters, bucket = (int(v) for v in key.split(","))
        cfg = decode_config(bucket, n_tiles128, num_k_iters, arch, SM_COUNT)
        assert cfg.dpf and "_dpf" in cfg.kernel_key, key
        assert cfg.kernel_key in required and cfg.kernel_key_for(True) in required
        dpf_cells += 1
    assert dpf_cells == {"sm_100a": 0, "sm_103a": 3}[arch]
    assert any("_dpf" in key for key in required) == (dpf_cells > 0)
    if MODULES:
        missing = sorted(key for key in required if not route_available(arch, (key,)))
        assert not missing, f"{arch} lacks {missing}"
        quant_programs = set()
        for key in required:
            name, defines = kernel_program(arch, key)
            assert arch in MODULES[name]["arches"]
            if key.startswith("quant:"):
                quant_programs.add(name)
                assert dict(defines) == {
                    "QUANT_UNITS": int(key.removeprefix("quant:u"))
                }
            else:
                assert defines == ()
        # The three quantization widths are one program, specialized on the compile line.
        assert len(quant_programs) == 1
        # The registry is the union of the per-architecture key maps (round 7): every architecture's required keys
        # are registered and no registered key is unused by every architecture.
        assert set(required) <= set(KERNELS)
        assert set(KERNELS) == set().union(
            *(required_kernel_keys(a, SM_COUNT) for a in ARCHES)
        )


# ---------------------------------------------------------------------------
# GPU correctness
# ---------------------------------------------------------------------------


def _gpu_arch():
    if not torch.cuda.is_available():
        return None
    return SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(0))


def _require_program():
    arch = _gpu_arch()
    if arch is None:
        pytest.skip("the Kimi-K3 FP8 projection requires an SM100/SM103 GPU")
    device = torch.device("cuda", 0)
    if not cb.generated_program_available(device):
        pytest.skip(
            f"no generated Kimi-K3 FP8 projection program registered for {arch}"
        )
    return device


def _make_case(tp, module, M, stride_pad, device, seed):
    n_valid, K = PROJECTION_FAMILIES[tp][module]
    weight, scale = make_weight(n_valid, K, device, seed)
    x = make_activation(M, K, device, seed)
    buf = torch.full(
        (M, n_valid + stride_pad), float("nan"), dtype=torch.bfloat16, device=device
    )
    return weight, scale, x, buf, buf[:, :n_valid], n_valid


def _run_case(tp, module, M, stride_pad, seed):
    device = _require_program()
    weight, scale, x, buf, out, n_valid = _make_case(
        tp, module, M, stride_pad, device, seed
    )
    prepared = prepare_kimi_k3_fp8_projection_weights(weight, scale, n_valid)
    workspace = allocate_kimi_k3_fp8_projection_workspace(prepared, M)
    runner = prepare_kimi_k3_fp8_projection(x, prepared, out, workspace)
    result = runner()
    torch.cuda.synchronize()
    assert result is out
    expected = reference(x, weight, scale, n_valid)
    assert_matches(out, expected)
    if stride_pad:
        assert torch.isnan(buf[:, n_valid:].float()).all()
    return runner, prepared, x, out, expected


@pytest.mark.parametrize("tp,module,M,stride_pad", GPU_ROWS)
def test_projection_matches_reference(tp, module, M, stride_pad):
    runner, _prepared, _x, _out, _expected = _run_case(
        tp, module, M, stride_pad, seed=622
    )
    plan = runner.plan
    assert plan.route in ("decode", "gemm")
    if plan.route == "gemm":
        aligned = cb.gemm_tma_store_eligible(
            _out.data_ptr(), _out.stride(0), _prepared.n_valid
        )
        # Round 5: 8-byte-aligned (but not TMA-store-eligible) output rows take the staged register epilogue.
        staged = cb.gemm_reg_staged_eligible(
            aligned, cb._store_vec(_out.data_ptr(), _out.stride(0))
        )
        expected_kernel = (
            "gemm_tstore" if aligned else ("gemm_rstaged" if staged else "gemm")
        )
        if plan.gemm_cs is not None:
            # Round 10 (CAKE-985 S1, lever _cs<S>): a tabulated ``gemm_cs`` cell launches the cluster split-K program of
            # the 192-wide 128-K TMA-store / staged-register epilogue (one cluster of 2S CTAs per output tile); round 11
            # (T1a): its ``_pdlb`` form first when the cell carries ``gemm_pdlb``.  No other token composes with it.
            assert expected_kernel in ("gemm_tstore", "gemm_rstaged")
            assert cb.gemm_cluster_split(
                M, _prepared.n_tiles128, _prepared.num_k_iters, plan.arch
            )
            cs_stem = expected_kernel + "_n192_k128" + plan.gemm_cs.program_suffix
            assert plan.kernels[-1] in (cb._pdlb_stem(cs_stem), cs_stem)
            assert plan.gemm_pdlb == plan.kernels[-1].endswith("_pdlb")
            assert (plan.gemm_bn, plan.gemm_bk) == (192, 128)
            assert plan.gemm_n_tiles == -(-_prepared.n_valid // 192)
            assert runner.launch_count == 2 and plan.kernels[0].startswith("quant:u")
            return
        # Round 6 (lever L5): the tabulated ``gemm_bn`` rows launch the 192-wide instance of the same epilogue over
        # ceil(n_valid / 192) N tiles; lever GP adds the prefetch distance of the tabulated aligned rows.
        bn = cb.gemm_block_n(
            M,
            _prepared.n_tiles128,
            _prepared.num_k_iters,
            plan.arch,
            _prepared.n_valid,
            _prepared.n_pad,
        )
        # Round 10 (CAKE-985 S3): the tabulated ``gemm_bn`` 384 rows launch the 128 x 384 pair-tile instance of the
        # TMA-store / staged-register epilogues (``_n384_k128``: always the 128-K ring, no weight prefetch and no stage /
        # slot / stream-K token, as ``gemm_n384_cell_conflicts`` enforces); the predicated register epilogue keeps 256.
        n384 = bn == cb.GEMM_BLOCK_N_WIDE and expected_kernel != "gemm"
        wide = "_n192" if bn == 192 else ("_n384" if n384 else "")
        gpf = (
            cb.gemm_prefetch_distance(
                M, _prepared.n_tiles128, _prepared.num_k_iters, plan.arch
            )
            if aligned and not n384
            else 0
        )
        # Round 8 (lever L8): the tabulated ``gemm_bk`` rows launch the ``_k128`` form (128-K operand hand-off, same output
        # bits) of the TMA-store / staged-register epilogues; the plain register program has no such form.
        bk = (
            128
            if n384
            else cb.gemm_block_k(
                M, _prepared.n_tiles128, _prepared.num_k_iters, plan.arch
            )
            if expected_kernel in ("gemm_tstore", "gemm_rstaged")
            else cb.BLOCK_K
        )
        k128 = "_k128" if bk == 128 else ""
        # Round 9 (K6 arm G1): the tabulated ``gemm_stages`` / ``gemm_ts_slots`` rows launch the ``_s<n>`` / ``_ts<n>`` form
        # of the TMA-store program (two operand stages, four staging slots; same output bits); the register epilogues
        # keep the (3, 1) programs.  The tokens follow ``_k128`` and precede the stream-K suffix.
        stages, slots = (
            cb.gemm_stages_ts_slots(
                M, _prepared.n_tiles128, _prepared.num_k_iters, plan.arch
            )
            if expected_kernel == "gemm_tstore"
            else (cb.GEMM_NUM_STAGES, cb.GEMM_TS_SLOTS)
        )
        isfx = cb.gemm_instance_suffix(stages, slots)
        # Round 10 (CAKE-985 S3, lever _g64): the tabulated ``gemm_gm`` 64 rows launch the raster-group form of the TMA-store
        # / staged-register programs (the token sits after the bn token, before ``_k128``; same output bits).
        ggm = (
            cb.gemm_group_m(M, _prepared.n_tiles128, _prepared.num_k_iters, plan.arch)
            if expected_kernel != "gemm"
            else cb.GEMM_GROUP_M
        )
        g64 = "_g64" if ggm == 64 else ""
        # Round 11 (CAKE-985 T1a, lever _pdlb): the tabulated ``gemm_pdlb`` rows launch the PDL-wait-after-weights form of
        # the TMA-store / staged-register programs of the two-launch route (never the 384-wide program); the token closes
        # the stem, after the stream-K suffix and before ``_pf<D>``.
        pdlb = (
            cb.gemm_pdlb(M, _prepared.n_tiles128, _prepared.num_k_iters, plan.arch)
            if expected_kernel != "gemm" and not n384
            else False
        )
        n_tiles_bn = _prepared.n_tiles if bn == 256 else -(-_prepared.n_valid // bn)
        # Round 11 (CAKE-985 T2-e, lever _st): the tabulated ``gemm_static`` rows launch the static one-tile form of the
        # plain 192-wide 128-K TMA-store / staged-register programs (the token takes the ``_cs<S>`` slot, after ``_k128``).
        static = (
            cb.gemm_static_tile(
                M,
                _prepared.n_tiles128,
                _prepared.num_k_iters,
                plan.arch,
                plan.sm_count,
                cb._m_tiles(M),
                n_tiles_bn,
                bn,
            )
            if expected_kernel != "gemm" and not n384
            else False
        )
        # Round 11 (CAKE-985 T6, lever _gs): the tabulated ``gemm_gs`` rows launch the static grid-stride form of the plain
        # 128-K TMA-store program at bn 192 / 256 (same slot as ``_st``; TMA-store stems of the two-launch route only).
        gs_pairs = (
            cb.gemm_gs_launch_pairs(
                M,
                _prepared.n_tiles128,
                _prepared.num_k_iters,
                plan.arch,
                plan.sm_count,
                cb._m_tiles(M),
                n_tiles_bn,
                bk == 128
                and bn in (192, 256)
                and not isfx
                and not gpf
                and not pdlb
                and not static,
            )
            if expected_kernel == "gemm_tstore" and not n384
            else 0
        )
        stem = expected_kernel + wide + g64 + k128
        slot = (
            "_st"
            if static and cb._st_eligible(stem)
            else "_gs"
            if gs_pairs and cb._gs_eligible(stem)
            else ""
        )
        stem = stem + slot + isfx
        # Round 6 continuation 12 (lever SKO): the tabulated ``gemm_sk`` rows inside the stream-K wave window launch the
        # ``_sk`` instance over 2 x dp CTAs with the hand-off area behind the activation scale tiles.
        sk = cb.gemm_stream_k_plan(
            M,
            _prepared.n_tiles128,
            _prepared.num_k_iters,
            plan.arch,
            plan.sm_count,
            cb._m_tiles(M),
            plan.gemm_n_tiles,
        )
        sk_key = cb.gemm_kernel_key(stem + "_sk", gpf)
        if sk is not None and not cb.route_available(plan.arch, (sk_key,)):
            sk = None  # only the TMA-store stream-K program ships: other views keep the plain program
        assert plan.gemm_sk == sk
        # Round 10 (CAKE-985 S2, lever SKP): the tabulated ``gemm_skp`` rows launch the prefetched stream-K instance
        # (``_skp`` / ``_skp4`` / ``_skp8`` by the cell's slot bound) when the geometry admits a plan; TMA-store stems
        # only, after the ordered form and before the fix-up form (the Cake launcher precedence).
        skp = (
            cb.gemm_stream_k_prefetch_plan(
                M,
                _prepared.n_tiles128,
                _prepared.num_k_iters,
                plan.arch,
                plan.sm_count,
                cb._m_tiles(M),
                plan.gemm_n_tiles,
                bn,
            )
            if sk is None and expected_kernel == "gemm_tstore"
            else None
        )
        skp_key = cb.gemm_kernel_key(
            stem + (skp.program_suffix if skp is not None else ""), gpf
        )
        if skp is not None and not cb.route_available(plan.arch, (skp_key,)):
            skp = None
        assert plan.gemm_skp == skp
        # Round 6 continuation 17/18 (lever SKF): the tabulated ``gemm_skf`` rows with a fractional wave launch the
        # ``_skf`` TMA-store instance (grid = 2 x the data-parallel tiles before the SK region, or one wave)
        skf = (
            cb.gemm_stream_k_fixup_plan(
                M,
                _prepared.n_tiles128,
                _prepared.num_k_iters,
                plan.arch,
                plan.sm_count,
                cb._m_tiles(M),
                plan.gemm_n_tiles,
                bn,
            )
            if sk is None and skp is None and expected_kernel == "gemm_tstore"
            else None
        )
        skf_key = cb.gemm_kernel_key(stem + "_skf", gpf)
        if skf is not None and not cb.route_available(plan.arch, (skf_key,)):
            skf = None
        assert plan.gemm_skf == skf
        full_stem = stem + (
            "_sk"
            if sk is not None
            else skp.program_suffix
            if skp is not None
            else "_skf"
            if skf is not None
            else ""
        )
        # the ``_pdlb`` form first, then the record-order program when it is not registered (route_plan's fallback)
        expected_key = cb.gemm_kernel_key(full_stem, gpf)
        if pdlb:
            pdlb_key = cb.gemm_kernel_key(cb._pdlb_stem(full_stem), gpf)
            if cb.route_available(plan.arch, (pdlb_key,)):
                expected_key = pdlb_key
        assert plan.kernels[-1] == expected_key
        launched = expected_key.split("_pf")[0]
        assert plan.gemm_pdlb == launched.endswith("_pdlb")
        assert plan.gemm_static == launched.endswith("_st")
        assert plan.gemm_gs == launched.endswith("_gs")
        assert plan.gemm_gm == (64 if "_g64" in launched else cb.GEMM_GROUP_M)
        assert (plan.gemm_bn, plan.gemm_n_tiles, plan.gemm_bk) == (
            bn,
            n_tiles_bn,
            bk,
        )
        assert (plan.gemm_stages, plan.gemm_ts_slots) == (stages, slots)
        # The prefetched stream-K instance launches its persistent CTA-pair grid (``skp.grid`` = 2 pairs, regime U,
        # or 2 x the data-parallel tiles, regime F), like the ordered / fix-up forms launch theirs; round 11 (CAKE-985 T6,
        # lever _gs): the static grid-stride form launches 2 P' CTAs (P' = min(T, sm_count // 2), its ``sk_pairs`` scalar).
        assert plan.gemm_gs_pairs == (gs_pairs if plan.gemm_gs else 0)
        assert plan.grids[-1] == (
            sk.grid
            if sk is not None
            else skp.grid
            if skp is not None
            else skf.grid
            if skf is not None
            else cb.CTA_GROUP * gs_pairs
            if plan.gemm_gs
            else cb._gemm_grid(cb._m_tiles(M), plan.gemm_n_tiles)
        )
        if sk is not None:
            assert (
                _workspace_sf_numel(runner)
                >= plan.counters_offset
                + cb.GEMM_SK_FLAG_BYTES
                + sk.rem * cb.GEMM_SK_TILE_BYTES
            )
        if skf is not None:
            assert (
                _workspace_sf_numel(runner)
                >= plan.counters_offset + cb.GEMM_SK_FLAG_BYTES + skf.partial_bytes
            )
    if plan.route == "decode":
        # Lever E1 / round 8: the TMA-store program is launched only on a 16-byte-aligned output view of a ``tstore``
        # cell whose ``_tso`` program is registered; an 8-byte-aligned row stride takes the register-epilogue twin.
        aligned = cb.gemm_tma_store_eligible(
            _out.data_ptr(), _out.stride(0), _prepared.n_valid
        )
        tso_key = plan.decode.kernel_key_for(True)
        expected_tma_store = (
            aligned
            and tso_key != plan.decode.kernel_key
            and cb.route_available(plan.arch, (tso_key,))
        )
        assert plan.decode_tma_store == expected_tma_store
        assert plan.kernels[-1] == plan.decode.kernel_key_for(plan.decode_tma_store)
        if stride_pad:
            assert not plan.decode_tma_store and not plan.kernels[-1].endswith("_tso")
    if plan.route == "decode" and plan.decode.fused:
        assert runner.launch_count == 1
    else:
        assert runner.launch_count == 2 and plan.kernels[0].startswith("quant:u")


def _workspace_sf_numel(runner) -> int:
    return int(runner.workspace.sf.numel())


def test_allocating_api_matches_reference():
    device = _require_program()
    weight, scale, x, _buf, _out, n_valid = _make_case(
        "tp8", "q_proj", 64, 0, device, seed=7
    )
    prepared = prepare_kimi_k3_fp8_projection_weights(weight, scale, n_valid)
    out = kimi_k3_fp8_projection(x, prepared)
    torch.cuda.synchronize()
    assert tuple(out.shape) == (64, n_valid)
    assert_matches(out, reference(x, weight, scale, n_valid))


def test_fused_output_views():
    device = _require_program()
    weight, scale, x, _buf, out, n_valid = _make_case(
        "tp8", "fused_qkvg", 8, 0, device, seed=11
    )
    splits = (1536, 1536, 1536, 1536)
    prepared = prepare_kimi_k3_fp8_projection_weights(
        weight, scale, n_valid, splits=splits
    )
    views = prepared.output_views(out)
    assert [v.shape[1] for v in views] == list(splits)
    kimi_k3_fp8_projection(x, prepared, out)
    torch.cuda.synchronize()
    expected = reference(x, weight, scale, n_valid)
    c = 0
    for view, width in zip(views, splits, strict=True):
        assert_matches(view, expected[:, c : c + width])
        c += width


@pytest.mark.parametrize(
    "tp,module,M,stride_pad",
    [("tp8", "q_proj", 8, 0), ("tp8", "kv_a", 64, 0), ("tp8", "q_b", 4096, 0)],
)
def test_graph_replay_follows_device_inputs(tp, module, M, stride_pad):
    """Capture once, replay with new activations written into the same buffer."""
    runner, prepared, x, out, _expected = _run_case(tp, module, M, stride_pad, seed=21)
    n_valid, K = PROJECTION_FAMILIES[tp][module]
    weight, scale = make_weight(n_valid, K, x.device, 21)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            runner()
    torch.cuda.synchronize()
    for round_index in range(3):
        x.copy_(make_activation(M, K, x.device, 1000 + round_index))
        out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        assert_matches(out, reference(x, weight, scale, n_valid))


def test_launcher_matches_prepare_path():
    """Round 7 (host path): the cached launcher reproduces the prepare() + launch() output bit for bit
    for fresh and strided output views, follows new activations, and binds only the call's tensors after the
    first call of an (M, output class)."""
    device = _require_program()
    n_valid, K = PROJECTION_FAMILIES["tp8"]["fused_qkv_a"]
    weight, scale = make_weight(n_valid, K, device, 31)
    prepared = prepare_kimi_k3_fp8_projection_weights(weight, scale, n_valid)
    launcher = cb.kimi_k3_fp8_projection_launcher(prepared, max_workspaces=2)
    for M in (8, 1024, 1025):
        x = make_activation(M, K, device, 100 + M)
        expected = kimi_k3_fp8_projection(x, prepared)
        got = launcher(x)
        torch.cuda.synchronize()
        assert torch.equal(got, expected), M
        buf = torch.full(
            (M, n_valid + 48), float("nan"), dtype=torch.bfloat16, device=device
        )
        view = buf[:, :n_valid]
        assert launcher(x, view) is view
        torch.cuda.synchronize()
        assert torch.equal(buf[:, :n_valid], expected), M
        assert torch.isnan(buf[:, n_valid:].float()).all()
        x2 = make_activation(M, K, device, 200 + M)
        assert torch.equal(launcher(x2), kimi_k3_fp8_projection(x2, prepared))
    # workspace cache: least recently used M evicted beyond max_workspaces
    assert launcher.cached_rows == (1024, 1025)
    out = torch.empty((1024, n_valid), dtype=torch.bfloat16, device=device)
    x = make_activation(1024, K, device, 3)
    launcher(x, out)
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    launcher(x, out)
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] - before == 0
    assert (
        launcher.plan(x, out).kernels
        == prepare_kimi_k3_fp8_projection(
            x, prepared, out, launcher.workspace(1024)
        ).plan.kernels
    )


def test_launcher_graph_replay_follows_device_inputs():
    device = _require_program()
    n_valid, K = PROJECTION_FAMILIES["tp8"]["q_b"]
    weight, scale = make_weight(n_valid, K, device, 41)
    prepared = prepare_kimi_k3_fp8_projection_weights(weight, scale, n_valid)
    launcher = cb.kimi_k3_fp8_projection_launcher(prepared, max_workspaces=1)
    M = 64
    x = make_activation(M, K, device, 41)
    out = torch.empty((M, n_valid), dtype=torch.bfloat16, device=device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launcher(x, out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            launcher(x, out)
    torch.cuda.synchronize()
    # another M evicts nothing the graph uses: the captured M stays cached (max_workspaces=1 grows instead)
    launcher(make_activation(8, K, device, 1))
    torch.cuda.synchronize()
    assert 64 in launcher.cached_rows
    for round_index in range(3):
        x.copy_(make_activation(M, K, device, 1000 + round_index))
        out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        assert_matches(out, reference(x, weight, scale, n_valid))


def test_launcher_rejects_bad_bindings():
    device = _require_program()
    weight, scale, x, _buf, out, n_valid = _make_case(
        "tp8", "q_proj", 16, 0, device, seed=3
    )
    prepared = prepare_kimi_k3_fp8_projection_weights(weight, scale, n_valid)
    launcher = cb.kimi_k3_fp8_projection_launcher(prepared)
    with pytest.raises(ValueError, match="contiguous bf16"):
        launcher(x.float())
    with pytest.raises(ValueError, match="unit column stride"):
        launcher(x, out.t())
    with pytest.raises(ValueError, match="max_workspaces"):
        cb.kimi_k3_fp8_projection_launcher(prepared, max_workspaces=0)


def test_launcher_receipts_keep_placeholder_output_descriptor():
    """Round 8 (W4): output views the TMA store cannot address (row stride not a multiple of 16 bytes) run
    the register-epilogue GEMM programs, whose ``OUT`` descriptor is the per-device placeholder; the receipt launcher
    must not re-encode it from the caller's view (``cuTensorMapEncodeTiled`` rejects such strides) and stays
    bit-identical to the one-shot path on the plain and the staged register epilogue."""
    device = _require_program()
    for tp, module, M, stride_pad, seed in (
        ("tp1", "kv_b", 129, 2, 61),  # 4-byte rows: plain register epilogue
        (
            "tp8",
            "in_proj_qkvgfab",
            512,
            0,
            62,
        ),  # 8-byte rows (n_valid = 6284): staged register epilogue
    ):
        weight, scale, x, buf, out, n_valid = _make_case(
            tp, module, M, stride_pad, device, seed
        )
        prepared = prepare_kimi_k3_fp8_projection_weights(weight, scale, n_valid)
        workspace = allocate_kimi_k3_fp8_projection_workspace(prepared, M)
        runner = prepare_kimi_k3_fp8_projection(x, prepared, out, workspace)
        assert not runner.plan.gemm_tma_store and runner.plan.decode is None
        runner()
        launcher = cb.kimi_k3_fp8_projection_launcher(prepared)
        bufs = [torch.full_like(buf, float("nan")) for _ in range(2)]
        for (
            b
        ) in bufs:  # two output addresses of the same alignment class share one receipt
            launcher(x, b[:, :n_valid])
        torch.cuda.synchronize()
        assert launcher.cached_receipts == 1
        # the register-epilogue GEMM receipt re-binds the flat ``out`` pointer (call tensor 1) but not the ``OUT``
        # descriptor position, which carries the per-device placeholder map
        (template,) = launcher._templates.values()
        gemm_stage = template.stages[-1]
        assert sum(1 for _pos, src in gemm_stage.rebinds if src == 1) == 1
        for b in bufs:
            assert torch.equal(b[:, :n_valid], out)
            if stride_pad:
                assert torch.isnan(b[:, n_valid:].float()).all()


@pytest.mark.parametrize(
    "tp,module,M,col_offset",
    [
        (
            "tp8",
            "q_b",
            1024,
            0,
        ),  # quant:u2 + tabulated decode TMA-store cell (sm_100a) / decode (sm_103a)
        (
            "tp8",
            "q_proj",
            4096,
            0,
        ),  # quant:u4 + ordered stream-K TMA-store GEMM (gemm_sk cell)
        (
            "tp8",
            "kv_a",
            16384,
            0,
        ),  # 192-wide fix-up stream-K TMA-store GEMM (gemm_skf + gemm_bn cell)
        (
            "tp1",
            "fused_qkvg",
            256,
            0,
        ),  # 256-wide fix-up stream-K TMA-store GEMM (gemm_skf cell)
        ("tp1", "q_proj", 256, 0),  # prefetching TMA-store GEMM (gemm_pf cell)
        (
            "tp8",
            "f_b",
            256,
            0,
        ),  # fused decode TMA-store instance (XB + OUT re-bound from one call)
        (
            "tp8",
            "fused_qkv_a",
            1025,
            8,
        ),  # 16-byte base at column 8 (data_ptr % 32 == 16): TMA-store, 8-byte stores
    ],
)
def test_launcher_matches_one_shot_on_program_classes(tp, module, M, col_offset):
    """Round 8 (W4): the receipt launcher reproduces the one-shot path bit for bit on every shipped
    program class the representative rows reach -- the stream-K / fix-up / 192-wide / prefetching TMA-store GEMMs
    (workspace hand-off areas frozen into the receipt), the fused decode TMA-store instance (four re-bound slots),
    the two-unit quantization program and an output base of the 16-byte-but-not-32-byte alignment class."""
    device = _require_program()
    n_valid, K = PROJECTION_FAMILIES[tp][module]
    weight, scale = make_weight(n_valid, K, device, 41)
    x = make_activation(M, K, device, 42)
    prepared = prepare_kimi_k3_fp8_projection_weights(weight, scale, n_valid)
    pad = 16 if col_offset else 0
    buf = torch.full(
        (M, n_valid + pad), float("nan"), dtype=torch.bfloat16, device=device
    )
    out = buf[:, col_offset : col_offset + n_valid]
    assert out.data_ptr() % 32 == (2 * col_offset) % 32
    expected = kimi_k3_fp8_projection(x, prepared)
    launcher = cb.kimi_k3_fp8_projection_launcher(prepared)
    assert launcher(x, out) is out
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
    assert torch.isnan(buf[:, :col_offset].float()).all()
    assert torch.isnan(buf[:, col_offset + n_valid :].float()).all()
    plan = launcher.plan(x, out)
    assert (
        plan.kernels
        == prepare_kimi_k3_fp8_projection(
            x, prepared, out, launcher.workspace(M)
        ).plan.kernels
    )
    x2 = make_activation(M, K, device, 43)
    assert torch.equal(launcher(x2, out), kimi_k3_fp8_projection(x2, prepared))
    assert launcher.cached_receipts == 1


def test_misaligned_activation_rejected():
    """Round 8: an activation whose base is not 16-byte aligned (a contiguous view at an odd 8-byte offset) is
    refused by the one-shot path and the launcher with a ValueError instead of a driver error at launch."""
    device = _require_program()
    n_valid, K = PROJECTION_FAMILIES["tp8"]["q_proj"]
    weight, scale = make_weight(n_valid, K, device, 51)
    prepared = prepare_kimi_k3_fp8_projection_weights(weight, scale, n_valid)
    big = torch.empty(64 * K + 8, dtype=torch.bfloat16, device=device)
    x = big[4 : 4 + 64 * K].view(64, K)
    assert x.is_contiguous() and x.data_ptr() % 16 == 8
    with pytest.raises(ValueError, match="16-byte aligned"):
        kimi_k3_fp8_projection(x, prepared)
    with pytest.raises(ValueError, match="16-byte aligned"):
        cb.kimi_k3_fp8_projection_launcher(prepared)(x)


def test_launcher_short_rows_take_tma_store():
    """Round 8: the bucket-256 GEMM cells route 65 <= M <= 256 to the GEMM; for M < 128 the TMA-store ``OUT``
    box (128 rows) is taller than the output view and the TMA unit clips the rows >= M (``allow_oob_box`` on the
    row axis of the descriptor).  The one-shot path and the receipt launcher plan the TMA-store program on a fresh
    contiguous output and stay bit-identical to the register epilogue of a 4-byte-stride view of the same rows."""
    device = _require_program()
    for tp, module, M, seed in (
        ("tp1", "kv_b", 100, 71),
        ("tp1", "q_b", 65, 72),
        ("tp1", "q_proj", 127, 73),
    ):
        weight, scale, x, buf, reg_out, n_valid = _make_case(
            tp, module, M, 2, device, seed
        )
        prepared = prepare_kimi_k3_fp8_projection_weights(weight, scale, n_valid)
        reg_runner = prepare_kimi_k3_fp8_projection(
            x, prepared, reg_out, allocate_kimi_k3_fp8_projection_workspace(prepared, M)
        )
        assert reg_runner.plan.route == "gemm" and not reg_runner.plan.gemm_tma_store
        reg_runner()
        out = torch.empty((M, n_valid), dtype=torch.bfloat16, device=device)
        runner = prepare_kimi_k3_fp8_projection(
            x, prepared, out, allocate_kimi_k3_fp8_projection_workspace(prepared, M)
        )
        assert runner.plan.route == "gemm" and runner.plan.gemm_tma_store, (
            tp,
            module,
            M,
        )
        runner()
        launcher = cb.kimi_k3_fp8_projection_launcher(prepared)
        got = launcher(x)
        torch.cuda.synchronize()
        assert torch.equal(out, reg_out), (tp, module, M)
        assert torch.equal(got, reg_out), (tp, module, M)
        assert torch.isnan(buf[:, n_valid:].float()).all()
        assert_matches(out, reference(x, weight, scale, n_valid))


def test_launcher_receipts_follow_output_addresses():
    """Round 8 (W4): one planned launch receipt per (M, row stride, output address class); every call re-binds
    its activation and output view into it (new addresses included), the receipt cache is bounded (least recently
    used first) and the result stays bit-identical to the one-shot path."""
    device = _require_program()
    n_valid, K = PROJECTION_FAMILIES["tp8"]["q_proj"]
    weight, scale = make_weight(n_valid, K, device, 51)
    prepared = prepare_kimi_k3_fp8_projection_weights(weight, scale, n_valid)
    launcher = cb.kimi_k3_fp8_projection_launcher(prepared, max_receipts=2)
    M = 64
    outs = [
        torch.empty((M, n_valid), dtype=torch.bfloat16, device=device) for _ in range(3)
    ]
    x = make_activation(M, K, device, 7)
    launcher(x, outs[0])
    assert launcher.cached_receipts == 1
    x8 = make_activation(M, K, device, 8)
    for out in outs[
        1:
    ]:  # new output addresses of the same alignment class re-bind into the same receipt
        launcher(x8, out)
    assert launcher.cached_receipts == 1
    torch.cuda.synchronize()
    assert torch.equal(outs[0], kimi_k3_fp8_projection(x, prepared))
    for out in outs[1:]:
        assert torch.equal(out, kimi_k3_fp8_projection(x8, prepared))
    for m in (
        256,
        1024,
    ):  # other M: more receipts; the third plan evicts the least recently used (M = 64)
        xm = make_activation(m, K, device, 10 + m)
        om = torch.empty((m, n_valid), dtype=torch.bfloat16, device=device)
        launcher(xm, om)
        torch.cuda.synchronize()
        assert torch.equal(om, kimi_k3_fp8_projection(xm, prepared))
    assert launcher.cached_receipts == 2
    x2 = make_activation(M, K, device, 9)
    launcher(x2, outs[0])  # re-planned after eviction
    torch.cuda.synchronize()
    assert torch.equal(outs[0], kimi_k3_fp8_projection(x2, prepared))
    launcher.pin(M)
    assert M in launcher.cached_rows
    with pytest.raises(ValueError, match="max_receipts"):
        cb.kimi_k3_fp8_projection_launcher(prepared, max_receipts=0)


def _planned_receipt(launcher, x, out):
    """The one receipt a fresh launcher planned for ``(x, out)`` (after one call)."""
    (receipt,) = launcher._receipts.values()
    assert receipt.state is launcher._workspaces[int(x.shape[0])]
    return receipt


def test_launcher_two_launch_rows_take_one_chain_call():
    """Round 8 (W4, chain): a two-launch row (quantization program + GEMM / decode program) issues ONE FFI call per
    projection call -- the first program's ``run_chain`` replays both receipts through the callees each receipt names
    -- and stays bit-identical to the one-shot path eagerly and after a CUDA-graph capture on a side stream; a
    fused single-launch row keeps ``run_launch``.  Neither path allocates on a repeat call."""
    device = _require_program()
    for tp, module, M, two_stage in (
        ("tp8", "q_b", 1024, True),  # quant:u2 + tabulated decode (two launches)
        (
            "tp8",
            "q_proj",
            4096,
            True,
        ),  # quant:u4 + stream-K TMA-store GEMM (two launches)
        ("tp8", "f_b", 256, False),  # fused decode TMA-store instance (one launch)
    ):
        n_valid, K = PROJECTION_FAMILIES[tp][module]
        weight, scale = make_weight(n_valid, K, device, 81)
        prepared = prepare_kimi_k3_fp8_projection_weights(weight, scale, n_valid)
        x = make_activation(M, K, device, 82)
        expected = kimi_k3_fp8_projection(x, prepared)
        launcher = cb.kimi_k3_fp8_projection_launcher(prepared)
        out = torch.empty((M, n_valid), dtype=torch.bfloat16, device=device)
        assert launcher(x, out) is out
        torch.cuda.synchronize()
        receipt = _planned_receipt(launcher, x, out)
        assert (len(receipt.stages) == 2) == two_stage, (tp, module, M)
        assert (receipt.chain is not None) == two_stage, (tp, module, M)
        assert torch.equal(out, expected), (tp, module, M)
        if two_stage:
            entry, record_0, codes_0, record_1, codes_1 = receipt.chain
            assert record_0 is receipt.stages[0][1] and record_1 is receipt.stages[1][1]
            assert codes_0 == cb._slot_codes(
                receipt.stages[0][2]
            ) and codes_1 == cb._slot_codes(receipt.stages[1][2])
            assert codes_0 != 0 and codes_1 != 0
        # repeat call: no allocation, no new receipt
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        launcher(x, out)
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] - before == 0
        assert launcher.cached_receipts == 1
        # fresh activation and output addresses re-bind into the same receipt(s)
        x2 = make_activation(M, K, device, 83)
        out2 = torch.empty((M, n_valid), dtype=torch.bfloat16, device=device)
        launcher(x2, out2)
        torch.cuda.synchronize()
        assert torch.equal(out2, kimi_k3_fp8_projection(x2, prepared)), (tp, module, M)
        assert launcher.cached_receipts == 1
        # side-stream graph capture of the chain / launch call, replayed into a NaN-filled output
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            launcher(x, out)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                launcher(x, out)
        torch.cuda.synchronize()
        out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected), (tp, module, M)
        assert launcher.cached_receipts == 1


def test_chain_entry_propagates_errors_across_modules():
    """Round 8 (W4, chain): the packed ``run_chain`` entry validates every receipt before it issues anything and
    returns the error a stage's own callee raised -- a slot code naming a frozen position of the SECOND program is
    refused by that program's ``ChainRebind`` (another module than the entry's) with the launcher's own message,
    and the second launch never happens; a truncated or zeroed receipt is refused before any launch."""
    device = _require_program()
    n_valid, K = PROJECTION_FAMILIES["tp8"]["q_b"]
    weight, scale = make_weight(n_valid, K, device, 91)
    prepared = prepare_kimi_k3_fp8_projection_weights(weight, scale, n_valid)
    M = 1024
    x = make_activation(M, K, device, 92)
    expected = kimi_k3_fp8_projection(x, prepared)
    launcher = cb.kimi_k3_fp8_projection_launcher(prepared)
    out = torch.empty((M, n_valid), dtype=torch.bfloat16, device=device)
    launcher(x, out)
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
    receipt = _planned_receipt(launcher, x, out)
    assert receipt.chain is not None, "the row must take two launches"
    entry, record_0, codes_0, record_1, codes_1 = receipt.chain
    (template,) = launcher._templates.values()
    stream = torch.cuda.current_stream(device).cuda_stream

    def frozen_code(stage_index):
        # the trailing ("grid", "grid_z") position of a program's argument plan is never a re-bindable slot
        position = len(template.stages[stage_index].slots) - 1
        assert position not in {
            pos for pos, _src in template.stages[stage_index].rebinds
        }
        return 1 + ((position << 1) | 1)

    # the second program refuses the re-bind (cross-module status propagation); the first launch already ran into
    # the workspace, the second never did: the output stays untouched
    out.fill_(float("nan"))
    torch.cuda.synchronize()
    with pytest.raises(ValueError, match="re-bindable"):
        entry(stream, x, out, record_0, codes_0, record_1, frozen_code(1))
    torch.cuda.synchronize()
    assert torch.isnan(out.float()).all()
    # the first program refuses its own re-bind before anything is issued
    with pytest.raises(ValueError, match="re-bindable"):
        entry(stream, x, out, record_0, frozen_code(0), record_1, codes_1)
    torch.cuda.synchronize()
    assert torch.isnan(out.float()).all()
    # receipts are validated before any launch: a truncated second receipt, a receipt of no launcher at all
    with pytest.raises(ValueError, match="needs at least"):
        entry(stream, x, out, record_0, codes_0, record_1[:16].clone(), codes_1)
    with pytest.raises(ValueError, match="not planned"):
        entry(stream, x, out, record_0, codes_0, torch.zeros_like(record_1), codes_1)
    with pytest.raises(ValueError, match="not planned"):
        entry(stream, x, out, torch.zeros_like(record_0), codes_0, record_1, codes_1)
    with pytest.raises(TypeError):
        entry(
            stream, x, out, record_0, codes_0, record_1
        )  # a receipt without its slot codes
    torch.cuda.synchronize()
    assert torch.isnan(out.float()).all()
    # the receipts are intact: the next call re-binds and reproduces the one-shot result
    assert torch.equal(launcher(x, out), expected)
    assert launcher.cached_receipts == 1


def test_launch_makes_no_allocation():
    runner, *_ = _run_case("tp8", "fused_qkv_a", 256, 0, seed=5)
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    runner()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] - before["allocation.all.allocated"] == 0


def test_prepare_rejects_bad_bindings():
    device = _require_program()
    weight, scale, x, _buf, out, n_valid = _make_case(
        "tp8", "q_proj", 16, 0, device, seed=3
    )
    prepared = prepare_kimi_k3_fp8_projection_weights(weight, scale, n_valid)
    workspace = allocate_kimi_k3_fp8_projection_workspace(prepared, 16)
    small = allocate_kimi_k3_fp8_projection_workspace(prepared, 8)
    with pytest.raises(ValueError, match="workspace.q"):
        prepare_kimi_k3_fp8_projection(x, prepared, out, small)
    with pytest.raises(ValueError, match="unit column stride"):
        prepare_kimi_k3_fp8_projection(x, prepared, out.t(), workspace)
    with pytest.raises(ValueError, match="contiguous bf16"):
        prepare_kimi_k3_fp8_projection(x.float(), prepared, out, workspace)
    with pytest.raises(ValueError, match="backend"):
        prepare_kimi_k3_fp8_projection(x, prepared, out, workspace, backend="cutlass")
    with pytest.raises(ValueError, match="n_valid"):
        prepare_kimi_k3_fp8_projection_weights(weight, scale, 15)
    with pytest.raises(ValueError, match="splits"):
        prepare_kimi_k3_fp8_projection_weights(weight, scale, n_valid, splits=(1, 2))
