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
# Measured SM counts of the dispatch tables (round 7, CAKE-985): B200 148, GB300 152 (JHB nodes).
SM_COUNTS = {"sm_100a": 148, "sm_103a": 152}
# Round-6 sm_103a cell whose ordered stream-K window (planned at 148 SMs) never opens at the measured 152: plain GEMM
# in production (the sealed export measured it so); pruning it is a table-hygiene follow-up.
STREAM_K_CLOSED_CELLS = {"sm_103a": {"56,48,4096"}}

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
    ("tp8", "q_proj", 1000, 0),
    # 16-byte row stride (6288) with a column edge (6284) that is not: register epilogue, padding untouched
    ("tp8", "in_proj_qkvgfab", 1000, 4),
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
    with pytest.raises(ValueError):
        cb.weight_scale_tiles_bn(sf, K, 100)


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
        assert (f"_px{cfg.pfx}" in cfg.kernel_key) == (cfg.pfx > 0)
        assert (f"_pi{cfg.pfi}" in cfg.kernel_key) == (cfg.pfi > 0)
        core_key = re.sub(r"(_mc\d+)?(_pf\d+)?(_px\d+)?(_pi\d+)?$", "", cfg.kernel_key)
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
        else:
            assert cfg.xb_stages == 0 and not re.search(r"_r\d", cfg.kernel_key)
            assert not cfg.xbh and "_xh" not in cfg.kernel_key
            assert not cfg.qer and "_qe" not in cfg.kernel_key
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
    assert cfg.kernel_key == "decode:t64_p2_fused_w16_r3_xh_qe_q4_pf4_pi2"
    # 16-token tiles cannot keep eight 4-lane groups busy per stage: the table's 4 lanes widen to 8, coupled staging.
    # Round 5: the 24-tile M = 256 row moves to a 4-CTA cluster split-K route (each CTA owns a quarter of K, FP32 partials
    # are exchanged through distributed shared memory in one round); the small dedicated inbox is used (no aliasing).
    cfg = decode_config(256, 1, 28, arch, SM_COUNT)
    assert (cfg.tok, cfg.split, cfg.csplit, cfg.cs_alias, cfg.fused) == (
        16,
        4,
        4,
        False,
        True,
    )
    assert cfg.kernel_key == "decode:t16_p4_fused_cs4"


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
            assert cfg.kernel_key_for(True) == cfg.kernel_key + "_tso"
        assert cfg.kernel_key_for(False) == cfg.kernel_key
    assert n_tstore == 17
    # Round-6 next loop (lever PX-S): eight small fused buckets per architecture prefetch their BF16 token tile one stage
    # ahead of its TMA load (``pfx: 1`` -> the ``_px1`` program); a prefetch changes no data path and no launch argument.
    n_pfx = 0
    for key, entry in decode_table(arch).items():
        if entry["route"] != "decode" or not entry.get("pfx"):
            continue
        n_tiles128, num_k_iters, bucket = (int(v) for v in key.split(","))
        cfg = decode_config(bucket, n_tiles128, num_k_iters, arch, SM_COUNT)
        assert (
            cfg.fused
            and not cfg.resident
            and cfg.pfx == 1
            and cfg.kernel_key.endswith("_px1")
        )
        n_pfx += 1
    assert n_pfx == 8
    assert decode_config(8, 24, 2, arch, SM_COUNT).kernel_key.endswith(
        "_px1"
    )  # tp8 kv_b M = 8: 1.076-1.083x B200 / 1.034-1.042x B300
    assert not decode_config(1, 24, 2, arch, SM_COUNT).kernel_key.endswith(
        "_px1"
    )  # tp8 kv_b M = 1 keeps the plain instance (B300-only win)
    cfg = decode_config(
        256, 56, 6, arch, SM_COUNT
    )  # tp8 o_proj M = 256: 1.07x on both GPUs
    assert cfg.tstore and cfg.kernel_key_for(True) == "decode:t128_p3_tso"
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
        assert cfg.module_stages == (3 if c >= 7 else 4), cfg
        # round-6 next loop (lever PX): the M = 8 buckets of this family carry the `_px1` suffix
        assert cfg.kernel_key == f"decode:t16_p{cfg.module_stages}_fused_cs{c}" + (
            f"_px{cfg.pfx}" if cfg.pfx else ""
        )
    assert (
        cb.decode_cluster_capacity(arch, 14) == 7
        and cb.decode_cluster_capacity(arch, 9) == 15
    )
    # The plan carries both programs of every tstore row (the register program is the fallback of unaligned views).
    required = set(cb.required_kernel_keys(arch, SM_COUNT))
    assert {
        "decode:t128_p3",
        "decode:t128_p3_tso",
        "decode:t16_p3_fused_cs14",
        "decode:t16_p3_fused_cs7",
    } <= required
    assert not {
        k
        for k in required
        if k.endswith("_tso") and k.removesuffix("_tso") not in required
    }
    assert "decode:t16_p4_fused_cs14" not in required
    # Lever L5: the N = 576 kv_a rows at M > 256 (buckets 4096 and 16384) take the 192-wide GEMM N tile: three 192-column
    # tiles stream and multiply no padded columns (1.04-1.09x on both GPUs, bit-exact with the 256-wide output); the
    # fused_qkv_a family (N = 2112) measured slower with it and stays 256-wide, as does every untabulated shape.
    narrow = {k: e for k, e in decode_table(arch).items() if "gemm_bn" in e}
    # round 7 (CAKE-985) adds measured 192-wide cells on the 512 / 1024 / 2048 buckets; the round-6 kv_a cells stay
    assert {"5,28,16384", "5,28,4096"} <= set(narrow)
    assert all(e["route"] == "gemm" and e["gemm_bn"] == 192 for e in narrow.values())
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
    assert {"gemm_tstore_n192", "gemm_rstaged_n192"} <= required
    # Round 6 continuation 12 (lever SKO): the tabulated ``gemm_sk`` rows launch the ordered stream-K instance when the
    # launch has one full wave of CTA pairs plus at most half a wave of tail tiles: the head pair of each tail tile runs
    # the first K half and hands its FP32 partial to the tail pair, which continues the same accumulation (bit-exact).
    sk_rows = {k: e for k, e in decode_table(arch).items() if "gemm_sk" in e}
    # round 7 (CAKE-985) adds measured stream-K cells on the 512 / 1024 / 2048 buckets; every cell's window opens at its bucket
    assert {"12,28,4096", "56,48,4096"} <= set(sk_rows)
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
        if key in STREAM_K_CLOSED_CELLS.get(arch, ()):
            assert plan is None
            continue
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
    # 96 tiles over 74 pairs: 22 tail tiles, head 16 of 28 K iterations (table key gemm_sk_ksplit, continuation 13); below one wave or untabulated: plain schedule
    assert cb.gemm_stream_k_plan(4096, 12, 28, arch, 148, cb._m_tiles(4096), 6) == (
        cb.StreamKPlan(74, 22, 16, 74, 148) if "12,28,4096" in sk_rows else None
    )
    # 448 tiles over 74 pairs: 6 full waves (888 CTAs, one pair per data-parallel tile) + 4 tail tiles, head 28 of 48 (table key gemm_sk_ksplit)
    assert cb.gemm_stream_k_plan(4096, 56, 48, arch, 148, cb._m_tiles(4096), 28) == (
        cb.StreamKPlan(74, 4, 28, 444, 888) if "56,48,4096" in sk_rows else None
    )
    assert cb.gemm_stream_k_plan(2048, 12, 28, arch, 148, cb._m_tiles(2048), 6) is None
    assert cb.gemm_stream_k_plan(4096, 96, 28, arch, 148, cb._m_tiles(4096), 48) is None
    assert "gemm_tstore_sk" in required and "gemm_rstaged_sk" not in required
    # Round 6 continuation 17/18 (lever SKF): the tabulated ``gemm_skf`` rows launch the fix-up stream-K instance when
    # the launch has a fractional wave: the last full wave plus the fraction (118 / 119 tiles x 28 K iterations) is cut
    # into 74 equal K ranges; the pair finishing a tile adds the other contributors' FP32 partials in ordinal order
    # (reduction order of those tiles differs from the plain schedule; accepted by the user, max_abs_err 0.03125).
    skf_rows = {k: e for k, e in decode_table(arch).items() if "gemm_skf" in e}
    # round 7 (CAKE-985) adds measured fix-up stream-K cells on the 512 / 1024 / 2048 buckets; the round-6 cells stay
    assert {"384,28,256", "386,28,256", "5,28,16384"} <= set(skf_rows)
    assert all(
        e["route"] == "gemm" and e["gemm_skf"] == 1 and "gemm_sk" not in e
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
    assert {"gemm_tstore_skf", "gemm_tstore_n192_skf"} <= required
    assert not any(
        k.startswith(("gemm_rstaged", "gemm_n192", "gemm_pf")) and k.endswith("_skf")
        for k in required
    )


def test_decode_module_stage_clamp():
    # 128-token unfused: 4 stages of 66 KB do not fit the pool -> 3.
    assert decode_module_stages(128, 4, False, False) == 3
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


@pytest.mark.parametrize("arch", ARCHES)
def test_required_kernel_keys_are_registered_when_programs_exist(arch):
    required = required_kernel_keys(arch, SM_COUNT)
    assert "gemm" in required and "quant:u1" in required and "quant:u4" in required
    assert any(key.startswith("decode:") for key in required)
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
        gpf = (
            cb.gemm_prefetch_distance(
                M, _prepared.n_tiles128, _prepared.num_k_iters, plan.arch
            )
            if aligned
            else 0
        )
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
        sk_key = cb.gemm_kernel_key(
            expected_kernel + ("_n192" if bn == 192 else "") + "_sk", gpf
        )
        if sk is not None and not cb.route_available(plan.arch, (sk_key,)):
            sk = None  # only the TMA-store stream-K program ships: other views keep the plain program
        assert plan.gemm_sk == sk
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
            if sk is None and expected_kernel == "gemm_tstore"
            else None
        )
        skf_key = cb.gemm_kernel_key(
            expected_kernel + ("_n192" if bn == 192 else "") + "_skf", gpf
        )
        if skf is not None and not cb.route_available(plan.arch, (skf_key,)):
            skf = None
        assert plan.gemm_skf == skf
        assert plan.kernels[-1] == cb.gemm_kernel_key(
            expected_kernel
            + ("_n192" if bn == 192 else "")
            + ("_sk" if sk is not None else "_skf" if skf is not None else ""),
            gpf,
        )
        assert (plan.gemm_bn, plan.gemm_n_tiles) == (
            bn,
            _prepared.n_tiles if bn == 256 else -(-_prepared.n_valid // 192),
        )
        assert plan.grids[-1] == (
            sk.grid
            if sk is not None
            else skf.grid
            if skf is not None
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
    """Round 7 (CAKE-949 host path): the cached launcher reproduces the prepare() + launch() output bit for bit
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
