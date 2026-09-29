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
from flashinfer.experimental.kimi_k3_fp8_projection.cake_jit import KERNELS, MODULES
from flashinfer.experimental.kimi_k3_fp8_projection.decode_table import DECODE_TABLE
from flashinfer.gemm import (
    allocate_kimi_k3_fp8_projection_workspace,
    kimi_k3_fp8_projection,
    prepare_kimi_k3_fp8_projection,
    prepare_kimi_k3_fp8_projection_weights,
)

ATOL = 1e-2
RTOL = 1e-2
SM_COUNT = 148

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
        assert quant_units(M, 56) == 1
    assert quant_units(4096, 56) == 4  # K = 7168
    assert quant_units(4096, 1) == 1  # K = 128
    assert (
        quant_units(4096, 4) == 2
    )  # K = 512: four blocks per half warp would leave < 4 CTAs/SM
    assert quant_units(16384, 4) == 4  # K = 512
    assert quant_units(1000, 12) == 2  # K = 1536, M = 1000
    assert quant_units(16384, 96) == 4  # K = 12288


@pytest.mark.parametrize("arch", ARCHES)
def test_decode_table_covers_every_family(arch):
    for tp, modules in PROJECTION_FAMILIES.items():
        for name, (n_valid, K) in modules.items():
            n_tiles128 = -(-n_valid // BLOCK)
            num_k_iters = -(-K // 256)
            for bucket in DECODE_TABLE_BUCKETS:
                entry = cb.decode_table_entry(bucket, n_tiles128, num_k_iters, arch)
                if bucket > DECODE_MAX_M:
                    # Large buckets list only the families measured faster on the decode kernel.
                    assert entry is None or entry["route"] == "decode"
                    continue
                assert entry is not None, f"{arch} {tp}:{name} bucket {bucket}"
                assert entry["route"] in ("decode", "gemm")


@pytest.mark.parametrize("arch", ARCHES)
def test_decode_config_rules(arch):
    # Above DECODE_MAX_M only tabulated (narrow-N) families take the decode route; the table decides below.
    assert decode_config(257, 12, 28, arch, SM_COUNT) is None
    assert decode_config(4096, 12, 28, arch, SM_COUNT) is None
    assert decode_config(4096, 1, 28, arch, SM_COUNT) is not None
    for key, entry in DECODE_TABLE[arch].items():
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
        core_key = re.sub(r"(_mc\d+)?(_pf\d+)?$", "", cfg.kernel_key)
        # Round-3 fused knobs: a decoupled ring only for fused, non-resident rows; narrow units divide evenly.
        if entry.get("xb_stages"):
            assert (
                cfg.fused
                and not cfg.resident
                and 1 <= cfg.xb_stages <= cb.DEC_XB_MAX_STAGES
            )
            assert f"_r{cfg.xb_stages}" in cfg.kernel_key
        else:
            assert cfg.xb_stages == 0 and not re.search(r"_r\d", cfg.kernel_key)
        assert cfg.qlanes in (4, 8, 16)
        assert (2 * cfg.tok) % (cb.DEC_QUANT_WARPS * (32 // cfg.qlanes)) == 0
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
    assert cfg.kernel_key == "decode:t64_p2_fused_r3_q4_pf4"
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
    # Lever P alone on the M = 64 bucket of the same family (two stages ahead).
    cfg = decode_config(64, 50, 28, arch, SM_COUNT)
    assert (cfg.tok, cfg.mc, cfg.pf) == (64, 1, 2)
    assert cfg.kernel_key == "decode:t64_p4_pf2"
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
    for key, entry in DECODE_TABLE[arch].items():
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
    cfg = decode_config(256, 56, 6, arch, SM_COUNT)  # tp8 o_proj M = 256: 1.07x on both GPUs
    assert cfg.tstore and cfg.kernel_key_for(True) == "decode:t128_p3_tso"
    # Lever C16: the M <= 64 rows that used a 12-28-way global split-K now run one cluster per output tile (14 CTAs = two
    # 256-K stages each; non-portable cluster), the 12-tile family a 7-CTA cluster (12 clusters <= capacity 15) and the
    # 17-tile family a 4-CTA cluster; the grid is whole clusters within the measured co-resident capacity.
    for (M, n_tiles, c, clusters) in ((1, 1, 14, 1), (64, 1, 14, 4), (1, 5, 14, 5), (1, 12, 7, 12), (1, 17, 4, 17)):
        cfg = decode_config(M, n_tiles, 28, arch, SM_COUNT)
        assert (cfg.tok, cfg.split, cfg.csplit, cfg.fused) == (16, c, c, True), (M, n_tiles, cfg)
        assert clusters <= cb.decode_cluster_capacity(arch, c)
        assert cfg.grid == c * clusters and cfg.total_work == cfg.grid
        assert f"_cs{c}" in cfg.kernel_key and not cfg.tstore
    assert cb.decode_cluster_capacity(arch, 14) == 7 and cb.decode_cluster_capacity(arch, 9) == 15


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
    if arch in KERNELS:
        missing = sorted(set(required) - set(KERNELS[arch]))
        assert not missing, f"{arch} lacks {missing}"
        for module_name in KERNELS[arch].values():
            assert MODULES[module_name]["arch"] == arch


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
        assert plan.kernels[-1] == expected_kernel
    if plan.route == "decode" and plan.decode.fused:
        assert runner.launch_count == 1
    else:
        assert runner.launch_count == 2 and plan.kernels[0].startswith("quant:u")


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
