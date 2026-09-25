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

import pytest
import torch
import torch.nn.functional as F

from flashinfer.experimental.kimi_k3_vision_tower import cake_backend as cb
from flashinfer.experimental.kimi_k3_vision_tower.cake_backend import (
    FFN,
    GEMM_BLOCK_M,
    HEAD_DIM,
    HEADS,
    HIDDEN,
    MERGE_KERNEL,
    MERGED_DIM,
    MODE_SPLIT_KV,
    MODE_TWO_TILE,
    NORM_EPS,
    PATCH,
    PATCH_DIM,
    PATCH_DIM_PAD,
    POS_SHIFT,
    PROJECTOR_EPS,
    QKV_HIDDEN,
    QKV_N,
    REQUIRED_KERNEL_KEYS,
    SOFTMAX_SCALE,
    SUPPORTED_COMPUTE_CAPABILITIES,
    TEXT_HIDDEN,
    TILE_CONFIGS,
    build_attention_plan,
    build_kimi_k3_vision_plan,
    build_merge_table_host,
    cu_seqlens_of,
    gemm_launch_geometry,
    merged_tokens,
    pos_emb_rows,
    prepare_kimi_k3_vision_tower,
    prepare_kimi_k3_vision_weights,
    rope_cos_sin,
    select_tile_config,
    sincos_time_table,
    validate_grid_thws,
)

# Tolerance of the Cake evaluation contract for per-operator checks (never looser).
ATOL = RTOL = 1e-2
# Persistent-grid capacity of B200 / B300 in 2-CTA clusters (148 SMs).
GRID_CLUSTERS = 74
SM_COUNT = 148

# grid_thws batches: single images, a ragged multi-segment batch and a t = 3 video group.
SMALL_GRIDS = [
    [(1, 2, 2)],
    [(1, 2, 6), (1, 10, 4), (2, 4, 4), (1, 6, 30)],
    [(3, 8, 8), (1, 12, 14)],
    [(1, 16, 16)],
]
CONTRACT_GRIDS = {
    "img_224": [(1, 16, 16)],
    "img_448": [(1, 32, 32)],
    "img_1920x1080": [(1, 78, 138)],
    "img_max_4096sq": [(1, 258, 258)],
    "batch8_448": [(1, 32, 32)] * 8,
    "video_720p_32f": [(4, 52, 92)] * 8,
    "video_480p_64f": [(4, 36, 46)] * 16,
}


def _ceil_div(a, b):
    return -(-a // b)


# ---------------------------------------------------------------------------
# Host plan (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "grid",
    [[(0, 2, 2)], [(1, 3, 2)], [(1, 2, 514)], [(5, 2, 2)], [], [(1, 2)]],
)
def test_rejects_invalid_grid_thws(grid):
    with pytest.raises(ValueError):
        validate_grid_thws(grid)


def test_token_counts():
    grids = [(1, 2, 6), (1, 10, 4), (2, 4, 4), (1, 6, 30)]
    assert cu_seqlens_of(grids) == (0, 12, 52, 84, 264)
    assert merged_tokens(grids) == 3 + 10 + 4 + 45


@pytest.mark.parametrize("grids", SMALL_GRIDS)
def test_merge_table(grids):
    table = build_merge_table_host(grids)
    rows = [tuple(table[i : i + 4]) for i in range(0, len(table), 4)]
    assert len(rows) == merged_tokens(grids)
    start = 0
    index = 0
    for t, h, w in grids:
        for ny in range(h // MERGE_KERNEL[0]):
            for nx in range(w // MERGE_KERNEL[1]):
                first, row_stride, frame_stride, frames = rows[index]
                assert (row_stride, frame_stride, frames) == (w, h * w, t)
                # The 2x2 window tokens of every frame stay inside the segment.
                for f in range(t):
                    for dy in range(2):
                        for dx in range(2):
                            tok = first + f * frame_stride + dy * row_stride + dx
                            assert start <= tok < start + t * h * w
                            assert (tok - start) % (h * w) // w == ny * 2 + dy
                            assert (tok - start) % w == nx * 2 + dx
                index += 1
        start += t * h * w


def test_rope_tables():
    grids = [(2, 4, 6)]
    cos, sin = rope_cos_sin(grids)
    T = 2 * 4 * 6
    assert cos.shape == sin.shape == (T, HEAD_DIM // 2)
    assert cos.dtype == sin.dtype == torch.float32
    # Token (t=1, y=2, x=5): pair 2i rotates by x * f_i, pair 2i + 1 by y * f_i.
    tok = 1 * 24 + 2 * 6 + 5
    freqs = 1.0 / (10000.0 ** (torch.arange(0, HEAD_DIM, 4).float() / HEAD_DIM))
    torch.testing.assert_close(cos[tok, 0::2], torch.cos(5 * freqs))
    torch.testing.assert_close(sin[tok, 1::2], torch.sin(2 * freqs))
    # The table repeats over t.
    torch.testing.assert_close(cos[:24], cos[24:])


def test_sincos_time_table():
    table = sincos_time_table()
    assert table.shape == (4, HIDDEN)
    torch.testing.assert_close(table[0, : HIDDEN // 2], torch.zeros(HIDDEN // 2))
    torch.testing.assert_close(table[0, HIDDEN // 2 :], torch.ones(HIDDEN // 2))


def test_select_tile_config_buckets():
    expect = {
        "pos": ("xs", "xs", "s_e8", "s_e8"),
        "norm_qkv_rope": ("xs", "s", "l", "l"),
        "residual_wo": ("xs", "xs", "m", "m"),
        "residual_fc1": ("xs_k4", "xs", "l_e8", "m"),
        "norm_gelu": ("xs", "s", "l", "l"),
        "gelu_erf": ("xs", "s", "l", "l"),
        "rmsnorm": ("s", "s", "l", "l"),
    }
    for variant, names in expect.items():
        got = tuple(select_tile_config(variant, m).name for m in (4, 1024, 4144, 10764))
        assert got == names, (variant, got)
    for cfg in TILE_CONFIGS.values():
        assert cfg.cluster_x == cfg.cta_group * cfg.ksplit


def test_gemm_launch_geometry():
    # pos at T = 4: two 128-row parity tiles of one 256-row block, 16 column tiles.
    grid, m_tiles = gemm_launch_geometry("pos", TILE_CONFIGS["xs"], 4, SM_COUNT)
    assert (grid, m_tiles) == ((32, 1, 1), 2)
    # Split-K residual GEMM: one 4-CTA cluster per output tile (non-persistent).
    grid, m_tiles = gemm_launch_geometry(
        "residual_fc1", TILE_CONFIGS["xs_k4"], 256, SM_COUNT
    )
    assert (grid, m_tiles) == ((2 * 16 * 4, 1, 1), 2)
    # Pair tile at large M: persistent grid of SM/2 clusters x 2 CTAs, even m_tiles.
    grid, m_tiles = gemm_launch_geometry(
        "norm_qkv_rope", TILE_CONFIGS["l"], 4144, SM_COUNT
    )
    assert m_tiles == _ceil_div(4144, GEMM_BLOCK_M) + 1
    assert grid == (2 * min((m_tiles // 2) * (QKV_N // 256), SM_COUNT // 2), 1, 1)


def test_required_kernel_keys():
    assert "attention:tiles1" in REQUIRED_KERNEL_KEYS
    assert "attention:tiles2" in REQUIRED_KERNEL_KEYS
    assert "merge" in REQUIRED_KERNEL_KEYS
    assert "rmsnorm_apply" in REQUIRED_KERNEL_KEYS
    gemm_keys = [k for k in REQUIRED_KERNEL_KEYS if k.startswith("gemm:")]
    assert "gemm:pos:xs" in gemm_keys and "gemm:residual_fc1:xs_k4" in gemm_keys
    assert len(gemm_keys) == len(set(gemm_keys)) == 19


@pytest.mark.parametrize("label,grids", list(CONTRACT_GRIDS.items()))
def test_attention_plan(label, grids):
    cu = cu_seqlens_of(grids)
    plan = build_attention_plan(
        cu, torch.device("cpu"), HEADS, grid_clusters=GRID_CLUSTERS
    )
    lens = [b - a for a, b in zip(cu, cu[1:], strict=False) if b > a]
    rows = 2 * plan.tiles_per_cta * 128
    clusters = [_ceil_div(n, rows) for n in lens]
    assert plan.num_segments == len(lens)
    assert plan.total_clusters == sum(clusters)
    assert plan.total_tiles == HEADS * plan.total_clusters
    assert plan.num_clusters == min(GRID_CLUSTERS, max(plan.total_tiles, 1))
    assert plan.seg_len.tolist() == lens
    table = plan.unit_table.tolist()
    decoded = sorted(
        (table[2 * u], table[2 * u + 1] >> 16, table[2 * u + 1] & 0xFFFF)
        for u in range(plan.total_tiles)
    )
    expected = sorted(
        (s, h, c)
        for s, n in enumerate(clusters)
        for h in range(HEADS)
        for c in range(n)
    )
    assert decoded == expected
    # The LPT makespan of the chosen layout is the smaller one (ties keep two tiles).
    two, split = plan.makespan[MODE_TWO_TILE], plan.makespan[MODE_SPLIT_KV]
    assert plan.tiles_per_cta == (
        MODE_SPLIT_KV if split < two * 0.95 else MODE_TWO_TILE
    )


def test_attention_plan_layout_rule():
    """Small images run the SPLIT_KV layout, the 4K image the two-tile layout."""
    for label, mode in (("img_224", MODE_SPLIT_KV), ("img_max_4096sq", MODE_TWO_TILE)):
        plan = build_attention_plan(
            cu_seqlens_of(CONTRACT_GRIDS[label]),
            torch.device("cpu"),
            HEADS,
            grid_clusters=GRID_CLUSTERS,
        )
        assert plan.tiles_per_cta == mode, label
    forced = build_attention_plan(
        [0, 256],
        torch.device("cpu"),
        HEADS,
        grid_clusters=GRID_CLUSTERS,
        tiles_per_cta=2,
    )
    assert forced.tiles_per_cta == 2 and forced.total_clusters == 1


def test_attention_plan_drops_empty_segments():
    plan = build_attention_plan(
        [0, 0, 640, 640, 1200], torch.device("cpu"), HEADS, grid_clusters=5
    )
    assert plan.num_segments == 2
    assert plan.seg_begin.tolist() == [0, 640]
    assert plan.num_clusters == 5


def _make_weights(device, layers, seed=6230, dtype=torch.bfloat16):
    g = torch.Generator(device=device).manual_seed(seed)

    def normal(shape, std):
        return (
            torch.empty(shape, dtype=torch.float32, device=device)
            .normal_(0.0, std, generator=g)
            .to(dtype)
        )

    def uniform(shape, lo, hi):
        return (
            torch.empty(shape, dtype=torch.float32, device=device)
            .uniform_(lo, hi, generator=g)
            .to(dtype)
        )

    weights = {
        "patch_proj": normal((HIDDEN, PATCH_DIM), 0.02),
        "pos_emb": normal((64, 64, HIDDEN), 0.02),
        "time_weight": sincos_time_table().to(device=device, dtype=dtype),
        "final_norm": uniform((HIDDEN,), 0.9, 1.1),
        "merger_proj0": normal((MERGED_DIM, MERGED_DIM), math.sqrt(2.0 / MERGED_DIM)),
        "merger_proj1": normal((TEXT_HIDDEN, MERGED_DIM), math.sqrt(2.0 / MERGED_DIM)),
        "post_norm": uniform((TEXT_HIDDEN,), 0.9, 1.1),
        "layers": [],
    }
    for _ in range(layers):
        weights["layers"].append(
            {
                "norm0": uniform((HIDDEN,), 0.9, 1.1),
                "wqkv": normal((QKV_N, HIDDEN), 0.02),
                "wo": normal((HIDDEN, QKV_HIDDEN), 0.02),
                "norm1": uniform((HIDDEN,), 0.9, 1.1),
                "fc0": normal((FFN, HIDDEN), math.sqrt(2.0 / HIDDEN)),
                "fc1": normal((HIDDEN, FFN), math.sqrt(2.0 / FFN)),
            }
        )
    return weights


def test_prepare_weights_folding():
    weights = _make_weights("cpu", layers=1)
    prepared = prepare_kimi_k3_vision_weights(weights)
    w_pe = weights["patch_proj"]
    assert prepared.patch_proj.shape == (2 * HIDDEN, PATCH_DIM_PAD)
    assert torch.equal(prepared.patch_proj[:HIDDEN, :PATCH_DIM], w_pe)
    assert torch.equal(
        prepared.patch_proj[HIDDEN:, POS_SHIFT : POS_SHIFT + PATCH_DIM], w_pe
    )
    assert not prepared.patch_proj[:HIDDEN, PATCH_DIM:].any()
    assert not prepared.patch_proj[HIDDEN:, :POS_SHIFT].any()
    lw = weights["layers"][0]
    folded = (lw["wqkv"].float() * lw["norm0"].float()[None, :]).to(torch.bfloat16)
    assert torch.equal(prepared.layers[0]["wqkv_folded"], folded)
    folded = (lw["fc0"].float() * lw["norm1"].float()[None, :]).to(torch.bfloat16)
    assert torch.equal(prepared.layers[0]["fc0_folded"], folded)
    assert prepared.num_layers == 1
    with pytest.raises(ValueError):
        prepare_kimi_k3_vision_weights(
            {k: v for k, v in weights.items() if k != "post_norm"}
        )


def test_pos_emb_rows():
    weights = _make_weights("cpu", layers=1)
    rows = pos_emb_rows(
        weights["pos_emb"], weights["time_weight"], [(2, 4, 6), (1, 64, 64)]
    )
    assert rows.shape == (2 * 24 + 4096, HIDDEN) and rows.dtype == torch.bfloat16
    # The native 64x64 grid is the table itself; frames of a t > 1 grid add the time rows.
    assert torch.equal(rows[48:], weights["pos_emb"].reshape(-1, HIDDEN))
    emb2d = (
        F.interpolate(
            weights["pos_emb"].permute(2, 0, 1).unsqueeze(0),
            size=(4, 6),
            mode="bilinear",
        )
        .squeeze(0)
        .permute(1, 2, 0)
        .reshape(-1, HIDDEN)
    )
    assert torch.equal(rows[:24], emb2d + weights["time_weight"][0])
    assert torch.equal(rows[24:48], emb2d + weights["time_weight"][1])


def test_plan_on_cpu_device_needs_sm_count():
    grids = SMALL_GRIDS[1]
    plan = build_kimi_k3_vision_plan(grids, "cpu", num_layers=2, sm_count=SM_COUNT)
    assert plan.total_tokens == 264 and plan.merged_tokens == 62
    assert plan.gemm_configs == {
        "pos": "xs",
        "norm_qkv_rope": "s",
        "residual_wo": "xs",
        "residual_fc1": "xs",
        "norm_gelu": "s",
        "gelu_erf": "xs",
        "rmsnorm": "s",
    }
    assert plan.attention.grid_clusters == GRID_CLUSTERS
    assert plan.workspace["x"].shape == (264, HIDDEN)


# ---------------------------------------------------------------------------
# FP32 oracles and the BF16 reference chain (HF round points)
# ---------------------------------------------------------------------------


def _rms_norm(x, weight, eps):
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * rstd * weight.float()).to(x.dtype)


def _apply_rope(q, k, cos, sin):
    def rot(x):
        xf = x.float().reshape(*x.shape[:-1], HEAD_DIM // 2, 2)
        a, b = xf[..., 0], xf[..., 1]
        c = cos.reshape(cos.shape[0], 1, HEAD_DIM // 2)
        s = sin.reshape(sin.shape[0], 1, HEAD_DIM // 2)
        return (
            torch.stack([a * c - b * s, a * s + b * c], dim=-1)
            .reshape(x.shape)
            .to(x.dtype)
        )

    return rot(q), rot(k)


def _attention_fp32(q, k, v, cu, out_dtype):
    """Exact FP32 noncausal segment attention."""
    out = torch.empty(q.shape, dtype=out_dtype, device=q.device)
    for a, b in zip(cu, cu[1:], strict=False):
        if b <= a:
            continue
        logits = (
            torch.einsum("qhd,khd->hqk", q[a:b].float(), k[a:b].float()) * SOFTMAX_SCALE
        )
        probs = torch.softmax(logits, dim=-1)
        out[a:b] = torch.einsum("hqk,khd->qhd", probs, v[a:b].float()).to(out_dtype)
    return out


def _attention_bf16(q, k, v, cu, out_dtype=None):
    """BF16 tensor-core attention per segment (P rounded to BF16 before PV, like every flash kernel)."""
    out = torch.empty_like(q)
    for a, b in zip(cu, cu[1:], strict=False):
        if b <= a:
            continue
        qs, ks, vs = (t[a:b].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        out[a:b] = (
            F.scaled_dot_product_attention(qs, ks, vs, scale=SOFTMAX_SCALE)
            .squeeze(0)
            .transpose(0, 1)
        )
    return out


def _tpool_merge(x, grids):
    outputs = []
    start = 0
    for t, h, w in grids:
        n = t * h * w
        seq = x[start : start + n]
        start += n
        nh, nw = h // 2, w // 2
        r = (
            seq.view(t, nh, 2, nw, 2, x.shape[-1])
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .mean(dim=0)
        )
        outputs.append(r.reshape(nh * nw, 4 * x.shape[-1]))
    return torch.cat(outputs, dim=0)


def _tower(pixels, grids, weights, cos, sin, pos_rows, *, fp32, attention):
    """The HF chain; ``fp32=True`` keeps every parameter and activation in FP32 (the oracle)."""
    cu = cu_seqlens_of(grids)
    cast = (lambda t: t.float()) if fp32 else (lambda t: t)
    x = F.linear(
        cast(pixels).reshape(pixels.shape[0], PATCH_DIM), cast(weights["patch_proj"])
    ) + cast(pos_rows)
    for lw in weights["layers"]:
        n = _rms_norm(x, cast(lw["norm0"]), NORM_EPS)
        qkv = F.linear(n, cast(lw["wqkv"])).view(x.shape[0], 3, HEADS, HEAD_DIM)
        q, k, v = qkv.unbind(dim=1)
        q, k = _apply_rope(q, k, cos, sin)
        a = attention(q.contiguous(), k.contiguous(), v.contiguous(), cu, x.dtype)
        x = x + F.linear(a.reshape(x.shape[0], QKV_HIDDEN), cast(lw["wo"]))
        n = _rms_norm(x, cast(lw["norm1"]), NORM_EPS)
        x = x + F.linear(
            F.gelu(F.linear(n, cast(lw["fc0"])), approximate="tanh"), cast(lw["fc1"])
        )
    x = _rms_norm(x, cast(weights["final_norm"]), NORM_EPS)
    m = _tpool_merge(x, grids)
    y = F.linear(
        F.gelu(F.linear(m, cast(weights["merger_proj0"]))),
        cast(weights["merger_proj1"]),
    )
    return _rms_norm(y, cast(weights["post_norm"]), PROJECTOR_EPS)


def _fairness(actual, chain, oracle):
    """Oracle-fairness gate of the Cake contract: no worse than the BF16 reference chain."""
    tol = ATOL + RTOL * oracle.abs()
    err_a = (actual.float() - oracle).abs()
    err_c = (chain.float() - oracle).abs()
    viol_a, viol_c = int((err_a > tol).sum()), int((err_c > tol).sum())
    return dict(
        finite=bool(torch.isfinite(actual.float()).all()),
        violations=(viol_a, viol_c),
        mean=(float(err_a.mean()), float(err_c.mean())),
        max=(float(err_a.max()), float(err_c.max())),
        passed=bool(torch.isfinite(actual.float()).all())
        and viol_a <= 1.1 * viol_c + 16
        and float(err_a.mean()) <= 1.05 * max(float(err_c.mean()), 1e-12)
        and float(err_a.max()) <= 1.5 * max(float(err_c.max()), 1e-6),
    )


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


def _require_program():
    if not torch.cuda.is_available():
        pytest.skip("Kimi-K3 vision tower requires an SM100/SM103 GPU")
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(0))
    if arch is None:
        pytest.skip("Kimi-K3 vision tower requires an SM100/SM103 GPU")
    if not cb.generated_program_available(torch.device("cuda", 0)):
        pytest.skip(f"no generated Kimi-K3 vision tower program registered for {arch}")
    return arch


def _inputs(grids, layers, seed):
    device = torch.device("cuda", 0)
    weights = _make_weights(device, layers, seed=seed)
    T = cu_seqlens_of(grids)[-1]
    g = torch.Generator(device=device).manual_seed(seed + 1)
    pixels = torch.empty(
        (T, 3, PATCH, PATCH), dtype=torch.bfloat16, device=device
    ).uniform_(-1.0, 1.0, generator=g)
    cos, sin = rope_cos_sin(grids, device)
    pos_rows = pos_emb_rows(weights["pos_emb"], weights["time_weight"], grids)
    out = torch.full(
        (merged_tokens(grids), TEXT_HIDDEN),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    return device, weights, pixels, cos, sin, pos_rows, out


def _close(actual, expected):
    assert torch.isfinite(actual.float()).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("grids", SMALL_GRIDS)
def test_stages_match_fp32_oracles(grids):
    """Every stage on the reference chain's intermediates against the FP32 oracle of that operator."""
    _require_program()
    device, weights, pixels, cos, sin, pos_rows, out = _inputs(grids, layers=1, seed=11)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        runner = prepare_kimi_k3_vision_tower(pixels, grids, weights, out)
        ws = runner.plan.workspace
        stages = runner.stages
        cu = cu_seqlens_of(grids)
        T = cu[-1]
        lw = weights["layers"][0]
        f = lambda w: w.float()  # noqa: E731

        stages["patch_embed"]()
        torch.cuda.synchronize()
        x0_oracle = (
            F.linear(pixels.float().view(T, PATCH_DIM), f(weights["patch_proj"]))
            + pos_rows.float()
        )
        _close(ws["x"], x0_oracle)
        x0 = x0_oracle.to(torch.bfloat16)

        ws["x"].copy_(x0)
        stages["layer_norm_qkv_rope"]()
        torch.cuda.synchronize()
        n = _rms_norm(x0.float(), f(lw["norm0"]), NORM_EPS)
        qkv = F.linear(n, f(lw["wqkv"])).view(T, 3, HEADS, HEAD_DIM)
        q_o, k_o, v_o = qkv.unbind(dim=1)
        q_o, k_o = _apply_rope(q_o, k_o, cos, sin)
        _close(ws["q"], q_o)
        _close(ws["k"], k_o)
        _close(ws["v"], v_o)

        stages["layer_attention"]()
        torch.cuda.synchronize()
        a_o = _attention_fp32(ws["q"], ws["k"], ws["v"], cu, torch.float32)
        _close(ws["attn_out"], a_o)
        a = a_o.to(torch.bfloat16)

        ws["attn_out"].copy_(a)
        ws["x"].copy_(x0)
        stages["layer_out_proj"]()
        torch.cuda.synchronize()
        x1_oracle = x0.float() + F.linear(a.float().view(T, QKV_HIDDEN), f(lw["wo"]))
        _close(ws["x"], x1_oracle)
        x1 = x1_oracle.to(torch.bfloat16)

        ws["x"].copy_(x1)
        stages["layer_norm_fc0_gelu"]()
        torch.cuda.synchronize()
        n1 = _rms_norm(x1.float(), f(lw["norm1"]), NORM_EPS)
        ffn_oracle = F.gelu(F.linear(n1, f(lw["fc0"])), approximate="tanh")
        _close(ws["ffn"], ffn_oracle)
        ffn = ffn_oracle.to(torch.bfloat16)

        ws["ffn"].copy_(ffn)
        ws["x"].copy_(x1)
        stages["layer_fc1"]()
        torch.cuda.synchronize()
        x2_oracle = x1.float() + F.linear(ffn.float(), f(lw["fc1"]))
        _close(ws["x"], x2_oracle)
        x2 = x2_oracle.to(torch.bfloat16)

        ws["x"].copy_(x2)
        stages["final_norm_merge"]()
        torch.cuda.synchronize()
        m_oracle = _tpool_merge(
            _rms_norm(x2.float(), f(weights["final_norm"]), NORM_EPS), grids
        )
        _close(ws["m"], m_oracle)
        m = m_oracle.to(torch.bfloat16)

        ws["m"].copy_(m)
        stages["merger_gemm0"]()
        torch.cuda.synchronize()
        h_oracle = F.gelu(F.linear(m.float(), f(weights["merger_proj0"])))
        _close(ws["h"], h_oracle)
        h = h_oracle.to(torch.bfloat16)

        ws["h"].copy_(h)
        stages["merger_gemm1"]()
        stages["merger_rmsnorm_apply"]()
        torch.cuda.synchronize()
        y_oracle = _rms_norm(
            F.linear(h.float(), f(weights["merger_proj1"])),
            f(weights["post_norm"]),
            PROJECTOR_EPS,
        )
        _close(out, y_oracle)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


@pytest.mark.parametrize("grids", SMALL_GRIDS[1:])
def test_tower_fairness_vs_bf16_chain(grids):
    """Complete call: error against the FP32 oracle no worse than the HF BF16 chain's."""
    _require_program()
    device, weights, pixels, cos, sin, pos_rows, out = _inputs(grids, layers=3, seed=23)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        runner = prepare_kimi_k3_vision_tower(pixels, grids, weights, out)
        result = runner.launch()
        torch.cuda.synchronize()
        assert result is out
        oracle = _tower(
            pixels,
            grids,
            weights,
            cos,
            sin,
            pos_rows,
            fp32=True,
            attention=_attention_fp32,
        )
        chain = _tower(
            pixels,
            grids,
            weights,
            cos,
            sin,
            pos_rows,
            fp32=False,
            attention=_attention_bf16,
        )
        report = _fairness(out, chain, oracle)
        assert report["passed"], report
        # Idempotent: a second launch reproduces the output bitwise.
        first = out.clone()
        out.fill_(float("nan"))
        runner.launch()
        torch.cuda.synchronize()
        assert torch.equal(first, out)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def test_prepared_runner_graph_replay_and_no_allocation():
    _require_program()
    grids = SMALL_GRIDS[1]
    device, weights, pixels, cos, sin, pos_rows, out = _inputs(grids, layers=2, seed=5)
    prepared = prepare_kimi_k3_vision_weights(weights)
    plan = build_kimi_k3_vision_plan(grids, device, num_layers=2)
    runner = prepare_kimi_k3_vision_tower(
        pixels, grids, prepared, out, plan=plan, pos_rows=pos_rows
    )
    assert runner.plan is plan and runner.launch_count == 1 + 2 * 5 + 4
    assert set(runner.stage_modules) == set(cb.STAGE_NAMES)
    runner.launch()
    torch.cuda.synchronize()
    eager = out.clone()
    before = torch.cuda.memory_stats()
    runner.launch()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        runner.launch()
    torch.cuda.current_stream().wait_stream(stream)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
    # New pixel values through the same graph: replay tracks the buffer contents.
    pixels.mul_(-1.0)
    runner.launch()
    torch.cuda.synchronize()
    expected = out.clone()
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
    metadata = runner.route_metadata
    assert metadata["layers"] == 2 and metadata["segment_count"] == 4
    assert metadata["tiles_per_cta"] in (1, 2)


def test_one_shot_api():
    _require_program()
    from flashinfer.kimi_k3_vision import kimi_k3_vision_tower

    grids = SMALL_GRIDS[2]
    device, weights, pixels, cos, sin, pos_rows, out = _inputs(grids, layers=1, seed=3)
    result = kimi_k3_vision_tower(pixels, grids, weights)
    torch.cuda.synchronize()
    assert (
        result.shape == (merged_tokens(grids), TEXT_HIDDEN)
        and result.dtype == torch.bfloat16
    )
    assert torch.isfinite(result.float()).all()
    second = kimi_k3_vision_tower(pixels, grids, weights, out)
    torch.cuda.synchronize()
    assert second is out and torch.equal(out, result)


def test_rejects_bad_inputs():
    _require_program()
    grids = SMALL_GRIDS[0]
    device, weights, pixels, cos, sin, pos_rows, out = _inputs(grids, layers=1, seed=1)
    prepared = prepare_kimi_k3_vision_weights(weights)
    with pytest.raises(ValueError):
        prepare_kimi_k3_vision_tower(pixels.float(), grids, prepared, out)
    with pytest.raises(ValueError):
        prepare_kimi_k3_vision_tower(pixels, [(1, 2, 4)], prepared, out)
    with pytest.raises(ValueError):
        prepare_kimi_k3_vision_tower(
            pixels, grids, prepared, out[:, :HIDDEN].contiguous()
        )
    with pytest.raises(ValueError):
        prepare_kimi_k3_vision_tower(pixels, grids, prepared, out, backend="torch")
    other = build_kimi_k3_vision_plan(SMALL_GRIDS[1], device, num_layers=1)
    with pytest.raises(ValueError):
        prepare_kimi_k3_vision_tower(pixels, grids, prepared, out, plan=other)
