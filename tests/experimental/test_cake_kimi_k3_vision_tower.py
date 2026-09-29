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
    launch_tile_config,
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
    # The K = 1024 norm GEMMs take the eight-warp epilogue above 1024 rows; the
    # handoff forms follow their base variant's rule.  Production twins: the
    # residual / pos forms run the packed epilogue (``_p`` / ``_pf`` with the
    # residual prefetch), norm_qkv_rope the packed f16x2 RoPE table (``_cs``).
    expect = {
        "pos": ("xs_pf", "xs_pf", "s_e8_pf", "s_e8_pf"),
        "pos_sqxw": ("xs_pf", "xs_pf", "s_e8_pf", "s_e8_pf"),
        "norm_qkv_rope": ("xs_cs_pf", "s_cs", "l_e8_cs", "l_e8_cs"),
        "residual_wo": ("xs_pf", "xs_pf", "s_e8_pf", "m_tma1"),
        "residual_wo_sqxw": ("xs_pf", "xs_pf", "s_e8_pf", "m_tma1"),
        "residual_fc1": ("xs_k4_pf", "xs_pf", "l_e8_pf", "m_p"),
        "residual_fc1_sqxw": ("xs_k4_pf", "xs_pf", "l_e8_pf", "m_p"),
        "norm_gelu": ("xs", "s", "l_e8", "l_e8"),
        # Round 4: the projector GEMMs take the stream-K twin ``l_sk`` inside the tile-census
        # window (gelu_erf at 10764: 688 pair tiles on 74 clusters, tail 22 = 30 %, saving 7 %).
        "gelu_erf": ("xs", "s", "l", "l_sk"),
        "rmsnorm": ("s", "s", "l", "l"),
    }
    for variant, names in expect.items():
        got = tuple(select_tile_config(variant, m).name for m in (4, 1024, 4144, 10764))
        assert got == names, (variant, got)
    # The census is per device: on 160 SMs (80 clusters) the same M leaves a 48-tile tail (60 %), no twin.
    assert select_tile_config("gelu_erf", 10764, sm_count=160).name == "l"
    # Round-3 per-form boundaries (Cake ``kimi_k3_vision_gemm`` r3 tile-boundary policy):
    # (first M of each interval, tile) over M = 1 .. 20000.
    boundaries = {
        "pos": [(1, "xs_pf"), (1025, "s_e8_pf")],
        "pos_sqxw": [(1, "xs_pf"), (1025, "s_e8_pf")],
        # Round 5: the eight-warp 256 x 128 pair tile inside (QKV_M_E8_MIN_M, M_E8_MAX_M] (norm_qkv_rope) /
        # (NG_S_LIMIT, M_E8_MAX_M] (norm_gelu); Cake ``_m_e8_twin``.
        "norm_qkv_rope": [
            (1, "xs_cs_pf"),
            (769, "s_cs"),
            (1537, "l_e8_cs"),
            (2305, "m_e8_cs"),
            (4097, "l_e8_cs"),
        ],
        "residual_wo": [(1, "xs_pf"), (1025, "s_e8_pf"), (6145, "m_tma1")],
        "residual_wo_sqxw": [(1, "xs_pf"), (1025, "s_e8_pf"), (6145, "m_tma1")],
        # Round 5 (Cake ``_fc1_sk_twin``): ``m_sk`` where the 256 x 256 census leaves 1..4 tiles in an extra round
        # (4609..4864 -> 76 pair tiles on 74 clusters), ``l_sk`` from 3.4 ideal rounds (15873) up to FC1_SK_L_MAX_M.
        "residual_fc1": [
            (1, "xs_k4_pf"),
            (257, "xs_pf"),
            (1025, "s_e8_pf"),
            (2305, "l_e8_pf"),
            (4609, "m_sk"),
            (4865, "l_e8_pf"),
            (8193, "m_p"),
            (15873, "l_sk"),
        ],
        "residual_fc1_sqxw": [
            (1, "xs_k4_pf"),
            (257, "xs_pf"),
            (1025, "s_e8_pf"),
            (2305, "l_e8_pf"),
            (4609, "m_sk"),
            (4865, "l_e8_pf"),
            (8193, "m_p"),
            (15873, "l_sk"),
        ],
        "norm_gelu": [
            (1, "xs"),
            (257, "s"),
            (513, "xs"),
            (769, "s"),
            (1657, "m_e8"),
            (4097, "l_e8"),
        ],
        # Round 4 / 5: ``l`` <-> ``l_sk`` alternate per 128-row tile pair inside the pair-tile range wherever
        # the census leaves a tail of <= 30 % of the 74 clusters worth >= 6.5 % of the rounds OR (round 5) a
        # tail of <= 60 % worth >= 9 % (Cake ``_sk_twin``; identical to the Cake rule over M = 1 .. 65536 at the
        # round-5 head, copied from ``exports/kimi_k3_vision_tower`` ``_gemm_tile``).
        "gelu_erf": [
            (1, "xs"),
            (257, "s"),
            (513, "xs"),
            (769, "s"),
            (1657, "l_sk"),
            (1793, "l"),
            (2305, "l_sk"),
            (3073, "l"),
            (3329, "l_sk"),
            (4097, "l"),
            (4609, "l_sk"),
            (5377, "l"),
            (5889, "l_sk"),
            (6401, "l"),
            (6913, "l_sk"),
            (7425, "l"),
            (8193, "l_sk"),
            (8449, "l"),
            (9473, "l_sk"),
            (9729, "l"),
            (10497, "l_sk"),
            (11009, "l"),
            (11777, "l_sk"),
            (12033, "l"),
            (12801, "l_sk"),
            (13057, "l"),
            (14081, "l_sk"),
            (14337, "l"),
        ],
        "rmsnorm": [
            (1, "s"),
            (1025, "l"),
            (1281, "l_sk"),
            (1537, "l"),
            (1793, "l_sk"),
            (2305, "l"),
            (2561, "l_sk"),
            (3073, "l"),
            (3329, "l_sk"),
            (3585, "l"),
            (3841, "l_sk"),
            (4097, "l"),
            (4609, "l_sk"),
            (4865, "l"),
            (5889, "l_sk"),
            (6145, "l"),
            (6657, "l_sk"),
            (6913, "l"),
            (7937, "l_sk"),
            (8193, "l"),
        ],
    }
    for variant, segments in boundaries.items():
        got, prev = [], None
        for m in range(1, 20001):
            name = select_tile_config(variant, m).name
            if name != prev:
                got.append((m, name))
                prev = name
        assert got == segments, (variant, got)
    for cfg in TILE_CONFIGS.values():
        assert cfg.cluster_x == cfg.cta_group * cfg.ksplit
        assert not cfg.tail or cfg.ksplit == 1
        assert not (cfg.sk and (cfg.tail or cfg.ksplit != 1 or cfg.cta_group != 2))
    # The stream-K twins exist for the pair tiles of the projector GEMMs (census window; above 16 rounds,
    # M > 17792 for gelu_erf, the window can no longer be met) and, round 5, of residual_fc1 (m_sk at the
    # 4609..4864 census point, l_sk for 15873 <= M <= FC1_SK_L_MAX_M); no other form takes one.
    for variant in expect:
        if variant not in ("gelu_erf", "rmsnorm", "residual_fc1", "residual_fc1_sqxw"):
            assert not any(
                select_tile_config(variant, m).sk for m in range(1, 65537, 128)
            ), variant
    assert not any(
        select_tile_config("gelu_erf", m).sk for m in range(17793, 65537, 128)
    )
    fc1 = [
        (m, select_tile_config("residual_fc1_sqxw", m).name)
        for m in range(1, cb.GEMM_CENSUS_LIMIT + 1, 128)
    ]
    assert [m for m, name in fc1 if name == "m_sk"] == [4609, 4737]
    l_sk = [m for m, name in fc1 if name == "l_sk"]
    assert (l_sk[0], l_sk[-1], len(l_sk)) == (15873, 105857, 704)
    assert select_tile_config("residual_fc1_sqxw", cb.FC1_SK_L_MAX_M).name == "l_sk"
    assert select_tile_config("residual_fc1_sqxw", cb.FC1_SK_L_MAX_M + 1).name == "m_p"
    assert select_tile_config("residual_fc1_sqxw", 153088).name == "m_p"
    # The census counts cover every routing window of the module (fc1 l_sk up to 105984, PDL windows) plus
    # the rule boundaries and the PDL window edges on M.
    assert cb.GEMM_CENSUS_LIMIT == 105984 + GEMM_BLOCK_M
    counts = cb.gemm_census_counts("gelu_erf")
    assert counts[:3] == (1, 129, 143) and {
        143,
        144,
        638,
        639,
        1195,
        1196,
        1197,
    } <= set(counts)
    assert {4096, 4097, 2304, 2305, 105984, 105985} <= set(
        cb.gemm_census_counts("norm_gelu")
    )
    # The tail split-K twins are opt-in in Cake and never selected.
    for variant in expect:
        for m in (4, 256, 257, 1024, 1025, 8192, 8193, 153088):
            assert not select_tile_config(variant, m).tail
    # Without a packed table the ``_cs`` tiles fall back to their FP32-table base.
    assert (
        launch_tile_config("norm_qkv_rope", TILE_CONFIGS["xs_cs_pf"]).name == "xs_cs_pf"
    )
    assert (
        launch_tile_config(
            "norm_qkv_rope", TILE_CONFIGS["xs_cs_pf"], rope_table=False
        ).name
        == "xs"
    )


def test_gemm_launch_geometry():
    # pos at T = 4: two 128-row parity tiles of one 256-row block, 16 column tiles.
    geo = gemm_launch_geometry("pos", TILE_CONFIGS["xs_pf"], 4, SM_COUNT)
    assert (geo.grid, geo.m_tiles) == ((32, 1, 1), 2)
    # No tail split-K on the production tiles: full_tiles = cluster tiles, tail_split = 1.
    assert (geo.cluster_tiles, geo.full_tiles, geo.tail_split) == (32, 32, 1)
    # Split-K residual GEMM: one 4-CTA cluster per output tile (non-persistent).
    geo = gemm_launch_geometry("residual_fc1", TILE_CONFIGS["xs_k4_pf"], 256, SM_COUNT)
    assert (geo.grid, geo.m_tiles) == ((2 * 16 * 4, 1, 1), 2)
    # Pair tile at large M: persistent grid of SM/2 clusters x 2 CTAs, even m_tiles.
    geo = gemm_launch_geometry("norm_qkv_rope", TILE_CONFIGS["l_e8_cs"], 4144, SM_COUNT)
    assert geo.m_tiles == _ceil_div(4144, GEMM_BLOCK_M) + 1
    assert geo.cluster_tiles == (geo.m_tiles // 2) * (QKV_N // 256)
    assert geo.grid == (2 * min(geo.cluster_tiles, SM_COUNT // 2), 1, 1)
    assert (geo.full_tiles, geo.tail_split) == (geo.cluster_tiles, 1)
    with pytest.raises(NotImplementedError):
        gemm_launch_geometry("norm_qkv_rope", TILE_CONFIGS["l_e8_t"], 4144, SM_COUNT)
    # Stream-K twin (round 4): the full rounds stay data-parallel; the 22-tile tail of gelu_erf
    # at M = 10764 (688 pair tiles on 74 clusters) is cut into q = 22 k-steps per cluster
    # (ceil(22 x 64 / 74) = 20, raised to ceil(64 / (SK_MAX_CONTRIB - 1)) = 22).
    cfg = select_tile_config("gelu_erf", 10764)
    assert cfg.name == "l_sk" and cfg.sk
    geo = gemm_launch_geometry("gelu_erf", cfg, 10764, SM_COUNT)
    assert (geo.grid, geo.m_tiles, geo.cluster_tiles) == ((148, 1, 1), 86, 688)
    assert (geo.full_tiles, geo.tail_split) == (666, 22)
    # A tail-free census (37 pair rows x 16 column tiles = 8 x 74) keeps every piece empty: q = 0.
    geo = gemm_launch_geometry("gelu_erf", TILE_CONFIGS["l_sk"], 9300, SM_COUNT)
    assert (geo.cluster_tiles, geo.full_tiles, geo.tail_split) == (592, 592, 0)
    assert cb.stream_k_workspace_ctas(SM_COUNT) == 150
    # Round 5 fc1 twins: m_sk at 4784 (19 pair rows x 8 column tiles = 152 tiles on 74 clusters, tail 4 ->
    # q = ceil(64 / 3) = 22); l_sk at 16576 (65 x 4 = 260 tiles, tail 38 -> q = ceil(38 x 64 / 74) = 33).
    cfg = select_tile_config("residual_fc1_sqxw", 4784)
    assert cfg.name == "m_sk" and cfg.sk
    geo = gemm_launch_geometry("residual_fc1_sqxw", cfg, 4784, SM_COUNT)
    assert (geo.grid, geo.m_tiles, geo.cluster_tiles) == ((148, 1, 1), 38, 152)
    assert (geo.full_tiles, geo.tail_split) == (148, 22)
    cfg = select_tile_config("residual_fc1_sqxw", 16576)
    assert cfg.name == "l_sk"
    geo = gemm_launch_geometry("residual_fc1_sqxw", cfg, 16576, SM_COUNT)
    assert (geo.grid, geo.m_tiles, geo.cluster_tiles) == ((148, 1, 1), 130, 260)
    assert (geo.full_tiles, geo.tail_split) == (222, 33)


def test_required_kernel_keys():
    # The plain SPLIT_KV form is reachable only on sm_103a (segments < 576 tokens or > 10764
    # tokens); on sm_100a every SPLIT_KV row runs the ring3 form.
    assert set(REQUIRED_KERNEL_KEYS) == {"sm_100a", "sm_103a"}
    assert "attention:tiles1" not in REQUIRED_KERNEL_KEYS["sm_100a"]
    assert "attention:tiles1" in REQUIRED_KERNEL_KEYS["sm_103a"]
    for keys in REQUIRED_KERNEL_KEYS.values():
        assert "attention:ring3" in keys
        assert "attention:tiles2" in keys
        assert "merge" in keys
        assert "rmsnorm_apply" in keys
        gemm_keys = [k for k in keys if k.startswith("gemm:")]
        # Small-M tiles of the per-layer forms are always inside their PDL_EARLY window: only the pdle
        # binary is reachable; the large-M tiles exist in both forms (window bound inside their range).
        assert (
            "gemm:pos_sqxw:xs_pf" in gemm_keys
            and "gemm:pos_sqxw:xs_pf:pdle" not in gemm_keys
        )
        assert "gemm:residual_fc1_sqxw:xs_k4_pf:pdle" in gemm_keys
        assert "gemm:residual_fc1:xs_k4_pf:pdle" in gemm_keys
        assert "gemm:residual_wo_sqxw:m_tma1" in gemm_keys
        assert "gemm:residual_wo_sqxw:m_tma1:pdle" in gemm_keys
        assert "gemm:residual_wo_sqxw:s_e8_pf:pdle" in gemm_keys
        assert "gemm:residual_fc1:s_e8_pf:pdle" in gemm_keys
        assert (
            "gemm:norm_qkv_rope:l_e8_cs" in gemm_keys
            and "gemm:norm_qkv_rope:l_e8_cs:pdle" in gemm_keys
            and "gemm:gelu_erf:l" in gemm_keys
        )
        # Round 4 / 5: the stream-K twins of the projector GEMMs and (round 5) of residual_fc1.
        assert "gemm:gelu_erf:l_sk" in gemm_keys and "gemm:rmsnorm:l_sk" in gemm_keys
        assert (
            "gemm:residual_fc1_sqxw:m_sk" in gemm_keys
            and "gemm:residual_fc1:l_sk" in gemm_keys
        )
        assert not any(
            k.split(":")[2].endswith("_sk")
            for k in gemm_keys
            if k.split(":")[1]
            not in ("gelu_erf", "rmsnorm", "residual_fc1", "residual_fc1_sqxw")
        )
        # Round 5: the PDL_EARLY binaries inside the census windows; never on stream-K / pos tiles, and the
        # plain PDL binary of a tile is registered only where it is reachable (e.g. norm_qkv_rope s_cs at
        # 769..1536 is always inside the window -> only its pdle form exists).
        pdle = [k for k in gemm_keys if k.endswith(":pdle")]
        assert len(pdle) == 25 and len(gemm_keys) == len(set(gemm_keys)) == 43
        assert (
            "gemm:norm_gelu:m_e8:pdle" in pdle
            and "gemm:norm_qkv_rope:m_e8_cs:pdle" in pdle
        )
        assert (
            "gemm:norm_qkv_rope:s_cs:pdle" in pdle
            and "gemm:norm_qkv_rope:s_cs" not in gemm_keys
        )
        assert "gemm:gelu_erf:l:pdle" not in pdle and "gemm:gelu_erf:l" in gemm_keys
        assert not any(
            TILE_CONFIGS[k.split(":")[2]].sk or k.split(":")[1] == "pos_sqxw"
            for k in pdle
        )
        # Every registered GEMM tile is a production (non-tail) config.
        assert not any(TILE_CONFIGS[k.split(":")[2]].tail for k in gemm_keys)
        # Only the launched variants (the ``_sq``-only forms are not part of the tower).
        assert not any(k.split(":")[1].endswith("_sq") for k in gemm_keys)


def test_pdl_early_window_and_exclusions():
    # Window edges (inclusive, on the GEMM's own M) follow the Cake table ``PDL_EARLY_WINDOW``.
    for variant, ranges in cb.PDL_EARLY_WINDOW.items():
        for lo, hi in ranges:
            assert cb.pdl_early_on(variant, lo) and cb.pdl_early_on(variant, hi)
            assert not cb.pdl_early_on(variant, hi + 1)
            if lo > 1:
                assert not cb.pdl_early_on(variant, lo - 1)
    # Variants without an entry use the default window: the last layer's plain residual_fc1 and pos.
    assert cb.pdl_early_on("residual_fc1", 43056) and not cb.pdl_early_on(
        "residual_fc1", 43057
    )
    assert cb.pdl_early_on("pos_sqxw", 4)
    # ... but the excluded configs never take the early binary: stream-K twins, tail twins, multicast, pos.
    assert not cb.pdl_early_selected("pos_sqxw", TILE_CONFIGS["xs_pf"], 4)
    assert not cb.pdl_early_selected("residual_fc1_sqxw", TILE_CONFIGS["m_sk"], 4784)
    assert not cb.pdl_early_selected("residual_fc1_sqxw", TILE_CONFIGS["l_sk"], 16576)
    assert not cb.pdl_early_selected("norm_gelu", TILE_CONFIGS["l_e8_t"], 4144)
    assert cb.pdl_early_selected("residual_fc1_sqxw", TILE_CONFIGS["l_e8_pf"], 4144)
    assert cb.gemm_stage_key("residual_fc1_sqxw", 4784) == "gemm:residual_fc1_sqxw:m_sk"
    assert (
        cb.gemm_stage_key("residual_fc1_sqxw", 4144)
        == "gemm:residual_fc1_sqxw:l_e8_pf:pdle"
    )
    assert cb.gemm_stage_key("gelu_erf", 143) == "gemm:gelu_erf:xs"
    assert cb.gemm_stage_key("gelu_erf", 144) == "gemm:gelu_erf:xs:pdle"
    assert cb.gemm_stage_key("rmsnorm", 1196) == "gemm:rmsnorm:l:pdle"
    assert cb.gemm_stage_key("rmsnorm", 1197) == "gemm:rmsnorm:l"
    assert cb.gemm_stage_key("norm_qkv_rope", 8192) == "gemm:norm_qkv_rope:l_e8_cs:pdle"
    assert cb.gemm_stage_key("norm_qkv_rope", 8193) == "gemm:norm_qkv_rope:l_e8_cs"
    # The census is per device (the fc1 twin's tail is counted on sm_count // 2 clusters).
    assert (
        cb.gemm_stage_key("residual_fc1_sqxw", 4784, sm_count=160)
        == "gemm:residual_fc1_sqxw:l_e8_pf:pdle"
    )


# Logical GEMM kernel key per launched variant of every contract row (T = total tokens, N = merged tokens),
# copied ONCE from the Cake export protocol at the round-5 head (``exports/kimi_k3_vision_tower/export.py``
# ``stage_kernels`` on ``kimi_k3_vision_gemm`` 2a311f42769; arch-independent).  The mirror must reproduce it
# exactly: a disagreement here means the host launches another binary than the source route.
CONTRACT_ROW_GEMM_KEYS = {
    # label: (T, N, {variant: key})
    "smoke_2x2": (
        4,
        1,
        (
            "xs_pf",
            "xs_cs_pf:pdle",
            "xs_pf:pdle",
            "xs:pdle",
            "xs_k4_pf:pdle",
            "xs_k4_pf:pdle",
            "xs",
            "s",
        ),
    ),
    "smoke_ragged": (
        264,
        62,
        (
            "xs_pf",
            "xs_cs_pf:pdle",
            "xs_pf:pdle",
            "s:pdle",
            "xs_pf:pdle",
            "xs_pf:pdle",
            "xs",
            "s",
        ),
    ),
    "smoke_t3": (
        360,
        58,
        (
            "xs_pf",
            "xs_cs_pf:pdle",
            "xs_pf:pdle",
            "s:pdle",
            "xs_pf:pdle",
            "xs_pf:pdle",
            "xs",
            "s",
        ),
    ),
    "img_224": (
        256,
        64,
        (
            "xs_pf",
            "xs_cs_pf:pdle",
            "xs_pf:pdle",
            "xs:pdle",
            "xs_k4_pf:pdle",
            "xs_k4_pf:pdle",
            "xs",
            "s",
        ),
    ),
    "img_336": (
        576,
        144,
        (
            "xs_pf",
            "xs_cs_pf:pdle",
            "xs_pf:pdle",
            "xs:pdle",
            "xs_pf:pdle",
            "xs_pf:pdle",
            "xs:pdle",
            "s:pdle",
        ),
    ),
    "img_448": (
        1024,
        256,
        (
            "xs_pf",
            "s_cs:pdle",
            "xs_pf:pdle",
            "s:pdle",
            "xs_pf:pdle",
            "xs_pf:pdle",
            "xs:pdle",
            "s:pdle",
        ),
    ),
    "img_640x480": (
        1656,
        414,
        (
            "s_e8_pf",
            "l_e8_cs:pdle",
            "s_e8_pf:pdle",
            "s:pdle",
            "s_e8_pf:pdle",
            "s_e8_pf:pdle",
            "s:pdle",
            "s:pdle",
        ),
    ),
    "img_800x600": (
        2552,
        638,
        (
            "s_e8_pf",
            "m_e8_cs:pdle",
            "s_e8_pf:pdle",
            "m_e8:pdle",
            "l_e8_pf:pdle",
            "l_e8_pf:pdle",
            "xs:pdle",
            "s:pdle",
        ),
    ),
    "img_1024x768": (
        4144,
        1036,
        (
            "s_e8_pf",
            "l_e8_cs:pdle",
            "s_e8_pf:pdle",
            "l_e8:pdle",
            "l_e8_pf:pdle",
            "l_e8_pf:pdle",
            "s",
            "l",
        ),
    ),
    "img_1280x720": (
        4784,
        1196,
        (
            "s_e8_pf",
            "l_e8_cs:pdle",
            "s_e8_pf:pdle",
            "l_e8:pdle",
            "m_sk",
            "m_sk",
            "s:pdle",
            "l:pdle",
        ),
    ),
    "batch8_448": (
        8192,
        2048,
        (
            "s_e8_pf",
            "l_e8_cs:pdle",
            "m_tma1:pdle",
            "l_e8:pdle",
            "l_e8_pf:pdle",
            "l_e8_pf:pdle",
            "l",
            "l_sk",
        ),
    ),
    "img_1920x1080": (
        10764,
        2691,
        (
            "s_e8_pf",
            "l_e8_cs",
            "m_tma1:pdle",
            "l_e8:pdle",
            "m_p:pdle",
            "m_p:pdle",
            "l_sk",
            "l_sk",
        ),
    ),
    "doc_1240x1754": (
        11340,
        2835,
        (
            "s_e8_pf",
            "l_e8_cs",
            "m_tma1:pdle",
            "l_e8:pdle",
            "m_p:pdle",
            "m_p:pdle",
            "l_sk",
            "l_sk",
        ),
    ),
    "mixed_1080p_xga_448_336": (
        16508,
        4127,
        ("s_e8_pf", "l_e8_cs", "m_tma1:pdle", "l_e8:pdle", "l_sk", "l_sk", "l", "l"),
    ),
    "batch4_1024x768": (
        16576,
        4144,
        ("s_e8_pf", "l_e8_cs", "m_tma1:pdle", "l_e8:pdle", "l_sk", "l_sk", "l", "l"),
    ),
    "img_2560x1440": (
        19136,
        4784,
        (
            "s_e8_pf",
            "l_e8_cs",
            "m_tma1:pdle",
            "l_e8:pdle",
            "l_sk",
            "l_sk",
            "l_sk",
            "l_sk",
        ),
    ),
    "video_720p_4f": (
        19136,
        1196,
        (
            "s_e8_pf",
            "l_e8_cs",
            "m_tma1:pdle",
            "l_e8:pdle",
            "l_sk",
            "l_sk",
            "s:pdle",
            "l:pdle",
        ),
    ),
    "img_3840x2160": (
        43056,
        10764,
        ("s_e8_pf", "l_e8_cs", "m_tma1:pdle", "l_e8", "l_sk", "l_sk", "l_sk", "l"),
    ),
    "video_1080p_4f": (
        43056,
        2691,
        ("s_e8_pf", "l_e8_cs", "m_tma1:pdle", "l_e8", "l_sk", "l_sk", "l_sk", "l_sk"),
    ),
    "img_max_4096sq": (
        66564,
        16641,
        ("s_e8_pf", "l_e8_cs", "m_tma1", "l_e8", "l_sk", "l_sk", "l", "l"),
    ),
    "video_480p_64f": (
        105984,
        6624,
        ("s_e8_pf", "l_e8_cs", "m_tma1", "l_e8", "l_sk", "l_sk", "l", "l"),
    ),
    "video_720p_32f": (
        153088,
        9568,
        ("s_e8_pf", "l_e8_cs", "m_tma1", "l_e8", "m_p", "m_p", "l_sk", "l"),
    ),
}
CONTRACT_ROW_VARIANTS = (
    "pos_sqxw",
    "norm_qkv_rope",
    "residual_wo_sqxw",
    "norm_gelu",
    "residual_fc1_sqxw",
    "residual_fc1",
    "gelu_erf",
    "rmsnorm",
)


def test_contract_row_kernel_keys_match_cake_protocol():
    assert set(CONTRACT_ROW_VARIANTS) == set(cb.PRODUCTION_GEMM_VARIANTS)
    for label, (total, merged, tiles) in CONTRACT_ROW_GEMM_KEYS.items():
        expected = {
            variant: f"gemm:{variant}:{tile}"
            for variant, tile in zip(CONTRACT_ROW_VARIANTS, tiles, strict=True)
        }
        assert cb.gemm_kernel_keys_for(total, merged, SM_COUNT) == expected, label
        # Every key of every row is registered as reachable on both architectures.
        for arch in SUPPORTED_COMPUTE_CAPABILITIES.values():
            assert set(expected.values()) <= set(REQUIRED_KERNEL_KEYS[arch]), (
                label,
                arch,
            )
    # ... and the census names nothing the rows cannot reach (the exporter refuses both directions).
    reachable = {
        key
        for _t, _n, tiles in CONTRACT_ROW_GEMM_KEYS.values()
        for key in (
            f"gemm:{variant}:{tile}"
            for variant, tile in zip(CONTRACT_ROW_VARIANTS, tiles, strict=True)
        )
    }
    for arch in SUPPORTED_COMPUTE_CAPABILITIES.values():
        assert {
            k for k in REQUIRED_KERNEL_KEYS[arch] if k.startswith("gemm:")
        } == reachable, arch


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
        assert plan.ring3 is False  # no arch: the plain form
    # Round 3: the SPLIT_KV form is the shared-O ring per arch / longest segment
    # (every SPLIT_KV row on sm_100a; 576 .. 10764-token segments on sm_103a).
    # Round 4: sm_100a prices the split layout with its own cost model (the ring loop at
    # 0.95 blocks per K/V block, a one-block unit boundary, no margin) and routes the long
    # rows to the ring too; sm_103a keeps the round-3 model (two tiles above 10764 tokens).
    for label, arch, ring3 in (
        ("img_224", "sm_100a", True),
        ("img_224", "sm_103a", False),
        ("img_448", "sm_100a", True),
        ("img_448", "sm_103a", True),
        ("img_1920x1080", "sm_100a", True),
        ("img_1920x1080", "sm_103a", True),
        ("batch8_448", "sm_100a", True),
        ("batch8_448", "sm_103a", False),
        ("img_max_4096sq", "sm_100a", True),
        ("img_max_4096sq", "sm_103a", False),
    ):
        plan = build_attention_plan(
            cu_seqlens_of(CONTRACT_GRIDS[label]),
            torch.device("cpu"),
            HEADS,
            grid_clusters=GRID_CLUSTERS,
            arch=arch,
        )
        assert plan.ring3 is ring3, (label, arch, plan.tiles_per_cta)
        assert not (plan.ring3 and plan.tiles_per_cta != MODE_SPLIT_KV)
    forced = build_attention_plan(
        [0, 256],
        torch.device("cpu"),
        HEADS,
        grid_clusters=GRID_CLUSTERS,
        tiles_per_cta=2,
    )
    assert forced.tiles_per_cta == 2 and forced.total_clusters == 1
    # The per-arch split cost model: eight 1024-token segments (8 K/V blocks each) cost 30 blocks
    # per cluster on two tiles (192 units x (8 + 2) over 74 clusters, 3 rounds); the split layout
    # (384 half-units, 6 rounds) costs 36 under the H3 model (4 + 2 per unit; sm_103a / no arch:
    # two tiles) and 28.8 under the sm_100a model (0.95 x 4 + 1: the ring).
    lens = [1024] * 8
    for arch, mode, split in (
        ("sm_100a", MODE_SPLIT_KV, 28.8),
        ("sm_103a", MODE_TWO_TILE, 36.0),
        (None, MODE_TWO_TILE, 36.0),
    ):
        sel = cb.select_tiles_per_cta(lens, HEADS, GRID_CLUSTERS, arch)
        assert sel["tiles_per_cta"] == mode, (arch, sel)
        assert sel["makespan"][MODE_TWO_TILE] == pytest.approx(30.0), (arch, sel)
        assert sel["makespan"][MODE_SPLIT_KV] == pytest.approx(split), (arch, sel)
    assert cb.split_cost_model(None) is cb.SPLIT_COST_MODELS["r3"]
    assert cb.split_cost_model("sm_103a") is cb.SPLIT_COST_MODELS["r3"]


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


def test_prepare_weights_no_folding():
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
    # The RMSNorm weights stay separate (applied on the activation side by the
    # residual epilogues); every layer tensor is a contiguous copy of the input.
    assert set(prepared.layers[0]) == {"norm0", "wqkv", "wo", "norm1", "fc0", "fc1"}
    for name, tensor in prepared.layers[0].items():
        assert torch.equal(tensor, lw[name]) and tensor.is_contiguous()
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
        "pos_sqxw": "xs_pf",
        "norm_qkv_rope": "xs_cs_pf",
        "residual_wo_sqxw": "xs_pf",
        "norm_gelu": "s",
        "residual_fc1_sqxw": "xs_pf",
        "residual_fc1": "xs_pf",
        "gelu_erf": "xs",
        "rmsnorm": "s",
    }
    assert plan.attention.grid_clusters == GRID_CLUSTERS
    # Packed f16x2 (cos, sin) words of the FP32 tables, one per pair.
    assert (
        plan.rope_cs.shape == (264, cb.ROPE_PAIRS)
        and plan.rope_cs.dtype == torch.uint32
    )
    halves = plan.rope_cs.view(torch.float16).float().view(264, cb.ROPE_PAIRS, 2)
    torch.testing.assert_close(halves[..., 0], plan.cos.half().float())
    torch.testing.assert_close(halves[..., 1], plan.sin.half().float())
    assert plan.workspace["u32_dummy"].dtype == torch.uint32
    assert plan.workspace["x"].shape == (264, HIDDEN)
    assert plan.workspace["xw"].shape == (264, HIDDEN)
    # No stream-K tile in this plan: no partial workspace.
    assert "sk_ws" not in plan.workspace and "sk_flags" not in plan.workspace
    # img_1920x1080 (T = 10764, 2691 merged tokens): the projector RMSNorm GEMM takes the
    # stream-K twin, so the plan owns the FP32 partial workspace (150 CTAs x 3 pieces x
    # 128 x 256) and the zeroed arrival counters of the Cake launcher's ``_tail_buffers``.
    sk_plan = build_kimi_k3_vision_plan(
        CONTRACT_GRIDS["img_1920x1080"], "cpu", num_layers=1, sm_count=SM_COUNT
    )
    assert sk_plan.merged_tokens == 2691
    assert sk_plan.gemm_configs["rmsnorm"] == "l_sk"
    # Round 5: the widened stream-K window routes gelu_erf at 2691 (tail 28 / 74, saving 20.7 %) to l_sk too.
    assert sk_plan.gemm_configs["gelu_erf"] == "l_sk"
    assert sk_plan.gemm_kernel_keys["gelu_erf"] == "gemm:gelu_erf:l_sk"
    # The per-layer forms at T = 10764: qkv outside its PDL_EARLY window (8192), the others inside.
    assert sk_plan.gemm_kernel_keys["norm_qkv_rope"] == "gemm:norm_qkv_rope:l_e8_cs"
    assert (
        sk_plan.gemm_kernel_keys["residual_wo_sqxw"]
        == "gemm:residual_wo_sqxw:m_tma1:pdle"
    )
    assert (
        sk_plan.gemm_kernel_keys["residual_fc1_sqxw"]
        == "gemm:residual_fc1_sqxw:m_p:pdle"
    )
    assert plan.gemm_kernel_keys == {
        "pos_sqxw": "gemm:pos_sqxw:xs_pf",
        "norm_qkv_rope": "gemm:norm_qkv_rope:xs_cs_pf:pdle",
        "residual_wo_sqxw": "gemm:residual_wo_sqxw:xs_pf:pdle",
        "norm_gelu": "gemm:norm_gelu:s:pdle",
        "residual_fc1_sqxw": "gemm:residual_fc1_sqxw:xs_pf:pdle",
        "residual_fc1": "gemm:residual_fc1:xs_pf:pdle",
        "gelu_erf": "gemm:gelu_erf:xs",
        "rmsnorm": "gemm:rmsnorm:s",
    }
    # img_1280x720 (T = 4784): the FC1 stream-K twin m_sk owns the partial workspace (a per-layer form, so the
    # plan of a row without any projector twin still carries sk_ws / sk_flags).
    fc1_plan = build_kimi_k3_vision_plan(
        [(1, 46, 104)], "cpu", num_layers=1, sm_count=SM_COUNT
    )
    assert fc1_plan.total_tokens == 4784 and fc1_plan.merged_tokens == 1196
    assert fc1_plan.gemm_configs["residual_fc1_sqxw"] == "m_sk"
    assert (
        fc1_plan.gemm_kernel_keys["residual_fc1_sqxw"] == "gemm:residual_fc1_sqxw:m_sk"
    )
    assert fc1_plan.gemm_kernel_keys["rmsnorm"] == "gemm:rmsnorm:l:pdle"
    assert (
        fc1_plan.gemm_configs["gelu_erf"] == "s"
        and fc1_plan.gemm_configs["rmsnorm"] == "l"
    )
    assert fc1_plan.workspace["sk_ws"].shape == (150 * 3 * 128 * 256,)
    assert fc1_plan.workspace["sk_flags"].shape == (150,)
    assert sk_plan.workspace["sk_ws"].shape == (150 * 3 * 128 * 256,)
    assert sk_plan.workspace["sk_ws"].dtype == torch.float32
    assert sk_plan.workspace["sk_flags"].shape == (150,)
    assert sk_plan.workspace["sk_flags"].dtype == torch.uint32
    assert int(sk_plan.workspace["sk_flags"].sum()) == 0
    assert plan.workspace["stats"].shape == (264, cb.STATS_PARTS)
    assert plan.workspace["stats"].dtype == torch.float32


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
    device, weights, pixels, cos, sin, pos_rows, out = _inputs(grids, layers=2, seed=11)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        runner = prepare_kimi_k3_vision_tower(pixels, grids, weights, out)
        ws = runner.plan.workspace
        stages = runner.stages
        cu = cu_seqlens_of(grids)
        T = cu[-1]
        lw = weights["layers"][0]
        lw1 = weights["layers"][1]
        f = lambda w: w.float()  # noqa: E731

        def handoff(x, w_next):
            """The residual epilogue's handoff of the BF16 row set ``x``: exact values."""
            xw = (x.float() * w_next.float()[None, :]).to(torch.bfloat16)
            stats = x.float().view(T, cb.STATS_PARTS, HIDDEN // cb.STATS_PARTS)
            return xw, stats.square().sum(dim=-1)

        def check_handoff(x, w_next):
            xw, stats = handoff(x, w_next)
            _close(ws["xw"], xw)
            torch.testing.assert_close(ws["stats"], stats, atol=1e-2, rtol=1e-2)

        def set_handoff(x, w_next):
            xw, stats = handoff(x, w_next)
            ws["x"].copy_(x)
            ws["xw"].copy_(xw)
            ws["stats"].copy_(stats)

        stages["patch_embed"]()
        torch.cuda.synchronize()
        x0_oracle = (
            F.linear(pixels.float().view(T, PATCH_DIM), f(weights["patch_proj"]))
            + pos_rows.float()
        )
        _close(ws["x"], x0_oracle)
        x0 = x0_oracle.to(torch.bfloat16)
        check_handoff(ws["x"], lw["norm0"])

        set_handoff(x0, lw["norm0"])
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
        check_handoff(ws["x"], lw["norm1"])

        set_handoff(x1, lw["norm1"])
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
        check_handoff(ws["x"], lw1["norm0"])

        # The last layer's FC1 (no consumer): plain residual form on layer 1's weights.
        ws["ffn"].copy_(ffn)
        ws["x"].copy_(x1)
        ws["xw"].zero_()
        ws["stats"].zero_()
        stages["layer_fc1_last"]()
        torch.cuda.synchronize()
        _close(ws["x"], x1.float() + F.linear(ffn.float(), f(lw1["fc1"])))
        assert not ws["xw"].any() and not ws["stats"].any()

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
