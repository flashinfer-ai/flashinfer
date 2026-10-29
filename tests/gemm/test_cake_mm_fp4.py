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

from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
from flashinfer.experimental.cake_nvfp4_per_token import cake_backend as cb
from flashinfer.experimental.cake_nvfp4_per_token.cake_jit import KERNELS
from tests.test_helpers.cake_nvfp4_per_token import (
    VALIDATED_GEMM_FAMILIES,
    VALIDATED_OFF_MATRIX_FAMILY,
    required_kernel_keys,
)

GLOBAL_SCALE_INV = 1.0 / (448.0 * 6.0)


# ---------------------------------------------------------------------------
# Host rules (CPU)
# ---------------------------------------------------------------------------


def test_default_tactic_rules():
    # M <= 32: swapped orientation; narrow weights take cluster split-K.
    t = cb.default_tactic(1, 2112, 7168, 148)
    assert t["alpha_n"] and t["tile_n"] == 8 and t["split_k"] == 4
    t = cb.default_tactic(17, 2112, 7168, 148)
    assert t["alpha_n"] and t["tile_n"] == 8 and t["split_k"] == 2
    t = cb.default_tactic(32, 8192, 8192, 148)
    assert t["alpha_n"] and t["tile_n"] == 32 and "split_k" not in t
    assert t["a_hint"] == "evict_first" and not t["deep_k"] and "num_stages" not in t
    # Single-wave deep-K rows run three mainloop stages.
    t = cb.default_tactic(8, 18432, 7168, 148)
    assert t["alpha_n"] and t["deep_k"] and t["num_stages"] == 3
    # Deep-K rows whose weight tiles fill at most half the SMs split K in two; the
    # 8 / 16-token tiles of the deeper rows stay unsplit, as do the wide-N rows.
    t = cb.default_tactic(17, 7168, 16384, 152)
    assert t["alpha_n"] and t["tile_n"] == 32 and t["split_k"] == 2
    assert t["a_hint"] == "evict_first" and not t["deep_k"]
    assert cb.default_tactic(1, 7168, 16384, 148)["split_k"] == 2
    assert cb.default_tactic(32, 7168, 18432, 148)["split_k"] == 2
    assert "split_k" not in cb.default_tactic(8, 7168, 18432, 148)
    assert "split_k" not in cb.default_tactic(32, 18432, 7168, 148)
    # Three K slices when the whole cluster grid (weight tiles x token tiles) is co-resident:
    # 7168x1536 M = 17 needs 36 clusters of 3 (capacity 45 on 148 SMs, 46 on 152); the
    # 48- and 51-cluster launches of M = 32 / N = 1536 and M = 17 / N = 2112 keep two
    # 8-token slices outside the 16-token rows, as does a part without a measured
    # capacity table.
    for sm_count in (148, 152):
        t = cb.default_tactic(17, 1536, 7168, sm_count)
        assert t["tile_n"] == 8 and t["split_k"] == 3
    t = cb.default_tactic(32, 1536, 7168, 148, out_f16=True)
    assert t["tile_n"] == 8 and t["split_k"] == 2
    assert cb.default_tactic(17, 2112, 7168, 148)["split_k"] == 2
    assert cb.default_tactic(17, 1536, 7168, 132)["split_k"] == 2
    assert set(cb.CLUSTER_CAPACITY_BY_SM_COUNT) == {148, 152}
    # One 128-token tile over 128-wide weight tiles runs without the L2 promotion on both
    # parts (7168x18432 M = 128: 144 tiles, one wave; 8192x28672 M = 128 on 148 SMs: two
    # waves of the same program).
    for sm_count in (148, 152):
        t = cb.default_tactic(128, 18432, 7168, sm_count)
        assert t["tile_n"] == 128 and t["l2_promo"] is None and "two_cta" not in t
    assert cb.default_tactic(128, 28672, 8192, 148)["l2_promo"] is None
    # More weight tiles than SMs: shallow K; two CTAs per SM on the 8-token tile or on
    # the 152-SM part, one CTA per SM for the 32-token tile on 148 SMs.
    t = cb.default_tactic(8, 28672, 8192, 148)
    assert (
        t["alpha_n"]
        and t["tile_n"] == 8
        and not t["deep_k"]
        and t["blocks_per_sm"] == 2
    )
    t = cb.default_tactic(17, 28672, 8192, 148)
    assert t["tile_n"] == 32 and not t["deep_k"] and "blocks_per_sm" not in t
    t = cb.default_tactic(17, 28672, 8192, 152)
    assert t["tile_n"] == 32 and not t["deep_k"] and t["blocks_per_sm"] == 2
    # One token tile over more 128-wide weight tiles than SMs: the 128-wide persistent tile
    # replaces the scorer's wider single-wave tile on the 148-SM part only.
    t = cb.default_tactic(128, 28672, 8192, 148)
    assert (
        t["tile_n"] == 128
        and not t["alpha_n"]
        and "two_cta" not in t
        and not t["deep_k"]
    )
    assert (
        t["a_hint"] is None and t["b_hint"] == "evict_first" and t["l2_promo"] is None
    )
    t = cb.default_tactic(128, 28672, 8192, 152)
    assert t["tile_n"] == 192 and "two_cta" not in t and t["l2_promo"] == "l2_256b"
    # One token tile, one wave of 128-wide tiles: no L2 promotion on either part.
    assert cb.default_tactic(128, 18432, 7168, 152)["l2_promo"] is None
    assert cb.default_tactic(128, 18432, 7168, 148)["l2_promo"] is None
    # M > 32: m orientation.  One 128-token tile over a few 64-wide weight tiles takes the
    # 2-CTA 256x64 pair (no grouped raster, no CLC); the weights stay evict_first.
    t = cb.default_tactic(128, 2112, 7168, 148)
    assert not t["alpha_n"] and t["tile_n"] == 64 and t["two_cta"] and t["half_m"]
    assert t["a_hint"] is None and t["b_hint"] == "evict_first"
    assert "split_k" not in t and "raster_group" not in t and "sched" not in t
    # The half-M pair (64 token rows per CTA) holds while its grid fits 1.5 waves of pairs:
    # 7168x2112 M=128 / 130 (66 / 99 tiles on 74 / 76 pairs) and 7168x1536 M=128 / 130 (48)
    # yes; 7168x1536 M=257 (120, 1.6 waves) and 7168x2112 M=257 (165, 2.2 waves) no.
    t = cb.default_tactic(130, 2112, 7168, 148)
    assert t["two_cta"] and t["tile_n"] == 64 and t["half_m"] and "split_k" not in t
    for sm_count in (148, 152):
        assert cb.default_tactic(130, 1536, 7168, sm_count)["half_m"]
        for n in (1536, 2112):
            t = cb.default_tactic(257, n, 7168, sm_count)
            assert t["two_cta"] and t["tile_n"] == 64 and "half_m" not in t
        assert "half_m" not in cb.default_tactic(512, 1536, 7168, sm_count)
    # Single-token-tile wide rows keep the scorer's 1-CTA tile (no A-multicast, no split-K).
    t = cb.default_tactic(128, 8192, 8192, 148)
    assert not t["alpha_n"] and t["tile_n"] == 64 and "two_cta" not in t
    assert "amc" not in t and "split_k" not in t
    t = cb.default_tactic(128, 18432, 7168, 152)
    assert (
        not t["alpha_n"]
        and t["tile_n"] == 128
        and "two_cta" not in t
        and "amc" not in t
    )
    # Large M: 2-CTA tiles, grouped raster for <= 32 weight tiles, CLC scheduler
    # once the pairs average >= 10 tiles.
    t = cb.default_tactic(8192, 7168, 16384, 148)
    assert t["two_cta"] and t["tile_n"] == 256 and t["raster_group"] == 16
    assert t["sched"] == "clc"
    # 16-tile groups only while the group's operands fit L2 (28672x8192: 176 MB -> 8),
    # and only past one 8-tile group of token tiles.
    assert cb.default_tactic(8192, 8192, 28672, 148)["raster_group"] == 8
    assert cb.default_tactic(8192, 8192, 8192, 152)["raster_group"] == 16
    assert cb.default_tactic(8192, 1536, 7168, 148)["raster_group"] == 8
    # Multi-wave rows re-pick the 2-CTA width from the measured wave model: 3.4 waves of
    # 256-wide tiles become 5 cheaper waves of 192-wide tiles (43 weight tiles: no group);
    # 3.03 waves on 148 SMs likewise, while 152 SMs fit the same grid in 3 full waves.
    for sms in (148, 152):
        t = cb.default_tactic(2048, 8192, 8192, sms)
        assert t["two_cta"] and t["tile_n"] == 192 and "raster_group" not in t
        assert cb.default_tactic(8192, 8192, 8192, sms)["tile_n"] == 256
    assert cb.default_tactic(2048, 7168, 16384, 148)["tile_n"] == 192
    assert cb.default_tactic(2048, 7168, 16384, 152)["tile_n"] == 256
    assert cb.default_tactic(257, 28672, 8192, 148)["tile_n"] == 192
    assert cb.default_tactic(257, 28672, 8192, 152)["tile_n"] == 256
    # A grid that fits one wave runs ungrouped (every tile is resident at once).
    for sms in (148, 152):
        t = cb.default_tactic(2048, 1536, 7168, sms)
        assert t["two_cta"] and t["tile_n"] == 192 and "raster_group" not in t
        assert "raster_group" not in cb.default_tactic(512, 8192, 8192, sms)
        assert cb.default_tactic(2048, 8192, 28672, sms)["tile_n"] == 192
    t = cb.default_tactic(2048, 18432, 7168, 148)
    assert t["two_cta"] and "raster_group" not in t and "sched" not in t
    t = cb.default_tactic(8192, 18432, 7168, 148)
    assert t["two_cta"] and t["sched"] == "clc" and "raster_group" not in t


def test_n8_w64_rows_only_where_registered():
    # Only sm_103a registers the 64-row weight-tile programs: sm_100a keeps 128-row weight
    # tiles at two CTAs per SM, also with 152 SMs.
    for sm_count in (148, 152):
        for n, k in ((28672, 8192), (18432, 7168), (7168, 16384), (2112, 7168)):
            for m in (1, 8, 17, 32):
                for out_f16 in (False, True):
                    plan = cb.gemm_plan(m, n, k, out_f16, "sm_100a", sm_count)
                    assert "_w64" not in plan.kernel_key, plan.kernel_key
    for m in (1, 8):
        t = cb.default_tactic(m, 28672, 8192, 152, n8_wide_w_rows=False)
        assert t["blocks_per_sm"] == 2 and "w_rows" not in t
        for out_f16, out in ((False, "bf16"), (True, "f16")):
            plan = cb.gemm_plan(m, 28672, 8192, out_f16, "sm_100a", 152)
            assert plan.kernel_key == f"gemm:n8_{out}_aF_l2256b_o2", plan.kernel_key
            assert plan.kernel_key in KERNELS["sm_100a"]


def test_n8_w64_rows():
    # The 8-token tile with more 128-row weight tiles than SMs on the 152-SM part:
    # 64-row weight tiles at one CTA per SM (8192x28672: 448 tiles, about three waves).
    for m in (1, 8):
        t = cb.default_tactic(m, 28672, 8192, 152)
        assert t["tile_n"] == 8 and t["w_rows"] == 64 and "blocks_per_sm" not in t
        t = cb.default_tactic(m, 28672, 8192, 148)
        assert t["blocks_per_sm"] == 2 and "w_rows" not in t
    assert "w_rows" not in cb.default_tactic(8, 18432, 7168, 152)  # 144 tiles
    assert "w_rows" not in cb.default_tactic(17, 28672, 8192, 152)
    for out_f16, out in ((False, "bf16"), (True, "f16")):
        plan = cb.gemm_plan(8, 28672, 8192, out_f16, "sm_103a", 152)
        assert plan.kernel_key == f"gemm:n8_{out}_aF_l2256b_w64", plan.kernel_key
        assert plan.w_tile == 64 and plan.w_tiles == 448 and plan.grid == 152
    with pytest.raises(ValueError, match="one CTA per SM"):
        cb.gemm_plan(
            8,
            28672,
            8192,
            False,
            "sm_103a",
            152,
            tactic={**cb.default_tactic(8, 28672, 8192, 152), "blocks_per_sm": 2},
        )


def test_m64_ring_overlay_rows():
    # The 64-wide 1-CTA m tile stages its output over the start of the mainloop ring on
    # both parts: the single-token-tile rows over more than 40 64-wide weight tiles.
    for arch, sm_count in (("sm_100a", 148), ("sm_103a", 152)):
        for n, k in ((7168, 16384), (7168, 18432), (8192, 28672), (8192, 8192)):
            t = cb.default_tactic(128, n, k, sm_count)
            assert t["tile_n"] == 64 and "two_cta" not in t and t["ring_overlay"]
            for out_f16, out in ((False, "bf16"), (True, "f16")):
                plan = _registered(
                    cb.gemm_plan(128, n, k, out_f16, arch, sm_count), arch
                )
                assert plan.kernel_key == f"gemm:m64_{out}_bF_l2256b_ro", (
                    plan.kernel_key
                )
    # The 2-CTA 256x64 pair, the 128-wide tile and the swapped orientation keep their
    # staging; the knob belongs to the 1-CTA m orientation.
    for sm_count in (148, 152):
        assert "ring_overlay" not in cb.default_tactic(128, 2112, 7168, sm_count)
        assert "ring_overlay" not in cb.default_tactic(128, 18432, 7168, sm_count)
        assert "ring_overlay" not in cb.default_tactic(8, 28672, 8192, sm_count)
    with pytest.raises(ValueError, match="1-CTA m orientation"):
        cb.gemm_plan(
            8,
            28672,
            8192,
            False,
            "sm_103a",
            152,
            tactic={**cb.default_tactic(8, 28672, 8192, 152), "ring_overlay": True},
        )


def test_m128_ring_overlay_rows():
    # One 128-token tile over more 128-wide weight tiles than SMs (8192x28672 M = 128 on
    # 148 SMs: 224 tiles) takes the overlay program of the 128-wide 1-CTA m tile, which
    # only sm_100a registers; one wave of 128-wide tiles keeps its staging.
    for out_f16, out in ((False, "bf16"), (True, "f16")):
        plan = _registered(
            cb.gemm_plan(128, 28672, 8192, out_f16, "sm_100a", 148), "sm_100a"
        )
        assert plan.kernel_key == f"gemm:m128_{out}_bF_ro", plan.kernel_key
        plan = cb.gemm_plan(128, 18432, 7168, out_f16, "sm_100a", 148)
        assert plan.kernel_key == f"gemm:m128_{out}_bF", plan.kernel_key
    t = cb.default_tactic(128, 28672, 8192, 148, m128_ring_overlay=True)
    assert t["tile_n"] == 128 and "two_cta" not in t and t["ring_overlay"]
    assert "ring_overlay" not in cb.default_tactic(128, 28672, 8192, 148)
    t = cb.default_tactic(128, 18432, 7168, 148, m128_ring_overlay=True)
    assert "ring_overlay" not in t


def test_token_remainder_fold_rows():
    # On a FOLD_CELLS (arch, K, N) cell at the measured SM count, 128 < M <= 136
    # runs one 128-token tile of the 1-CTA m128 program that carries the remainder
    # rows (``_fold``), and the chain takes its entry L2 prefetch form
    # (``_ep4_fold``); on a B2_FOLD_CELLS cell, 256 < M <= 272 runs one pair token
    # tile of the static 2-CTA pair program that carries the remainder rows
    # (``_b2fold``).  The token counts next to the ranges, the other SM count of the
    # architecture and the other shapes keep their plans, and the fold knobs refuse
    # other programs and token counts.
    other = {"sm_100a": 152, "sm_103a": 148}
    for arch, k, n in sorted(cb.FOLD_CELLS):
        sm_count = cb.MEASURED_SM_COUNT[arch]
        for out_f16, out in ((False, "bf16"), (True, "f16")):
            for m in range(129, 137):
                plan = _registered(cb.gemm_plan(m, n, k, out_f16, arch, sm_count), arch)
                assert plan.kernel_key == f"gemm:m128_{out}_bF_fold", (arch, m)
                assert plan.tok_tiles == 1 and plan.num_tiles == plan.w_tiles
                assert cb.gemm_alone_tactic(plan) is None
                tactic = cb.chain_gemm_tactic(plan)
                assert tactic is not None, (arch, m, out_f16)
                chain = _registered(
                    cb.gemm_plan(m, n, k, out_f16, arch, sm_count, tactic=tactic),
                    arch,
                )
                assert chain.kernel_key == f"gemm:m128_{out}_bF_ep4_fold"
                assert (chain.grid, chain.num_tiles) == (plan.grid, plan.num_tiles)
            for m in (128, 137):
                plan = cb.gemm_plan(m, n, k, out_f16, arch, sm_count)
                assert "fold" not in plan.kernel_key, (arch, m, plan.kernel_key)
            for m in (129, 136):
                plan = cb.gemm_plan(m, n, k, out_f16, arch, other[arch])
                assert "fold" not in plan.kernel_key, (arch, m, plan.kernel_key)
    for arch, k, n in sorted(cb.B2_FOLD_CELLS):
        sm_count = cb.MEASURED_SM_COUNT[arch]
        for out_f16, out in ((False, "bf16"), (True, "f16")):
            for m in range(257, 273):
                plan = _registered(cb.gemm_plan(m, n, k, out_f16, arch, sm_count), arch)
                assert plan.kernel_key == f"gemm:m256_2cta_{out}_bF_l2256b_b2fold"
                assert plan.tok_tiles == 1
                assert cb.chain_gemm_tactic(plan) is None
                assert cb.gemm_alone_tactic(plan) is None
            for m in (256, 273):
                plan = cb.gemm_plan(m, n, k, out_f16, arch, sm_count)
                assert "fold" not in plan.kernel_key, (arch, m, plan.kernel_key)
            for m in (257, 272):
                plan = cb.gemm_plan(m, n, k, out_f16, arch, other[arch])
                assert "fold" not in plan.kernel_key, (arch, m, plan.kernel_key)
    cells = cb.FOLD_CELLS | cb.B2_FOLD_CELLS
    for arch, sm_count in cb.MEASURED_SM_COUNT.items():
        for k, n in (*VALIDATED_GEMM_FAMILIES, VALIDATED_OFF_MATRIX_FAMILY):
            if (arch, k, n) in cells:
                continue
            for m in (129, 130, 136, 257, 264, 272):
                plan = cb.gemm_plan(m, n, k, False, arch, sm_count)
                assert "fold" not in plan.kernel_key, (arch, m, n, k, plan.kernel_key)
    for m in (128, 137):
        with pytest.raises(ValueError, match="fold"):
            cb.gemm_plan(
                m, 18432, 7168, False, "sm_100a", 148, tactic=dict(cb.FOLD_TACTIC)
            )
    with pytest.raises(ValueError, match="fold"):
        cb.gemm_plan(
            130,
            18432,
            7168,
            False,
            "sm_100a",
            148,
            tactic={**cb.FOLD_TACTIC, "ring_overlay": True},
        )
    b2 = cb.gemm_plan(257, 18432, 7168, False, "sm_100a", 148).tactic
    assert b2["b2"] == "fold"
    for m in (256, 273):
        with pytest.raises(ValueError, match="b2"):
            cb.gemm_plan(m, 18432, 7168, False, "sm_100a", 148, tactic=b2)


def test_chain_entry_prefetch_rows():
    # The quantize + GEMM chain runs the entry L2 prefetch program (``_ep4``) of the GEMM
    # plan exactly on the cells of CHAIN_EP_RULES (program family and token count per
    # architecture) and on the (K, N) shapes of CHAIN_EP_MEASURED_CELLS at the measured
    # SM count; the default-tactic plan never does (the GEMM alone takes it only on the
    # GEMM_ALONE_EP_RULES cells), and the prefetch keeps the plan's grid.
    for arch in sorted(set(cb.CHAIN_EP_RULES) | set(cb.CHAIN_EP_MEASURED_CELLS)):
        cells = cb.CHAIN_EP_RULES.get(arch, {})
        measured_cells = cb.CHAIN_EP_MEASURED_CELLS.get(arch, {})
        sm_count = cb.MEASURED_SM_COUNT[arch]
        served, measured_served = set(), set()
        for k, n in VALIDATED_GEMM_FAMILIES:
            for m in _EDGE_M:
                for out_f16 in (False, True):
                    plan = cb.gemm_plan(m, n, k, out_f16, arch, sm_count)
                    assert "_ep" not in plan.kernel_key, plan.kernel_key
                    if plan.kernel_key not in KERNELS[arch]:
                        continue
                    family = "_".join(
                        p
                        for p in plan.kernel_key.removeprefix("gemm:").split("_")
                        if p not in ("bf16", "f16")
                    )
                    cell = cells.get(family)
                    inside = cell is not None and cell[0] <= m <= cell[1]
                    measured = measured_cells.get(family)
                    in_measured = (
                        measured is not None
                        and (k, n) in measured[2]
                        and measured[0] <= m <= measured[1]
                    )
                    tactic = cb.chain_gemm_tactic(plan)
                    if tactic is not None and not tactic.get("ep_stages"):
                        tactic = None  # a scale-factor ring row: test_sf_ring_rows
                    assert (tactic is not None) == (inside or in_measured), (
                        arch,
                        m,
                        n,
                        k,
                        out_f16,
                    )
                    if tactic is None:
                        continue
                    chain = _registered(
                        cb.gemm_plan(m, n, k, out_f16, arch, sm_count, tactic=tactic),
                        arch,
                    )
                    assert chain.kernel_key == _ep4_key(plan.kernel_key)
                    assert (chain.grid, chain.num_tiles) == (plan.grid, plan.num_tiles)
                    (served if inside else measured_served).add(family)
        assert served == set(cells), (arch, sorted(set(cells) - served))
        assert measured_served == set(measured_cells), (
            arch,
            sorted(set(measured_cells) - measured_served),
        )
    # The prefetch is a knob of the 1-CTA program and the swapped-orientation split-K
    # program.
    with pytest.raises(ValueError, match="ep_stages"):
        cb.gemm_plan(
            512,
            8192,
            8192,
            False,
            "sm_100a",
            148,
            tactic={**cb.default_tactic(512, 8192, 8192, 148), "ep_stages": 4},
        )


def test_chain_entry_prefetch_measured_cells_keep_other_sm_counts():
    # A CHAIN_EP_MEASURED_CELLS cell takes the prefetch only at the SM count it was timed
    # on, where the GEMM alone keeps its program; the same shapes on the other SM count of
    # the architecture (sm_100a on 152 SMs, sm_103a on 148) prefetch only inside a
    # CHAIN_EP_RULES cell.
    def family_of(plan):
        return "_".join(
            p
            for p in plan.kernel_key.removeprefix("gemm:").split("_")
            if p not in ("bf16", "f16")
        )

    other = {"sm_100a": 152, "sm_103a": 148}
    for arch, cells in cb.CHAIN_EP_MEASURED_CELLS.items():
        for family, (lo, hi, shapes) in cells.items():
            for k, n in shapes:
                for m in sorted({lo, hi}):
                    for out_f16 in (False, True):
                        sm_count = cb.MEASURED_SM_COUNT[arch]
                        plan = cb.gemm_plan(m, n, k, out_f16, arch, sm_count)
                        assert family_of(plan) == family, plan.kernel_key
                        assert cb.chain_gemm_tactic(plan) is not None, (arch, m, n, k)
                        assert cb.gemm_alone_tactic(plan) is None, (arch, m, n, k)
                        plan = cb.gemm_plan(m, n, k, out_f16, arch, other[arch])
                        rule = cb.CHAIN_EP_RULES.get(arch, {}).get(family_of(plan))
                        inside = rule is not None and rule[0] <= m <= rule[1]
                        tactic = cb.chain_gemm_tactic(plan)
                        assert (tactic is not None) == inside, (arch, m, n, k, out_f16)


def test_gemm_alone_entry_prefetch_rows():
    # The GEMM alone runs the entry L2 prefetch program exactly on the cells of
    # GEMM_ALONE_EP_RULES, each inside a CHAIN_EP_RULES cell, with the plan's grid.
    for arch, cells in cb.GEMM_ALONE_EP_RULES.items():
        for family, (lo, hi) in cells.items():
            chain = cb.CHAIN_EP_RULES[arch][family]
            assert chain[0] <= lo <= hi <= chain[1], (arch, family)
        sm_count = cb.MEASURED_SM_COUNT[arch]
        served = set()
        for k, n in VALIDATED_GEMM_FAMILIES:
            for m in _EDGE_M:
                for out_f16 in (False, True):
                    plan = cb.gemm_plan(m, n, k, out_f16, arch, sm_count)
                    if plan.kernel_key not in KERNELS[arch]:
                        continue
                    family = "_".join(
                        p
                        for p in plan.kernel_key.removeprefix("gemm:").split("_")
                        if p not in ("bf16", "f16")
                    )
                    cell = cells.get(family)
                    inside = cell is not None and cell[0] <= m <= cell[1]
                    tactic = cb.gemm_alone_tactic(plan)
                    assert (tactic is not None) == inside, (arch, m, n, k, out_f16)
                    if tactic is None:
                        continue
                    assert tactic == cb.chain_gemm_tactic(plan)
                    alone = _registered(
                        cb.gemm_plan(m, n, k, out_f16, arch, sm_count, tactic=tactic),
                        arch,
                    )
                    assert alone.kernel_key == plan.kernel_key + "_ep4"
                    assert (alone.grid, alone.num_tiles) == (plan.grid, plan.num_tiles)
                    served.add(family)
        assert served == set(cells), (arch, sorted(set(cells) - served))
    for arch in cb.MEASURED_SM_COUNT:
        if arch not in cb.GEMM_ALONE_EP_RULES:
            plan = cb.gemm_plan(
                128, 8192, 8192, False, arch, cb.MEASURED_SM_COUNT[arch]
            )
            assert cb.gemm_alone_tactic(plan) is None


def test_n16_split3_rows():
    # The 16-token tile with three K slices on the measured rows of each SM count and
    # output dtype: M = 17 stands for 17..24 and M = 32 for 25..32 where both dtypes
    # agree, else only the measured token count.
    def n16sk3(m, n, sm_count, out_f16, k=7168):
        t = cb.default_tactic(m, n, k, sm_count, out_f16=out_f16)
        assert t["alpha_n"] and t.get("split_k") in (2, 3), (m, n, sm_count, t)
        return (t["tile_n"], t["split_k"]) == (16, 3)

    tail = list(range(25, 33))
    for out_f16 in (False, True):
        assert all(n16sk3(m, 2112, 152, out_f16) for m in range(17, 33))
        assert [m for m in range(17, 33) if n16sk3(m, 1536, 152, out_f16)] == tail
    assert [m for m in range(17, 33) if n16sk3(m, 2112, 148, False)] == tail
    assert [m for m in range(17, 33) if n16sk3(m, 2112, 148, True)] == [17, *tail]
    assert [m for m in range(17, 33) if n16sk3(m, 1536, 148, False)] == [32]
    assert not any(n16sk3(m, 1536, 148, True) for m in range(17, 33))
    # Other parts and depths keep the 8-token tile.
    for out_f16 in (False, True):
        assert not any(n16sk3(m, 2112, 132, out_f16) for m in range(17, 33))
        assert not any(n16sk3(m, 2112, 152, out_f16, k=4096) for m in range(17, 33))
    # gemm_plan passes the output dtype to the rule.
    for out_f16, out in ((False, "bf16"), (True, "f16")):
        for arch in ("sm_100a", "sm_103a"):
            plan = cb.gemm_plan(32, 2112, 7168, out_f16, arch, 152)
            assert plan.kernel_key == f"gemm:n16_sk3_{out}_l2256b", plan.kernel_key
    assert cb.gemm_plan(17, 2112, 7168, True, "sm_100a", 148).tactic["tile_n"] == 16
    assert cb.gemm_plan(17, 2112, 7168, False, "sm_100a", 148).tactic["tile_n"] == 8


def _registered(plan, arch):
    """The plan's kernel resolves to a generated program of ``arch``."""
    assert plan.kernel_key in KERNELS[arch], plan.kernel_key
    return plan


def _ep4_key(key):
    """The entry L2 prefetch key of the program ``key``: ``_ep4`` goes before a trailing
    token-remainder ``_fold``."""
    if key.endswith("_fold"):
        return key.removesuffix("_fold") + "_ep4_fold"
    return key + "_ep4"


def test_gemm_plan_geometry():
    plan = _registered(
        cb.gemm_plan(8192, 7168, 16384, False, "sm_100a", 148), "sm_100a"
    )
    assert plan.tactic["two_cta"] and plan.tactic["sched"] == "clc"
    assert plan.tok_tile == 256 and plan.w_tile == 256
    assert plan.tok_tiles == 32 and plan.w_tiles == 28
    assert plan.grid == 2 * plan.num_tiles  # CLC launches the whole tile domain
    plan = _registered(cb.gemm_plan(1, 2112, 7168, True, "sm_100a", 148), "sm_100a")
    assert plan.tok_tile == 8 and plan.w_tile == 128 and plan.num_tiles == 17
    assert plan.grid == 4 * 17  # four K slices per weight tile
    # Half-M pairs: 64 token rows per CTA, 128 per pair tile.
    plan = _registered(cb.gemm_plan(130, 2112, 7168, False, "sm_100a", 148), "sm_100a")
    assert plan.tactic["half_m"] and plan.tok_tile == 128 and plan.tok_tiles == 2
    assert plan.num_tiles == 66 and plan.grid == 2 * min(74, plan.num_tiles)
    plan = _registered(cb.gemm_plan(128, 2112, 7168, False, "sm_100a", 148), "sm_100a")
    assert plan.tactic["half_m"] and plan.tok_tiles == 1 and plan.w_tiles == 33
    assert plan.grid == 2 * 33
    plan = _registered(cb.gemm_plan(257, 2112, 7168, False, "sm_100a", 148), "sm_100a")
    assert "half_m" not in plan.tactic and plan.tok_tile == 256
    plan = _registered(cb.gemm_plan(128, 18432, 7168, False, "sm_100a", 148), "sm_100a")
    assert plan.w_tiles == 144 and plan.grid == 144
    plan = _registered(cb.gemm_plan(8, 28672, 8192, False, "sm_100a", 148), "sm_100a")
    assert plan.num_tiles == 224 and plan.grid == 224  # min(2 x 148, 224)
    plan = _registered(cb.gemm_plan(8, 18432, 7168, True, "sm_103a", 152), "sm_103a")
    assert plan.grid == 144
    # The small token counts between the swapped-orientation tiles and an off-matrix
    # (K, N) pair resolve to registered programs on every architecture, including a part
    # whose SM count is outside the validated set.
    for arch, sm_count in (("sm_100a", 148), ("sm_103a", 152), ("sm_100a", 132)):
        for n, k in ((2112, 7168), (8192, 8192), (18432, 7168), (3200, 4096)):
            for m in (9, 12, 16):
                _registered(cb.gemm_plan(m, n, k, False, arch, sm_count), arch)
    # Rows past the small-token tiles are generated for the validated SM count of each
    # architecture (:func:`required_kernel_keys`).  A part with a different SM count can
    # resolve such a row to a tile the package does not carry -- the plain 2-CTA m192
    # schedule, which the stream-K tail replaces at 148 and 152 SMs -- and the backend
    # then names the missing key instead of launching something else.
    for arch, sm_count in (("sm_100a", 148), ("sm_103a", 152)):
        for n, k in ((2112, 7168), (8192, 8192), (18432, 7168), (3200, 4096)):
            _registered(cb.gemm_plan(300, n, k, False, arch, sm_count), arch)
    # Stream-K tail of the static 2-CTA schedule: each tile of the partial last wave is cut into
    # equal K slices over consecutive pairs; the plan resolves the slice count into the tactic
    # (it is baked into the program) and carries the geometry the kernel re-derives.
    sk_tactic = {**cb.default_tactic(2048, 7168, 16384, 148), "stream_k": True}
    plan = cb.gemm_plan(2048, 7168, 16384, False, "sm_100a", 148, tactic=sk_tactic)
    assert plan.num_tiles == 304 and plan.grid == 2 * 74 and plan.stream_k
    assert (plan.sk_tiles, plan.sk_slices, plan.sk_pairs) == (8, 6, 48)
    assert plan.tactic["sk_split"] == 6
    assert plan.kernel_key == "gemm:m192_2cta_skt6_bf16_aL_bL_l2256b"
    assert (
        plan.sk_flag_words == 8 * 6 * 2
        and plan.sk_workspace_floats == 8 * 6 * 2 * 128 * 192
    )
    plan = cb.gemm_plan(
        2048, 7168, 16384, False, "sm_100a", 148, tactic={**sk_tactic, "sk_split": 4}
    )
    assert (plan.sk_tiles, plan.sk_slices, plan.sk_pairs) == (8, 4, 32)
    assert plan.kernel_key == "gemm:m192_2cta_skt4_bf16_aL_bL_l2256b"
    # fewer than SK_MIN_SLICES slices are not worth the exchange: the plain program
    plan = cb.gemm_plan(
        2048, 7168, 16384, False, "sm_100a", 148, tactic={**sk_tactic, "sk_split": 2}
    )
    assert not plan.stream_k and "sk_split" not in plan.tactic
    assert plan.kernel_key == "gemm:m192_2cta_bf16_aL_bL_l2256b"
    # the default tactic of every static 2-CTA row carries the knob; the key needs the plan
    assert cb.default_tactic(2048, 7168, 16384, 148)["stream_k"]
    assert "stream_k" not in cb.default_tactic(128, 1536, 7168, 148)  # half_m pairs
    # static 2x256 pairs (192 tiles < 10 per pair) vs the CLC scheduler (896 tiles)
    assert cb.default_tactic(8192, 1536, 7168, 148)["stream_k"]
    assert "stream_k" not in cb.default_tactic(8192, 7168, 16384, 148)
    with pytest.raises(ValueError, match="resolved sk_split"):
        cb.gemm_kernel_key(cb.default_tactic(2048, 7168, 16384, 148), False)
    plan = cb.gemm_plan(2048, 7168, 16384, False, "sm_100a", 148)
    assert plan.kernel_key == "gemm:m192_2cta_skt6_bf16_aL_bL_l2256b"
    # 40 tail tiles over 76 pairs cannot be split: the plain program (tactic loses the knob)
    plan = cb.gemm_plan(
        2048,
        8192,
        8192,
        False,
        "sm_103a",
        152,
        tactic={**cb.default_tactic(2048, 8192, 8192, 152), "stream_k": True},
    )
    assert plan.num_tiles == 344 and not plan.stream_k and "sk_split" not in plan.tactic
    assert (plan.sk_tiles, plan.sk_slices, plan.sk_pairs) == (0, 0, 0)
    assert plan.kernel_key == "gemm:m192_2cta_bf16_aL_bL_l2256b"
    plan = cb.gemm_plan(
        2048,
        2112,
        7168,
        False,
        "sm_100a",
        148,
        tactic={**cb.default_tactic(2048, 2112, 7168, 148), "stream_k": True},
    )
    assert plan.num_tiles == 72 and not plan.stream_k
    assert cb.stream_k_tail(344, 76, 32) == (40, 1, 0)
    assert cb.stream_k_tail(300, 74, 32, 8, 6) == (4, 6, 24)
    assert cb.stream_k_tail(256, 76, 32, 8, 8) == (
        28,
        1,
        0,
    )  # two slices fit, below SK_MIN_SLICES
    assert cb.stream_k_tail(256, 76, 32, 8, 8, min_slices=2) == (28, 2, 56)
    assert cb.stream_k_tail(152, 76, 32) == (0, 0, 0)
    with pytest.raises(ValueError, match="static 2-CTA"):
        cb.gemm_plan(
            8192,
            7168,
            16384,
            False,
            "sm_100a",
            148,
            tactic={**cb.default_tactic(8192, 7168, 16384, 148), "stream_k": True},
        )
    with pytest.raises(ValueError, match="1-CTA persistent"):
        cb.gemm_plan(
            8,
            2112,
            7168,
            False,
            "sm_100a",
            148,
            tactic={**cb.default_tactic(8, 2112, 7168, 148), "blocks_per_sm": 2},
        )
    with pytest.raises(ValueError, match="K="):
        cb.gemm_plan(128, 2112, 7000, False, "sm_100a", 148)
    with pytest.raises(ValueError, match="N % 8"):
        cb.gemm_plan(
            8,
            130,
            7168,
            False,
            "sm_100a",
            148,
            tactic={**cb.default_tactic(8, 2112, 7168, 148), "split_k": 1},
        )
    # An explicit override with a key no tactic uses is rejected by name instead
    # of being planned without it (the plan above would otherwise ignore it).
    with pytest.raises(ValueError, match="unknown tactic keys: amc"):
        cb.gemm_plan(
            128,
            18432,
            7168,
            False,
            "sm_100a",
            148,
            tactic={**cb.default_tactic(128, 18432, 7168, 148), "amc": 2},
        )


def test_sm103_k96_width_rule():
    # On the 152-SM sm_103a part the 2-CTA width re-pick takes the K = 96 programs' wave
    # cost when the re-picked plan is one of them: three multi-wave rows move from five
    # waves of 192-wide pairs to four waves of the 256-wide grouped-raster pair.
    for m, n, k in ((8192, 2112, 7168), (2048, 8192, 8192), (2048, 8192, 28672)):
        for out_f16, out in ((False, "bf16"), (True, "f16")):
            plan = _registered(
                cb.gemm_plan(m, n, k, out_f16, "sm_103a", 152), "sm_103a"
            )
            assert plan.kernel_key == f"gemm:m256_2cta_{out}_aL_bL_l2256b_g8"
            assert plan.kernel_key in cb.K96_KERNEL_KEYS
            for arch, sm_count in (("sm_100a", 148), ("sm_100a", 152)):
                base = cb.gemm_plan(m, n, k, out_f16, arch, sm_count)
                assert base.tactic["tile_n"] == 192, (arch, sm_count, base.kernel_key)
    # A row whose re-picked plan is not a K = 96 program keeps the K = 64 plan.
    for m, n, k in ((2048, 1536, 7168), (8192, 8192, 8192), (512, 8192, 8192)):
        assert (
            cb.gemm_plan(m, n, k, False, "sm_103a", 152).kernel_key
            == cb.gemm_plan(m, n, k, False, "sm_100a", 152).kernel_key
        )


def test_sf_ring_rows():
    # On sm_103a at 152 SMs the GEMM alone and the chain run the scale-factor ring
    # program (``_sfr``) of every row of the CLC K = 96 programs and of the static K = 96
    # programs on the SF_RING_G8_CELLS ranges of their kind, with the plan's grid.  The
    # token counts next to a static range keep the plan's program, the chain keeps it on
    # the cells of the GEMM alone only, the other parts never take the ring, and the knob
    # refuses other programs and architectures.  The cells of SF_RING_MC42_CELLS run the
    # multicast form (test_sf_ring_mc42_rows).
    arch = cb.SF_RING_ARCH
    sm_count = cb.MEASURED_SM_COUNT[arch]

    def runs(plan, kind):
        hook = cb.gemm_alone_tactic if kind == "gemm" else cb.chain_gemm_tactic
        tactic = hook(plan)
        if tactic is None:
            return plan
        return _registered(
            cb.gemm_plan(
                plan.M, plan.N, plan.K, plan.out_f16, arch, sm_count, tactic=tactic
            ),
            arch,
        )

    clc = 0
    for k, n in VALIDATED_GEMM_FAMILIES:
        for m in (*_EDGE_M, 5889, 6913, 12288):
            for out_f16 in (False, True):
                plan = cb.gemm_plan(m, n, k, out_f16, arch, sm_count)
                if plan.kernel_key not in cb.SF_RING_CLC_KEYS:
                    continue
                clc += 1
                for kind in ("gemm", "chain"):
                    if _mc42_cell(kind, m, n, k):
                        continue
                    run = runs(plan, kind)
                    assert run.kernel_key == plan.kernel_key + "_sfr", (kind, m, n, k)
                    assert (run.grid, run.num_tiles) == (plan.grid, plan.num_tiles)
    assert clc
    for kind, cells in cb.SF_RING_G8_CELLS.items():
        for (k, n), (lo, hi) in cells.items():
            for out_f16 in (False, True):
                for m in (lo, (lo + hi) // 2, hi):
                    plan = cb.gemm_plan(m, n, k, out_f16, arch, sm_count)
                    assert plan.kernel_key in cb.SF_RING_G8_KEYS, (m, n, k)
                    if _mc42_cell(kind, m, n, k):
                        continue
                    run = runs(plan, kind)
                    assert run.kernel_key == plan.kernel_key + "_sfr", (kind, m, n, k)
                    assert (run.grid, run.num_tiles) == (plan.grid, plan.num_tiles)
                for m in (lo - 1, hi + 1):
                    plan = cb.gemm_plan(m, n, k, out_f16, arch, sm_count)
                    assert not runs(plan, kind).kernel_key.endswith("_sfr"), (kind, m)
                if kind == "gemm" and (k, n) not in cb.SF_RING_G8_CELLS["chain"]:
                    plan = cb.gemm_plan(hi, n, k, out_f16, arch, sm_count)
                    assert cb.chain_gemm_tactic(plan) == plan.tactic
                    assert runs(plan, "chain").kernel_key == plan.kernel_key
    for other, other_sms in (("sm_103a", 148), ("sm_100a", 148), ("sm_100a", 152)):
        for k, n in VALIDATED_GEMM_FAMILIES:
            for m in (1281, 1537, 2048, 2304, 5889, 6913, 7937, 8192, 9728, 16384):
                for out_f16 in (False, True):
                    plan = cb.gemm_plan(m, n, k, out_f16, other, other_sms)
                    for hook in (cb.gemm_alone_tactic, cb.chain_gemm_tactic):
                        tactic = hook(plan)
                        assert tactic is None or not tactic.get("sf_ring"), (other, m)
    plan = cb.gemm_plan(8192, 7168, 16384, False, arch, sm_count)
    assert plan.kernel_key in cb.SF_RING_CLC_KEYS
    ring = {**plan.tactic, "sf_ring": True}
    for other, tactic in (
        ("sm_100a", ring),
        (arch, {**ring, "raster_group": 8}),
        (arch, {**ring, "tile_n": 192}),
        (arch, {**ring, "b_hint": "evict_first"}),
    ):
        with pytest.raises(ValueError, match="sf_ring"):
            cb.gemm_plan(8192, 7168, 16384, False, other, sm_count, tactic=tactic)


def test_sf_ring_form_rows():
    # On sm_103a at 152 SMs the GEMM alone and the chain run the ring program of the
    # cell's form on the SF_RING_FORM_CELLS ranges of their kind (the plan's static
    # program without raster groups, or CLC program without or with raster groups of
    # 8); the token counts next to a range keep the plan's program, the chain keeps it
    # on the cells of the GEMM alone only, and the other parts never take the ring.  The
    # cells of SF_RING_MC42_CELLS run the multicast form (test_sf_ring_mc42_rows).
    arch = cb.SF_RING_ARCH
    sm_count = cb.MEASURED_SM_COUNT[arch]

    def runs(plan, kind):
        hook = cb.gemm_alone_tactic if kind == "gemm" else cb.chain_gemm_tactic
        tactic = hook(plan)
        if tactic is None:
            return plan
        return _registered(
            cb.gemm_plan(
                plan.M, plan.N, plan.K, plan.out_f16, arch, sm_count, tactic=tactic
            ),
            arch,
        )

    def ring_key(plan, form):
        tactic = {
            name: value
            for name, value in plan.tactic.items()
            if name not in ("sched", "stream_k", "sk_split")
        }
        tactic.update(cb.SF_RING_FORMS[form], sf_ring=True)
        return cb.gemm_kernel_key(tactic, plan.out_f16)

    checked = 0
    for kind, cells in cb.SF_RING_FORM_CELLS.items():
        for (k, n), ranges in cells.items():
            for lo, hi, form in ranges:
                top = hi if hi is not None else 16384
                for out_f16 in (False, True):
                    for m in (lo, (lo + top) // 2, top):
                        plan = cb.gemm_plan(m, n, k, out_f16, arch, sm_count)
                        assert plan.kernel_key in cb.SF_RING_FORM_KEYS, (m, n, k)
                        if _mc42_cell(kind, m, n, k):
                            continue
                        run = runs(plan, kind)
                        assert run.kernel_key == ring_key(plan, form), (kind, m, n, k)
                        assert run.kernel_key.endswith("_sfr")
                        assert run.num_tiles == plan.num_tiles
                        same_sched = (plan.tactic.get("sched") == "clc") == (
                            cb.SF_RING_FORMS[form].get("sched") == "clc"
                        )
                        assert run.grid == plan.grid or not same_sched
                        checked += 1
                    for m in (lo - 1,) + ((hi + 1,) if hi is not None else ()):
                        plan = cb.gemm_plan(m, n, k, out_f16, arch, sm_count)
                        covered = any(
                            a <= m and (b is None or m <= b) for a, b, _ in ranges
                        )
                        if plan.kernel_key in cb.SF_RING_FORM_KEYS and not covered:
                            run = runs(plan, kind)
                            assert not run.kernel_key.endswith("_sfr"), (kind, m)
                    chain_ranges = cb.SF_RING_FORM_CELLS["chain"].get((k, n), ())
                    if kind == "gemm" and not any(
                        a <= lo and (b is None or lo <= b) for a, b, _ in chain_ranges
                    ):
                        plan = cb.gemm_plan(lo, n, k, out_f16, arch, sm_count)
                        assert cb.chain_gemm_tactic(plan) == plan.tactic
                        assert runs(plan, "chain").kernel_key == plan.kernel_key
    assert checked
    for other, other_sms in (("sm_103a", 148), ("sm_100a", 148), ("sm_100a", 152)):
        for cells in cb.SF_RING_FORM_CELLS.values():
            for (k, n), ranges in cells.items():
                for lo, hi, _ in ranges:
                    for m in (lo, hi if hi is not None else 16384):
                        for out_f16 in (False, True):
                            plan = cb.gemm_plan(m, n, k, out_f16, other, other_sms)
                            for hook in (cb.gemm_alone_tactic, cb.chain_gemm_tactic):
                                tactic = hook(plan)
                                assert tactic is None or not tactic.get("sf_ring")


def _mc42_cell(kind, m, n, k):
    """``M`` lies in a token range of the ``(K, N)`` cell of
    ``SF_RING_MC42_CELLS[kind]``."""
    ranges = cb.SF_RING_MC42_CELLS[kind].get((k, n), ())
    return any(lo <= m <= hi for lo, hi in ranges)


def test_sf_ring_mc42_rows():
    # On sm_103a at 152 SMs the GEMM alone and the chain run the cluster-of-8 multicast
    # form of the CLC ring program without raster groups (``_clc_sfr_mc42``) on the
    # SF_RING_MC42_CELLS ranges of their kind, with the plan's tiles and the CLC grid
    # (one CTA pair per tile).  The token counts next to a range keep their programs,
    # the other parts never take the form, and the knob refuses odd tile counts and
    # the programs other than the CLC ring program without raster groups.
    arch = cb.SF_RING_ARCH
    sm_count = cb.MEASURED_SM_COUNT[arch]
    checked = 0
    for kind, cells in cb.SF_RING_MC42_CELLS.items():
        hook = cb.gemm_alone_tactic if kind == "gemm" else cb.chain_gemm_tactic
        for (k, n), ranges in cells.items():
            for (lo, hi), out_f16 in ((r, f) for r in ranges for f in (False, True)):
                out = "f16" if out_f16 else "bf16"
                key = f"gemm:m256_2cta_{out}_aL_bL_l2256b_clc_sfr_mc42"
                for m in (lo, (lo + hi) // 2, hi):
                    plan = cb.gemm_plan(m, n, k, out_f16, arch, sm_count)
                    tactic = hook(plan)
                    assert tactic is not None and tactic.get("mc42"), (kind, m, n, k)
                    run = _registered(
                        cb.gemm_plan(m, n, k, out_f16, arch, sm_count, tactic=tactic),
                        arch,
                    )
                    assert run.kernel_key == key, (kind, m, n, k)
                    tiles = (run.tok_tiles, run.w_tiles)
                    assert tiles == (plan.tok_tiles, plan.w_tiles)
                    assert run.grid == 2 * run.num_tiles
                    checked += 1
                for m in (lo - 1, hi + 1):
                    plan = cb.gemm_plan(m, n, k, out_f16, arch, sm_count)
                    tactic = hook(plan)
                    assert tactic is None or not tactic.get("mc42"), (kind, m, n, k)
    assert checked
    # The rows next to a range have odd token tile counts; every row between the
    # (8192, 8192) ranges whose program the multicast form could take (a ring program
    # with even token and weight tile counts) keeps its program without that form.
    ring = cb.SF_RING_FORM_KEYS | cb.SF_RING_G8_KEYS | cb.SF_RING_CLC_KEYS
    (_, end), (start, _) = cb.SF_RING_MC42_CELLS["gemm"][(8192, 8192)]
    between = 0
    for m, out_f16 in ((m, f) for m in range(end + 1, start) for f in (False, True)):
        plan = cb.gemm_plan(m, 8192, 8192, out_f16, arch, sm_count)
        if plan.kernel_key not in ring or plan.tok_tiles % 2 or plan.w_tiles % 2:
            continue
        for hook in (cb.gemm_alone_tactic, cb.chain_gemm_tactic):
            tactic = hook(plan)
            assert tactic is None or not tactic.get("mc42"), (hook, m)
            if plan.kernel_key in cb.SF_RING_CLC_KEYS:
                assert tactic is not None and tactic.get("sf_ring"), (hook, m)
        between += 1
    assert between
    for other, other_sms in (("sm_103a", 148), ("sm_100a", 148), ("sm_100a", 152)):
        for cells in cb.SF_RING_MC42_CELLS.values():
            for (k, n), ranges in cells.items():
                for m in (x for lo_hi in ranges for x in lo_hi):
                    for out_f16 in (False, True):
                        plan = cb.gemm_plan(m, n, k, out_f16, other, other_sms)
                        for hook in (cb.gemm_alone_tactic, cb.chain_gemm_tactic):
                            tactic = hook(plan)
                            assert tactic is None or not tactic.get("mc42"), (other, m)
    plan = cb.gemm_plan(2048, 8192, 8192, False, arch, sm_count)
    mc42 = cb.gemm_alone_tactic(plan)
    assert mc42 is not None and mc42.get("mc42")
    # Odd token (M = 2049: 9 tiles) or weight (N = 8448: 33 tiles) tile counts.
    for m, n in ((2049, 8192), (2048, 8448), (2049, 8448)):
        with pytest.raises(ValueError, match="mc42"):
            cb.gemm_plan(m, n, 8192, False, arch, sm_count, tactic=mc42)
    for tactic in (
        {**mc42, "sf_ring": False},
        {**mc42, "raster_group": 16},
        {**mc42, "sched": "static", "raster_group": 8},
    ):
        with pytest.raises(ValueError, match="mc42"):
            cb.gemm_plan(2048, 8192, 8192, False, arch, sm_count, tactic=tactic)


def test_cake_is_registered_but_never_auto_selected():
    assert "cake" in mm_fp4.experimental_backends
    assert mm_fp4.is_backend_supported("cake", 100)
    assert mm_fp4.is_backend_supported("cake", 103)
    assert not mm_fp4.is_backend_supported("cake", 120)


@pytest.mark.parametrize("arch", cb.ARCHES)
def test_required_gemm_keys_registered_when_programs_exist(arch):
    required = [k for k in required_kernel_keys(arch) if k.startswith("gemm:")]
    assert required
    if arch in KERNELS:
        missing = sorted(set(required) - set(KERNELS[arch]))
        assert not missing, f"{arch} lacks {missing}"


# Token counts at the edges of the dispatch rules: the swapped-orientation tiles, the
# single-token-tile and 2-CTA limits, the validated rows and their neighbours.
_EDGE_M = (1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 129, 130, 255, 256)
_EDGE_M += (257, 511, 512, 513, 1000, 1024, 2047, 2048, 2049, 4096, 4097, 8191, 8192)
_EDGE_M += (8193, 16384)


@pytest.mark.parametrize("sm_count", (148, 152))
@pytest.mark.parametrize("arch", cb.ARCHES)
def test_plans_resolve_on_148_and_152_sms(arch, sm_count):
    # Each architecture ships on a 148- and a 152-SM part (sm_100a: B200 / GB200,
    # sm_103a: B300 / GB300) and registers the kernels of its measured part.  The
    # validated matrix resolves on both parts; on the other part an edge row whose own
    # tactic selects an unregistered kernel runs the measured part's default kernel (the
    # FOLD_CELLS / B2_FOLD_CELLS rules hold at the measured SM count only), and every
    # other row keeps its own.
    registered = KERNELS[arch]
    missing = sorted(set(required_kernel_keys(arch, sm_count)) - set(registered))
    assert not missing, f"{arch} with {sm_count} SMs lacks {missing}"
    measured = cb.MEASURED_SM_COUNT[arch]
    served = 0
    for k, n in (*VALIDATED_GEMM_FAMILIES, VALIDATED_OFF_MATRIX_FAMILY):
        for m in _EDGE_M:
            for out_f16 in (False, True):
                reference = cb.gemm_plan(m, n, k, out_f16, arch, measured)
                if reference.unregistered_key is not None:
                    continue  # outside the registered matrix on the measured part too
                served += 1
                plan = cb.gemm_plan(m, n, k, out_f16, arch, sm_count)
                key = plan.kernel_key
                assert key in registered, (m, n, k, out_f16, key)
                assert plan.sm_count == sm_count
                if sm_count == measured or plan.unfit_key is not None:
                    continue
                own = cb.default_tactic(
                    m,
                    n,
                    k,
                    sm_count,
                    out_f16=out_f16,
                    n8_wide_w_rows=arch == cb.N8_WIDE_W_ROWS_ARCH,
                )
                own_key = cb.gemm_plan(
                    m, n, k, out_f16, arch, sm_count, tactic=own
                ).kernel_key
                if own_key in registered:
                    assert key == own_key, (m, n, k, out_f16)
                else:
                    default = cb._gemm_plan(m, n, k, out_f16, arch, measured)
                    assert key == default.kernel_key, (m, n, k, out_f16)
    assert served


def test_other_sm_count_runs_the_measured_kernel():
    # GB200 (sm_100a, 152 SMs): two CTAs per SM on the 16 / 32-token tiles, the
    # 128 x 192 one-CTA tile and the plain 192-wide 2-CTA pair are registered for
    # sm_103a only.  B300 (sm_103a, 148 SMs): the stream-K tail programs are registered
    # for sm_100a only.  These rows run the measured part's kernel with the grid of
    # their own SM count.
    rows = (
        ("sm_100a", 152, 16, 28672, 8192, "gemm:n16_bf16_aF_l2256b", 152),
        ("sm_100a", 152, 32, 28672, 8192, "gemm:n32_bf16_aF_l2256b", 152),
        ("sm_100a", 152, 128, 28672, 8192, "gemm:m128_bf16_bF_ro", 152),
        ("sm_100a", 152, 257, 7168, 16384, "gemm:m256_2cta_bf16_bF_l2256b", 112),
        ("sm_103a", 148, 512, 28672, 8192, "gemm:m256_2cta_bf16_bF_l2256b", 148),
        ("sm_103a", 148, 2048, 7168, 16384, "gemm:m256_2cta_bf16_aL_bL_l2256b_g8", 148),
    )
    for arch, sm_count, m, n, k, key, grid in rows:
        own = cb.default_tactic(
            m, n, k, sm_count, n8_wide_w_rows=arch == cb.N8_WIDE_W_ROWS_ARCH
        )
        own_key = cb.gemm_plan(m, n, k, False, arch, sm_count, tactic=own).kernel_key
        assert own_key not in KERNELS[arch], own_key
        plan = cb.gemm_plan(m, n, k, False, arch, sm_count)
        measured = cb.gemm_plan(m, n, k, False, arch, cb.MEASURED_SM_COUNT[arch])
        assert plan.kernel_key == measured.kernel_key == key, plan.kernel_key
        assert plan.sm_count == sm_count and plan.grid == grid, plan.grid
    # An explicit tactic is planned as given.
    t = cb.default_tactic(32, 28672, 8192, 152)
    plan = cb.gemm_plan(32, 28672, 8192, False, "sm_100a", 152, tactic=t)
    assert plan.kernel_key == "gemm:n32_bf16_aF_l2256b_o2"


@pytest.mark.parametrize("sm_count", (148, 152))
@pytest.mark.parametrize("arch", cb.ARCHES)
def test_every_default_plan_runs_a_registered_program(arch, sm_count):
    # Token counts between the validated rows select tactics whose kernels are not
    # registered on either part (for example the 16-wide raster group without the CLC
    # scheduler, or a stream-K tail of another slice count).  Every default plan runs a
    # registered program: its own when registered, else the measured part's kernel,
    # else the nearest registered program (on the other part: the one the measured
    # part runs, when its grid reaches it), which records the unregistered key.
    registered = KERNELS[arch]
    measured = cb.MEASURED_SM_COUNT[arch]
    nearest = 0
    for k, n in (*VALIDATED_GEMM_FAMILIES, VALIDATED_OFF_MATRIX_FAMILY):
        for m in (*range(1, 16385, 29), *_EDGE_M):
            for out_f16 in (False, True):
                own = _route(m, n, k, out_f16, arch, sm_count, nearest=False)
                plan = cb.gemm_plan(m, n, k, out_f16, arch, sm_count)
                assert plan.kernel_key in registered, (m, n, k, out_f16, own.kernel_key)
                assert plan.sm_count == sm_count
                assert cb._tma_boxes_fit(plan), (m, n, k, out_f16, plan.kernel_key)
                if plan.unfit_key is not None:
                    # the registered program's TMA boxes do not fit (tested below)
                    route = _route(m, n, k, out_f16, arch, sm_count)
                    assert plan.unfit_key == route.kernel_key, (m, n, k, out_f16)
                    continue
                if own.kernel_key in registered:
                    assert plan.unregistered_key is None
                    assert (plan.kernel_key, plan.grid) == (own.kernel_key, own.grid)
                    continue
                reference = cb.gemm_plan(m, n, k, out_f16, arch, measured)
                if plan.unregistered_key is None:
                    # the measured part's kernel (see the test above)
                    assert sm_count != measured
                    assert plan.kernel_key == reference.kernel_key
                    continue
                nearest += 1
                assert plan.unregistered_key == own.kernel_key, (m, n, k, out_f16)
                if plan.kernel_key != reference.kernel_key:
                    assert (plan.alpha_n, plan.two_cta) == (own.alpha_n, own.two_cta)
    assert nearest


def test_off_matrix_rows_run_the_nearest_registered_program():
    # The tile width, orientation and CTA group of the default tactic are kept; the raster
    # group, the stream-K slice count, the scheduler and the operand cache hints move to
    # the closest registered program.  The other part runs the measured part's program.
    rows = (
        (2049, 8192, 8192, False, "gemm:m256_2cta_bf16_aL_bL_l2256b_g8"),
        (9729, 1536, 7168, False, "gemm:m192_2cta_bf16_aL_bL_l2256b_g8"),
        (600, 1536, 7168, True, "gemm:m128_2cta_f16_bF_l2256b"),
        (12, 2112, 7168, True, "gemm:n16_sk3_f16_l2256b"),
    )
    for m, n, k, out_f16, key in rows:
        own = cb._gemm_plan(m, n, k, out_f16, "sm_100a", 148)
        assert own.kernel_key not in KERNELS["sm_100a"], own.kernel_key
        plan = cb.gemm_plan(m, n, k, out_f16, "sm_100a", 148)
        assert plan.kernel_key == key and plan.unregistered_key == own.kernel_key
        other = cb.gemm_plan(m, n, k, out_f16, "sm_100a", 152)
        assert other.kernel_key == key and other.sm_count == 152
    plan = cb.gemm_plan(4865, 7168, 16384, False, "sm_103a", 152)
    assert plan.unregistered_key == "gemm:m192_2cta_bf16_aL_bL_l2256b_clc"
    assert plan.kernel_key == "gemm:m192_2cta_bf16_aL_bL_l2256b"
    # An explicit tactic is planned as given, registered or not.
    own = cb._gemm_plan(2049, 8192, 8192, False, "sm_100a", 148)
    plan = cb.gemm_plan(2049, 8192, 8192, False, "sm_100a", 148, tactic=own.tactic)
    assert plan.kernel_key == own.kernel_key and plan.unregistered_key is None


def _route(m, n, k, out_f16, arch, sm_count, nearest=True):
    """:func:`gemm_plan` without the TMA box check (the program the shape names, whether
    or not its host binding accepts the shape) and, with ``nearest=False``, without the
    nearest-registered-program fallback (the default plan, registered or not)."""
    saved = cb._tma_fitting_plan, cb._nearest_registered_plan
    cb._tma_fitting_plan = lambda plan: plan
    if not nearest:
        cb._nearest_registered_plan = lambda plan: plan
    try:
        return cb.gemm_plan(m, n, k, out_f16, arch, sm_count)
    finally:
        cb._tma_fitting_plan, cb._nearest_registered_plan = saved


def _binding_boxes(arch, key):
    """``{descriptor: (box, checked dims)}`` of the TMA descriptors the generated host
    binding of ``key``'s program on ``arch`` encodes (read from its source)."""
    import re
    from pathlib import Path

    from flashinfer.experimental.cake_nvfp4_per_token.cake_jit import MODULES

    root = Path(cb.__file__).resolve().parent / "csrc"
    text = (root / MODULES[KERNELS[arch][key]]["sources"][1]).read_text()
    common = root / "cake_nvfp4_per_token" / "cake_nvfp4_per_token_host_common.cuh"
    encoder = r"CUtensorMap EncodeTma_(\w+)\(const TensorView& t\) \{(.*?)\n\}"
    bodies = dict(re.findall(encoder, text, re.S))
    for name, variant in re.findall(r"#define CAKE_TMA_(\w+) (\d+)", text):
        variant_re = rf"#(?:el)?if CAKE_TMA_{name} == {variant}\n.*?{encoder}"
        found = re.search(variant_re, common.read_text(), re.S)
        bodies[found[1]] = found[2]
    boxes = {}
    for name, body in bodies.items():
        box = re.search(r"box_dim\[\d\] = \{(.*?)\};", body)[1]
        oob = re.search(r"allow_oob_box\[\d\] = \{(.*?)\};", body)
        if oob:  # a box may exceed the axes listed as out-of-bounds tiles
            flags = [v.strip() for v in oob[1].split(",")]
            dims = tuple(i for i, v in enumerate(flags) if v == "false")
        else:
            dims = tuple(
                int(i) for i in re.findall(r"box_dim\[(\d)\] <= global_dim", body)
            )
        boxes[name] = (tuple(int(v) for v in re.findall(r"(\d+)u", box)), dims)
    return boxes


def _binding_accepts(boxes, plan):
    """The binding's box checks against the operand extents of ``plan``'s launch
    (``cake_backend._gemm_kwargs``), in the descriptors' global-dimension order."""
    m, n, k = plan.M, plan.N, plan.K
    rows_a, rows_b = (n, m) if plan.alpha_n else (m, n)
    cols = cb.padded_sf_cols(k) // 4
    atoms_a, atoms_b = (cb.padded_rows(r) // cb.ROW_TILE for r in (rows_a, rows_b))
    extents = {
        "A": (128, rows_a, k // 256),
        "B": (128, rows_b, k // 256),
        "SFA": (128, cols, atoms_a),
        "SFB": (128, cols, atoms_b),
        "out": (n, m),
    }
    if plan.alpha_n and int(plan.tactic["tile_n"]) < 32:
        extents["SFB"] = (32, cols, 4, atoms_b)  # eight-row scale groups
    return all(
        box[i] <= extents[name][i] for name, (box, dims) in boxes.items() for i in dims
    )


_GEMM_DESCRIPTORS = {"A", "B", "SFA", "SFB", "out"}


@pytest.mark.parametrize("arch", cb.ARCHES)
def test_tma_box_rule_matches_the_host_bindings(arch):
    # Each generated host binding refuses an operand smaller than one of its TMA boxes in
    # a dimension it checks (ValueError at prepare).  The planner's rule agrees with the
    # boxes of every registered GEMM program the fallback can select (a key of the
    # tactic form, encoding the five descriptors of the launch).
    checked = 0
    for key in KERNELS[arch]:
        parsed = cb._tactic_of_key(key) if key.startswith("gemm:") else None
        if parsed is None:
            continue
        tactic, out_f16 = parsed
        boxes = _binding_boxes(arch, key)
        assert set(boxes) == _GEMM_DESCRIPTORS, (key, sorted(boxes))
        for n in (8, 24, 32, 56, 64, 128, 136, 1536):
            for m in (1, 8, 32, 33, 63, 64, 127, 128, 129, 1024):
                sm_count = cb.MEASURED_SM_COUNT[arch]
                try:
                    plan = cb._gemm_plan(
                        m, n, 7168, out_f16, arch, sm_count, tactic=tactic
                    )
                except ValueError:
                    continue  # the swapped orientation holds M <= 32 only
                if plan.kernel_key != key:
                    continue
                checked += 1
                fits = cb._tma_boxes_fit(plan)
                assert fits == _binding_accepts(boxes, plan), (key, m, n, fits)
    assert checked


@pytest.mark.parametrize("sm_count", (148, 152))
@pytest.mark.parametrize("arch", cb.ARCHES)
def test_plans_whose_tma_boxes_do_not_fit_run_a_program_that_fits(arch, sm_count):
    # Narrow outputs and few tokens can name a registered program whose TMA boxes exceed
    # the operands (the 64-column output box of the swapped orientation for N < 64, the
    # 2-CTA output box rows for M < 128 / 64, the two-atom weight scale box of the wider
    # 2-CTA pairs for N <= 128).  Such a plan runs the nearest registered program whose
    # boxes fit; every other plan is unchanged.  No program fits N < 32 (each output box
    # is at least 32 columns wide): those plans stay as they are.
    registered = KERNELS[arch]
    refit = 0
    for n in (8, 16, 24, 32, 40, 48, 56, 64, 72, 96, 120, 128, 136, 192, 256, 1536):
        for m in (*range(1, 65), 96, 127, 128, 129, 192, 256, 512, 1024, 4096, 16384):
            for out_f16 in (False, True):
                route = _route(m, n, 7168, out_f16, arch, sm_count)
                plan = cb.gemm_plan(m, n, 7168, out_f16, arch, sm_count)
                if cb._tma_boxes_fit(route) or n < 32:
                    assert plan == route and plan.unfit_key is None, (m, n, out_f16)
                if cb._tma_boxes_fit(route):
                    continue
                # the binding of the program the plan replaces does refuse the shape
                boxes = _binding_boxes(arch, route.kernel_key)
                if set(boxes) == _GEMM_DESCRIPTORS:
                    assert not _binding_accepts(boxes, route), (m, n, route.kernel_key)
                if n < 32:
                    continue
                refit += 1
                assert plan.kernel_key in registered and cb._tma_boxes_fit(plan)
                assert plan.unfit_key == route.kernel_key, (m, n, out_f16)
                assert plan.unregistered_key == route.unregistered_key
    assert refit


def test_rows_whose_tma_boxes_do_not_fit_run_the_one_cta_m64_program():
    # The half-M 2-CTA pair stores 64-row output boxes and checks them against M: the
    # validated 7168 x 1536 / 7168 x 2112 rows of 33-63 tokens run the 1-CTA 64-wide
    # program instead.  The swapped orientation stores 64-column output boxes: N = 32-56
    # rows of up to 32 tokens run the same program (directly when their own swapped
    # program is not registered either).  An output narrower than 32 columns keeps its
    # plan (no registered program fits).
    rows = (
        (33, 1536, False, "gemm:m64_2cta_hm_bf16_bF_l2256b"),
        (63, 2112, True, "gemm:m64_2cta_hm_f16_bF_l2256b"),
        (48, 96, False, "gemm:m64_2cta_hm_bf16_bF_l2256b"),
        (1, 32, False, "gemm:n8_"),
        (16, 40, True, "gemm:n"),
        (32, 56, False, "gemm:n"),
    )
    for arch in cb.ARCHES:
        for sm_count in (148, 152):
            for m, n, out_f16, replaced in rows:
                plan = cb.gemm_plan(m, n, 7168, out_f16, arch, sm_count)
                key = "gemm:m64_f16_bF_l2256b" if out_f16 else "gemm:m64_bf16_bF_l2256b"
                assert plan.kernel_key == key, (arch, sm_count, m, n, plan.kernel_key)
                assert plan.sm_count == sm_count and cb._tma_boxes_fit(plan)
                own = plan.unfit_key or plan.unregistered_key
                assert own.startswith(replaced), (arch, sm_count, m, n, own)
            plan = cb.gemm_plan(40, 24, 7168, False, arch, sm_count)
            assert plan.unfit_key is None and not cb._tma_boxes_fit(plan)


# ---------------------------------------------------------------------------
# GPU correctness
# ---------------------------------------------------------------------------


def _require_program():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    device = torch.device("cuda", 0)
    if not cb.generated_program_available(device):
        pytest.skip("no generated per-token NVFP4 program registered for this GPU")
    return device


def _dequantize(fp4, sf):
    """FP32 dequantisation of packed E2M1 + swizzled 128x4 E4M3 scales (rows x K)."""
    fp4 = fp4.view(torch.uint8)
    rows, kh = fp4.shape
    k = 2 * kh
    e2m1 = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=fp4.device,
    )
    nib = torch.stack([fp4 & 0xF, fp4 >> 4], dim=-1).reshape(rows, k).long()
    sf_logical = cb.unswizzle_sf_128x4(sf.view(torch.uint8), rows, k)
    scales = sf_logical.view(torch.float8_e4m3fn).float()
    return (e2m1[nib].view(rows, k // 16, 16) * scales[:, :, None]).view(rows, k)


def _operands(m, n, k, x_dtype, device, seed, fold=None):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(m, k, device=device, dtype=x_dtype, generator=g)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16, generator=g)
    w_global_sf = (448.0 * 6.0) / w.float().abs().max()
    w_fp4, w_sf = nvfp4_quantize(
        w, w_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    gs_inv = torch.tensor([GLOBAL_SCALE_INV], dtype=torch.float32, device=device)
    w_scale = (1.0 / w_global_sf).reshape(1).float()
    if fold is None:
        fold = x_dtype == torch.bfloat16
    if not fold:
        # The validated fp16-activation rows (and the off-matrix K rows, whose
        # folded-scale quantizer programs are a recorded follow-up) quantize
        # without the folded output scale; fold the weight scale into alpha on
        # the host instead.
        a_fp4, a_sf, alpha = nvfp4_quantize(
            x, gs_inv, per_token_activation=True, backend="cake"
        )
        alpha = alpha * w_scale
    else:
        a_fp4, a_sf, alpha = nvfp4_quantize(
            x, gs_inv, per_token_activation=True, backend="cake", out_scale=w_scale
        )
    return x, w, a_fp4, a_sf, alpha, w_fp4, w_sf, gs_inv, w_scale


def _assert_matches(out, ref):
    got = out.float()
    err = (got - ref).abs()
    ulp = ref.abs() * (2.0**-8 if out.dtype == torch.bfloat16 else 2.0**-11)
    bound = 1e-2 + 1e-2 * ref.abs() + ulp
    assert torch.isfinite(got).all()
    bad = int((err > bound).sum())
    assert bad == 0, f"{bad} elements exceed |ref| * 1e-2 + 1e-2 + half ulp"


# Every M below 33 runs the swapped orientation (alpha along the MMA N extent);
# larger M the m orientation (alpha per accumulator row); 17 / 130 / 257 are tails.
# The 16-token tiles of M in 9..16 are registered for bf16 output (the validated
# coverage rows); their fp16-output variants are a recorded follow-up (see below).
@pytest.mark.parametrize(
    "m,out_dtype",
    [
        (m, dt)
        for m in (1, 8, 17, 32, 130, 257, 512)
        for dt in (torch.bfloat16, torch.float16)
    ]
    + [(m, torch.bfloat16) for m in (9, 12, 16)],
)
@pytest.mark.parametrize("n,k", [(2112, 7168)])
def test_mm_fp4_cake_matches_reference(m, n, k, out_dtype):
    device = _require_program()
    x, w, a_fp4, a_sf, alpha, w_fp4, w_sf, _, _ = _operands(
        m, n, k, torch.bfloat16, device, seed=100 + m
    )
    out = mm_fp4(
        a_fp4,
        w_fp4.T,
        a_sf,
        w_sf.T,
        alpha,
        out_dtype,
        None,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="cake",
        use_nvfp4=True,
    )
    torch.cuda.synchronize()
    assert out.shape == (m, n) and out.dtype == out_dtype
    ref = (_dequantize(a_fp4, a_sf) @ _dequantize(w_fp4, w_sf).T) * alpha[:, None]
    _assert_matches(out, ref)
    dense = x.float() @ w.float().T
    cos = torch.nn.functional.cosine_similarity(
        out.float().reshape(-1), dense.reshape(-1), dim=0
    )
    assert cos > 0.98


@pytest.mark.parametrize(
    "m,n,k,x_dtype",
    [
        (8, 18432, 7168, torch.bfloat16),  # deep-K swapped orientation
        (17, 7168, 16384, torch.bfloat16),  # split-K tail
        (257, 2112, 7168, torch.float16),  # fp16 activations (validated f16 row)
        (130, 8192, 8192, torch.bfloat16),  # 2-CTA m orientation
        (1000, 2112, 7168, torch.bfloat16),  # ragged M, grouped raster
        (9, 8192, 8192, torch.bfloat16),  # 16-token swapped tile, wide N
        (16, 18432, 7168, torch.bfloat16),  # 16-token tile, deep K
        (16, 3200, 4096, torch.bfloat16),  # (K, N) outside the measured families
        (300, 3200, 4096, torch.bfloat16),  # off-matrix, m orientation
    ],
)
def test_mm_fp4_cake_routes(m, n, k, x_dtype):
    device = _require_program()
    _, _, a_fp4, a_sf, alpha, w_fp4, w_sf, _, _ = _operands(
        m, n, k, x_dtype, device, seed=200 + m
    )
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
    result = mm_fp4(
        a_fp4, w_fp4.T, a_sf, w_sf.T, alpha, torch.bfloat16, out, backend="cake"
    )
    torch.cuda.synchronize()
    assert result is out
    ref = (_dequantize(a_fp4, a_sf) @ _dequantize(w_fp4, w_sf).T) * alpha[:, None]
    _assert_matches(out, ref)


def test_prepared_chain_runner_graph_replay_and_no_allocation():
    device = _require_program()
    m, n, k = 257, 2112, 7168
    x, _, _, _, _, w_fp4, w_sf, gs_inv, w_scale = _operands(
        m, n, k, torch.bfloat16, device, seed=5
    )
    ws = cb.allocate_nvfp4_per_token_quantize_outputs(m, k, device)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    runner = cb.prepare_nvfp4_per_token_chain(
        x, gs_inv, w_fp4.view(torch.uint8), w_sf, out, ws, out_scale=w_scale
    )
    assert runner.launch_count == 2
    assert runner.kernels[0].startswith("quant:")
    assert runner.kernels[1].startswith("gemm:")
    runner()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    runner()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] - before["allocation.all.allocated"] == 0
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            runner()
    torch.cuda.synchronize()
    g = torch.Generator(device=device).manual_seed(77)
    for _ in range(2):
        x.copy_(torch.randn(m, k, device=device, dtype=torch.bfloat16, generator=g))
        out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        replayed = out.clone()
        out.fill_(float("nan"))
        runner()
        torch.cuda.synchronize()
        assert torch.equal(replayed, out)
        ref = _dequantize(ws.fp4, ws.sf) @ _dequantize(w_fp4, w_sf).T
        ref = ref * ws.scale[:, None]
        _assert_matches(out, ref)


# Rows whose contiguous chain quantizer splits over a cluster on some architecture.
@pytest.mark.parametrize(
    "m,n,k",
    [(8, 7168, 16384), (32, 7168, 18432), (2, 8192, 28672), (17, 8192, 8192)],
)
@pytest.mark.parametrize("fold", [False, True])
def test_chain_reads_row_strided_activations_in_place(m, n, k, fold):
    """The chain quantizes a row-strided view with one CTA per padded row and writes the
    same quantized activation (padding included) and GEMM output as for the contiguous
    copy, bit for bit."""
    device = _require_program()
    x, _, _, _, _, w_fp4, w_sf, gs_inv, w_scale = _operands(
        m, n, k, torch.bfloat16, device, seed=900 + m
    )
    full = torch.zeros((m, k + 64), dtype=torch.bfloat16, device=device)
    full[:, :k] = x
    strided = full[:, :k]
    assert not strided.is_contiguous()
    out_scale = w_scale if fold else None
    outs, plans = [], []
    for activation in (strided, x):
        ws = cb.allocate_nvfp4_per_token_quantize_outputs(m, k, device)
        for t in ws:
            t.view(torch.uint8).fill_(0xFF)
        out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
        runner = cb.prepare_nvfp4_per_token_chain(
            activation,
            gs_inv,
            w_fp4.view(torch.uint8),
            w_sf,
            out,
            ws,
            out_scale=out_scale,
        )
        runner()
        outs.append((*ws, out))
        plans.append(runner.quantize.plan)
    torch.cuda.synchronize()
    for a, b in zip(*outs, strict=True):
        assert torch.equal(
            a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)
        )
    fp4, sf, scale, out = outs[0]
    assert torch.isfinite(out).all()
    ref = _dequantize(fp4, sf) @ _dequantize(w_fp4, w_sf).T
    _assert_matches(out, ref * scale[:, None])
    assert plans[0].cluster_split == 1


@pytest.mark.parametrize("m,n,k", [(8, 7168, 16384), (8, 2112, 7168)])
@pytest.mark.parametrize("kind", ["row_stride", "base", "contiguous_base"])
def test_chain_rejects_rows_off_32_byte_boundaries(m, n, k, kind):
    """The chain reads ``x`` in place at every launch, so it refuses a view whose
    rows do not all start on 32-byte boundaries."""
    device = _require_program()
    x, _, _, _, _, w_fp4, w_sf, gs_inv, w_scale = _operands(
        m, n, k, torch.bfloat16, device, seed=950 + m
    )
    if kind == "row_stride":  # row stride K + 8: every other row starts 16 bytes off
        view = torch.zeros((m, k + 8), dtype=torch.bfloat16, device=device)[:, :k]
    elif kind == "base":  # row stride K + 16, every row starts 16 bytes off
        view = torch.zeros((m, k + 16), dtype=torch.bfloat16, device=device)[:, 8:-8]
    else:  # contiguous, 16 bytes off
        view = torch.zeros(m * k + 8, dtype=torch.bfloat16, device=device)[8:]
        view = view.view(m, k)
    view.copy_(x)
    ws = cb.allocate_nvfp4_per_token_quantize_outputs(m, k, device)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="32-byte boundaries"):
        cb.prepare_nvfp4_per_token_chain(
            view, gs_inv, w_fp4.view(torch.uint8), w_sf, out, ws, out_scale=w_scale
        )


@pytest.mark.parametrize(
    "m,n,k,out_dtype",
    [
        (1, 1536, 7168, torch.bfloat16),  # 8-token tile, split-K 4
        (8, 8192, 8192, torch.float16),  # 8-token tile
        (17, 7168, 16384, torch.bfloat16),  # 32-token tile, split-K 2
        (32, 8192, 8192, torch.bfloat16),  # 32-token tile
        (24, 28672, 8192, torch.bfloat16),  # 32-token tile, two CTAs per SM on 152 SMs
        (32, 28672, 8192, torch.float16),  # 32-token tile, two CTAs per SM on 152 SMs
        (128, 18432, 7168, torch.float16),  # one 128-token tile, 128-wide weight tiles
        (130, 18432, 7168, torch.bfloat16),  # one 128-token tile + the remainder fold
        (136, 18432, 7168, torch.float16),  # one 128-token tile + the remainder fold
    ],
)
def test_chain_entry_prefetch_matches_gemm_alone(m, n, k, out_dtype):
    # On a CHAIN_EP_RULES cell the chain's GEMM is the entry L2 prefetch program, bitwise
    # equal to the program the GEMM alone runs on the same quantized activation.
    device = _require_program()
    x, _, _, _, _, w_fp4, w_sf, gs_inv, w_scale = _operands(
        m, n, k, torch.bfloat16, device, seed=300 + m
    )
    ws = cb.allocate_nvfp4_per_token_quantize_outputs(m, k, device)
    out = torch.full((m, n), float("nan"), dtype=out_dtype, device=device)
    runner = cb.prepare_nvfp4_per_token_chain(
        x, gs_inv, w_fp4.view(torch.uint8), w_sf, out, ws, out_scale=w_scale
    )
    plan = cb.gemm_plan(
        m,
        n,
        k,
        out_dtype == torch.float16,
        cb._device_arch(device),
        cb._sm_count(device),
    )
    if cb.chain_gemm_tactic(plan) is None:
        pytest.skip(f"{plan.kernel_key} has no chain cell on this part")
    assert runner.kernels[1] == _ep4_key(plan.kernel_key)
    runner()
    torch.cuda.synchronize()
    alone = torch.full_like(out, float("nan"))
    cb.prepare_mm_fp4_per_token(
        ws.fp4, ws.sf, w_fp4.view(torch.uint8), w_sf, ws.scale, alone
    )()
    torch.cuda.synchronize()
    assert torch.equal(out, alone)
    ref = _dequantize(ws.fp4, ws.sf) @ _dequantize(w_fp4, w_sf).T
    _assert_matches(out, ref * ws.scale[:, None])


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n,k", [(8192, 8192), (7168, 16384)])
def test_gemm_alone_entry_prefetch_matches_plain_program(n, k, out_dtype):
    # On a GEMM_ALONE_EP_RULES cell mm_fp4 and the prepared GEMM run the entry L2 prefetch
    # program, bitwise equal to the plain program of the same plan.
    device = _require_program()
    m = 128
    _, _, a_fp4, a_sf, alpha, w_fp4, w_sf, _, _ = _operands(
        m, n, k, torch.bfloat16, device, seed=400 + n
    )
    plan = cb.gemm_plan(
        m,
        n,
        k,
        out_dtype == torch.float16,
        cb._device_arch(device),
        cb._sm_count(device),
    )
    if cb.gemm_alone_tactic(plan) is None:
        pytest.skip(f"{plan.kernel_key} has no GEMM-alone prefetch cell on this part")
    a_u8, w_u8 = a_fp4.view(torch.uint8), w_fp4.view(torch.uint8)
    out = torch.full((m, n), float("nan"), dtype=out_dtype, device=device)
    runner = cb.prepare_mm_fp4_per_token(a_u8, a_sf, w_u8, w_sf, alpha, out)
    assert runner.plan.kernel_key == plan.kernel_key + "_ep4"
    runner()
    plain = torch.full_like(out, float("nan"))
    cb.prepare_mm_fp4_per_token(
        a_u8, a_sf, w_u8, w_sf, alpha, plain, tactic=plan.tactic
    )()
    public = mm_fp4(
        a_fp4,
        w_fp4.T,
        a_sf,
        w_sf.T,
        alpha,
        out_dtype,
        None,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="cake",
        use_nvfp4=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(out, plain) and torch.equal(public, plain)
    ref = (_dequantize(a_fp4, a_sf) @ _dequantize(w_fp4, w_sf).T) * alpha[:, None]
    _assert_matches(out, ref)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [129, 130, 136, 257, 264, 272])
@pytest.mark.parametrize("n,k", [(18432, 7168), (28672, 8192)])
def test_token_remainder_fold_matches_route(m, n, k, out_dtype):
    # On a token-remainder fold cell (a FOLD_CELLS / B2_FOLD_CELLS (K, N) at the
    # measured SM count) mm_fp4 and the prepared GEMM run the fold program, bitwise
    # equal to the program of the plan without the fold (the 2-CTA pair program over
    # the plain token tiles).
    device = _require_program()
    out_f16 = out_dtype == torch.float16
    _, _, a_fp4, a_sf, alpha, w_fp4, w_sf, _, _ = _operands(
        m, n, k, torch.bfloat16, device, seed=500 + m
    )
    arch, sm_count = cb._device_arch(device), cb._sm_count(device)
    plan = cb.gemm_plan(m, n, k, out_f16, arch, sm_count)
    if not plan.kernel_key.endswith("fold"):
        pytest.skip(f"{plan.kernel_key}: no token-remainder fold cell on this part")
    route = cb._gemm_plan(m, n, k, out_f16, arch, sm_count)
    assert "fold" not in route.kernel_key and route.kernel_key in KERNELS[arch]
    a_u8, w_u8 = a_fp4.view(torch.uint8), w_fp4.view(torch.uint8)
    out = torch.full((m, n), float("nan"), dtype=out_dtype, device=device)
    runner = cb.prepare_mm_fp4_per_token(a_u8, a_sf, w_u8, w_sf, alpha, out)
    assert runner.plan.kernel_key == plan.kernel_key
    runner()
    plain = torch.full_like(out, float("nan"))
    cb.prepare_mm_fp4_per_token(
        a_u8, a_sf, w_u8, w_sf, alpha, plain, tactic=route.tactic
    )()
    public = mm_fp4(
        a_fp4,
        w_fp4.T,
        a_sf,
        w_sf.T,
        alpha,
        out_dtype,
        None,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="cake",
        use_nvfp4=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(out, plain) and torch.equal(public, plain)
    ref = (_dequantize(a_fp4, a_sf) @ _dequantize(w_fp4, w_sf).T) * alpha[:, None]
    _assert_matches(out, ref)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "m,n,k",
    [
        (6913, 7168, 16384),  # CLC program, the first row it serves
        (8192, 8192, 8192),  # CLC program, a multicast row of SF_RING_MC42_CELLS
        (1537, 7168, 18432),  # static program, the first row of its cell
        (2304, 8192, 28672),  # static program, the last row of its cell
        (7937, 1536, 7168),  # static program, a cell of the GEMM alone only
    ],
)
def test_sf_ring_matches_route(m, n, k, out_dtype):
    # Where the scale-factor ring rules take it, mm_fp4, the prepared GEMM and the
    # chain's GEMM run the ring program (``_sfr``), bitwise equal to the program of the
    # plan without the ring; the chain keeps the plan's program on a cell of the GEMM
    # alone only. On a row of SF_RING_MC42_CELLS they run the ring program's
    # cluster-of-8 multicast form (``_clc_sfr_mc42``), also bitwise equal to it.
    device = _require_program()
    out_f16 = out_dtype == torch.float16
    plan = cb.gemm_plan(m, n, k, out_f16, cb._device_arch(device), cb._sm_count(device))
    alone = cb.gemm_alone_tactic(plan)
    if alone is None or not alone.get("sf_ring"):
        pytest.skip(f"{plan.kernel_key}: no scale-factor ring cell on this part")
    x, _, a_fp4, a_sf, alpha, w_fp4, w_sf, gs_inv, w_scale = _operands(
        m, n, k, torch.bfloat16, device, seed=600 + m
    )
    a_u8, w_u8 = a_fp4.view(torch.uint8), w_fp4.view(torch.uint8)
    out = torch.full((m, n), float("nan"), dtype=out_dtype, device=device)
    runner = cb.prepare_mm_fp4_per_token(a_u8, a_sf, w_u8, w_sf, alpha, out)
    if alone.get("mc42"):
        assert runner.plan.kernel_key == cb.gemm_kernel_key(alone, out_f16)
        assert runner.plan.kernel_key.endswith("_clc_sfr_mc42")
        assert runner.plan.grid == 2 * plan.num_tiles
    else:
        assert runner.plan.kernel_key == plan.kernel_key + "_sfr"
        assert runner.plan.grid == plan.grid
    runner()
    plain = torch.full_like(out, float("nan"))
    cb.prepare_mm_fp4_per_token(
        a_u8, a_sf, w_u8, w_sf, alpha, plain, tactic=plan.tactic
    )()
    public = mm_fp4(
        a_fp4,
        w_fp4.T,
        a_sf,
        w_sf.T,
        alpha,
        out_dtype,
        None,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="cake",
        use_nvfp4=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(out, plain) and torch.equal(public, plain)
    ref = (_dequantize(a_fp4, a_sf) @ _dequantize(w_fp4, w_sf).T) * alpha[:, None]
    _assert_matches(out, ref)
    ws = cb.allocate_nvfp4_per_token_quantize_outputs(m, k, device)
    chained = torch.full_like(out, float("nan"))
    chain = cb.prepare_nvfp4_per_token_chain(
        x, gs_inv, w_u8, w_sf, chained, ws, out_scale=w_scale
    )
    tactic = cb.chain_gemm_tactic(plan)
    if tactic.get("mc42"):
        expected = cb.gemm_kernel_key(tactic, out_f16)
        assert expected.endswith("_clc_sfr_mc42")
    else:
        expected = plan.kernel_key + ("_sfr" if tactic.get("sf_ring") else "")
    assert chain.kernels[1] == expected
    chain()
    chain_plain = torch.full_like(out, float("nan"))
    cb.prepare_mm_fp4_per_token(
        ws.fp4, ws.sf, w_u8, w_sf, ws.scale, chain_plain, tactic=plan.tactic
    )()
    torch.cuda.synchronize()
    assert torch.equal(chained, chain_plain)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "m,n,k",
    [
        (513, 18432, 7168),  # g16_clc_sfr, first row (GEMM alone and chain)
        (2560, 18432, 7168),  # g16_clc_sfr, last row (GEMM alone and chain)
        (1537, 28672, 8192),  # g16_clc_sfr, first row (GEMM alone and chain)
        (2560, 28672, 8192),  # g16_clc_sfr, last row (GEMM alone and chain)
        (5889, 8192, 28672),  # g8_sfr, first row (GEMM alone and chain)
        (8192, 8192, 28672),  # g8_sfr, last row (GEMM alone and chain)
        (2561, 28672, 8192),  # clc_sfr, first row (GEMM alone and chain)
        (8192, 28672, 8192),  # clc_sfr, last row (GEMM alone and chain)
    ],
)
def test_sf_ring_form_matches_route(m, n, k, out_dtype):
    # Where SF_RING_FORM_CELLS take a ring program, mm_fp4, the prepared GEMM and the
    # chain's GEMM run it, bitwise equal to the plan's program without the ring.
    device = _require_program()
    out_f16 = out_dtype == torch.float16
    plan = cb.gemm_plan(m, n, k, out_f16, cb._device_arch(device), cb._sm_count(device))
    alone = cb.gemm_alone_tactic(plan)
    chain_tactic = cb.chain_gemm_tactic(plan)
    if not any(t is not None and t.get("sf_ring") for t in (alone, chain_tactic)):
        pytest.skip(f"{plan.kernel_key}: no scale-factor ring cell on this part")
    x, _, a_fp4, a_sf, alpha, w_fp4, w_sf, gs_inv, w_scale = _operands(
        m, n, k, torch.bfloat16, device, seed=700 + m
    )
    a_u8, w_u8 = a_fp4.view(torch.uint8), w_fp4.view(torch.uint8)
    out = torch.full((m, n), float("nan"), dtype=out_dtype, device=device)
    runner = cb.prepare_mm_fp4_per_token(a_u8, a_sf, w_u8, w_sf, alpha, out)
    alone_key = plan.kernel_key if alone is None else cb.gemm_kernel_key(alone, out_f16)
    assert runner.plan.kernel_key == alone_key
    assert runner.plan.num_tiles == plan.num_tiles
    runner()
    plain = torch.full_like(out, float("nan"))
    cb.prepare_mm_fp4_per_token(
        a_u8, a_sf, w_u8, w_sf, alpha, plain, tactic=plan.tactic
    )()
    public = mm_fp4(
        a_fp4,
        w_fp4.T,
        a_sf,
        w_sf.T,
        alpha,
        out_dtype,
        None,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="cake",
        use_nvfp4=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(out, plain) and torch.equal(public, plain)
    ref = (_dequantize(a_fp4, a_sf) @ _dequantize(w_fp4, w_sf).T) * alpha[:, None]
    _assert_matches(out, ref)
    ws = cb.allocate_nvfp4_per_token_quantize_outputs(m, k, device)
    chained = torch.full_like(out, float("nan"))
    chain = cb.prepare_nvfp4_per_token_chain(
        x, gs_inv, w_u8, w_sf, chained, ws, out_scale=w_scale
    )
    chain_key = (
        alone_key if chain_tactic is None else cb.gemm_kernel_key(chain_tactic, out_f16)
    )
    assert chain.kernels[1] == chain_key
    chain()
    chain_plain = torch.full_like(out, float("nan"))
    cb.prepare_mm_fp4_per_token(
        ws.fp4, ws.sf, w_u8, w_sf, ws.scale, chain_plain, tactic=plan.tactic
    )()
    torch.cuda.synchronize()
    assert torch.equal(chained, chain_plain)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [1793, 2048])
@pytest.mark.parametrize("n,k", [(8192, 8192), (18432, 7168), (28672, 8192)])
def test_sf_ring_mc42_matches_route(m, n, k, out_dtype):
    # Where SF_RING_MC42_CELLS take the cluster-of-8 multicast form (the first row of
    # each cell, whose last token tile holds one row, and the last row), mm_fp4, the
    # prepared GEMM and the chain's GEMM run it, bitwise equal to the same tactic
    # without the form (the CLC ring program without raster groups).
    device = _require_program()
    out_f16 = out_dtype == torch.float16
    plan = cb.gemm_plan(m, n, k, out_f16, cb._device_arch(device), cb._sm_count(device))
    alone = cb.gemm_alone_tactic(plan)
    chain_tactic = cb.chain_gemm_tactic(plan)
    if alone is None or not alone.get("mc42"):
        pytest.skip(f"{plan.kernel_key}: no multicast ring cell on this part")
    assert chain_tactic is not None and chain_tactic.get("mc42")
    x, _, a_fp4, a_sf, alpha, w_fp4, w_sf, gs_inv, w_scale = _operands(
        m, n, k, torch.bfloat16, device, seed=800 + m
    )
    a_u8, w_u8 = a_fp4.view(torch.uint8), w_fp4.view(torch.uint8)
    out = torch.full((m, n), float("nan"), dtype=out_dtype, device=device)
    runner = cb.prepare_mm_fp4_per_token(a_u8, a_sf, w_u8, w_sf, alpha, out)
    assert runner.plan.kernel_key == cb.gemm_kernel_key(alone, out_f16)
    assert runner.plan.kernel_key.endswith("_clc_sfr_mc42")
    assert runner.plan.grid == 2 * plan.num_tiles
    runner()
    pair = {name: value for name, value in alone.items() if name != "mc42"}
    plain = torch.full_like(out, float("nan"))
    cb.prepare_mm_fp4_per_token(a_u8, a_sf, w_u8, w_sf, alpha, plain, tactic=pair)()
    public = mm_fp4(
        a_fp4,
        w_fp4.T,
        a_sf,
        w_sf.T,
        alpha,
        out_dtype,
        None,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="cake",
        use_nvfp4=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(out, plain) and torch.equal(public, plain)
    ref = (_dequantize(a_fp4, a_sf) @ _dequantize(w_fp4, w_sf).T) * alpha[:, None]
    _assert_matches(out, ref)
    ws = cb.allocate_nvfp4_per_token_quantize_outputs(m, k, device)
    chained = torch.full_like(out, float("nan"))
    chain = cb.prepare_nvfp4_per_token_chain(
        x, gs_inv, w_u8, w_sf, chained, ws, out_scale=w_scale
    )
    assert chain.kernels[1] == cb.gemm_kernel_key(chain_tactic, out_f16)
    chain()
    chain_pair = {name: value for name, value in chain_tactic.items() if name != "mc42"}
    chain_plain = torch.full_like(out, float("nan"))
    cb.prepare_mm_fp4_per_token(
        ws.fp4, ws.sf, w_u8, w_sf, ws.scale, chain_plain, tactic=chain_pair
    )()
    torch.cuda.synchronize()
    assert torch.equal(chained, chain_plain)


@pytest.mark.parametrize("m", [9, 12, 16])
def test_mm_fp4_cake_fp16_output_small_m_runs_a_registered_program(m):
    # The fp16-output 16-token swapped-orientation programs of the default tactic are
    # not generated: mm_fp4 runs the nearest registered program (the 16-token fp16
    # program with the cluster split-K of three) and matches the reference.
    device = _require_program()
    n, k = 2112, 7168
    _, _, a_fp4, a_sf, alpha, w_fp4, w_sf, _, _ = _operands(
        m, n, k, torch.bfloat16, device, seed=300 + m
    )
    out = mm_fp4(a_fp4, w_fp4.T, a_sf, w_sf.T, alpha, torch.float16, backend="cake")
    torch.cuda.synchronize()
    ref = (_dequantize(a_fp4, a_sf) @ _dequantize(w_fp4, w_sf).T) * alpha[:, None]
    _assert_matches(out, ref)


@pytest.mark.parametrize("chain", [False, True])
@pytest.mark.parametrize(
    "m,n,k,out_dtype",
    [
        (12, 2112, 7168, torch.float16),  # 16-token tile, fp16 output
        (600, 1536, 7168, torch.float16),  # 2-CTA 128-wide tile, fp16 output
        (2049, 8192, 8192, torch.bfloat16),  # 16-wide raster group, static schedule
        (769, 28672, 8192, torch.float16),  # stream-K tail of 8 slices
        (9729, 1536, 7168, torch.bfloat16),  # stream-K tail with the 8-wide raster
    ],
)
def test_off_matrix_rows_run_a_registered_program(m, n, k, out_dtype, chain):
    # Rows whose own tactic selects an unregistered kernel run the nearest registered
    # program (alone and in the quantize + GEMM chain) and match the reference.
    device = _require_program()
    x, _, a_fp4, a_sf, alpha, w_fp4, w_sf, gs_inv, w_scale = _operands(
        m, n, k, torch.bfloat16, device, seed=300 + m
    )
    w_u8 = w_fp4.view(torch.uint8)
    out = torch.full((m, n), float("nan"), dtype=out_dtype, device=device)
    if chain:
        ws = cb.allocate_nvfp4_per_token_quantize_outputs(m, k, device)
        runner = cb.prepare_nvfp4_per_token_chain(
            x, gs_inv, w_u8, w_sf, out, ws, out_scale=w_scale
        )
        a_fp4, a_sf, alpha = ws.fp4, ws.sf, ws.scale
        key = runner.kernels[1]
    else:
        a_u8 = a_fp4.view(torch.uint8)
        runner = cb.prepare_mm_fp4_per_token(a_u8, a_sf, w_u8, w_sf, alpha, out)
        key = runner.plan.kernel_key
    assert key in KERNELS[cb._device_arch(device)]
    runner()
    torch.cuda.synchronize()
    ref = (_dequantize(a_fp4, a_sf) @ _dequantize(w_fp4, w_sf).T) * alpha[:, None]
    _assert_matches(out, ref)
    if not chain:
        public = mm_fp4(
            a_fp4,
            w_fp4.T,
            a_sf,
            w_sf.T,
            alpha,
            out_dtype,
            None,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="cake",
            use_nvfp4=True,
        )
        torch.cuda.synchronize()
        assert torch.equal(public, out)


@pytest.mark.parametrize("chain", [False, True])
@pytest.mark.parametrize(
    "m,n,out_dtype",
    [
        (33, 2112, torch.bfloat16),  # validated family, half-M pair box rows > M
        (40, 1536, torch.float16),
        (63, 1536, torch.bfloat16),
        (48, 96, torch.float16),
        (1, 32, torch.bfloat16),  # swapped orientation, 64-column output box > N
        (16, 40, torch.float16),
        (32, 56, torch.bfloat16),
    ],
)
def test_rows_whose_tma_boxes_do_not_fit_match_the_reference(m, n, out_dtype, chain):
    # These rows name a registered program whose host binding refuses the shape; they
    # run the 1-CTA 64-wide program (alone and in the quantize + GEMM chain).
    device = _require_program()
    k = 7168
    x, _, a_fp4, a_sf, alpha, w_fp4, w_sf, gs_inv, w_scale = _operands(
        m, n, k, torch.bfloat16, device, seed=500 + m
    )
    w_u8 = w_fp4.view(torch.uint8)
    out = torch.full((m, n), float("nan"), dtype=out_dtype, device=device)
    if chain:
        ws = cb.allocate_nvfp4_per_token_quantize_outputs(m, k, device)
        runner = cb.prepare_nvfp4_per_token_chain(
            x, gs_inv, w_u8, w_sf, out, ws, out_scale=w_scale
        )
        a_fp4, a_sf, alpha = ws.fp4, ws.sf, ws.scale
        key = runner.kernels[1]
    else:
        a_u8 = a_fp4.view(torch.uint8)
        runner = cb.prepare_mm_fp4_per_token(a_u8, a_sf, w_u8, w_sf, alpha, out)
        key = runner.plan.kernel_key
        assert runner.plan.unfit_key or runner.plan.unregistered_key
    assert key.startswith("gemm:m64_") and "2cta" not in key, key
    runner()
    torch.cuda.synchronize()
    ref = (_dequantize(a_fp4, a_sf) @ _dequantize(w_fp4, w_sf).T) * alpha[:, None]
    _assert_matches(out, ref)
    if not chain:
        public = mm_fp4(
            a_fp4,
            w_fp4.T,
            a_sf,
            w_sf.T,
            alpha,
            out_dtype,
            None,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="cake",
            use_nvfp4=True,
        )
        torch.cuda.synchronize()
        assert torch.equal(public, out)


def test_quantize_fold_off_matrix_k_runs():
    # The folded-output-scale quantizer of the one-block-per-thread 256-thread CTA
    # shape (K = 4096 at M > 1) is generated: the scale fold matches the reference.
    device = _require_program()
    x = torch.randn(16, 4096, device=device, dtype=torch.bfloat16)
    gs_inv = torch.tensor([GLOBAL_SCALE_INV], dtype=torch.float32, device=device)
    out_scale = torch.tensor([0.5], dtype=torch.float32, device=device)
    fp4, sf, scale = nvfp4_quantize(
        x, gs_inv, per_token_activation=True, backend="cake", out_scale=out_scale
    )
    _, _, plain_scale = nvfp4_quantize(
        x, gs_inv, per_token_activation=True, backend="cake"
    )
    torch.testing.assert_close(scale, plain_scale * 0.5, rtol=0, atol=0)


def test_mm_fp4_cake_repeated_call_allocates_only_its_output():
    device = _require_program()
    m, n, k = 130, 2112, 7168
    _, _, a_fp4, a_sf, alpha, w_fp4, w_sf, _, _ = _operands(
        m, n, k, torch.bfloat16, device, seed=9
    )
    out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    call = lambda: mm_fp4(  # noqa: E731
        a_fp4, w_fp4.T, a_sf, w_sf.T, alpha, torch.bfloat16, out, backend="cake"
    )
    call()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    ref = (_dequantize(a_fp4, a_sf) @ _dequantize(w_fp4, w_sf).T) * alpha[:, None]
    _assert_matches(out, ref)


def test_scalar_alpha_and_bad_layouts_are_rejected():
    device = _require_program()
    m, n, k = 16, 2112, 7168
    _, _, a_fp4, a_sf, alpha, w_fp4, w_sf, _, _ = _operands(
        m, n, k, torch.bfloat16, device, seed=3
    )
    with pytest.raises(ValueError, match="per-token alpha"):
        mm_fp4(a_fp4, w_fp4.T, a_sf, w_sf.T, alpha[:1], torch.bfloat16, backend="cake")
    with pytest.raises(ValueError, match="column-major"):
        mm_fp4(
            a_fp4,
            w_fp4.T.contiguous(),
            a_sf,
            w_sf.T,
            alpha,
            torch.bfloat16,
            backend="cake",
        )
    with pytest.raises(ValueError, match="dependent launch"):
        mm_fp4(
            a_fp4,
            w_fp4.T,
            a_sf,
            w_sf.T,
            alpha,
            torch.bfloat16,
            backend="cake",
            enable_pdl=False,
        )
