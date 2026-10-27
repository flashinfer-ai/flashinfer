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
    # tactic selects an unregistered kernel runs the measured part's kernel, and every
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
                if reference.kernel_key not in registered:
                    continue  # outside the registered matrix on the measured part too
                served += 1
                plan = cb.gemm_plan(m, n, k, out_f16, arch, sm_count)
                key = plan.kernel_key
                assert key in registered, (m, n, k, out_f16, key)
                assert plan.sm_count == sm_count
                if sm_count == measured:
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
                    assert key == reference.kernel_key, (m, n, k, out_f16)
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


@pytest.mark.parametrize(
    "m,n,k,out_dtype",
    [
        (1, 1536, 7168, torch.bfloat16),  # 8-token tile, split-K 4
        (8, 8192, 8192, torch.float16),  # 8-token tile
        (17, 7168, 16384, torch.bfloat16),  # 32-token tile, split-K 2
        (32, 8192, 8192, torch.bfloat16),  # 32-token tile
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
def test_token_remainder_fold_matches_route(m, out_dtype):
    # On a token-remainder fold cell (K 7168, N 18432 at the measured SM count)
    # mm_fp4 and the prepared GEMM run the fold program, bitwise equal to the program
    # of the plan without the fold (the 2-CTA pair program over the plain token tiles).
    device = _require_program()
    n, k = 18432, 7168
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


@pytest.mark.parametrize("m", [9, 12, 16])
def test_mm_fp4_cake_fp16_output_small_m_names_the_missing_program(m):
    # The fp16-output 16-token swapped-orientation programs are not generated yet:
    # the backend raises at preparation and names the kernel key it would need.
    device = _require_program()
    n, k = 2112, 7168
    _, _, a_fp4, a_sf, alpha, w_fp4, w_sf, _, _ = _operands(
        m, n, k, torch.bfloat16, device, seed=300 + m
    )
    with pytest.raises(NotImplementedError, match=r"gemm:n16_.*_f16"):
        mm_fp4(a_fp4, w_fp4.T, a_sf, w_sf.T, alpha, torch.float16, backend="cake")


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
