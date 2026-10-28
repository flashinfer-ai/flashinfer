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

import gc
import math
import re
import threading
from pathlib import Path

import pytest
import torch

from flashinfer.experimental.kimi_k3_latent_moe import cake_backend as cb
from flashinfer.experimental.kimi_k3_latent_moe import cake_jit
from flashinfer.experimental.kimi_k3_latent_moe.cake_jit import KERNELS, MODULES
from flashinfer.experimental.kimi_k3_latent_moe.cake_backend import (
    DECODE_MAX_T,
    HIDDEN,
    LATENT,
    NUM_EXPERTS,
    QUALIFIED_SM_COUNTS,
    RMS_EPS,
    SHARED_INTERMEDIATE,
    SM_COUNT,
    SUPPORTED_COMPUTE_CAPABILITIES,
    SUPPORTED_TP,
    decode_front_plan,
    decode_kernel_key,
    decode_tail_plan,
    front_split_plan,
    i_local_for_tp,
    prefill_front_plan,
    prefill_tail_plan,
    route_kernel_keys,
    split_plan,
)
from flashinfer.kimi_k3_latent_moe import (
    kimi_k3_latent_moe_front,
    kimi_k3_latent_moe_tail,
    prepare_kimi_k3_latent_moe_front,
    prepare_kimi_k3_latent_moe_tail,
)

# Tolerances of the Cake evaluation contract (never looser): BF16 outputs 1e-2, FP32 logits 1e-3.
ATOL = RTOL = 1e-2
LOGITS_ATOL = LOGITS_RTOL = 1e-3
WEIGHT_SEED = 621
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0
# Representative subset of the 60 validated rows: both stages x TP {1, 8} x these token counts.
SMOKE_TOKENS = (
    1,
    8,
    16,
    128,
    256,
    4096,
    8192,
    16384,
)  # decode + prefill routes, incl. the power-capped GEMM rows


# ---------------------------------------------------------------------------
# Host plan (CPU)
# ---------------------------------------------------------------------------


def test_decode_plan_rules():
    # Front: TP1 streams 7 + 28 + 96 = 131 tiles (one CTA per tile), TP8 7 + 28 + 12 = 47 tiles as 94 cluster-pair CTAs.
    tp1 = decode_front_plan(1, i_local_for_tp(1))
    assert (tp1["tiles"], tp1["grid"], tp1["cluster"], tp1["n_pad"]) == (131, 131, 1, 8)
    tp8 = decode_front_plan(1, i_local_for_tp(8))
    assert (tp8["tiles"], tp8["grid"], tp8["cluster"], tp8["n_pad"]) == (47, 94, 2, 8)
    assert decode_front_plan(9, i_local_for_tp(8))["n_pad"] == 16
    assert decode_front_plan(128, i_local_for_tp(1))["n_pad"] == 128
    # Front K depth 2 while its ring keeps >= 4 (one CTA per tile) / 6 (cluster pairs) stages: TP1 T <= 64, TP8 T <= 16.
    for plan in (tp1, tp8):
        assert (
            plan["kdepth"] == 2
            and 1 <= plan["stages"] <= cb.MAX_STAGES
            and not plan["fused"]
        )
    assert decode_front_plan(64, i_local_for_tp(1))["kdepth"] == 2
    assert decode_front_plan(128, i_local_for_tp(1))["kdepth"] == 1
    assert decode_front_plan(16, i_local_for_tp(8))["kdepth"] == 2
    assert decode_front_plan(32, i_local_for_tp(8))["kdepth"] == 1
    assert decode_front_plan(32, i_local_for_tp(8))["stages"] == cb.plan_stages(
        n_pad=32, kdepth=1, cluster=2, split_fin=True
    )
    # Tail: 56 output tiles -> 112 cluster-pair CTAs; the fused norm is always on.
    for tp in SUPPORTED_TP:
        for tokens in (1, 8, 16, 128):
            plan = decode_tail_plan(tokens, i_local_for_tp(tp), tp)
            assert (plan["tiles"], plan["grid"], plan["cluster"]) == (56, 112, 2)
            assert (
                plan["fused"]
                and plan["k1"] == LATENT // tp // 64
                and plan["k2"] == SHARED_INTERMEDIATE // tp // 64
            )
            if plan["rows_smem"]:
                assert plan["smem_b1"]
    # The resident B operand with staged rows serves the TP8 tails up to T16 (5 ring stages left); T32 falls back to
    # the global-y protocol, and so does TP1 (long shared-down window).
    assert decode_tail_plan(1, i_local_for_tp(8), 8)["smem_b1"]
    assert decode_tail_plan(1, i_local_for_tp(8), 8)["rows_smem"]
    assert decode_tail_plan(8, i_local_for_tp(8), 8)["smem_b1"]
    t16 = decode_tail_plan(16, i_local_for_tp(8), 8)
    assert t16["smem_b1"] and t16["rows_smem"] and t16["stages"] == 5
    assert (
        not t16["tmap_prefetch"]
        and decode_tail_plan(8, i_local_for_tp(8), 8)["tmap_prefetch"]
    )
    # Round-7 lever 3b: the same instance defers the cluster rendezvous to its epilogue warps.
    assert (
        t16["wait_warps"]
        and not decode_tail_plan(8, i_local_for_tp(8), 8)["wait_warps"]
    )
    assert not decode_tail_plan(32, i_local_for_tp(8), 8)["wait_warps"]
    assert not decode_front_plan(16, i_local_for_tp(8))["wait_warps"]
    # one ring stage fewer than the smem budget for the small-T tail (T16 keeps its 5)
    assert decode_tail_plan(1, i_local_for_tp(8), 8)["stages"] == 8
    assert decode_tail_plan(8, i_local_for_tp(8), 8)["stages"] == 8
    assert decode_tail_plan(1, i_local_for_tp(1), 1)["stages"] == 10
    assert not decode_tail_plan(32, i_local_for_tp(8), 8)["smem_b1"]
    assert not decode_tail_plan(32, i_local_for_tp(1), 1)["smem_b1"]
    assert not decode_tail_plan(16, i_local_for_tp(1), 1)["rows_smem"]
    # Round-7 landing-zone alias (lever 6a): only the N_PAD 128 cluster instances streaming >= 32 chunk units per CTA
    # (tail TP1 T=128: 76 units, front TP8 T=128: 56) alias the DSMEM landing zone on the drained A ring and gain two
    # ring stages (5 -> 7); the short TP8 tail (9 units) and the single-CTA TP1 front keep their own zone.
    la_tail = decode_tail_plan(128, i_local_for_tp(1), 1)
    assert la_tail["land_alias"] and la_tail["cluster"] == 2 and la_tail["stages"] == 7
    la_front = decode_front_plan(128, i_local_for_tp(8))
    assert (
        la_front["land_alias"] and la_front["cluster"] == 2 and la_front["stages"] == 7
    )
    assert not decode_tail_plan(128, i_local_for_tp(8), 8)["land_alias"]
    assert not decode_tail_plan(64, i_local_for_tp(1), 1)["land_alias"]
    assert not decode_front_plan(128, i_local_for_tp(1))["land_alias"]
    assert cb.land_alias_auto(128, 152) and not cb.land_alias_auto(128, 19)
    assert not cb.land_alias_auto(64, 152)
    # Evict-first weight streams: every front instance and the TP1 tail; the TP8 tail keeps the default L2 policy.
    assert la_front["a_evict_first"]
    assert decode_front_plan(8, i_local_for_tp(1))["a_evict_first"]
    assert decode_tail_plan(64, i_local_for_tp(1), 1)["a_evict_first"]
    assert la_tail["a_evict_first"]
    assert decode_kernel_key(la_tail).endswith("_f_la_ef_p256b_sf_ea")
    assert not decode_tail_plan(64, i_local_for_tp(8), 8)["a_evict_first"]
    # 256-byte L2 promotion of the weight maps: the TP1 tail at every N_PAD and the TP8 front up to N_PAD 64.
    for tokens in (1, 16, 64, 128):
        tail1 = decode_tail_plan(tokens, i_local_for_tp(1), 1)
        assert tail1["a_l2_promotion"] == "l2_256b"
        assert "_ef_p256b" in decode_kernel_key(tail1)
        tail8 = decode_tail_plan(tokens, i_local_for_tp(8), 8)
        assert tail8["a_l2_promotion"] == "none"
        assert "_p256b" not in decode_kernel_key(tail8)
        front1 = decode_front_plan(tokens, i_local_for_tp(1))
        assert front1["a_l2_promotion"] == "none"
        assert decode_kernel_key(front1).endswith("_ef")
    for tokens in (1, 16, 32, 64):
        front8 = decode_front_plan(tokens, i_local_for_tp(8))
        assert front8["a_l2_promotion"] == "l2_256b"
        assert "_ef_p256b" in decode_kernel_key(front8)
    assert la_front["a_l2_promotion"] == "none"
    assert decode_kernel_key(la_front).endswith("_la_ef_sf")
    # A second routed partial disables the staged rows (P == 1 only).
    assert not decode_tail_plan(1, i_local_for_tp(8), 8, num_partials=2)["rows_smem"]
    with pytest.raises(ValueError):
        cb.n_pad_for(DECODE_MAX_T + 1)


def test_decode_split_finalize_rules():
    """Split finalize (``_sf``): the cluster-pair decode instances -- the TP8 front from N_PAD 16 and the fused-norm
    tails of one routed partial from N_PAD 32, on both qualified SM counts -- with the early grid arrival on the TP1
    tail (``_ea``) and the up-unit L2 prefetch on the TP8 tail at N_PAD 32 / 64 (``_upf``); each CTA's landing zone
    holds half the partial."""
    assert cb.SPLIT_FIN_MIN_N_PAD == 16 and cb.TAIL_SPLIT_FIN_MIN_N_PAD == 32
    for sm_count in QUALIFIED_SM_COUNTS:
        for tokens in (1, 8):
            assert not decode_front_plan(tokens, i_local_for_tp(8), sm_count)[
                "split_fin"
            ]
            for tp in SUPPORTED_TP:
                plan = decode_tail_plan(tokens, i_local_for_tp(tp), tp, 1, sm_count)
                assert not plan["split_fin"]
        for tokens in (9, 16, 32, 64, 128):
            front8 = decode_front_plan(tokens, i_local_for_tp(8), sm_count)
            assert front8["split_fin"] and front8["cluster"] == 2
            assert not (front8["early_arrive"] or front8["up_prefetch"])
            assert decode_kernel_key(front8).endswith("_sf")
            front1 = decode_front_plan(tokens, i_local_for_tp(1), sm_count)
            assert front1["cluster"] == 1 and not front1["split_fin"]
            tail1 = decode_tail_plan(tokens, i_local_for_tp(1), 1, 1, sm_count)
            split1 = tail1["n_pad"] > 16
            assert tail1["split_fin"] == tail1["early_arrive"] == split1
            assert not tail1["up_prefetch"]
            assert decode_kernel_key(tail1).endswith(
                "_ef_p256b_sf_ea" if split1 else "_ef_p256b"
            )
            tail8 = decode_tail_plan(tokens, i_local_for_tp(8), 8, 1, sm_count)
            assert tail8["split_fin"] == (tail8["n_pad"] > 16)
            assert not tail8["early_arrive"]
            assert tail8["up_prefetch"] == (tail8["n_pad"] in (32, 64))
            assert decode_kernel_key(tail8).endswith(
                "_sf_upf"
                if tail8["up_prefetch"]
                else "_sf"
                if tail8["split_fin"]
                else "_ww"
            )
            for tp in SUPPORTED_TP:
                for num_partials in (2, 4):
                    many = decode_tail_plan(
                        tokens, i_local_for_tp(tp), tp, num_partials, sm_count
                    )
                    assert not (
                        many["split_fin"] or many["early_arrive"] or many["up_prefetch"]
                    )
                    assert "_sf" not in decode_kernel_key(many)
        # The tails at N_PAD 16 keep the single finisher, the staged-rows TP8 tail its 5 stages.
        t16 = decode_tail_plan(16, i_local_for_tp(8), 8, 1, sm_count)
        assert t16["rows_smem"] and t16["stages"] == 5 and not t16["split_fin"]
        assert decode_kernel_key(t16).endswith("_sb_rs_np_st2_ww")
        t16_tp1 = decode_tail_plan(16, i_local_for_tp(1), 1, 1, sm_count)
        assert decode_kernel_key(t16_tp1).endswith("_ef_p256b")
    # Half the landing zone buys the TP8 tail at N_PAD 128 a sixth ring stage; the other instances keep their depth.
    assert cb.plan_stages(n_pad=128, cluster=2) == 5
    assert cb.plan_stages(n_pad=128, cluster=2, split_fin=True) == 6
    assert decode_tail_plan(128, i_local_for_tp(8), 8)["stages"] == 6
    assert decode_tail_plan(128, i_local_for_tp(8), 8, 2)["stages"] == 5
    for tp, tokens in ((1, 16), (1, 32), (1, 64), (1, 128), (8, 32), (8, 64)):
        assert (
            decode_tail_plan(tokens, i_local_for_tp(tp), tp)["stages"]
            == decode_tail_plan(tokens, i_local_for_tp(tp), tp, 2)["stages"]
        )


def test_split_plan_rules():
    # No remainder wave, short K loops (TP8: 19 iterations) or a small-reuse region keep whole tiles.
    whole = split_plan(148, 152, SM_COUNT, 1)
    assert whole == dict(
        num_items=148, full_items=148, sk_ipc=152, sk_max_seg=1, sk_total=0, sk_tiles=0
    )
    assert split_plan(56, 19, SM_COUNT, 1)["sk_tiles"] == 0
    assert split_plan(60, 152, SM_COUNT, 4)["sk_tiles"] == 0
    # Partial reuse (2-4 pair rows per column): only a remainder wave after a full wave is split, and only
    # when it fills at most 55 % of the clusters (TP1 T=1024: 74 whole + 38 tiles as 74 x 79 iterations).
    rem = split_plan(112, 152, SM_COUNT, 4)
    assert rem["full_items"] == 74 and rem["sk_tiles"] == 38
    assert rem["sk_ipc"] == 79 and rem["num_items"] == 148
    assert split_plan(56, 152, SM_COUNT, 2)["sk_tiles"] == 0
    assert split_plan(140, 152, SM_COUNT, 5)["sk_tiles"] == 0
    # TP1 T=2048 (224 pair tiles on 74 resident clusters): 3 x 74 whole, then the 2 remainder tiles as 8 stream-K
    # ranges of 38 K iterations; the last full wave stays whole.
    sk = split_plan(224, 152, SM_COUNT, 8)
    assert sk == dict(
        num_items=222 + 8,
        full_items=222,
        sk_ipc=38,
        sk_max_seg=4,
        sk_total=2 * 152,
        sk_tiles=2,
    )
    # Minimum stream-K saving: on 152 SMs the TP1 T=4096 / T=8192 regions (448 / 896 pair tiles on 76
    # resident clusters) leave a near-whole remainder wave whose cut saves less than SK_MIN_SAVING_ITERS on paper (a
    # cut there measured 3.6-4.8 % slower), so they stay whole-tile; T=16384 (1792 tiles, saving 40) and the 148-SM
    # rows above keep their cut.
    assert (
        split_plan(448, 152, 152, 8)["sk_tiles"] == 0
        and split_plan(448, 152, 152, 8)["num_items"] == 448
    )
    assert (
        split_plan(896, 152, 152, 8)["sk_tiles"] == 0
        and split_plan(896, 152, 152, 8)["num_items"] == 896
    )
    assert split_plan(1792, 152, 152, 8)["sk_tiles"] > 0
    assert (
        split_plan(448, 152, SM_COUNT, 8)["sk_tiles"] > 0
        and split_plan(896, 152, SM_COUNT, 8)["sk_tiles"] > 0
    )


def test_front_split_plan_rules():
    # Round-8 rule: the trailing wave's tiles become two aligned K halves when the halves fit one wave of the
    # 74 resident clusters and the modelled saving is >= 4 % of the whole-tile cost; every other shape runs whole.
    whole = dict(
        num_items=48, full_items=48, sk_ipc=112, sk_max_seg=1, sk_total=0, sk_tiles=0
    )
    assert (
        front_split_plan(48, SM_COUNT) == whole
    )  # TP8 T=512: 96 halves would need two waves
    assert front_split_plan(24, SM_COUNT) == dict(
        num_items=48,
        full_items=0,
        sk_ipc=56,
        sk_max_seg=2,
        sk_total=24 * 112,
        sk_tiles=24,
    )  # TP8 T=256
    assert front_split_plan(96, SM_COUNT) == dict(
        num_items=118,
        full_items=74,
        sk_ipc=56,
        sk_max_seg=2,
        sk_total=22 * 112,
        sk_tiles=22,
    )  # TP8 T=1024
    assert (
        front_split_plan(384, SM_COUNT)["sk_tiles"] == 14
    )  # TP8 T=4096: 370 whole + 14 x 2
    assert (
        front_split_plan(528, SM_COUNT)["sk_tiles"] == 10
    )  # TP1 T=2048: 518 whole + 10 x 2
    assert (
        front_split_plan(768, SM_COUNT)["sk_tiles"] == 0
    )  # TP8 T=8192: 3.1 % modelled -> whole
    assert (
        front_split_plan(1056, SM_COUNT)["sk_tiles"] == 0
    )  # TP1 T=4096: 2.3 % modelled -> whole
    for tiles in (66, 132, 264, 192, 1536, 2112, 4224):
        assert front_split_plan(tiles, SM_COUNT)["sk_max_seg"] == 1
    plan = prefill_front_plan(1024, i_local_for_tp(8))
    assert (
        plan["cluster_tiles"] == 96
        and plan["grid"] == 118 * 2
        and not plan["evict_first"]
    )
    assert prefill_front_plan(256, i_local_for_tp(1))["evict_first"]
    assert prefill_front_plan(512, i_local_for_tp(8))["evict_first"]
    assert not prefill_front_plan(1024, i_local_for_tp(1))["evict_first"]
    assert prefill_front_plan(256, i_local_for_tp(1))["grid"] == cb.front_grid(
        cb.m_tiles_for(256), i_local_for_tp(1)
    )
    # evict_first weights up to two pair rows; the TP8 front up to three (T <= 768); on 152 SMs the
    # TP8 front up to 64 pair rows (T <= 16384) and the TP1 front up to eight (T <= 2048).
    assert cb.front_evict_first(4, 6144) and not cb.front_evict_first(6, 6144)
    assert cb.front_evict_first(6, 768) and not cb.front_evict_first(8, 768)
    assert cb.front_evict_first(6, 768, 148) and not cb.front_evict_first(8, 768, 148)
    assert cb.front_evict_first(128, 768, 152)
    assert not cb.front_evict_first(130, 768, 152)
    assert cb.front_evict_first(16, 6144, 152)
    assert not cb.front_evict_first(18, 6144, 152)
    assert prefill_front_plan(768, i_local_for_tp(8))["evict_first"]
    assert prefill_front_plan(513, i_local_for_tp(8))["evict_first"]
    assert not prefill_front_plan(769, i_local_for_tp(8))["evict_first"]
    assert not prefill_front_plan(768, i_local_for_tp(1))["evict_first"]
    assert prefill_front_plan(16384, i_local_for_tp(8), 152)["evict_first"]
    assert not prefill_front_plan(16385, i_local_for_tp(8), 152)["evict_first"]
    assert prefill_front_plan(2048, i_local_for_tp(1), 152)["evict_first"]
    assert not prefill_front_plan(2049, i_local_for_tp(1), 152)["evict_first"]
    for sm_count in QUALIFIED_SM_COUNTS:
        assert route_kernel_keys("front", 8, 768, sm_count) == ("front:i768e1",)
    assert route_kernel_keys("front", 1, 768, 148) == ("front:i6144",)
    assert route_kernel_keys("front", 1, 768, 152) == ("front:i6144e1",)
    assert route_kernel_keys("front", 8, 4096, 148) == ("front:i768",)
    assert route_kernel_keys("front", 8, 4096, 152) == ("front:i768e1",)


def test_front_fold_plan_rules():
    """The fold form serves the TP8 rows of 16 row tiles (T = 1793-2048) on the 152-SM parts, sm_100a and sm_103a,
    and on sm_100a 148-SM parts: nine lanes of eight clusters in the carry mode (the trailing-wave plan runs those 192
    pair tiles whole); and the TP8 rows of 8 row tiles (T = 769-1024) on sm_103a 152-SM parts: 19 lanes of four
    clusters keeping the trailing-wave plan's 20 split tiles.  sm_103a 148-SM parts keep the trailing-wave plan on
    every row, every other row keeps it too, and so does a device whose co-resident cluster count does not hold the
    fold grid."""
    il = i_local_for_tp(8)
    carry = dict(
        num_items=72,
        full_items=8,
        sk_ipc=9,
        sk_max_seg=2 * 192,
        sk_total=24 * 112,
        sk_tiles=0,
        fold=True,
        ws_slots=2 * 144,
    )
    assert front_split_plan(192, 152)["sk_tiles"] == 0
    assert front_split_plan(192, 148)["sk_tiles"] == 0
    for arch in SUPPORTED_COMPUTE_CAPABILITIES.values():
        fold_parts = (152, 148) if arch == "sm_100a" else (152,)
        for sm_count in fold_parts:
            assert cb.front_fold_plan(16, il, sm_count, arch) == carry
        if arch != "sm_100a":
            assert cb.front_fold_plan(16, il, 148, arch) is None
        for tokens in (1793, 1800, 2000, 2047, 2048):
            for sm_count in fold_parts:
                plan = prefill_front_plan(tokens, il, sm_count, arch)
                assert plan["m_tiles"] == 16 and plan["grid"] == 144
                assert plan["evict_first"] and {k: plan[k] for k in carry} == carry
                assert route_kernel_keys("front", 8, tokens, sm_count, arch=arch) == (
                    "front:i768e1fold",
                )
            if arch == "sm_100a":
                continue
            # sm_103a 148-SM parts: the trailing-wave plan, 192 whole pair tiles with the default weight policy.
            whole = prefill_front_plan(tokens, il, 148, arch)
            assert not whole["fold"] and whole["ws_slots"] == 0
            assert whole["num_items"] == 192 and whole["sk_tiles"] == 0
            assert route_kernel_keys("front", 8, tokens, 148, arch=arch) == (
                "front:i768",
            )
        for sm_count in QUALIFIED_SM_COUNTS:
            for tokens in (1792, 2049, 4096):
                assert not prefill_front_plan(tokens, il, sm_count, arch)["fold"]
                key = route_kernel_keys("front", 8, tokens, sm_count, arch=arch)[0]
                assert "fold" not in key
            assert cb.front_fold_plan(16, i_local_for_tp(1), sm_count, arch) is None
            for m_tiles in (2, 4, 6, 14, 18, 32):
                assert cb.front_fold_plan(m_tiles, il, sm_count, arch) is None
    split = dict(
        num_items=76,
        full_items=4,
        sk_ipc=19,
        sk_max_seg=2 * 76,
        sk_total=24 * 112,
        sk_tiles=0,
        fold=True,
        ws_slots=2 * 152,
    )
    assert cb.front_fold_plan(8, il, 152, "sm_103a") == split
    # t0 = the trailing-wave plan's 76 whole tiles; its 20 split tiles keep their two K runs.
    assert front_split_plan(96, 152)["full_items"] == 76
    assert front_split_plan(96, 152)["sk_tiles"] == 20
    for sm_count, arch in ((148, "sm_100a"), (148, "sm_103a"), (152, "sm_100a")):
        assert cb.front_fold_plan(8, il, sm_count, arch) is None
    for tokens in (769, 800, 897, 1000, 1024):
        assert route_kernel_keys("front", 8, tokens, 152, arch="sm_103a") == (
            "front:i768e1fold",
        )
        assert route_kernel_keys("front", 8, tokens, 152, arch="sm_100a") == (
            "front:i768e1",
        )
        assert route_kernel_keys("front", 8, tokens, 148, arch="sm_103a") == (
            "front:i768",
        )
    # Occupancy guard: fewer co-resident clusters than the fold grid keep the trailing-wave plan.
    assert prefill_front_plan(2048, il, 152, "sm_100a", 72)["fold"]
    fallback = prefill_front_plan(2048, il, 152, "sm_100a", 71)
    assert not fallback["fold"] and fallback["ws_slots"] == 0
    assert fallback["num_items"] == 192 and fallback["sk_tiles"] == 0
    assert cb.prefill_front_kernel_key(fallback) == "front:i768e1"
    assert prefill_front_plan(2048, il, 148, "sm_100a", 72)["fold"]
    assert not prefill_front_plan(2048, il, 148, "sm_100a", 71)["fold"]
    assert not prefill_front_plan(2048, il, 148, "sm_103a", 74)["fold"]
    fallback = prefill_front_plan(1024, il, 152, "sm_103a", 75)
    assert not fallback["fold"] and fallback["sk_tiles"] == 20
    assert cb.prefill_front_kernel_key(fallback) == "front:i768e1"
    assert cb.front_kernel_key(il, True, fold=True) == "front:i768e1fold"


def test_prefill_tail_plan_and_trigger():
    plan = prefill_tail_plan(256, 8)
    assert (
        plan["m_tiles"] == 2 and plan["cluster_tiles"] == 28 and plan["norm_grid"] == 64
    )
    assert plan["num_k"] == (LATENT // 8 + SHARED_INTERMEDIATE // 8) // 64 == 19
    assert plan["sk_tiles"] == 0 and plan["gemm_grid"] == 56 and plan["early_trigger"]
    # TP1 T=256: 112 GEMM CTAs next to 64 norm CTAs do not fit 148 SMs -> late trigger.
    tp1 = prefill_tail_plan(256, 1)
    assert tp1["gemm_grid"] == (tp1["num_items"]) * 2 and not tp1["early_trigger"]
    assert plan["weights_evict_first"] and tp1["weights_evict_first"]
    assert not prefill_tail_plan(2048, 1)["weights_evict_first"]
    # Round-7 N128 rule (TP8 T=512): 128-wide pair tile, 56 column tiles, 224 CTAs, 9-deep ring, no stream-K region.
    n128 = prefill_tail_plan(512, 8)
    assert (n128["block_n"], n128["n_tiles"], n128["num_stages"]) == (128, 56, 9)
    assert (
        n128["cluster_tiles"] == 112
        and n128["gemm_grid"] == 224
        and n128["sk_tiles"] == 0
    )
    assert (
        not n128["weights_evict_first"]
        and n128["early_trigger"]
        and not n128["fused_norm"]
    )
    assert (
        prefill_tail_plan(256, 8)["block_n"] == 256
        and prefill_tail_plan(256, 8)["n_tiles"] == 28
    )
    # The packed norm instance: TP8, one routed partial, from 2048 tokens.
    assert not n128["packed_norm"] and not plan["packed_norm"]
    assert prefill_tail_plan(2048, 8)["packed_norm"]
    assert not prefill_tail_plan(2048, 8, SM_COUNT, 2)["packed_norm"]
    assert not prefill_tail_plan(2047, 8)["packed_norm"]
    assert prefill_tail_plan(16384, 8)["packed_norm"]
    assert not tp1["packed_norm"] and not prefill_tail_plan(4096, 1)["packed_norm"]
    # Round-10 rule: every 256-wide instance stages the CTA's final item through a TMA store (``final_ts``, a
    # separate kernel instance that binds the ``out`` tensor map beside the pointer); the 128-wide TP8 T=512
    # instance keeps the direct stores.
    assert (
        plan["final_ts"] and tp1["final_ts"] and prefill_tail_plan(2048, 1)["final_ts"]
    )
    assert not n128["final_ts"]
    assert cb.tail_gemm_final_ts(1024, 8) and not cb.tail_gemm_final_ts(512, 8)
    assert not cb.tail_gemm_final_ts(1024, 8, 152)
    # Fused norm: TP1 T = 256 / 512 on the single-wave grid (K2 = 96 blocks), larger TP1 rows on the claimed resident
    # grid; TP8 (K2 = 12) does not fuse.
    assert tp1["fused_norm"] and prefill_tail_plan(512, 1)["fused_norm"]
    assert not tp1["claim_items"] and not prefill_tail_plan(512, 1)["claim_items"]
    assert (
        prefill_tail_plan(1024, 1)["fused_norm"]
        and prefill_tail_plan(1024, 1)["claim_items"]
    )
    assert not plan["fused_norm"] and not plan["claim_items"]
    assert cb.use_fused_norm(256, 112, SM_COUNT, 96) and not cb.use_fused_norm(
        256, 56, SM_COUNT, 12
    )
    assert cb.weights_evict_first(112, SM_COUNT) and not cb.weights_evict_first(
        224, SM_COUNT
    )
    assert cb.norm_early_trigger(300, 64, SM_COUNT) and not cb.norm_early_trigger(
        112, 64, SM_COUNT
    )
    assert cb.front_grid(cb.m_tiles_for(256), i_local_for_tp(1)) == 1 * 66 * 2
    assert cb.front_grid(cb.m_tiles_for(300), i_local_for_tp(8)) == 2 * 24 * 2


# Every token count the public entry points accept, both tensor-parallel degrees and the partial
# counts the decode tail plans on (P >= 2 selects the un-staged instances).
COVERAGE_TOKENS = tuple(range(1, 1025)) + tuple(range(1025, 16385, 97)) + (16384,)
COVERAGE_PARTIALS = (1, 2, 4, 8)
# The few-token multi-partial tail cells served by the global-y instances with the resident-B arithmetic.
SB_NORM_ROUTES = {
    1: (32, {1: 5, 2: 5, 3: 5, 4: 5, 5: 3, 6: 3, 7: 3, 8: 3}),
    8: (8, {}),
}
# The few-token multi-partial tail cells served with the routed partials prefetched into L2: (first, last, sb_norm),
# ``last`` None: from ``first`` up.
PARTIAL_PREFETCH_ROUTES = {
    1: {
        1: ((4, 6, False), (7, None, True)),
        2: ((4, 6, False), (7, None, True)),
        3: ((4, 6, False), (7, None, True)),
        4: ((4, 6, False), (7, None, True)),
        5: ((3, 3, False), (5, None, True)),
        6: ((3, 3, False), (5, None, True)),
        7: ((3, 3, False), (5, None, True)),
        8: ((3, 3, False), (5, None, True)),
    },
    8: {
        1: ((4, None, True),),
        2: ((4, None, True),),
        3: ((4, None, True),),
        4: ((3, None, True),),
    },
}


def _reached_programs(sm_count, arch):
    """The registered programs the routes of the public contract select on an ``arch`` device of ``sm_count``
    SMs."""
    keys = set()
    for stage in ("front", "tail"):
        for tp in SUPPORTED_TP:
            for tokens in COVERAGE_TOKENS:
                keys.update(route_kernel_keys(stage, tp, tokens, sm_count, arch=arch))
    for tp in SUPPORTED_TP:
        for num_partials in COVERAGE_PARTIALS:
            for tokens in (
                *range(1, DECODE_MAX_T + 1),
                129,
                300,
                511,
                512,
                1024,
                2047,
                2048,
                4096,
                16384,
            ):
                keys.update(
                    route_kernel_keys("tail", tp, tokens, sm_count, num_partials)
                )
    return {KERNELS[key] for key in keys if key in KERNELS}


@pytest.mark.parametrize("arch", tuple(SUPPORTED_COMPUTE_CAPABILITIES.values()))
@pytest.mark.parametrize("sm_count", QUALIFIED_SM_COUNTS)
def test_route_keys_are_registered_for_every_token_count(sm_count, arch):
    """Every route of the public contract resolves to a registered program on every qualified SM
    count (no token-range holes; the 152-SM plans differ from the 148-SM plans in runtime values --
    stream-K windows, resident pairs -- and in the TP1 tail rows that take the 256 x 512 pair tile
    from 4096 tokens there instead of 8192)."""
    assert MODULES and KERNELS
    arches = cake_jit.registered_arches()
    assert set(arches) <= set(SUPPORTED_COMPUTE_CAPABILITIES.values()) and arches
    reached = set()
    for stage in ("front", "tail"):
        for tp in SUPPORTED_TP:
            for tokens in COVERAGE_TOKENS:
                keys = route_kernel_keys(stage, tp, tokens, sm_count, arch=arch)
                assert 1 <= len(keys) <= 2
                for key in keys:
                    assert key in KERNELS, (
                        f"{stage} tp{tp} T={tokens} on {arch} with {sm_count} SMs: {key} is not registered"
                    )
                    reached.add(KERNELS[key])
                for registered in arches:
                    assert cake_jit.route_available(registered, keys)
    for tp in SUPPORTED_TP:
        for num_partials in COVERAGE_PARTIALS:
            for tokens in range(1, DECODE_MAX_T + 1):
                keys = route_kernel_keys("tail", tp, tokens, sm_count, num_partials)
                assert keys == (
                    decode_kernel_key(
                        decode_tail_plan(
                            tokens, i_local_for_tp(tp), tp, num_partials, sm_count
                        )
                    ),
                )
                assert keys[0] in KERNELS, (
                    f"tail tp{tp} T={tokens} P={num_partials} on {sm_count} SMs: {keys[0]}"
                )
                reached.add(KERNELS[keys[0]])
            # The prefill tail's norm instance follows the partial count too (packed for one partial).
            for tokens in (129, 300, 511, 512, 1024, 2047, 2048, 4096, 16384):
                keys = route_kernel_keys("tail", tp, tokens, sm_count, num_partials)
                for key in keys:
                    assert key in KERNELS, (
                        f"tail tp{tp} T={tokens} P={num_partials} on {sm_count} SMs: {key}"
                    )
                    reached.add(KERNELS[key])
    # Every registered program is reachable on some qualified part (on 152 SMs the TP1 tail rows of
    # 129..512 tokens take the 192-wide instance, so the 256-wide ring-6 / ring-7 program serves 148 SMs
    # only; the TP1 front rows there take the 224-wide instance), serves every registered architecture once
    # and carries a complete launch contract.
    assert reached <= set(MODULES)
    assert set().union(
        *(
            _reached_programs(count, part_arch)
            for count in QUALIFIED_SM_COUNTS
            for part_arch in SUPPORTED_COMPUTE_CAPABILITIES.values()
        )
    ) == set(MODULES)
    for name, record in MODULES.items():
        assert tuple(record["arches"]) == arches, name
        assert len(record["sources"]) == 2 and record["ffi_entry"] == "run"
        assert {"block", "cluster", "cooperative", "dynamic_smem_bytes"} <= set(
            record["launch"]
        )
        assert all(
            kind in {"tma_buffer", "buffer", "parameter", "grid"}
            for kind, _ in record["arg_plan"]
        )
    for key, defines in cake_jit.SPECIALIZATIONS.items():
        assert (
            key in KERNELS
            and defines
            and all(isinstance(v, int) for v in defines.values())
        )
    with pytest.raises(ValueError):
        route_kernel_keys("front", 4, 8)
    with pytest.raises(ValueError):
        route_kernel_keys("block", 1, 8)
    with pytest.raises(ValueError):
        route_kernel_keys("tail", 8, 8, sm_count, 0)


def _waits_grid_wide(key):
    """Whether the program of route ``key`` has CTAs that wait for words other CTAs of its grid release: the
    fused decode tail instances (the norm's grid counter) except the staged-rows ones (``_sb``), the fused
    tail GEMMs (row-release semaphores; the claimed ones also their item ticket) and the front's fold form
    (carries and partial runs of other lanes)."""
    stage, name = key.split(":", 1)
    if stage == "decode":
        segments = name.split("_")
        return "f" in segments and "sb" not in segments
    if stage == "tail_gemm":
        return re.match(r"tp\d+e[01]f1", name) is not None
    if stage == "front":
        return name.endswith("fold")
    return False


def test_cooperative_launch_follows_grid_wide_waits():
    """A program whose CTAs wait for words other CTAs of its grid release launches cooperatively (the driver
    holds the whole grid resident or refuses the launch); every other program launches without the attribute.
    No cooperative program uses cluster launch control, whose cancel of a cluster that has not started
    would leave that cluster's work unrun."""
    assert MODULES and KERNELS
    csrc = Path(cake_jit.__file__).resolve().parent / "csrc"
    cooperative = set()
    for key, name in KERNELS.items():
        record = MODULES[name]
        waits = _waits_grid_wide(key)
        assert bool(record["launch"]["cooperative"]) is waits, (key, name)
        if waits:
            cooperative.add(key.split(":", 1)[0])
            for source in record["sources"]:
                text = (csrc / source).read_text()
                assert "clusterlaunchcontrol" not in text, (key, name, source)
    assert cooperative == {"decode", "tail_gemm", "front"}
    # The front base instances (256- and 224-wide) and the unfused tail GEMMs hand out work items through
    # cluster launch control and never wait on another cluster: an ordinary launch.
    for key in ("front:i6144", "front:i6144e1", "front:i768", "front:i768e1"):
        assert not MODULES[KERNELS[key]]["launch"]["cooperative"], key


def test_plans_follow_the_device_sm_count():
    """152-SM parts (GB200 / GB300) re-plan the resident-pair windows from their own count; the
    decode grids are tile-bound and identical on both qualified counts."""
    for tp in SUPPORTED_TP:
        il = i_local_for_tp(tp)
        for tokens in (1, 8, 16, 64, 128):
            assert decode_front_plan(tokens, il, 148) == decode_front_plan(
                tokens, il, 152
            )
            for num_partials in COVERAGE_PARTIALS:
                assert decode_tail_plan(
                    tokens, il, tp, num_partials, 148
                ) == decode_tail_plan(tokens, il, tp, num_partials, 152)
    assert decode_front_plan(1, i_local_for_tp(8), 152)["grid"] == 94
    assert decode_front_plan(1, i_local_for_tp(1), 152)["grid"] == 131
    assert decode_tail_plan(1, i_local_for_tp(8), 8, 1, 152)["grid"] == 112
    # Prefill: 76 resident pairs instead of 74 change the trailing-wave plan of these contract
    # rows (the programs stay the same; the values are launch arguments).
    assert prefill_tail_plan(2048, 1, 148)["sk_tiles"] == 2
    assert prefill_tail_plan(2048, 1, 152)["sk_tiles"] == 0
    assert prefill_tail_plan(2048, 1, 152)["full_items"] == 224
    assert prefill_tail_plan(1024, 1, 152)["full_items"] == 76
    assert prefill_front_plan(1024, i_local_for_tp(1), 148)["sk_tiles"] == 0
    assert prefill_front_plan(1024, i_local_for_tp(1), 152)["sk_tiles"] == 36
    for tp in SUPPORTED_TP:
        for tokens in (256, 512, 300, 1024, 2048, 4096, 8192, 16384):
            same_front = route_kernel_keys(
                "front", tp, tokens, 148
            ) == route_kernel_keys("front", tp, tokens, 152)
            # The TP1 front rows of 129..512 tokens take the 224-wide tile on 152 SMs, which also
            # stream the front weights evict_first further: TP8 up to 16384 tokens, TP1 up to 2048;
            # the fold form serves the TP8 rows of 16 row tiles on both counts of sm_100a.
            differs = (tp == 1 and DECODE_MAX_T < tokens <= 2048) or (
                tp == 8 and 768 < tokens <= 16384 and cb.m_tiles_for(tokens) != 16
            )
            assert same_front == (not differs)
            same_tail = route_kernel_keys("tail", tp, tokens, 148) == route_kernel_keys(
                "tail", tp, tokens, 152
            )
            # The 256 x 512 tail tile starts at 4096 TP1 tokens on 152 SMs, at 8192 on 148; the TP1 rows of
            # 129..512 tokens take the 192-wide tile on 152 SMs, and the 128-wide TP8 tile serves T=1024 on
            # 152 SMs only.
            assert same_tail == (
                not (
                    tp == 1 and (4096 <= tokens < 8192 or DECODE_MAX_T < tokens <= 512)
                )
                and not (tp == 8 and tokens == 1024)
            )


def test_tail_n192_rule_on_152_sms():
    """152-SM parts (GB200 / GB300) run the TP1 tail rows of 129..512 tokens on the 192-wide pair tile with
    an 8-deep ring: 38 column tiles (the last one 64 columns wide), so every one of the 76 resident clusters
    takes one item (one pair row cuts its 38 tiles into two K halves); 148-SM parts keep the 256-wide tile."""
    one_row = dict(
        cluster_tiles=38,
        sk_tiles=38,
        full_items=0,
        num_items=76,
        sk_ipc=76,
        sk_max_seg=2,
    )
    two_rows = dict(
        cluster_tiles=76,
        sk_tiles=0,
        full_items=76,
        num_items=76,
        sk_ipc=152,
        sk_max_seg=1,
    )
    for tokens, expect in (
        (129, one_row),
        (256, one_row),
        (300, two_rows),
        (511, two_rows),
        (512, two_rows),
    ):
        for num_partials in (1, 3, 8):
            plan = prefill_tail_plan(tokens, 1, 152, num_partials)
            assert (plan["num_stages"], plan["block_n"], plan["n_tiles"]) == (
                8,
                192,
                38,
            )
            assert {k: plan[k] for k in expect} == expect, tokens
            assert (
                plan["gemm_grid"] == 152
                and plan["fused_norm"]
                and not plan["claim_items"]
            )
            assert plan["weights_evict_first"] and plan["final_ts"]
            assert route_kernel_keys("tail", 1, tokens, 152, num_partials) == (
                "tail_gemm:tp1e1f1s8n192t1",
            )
            assert prefill_tail_plan(tokens, 1, 148, num_partials)["block_n"] == 256
    for tokens in (128, 513, 1024):
        assert prefill_tail_plan(tokens, 1, 152)["block_n"] == 256
        assert "n192" not in "".join(route_kernel_keys("tail", 1, tokens, 152))
    for tokens in (129, 256, 512):
        assert prefill_tail_plan(tokens, 8, 152)["block_n"] != 192


def test_front_n224_rule_on_152_sms():
    """152-SM parts (GB200 / GB300) run the TP1 front rows of 129..512 tokens on the 224-wide pair tile: 75 column
    tiles per pair row (4 router + 16 latent + 55 shared, the last one 96 columns wide), whole evict_first tiles in
    one wave at one pair row and two at two; 148-SM parts and the TP8 rows keep the 256-wide tile."""
    for tokens, pair_rows in ((129, 1), (256, 1), (300, 2), (511, 2), (512, 2)):
        plan = prefill_front_plan(tokens, i_local_for_tp(1), 152)
        assert (plan["block_n"], plan["n_tiles"], plan["evict_first"]) == (
            224,
            75,
            True,
        )
        assert plan["cluster_tiles"] == 75 * pair_rows
        assert plan["num_items"] == plan["full_items"] == plan["cluster_tiles"]
        assert plan["sk_max_seg"] == 1 and plan["grid"] == 2 * 75 * pair_rows
        assert route_kernel_keys("front", 1, tokens, 152) == ("front:i6144e1n224",)
        assert prefill_front_plan(tokens, i_local_for_tp(1), 148)["block_n"] == 256
        assert route_kernel_keys("front", 1, tokens, 148) == ("front:i6144e1",)
    for tokens in (128, 513, 1024):
        assert "n224" not in "".join(route_kernel_keys("front", 1, tokens, 152))
    for tokens in (129, 256, 512):
        assert prefill_front_plan(tokens, i_local_for_tp(8), 152)["block_n"] == 256


def test_generated_program_available_follows_partials_and_sm_count(monkeypatch):
    """The availability answer resolves the exact route ``prepare`` would require: the caller's
    routed-partial count selects the decode tail instance (flashinfer-ai/flashinfer#4568)
    and an unqualified SM count is never admitted."""
    device = torch.device("cuda", 0)
    for sm_count in QUALIFIED_SM_COUNTS:
        monkeypatch.setattr(cb, "_device_facts", lambda index, c=sm_count: ((10, 3), c))
        if MODULES:
            assert cb.generated_program_available(device)
            for num_partials in COVERAGE_PARTIALS:
                expected = cake_jit.route_available(
                    "sm_103a", route_kernel_keys("tail", 8, 16, sm_count, num_partials)
                )
                assert (
                    cb.generated_program_available(device, "tail", 8, 16, num_partials)
                    is expected
                )
        assert cb.generated_program_available(device, "front", 4, 16) is False
        assert cb._device_sm_count(device) == (0, sm_count)
    monkeypatch.setattr(cb, "_device_facts", lambda index: ((10, 3), 132))
    assert cb.generated_program_available(device) is False
    assert cb.generated_program_available(device, "tail", 8, 16) is False
    with pytest.raises(NotImplementedError, match="132"):
        cb._device_sm_count(device)


def test_route_rules_select_the_expected_programs():
    """Planner rules (behaviour, not registry text): route lengths, fused norm, cache policy, ring depth."""
    assert route_kernel_keys("front", 1, 128)[0].startswith("decode:")
    assert len(route_kernel_keys("front", 1, 1024)) == 1
    assert cb.prefill_front_plan(256, i_local_for_tp(1))["evict_first"]
    assert not cb.prefill_front_plan(1024, i_local_for_tp(1))["evict_first"]
    tail = cb.prefill_tail_plan(256, 1)
    assert tail["fused_norm"] and len(route_kernel_keys("tail", 1, 256)) == 1
    assert cb.tail_gemm_num_stages(256, 1) == 6 and cb.tail_gemm_num_stages(512, 1) == 7
    assert cb.tail_gemm_config(512, 8) == (9, 128, False)
    assert cb.tail_gemm_config(512, 1) == (7, 256, True)
    # Round-4 R128: the 128-wide TP8 tile also serves T=1024 on 152 SMs (3 waves of 76 clusters); 148 SMs keep 256.
    assert cb.tail_n128_tokens(152) == (512, 1024)
    assert cb.tail_n128_tokens(148) == (512,)
    assert cb.tail_gemm_config(1024, 8, 152) == (9, 128, False)
    assert cb.tail_gemm_config(1024, 8, 148) == (7, 256, True)
    assert cb.tail_gemm_config(512, 8, 152) == (9, 128, False)
    n128_1024 = cb.prefill_tail_plan(1024, 8, 152)
    assert n128_1024["cluster_tiles"] == 224 and n128_1024["gemm_grid"] == 448
    assert n128_1024["sk_tiles"] == 0 and n128_1024["early_trigger"]
    assert not n128_1024["weights_evict_first"] and not n128_1024["fused_norm"]
    assert route_kernel_keys("tail", 8, 1024, 152) == (
        "tail_norm:e1",
        "tail_gemm:tp8e0f0s9n128",
    )
    for tokens in (300, 511):
        plan = cb.prefill_tail_plan(tokens, 8)
        assert not plan["fused_norm"] and not plan["early_trigger"]
        assert plan["weights_evict_first"]
        assert len(route_kernel_keys("tail", 8, tokens)) == 2
        assert route_kernel_keys("tail", 8, tokens)[0] == "tail_norm:e0"
    assert cb.prefill_tail_plan(512, 8)["early_trigger"]
    assert len(route_kernel_keys("tail", 1, 16384)) == 1
    assert len(route_kernel_keys("tail", 8, 16384)) == 2
    # TP8 prefill norm: the packed instance for one routed partial from 2048 tokens, the summing one otherwise.
    assert route_kernel_keys("tail", 8, 256)[0] == "tail_norm:e1"
    assert route_kernel_keys("tail", 8, 1024)[0] == "tail_norm:e1"
    assert route_kernel_keys("tail", 8, 2048)[0] == "tail_norm:e1pk"
    assert route_kernel_keys("tail", 8, 16384)[0] == "tail_norm:e1pk"
    assert route_kernel_keys("tail", 8, 16384, SM_COUNT, 2)[0] == "tail_norm:e1"
    # P >= 2 routed partials leave the staged-rows / resident-B decode instances (T <= 16, TP8).
    one = decode_tail_plan(16, i_local_for_tp(8), 8, 1)
    two = decode_tail_plan(16, i_local_for_tp(8), 8, 2)
    assert one["rows_smem"] and one["stagger"] == cb.TAIL_ROWS_STAGGER
    assert not two["rows_smem"] and not two["smem_b1"] and two["stagger"] == 0
    assert decode_kernel_key(one) != decode_kernel_key(two)
    # From N_PAD 32 one routed partial also takes the split finalize; P >= 2 keeps the single finisher.
    assert decode_kernel_key(decode_tail_plan(32, i_local_for_tp(8), 8, 1)) == (
        decode_kernel_key(decode_tail_plan(32, i_local_for_tp(8), 8, 2)) + "_sf_upf"
    )


def test_tp1_tail_prefill_routes():
    """TP1 prefill tails past the single-wave fused norm run one claimed launch on the resident grid
    (one CTA pair per resident cluster, default weight L2 policy); the 256 x 512 pair tile starts
    at 4096 tokens on 152 SMs and at 8192 on 148; the single-wave rows take the 192-wide pair tile on
    152 SMs."""
    assert cb.tail_wide_min_tokens(152) == 4096 and cb.tail_wide_min_tokens(148) == 8192
    for sm_count in QUALIFIED_SM_COUNTS:
        wide = cb.tail_wide_min_tokens(sm_count)
        for tokens in (129, 256, 300, 512, 513, 1024, 2048, 4095, 4096, 8192, 16384):
            plan = prefill_tail_plan(tokens, 1, sm_count)
            claimed = tokens > 512
            assert plan["fused_norm"] and plan["claim_items"] == claimed
            n192 = sm_count == 152 and not claimed
            assert plan["block_n"] == (512 if tokens >= wide else 192 if n192 else 256)
            assert len(route_kernel_keys("tail", 1, tokens, sm_count)) == 1
            if claimed:
                assert not plan["weights_evict_first"]
                assert plan["gemm_grid"] == cb.resident_grid(
                    plan["num_items"], sm_count
                )
                assert plan["gemm_grid"] <= sm_count
    assert cb.resident_grid(3, 148) == 6 and cb.resident_grid(500, 152) == 152


# ---------------------------------------------------------------------------
# Torch reference (nvidia/Kimi-K3-NVFP4 modeling_kimi_linear.py semantics)
# ---------------------------------------------------------------------------


def make_weights(device, seed=WEIGHT_SEED):
    """Synthetic checkpoint-shaped BF16 weights (variance ~ 1 / fan_in)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def lin(n, k):
        return (
            (torch.randn(n, k, generator=gen, dtype=torch.float32) * (k**-0.5))
            .to(torch.bfloat16)
            .to(device)
        )

    gate_weight = lin(NUM_EXPERTS, HIDDEN)
    torch.randn(
        NUM_EXPERTS, generator=gen, dtype=torch.float32
    )  # gate bias (router only; keeps the seed stream)
    norm_weight = (
        (1.0 + 0.1 * torch.randn(LATENT, generator=gen, dtype=torch.float32))
        .to(torch.bfloat16)
        .to(device)
    )
    return dict(
        gate_weight=gate_weight,
        norm_weight=norm_weight,
        down_weight=lin(LATENT, HIDDEN),
        up_weight=lin(HIDDEN, LATENT),
        shared_gate_weight=lin(SHARED_INTERMEDIATE, HIDDEN),
        shared_up_weight=lin(SHARED_INTERMEDIATE, HIDDEN),
        shared_down_weight=lin(HIDDEN, SHARED_INTERMEDIATE),
    )


def shard(weights, tp, rank):
    i_local = SHARED_INTERMEDIATE // tp
    rows = slice(rank * i_local, (rank + 1) * i_local)
    return dict(
        weights,
        shared_gate_weight=weights["shared_gate_weight"][rows].contiguous(),
        shared_up_weight=weights["shared_up_weight"][rows].contiguous(),
        shared_down_weight=weights["shared_down_weight"][:, rows].contiguous(),
    )


def situ_and_mul(gate_up):
    d = gate_up.shape[-1] // 2
    gate = gate_up[..., :d].float()
    up = gate_up[..., d:].float()
    a = SITU_BETA * torch.tanh(gate / SITU_BETA) * torch.sigmoid(gate)
    b = SITU_LINEAR_BETA * torch.tanh(up / SITU_LINEAR_BETA)
    return (a * b).to(gate_up.dtype)


def rmsnorm(x, weight, eps=RMS_EPS):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


def front_reference(x, w):
    gate_up = torch.cat(
        [
            torch.nn.functional.linear(x, w["shared_gate_weight"]),
            torch.nn.functional.linear(x, w["shared_up_weight"]),
        ],
        dim=-1,
    )
    return dict(
        logits=torch.nn.functional.linear(x.float(), w["gate_weight"].float()),
        latent=torch.nn.functional.linear(x, w["down_weight"]),
        shared_act=situ_and_mul(gate_up),
    )


def tail_reference(routed, shared_act, w, tp, rank):
    acc = routed[0].float()
    for p in range(1, routed.shape[0]):
        acc = acc + routed[p].float()
    x = acc.to(torch.bfloat16)
    y = rmsnorm(x, w["norm_weight"])
    k_up = LATENT // tp
    cols = slice(rank * k_up, (rank + 1) * k_up)
    up = torch.nn.functional.linear(y[:, cols].float(), w["up_weight"][:, cols].float())
    shared = torch.nn.functional.linear(
        shared_act.float(), w["shared_down_weight"].float()
    )
    return dict(x=x, y=y, out=(up + shared).to(torch.bfloat16))


def _gpu_arch():
    if not torch.cuda.is_available():
        return None
    return SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(0))


def _require_program(stage, tp, tokens, num_partials=1):
    arch = _gpu_arch()
    if arch is None:
        pytest.skip("the Kimi-K3 LatentMoE programs require an SM100/SM103 GPU")
    device = torch.device("cuda", 0)
    sm_count = int(torch.cuda.get_device_properties(0).multi_processor_count)
    if sm_count not in QUALIFIED_SM_COUNTS:
        pytest.skip(
            f"the programs are qualified for {QUALIFIED_SM_COUNTS} SMs, not {sm_count}"
        )
    assert cb.generated_program_available(device, stage, tp, tokens, num_partials), (
        f"the generated {stage} program for {arch} (tp {tp}, T {tokens}, P {num_partials}) "
        "is not registered"
    )
    return device


_WEIGHTS = {}


def _weights(device):
    key = str(device)
    if key not in _WEIGHTS:
        _WEIGHTS[key] = make_weights(device)
    return _WEIGHTS[key]


def _graph_replay(runner):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner()
    return graph


def _assert_close(name, actual, expected, atol, rtol):
    assert torch.isfinite(actual.float()).all(), f"{name}: non-finite values"
    err = (actual.float() - expected.float()).abs()
    tol = atol + rtol * expected.float().abs()
    bad = int((err > tol).sum())
    assert bad == 0, (
        f"{name}: {bad} elements outside atol={atol} rtol={rtol} (max abs err {float(err.max()):.4g})"
    )


# Error bounds of the decode norm's FP32 row statistic (``_decode_norm_window``).
_F32_U = 2.0**-24  # FP32 unit roundoff
# Roundings a square meets on its way into the decode programs' row sum: 111 in its lane's chain
# of LATENT / 32 = 112 squares, 5 in the warp's XOR butterfly.
_DECODE_SUM_ROUNDINGS = LATENT // 32 - 1 + 5
_MEAN_REL_ERR = 4 * _F32_U
_RSQRT_REL_ERR = 2.0**-21
_BOUND_SLACK = 2.0**-40


def _f32_bits(t):
    """FP32 bit patterns (int64) of ``t`` rounded to FP32."""
    return t.float().view(torch.int32).long()


def _decode_norm_window(x, eps=RMS_EPS):
    """Per row of ``x`` ([rows, LATENT]): the FP32 rstd values the decode norm can compute.

    The window follows the reduction structure that every decode program shares: one warp reduces
    a row, lane l adds the exact squares of columns (l + 32 i) * 8 + j (i = 0..13, j = 0..7) one
    after another in FP32 (112 terms), and an XOR butterfly over lane offsets 16, 8, 4, 2, 1 adds
    the 32 lane sums.  A square meets at most k = 111 + 5 = 116 roundings (117 when x is not
    BF16 and its square rounds too), and every addend is non-negative, so the computed sum s
    satisfies |s - S| <= gamma_k S (u = 2^-24, gamma_k = k u / (1 - k u)).  S is the FP64
    ``math.fsum`` of the exact squares (correctly rounded); every nonzero square is normal
    (|x| >= 2^-63), so flush-to-zero never acts.  A decode program with a different reduction
    structure must update this bound (``_DECODE_SUM_ROUNDINGS``).

    The mean s / n carries a relative error of at most 4u: s * fl(1 / n) is within 2u + u^2, and
    a division is within 2 ulp (the CUDA Programming Guide gives "x / y -> __fdividef(x, y) :
    2 ULP" under -prec-div=false).  Adding e = fl32(eps) rounds once more (u; an FFMA folds the
    product's rounding into it).  The rsqrt is taken at 2^-21 relative error, a margin over its
    documented bounds: the CUDA Programming Guide lists "rsqrtf(x) 2 ULP" (2^-22 relative), and
    the PTX ISA states "The maximum relative error for rsqrt.f32 over the entire positive finite
    floating-point range is 2^-22.9".  So

        m_lo = (S (1 - gamma_k) (1 - 4u) / n + e) (1 - u)
        m_hi = (S (1 + gamma_k) (1 + 4u) / n + e) (1 + u)
        rstd in [(1 - 2^-21) / sqrt(m_hi), (1 + 2^-21) / sqrt(m_lo)],

    evaluated in FP64 and widened by 2^-40 relative.  With gamma_116 = 6.914e-6 the window is
    +-4.083e-6 of the exact rstd (+-34 to +-69 FP32 ulps) where the mean square dominates eps,
    and narrower where they are comparable: +-2.04e-6 (+-17 to +-34 ulps) at a mean square of
    0.75 eps.

    Returns the FP32 bit patterns (int64, CPU) of the anchor fl32((S / n + e)^-1/2) and of the
    lowest and highest FP32 value in the window, and S.
    """
    n = x.shape[-1]
    assert n == LATENT, "the bound is for the decode programs' LATENT columns"
    x64 = x.double()
    assert bool((x64.abs() >= 2.0**-63)[x64 != 0].all()), (
        "x has a nonzero element below 2^-63 (its FP32 square is not normal)"
    )
    squares = (x64 * x64).cpu().tolist()
    exact = torch.tensor([math.fsum(row) for row in squares], dtype=torch.float64)
    assert bool((exact > 0).all()), "a zero row matches every rstd"
    assert bool((exact < 2.0**100).all()), "the sum of squares leaves the FP32 range"
    u = _F32_U
    k = _DECODE_SUM_ROUNDINGS + (x.dtype != torch.bfloat16)
    gamma = k * u / (1 - k * u)
    e = float(torch.tensor(eps, dtype=torch.float32))
    m_lo = (exact * (1 - gamma) * (1 - _MEAN_REL_ERR) / n + e) * (1 - u)
    m_hi = (exact * (1 + gamma) * (1 + _MEAN_REL_ERR) / n + e) * (1 + u)
    r_lo = (1 - _RSQRT_REL_ERR) * (1 - _BOUND_SLACK) / m_hi.sqrt()
    r_hi = (1 + _RSQRT_REL_ERR) * (1 + _BOUND_SLACK) / m_lo.sqrt()
    lo = _f32_bits(r_lo) + (r_lo.float().double() < r_lo).long()
    hi = _f32_bits(r_hi) - (r_hi.float().double() > r_hi).long()
    anchor = _f32_bits((exact / n + e).rsqrt())
    return dict(anchor=anchor, lo=lo, hi=hi, exact=exact)


def _decode_norm_y(x, weight, rstd):
    """The decode norm's y for FP32 rstd candidates ``rstd`` ([rows, k]): weight *
    bf16(fp32(x) * rstd), the product of the two BF16 values rounded to BF16 once."""
    return weight * (x.float()[:, None, :] * rstd[:, :, None]).to(torch.bfloat16)


def _decode_norm_matches(y, x, weight, eps=RMS_EPS, window=None, y_model=None):
    """The FP32 values r of each row's window (``_decode_norm_window``, enumerated by bit
    pattern) for which ``y_model(x, weight, r)`` (``_decode_norm_y``) is the row of ``y``
    bit for bit.  Per row (int64, CPU): ``count`` of such r, the lowest and highest of
    them as FP32 ulp offsets from the anchor (``lo`` / ``hi``, 0 without a match) and the
    window ``size``."""
    window = _decode_norm_window(x, eps) if window is None else window
    y_model = _decode_norm_y if y_model is None else y_model
    device = y.device
    lo, hi, anchor = (window[key].to(device) for key in ("lo", "hi", "anchor"))
    r_lo, r_hi = (b.to(torch.int32).view(torch.float32).double() for b in (lo, hi))
    # Every nonzero x * r and weight * bf16(x * r) normal and every value finite, so the
    # kernel's flush-to-zero arithmetic and torch's round alike.
    xa = x.double().abs()
    wa = weight.double().abs().expand_as(xa)
    low = xa * torch.where(wa > 0, wa.clamp(max=1.0), torch.ones_like(wa))
    high = xa * wa.clamp(min=1.0)
    assert bool((low * r_lo[:, None] >= 2.0**-125)[xa > 0].all()), (
        "x * rstd or weight * bf16(x * rstd) is not a normal FP32 value"
    )
    assert bool((high * r_hi[:, None] < 2.0**126).all()), "y leaves the BF16 range"
    rows, n = y.shape
    y_bits = y.view(torch.int16)[:, None, :]
    count = torch.zeros(rows, dtype=torch.int64, device=device)
    first = torch.full((rows,), 2**31, dtype=torch.int64, device=device)
    last = torch.full((rows,), -1, dtype=torch.int64, device=device)
    step = max(1, (1 << 24) // (rows * n))
    for k0 in range(0, int((hi - lo).max()) + 1, step):
        cand = lo[:, None] + torch.arange(k0, k0 + step, device=device)
        valid = cand <= hi[:, None]
        r = torch.minimum(cand, hi[:, None]).to(torch.int32).view(torch.float32)
        ok = (y_model(x, weight, r).view(torch.int16) == y_bits).all(-1) & valid
        count += ok.sum(-1)
        first = torch.minimum(first, torch.where(ok, cand, 2**31).amin(-1))
        last = torch.maximum(last, torch.where(ok, cand, -1).amax(-1))
    hit = count > 0
    return dict(
        count=count.cpu(),
        lo=torch.where(hit, first - anchor, 0).cpu(),
        hi=torch.where(hit, last - anchor, 0).cpu(),
        size=(hi - lo + 1).cpu(),
    )


def _assert_decode_norm_rows(y, x, weight, eps=RMS_EPS):
    """Every row of the decode norm's ``y`` is weight * bf16(fp32(x) * r) for an FP32 r in
    the row's error window of the exact statistic (``_decode_norm_window``): the reference's
    rounding points exactly, its FP32 sum of squares in the decode programs' reduction order.
    ``x`` is the reference's BF16 row (the FP32 sum of the routed partials in partial order,
    rounded once)."""
    assert y.shape == x.shape and y.dtype == torch.bfloat16
    found = _decode_norm_matches(y, x, weight, eps)
    bad = (found["count"] == 0).nonzero().flatten().tolist()
    assert not bad, (
        f"y rows {bad[:8]} ({len(bad)} of {y.shape[0]}) are weight * bf16(x * r) "
        "for no FP32 r in the error window of the exact statistic"
    )


# ---------------------------------------------------------------------------
# GPU correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tokens", SMOKE_TOKENS)
@pytest.mark.parametrize("tp", SUPPORTED_TP)
def test_front_matches_reference(tp, tokens):
    device = _require_program("front", tp, tokens)
    w = shard(_weights(device), tp, 0)
    i_local = SHARED_INTERMEDIATE // tp
    gen = torch.Generator(device="cpu").manual_seed(621 + tokens)
    x = (
        torch.randn(tokens, HIDDEN, generator=gen, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(device)
    )
    shared_gate_up = torch.cat(
        [w["shared_gate_weight"], w["shared_up_weight"]], dim=0
    ).contiguous()
    logits = torch.full(
        (tokens, NUM_EXPERTS), float("nan"), dtype=torch.float32, device=device
    )
    latent = torch.full(
        (tokens, LATENT), float("nan"), dtype=torch.bfloat16, device=device
    )
    shared_act = torch.full(
        (tokens, i_local), float("nan"), dtype=torch.bfloat16, device=device
    )
    runner = prepare_kimi_k3_latent_moe_front(
        x,
        w["gate_weight"],
        w["down_weight"],
        shared_gate_up,
        logits,
        latent,
        shared_act,
    )
    assert runner.route == ("decode" if tokens <= DECODE_MAX_T else "prefill")
    assert runner.kernel_keys == route_kernel_keys(
        "front", tp, tokens, runner.sm_count, arch=runner.arch
    )
    runner()
    torch.cuda.synchronize()
    expected = front_reference(x, w)
    _assert_close("logits", logits, expected["logits"], LOGITS_ATOL, LOGITS_RTOL)
    _assert_close("latent", latent, expected["latent"], ATOL, RTOL)
    _assert_close("shared_act", shared_act, expected["shared_act"], ATOL, RTOL)
    first = (logits.clone(), latent.clone(), shared_act.clone())
    # Idempotent re-launch and bit-identical CUDA-graph replay (the deployment form).
    runner()
    torch.cuda.synchronize()
    assert all(
        torch.equal(a, b)
        for a, b in zip(first, (logits, latent, shared_act), strict=True)
    )
    graph = _graph_replay(runner)
    for t in (logits, latent, shared_act):
        t.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert all(
        torch.equal(a, b)
        for a, b in zip(first, (logits, latent, shared_act), strict=True)
    )
    # The one-shot entry point writes the same values.
    for t in (logits, latent, shared_act):
        t.fill_(float("nan"))
    kimi_k3_latent_moe_front(
        x,
        w["gate_weight"],
        w["down_weight"],
        shared_gate_up,
        logits,
        latent,
        shared_act,
    )
    torch.cuda.synchronize()
    assert all(
        torch.equal(a, b)
        for a, b in zip(first, (logits, latent, shared_act), strict=True)
    )


def _front_case(device, tp, tokens, seed):
    w = shard(_weights(device), tp, 0)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = (
        torch.randn(tokens, HIDDEN, generator=gen, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(device)
    )
    shared_gate_up = torch.cat(
        [w["shared_gate_weight"], w["shared_up_weight"]], dim=0
    ).contiguous()
    return w, x, shared_gate_up


def _front_run(w, x, shared_gate_up, tp):
    """Prepare and launch the front on fresh NaN-filled outputs; the runner and its outputs."""
    tokens, device = x.shape[0], x.device
    outs = (
        torch.full((tokens, NUM_EXPERTS), float("nan"), device=device),
        torch.full((tokens, LATENT), float("nan"), dtype=torch.bfloat16, device=device),
        torch.full(
            (tokens, SHARED_INTERMEDIATE // tp),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        ),
    )
    runner = prepare_kimi_k3_latent_moe_front(
        x, w["gate_weight"], w["down_weight"], shared_gate_up, *outs
    )
    runner()
    torch.cuda.synchronize()
    return runner, outs


@pytest.mark.parametrize(("tp", "tokens"), ((8, 640), (8, 768), (8, 4096), (1, 1024)))
def test_front_evict_first_rows_are_a_reroute(monkeypatch, tp, tokens):
    """The rows past the two-pair-row bound that stream the weights evict_first (TP8 up to three
    pair rows; on 152 SMs every TP8 row and TP1 up to eight): the same outputs, bitwise, as the
    default-policy instance."""
    device = _require_program("front", tp, tokens)
    il = i_local_for_tp(tp)
    sm_count = int(torch.cuda.get_device_properties(0).multi_processor_count)
    if not cb.front_evict_first(cb.m_tiles_for(tokens), il, sm_count):
        pytest.skip(
            f"T={tokens} TP{tp} keeps the default weight policy on {sm_count} SMs"
        )
    w, x, shared_gate_up = _front_case(device, tp, tokens, 2700 + tokens)
    runner, outs = _front_run(w, x, shared_gate_up, tp)
    assert runner.kernel_keys == (f"front:i{il}e1",)
    monkeypatch.setattr(cb, "FRONT_EVICT_FIRST_MAX_PAIR_ROWS_BY_I_LOCAL", {})
    monkeypatch.setattr(cb, "FRONT_EVICT_FIRST_MAX_PAIR_ROWS_BY_PART", {})
    cb.prefill_front_plan.cache_clear()
    try:
        default, default_outs = _front_run(w, x, shared_gate_up, tp)
    finally:
        cb.prefill_front_plan.cache_clear()
    assert default.kernel_keys == (f"front:i{il}",)
    assert all(torch.equal(a, b) for a, b in zip(outs, default_outs, strict=True))


@pytest.mark.parametrize("tokens", (1000, 1024, 1800, 2000, 2048))
def test_front_fold_matches_the_trailing_wave_plan(monkeypatch, tokens):
    """The TP8 rows the fold form serves (16 row tiles on 152-SM parts and on sm_100a 148-SM parts, 8 on sm_103a
    152-SM parts): the device holds the fold grid, the outputs match the reference and are bitwise those of the
    trailing-wave plan the occupancy guard falls back to, and re-launches, CUDA-graph replays and the one-shot entry
    point reproduce them with every counter back at zero."""
    tp = 8
    il = i_local_for_tp(tp)
    device = _require_program("front", tp, tokens)
    arch = _gpu_arch()
    sm_count = int(torch.cuda.get_device_properties(0).multi_processor_count)
    m_tiles = cb.m_tiles_for(tokens)
    fold = cb.front_fold_plan(m_tiles, il, sm_count, arch)
    if fold is None:
        pytest.skip(f"no fold route for T={tokens} on {arch} with {sm_count} SMs")
    resident = cb._front_resident_clusters(0, arch, il)
    assert sm_count // 2 >= resident >= fold["num_items"]
    w, x, shared_gate_up = _front_case(device, tp, tokens, 2300 + tokens)
    runner, outs = _front_run(w, x, shared_gate_up, tp)
    assert runner.kernel_keys == ("front:i768e1fold",)
    assert runner.plan["fold"] and runner.plan["grid"] == 2 * fold["num_items"]
    launch = runner.launches[0]
    assert launch.cooperative_ctas == runner.plan["grid"]
    assert launch.ws_numel == 2 * runner.plan["grid"] * cb.BLOCK_M * cb.FRONT_BLOCK_N
    expected = front_reference(x, w)
    _assert_close("logits", outs[0], expected["logits"], LOGITS_ATOL, LOGITS_RTOL)
    _assert_close("latent", outs[1], expected["latent"], ATOL, RTOL)
    _assert_close("shared_act", outs[2], expected["shared_act"], ATOL, RTOL)
    first = _snapshot(outs)
    _assert_words_zero(runner, "fold launch")
    for rnd in range(3):
        _nan_fill(outs)
        runner()
        torch.cuda.synchronize()
        _assert_bitwise(outs, first, f"fold re-launch {rnd}")
        _assert_words_zero(runner, f"fold re-launch {rnd}")
    graph = _graph_replay(runner)
    for rnd in range(3):
        _nan_fill(outs)
        graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise(outs, first, f"fold replay {rnd}")
        _assert_words_zero(runner, f"fold replay {rnd}")
    _nan_fill(outs)
    kimi_k3_latent_moe_front(
        x, w["gate_weight"], w["down_weight"], shared_gate_up, *outs
    )
    torch.cuda.synchronize()
    _assert_bitwise(outs, first, "fold one-shot entry point")
    monkeypatch.setattr(
        cb, "_front_resident_clusters", lambda *args: fold["num_items"] - 1
    )
    whole, whole_outs = _front_run(w, x, shared_gate_up, tp)
    assert not whole.plan["fold"]
    assert whole.kernel_keys == (
        cb.front_kernel_key(il, cb.front_evict_first(m_tiles, il, sm_count)),
    )
    _assert_bitwise(whole_outs, first, "trailing-wave plan")


@pytest.mark.parametrize("tokens", SMOKE_TOKENS)
@pytest.mark.parametrize("tp", SUPPORTED_TP)
def test_tail_matches_reference(tp, tokens):
    device = _require_program("tail", tp, tokens)
    rank = 0
    w = shard(_weights(device), tp, rank)
    i_local = SHARED_INTERMEDIATE // tp
    gen = torch.Generator(device="cpu").manual_seed(628 + tokens)
    routed = (
        (torch.randn(1, tokens, LATENT, generator=gen) * 0.7)
        .to(torch.bfloat16)
        .to(device)
        .contiguous()
    )
    shared_act = (
        (torch.randn(tokens, i_local, generator=gen) * 0.6)
        .to(torch.bfloat16)
        .to(device)
        .contiguous()
    )
    out = torch.full(
        (tokens, HIDDEN), float("nan"), dtype=torch.bfloat16, device=device
    )
    y = torch.full((tokens, LATENT), float("nan"), dtype=torch.bfloat16, device=device)
    runner = prepare_kimi_k3_latent_moe_tail(
        routed,
        w["norm_weight"],
        w["up_weight"],
        shared_act,
        w["shared_down_weight"],
        out,
        tp=tp,
        rank=rank,
        y_workspace=y,
    )
    assert runner.route == ("decode" if tokens <= DECODE_MAX_T else "prefill")
    assert runner.kernel_keys == route_kernel_keys("tail", tp, tokens, runner.sm_count)
    assert runner.launch_count == (
        1 if tokens <= DECODE_MAX_T or runner.plan.get("fused_norm") else 2
    )
    runner()
    torch.cuda.synchronize()
    expected = tail_reference(routed, shared_act, w, tp, rank)
    # Decode route: the fused norm keeps the reference's rounding points exactly (BF16 of the FP32
    # partial sum, BF16 of x * rstd, one BF16 rounding of the weight product).  Its FP32 statistic
    # sums the squares in the decode programs' fixed order (a chain of 112 squares per lane, then a
    # warp butterfly) and takes an approximate rsqrt, so each row must equal weight * bf16(x * r)
    # bit for bit for an FP32 r in the error window of that order around the exact statistic
    # (``_assert_decode_norm_rows``).  At these inputs the window spans +-4.08e-6 of the exact rstd,
    # +-34 to +-69 FP32 ulps.  Prefill route: the one-pass RMSNorm kernel reduces the row in a
    # different FP32 order and lands within one BF16 ulp on a few elements (the Cake contract
    # receipts record the same for the production kernel), so it is held to the BF16 tolerance.
    # Both routes are bit-identical across re-launch and CUDA-graph replay below.
    _assert_close("y", y, expected["y"], ATOL, RTOL)
    if tokens <= DECODE_MAX_T:
        _assert_decode_norm_rows(y, expected["x"], w["norm_weight"])
    _assert_close("out", out, expected["out"], ATOL, RTOL)
    first = (y.clone(), out.clone())
    runner()
    torch.cuda.synchronize()
    assert torch.equal(first[0], y) and torch.equal(first[1], out)
    graph = _graph_replay(runner)
    y.fill_(float("nan"))
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(first[0], y) and torch.equal(first[1], out)
    y.fill_(float("nan"))
    out.fill_(float("nan"))
    kimi_k3_latent_moe_tail(
        routed,
        w["norm_weight"],
        w["up_weight"],
        shared_act,
        w["shared_down_weight"],
        out,
        tp=tp,
        rank=rank,
        y_workspace=y,
    )
    torch.cuda.synchronize()
    assert torch.equal(first[0], y) and torch.equal(first[1], out)


def test_tail_rank_slice_tp8():
    """A non-zero rank multiplies its own latent slice (k1_off) and its own shared-down shard."""
    tokens, tp, rank = 4, 8, 5
    device = _require_program("tail", tp, tokens)
    w = shard(_weights(device), tp, rank)
    i_local = SHARED_INTERMEDIATE // tp
    gen = torch.Generator(device="cpu").manual_seed(777)
    routed = (
        (torch.randn(1, tokens, LATENT, generator=gen) * 0.7)
        .to(torch.bfloat16)
        .to(device)
        .contiguous()
    )
    shared_act = (
        (torch.randn(tokens, i_local, generator=gen) * 0.6)
        .to(torch.bfloat16)
        .to(device)
        .contiguous()
    )
    out = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=device)
    y = torch.empty((tokens, LATENT), dtype=torch.bfloat16, device=device)
    kimi_k3_latent_moe_tail(
        routed,
        w["norm_weight"],
        w["up_weight"],
        shared_act,
        w["shared_down_weight"],
        out,
        tp=tp,
        rank=rank,
        y_workspace=y,
    )
    torch.cuda.synchronize()
    expected = tail_reference(routed, shared_act, w, tp, rank)
    _assert_decode_norm_rows(y, expected["x"], w["norm_weight"])
    _assert_close("out", out, expected["out"], ATOL, RTOL)


def _tail_case(device, tp, rank, tokens, num_partials, seed):
    w = shard(_weights(device), tp, rank)
    i_local = SHARED_INTERMEDIATE // tp
    gen = torch.Generator(device="cpu").manual_seed(seed)
    routed = (
        (torch.randn(num_partials, tokens, LATENT, generator=gen) * 0.7)
        .to(torch.bfloat16)
        .to(device)
        .contiguous()
    )
    shared_act = (
        (torch.randn(tokens, i_local, generator=gen) * 0.6)
        .to(torch.bfloat16)
        .to(device)
        .contiguous()
    )
    out = torch.full(
        (tokens, HIDDEN), float("nan"), dtype=torch.bfloat16, device=device
    )
    y = torch.full((tokens, LATENT), float("nan"), dtype=torch.bfloat16, device=device)
    return w, routed, shared_act, out, y


@pytest.mark.parametrize("tokens", (300, 511))
def test_tail_tp8_between_the_single_wave_rows(tokens):
    """TP8 tails for 256 < T < 512 run the norm launch with the late trigger before the GEMM."""
    tp, rank = 8, 0
    device = _require_program("tail", tp, tokens)
    w, routed, shared_act, out, y = _tail_case(
        device, tp, rank, tokens, 1, 900 + tokens
    )
    runner = prepare_kimi_k3_latent_moe_tail(
        routed,
        w["norm_weight"],
        w["up_weight"],
        shared_act,
        w["shared_down_weight"],
        out,
        tp=tp,
        rank=rank,
        y_workspace=y,
    )
    assert runner.route == "prefill" and runner.launch_count == 2
    assert not runner.plan["early_trigger"] and not runner.plan["fused_norm"]
    runner()
    torch.cuda.synchronize()
    expected = tail_reference(routed, shared_act, w, tp, rank)
    _assert_close("y", y, expected["y"], ATOL, RTOL)
    _assert_close("out", out, expected["out"], ATOL, RTOL)
    first = (y.clone(), out.clone())
    graph = _graph_replay(runner)
    y.fill_(float("nan"))
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(first[0], y) and torch.equal(first[1], out)


@pytest.mark.parametrize("tokens", (1024, 2048))
@pytest.mark.parametrize("num_partials", (1, 2))
def test_tail_tp8_prefill_norm_follows_partials(tokens, num_partials):
    """TP8 prefill tails of one routed partial from 2048 tokens run the packed norm instance, fewer
    tokens or more partials the summing one; both reproduce across re-launch and CUDA-graph replay."""
    tp, rank = 8, 0
    device = _require_program("tail", tp, tokens, num_partials)
    w, routed, shared_act, out, y = _tail_case(
        device, tp, rank, tokens, num_partials, 1300 + tokens + num_partials
    )
    runner = prepare_kimi_k3_latent_moe_tail(
        routed,
        w["norm_weight"],
        w["up_weight"],
        shared_act,
        w["shared_down_weight"],
        out,
        tp=tp,
        rank=rank,
        y_workspace=y,
    )
    assert runner.route == "prefill" and runner.launch_count == 2
    packed = num_partials == 1 and tokens >= cb.PACKED_NORM_MIN_T
    assert runner.plan["packed_norm"] == packed
    assert runner.kernel_keys == route_kernel_keys(
        "tail", tp, tokens, runner.sm_count, num_partials
    )
    assert runner.kernel_keys[0].endswith("pk") == packed
    runner()
    torch.cuda.synchronize()
    expected = tail_reference(routed, shared_act, w, tp, rank)
    _assert_close("y", y, expected["y"], ATOL, RTOL)
    _assert_close("out", out, expected["out"], ATOL, RTOL)
    first = (y.clone(), out.clone())
    runner()
    torch.cuda.synchronize()
    assert torch.equal(first[0], y) and torch.equal(first[1], out)
    graph = _graph_replay(runner)
    y.fill_(float("nan"))
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(first[0], y) and torch.equal(first[1], out)


@pytest.mark.parametrize("tokens", (1024, 2048))
def test_tail_tp1_claimed_resident_grid(tokens):
    """TP1 tails past the single-wave fused norm run one claimed launch on the resident grid; the
    kernel re-arms its claim counter, so re-launch and CUDA-graph replay reproduce the result."""
    tp, rank = 1, 0
    device = _require_program("tail", tp, tokens)
    w, routed, shared_act, out, y = _tail_case(
        device, tp, rank, tokens, 1, 1100 + tokens
    )
    runner = prepare_kimi_k3_latent_moe_tail(
        routed,
        w["norm_weight"],
        w["up_weight"],
        shared_act,
        w["shared_down_weight"],
        out,
        tp=tp,
        rank=rank,
        y_workspace=y,
    )
    assert runner.route == "prefill" and runner.launch_count == 1
    assert runner.plan["claim_items"] and runner.plan["fused_norm"]
    assert runner.plan["gemm_grid"] <= runner.sm_count
    runner()
    torch.cuda.synchronize()
    expected = tail_reference(routed, shared_act, w, tp, rank)
    _assert_close("y", y, expected["y"], ATOL, RTOL)
    _assert_close("out", out, expected["out"], ATOL, RTOL)
    first = (y.clone(), out.clone())
    for _ in range(2):
        y.fill_(float("nan"))
        out.fill_(float("nan"))
        runner()
        torch.cuda.synchronize()
        assert torch.equal(first[0], y) and torch.equal(first[1], out)
    graph = _graph_replay(runner)
    for _ in range(2):
        y.fill_(float("nan"))
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(first[0], y) and torch.equal(first[1], out)


@pytest.mark.parametrize("tokens", (3, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128))
@pytest.mark.parametrize("num_partials", (2, 3))
def test_tail_decode_sums_routed_partials(tokens, num_partials):
    """The fused decode tail reduces P >= 2 routed partials on device before the norm."""
    tp, rank = 8, 1
    device = _require_program("tail", tp, tokens, num_partials)
    w, routed, shared_act, out, y = _tail_case(
        device, tp, rank, tokens, num_partials, 1000 + tokens * 10 + num_partials
    )
    runner = prepare_kimi_k3_latent_moe_tail(
        routed,
        w["norm_weight"],
        w["up_weight"],
        shared_act,
        w["shared_down_weight"],
        out,
        tp=tp,
        rank=rank,
        y_workspace=y,
    )
    assert runner.route == "decode" and runner.launch_count == 1
    runner()
    torch.cuda.synchronize()
    expected = tail_reference(routed, shared_act, w, tp, rank)
    _assert_decode_norm_rows(y, expected["x"], w["norm_weight"])
    _assert_close("out", out, expected["out"], ATOL, RTOL)
    first = (y.clone(), out.clone())
    runner()
    torch.cuda.synchronize()
    assert torch.equal(first[0], y) and torch.equal(first[1], out)


# (tp, rank, tokens, num_partials) of the eps-scale decode cases: TP1 and TP8, P = 1 to 5, T 1 to
# 128.  On every qualified part each decode tail program the package routes is served by at least
# one row (``test_eps_scale_rows_reach_every_decode_tail_program``).  The resident-B routes of more
# than four tokens appear at their largest token count, so the norm code of the rows past the first
# row of each of the four epilogue warps runs too.
_EPS_SCALE_ROWS = (
    (1, 0, 1, 1),
    (1, 0, 8, 2),
    (1, 0, 6, 3),
    (1, 0, 12, 2),
    (1, 0, 20, 1),
    (1, 0, 24, 2),
    (1, 0, 33, 1),
    (1, 0, 48, 3),
    (1, 0, 128, 1),
    (1, 0, 100, 2),
    (8, 0, 8, 1),
    (8, 1, 16, 1),
    (8, 1, 3, 2),
    (8, 1, 6, 2),
    (8, 1, 4, 4),
    (8, 5, 2, 5),
    (8, 1, 12, 3),
    (8, 1, 24, 1),
    (8, 1, 24, 3),
    (8, 1, 33, 1),
    (8, 1, 64, 3),
    (8, 1, 100, 1),
    (8, 5, 128, 2),
    (1, 0, 8, 3),
    (1, 0, 8, 4),
    (1, 0, 8, 5),
)


def test_eps_scale_rows_reach_every_decode_tail_program():
    """On every qualified SM count and architecture the eps-scale rows reach exactly the decode
    tail programs the package routes for T <= DECODE_MAX_T, TP1 and TP8 and P = 1 to 16 (the
    decode tail plan depends on P only through P = 1, the partial counts of
    ``TAIL_SB_NORM_ROUTES`` and the partial ranges of ``TAIL_PARTIAL_PREFETCH_ROUTES``, all below
    16 or open above), so the near-eps check runs on every decode tail program."""
    partials = range(1, 17)
    for _, cells in cb.TAIL_SB_NORM_ROUTES.values():
        assert max(cells.values(), default=1) < max(partials)
    for cells in cb.TAIL_PARTIAL_PREFETCH_ROUTES.values():
        for ranges in cells.values():
            assert all(first < max(partials) for first, _, _ in ranges)
            assert all(last is None or last < max(partials) for _, last, _ in ranges)
    for sm_count in QUALIFIED_SM_COUNTS:
        for arch in SUPPORTED_COMPUTE_CAPABILITIES.values():
            routed = {
                key
                for tp in SUPPORTED_TP
                for tokens in range(1, DECODE_MAX_T + 1)
                for num_partials in partials
                for key in route_kernel_keys(
                    "tail", tp, tokens, sm_count, num_partials, arch=arch
                )
            }
            reached = {
                key
                for tp, _, tokens, num_partials in _EPS_SCALE_ROWS
                for key in route_kernel_keys(
                    "tail", tp, tokens, sm_count, num_partials, arch=arch
                )
            }
            assert all(key.startswith("decode:") for key in routed)
            assert reached == routed, (
                f"{sm_count} SMs, {arch}: no eps-scale row reaches {sorted(routed - reached)}; "
                f"{sorted(reached - routed)} are not decode tail routes"
            )


@pytest.mark.parametrize(("tp", "rank", "tokens", "num_partials"), _EPS_SCALE_ROWS)
def test_tail_decode_norm_near_eps(tp, rank, tokens, num_partials):
    """Routed partials scaled by 2^-8 put each row's mean square near eps (about 0.75 eps per
    partial), where the eps term moves rstd by 13 % (P = 5) to 53 % (P = 1): the decode norm
    rows still equal weight * bf16(x * r) bit for bit for an FP32 r in the error window of the
    exact statistic (``_assert_decode_norm_rows``)."""
    device = _require_program("tail", tp, tokens, num_partials)
    w, routed, shared_act, out, y = _tail_case(
        device, tp, rank, tokens, num_partials, 2000 + tokens * 10 + num_partials
    )
    routed.mul_(2.0**-8)  # exact: a power-of-two scale of normal BF16 values
    runner = prepare_kimi_k3_latent_moe_tail(
        routed,
        w["norm_weight"],
        w["up_weight"],
        shared_act,
        w["shared_down_weight"],
        out,
        tp=tp,
        rank=rank,
        y_workspace=y,
    )
    assert runner.route == "decode" and runner.launch_count == 1
    runner()
    torch.cuda.synchronize()
    expected = tail_reference(routed, shared_act, w, tp, rank)
    mean_square = expected["x"].float().pow(2).mean(-1)
    assert bool(((mean_square > RMS_EPS / 2) & (mean_square < 8 * RMS_EPS)).all())
    _assert_decode_norm_rows(y, expected["x"], w["norm_weight"])
    _assert_close("y", y, expected["y"], ATOL, RTOL)
    _assert_close("out", out, expected["out"], ATOL, RTOL)


def _sb_norm_rows():
    """The (tp, tokens, num_partials) cells ``TAIL_SB_NORM_ROUTES`` sends to the global-y instance with the resident-B
    arithmetic, at the fewest partials each serves."""
    return tuple(
        (tp, tokens, least)
        for tp, (_, cells) in sorted(cb.TAIL_SB_NORM_ROUTES.items())
        for tokens, least in sorted(cells.items())
    )


@pytest.fixture
def no_partial_prefetch(monkeypatch):
    """The routes without the prefetch cells of ``TAIL_PARTIAL_PREFETCH_ROUTES`` (which overlay
    ``TAIL_SB_NORM_ROUTES``; their own rules and outputs are tested below), the plan cache cleared around."""
    monkeypatch.setattr(cb, "TAIL_PARTIAL_PREFETCH_ROUTES", {})
    decode_tail_plan.cache_clear()
    yield
    monkeypatch.undo()
    decode_tail_plan.cache_clear()


def test_tail_sb_norm_rules(monkeypatch, no_partial_prefetch):
    """The few-token multi-partial tails of ``TAIL_SB_NORM_ROUTES`` take the global-y instance of the table's N_PAD with
    the resident-B arithmetic (``sb_norm``, ``_sbn``): its plan is that instance's but for the flag, on both qualified
    SM counts and from the table's partial count up; every other row keeps its plan, and the rows taken are rows of the
    resident-B instance (``_sb``) the planner chooses without the table."""
    assert cb.TAIL_SB_NORM_ROUTES == SB_NORM_ROUTES
    taken = {}
    for sm_count in QUALIFIED_SM_COUNTS:
        for tp in SUPPORTED_TP:
            n_pad, least = cb.TAIL_SB_NORM_ROUTES[tp]
            for tokens in range(1, DECODE_MAX_T + 1):
                for num_partials in range(1, 17):
                    plan = decode_tail_plan(
                        tokens, i_local_for_tp(tp), tp, num_partials, sm_count
                    )
                    routed = tokens in least and num_partials >= least[tokens]
                    assert plan["sb_norm"] is routed, (tp, tokens, num_partials)
                    assert cb.tail_sb_norm_n_pad(tokens, tp, num_partials) == (
                        n_pad if routed else 0
                    )
                    key = decode_kernel_key(plan)
                    if not routed:
                        assert not key.endswith("_sbn")
                        continue
                    base = decode_tail_plan(
                        n_pad, i_local_for_tp(tp), tp, num_partials, sm_count
                    )
                    assert not (base["sb_norm"] or base["smem_b1"] or base["split_fin"])
                    assert plan == dict(base, sb_norm=True)
                    assert key == decode_kernel_key(base) + "_sbn" and key in KERNELS
                    taken[sm_count, tp, tokens, num_partials] = key
    assert taken
    monkeypatch.setattr(cb, "TAIL_SB_NORM_ROUTES", {})
    decode_tail_plan.cache_clear()
    try:
        for sm_count, tp, tokens, num_partials in taken:
            default = decode_tail_plan(
                tokens, i_local_for_tp(tp), tp, num_partials, sm_count
            )
            assert default["smem_b1"] and not default["rows_smem"]
            assert not default["sb_norm"] and "_sb" in decode_kernel_key(default)
    finally:
        monkeypatch.undo()
        decode_tail_plan.cache_clear()


@pytest.mark.parametrize(("tp", "tokens", "num_partials"), _sb_norm_rows())
def test_tail_sb_norm_is_bitwise_the_resident_b_instance(
    monkeypatch, no_partial_prefetch, tp, tokens, num_partials
):
    """The global-y instance with the resident-B arithmetic reproduces the resident-B instance's ``y`` and ``out`` bit for
    bit -- partial sums that flush to ``-0.0`` and subnormal partials of both signs included -- in the first launch,
    re-launches over NaN-filled outputs and graph replays, with the control words back at zero."""
    rank = tp - 1
    device = _require_program("tail", tp, tokens, num_partials)
    seed = 4100 + 97 * tp + 13 * tokens + num_partials
    w, routed, shared_act, out, y = _tail_case(
        device, tp, rank, tokens, num_partials, seed
    )
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    parts = routed.double().cpu()
    pick = torch.rand(tokens, LATENT, generator=gen)
    # A quarter of the elements: partials 0 and 1 cancel to -m * 2^-133, a negative FP32 subnormal the
    # flush-to-zero add turns into -0.0, and the other partials are -0.0 (the sum keeps its sign).
    a = 2.0**-126 * (
        1.0 + torch.randint(16, 128, (tokens, LATENT), generator=gen).double() / 128.0
    )
    m = torch.randint(1, 16, (tokens, LATENT), generator=gen).double()
    cancel = pick < 0.25
    parts[0] = torch.where(cancel, -a, parts[0])
    parts[1] = torch.where(cancel, a - m * 2.0**-133, parts[1])
    parts[2:] = torch.where(cancel, torch.full_like(parts[2:], -0.0), parts[2:])
    # Another quarter: bf16 subnormal partials of both signs.
    parts = torch.where((pick >= 0.25) & (pick < 0.5), parts * 2.0**-130, parts)
    routed.copy_(parts.to(torch.float32).to(torch.bfloat16).to(device))
    runner = prepare_kimi_k3_latent_moe_tail(
        routed,
        w["norm_weight"],
        w["up_weight"],
        shared_act,
        w["shared_down_weight"],
        out,
        tp=tp,
        rank=rank,
        y_workspace=y,
    )
    assert runner.plan["sb_norm"] and runner.kernel_keys[0].endswith("_sbn")
    runner()
    torch.cuda.synchronize()
    first = _snapshot((y, out))
    assert int((first[0] == -32768).sum()) > 0, "no -0.0 in y"
    _assert_words_zero(runner, "first launch")
    for rnd in range(3):
        _nan_fill((y, out))
        runner()
        torch.cuda.synchronize()
        _assert_bitwise((y, out), first, f"re-launch {rnd}")
    graph = _graph_replay(runner)
    for rnd in range(3):
        _nan_fill((y, out))
        graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise((y, out), first, f"replay {rnd}")
        _assert_words_zero(runner, f"replay {rnd}")
    monkeypatch.setattr(cb, "TAIL_SB_NORM_ROUTES", {})
    decode_tail_plan.cache_clear()
    try:
        out_sb = torch.full_like(out, float("nan"))
        y_sb = torch.full_like(y, float("nan"))
        resident = prepare_kimi_k3_latent_moe_tail(
            routed,
            w["norm_weight"],
            w["up_weight"],
            shared_act,
            w["shared_down_weight"],
            out_sb,
            tp=tp,
            rank=rank,
            y_workspace=y_sb,
        )
        assert resident.plan["smem_b1"] and not resident.plan["sb_norm"]
        resident()
        torch.cuda.synchronize()
        _assert_bitwise((y_sb, out_sb), first, "resident-B instance")
    finally:
        monkeypatch.undo()
        decode_tail_plan.cache_clear()


def _partial_prefetch_rows():
    """The (tp, tokens, num_partials, sb_norm) cells of ``TAIL_PARTIAL_PREFETCH_ROUTES``: the first and the last partial
    count of every range, a range open above at 17 partials (past the measured 2 to 16)."""
    rows = []
    for tp, cells in sorted(cb.TAIL_PARTIAL_PREFETCH_ROUTES.items()):
        for tokens, ranges in sorted(cells.items()):
            for first, last, sb_norm in ranges:
                for num_partials in sorted({first, 17 if last is None else last}):
                    rows.append((tp, tokens, num_partials, sb_norm))
    return tuple(rows)


def _without_partial_prefetch(tokens, tp, num_partials, sm_count, sb_norm):
    """The plan of the instance a prefetch cell names, without the prefetch: the global-y instance of
    ``TAIL_SB_NORM_ROUTES`` with the resident-B arithmetic, or the resident-B instance."""
    if sb_norm:
        n_pad = SB_NORM_ROUTES[tp][0]
        base = decode_tail_plan(n_pad, i_local_for_tp(tp), tp, num_partials, sm_count)
        assert not (base["sb_norm"] or base["smem_b1"] or base["split_fin"])
        return dict(base, sb_norm=True)
    base = decode_tail_plan(tokens, i_local_for_tp(tp), tp, num_partials, sm_count)
    assert base["smem_b1"] and not base["rows_smem"] and not base["sb_norm"]
    return base


def test_tail_partial_prefetch_rules(monkeypatch):
    """The cells of ``TAIL_PARTIAL_PREFETCH_ROUTES`` take the instance the table names (the global-y instance with the
    resident-B arithmetic, or the resident-B instance) with the routed partials prefetched into L2 (``_ppf``): its plan
    is that instance's but for the flag, on both qualified SM counts and up to 17 partials (a range open above
    included); every other row keeps its plan."""
    assert cb.TAIL_PARTIAL_PREFETCH_ROUTES == PARTIAL_PREFETCH_ROUTES
    taken = {}
    for sm_count in QUALIFIED_SM_COUNTS:
        for tp in SUPPORTED_TP:
            for tokens in range(1, DECODE_MAX_T + 1):
                for num_partials in range(1, 18):
                    plan = decode_tail_plan(
                        tokens, i_local_for_tp(tp), tp, num_partials, sm_count
                    )
                    route = cb.tail_partial_prefetch(tokens, tp, num_partials)
                    key = decode_kernel_key(plan)
                    assert plan["partial_prefetch"] is (route is not None)
                    assert key.endswith("_ppf") is (route is not None)
                    if route is not None:
                        assert key in KERNELS, key
                        taken[sm_count, tp, tokens, num_partials] = (route, plan)
    assert taken or not PARTIAL_PREFETCH_ROUTES
    monkeypatch.setattr(cb, "TAIL_PARTIAL_PREFETCH_ROUTES", {})
    decode_tail_plan.cache_clear()
    try:
        for (sm_count, tp, tokens, num_partials), (route, plan) in taken.items():
            monkeypatch.setattr(cb, "TAIL_SB_NORM_ROUTES", {})
            decode_tail_plan.cache_clear()
            base = _without_partial_prefetch(tokens, tp, num_partials, sm_count, route)
            monkeypatch.setattr(cb, "TAIL_SB_NORM_ROUTES", SB_NORM_ROUTES)
            decode_tail_plan.cache_clear()
            cell = (tp, tokens, num_partials)
            assert plan == dict(base, partial_prefetch=True), cell
            assert decode_kernel_key(plan) == decode_kernel_key(base) + "_ppf"
    finally:
        monkeypatch.undo()
        decode_tail_plan.cache_clear()


@pytest.mark.parametrize(
    ("tp", "tokens", "num_partials", "sb_norm"), _partial_prefetch_rows()
)
def test_tail_partial_prefetch_is_bitwise_the_resident_b_instance(
    monkeypatch, tp, tokens, num_partials, sb_norm
):
    """The routed-partial prefetch changes no output: the ``_ppf`` instance reproduces the resident-B instance the
    planner takes without ``TAIL_SB_NORM_ROUTES`` and ``TAIL_PARTIAL_PREFETCH_ROUTES`` -- for a resident-B cell the same
    instance without the prefetch; the global-y instance with the resident-B arithmetic is bitwise that instance too
    (``test_tail_sb_norm_is_bitwise_the_resident_b_instance``) -- bit for bit, partial sums that flush to ``-0.0`` and
    subnormal partials of both signs included, in the first launch, re-launches over NaN-filled outputs and graph
    replays, with the control words back at zero."""
    rank = tp - 1
    device = _require_program("tail", tp, tokens, num_partials)
    seed = 5200 + 97 * tp + 13 * tokens + num_partials
    w, routed, shared_act, out, y = _tail_case(
        device, tp, rank, tokens, num_partials, seed
    )
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    parts = routed.double().cpu()
    pick = torch.rand(tokens, LATENT, generator=gen)
    a = 2.0**-126 * (
        1.0 + torch.randint(16, 128, (tokens, LATENT), generator=gen).double() / 128.0
    )
    m = torch.randint(1, 16, (tokens, LATENT), generator=gen).double()
    cancel = pick < 0.25
    parts[0] = torch.where(cancel, -a, parts[0])
    parts[1] = torch.where(cancel, a - m * 2.0**-133, parts[1])
    parts[2:] = torch.where(cancel, torch.full_like(parts[2:], -0.0), parts[2:])
    parts = torch.where((pick >= 0.25) & (pick < 0.5), parts * 2.0**-130, parts)
    routed.copy_(parts.to(torch.float32).to(torch.bfloat16).to(device))
    args = (
        routed,
        w["norm_weight"],
        w["up_weight"],
        shared_act,
        w["shared_down_weight"],
    )
    runner = prepare_kimi_k3_latent_moe_tail(
        *args, out, tp=tp, rank=rank, y_workspace=y
    )
    assert runner.plan["partial_prefetch"] and runner.kernel_keys[0].endswith("_ppf")
    assert runner.plan["sb_norm"] is sb_norm
    runner()
    torch.cuda.synchronize()
    first = _snapshot((y, out))
    _assert_words_zero(runner, "first launch")
    for rnd in range(3):
        _nan_fill((y, out))
        runner()
        torch.cuda.synchronize()
        _assert_bitwise((y, out), first, f"re-launch {rnd}")
    graph = _graph_replay(runner)
    for rnd in range(3):
        _nan_fill((y, out))
        graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise((y, out), first, f"replay {rnd}")
        _assert_words_zero(runner, f"replay {rnd}")
    monkeypatch.setattr(cb, "TAIL_PARTIAL_PREFETCH_ROUTES", {})
    monkeypatch.setattr(cb, "TAIL_SB_NORM_ROUTES", {})
    decode_tail_plan.cache_clear()
    try:
        out_b = torch.full_like(out, float("nan"))
        y_b = torch.full_like(y, float("nan"))
        base = prepare_kimi_k3_latent_moe_tail(
            *args, out_b, tp=tp, rank=rank, y_workspace=y_b
        )
        assert not (base.plan["partial_prefetch"] or base.plan["sb_norm"])
        assert base.plan["smem_b1"] and not base.plan["rows_smem"]
        if not sb_norm:
            assert base.kernel_keys[0] + "_ppf" == runner.kernel_keys[0]
        base()
        torch.cuda.synchronize()
        _assert_bitwise((y_b, out_b), first, "resident-B instance")
    finally:
        monkeypatch.undo()
        decode_tail_plan.cache_clear()


@pytest.mark.parametrize("tokens", (256, 300, 512))
def test_tail_prefill_on_a_side_stream(tokens):
    """The TP1 prefill tail rows of one wave (the fused single-wave GEMMs, launched cooperatively and
    programmatic-dependent) run on a side stream behind other work on it, eagerly and captured there, with the
    outputs of the current stream, bitwise, and every control word back at zero."""
    tp = 1
    device = _require_program("tail", tp, tokens)
    args, kwargs = _tail_args(device, tp, tokens, 1, 3100 + tokens)
    runner = prepare_kimi_k3_latent_moe_tail(*args, **kwargs)
    gemm = [launch for launch in runner.launches if launch.key.startswith("tail_gemm:")]
    assert gemm and all(launch.cooperative_ctas for launch in gemm), runner.kernel_keys
    runner()
    torch.cuda.synchronize()
    reference = _snapshot(runner.outputs)
    side = torch.cuda.Stream()
    for rnd in range(3):
        _nan_fill(runner.outputs)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            torch.cuda._sleep(100_000)
            runner()
        torch.cuda.synchronize()
        _assert_bitwise(runner.outputs, reference, f"T={tokens} side stream {rnd}")
        _assert_words_zero(runner, f"T={tokens} side stream {rnd}")
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        runner()
    for rnd in range(3):
        _nan_fill(runner.outputs)
        torch.cuda.synchronize()
        with torch.cuda.stream(side):
            graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise(
            runner.outputs, reference, f"T={tokens} side-stream replay {rnd}"
        )
        _assert_words_zero(runner, f"T={tokens} side-stream replay {rnd}")


def test_rejects_invalid_operands():
    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )
    x = torch.zeros(4, HIDDEN, dtype=torch.bfloat16, device=device)
    gate = torch.zeros(NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16, device=device)
    down = torch.zeros(LATENT, HIDDEN, dtype=torch.bfloat16, device=device)
    sgu = torch.zeros(2 * 768, HIDDEN, dtype=torch.bfloat16, device=device)
    logits = torch.zeros(4, NUM_EXPERTS, dtype=torch.float32, device=device)
    latent = torch.zeros(4, LATENT, dtype=torch.bfloat16, device=device)
    act = torch.zeros(4, 768, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError):
        prepare_kimi_k3_latent_moe_front(
            x.float(), gate, down, sgu, logits, latent, act
        )
    with pytest.raises(ValueError):
        prepare_kimi_k3_latent_moe_front(
            x,
            gate,
            down,
            torch.zeros(2 * 512, HIDDEN, dtype=torch.bfloat16, device=device),
            logits,
            latent,
            act,
        )
    with pytest.raises(ValueError):
        prepare_kimi_k3_latent_moe_front(
            x, gate, down, sgu, logits.to(torch.bfloat16), latent, act
        )
    routed = torch.zeros(1, 4, LATENT, dtype=torch.bfloat16, device=device)
    nw = torch.zeros(LATENT, dtype=torch.bfloat16, device=device)
    up = torch.zeros(HIDDEN, LATENT, dtype=torch.bfloat16, device=device)
    sd = torch.zeros(HIDDEN, 768, dtype=torch.bfloat16, device=device)
    out = torch.zeros(4, HIDDEN, dtype=torch.bfloat16, device=device)
    y = torch.zeros(4, LATENT, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError):
        prepare_kimi_k3_latent_moe_tail(
            routed, nw, up, act, sd, out, tp=4, rank=0, y_workspace=y
        )
    with pytest.raises(ValueError):
        prepare_kimi_k3_latent_moe_tail(
            routed, nw, up, act, sd, out, tp=8, rank=8, y_workspace=y
        )
    with pytest.raises(ValueError):
        prepare_kimi_k3_latent_moe_tail(
            routed[0], nw, up, act, sd, out, tp=8, rank=0, y_workspace=y
        )
    with pytest.raises(ValueError):
        prepare_kimi_k3_latent_moe_tail(
            routed, nw, up, act, sd, out, tp=8, rank=0, y_workspace=y[:, : LATENT // 2]
        )
    with pytest.raises(ValueError):
        kimi_k3_latent_moe_front(
            x, gate, down, sgu, logits, latent, act, backend="cutlass"
        )


# ---------------------------------------------------------------------------
# Device scratch ownership and concurrency (per-runner control words, per-stream partials)
# ---------------------------------------------------------------------------


def test_partial_workspace_and_control_word_sizes():
    """Split prefill plans use one ``128 x block_n`` FP32 slot per contributor CTA (whole plans none), the front
    fold form two ``128 x 256`` slots and two counters per CTA, and every runner's control words fit one KiB in
    128-byte slots (the fold form's 1.25 KiB on 152 SMs)."""
    for sm_count in QUALIFIED_SM_COUNTS:
        for arch in SUPPORTED_COMPUTE_CAPABILITIES.values():
            for tp in SUPPORTED_TP:
                for tokens in sorted(set(cb.ROW_TOKENS) | {129, 300, 511, 1000, 3000}):
                    if tokens <= DECODE_MAX_T:
                        continue
                    tail = prefill_tail_plan(tokens, tp, sm_count)
                    front = prefill_front_plan(
                        tokens, i_local_for_tp(tp), sm_count, arch
                    )
                    for stage, plan, block_n, n_u32 in (
                        ("tail", tail, tail["block_n"], cb.NORM_COUNTER_WORDS),
                        ("front", front, cb.FRONT_BLOCK_N, 0),
                    ):
                        if plan.get("fold"):
                            slots = expected_words = 2 * plan["grid"]
                            max_bytes = 1280
                        else:
                            slots = plan["sk_tiles"] * plan["sk_max_seg"] * cb.CTA_GROUP
                            expected_words = max(2, plan["sk_tiles"] * cb.CTA_GROUP)
                            max_bytes = 1024
                        assert cb._partial_ws_numel(stage, plan) == (
                            slots * cb.BLOCK_M * block_n
                        )
                        words = cb._arrival_counter_words(plan)
                        assert words == expected_words
                        u32_off, slot_words = cb._ctrl_layout(words, n_u32)
                        assert u32_off >= words and u32_off % cb.CTRL_SLOT_WORDS == 0
                        assert slot_words % cb.CTRL_SLOT_WORDS == 0
                        assert slot_words >= u32_off + n_u32
                        assert 4 * slot_words <= max_bytes
    assert cb._ctrl_layout(0, cb.DECODE_COUNTER_WORDS) == (0, cb.CTRL_SLOT_WORDS)
    # A plan that carries partials between its items states its slots in ``ws_slots``: the front fold form's two
    # slots and two counters per CTA.
    fold = dict(sk_tiles=0, sk_max_seg=384, ws_slots=288)
    assert cb._partial_ws_numel("front", fold) == 288 * cb.BLOCK_M * cb.FRONT_BLOCK_N
    assert cb._arrival_counter_words(fold) == 288


def test_ws_programs_never_trigger_dependents():
    """A program bound to a partial workspace (``('buffer', 'ws')`` in its argument plan) must not trigger
    ``griddepcontrol.launch_dependents``: a dependent launched early on the same stream could write the stream's
    workspace (or a recycled block) while the program still uses it."""
    csrc = Path(cake_jit.__file__).resolve().parent / "csrc"
    ws_programs = [
        name
        for name, record in MODULES.items()
        if ["buffer", "ws"] in [list(arg) for arg in record["arg_plan"]]
    ]
    assert ws_programs
    for name in ws_programs:
        for source in MODULES[name]["sources"]:
            text = (csrc / source).read_text()
            assert "griddepcontrol.launch_dependents" not in text, (name, source)
            assert "cudaTriggerProgrammaticLaunchCompletion" not in text, (
                name,
                source,
            )


def _bits(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def _snapshot(outputs):
    return [_bits(t).clone() for t in outputs]


def _assert_bitwise(outputs, reference, what):
    for i, (t, r) in enumerate(zip(outputs, reference, strict=True)):
        bad = int((_bits(t) != r).sum())
        assert bad == 0, f"{what}: output {i} differs in {bad} elements"


def _nan_fill(outputs):
    for t in outputs:
        t.fill_(float("nan"))


def _control_tensors(runner):
    """The runner's bound control words (``counters`` / ``norm_counter`` of every launch)."""
    return [
        launch.kwargs[name]
        for launch in runner.launches
        for name in ("counters", "norm_counter")
        if name in launch.kwargs
    ]


def _byte_range(t):
    start = t.data_ptr()
    return start, start + t.numel() * t.element_size()


def _assert_words_zero(runner, what):
    for t in _control_tensors(runner):
        assert int(torch.count_nonzero(t.view(torch.int32))) == 0, (
            f"{what}: control words not zero"
        )


def _front_args(device, tp, tokens, seed):
    w = shard(_weights(device), tp, 0)
    i_local = SHARED_INTERMEDIATE // tp
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = (
        torch.randn(tokens, HIDDEN, generator=gen, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(device)
    )
    shared_gate_up = torch.cat(
        [w["shared_gate_weight"], w["shared_up_weight"]], dim=0
    ).contiguous()

    def nan(*shape, dtype=torch.bfloat16):
        return torch.full(shape, float("nan"), dtype=dtype, device=device)

    return (
        x,
        w["gate_weight"],
        w["down_weight"],
        shared_gate_up,
        nan(tokens, NUM_EXPERTS, dtype=torch.float32),
        nan(tokens, LATENT),
        nan(tokens, i_local),
    )


def _tail_args(device, tp, tokens, num_partials, seed):
    w, routed, shared_act, out, y = _tail_case(
        device, tp, 0, tokens, num_partials, seed
    )
    args = (
        routed,
        w["norm_weight"],
        w["up_weight"],
        shared_act,
        w["shared_down_weight"],
        out,
    )
    return args, dict(tp=tp, rank=0, y_workspace=y)


def _prepare(device, stage, tp, tokens, num_partials=1, seed=0):
    if stage == "front":
        return prepare_kimi_k3_latent_moe_front(*_front_args(device, tp, tokens, seed))
    args, kwargs = _tail_args(device, tp, tokens, num_partials, seed)
    return prepare_kimi_k3_latent_moe_tail(*args, **kwargs)


def _force_tail_plan(monkeypatch, **overrides):
    """Route the TP1 T=256 tail to another registered program of the 256-wide tile and its stream-K plan for the
    device's SM count.  152-SM parts run that row on the 192-wide tile (``tail_gemm_config``); the forced plan is
    the 256-wide tile's there too."""
    plan_of = getattr(cb.prefill_tail_plan, "__wrapped__", cb.prefill_tail_plan)
    config_of = cb.tail_gemm_config

    def config_256(M, tp, sm_count=SM_COUNT):
        num_stages, block_n, final_ts = config_of(M, tp, sm_count)
        if block_n == cb.TAIL_N192_BLOCK_N:
            return cb.TAIL_NUM_STAGES, cb.BLOCK_N, cb.TAIL_FINAL_TS
        return num_stages, block_n, final_ts

    def plan(M, tp, sm_count=SM_COUNT, num_partials=1):
        with monkeypatch.context() as m:
            m.setattr(cb, "tail_gemm_config", config_256)
            p = dict(plan_of(M, tp, sm_count, num_partials))
        assert p["block_n"] == cb.BLOCK_N
        p.update(
            weights_evict_first=False,
            packed_norm=False,
            num_stages=cb.TAIL_NUM_STAGES,
            final_ts=True,
            **overrides,
        )
        p["gemm_grid"] = (
            cb.resident_grid(p["num_items"], sm_count)
            if p["claim_items"]
            else p["num_items"] * cb.CTA_GROUP
        )
        return p

    monkeypatch.setattr(cb, "prefill_tail_plan", plan)


# Rows whose launches make progress side by side on every qualified part with the registered programs, so a
# concurrent mismatch can only come from shared device state:
#   unfused_split: the TP1 T=256 tail on the non-fused stream-K program (tail_gemm:tp1e0f0t1): its split tiles
#     use the last-arriver fix-up only (no cross-CTA wait); the isolation row of the per-stream partials and the
#     per-runner arrival counters;
#   claimed_112: the TP1 T=256 tail on the claimed program (tail_gemm:tp1e0f1t1ci, cooperative 112-CTA grid): a
#     second grid starts while the first one's last CTAs retire (claim ticket and retirement words);
#   decode_coop: the TP1 T=16 decode tail (cooperative fused norm, grid counter);
#   front_fold: the TP8 T=2048 front on the fold program of the 152-SM and sm_100a 148-SM parts (front:i768e1fold,
#     cooperative 144-CTA grid): its CTAs wait for the carries other CTAs release (per-runner counters, per-stream
#     partials).
_X2_ROWS = ("unfused_split", "claimed_112", "decode_coop", "front_fold")


def _x2_front_fold_runner(device, seed):
    tp, tokens = 8, 2048
    _require_program("front", tp, tokens)
    arch = _gpu_arch()
    sm_count = int(torch.cuda.get_device_properties(0).multi_processor_count)
    if (
        cb.front_fold_plan(cb.m_tiles_for(tokens), i_local_for_tp(tp), sm_count, arch)
        is None
    ):
        pytest.skip(
            f"no fold route on {arch} with {sm_count} SMs (the trailing-wave plan there)"
        )
    args = _front_args(device, tp, tokens, seed)
    runner = prepare_kimi_k3_latent_moe_front(*args)
    assert runner.kernel_keys == ("front:i768e1fold",)
    runner()
    torch.cuda.synchronize()
    x, gate_weight, down_weight, shared_gate_up, logits, latent, shared_act = args
    i_local = SHARED_INTERMEDIATE // tp
    expected = front_reference(
        x,
        dict(
            gate_weight=gate_weight,
            down_weight=down_weight,
            shared_gate_weight=shared_gate_up[:i_local],
            shared_up_weight=shared_gate_up[i_local:],
        ),
    )
    _assert_close("logits", logits, expected["logits"], LOGITS_ATOL, LOGITS_RTOL)
    _assert_close("latent", latent, expected["latent"], ATOL, RTOL)
    _assert_close("shared_act", shared_act, expected["shared_act"], ATOL, RTOL)
    return runner


def _x2_runner(device, monkeypatch, row, seed):
    """A runner of ``row`` (``_X2_ROWS``) on its own inputs, checked once against the torch reference."""
    if row == "front_fold":
        return _x2_front_fold_runner(device, seed)
    tokens = 16 if row == "decode_coop" else 256
    if row == "unfused_split":
        _force_tail_plan(monkeypatch, fused_norm=False, claim_items=False)
    elif row == "claimed_112":
        _force_tail_plan(monkeypatch, fused_norm=True, claim_items=True)
    _require_program("tail", 1, tokens)
    args, kwargs = _tail_args(device, 1, tokens, 1, seed)
    runner = prepare_kimi_k3_latent_moe_tail(*args, **kwargs)
    if row == "unfused_split":
        assert runner.kernel_keys[-1] == "tail_gemm:tp1e0f0t1"
        assert runner.plan["sk_tiles"] > 0 and runner.launches[-1].ws_numel > 0
    elif row == "claimed_112":
        assert runner.kernel_keys == ("tail_gemm:tp1e0f1t1ci",)
        assert runner.plan["gemm_grid"] < runner.sm_count
    runner()
    torch.cuda.synchronize()
    routed, norm_weight, up_weight, shared_act, shared_down_weight, out = args
    expected = tail_reference(
        routed,
        shared_act,
        dict(
            norm_weight=norm_weight,
            up_weight=up_weight,
            shared_down_weight=shared_down_weight,
        ),
        1,
        0,
    )
    _assert_close("y", kwargs["y_workspace"], expected["y"], ATOL, RTOL)
    _assert_close("out", out, expected["out"], ATOL, RTOL)
    return runner


def _x2_pair(device, monkeypatch, row):
    r1 = _x2_runner(device, monkeypatch, row, 11)
    r2 = _x2_runner(device, monkeypatch, row, 12)
    refs = [_snapshot(r.outputs) for r in (r1, r2)]
    assert any(not torch.equal(a, b) for a, b in zip(*refs, strict=True))
    return (r1, r2), refs


def _gate(streams):
    """Hold ``streams`` behind one short device spin, so the launches queued after it start together."""
    gate = torch.cuda.Stream()
    with torch.cuda.stream(gate):
        torch.cuda._sleep(200_000)
    event = torch.cuda.Event()
    event.record(gate)
    for s in streams:
        s.wait_event(event)


@pytest.mark.parametrize(
    "stage, tp, tokens, num_partials",
    (
        ("front", 8, 16, 1),
        ("front", 8, 256, 1),
        ("front", 1, 1024, 1),
        ("front", 8, 2048, 1),
        ("tail", 8, 8, 2),
        ("tail", 1, 16, 1),
        ("tail", 1, 256, 1),
        ("tail", 1, 1024, 1),
        ("tail", 8, 256, 1),
    ),
)
def test_runner_control_words_private(stage, tp, tokens, num_partials):
    """Two identically prepared runners own disjoint, zero control words that every launch returns to zero; a
    dropped runner's words are never handed out again; split plans size their partial workspace from the plan."""
    device = _require_program(stage, tp, tokens, num_partials)
    r1 = _prepare(device, stage, tp, tokens, num_partials)
    r2 = _prepare(device, stage, tp, tokens, num_partials)
    words1, words2 = _control_tensors(r1), _control_tensors(r2)
    assert words1 and len(words1) == len(words2)
    ranges1 = [_byte_range(t) for t in words1]
    for a in ranges1:
        for b in map(_byte_range, words2):
            assert a[1] <= b[0] or b[1] <= a[0], "two runners share control words"
    for r in (r1, r2):
        _assert_words_zero(r, "after prepare")
        assert r.device_index == device.index
        for launch in r.launches:
            if "ws" in launch.kwargs:
                ws_numel = cb._partial_ws_numel(stage, r.plan)
                assert launch.ws_numel == ws_numel
                assert (ws_numel > 0) == (
                    r.plan["sk_tiles"] > 0 or r.plan.get("ws_slots", 0) > 0
                )
                assert launch.kwargs["ws"].data_ptr() == (
                    cb._read_never(device.index)[0].data_ptr()
                )
            else:
                assert launch.ws_numel == 0
    for _ in range(2):
        r1()
        r2()
    torch.cuda.synchronize()
    for r in (r1, r2):
        _assert_words_zero(r, "after launch")
    del r1, words1
    gc.collect()
    r3 = _prepare(device, stage, tp, tokens, num_partials)
    for a in ranges1:
        for b in map(_byte_range, _control_tensors(r3)):
            assert a[1] <= b[0] or b[1] <= a[0], "a dropped runner's words were reused"


def test_no_shared_mutable_scratch():
    """The package holds no mutable device tensor shared across streams: module-level device memory is the
    never-read buffers, the control-word slabs, whose slots belong to one runner (or one functional-call lane)
    each, and one partial workspace per (device, stream)."""
    device = _require_program("tail", 8, 256)
    _require_program("front", 8, 256)
    _require_program("tail", 8, 16)
    runners = [
        _prepare(device, "front", 8, 256),
        _prepare(device, "tail", 8, 256),
        _prepare(device, "tail", 8, 16),
        _prepare(device, "front", 8, 16),
    ]
    for name in ("_SCRATCH", "_TAIL_WS", "_FRONT_WS"):
        assert not hasattr(cb, name)
    for r in runners:
        r()
    torch.cuda.synchronize()
    never = {t.data_ptr() for pair in cb._READ_NEVER.values() for t in pair}
    lanes = {workspace.data_ptr() for _, workspace in cb._WS_LANES.values()}
    assert lanes and len(lanes) == len(cb._WS_LANES)
    chunks = [_byte_range(t) for slab in cb._CTRL_SLABS.values() for t in slab._pinned]

    def in_slab(t):
        lo, hi = _byte_range(t)
        return any(c[0] <= lo and hi <= c[1] for c in chunks)

    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for v in value.values():
                yield from tensors(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                yield from tensors(v)
        elif isinstance(value, cb._CtrlSlab):
            yield from tensors(value._pinned)

    for name, value in vars(cb).items():
        for t in tensors(value):
            if t.is_cuda:
                assert t.data_ptr() in never or in_slab(t) or t.data_ptr() in lanes, (
                    name
                )
    seen = []
    for r in runners:
        for t in _control_tensors(r):
            assert in_slab(t)
            lo, hi = _byte_range(t)
            assert all(hi <= a or b <= lo for a, b in seen)
            seen.append((lo, hi))
        for launch in r.launches:
            for name in ("ws", "tl"):
                if name in launch.kwargs:
                    assert launch.kwargs[name].data_ptr() in never
        if r.stage == "tail" and r.route == "decode":
            assert r.launches[0].kwargs["out_r"].data_ptr() in never


def test_partial_workspace_stream_lanes(monkeypatch):
    """Eager split launches reuse one partial workspace per (device, stream), grown to the largest plan launched
    there and not reallocated once it fits; distinct streams use distinct workspaces; a launch under capture takes
    a graph-pool block and leaves the stream workspaces alone; eager launches beside a replaying graph of another
    runner stay bitwise."""
    device = _require_program("tail", 1, 1024)
    big = _prepare(device, "tail", 1, 1024, seed=31)
    small = _x2_runner(device, monkeypatch, "unfused_split", 11)
    small_ws, big_ws = small.launches[-1].ws_numel, big.launches[-1].ws_numel
    if not 0 < small_ws < big_ws:
        pytest.skip(
            f"needs two split plans of different sizes, got {small_ws}, {big_ws}"
        )
    big()
    torch.cuda.synchronize()
    refs = {id(r): _snapshot(r.outputs) for r in (small, big)}

    def check(what):
        torch.cuda.synchronize()
        for r in (small, big):
            _assert_bitwise(r.outputs, refs[id(r)], f"{what} {r.num_tokens}")
            _assert_words_zero(r, what)

    monkeypatch.setattr(cb, "_WS_LANES", {})
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(s1):
        small()
    first = cb._WS_LANES[(device.index, s1.cuda_stream)]
    assert first[0] == small_ws
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated(device)
    with torch.cuda.stream(s1):
        small()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated(device) == allocated
    assert cb._WS_LANES[(device.index, s1.cuda_stream)][1] is first[1]
    with torch.cuda.stream(s1):
        big()
        grown = cb._WS_LANES[(device.index, s1.cuda_stream)]
        small()
    assert grown[0] == big_ws and grown[1] is not first[1]
    assert cb._WS_LANES[(device.index, s1.cuda_stream)][1] is grown[1]
    del first
    check("one stream")
    with torch.cuda.stream(s2):
        small()
    other = cb._WS_LANES[(device.index, s2.cuda_stream)]
    assert other[1].data_ptr() != grown[1].data_ptr()
    check("second stream")
    lanes = {key: id(lane[1]) for key, lane in cb._WS_LANES.items()}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        big()
    assert {key: id(lane[1]) for key, lane in cb._WS_LANES.items()} == lanes
    for rnd in range(20):
        _nan_fill(small.outputs)
        _nan_fill(big.outputs)
        torch.cuda.synchronize()
        _gate((s1, s2))
        with torch.cuda.stream(s2):
            graph.replay()
        with torch.cuda.stream(s1):
            small()
        check(f"eager beside replay {rnd}")


@pytest.mark.parametrize("row", _X2_ROWS)
def test_concurrent_streams_x2(monkeypatch, row):
    """Two runners of one route with different inputs, launched together on two streams (alternating issue
    order), reproduce their lone results bitwise and leave their control words at zero."""
    device = torch.device("cuda", 0) if torch.cuda.is_available() else None
    if device is None:
        pytest.skip("requires a GPU")
    (r1, r2), refs = _x2_pair(device, monkeypatch, row)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    for rnd in range(40):
        _nan_fill(r1.outputs)
        _nan_fill(r2.outputs)
        torch.cuda.synchronize()
        _gate((s1, s2))
        order = ((s1, r1), (s2, r2)) if rnd % 2 == 0 else ((s2, r2), (s1, r1))
        for stream, runner in order:
            with torch.cuda.stream(stream):
                runner()
        torch.cuda.synchronize()
        _assert_bitwise(r1.outputs, refs[0], f"{row} round {rnd} runner 1")
        _assert_bitwise(r2.outputs, refs[1], f"{row} round {rnd} runner 2")
    for r in (r1, r2):
        _assert_words_zero(r, row)


@pytest.mark.parametrize("row", ("unfused_split", "decode_coop", "front_fold"))
def test_graphs_replayed_concurrently(monkeypatch, row):
    """Graphs capturing two runners (default private pools) replay concurrently on two streams bitwise."""
    device = torch.device("cuda", 0) if torch.cuda.is_available() else None
    if device is None:
        pytest.skip("requires a GPU")
    (r1, r2), refs = _x2_pair(device, monkeypatch, row)
    graphs = [_graph_replay(r) for r in (r1, r2)]
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    for rnd in range(40):
        _nan_fill(r1.outputs)
        _nan_fill(r2.outputs)
        torch.cuda.synchronize()
        _gate((s1, s2))
        for stream, graph in ((s1, graphs[0]), (s2, graphs[1])):
            with torch.cuda.stream(stream):
                graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise(r1.outputs, refs[0], f"{row} replay {rnd} graph 1")
        _assert_bitwise(r2.outputs, refs[1], f"{row} replay {rnd} graph 2")
    for r in (r1, r2):
        _assert_words_zero(r, row)


def test_multistream_capture(monkeypatch):
    """One graph capturing two split-plan runners on forked capture streams replays bitwise."""
    device = torch.device("cuda", 0) if torch.cuda.is_available() else None
    if device is None:
        pytest.skip("requires a GPU")
    (r1, r2), refs = _x2_pair(device, monkeypatch, "unfused_split")
    fork = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        capture = torch.cuda.current_stream()
        fork.wait_stream(capture)
        r1()
        with torch.cuda.stream(fork):
            r2()
        capture.wait_stream(fork)
    for rnd in range(20):
        _nan_fill(r1.outputs)
        _nan_fill(r2.outputs)
        graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise(r1.outputs, refs[0], f"replay {rnd} branch 1")
        _assert_bitwise(r2.outputs, refs[1], f"replay {rnd} branch 2")
    for r in (r1, r2):
        _assert_words_zero(r, "multistream capture")


@pytest.mark.parametrize(
    "stage, tp, tokens, row",
    (
        ("tail", 8, 128, None),
        ("front", 8, 256, None),
        ("tail", 1, 256, "unfused_split"),
    ),
)
def test_cross_stream_first_launch(monkeypatch, stage, tp, tokens, row):
    """A runner prepared on one stream and launched at once on a fresh stream, with no user synchronisation,
    reproduces the reference bitwise (its words are zero and visible to every stream when the slab hands them
    out); a small slab chunk makes most of the runners start a new chunk."""
    device = _require_program(stage, tp, tokens)
    if row is not None:
        _force_tail_plan(monkeypatch, fused_norm=False, claim_items=False)
    reference_runner = _prepare(device, stage, tp, tokens, seed=21)
    reference_runner()
    torch.cuda.synchronize()
    reference = _snapshot(reference_runner.outputs)
    monkeypatch.setattr(cb, "_CTRL_SLABS", {})
    monkeypatch.setattr(cb, "CTRL_CHUNK_WORDS", 4 * cb.CTRL_SLOT_WORDS)
    for i in range(40):
        runner = _prepare(device, stage, tp, tokens, seed=21)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            runner()
        stream.synchronize()
        _assert_bitwise(runner.outputs, reference, f"fresh runner {i}")
        _assert_words_zero(runner, f"fresh runner {i}")


@pytest.mark.parametrize("row", ("unfused_split", "decode_coop"))
def test_functional_two_streams(monkeypatch, row):
    """The one-shot entry point called concurrently on two streams (one control-word lane per stream), and graphs
    capturing it (fresh words per captured call) replayed concurrently, reproduce the lone results bitwise."""
    device = torch.device("cuda", 0) if torch.cuda.is_available() else None
    if device is None:
        pytest.skip("requires a GPU")
    (r1, r2), refs = _x2_pair(device, monkeypatch, row)
    tokens = 16 if row == "decode_coop" else 256
    inputs = [_tail_args(device, 1, tokens, 1, seed) for seed in (11, 12)]
    calls = [(lambda a=a, k=k: kimi_k3_latent_moe_tail(*a, **k)) for a, k in inputs]
    outputs = [(k["y_workspace"], a[5]) for a, k in inputs]
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    lanes_before = set(cb._CTRL_LANES)
    for rnd in range(20):
        for o in outputs:
            _nan_fill(o)
        torch.cuda.synchronize()
        _gate((s1, s2))
        for stream, call in ((s1, calls[0]), (s2, calls[1])):
            with torch.cuda.stream(stream):
                call()
        torch.cuda.synchronize()
        for o, ref in zip(outputs, refs, strict=True):
            _assert_bitwise(o, ref, f"{row} eager round {rnd}")
    # One lane per stream, reused by every call on it.
    new_lanes = set(cb._CTRL_LANES) - lanes_before
    assert len(new_lanes) <= 2
    lanes = [
        key for key in cb._CTRL_LANES if key[1] in (s1.cuda_stream, s2.cuda_stream)
    ]
    assert {key[1] for key in lanes} == {s1.cuda_stream, s2.cuda_stream}
    for key in lanes:
        for words in cb._CTRL_LANES[key]:
            if words is not None:
                assert int(torch.count_nonzero(words)) == 0
    graphs = []
    for call in calls:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            call()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graphs.append(graph)
    for rnd in range(20):
        for o in outputs:
            _nan_fill(o)
        torch.cuda.synchronize()
        _gate((s1, s2))
        for stream, graph in ((s1, graphs[0]), (s2, graphs[1])):
            with torch.cuda.stream(stream):
                graph.replay()
        torch.cuda.synchronize()
        for o, ref in zip(outputs, refs, strict=True):
            _assert_bitwise(o, ref, f"{row} graph round {rnd}")


@pytest.mark.parametrize(
    "stage, tp, tokens", (("tail", 8, 16), ("tail", 1, 256), ("front", 8, 256))
)
def test_functional_first_call_on_fresh_slab(monkeypatch, stage, tp, tokens):
    """A one-shot call that is the process's first control-word request (no slab yet) creates the slab, takes
    its lane and reproduces a prepared runner bitwise (no lock re-entry)."""
    device = torch.device("cuda", 0) if torch.cuda.is_available() else None
    if device is None:
        pytest.skip("requires a GPU")
    if stage == "front":
        args, kwargs = _front_args(device, tp, tokens, 31), {}
        call, prepare = kimi_k3_latent_moe_front, prepare_kimi_k3_latent_moe_front
        outputs = args[4:]
    else:
        args, kwargs = _tail_args(device, tp, tokens, 1, 31)
        call, prepare = kimi_k3_latent_moe_tail, prepare_kimi_k3_latent_moe_tail
        outputs = (kwargs["y_workspace"], args[5])
    runner = prepare(*args, **kwargs)
    runner()
    torch.cuda.synchronize()
    reference = _snapshot(outputs)
    monkeypatch.setattr(cb, "_CTRL_SLABS", {})
    monkeypatch.setattr(cb, "_CTRL_LANES", {})
    _nan_fill(outputs)
    torch.cuda.synchronize()
    errors = []

    def first_call():
        try:
            call(*args, **kwargs)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    worker = threading.Thread(target=first_call, daemon=True)
    worker.start()
    worker.join(timeout=120)
    assert not worker.is_alive(), "the first one-shot call did not return"
    assert not errors, errors
    assert set(cb._CTRL_SLABS) == {0} and len(cb._CTRL_LANES) == 1
    _assert_bitwise(outputs, reference, f"{stage} first one-shot call")


@pytest.mark.parametrize("row", ("decode_coop", "claimed_112", "front_fold"))
def test_cooperative_refusal_in_small_green_context(monkeypatch, row):
    """A cooperative route launched eagerly on a green context with fewer SMs than it needs raises a clean error
    naming the route and its minimum SMs and leaves no CUDA error behind: the next torch op and a full-device
    launch succeed.  Captured inside that green context, the launch fails the capture, with no lasting error
    either."""
    device = torch.device("cuda", 0) if torch.cuda.is_available() else None
    if device is None:
        pytest.skip("requires a GPU")
    try:
        from torch.cuda.green_contexts import GreenContext
    except ImportError:
        pytest.skip("requires torch.cuda.green_contexts")
    runner = _x2_runner(device, monkeypatch, row, 11)
    reference = _snapshot(runner.outputs)
    cooperative = [launch for launch in runner.launches if launch.cooperative_ctas]
    assert cooperative, runner.kernel_keys
    launch = cooperative[0]
    need = launch.min_sms()
    # Every registered cooperative program holds more than half an SM's shared memory: one CTA per SM.
    assert need == launch.cooperative_ctas
    sms = 16
    assert sms < need
    green = GreenContext(num_sms=sms, device_id=0)
    stream = green.Stream()

    def torch_op_ok():
        return float(torch.ones(1 << 20, device=device).sum()) == float(1 << 20)

    _nan_fill(runner.outputs)
    torch.cuda.synchronize()
    message = (
        rf"{re.escape(launch.key)} \(.*\): the cooperative launch of {launch.cooperative_ctas} CTAs "
        rf"needs at least {need} SMs"
    )
    with pytest.raises(RuntimeError, match=message), torch.cuda.stream(stream):
        runner()
    # No stale CUDA error: the next torch launch succeeds at the first attempt.
    assert torch_op_ok()
    torch.cuda.synchronize()
    _assert_words_zero(runner, f"{row} after the refused launch")
    runner()
    torch.cuda.synchronize()
    _assert_bitwise(runner.outputs, reference, f"{row} after the refused launch")
    # Graph form: capturing the launch inside the green context fails the capture.
    graph = torch.cuda.CUDAGraph()
    current = torch.cuda.current_stream()
    try:
        with pytest.raises(RuntimeError), torch.cuda.graph(graph, stream=stream):
            runner()
    finally:
        # torch.cuda.graph leaves its capture stream current when the capture fails
        torch.cuda.set_stream(current)
    del graph
    assert torch_op_ok()
    _nan_fill(runner.outputs)
    runner()
    torch.cuda.synchronize()
    _assert_bitwise(runner.outputs, reference, f"{row} after the refused capture")
    _assert_words_zero(runner, f"{row} after the refused capture")


@pytest.mark.parametrize("exhausted", (False, True))
def test_prepare_under_capture(monkeypatch, exhausted):
    """A runner prepared during CUDA Graph capture replays bitwise: it takes a free slab slot, or, with no slab
    slot left, words zeroed inside the capture."""
    device = torch.device("cuda", 0) if torch.cuda.is_available() else None
    if device is None:
        pytest.skip("requires a GPU")
    (r1, _), refs = _x2_pair(device, monkeypatch, "unfused_split")
    args, kwargs = _tail_args(device, 1, 256, 1, 11)
    outputs = (kwargs["y_workspace"], args[5])
    monkeypatch.setattr(cb, "_CTRL_SLABS", {})
    if not exhausted:
        _prepare(device, "tail", 1, 256, seed=11)  # a fresh chunk with free slots
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner = prepare_kimi_k3_latent_moe_tail(*args, **kwargs)
        runner()
    slab = cb._CTRL_SLABS[device.index]
    words = _control_tensors(runner)
    if exhausted:
        assert slab._chunk is None and len(slab._pinned) == 1
        assert all(t.data_ptr() >= slab._pinned[0].data_ptr() for t in words)
    else:
        assert slab._chunk is not None and len(slab._pinned) == 1
    for rnd in range(3):
        _nan_fill(outputs)
        graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise(outputs, refs[0], f"replay {rnd}")
        _assert_words_zero(runner, f"replay {rnd}")
