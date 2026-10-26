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
    assert cb.front_evict_first(4) and not cb.front_evict_first(6)


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
COVERAGE_PARTIALS = (1, 2, 4)


@pytest.mark.parametrize("sm_count", QUALIFIED_SM_COUNTS)
def test_route_keys_are_registered_for_every_token_count(sm_count):
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
                keys = route_kernel_keys(stage, tp, tokens, sm_count)
                assert 1 <= len(keys) <= 2
                for key in keys:
                    assert key in KERNELS, (
                        f"{stage} tp{tp} T={tokens} on {sm_count} SMs: {key} is not registered"
                    )
                    reached.add(KERNELS[key])
                for arch in arches:
                    assert cake_jit.route_available(arch, keys)
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
    # Every registered program is reachable, serves every registered architecture once and carries
    # a complete launch contract.
    assert reached == set(MODULES)
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
            assert route_kernel_keys("front", tp, tokens, 148) == route_kernel_keys(
                "front", tp, tokens, 152
            )
            same_tail = route_kernel_keys("tail", tp, tokens, 148) == route_kernel_keys(
                "tail", tp, tokens, 152
            )
            # The 256 x 512 tail tile starts at 4096 TP1 tokens on 152 SMs, at 8192 on 148.
            assert same_tail == (tp != 1 or not 4096 <= tokens < 8192)


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
    at 4096 tokens on 152 SMs and at 8192 on 148."""
    assert cb.tail_wide_min_tokens(152) == 4096 and cb.tail_wide_min_tokens(148) == 8192
    for sm_count in QUALIFIED_SM_COUNTS:
        wide = cb.tail_wide_min_tokens(sm_count)
        for tokens in (129, 256, 300, 512, 513, 1024, 2048, 4095, 4096, 8192, 16384):
            plan = prefill_tail_plan(tokens, 1, sm_count)
            claimed = tokens > 512
            assert plan["fused_norm"] and plan["claim_items"] == claimed
            assert plan["block_n"] == (512 if tokens >= wide else 256)
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
    y = rmsnorm(acc.to(torch.bfloat16), w["norm_weight"])
    k_up = LATENT // tp
    cols = slice(rank * k_up, (rank + 1) * k_up)
    up = torch.nn.functional.linear(y[:, cols].float(), w["up_weight"][:, cols].float())
    shared = torch.nn.functional.linear(
        shared_act.float(), w["shared_down_weight"].float()
    )
    return dict(y=y, out=(up + shared).to(torch.bfloat16))


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
    assert runner.kernel_keys == route_kernel_keys("front", tp, tokens)
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
    # Decode route: the fused norm reproduces the reference rounding exactly (FP32 statistics, BF16
    # round, BF16 weight product).  Prefill route: the one-pass RMSNorm kernel reduces the row in a
    # different FP32 order and lands within one BF16 ulp on a few elements (the Cake contract
    # receipts record the same for the production kernel), so it is held to the BF16 tolerance.
    # Both routes are bit-identical across re-launch and CUDA-graph replay below.
    _assert_close("y", y, expected["y"], ATOL, RTOL)
    if tokens <= DECODE_MAX_T:
        assert torch.equal(y, expected["y"]), (
            f"y differs from the reference in {int((y != expected['y']).sum())} elements"
        )
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
    assert torch.equal(y, expected["y"])
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


@pytest.mark.parametrize("tokens", (3, 8, 16, 64))
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
    assert torch.equal(y, expected["y"]), (
        f"y differs from the reference in {int((y != expected['y']).sum())} elements"
    )
    _assert_close("out", out, expected["out"], ATOL, RTOL)
    first = (y.clone(), out.clone())
    runner()
    torch.cuda.synchronize()
    assert torch.equal(first[0], y) and torch.equal(first[1], out)


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
# Device scratch ownership and concurrency (per-runner control words, per-launch partials)
# ---------------------------------------------------------------------------


def test_partial_workspace_and_control_word_sizes():
    """Split prefill plans take one ``128 x block_n`` FP32 slot per contributor CTA per launch (whole plans none) and
    every runner's control words fit one KiB in 128-byte slots."""
    for sm_count in QUALIFIED_SM_COUNTS:
        for tp in SUPPORTED_TP:
            for tokens in sorted(set(cb.ROW_TOKENS) | {129, 300, 511, 1000, 3000}):
                if tokens <= DECODE_MAX_T:
                    continue
                tail = prefill_tail_plan(tokens, tp, sm_count)
                front = prefill_front_plan(tokens, i_local_for_tp(tp), sm_count)
                for stage, plan, block_n, n_u32 in (
                    ("tail", tail, tail["block_n"], cb.NORM_COUNTER_WORDS),
                    ("front", front, cb.FRONT_BLOCK_N, 0),
                ):
                    slots = plan["sk_tiles"] * plan["sk_max_seg"] * cb.CTA_GROUP
                    assert cb._partial_ws_numel(stage, plan) == (
                        slots * cb.BLOCK_M * block_n
                    )
                    words = cb._arrival_counter_words(plan)
                    assert words == max(2, plan["sk_tiles"] * cb.CTA_GROUP)
                    u32_off, slot_words = cb._ctrl_layout(words, n_u32)
                    assert u32_off >= words and u32_off % cb.CTRL_SLOT_WORDS == 0
                    assert slot_words % cb.CTRL_SLOT_WORDS == 0
                    assert slot_words >= u32_off + n_u32 and 4 * slot_words <= 1024
    assert cb._ctrl_layout(0, cb.DECODE_COUNTER_WORDS) == (0, cb.CTRL_SLOT_WORDS)
    # A plan that carries partials between its items states them the same way: the front lane form's sk_tiles
    # clusters of sk_max_seg 1 take one carry slot and one carry counter per CTA.
    lane = dict(sk_tiles=72, sk_max_seg=1)
    assert cb._partial_ws_numel("front", lane) == 144 * cb.BLOCK_M * cb.FRONT_BLOCK_N
    assert cb._arrival_counter_words(lane) == 144


def test_ws_programs_never_trigger_dependents():
    """A program bound to a per-launch partial workspace (``('buffer', 'ws')`` in its argument plan) must not
    trigger ``griddepcontrol.launch_dependents``: a dependent launched early could be handed the recycled block
    while the program still writes it."""
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
    """Route the TP1 T=256 tail to another registered program of the same tile and stream-K plan."""
    plan_of = cb.prefill_tail_plan

    def plan(M, tp, sm_count=SM_COUNT, num_partials=1):
        p = plan_of(M, tp, sm_count, num_partials)
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
#     use the last-arriver fix-up only (no cross-CTA wait); the isolation row of the per-launch partials and the
#     per-runner arrival counters;
#   claimed_112: the TP1 T=256 tail on the claimed program (tail_gemm:tp1e0f1t1ci, cooperative 112-CTA grid): a
#     second grid starts while the first one's last CTAs retire (claim ticket and retirement words);
#   decode_coop: the TP1 T=16 decode tail (cooperative fused norm, grid counter).
_X2_ROWS = ("unfused_split", "claimed_112", "decode_coop")


def _x2_runner(device, monkeypatch, row, seed):
    """A runner of ``row`` (``_X2_ROWS``) on its own inputs, checked once against the torch reference."""
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
        ("tail", 8, 8, 2),
        ("tail", 1, 16, 1),
        ("tail", 1, 256, 1),
        ("tail", 1, 1024, 1),
        ("tail", 8, 256, 1),
    ),
)
def test_runner_control_words_private(stage, tp, tokens, num_partials):
    """Two identically prepared runners own disjoint, zero control words that every launch returns to zero; a
    dropped runner's words are never handed out again; split plans size their per-launch partials from the plan."""
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
                assert (ws_numel > 0) == (r.plan["sk_tiles"] > 0)
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
    """The package holds no mutable device tensor shared by runners: module-level device memory is the never-read
    buffers and the control-word slabs, whose slots belong to one runner (or one functional-call lane) each."""
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
    never = {t.data_ptr() for pair in cb._READ_NEVER.values() for t in pair}
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
                assert t.data_ptr() in never or in_slab(t), name
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


@pytest.mark.parametrize("row", ("unfused_split", "decode_coop"))
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


@pytest.mark.parametrize("row", ("decode_coop", "claimed_112"))
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
