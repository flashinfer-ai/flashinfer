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

CPU-only checks of the Blackwell MSA route selection: every registered route
is one the dispatcher derives from its inputs, and the input-derived
predicates select exactly the registered programs.
"""

import threading
import types
from unittest import mock

import pytest
import torch

from flashinfer.jit import blackwell_msa as loader
from flashinfer.msa_ops import _blackwell_sm100 as backend

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
}


def _route_keys(target):
    return sorted(loader.ROUTES[target])


def _derive(route):
    """Re-derive one registered ``<kind>:...`` route from the inputs it encodes."""
    kind, *fields = route.split(":")
    if kind == "topk_select":
        # the 16-query radix select, pre-T9 instance: the k16384 row T2 H8 nvp16381 (total_q 2 < TOPK_Q32_MIN_Q keeps
        # the 16-query program; its slice -- 1024 at S = 16 under the round-6 plan refit, 2048 at S = 8 under the
        # round-4 rule -- is above _TOPK_KEQ_MAX_SLEN either way)
        return backend._topk_route(2, 8, 16381, 16384)
    if kind == "topk_select_keq":
        # the radix select's lever T9 instance: the tied t256 row T256 H4 nvp782 (round-6 plan S = 2 / slice 391:
        # streamed, within (_TOPK_CAND_CAP, _TOPK_KEQ_MAX_SLEN]; its 32-query plan slice 391 < _TOPK_Q32_MIN_SLICE
        # keeps the 16-query program)
        return backend._topk_route(256, 4, 782, 8192)
    if kind == "topk_select_streaming":
        # small latency-bound multi-pass grid (the tied T64 H4 nvp782 row): 256 rows, slice 98 > cap 64
        return backend._topk_route(64, 4, 782, 8192)
    if kind == "topk_select_q32":
        # the 32-query instance (lever T4): the LOSS shape T1024 H8 nvp8192 (32-query plan S = 1, slice 8192)
        return backend._topk_route(1024, 8, 8192, 8192)
    if kind == "prefill_union":
        q_dtype, kv_dtype, layout, gqa, variant = fields
        causal = variant != "any"
        max_pages = 64 if variant == "causal_mask64" else 65 if layout == "paged" else 0
        return backend._prefill_route(
            q_dtype=_DTYPES[q_dtype],
            k_dtype=_DTYPES[kv_dtype],
            paged=layout == "paged",
            folded_gqa_group=int(gqa[3:]),
            causal=causal,
            max_pages=max_pages,
        )
    if kind == "decode_m16" and len(fields) == 4:
        # the L2 evict-first form of the paged BF16 decode program: an
        # under-filled long-context batch on compute capability 10.3 (eight
        # requests of eight queries over four KV heads = 256 work items at
        # 152 SMs, a 465-page table)
        q_dtype, kv_dtype, layout, form = fields
        assert form == "l2ef", route
        return backend._decode_route(
            q_dtype=_DTYPES[q_dtype],
            k_dtype=_DTYPES[kv_dtype],
            paged=layout == "paged",
            l2ef=backend._use_decode_l2ef(
                target="sm103a",
                paged=layout == "paged",
                q_dtype=_DTYPES[q_dtype],
                k_dtype=_DTYPES[kv_dtype],
                items=256,
                num_sms=152,
                max_pages=465,
            ),
        )
    if kind == "decode_m16":
        q_dtype, kv_dtype, layout = fields
        return backend._decode_route(
            q_dtype=_DTYPES[q_dtype], k_dtype=_DTYPES[kv_dtype], paged=layout == "paged"
        )
    if kind == "long_bf16_reverse":
        layout, gqa = fields
        return backend._long_prefill_route(
            paged=layout == "paged", group_size=int(gqa[3:])
        )
    if kind == "kmaj_p":
        layout, gqa = fields
        return backend._kmaj_p_route(paged=layout == "paged", group_size=int(gqa[3:]))
    if kind == "decode_fp8_q1":
        (schedule,) = fields
        paged = schedule == "q1_paged_xform2"
        return "decode_fp8_q1:" + backend._fp8_q1_schedule(
            capturing=False,
            paged=paged,
            force_fused=True,
            causal=True,
            q_offset_is_none=True,
            q_dtype=torch.bfloat16,
            k_dtype=torch.float8_e4m3fn,
            batch_size=128 if paged else 32,
            total_q=128 if paged else 32,
            seqlen_q=1,
            num_q_heads=64,
            num_kv_heads=4,
            k_outer_dim=4096 if paged else 262144,
            max_pages=32 if paged else 0,
        )
    if kind == "decode_uniform_fp8" and fields == ["paged", "fuseq8"]:
        # the fused program: a launch inside the narrow box (64 requests of
        # eight tokens, 32 Q heads over 4 KV heads, contexts up to 1000)
        assert backend._fuseq8_narrow_applies(
            target="sm100a",
            total_q=64 * 8,
            seqlen_q=8,
            num_q_heads=32,
            num_kv_heads=4,
            max_kv_len=1000,
        )
        return backend._FUSEQ8_ROUTE
    if kind == "decode_uniform_fp8":
        # the lean paged program of one launch tier: every registered tier is
        # reached by a launch (items, SMs, page-table width, longest context)
        # the host rule resolves it from -- ``paged:lean:s<splits>:kv<ring>``
        # on compute capability 10.0, the occupancy-2 entry on 10.3
        assert fields[:2] == ["paged", "lean"], route
        if fields[2:] == ["occ2"]:
            launch = dict(
                target="sm103a", items=304, num_sms=152, max_pages=64, max_kv_len=8000
            )
        else:
            splits, ring = int(fields[2][1:]), int(fields[3][2:])
            launch = dict(
                target="sm100a",
                items={1: 100, 2: 64, 4: 32, 8: 16}[splits],
                num_sms=148,
                max_pages=16,
                max_kv_len=1000 if ring == 11 else 20000,
            )
        return backend._lean_route(backend._lean_launch_plan(**launch))
    raise AssertionError(f"unknown route kind {kind!r} in {route}")


@pytest.mark.cpu_only
@pytest.mark.parametrize("target", ["sm100a", "sm103a"])
def test_every_registered_route_is_input_derived(target):
    keys = _route_keys(target)
    assert keys
    for key in keys:
        route, stage = key.rsplit(":", 1)
        assert stage in {"main", "reduce"}, key
        assert _derive(route) == route, key
        assert loader.route_program(key, target) in loader.MODULES


@pytest.mark.cpu_only
def test_fp8_q1_specializations_are_selected_only_on_their_coordinates():
    common = dict(
        capturing=False,
        force_fused=True,
        causal=True,
        q_offset_is_none=True,
        q_dtype=torch.bfloat16,
        k_dtype=torch.float8_e4m3fn,
        seqlen_q=1,
        num_q_heads=64,
        num_kv_heads=4,
    )
    paged = dict(
        common, paged=True, batch_size=128, total_q=128, k_outer_dim=4096, max_pages=32
    )
    flat = dict(
        common, paged=False, batch_size=32, total_q=32, k_outer_dim=262144, max_pages=0
    )
    assert backend._fp8_q1_schedule(**paged) == "q1_paged_xform2"
    assert backend._fp8_q1_schedule(**flat) == "q1_flat_xform2"
    assert backend._fp8_q1_schedule(**{**paged, "max_pages": 33}) == ""
    assert backend._fp8_q1_schedule(**{**flat, "k_outer_dim": 262145}) == ""
    assert backend._fp8_q1_schedule(**{**paged, "capturing": True}) == ""
    assert backend._fp8_q1_schedule(**{**paged, "force_fused": None}) == ""
    assert backend._fp8_q1_schedule(**{**flat, "k_dtype": torch.bfloat16}) == ""


@pytest.mark.cpu_only
def test_long_prefill_predicate_covers_flat_and_paged_boundaries():
    common = dict(
        q_dtype=torch.bfloat16,
        k_dtype=torch.bfloat16,
        v_dtype=torch.bfloat16,
        causal=True,
        q_offset_is_none=True,
        return_temperature_lse=False,
        lse_temperature_scale=1.0,
    )
    flat = dict(paged=False, group_size=16, max_pages=0)
    assert backend._use_long_prefill(**common, **flat, q_lens=(8192,), k_lens=(8192,))
    assert backend._use_long_prefill(
        **common, **flat, q_lens=(8192, 8320), k_lens=(8192, 8320)
    )
    assert backend._use_long_prefill(
        **common,
        paged=True,
        group_size=8,
        max_pages=64,
        q_lens=(8192,),
        k_lens=(64 * 128,),
    )
    assert backend._use_long_prefill(
        **common,
        paged=True,
        group_size=16,
        max_pages=65,
        q_lens=(8192,),
        k_lens=(65 * 128,),
    )
    # Paged K/V qualifies through the page-table width, not the KV lengths.
    assert backend._use_long_prefill(
        **common,
        paged=True,
        group_size=16,
        max_pages=65,
        q_lens=(8192, 8192),
        k_lens=(4096, 8320),
    )
    assert not backend._use_long_prefill(
        **common,
        paged=True,
        group_size=16,
        max_pages=64,
        q_lens=(8192,),
        k_lens=(64 * 128,),
    )
    assert not backend._use_long_prefill(
        **common, paged=False, group_size=8, max_pages=0, q_lens=(8192,), k_lens=(8192,)
    )
    assert not backend._use_long_prefill(
        **common, **flat, q_lens=(8191,), k_lens=(8192,)
    )
    # One short request declines the whole batch; so does one short flat KV.
    assert not backend._use_long_prefill(
        **common, **flat, q_lens=(8192, 4096), k_lens=(8192, 8192)
    )
    assert not backend._use_long_prefill(
        **common, **flat, q_lens=(8192, 8192), k_lens=(8192, 8191)
    )
    assert not backend._use_long_prefill(
        **{**common, "return_temperature_lse": True, "lse_temperature_scale": 0.7},
        **flat,
        q_lens=(8192,),
        k_lens=(8192,),
    )
    with pytest.raises(ValueError):
        backend._use_long_prefill(**common, **flat, q_lens=(8192, 8192), k_lens=(8192,))
    with pytest.raises(ValueError):
        backend._use_long_prefill(**common, **flat, q_lens=(), k_lens=())


@pytest.mark.cpu_only
def test_prefill_route_selects_the_gqa16_paged_mask_family():
    common = dict(
        q_dtype=torch.bfloat16, k_dtype=torch.bfloat16, paged=True, folded_gqa_group=16
    )
    assert backend._prefill_route(**common, causal=True, max_pages=64).endswith(
        ":causal_mask64"
    )
    assert backend._prefill_route(**common, causal=True, max_pages=65).endswith(
        ":causal_large"
    )
    assert backend._prefill_route(**common, causal=False, max_pages=64).endswith(":any")
    assert backend._prefill_route(
        **{**common, "paged": False}, causal=True, max_pages=0
    ).endswith(":any")


# (items, max_pages, max_kv_len) -> (splits, kv_stages, grid) on 148 SMs of compute
# capability 10.0: the kernel's own pre-registered samples (items = batch x query
# tokens x KV heads).
_LEAN_SAMPLES_148 = [
    ((32, 8, 1000), (4, 11, 128)),
    ((32, 32, 4000), (4, 11, 128)),
    ((32, 63, 8000), (4, 11, 128)),
    ((32, 250, 32000), (4, 8, 128)),
    ((64, 10, 1200), (2, 11, 128)),
    ((64, 34, 4294), (2, 11, 128)),
    ((64, 295, 37720), (2, 8, 128)),
    ((128, 8, 1000), (1, 11, 128)),
    ((1024, 128, 16384), (1, 11, 148)),
    ((1024, 129, 16385), (1, 8, 148)),
    ((4096, 39, 4992), (1, 11, 148)),
    ((4, 16, 2048), (8, 11, 32)),
    ((4, 128, 16384), (8, 11, 32)),
    ((4, 2, 256), (1, 11, 4)),
    ((32, 16, 2048), (4, 11, 128)),
    ((32, 1, 128), (1, 11, 32)),
]
# (items, max_pages, max_kv_len) -> the plan (splits, kv_stages, grid, min_blocks)
# on 152 SMs of compute capability 10.3 and on 10.0: the kernel's occupancy-2
# rows (the first branch at kv <= 8192, the long band 8192 < kv <= 16384 with
# items <= 2 x SMs or >= 8 x SMs, the guards at each boundary).
_LEAN_OCC2_ROWS_152 = [
    ((256, 10, 1250), (1, 5, 256, 2), (1, 11, 152, 1)),
    ((1024, 39, 4989), (1, 5, 304, 2), (1, 11, 152, 1)),
    ((256, 37, 4633), (1, 5, 256, 2), (1, 11, 152, 1)),
    ((256, 49, 6185), (1, 5, 256, 2), (1, 11, 152, 1)),
    ((1024, 10, 1247), (1, 5, 304, 2), (1, 11, 152, 1)),
    ((1024, 49, 6166), (1, 5, 304, 2), (1, 11, 152, 1)),
    ((2048, 10, 1250), (1, 5, 304, 2), (1, 11, 152, 1)),
    ((2048, 39, 4983), (1, 5, 304, 2), (1, 11, 152, 1)),
    ((2048, 49, 6205), (1, 5, 304, 2), (1, 11, 152, 1)),
    ((4096, 10, 1248), (1, 5, 304, 2), (1, 11, 152, 1)),
    ((4096, 39, 4992), (1, 5, 304, 2), (1, 11, 152, 1)),
    ((4096, 49, 6223), (1, 5, 304, 2), (1, 11, 152, 1)),
    ((256, 96, 12229), (1, 5, 256, 2), (1, 11, 152, 1)),
    ((1024, 96, 12245), (1, 11, 152, 1), (1, 11, 152, 1)),
    ((4096, 98, 12491), (1, 5, 304, 2), (1, 11, 152, 1)),
    ((2048, 98, 12500), (1, 5, 304, 2), (1, 11, 152, 1)),
    ((32, 32, 4000), (4, 11, 128, 1), (4, 11, 128, 1)),
    ((32, 63, 8000), (4, 11, 128, 1), (4, 11, 128, 1)),
    ((32, 250, 32000), (4, 8, 128, 1), (4, 8, 128, 1)),
    ((128, 35, 4415), (1, 11, 128, 1), (1, 11, 128, 1)),
    ((4096, 53, 6750), (1, 5, 304, 2), (1, 11, 152, 1)),
    ((4096, 63, 8000), (1, 5, 304, 2), (1, 11, 152, 1)),
    ((4096, 71, 9000), (1, 5, 304, 2), (1, 11, 152, 1)),
    ((192, 39, 4992), (1, 5, 192, 2), (1, 11, 152, 1)),
    ((320, 39, 4992), (1, 5, 304, 2), (1, 11, 152, 1)),
    ((1024, 64, 8192), (1, 5, 304, 2), (1, 11, 152, 1)),
    ((2048, 64, 8192), (1, 5, 304, 2), (1, 11, 152, 1)),
    ((4096, 64, 8192), (1, 5, 304, 2), (1, 11, 152, 1)),
    ((1024, 128, 16384), (1, 11, 152, 1), (1, 11, 152, 1)),
    ((2048, 128, 16384), (1, 5, 304, 2), (1, 11, 152, 1)),
    ((4096, 128, 16384), (1, 5, 304, 2), (1, 11, 152, 1)),
    ((4096, 133, 17000), (1, 8, 152, 1), (1, 8, 152, 1)),
    ((1216, 96, 12245), (1, 5, 304, 2), (1, 11, 152, 1)),
]
_LEAN_KEYS = [
    f"decode_uniform_fp8:paged:lean:s{splits}:kv{ring}"
    for ring in (8, 11)
    for splits in (1, 2, 4, 8)
] + ["decode_uniform_fp8:paged:lean:occ2"]


@pytest.mark.cpu_only
def test_lean_constants_mirror_the_kernel_host_rule():
    assert backend._LEAN_ROUTE == "decode_uniform_fp8:paged:lean"
    assert backend._LEAN_TOPK == 16
    assert backend._LEAN_SPLIT_COUNTS == (1, 2, 4, 8)
    assert backend._LEAN_CLUSTER_SPLITS == (4, 8)
    assert (backend._LEAN_KV_STAGES, backend._LEAN_KV_STAGES_SHORT) == (8, 11)
    assert backend._LEAN_KV_TIER_MAX_KV_LEN == 16384
    assert backend._LEAN_OCC2_TARGETS == ("sm103a",)
    assert backend._LEAN_OCC2_MAX_KV_LEN == 8192
    assert backend._LEAN_OCC2_ITEMS_FACTOR == (1, 1)
    assert (backend._LEAN_OCC2_KV_STAGES, backend._LEAN_OCC2_MIN_BLOCKS) == (5, 2)
    assert backend._LEAN_OCC2_LONG_MAX_KV_LEN == 16384
    assert (
        backend._LEAN_OCC2_LONG_WAVE_FACTOR,
        backend._LEAN_OCC2_LONG_ITEMS_FACTOR,
    ) == (2, 8)
    assert backend._LEAN_PARTIAL_ROWS_PER_SPLIT == 128


@pytest.mark.parametrize(
    ("inputs", "expected"),
    _LEAN_SAMPLES_148,
    ids=[f"items{i}-pages{p}-kv{k}" for (i, p, k), _ in _LEAN_SAMPLES_148],
)
@pytest.mark.cpu_only
def test_lean_launch_plan_matches_the_kernel_samples_on_148_sms(inputs, expected):
    items, max_pages, max_kv_len = inputs
    plan = backend._lean_launch_plan(
        target="sm100a",
        items=items,
        num_sms=148,
        max_pages=max_pages,
        max_kv_len=max_kv_len,
    )
    assert (plan.splits, plan.kv_stages, plan.grid) == expected
    assert plan.min_blocks == 1
    route = backend._lean_route(plan)
    assert route == f"decode_uniform_fp8:paged:lean:s{plan.splits}:kv{plan.kv_stages}"
    assert route in _LEAN_KEYS


@pytest.mark.parametrize(
    ("inputs", "routed", "plain"),
    _LEAN_OCC2_ROWS_152,
    ids=[f"items{i}-pages{p}-kv{k}" for (i, p, k), _, _ in _LEAN_OCC2_ROWS_152],
)
@pytest.mark.cpu_only
def test_lean_occupancy_rows_on_152_sms(inputs, routed, plain):
    items, max_pages, max_kv_len = inputs
    launch = dict(items=items, num_sms=152, max_pages=max_pages, max_kv_len=max_kv_len)
    on_103 = backend._lean_launch_plan(target="sm103a", **launch)
    on_100 = backend._lean_launch_plan(target="sm100a", **launch)
    assert tuple(on_103) == routed
    assert tuple(on_100) == plain
    assert on_100.min_blocks == 1, "the occupancy-2 entry is routed on 10.3 only"
    if on_103.min_blocks == 2:
        assert backend._lean_route(on_103) == "decode_uniform_fp8:paged:lean:occ2"
        assert (on_103.splits, on_103.kv_stages) == (1, 5)
        assert on_103.grid == min(items, 2 * 152)
    else:
        assert backend._lean_route(on_103) == backend._lean_route(on_100)


@pytest.mark.cpu_only
def test_lean_split_rule_boundaries():
    def splits(items, max_pages, num_sms=148):
        return backend._lean_splits(items=items, num_sms=num_sms, max_pages=max_pages)

    assert splits(32, 7) == 4, "7 pages are 4 pairs: the floor"
    assert splits(32, 6) == 1, "6 pages are 3 pairs: below the floor"
    assert splits(74, 16) == 2, "74 x 2 = 148 fits"
    assert splits(75, 16) == 1, "75 x 2 > 148"
    assert splits(37, 16) == 4, "37 x 4 = 148 fits, 37 x 8 does not"
    assert splits(18, 15) == 8, "15 pages are 8 pairs, 18 x 8 = 144 fits"
    assert splits(19, 15) == 4, "19 x 8 > 148"
    assert splits(18, 14) == 4, "14 pages are 7 pairs: no eighth pair"
    assert splits(18, 4096) == 8, "the pair count saturates at 16 selected blocks"
    assert splits(19, 15, num_sms=152) == 8, "19 x 8 = 152 fits a 152-SM device"
    assert backend._lean_kv_stages(max_kv_len=1) == 11
    assert backend._lean_kv_stages(max_kv_len=16384) == 11
    assert backend._lean_kv_stages(max_kv_len=16385) == 8
    for kwargs in ({"items": 0}, {"num_sms": 0}, {"max_pages": 0}, {"max_kv_len": 0}):
        launch = dict(
            target="sm100a", items=32, num_sms=148, max_pages=8, max_kv_len=1000
        )
        with pytest.raises(ValueError, match="positive"):
            backend._lean_launch_plan(**{**launch, **kwargs})


@pytest.mark.cpu_only
def test_lean_grid_is_exact_for_the_cluster_tiers_and_persistent_otherwise():
    def plan(target, items, num_sms, max_pages=16, max_kv_len=1000):
        return tuple(
            backend._lean_launch_plan(
                target=target,
                items=items,
                num_sms=num_sms,
                max_pages=max_pages,
                max_kv_len=max_kv_len,
            )
        )

    assert plan("sm100a", 36, 148) == (4, 11, 144, 1)
    assert plan("sm100a", 37, 148) == (4, 11, 148, 1)
    assert plan("sm100a", 38, 148) == (2, 11, 76, 1)
    assert plan("sm100a", 74, 148) == (2, 11, 148, 1)
    assert plan("sm100a", 75, 148) == (1, 11, 75, 1)
    assert plan("sm100a", 4, 148) == (8, 11, 32, 1)
    assert plan("sm100a", 4096, 148) == (1, 11, 148, 1), "persistent: the SM count"
    assert plan("sm103a", 4096, 152) == (1, 5, 304, 2), "two CTAs per SM"
    assert plan("sm103a", 256, 152) == (1, 5, 256, 2)


@pytest.mark.cpu_only
def test_lean_route_keys_and_the_shared_plan_are_registered_on_both_targets():
    for target in ("sm100a", "sm103a"):
        keys = _route_keys(target)
        assert "decode_uniform_fp8:paged:main" not in keys, (
            "the natural program is retired"
        )
        for route in _LEAN_KEYS:
            assert f"{route}:main" in keys, (target, route)
        programs = [
            loader.route_program(f"{route}:main", target) for route in _LEAN_KEYS
        ]
        assert len(set(programs)) == len(_LEAN_KEYS), "one program per tier"
        for program in programs:
            assert loader.MODULES[program]["arg_plan"] == "decode_uniform_fp8"
            assert loader.MODULES[program]["arches"] == ["sm_100a", "sm_103a"]
    names = [name for _kind, name in loader.ARG_PLANS["decode_uniform_fp8"]]
    assert {"K_scale", "V_scale", "partial_O", "partial_M", "partial_D"} <= set(names)
    assert {"split_completion", "total_q", "seqlen_q", "msa_max_pages"} <= set(names)


@pytest.mark.cpu_only
def test_fuseq8_constants_mirror_the_kernel_host_selector():
    assert backend._FUSEQ8_ROUTE == "decode_uniform_fp8:paged:fuseq8"
    assert backend._FUSEQ8_TARGETS == ("sm100a", "sm103a")
    assert (backend._FUSEQ8_NUM_Q_HEADS, backend._FUSEQ8_NUM_KV_HEADS) == (32, 4)
    assert backend._FUSEQ8_SEQLEN_Q == 8
    assert backend._FUSEQ8_MIN_BATCH == 64
    assert backend._FUSEQ8_KV_MAX == 2048


def _fuseq8(target="sm100a", batch=64, seqlen_q=8, hq=32, hkv=4, max_kv_len=1250):
    return backend._fuseq8_narrow_applies(
        target=target,
        total_q=batch * seqlen_q,
        seqlen_q=seqlen_q,
        num_q_heads=hq,
        num_kv_heads=hkv,
        max_kv_len=max_kv_len,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("target", ["sm100a", "sm103a"])
def test_fuseq8_box_takes_the_two_target_rows_and_keeps_the_guards_on_lean(target):
    # the two rows the fused program serves
    assert _fuseq8(target, batch=64, max_kv_len=1250)
    assert _fuseq8(target, batch=128, max_kv_len=1248)
    # the guard rows stay on the lean tiers
    assert not _fuseq8(target, batch=32, max_kv_len=1247)
    assert not _fuseq8(target, batch=32, max_kv_len=4989)
    assert not _fuseq8(target, batch=64, hq=64, max_kv_len=1250)
    # each clause at its boundary
    assert _fuseq8(target, batch=64, max_kv_len=2048)
    assert not _fuseq8(target, batch=64, max_kv_len=2049)
    assert not _fuseq8(target, batch=63)
    assert not _fuseq8(target, seqlen_q=1)
    assert not _fuseq8(target, seqlen_q=16)
    assert not _fuseq8(target, hq=16, hkv=2), "a head group of 8 through other counts"
    assert not _fuseq8(target, hq=32, hkv=8)


@pytest.mark.cpu_only
def test_fuseq8_box_refuses_what_it_cannot_resolve():
    with pytest.raises(ValueError, match="whole number"):
        backend._fuseq8_narrow_applies(
            target="sm100a",
            total_q=65,
            seqlen_q=8,
            num_q_heads=32,
            num_kv_heads=4,
            max_kv_len=1000,
        )
    with pytest.raises(ValueError, match="positive"):
        _fuseq8(max_kv_len=0)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "hq, hkv, seqlen_q",
    [(64, 4, 8), (32, 8, 8), (32, 4, 1), (16, 2, 8)],
    ids=["hq64", "hkv8", "q1", "hq16-hkv2"],
)
def test_fuseq8_box_leaves_other_launches_unchecked(hq, hkv, seqlen_q):
    # outside the head counts and tokens per request of the box the lean tier
    # is unchanged: nothing is checked, so nothing raises
    for kwargs in (
        {"max_kv_len": 0},
        {"total_q": 4 * seqlen_q + 1},
        {"total_q": 0},
        {"target": "sm90a"},
    ):
        launch = dict(
            target="sm100a",
            total_q=64 * seqlen_q,
            seqlen_q=seqlen_q,
            num_q_heads=hq,
            num_kv_heads=hkv,
            max_kv_len=1000,
        )
        launch.update(kwargs)
        assert backend._fuseq8_narrow_applies(**launch) is False, kwargs


@pytest.mark.cpu_only
def test_fuseq8_box_raises_on_the_same_conditions_inside_the_box():
    for kwargs, match in (
        ({"max_kv_len": 0}, "positive"),
        ({"total_q": 4 * 8 + 1}, "whole number"),
        ({"total_q": 0}, "positive"),
    ):
        launch = dict(
            target="sm100a",
            total_q=64 * 8,
            seqlen_q=8,
            num_q_heads=32,
            num_kv_heads=4,
            max_kv_len=1000,
        )
        launch.update(kwargs)
        with pytest.raises(ValueError, match=match):
            backend._fuseq8_narrow_applies(**launch)


@pytest.mark.cpu_only
@pytest.mark.parametrize("target", ["sm107a", "sm90a", "sm120a"])
def test_fuseq8_box_on_other_targets_is_outside_the_box_and_unchecked(target):
    # only sm100a / sm103a take part: on any other target the box shape keeps
    # the lean path, and nothing is checked, so nothing raises
    for kwargs in ({}, {"max_kv_len": 0}, {"total_q": 4 * 8 + 1}, {"total_q": 0}):
        launch = dict(
            target=target,
            total_q=64 * 8,
            seqlen_q=8,
            num_q_heads=32,
            num_kv_heads=4,
            max_kv_len=1250,
        )
        launch.update(kwargs)
        assert backend._fuseq8_narrow_applies(**launch) is False, kwargs
        assert not backend._uniform_fp8_uses_fuseq8(
            workspace=None, capturing=False, **launch
        ), kwargs


# the refusal ``_program`` raises for the dense routes on compute capability
# 10.7, unchanged by the fused program
_SM107A_DENSE_REFUSAL = (
    "the dense SM100/SM103 MSA backend is not qualified on compute "
    "capability 10.7 (Rubin); only the packed-NVFP4 paged-KV routes are "
    "enabled there"
)


def _uniform_fp8_call(target, *, batch, seqlen_q, hq, hkv, kv_lens, max_pages=16):
    """``_run_fp8_direct`` arguments of one eager uniform FP8 decode call without
    a workspace (CPU tensors; the program is never loaded)."""
    total_q = batch * seqlen_q
    cu_k = torch.zeros(batch + 1, dtype=torch.int32)
    return dict(
        route=backend._LEAN_ROUTE,
        target=target,
        q=torch.empty(total_q, hq, backend._HEAD_DIM, dtype=torch.float8_e4m3fn),
        k=torch.empty(4, hkv, 64, backend._HEAD_DIM, dtype=torch.float8_e4m3fn),
        v=torch.empty(4, hkv, 64, backend._HEAD_DIM, dtype=torch.float8_e4m3fn),
        out=torch.empty(total_q, hq, backend._HEAD_DIM, dtype=torch.bfloat16),
        lse=torch.empty(total_q, hq, dtype=torch.float32),
        q2k_indices=torch.zeros(total_q, hkv, 16, dtype=torch.int32),
        cu_k=cu_k,
        q_offsets=cu_k,
        kv_lens=torch.tensor(kv_lens, dtype=torch.int32),
        page_table=torch.zeros(batch, max_pages, dtype=torch.int32),
        paged=True,
        max_pages=max_pages,
        seqlen_q=seqlen_q,
        softmax_scale_log2=1.0,
        output_scale=1.0,
        workspace=None,
        capturing=False,
    )


class _ProgramRequested(Exception):
    pass


def _selected_route(monkeypatch, call, num_sms=148):
    """The route key ``_run_fp8_direct`` requests for ``call``: the program
    lookup records it and stops the launch before any program loads (a
    two-split launch takes its completion counter from a fresh CPU cache)."""
    monkeypatch.setattr(backend, "_num_sms", lambda _device: num_sms)
    monkeypatch.setattr(backend, "_device_index", lambda _device: 0)
    monkeypatch.setattr(backend, "_stream_ptr", lambda _device: 7)
    monkeypatch.setattr(backend, "_lean_completion_counters", {})
    received = []

    def program_must_not_load(route, target):
        received.append((route, target))
        raise _ProgramRequested

    monkeypatch.setattr(backend, "_program", program_must_not_load)
    with pytest.raises(_ProgramRequested):
        backend._run_fp8_direct(**call)
    ((route, target),) = received
    assert target == call["target"]
    return route


@pytest.mark.cpu_only
def test_fuseq8_box_on_sm107a_keeps_the_lean_path_and_its_runtime_error(monkeypatch):
    # an eager call without a workspace whose shape is in the box: on sm107a it
    # takes the lean tier and fails in ``_program`` exactly as before
    kv_lens = [750 + (37 * i) % 501 for i in range(64)]
    assert _fuseq8(target="sm100a", max_kv_len=max(kv_lens))
    call = _uniform_fp8_call(
        "sm107a", batch=64, seqlen_q=8, hq=32, hkv=4, kv_lens=kv_lens
    )
    monkeypatch.setattr(backend, "_num_sms", lambda _device: 148)
    program = backend._program
    received = []

    def recording_program(route, target):
        received.append((route, target))
        return program(route, target)

    monkeypatch.setattr(backend, "_program", recording_program)
    with pytest.raises(RuntimeError) as raised:
        backend._run_fp8_direct(**call)
    assert str(raised.value) == _SM107A_DENSE_REFUSAL
    ((route, target),) = received
    assert target == "sm107a"
    assert route.startswith("decode_uniform_fp8:paged:lean:"), route


@pytest.mark.cpu_only
@pytest.mark.parametrize("target", ["sm100a", "sm103a"])
def test_fuseq8_is_never_selected_under_capture_or_with_a_workspace(target):
    # the narrow-box shape: eager without a workspace runs the fused program;
    # graph capture and the workspace's eager warm-up keep the lean tier, since
    # the fused program has a hard context bound a replay does not re-check
    launch = dict(
        target=target,
        total_q=64 * 8,
        seqlen_q=8,
        num_q_heads=32,
        num_kv_heads=4,
        max_kv_len=1250,
    )
    workspace_stand_in = object()
    assert backend._uniform_fp8_uses_fuseq8(workspace=None, capturing=False, **launch)
    assert not backend._uniform_fp8_uses_fuseq8(
        workspace=None, capturing=True, **launch
    )
    assert not backend._uniform_fp8_uses_fuseq8(
        workspace=workspace_stand_in, capturing=False, **launch
    )
    assert not backend._uniform_fp8_uses_fuseq8(
        workspace=workspace_stand_in, capturing=True, **launch
    )
    # under capture nothing is checked, so an unresolvable launch does not raise
    assert not backend._uniform_fp8_uses_fuseq8(
        workspace=workspace_stand_in, capturing=True, **{**launch, "max_kv_len": 0}
    )


@pytest.mark.cpu_only
def test_fuseq8_grid_is_persistent():
    assert backend._fuseq8_grid(items=64 * 4, num_sms=148) == 148
    assert backend._fuseq8_grid(items=100, num_sms=148) == 100
    assert backend._fuseq8_grid(items=128 * 4, num_sms=152) == 152


@pytest.mark.cpu_only
def test_fuseq8_route_key_is_registered_on_both_targets_with_the_shared_plan():
    for target in ("sm100a", "sm103a"):
        key = "decode_uniform_fp8:paged:fuseq8:main"
        assert key in _route_keys(target), target
        program = loader.route_program(key, target)
        assert loader.MODULES[program]["arg_plan"] == "decode_uniform_fp8"
        assert loader.MODULES[program]["arches"] == ["sm_100a", "sm_103a"]
        lean = {loader.route_program(f"{route}:main", target) for route in _LEAN_KEYS}
        assert program not in lean, "the fused program is its own program"


@pytest.mark.cpu_only
@pytest.mark.parametrize("target", ["sm100a", "sm103a"])
def test_lean_context_bound_is_the_page_table_width_and_no_call_reads_the_device(
    target, monkeypatch
):
    # the tier's context bound is the page-table width x 128 tokens, a host
    # quantity: neither an eager call nor a capturing call, with or without a
    # workspace, reads seqused_k to the host
    assert not hasattr(backend, "_lean_max_kv_len")
    for pages in (1, 9, 16, 83, 160):
        assert backend._lean_kv_len_bound(pages) == pages * backend._BLOCK_SIZE
    for pages in (0, -1):
        with pytest.raises(ValueError, match="max_pages"):
            backend._lean_kv_len_bound(pages)

    def no_device_read(*_args, **_kwargs):
        raise AssertionError("the uniform FP8 route read a tensor to the host")

    for name in ("item", "tolist", "cpu", "max"):
        monkeypatch.setattr(torch.Tensor, name, no_device_read)
    monkeypatch.setattr(backend, "_check_warmed_launch", lambda *_a, **_k: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    kv_lens = [750 + (37 * i) % 501 for i in range(64)]
    lean = backend._lean_launch_plan(
        target=target,
        items=64 * 8 * 4,
        num_sms=148,
        max_pages=16,
        max_kv_len=16 * backend._BLOCK_SIZE,
    )
    for workspace in (None, types.SimpleNamespace(_buffers={})):
        for capturing in (False, True):
            call = _uniform_fp8_call(
                target, batch=64, seqlen_q=8, hq=32, hkv=4, kv_lens=kv_lens
            )
            call.update(workspace=workspace, capturing=capturing)
            want = backend._lean_route(lean)
            if workspace is None and not capturing:
                want = backend._FUSEQ8_ROUTE
            route = _selected_route(monkeypatch, call)
            assert route == f"{want}:main", (workspace, capturing)


@pytest.mark.cpu_only
def test_every_tier_threshold_is_a_multiple_of_the_block_size():
    # the context bound is width x 128 tokens; with every threshold a multiple
    # of 128, a page table of ceil(longest context / 128) pages resolves the
    # tier (and the fused-program box) the longest context resolves
    names = (
        "_LEAN_KV_TIER_MAX_KV_LEN",
        "_LEAN_OCC2_MAX_KV_LEN",
        "_LEAN_OCC2_LONG_MAX_KV_LEN",
        "_FUSEQ8_KV_MAX",
    )
    for name in names:
        assert getattr(backend, name) % backend._BLOCK_SIZE == 0, name
    for target in ("sm100a", "sm103a"):
        for items, num_sms in ((64, 148), (300, 152), (2048, 148)):
            for longest in range(1, 40001):
                pages = -(-longest // backend._BLOCK_SIZE)
                launch = dict(
                    target=target, items=items, num_sms=num_sms, max_pages=pages
                )
                width = backend._lean_kv_len_bound(pages)
                bound = backend._lean_launch_plan(**launch, max_kv_len=width)
                exact = backend._lean_launch_plan(**launch, max_kv_len=longest)
                assert bound == exact, (target, items, longest)
        for longest in range(1, 8193):
            pages = -(-longest // backend._BLOCK_SIZE)
            box = dict(
                target=target,
                total_q=64 * 8,
                seqlen_q=8,
                num_q_heads=32,
                num_kv_heads=4,
            )
            width = backend._lean_kv_len_bound(pages)
            bound = backend._fuseq8_narrow_applies(**box, max_kv_len=width)
            exact = backend._fuseq8_narrow_applies(**box, max_kv_len=longest)
            assert bound == exact, (target, longest)


@pytest.fixture
def lean_counters(monkeypatch):
    """A fresh workspace-free completion-counter cache on CPU tensors, keyed by
    the device index and stream the test sets in the returned launch state."""
    launch = types.SimpleNamespace(index=0, stream=0x1000)
    monkeypatch.setattr(backend, "_num_sms", lambda _device: 148)
    monkeypatch.setattr(backend, "_device_index", lambda _device: launch.index)
    monkeypatch.setattr(backend, "_stream_ptr", lambda _device: launch.stream)
    monkeypatch.setattr(backend, "_lean_completion_counters", {})
    return launch


@pytest.mark.cpu_only
def test_lean_split_pools_are_fp32_partials_at_two_splits_and_aliases_otherwise(
    lean_counters,
):
    lse = torch.zeros(8, 4)
    q2k = torch.zeros(8, 16, dtype=torch.int32)
    cpu = torch.device("cpu")
    pools = backend._lean_split_pools(
        None, items=64, splits=2, lse=lse, q2k_indices=q2k, device=cpu
    )
    assert tuple(pools["partial_O"].shape) == (64 * 2 * 128, 128)
    assert pools["partial_O"].dtype == torch.float32
    assert (
        tuple(pools["partial_M"].shape)
        == (64 * 2 * 128,)
        == tuple(pools["partial_D"].shape)
    )
    assert pools["partial_M"].dtype == pools["partial_D"].dtype == torch.float32
    assert tuple(pools["split_completion"].shape) == (64,)
    assert pools["split_completion"].dtype == torch.int32
    assert int(pools["split_completion"].abs().sum()) == 0, "zeroed without a workspace"
    # with a workspace the counter is zeroed when it is created and only
    # re-bound afterwards: the program resets each item's counter after its merge
    workspace = types.SimpleNamespace(_buffers={})
    with mock.patch.object(
        torch.cuda, "is_current_stream_capturing", return_value=False
    ):
        first = backend._lean_split_pools(
            workspace, items=64, splits=2, lse=lse, q2k_indices=q2k, device=cpu
        )
        assert int(first["split_completion"].abs().sum()) == 0, "zeroed on creation"
        first["split_completion"].fill_(7)
        again = backend._lean_split_pools(
            workspace, items=64, splits=2, lse=lse, q2k_indices=q2k, device=cpu
        )
        assert again["split_completion"] is first["split_completion"]
        assert int(again["split_completion"].sum()) == 7 * 64, "not zeroed again"
        grown = backend._lean_split_pools(
            workspace, items=96, splits=2, lse=lse, q2k_indices=q2k, device=cpu
        )
        assert tuple(grown["split_completion"].shape) == (96,)
        assert int(grown["split_completion"].abs().sum()) == 0, "a new buffer is zeroed"
    for splits in (1, 4, 8):
        alias = backend._lean_split_pools(
            None, items=64, splits=splits, lse=lse, q2k_indices=q2k, device=cpu
        )
        assert alias["partial_O"].data_ptr() == lse.data_ptr()
        assert (
            alias["partial_M"].data_ptr()
            == alias["partial_D"].data_ptr()
            == lse.data_ptr()
        )
        assert alias["split_completion"].data_ptr() == q2k.data_ptr()
    # the aliases and the workspace's counter never enter the eager cache
    assert len(backend._lean_completion_counters) == 1


@pytest.mark.cpu_only
def test_lean_completion_counter_is_zeroed_once_per_device_and_stream(lean_counters):
    # without a workspace the two-split counter is cached per (device, stream):
    # zeroed when it is created, then only re-bound, since the program resets
    # each item's counter after its merge
    lse = torch.zeros(8, 4)
    q2k = torch.zeros(8, 16, dtype=torch.int32)
    cpu = torch.device("cpu")

    def counter(items=64):
        pools = backend._lean_split_pools(
            None, items=items, splits=2, lse=lse, q2k_indices=q2k, device=cpu
        )
        return pools["split_completion"]

    first = counter()
    assert tuple(first.shape) == (64,)
    assert first.dtype == torch.int32
    (storage,) = backend._lean_completion_counters.values()
    assert tuple(storage.shape) == (148 // 2,), "num_sms // 2 entries"
    assert int(storage.abs().sum()) == 0, "zeroed on creation"
    assert first.data_ptr() == storage.data_ptr()
    first.fill_(7)
    again = counter()
    assert again.data_ptr() == first.data_ptr(), "the same key, the same storage"
    assert int(again.sum()) == 7 * 64, "not zeroed again"
    shorter = counter(items=32)
    assert shorter.data_ptr() == first.data_ptr()
    assert tuple(shorter.shape) == (32,)
    assert int(storage[64:].abs().sum()) == 0, "the tail is never written"
    widest = counter(items=148 // 2)
    assert widest.data_ptr() == first.data_ptr()
    with pytest.raises(RuntimeError, match="num_sms // 2"):
        counter(items=148 // 2 + 1)
    assert len(backend._lean_completion_counters) == 1

    # another stream and another device each get their own zeroed storage
    seen = {first.data_ptr()}
    lean_counters.stream = 0x2000
    other_stream = counter()
    lean_counters.index, lean_counters.stream = 1, 0x1000
    other_device = counter()
    for distinct in (other_stream, other_device):
        assert distinct.data_ptr() not in seen
        assert int(distinct.abs().sum()) == 0
        seen.add(distinct.data_ptr())
    lean_counters.index = 0
    assert counter().data_ptr() == first.data_ptr()
    assert len(backend._lean_completion_counters) == 3


@pytest.mark.cpu_only
def test_lean_completion_counter_of_the_per_thread_stream_is_per_thread(
    lean_counters,
):
    # cudaStreamPerThread (handle 2) names another stream in each thread, so
    # each thread gets its own counter; other handles are shared by threads
    lse = torch.zeros(8, 4)
    q2k = torch.zeros(8, 16, dtype=torch.int32)
    cpu = torch.device("cpu")

    def counter_ptr():
        pools = backend._lean_split_pools(
            None, items=64, splits=2, lse=lse, q2k_indices=q2k, device=cpu
        )
        return pools["split_completion"].data_ptr()

    def ptrs_of_two_threads():
        # both threads stay alive until both have their counter: a finished
        # thread's identifier may be reused by the next one
        found = []
        both = threading.Barrier(2)

        def run():
            found.append(counter_ptr())
            both.wait(timeout=60)

        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(found) == 2
        return found

    lean_counters.stream = 2
    here = counter_ptr()
    assert counter_ptr() == here
    per_thread = ptrs_of_two_threads()
    assert len({here, *per_thread}) == 3
    lean_counters.stream = 0x3000
    shared = ptrs_of_two_threads()
    assert shared[0] == shared[1] == counter_ptr()


@pytest.mark.cpu_only
@pytest.mark.parametrize("target", ["sm100a", "sm103a"])
def test_uniform_fp8_route_is_the_lean_tier_or_the_fused_program(target, monkeypatch):
    # no single-token program is registered: every uniform FP8 launch takes the
    # lean tier its launch plan resolves, and the narrow box the fused program
    assert not hasattr(backend, "_UNIFORM_FP8_SINGLE_ROUTE")
    keys = [key for key in _route_keys(target) if key.startswith("decode_uniform_fp8:")]
    lean = [f"{route}:main" for route in _LEAN_KEYS]
    assert keys == sorted([*lean, f"{backend._FUSEQ8_ROUTE}:main"])
    kv_lens = [1000 + i for i in range(4)]
    for seqlen_q in (1, 4, 8, 32):
        call = _uniform_fp8_call(
            target, batch=4, seqlen_q=seqlen_q, hq=32, hkv=4, kv_lens=kv_lens
        )
        plan = backend._lean_launch_plan(
            target=target,
            items=4 * seqlen_q * 4,
            num_sms=148,
            max_pages=16,
            max_kv_len=backend._lean_kv_len_bound(16),
        )
        route = _selected_route(monkeypatch, call)
        assert route == f"{backend._lean_route(plan)}:main", seqlen_q
        assert route in lean
    box = _uniform_fp8_call(
        target,
        batch=64,
        seqlen_q=8,
        hq=32,
        hkv=4,
        kv_lens=[750 + (37 * i) % 501 for i in range(64)],
    )
    assert _selected_route(monkeypatch, box) == f"{backend._FUSEQ8_ROUTE}:main"
    # the tier follows the page-table width, not the contexts: one page more
    # than 2048 tokens leaves the box, and a table wider than 16384 tokens
    # takes the 8-stage ring although every context is about 1000 tokens
    wide_box = dict(
        box, max_pages=17, page_table=torch.zeros(64, 17, dtype=torch.int32)
    )
    wide_route = _selected_route(monkeypatch, wide_box)
    assert wide_route.startswith(f"{backend._LEAN_ROUTE}:")
    wide = _uniform_fp8_call(
        target, batch=4, seqlen_q=1, hq=32, hkv=4, kv_lens=kv_lens, max_pages=160
    )
    plan = backend._lean_launch_plan(
        target=target, items=16, num_sms=148, max_pages=160, max_kv_len=160 * 128
    )
    assert plan.kv_stages == 8
    assert _selected_route(monkeypatch, wide) == f"{backend._lean_route(plan)}:main"


_KMAJ_COMMON = dict(
    q_dtype=torch.bfloat16,
    k_dtype=torch.bfloat16,
    v_dtype=torch.bfloat16,
    causal=True,
    q_offset_is_none=True,
    return_temperature_lse=False,
    lse_temperature_scale=1.0,
)
_KMAJ_PAGED16 = dict(paged=True, group_size=16, max_pages=200)
_KMAJ_FLAT16 = dict(paged=False, group_size=16, max_pages=0)


def _kmaj(q_lens, k_lens, **layout):
    return backend._use_kmaj_p_prefill(
        **_KMAJ_COMMON, **layout, q_lens=q_lens, k_lens=k_lens
    )


@pytest.mark.cpu_only
def test_persistent_route_takes_the_short_request_batches_the_long_route_declines():
    # Every request at or above the long thresholds: the long route, never the
    # persistent one; one request below them hands the whole batch over.
    assert backend._use_long_prefill(
        **_KMAJ_COMMON, **_KMAJ_PAGED16, q_lens=(8192, 8192), k_lens=(25000, 25000)
    )
    assert not _kmaj((8192, 8192), (25000, 25000), **_KMAJ_PAGED16)
    assert not backend._use_long_prefill(
        **_KMAJ_COMMON, **_KMAJ_PAGED16, q_lens=(8192, 100), k_lens=(85362, 11008)
    )
    assert _kmaj((8192, 100), (85362, 11008), **_KMAJ_PAGED16)
    # The length-independent admission is the long route's, plus paged GQA-16
    # batches of exactly 64 pages; below 64 pages, a non-causal batch or FP16
    # operands stay on the union route.
    assert _kmaj((100, 100), (8192, 8192), paged=True, group_size=16, max_pages=64)
    assert not _kmaj((100, 100), (8000, 8000), paged=True, group_size=16, max_pages=63)
    assert _kmaj((100, 100), (8192, 8192), paged=True, group_size=8, max_pages=64)
    assert not backend._use_kmaj_p_prefill(
        **{**_KMAJ_COMMON, "causal": False},
        **_KMAJ_PAGED16,
        q_lens=(100, 100),
        k_lens=(25000, 25000),
    )
    assert not backend._use_kmaj_p_prefill(
        **{**_KMAJ_COMMON, "q_dtype": torch.float16, "k_dtype": torch.float16},
        **_KMAJ_PAGED16,
        q_lens=(100, 100),
        k_lens=(25000, 25000),
    )
    with pytest.raises(ValueError):
        _kmaj((100, 100), (25000,), **_KMAJ_PAGED16)


@pytest.mark.cpu_only
def test_persistent_route_shape_guards():
    # Flat KV floor: the longest request spans more than 64 KV blocks.
    assert not _kmaj((1024,), (64 * 128,), **_KMAJ_FLAT16)
    assert _kmaj((1024,), (64 * 128 + 1,), **_KMAJ_FLAT16)
    assert _kmaj((1024, 1500), (4096, 8320), **_KMAJ_FLAT16)
    # Longest-request box: a request of 14336+ queries stays on the union route
    # only when the batch's shortest KV is at most 12288 tokens; 14208 queries
    # pass, and long requests over longer KV take the persistent route.
    assert _kmaj((114, 14208), (11008, 85362), **_KMAJ_PAGED16)
    assert not _kmaj((114, 14336), (11008, 85362), **_KMAJ_PAGED16)
    assert not _kmaj((114, 15318), (11008, 85362), **_KMAJ_PAGED16)
    assert not _kmaj((114, 14336), (85362, 12288), **_KMAJ_PAGED16)
    assert _kmaj((114, 14336), (85362, 12289), **_KMAJ_PAGED16)
    assert _kmaj((186, 16128), (26112, 50874), **_KMAJ_PAGED16)
    assert _kmaj((574, 15744), (25728, 26942), **_KMAJ_PAGED16)
    # No KV/request box: short requests over short KV take the persistent route.
    assert _kmaj((2548, 4168), (14792, 20084), **_KMAJ_PAGED16)
    assert _kmaj((2048,), (20480,), **_KMAJ_PAGED16)
    assert _kmaj((182, 1623), (17718, 18007), **_KMAJ_PAGED16)
    assert _kmaj((2048,), (8192,), paged=True, group_size=8, max_pages=64)
    assert _kmaj(
        (2048, 2048, 2048, 2048), (8192,) * 4, paged=True, group_size=16, max_pages=64
    )
    assert backend._kmaj_p_admits(q_lens=(100,), k_lens=(8192,), paged=True)
    assert not backend._kmaj_p_admits(q_lens=(100,), k_lens=(8192,), paged=False)


@pytest.mark.cpu_only
def test_persistent_route_keys_and_plans_are_registered():
    # The dispatcher requests these keys for every admitted batch; the plans
    # are the producer's and the BF16 combine's argument orders.
    for target in ("sm100a", "sm103a"):
        keys = _route_keys(target)
        for layout, group in (("flat", 16), ("paged", 16), ("paged", 8)):
            route = backend._kmaj_p_route(paged=layout == "paged", group_size=group)
            assert route == f"kmaj_p:{layout}:gqa{group}"
            for stage in ("main", "reduce"):
                assert f"{route}:{stage}" in keys, (target, route, stage)
    assert {"kmaj_p_main", "kmaj_p_reduce"} <= set(loader.ARG_PLANS)
    main = [name for _kind, name in loader.ARG_PLANS["kmaj_p_main"]]
    assert "partial_scale" not in main and "num_work_items" in main
    reduce = [name for _kind, name in loader.ARG_PLANS["kmaj_p_reduce"]]
    assert "partial_scale" not in reduce and "split_counts" in reduce


def _l2ef(batch, seqlen_q, num_kv_heads, kv_len, **over):
    # items = batch x seqlen_q x KV heads (the persistent work items of the M16
    # decode); pages = ceil(kv / 128), the width of a tight page table
    inputs = dict(
        target="sm103a",
        paged=True,
        q_dtype=torch.bfloat16,
        k_dtype=torch.bfloat16,
        items=batch * seqlen_q * num_kv_heads,
        num_sms=152,
        max_pages=-(-kv_len // backend._BLOCK_SIZE),
    )
    return backend._use_decode_l2ef(**{**inputs, **over})


def _l2ef_100(batch, seqlen_q, num_kv_heads, kv_len, **over):
    # the same launch judged for compute capability 10.0 (148 SMs)
    over = {"target": "sm100a", "num_sms": 148, **over}
    return _l2ef(batch, seqlen_q, num_kv_heads, kv_len, **over)


@pytest.mark.cpu_only
def test_decode_l2ef_constants_and_truth_table():
    assert backend._DECODE_L2EF_TARGETS == ("sm100a", "sm103a")
    assert backend._DECODE_L2EF_ITEMS_FACTOR == (2, 1)
    assert backend._DECODE_L2EF_MIN_KV_LEN == 40960
    assert backend._BLOCK_SIZE == 128
    assert backend._DECODE_L2EF_MIN_PAGES == 320
    # (batch, seqlen_q, KV heads, longest context): the decode rows the hinted
    # program was measured on; at 152 SMs (compute capability 10.3) and at 148
    # (10.0) alike it takes the under-filled rows whose longest context reaches
    # 40960 tokens (three of nineteen) and no other
    universe = [
        *((8, 8, 4, kv) for kv in (1250, 4633, 6185, 12229, 59482, 119999, 243998)),
        *((1, 8, 4, kv) for kv in (1000, 4000, 8000, 32000)),
        *((2, 8, 4, kv) for kv in (1200, 4294, 8727, 37720)),
        *((4, 8, 4, kv) for kv in (1234, 4415, 9550, 37800)),
    ]
    for batch, seqlen_q, kv_heads, kv in universe:
        items = batch * seqlen_q * kv_heads
        expected = items <= 2 * 152 and -(-kv // 128) >= 320
        assert _l2ef(batch, seqlen_q, kv_heads, kv) is expected, (batch, kv)
        expected_100 = items <= 2 * 148 and -(-kv // 128) >= 320
        assert _l2ef_100(batch, seqlen_q, kv_heads, kv) is expected_100, (batch, kv)
    assert sum(_l2ef(*row) for row in universe) == 3
    assert sum(_l2ef_100(*row) for row in universe) == 3
    # batches deeper than two waves never take it, whatever the context
    for batch, seqlen_q, kv_heads, kv in (
        (16, 8, 4, 243998),
        (32, 8, 4, 244115),
        (32, 8, 4, 12245),
        (64, 8, 4, 249393),
        (128, 8, 4, 249728),
        (32, 8, 4, 8192),
        (64, 8, 4, 16384),
        (128, 8, 4, 65536),
        (32, 8, 4, 65536),
        (64, 8, 4, 65536),
    ):
        assert _l2ef(batch, seqlen_q, kv_heads, kv) is False, (batch, kv)


@pytest.mark.cpu_only
def test_decode_l2ef_items_and_pages_boundaries():
    # items at twice the SM count (inclusive) ...
    assert _l2ef(1, 1, 304, 200000) is True
    assert _l2ef(1, 1, 305, 200000) is False
    assert _l2ef(8, 8, 4, 200000) is True  # 256 items: the measured row
    assert _l2ef(16, 8, 4, 200000) is False  # 512 items
    assert _l2ef_100(1, 1, 296, 200000) is True  # 148 SMs
    assert _l2ef_100(1, 1, 297, 200000) is False
    # ... and the page-table width at the 40960-token floor (320 pages)
    assert _l2ef(8, 8, 4, 40960) is True
    assert _l2ef(8, 8, 4, 40833) is True  # 320 pages, the last one partial
    assert _l2ef(8, 8, 4, 40832) is False  # 319 pages
    assert _l2ef(8, 8, 4, 0) is False
    assert _l2ef(8, 8, 4, 200000, max_pages=320) is True
    assert _l2ef(8, 8, 4, 200000, max_pages=319) is False


@pytest.mark.cpu_only
def test_decode_l2ef_target_layout_and_dtype_exclude():
    assert _l2ef(8, 8, 4, 200000) is True
    assert _l2ef_100(8, 8, 4, 200000) is True
    assert _l2ef(8, 8, 4, 200000, target="sm107a") is False
    assert _l2ef(8, 8, 4, 200000, paged=False) is False
    assert _l2ef(8, 8, 4, 200000, k_dtype=torch.float8_e4m3fn) is False
    assert _l2ef(8, 8, 4, 200000, q_dtype=torch.float16, k_dtype=torch.float16) is False
    for bad in ({"items": 0}, {"items": -8}, {"num_sms": 0}, {"max_pages": -1}):
        with pytest.raises(ValueError):
            _l2ef(8, 8, 4, 200000, **bad)


@pytest.mark.cpu_only
def test_decode_l2ef_key_and_program_are_registered_on_both_targets():
    bf16 = dict(q_dtype=torch.bfloat16, k_dtype=torch.bfloat16, paged=True)
    plain = backend._decode_route(**bf16)
    hinted = backend._decode_route(**bf16, l2ef=True)
    assert plain == "decode_m16:bfloat16:bfloat16:paged"
    assert hinted == "decode_m16:bfloat16:bfloat16:paged:l2ef"
    assert backend._decode_route(**bf16, l2ef=False) == plain
    for target in ("sm100a", "sm103a"):
        keys = _route_keys(target)
        assert f"{plain}:main" in keys and f"{hinted}:main" in keys
        unhinted = loader.route_program(f"{plain}:main", target)
        program = loader.route_program(f"{hinted}:main", target)
        # a program of its own, with the paged program's argument plan, built
        # for both targets and selected on both by the predicate alone
        assert program != unhinted
        assert loader.MODULES[program]["arg_plan"] == "decode_m16"
        assert loader.MODULES[unhinted]["arg_plan"] == "decode_m16"
        assert loader.MODULES[program]["arches"] == ["sm_100a", "sm_103a"]


@pytest.mark.cpu_only
def test_uniform_fp8_grid_is_the_lean_launch_plan_grid():
    # The single-token program took an even 128-CTA wave when it matched the
    # default wave count (seqlen_q >= 4).  The lean tiers have no even-wave rule
    # and no seqlen_q input: one and two splits run the persistent grid
    # min(items x splits, SMs x CTAs per SM), four and eight splits exactly
    # items x splits.  The inputs below are the former test's.
    def grid(target, items, num_sms=148, max_pages=16):
        return backend._lean_launch_plan(
            target=target,
            items=items,
            num_sms=num_sms,
            max_pages=max_pages,
            max_kv_len=1000,
        ).grid

    assert grid("sm100a", 256) == 148, "the persistent wave, not an even 128"
    assert grid("sm100a", 3) == 24, "eight splits, one cluster per item"
    assert grid("sm100a", 3, max_pages=6) == 3, "under four page pairs: one split"
    assert grid("sm103a", 256, num_sms=152) == 256, "two CTAs per SM"
