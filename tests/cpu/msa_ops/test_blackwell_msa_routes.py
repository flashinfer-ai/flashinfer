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
    if kind == "decode_uniform_fp8":
        # the single-token program serves every seqlen_q (no other uniform-FP8 route is registered)
        return backend._UNIFORM_FP8_SINGLE_ROUTE
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


@pytest.mark.cpu_only
def test_uniform_fp8_route_is_the_single_token_program():
    # every seqlen_q takes the single-token program; no other uniform-FP8 route is registered
    single = backend._UNIFORM_FP8_SINGLE_ROUTE
    assert single == "decode_uniform_fp8:paged"
    for target in ("sm100a", "sm103a"):
        keys = _route_keys(target)
        assert [key for key in keys if key.startswith("decode_uniform_fp8:")] == [
            f"{single}:main"
        ]


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
def test_uniform_fp8_grid_uses_the_full_even_wave_when_available():
    assert (
        backend._uniform_fp8_decode_grid(total_work_items=256, num_sms=148, seqlen_q=4)
        == 128
    )
    assert (
        backend._uniform_fp8_decode_grid(total_work_items=256, num_sms=148, seqlen_q=1)
        == 148
    )
    assert (
        backend._uniform_fp8_decode_grid(total_work_items=3, num_sms=148, seqlen_q=4)
        == 3
    )
