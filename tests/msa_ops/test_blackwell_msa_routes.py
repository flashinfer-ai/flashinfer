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
        # the single-token program serves every seqlen_q while lever D4b(dynpair) ships with its route off;
        # a registered ``paged_dynpair`` route must be selectable by the mirror (even seqlen_q)
        return backend._uniform_fp8_route(8 if fields == ["paged_dynpair"] else 1)
    raise AssertionError(f"unknown route kind {kind!r} in {route}")


@pytest.mark.parametrize("target", ["sm100a", "sm103a"])
def test_every_registered_route_is_input_derived(target):
    keys = _route_keys(target)
    assert keys
    for key in keys:
        route, stage = key.rsplit(":", 1)
        assert stage in {"main", "reduce"}, key
        assert _derive(route) == route, key
        assert loader.route_program(key, target) in loader.MODULES


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


def test_uniform_fp8_route_is_the_single_token_program_unless_dynpair_is_opted_in():
    # Lever D4b(dynpair) ships with its route off by default: every
    # seqlen_q takes the single-token program and only that route is registered.  The
    # mirror switch keeps the dispatcher's predicate (even seqlen_q in [2, 32]); turning
    # it on needs a re-export that registers the ``paged_dynpair`` route again.
    assert backend._UNIFORM_FP8_DYNPAIR_ROUTE is False
    single = "decode_uniform_fp8:paged"
    for seqlen_q in range(1, 35):
        assert backend._uniform_fp8_route(seqlen_q) == single, seqlen_q
    with mock.patch.object(backend, "_UNIFORM_FP8_DYNPAIR_ROUTE", True):
        for seqlen_q in range(1, 33):
            expected = (
                "decode_uniform_fp8:paged_dynpair"
                if seqlen_q % 2 == 0
                else "decode_uniform_fp8:paged"
            )
            assert backend._uniform_fp8_route(seqlen_q) == expected, seqlen_q
        assert backend._uniform_fp8_route(34) == "decode_uniform_fp8:paged"
    for target in ("sm100a", "sm103a"):
        assert "decode_uniform_fp8:paged:main" in _route_keys(target)
        assert "decode_uniform_fp8:paged_dynpair:main" not in _route_keys(target)


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
