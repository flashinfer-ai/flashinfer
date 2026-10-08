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
        return "topk_select"
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
        layout, gqa, *direct = fields
        return backend._long_prefill_route(
            paged=layout == "paged", group_size=int(gqa[3:]), direct_group=bool(direct)
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
        return "decode_uniform_fp8:paged"
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
        batch_size=1,
        total_q=8192,
        q_dtype=torch.bfloat16,
        k_dtype=torch.bfloat16,
        v_dtype=torch.bfloat16,
        causal=True,
        q_offset_is_none=True,
        return_temperature_lse=False,
        lse_temperature_scale=1.0,
    )
    assert backend._use_long_prefill(
        **common, paged=False, group_size=16, max_pages=0, k_outer_dim=8192
    )
    assert backend._use_long_prefill(
        **common, paged=True, group_size=8, max_pages=64, k_outer_dim=64
    )
    assert backend._use_long_prefill(
        **common, paged=True, group_size=16, max_pages=65, k_outer_dim=65
    )
    assert not backend._use_long_prefill(
        **common, paged=True, group_size=16, max_pages=64, k_outer_dim=64
    )
    assert not backend._use_long_prefill(
        **common, paged=False, group_size=8, max_pages=0, k_outer_dim=8192
    )
    assert not backend._use_long_prefill(
        **{**common, "total_q": 8191},
        paged=False,
        group_size=16,
        max_pages=0,
        k_outer_dim=8192,
    )
    assert not backend._use_long_prefill(
        **{**common, "return_temperature_lse": True, "lse_temperature_scale": 0.7},
        paged=False,
        group_size=16,
        max_pages=0,
        k_outer_dim=8192,
    )


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
