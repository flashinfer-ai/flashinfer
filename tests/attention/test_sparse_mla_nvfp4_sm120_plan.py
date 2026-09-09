# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure-Python tests for the independently calibrated NVFP4 planner."""

from __future__ import annotations

import pytest
import torch

from flashinfer.mla import _sparse_mla_nvfp4_sm120_plan as plan_mod
from flashinfer.mla import _sparse_mla_sm120_cpb as cpb_mod


@pytest.fixture
def planner_state(monkeypatch):
    crossover: dict[str, int] = {}
    cpb: dict[str, int] = {}
    monkeypatch.setattr(cpb_mod, "_device_key", lambda _device: "0:Fake SM120")
    monkeypatch.setattr(cpb_mod, "_maybe_load_disk", lambda: None)
    monkeypatch.setattr(
        cpb_mod,
        "get_decode_max_tokens",
        lambda _device, family, heads, topk: crossover.get(f"{family}|{heads}|{topk}"),
    )
    monkeypatch.setattr(
        cpb_mod,
        "get_cpb_override",
        lambda _device, family, heads, topk, tokens: cpb.get(
            f"{family}|{heads}|{topk}|{tokens}"
        ),
    )
    # Unit tests inject calibration records directly; GPU timing is covered by
    # the SM120 integration/benchmark path.
    monkeypatch.setattr(plan_mod, "_maybe_calibrate", lambda **_kwargs: None)
    plan_mod._plan_memo.clear()
    yield crossover, cpb
    plan_mod._plan_memo.clear()


def _family(**kwargs) -> str:
    defaults = dict(
        primary_page_size=64,
        extra_topk=0,
        extra_page_size=0,
        has_topk_length=True,
        has_extra_topk_length=False,
        has_attn_sink=False,
    )
    defaults.update(kwargs)
    return plan_mod._family_key(**defaults)


def _plan(tokens: int, **kwargs):
    defaults = dict(
        num_tokens=tokens,
        num_heads=64,
        topk=128,
        primary_page_size=64,
        device=torch.device("cpu"),
        has_topk_length=True,
    )
    defaults.update(kwargs)
    return plan_mod.plan_nvfp4_sparse_mla_sm120(**defaults)


def _phase_key(family: str, tokens: int, *, heads: int = 64, topk: int = 128) -> str:
    return f"{plan_mod._phase_family(family, tokens)}|{heads}|{topk}"


def test_nvfp4_unknown_crossover_keeps_safe_decode_fallback(planner_state) -> None:
    planned = _plan(32)
    assert planned is not None
    assert planned.variant is plan_mod.NVFP4KernelVariant.DECODE_SPLITK
    assert planned.cpb == 0


def test_nvfp4_calibrated_crossover_replaces_fixed_64_policy(planner_state) -> None:
    crossover, cpb = planner_state
    family = _family()
    crossover[_phase_key(family, 8)] = 8
    crossover[_phase_key(family, 16)] = 0
    cpb[f"{family}|64|128|8"] = 2
    plan_mod._plan_memo.clear()

    at_crossover = _plan(8)
    above_crossover = _plan(9)
    assert at_crossover is not None
    assert at_crossover.variant is plan_mod.NVFP4KernelVariant.DECODE_SPLITK
    assert at_crossover.cpb == 2
    assert above_crossover is not None
    assert above_crossover.variant is plan_mod.NVFP4KernelVariant.PREFILL_STREAMING


def test_nvfp4_decode_cpb_uses_calibrated_token_bucket(planner_state) -> None:
    crossover, cpb = planner_state
    family = _family()
    crossover[_phase_key(family, 16)] = 16
    cpb[f"{family}|64|128|16"] = 3
    plan_mod._plan_memo.clear()

    planned = _plan(9)
    assert planned is not None
    assert planned.variant is plan_mod.NVFP4KernelVariant.DECODE_SPLITK
    assert planned.cpb == 3


def test_nvfp4_64_is_only_decode_envelope(planner_state) -> None:
    crossover, _ = planner_state
    family = _family()
    crossover[_phase_key(family, 64)] = 64
    plan_mod._plan_memo.clear()

    assert _plan(64).variant is plan_mod.NVFP4KernelVariant.DECODE_SPLITK
    assert _plan(65).variant is plan_mod.NVFP4KernelVariant.PREFILL_STREAMING
    assert _plan(8192).variant is plan_mod.NVFP4KernelVariant.PREFILL_STREAMING


def test_nvfp4_phase_table_preserves_non_monotonic_wave_boundaries(
    planner_state,
) -> None:
    crossover, cpb = planner_state
    family = _family()
    crossover[_phase_key(family, 48)] = 0
    crossover[_phase_key(family, 64)] = 64
    cpb[f"{family}|64|128|64"] = 7
    plan_mod._plan_memo.clear()

    at_48 = _plan(48)
    at_64 = _plan(64)
    assert at_48.variant is plan_mod.NVFP4KernelVariant.PREFILL_STREAMING
    assert at_64.variant is plan_mod.NVFP4KernelVariant.DECODE_SPLITK
    assert at_64.cpb == 7
    assert plan_mod._monotonic_decode_max_tokens({48: False, 64: True}) is None


def test_nvfp4_calibration_namespace_is_format_and_shape_specific() -> None:
    main = _family()
    dual_p2 = _family(
        extra_topk=512,
        extra_page_size=2,
        has_extra_topk_length=True,
    )
    dual_p64 = _family(
        extra_topk=512,
        extra_page_size=64,
        has_extra_topk_length=True,
    )
    with_sink = _family(has_attn_sink=True)

    assert main.startswith("dsv4_nvfp4_v")
    assert len({main, dual_p2, dual_p64, with_sink}) == 4
    assert main != "dsv4"  # An FP8 record can never satisfy this lookup.


def test_nvfp4_planner_rejects_shapes_outside_both_envelopes(
    planner_state,
) -> None:
    assert _plan(8, num_heads=8) is None
    assert _plan(8, topk=256) is None
    assert _plan(8, primary_page_size=32) is None
    assert _plan(8, extra_topk=512, extra_page_size=1) is None
    with pytest.raises(ValueError, match="has_extra_topk_length"):
        _plan(8, has_extra_topk_length=True)


def test_nvfp4_plan_rechecks_lookup_after_lazy_calibration(
    planner_state, monkeypatch
) -> None:
    crossover, cpb = planner_state
    family = _family()
    calls = []

    def inject(**kwargs) -> None:
        calls.append(kwargs)
        crossover[_phase_key(family, 16)] = 16
        cpb[f"{family}|64|128|16"] = 4

    monkeypatch.setattr(plan_mod, "_maybe_calibrate", inject)
    planned = _plan(12)

    assert len(calls) == 1
    assert planned is not None
    assert planned.variant is plan_mod.NVFP4KernelVariant.DECODE_SPLITK
    assert planned.cpb == 4
