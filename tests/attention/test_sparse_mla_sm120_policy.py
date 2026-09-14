# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Dispatch-table and config-query tests for sparse-MLA SM120.

Config queries and diagnostics use real compiled capabilities without a GPU.
Tests using ``known_crossover`` or ``planner_state`` inject recorded capability
answers and reject module loading. These answers are checked against compiled
modules in ``test_sparse_mla_sm120_execution.py``.
Actual dispatch and launch behavior is covered on SM12x hardware.
"""

from __future__ import annotations

import re

import pytest
import torch

import flashinfer
from flashinfer.mla import _sparse_mla_sm120_dsv4_nvfp4_policy as native_policy
from flashinfer.mla import (
    SparseMLASm120DecodeConfig,
    supported_sparse_mla_sm120_configs,
)
from flashinfer.mla._sparse_mla_sm120 import (
    _DECODE_DSV3_2_DISPATCH,
    _DECODE_DSV4_DISPATCH,
    _DECODE_DSV4_1_DISPATCH,
    _DECODE_GLM53_NOPE_DISPATCH,
    _DECODE_MAX_TOKENS,
    _DECODE_DOTS3_SWA_DISPATCH,
    _MODEL_TYPE_DSV3_2,
    _MODEL_TYPE_DSV4,
    _MODEL_TYPE_DSV4_1,
    _MODEL_TYPE_GLM_NSA,
    _MODEL_TYPE_GLM53_NOPE,
    _MODEL_TYPE_DOTS3_SWA,
    _decode_scratch_views,
    _decode_dispatch_error_message,
    _resolve_model_type,
)
from flashinfer.mla._sparse_mla_sm120_policy import (
    _PREFILL_IMPL_SWAPAB,
    _normalize_prefill_impl,
    decode_splitk_eligible,
    plan,
)


"""Recorded compiled answers for policy unit tests, not an eligibility model."""

ORDINARY_CANDIDATES = {
    (0, 16, 1000, 64, False): (0,),
    (0, 32, 2048, 64, False): (0, 2),
    (0, 64, 1000, 64, False): (0,),
    (0, 64, 1024, 64, False): (0, 2, 4),
    (0, 64, 2048, 64, False): (0, 2, 4),
    (0, 256, 2048, 64, False): (),
    (1, 12, 256, 64, False): (0,),
    (1, 24, 256, 64, False): (0,),
    (1, 64, 100, 64, False): (0,),
    (1, 64, 128, 64, True): (0, 3),
    (1, 64, 256, 64, True): (0, 3),
    (1, 64, 384, 64, False): (0, 2),
    (1, 64, 512, 32, False): (),
    (1, 64, 512, 64, False): (0, 2),
    (1, 64, 1000, 64, True): (0,),
    (1, 64, 2048, 64, False): (0, 2),
    (1, 80, 256, 64, False): (0,),
    (1, 256, 256, 64, False): (),
    (2, 16, 384, 64, False): (0, 1),
    (3, 32, 2176, 64, False): (0, 2),
    (3, 64, 2048, 64, False): (0, 2, 4),
    (3, 64, 2176, 64, False): (0, 2, 4),
    (3, 128, 2048, 64, False): (0, 2, 4),
    (4, 8, 576, 64, False): (0, 1),
    (4, 16, 576, 64, False): (0, 1),
    (4, 32, 576, 64, False): (0, 1),
    (4, 64, 448, 64, False): (),
    (4, 64, 512, 64, False): (),
    (4, 64, 576, 64, False): (0, 1),
    (4, 64, 640, 64, False): (0, 1),
    (5, 8, 512, 64, False): (0, 1),
    (5, 16, 128, 64, True): (0, 1),
    (5, 16, 512, 64, False): (0, 1),
    (5, 32, 512, 64, False): (0, 1),
    (5, 64, 512, 64, False): (0, 1),
    (5, 64, 512, 64, True): (0, 1),
}

DSV4_NVFP4_SUPPORT = {
    (64, 128, 64, 0, 0): True,
    (8, 128, 64, 0, 0): False,
    (64, 256, 64, 0, 0): False,
    (64, 128, 32, 0, 0): False,
    (64, 128, 64, 512, 1): False,
}


def block_module_loading(monkeypatch):
    from flashinfer.jit.core import JitSpec
    from flashinfer.mla import _sparse_mla_sm120_calibration as calibration

    monkeypatch.setattr(calibration, "refresh_store", lambda: None)

    def unexpected(*args, **kwargs):
        raise AssertionError("policy unit test attempted to load a compiled module")

    monkeypatch.setattr(JitSpec, "build_and_load", unexpected)


def recorded_candidates(
    model, heads, topk, page, dual, extra_page=64, precision="default"
):
    assert extra_page == 64 and precision == "default"
    return frozenset(ORDINARY_CANDIDATES[model, heads, topk, page, dual])


def test_wrapper_construction_does_not_load_module(monkeypatch):
    """Constructing the wrapper must not build or load the JIT module."""
    block_module_loading(monkeypatch)
    from flashinfer.mla._sparse_mla_sm120 import _SparseMLAPagedAttentionRunner

    _SparseMLAPagedAttentionRunner()
    _SparseMLAPagedAttentionRunner(kv_cache_format="nvfp4")
    _SparseMLAPagedAttentionRunner(
        kv_scale_format="ue8m0_g32", extra_kv_fp4=True, compute_precision="bf16"
    )


def test_supported_configs_families() -> None:
    """The query API mirrors the decode dispatch envelopes exactly."""
    configs = supported_sparse_mla_sm120_configs()
    assert set(configs) == {
        "dsv4",
        "dsv3_2",
        "glm_nsa",
        "glm53_nope",
        "dots3_swa",
        "dsv4_1",
    }
    assert all(
        isinstance(config, SparseMLASm120DecodeConfig) for config in configs.values()
    )

    dsv4 = configs["dsv4"]
    assert dsv4.d_qk == 512
    assert dsv4.page_block_size == 64
    assert dsv4.max_num_tokens == _DECODE_MAX_TOKENS
    assert dsv4.max_num_heads == 128
    assert dsv4.topks == frozenset({128, 192, 256, 512, 1024})  # calibrated values
    assert dsv4.min_topk == 1
    # The dispatch envelope is a membership predicate (topk is a runtime
    # kernel argument): any H in [1, 128] at any topk >= min_topk. vLLM
    # probes ``(num_heads, topk) in _DECODE_DSV4_DISPATCH`` directly.
    assert (64, 128) in _DECODE_DSV4_DISPATCH
    assert (48, 384) in _DECODE_DSV4_DISPATCH  # off the calibrated grid
    assert (256, 128) not in _DECODE_DSV4_DISPATCH
    assert (64, 0) not in _DECODE_DSV4_DISPATCH

    dsv3_2 = configs["dsv3_2"]
    assert dsv3_2.d_qk == 576
    assert dsv3_2.page_block_size == 64
    assert dsv3_2.topks == frozenset({128, 512, 1024, 2048})
    assert dsv3_2.min_topk == 1
    assert (96, 640) in _DECODE_DSV3_2_DISPATCH

    # GLM-NSA shares the DSv3.2 decode instantiations (same config object).
    assert configs["glm_nsa"] is dsv3_2

    glm53 = configs["glm53_nope"]
    assert glm53.d_qk == 512
    assert glm53.page_block_size == 64
    assert glm53.topks == frozenset({2176})
    assert glm53.min_topk == 1
    assert (64, 2176) in _DECODE_GLM53_NOPE_DISPATCH

    dots3_swa = configs["dots3_swa"]
    assert dots3_swa.d_qk == 1088
    assert dots3_swa.page_block_size == 64
    assert dots3_swa.topks == frozenset({576})
    assert dots3_swa.min_topk == 513  # the sliding window must fit the buffer
    assert (64, 513) in _DECODE_DOTS3_SWA_DISPATCH
    assert (64, 576) in _DECODE_DOTS3_SWA_DISPATCH
    assert (64, 512) not in _DECODE_DOTS3_SWA_DISPATCH

    dsv4_1 = configs["dsv4_1"]
    assert dsv4_1.d_qk == 512
    assert dsv4_1.page_block_size == 64
    assert dsv4_1.topks == frozenset({512})  # the V4.1 indexer topk
    assert dsv4_1.min_topk == 1
    assert dsv4_1.bytes_per_token == 528
    assert (64, 512) in _DECODE_DSV4_1_DISPATCH
    assert (48, 384) in _DECODE_DSV4_1_DISPATCH  # off the calibrated grid
    assert (256, 512) not in _DECODE_DSV4_1_DISPATCH

    # The lazy export resolves through the public flashinfer.mla namespace.
    assert (
        flashinfer.mla.supported_sparse_mla_sm120_configs
        is supported_sparse_mla_sm120_configs
    )
    assert "supported_sparse_mla_sm120_configs" in dir(flashinfer.mla)
    assert "SparseMLASm120DecodeConfig" in dir(flashinfer.mla)


def test_supported_configs_nvfp4_envelope() -> None:
    """The shared query API exposes the exact, independently keyed NVFP4 set."""
    configs = supported_sparse_mla_sm120_configs(kv_cache_format="nvfp4")
    assert set(configs) == {"dsv4"}
    dsv4 = configs["dsv4"]
    assert dsv4.kv_cache_format == "nvfp4"
    assert dsv4.bytes_per_token == 384
    assert dsv4.supported_num_heads() == (16, 32, 64, 128)
    assert dsv4.supported_topk() == (128, 512)
    assert dsv4.extra_page_block_sizes == frozenset({2, 64})
    assert dsv4.supports_decode(64, 128)
    assert not dsv4.supports_decode(8, 128)
    assert not dsv4.supports_decode(64, 256)

    with pytest.raises(ValueError, match="kv_cache_format"):
        supported_sparse_mla_sm120_configs(kv_cache_format="int4")


def test_nvfp4_exact_head_scratch_view() -> None:
    """The shared scratch slicer also supports NVFP4's exact-head ABI."""
    mid_out = torch.empty((8, 64, 9, 512), dtype=torch.bfloat16, device="meta")
    mid_lse = torch.empty((8, 64, 9), dtype=torch.float32, device="meta")

    out_view, lse_view = _decode_scratch_views(
        mid_out,
        mid_lse,
        num_tokens=2,
        num_heads=16,
        num_splits=2,
        d_v=512,
        scratch_heads=16,
    )

    assert out_view.shape == (2, 16, 2, 512)
    assert lse_view.shape == (2, 16, 2)


def test_supported_helpers() -> None:
    """supported_num_heads / supported_topk return sorted tuples."""
    dsv4 = supported_sparse_mla_sm120_configs()["dsv4"]
    # Runtime-H instantiation: every head count in [1, 128] is served.
    assert dsv4.supported_num_heads() == tuple(range(1, 129))
    assert dsv4.supported_topk(64) == (128, 192, 256, 512, 1024)
    assert dsv4.supported_topk() == (128, 192, 256, 512, 1024)
    assert dsv4.supported_topk(48) == (128, 192, 256, 512, 1024)  # any H <= 128
    assert dsv4.supported_topk(256) == ()  # beyond the runtime-H ceiling

    dsv3_2 = supported_sparse_mla_sm120_configs()["dsv3_2"]
    assert dsv3_2.supported_topk(64) == (128, 512, 1024, 2048)


def test_supports_decode_matches_dispatch_predicates() -> None:
    """config.supports_decode agrees with the plan-layer decode predicate
    (model_type resolved from d_qk), on and off the calibrated grid (topk is
    a runtime kernel argument)."""
    configs = supported_sparse_mla_sm120_configs()
    dsv4 = configs["dsv4"]
    for num_heads in (8, 16, 64, 128, 48, 3):
        for topk in (128, 384, 256, 640, 1024, 4096):
            assert dsv4.supports_decode(num_heads, topk)
            for t in (1, _DECODE_MAX_TOKENS):
                assert decode_splitk_eligible(
                    _resolve_model_type(512, "auto"), num_heads, topk, 64, False, t
                )

    dsv3_2 = configs["dsv3_2"]
    for num_heads in (8, 64, 128, 24):
        for topk in (128, 512, 1024, 2048, 3000):
            assert dsv3_2.supports_decode(num_heads, topk)
            assert decode_splitk_eligible(
                _resolve_model_type(576, "auto"), num_heads, topk, 64, False, 1
            )


def test_supports_decode_rejects_mismatches() -> None:
    """supports_decode is False only outside the envelope axes."""
    dsv4 = supported_sparse_mla_sm120_configs()["dsv4"]
    assert dsv4.supports_decode(64, 384)  # arbitrary topk rides runtime-topk
    assert not dsv4.supports_decode(64, 0)  # topk below min_topk=1
    assert not dsv4.supports_decode(256, 256)  # num_heads past the runtime-H ceiling
    assert dsv4.supports_decode(48, 256)  # arbitrary H <= 128 rides runtime-H
    assert not dsv4.supports_decode(64, 256, page_block_size=32)
    assert not dsv4.supports_decode(64, 256, num_tokens=_DECODE_MAX_TOKENS + 1)
    assert dsv4.supports_decode(64, 256, num_tokens=_DECODE_MAX_TOKENS)
    assert not decode_splitk_eligible(
        _resolve_model_type(512, "auto"), 64, 0, 64, False, 1
    )
    assert not decode_splitk_eligible(
        _resolve_model_type(512, "auto"), 64, 256, 64, False, _DECODE_MAX_TOKENS + 1
    )
    # DOTS3_SWA's window constraint: topk < 513 is out of envelope.
    dots3 = supported_sparse_mla_sm120_configs()["dots3_swa"]
    assert dots3.supports_decode(64, 513)
    assert not dots3.supports_decode(64, 512)


def test_error_message_names_topk_mismatch() -> None:
    """A sub-minimum topk is named with the family minimum."""
    msg = _decode_dispatch_error_message(
        num_tokens=5,
        num_heads=64,
        topk=512,
        d_qk=1088,
        page_block_size=64,
        model_type=_MODEL_TYPE_DOTS3_SWA,
        extra_topk=0,
    )
    assert "topk=512 is below the dots3_swa decode minimum (topk >= 513" in msg
    assert "sliding window must fit the indices buffer" in msg
    assert "prefill envelope both reject" in msg
    assert "supported_sparse_mla_sm120_configs" in msg
    # The matching page size must not be blamed.
    assert "page_block_size=64 is unsupported" not in msg


def test_error_message_names_num_heads_mismatch() -> None:
    """A head count past the runtime-H ceiling names the envelope."""
    msg = _decode_dispatch_error_message(
        num_tokens=1,
        num_heads=256,
        topk=256,
        d_qk=512,
        page_block_size=64,
        model_type=_MODEL_TYPE_DSV4,
        extra_topk=0,
    )
    assert "num_heads=256 exceeds the decode envelope [1, 128]" in msg


def test_error_message_names_both_mismatches() -> None:
    """Both num_heads and topk out of envelope are reported together."""
    msg = _decode_dispatch_error_message(
        num_tokens=1,
        num_heads=256,
        topk=512,
        d_qk=1088,
        page_block_size=64,
        model_type=_MODEL_TYPE_DOTS3_SWA,
        extra_topk=0,
    )
    assert "topk=512 is below the dots3_swa decode minimum (topk >= 513" in msg
    assert "num_heads=256 exceeds the decode envelope [1, 128]" in msg


def test_error_message_names_page_block_size_mismatch() -> None:
    """An uninstantiated page size is named; valid pairs are not blamed."""
    msg = _decode_dispatch_error_message(
        num_tokens=1,
        num_heads=64,
        topk=256,
        d_qk=512,
        page_block_size=32,
        model_type=_MODEL_TYPE_DSV4,
        extra_topk=0,
    )
    assert "page_block_size=32 is unsupported" in msg
    assert "instantiated only for page_block_size=64" in msg
    # (num_heads=64, topk=256) is inside the envelope, so the Mismatch
    # reasons blame neither (the shape summary echo is expected).
    assert "num_heads=64 exceeds" not in msg
    assert "topk=256 is below" not in msg


def test_error_message_dsv3_2_family() -> None:
    """DSv3.2 and GLM-NSA report their family name; a legal topk is not
    blamed (any width >= 1 is served)."""
    for model_type, family in (
        (_MODEL_TYPE_DSV3_2, "dsv3_2"),
        (_MODEL_TYPE_GLM_NSA, "glm_nsa"),
    ):
        msg = _decode_dispatch_error_message(
            num_tokens=2,
            num_heads=256,
            topk=192,
            d_qk=576,
            page_block_size=64,
            model_type=model_type,
            extra_topk=0,
        )
        assert f"model_type={family}" in msg
        assert "num_heads=256 exceeds the decode envelope [1, 128]" in msg
        assert "is below" not in msg  # topk=192 is legal; not blamed


def test_error_message_keeps_shape_summary_format() -> None:
    """Callers grepping for the pre-existing message shape keep working.

    (The shape itself is decode-eligible under runtime-topk; the builder is
    exercised directly here purely for the message format.)"""
    msg = _decode_dispatch_error_message(
        num_tokens=1,
        num_heads=16,
        topk=384,
        d_qk=512,
        page_block_size=64,
        model_type=_MODEL_TYPE_DSV4,
        extra_topk=0,
    )
    assert re.search(r"no decode kernel.*num_tokens=1, num_heads=16, topk=384", msg)


def test_plan_swapab_rejections(known_crossover) -> None:
    """Forced swapab raises at the plan layer, before any launch: non-V32
    family, dual-cache call, unknown prefill_impl string."""
    dev = torch.device("cpu")
    with pytest.raises(ValueError, match="V32-family"):
        plan(128, 64, 512, _MODEL_TYPE_DSV4, 64, False, _PREFILL_IMPL_SWAPAB, dev)
    with pytest.raises(ValueError, match="dual-cache"):
        plan(
            128,
            64,
            128,
            _MODEL_TYPE_DSV4,
            64,
            True,
            _PREFILL_IMPL_SWAPAB,
            dev,
            extra_topk=64,
        )
    with pytest.raises(ValueError, match="prefill_impl"):
        _normalize_prefill_impl("sg")


def test_extra_cache_args_pairing() -> None:
    """extra_kv_cache / extra_indices / extra_topk_length are an
    all-or-nothing group: a mismatch must raise before planning (previously
    extra_indices alone reached the planner as has_extra=False with
    extra_topk>0). The check precedes any module load, so CPU tensors
    suffice; this exercises the same funnel the runner path calls."""
    from flashinfer.mla._sparse_mla_sm120 import _sparse_mla_sm120_paged_attention

    q = torch.zeros(2, 16, 512, dtype=torch.bfloat16)
    kv = torch.zeros(4, 64, 584, dtype=torch.uint8)
    idx = torch.zeros(2, 128, dtype=torch.int32)
    out = torch.zeros(2, 16, 512, dtype=torch.bfloat16)
    lse = torch.zeros(2, 16, dtype=torch.float32)
    extra_idx = torch.zeros(2, 64, dtype=torch.int32)
    with pytest.raises(ValueError, match="must be provided together"):
        _sparse_mla_sm120_paged_attention(
            q, kv, idx, out, lse, 1.0, extra_indices=extra_idx
        )
    with pytest.raises(ValueError, match="must be provided together"):
        _sparse_mla_sm120_paged_attention(q, kv, idx, out, lse, 1.0, extra_kv_cache=kv)
    with pytest.raises(ValueError, match="requires extra_kv_cache"):
        _sparse_mla_sm120_paged_attention(
            q,
            kv,
            idx,
            out,
            lse,
            1.0,
            extra_topk_length=torch.zeros(2, dtype=torch.int32),
        )


# Crossover-aware decode-form routing via the dispatch planner (pure Python;
# no GPU).


@pytest.fixture
def known_crossover(monkeypatch):
    """Inject a decode_max_tokens lookup without touching disk/GPU state."""
    from flashinfer.mla import _sparse_mla_sm120_calibration as cpb_mod
    from flashinfer.mla import _sparse_mla_sm120_policy as plan_mod

    block_module_loading(monkeypatch)
    monkeypatch.setattr(plan_mod, "_candidates", recorded_candidates)
    monkeypatch.setattr(plan_mod, "format_info", lambda model: {"page_size": 64})
    table = {}
    monkeypatch.setattr(cpb_mod, "_device_key", lambda device: "0:Fake GPU")
    monkeypatch.setattr(cpb_mod, "_maybe_load_disk", lambda: None)
    monkeypatch.setattr(cpb_mod, "get_constants", lambda device, family: None)
    monkeypatch.setattr(
        cpb_mod,
        "get_decode_max_tokens",
        lambda device, family, num_heads, topk: table.get(
            f"{family}|{num_heads}|{topk}"
        ),
    )
    yield plan_mod, table


def test_plan_off_grid_topk_rides_runtime_topk(known_crossover) -> None:
    """topk is a runtime kernel argument: off-grid widths (dsv4 topk=2048,
    384) are decode-eligible and take the decode-first default when
    uncalibrated. (topk=2048 was prefill-routed before runtime-topk.)"""
    plan_mod, _ = known_crossover
    for topk in (384, 2048):
        planned = plan_mod.plan(
            32,
            64,
            topk,
            _MODEL_TYPE_DSV4,
            64,
            False,
            plan_mod._PREFILL_IMPL_AUTO,
            torch.device("cpu"),
        )
        assert planned is not None
        assert planned.variant is plan_mod.KernelVariant.DECODE_SPLITK
        assert planned.cpb == -1  # no calibrated constants


def test_plan_neither_envelope_returns_none(known_crossover) -> None:
    """A shape rejected by both envelopes plans to None (caller raises)."""
    plan_mod, _ = known_crossover
    # num_heads=256 is past the runtime-H decode ceiling (128) and outside
    # every prefill head set.
    assert (
        plan_mod.plan(
            8,
            256,
            2048,
            _MODEL_TYPE_DSV3_2,
            64,
            False,
            plan_mod._PREFILL_IMPL_AUTO,
            torch.device("cpu"),
        )
        is None
    )


def test_plan_unknown_crossover_keeps_decode(known_crossover) -> None:
    """Instantiated shape, no calibration: the decode-first default holds."""
    plan_mod, _ = known_crossover
    for num_tokens in (1, _DECODE_MAX_TOKENS):
        planned = plan_mod.plan(
            num_tokens,
            64,
            512,
            _MODEL_TYPE_DSV4,
            64,
            False,
            plan_mod._PREFILL_IMPL_AUTO,
            torch.device("cpu"),
        )
        assert planned is not None
        assert planned.variant is plan_mod.KernelVariant.DECODE_SPLITK
        assert planned.cpb == -1  # no calibrated constants


def test_plan_arbitrary_num_heads_rides_runtime_h(known_crossover) -> None:
    """Off-grid head counts (runtime-H instantiation) plan to DECODE_SPLITK
    with the decode-first default (no calibrated crossover entry)."""
    plan_mod, _ = known_crossover
    for num_heads in (12, 24, 80):
        planned = plan_mod.plan(
            4,
            num_heads,
            256,
            _MODEL_TYPE_DSV4,
            64,
            False,
            plan_mod._PREFILL_IMPL_AUTO,
            torch.device("cpu"),
        )
        assert planned is not None
        assert planned.variant is plan_mod.KernelVariant.DECODE_SPLITK
        assert planned.cpb == -1  # no calibrated constants
    # Past the runtime-H ceiling neither envelope serves the shape.
    assert (
        plan_mod.plan(
            4,
            256,
            256,
            _MODEL_TYPE_DSV4,
            64,
            False,
            plan_mod._PREFILL_IMPL_AUTO,
            torch.device("cpu"),
        )
        is None
    )


@pytest.mark.parametrize(
    "family_key,model_type,num_heads,topk,above_variant",
    [
        ("dsv4", _MODEL_TYPE_DSV4, 64, 512, "PREFILL_MG"),
        ("glm53_nope", _MODEL_TYPE_GLM53_NOPE, 32, 2176, "PREFILL_MG"),
        ("dots3_swa", _MODEL_TYPE_DOTS3_SWA, 64, 576, "PREFILL_SG"),
        ("dsv4_1", _MODEL_TYPE_DSV4_1, 64, 512, "PREFILL_SG"),
    ],
)
def test_plan_crossover_injection(
    known_crossover, family_key, model_type, num_heads, topk, above_variant
) -> None:
    """An injected crossover=8 applies to the family key space: T<=8 decodes,
    T>8 prefills (MG where instantiated, SG for the SG-only family)."""
    plan_mod, table = known_crossover
    table[f"{family_key}|{num_heads}|{topk}"] = 8
    planned = plan_mod.plan(
        8,
        num_heads,
        topk,
        model_type,
        64,
        False,
        plan_mod._PREFILL_IMPL_AUTO,
        torch.device("cpu"),
    )
    assert (
        planned is not None and planned.variant is plan_mod.KernelVariant.DECODE_SPLITK
    )
    planned = plan_mod.plan(
        16,
        num_heads,
        topk,
        model_type,
        64,
        False,
        plan_mod._PREFILL_IMPL_AUTO,
        torch.device("cpu"),
    )
    assert planned is not None
    assert planned.variant is getattr(plan_mod.KernelVariant, above_variant)


def test_plan_crossover_zero_always_prefill(known_crossover) -> None:
    """decode_max_tokens=0 (decode never wins) routes even T=1 to prefill."""
    plan_mod, table = known_crossover
    table["dsv3_2|64|2048"] = 0
    planned = plan_mod.plan(
        1,
        64,
        2048,
        _MODEL_TYPE_DSV3_2,
        64,
        False,
        plan_mod._PREFILL_IMPL_AUTO,
        torch.device("cpu"),
    )
    # Auto prefill prefers swapAB at this shape.
    assert (
        planned is not None and planned.variant is plan_mod.KernelVariant.PREFILL_SWAPAB
    )


def test_plan_page_block_size_mismatch(known_crossover) -> None:
    """A non-64 page size is served by neither envelope (pbs=64 is hardwired
    in every instantiation); the planner returns None instead of letting C++
    launch with a mismatched stride."""
    plan_mod, _ = known_crossover
    assert (
        plan_mod.plan(
            8,
            64,
            512,
            _MODEL_TYPE_DSV4,
            32,
            False,
            plan_mod._PREFILL_IMPL_AUTO,
            torch.device("cpu"),
        )
        is None
    )


def test_plan_prefill_impl_pref(known_crossover) -> None:
    """prefill_impl='mg' excludes swapAB; 'swapab' forces it or raises."""
    plan_mod, _ = known_crossover
    planned = plan_mod.plan(
        128,
        64,
        2048,
        _MODEL_TYPE_DSV3_2,
        64,
        False,
        plan_mod._PREFILL_IMPL_AUTO,
        torch.device("cpu"),
    )
    assert (
        planned is not None and planned.variant is plan_mod.KernelVariant.PREFILL_SWAPAB
    )
    planned = plan_mod.plan(
        128,
        64,
        2048,
        _MODEL_TYPE_DSV3_2,
        64,
        False,
        plan_mod._PREFILL_IMPL_MG,
        torch.device("cpu"),
    )
    assert planned is not None and planned.variant is plan_mod.KernelVariant.PREFILL_MG
    with pytest.raises(ValueError, match="num_heads"):
        plan_mod.plan(
            128,
            32,
            2048,
            _MODEL_TYPE_DSV3_2,
            64,
            False,
            plan_mod._PREFILL_IMPL_SWAPAB,
            torch.device("cpu"),
        )


def test_plan_large_t_selects_prefill(known_crossover) -> None:
    plan_mod, _ = known_crossover
    for tokens in (65, 8192):
        planned = plan_mod.plan(
            tokens,
            64,
            512,
            _MODEL_TYPE_DSV4,
            64,
            False,
            plan_mod._PREFILL_IMPL_AUTO,
            torch.device("cpu"),
        )
        assert planned is not None
        assert planned.variant is plan_mod.KernelVariant.PREFILL_MG


def test_plan_glm53_nope_decode_and_prefill(known_crossover) -> None:
    """GLM53_NOPE: decode at (32|64, 2176) for T<=64; prefill MG above."""
    plan_mod, _ = known_crossover
    for num_heads in (32, 64):
        planned = plan_mod.plan(
            4,
            num_heads,
            2176,
            _MODEL_TYPE_GLM53_NOPE,
            64,
            False,
            plan_mod._PREFILL_IMPL_AUTO,
            torch.device("cpu"),
        )
        assert planned is not None
        assert planned.variant is plan_mod.KernelVariant.DECODE_SPLITK
    planned = plan_mod.plan(
        65,
        32,
        2176,
        _MODEL_TYPE_GLM53_NOPE,
        64,
        False,
        plan_mod._PREFILL_IMPL_AUTO,
        torch.device("cpu"),
    )
    assert planned is not None and planned.variant is plan_mod.KernelVariant.PREFILL_MG
    # (64, 2048): decode-eligible under runtime-topk (any width >= 1), and
    # prefill-eligible too (2048 % 64 == 0 — NOPE prefill serves any
    # whole-tile width). Decode-first default applies.
    planned = plan_mod.plan(
        4,
        64,
        2048,
        _MODEL_TYPE_GLM53_NOPE,
        64,
        False,
        plan_mod._PREFILL_IMPL_AUTO,
        torch.device("cpu"),
    )
    assert planned is not None
    assert planned.variant is plan_mod.KernelVariant.DECODE_SPLITK


def test_plan_glm53_nope_swapab(known_crossover) -> None:
    """swapAB serves GLM53_NOPE at topk=2176: auto prefers it at H>=64, and
    forcing it works; an ineligible head count still raises."""
    plan_mod, _ = known_crossover
    for impl in (plan_mod._PREFILL_IMPL_AUTO, plan_mod._PREFILL_IMPL_SWAPAB):
        planned = plan_mod.plan(
            128,
            64,
            2176,
            _MODEL_TYPE_GLM53_NOPE,
            64,
            False,
            impl,
            torch.device("cpu"),
        )
        assert (
            planned is not None
            and planned.variant is plan_mod.KernelVariant.PREFILL_SWAPAB
        )
    with pytest.raises(ValueError, match="num_heads"):
        plan_mod.plan(
            128,
            32,
            2176,
            _MODEL_TYPE_GLM53_NOPE,
            64,
            False,
            plan_mod._PREFILL_IMPL_SWAPAB,
            torch.device("cpu"),
        )


def test_plan_dots3_swa_decode_and_prefill(known_crossover) -> None:
    """DOTS3_SWA: decode at (H, 576) for T<=64; SG-only prefill above."""
    plan_mod, _ = known_crossover
    for num_heads in (8, 16, 32, 64):
        planned = plan_mod.plan(
            4,
            num_heads,
            576,
            _MODEL_TYPE_DOTS3_SWA,
            64,
            False,
            plan_mod._PREFILL_IMPL_AUTO,
            torch.device("cpu"),
        )
        assert planned is not None
        assert planned.variant is plan_mod.KernelVariant.DECODE_SPLITK
    # Past the decode-form cutoff every supported head count routes to SG —
    # DOTS3_SWA has no MG/swapAB form, so H=64 must not route to swapAB even
    # though 64 is in the swapAB head set for the V32 family.
    for num_heads in (8, 16, 32, 64):
        planned = plan_mod.plan(
            65,
            num_heads,
            576,
            _MODEL_TYPE_DOTS3_SWA,
            64,
            False,
            plan_mod._PREFILL_IMPL_AUTO,
            torch.device("cpu"),
        )
        assert planned is not None
        assert planned.variant is plan_mod.KernelVariant.PREFILL_SG
    # topk=512 is below the 513 sliding-window floor, so neither side serves
    # it (prefill otherwise takes any topk >= 513 with topk % 64 == 0).
    assert (
        plan_mod.plan(
            4,
            64,
            512,
            _MODEL_TYPE_DOTS3_SWA,
            64,
            False,
            plan_mod._PREFILL_IMPL_AUTO,
            torch.device("cpu"),
        )
        is None
    )
    # A wider window-sized buffer (640 >= 513, whole tiles) is served by
    # prefill past the decode-form cutoff.
    planned = plan_mod.plan(
        65,
        64,
        640,
        _MODEL_TYPE_DOTS3_SWA,
        64,
        False,
        plan_mod._PREFILL_IMPL_AUTO,
        torch.device("cpu"),
    )
    assert planned is not None
    assert planned.variant is plan_mod.KernelVariant.PREFILL_SG
    # Forcing swapAB on the SG-only family raises.
    with pytest.raises(ValueError, match="V32-family"):
        plan_mod.plan(
            128,
            64,
            576,
            _MODEL_TYPE_DOTS3_SWA,
            64,
            False,
            plan_mod._PREFILL_IMPL_SWAPAB,
            torch.device("cpu"),
        )


def test_resolve_model_type_dsv4_1_explicit_only() -> None:
    """DSV4_1 shares d_qk=512 with DSV4 and its 528B payload with GLM53_NOPE,
    so it is reachable only through the explicit ue8m0_g32 scale format."""
    assert _resolve_model_type(512, "auto") == _MODEL_TYPE_DSV4
    assert _resolve_model_type(512, "arbitrary_fp32") == _MODEL_TYPE_GLM53_NOPE
    assert _resolve_model_type(512, "ue8m0_g32") == _MODEL_TYPE_DSV4_1
    # The format is pinned to the 512-wide layout; other widths reject it.
    with pytest.raises(ValueError, match="kv_scale_format"):
        _resolve_model_type(576, "ue8m0_g32")
    with pytest.raises(ValueError, match="kv_scale_format"):
        _resolve_model_type(1088, "ue8m0_g32")


def test_plan_dsv4_1_dual_isolates_single_cache_calibration(
    known_crossover, monkeypatch
) -> None:
    plan_mod, table = known_crossover
    table["dsv4_1|16|128"] = 0

    def unexpected_cpb(*args, **kwargs):
        raise AssertionError("dual-cache call reused single-cache cpb calibration")

    monkeypatch.setattr(plan_mod, "_resolve_cpb", unexpected_cpb)
    for nt, variant in [
        (4, plan_mod.KernelVariant.DECODE_SPLITK),
        (65, plan_mod.KernelVariant.PREFILL_SG),
    ]:
        planned = plan_mod.plan(
            nt,
            16,
            128,
            _MODEL_TYPE_DSV4_1,
            64,
            True,
            plan_mod._PREFILL_IMPL_AUTO,
            torch.device("cpu"),
            extra_topk=77,
        )
        assert planned is not None
        assert planned.variant is variant
        assert planned.cpb == -1


def test_plan_dsv4_1_decode_and_prefill(known_crossover) -> None:
    """DSV4_1: decode (incl. dual-cache) at (H, 512) for T<=64; SG-only
    prefill above (its 32-wide quant groups floor the MG XV warp split)."""
    plan_mod, _ = known_crossover
    for num_heads in (8, 16, 32, 64):
        planned = plan_mod.plan(
            4,
            num_heads,
            512,
            _MODEL_TYPE_DSV4_1,
            64,
            False,
            plan_mod._PREFILL_IMPL_AUTO,
            torch.device("cpu"),
        )
        assert planned is not None
        assert planned.variant is plan_mod.KernelVariant.DECODE_SPLITK
    # Dual-cache decode stays on the decode-dsv4 kernel (same as DSV4).
    planned = plan_mod.plan(
        4,
        64,
        512,
        _MODEL_TYPE_DSV4_1,
        64,
        True,
        plan_mod._PREFILL_IMPL_AUTO,
        torch.device("cpu"),
        extra_topk=512,
    )
    assert planned is not None
    assert planned.variant is plan_mod.KernelVariant.DECODE_SPLITK
    # Past the decode-form cutoff every supported head count routes to SG.
    for num_heads in (8, 16, 32, 64):
        planned = plan_mod.plan(
            65,
            num_heads,
            512,
            _MODEL_TYPE_DSV4_1,
            64,
            False,
            plan_mod._PREFILL_IMPL_AUTO,
            torch.device("cpu"),
        )
        assert planned is not None
        assert planned.variant is plan_mod.KernelVariant.PREFILL_SG
    planned = plan_mod.plan(
        65,
        64,
        512,
        _MODEL_TYPE_DSV4_1,
        64,
        True,
        plan_mod._PREFILL_IMPL_AUTO,
        torch.device("cpu"),
        extra_topk=512,
    )
    assert planned is not None
    assert planned.variant is plan_mod.KernelVariant.PREFILL_SG
    # Forcing swapAB on the SG-only family raises.
    with pytest.raises(ValueError, match="V32-family"):
        plan_mod.plan(
            128,
            64,
            512,
            _MODEL_TYPE_DSV4_1,
            64,
            False,
            plan_mod._PREFILL_IMPL_SWAPAB,
            torch.device("cpu"),
        )


def test_plan_prefill_runtime_topk_widths(known_crossover) -> None:
    """Prefill topk is a runtime kernel argument: every variant serves any
    whole-tile width (topk >= 1, topk % 64 == 0; DOTS3_SWA also >= 513), and
    ragged widths fall out of the envelope."""
    plan_mod, _ = known_crossover
    # Widths that were template-pinned before (V32 2048/2176, DSV4
    # {128..2048}, MG_DUAL 128, DOTS3_SWA 576) now ride one instantiation.
    assert plan_mod.prefill_swapab_eligible(_MODEL_TYPE_DSV3_2, 64, 1024, 64, False)
    assert plan_mod.prefill_swapab_eligible(
        _MODEL_TYPE_GLM53_NOPE, 128, 2048, 64, False
    )
    assert plan_mod.prefill_sg_eligible(_MODEL_TYPE_GLM_NSA, 16, 384, 64, False)
    assert plan_mod.prefill_mg_eligible(_MODEL_TYPE_DSV4, 64, 384, 64, False)
    assert plan_mod.prefill_mg_dual_eligible(_MODEL_TYPE_DSV4, 64, 256, 64, True)
    assert plan_mod.prefill_sg_eligible(_MODEL_TYPE_DOTS3_SWA, 64, 640, 64, False)
    # Ragged widths (not a whole number of 64-wide tiles) are not served.
    assert not plan_mod.prefill_swapab_eligible(_MODEL_TYPE_DSV3_2, 64, 1000, 64, False)
    assert not plan_mod.prefill_sg_eligible(_MODEL_TYPE_DSV3_2, 16, 1000, 64, False)
    assert not plan_mod.prefill_mg_eligible(_MODEL_TYPE_DSV4, 64, 100, 64, False)
    assert not plan_mod.prefill_mg_dual_eligible(_MODEL_TYPE_DSV4, 64, 1000, 64, True)
    # DOTS3_SWA's window floor: 448 is a whole-tile width below 513.
    assert not plan_mod.prefill_sg_eligible(_MODEL_TYPE_DOTS3_SWA, 64, 448, 64, False)
    # Auto routing picks the same variants it had at the pinned widths.
    planned = plan_mod.plan(
        128,
        64,
        1024,
        _MODEL_TYPE_DSV3_2,
        64,
        False,
        plan_mod._PREFILL_IMPL_AUTO,
        torch.device("cpu"),
    )
    assert planned is not None
    assert planned.variant is plan_mod.KernelVariant.PREFILL_SWAPAB
    # Forced swapab accepts an off-pin whole-tile width now.
    planned = plan_mod.plan(
        128,
        64,
        1024,
        _MODEL_TYPE_DSV3_2,
        64,
        False,
        plan_mod._PREFILL_IMPL_SWAPAB,
        torch.device("cpu"),
    )
    assert planned is not None
    assert planned.variant is plan_mod.KernelVariant.PREFILL_SWAPAB
    # ...but still rejects a ragged width loudly at the plan layer.
    with pytest.raises(ValueError, match="topk % 64"):
        plan_mod.plan(
            128,
            64,
            1000,
            _MODEL_TYPE_DSV3_2,
            64,
            False,
            plan_mod._PREFILL_IMPL_SWAPAB,
            torch.device("cpu"),
        )


def test_sparse_mla_sm120_wrapper_public_export() -> None:
    """The runner alias is public via ``flashinfer.mla`` and is the impl class."""
    from flashinfer.mla import SparseMLASm120Wrapper
    from flashinfer.mla._sparse_mla_sm120 import _SparseMLAPagedAttentionRunner

    assert SparseMLASm120Wrapper is _SparseMLAPagedAttentionRunner
    assert "SparseMLASm120Wrapper" in dir(flashinfer.mla)


def test_actual_metadata_policy_filters_measured_prefill(monkeypatch):
    from flashinfer.mla import _sparse_mla_sm120_execution as execution
    from flashinfer.mla import _sparse_mla_sm120_policy as policy

    metadata = execution.AttentionMetadata(
        5,
        4,
        16,
        128,
        64,
        61,
        53,
        61 * 528,
        53 * 288,
        528,
        128,
        64,
        16,
        True,
        True,
        True,
        True,
        0,
    )
    monkeypatch.setattr(
        execution,
        "metadata_candidates",
        lambda m, *args: {0: 3} if m.tokens <= 64 else {},
    )
    selected = policy.PlannedCall(policy.KernelVariant.PREFILL_SG, -1)
    choice = policy.filter_metadata_selection(selected, metadata, "fp8", 148, 1000000)
    assert choice.variant is policy.KernelVariant.DECODE_SPLITK and choice.cpb == 1
    assert (
        policy.filter_metadata_selection(
            selected, metadata._replace(tokens=65), "fp8", 148, 1000000
        )
        is None
    )
    monkeypatch.setattr(execution, "metadata_candidates", lambda *args: {0: 3, 1: 3})
    assert (
        policy.filter_metadata_selection(selected, metadata, "bf16", 148, 1000000)
        == selected
    )


@pytest.fixture
def planner_state(monkeypatch):
    from flashinfer.mla import _sparse_mla_sm120_calibration as cpb_mod

    block_module_loading(monkeypatch)
    monkeypatch.setattr(
        native_policy, "_supports", lambda *args: DSV4_NVFP4_SUPPORT[args]
    )
    profiles: dict[str, dict] = {}
    monkeypatch.setattr(cpb_mod, "_device_key", lambda _device: "0:Fake SM120")
    monkeypatch.setattr(cpb_mod, "_maybe_load_disk", lambda: None)
    monkeypatch.setattr(cpb_mod, "get_profile", lambda key, _device: profiles.get(key))
    monkeypatch.setattr(native_policy, "_maybe_calibrate", lambda **_kwargs: None)
    monkeypatch.setattr(native_policy, "_plan_memo", {})
    return profiles


def _profile_key(**kwargs) -> str:
    defaults = dict(
        num_heads=64,
        topk=128,
        primary_page_size=64,
        extra_topk=0,
        extra_page_size=0,
        has_topk_length=True,
        has_extra_topk_length=False,
        has_attn_sink=False,
    )
    defaults.update(kwargs)
    return native_policy._request_key(**defaults)


def _profile(phases: dict[int, str], cpbs: dict[int, int] | None = None) -> dict:
    """Full-grid profile record; unlisted buckets default to decode at cpb 1."""
    cpbs = cpbs or {}
    return {
        "request": {"family": "dsv4_nvfp4"},
        "buckets": {
            str(t): {
                "variant": (
                    "decode_splitk"
                    if phases.get(t, "decode") == "decode"
                    else "prefill_streaming"
                ),
                "cpb": cpbs.get(t, 1),
                "decode_s": 1e-5,
                "prefill_s": 1e-5,
            }
            for t in native_policy._CROSSOVER_PROBED_T
        },
    }


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
    return native_policy.plan_nvfp4_sparse_mla_sm120(**defaults)


def test_nvfp4_missing_profile_keeps_safe_decode_fallback(planner_state) -> None:
    planned = _plan(32)
    assert planned is not None
    assert planned.variant is native_policy.NVFP4KernelVariant.DECODE_SPLITK
    assert planned.cpb == 0


def test_nvfp4_calibrated_crossover_replaces_fixed_64_policy(planner_state) -> None:
    planner_state[_profile_key()] = _profile({16: "prefill"}, {8: 2})
    native_policy._plan_memo.clear()
    at_crossover = _plan(8)
    above_crossover = _plan(9)
    assert at_crossover is not None
    assert at_crossover.variant is native_policy.NVFP4KernelVariant.DECODE_SPLITK
    assert at_crossover.cpb == 2
    assert above_crossover is not None
    assert above_crossover.variant is native_policy.NVFP4KernelVariant.PREFILL_STREAMING


def test_nvfp4_decode_cpb_uses_calibrated_token_bucket(planner_state) -> None:
    planner_state[_profile_key()] = _profile({}, {16: 3})
    native_policy._plan_memo.clear()
    planned = _plan(9)
    assert planned is not None
    assert planned.variant is native_policy.NVFP4KernelVariant.DECODE_SPLITK
    assert planned.cpb == 3


def test_nvfp4_64_is_only_decode_envelope(planner_state) -> None:
    planner_state[_profile_key()] = _profile({})
    native_policy._plan_memo.clear()
    assert _plan(64).variant is native_policy.NVFP4KernelVariant.DECODE_SPLITK
    assert _plan(65).variant is native_policy.NVFP4KernelVariant.PREFILL_STREAMING
    assert _plan(8192).variant is native_policy.NVFP4KernelVariant.PREFILL_STREAMING


def test_nvfp4_bucket_table_preserves_non_monotonic_wave_boundaries(
    planner_state,
) -> None:
    planner_state[_profile_key()] = _profile({48: "prefill"}, {64: 7})
    native_policy._plan_memo.clear()
    at_48 = _plan(48)
    at_64 = _plan(64)
    assert at_48.variant is native_policy.NVFP4KernelVariant.PREFILL_STREAMING
    assert at_64.variant is native_policy.NVFP4KernelVariant.DECODE_SPLITK
    assert at_64.cpb == 7


def test_nvfp4_calibration_namespace_is_format_and_shape_specific() -> None:
    main = _profile_key()
    dual_p2 = _profile_key(
        extra_topk=512, extra_page_size=2, has_extra_topk_length=True
    )
    dual_p64 = _profile_key(
        extra_topk=512, extra_page_size=64, has_extra_topk_length=True
    )
    with_sink = _profile_key(has_attn_sink=True)
    assert len({main, dual_p2, dual_p64, with_sink}) == 4
    assert main != "dsv4"


def test_nvfp4_planner_rejects_shapes_outside_both_envelopes(planner_state) -> None:
    assert _plan(8, num_heads=8) is None
    assert _plan(8, topk=256) is None
    assert _plan(8, primary_page_size=32) is None
    assert _plan(8, extra_topk=512, extra_page_size=1) is None
    with pytest.raises(ValueError, match="has_extra_topk_length"):
        _plan(8, has_extra_topk_length=True)


def test_nvfp4_plan_rechecks_lookup_after_lazy_calibration(
    planner_state, monkeypatch
) -> None:
    calls = []

    def inject(**kwargs) -> None:
        calls.append(kwargs)
        planner_state[kwargs["key"]] = _profile({}, {16: 4})

    monkeypatch.setattr(native_policy, "_maybe_calibrate", inject)
    planned = _plan(12)
    assert len(calls) == 1
    assert planned is not None
    assert planned.variant is native_policy.NVFP4KernelVariant.DECODE_SPLITK
    assert planned.cpb == 4


def test_nvfp4_plan_refines_exact_tokens_in_tuning(planner_state, monkeypatch) -> None:
    from types import SimpleNamespace

    planner_state[_profile_key()] = _profile({})
    native_policy._plan_memo.clear()
    monkeypatch.setattr(
        native_policy.AutoTuner,
        "get",
        lambda: SimpleNamespace(is_tuning_mode=True, _get_skip_ops_stack=lambda: []),
    )

    refined = []

    def fake_refine(device, *, tokens, **fields):
        refined.append(tokens)
        current = planner_state[native_policy._request_key(**fields)]
        current["buckets"][str(tokens)] = {
            "variant": "decode_splitk",
            "cpb": 4,
            "decode_s": 2e-5,
            "prefill_s": 1e-5,
        }
        return current["buckets"][str(tokens)]

    monkeypatch.setattr(native_policy, "refine_nvfp4", fake_refine)
    planned = _plan(12)
    assert refined == [12]
    assert planned is not None
    assert planned.variant is native_policy.NVFP4KernelVariant.DECODE_SPLITK
    assert planned.cpb == 4
    # The stored exact entry serves later calls without refining again.
    assert _plan(12).cpb == 4
    assert refined == [12]
    # Above the decode envelope no refinement is attempted.
    assert _plan(8192).variant is native_policy.NVFP4KernelVariant.PREFILL_STREAMING
    assert refined == [12]
