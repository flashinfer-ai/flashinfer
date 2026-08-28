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

Pure-Python coverage of :func:`supported_sparse_mla_sm120_configs` and the
decode dispatch-miss diagnostics (flashinfer-ai/flashinfer#4541). No GPU or
JIT build required; the raise path inside the kernel dispatcher is covered by
``test_sparse_mla_sm120.py`` on SM12x hardware.
"""

from __future__ import annotations

import re

import pytest
import torch

import flashinfer
from flashinfer.mla import (
    SparseMLASm120DecodeConfig,
    supported_sparse_mla_sm120_configs,
)
from flashinfer.mla._sparse_mla_sm120 import (
    _DECODE_DSV3_2_DISPATCH,
    _DECODE_DSV4_DISPATCH,
    _DECODE_GLM53_NOPE_DISPATCH,
    _DECODE_MAX_TOKENS,
    _DECODE_DOTS3_SWA_DISPATCH,
    _MODEL_TYPE_DSV3_2,
    _MODEL_TYPE_DSV4,
    _MODEL_TYPE_GLM_NSA,
    _MODEL_TYPE_GLM53_NOPE,
    _MODEL_TYPE_DOTS3_SWA,
    _decode_dispatch_error_message,
    _decode_dsv3_2_dispatchable,
    _decode_dsv4_dispatchable,
)


def test_supported_configs_families() -> None:
    """The query API mirrors the private dispatch tables exactly."""
    configs = supported_sparse_mla_sm120_configs()
    assert set(configs) == {"dsv4", "dsv3_2", "glm_nsa", "glm53_nope", "dots3_swa"}
    assert all(
        isinstance(config, SparseMLASm120DecodeConfig) for config in configs.values()
    )

    dsv4 = configs["dsv4"]
    assert dsv4.d_qk == 512
    assert dsv4.page_block_size == 64
    assert dsv4.max_num_tokens == _DECODE_MAX_TOKENS
    assert dsv4.head_topk_pairs == _DECODE_DSV4_DISPATCH

    dsv3_2 = configs["dsv3_2"]
    assert dsv3_2.d_qk == 576
    assert dsv3_2.page_block_size == 64
    assert dsv3_2.head_topk_pairs == _DECODE_DSV3_2_DISPATCH

    # GLM-NSA shares the DSv3.2 decode instantiations (same config object).
    assert configs["glm_nsa"] is dsv3_2

    glm53 = configs["glm53_nope"]
    assert glm53.d_qk == 512
    assert glm53.page_block_size == 64
    assert glm53.head_topk_pairs == _DECODE_GLM53_NOPE_DISPATCH

    dots3_swa = configs["dots3_swa"]
    assert dots3_swa.d_qk == 1088
    assert dots3_swa.page_block_size == 64
    assert dots3_swa.head_topk_pairs == _DECODE_DOTS3_SWA_DISPATCH

    # The lazy export resolves through the public flashinfer.mla namespace.
    assert (
        flashinfer.mla.supported_sparse_mla_sm120_configs
        is supported_sparse_mla_sm120_configs
    )
    assert "supported_sparse_mla_sm120_configs" in dir(flashinfer.mla)
    assert "SparseMLASm120DecodeConfig" in dir(flashinfer.mla)


def test_supported_helpers() -> None:
    """supported_num_heads / supported_topk return sorted tuples."""
    dsv4 = supported_sparse_mla_sm120_configs()["dsv4"]
    assert dsv4.supported_num_heads() == (8, 16, 32, 64, 128)
    assert dsv4.supported_topk(64) == (128, 192, 256, 512, 1024)
    assert dsv4.supported_topk() == (128, 192, 256, 512, 1024)
    assert dsv4.supported_topk(48) == ()

    dsv3_2 = supported_sparse_mla_sm120_configs()["dsv3_2"]
    assert dsv3_2.supported_topk(64) == (128, 512, 1024, 2048)


def test_supports_decode_matches_dispatch_predicates() -> None:
    """config.supports_decode agrees with the internal dispatch predicates."""
    configs = supported_sparse_mla_sm120_configs()
    dsv4 = configs["dsv4"]
    for num_heads, topk in sorted(dsv4.head_topk_pairs):
        assert dsv4.supports_decode(num_heads, topk)
        assert _decode_dsv4_dispatchable(1, num_heads, topk, 512, 64)
        assert _decode_dsv4_dispatchable(_DECODE_MAX_TOKENS, num_heads, topk, 512, 64)

    dsv3_2 = configs["dsv3_2"]
    for num_heads, topk in sorted(dsv3_2.head_topk_pairs):
        assert dsv3_2.supports_decode(num_heads, topk)
        assert _decode_dsv3_2_dispatchable(1, num_heads, topk, 576, 64)


def test_supports_decode_rejects_mismatches() -> None:
    """supports_decode is False for any out-of-set shape parameter."""
    dsv4 = supported_sparse_mla_sm120_configs()["dsv4"]
    assert not dsv4.supports_decode(64, 384)  # topk not instantiated
    assert not dsv4.supports_decode(48, 256)  # num_heads not instantiated
    assert not dsv4.supports_decode(64, 256, page_block_size=32)
    assert not dsv4.supports_decode(64, 256, num_tokens=_DECODE_MAX_TOKENS + 1)
    assert dsv4.supports_decode(64, 256, num_tokens=_DECODE_MAX_TOKENS)
    assert not _decode_dsv4_dispatchable(1, 64, 384, 512, 64)
    assert not _decode_dsv4_dispatchable(_DECODE_MAX_TOKENS + 1, 64, 256, 512, 64)


def test_error_message_names_topk_mismatch() -> None:
    """The issue-#4541 scenario: decode-form call, uninstantiated topk."""
    msg = _decode_dispatch_error_message(
        num_tokens=5,
        num_heads=64,
        topk=384,
        d_qk=512,
        page_block_size=64,
        model_type=_MODEL_TYPE_DSV4,
        extra_topk=0,
    )
    assert "topk=384 is not instantiated for num_heads=64" in msg
    assert "available topk: [128, 192, 256, 512, 1024]" in msg
    assert "prefill envelope both reject" in msg
    assert "supported_sparse_mla_sm120_configs" in msg
    # The matching page size must not be blamed.
    assert "page_block_size=64 is unsupported" not in msg


def test_error_message_names_num_heads_mismatch() -> None:
    """An uninstantiated head count is named with the available ones."""
    msg = _decode_dispatch_error_message(
        num_tokens=1,
        num_heads=48,
        topk=256,
        d_qk=512,
        page_block_size=64,
        model_type=_MODEL_TYPE_DSV4,
        extra_topk=0,
    )
    assert "num_heads=48 is not instantiated for topk=256" in msg
    assert "available num_heads: [8, 16, 32, 64, 128]" in msg


def test_error_message_names_both_mismatches() -> None:
    """Both num_heads and topk out of set are reported together."""
    msg = _decode_dispatch_error_message(
        num_tokens=1,
        num_heads=48,
        topk=384,
        d_qk=512,
        page_block_size=64,
        model_type=_MODEL_TYPE_DSV4,
        extra_topk=0,
    )
    assert "neither num_heads=48 nor topk=384 is instantiated" in msg
    assert "available num_heads: [8, 16, 32, 64, 128]" in msg
    assert "available topk: [128, 192, 256, 512, 1024]" in msg


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
    # (num_heads=64, topk=256) is instantiated, so it must not be blamed.
    assert "topk=256 is not instantiated" not in msg


def test_error_message_dsv3_2_family() -> None:
    """DSv3.2 and GLM-NSA report their family name and topk set."""
    for model_type, family in (
        (_MODEL_TYPE_DSV3_2, "dsv3_2"),
        (_MODEL_TYPE_GLM_NSA, "glm_nsa"),
    ):
        msg = _decode_dispatch_error_message(
            num_tokens=2,
            num_heads=128,
            topk=192,
            d_qk=576,
            page_block_size=64,
            model_type=model_type,
            extra_topk=0,
        )
        assert f"model_type={family}" in msg
        assert "topk=192 is not instantiated for num_heads=128" in msg
        assert "available topk: [128, 512, 1024, 2048]" in msg


def test_error_message_keeps_shape_summary_format() -> None:
    """Callers grepping for the pre-existing message shape keep working."""
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
    assert re.search(r"no decode kernel.*num_tokens=1, num_heads=16, topk=384", msg)


# Crossover-aware decode-form routing via the dispatch planner (pure Python;
# no GPU).


@pytest.fixture
def known_crossover(monkeypatch):
    """Inject a decode_max_tokens lookup without touching disk/GPU state."""
    from flashinfer.mla import _sparse_mla_sm120_cpb as cpb_mod
    from flashinfer.mla import _sparse_mla_sm120_plan as plan_mod

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
    plan_mod._plan_memo.clear()
    yield plan_mod, table
    plan_mod._plan_memo.clear()


def test_plan_uninstantiated_goes_prefill(known_crossover) -> None:
    """A shape outside the decode sets routes to a prefill variant."""
    plan_mod, _ = known_crossover
    planned = plan_mod.plan(
        32,
        64,
        2048,
        _MODEL_TYPE_DSV4,
        64,
        False,
        plan_mod._PREFILL_IMPL_AUTO,
        torch.device("cpu"),
    )
    assert planned is not None
    assert planned.variant is plan_mod.KernelVariant.PREFILL_MG


def test_plan_neither_envelope_returns_none(known_crossover) -> None:
    """A shape rejected by both envelopes plans to None (caller raises)."""
    plan_mod, _ = known_crossover
    # num_heads=48 is in no instantiation.
    assert (
        plan_mod.plan(
            8,
            48,
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


def test_plan_honors_crossover(known_crossover) -> None:
    """Instantiated shape with crossover=8: T<=8 decodes, T>8 prefills."""
    plan_mod, table = known_crossover
    table["dsv4|64|512"] = 8
    plan_mod._plan_memo.clear()
    planned = plan_mod.plan(
        8,
        64,
        512,
        _MODEL_TYPE_DSV4,
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
        64,
        512,
        _MODEL_TYPE_DSV4,
        64,
        False,
        plan_mod._PREFILL_IMPL_AUTO,
        torch.device("cpu"),
    )
    assert planned is not None and planned.variant is plan_mod.KernelVariant.PREFILL_MG


def test_plan_crossover_zero_always_prefill(known_crossover) -> None:
    """decode_max_tokens=0 (decode never wins) routes even T=1 to prefill."""
    plan_mod, table = known_crossover
    table["dsv3_2|64|2048"] = 0
    plan_mod._plan_memo.clear()
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


def test_plan_above_decode_form_cutoff(known_crossover) -> None:
    """num_tokens > 64 is prefill regardless of instantiation/crossover."""
    plan_mod, _ = known_crossover
    planned = plan_mod.plan(
        _DECODE_MAX_TOKENS + 1,
        64,
        512,
        _MODEL_TYPE_DSV4,
        64,
        False,
        plan_mod._PREFILL_IMPL_AUTO,
        torch.device("cpu"),
    )
    assert planned is not None and planned.variant is plan_mod.KernelVariant.PREFILL_MG


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


def test_plan_memo_buckets_large_t(known_crossover) -> None:
    """All T > 64 share one memo entry (the plan is T-independent there)."""
    plan_mod, _ = known_crossover
    plan_mod.plan(
        65,
        64,
        512,
        _MODEL_TYPE_DSV4,
        64,
        False,
        plan_mod._PREFILL_IMPL_AUTO,
        torch.device("cpu"),
    )
    size_after_first = len(plan_mod._plan_memo)
    plan_mod.plan(
        8192,
        64,
        512,
        _MODEL_TYPE_DSV4,
        64,
        False,
        plan_mod._PREFILL_IMPL_AUTO,
        torch.device("cpu"),
    )
    assert len(plan_mod._plan_memo) == size_after_first


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
    # (64, 2048) is neither a NOPE decode instantiation nor a NOPE prefill
    # shape (NOPE prefill serves topk=2176 only).
    assert (
        plan_mod.plan(
            4,
            64,
            2048,
            _MODEL_TYPE_GLM53_NOPE,
            64,
            False,
            plan_mod._PREFILL_IMPL_AUTO,
            torch.device("cpu"),
        )
        is None
    )


def test_plan_glm53_nope_crossover(known_crossover) -> None:
    """Injected crossover applies to the glm53_nope key space."""
    plan_mod, table = known_crossover
    table["glm53_nope|32|2176"] = 8
    plan_mod._plan_memo.clear()
    planned = plan_mod.plan(
        8,
        32,
        2176,
        _MODEL_TYPE_GLM53_NOPE,
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
        32,
        2176,
        _MODEL_TYPE_GLM53_NOPE,
        64,
        False,
        plan_mod._PREFILL_IMPL_AUTO,
        torch.device("cpu"),
    )
    assert planned is not None and planned.variant is plan_mod.KernelVariant.PREFILL_MG


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
    # topk != 576 is uninstantiated on both sides.
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


def test_plan_dots3_swa_crossover(known_crossover) -> None:
    """Injected crossover applies to the dots3_swa key space."""
    plan_mod, table = known_crossover
    table["dots3_swa|64|576"] = 8
    plan_mod._plan_memo.clear()
    planned = plan_mod.plan(
        8,
        64,
        576,
        _MODEL_TYPE_DOTS3_SWA,
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
        64,
        576,
        _MODEL_TYPE_DOTS3_SWA,
        64,
        False,
        plan_mod._PREFILL_IMPL_AUTO,
        torch.device("cpu"),
    )
    assert planned is not None and planned.variant is plan_mod.KernelVariant.PREFILL_SG


def test_sparse_mla_sm120_wrapper_public_export() -> None:
    """The runner alias is public via ``flashinfer.mla`` and is the impl class."""
    from flashinfer.mla import SparseMLASm120Wrapper
    from flashinfer.mla._sparse_mla_sm120 import _SparseMLAPagedAttentionRunner

    assert SparseMLASm120Wrapper is _SparseMLAPagedAttentionRunner
    assert "SparseMLASm120Wrapper" in dir(flashinfer.mla)
