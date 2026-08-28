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
    _DECODE_MAX_TOKENS,
    _MODEL_TYPE_DSV3_2,
    _MODEL_TYPE_DSV4,
    _MODEL_TYPE_GLM_NSA,
    _decode_dispatch_error_message,
    _decode_dsv3_2_dispatchable,
    _decode_dsv4_dispatchable,
)


def test_supported_configs_families() -> None:
    """The query API mirrors the private dispatch tables exactly."""
    configs = supported_sparse_mla_sm120_configs()
    assert set(configs) == {"dsv4", "dsv3_2", "glm_nsa"}
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


# Crossover-aware decode-form routing (pure Python; no GPU).


@pytest.fixture
def known_crossover(monkeypatch):
    """Inject a decode_max_tokens lookup without touching disk state."""
    import flashinfer.mla._sparse_mla_sm120 as sm
    from flashinfer.mla import _sparse_mla_sm120_cpb as cpb_mod

    table = {}
    monkeypatch.setattr(
        cpb_mod,
        "get_decode_max_tokens",
        lambda device, family, num_heads, topk: table.get(
            f"{family}|{num_heads}|{topk}"
        ),
    )
    return sm, table


def test_route_decode_form_uninstantiated_goes_prefill(known_crossover) -> None:
    """A shape outside the decode sets routes to prefill, not an error."""
    sm, _ = known_crossover
    assert not sm._route_decode_form(
        _MODEL_TYPE_DSV4, 64, 2048, 64, 32, False, torch.device("cpu")
    )
    assert not sm._route_decode_form(
        _MODEL_TYPE_DSV3_2, 48, 2048, 64, 8, False, torch.device("cpu")
    )


def test_route_decode_form_unknown_crossover_keeps_decode(known_crossover) -> None:
    """Instantiated shape, no calibration: the decode-first default holds."""
    sm, _ = known_crossover
    for num_tokens in (1, _DECODE_MAX_TOKENS):
        assert sm._route_decode_form(
            _MODEL_TYPE_DSV4, 64, 512, 64, num_tokens, False, torch.device("cpu")
        )


def test_route_decode_form_honors_crossover(known_crossover) -> None:
    """Instantiated shape with crossover=8: T<=8 decodes, T>8 prefills."""
    sm, table = known_crossover
    table["dsv4|64|512"] = 8
    assert sm._route_decode_form(
        _MODEL_TYPE_DSV4, 64, 512, 64, 8, False, torch.device("cpu")
    )
    assert not sm._route_decode_form(
        _MODEL_TYPE_DSV4, 64, 512, 64, 16, False, torch.device("cpu")
    )


def test_route_decode_form_crossover_zero_always_prefill(known_crossover) -> None:
    """decode_max_tokens=0 (decode never wins) routes even T=1 to prefill."""
    sm, table = known_crossover
    table["dsv3_2|64|2048"] = 0
    assert not sm._route_decode_form(
        _MODEL_TYPE_DSV3_2, 64, 2048, 64, 1, False, torch.device("cpu")
    )


def test_route_decode_form_above_decode_form_cutoff(known_crossover) -> None:
    """num_tokens > 64 is prefill regardless of instantiation/crossover."""
    sm, _ = known_crossover
    assert not sm._route_decode_form(
        _MODEL_TYPE_DSV4,
        64,
        512,
        64,
        _DECODE_MAX_TOKENS + 1,
        False,
        torch.device("cpu"),
    )


def test_route_decode_form_page_block_size_mismatch(known_crossover) -> None:
    """A non-64 page size is decode-uninstantiated and routes to prefill."""
    sm, _ = known_crossover
    assert not sm._route_decode_form(
        _MODEL_TYPE_DSV4, 64, 512, 32, 8, False, torch.device("cpu")
    )
