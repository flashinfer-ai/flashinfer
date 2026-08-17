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

CPU contracts for the calibrated SM90 push BF16 tactic selector.
"""

from __future__ import annotations

import inspect

import pytest

from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_tactics import (
    _LARGE_CLUSTER_ENABLED,
    _M128_MAX_EXPECTED_M,
    _M64_MAX_EXPECTED_M,
    _SELECTOR_CALIBRATION,
    _SELECTOR_CALIBRATION_ROUTING,
    _SELECTOR_STAGES,
    DEFAULT_BF16_GEMM_TACTIC,
    select_sm90_push_bf16_gemm_tactic,
)


_EP4_CASES = (
    ("DSV3", 64, 8, 64, 7168, 2048, "m64", 3),
    ("DSV3", 512, 8, 64, 7168, 2048, "m64", 3),
    ("DSV3", 2048, 8, 64, 7168, 2048, "dual", 3),
    ("DSV3", 8192, 8, 64, 7168, 2048, "dual", 3),
    ("DSV4_FLASH", 64, 6, 64, 4096, 2048, "m64", 3),
    ("DSV4_FLASH", 512, 6, 64, 4096, 2048, "m64", 3),
    ("DSV4_FLASH", 2048, 6, 64, 4096, 2048, "dual", 3),
    ("DSV4_FLASH", 8192, 6, 64, 4096, 2048, "dual", 3),
    ("DSV4_PRO", 64, 6, 96, 7168, 3072, "m64", 3),
    ("DSV4_PRO", 512, 6, 96, 7168, 3072, "m64", 3),
    ("DSV4_PRO", 2048, 6, 96, 7168, 3072, "m128", 3),
    ("DSV4_PRO", 8192, 6, 96, 7168, 3072, "dual", 3),
    ("KIMI_K2_6", 64, 8, 96, 7168, 2048, "m64", 3),
    ("KIMI_K2_6", 512, 8, 96, 7168, 2048, "m64", 3),
    ("KIMI_K2_6", 2048, 8, 96, 7168, 2048, "dual", 3),
    ("KIMI_K2_6", 8192, 8, 96, 7168, 2048, "dual", 3),
)


@pytest.mark.parametrize(
    (
        "geometry",
        "tokens",
        "top_k",
        "local_experts",
        "n",
        "k",
        "family_mode",
        "stages",
    ),
    _EP4_CASES,
)
def test_ep4_calibration_selects_c1_pingpong(
    geometry: str,
    tokens: int,
    top_k: int,
    local_experts: int,
    n: int,
    k: int,
    family_mode: str,
    stages: int,
) -> None:
    del geometry
    expected_m = tokens * top_k / local_experts
    tactic, reason = select_sm90_push_bf16_gemm_tactic(
        expected_m=expected_m,
        n=n,
        k=k,
        sm_count=132,
    )

    assert tactic.family_mode == family_mode
    assert {family.cluster_m for family in tactic.families} == {1}
    assert {family.schedule for family in tactic.families} == {"pingpong"}
    assert {family.stages for family in tactic.families} == {stages}
    assert (
        f"calibration={_SELECTOR_CALIBRATION}/{_SELECTOR_CALIBRATION_ROUTING}" in reason
    )


def test_ep4_calibration_pins_family_thresholds() -> None:
    assert _M64_MAX_EXPECTED_M == 64.0
    assert _M128_MAX_EXPECTED_M == 128.0
    assert _SELECTOR_STAGES == 3


def test_production_tactic_uses_c1_pingpong() -> None:
    assert DEFAULT_BF16_GEMM_TACTIC.family_mode == "dual"
    assert {family.cluster_m for family in DEFAULT_BF16_GEMM_TACTIC.families} == {1}
    assert {family.schedule for family in DEFAULT_BF16_GEMM_TACTIC.families} == {
        "pingpong"
    }
    assert {family.stages for family in DEFAULT_BF16_GEMM_TACTIC.families} == {3}


@pytest.mark.parametrize("expected_m", [96.0, 128.0, 129.0, 512.0])
def test_large_expected_m_does_not_select_c2_cooperative(expected_m: float) -> None:
    tactic, _ = select_sm90_push_bf16_gemm_tactic(
        expected_m=expected_m,
        n=7168,
        k=3072,
        sm_count=132,
    )

    assert _LARGE_CLUSTER_ENABLED is False
    assert all(family.cluster_m == 1 for family in tactic.families)
    assert all(family.schedule == "pingpong" for family in tactic.families)


def test_selection_reason_is_part_of_tactic_provenance() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_gemm import (
        Sm90PushBf16GroupedGemm,
    )

    source = inspect.getsource(Sm90PushBf16GroupedGemm.tactic_provenance)
    assert '"selection_reason": self.selection_reason' in source
