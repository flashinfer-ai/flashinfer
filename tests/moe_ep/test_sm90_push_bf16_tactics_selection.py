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
    _FAMILY_M128_MAX_EXPECTED_M,
    _FAMILY_M64_MAX_EXPECTED_M,
    _SELECTOR_CALIBRATION,
    _SELECTOR_CALIBRATION_ROUTING,
    DEFAULT_BF16_GEMM_TACTIC,
    select_sm90_push_bf16_gemm_tactic,
)


_EP4_CASES = (
    ("DSV3_T64", 8.0, 4096, 7168, "m64", 2),
    ("DSV3_T512", 64.0, 4096, 7168, "m64", 3),
    ("DSV3_T2048", 256.0, 4096, 7168, "dual", 3),
    ("DSV3_T8192", 1024.0, 4096, 7168, "dual", 3),
    ("DSV4_FLASH_T64", 6.0, 4096, 4096, "m64", 2),
    ("DSV4_FLASH_T512", 48.0, 4096, 4096, "m64", 3),
    ("DSV4_FLASH_T2048", 192.0, 4096, 4096, "dual", 3),
    ("DSV4_FLASH_T8192", 768.0, 4096, 4096, "dual", 3),
    ("DSV4_PRO_T64", 4.0, 6144, 7168, "m64", 2),
    ("DSV4_PRO_T512", 32.0, 6144, 7168, "m64", 2),
    ("DSV4_PRO_T2048", 128.0, 6144, 7168, "m128", 3),
    ("DSV4_PRO_T8192", 512.0, 6144, 7168, "dual", 3),
    ("KIMI_K2_6_T64", 64.0 * 8 / 96, 4096, 7168, "m64", 2),
    ("KIMI_K2_6_T512", 512.0 * 8 / 96, 4096, 7168, "m64", 3),
    ("KIMI_K2_6_T2048", 2048.0 * 8 / 96, 4096, 7168, "dual", 3),
    ("KIMI_K2_6_T8192", 8192.0 * 8 / 96, 4096, 7168, "dual", 3),
)


@pytest.mark.parametrize(
    (
        "geometry",
        "expected_m",
        "n",
        "k",
        "family_mode",
        "stages",
    ),
    _EP4_CASES,
)
def test_selector_matches_ep4_calibration(
    geometry: str,
    expected_m: float,
    n: int,
    k: int,
    family_mode: str,
    stages: int,
) -> None:
    del geometry
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
    assert _FAMILY_M64_MAX_EXPECTED_M == 64.0
    assert _FAMILY_M128_MAX_EXPECTED_M == 128.0


def test_production_tactic_uses_c1_pingpong() -> None:
    assert DEFAULT_BF16_GEMM_TACTIC.family_mode == "dual"
    assert {family.cluster_m for family in DEFAULT_BF16_GEMM_TACTIC.families} == {1}
    assert {family.schedule for family in DEFAULT_BF16_GEMM_TACTIC.families} == {
        "pingpong"
    }
    assert {family.stages for family in DEFAULT_BF16_GEMM_TACTIC.families} == {3}


def test_large_cluster_never_selected() -> None:
    for expected_m in (0.0, 32.0, 64.0, 96.0, 128.0, 129.0, 512.0):
        for n in (64, 2048, 4096, 7168):
            for k in (64, 2048, 3072, 7168):
                tactic, _ = select_sm90_push_bf16_gemm_tactic(
                    expected_m=expected_m,
                    n=n,
                    k=k,
                    sm_count=132,
                )
                assert all(family.cluster_m == 1 for family in tactic.families)
                assert all(family.schedule == "pingpong" for family in tactic.families)


def test_selection_reason_is_part_of_tactic_provenance() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_gemm import (
        Sm90PushBf16GroupedGemm,
    )

    source = inspect.getsource(Sm90PushBf16GroupedGemm.tactic_provenance)
    assert '"selection_reason": self.selection_reason' in source
