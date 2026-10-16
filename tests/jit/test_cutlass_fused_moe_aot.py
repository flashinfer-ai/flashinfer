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
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    ("capabilities", "expected_count"),
    [
        pytest.param({"sm110": True}, 1, id="sm110-only"),
        pytest.param({"sm100": True}, 1, id="sm100-only"),
        pytest.param({"sm100": True, "sm110": True}, 1, id="sm100-and-sm110"),
        pytest.param({"sm80": True}, 0, id="unsupported-target"),
    ],
)
@pytest.mark.parametrize("add_moe", [False, True])
def test_aot_registers_cutlass_fused_moe_for_sm110(
    monkeypatch, capabilities, expected_count, add_moe
):
    from flashinfer import aot

    # Exercise registration without generating sources or probing CUDA.
    for name in dir(aot):
        if name.startswith("gen_") and name != "gen_all_modules":
            monkeypatch.setattr(
                aot, name, Mock(return_value=SimpleNamespace(name=name))
            )
    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    fused_moe = aot.gen_cutlass_fused_moe_sm100_module
    fused_moe.return_value = SimpleNamespace(name="fused_moe_100")

    specs = aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [],
        [],
        capabilities,
        add_comm=False,
        add_gemma=False,
        add_oai_oss=False,
        add_moe=add_moe,
        add_act=False,
        add_misc=False,
        add_xqa=False,
    )

    expected_count = expected_count if add_moe else 0
    # Check calls too: gen_all_modules deduplicates its returned specs by name.
    assert fused_moe.call_count == expected_count
    assert [spec.name for spec in specs].count("fused_moe_100") == expected_count
