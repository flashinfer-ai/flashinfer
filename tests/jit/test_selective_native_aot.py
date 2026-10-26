# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from packaging.version import Version


@pytest.mark.parametrize(
    "target", ["sm80", "sm90", "sm100", "sm100f", "sm103", "sm107", "sm120"]
)
@pytest.mark.parametrize("cuda_version", ["12.3", "12.8"])
def test_selective_native_aot_target_gate(monkeypatch, target, cuda_version):
    from flashinfer import aot

    for name in dir(aot):
        if name.startswith("gen_") and name != "gen_all_modules":
            monkeypatch.setattr(
                aot, name, Mock(return_value=SimpleNamespace(name=name))
            )
    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    monkeypatch.setattr(aot, "get_cuda_version", lambda: Version(cuda_version))
    aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [],
        [],
        {target: True},
        add_comm=False,
        add_gemma=False,
        add_oai_oss=False,
        add_moe=False,
        add_act=False,
        add_misc=True,
        add_xqa=False,
    )
    supported = target in {"sm100", "sm100f", "sm103", "sm107"}
    assert aot.gen_selective_sample_ladder_module.call_count == int(supported)
    assert aot.gen_selective_repair_if_module.call_count == int(
        supported and Version(cuda_version) >= Version("12.4")
    )
