# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for paged FP8/FP4 MQA scoring and selective TopK.

Reference implementations are adapted from TRT-LLM test scripts.
"""

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
    modules = aot.gen_all_modules(
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
    for name in ("selective_sample_ladder", "selective_repair_if"):
        assert not hasattr(aot, f"gen_{name}_module")
        assert all(name not in module.name for module in modules)


@pytest.mark.parametrize("name", ["selective_sample_ladder", "selective_repair_if"])
def test_selective_native_jit_target_gate(monkeypatch, name):
    import importlib

    module = importlib.import_module(f"flashinfer.experimental.attn_scores.jit.{name}")
    spec = Mock()
    monkeypatch.setattr(module, "gen_jit_spec", spec)
    flags = Mock(return_value=["-gencode=arch=compute_100a,code=sm_100a"])
    monkeypatch.setattr(
        module,
        "current_compilation_context",
        SimpleNamespace(get_nvcc_flags_list=flags),
    )
    if name == "selective_repair_if":
        monkeypatch.setattr(module, "get_cuda_version", lambda: Version("13.4"))
    getattr(module, f"gen_{name}_module")()
    flags.assert_called_once_with(supported_major_versions=[10])
    assert (
        "-gencode=arch=compute_100a,code=sm_100a"
        in spec.call_args.kwargs["extra_cuda_cflags"]
    )
