# Copyright (c) 2026 by FlashInfer team.
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

"""CPU-only tests for PCIe IPC ahead-of-time build registration."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("add_comm", [False, True])
def test_aot_registers_pcie_ipc_modules_without_architecture_gate(
    monkeypatch, add_comm
):
    from flashinfer import aot
    from flashinfer.jit import comm as jit_comm

    generated = []

    def spec(name, sources=(), **kwargs):
        result = SimpleNamespace(name=name, sources=sources)
        generated.append(result)
        return result

    monkeypatch.setattr(aot, "gen_spdlog_module", lambda: spec("spdlog"))
    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    monkeypatch.setattr(aot, "gen_cudnn_fmha_module", lambda: spec("cudnn"))
    monkeypatch.setattr(jit_comm, "gen_comm_alltoall_module", lambda: spec("comm"))
    monkeypatch.setattr(jit_comm, "gen_vllm_comm_module", lambda: spec("vllm"))
    monkeypatch.setattr(jit_comm, "gen_jit_spec", spec)

    modules = aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [],
        [],
        {},
        add_comm=add_comm,
        add_gemma=False,
        add_oai_oss=False,
        add_moe=False,
        add_act=False,
        add_misc=False,
        add_xqa=False,
    )
    expected_sources = {
        "pcie_ipc_comm": ["pcie_ipc_all_reduce.cu"],
        "pcie_ipc_ag_rs": ["pcie_ipc_all_gather.cu", "pcie_ipc_reduce_scatter.cu"],
    }
    for name, sources in expected_sources.items():
        assert sum(module.name == name for module in generated) == int(add_comm)
        registered = [module for module in modules if module.name == name]
        assert len(registered) == int(add_comm)
        if add_comm:
            assert [source.name for source in registered[0].sources] == sources
