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

import threading

import pytest
import torch

from flashinfer.prims_ts.batched_gemm import batched_gemm_run
from flashinfer.prims_ts.moe import compile_cache
from flashinfer.prims_ts.moe import runner as moe_runner


def test_prims_ts_moe_precompile_tactics_stays_on_calling_thread(monkeypatch):
    calling_thread = threading.get_ident()
    compiled = []

    class FakeTensor:
        device = torch.device("cpu")

    class FakeInputs:
        hidden_states = FakeTensor()

    class FakeRunner:
        def _precompile_tactic_compile_only(self, _inputs, tactic, **_kwargs):
            compiled.append((tactic, threading.get_ident()))

    monkeypatch.setattr(
        moe_runner.MoeRunnerInputs, "from_list", lambda _inputs: FakeInputs()
    )

    assert moe_runner._PrimsTsMoERunnerMixin.precompile_tactics(
        FakeRunner(), [], [3, 1, 4], profile=object()
    )
    assert compiled == [(3, calling_thread), (1, calling_thread), (4, calling_thread)]


def test_prims_ts_moe_precompile_tactics_surfaces_candidate_failure(monkeypatch):
    class FakeInputs:
        hidden_states = object()

    class FakeRunner:
        def _precompile_tactic_compile_only(self, _inputs, tactic, **_kwargs):
            raise ValueError(f"broken tactic {tactic}")

    monkeypatch.setattr(
        moe_runner.MoeRunnerInputs, "from_list", lambda _inputs: FakeInputs()
    )

    with pytest.raises(RuntimeError, match=r"tactic \[32, 7\]") as exc_info:
        moe_runner._PrimsTsMoERunnerMixin.precompile_tactics(
            FakeRunner(), [], [[32, 7]], profile=object()
        )
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_compiled_gemm_cache_is_partitioned_by_device_and_arch(monkeypatch, request):
    compiled = []
    target = [(0, "sm100a")]

    def fake_compile(io, stream):
        result = object()
        compiled.append((io, stream, result))
        return result

    monkeypatch.setattr(compile_cache, "_compile_target_key", lambda _io: target[0])
    monkeypatch.setattr(batched_gemm_run, "_compile_for_launch", fake_compile)
    compile_cache._COMPILED_GEMM_CACHE.clear()
    request.addfinalizer(compile_cache._COMPILED_GEMM_CACHE.clear)

    io = {"cfg": object()}
    first = compile_cache.get_compiled_gemm("cfg", "fc1", io, "stream")
    assert compile_cache.get_compiled_gemm("cfg", "fc1", io, "stream") is first

    target[0] = (1, "sm100a")
    second = compile_cache.get_compiled_gemm("cfg", "fc1", io, "stream")
    target[0] = (1, "sm103a")
    third = compile_cache.get_compiled_gemm("cfg", "fc1", io, "stream")

    assert first is not second
    assert second is not third
    assert len(compiled) == 3
