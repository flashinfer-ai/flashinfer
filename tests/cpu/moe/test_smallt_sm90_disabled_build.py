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

import torch

from flashinfer.fused_moe import SpecializedFp8BlockSmallTRunner

DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"


def _inputs():
    return dict(
        input=torch.zeros((4, 128), dtype=torch.bfloat16),
        token_selected_experts=torch.zeros((4, 1), dtype=torch.int32),
        token_final_scales=torch.ones((4, 1)),
        fc1_expert_weights=torch.zeros((1, 256, 128), dtype=torch.float8_e4m3fn),
        fc2_expert_weights=torch.zeros((1, 128, 128), dtype=torch.float8_e4m3fn),
        output_dtype=torch.bfloat16,
        quant_scales=[torch.ones((1, 2, 1)), torch.ones((1, 1, 1))],
        output=None,
        flags={},
    )


def test_disabled_build_hook_then_eager_warmup(monkeypatch):
    # CPU unit boundary: invoke the real constructor build hook, then the
    # real dispatch/module cache. Stub device eligibility and native code.
    from contextlib import nullcontext
    from unittest.mock import Mock

    import flashinfer.fused_moe.smallt_sm90 as smallt
    import flashinfer.jit.moe_fp8_grouped_smallt_sm90 as recipe

    monkeypatch.delenv("FLASHINFER_DISABLE_MOE_FP8_GROUPED_SMALLT_SM90", raising=False)
    monkeypatch.setattr(smallt, "_announced", False)
    monkeypatch.setattr(smallt, "_counters", dict(smallt._counters))
    monkeypatch.setattr(smallt, "_fallback_reasons", {})
    monkeypatch.setattr(smallt, "_backend_dispatches", dict(smallt._backend_dispatches))
    native = Mock()
    native.plan_workspace_bytes.return_value = 64
    generate = Mock()
    generate.return_value.build_and_load.return_value = native
    monkeypatch.setattr(recipe, "gen_moe_fp8_grouped_smallt_sm90_module", generate)
    monkeypatch.setattr(smallt, "_module", None)
    monkeypatch.setattr(smallt, "_module_error", None)
    monkeypatch.setattr(smallt, "_eligibility", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    capturing = False
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)

    monkeypatch.setenv(DISABLE_ENV, "1")
    runner = object.__new__(SpecializedFp8BlockSmallTRunner)
    runner._build()
    args = _inputs()
    assert smallt._moe_smallt_sm90_dispatch(**args) is None
    generate.assert_not_called()
    assert smallt._module is None

    monkeypatch.delenv(DISABLE_ENV)
    capturing = True
    assert smallt._moe_smallt_sm90_dispatch(**args) is None
    generate.assert_not_called()
    capturing = False
    assert smallt._moe_smallt_sm90_dispatch(**args) is not None
    monkeypatch.setenv(DISABLE_ENV, "1")
    assert smallt._moe_smallt_sm90_dispatch(**args) is None
    monkeypatch.delenv(DISABLE_ENV)
    assert smallt._moe_smallt_sm90_dispatch(**args) is not None
    generate.assert_called_once_with()
    generate.return_value.build_and_load.assert_called_once_with()
    assert native.run.call_count == 2


def test_cold_capture_query_uses_input_device(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from unittest.mock import Mock

    import flashinfer.fused_moe.smallt_sm90 as smallt

    monkeypatch.delenv(DISABLE_ENV, raising=False)
    monkeypatch.delenv("FLASHINFER_DISABLE_MOE_FP8_GROUPED_SMALLT_SM90", raising=False)
    monkeypatch.setattr(smallt, "_module", None)
    monkeypatch.setattr(smallt, "_eligibility", lambda *args: None)
    monkeypatch.setattr(smallt, "_counters", dict(smallt._counters))
    monkeypatch.setattr(smallt, "_fallback_reasons", {})
    monkeypatch.setattr(smallt, "_backend_dispatches", dict(smallt._backend_dispatches))
    build = Mock(side_effect=AssertionError("must not compile during capture"))
    monkeypatch.setattr(smallt, "_get_module", build)
    current = [0]
    queried = []

    @contextmanager
    def device_guard(device):
        previous = current[0]
        current[0] = device.index
        try:
            yield
        finally:
            current[0] = previous

    def capturing():
        queried.append(current[0])
        return current[0] == 1

    monkeypatch.setattr(torch.cuda, "device", device_guard)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", capturing)
    args = _inputs()
    args["input"] = SimpleNamespace(device=torch.device("cuda:1"))
    assert smallt._moe_smallt_sm90_dispatch(**args) is None
    assert queried == [1] and current == [0]
    build.assert_not_called()
