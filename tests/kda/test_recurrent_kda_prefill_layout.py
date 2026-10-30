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

"""Provider-owned Q/K packing must preserve routing and caller storage contracts."""

import importlib

import pytest
import torch

from flashinfer.kda_prefill import RecurrentKDAPrefillWorkspace
from flashinfer.utils import get_compute_capability
from tests.test_helpers.kda_prefill import packed_prefill_inputs

kda = importlib.import_module("flashinfer.kda")


def _inputs(device="cpu", *, strided=True):
    values = packed_prefill_inputs(device, seq_lens=[64], num_heads=4, seed=31)
    if strided:
        q = values["q"]
        projection = q.shape[2] * q.shape[3]
        carrier = torch.empty(
            (q.shape[1], 3 * projection), device=device, dtype=q.dtype
        )
        # GLM's post-convolution QKV carrier. V retains the caller's existing
        # compact conversion; this change only takes ownership of Q/K packing.
        q_view, k_view, _ = (
            part.view(q.shape) for part in carrier.split(projection, dim=-1)
        )
        q_view.copy_(q)
        k_view.copy_(values["k"])
        values.update(q=q_view, k=k_view)
    values["initial_state"] = torch.zeros(
        (1, 4, 128, 128), dtype=torch.bfloat16, device=device
    )
    values["output"] = torch.empty(
        values["q"].shape, dtype=torch.bfloat16, device=device
    )
    values["output_final_state"] = True
    return values


def _stub_auto_prefill(monkeypatch, seen):
    # These host-only tests control provider availability, not its scheduling
    # thresholds. A matching GPU test below executes the selected real kernel.
    monkeypatch.setattr(kda, "is_cute_dsl_available", lambda: False)
    monkeypatch.setattr(
        kda._kda_prefill, "_sm120_kda_prefill_is_eligible", lambda **kwargs: False
    )

    def eligible(**kwargs):
        seen["eligibility"] = kwargs
        return kwargs["q"].is_contiguous() and kwargs["k"].is_contiguous()

    def run(**kwargs):
        seen["launch"] = kwargs
        return kwargs["output"], kwargs["initial_state"]

    monkeypatch.setattr(
        kda._kda_prefill_cute, "_is_cute_dsl_kda_prefill_eligible", eligible
    )
    monkeypatch.setattr(kda._kda_prefill_cute, "_run_cute_dsl_kda_prefill", run)


@pytest.mark.parametrize("strided", [False, True])
def test_auto_prefill_packs_only_qk_when_needed(monkeypatch, strided):
    values = _inputs(strided=strided)
    seen = {}
    _stub_auto_prefill(monkeypatch, seen)
    result = kda.recurrent_kda(**values, backend="auto")
    bound = seen["launch"]
    for name in ("q", "k"):
        assert bound[name].is_contiguous()
        assert (bound[name] is values[name]) is (not strided)
        torch.testing.assert_close(bound[name], values[name], rtol=0, atol=0)
        assert seen["eligibility"][name] is bound[name]
    for name in ("v", "g", "beta", "initial_state", "output"):
        assert bound[name] is values[name]
    assert result[0] is values["output"]
    assert result[1] is values["initial_state"]


def test_cudnn_prefill_receives_original_qk_strides(monkeypatch):
    values = _inputs()
    seen = {}

    def run(q, k, v, g, beta, **kwargs):
        seen.update(q=q, k=k)
        return kwargs["output"], kwargs["initial_state"]

    monkeypatch.setattr(
        importlib.import_module("flashinfer.cudnn"), "cudnn_recurrent_kda", run
    )
    kda.recurrent_kda(**values, backend="cudnn")
    assert seen["q"] is values["q"]
    assert seen["k"] is values["k"]
    assert seen["q"].stride(1) == 3 * 4 * 128


@pytest.mark.parametrize("alias", ["q", "k"])
def test_prefill_rejects_original_input_output_alias_before_packing(monkeypatch, alias):
    values = _inputs()
    # A compact output view into the QKV allocation overlaps the original Q/K
    # span, but would not overlap newly packed Q/K copies.
    values["output"] = values[alias].as_strided(
        values[alias].shape, torch.empty(values[alias].shape).stride()
    )
    seen = {}
    _stub_auto_prefill(monkeypatch, seen)
    with pytest.raises(ValueError, match="output must not overlap"):
        kda.recurrent_kda(**values, backend="auto")
    assert not seen


def test_prefill_fallback_preserves_decode_qk_layout(monkeypatch):
    values = _inputs()
    monkeypatch.setattr(kda, "is_cute_dsl_available", lambda: False)
    monkeypatch.setattr(
        kda._kda_prefill, "_sm120_kda_prefill_is_eligible", lambda **kwargs: False
    )
    monkeypatch.setattr(
        kda._kda_prefill_cute,
        "_is_cute_dsl_kda_prefill_eligible",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        kda._kda_prefill, "_flash_kda_prefill_is_eligible", lambda **kwargs: False
    )
    seen = {}

    def decode(**kwargs):
        seen.update(kwargs)
        return kwargs["output"], kwargs["initial_state"]

    monkeypatch.setattr(kda._kda_decode, "_dispatch_recurrent_kda_decode", decode)
    kda.recurrent_kda(**values, backend="auto")
    assert seen["q"] is values["q"]
    assert seen["k"] is values["k"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_auto_prefill_strided_qk_capture_reads_changed_inputs():
    device = torch.device("cuda")
    if get_compute_capability(device) not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100/SM103 KDA prefill")
    values = _inputs(device)
    state_seed = values["initial_state"].clone()
    reference_state = state_seed.clone()
    reference_output = torch.empty_like(values["output"])
    workspace = RecurrentKDAPrefillWorkspace(device=device)
    reference_workspace = RecurrentKDAPrefillWorkspace(device=device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        kda.recurrent_kda(**values, prefill_workspace=workspace, backend="auto")
        stream.synchronize()
        values["initial_state"].copy_(state_seed)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = kda.recurrent_kda(
                **values, prefill_workspace=workspace, backend="auto"
            )
        # The facade's packed Q/K temporaries have no Python owner now. Replay
        # must retain their captured storage and recopy the current input views.
        for delta in (0.0, 0.125):
            values["q"].add_(delta)
            values["k"].sub_(delta)
            reference_state.copy_(state_seed)
            reference = dict(
                values,
                q=values["q"].contiguous(),
                k=values["k"].contiguous(),
                initial_state=reference_state,
                output=reference_output,
            )
            expected = kda.recurrent_kda(
                **reference, prefill_workspace=reference_workspace, backend="auto"
            )
            churn = [
                torch.empty_like(reference["q"]).fill_(float("nan")) for _ in range(4)
            ]
            values["initial_state"].copy_(state_seed)
            values["output"].fill_(float("nan"))
            graph.replay()
            stream.synchronize()
            for actual, target in zip(result, expected, strict=True):
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, target, rtol=1e-2, atol=1e-2)
            del churn
