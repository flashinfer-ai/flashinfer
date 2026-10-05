# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU compatibility contracts; GPU numerics live in the adjacent KDA suite."""

import importlib.util
import sys
from types import SimpleNamespace

import pytest
import torch

from flashinfer.cudnn import linear_attention


@pytest.fixture
def adapter(monkeypatch):
    if not linear_attention.CUDNN_AVAILABLE:
        pytest.skip("requires the cudnn-frontend python package")
    # A fresh module gives each compatibility scenario its own graph cache.
    name = "flashinfer.cudnn._test_linear_attention_contract"
    spec = importlib.util.spec_from_file_location(name, linear_attention.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_device_index", lambda device: 0)
    monkeypatch.setattr(module, "_get_cache_buf", lambda *args: torch.empty(0))
    monkeypatch.setattr(module, "_create_cudnn_handle", lambda stream: stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: "cpu-test")
    return module


@pytest.mark.parametrize(
    "version,ordered,decline",
    [
        ("1.29.0", False, None),
        ("1.30.0", True, None),
        ("1.30.0", True, "NotImplementedError"),
        ("1.30.0", True, "cudnnGraphNotSupportedError"),
    ],
)
@pytest.mark.parametrize("return_state", [False, True])
def test_kda_state_update_survives_older_or_declining_frontend(
    monkeypatch, adapter, version, ordered, decline, return_state
):
    monkeypatch.setattr(adapter.cudnn, "__version__", version)
    uid = adapter.UIDs
    rejected = []

    def create(family, q, k, v, g, beta, cu, out, **kwargs):
        overwrite = kwargs.get("overwrite_initial_state", False)
        if version == "1.29.0" and overwrite:
            raise AssertionError("overwrite was requested from an older frontend")
        if overwrite and decline is not None:
            rejected.append(True)
            error = (
                NotImplementedError
                if decline == "NotImplementedError"
                else adapter.cudnn.cudnnGraphNotSupportedError
            )
            raise error("the plan cannot overwrite its incoming state")

        def compute(pack):
            state = pack[uid.INITIAL_STATE_UID.value]
            final = pack[uid.FINAL_STATE_UID.value]
            if not overwrite:
                assert state.data_ptr() != final.data_ptr(), "fallback needs scratch"
            # A small CPU operation tests binding and copy-back, not KDA numerics.
            pack[uid.O_UID.value].copy_(pack[uid.Q_UID.value] + pack[uid.V_UID.value])
            final.copy_(state + 1)

        if ordered:

            def execute(buffers, workspace, handle, tensor_uids):
                compute(dict(zip(tensor_uids, buffers, strict=True)))
        else:

            def execute(pack, workspace, handle):
                compute(pack)

        return SimpleNamespace(
            execute=execute,
            get_workspace_size=lambda: 0,
            _fi_la_uids=tuple(
                value.value
                for value in (
                    uid.Q_UID,
                    uid.K_UID,
                    uid.V_UID,
                    uid.G_UID,
                    uid.BETA_UID,
                    uid.CU_SEQLENS_UID,
                    uid.O_UID,
                    uid.INITIAL_STATE_UID,
                    uid.FINAL_STATE_UID,
                )
            ),
        ), []

    monkeypatch.setattr(adapter, "_create_la_graph", create)
    retained = []
    for value in (0.25, -0.5):
        q = torch.full((7, 2, 8), value)
        k, v, g = q.clone(), q.clone(), q.clone()
        beta = torch.ones(7, 2)
        state = torch.full((1, 2, 8, 8), value)
        out = torch.empty_like(q)
        result, final = adapter.cudnn_recurrent_kda(
            q,
            k,
            v,
            g,
            beta,
            initial_state=state,
            output_final_state=return_state,
            output=out,
            cu_seqlens=torch.tensor([0, 7], dtype=torch.int32),
        )
        assert result.data_ptr() == out.data_ptr()
        assert (final is not None) == return_state
        if return_state:
            assert final.data_ptr() == state.data_ptr()
        torch.testing.assert_close(out, q + v)
        torch.testing.assert_close(state, torch.full_like(state, value + 1))
        retained.append((q, k, v, g, beta, state, out))
    if decline is not None:
        assert rejected, "the simulated unsupported overwrite path was not exercised"


def test_kda_does_not_hide_an_unexpected_graph_build_failure(monkeypatch, adapter):
    monkeypatch.setattr(adapter.cudnn, "__version__", "1.30.0")

    def invalid_graph(*args, **kwargs):
        raise ValueError("invalid graph descriptor")

    monkeypatch.setattr(adapter, "_create_la_graph", invalid_graph)
    q = torch.ones(7, 2, 8)
    with pytest.raises(ValueError, match="invalid graph descriptor"):
        adapter.cudnn_recurrent_kda(
            q,
            q,
            q,
            q,
            torch.ones(7, 2),
            initial_state=torch.zeros(1, 2, 8, 8),
            cu_seqlens=torch.tensor([0, 7], dtype=torch.int32),
        )


@pytest.mark.parametrize("inference_mode", [False, True])
def test_kda_preserves_inference_tensor_mutation_rules(
    monkeypatch, adapter, inference_mode
):
    def execute(*args, final_state, **kwargs):
        # Model a raw kernel write: it does not perform copy_'s caller-side
        # inference-mode check. The adapter must preserve that check itself.
        with torch.inference_mode():
            final_state.fill_(2)
        return final_state

    monkeypatch.setattr(adapter, "_run_la_graph", execute)
    q = torch.ones(7, 2, 8)
    with torch.inference_mode():
        state = torch.zeros(1, 2, 8, 8)
    kwargs = dict(
        initial_state=state,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, 7], dtype=torch.int32),
    )
    with torch.inference_mode(inference_mode):
        if inference_mode:
            _, final = adapter.cudnn_recurrent_kda(
                q, q, q, q, torch.ones(7, 2), **kwargs
            )
            assert final.data_ptr() == state.data_ptr()
            torch.testing.assert_close(state, torch.full_like(state, 2))
        else:
            with pytest.raises(RuntimeError, match="[Ii]nference[Mm]ode"):
                adapter.cudnn_recurrent_kda(q, q, q, q, torch.ones(7, 2), **kwargs)
