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

import importlib
import math
import threading
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from packaging.version import Version

import flashinfer
from flashinfer.kda import RecurrentKDAPrefillWrapper, recurrent_kda
from flashinfer.kda_prefill import RecurrentKDAPrefillWorkspace
from flashinfer.utils import get_compute_capability

kda_decode_api = importlib.import_module("flashinfer.kda_decode")
kda_api = importlib.import_module("flashinfer.kda")
kda_prefill_api = importlib.import_module("flashinfer.kda_prefill")
kda_prefill_cute_api = importlib.import_module("flashinfer.kda_prefill_cute")


def test_public_api_uses_phase_neutral_facade_and_prefill_workspace():
    assert flashinfer.recurrent_kda is kda_api.recurrent_kda
    assert (
        flashinfer.RecurrentKDAPrefillWorkspace
        is kda_prefill_api.RecurrentKDAPrefillWorkspace
    )
    assert flashinfer.RecurrentKDAPrefillWrapper is RecurrentKDAPrefillWrapper


def test_prefill_wrapper_plan_builds_stable_device_metadata(cuda_device):
    wrapper = RecurrentKDAPrefillWrapper(cuda_device)
    wrapper.plan(torch.tensor([0, 0, 7, 7, 12], device=cuda_device))

    cu_seqlens_ptr = wrapper._cu_seqlens_buf.data_ptr()
    seq_order_ptr = wrapper._seq_order_buf.data_ptr()
    cu_chunks_ptr = wrapper._cu_chunks_buf.data_ptr()
    assert wrapper._cu_seqlens_buf.dtype == torch.int64
    assert wrapper._cu_seqlens_buf.tolist() == [0, 0, 7, 7, 12]
    assert wrapper._seq_order_buf.tolist() == [1, 3, 0, 2]
    assert wrapper._cu_chunks_buf.tolist() == [0, 0, 1, 1, 2]
    assert wrapper._workspace._cute_dsl_total_chunks == 2

    wrapper.plan(torch.tensor([0, 0, 2, 2, 12], device=cuda_device))
    assert wrapper._cu_seqlens_buf.data_ptr() == cu_seqlens_ptr
    assert wrapper._seq_order_buf.data_ptr() == seq_order_ptr
    assert wrapper._cu_chunks_buf.data_ptr() == cu_chunks_ptr
    assert wrapper._seq_order_buf.tolist() == [3, 1, 0, 2]

    with pytest.raises(ValueError, match="total token count is fixed"):
        wrapper.plan(torch.tensor([0, 0, 2, 2, 13], device=cuda_device))

    with pytest.raises(ValueError, match="number of sequences is fixed"):
        wrapper.plan(torch.tensor([0, 2, 12], device=cuda_device))

    chunk_wrapper = RecurrentKDAPrefillWrapper(cuda_device)
    chunk_wrapper.plan(torch.tensor([0, 16, 16, 32], device=cuda_device))
    with pytest.raises(ValueError, match="chunk count is fixed"):
        chunk_wrapper.plan(torch.tensor([0, 1, 17, 32], device=cuda_device))

    with pytest.raises(ValueError, match="non-decreasing"):
        RecurrentKDAPrefillWrapper(cuda_device).plan(
            torch.tensor([0, 2, 1, 12], device=cuda_device)
        )


def test_prefill_wrapper_run_forwards_planned_buffers(cuda_device, monkeypatch):
    wrapper = RecurrentKDAPrefillWrapper(cuda_device)
    wrapper.plan(torch.tensor([0, 1, 3], device=cuda_device))
    calls = []
    sentinel = (object(), object())
    monkeypatch.setattr(
        kda_api,
        "recurrent_kda",
        lambda **kwargs: calls.append(kwargs) or sentinel,
    )
    tensors = _cpu_route_tensors(token_count=3)
    tensors = {
        key: value.to(cuda_device) if isinstance(value, torch.Tensor) else value
        for key, value in tensors.items()
    }

    assert wrapper.run(**tensors) is sentinel
    assert calls[0]["cu_seqlens"] is wrapper._cu_seqlens_buf
    assert calls[0]["seq_order"] is wrapper._seq_order_buf
    assert calls[0]["prefill_workspace"] is wrapper._workspace
    assert calls[0]["backend"] == "cute-dsl"
    assert wrapper._workspace._cute_dsl_cu_chunks is wrapper._cu_chunks_buf
    assert wrapper._workspace._cute_dsl_total_chunks == 2


def _cpu_route_tensors(token_count=2):
    shape = (1, token_count, 1, 128)
    return {
        "q": torch.empty(shape, dtype=torch.bfloat16),
        "k": torch.empty(shape, dtype=torch.bfloat16),
        "v": torch.empty(shape, dtype=torch.bfloat16),
        "g": torch.empty(shape, dtype=torch.bfloat16),
        "beta": torch.empty((1, token_count, 1), dtype=torch.bfloat16),
        "A_log": torch.empty(1, dtype=torch.float32),
        "dt_bias": torch.empty((1, 128), dtype=torch.float32),
        "use_gate_in_kernel": True,
        "lower_bound": -5.0,
        "beta_is_logit": True,
    }


def test_public_prefill_backend_option_routes_to_cute_dsl(monkeypatch):
    sentinel = (object(), object())
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_is_cute_dsl_kda_prefill_eligible",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_run_cute_dsl_kda_prefill",
        lambda **kwargs: sentinel,
    )

    assert recurrent_kda(**_cpu_route_tensors(), backend="cute-dsl") is sentinel


def test_public_prefill_auto_prefers_cute_dsl(monkeypatch):
    sentinel = (object(), object())
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_is_cute_dsl_kda_prefill_eligible",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_run_cute_dsl_kda_prefill",
        lambda **kwargs: sentinel,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_prefill_is_eligible",
        lambda **kwargs: pytest.fail("auto should not probe Cake after a CuTe match"),
    )

    assert recurrent_kda(**_cpu_route_tensors()) is sentinel


def test_public_prefill_forwards_sequence_order_to_cute_dsl(monkeypatch):
    calls = []
    sentinel = (object(), object())
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_is_cute_dsl_kda_prefill_eligible",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_run_cute_dsl_kda_prefill",
        lambda **kwargs: calls.append(kwargs) or sentinel,
    )

    seq_order = torch.tensor([1, 0], dtype=torch.int32)
    assert (
        recurrent_kda(
            **_cpu_route_tensors(token_count=3),
            cu_seqlens=torch.tensor([0, 1, 3], dtype=torch.int64),
            seq_order=seq_order,
        )
        is sentinel
    )
    assert calls[0]["seq_order"] is seq_order


def test_public_prefill_auto_falls_back_to_cake(monkeypatch):
    sentinel = (object(), object())
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_is_cute_dsl_kda_prefill_eligible",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_prefill_is_eligible",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_run_flash_kda_prefill",
        lambda **kwargs: sentinel,
    )

    assert recurrent_kda(**_cpu_route_tensors()) is sentinel


def test_public_prefill_explicit_cake_skips_cute_dsl_probe_with_checkpoints(
    monkeypatch,
):
    sentinel = (object(), object(), object())
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_is_cute_dsl_kda_prefill_eligible",
        lambda **kwargs: pytest.fail("backend='cake' must not probe CuTe DSL"),
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_prefill_is_eligible",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_run_flash_kda_prefill",
        lambda **kwargs: sentinel,
    )

    checkpoint_state = torch.empty((1, 1, 128, 128), dtype=torch.bfloat16)
    checkpoint_starts = torch.tensor([0, 1], dtype=torch.int64)
    assert (
        recurrent_kda(
            **_cpu_route_tensors(),
            state_checkpoints=checkpoint_state,
            checkpoint_cu_starts=checkpoint_starts,
            checkpoint_every_n_tokens=32,
            backend="cake",
        )
        is sentinel
    )


def test_public_prefill_auto_routes_supported_checkpoints_to_cute_dsl(monkeypatch):
    calls = []
    sentinel = (object(), object(), object())
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_is_cute_dsl_kda_prefill_eligible",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_run_cute_dsl_kda_prefill",
        lambda **kwargs: calls.append(kwargs) or sentinel,
    )

    checkpoints = torch.empty((1, 1, 128, 128), dtype=torch.bfloat16)
    starts = torch.tensor([0, 1], dtype=torch.int64)
    assert (
        recurrent_kda(
            **_cpu_route_tensors(),
            state_checkpoints=checkpoints,
            checkpoint_cu_starts=starts,
            checkpoint_every_n_tokens=32,
        )
        is sentinel
    )
    assert calls[0]["state_checkpoints"] is checkpoints
    assert calls[0]["checkpoint_cu_starts"] is starts
    assert calls[0]["checkpoint_every_n_tokens"] == 32


def test_public_prefill_cake_backend_is_strict(monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_prefill_is_eligible",
        lambda **kwargs: False,
    )

    with pytest.raises(ValueError, match="backend='cake' does not support"):
        recurrent_kda(**_cpu_route_tensors(), backend="cake")


def test_public_decode_backend_option_forwards_to_decode_layer(monkeypatch):
    calls = []
    sentinel = (object(), object())

    def run(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(kda_decode_api, "_run_recurrent_kda", run)
    assert (
        recurrent_kda(**_cpu_route_tensors(token_count=1), backend="cake") is sentinel
    )
    assert calls[0]["backend"] == "cake"


def test_public_backend_option_rejects_unknown_value():
    with pytest.raises(ValueError, match="backend must be"):
        recurrent_kda(**_cpu_route_tensors(), backend="unknown")


def test_cute_dsl_prefill_adapter_preserves_indexed_in_place_state_semantics(
    monkeypatch,
):
    calls = []
    compile_args = []
    identity_order = torch.tensor([0], dtype=torch.int32)

    class Compiled:
        def workspace_size(self, cu_seqlens, heads, **kwargs):
            assert cu_seqlens is None
            assert heads == 1
            assert kwargs == {"batch": 1, "seqlen": 2}
            return 0

        def __call__(self, *args, **kwargs):
            calls.append((args, kwargs))

    def get_compiled(**kwargs):
        compile_args.append(kwargs)
        return Compiled()

    monkeypatch.setattr(
        kda_prefill_cute_api, "_get_compiled_cute_dsl_kda", get_compiled
    )
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_identity_seq_order",
        lambda **kwargs: identity_order,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device=None: SimpleNamespace(cuda_stream=7)
    )

    inputs = _cpu_route_tensors()
    state = torch.empty((3, 1, 128, 128), dtype=torch.bfloat16)
    state_indices = torch.tensor([2], dtype=torch.int32)
    output = torch.empty_like(inputs["q"])
    result = kda_prefill_cute_api._run_cute_dsl_kda_prefill(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        scale=None,
        initial_state=state,
        output_final_state=False,
        lower_bound=-5.0,
        cu_seqlens=None,
        seq_order=None,
        output=output,
        prefill_workspace=None,
        state_checkpoints=None,
        checkpoint_cu_starts=None,
        checkpoint_every_n_tokens=0,
        state_indices=state_indices,
    )

    assert result[0] is output
    assert result[1] is None
    assert compile_args == [
        {
            "lower_bound": -5.0,
            "has_state_in": True,
            "has_state_out": True,
            "has_state_ckpt": False,
            "has_state_indices": True,
        }
    ]
    args, kwargs = calls[0]
    assert args[8] is state
    assert args[10] is state
    assert args[11] is None
    assert args[12] == 7
    assert kwargs == {
        "seq_order": identity_order,
        "state_indices": state_indices,
        "planned_cu_chunks": None,
        "planned_total_chunks": None,
    }


@pytest.mark.parametrize("explicit_order", [False, True])
def test_cute_dsl_prefill_adapter_forwards_packed_sequence_order(
    monkeypatch, explicit_order
):
    calls = []

    class Compiled:
        def workspace_size(self, cu_seqlens, heads, **kwargs):
            assert cu_seqlens.tolist() == [0, 1, 2]
            assert heads == 1
            assert kwargs == {}
            return 0

        def __call__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_get_compiled_cute_dsl_kda",
        lambda **kwargs: Compiled(),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device=None: SimpleNamespace(cuda_stream=7)
    )

    inputs = _cpu_route_tensors()
    output = torch.empty_like(inputs["q"])
    cu_seqlens = torch.tensor([0, 1, 2], dtype=torch.int64)
    seq_order = torch.tensor([1, 0], dtype=torch.int32) if explicit_order else None
    result = kda_prefill_cute_api._run_cute_dsl_kda_prefill(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        scale=None,
        initial_state=None,
        output_final_state=False,
        lower_bound=-5.0,
        cu_seqlens=cu_seqlens,
        seq_order=seq_order,
        output=output,
        prefill_workspace=None,
        state_checkpoints=None,
        checkpoint_cu_starts=None,
        checkpoint_every_n_tokens=0,
    )

    assert result[0] is output
    assert result[1] is None
    args, kwargs = calls[0]
    assert args[7] is cu_seqlens
    assert set(kwargs) == {
        "state_indices",
        "seq_order",
        "planned_cu_chunks",
        "planned_total_chunks",
    }
    assert kwargs["seq_order"] is seq_order
    assert kwargs["state_indices"] is None
    assert kwargs["planned_cu_chunks"] is None
    assert kwargs["planned_total_chunks"] is None


def test_cute_dsl_lpt_sequence_order_is_content_cached(monkeypatch):
    kernel_module = importlib.import_module("flashinfer.kda_kernels.kda_chunked_bt16")
    monkeypatch.setattr(kernel_module, "_CU_CONTENTS_MEMO", {})
    monkeypatch.setattr(kernel_module, "_LPT_SEQUENCE_ORDER_CACHE", {})
    cu_seqlens = torch.tensor(
        [0, 1300, 1847, 3895, 4858, 5129, 8192], dtype=torch.int64
    )

    first = kernel_module._lpt_sequence_order(cu_seqlens)
    second = kernel_module._lpt_sequence_order(cu_seqlens.clone())

    assert first.tolist() == [5, 2, 0, 3, 1, 4]
    assert second.data_ptr() == first.data_ptr()


def test_cute_dsl_unplanned_packed_engine_rejects_graph_capture(monkeypatch):
    kernel_module = importlib.import_module("flashinfer.kda_kernels.kda_chunked_bt16")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(kernel_module, "_device_sm_count", lambda device: 148)
    monkeypatch.setattr(
        kernel_module,
        "_route_for_workspace",
        lambda n_seq, heads, device, mode: "engine",
    )
    compiled = kernel_module._make_call(
        lambda *args, **kwargs: None,
        {
            "mode": None,
            "dtype": object(),
            "state_dtype": object(),
            "gate_dtype": object(),
            "safe_gate": True,
            "gate_lower_bound": -5.0,
            "has_state_in": False,
            "has_state_out": False,
            "has_state_ckpt": False,
            "has_state_indices": False,
        },
    )
    inputs = _cpu_route_tensors()
    cu_seqlens = torch.tensor([0, 1, 2], dtype=torch.int64)

    with pytest.raises(RuntimeError, match=r"Wrapper\.plan\(\)"):
        compiled(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["A_log"],
            inputs["dt_bias"],
            inputs["beta"],
            cu_seqlens,
            None,
            torch.empty_like(inputs["q"]),
            None,
            torch.empty(0, dtype=torch.uint8),
            0,
        )


def test_cute_dsl_engine_workspace_query_does_not_read_device_offsets(monkeypatch):
    kernel_module = importlib.import_module("flashinfer.kda_kernels.kda_chunked_bt16")
    monkeypatch.setattr(kernel_module, "_device_sm_count", lambda device: 148)
    monkeypatch.setattr(
        kernel_module,
        "_cu_seqlens_contents",
        lambda tensor: pytest.fail("engine workspace query must not read offsets"),
    )

    cu_seqlens = torch.tensor([0, 3, 7, 12, 18, 25], dtype=torch.int64)
    assert kernel_module.workspace_size(cu_seqlens, heads=64) == 0


def _strict_prefill_kwargs(inputs):
    return {
        **inputs,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "lower_bound": -5.0,
        "beta_is_logit": True,
    }


def _make_inputs(
    *,
    seq_lens,
    num_heads: int,
    packed: bool,
    initial_state: bool = False,
    seed: int = 0,
):
    torch.manual_seed(seed)
    if packed:
        batch_size = 1
        seq_len = sum(seq_lens)
    else:
        if len(set(seq_lens)) != 1:
            raise ValueError("fixed test inputs require equal sequence lengths")
        batch_size = len(seq_lens)
        seq_len = seq_lens[0]
    shape = (batch_size, seq_len, num_heads, 128)
    q = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    g = (0.1 * torch.randn(shape, dtype=torch.float32, device="cuda")).to(
        torch.bfloat16
    )
    beta = torch.randn(
        (batch_size, seq_len, num_heads),
        dtype=torch.bfloat16,
        device="cuda",
    )
    A_log = 0.1 * torch.randn(num_heads, dtype=torch.float32, device="cuda")
    dt_bias = 0.1 * torch.randn((num_heads, 128), dtype=torch.float32, device="cuda")
    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)
    state = None
    if initial_state:
        state = (
            0.1
            * torch.randn(
                (len(seq_lens), num_heads, 128, 128),
                dtype=torch.float32,
                device="cuda",
            )
        ).to(torch.bfloat16)
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "initial_state": state,
        "cu_seqlens": (
            torch.tensor(offsets, dtype=torch.int64, device="cuda") if packed else None
        ),
    }


@pytest.mark.parametrize(
    "field",
    (
        "beta",
        "cu_seqlens",
        "seq_order",
        "ssm_state_indices",
        "initial_state",
        "output",
    ),
)
def test_public_prefill_auto_falls_back_for_non_tensor_arguments(
    flash_kda_device,
    monkeypatch,
    field,
):
    with torch.cuda.device(flash_kda_device):
        inputs = _make_inputs(
            seq_lens=[3, 5],
            num_heads=1,
            packed=True,
            initial_state=True,
        )
    kwargs = _strict_prefill_kwargs(inputs)
    kwargs[field] = object()
    sentinel = (object(), object())
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_prefill_is_eligible",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_run_cute_dsl_kda_prefill",
        lambda **kwargs: pytest.fail("ineligible call must not run CuTe DSL"),
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_run_flash_kda_prefill",
        lambda **kwargs: sentinel,
    )

    assert recurrent_kda(**kwargs) is sentinel


def _reference(inputs, *, lower_bound=-5.0, scale=None):
    q = inputs["q"]
    batch_size, seq_len, num_heads, head_dim = q.shape
    scale = head_dim**-0.5 if scale is None else scale
    q_flat = F.normalize(q.float(), dim=-1).reshape(-1, num_heads, head_dim)
    k_flat = F.normalize(inputs["k"].float(), dim=-1).reshape(-1, num_heads, head_dim)
    v_flat = inputs["v"].float().reshape(-1, num_heads, head_dim)
    g_flat = inputs["g"].float().reshape(-1, num_heads, head_dim)
    beta_flat = torch.sigmoid(inputs["beta"].float().reshape(-1, num_heads))
    gate = lower_bound * torch.sigmoid(
        torch.exp(inputs["A_log"]).reshape(1, num_heads, 1)
        * (g_flat + inputs["dt_bias"].reshape(1, num_heads, head_dim))
    )
    decay = torch.exp(gate)
    if inputs["cu_seqlens"] is None:
        offsets = [index * seq_len for index in range(batch_size + 1)]
    else:
        offsets = [int(value) for value in inputs["cu_seqlens"].tolist()]
    if inputs["initial_state"] is None:
        state = torch.zeros(
            (len(offsets) - 1, num_heads, head_dim, head_dim),
            dtype=torch.bfloat16,
            device=q.device,
        )
    else:
        state = inputs["initial_state"].clone()
    out = torch.empty_like(q_flat)
    for sequence in range(len(offsets) - 1):
        for token in range(offsets[sequence], offsets[sequence + 1]):
            state_f32 = state[sequence].float()
            decayed = state_f32 * decay[token].unsqueeze(1)
            predicted = torch.einsum("hk,hvk->hv", k_flat[token], decayed)
            residual = beta_flat[token].unsqueeze(-1) * (v_flat[token] - predicted)
            updated = decayed + residual.unsqueeze(-1) * k_flat[token].unsqueeze(1)
            state[sequence] = updated.to(torch.bfloat16)
            projected = torch.einsum(
                "hk,hvk->hv", q_flat[token], state[sequence].float()
            )
            out[token] = (scale * projected).to(torch.bfloat16)
    return out.reshape_as(q), state


def _h12_bf16_residual_carriers(torch, *, value, prediction, beta_logit):
    """Apply the four BF16 residual carriers selected by the public H12 ABI."""

    prediction_carrier = prediction.to(torch.bfloat16).float()
    delta_carrier = (value - prediction_carrier).to(torch.bfloat16).float()
    beta_carrier = torch.sigmoid(beta_logit).to(torch.bfloat16).float()
    update_carrier = (
        (beta_carrier.unsqueeze(-1) * delta_carrier).to(torch.bfloat16).float()
    )
    return prediction_carrier, delta_carrier, beta_carrier, update_carrier


def test_h12_smoke_reference_residual_carriers_round_every_boundary_on_cpu():
    prediction = torch.tensor(
        [[-15.22768497, -1.95509577, 3.25501537, 0.3333]],
        dtype=torch.float32,
    )
    value = torch.tensor(
        [[3.81683922, -9.65635967, -4.79144955, 0.7123]],
        dtype=torch.float32,
    )
    beta_logit = torch.tensor([-1.02760863], dtype=torch.float32)

    prediction_carrier, delta_carrier, beta_carrier, update_carrier = (
        _h12_bf16_residual_carriers(
            torch,
            value=value,
            prediction=prediction,
            beta_logit=beta_logit,
        )
    )
    expected_prediction = prediction.to(torch.bfloat16).float()
    unrounded_delta = value - expected_prediction
    expected_delta = unrounded_delta.to(torch.bfloat16).float()
    unrounded_beta = torch.sigmoid(beta_logit)
    expected_beta = unrounded_beta.to(torch.bfloat16).float()
    unrounded_update = expected_beta.unsqueeze(-1) * expected_delta
    expected_update = unrounded_update.to(torch.bfloat16).float()

    assert torch.equal(prediction_carrier, expected_prediction)
    assert torch.equal(delta_carrier, expected_delta)
    assert torch.equal(beta_carrier, expected_beta)
    assert torch.equal(update_carrier, expected_update)
    assert not torch.equal(prediction_carrier, prediction)
    assert not torch.equal(delta_carrier, unrounded_delta)
    assert not torch.equal(beta_carrier, unrounded_beta)
    assert not torch.equal(update_carrier, unrounded_update)


def _chunk16_debug_reference(
    inputs, *, lower_bound=-5.0, scale=None, checkpoint_every_n_tokens=0
):
    """Clean-room H12 smoke reference for focused numerical diagnostics.

    The recurrent state carrier stays in FP32 within each 16-token chunk, but
    the state/K prediction, V-minus-prediction delta, sigmoid beta, and
    post-beta update carrier each round through BF16.  A BF16 state snapshot
    becomes the next chunk's carrier, while each output projects the unrounded
    FP32 state for its token.  The public benchmark separately compares output
    and complete final state against the pinned FlashKDA implementation.
    """

    q = inputs["q"]
    batch_size, seq_len, num_heads, head_dim = q.shape
    scale = head_dim**-0.5 if scale is None else scale
    q_flat = F.normalize(q.float(), dim=-1).reshape(-1, num_heads, head_dim)
    k_flat = F.normalize(inputs["k"].float(), dim=-1).reshape(-1, num_heads, head_dim)
    v_flat = inputs["v"].float().reshape(-1, num_heads, head_dim)
    g_flat = inputs["g"].float().reshape(-1, num_heads, head_dim)
    beta_logits_flat = inputs["beta"].float().reshape(-1, num_heads)
    gate = lower_bound * torch.sigmoid(
        torch.exp(inputs["A_log"]).reshape(1, num_heads, 1)
        * (g_flat + inputs["dt_bias"].reshape(1, num_heads, head_dim))
    )
    decay = torch.exp(gate)
    if inputs["cu_seqlens"] is None:
        offsets = [index * seq_len for index in range(batch_size + 1)]
    else:
        offsets = [int(value) for value in inputs["cu_seqlens"].tolist()]
    if inputs["initial_state"] is None:
        state = torch.zeros(
            (len(offsets) - 1, num_heads, head_dim, head_dim),
            dtype=torch.bfloat16,
            device=q.device,
        )
    else:
        state = inputs["initial_state"].clone()
    out = torch.empty_like(q_flat)
    checkpoints = []
    for sequence in range(len(offsets) - 1):
        if checkpoint_every_n_tokens:
            checkpoints.append(state[sequence].clone())
        carrier = state[sequence].float()
        sequence_length = offsets[sequence + 1] - offsets[sequence]
        for local_token, token in enumerate(
            range(offsets[sequence], offsets[sequence + 1]), start=1
        ):
            decayed = carrier * decay[token].unsqueeze(1)
            predicted = torch.einsum("hk,hvk->hv", k_flat[token], decayed)
            _, _, _, update_carrier = _h12_bf16_residual_carriers(
                torch,
                value=v_flat[token],
                prediction=predicted,
                beta_logit=beta_logits_flat[token],
            )
            updated = decayed + update_carrier.unsqueeze(-1) * k_flat[token].unsqueeze(
                1
            )
            state[sequence] = updated.to(torch.bfloat16)
            projected = torch.einsum("hk,hvk->hv", q_flat[token], updated)
            out[token] = (scale * projected).to(torch.bfloat16)
            carrier = state[sequence].float() if local_token % 16 == 0 else updated
            if (
                checkpoint_every_n_tokens
                and local_token % checkpoint_every_n_tokens == 0
                and local_token < sequence_length
            ):
                checkpoints.append(state[sequence].clone())
    result = (out.reshape_as(q), state)
    if checkpoint_every_n_tokens:
        return (*result, torch.stack(checkpoints))
    return result


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda")


@pytest.fixture
def flash_kda_device(cuda_device):
    if get_compute_capability(cuda_device) not in ((10, 0), (10, 3)):
        pytest.skip(
            "frozen recurrent KDA prefill requires CC 10.0 "
            "(SM100a; B200/GB200) or CC 10.3 (SM103a; B300/GB300)"
        )
    return cuda_device


@pytest.mark.parametrize(
    ("compute_capability", "cuda_version", "expected_target", "error_match"),
    [
        ((10, 0), "12.8", "sm100a", None),
        ((10, 0), "12.9", "sm100f", None),
        ((10, 3), "12.8", None, "10.3 requires CUDA 12.9"),
        ((10, 3), "12.9", "sm100f", None),
        ((12, 0), "13.0", None, "requires compute capability 10.0"),
        ((10, 0), "12.7", None, "10.0 requires CUDA 12.8"),
    ],
)
def test_flash_kda_target_resolution(
    monkeypatch,
    compute_capability,
    cuda_version,
    expected_target,
    error_match,
):
    monkeypatch.setattr(
        kda_prefill_api,
        "get_compute_capability",
        lambda device: compute_capability,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_is_cuda_version_at_least",
        lambda required: Version(cuda_version) >= Version(required),
    )
    device = torch.device("cuda")
    if error_match is not None:
        with pytest.raises(RuntimeError, match=error_match):
            kda_prefill_api._select_flash_kda_prefill_target(device)
    else:
        assert (
            kda_prefill_api._select_flash_kda_prefill_target(device) == expected_target
        )


def test_flash_kda_sm_count_is_cached_per_device(monkeypatch):
    calls = []

    def get_device_properties(device):
        resolved = torch.device(device)
        calls.append(resolved)
        return SimpleNamespace(
            multi_processor_count=148 if resolved.index == 0 else 152
        )

    kda_prefill_api._flash_kda_device_sm_count.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)
    try:
        assert kda_prefill_api._flash_kda_device_sm_count(torch.device("cuda:0")) == 148
        assert kda_prefill_api._flash_kda_device_sm_count(torch.device("cuda:0")) == 148
        assert kda_prefill_api._flash_kda_device_sm_count(torch.device("cuda:1")) == 152
        assert kda_prefill_api._flash_kda_device_sm_count(torch.device("cuda:1")) == 152
        assert calls == [torch.device("cuda:0"), torch.device("cuda:1")]
    finally:
        kda_prefill_api._flash_kda_device_sm_count.cache_clear()


def test_persistent_policy_uses_physical_arch_and_sm_count_independently():
    for compute_capability, sm_count, expected in (
        ((10, 0), 148, True),
        ((10, 0), 152, True),
        ((10, 3), 148, False),
        ((10, 3), 152, False),
    ):
        assert (
            kda_prefill_api._uses_measured_sm100_persistent_policy(
                compute_capability=compute_capability,
                sm_count=sm_count,
            )
            is expected
        )

    uniform = kda_prefill_api._persistent_task_plan(
        (8192,) * 8,
        num_heads=96,
        sm_count=148,
    )
    assert uniform is not None
    sequence_order, task_ids, task_offsets = uniform
    assert sequence_order == tuple(range(8))
    assert sorted(task_ids) == list(range(8 * 96))
    assert len(task_offsets) == 129
    assert {
        right - left
        for left, right in zip(task_offsets, task_offsets[1:], strict=False)
    } == {6}

    mixed = kda_prefill_api._persistent_task_plan(
        (3063, 2048, 1300, 963, 547, 271),
        num_heads=96,
        sm_count=148,
    )
    assert mixed is not None
    _, mixed_ids, mixed_offsets = mixed
    assert sorted(mixed_ids) == list(range(6 * 96))
    assert len(mixed_offsets) == 149
    large_sm_count_uniform = kda_prefill_api._persistent_task_plan(
        (8192,) * 8,
        num_heads=96,
        sm_count=152,
    )
    assert large_sm_count_uniform is not None
    assert len(large_sm_count_uniform[2]) == 129

    large_sm_count_mixed = kda_prefill_api._persistent_task_plan(
        (3063, 2048, 1300, 963, 547, 271),
        num_heads=96,
        sm_count=152,
    )
    assert large_sm_count_mixed is not None
    assert len(large_sm_count_mixed[2]) == 153
    assert (
        kda_prefill_api._persistent_task_plan(
            (3063, 2048, 1300, 963, 547, 271),
            num_heads=64,
            sm_count=152,
        )
        is None
    )
    assert (
        kda_prefill_api._persistent_task_plan(
            (8192,) * 8,
            num_heads=96,
            sm_count=150,
        )
        is None
    )


def test_variant_selector_exposes_specialized_routes_only_when_requested():
    assert (
        kda_prefill_api._select_flash_kda_prefill_variant(
            fixed_layout=True,
            num_sequences=1,
            num_heads=8,
            use_persistent_m128=True,
            use_small_bh_m128=True,
        )
        == "small_bh_m128"
    )
    assert (
        kda_prefill_api._select_flash_kda_prefill_variant(
            fixed_layout=False,
            num_sequences=8,
            num_heads=96,
            use_persistent_m128=True,
        )
        == "persistent_m128"
    )
    assert (
        kda_prefill_api._select_flash_kda_prefill_variant(
            fixed_layout=False,
            num_sequences=128,
            num_heads=96,
            use_persistent_m128=True,
            use_exact_n16=True,
        )
        == "m128_n16"
    )
    assert (
        kda_prefill_api._select_flash_kda_prefill_variant(
            fixed_layout=False,
            num_sequences=8,
            num_heads=12,
            use_persistent_m128=True,
        )
        == "m128_n16"
    )


@pytest.mark.parametrize(
    (
        "compute_capability",
        "sm_count",
        "num_sequences",
        "num_heads",
        "sequence_length",
        "expected",
    ),
    [
        ((10, 0), 148, 1, 8, 2048, True),
        ((10, 3), 152, 2, 4, 65536, True),
        ((10, 3), 64, 8, 1, 131072, True),
        ((10, 0), 63, 8, 1, 2048, False),
        ((10, 0), 148, 1, 8, 2047, False),
        ((10, 0), 148, 3, 3, 2048, False),
        ((10, 0), 148, 1, 9, 2048, False),
    ],
)
def test_small_bh_owner_helper_policy_matches_residency_contract(
    compute_capability,
    sm_count,
    num_sequences,
    num_heads,
    sequence_length,
    expected,
):
    assert (
        kda_prefill_api._should_use_small_bh_owner_helper(
            compute_capability=compute_capability,
            sm_count=sm_count,
            num_sequences=num_sequences,
            num_heads=num_heads,
            sequence_length=sequence_length,
        )
        is expected
    )


@pytest.mark.parametrize(
    (
        "fixed_layout",
        "num_sequences",
        "num_heads",
        "uniform_sequences",
        "max_sequence_length",
        "expected_route",
    ),
    [
        (True, 1, 64, True, 4096, "bt16_prepare_chain_m64"),
        (True, 1, 12, True, 512, "bt16_prepare_chain_m64"),
        (True, 8, 12, True, 1024, "direct_m128"),
        (False, 8, 12, False, 3072, "bt16_prepare_chain_m64"),
        (True, 1, 4, True, 65_536, "bt16_prepare_chain_m64"),
        (True, 1, 1, True, 65_535, "small_bh_owner_helper_m128"),
        (True, 1, 1, True, 65_536, "bt16_prepare_chain_m64"),
        (True, 1, 64, True, 512, "independent_dvsplit_m64"),
    ],
)
def test_bt16_route_policy_matches_measured_crossovers(
    fixed_layout,
    num_sequences,
    num_heads,
    uniform_sequences,
    max_sequence_length,
    expected_route,
):
    assert (
        kda_prefill_api._select_flash_kda_bf16_route(
            compute_capability=(10, 3),
            sm_count=152,
            fixed_layout=fixed_layout,
            num_sequences=num_sequences,
            num_heads=num_heads,
            uniform_sequences=uniform_sequences,
            max_sequence_length=max_sequence_length,
        )
        == expected_route
    )


def test_bt16_prepare_walk_and_physical_variants_match_production_policy():
    assert (
        kda_prefill_api._direct_m128_route(num_heads=64, max_sequence_length=16)
        == "direct_m128_n16"
    )
    assert (
        kda_prefill_api._direct_m128_route(num_heads=64, max_sequence_length=17)
        == "direct_m128"
    )
    assert (
        kda_prefill_api._direct_m128_route(num_heads=96, max_sequence_length=16)
        == "direct_m128"
    )
    assert (
        kda_prefill_api._direct_m128_route(num_heads=12, max_sequence_length=16)
        == "direct_m128_n16"
    )
    assert (
        kda_prefill_api._bt16_chunks_per_prepare_cta(num_heads=12, total_chunks=128)
        == 1
    )
    assert (
        kda_prefill_api._bt16_chunks_per_prepare_cta(num_heads=12, total_chunks=129)
        == 4
    )
    assert (
        kda_prefill_api._bt16_chunks_per_prepare_cta(num_heads=64, total_chunks=255)
        == 6
    )
    assert (
        kda_prefill_api._bt16_chunks_per_prepare_cta(num_heads=64, total_chunks=256)
        == 8
    )

    assert kda_prefill_api._select_bt16_physical_variants(
        compute_capability=(10, 3),
        sm_count=152,
        fixed_layout=True,
        num_sequences=1,
        num_heads=64,
        max_sequence_length=4096,
    ) == ("bt16_prepare_beta_tma", "bt16_chain_m64_s8", True)
    assert kda_prefill_api._select_bt16_physical_variants(
        compute_capability=(10, 3),
        sm_count=152,
        fixed_layout=True,
        num_sequences=1,
        num_heads=4,
        max_sequence_length=65_536,
    ) == ("bt16_prepare", "bt16_chain_m64_s9", False)
    assert kda_prefill_api._select_bt16_physical_variants(
        compute_capability=(10, 3),
        sm_count=152,
        fixed_layout=True,
        num_sequences=1,
        num_heads=8,
        max_sequence_length=65_536,
    ) == ("bt16_prepare", "bt16_chain_m64_s9", False)
    assert kda_prefill_api._select_bt16_physical_variants(
        compute_capability=(10, 0),
        sm_count=148,
        fixed_layout=False,
        num_sequences=8,
        num_heads=12,
        max_sequence_length=3072,
    ) == ("bt16_prepare", "bt16_chain_m64_s7", False)


def test_bt16_two_stage_adapter_forwards_stable_wrapper_abis(monkeypatch):
    prepare_module = _RecorderModule()
    chain_module = _RecorderModule()
    modules = {
        "bt16_prepare_beta_tma": prepare_module,
        "bt16_chain_m64_s8": chain_module,
    }
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, target: modules[variant],
    )
    q = torch.empty((1, 1, 1, 1), dtype=torch.bfloat16)
    factor = torch.empty((1, 1, 1, 1), dtype=torch.bfloat16)
    qk = torch.empty((1, 1, 1, 1, 1), dtype=torch.bfloat16)
    diag = torch.empty((1, 1, 1, 1), dtype=torch.float32)
    cu_chunks = torch.tensor([0, 256], dtype=torch.int32)
    chunk_to_seq = torch.zeros(256, dtype=torch.int32)
    monkeypatch.setattr(
        kda_prefill_api,
        "_bt16_workspace",
        lambda **kwargs: (
            cu_chunks,
            chunk_to_seq,
            factor,
            factor.clone(),
            factor.clone(),
            qk,
            diag,
            256,
            760,
        ),
    )
    workspace = SimpleNamespace(
        _descriptor_signatures={},
        _descriptor_storages={
            variant: torch.empty(896, dtype=torch.uint8) for variant in modules
        },
    )
    cu_seqlens = torch.tensor([0, 4096], dtype=torch.int64)
    seq_order = torch.tensor([0], dtype=torch.int32)
    state = torch.empty((1, 1, 1, 1), dtype=torch.bfloat16)
    output = torch.empty_like(q)

    kda_prefill_api._run_bt16_prepare_chain(
        workspace=workspace,
        target="sm100f",
        q=q,
        k=q,
        v=q,
        g=q,
        beta=torch.empty((1, 1, 1), dtype=torch.bfloat16),
        A_log=torch.empty(64, dtype=torch.float32),
        dt_bias=torch.empty((64, 128), dtype=torch.float32),
        cu_seqlens=cu_seqlens,
        seq_order=seq_order,
        initial_state=state,
        out=output,
        final_state=state,
        offsets=(0, 4096),
        num_heads=64,
        sm_count=152,
        compute_capability=(10, 3),
        fixed_layout=True,
        max_sequence_length=4096,
        use_initial_state=True,
        store_final_state=True,
        scale=0.125,
        lower_bound=-5.0,
        stream_ptr=17,
        capturing=False,
    )

    (prepare_args,) = prepare_module.calls
    assert len(prepare_args) == 21
    assert prepare_args[6] is cu_seqlens
    assert prepare_args[7] is cu_chunks
    assert prepare_args[8] is chunk_to_seq
    assert prepare_args[15] == 1
    assert prepare_args[16:21] == (256, 64, -5.0, 760, 17)
    (chain_args,) = chain_module.calls
    assert len(chain_args) == 20
    assert chain_args[6] is cu_seqlens
    assert chain_args[7] is cu_chunks
    assert chain_args[8] is seq_order
    assert chain_args[12].dtype == torch.uint8
    assert chain_args[13:20] == (1, 64, 1, 1, 0.125, 128, 17)


def test_h96_uniform_n128_uses_exact_n16_only_on_148_sm():
    for sm_count in (148, 152):
        assert kda_prefill_api._requires_exact_n16_recurrence(
            sm_count=sm_count,
            fixed_layout=False,
            num_sequences=128,
            num_heads=96,
            uniform_sequences=True,
        ) is (sm_count == 148)


class _RecorderModule:
    def __init__(self, *, final_value=None):
        self.calls = []
        self.final_value = final_value

    def run(self, *args):
        self.calls.append(args)
        if self.final_value is not None:
            if len(args) == 21:
                store_final_state = args[17]
                final_state = args[12]
            elif len(args) == 23:
                store_final_state = args[19]
                final_state = args[14]
            elif len(args) == 25:
                store_final_state = args[21]
                final_state = args[12]
            else:
                store_final_state = args[23]
                final_state = args[13]
            if bool(store_final_state):
                final_state.fill_(self.final_value)


def test_decode_and_spec_stay_on_existing_backend(monkeypatch):
    sentinel = (object(), object())
    calls = []

    def old_backend(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(kda_decode_api, "_run_recurrent_kda", old_backend)
    monkeypatch.setattr(
        kda_decode_api,
        "recurrent_kda",
        lambda *args, **kwargs: pytest.fail("facade nested the decorated decode API"),
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, arch: pytest.fail(f"unexpected frozen route {variant}/{arch}"),
    )
    q = torch.empty((2, 1, 4, 128), dtype=torch.bfloat16)
    result = recurrent_kda(q, q, q, q, torch.empty((2, 1, 4)))
    assert result is sentinel
    result = recurrent_kda(
        q.expand(2, 2, 4, 128),
        q.expand(2, 2, 4, 128),
        q.expand(2, 2, 4, 128),
        q.expand(2, 2, 4, 128),
        torch.empty((2, 2, 4)),
        num_spec_tokens=1,
    )
    assert result is sentinel
    assert len(calls) == 2


def test_multi_token_gqa_stays_on_existing_backend(cuda_device, monkeypatch):
    sentinel = (object(), object())
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(kda_decode_api, "_run_recurrent_kda", lambda **kwargs: sentinel)
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, arch: pytest.fail(f"unexpected frozen route {variant}/{arch}"),
    )
    q = torch.randn((1, 2, 2, 128), dtype=torch.bfloat16, device=cuda_device)
    v = torch.randn((1, 2, 4, 128), dtype=torch.bfloat16, device=cuda_device)
    result = recurrent_kda(
        q,
        q.clone(),
        v,
        v.clone(),
        torch.randn((1, 2, 4), dtype=torch.bfloat16, device=cuda_device),
        A_log=torch.randn(2, device=cuda_device),
        dt_bias=torch.randn((2, 128), device=cuda_device),
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        beta_is_logit=True,
    )
    assert result is sentinel


@pytest.mark.parametrize(
    ("packed", "num_heads", "expected_variant"),
    [
        (False, 64, "m128_n16"),
        (True, 64, "m128_n16"),
        (True, 2, "m128_n16"),
        (False, 12, "m128_n16"),
    ],
)
@pytest.mark.parametrize(
    ("compute_capability", "expected_target"),
    [((10, 0), "sm100f"), ((10, 3), "sm100f")],
)
def test_frozen_route_and_ffi_abi(
    cuda_device,
    monkeypatch,
    packed,
    num_heads,
    expected_variant,
    compute_capability,
    expected_target,
):
    monkeypatch.setattr(
        kda_prefill_api,
        "get_compute_capability",
        lambda device: compute_capability,
    )
    monkeypatch.setattr(
        kda_prefill_api, "_is_cuda_version_at_least", lambda version: True
    )
    monkeypatch.setattr(kda_prefill_api, "_flash_kda_stream_workspaces", {})
    modules = {}
    routes = []

    def get_module(variant, target):
        routes.append((variant, target))
        modules.setdefault(variant, _RecorderModule())
        return modules[variant]

    monkeypatch.setattr(kda_prefill_api, "_get_flash_kda_prefill_module", get_module)
    inputs = _make_inputs(
        seq_lens=[1, 2] if packed else [2],
        num_heads=num_heads,
        packed=packed,
    )
    if packed and num_heads == 2:
        inputs["cu_seqlens"] = inputs["cu_seqlens"].to(torch.int32)
    output = torch.zeros_like(inputs["q"])
    seq_order = (
        torch.tensor([1, 0], dtype=torch.int32, device="cuda") if packed else None
    )
    actual, state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        seq_order=seq_order,
        backend="cake",
    )
    assert actual.data_ptr() == output.data_ptr()
    assert state is None
    assert set(modules) == {expected_variant}
    assert routes == [(expected_variant, expected_target)]
    (args,) = modules[expected_variant].calls
    expected_arg_count = 21 if expected_variant == "m64" else 28
    assert len(args) == expected_arg_count
    assert args[0].data_ptr() == inputs["q"].data_ptr()
    assert args[4].data_ptr() == inputs["beta"].data_ptr()
    assert args[5].shape == (
        max(inputs["q"].numel() // (num_heads * 128), 32),
        (num_heads + 7) // 8 * 8,
    )
    assert args[8].dtype == torch.int64
    assert args[9].dtype == torch.int32
    if packed:
        assert args[9].data_ptr() == seq_order.data_ptr()
    if expected_variant == "m64":
        assert args[10].data_ptr() == args[12].data_ptr()
        assert args[13].dtype == torch.uint8
        assert args[13].shape == (768,)
        assert args[14] == 1
        assert args[15] == num_heads
        assert args[16] == 0
        assert args[17] == 0
        assert math.isclose(args[18], 128**-0.5)
        assert args[19] == -5.0
        assert args[20] == int(torch.cuda.current_stream(cuda_device).cuda_stream)
    else:
        assert args[11].data_ptr() == args[13].data_ptr()
        assert args[16].dtype == torch.uint8
        assert args[16].shape == (768,)
        assert args[17] == 1
        assert args[18] == num_heads
        assert args[19] == inputs["beta"].stride(-2)
        assert args[21] == 0
        assert args[22] == 0
        assert args[23] == 0
        assert args[24] == 0
        assert math.isclose(args[25], 128**-0.5)
        assert args[26] == -5.0
        assert args[27] == int(torch.cuda.current_stream(cuda_device).cuda_stream)
    if num_heads % 8 != 0:
        assert args[5].data_ptr() != inputs["beta"].data_ptr()


@pytest.mark.parametrize("sm_count", [148, 152])
def test_sm100_uniform_prefill_reaches_persistent_worker_abi(
    cuda_device,
    monkeypatch,
    sm_count,
):
    monkeypatch.setattr(
        kda_prefill_api,
        "get_compute_capability",
        lambda device: (10, 0),
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_is_cuda_version_at_least",
        lambda version: True,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_device_sm_count",
        lambda device: sm_count,
    )
    monkeypatch.setattr(kda_prefill_api, "_flash_kda_stream_workspaces", {})
    module = _RecorderModule()
    routes = []

    def get_module(variant, target):
        routes.append((variant, target))
        return module

    monkeypatch.setattr(kda_prefill_api, "_get_flash_kda_prefill_module", get_module)
    inputs = _make_inputs(
        seq_lens=[2, 2],
        num_heads=96,
        packed=True,
        initial_state=True,
    )
    output, state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        backend="cake",
    )
    assert output.shape == inputs["q"].shape
    assert state is None
    assert routes == [("persistent_m128", "sm100f")]
    (args,) = module.calls
    assert len(args) == 23
    assert args[9].tolist() == [0, 1]
    assert sorted(args[10].tolist()) == list(range(2 * 96))
    assert args[11].numel() == sm_count + 1
    assert args[11][0].item() == 0
    assert args[11][-1].item() == 2 * 96
    assert args[15].dtype == torch.uint8
    assert args[15].shape == (768,)
    assert args[16] == 1
    assert args[17] == 96


def test_explicit_seq_order_keeps_direct_worker_and_reaches_ffi(
    cuda_device,
    monkeypatch,
):
    monkeypatch.setattr(
        kda_prefill_api,
        "get_compute_capability",
        lambda device: (10, 0),
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_is_cuda_version_at_least",
        lambda version: True,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_device_sm_count",
        lambda device: 148,
    )
    monkeypatch.setattr(kda_prefill_api, "_flash_kda_stream_workspaces", {})
    module = _RecorderModule()
    routes = []

    def get_module(variant, target):
        routes.append((variant, target))
        return module

    monkeypatch.setattr(kda_prefill_api, "_get_flash_kda_prefill_module", get_module)
    inputs = _make_inputs(
        seq_lens=[2, 2],
        num_heads=96,
        packed=True,
        initial_state=True,
    )
    seq_order = torch.tensor([1, 0], dtype=torch.int32, device=cuda_device)
    recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        seq_order=seq_order,
        backend="cake",
    )

    assert routes == [("m128", "sm100f")]
    (args,) = module.calls
    assert len(args) == 28
    assert args[9].data_ptr() == seq_order.data_ptr()


def test_b200_prefill_without_initial_state_stays_direct(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api,
        "get_compute_capability",
        lambda device: (10, 0),
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_is_cuda_version_at_least",
        lambda version: True,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_device_sm_count",
        lambda device: 148,
    )
    monkeypatch.setattr(kda_prefill_api, "_flash_kda_stream_workspaces", {})
    module = _RecorderModule()
    routes = []

    def get_module(variant, target):
        routes.append((variant, target))
        return module

    monkeypatch.setattr(kda_prefill_api, "_get_flash_kda_prefill_module", get_module)
    inputs = _make_inputs(
        seq_lens=[3, 1],
        num_heads=96,
        packed=True,
        initial_state=False,
    )
    recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        backend="cake",
    )
    assert routes == [("m128", "sm100f")]
    (args,) = module.calls
    assert args[9].tolist() == [0, 1]


@pytest.mark.parametrize(
    ("compute_capability", "sm_count"),
    [((10, 0), 148), ((10, 3), 152)],
)
def test_fixed_small_bh_prefill_reaches_owner_helper_abi(
    cuda_device,
    monkeypatch,
    compute_capability,
    sm_count,
):
    monkeypatch.setattr(
        kda_prefill_api,
        "get_compute_capability",
        lambda device: compute_capability,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_is_cuda_version_at_least",
        lambda version: True,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_device_sm_count",
        lambda device: sm_count,
    )
    monkeypatch.setattr(kda_prefill_api, "_flash_kda_stream_workspaces", {})
    module = _RecorderModule(final_value=0.5)
    routes = []

    def get_module(variant, target):
        routes.append((variant, target))
        return module

    monkeypatch.setattr(kda_prefill_api, "_get_flash_kda_prefill_module", get_module)
    inputs = _make_inputs(
        seq_lens=[2048],
        num_heads=1,
        packed=False,
    )
    output, state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        backend="cake",
    )

    assert output.shape == inputs["q"].shape
    assert state is not None
    assert torch.all(state == 0.5)
    assert routes == [("small_bh_m128", "sm100f")]
    (args,) = module.calls
    assert len(args) == 25
    assert args[13].dtype == torch.uint8
    assert args[13].shape == (896,)
    assert args[14].dtype == torch.bfloat16
    assert args[14].shape == (35 * 123, 128)
    assert args[15].dtype == torch.uint32
    assert args[15].shape == (35,)
    assert args[16].dtype == torch.uint32
    assert args[16].shape == (35,)
    assert args[17].dtype == torch.uint32
    assert args[17].shape == (1,)
    assert args[18] == 1
    assert args[19] == 1
    assert args[21] == 1
    assert math.isclose(args[22], 128**-0.5)
    assert args[23] == -5.0
    assert args[24] == int(torch.cuda.current_stream(cuda_device).cuda_stream)


def test_b200_packed_metadata_is_cached_for_unchanged_offsets(cuda_device):
    workspace = kda_prefill_api._FlashKDAStreamWorkspace(cuda_device)
    offsets = torch.tensor([0, 3, 6], dtype=torch.int64, device=cuda_device)
    first = kda_prefill_api._cached_packed_task_metadata(
        workspace,
        offsets,
        total_tokens=6,
        num_heads=96,
        sm_count=148,
        build_persistent_plan=True,
    )
    second = kda_prefill_api._cached_packed_task_metadata(
        workspace,
        offsets,
        total_tokens=6,
        num_heads=96,
        sm_count=148,
        build_persistent_plan=True,
    )
    assert first[0] == (0, 1)
    assert first[1] is not None
    assert first[3] == (0, 3, 6)
    assert first[4] == (3, 3)
    assert first is second


def test_packed_metadata_is_self_contained_across_threads(cuda_device):
    workspace = kda_prefill_api._FlashKDAStreamWorkspace(cuda_device)
    layouts = {
        "short_first": ((0, 1, 6), (1, 5)),
        "long_first": ((0, 4, 6), (4, 2)),
    }
    barrier = threading.Barrier(len(layouts))
    results = {}
    failures = []

    def build_metadata(name, expected):
        try:
            torch.cuda.set_device(cuda_device)
            offsets, _ = expected
            cu_seqlens = torch.tensor(
                offsets, dtype=torch.int64, device=cuda_device
            )
            metadata = kda_prefill_api._cached_packed_task_metadata(
                workspace,
                cu_seqlens,
                total_tokens=6,
                num_heads=96,
                sm_count=148,
                build_persistent_plan=False,
            )
            barrier.wait(timeout=10)
            results[name] = metadata
        except BaseException as error:
            failures.append(error)

    threads = [
        threading.Thread(target=build_metadata, args=(name, expected))
        for name, expected in layouts.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert all(not thread.is_alive() for thread in threads)
    assert not failures
    for name, (expected_offsets, expected_lengths) in layouts.items():
        metadata = results[name]
        assert metadata[3] == expected_offsets
        assert metadata[4] == expected_lengths


def test_direct_packed_prefill_automatically_sorts_sequences(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api,
        "get_compute_capability",
        lambda device: (10, 3),
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_is_cuda_version_at_least",
        lambda version: True,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_device_sm_count",
        lambda device: 152,
    )
    monkeypatch.setattr(kda_prefill_api, "_flash_kda_stream_workspaces", {})
    module = _RecorderModule()
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, target: module,
    )
    inputs = _make_inputs(
        seq_lens=[1, 3, 2],
        num_heads=96,
        packed=True,
    )
    recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        backend="cake",
    )
    (args,) = module.calls
    assert args[9].tolist() == [1, 2, 0]


def test_strided_beta_indexed_state_and_checkpoints_reach_native_ffi(
    cuda_device, monkeypatch
):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(
        kda_prefill_api, "_is_cuda_version_at_least", lambda version: True
    )
    module = _RecorderModule()
    routes = []

    def get_module(variant, target):
        routes.append((variant, target))
        return module

    monkeypatch.setattr(kda_prefill_api, "_get_flash_kda_prefill_module", get_module)
    inputs = _make_inputs(
        seq_lens=[65, 131],
        num_heads=12,
        packed=True,
        initial_state=True,
    )
    total_tokens = inputs["q"].shape[1]
    beta_carrier = torch.empty(
        (total_tokens, 32), dtype=torch.bfloat16, device=cuda_device
    )
    beta_carrier[:, 8:20].copy_(inputs["beta"][0])
    inputs["beta"] = beta_carrier[None, :, 8:20]
    assert not inputs["beta"].is_contiguous()

    state_slot_numel = 12 * 128 * 128
    state_storage = torch.zeros(
        (5, state_slot_numel + 64), dtype=torch.bfloat16, device=cuda_device
    )
    state_pool = state_storage.as_strided(
        (5, 12, 128, 128),
        (state_storage.stride(0), 128 * 128, 128, 1),
    )
    state_indices = torch.tensor([1, 3], dtype=torch.int32, device=cuda_device)
    state_pool[state_indices.to(torch.int64)] = inputs["initial_state"]
    inputs["initial_state"] = state_pool

    checkpoint_cu_starts = torch.tensor(
        [0, 5, 14], dtype=torch.int64, device=cuda_device
    )
    state_checkpoints = torch.empty(
        (14, 12, 128, 128), dtype=torch.bfloat16, device=cuda_device
    )
    output, returned_state, returned_checkpoints = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        ssm_state_indices=state_indices,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=16,
    )
    assert output.shape == inputs["q"].shape
    assert returned_state is state_pool
    assert returned_checkpoints is state_checkpoints
    assert routes == [("m128_n16_checkpoint", "sm100f")]
    (args,) = module.calls
    assert len(args) == 28
    assert args[4].data_ptr() == inputs["beta"].data_ptr()
    assert args[5].data_ptr() == inputs["beta"].data_ptr()
    assert args[10].data_ptr() == state_indices.data_ptr()
    assert args[11].data_ptr() == state_pool.data_ptr()
    assert args[13].data_ptr() == state_pool.data_ptr()
    assert args[14].data_ptr() == state_checkpoints.data_ptr()
    assert args[15].data_ptr() == checkpoint_cu_starts.data_ptr()
    assert args[19] == inputs["beta"].stride(-2)
    assert args[20] == state_pool.stride(0)
    assert args[21:25] == (1, 1, 1, 16)


def test_unaligned_strided_beta_uses_internal_tma_workspace(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(
        kda_prefill_api, "_is_cuda_version_at_least", lambda version: True
    )
    module = _RecorderModule()
    monkeypatch.setattr(
        kda_prefill_api, "_get_flash_kda_prefill_module", lambda variant, target: module
    )
    inputs = _make_inputs(seq_lens=[32], num_heads=12, packed=True)
    beta_carrier = torch.empty((32, 32), dtype=torch.bfloat16, device=cuda_device)
    beta_carrier[:, 7:19].copy_(inputs["beta"][0])
    inputs["beta"] = beta_carrier[None, :, 7:19]

    recurrent_kda(
        **_strict_prefill_kwargs(inputs), output=torch.empty_like(inputs["q"])
    )

    (args,) = module.calls
    assert args[4].data_ptr() == inputs["beta"].data_ptr()
    assert args[5].data_ptr() != inputs["beta"].data_ptr()
    assert args[5].shape == (32, 16)


def test_aligned_h6_strided_beta_uses_head_padded_tma_workspace(
    cuda_device, monkeypatch
):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(
        kda_prefill_api, "_is_cuda_version_at_least", lambda version: True
    )
    module = _RecorderModule()
    monkeypatch.setattr(
        kda_prefill_api, "_get_flash_kda_prefill_module", lambda variant, target: module
    )
    inputs = _make_inputs(seq_lens=[128], num_heads=6, packed=True)
    beta_carrier = torch.empty((128, 32), dtype=torch.bfloat16, device=cuda_device)
    beta_carrier[:, 8:14].copy_(inputs["beta"][0])
    inputs["beta"] = beta_carrier[None, :, 8:14]
    assert inputs["beta"].data_ptr() % 16 == 0
    assert inputs["beta"].stride(-2) == 32

    recurrent_kda(
        **_strict_prefill_kwargs(inputs), output=torch.empty_like(inputs["q"])
    )

    (args,) = module.calls
    assert args[4].data_ptr() == inputs["beta"].data_ptr()
    assert args[5].data_ptr() != inputs["beta"].data_ptr()
    assert args[5].shape == (128, 8)


def test_frozen_route_passes_nondefault_stream(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    module = _RecorderModule()
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, arch: module,
    )
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False)
    stream = torch.cuda.Stream(device=cuda_device)
    stream.wait_stream(torch.cuda.current_stream(cuda_device))
    with torch.cuda.stream(stream):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=torch.empty_like(inputs["q"]),
            backend="cake",
        )
    (args,) = module.calls
    assert args[27] == int(stream.cuda_stream)


def test_frozen_route_rejects_output_overlap(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    module = _RecorderModule()
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, arch: module,
    )
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False)
    with pytest.raises(ValueError, match="output must not overlap q"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=inputs["q"].view_as(inputs["q"]),
        )
    assert module.calls == []


def test_initial_state_is_updated_in_place(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    module = _RecorderModule(final_value=0.25)
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, arch: module,
    )
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False, initial_state=True)
    original_state = inputs["initial_state"]
    actual, returned_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        backend="cake",
    )
    assert actual.shape == inputs["q"].shape
    assert returned_state is original_state
    (args,) = module.calls
    assert args[11].data_ptr() == original_state.data_ptr()
    assert args[13].data_ptr() == original_state.data_ptr()
    assert args[22] == 1
    assert args[23] == 1
    torch.testing.assert_close(
        original_state,
        torch.full_like(original_state, 0.25),
    )


def test_stream_workspace_does_not_allocate_state_scratch_for_inplace_update(
    cuda_device, monkeypatch
):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(kda_prefill_api, "_flash_kda_stream_workspaces", {})
    module = _RecorderModule(final_value=0.0)
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, arch: module,
    )
    cases = [
        _make_inputs(
            seq_lens=[2],
            num_heads=2,
            packed=False,
            initial_state=True,
        ),
        _make_inputs(
            seq_lens=[1, 1, 2],
            num_heads=2,
            packed=True,
            initial_state=True,
        ),
        _make_inputs(
            seq_lens=[2, 2],
            num_heads=2,
            packed=False,
            initial_state=True,
        ),
    ]
    for inputs in cases:
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=torch.empty_like(inputs["q"]),
            backend="cake",
        )

    assert len(kda_prefill_api._flash_kda_stream_workspaces) == 1
    (workspace,) = kda_prefill_api._flash_kda_stream_workspaces.values()
    assert workspace._state_scratch is None
    assert workspace._beta_padding.numel() == 32 * 8


@pytest.mark.parametrize(
    ("dtype", "size_delta"),
    [(torch.int64, 0), (torch.int32, 1)],
)
def test_packed_seq_order_validation(cuda_device, monkeypatch, dtype, size_delta):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, arch: _RecorderModule(),
    )
    inputs = _make_inputs(seq_lens=[1, 2], num_heads=2, packed=True)
    seq_order = torch.arange(2 + size_delta, dtype=dtype, device="cuda")
    with pytest.raises(ValueError, match="seq_order"):
        recurrent_kda(**_strict_prefill_kwargs(inputs), seq_order=seq_order)


def test_fixed_prefill_rejects_seq_order(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False)
    with pytest.raises(ValueError, match="only supported for packed"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            seq_order=torch.zeros(1, dtype=torch.int32, device=cuda_device),
        )


def test_graph_capture_requires_packed_int64_offsets(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    inputs = _make_inputs(seq_lens=[1, 2], num_heads=2, packed=True)
    inputs["cu_seqlens"] = inputs["cu_seqlens"].to(torch.int32)
    workspace = RecurrentKDAPrefillWorkspace(cuda_device)
    with pytest.raises(RuntimeError, match="requires int64 cu_seqlens"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            prefill_workspace=workspace,
            backend="cake",
        )


def test_graph_capture_requires_explicit_workspace(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False)
    with pytest.raises(
        RuntimeError, match="requires an explicit RecurrentKDAPrefillWorkspace"
    ):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=torch.empty_like(inputs["q"]),
        )


def test_explicit_workspace_descriptor_prepare_and_reuse(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    module = _RecorderModule()
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, arch: module,
    )
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False)
    output = torch.empty_like(inputs["q"])
    workspace = RecurrentKDAPrefillWorkspace(cuda_device)

    for _ in range(2):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=output,
            prefill_workspace=workspace,
            backend="cake",
        )
    assert [args[17] for args in module.calls] == [1, 0]
    assert (
        module.calls[0][16].data_ptr()
        == module.calls[1][16].data_ptr()
        == workspace._descriptor_storages["m128_n16"].data_ptr()
    )

    changed_output = torch.empty_like(output)
    recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=changed_output,
        prefill_workspace=workspace,
        backend="cake",
    )
    assert module.calls[-1][17] == 1


def test_captured_workspace_rejects_eager_reuse_and_capture_mismatch(
    cuda_device, monkeypatch
):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    module = _RecorderModule()
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, arch: module,
    )
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False)
    output = torch.empty_like(inputs["q"])
    workspace = RecurrentKDAPrefillWorkspace(cuda_device)
    recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        prefill_workspace=workspace,
        backend="cake",
    )

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="not warmed for the exact"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=torch.empty_like(output),
            prefill_workspace=workspace,
            backend="cake",
        )

    recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        prefill_workspace=workspace,
        backend="cake",
    )
    assert module.calls[-1][17] == 0
    assert workspace._captured

    with pytest.raises(RuntimeError, match="captured by another CUDA graph"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=output,
            prefill_workspace=workspace,
            backend="cake",
        )

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with pytest.raises(RuntimeError, match="cannot be reused eagerly"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=output,
            prefill_workspace=workspace,
            backend="cake",
        )


def test_workspace_rejects_a_different_stream(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    module = _RecorderModule()
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, arch: module,
    )
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False)
    output = torch.empty_like(inputs["q"])
    workspace = RecurrentKDAPrefillWorkspace(cuda_device)
    recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        prefill_workspace=workspace,
    )

    other_stream = torch.cuda.Stream(device=cuda_device)
    with (
        torch.cuda.stream(other_stream),
        pytest.raises(RuntimeError, match="different CUDA stream"),
    ):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=output,
            prefill_workspace=workspace,
        )


def test_flash_kda_jit_getter_is_importable():
    import flashinfer
    from flashinfer.jit.flash_kda import get_flash_kda_prefill_module

    assert callable(get_flash_kda_prefill_module)
    assert flashinfer.RecurrentKDAPrefillWorkspace is RecurrentKDAPrefillWorkspace


@pytest.mark.parametrize("non_default_stream", [False, True])
def test_frozen_small_bh_prefill_matches_direct_control(
    flash_kda_device,
    monkeypatch,
    non_default_stream,
):
    inputs = _make_inputs(
        seq_lens=[2048],
        num_heads=1,
        packed=False,
        initial_state=True,
        seed=2048,
    )
    direct_inputs = {
        **inputs,
        "initial_state": inputs["initial_state"].clone(),
    }
    small_output = torch.empty_like(inputs["q"])
    direct_output = torch.empty_like(inputs["q"])
    routes = []
    get_module = kda_prefill_api._get_flash_kda_prefill_module

    def recording_get_module(variant, target):
        routes.append((variant, target))
        return get_module(variant, target)

    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        recording_get_module,
    )

    if non_default_stream:
        stream = torch.cuda.Stream(device=flash_kda_device)
        stream.wait_stream(torch.cuda.current_stream(flash_kda_device))
        with torch.cuda.stream(stream):
            actual_output, actual_state = recurrent_kda(
                **_strict_prefill_kwargs(inputs),
                output=small_output,
                output_final_state=True,
                backend="cake",
            )
        stream.synchronize()
    else:
        actual_output, actual_state = recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=small_output,
            output_final_state=True,
            backend="cake",
        )

    monkeypatch.setattr(
        kda_prefill_api,
        "_should_use_small_bh_owner_helper",
        lambda **kwargs: False,
    )
    expected_output, expected_state = recurrent_kda(
        **_strict_prefill_kwargs(direct_inputs),
        output=direct_output,
        output_final_state=True,
        backend="cake",
    )

    expected_target = kda_prefill_api._select_flash_kda_prefill_target(flash_kda_device)
    assert routes == [
        ("small_bh_m128", expected_target),
        ("m128", expected_target),
    ]
    assert actual_output.data_ptr() == small_output.data_ptr()
    assert actual_state is inputs["initial_state"]
    assert expected_output.data_ptr() == direct_output.data_ptr()
    assert expected_state is direct_inputs["initial_state"]
    torch.testing.assert_close(
        actual_output.float(),
        expected_output.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual_state.float(),
        expected_state.float(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_frozen_small_bh_prefill_cuda_graph_replay_matches_direct_control(
    flash_kda_device,
    monkeypatch,
):
    inputs = _make_inputs(
        seq_lens=[2048],
        num_heads=1,
        packed=False,
        initial_state=False,
        seed=2049,
    )
    output = torch.empty_like(inputs["q"])
    workspace = RecurrentKDAPrefillWorkspace(flash_kda_device)
    capture_stream = torch.cuda.Stream(device=flash_kda_device)
    capture_stream.wait_stream(torch.cuda.current_stream(flash_kda_device))
    call_kwargs = {
        **_strict_prefill_kwargs(inputs),
        "output": output,
        "output_final_state": True,
        "prefill_workspace": workspace,
        "backend": "cake",
    }

    with torch.cuda.stream(capture_stream):
        recurrent_kda(**call_kwargs)
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output, captured_state = recurrent_kda(**call_kwargs)

    with torch.cuda.stream(capture_stream):
        inputs["q"].mul_(0.875)
        inputs["beta"].add_(0.125)
        output.fill_(float("nan"))
    capture_stream.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    replay_output = captured_output.clone()
    replay_state = captured_state.clone()

    monkeypatch.setattr(
        kda_prefill_api,
        "_should_use_small_bh_owner_helper",
        lambda **kwargs: False,
    )
    direct_output, direct_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(output),
        output_final_state=True,
        backend="cake",
    )

    assert workspace._captured
    assert captured_output.data_ptr() == output.data_ptr()
    assert captured_state.data_ptr() == workspace._state_scratch.data_ptr()
    torch.testing.assert_close(
        replay_output.float(),
        direct_output.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        replay_state.float(),
        direct_state.float(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_frozen_bt16_scalar_prepare_subgroup_heads_matches_direct_control(
    flash_kda_device,
    monkeypatch,
):
    inputs = _make_inputs(
        seq_lens=[32],
        num_heads=4,
        packed=False,
        initial_state=True,
        seed=2050,
    )
    initial_state_seed = inputs["initial_state"].clone()
    routes = []
    get_module = kda_prefill_api._get_flash_kda_prefill_module

    def recording_get_module(variant, target):
        routes.append(variant)
        return get_module(variant, target)

    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        recording_get_module,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_should_use_bt16_prepare_chain",
        lambda **kwargs: True,
    )
    actual_output, actual_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        backend="cake",
    )
    actual_output = actual_output.clone()
    actual_state = actual_state.clone()

    inputs["initial_state"].copy_(initial_state_seed)
    monkeypatch.setattr(
        kda_prefill_api,
        "_should_use_bt16_prepare_chain",
        lambda **kwargs: False,
    )
    expected_output, expected_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        backend="cake",
    )

    assert routes[:2] == ["bt16_prepare", "bt16_chain_m64_s9"]
    assert routes[-1] == "m128"
    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual_state.float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )


def test_frozen_bt16_scalar_prepare_subgroup_heads_cuda_graph_replay(
    flash_kda_device,
    monkeypatch,
):
    inputs = _make_inputs(
        seq_lens=[32],
        num_heads=4,
        packed=False,
        initial_state=False,
        seed=2051,
    )
    output = torch.empty_like(inputs["q"])
    workspace = RecurrentKDAPrefillWorkspace(flash_kda_device)
    capture_stream = torch.cuda.Stream(device=flash_kda_device)
    capture_stream.wait_stream(torch.cuda.current_stream(flash_kda_device))
    monkeypatch.setattr(
        kda_prefill_api,
        "_should_use_bt16_prepare_chain",
        lambda **kwargs: True,
    )
    call_kwargs = {
        **_strict_prefill_kwargs(inputs),
        "output": output,
        "output_final_state": True,
        "prefill_workspace": workspace,
        "backend": "cake",
    }

    with torch.cuda.stream(capture_stream):
        recurrent_kda(**call_kwargs)
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output, captured_state = recurrent_kda(**call_kwargs)

    with torch.cuda.stream(capture_stream):
        inputs["q"].mul_(0.875)
        inputs["beta"].add_(0.125)
        output.fill_(float("nan"))
    capture_stream.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    replay_output = captured_output.clone()
    replay_state = captured_state.clone()

    monkeypatch.setattr(
        kda_prefill_api,
        "_should_use_bt16_prepare_chain",
        lambda **kwargs: False,
    )
    direct_output, direct_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(output),
        output_final_state=True,
        backend="cake",
    )

    assert workspace._captured
    assert captured_output.data_ptr() == output.data_ptr()
    assert captured_state.data_ptr() == workspace._state_scratch.data_ptr()
    torch.testing.assert_close(
        replay_output.float(), direct_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        replay_state.float(), direct_state.float(), atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("non_default_stream", [False, True])
def test_frozen_prefill_matches_reference(flash_kda_device, packed, non_default_stream):
    inputs = _make_inputs(
        seq_lens=[3, 5] if packed else [4, 4],
        num_heads=2,
        packed=packed,
        initial_state=True,
        seed=2026,
    )
    reference_inputs = {
        **inputs,
        "initial_state": inputs["initial_state"].clone(),
    }
    expected_output, expected_state = _reference(reference_inputs)
    output = torch.empty_like(inputs["q"])
    state_identity = inputs["initial_state"]
    seq_order = (
        torch.tensor([1, 0], dtype=torch.int32, device=flash_kda_device)
        if packed
        else None
    )

    if non_default_stream:
        stream = torch.cuda.Stream(device=flash_kda_device)
        stream.wait_stream(torch.cuda.current_stream(flash_kda_device))
        with torch.cuda.stream(stream):
            actual_output, actual_state = recurrent_kda(
                **_strict_prefill_kwargs(inputs),
                output=output,
                output_final_state=True,
                seq_order=seq_order,
            )
        stream.synchronize()
    else:
        actual_output, actual_state = recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=output,
            output_final_state=True,
            seq_order=seq_order,
        )

    assert actual_output.data_ptr() == output.data_ptr()
    assert actual_state is state_identity
    torch.testing.assert_close(
        actual_output.float(),
        expected_output.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual_state.float(),
        expected_state.float(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_frozen_prefill_without_initial_or_final_state(flash_kda_device):
    inputs = _make_inputs(seq_lens=[3], num_heads=2, packed=False, initial_state=False)
    expected_output, _ = _reference(inputs)
    output = torch.empty_like(inputs["q"])
    actual_output, actual_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        output_final_state=False,
    )
    assert actual_output.data_ptr() == output.data_ptr()
    assert actual_state is None
    torch.testing.assert_close(
        actual_output.float(),
        expected_output.float(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_frozen_prefill_h6_full_tma_chunk_matches_reference(flash_kda_device):
    inputs = _make_inputs(
        seq_lens=[32],
        num_heads=6,
        packed=True,
        initial_state=True,
        seed=2032,
    )
    reference_inputs = {
        **inputs,
        "initial_state": inputs["initial_state"].clone(),
    }
    expected_output, expected_state = _reference(reference_inputs)
    output = torch.empty_like(inputs["q"])

    actual_output, actual_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        output_final_state=True,
        backend="cake",
    )

    assert actual_output.data_ptr() == output.data_ptr()
    assert actual_state is inputs["initial_state"]
    torch.testing.assert_close(
        actual_output.float(),
        expected_output.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual_state.float(),
        expected_state.float(),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("seq_len", [32, 33])
def test_frozen_prefill_h12_tma_chunks_match_reference(flash_kda_device, seq_len):
    inputs = _make_inputs(
        seq_lens=[seq_len],
        num_heads=12,
        packed=False,
        initial_state=True,
        seed=2012 + seq_len,
    )
    reference_inputs = {
        **inputs,
        "initial_state": inputs["initial_state"].clone(),
    }
    expected_output, expected_state = _chunk16_debug_reference(reference_inputs)
    output = torch.empty_like(inputs["q"])

    actual_output, actual_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        output_final_state=True,
        backend="cake",
    )

    assert actual_output.data_ptr() == output.data_ptr()
    assert actual_state is inputs["initial_state"]
    torch.testing.assert_close(
        actual_output.float(),
        expected_output.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual_state.float(),
        expected_state.float(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_frozen_prefill_h12_packed_matches_reference(flash_kda_device):
    inputs = _make_inputs(
        seq_lens=[32, 3],
        num_heads=12,
        packed=True,
        initial_state=True,
        seed=2047,
    )
    reference_inputs = {
        **inputs,
        "initial_state": inputs["initial_state"].clone(),
    }
    expected_output, expected_state = _chunk16_debug_reference(reference_inputs)
    output = torch.empty_like(inputs["q"])

    actual_output, actual_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        output_final_state=True,
        backend="cake",
    )

    assert actual_output.data_ptr() == output.data_ptr()
    assert actual_state is inputs["initial_state"]
    torch.testing.assert_close(
        actual_output.float(),
        expected_output.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual_state.float(),
        expected_state.float(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_frozen_prefill_h12_strided_beta_indexed_state_and_checkpoints_match_reference(
    flash_kda_device,
):
    checkpoint_interval = 16
    inputs = _make_inputs(
        seq_lens=[65, 131],
        num_heads=12,
        packed=True,
        initial_state=True,
        seed=2064,
    )
    compact_initial_state = inputs["initial_state"].clone()
    beta_carrier = torch.empty(
        (inputs["q"].shape[1], 32),
        dtype=torch.bfloat16,
        device=flash_kda_device,
    )
    beta_carrier[:, 8:20].copy_(inputs["beta"][0])
    inputs["beta"] = beta_carrier[None, :, 8:20]
    expected_output, expected_state, expected_checkpoints = _chunk16_debug_reference(
        {**inputs, "initial_state": compact_initial_state},
        checkpoint_every_n_tokens=checkpoint_interval,
    )

    state_slot_numel = 12 * 128 * 128
    state_storage = torch.zeros(
        (5, state_slot_numel + 64),
        dtype=torch.bfloat16,
        device=flash_kda_device,
    )
    state_pool = state_storage.as_strided(
        (5, 12, 128, 128),
        (state_storage.stride(0), 128 * 128, 128, 1),
    )
    state_indices = torch.tensor([1, 3], dtype=torch.int32, device=flash_kda_device)
    state_indices_i64 = state_indices.to(torch.int64)
    state_pool[state_indices_i64] = compact_initial_state
    untouched_before = state_pool[[0, 2, 4]].clone()
    inputs["initial_state"] = state_pool
    checkpoint_cu_starts = torch.tensor(
        [0, 5, 14], dtype=torch.int64, device=flash_kda_device
    )
    state_checkpoints = torch.empty(
        (14, 12, 128, 128), dtype=torch.bfloat16, device=flash_kda_device
    )

    actual_output, actual_state, actual_checkpoints = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        ssm_state_indices=state_indices,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_interval,
        backend="cake",
    )

    assert actual_state is state_pool
    assert actual_checkpoints is state_checkpoints
    assert inputs["beta"].data_ptr() == beta_carrier[:, 8:20].data_ptr()
    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        state_pool[state_indices_i64].float(),
        expected_state.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual_checkpoints.float(),
        expected_checkpoints.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(state_pool[[0, 2, 4]], untouched_before)


@pytest.mark.parametrize(
    ("seq_lens", "num_heads", "packed", "has_initial_state"),
    [
        ((33,), 96, False, False),
        ((33, 65), 12, True, True),
    ],
)
def test_cute_dsl_checkpoints_match_cake(
    flash_kda_device,
    seq_lens,
    num_heads,
    packed,
    has_initial_state,
):
    interval = 32
    inputs = _make_inputs(
        seq_lens=seq_lens,
        num_heads=num_heads,
        packed=packed,
        initial_state=has_initial_state,
        seed=2091 + num_heads,
    )
    counts = [(length + interval - 1) // interval for length in seq_lens]
    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count)
    checkpoint_cu_starts = torch.tensor(
        starts, dtype=torch.int64, device=flash_kda_device
    )

    results = {}
    for backend in ("cake", "cute-dsl"):
        backend_inputs = {
            **inputs,
            "initial_state": (
                inputs["initial_state"].clone()
                if inputs["initial_state"] is not None
                else None
            ),
        }
        checkpoints = torch.empty(
            starts[-1],
            num_heads,
            128,
            128,
            dtype=torch.bfloat16,
            device=flash_kda_device,
        )
        run_kwargs = {
            **_strict_prefill_kwargs(backend_inputs),
            "output": torch.empty_like(inputs["q"]),
            "output_final_state": True,
            "state_checkpoints": checkpoints,
            "checkpoint_cu_starts": checkpoint_cu_starts,
            "checkpoint_every_n_tokens": interval,
        }
        if backend == "cute-dsl" and packed:
            wrapper = RecurrentKDAPrefillWrapper(flash_kda_device)
            wrapper.plan(run_kwargs.pop("cu_seqlens"))
            results[backend] = wrapper.run(**run_kwargs)
        else:
            results[backend] = recurrent_kda(**run_kwargs, backend=backend)

    for cute_value, cake_value in zip(
        results["cute-dsl"], results["cake"], strict=True
    ):
        torch.testing.assert_close(
            cute_value.float(), cake_value.float(), atol=1e-2, rtol=1e-2
        )


@pytest.mark.parametrize(
    ("seq_lens", "num_heads", "packed"),
    [((17,), 96, False), ((17, 33), 12, True)],
)
def test_cute_dsl_padded_indexed_state_matches_cake(
    flash_kda_device, seq_lens, num_heads, packed
):
    inputs = _make_inputs(
        seq_lens=seq_lens,
        num_heads=num_heads,
        packed=packed,
        initial_state=True,
        seed=2117 + num_heads,
    )
    # Exercise Cake's minimum int32 alignment contract, including a +4-byte
    # contiguous view that is not 8-byte aligned.
    state_indices = torch.tensor(
        [0, 3, 1][: len(seq_lens) + 1],
        dtype=torch.int32,
        device=flash_kda_device,
    )[1:]

    def make_state_pool():
        slot_numel = num_heads * 128 * 128
        storage = torch.zeros(
            (4, slot_numel + 64),
            dtype=torch.bfloat16,
            device=flash_kda_device,
        )
        pool = storage.as_strided(
            (4, num_heads, 128, 128),
            (storage.stride(0), 128 * 128, 128, 1),
        )
        pool[state_indices.to(torch.int64)] = inputs["initial_state"]
        return pool

    results = {}
    for backend in ("cake", "cute-dsl"):
        backend_inputs = {**inputs, "initial_state": make_state_pool()}
        run_kwargs = {
            **_strict_prefill_kwargs(backend_inputs),
            "output": torch.empty_like(inputs["q"]),
            "output_final_state": True,
            "ssm_state_indices": state_indices,
        }
        if backend == "cute-dsl" and packed:
            wrapper = RecurrentKDAPrefillWrapper(flash_kda_device)
            wrapper.plan(run_kwargs.pop("cu_seqlens"))
            results[backend] = wrapper.run(**run_kwargs)
        else:
            results[backend] = recurrent_kda(**run_kwargs, backend=backend)

    for cute_value, cake_value in zip(
        results["cute-dsl"], results["cake"], strict=True
    ):
        torch.testing.assert_close(
            cute_value.float(), cake_value.float(), atol=1e-2, rtol=1e-2
        )


def test_frozen_prefill_m64_matches_reference(flash_kda_device):
    inputs = _make_inputs(
        seq_lens=[2],
        num_heads=64,
        packed=False,
        initial_state=True,
        seed=2027,
    )
    reference_inputs = {
        **inputs,
        "initial_state": inputs["initial_state"].clone(),
    }
    expected_output, expected_state = _reference(reference_inputs)
    output = torch.empty_like(inputs["q"])
    state_identity = inputs["initial_state"]

    actual_output, actual_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        output_final_state=True,
    )

    assert actual_output.data_ptr() == output.data_ptr()
    assert actual_state is state_identity
    torch.testing.assert_close(
        actual_output.float(),
        expected_output.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual_state.float(),
        expected_state.float(),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize(
    (
        "packed",
        "num_heads",
        "has_initial_state",
        "seq_lens",
        "output_final_state",
        "seed",
        "compare_eager_control",
    ),
    [
        (False, 64, True, (2,), True, 2028, False),
        (True, 2, False, (1, 2), True, 2028, False),
        (True, 96, True, (16,), False, 11018, True),
    ],
)
def test_frozen_prefill_cuda_graph_capture_and_replay(
    flash_kda_device,
    packed,
    num_heads,
    has_initial_state,
    seq_lens,
    output_final_state,
    seed,
    compare_eager_control,
):
    inputs = _make_inputs(
        seq_lens=seq_lens,
        num_heads=num_heads,
        packed=packed,
        initial_state=has_initial_state,
        seed=seed,
    )
    initial_state_seed = (
        inputs["initial_state"].clone() if inputs["initial_state"] is not None else None
    )
    expected_output, expected_state = _reference(
        {
            **inputs,
            "initial_state": (
                initial_state_seed.clone() if initial_state_seed is not None else None
            ),
        }
    )
    output = torch.empty_like(inputs["q"])
    seq_order = (
        torch.arange(
            len(seq_lens) - 1,
            -1,
            -1,
            dtype=torch.int32,
            device=flash_kda_device,
        )
        if packed
        else None
    )
    workspace = RecurrentKDAPrefillWorkspace(flash_kda_device)
    capture_stream = torch.cuda.Stream(device=flash_kda_device)
    capture_stream.wait_stream(torch.cuda.current_stream(flash_kda_device))

    call_kwargs = {
        **_strict_prefill_kwargs(inputs),
        "output": output,
        "output_final_state": output_final_state,
        "seq_order": seq_order,
        "prefill_workspace": workspace,
        "backend": "cake",
    }
    with torch.cuda.stream(capture_stream):
        warm_output, warm_state = recurrent_kda(**call_kwargs)
    capture_stream.synchronize()
    observed_warm_state = (
        inputs["initial_state"] if inputs["initial_state"] is not None else warm_state
    )
    assert observed_warm_state is not None
    with torch.cuda.stream(capture_stream):
        warm_output_control = warm_output.clone()
        warm_state_control = observed_warm_state.clone()
        if initial_state_seed is not None:
            inputs["initial_state"].copy_(initial_state_seed)
        output.zero_()
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output, captured_state = recurrent_kda(**call_kwargs)

    with torch.cuda.stream(capture_stream):
        if initial_state_seed is not None:
            inputs["initial_state"].copy_(initial_state_seed)
        output.fill_(float("nan"))
    capture_stream.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    assert captured_output.data_ptr() == output.data_ptr()
    if not output_final_state:
        assert captured_state is None
    elif inputs["initial_state"] is None:
        assert captured_state is not None
        assert captured_state.data_ptr() == workspace._state_scratch.data_ptr()
    else:
        assert captured_state is inputs["initial_state"]
    assert workspace._captured
    observed_captured_state = (
        inputs["initial_state"]
        if inputs["initial_state"] is not None
        else captured_state
    )
    assert observed_captured_state is not None
    if compare_eager_control:
        torch.testing.assert_close(
            warm_output_control.float(),
            expected_output.float(),
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            warm_state_control.float(),
            expected_state.float(),
            atol=1e-2,
            rtol=1e-2,
        )
        assert torch.equal(captured_output, warm_output_control)
        assert torch.equal(observed_captured_state, warm_state_control)
    else:
        torch.testing.assert_close(
            captured_output.float(),
            expected_output.float(),
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            observed_captured_state.float(),
            expected_state.float(),
            atol=1e-2,
            rtol=1e-2,
        )


@pytest.mark.parametrize("num_heads", [12, 64])
def test_cute_dsl_planned_zero_length_cuda_graph_capture_and_replay(
    flash_kda_device, num_heads
):
    inputs = _make_inputs(
        seq_lens=[0, 17, 0, 33],
        num_heads=num_heads,
        packed=True,
        initial_state=True,
        seed=2040 + num_heads,
    )
    initial_state_seed = inputs["initial_state"].clone()
    reference_inputs = {
        **inputs,
        "initial_state": initial_state_seed.clone(),
    }
    reference = _chunk16_debug_reference if num_heads == 12 else _reference
    expected_output, expected_state = reference(reference_inputs)

    wrapper = RecurrentKDAPrefillWrapper(flash_kda_device)
    wrapper.plan(inputs["cu_seqlens"])
    output = torch.empty_like(inputs["q"])
    run_kwargs = {
        **_strict_prefill_kwargs(inputs),
        "output": output,
        "output_final_state": True,
    }
    run_kwargs.pop("cu_seqlens")

    capture_stream = torch.cuda.Stream(device=flash_kda_device)
    capture_stream.wait_stream(torch.cuda.current_stream(flash_kda_device))
    with torch.cuda.stream(capture_stream):
        wrapper.run(**run_kwargs)
        inputs["initial_state"].copy_(initial_state_seed)
        output.zero_()
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output, captured_state = wrapper.run(**run_kwargs)

    with torch.cuda.stream(capture_stream):
        inputs["initial_state"].copy_(initial_state_seed)
        output.fill_(float("nan"))
    capture_stream.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    assert captured_output.data_ptr() == output.data_ptr()
    assert captured_state is inputs["initial_state"]
    assert wrapper._workspace._captured
    torch.testing.assert_close(
        captured_output.float(), expected_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        captured_state.float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("num_heads", [6, 12])
def test_frozen_prefill_non_aligned_heads_graph_refreshes_beta(
    flash_kda_device, num_heads
):
    inputs = _make_inputs(
        seq_lens=[32],
        num_heads=num_heads,
        packed=False,
        initial_state=True,
        seed=2033 + num_heads,
    )
    initial_state_seed = inputs["initial_state"].clone()
    output = torch.empty_like(inputs["q"])
    workspace = RecurrentKDAPrefillWorkspace(flash_kda_device)
    capture_stream = torch.cuda.Stream(device=flash_kda_device)
    capture_stream.wait_stream(torch.cuda.current_stream(flash_kda_device))
    call_kwargs = {
        **_strict_prefill_kwargs(inputs),
        "output": output,
        "output_final_state": True,
        "prefill_workspace": workspace,
    }

    with torch.cuda.stream(capture_stream):
        recurrent_kda(**call_kwargs)
        inputs["initial_state"].copy_(initial_state_seed)
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output, captured_state = recurrent_kda(**call_kwargs)

    # Establish an original-beta replay result before mutating graph inputs.
    with torch.cuda.stream(capture_stream):
        inputs["initial_state"].copy_(initial_state_seed)
        output.fill_(float("nan"))
    capture_stream.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    original_output = captured_output.clone()
    original_state = captured_state.clone()

    # The captured public call must repack the changed beta values on replay.
    with torch.cuda.stream(capture_stream):
        inputs["beta"].fill_(2.0)
        inputs["initial_state"].copy_(initial_state_seed)
        output.fill_(float("nan"))
    capture_stream.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    # Compare against a separate eager launch with distinct tensors/workspace.
    eager_inputs = {
        name: value.clone() if value is not None else None
        for name, value in inputs.items()
    }
    eager_inputs["initial_state"] = initial_state_seed.clone()
    eager_output_storage = torch.empty_like(output)
    eager_workspace = RecurrentKDAPrefillWorkspace(flash_kda_device)
    eager_output, eager_state = recurrent_kda(
        **_strict_prefill_kwargs(eager_inputs),
        output=eager_output_storage,
        output_final_state=True,
        prefill_workspace=eager_workspace,
    )
    torch.cuda.synchronize()

    assert captured_output.data_ptr() == output.data_ptr()
    assert captured_state is inputs["initial_state"]
    assert eager_output.data_ptr() == eager_output_storage.data_ptr()
    assert eager_state is eager_inputs["initial_state"]
    assert torch.equal(captured_output, eager_output)
    assert torch.equal(captured_state, eager_state)
    assert not torch.equal(captured_output, original_output)
    assert not torch.equal(captured_state, original_state)


def test_frozen_prefill_cuda_graph_workspaces_are_isolated(flash_kda_device):
    capture_stream = torch.cuda.Stream(device=flash_kda_device)
    launch_stream = torch.cuda.Stream(device=flash_kda_device)
    bundles = []

    for seed in (2030, 2031):
        inputs = _make_inputs(
            seq_lens=[2],
            num_heads=2,
            packed=False,
            initial_state=True,
            seed=seed,
        )
        state_seed = inputs["initial_state"].clone()
        expected_output, expected_state = _reference(
            {
                **inputs,
                "initial_state": state_seed.clone(),
            }
        )
        output = torch.empty_like(inputs["q"])
        workspace = RecurrentKDAPrefillWorkspace(flash_kda_device)
        call_kwargs = {
            **_strict_prefill_kwargs(inputs),
            "output": output,
            "output_final_state": True,
            "prefill_workspace": workspace,
        }
        capture_stream.wait_stream(torch.cuda.current_stream(flash_kda_device))
        with torch.cuda.stream(capture_stream):
            recurrent_kda(**call_kwargs)
            inputs["initial_state"].copy_(state_seed)
            output.zero_()
        capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            recurrent_kda(**call_kwargs)
        bundles.append(
            (
                graph,
                workspace,
                inputs,
                state_seed,
                output,
                expected_output,
                expected_state,
            )
        )

    assert bundles[0][1]._state_scratch is None
    assert bundles[1][1]._state_scratch is None
    assert (
        bundles[0][1]._descriptor_storages["m128"].data_ptr()
        != bundles[1][1]._descriptor_storages["m128"].data_ptr()
    )

    for bundle_index in (0, 1, 0, 1):
        (
            graph,
            _workspace,
            inputs,
            state_seed,
            output,
            expected_output,
            expected_state,
        ) = bundles[bundle_index]
        with torch.cuda.stream(launch_stream):
            inputs["initial_state"].copy_(state_seed)
            output.fill_(float("nan"))
        launch_stream.synchronize()
        with torch.cuda.stream(launch_stream):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output.float(),
            expected_output.float(),
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            inputs["initial_state"].float(),
            expected_state.float(),
            atol=1e-2,
            rtol=1e-2,
        )
