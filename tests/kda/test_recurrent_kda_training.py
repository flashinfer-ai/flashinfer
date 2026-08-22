# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import inspect
import math
import os

import pytest
import torch

from flashinfer import kda_training as kda_training_api
from flashinfer.kda_training import (
    RecurrentKDATrainingContext,
    recurrent_kda_training_backward,
    recurrent_kda_training_forward,
)


class _ForwardRecorder:
    def __init__(self):
        self.calls = []

    def run_forward(self, *args):
        self.calls.append(args)


class _BackwardRecorder:
    def __init__(self):
        self.calls = []

    def run_c16_backward(self, *args):
        self.calls.append(args)


class _FinalStateRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        kwargs["state_scratch"].copy_(kwargs["initial_state"])
        kwargs["final_state"].copy_(kwargs["state_scratch"])


def test_training_api_signatures_and_no_forward_recompute():
    forward_parameters = tuple(
        inspect.signature(recurrent_kda_training_forward).parameters
    )
    backward_parameters = tuple(
        inspect.signature(recurrent_kda_training_backward).parameters
    )
    assert forward_parameters[:9] == (
        "q",
        "k",
        "v",
        "g",
        "beta",
        "A_log",
        "dt_bias",
        "initial_state",
        "cu_seqlens",
    )
    assert backward_parameters == ("context", "do", "dfinal_state", "out")
    source = inspect.getsource(recurrent_kda_training_backward)
    assert "run_c16_backward" in source
    assert "run_forward" not in source
    assert "_get_training_module" not in source


def _require_blackwell():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
        pytest.skip("the recurrent KDA training route requires SM100a or SM103a")


def _make_inputs(seed=819208):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    token_shape = (1, 8192, 96, 128)
    state_shape = (8, 96, 128, 128)

    def bf16(shape, multiplier=1.0):
        return (torch.randn(shape, generator=generator, device="cuda") * multiplier).to(
            torch.bfloat16
        )

    return {
        "q": bf16(token_shape),
        "k": bf16(token_shape),
        "v": bf16(token_shape),
        "g": bf16(token_shape, 0.1),
        "beta": bf16(token_shape[:-1]),
        "A_log": torch.log(torch.rand((96,), generator=generator, device="cuda") + 1.0),
        "dt_bias": torch.randn((96, 128), generator=generator, device="cuda") * 0.1,
        "initial_state": torch.randn(state_shape, generator=generator, device="cuda")
        * 0.02,
        "cu_seqlens": torch.arange(0, 8193, 1024, dtype=torch.int64, device="cuda"),
        "do": bf16(token_shape, 0.1),
        "dfinal_state": torch.randn(state_shape, generator=generator, device="cuda")
        * 0.1,
    }


def _fla_reference(inputs):
    os.environ["FLA_FLASH_KDA"] = "0"
    kda_ops = pytest.importorskip("fla.ops.kda")
    names = ("q", "k", "v", "g", "beta", "A_log", "dt_bias", "initial_state")
    leaves = {
        name: inputs[name].detach().clone().requires_grad_(True) for name in names
    }
    leaves["dt_bias"] = (
        inputs["dt_bias"].detach().reshape(-1).clone().requires_grad_(True)
    )
    output, final_state = kda_ops.chunk_kda(
        leaves["q"],
        leaves["k"],
        leaves["v"],
        leaves["g"],
        leaves["beta"],
        scale=1.0 / math.sqrt(128),
        initial_state=leaves["initial_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        state_v_first=True,
        cu_seqlens=inputs["cu_seqlens"],
        cu_seqlens_cpu=inputs["cu_seqlens"].detach().cpu(),
        A_log=leaves["A_log"],
        dt_bias=leaves["dt_bias"],
        chunk_size=32,
    )
    gradients = torch.autograd.grad(
        (output, final_state),
        tuple(leaves[name] for name in names),
        grad_outputs=(inputs["do"], inputs["dfinal_state"]),
    )
    gradients = (
        *gradients[:-2],
        gradients[-2].reshape_as(inputs["dt_bias"]),
        gradients[-1],
    )
    return output, final_state, gradients


def test_training_forward_context_backward_matches_fla():
    _require_blackwell()
    inputs = _make_inputs()
    expected_output, expected_final, expected_gradients = _fla_reference(inputs)
    initial_state_before = inputs["initial_state"].clone()
    output, final_state, context = recurrent_kda_training_forward(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["A_log"],
        inputs["dt_bias"],
        inputs["initial_state"],
        inputs["cu_seqlens"],
    )
    assert isinstance(context, RecurrentKDATrainingContext)
    assert output.dtype == torch.bfloat16
    assert final_state.dtype == torch.float32
    assert torch.equal(inputs["initial_state"], initial_state_before)
    torch.testing.assert_close(output, expected_output, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(final_state, expected_final, atol=1e-2, rtol=1e-2)
    gradients = recurrent_kda_training_backward(
        context, inputs["do"], inputs["dfinal_state"]
    )
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


def test_paired_backward_ffi_does_not_rerun_forward(monkeypatch):
    _require_blackwell()
    forward_module = _ForwardRecorder()
    backward_module = _BackwardRecorder()
    final_state_runner = _FinalStateRecorder()
    monkeypatch.setattr(
        kda_training_api, "_get_training_module", lambda _: forward_module
    )
    monkeypatch.setattr(
        kda_training_api, "_get_backward_module", lambda _: backward_module
    )
    monkeypatch.setattr(
        kda_training_api, "_run_final_state_recurrence", final_state_runner
    )
    inputs = _make_inputs(seed=1024)
    output, final_state, context = recurrent_kda_training_forward(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["A_log"],
        inputs["dt_bias"],
        inputs["initial_state"],
        inputs["cu_seqlens"],
    )
    final_state_output_ptr = context._final_state_recurrence_output.data_ptr()
    final_state_scratch_ptr = context._final_state_bf16.data_ptr()
    assert final_state_output_ptr != output.data_ptr()
    assert len(final_state_runner.calls) == 1
    assert (
        final_state_runner.calls[0]["output_scratch"].data_ptr()
        == final_state_output_ptr
    )
    with pytest.raises(
        ValueError, match="out must not overlap context._final_state_recurrence_output"
    ):
        recurrent_kda_training_forward(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            inputs["A_log"],
            inputs["dt_bias"],
            inputs["initial_state"],
            inputs["cu_seqlens"],
            out=context._final_state_recurrence_output,
            final_state_out=final_state,
            context_out=context,
        )
    _, _, reused_context = recurrent_kda_training_forward(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["A_log"],
        inputs["dt_bias"],
        inputs["initial_state"],
        inputs["cu_seqlens"],
        out=output,
        final_state_out=final_state,
        context_out=context,
    )
    assert reused_context is context
    assert context._final_state_recurrence_output.data_ptr() == final_state_output_ptr
    assert context._final_state_bf16.data_ptr() == final_state_scratch_ptr
    assert len(final_state_runner.calls) == 2
    recurrent_kda_training_backward(context, inputs["do"], inputs["dfinal_state"])
    assert len(forward_module.calls) == 2
    assert len(backward_module.calls) == 1

    other_stream = torch.cuda.Stream(device=inputs["q"].device)
    with torch.cuda.stream(other_stream):
        with pytest.raises(RuntimeError, match="must run on the forward stream"):
            recurrent_kda_training_backward(
                context, inputs["do"], inputs["dfinal_state"]
            )
        with pytest.raises(RuntimeError, match="reused on its forward stream"):
            recurrent_kda_training_forward(
                inputs["q"],
                inputs["k"],
                inputs["v"],
                inputs["g"],
                inputs["beta"],
                inputs["A_log"],
                inputs["dt_bias"],
                inputs["initial_state"],
                inputs["cu_seqlens"],
                out=output,
                final_state_out=final_state,
                context_out=context,
            )
    assert len(forward_module.calls) == 2
    assert len(backward_module.calls) == 1

    inputs["q"][0, 0, 0, 0] = inputs["q"][0, 0, 0, 0]
    with pytest.raises(RuntimeError, match="input was modified after forward"):
        recurrent_kda_training_backward(context, inputs["do"], inputs["dfinal_state"])
    assert len(forward_module.calls) == 2
    assert len(backward_module.calls) == 1
