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


class _TrainingRecorder:
    def __init__(self):
        self.forward_calls = []
        self.backward_calls = []
        self.row_forward_calls = []
        self.row_backward_calls = []
        self.grouped_row_forward_calls = []
        self.grouped_row_backward_calls = []
        self.c32_forward_calls = []
        self.c32_backward_calls = []

    def run_training_forward(self, *args):
        assert len(args) == 37
        self.forward_calls.append(args)

    def run_training_backward(self, *args):
        assert len(args) == 43
        self.backward_calls.append(args)

    def run_training_row_forward(self, *args):
        assert len(args) == 28
        self.row_forward_calls.append(args)

    def run_training_row_backward(self, *args):
        assert len(args) == 33
        self.row_backward_calls.append(args)

    def run_training_grouped_row_forward(self, *args):
        assert len(args) == 31
        self.grouped_row_forward_calls.append(args)

    def run_training_grouped_row_backward(self, *args):
        assert len(args) == 38
        self.grouped_row_backward_calls.append(args)

    def run_training_c32_forward(self, *args):
        assert len(args) == 52
        self.c32_forward_calls.append(args)

    def run_training_c32_backward(self, *args):
        assert len(args) == 60
        self.c32_backward_calls.append(args)


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
    assert (
        inspect.signature(recurrent_kda_training_forward)
        .parameters["cu_seqlens"]
        .default
        is None
    )
    assert backward_parameters == ("context", "do", "dfinal_state", "out")
    source = inspect.getsource(recurrent_kda_training_backward)
    assert "run_training_backward" in source
    assert "run_training_forward" not in source
    assert ".run_low(" not in source
    assert ".run_high(" not in source


@pytest.mark.parametrize(
    ("seq_lens", "num_qk_heads", "num_v_heads", "tag", "family"),
    [
        ((1024,) * 8, 4, 8, "grouped_hybrid_c16_c32", "c16"),
        ((17, 33, 65), 4, 8, "grouped_row_split", "row_split"),
        ((17,), 1, 1, "row_split", "row_split"),
        ((1024,) * 8, 96, 96, "c16", "c16"),
        ((1300, 547, 2048, 963, 271, 3063), 96, 96, "c32", "c32"),
    ],
)
def test_full_training_dispatcher_route_selector(
    seq_lens, num_qk_heads, num_v_heads, tag, family
):
    """The analytical selector chooses a strict route without shape guards."""

    spec = kda_training_api._select_training_route(seq_lens, num_qk_heads, num_v_heads)
    assert spec.tag == tag
    assert spec.family == family


@pytest.mark.parametrize(
    ("batch_size", "seq_len", "num_heads", "tag"),
    [
        (2, 17, 1, "row_split"),
        (4, 33, 16, "row_split"),
        (8, 64, 32, "c16"),
    ],
)
def test_fixed_batch_route_selection_uses_semantic_sequences(
    batch_size, seq_len, num_heads, tag
):
    spec = kda_training_api._select_training_route(
        (seq_len,) * batch_size, num_heads, num_heads
    )
    assert spec.tag == tag


def test_public_contract_does_not_promote_fast_path_predicates_to_guards():
    source = inspect.getsource(kda_training_api._validate_forward_inputs)
    assert "divisible by 16" not in source
    assert "validated C16 production routes only" not in source
    assert "grouped Q/K heads only" not in source
    assert "torch.bfloat16" in source
    assert "torch.float32" in source


def test_analytical_selector_preserves_template_choice_and_strict_adapter():
    grouped = kda_training_api._select_training_route((3200,) * 9 + (3968,), 4, 8)
    equal = kda_training_api._select_training_route((2656,) * 11 + (3552,), 4, 4)
    assert grouped.selected_template == "checkpoint_recurrent_c16"
    assert grouped.tag == "grouped_hybrid_c16_c32"
    assert grouped.uses_parameter_context
    assert equal.selected_template == "checkpoint_recurrent_c16"
    assert equal.tag == "c32"
    assert not equal.uses_parameter_context


@pytest.mark.parametrize(
    ("seq_lens", "heads", "template"),
    [
        ((1024,) * 2, 96, "tensor_tape_c32"),
        ((1024,) * 4, 96, "checkpoint_recurrent_c16"),
        ((2048,) * 4, 96, "tensor_tape_c32"),
        ((2048,) * 5, 96, "checkpoint_recurrent_c16"),
        ((512,), 8, "row_warp_checkpoint"),
        ((17,), 1, "row_warp_checkpoint"),
    ],
)
def test_analytical_template_crossovers(seq_lens, heads, template):
    spec = kda_training_api._select_training_route(
        seq_lens, heads, heads, resident_sms=152
    )
    assert spec.selected_template == template


@pytest.mark.parametrize("resident_sms", [148, 152, 160])
def test_all_customer_layouts_select_c16_template(resident_sms):
    layouts = {
        "a": (3200,) * 9 + (3968,),
        "b": (2000,) * 8 + (2432,),
        "c": (2656,) * 11 + (3552,),
        "d": (1648,) * 10 + (1952,),
    }
    rows = (
        (4, 8, "a"),
        (2, 8, "b"),
        (4, 4, "c"),
        (2, 4, "d"),
        (4, 8, "b"),
        (2, 8, "a"),
        (2, 4, "b"),
        (4, 4, "d"),
        (2, 8, "c"),
        (4, 8, "d"),
        (4, 4, "a"),
        (2, 4, "c"),
        (4, 4, "b"),
        (2, 4, "a"),
        (4, 8, "c"),
        (2, 8, "d"),
    )
    for qk_heads, value_heads, layout in rows:
        spec = kda_training_api._select_training_route(
            layouts[layout],
            qk_heads,
            value_heads,
            resident_sms=resident_sms,
        )
        assert spec.selected_template == "checkpoint_recurrent_c16"


def test_context_has_an_explicit_route_tag():
    assert "_route" in RecurrentKDATrainingContext.__dataclass_fields__


def _require_blackwell():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
        pytest.skip("the recurrent KDA training route requires SM100a or SM103a")


def _make_inputs(
    seed=819208,
    *,
    seq_lens=(1024,) * 8,
    num_qk_heads=96,
    num_v_heads=None,
):
    num_v_heads = num_qk_heads if num_v_heads is None else num_v_heads
    generator = torch.Generator(device="cuda").manual_seed(seed)
    total_tokens = sum(seq_lens)
    qk_shape = (1, total_tokens, num_qk_heads, 128)
    value_shape = (1, total_tokens, num_v_heads, 128)
    state_shape = (len(seq_lens), num_v_heads, 128, 128)

    def bf16(shape, multiplier=1.0):
        return (torch.randn(shape, generator=generator, device="cuda") * multiplier).to(
            torch.bfloat16
        )

    return {
        "q": bf16(qk_shape),
        "k": bf16(qk_shape),
        "v": bf16(value_shape),
        "g": bf16(value_shape, 0.1),
        "beta": bf16(value_shape[:-1]),
        "A_log": torch.log(
            torch.rand((num_v_heads,), generator=generator, device="cuda") + 1.0
        ),
        "dt_bias": torch.randn((num_v_heads, 128), generator=generator, device="cuda")
        * 0.1,
        "initial_state": torch.randn(state_shape, generator=generator, device="cuda")
        * 0.02,
        "cu_seqlens": torch.tensor(
            [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
            dtype=torch.int64,
            device="cuda",
        ),
        "do": bf16(value_shape, 0.1),
        "dfinal_state": torch.randn(state_shape, generator=generator, device="cuda")
        * 0.1,
    }


def _make_fixed_inputs(
    batch_size,
    seq_len,
    num_heads,
    *,
    seed,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    token_shape = (batch_size, seq_len, num_heads, 128)
    state_shape = (batch_size, num_heads, 128, 128)

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
        "A_log": torch.log(
            torch.rand((num_heads,), generator=generator, device="cuda") + 1.0
        ),
        "dt_bias": torch.randn((num_heads, 128), generator=generator, device="cuda")
        * 0.1,
        "initial_state": torch.randn(state_shape, generator=generator, device="cuda")
        * 0.02,
        "cu_seqlens": None,
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
    cu_seqlens = inputs["cu_seqlens"]
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
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=(None if cu_seqlens is None else cu_seqlens.detach().cpu()),
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


def _assert_training_matches_fla(inputs, expected_route):
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
    assert context._route.tag == expected_route
    assert output.dtype == torch.bfloat16
    assert final_state.dtype == torch.float32
    assert output.shape == inputs["v"].shape
    assert final_state.shape == inputs["initial_state"].shape
    gradients = recurrent_kda_training_backward(
        context, inputs["do"], inputs["dfinal_state"]
    )
    assert torch.equal(inputs["initial_state"], initial_state_before)
    torch.testing.assert_close(output, expected_output, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(final_state, expected_final, atol=1e-2, rtol=1e-2)
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.arch_blackwell
def test_training_forward_context_backward_matches_fla():
    _require_blackwell()
    _assert_training_matches_fla(_make_inputs(), "c16")


@pytest.mark.arch_blackwell
def test_grouped_row_forward_context_backward_matches_fla():
    """The grouped WG8 row fallback is part of the public API."""

    _require_blackwell()
    inputs = _make_inputs(
        seed=24005,
        seq_lens=(17, 33, 65),
        num_qk_heads=4,
        num_v_heads=8,
    )
    _assert_training_matches_fla(inputs, "grouped_row_split")


@pytest.mark.arch_blackwell
def test_grouped_c32_forward_context_backward_matches_fla():
    """The grouped C32 template handles deeper unaligned recurrence."""

    _require_blackwell()
    inputs = _make_inputs(
        seed=24105,
        seq_lens=(4097,),
        num_qk_heads=1,
        num_v_heads=8,
    )
    _assert_training_matches_fla(inputs, "grouped_c32")


@pytest.mark.arch_blackwell
def test_grouped_hybrid_forward_context_backward_matches_fla():
    """The strict grouped route saves both contexts before backward."""

    _require_blackwell()
    inputs = _make_inputs(
        seed=24001,
        seq_lens=(1024,),
        num_qk_heads=4,
        num_v_heads=8,
    )
    _assert_training_matches_fla(inputs, "grouped_hybrid_c16_c32")


@pytest.mark.arch_blackwell
def test_high_head_mixed_row_forward_context_backward_matches_fla():
    """Mixed short packed lengths use the analytical row template."""

    _require_blackwell()
    inputs = _make_inputs(
        seed=24017,
        seq_lens=(17, 33),
        num_qk_heads=16,
        num_v_heads=16,
    )
    _assert_training_matches_fla(inputs, "row_split")


@pytest.mark.arch_blackwell
def test_short_high_head_row_forward_context_backward_matches_fla():
    """A short high-head row remains in the public legal domain."""

    _require_blackwell()
    inputs = _make_inputs(
        seed=24020,
        seq_lens=(17,),
        num_qk_heads=16,
        num_v_heads=16,
    )
    _assert_training_matches_fla(inputs, "row_split")


@pytest.mark.arch_blackwell
def test_split_c32_forward_context_backward_matches_fla():
    """A multi-work-item C32 route preserves the saved training tape."""

    _require_blackwell()
    inputs = _make_inputs(
        seed=24019,
        seq_lens=(2049,),
        num_qk_heads=16,
        num_v_heads=16,
    )
    _assert_training_matches_fla(inputs, "c32")


@pytest.mark.arch_blackwell
def test_packed_row_split_forward_context_backward_matches_fla():
    """The packed row-split benchmark shape is covered by exact correctness."""

    _require_blackwell()
    inputs = _make_inputs(
        seed=24018,
        seq_lens=(17, 33),
        num_qk_heads=1,
        num_v_heads=1,
    )
    _assert_training_matches_fla(inputs, "row_split")


@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    ("batch_size", "seq_len", "num_heads", "route"),
    [
        (2, 17, 1, "row_split"),
        (4, 33, 16, "row_split"),
        (8, 64, 32, "c16"),
    ],
)
def test_fixed_batch_normalization_output_shape_and_correctness(
    batch_size, seq_len, num_heads, route
):
    _require_blackwell()
    inputs = _make_fixed_inputs(batch_size, seq_len, num_heads, seed=9100 + batch_size)
    _assert_training_matches_fla(inputs, route)


def test_paired_backward_ffi_does_not_rerun_forward(monkeypatch):
    _require_blackwell()
    training_module = _TrainingRecorder()
    monkeypatch.setattr(
        kda_training_api, "_get_training_module", lambda _: training_module
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
    final_state_output_ptr = context._final_output_scratch.data_ptr()
    assert final_state_output_ptr != output.data_ptr()
    rejected_q = torch.empty_like(inputs["q"])
    with pytest.raises(ValueError, match="out must not overlap q"):
        recurrent_kda_training_forward(
            rejected_q,
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            inputs["A_log"],
            inputs["dt_bias"],
            inputs["initial_state"],
            inputs["cu_seqlens"],
            out=rejected_q,
            final_state_out=final_state,
            context_out=context,
        )
    assert context._q is inputs["q"]
    with pytest.raises(
        ValueError, match="out must not overlap context._final_output_scratch"
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
            out=context._final_output_scratch,
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
    assert context._final_output_scratch.data_ptr() == final_state_output_ptr
    assert training_module.forward_calls[-1][33] == 0
    context._final_descriptor_storage[0] = context._final_descriptor_storage[0]
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
    assert training_module.forward_calls[-1][33] == 1
    recurrent_kda_training_backward(context, inputs["do"], inputs["dfinal_state"])
    assert len(training_module.forward_calls) == 3
    assert len(training_module.backward_calls) == 1

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
    assert len(training_module.forward_calls) == 3
    assert len(training_module.backward_calls) == 1

    inputs["q"][0, 0, 0, 0] = inputs["q"][0, 0, 0, 0]
    with pytest.raises(RuntimeError, match="input was modified after forward"):
        recurrent_kda_training_backward(context, inputs["do"], inputs["dfinal_state"])
    assert len(training_module.forward_calls) == 3
    assert len(training_module.backward_calls) == 1


@pytest.mark.parametrize(
    ("route", "inputs"),
    [
        (
            "row_split",
            lambda: _make_inputs(
                seed=24018,
                seq_lens=(17, 33),
                num_qk_heads=1,
                num_v_heads=1,
            ),
        ),
        (
            "grouped_row_split",
            lambda: _make_inputs(
                seed=24005,
                seq_lens=(17, 33, 65),
                num_qk_heads=4,
                num_v_heads=8,
            ),
        ),
        (
            "grouped_c32",
            lambda: _make_inputs(
                seed=24105,
                seq_lens=(4097,),
                num_qk_heads=1,
                num_v_heads=8,
            ),
        ),
    ],
)
def test_fallback_ffi_consumes_route_context_without_recompute(
    monkeypatch, route, inputs
):
    _require_blackwell()
    training_module = _TrainingRecorder()
    monkeypatch.setattr(
        kda_training_api, "_get_training_module", lambda _: training_module
    )
    values = inputs()
    _, _, context = recurrent_kda_training_forward(
        values["q"],
        values["k"],
        values["v"],
        values["g"],
        values["beta"],
        values["A_log"],
        values["dt_bias"],
        values["initial_state"],
        values["cu_seqlens"],
    )
    assert context._route.tag == route
    recurrent_kda_training_backward(context, values["do"], values["dfinal_state"])
    if route == "row_split":
        assert len(training_module.row_forward_calls) == 1
        assert len(training_module.row_backward_calls) == 1
    elif route == "grouped_row_split":
        assert len(training_module.grouped_row_forward_calls) == 1
        assert len(training_module.grouped_row_backward_calls) == 1
        forward_args = training_module.grouped_row_forward_calls[0]
        assert forward_args[2] is context._route_tensors["q_value_heads"]
        assert forward_args[3] is context._route_tensors["k_value_heads"]
        assert forward_args[24] == context._shape.num_qk_heads
        assert forward_args[25] == context._shape.num_v_heads
    else:
        assert len(training_module.c32_forward_calls) == 1
        assert len(training_module.c32_backward_calls) == 1
        forward_args = training_module.c32_forward_calls[0]
        assert forward_args[12] is context._metadata["work_items"]
        assert forward_args[13] is context._metadata["seq_order"]
        assert forward_args[32].data_ptr() == forward_args[16].data_ptr()
        assert forward_args[32].data_ptr() != context._final_output_scratch.data_ptr()
        assert forward_args[43] == 0
        assert forward_args[44] == 1
    assert not training_module.forward_calls
    assert not training_module.backward_calls


def test_grouped_hybrid_materializes_both_contexts_during_forward(monkeypatch):
    _require_blackwell()
    training_module = _TrainingRecorder()
    monkeypatch.setattr(
        kda_training_api, "_get_training_module", lambda _: training_module
    )
    values = _make_inputs(
        seed=24001,
        seq_lens=(1024,),
        num_qk_heads=4,
        num_v_heads=8,
    )
    _, _, context = recurrent_kda_training_forward(
        values["q"],
        values["k"],
        values["v"],
        values["g"],
        values["beta"],
        values["A_log"],
        values["dt_bias"],
        values["initial_state"],
        values["cu_seqlens"],
    )
    assert context._route.tag == "grouped_hybrid_c16_c32"
    assert context._parameter_context is not None
    assert context._parameter_context._route.tag == "grouped_c32"
    assert len(training_module.forward_calls) == 1
    assert len(training_module.c32_forward_calls) == 1
    assert training_module.c32_forward_calls[0][47] == 0

    gradients = recurrent_kda_training_backward(
        context, values["do"], values["dfinal_state"]
    )
    assert len(training_module.backward_calls) == 1
    assert len(training_module.c32_backward_calls) == 1
    c32_outputs = training_module.c32_backward_calls[0]
    assert c32_outputs[43] is gradients[5]
    assert c32_outputs[44] is gradients[6]
    assert len(training_module.forward_calls) == 1
    assert len(training_module.c32_forward_calls) == 1


def test_training_rejects_cuda_graph_capture_before_ffi(monkeypatch):
    _require_blackwell()
    training_module = _TrainingRecorder()
    monkeypatch.setattr(
        kda_training_api, "_get_training_module", lambda _: training_module
    )
    inputs = _make_inputs(
        seed=1025,
        seq_lens=(1024,),
        num_qk_heads=4,
        num_v_heads=4,
    )
    _, _, context = recurrent_kda_training_forward(
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
    assert len(training_module.forward_calls) == 1
    assert len(training_module.backward_calls) == 0

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="does not support CUDA graph capture"):
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
        )
    with pytest.raises(RuntimeError, match="does not support CUDA graph capture"):
        recurrent_kda_training_backward(context, inputs["do"], inputs["dfinal_state"])

    assert len(training_module.forward_calls) == 1
    assert len(training_module.backward_calls) == 0


def test_saved_context_mutation_rejected_before_ffi(monkeypatch):
    _require_blackwell()
    training_module = _TrainingRecorder()
    monkeypatch.setattr(
        kda_training_api, "_get_training_module", lambda _: training_module
    )
    inputs = _make_inputs(
        seed=1026,
        seq_lens=(1024,),
        num_qk_heads=4,
        num_v_heads=4,
    )
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
    context.state_checkpoints.zero_()

    with pytest.raises(RuntimeError, match="context was modified after forward"):
        recurrent_kda_training_backward(context, inputs["do"], inputs["dfinal_state"])
    with pytest.raises(RuntimeError, match="context was modified after forward"):
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

    assert len(training_module.forward_calls) == 1
    assert len(training_module.backward_calls) == 0
