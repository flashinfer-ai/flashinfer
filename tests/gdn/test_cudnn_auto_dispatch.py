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

"""Raw-gate auto preserves semantics and retries only before execution.

Force the performance preference in contract tests so future tuning does not
require changing assertions about shapes or architecture thresholds.
"""

import importlib
import importlib.metadata
from types import SimpleNamespace

import pytest
import torch

from flashinfer.cudnn import linear_attention
from tests.test_helpers.cudnn_linear_attention import (
    assert_rel_close,
    requires_cudnn_linear_attention,
    serial_delta_rule,
)

gdn = importlib.import_module("flashinfer.gdn_prefill")
cudnn_adapter = importlib.import_module("flashinfer.cudnn")


def _inputs(device="cpu", lengths=(3, 5), dim=128, seed=71):
    generator = torch.Generator(device=device).manual_seed(seed)
    total = sum(lengths)

    def random(*shape):
        return torch.randn(shape, device=device, generator=generator)

    return dict(
        q=torch.nn.functional.normalize(random(total, 4, dim), dim=-1).bfloat16(),
        k=torch.nn.functional.normalize(random(total, 4, dim), dim=-1).bfloat16(),
        v=random(total, 8, dim).bfloat16(),
        g=random(total, 8).bfloat16(),
        beta=random(total, 8).bfloat16(),
        A_log=torch.full((8,), -2.0, device=device),
        dt_bias=torch.linspace(-0.2, 0.3, 8, device=device),
        initial_state=random(len(lengths), 8, dim, dim) * 0.05,
        output_state=torch.full((len(lengths), 8, dim, dim), -11.0, device=device),
        output=torch.full((total, 8, dim), -13.0, dtype=torch.bfloat16, device=device),
        cu_seqlens=torch.tensor([0, *lengths], device=device, dtype=torch.int64).cumsum(
            0
        ),
        output_final_state=True,
        use_gate_in_kernel=True,
        beta_is_logit=True,
    )


@pytest.fixture
def routes(monkeypatch):
    seen = []
    monkeypatch.setattr(gdn, "_prefer_cudnn_gdn_prefill", lambda *args: True)
    monkeypatch.setattr(gdn, "_cudnn_gdn_prefill_available", lambda: True)
    # Only CUDA metadata is emulated; all shape/dtype/layout/alias admission
    # checks and public gate validation run normally on real CPU tensors.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(gdn, "get_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(gdn, "get_device_sm_count", lambda device: 148)
    monkeypatch.setattr(gdn, "get_device_name", lambda device: "NVIDIA B200")
    monkeypatch.setattr(gdn, "should_use_cp_host", lambda *args, **kwargs: True)
    monkeypatch.setattr(torch.version, "cuda", "13.0")

    def native(output, output_state, q, k, v, g, beta, cu_seqlens, scale, **kwargs):
        seen.append(("native", dict(g=g, beta=beta, **kwargs)))
        output.copy_(v)
        if output_state is not None:
            output_state.copy_(kwargs["initial_state"] + 1)

    def non_cp(
        q,
        k,
        v,
        g,
        beta,
        output,
        cu_seqlens,
        initial_state,
        output_state,
        scale,
        **kwargs,
    ):
        native(
            output,
            output_state,
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            scale,
            initial_state=initial_state,
            **kwargs,
        )

    def cudnn(q, k, v, g, beta, scale, **kwargs):
        seen.append(("cudnn", dict(q=q, k=k, v=v, g=g, beta=beta, **kwargs)))
        kwargs["output"].copy_(v)
        kwargs["output_state"].copy_(kwargs["initial_state"] + 1)
        return kwargs["output"], kwargs["output_state"]

    monkeypatch.setattr(gdn, "cp_delta_rule_dsl_sm100", native)
    monkeypatch.setattr(gdn, "chunk_gated_delta_rule_sm100", non_cp)
    monkeypatch.setattr(cudnn_adapter, "cudnn_chunk_gated_delta_rule", cudnn)
    return seen


def test_raw_auto_preserves_fresh_bindings_and_input_state(routes, monkeypatch):
    gates = importlib.import_module("flashinfer.gdn_kernels.gates")

    def materialization_is_not_needed(*args):
        raise AssertionError("cuDNN must receive the original raw gates")

    monkeypatch.setattr(gates, "materialize_gates", materialization_is_not_needed)
    retained = []
    for seed in (71, 79):
        values = _inputs(seed=seed)
        retained.append(values)
        before = values["initial_state"].clone()
        output, state = gdn.chunk_gated_delta_rule(**values)
        provider, call = routes[-1]
        assert provider == "cudnn"
        for name in ("q", "k", "v", "g", "beta", "initial_state", "A_log", "dt_bias"):
            assert call[name] is values[name]
        assert output is values["output"] and state is values["output_state"]
        torch.testing.assert_close(values["initial_state"], before)
        torch.testing.assert_close(state, before + 1)
    assert retained[0]["g"].data_ptr() != retained[1]["g"].data_ptr()


@pytest.mark.parametrize(
    "option",
    [
        "legacy",
        "explicit_native",
        "cp_true",
        "cp_false",
        "cp_chunk",
        "pool",
        "checkpoint",
        "no_state_return",
        "int32",
        "no_output",
        "no_output_state",
        "normalization",
        "precomputed_gate",
        "precomputed_beta",
    ],
)
def test_unadmitted_contract_keeps_native(routes, monkeypatch, option):
    values = _inputs()
    if option == "legacy":
        values.update(
            use_gate_in_kernel=False, beta_is_logit=False, A_log=None, dt_bias=None
        )
        values["g"] = values["g"].float().sigmoid()
        values["beta"] = values["beta"].float().sigmoid()
    elif option == "explicit_native":
        values["backend"] = "flashinfer"
    elif option.startswith("cp_"):
        values.update(
            {"use_cp": option == "cp_true"}
            if option != "cp_chunk"
            else {"_cp_chunk_len": 64}
        )
    elif option == "pool":
        values["state_indices"] = torch.tensor([0, 1], dtype=torch.int32)
    elif option == "checkpoint":
        values.update(
            checkpoint_every_n_tokens=64,
            state_checkpoints=torch.empty(0, *values["initial_state"].shape[1:]),
            checkpoint_cu_starts=torch.zeros(3, dtype=torch.int64),
        )
    elif option == "no_state_return":
        values["output_final_state"] = False
    elif option == "int32":
        values["cu_seqlens"] = values["cu_seqlens"].int()
    elif option == "no_output":
        values["output"] = None
    elif option == "no_output_state":
        values["output_state"] = None
    elif option == "normalization":
        values["use_qk_l2norm_in_kernel"] = True
        norm = importlib.import_module("flashinfer.gdn_kernels.qk_l2norm")
        monkeypatch.setattr(norm, "normalize_qk", lambda q, k: (q, k))
    elif option == "precomputed_gate":
        values.update(use_gate_in_kernel=False, A_log=None, dt_bias=None)
        values["g"] = values["g"].float().sigmoid()
    elif option == "precomputed_beta":
        values["beta_is_logit"] = False
        values["beta"] = values["beta"].float().sigmoid()
    # This mock route deliberately isolates dispatch from CUDA kernel admission.
    gdn.chunk_gated_delta_rule(**values)
    assert [provider for provider, _ in routes] == ["native"]


@pytest.mark.parametrize(
    "problem",
    [
        "state_alias",
        "dlpack_alias",
        "shifted_dlpack",
        "output_alias",
        "outputs_alias",
        "strided_gate",
        "state_dtype",
        "grad",
        "inference_output",
    ],
)
def test_layout_and_alias_contracts_decline_before_provider(routes, problem):
    values = _inputs()
    if problem == "state_alias":
        values["output_state"] = values["initial_state"]
    elif problem == "dlpack_alias":
        values["output_state"] = torch.from_dlpack(values["initial_state"])
        assert not torch._C._overlaps(values["output_state"], values["initial_state"])
    elif problem == "shifted_dlpack":
        backing = torch.empty(3, *values["initial_state"].shape[1:])
        values["initial_state"] = backing[:-1]
        values["output_state"] = torch.from_dlpack(backing[1:])
        assert not torch._C._overlaps(values["output_state"], values["initial_state"])
    elif problem == "output_alias":
        values["output"] = values["v"]
    elif problem == "outputs_alias":
        values["output"] = (
            values["output_state"]
            .view(torch.bfloat16)
            .flatten()[: values["v"].numel()]
            .view_as(values["v"])
        )
    elif problem == "strided_gate":
        values["g"] = values["g"].t().contiguous().t()
    elif problem == "state_dtype":
        values["initial_state"] = values["initial_state"].bfloat16()
    elif problem == "grad":
        values["output_state"].requires_grad_(True)
    elif problem == "inference_output":
        with torch.inference_mode():
            values["output"] = torch.empty_like(values["output"])
    args = [
        values[name]
        for name in (
            "q",
            "k",
            "v",
            "g",
            "beta",
            "A_log",
            "dt_bias",
            "initial_state",
            "output_state",
            "output",
            "cu_seqlens",
        )
    ]
    assert not gdn._is_cudnn_gdn_auto_eligible(*args)
    assert not routes


@pytest.mark.parametrize(
    "problem", ["gate_shape", "missing_bias", "max_seqlen", "checkpoint"]
)
def test_common_validation_runs_before_auto(routes, problem):
    values = _inputs()
    if problem == "gate_shape":
        values["g"] = values["g"][:, :4]
    elif problem == "missing_bias":
        values["dt_bias"] = None
    elif problem == "max_seqlen":
        values["max_seqlen"] = 0
    elif problem == "checkpoint":
        values["checkpoint_every_n_tokens"] = 1
    with pytest.raises(ValueError):
        gdn.chunk_gated_delta_rule(**values)
    assert not routes


def test_unpreferred_shape_does_not_probe_runtime_or_full_metadata(routes, monkeypatch):
    monkeypatch.setattr(gdn, "_prefer_cudnn_gdn_prefill", lambda *args: False)

    def unexpected(*args):
        raise AssertionError("a policy decline must keep the native hot path short")

    monkeypatch.setattr(gdn, "_is_cudnn_gdn_auto_eligible", unexpected)
    monkeypatch.setattr(gdn, "_cudnn_gdn_prefill_available", unexpected)
    gdn.chunk_gated_delta_rule(**_inputs())
    assert routes[-1][0] == "native"


@pytest.mark.parametrize("frontend", [None, "1.30.0", "1.31.0rc1"])
def test_runtime_probe_rejects_older_or_missing_frontend(monkeypatch, frontend):
    probe = gdn._cudnn_gdn_prefill_available
    monkeypatch.setattr(linear_attention, "CUDNN_AVAILABLE", frontend is not None)
    monkeypatch.setattr(
        linear_attention, "cudnn", SimpleNamespace(__version__=frontend)
    )
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "4.7.0")
    probe.cache_clear()
    try:
        assert not probe()
    finally:
        probe.cache_clear()


def test_unavailable_runtime_keeps_native(routes, monkeypatch):
    monkeypatch.setattr(gdn, "_cudnn_gdn_prefill_available", lambda: False)
    gdn.chunk_gated_delta_rule(**_inputs())
    assert [provider for provider, _ in routes] == ["native"]


@pytest.mark.parametrize("decline", [True, False])
@pytest.mark.parametrize("backend", ["auto", "cudnn"])
def test_real_build_failure_falls_back_only_before_execution(
    routes, monkeypatch, decline, backend
):
    values = _inputs()
    initial = values["initial_state"].clone()
    original = (
        NotImplementedError("unsupported plan")
        if decline
        else TypeError("bad descriptor")
    )

    def build(*args, **kwargs):
        torch.testing.assert_close(kwargs["initial_state"], initial)
        assert torch.all(kwargs["final_state"] == -11)
        assert torch.all(values["output"] == -13)
        raise original

    monkeypatch.setattr(
        cudnn_adapter,
        "cudnn_chunk_gated_delta_rule",
        linear_attention.cudnn_chunk_gated_delta_rule,
    )
    monkeypatch.setattr(linear_attention, "_check_cudnn_frontend", lambda *args: None)
    monkeypatch.setattr(linear_attention, "_build_la_graph", build, raising=False)
    if decline and backend == "auto":
        gdn.chunk_gated_delta_rule(**values, backend=backend)
        assert [provider for provider, _ in routes] == ["native"]
        torch.testing.assert_close(values["output_state"], initial + 1)
        expected_alpha = (
            -values["A_log"].exp()
            * torch.nn.functional.softplus(values["g"].float() + values["dt_bias"])
        ).exp()
        expected_beta = values["beta"].float().sigmoid().bfloat16().float()
        torch.testing.assert_close(routes[-1][1]["g"], expected_alpha)
        torch.testing.assert_close(routes[-1][1]["beta"], expected_beta)
    else:
        with pytest.raises(type(original)) as caught:
            gdn.chunk_gated_delta_rule(**values, backend=backend)
        assert caught.value is original and not routes
        assert torch.all(values["output_state"] == -11)
    torch.testing.assert_close(values["initial_state"], initial)


@pytest.mark.parametrize(
    "error_type", linear_attention._LINEAR_ATTENTION_BUILD_ERRORS + (RuntimeError,)
)
def test_execute_failure_never_retries_native(routes, monkeypatch, error_type):
    values = _inputs()
    error = error_type("failure after state write")

    def execute(*args, **kwargs):
        kwargs["output_state"].add_(1)
        raise error

    monkeypatch.setattr(cudnn_adapter, "cudnn_chunk_gated_delta_rule", execute)
    with pytest.raises(error_type) as caught:
        gdn.chunk_gated_delta_rule(**values)
    assert caught.value is error and not routes
    assert torch.all(values["output_state"] == -10)


@requires_cudnn_linear_attention
@pytest.mark.parametrize("lengths", [(32,), (13, 19)])
def test_auto_raw_gates_fresh_inputs_and_capture(monkeypatch, lengths):
    if not gdn._cudnn_gdn_prefill_available():
        pytest.skip("requires cuDNN frontend 1.31 GA and CuTe DSL 4.7")
    monkeypatch.setattr(gdn, "_prefer_cudnn_gdn_prefill", lambda *args: True)
    actual_cudnn = cudnn_adapter.cudnn_chunk_gated_delta_rule
    launches = []

    def observe(*args, **kwargs):
        result = actual_cudnn(*args, **kwargs)
        launches.append(1)
        return result

    monkeypatch.setattr(cudnn_adapter, "cudnn_chunk_gated_delta_rule", observe)

    def reference(values):
        alpha = (
            -values["A_log"].exp()
            * torch.nn.functional.softplus(values["g"].float() + values["dt_bias"])
        ).exp()
        beta = values["beta"].float().sigmoid().to(values["beta"].dtype).float()
        return serial_delta_rule(
            values["q"],
            values["k"],
            values["v"],
            values["cu_seqlens"],
            alpha=alpha,
            beta=beta,
            initial_state=values["initial_state"],
            scale=128**-0.5,
        )

    retained = []
    for seed in (71, 79):
        values = _inputs("cuda", lengths, 128, seed)
        retained.append(values)
        initial = values["initial_state"].clone()
        expected = reference(values)
        before = len(launches)
        actual = gdn.chunk_gated_delta_rule(**values)
        assert len(launches) == before + 1, "cuDNN must execute, not silently fall back"
        for name, result, target in zip(
            ("output", "state"), actual, expected, strict=True
        ):
            assert_rel_close(name, result, target, 0.03)
        torch.testing.assert_close(values["initial_state"], initial, rtol=0, atol=0)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            gdn.chunk_gated_delta_rule(**values)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = gdn.chunk_gated_delta_rule(**values)
            values["g"].add_(0.125)
            values["beta"].sub_(0.25)
            values["A_log"].add_(0.03)
            values["dt_bias"].mul_(0.8)
            values["v"].add_(0.0625)
            values["initial_state"].add_(0.01)
            expected = reference(values)
            initial = values["initial_state"].clone()
            values["output"].fill_(float("nan"))
            values["output_state"].fill_(float("nan"))
            graph.replay()
            stream.synchronize()
            for name, result, target in zip(
                ("output", "state"), captured, expected, strict=True
            ):
                assert_rel_close(name, result, target, 0.03)
            torch.testing.assert_close(values["initial_state"], initial, rtol=0, atol=0)
        torch.cuda.current_stream().wait_stream(stream)
