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

"""Auto substitutes a provider without changing native prefill semantics.

The performance policy is forced on here; tests cover capabilities and mutation
boundaries rather than freeze the thresholds that future tuning may change.
"""

import importlib
import importlib.metadata
from types import SimpleNamespace

import pytest
import torch

from flashinfer.cudnn import linear_attention
from flashinfer.kda_prefill import RecurrentKDAPrefillWorkspace
from flashinfer.utils import get_compute_capability
from tests.test_helpers.cudnn_linear_attention import assert_rel_close
from tests.test_helpers.kda_prefill import packed_prefill_inputs

kda = importlib.import_module("flashinfer.kda")
cudnn_adapter = importlib.import_module("flashinfer.cudnn")


@pytest.fixture(autouse=True)
def clear_auto_declines():
    linear_attention._la_auto_declines.clear()
    linear_attention._la_auto_decline_scopes.clear()
    yield
    linear_attention._la_auto_declines.clear()
    linear_attention._la_auto_decline_scopes.clear()


def _inputs(device="cpu", state_dtype=torch.bfloat16, seed=31):
    values = packed_prefill_inputs(device, seq_lens=[64], num_heads=4, seed=seed)
    values["initial_state"] = torch.zeros(
        (1, 4, 128, 128), device=device, dtype=state_dtype
    )
    values["output"] = torch.empty_like(values["q"])
    values["output_final_state"] = True
    return values


def _force_native_route(monkeypatch, native, seen):
    monkeypatch.setattr(kda, "_cudnn_kda_prefill_available", lambda: True)
    monkeypatch.setattr(kda, "_prefer_cudnn_kda_prefill", lambda *args: True)
    monkeypatch.setattr(kda, "is_cute_dsl_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=10000),
    )
    monkeypatch.setattr(
        kda._kda_prefill_cute_small_bh,
        "_is_kda_prefill_cute_small_bh_eligible",
        lambda **kwargs: native == "small-bh",
    )
    monkeypatch.setattr(
        kda._kda_prefill, "_sm120_kda_prefill_is_eligible", lambda **kwargs: False
    )
    monkeypatch.setattr(
        kda._kda_prefill_cute,
        "_is_cute_dsl_kda_prefill_eligible",
        lambda **kwargs: native == "cute-dsl",
    )

    def run(**kwargs):
        seen.append(("native", kwargs))
        kwargs["output"].copy_(kwargs["q"] + kwargs["v"])
        kwargs["initial_state"].add_(1)
        return (
            kwargs["output"],
            kwargs["initial_state"] if kwargs["output_final_state"] else None,
        )

    monkeypatch.setattr(
        kda._kda_prefill_cute_small_bh, "_run_kda_prefill_cute_small_bh", run
    )
    monkeypatch.setattr(kda._kda_prefill_cute, "_run_cute_dsl_kda_prefill", run)

    def cudnn(q, k, v, g, beta, **kwargs):
        seen.append(("cudnn", dict(q=q, k=k, v=v, g=g, beta=beta, **kwargs)))
        kwargs["output"].copy_(q + v)
        kwargs["initial_state"].add_(1)
        return (
            kwargs["output"],
            kwargs["initial_state"] if kwargs["output_final_state"] else None,
        )

    monkeypatch.setattr(cudnn_adapter, "cudnn_recurrent_kda", cudnn)


@pytest.mark.parametrize(
    "native,state_dtype",
    [("small-bh", torch.bfloat16), ("cute-dsl", torch.float32)],
)
@pytest.mark.parametrize("return_state", [False, True])
def test_auto_preserves_math_dtype_and_fresh_bindings(
    monkeypatch, native, state_dtype, return_state
):
    seen = []
    _force_native_route(monkeypatch, native, seen)
    retained = []
    for seed in (19, 23):
        values = _inputs(state_dtype=state_dtype, seed=seed)
        values["output_final_state"] = return_state
        result = kda.recurrent_kda(**values, backend="auto")
        provider, bound = seen[-1]
        assert provider == "cudnn"
        assert bound["use_qk_l2norm_in_kernel"] is True
        assert "qk_l2norm_additive_epsilon" not in bound
        for name in ("q", "k", "v", "g", "beta", "initial_state", "output"):
            assert bound[name] is values[name]
        assert bound["initial_state"].dtype == state_dtype
        assert result[0] is values["output"]
        assert result[1] is (values["initial_state"] if return_state else None)
        torch.testing.assert_close(result[0], values["q"] + values["v"])
        torch.testing.assert_close(
            values["initial_state"], torch.ones_like(values["initial_state"])
        )
        retained.append(values)
    assert len(seen) == 2


@pytest.mark.parametrize(
    "excluded",
    [
        "prefill_workspace",
        "ssm_state_indices",
        "initial_state_source",
        "initial_state_indices",
        "num_accepted_tokens",
        "state_checkpoints",
        "checkpoint_cu_starts",
        "checkpoint_state_indices",
        "checkpoint_every_n_tokens",
        "strided_qk",
        "strided_state",
        "gate_capacity",
        "beta_capacity",
        "unpacked",
    ],
)
def test_auto_preserves_unadmitted_native_contracts(monkeypatch, excluded):
    seen = []
    _force_native_route(monkeypatch, "cute-dsl", seen)
    values = _inputs()
    if excluded == "prefill_workspace":
        # Only the dispatch type check runs here; the real constructor allocates
        # CUDA storage, whereas this host test substitutes the native launcher.
        values[excluded] = object.__new__(RecurrentKDAPrefillWorkspace)
    elif excluded == "checkpoint_every_n_tokens":
        values[excluded] = 32
    elif excluded == "strided_qk":
        for name in ("q", "k"):
            carrier = torch.empty(*values[name].shape, 2, dtype=values[name].dtype)
            view = carrier[..., 0]
            view.copy_(values[name])
            values[name] = view
    elif excluded == "strided_state":
        # Two sequences are needed to make a padded row pitch noncompact.
        values["cu_seqlens"] = torch.tensor([0, 32, 64], dtype=torch.int64)
        values["initial_state"] = torch.zeros(4, 4, 128, 128, dtype=torch.bfloat16)[::2]
    elif excluded in ("gate_capacity", "beta_capacity"):
        name = "g" if excluded == "gate_capacity" else "beta"
        values[name] = torch.cat((values[name], values[name]), dim=1)
    elif excluded == "unpacked":
        values["cu_seqlens"] = None
    else:
        values[excluded] = torch.tensor([0], dtype=torch.int32)
    kda.recurrent_kda(**values, backend="auto")
    assert [provider for provider, _ in seen] == ["native"]


@pytest.mark.parametrize("order", ["valid", "dtype", "shape", "device"])
def test_seq_order_uses_native_metadata_validation(monkeypatch, order):
    small_bh_eligible = (
        kda._kda_prefill_cute_small_bh._is_kda_prefill_cute_small_bh_eligible
    )
    cute_eligible = kda._kda_prefill_cute._is_cute_dsl_kda_prefill_eligible
    seen = []
    _force_native_route(monkeypatch, "small-bh", seen)
    # Exercise the actual metadata checks without executing or allocating CUDA.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    for module in (kda._kda_prefill_cute_small_bh, kda._kda_prefill_cute):
        monkeypatch.setattr(module, "get_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(
        kda._kda_prefill_cute_small_bh,
        "_is_kda_prefill_cute_small_bh_eligible",
        small_bh_eligible,
    )
    monkeypatch.setattr(
        kda._kda_prefill_cute, "_is_cute_dsl_kda_prefill_eligible", cute_eligible
    )
    monkeypatch.setattr(
        kda._kda_prefill_cute, "_is_cute_dsl_kda_runtime_available", lambda: True
    )
    monkeypatch.setattr(
        kda._kda_prefill, "_flash_kda_prefill_is_eligible", lambda **kwargs: False
    )
    values = _inputs()
    values["seq_order"] = torch.zeros(
        2 if order == "shape" else 1,
        dtype=torch.int64 if order == "dtype" else torch.int32,
        device="meta" if order == "device" else "cpu",
    )
    if order == "valid":
        kda.recurrent_kda(**values, backend="auto")
        assert [provider for provider, _ in seen] == ["cudnn"]
    else:
        with pytest.raises(ValueError, match="seq_order"):
            kda.recurrent_kda(**values, backend="auto")
        assert not seen


@pytest.mark.parametrize("alias", ["q", "k", "v", "g", "initial_state"])
def test_auto_rejects_output_alias_before_provider_launch(monkeypatch, alias):
    seen = []
    _force_native_route(monkeypatch, "cute-dsl", seen)
    values = _inputs()
    values["output"] = (
        values[alias].view(-1)[: values["q"].numel()].view(values["q"].shape)
    )
    with pytest.raises(ValueError, match="output must not overlap"):
        kda.recurrent_kda(**values, backend="auto")
    assert not seen


def test_unsupported_build_falls_back_before_mutation(monkeypatch):
    seen = []
    _force_native_route(monkeypatch, "small-bh", seen)
    values = _inputs()

    def decline(*args, **kwargs):
        assert torch.count_nonzero(kwargs["initial_state"]) == 0
        error = NotImplementedError("no supported plan")
        error.__dict__["_fi_la_build_unsupported"] = True
        raise error

    monkeypatch.setattr(cudnn_adapter, "cudnn_recurrent_kda", decline)
    kda.recurrent_kda(**values, backend="auto")
    assert [provider for provider, _ in seen] == ["native"]
    torch.testing.assert_close(
        values["initial_state"], torch.ones_like(values["initial_state"])
    )


@pytest.mark.parametrize("error_type", [NotImplementedError, "graph_not_supported"])
def test_repeated_unsupported_build_keeps_fallback_and_explicit_diagnostics(
    monkeypatch, error_type
):
    seen = []
    _force_native_route(monkeypatch, "small-bh", seen)
    monkeypatch.setattr(
        cudnn_adapter, "cudnn_recurrent_kda", linear_attention.cudnn_recurrent_kda
    )
    monkeypatch.setattr(linear_attention, "_check_cudnn_frontend", lambda *args: None)
    monkeypatch.setattr(linear_attention, "get_device_index", lambda device: 0)
    monkeypatch.setattr(linear_attention.cudnn, "__version__", "1.31.0")
    if error_type == "graph_not_supported":
        error_type = linear_attention.cudnn.cudnnGraphNotSupportedError
    error = error_type("no supported plan")
    builds = []

    def decline(*args, **kwargs):
        builds.append(kwargs.get("overwrite_initial_state", False))
        raise error

    monkeypatch.setattr(linear_attention, "_create_la_graph", decline)
    for seed in range(5):
        # New storage/values reuse the support decision, never live bindings.
        values = _inputs(seed=seed)
        out, state = kda.recurrent_kda(**values, backend="auto")
        torch.testing.assert_close(out, values["q"] + values["v"])
        torch.testing.assert_close(state, torch.ones_like(state))
    assert len(builds) == 2  # One overwrite + compatibility attempt total.
    assert [provider for provider, _ in seen] == ["native"] * 5
    with pytest.raises(error_type) as caught:
        kda.recurrent_kda(**_inputs(), backend="cudnn")
    assert caught.value is error
    assert len(builds) == 4  # Explicit calls do not consume auto declines.


def test_cached_decline_does_not_skip_output_alias_validation(monkeypatch):
    seen = []
    _force_native_route(monkeypatch, "small-bh", seen)

    def decline(*args, **kwargs):
        error = NotImplementedError("no supported plan")
        error._fi_la_build_unsupported = True
        raise error

    monkeypatch.setattr(cudnn_adapter, "cudnn_recurrent_kda", decline)
    values = _inputs()
    kda.recurrent_kda(**values, backend="auto")
    values["output"] = values["q"]
    with pytest.raises(ValueError, match="output must not overlap"):
        kda.recurrent_kda(**values, backend="auto")
    assert len(seen) == 1


@pytest.mark.parametrize("option", ["scale", "lower_bound"])
def test_tensor_scalar_values_do_not_share_decline(monkeypatch, option):
    seen = []
    _force_native_route(monkeypatch, "small-bh", seen)
    calls = []

    def decline(*args, **kwargs):
        calls.append(kwargs[option])
        error = NotImplementedError("unsupported scalar option")
        error._fi_la_build_unsupported = True
        raise error

    monkeypatch.setattr(cudnn_adapter, "cudnn_recurrent_kda", decline)
    for value in (-1.0, -2.0, -1.0):
        values = _inputs()
        values[option] = torch.tensor(value)
        kda.recurrent_kda(**values, backend="auto")
    assert calls == [-1.0, -2.0]
    assert all(isinstance(value, float) for value in calls)
    assert len(seen) == 3


def test_build_typeerror_propagates_without_mutating_state(monkeypatch):
    seen = []
    _force_native_route(monkeypatch, "small-bh", seen)
    values = _inputs()
    monkeypatch.setattr(
        cudnn_adapter, "cudnn_recurrent_kda", linear_attention.cudnn_recurrent_kda
    )
    monkeypatch.setattr(linear_attention, "_check_cudnn_frontend", lambda *args: None)

    def build(*args, **kwargs):
        assert "qk_l2norm_additive_epsilon" not in kwargs
        raise TypeError("invalid tensor descriptor")

    monkeypatch.setattr(linear_attention, "_build_la_graph", build)
    with pytest.raises(TypeError, match="invalid tensor descriptor"):
        kda.recurrent_kda(**values, backend="auto")
    assert not seen
    torch.testing.assert_close(
        values["initial_state"], torch.zeros_like(values["initial_state"])
    )


@pytest.mark.parametrize(
    "error", linear_attention._LINEAR_ATTENTION_BUILD_ERRORS + (RuntimeError,)
)
def test_execution_errors_do_not_retry_after_state_mutation(monkeypatch, error):
    seen = []
    _force_native_route(monkeypatch, "small-bh", seen)
    values = _inputs()

    def execute(*args, **kwargs):
        kwargs["initial_state"].add_(1)
        raise error("failure after launch")

    monkeypatch.setattr(cudnn_adapter, "cudnn_recurrent_kda", execute)
    with pytest.raises(error, match="failure after launch"):
        kda.recurrent_kda(**values, backend="auto")
    assert not seen
    torch.testing.assert_close(
        values["initial_state"], torch.ones_like(values["initial_state"])
    )


def test_declined_performance_policy_does_not_probe_optional_runtime(monkeypatch):
    seen = []
    _force_native_route(monkeypatch, "small-bh", seen)
    monkeypatch.setattr(kda, "_prefer_cudnn_kda_prefill", lambda *args: False)

    def unexpected_probe():
        raise AssertionError("unadmitted shapes must not probe the cuDNN runtime")

    monkeypatch.setattr(kda, "_cudnn_kda_prefill_available", unexpected_probe)
    kda.recurrent_kda(**_inputs(), backend="auto")
    assert [provider for provider, _ in seen] == ["native"]


@pytest.mark.parametrize(
    "available,fe,dsl,expected",
    [
        (False, "1.31.0", "4.7.0", False),
        (True, "1.30.0", "4.8.0", False),
        (True, "1.31.0rc1", "4.7.0", False),
        (True, "1.31.0", "4.6.2", False),
        (True, "1.31.0", "4.7.0", True),
    ],
)
def test_optional_runtime_floor(monkeypatch, available, fe, dsl, expected):
    monkeypatch.setattr(linear_attention, "CUDNN_AVAILABLE", available)
    monkeypatch.setattr(linear_attention, "cudnn", SimpleNamespace(__version__=fe))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: dsl)
    kda._cudnn_kda_prefill_available.cache_clear()
    try:
        assert kda._cudnn_kda_prefill_available() is expected
    finally:
        kda._cudnn_kda_prefill_available.cache_clear()


@pytest.mark.parametrize("frontend", [None, "1.30.0", "1.31.0rc1"])
def test_missing_or_older_frontend_keeps_native_auto(monkeypatch, frontend):
    probe = kda._cudnn_kda_prefill_available
    seen = []
    _force_native_route(monkeypatch, "small-bh", seen)
    monkeypatch.setattr(kda, "_cudnn_kda_prefill_available", probe)
    monkeypatch.setattr(linear_attention, "CUDNN_AVAILABLE", frontend is not None)
    monkeypatch.setattr(
        linear_attention, "cudnn", SimpleNamespace(__version__=frontend)
    )
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "4.7.0")
    probe.cache_clear()
    try:
        kda.recurrent_kda(**_inputs(), backend="auto")
        assert [provider for provider, _ in seen] == ["native"]
    finally:
        probe.cache_clear()


@pytest.mark.parametrize("requires_grad,inference", [(True, False), (False, True)])
def test_performance_policy_excludes_state_copy_compatibility_path(
    monkeypatch, requires_grad, inference
):
    q = SimpleNamespace(
        is_cuda=True, dtype=torch.bfloat16, shape=(1, 8192, 4, 128), device="cuda:0"
    )
    state = SimpleNamespace(
        dtype=torch.bfloat16,
        requires_grad=requires_grad,
        is_inference=lambda: inference,
    )
    offsets = SimpleNamespace(numel=lambda: 2)
    monkeypatch.setattr(kda, "get_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(torch, "is_inference_mode_enabled", lambda: False)
    assert not kda._prefer_cudnn_kda_prefill(q, state, offsets)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_auto_candidate_matches_native_with_fresh_pointers_and_replay(
    monkeypatch, state_dtype
):
    device = torch.device("cuda")
    if get_compute_capability(device) not in ((10, 0), (10, 3)):
        pytest.skip("auto candidate is scoped to SM100/SM103")
    if not kda._cudnn_kda_prefill_available():
        pytest.skip("requires cuDNN frontend 1.31 and CuTe DSL 4.7")
    retained = []
    native_cudnn = cudnn_adapter.cudnn_recurrent_kda
    seen = []

    def observe(*args, **kwargs):
        result = native_cudnn(*args, **kwargs)
        assert "qk_l2norm_additive_epsilon" not in kwargs
        seen.append(kwargs["use_qk_l2norm_in_kernel"])
        return result

    monkeypatch.setattr(cudnn_adapter, "cudnn_recurrent_kda", observe)
    for seed in (71, 79):
        values = _inputs(device, state_dtype, seed)
        values["seq_order"] = torch.zeros(1, dtype=torch.int32, device=device)
        for name in ("q", "k"):
            values[name][:, 0::4] = 0
            values[name][:, 1::4] *= 1e-5
        seed_state = values["initial_state"].clone()
        reference_state = seed_state.clone()
        reference_output = torch.empty_like(values["output"])
        reference = dict(values, initial_state=reference_state, output=reference_output)
        monkeypatch.setattr(kda, "_prefer_cudnn_kda_prefill", lambda *args: False)
        expected = kda.recurrent_kda(**reference, backend="auto")
        monkeypatch.setattr(kda, "_prefer_cudnn_kda_prefill", lambda *args: True)
        previous_launches = len(seen)
        actual = kda.recurrent_kda(**values, backend="auto")
        assert len(seen) == previous_launches + 1, (
            "cuDNN must execute with seq_order; a native fallback is not candidate coverage"
        )
        for name, result, target in zip(
            ("output", "state"), actual, expected, strict=True
        ):
            assert_rel_close(name, result, target, 0.03)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            values["initial_state"].copy_(seed_state)
            kda.recurrent_kda(**values, backend="auto")
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                values["initial_state"].copy_(seed_state)
                actual = kda.recurrent_kda(**values, backend="auto")
            values["q"][:, 1::4].add_(1e-5)
            values["v"].add_(0.0625)
            reference_state.copy_(seed_state)
            monkeypatch.setattr(kda, "_prefer_cudnn_kda_prefill", lambda *args: False)
            expected = kda.recurrent_kda(**reference, backend="auto")
            monkeypatch.setattr(kda, "_prefer_cudnn_kda_prefill", lambda *args: True)
            values["output"].fill_(float("nan"))
            graph.replay()
            stream.synchronize()
            for name, result, target in zip(
                ("output", "state"), actual, expected, strict=True
            ):
                assert_rel_close(name, result, target, 0.03)
        torch.cuda.current_stream().wait_stream(stream)
        # Keep old bindings alive so the next call cannot reuse their pointers.
        retained.append((values, reference, seed_state, graph, stream))
