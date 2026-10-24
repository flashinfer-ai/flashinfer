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
from contextlib import nullcontext
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


@pytest.fixture
def native_auto_policy():
    """This module exercises the auto resolution itself, so the native-only
    pin of the directory's conftest does not apply here."""
    yield


cudnn_adapter = importlib.import_module("flashinfer.cudnn")


@pytest.fixture(autouse=True)
def clear_auto_declines():
    linear_attention._la_auto_declines.clear()
    linear_attention._la_auto_decline_scopes.clear()
    yield
    linear_attention._la_auto_declines.clear()
    linear_attention._la_auto_decline_scopes.clear()


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
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
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
        output = (
            kwargs["output"] if kwargs["output"] is not None else torch.empty_like(v)
        )
        output.copy_(v)
        if not kwargs["output_final_state"]:
            return output
        initial = kwargs["initial_state"]
        state = (
            kwargs["output_state"]
            if kwargs["output_state"] is not None
            else torch.empty(
                kwargs["cu_seqlens"].numel() - 1,
                v.shape[1],
                v.shape[2],
                v.shape[2],
                dtype=torch.float32 if initial is None else initial.dtype,
            )
        )
        state.copy_((0 if initial is None else initial) + 1)
        return output, state

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
        "explicit_native",
        "cp_true",
        "cp_false",
        "cp_chunk",
        "pool_int64_slots",
        "pool_overlapping_output",
        "checkpoint",
    ],
)
def test_unadmitted_contract_keeps_native(routes, monkeypatch, option):
    values = _inputs()
    if option == "explicit_native":
        values["backend"] = "flashinfer"
    elif option.startswith("cp_"):
        values.update(
            {"use_cp": option == "cp_true"}
            if option != "cp_chunk"
            else {"_cp_chunk_len": 64}
        )
    elif option.startswith("pool_"):
        storage = torch.randn(7, 8, 128, 128) * 0.05
        values["initial_state"] = storage[:6]
        values["output_state"] = torch.full((6, 8, 128, 128), -11.0)
        values["state_indices"] = torch.tensor([4, 1], dtype=torch.int32)
        if option == "pool_int64_slots":
            values["state_indices"] = values["state_indices"].long()
        else:
            values["output_state"] = storage[1:7]
    elif option == "checkpoint":
        values.update(
            checkpoint_every_n_tokens=64,
            state_checkpoints=torch.empty(0, *values["initial_state"].shape[1:]),
            checkpoint_cu_starts=torch.zeros(3, dtype=torch.int64),
        )
    # This mock route deliberately isolates dispatch from CUDA kernel admission.
    gdn.chunk_gated_delta_rule(**values)
    assert [provider for provider, _ in routes] == ["native"]


@pytest.mark.parametrize(
    "option",
    [
        "legacy",
        "no_state_return",
        "int32",
        "no_output",
        "no_output_state",
        "no_initial_state",
        "bf16_state",
        "fp16_state",
        "fp16_inputs",
        "precomputed_gate",
        "precomputed_beta",
        "precomputed_log_gate",
        "absent_gates",
    ],
)
def test_admitted_contract_variants_take_cudnn(routes, monkeypatch, option):
    """Every contract cuDNN's GDN entry serves is substituted, gates untouched.

    Precomputed gates cross as they are (cuDNN reads ``gate_domain``), raw
    gates are fused by cuDNN; neither is materialized on the host first.
    """
    gates = importlib.import_module("flashinfer.gdn_kernels.gates")

    def materialization_is_not_needed(*args):
        raise AssertionError("cuDNN must receive the caller's gates")

    monkeypatch.setattr(gates, "materialize_gates", materialization_is_not_needed)
    values = _inputs()
    if option == "legacy":
        values.update(
            use_gate_in_kernel=False, beta_is_logit=False, A_log=None, dt_bias=None
        )
        values["g"] = values["g"].float().sigmoid()
        values["beta"] = values["beta"].float().sigmoid()
    elif option == "no_state_return":
        values["output_final_state"] = False
    elif option == "int32":
        values["cu_seqlens"] = values["cu_seqlens"].int()
    elif option == "no_output":
        values["output"] = None
    elif option == "no_output_state":
        values["output_state"] = None
    elif option == "no_initial_state":
        values["initial_state"] = None
    elif option == "bf16_state":
        values["initial_state"] = values["initial_state"].bfloat16()
        values["output_state"] = values["output_state"].bfloat16()
    elif option == "fp16_state":
        values["initial_state"] = values["initial_state"].half()
        values["output_state"] = values["output_state"].half()
    elif option == "fp16_inputs":
        for name in ("q", "k", "v", "g", "beta", "output"):
            values[name] = values[name].half()
    elif option == "precomputed_gate":
        values.update(use_gate_in_kernel=False, A_log=None, dt_bias=None)
        values["g"] = values["g"].float().sigmoid()
    elif option == "precomputed_beta":
        values["beta_is_logit"] = False
        values["beta"] = values["beta"].float().sigmoid()
    elif option == "precomputed_log_gate":
        values.update(
            use_gate_in_kernel=False, A_log=None, dt_bias=None, gate_domain="log"
        )
        values["g"] = -values["g"].float().abs()
    elif option == "absent_gates":
        values.update(
            use_gate_in_kernel=False,
            beta_is_logit=False,
            A_log=None,
            dt_bias=None,
            g=None,
            beta=None,
        )
    result = gdn.chunk_gated_delta_rule(**values)
    provider, call = routes[-1]
    assert provider == "cudnn"
    for name in ("q", "k", "v", "g", "beta", "initial_state", "A_log", "dt_bias"):
        assert call[name] is values[name]
    assert call["gate_domain"] == values.get("gate_domain", "linear")
    output = result[0] if isinstance(result, tuple) else result
    torch.testing.assert_close(output, values["v"])
    if values["output_final_state"]:
        state = result[1]
        initial = values["initial_state"]
        expected = (
            torch.ones_like(state) if initial is None else (initial + 1).to(state.dtype)
        )
        torch.testing.assert_close(state, expected)


@pytest.mark.parametrize("in_place", [False, True])
def test_state_pools_take_cudnn(routes, in_place):
    """A contiguous pool with contiguous int32 slots and its output pool,
    identical or disjoint, follows the same policy as the compact call and
    crosses as given (measured 2026-10-15: the cuDNN pool path costs what its
    dense path costs while the native pool route runs the CuTe DSL kernel)."""
    values = _inputs()
    pool = torch.randn(6, 8, 128, 128) * 0.05
    values["initial_state"] = pool
    values["output_state"] = pool if in_place else torch.full_like(pool, -11.0)
    values["state_indices"] = torch.tensor([4, 1], dtype=torch.int32)
    before = pool.clone()
    output, state = gdn.chunk_gated_delta_rule(**values)
    provider, call = routes[-1]
    assert provider == "cudnn"
    assert call["state_indices"] is values["state_indices"]
    assert call["initial_state"] is pool
    assert call["output_state"] is values["output_state"]
    assert output is values["output"] and state is values["output_state"]
    torch.testing.assert_close(output, values["v"])
    torch.testing.assert_close(state, before + 1)
    if not in_place:
        torch.testing.assert_close(pool, before)


def test_in_kernel_normalization_is_admitted_and_forwarded(routes, monkeypatch):
    """``use_qk_l2norm_in_kernel=True`` reaches cuDNN as its fused norm.

    cuDNN normalizes with the same additive ``1e-6`` formula as the native
    pre-pass, so the flag is forwarded and the pre-pass must not run first:
    the raw q / k are what cuDNN receives.
    """
    norm = importlib.import_module("flashinfer.gdn_kernels.qk_l2norm")

    def pre_pass_must_not_run(q, k):
        raise AssertionError("cuDNN fuses the normalization; no native pre-pass")

    monkeypatch.setattr(norm, "normalize_qk", pre_pass_must_not_run)
    values = _inputs()
    values["use_qk_l2norm_in_kernel"] = True
    gdn.chunk_gated_delta_rule(**values)
    provider, call = routes[-1]
    assert provider == "cudnn"
    assert call["use_qk_l2norm_in_kernel"] is True
    assert call["q"] is values["q"] and call["k"] is values["k"]


@pytest.mark.parametrize(
    "problem",
    [
        "state_alias",
        "dlpack_alias",
        "shifted_dlpack",
        "output_alias",
        "outputs_alias",
        "strided_gate",
        "mixed_state_dtype",
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
    elif problem == "mixed_state_dtype":
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


@pytest.mark.parametrize(
    "value_heads,lengths,raw_gates,expected",
    [
        (8, (1024,), False, True),
        (8, (4096,), False, True),
        (8, (8192,), False, True),
        (8, (16384,), False, True),
        (8, (2048,) * 8, False, True),
        (8, (1024,) * 8, False, True),
        (16, (8192,), False, True),
        (16, (6144, 2048), False, True),
        (16, (7168, 512, 512), False, True),
        (16, (1024,) * 8, False, True),
        (32, (4096,), False, True),
        (32, (8192,), False, True),
        (32, (6144, 2048), False, True),
        (32, (4096, 2048, 2048), False, True),
        (64, (1024,), False, True),
        (64, (6144, 2048), False, True),
        (8, (1024,), True, True),
        (16, (8192,), True, True),
    ],
)
def test_gdn_policy_region(monkeypatch, value_heads, lengths, raw_gates, expected):
    """cuDNN for every call the engine serves on SM100 / SM103 (the 2026-10-14
    sweep: faster or within 2 percent on 203 of 232 cells, 1 to 15 us behind
    on fixed-cost cells of 2k to 8k tokens); other devices keep the native
    kernels. The shapes are the 2026-10-13 region's boundary cases."""
    monkeypatch.setattr(gdn, "get_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    total = sum(lengths)
    q = torch.empty(total, 4, 128, dtype=torch.bfloat16)
    v = torch.empty(total, value_heads, 128, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, *lengths]).cumsum(0)
    assert gdn._prefer_cudnn_gdn_prefill(q, v, cu_seqlens, raw_gates) is expected
    monkeypatch.setattr(gdn, "get_compute_capability", lambda device: (12, 0))
    assert gdn._prefer_cudnn_gdn_prefill(q, v, cu_seqlens, raw_gates) is False


@pytest.mark.parametrize(
    "dims,expected",
    [
        ((64, 64), True),
        ((64, 128), True),
        ((128, 64), True),
        ((128, 128), True),
        ((256, 128), False),
        ((128, 96), False),
    ],
)
def test_gdn_policy_head_dims(monkeypatch, dims, expected):
    """cuDNN's engine serves K and V head dims of 64 or 128 in any pairing and
    the native kernels 128 x 128 only, so every served pairing is cuDNN's on
    SM100 and widths the engine does not serve stay native."""
    monkeypatch.setattr(gdn, "get_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    q = torch.empty(4096, 4, dims[0], dtype=torch.bfloat16)
    v = torch.empty(4096, 8, dims[1], dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 4096])
    assert gdn._prefer_cudnn_gdn_prefill(q, v, cu_seqlens, False) is expected
    monkeypatch.setattr(gdn, "get_compute_capability", lambda device: (12, 0))
    assert gdn._prefer_cudnn_gdn_prefill(q, v, cu_seqlens, False) is False


@pytest.mark.parametrize("dims", [(64, 64), (64, 128), (128, 64)])
def test_eligibility_admits_64_wide_heads(monkeypatch, dims):
    """The admission check sizes v, the states and the output by V and K
    separately: ``[tokens, heads, V]`` and ``[num_seqs, heads, V, K]``."""
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    values = _inputs(dim=dims[0])
    total, num_seqs = values["q"].shape[0], values["cu_seqlens"].numel() - 1
    values["v"] = torch.randn(total, 8, dims[1]).bfloat16()
    values["initial_state"] = torch.randn(num_seqs, 8, dims[1], dims[0]) * 0.05
    values["output_state"] = torch.empty(num_seqs, 8, dims[1], dims[0])
    values["output"] = torch.empty(total, 8, dims[1], dtype=torch.bfloat16)
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
    assert gdn._is_cudnn_gdn_auto_eligible(*args)


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


def test_repeated_build_declines_keep_fresh_fallback_and_explicit_error(
    routes, monkeypatch
):
    monkeypatch.setattr(
        cudnn_adapter,
        "cudnn_chunk_gated_delta_rule",
        linear_attention.cudnn_chunk_gated_delta_rule,
    )
    monkeypatch.setattr(linear_attention, "_check_cudnn_frontend", lambda *args: None)
    monkeypatch.setattr(linear_attention, "get_device_index", lambda device: 0)
    error = NotImplementedError("unsupported raw gates")
    builds = []

    def decline(*args, **kwargs):
        builds.append(1)
        raise error

    monkeypatch.setattr(linear_attention, "_create_la_graph", decline)
    for seed in range(5):
        values = _inputs(seed=seed)
        out, final = gdn.chunk_gated_delta_rule(**values)
        torch.testing.assert_close(out, values["v"])
        torch.testing.assert_close(final, values["initial_state"] + 1)
    assert builds == [1]
    assert [provider for provider, _ in routes] == ["native"] * 5
    with pytest.raises(NotImplementedError) as caught:
        gdn.chunk_gated_delta_rule(**_inputs(), backend="cudnn")
    assert caught.value is error and builds == [1, 1]


def test_tensor_scale_values_have_distinct_support_decisions(routes, monkeypatch):
    calls = []

    def decline(q, k, v, g, beta, scale, **kwargs):
        calls.append(scale)
        error = NotImplementedError("unsupported scale")
        error._fi_la_build_unsupported = True
        raise error

    monkeypatch.setattr(cudnn_adapter, "cudnn_chunk_gated_delta_rule", decline)
    for value in (1.0, 2.0, 1.0):
        gdn.chunk_gated_delta_rule(**_inputs(), scale=torch.tensor(value))
    assert calls == [1.0, 2.0]
    assert all(isinstance(value, float) for value in calls)
    assert len(routes) == 3


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
