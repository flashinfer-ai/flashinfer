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

import pytest
import torch

from flashinfer.gdn_kernels.gates import materialize_gates, validate_gate_inputs
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.trace.templates.gdn import gdn_prefill_trace


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_raw_gates_preserve_beta_storage_rounding(dtype):
    g = torch.tensor([[-20.0, 0.0, 20.0], [0.2, -0.7, 0.4]], dtype=dtype)
    beta = torch.tensor([[0.1, 0.3, 0.7], [-0.1, -0.3, -0.7]], dtype=dtype)
    a_log = torch.tensor([-1.0, 0.0, 1.0])
    bias = torch.tensor([0.3, -0.2, 0.1])
    alpha, actual_beta = materialize_gates(g, beta, "linear", True, a_log, bias, True)
    expected = (
        (
            -a_log.double().exp()
            * torch.nn.functional.softplus(g.double() + bias.double())
        )
        .exp()
        .float()
    )
    torch.testing.assert_close(alpha, expected, atol=1e-7, rtol=2e-6)
    torch.testing.assert_close(
        actual_beta, beta.float().sigmoid().to(dtype).float(), rtol=0, atol=0
    )
    log_alpha, _ = materialize_gates(
        expected.log(), beta, "log", False, None, None, False
    )
    torch.testing.assert_close(log_alpha, expected)


@pytest.mark.parametrize(
    "problem",
    [
        "domain",
        "missing_gate",
        "missing_bias",
        "unused_parameter",
        "beta_none",
        "beta_dtype",
        "gate_shape",
        "parameter_shape",
    ],
)
def test_invalid_gate_metadata(problem):
    q = torch.empty(3, 2, 8)
    v = torch.empty(3, 4, 8)
    g, beta = torch.empty(3, 4), torch.empty(3, 4)
    a_log, bias = torch.empty(4), torch.empty(4)
    domain, raw, logits = "linear", True, True
    if problem == "domain":
        domain = "raw"
    elif problem == "missing_gate":
        g = None
    elif problem == "missing_bias":
        bias = None
    elif problem == "unused_parameter":
        raw = False
    elif problem == "beta_none":
        beta = None
    elif problem == "beta_dtype":
        beta = beta.long()
    elif problem == "gate_shape":
        g = g[:, :2]
    elif problem == "parameter_shape":
        bias = bias[None]
    with pytest.raises(ValueError):
        validate_gate_inputs(q, v, g, beta, domain, raw, a_log, bias, logits)


def test_transformed_trace_preserves_default_definition_and_runs_standalone():
    default = gdn_prefill_trace()
    transformed = gdn_prefill_trace(use_gate_in_kernel=True)
    assert default.name_prefix == "gdn_prefill"
    assert "gate_domain" not in default.inputs
    assert transformed.name_prefix == "gdn_prefill_gates"
    assert gdn_prefill_trace(gate_domain="log") is transformed
    assert gdn_prefill_trace(beta_is_logit=True) is transformed
    args = transformed.init(
        total_seq_len=4,
        num_seqs=2,
        num_q_heads=2,
        num_k_heads=2,
        num_v_heads=4,
        head_size=8,
        device="cpu",
    )
    definition = chunk_gated_delta_rule.fi_trace(**args)
    namespace = {}
    exec(definition["reference"], namespace)
    output, state = namespace["_gdn_prefill_gates_reference"](
        args["q"],
        args["k"],
        args["v"],
        None,
        args["A_log"],
        args["g"],
        args["dt_bias"],
        args["beta"],
        args["cu_seqlens"],
        None,
        use_gate_in_kernel=True,
        beta_is_logit=True,
        output_final_state=True,
    )
    assert output.shape == args["v"].shape
    assert state.shape == (2, 4, 8, 8)
    assert torch.isfinite(output).all() and torch.isfinite(state).all()
    init_namespace = {}
    exec(definition["init"], init_namespace)
    replay = init_namespace["_gdn_prefill_gates_init"](
        total_seq_len=4,
        num_seqs=2,
        num_q_heads=2,
        num_k_heads=2,
        num_v_heads=4,
        head_size=8,
        device="cpu",
    )
    assert replay["use_gate_in_kernel"] and replay["beta_is_logit"]


@pytest.mark.parametrize("heads", [(4, 2, 2), (2, 2, 4)])
@pytest.mark.parametrize("empty_sequence", [False, True])
def test_gate_trace_reference_grouping_and_state_pool(heads, empty_sequence):
    from tests.test_helpers.cudnn_linear_attention import serial_delta_rule

    hq, hk, hv = heads
    template = gdn_prefill_trace(use_gate_in_kernel=True)
    args = template.init(
        total_seq_len=4,
        num_seqs=2,
        state_pool_rows=6,
        num_q_heads=hq,
        num_k_heads=hk,
        num_v_heads=hv,
        head_size=8,
        device="cpu",
    )
    args["initial_state"].normal_(0, 0.01)
    args["output_state"].fill_(17)
    args["state_indices"] = torch.tensor([5, 1], dtype=torch.int32)
    args["cu_seqlens"] = torch.tensor(
        [0, 0 if empty_sequence else 1, 4], dtype=torch.int64
    )
    definition = chunk_gated_delta_rule.fi_trace(**args)
    assert "state_indices" in definition["inputs"]
    assert definition["inputs"]["state"]["shape"][0] == "state_pool_rows"
    namespace = {}
    exec(definition["reference"], namespace)
    actual, pool = namespace["_gdn_prefill_gates_reference"](
        args["q"],
        args["k"],
        args["v"],
        args["initial_state"],
        args["A_log"],
        args["g"],
        args["dt_bias"],
        args["beta"],
        args["cu_seqlens"],
        None,
        use_gate_in_kernel=True,
        beta_is_logit=True,
        output_final_state=True,
        state_indices=args["state_indices"],
        output_state=args["output_state"],
    )
    alpha = (
        -args["A_log"].exp() * torch.nn.functional.softplus(args["g"] + args["dt_bias"])
    ).exp()
    beta = args["beta"].float().sigmoid().to(args["beta"].dtype).float()
    expected, state = serial_delta_rule(
        args["q"],
        args["k"],
        args["v"],
        args["cu_seqlens"],
        alpha=alpha,
        beta=beta,
        initial_state=args["initial_state"][args["state_indices"]],
        scale=8**-0.5,
    )
    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=0, atol=0)
    torch.testing.assert_close(pool[args["state_indices"]], state)
    assert torch.equal(pool[[0, 2, 3, 4]], args["output_state"][[0, 2, 3, 4]])


def test_pool_only_trace_selects_slot_aware_schema_and_reference():
    from tests.test_helpers.cudnn_linear_attention import serial_delta_rule

    indices = torch.tensor([5, 1], dtype=torch.int32)
    template = gdn_prefill_trace(state_indices=indices)
    assert template is not gdn_prefill_trace()
    args = template.init(
        total_seq_len=4,
        num_seqs=2,
        state_pool_rows=6,
        num_q_heads=2,
        num_k_heads=2,
        num_v_heads=4,
        head_size=8,
        device="cpu",
    )
    args.update(
        state_indices=indices,
        use_gate_in_kernel=False,
        beta_is_logit=False,
        A_log=None,
        dt_bias=None,
    )
    args["g"].fill_(0.9)
    args["beta"].fill_(0.5)
    args["initial_state"].normal_(0, 0.01)
    args["output_state"].fill_(17)
    definition = chunk_gated_delta_rule.fi_trace(**args)
    assert "state_indices" in definition["inputs"]
    assert definition["inputs"]["state"]["shape"][0] == "state_pool_rows"
    namespace = {}
    exec(definition["reference"], namespace)
    actual, pool = namespace["_gdn_prefill_gates_reference"](
        args["q"],
        args["k"],
        args["v"],
        args["initial_state"],
        None,
        args["g"],
        None,
        args["beta"],
        args["cu_seqlens"],
        None,
        output_final_state=True,
        state_indices=indices,
        output_state=args["output_state"],
    )
    expected, state = serial_delta_rule(
        args["q"],
        args["k"],
        args["v"],
        args["cu_seqlens"],
        alpha=args["g"],
        beta=args["beta"],
        initial_state=args["initial_state"][indices],
        scale=8**-0.5,
    )
    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=0, atol=0)
    torch.testing.assert_close(pool[indices], state)
    torch.testing.assert_close(pool[[0, 2, 3, 4]], args["output_state"][[0, 2, 3, 4]])
