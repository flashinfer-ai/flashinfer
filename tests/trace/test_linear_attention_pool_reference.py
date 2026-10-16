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

import json
from pathlib import Path

import pytest
import torch

from flashinfer.gdn2_prefill import chunk_gated_delta_rule2
from flashinfer.gdp_prefill import chunk_gated_delta_product
from flashinfer.trace.templates.gdn2 import gdn2_prefill_trace
from flashinfer.trace.templates.gdp import gdp_prefill_trace
from tests.test_helpers.cudnn_linear_attention import (
    serial_delta_product,
    serial_delta_rule2,
)


@pytest.mark.parametrize("definition_source", ["generated", "checked_in"])
def test_gdn2_exported_reference_reads_natural_log_decay(definition_source):
    q = torch.ones(1, 1, 1)
    zero = torch.zeros_like(q)
    g = torch.full_like(q, 0.5).log()
    state = torch.full((1, 1, 1, 1), 2.0)
    cu = torch.tensor([0, 1], dtype=torch.int64)
    if definition_source == "generated":
        definition = chunk_gated_delta_rule2.fi_trace(
            q=q,
            k=zero,
            v=zero,
            g=g,
            beta=zero,
            w=zero,
            initial_state=state,
            cu_seqlens=cu,
        )
    else:
        definition = json.loads(
            (
                Path(__file__).parent / "fi_trace_out/gdn2_prefill_qk4_v8_d128.json"
            ).read_text()
        )
    namespace = {}
    exec(definition["reference"], namespace)
    output, final = namespace["_gdn2_prefill_reference"](
        q, zero, zero, g, zero, zero, state, cu, 1.0
    )
    # No write/erase: a decay of 1/2 maps incoming state 2 to state/output 1.
    torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)
    torch.testing.assert_close(final, torch.ones_like(final), rtol=0, atol=0)


@pytest.mark.parametrize("definition_source", ["generated", "checked_in"])
def test_gdn2_exported_initializer_produces_finite_prefill(definition_source):
    if definition_source == "generated":
        q, v = torch.empty(1, 4, 128), torch.empty(1, 8, 128)
        definition = chunk_gated_delta_rule2.fi_trace(
            q=q, k=q, v=v, g=v, beta=v, w=v, cu_seqlens=torch.tensor([0, 1])
        )
    else:
        definition = json.loads(
            (
                Path(__file__).parent / "fi_trace_out/gdn2_prefill_qk4_v8_d128.json"
            ).read_text()
        )
    namespace = {}
    exec(definition["init"], namespace)
    exec(definition["reference"], namespace)
    args = namespace["_gdn2_prefill_init"](
        total_seq_len=512,
        num_seqs=1,
        num_q_heads=1,
        num_k_heads=1,
        num_v_heads=1,
        head_size=8,
        device="cpu",
    )
    # Positive g means amplifying decay; a long sequence exposes overflow.
    output, final = namespace["_gdn2_prefill_reference"](
        **args, initial_state=None, scale=None
    )
    assert torch.isfinite(output).all()
    assert torch.isfinite(final).all()
    assert (args["g"] <= 0).all()


@pytest.mark.parametrize("heads", [(2, 2), (4, 2), (2, 4)])
@pytest.mark.parametrize(
    "optional", [None, "g", "beta", "initial_state", "output_state"]
)
def test_gdp_pool_trace_resolves_output_heads(heads, optional):
    hq, hv = heads
    ho = max(heads)
    args = dict(
        q=torch.zeros(4, hq, 8),
        k=torch.zeros(8, hq, 8),
        v=torch.zeros(8, hv, 8),
        num_householder=2,
        cu_seqlens=torch.tensor([0, 4], dtype=torch.int64),
        state_indices=torch.tensor([5], dtype=torch.int32),
    )
    if optional in ("g", "beta"):
        args[optional] = torch.ones(4 if optional == "g" else 8, ho)
    elif optional is not None:
        args[optional] = torch.zeros(6, ho, 8, 8)

    template = gdp_prefill_trace(**args)
    # The registry uses the template directly; API tracing uses the dispatcher.
    for trace in (
        template.build_fi_trace_fn("flashinfer.gdp_prefill.chunk_gated_delta_product"),
        chunk_gated_delta_product.fi_trace,
    ):
        definition = trace(**args)
        axes = definition["axes"]
        assert axes["num_o_heads"]["value"] == ho
        assert axes["num_q_heads"]["value"] == hq
        assert axes["num_v_heads"]["value"] == hv
        assert all("value" in axis for axis in axes.values() if axis["type"] == "const")
        assert definition["outputs"]["output"]["shape"][1] == "num_o_heads"


@pytest.mark.parametrize("family", ["gdn2", "gdp"])
@pytest.mark.parametrize("heads", [(4, 2, 2), (2, 2, 4)])
@pytest.mark.parametrize("return_state", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_pool_export_preserves_slots_and_runs_standalone(
    family, heads, return_state, state_dtype
):
    api, factory = (
        (chunk_gated_delta_rule2, gdn2_prefill_trace)
        if family == "gdn2"
        else (chunk_gated_delta_product, gdp_prefill_trace)
    )
    hq, hk, hv = heads
    template = factory(state_indices=True)
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
    args["initial_state"] = args["initial_state"].to(state_dtype).normal_(0, 0.01)
    args["output_state"] = args["output_state"].to(state_dtype).fill_(17)
    args["state_indices"] = torch.tensor([5, 1], dtype=torch.int32)
    args["cu_seqlens"] = torch.tensor([0, 0, 4], dtype=torch.int64)
    args["output_final_state"] = return_state
    definition = api.fi_trace(**args)
    assert definition["inputs"]["initial_state"]["shape"][0] == "state_pool_rows"
    assert "state_indices" in definition["inputs"]
    namespace = {}
    exec(definition["reference"], namespace)
    actual, final = namespace[f"_{family}_prefill_pool_reference"](**args, scale=None)
    selected = args["initial_state"].index_select(0, args["state_indices"])
    common = dict(beta=args["beta"], initial_state=selected, scale=8**-0.5)
    if family == "gdn2":
        expected, state = serial_delta_rule2(
            args["q"],
            args["k"],
            args["v"],
            args["cu_seqlens"],
            alpha=args["g"].exp(),
            w=args["w"],
            **common,
        )
    else:
        expected, state = serial_delta_product(
            args["q"],
            args["k"],
            args["v"],
            args["cu_seqlens"],
            alpha=args["g"],
            num_householder=args["num_householder"],
            **common,
        )
    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=0, atol=0)
    if return_state:
        torch.testing.assert_close(
            final[args["state_indices"]], state.to(state_dtype), rtol=0, atol=0
        )
        assert torch.equal(final[[0, 2, 3, 4]], args["output_state"][[0, 2, 3, 4]])
    else:
        assert final is None
    # The serialized initializer must also run without importing FlashInfer.
    init_namespace = {}
    exec(definition["init"], init_namespace)
    initialized = init_namespace[f"_{family}_prefill_pool_init"](
        total_seq_len=4, num_seqs=2, state_pool_rows=6, head_size=8, device="cpu"
    )
    assert initialized["initial_state"].shape[0] == 6
    assert initialized["state_indices"].shape == (2,)
    assert "state_indices" not in factory().inputs
