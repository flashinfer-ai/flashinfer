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
