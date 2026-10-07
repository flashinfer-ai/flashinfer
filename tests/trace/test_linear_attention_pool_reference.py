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

import torch

from flashinfer.gdn2_prefill import chunk_gated_delta_rule2


def test_gdn2_exported_reference_reads_natural_log_decay():
    q = torch.ones(1, 1, 1)
    zero = torch.zeros_like(q)
    g = torch.full_like(q, 0.5).log()
    state = torch.full((1, 1, 1, 1), 2.0)
    cu = torch.tensor([0, 1], dtype=torch.int64)
    definition = chunk_gated_delta_rule2.fi_trace(
        q=q, k=zero, v=zero, g=g, beta=zero, w=zero, initial_state=state, cu_seqlens=cu
    )
    namespace = {}
    exec(definition["reference"], namespace)
    output, final = namespace["_gdn2_prefill_reference"](
        q, zero, zero, g, zero, zero, state, cu, 1.0
    )
    # No write/erase: a decay of 1/2 maps incoming state 2 to state/output 1.
    torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)
    torch.testing.assert_close(final, torch.ones_like(final), rtol=0, atol=0)
