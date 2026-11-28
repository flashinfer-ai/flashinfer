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

"""Die-local scheduling of the CuTe DSL KDA prefill (``sm_locality_domains``).

The variant only changes which block computes which (sequence, head), so its
output and state must be bitwise those of the stock build for any map that
names two domains; the map here alternates SM pairs.
"""

import pytest
import torch

from flashinfer.kda import RecurrentKDAPrefillWrapper, recurrent_kda
from flashinfer.kda_prefill_cute import (
    _SUPPORTED_COMPUTE_CAPABILITIES,
    _is_cute_dsl_kda_runtime_available,
)
from flashinfer.utils import get_compute_capability

from tests.kda.test_helpers.kda_prefill import packed_prefill_inputs

NUM_HEADS = 12
# Mixed lengths; 4 sequences take the two-kernel route, 20 the single kernel.
SEQ_LENS = {
    "decomp": [200, 48, 333, 16],
    "engine": [96, 16, 240, 64, 33, 128, 48, 16, 160, 80] * 2,
}


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if (
        get_compute_capability(device) not in _SUPPORTED_COMPUTE_CAPABILITIES
        or not _is_cute_dsl_kda_runtime_available()
    ):
        pytest.skip("the CuTe DSL BT=16 KDA prefill is not available here")
    return device


def _sm_locality_domains(device):
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    return [(sm >> 1) & 1 for sm in range(sm_count)]


def _inputs(device, seq_lens):
    inputs = packed_prefill_inputs(
        device, seq_lens=seq_lens, num_heads=NUM_HEADS, seed=5069
    )
    generator = torch.Generator(device=device).manual_seed(1)
    inputs["initial_state"] = 0.02 * torch.randn(
        (len(seq_lens), NUM_HEADS, 128, 128), device=device, generator=generator
    )
    inputs["ssm_state_indices"] = torch.arange(
        len(seq_lens), dtype=torch.int32, device=device
    )
    return inputs


@pytest.mark.parametrize("route", ["decomp", "engine"])
def test_die_local_prefill_is_bitwise_stock(device, route):
    seq_lens = SEQ_LENS[route]
    domains = _sm_locality_domains(device)

    expected = _inputs(device, seq_lens)
    expected_out, _ = recurrent_kda(**expected, backend="cute-dsl")

    wrapper = RecurrentKDAPrefillWrapper(device)
    num_bytes, ranges = wrapper.scratch_layout(seq_lens, NUM_HEADS)
    assert (num_bytes > 0) == (route == "decomp")
    assert sum(stop - start for seq in ranges for start, stop in seq) <= num_bytes
    if num_bytes:
        scratch = torch.empty(num_bytes, dtype=torch.uint8, device=device)
        wrapper.set_scratch(scratch)
    # Twice on one wrapper: the second launch reuses the claim counter.
    for _ in range(2):
        planned = _inputs(device, seq_lens)
        wrapper.plan(planned.pop("cu_seqlens"))
        planned_out, _ = wrapper.run(**planned, sm_locality_domains=domains)
        assert torch.equal(planned_out, expected_out)
        assert torch.equal(planned["initial_state"], expected["initial_state"])
        # The variant ran (the counter exists) and every item was claimed.
        assert int(wrapper._impl.workspace._cute_dsl_die_claim) == 0
    if num_bytes:
        assert wrapper._impl.workspace._cute_dsl_workspace is scratch

    eager = _inputs(device, seq_lens)
    eager_out, _ = recurrent_kda(
        **eager, backend="cute-dsl", sm_locality_domains=domains
    )
    assert torch.equal(eager_out, expected_out)
    assert torch.equal(eager["initial_state"], expected["initial_state"])


def test_set_scratch_rejects_a_buffer_that_is_too_small(device):
    seq_lens = SEQ_LENS["decomp"]
    wrapper = RecurrentKDAPrefillWrapper(device)
    num_bytes, _ = wrapper.scratch_layout(seq_lens, NUM_HEADS)
    wrapper.set_scratch(torch.empty(num_bytes - 1, dtype=torch.uint8, device=device))
    inputs = _inputs(device, seq_lens)
    wrapper.plan(inputs.pop("cu_seqlens"))

    with pytest.raises(ValueError, match="set_scratch"):
        wrapper.run(**inputs)


def test_set_scratch_rejects_a_misaligned_or_strided_buffer(device):
    wrapper = RecurrentKDAPrefillWrapper(device)
    storage = torch.empty(4096, dtype=torch.uint8, device=device)

    for scratch in (storage[1:], storage[::2]):
        with pytest.raises(ValueError, match="16-byte-aligned"):
            wrapper.set_scratch(scratch)
    wrapper.set_scratch(storage[16:])
