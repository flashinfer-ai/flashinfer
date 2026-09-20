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

"""CPU trace coverage for packed chunks; real collectives live in tests/comm/."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flashinfer.comm import UlyssesCommunicator
from flashinfer.trace.templates.comm import ulysses_exchange_chunks_trace


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.bfloat16])
def test_exchange_chunks_trace_preserves_payload_dtype(monkeypatch, dtype):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tracing must not construct a communicator or communicate")

    monkeypatch.setattr(UlyssesCommunicator, "__init__", forbidden)
    monkeypatch.setattr(torch.distributed, "all_to_all_single", forbidden)
    x = torch.arange(256).to(dtype).view(1, 1, 1, 256)
    metadata = SimpleNamespace(world_size=1)
    definition = UlyssesCommunicator.exchange_chunks.fi_trace(
        self=metadata, x=x, dtype=dtype
    )
    expected_dtype = str(dtype).removeprefix("torch.")
    assert definition["axes"]["world_size"]["value"] == 1
    assert definition["inputs"]["x"]["dtype"] == expected_dtype
    assert definition["outputs"]["output"]["dtype"] == expected_dtype
    axes = {key: entry["value"] for key, entry in definition["axes"].items()}
    assert all(eval(constraint, {}, axes) for constraint in definition["constraints"])

    # Serialized references must run independently of the source module and
    # preserve caller-owned output identity, including opaque uint8 payloads.
    namespace = {}
    exec(definition["reference"], namespace)
    reference = namespace["_ulysses_single_rank_reference"]
    result = reference(x, dtype=dtype)
    assert result.data_ptr() != x.data_ptr()
    assert torch.equal(result, x)
    out = torch.empty_like(x)
    assert reference(x, out=out, dtype=dtype) is out
    assert torch.equal(out, x)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_exchange_chunks_trace_does_not_model_multi_rank_as_identity(world_size):
    metadata = SimpleNamespace(world_size=world_size)
    x = torch.empty(1, 1, world_size, 256, dtype=torch.uint8)
    assert ulysses_exchange_chunks_trace(self=metadata, x=x) is None
    assert UlyssesCommunicator.exchange_chunks.fi_trace(self=metadata, x=x) == {}

    # trace_apply consults axis extractors separately from fi_trace dispatch.
    # It must see the actual group size, never the schema's fixed reference 1.
    template = ulysses_exchange_chunks_trace(self=SimpleNamespace(world_size=1))
    extract = template._build_axis_extractors()["world_size"]
    assert extract({"self": metadata}) == world_size


def test_exchange_chunks_committed_example_matches_schema():
    definition = UlyssesCommunicator.exchange_chunks.fi_trace(
        self=SimpleNamespace(world_size=1),
        x=torch.empty(1, 1, 1, 256, dtype=torch.uint8),
        dtype=torch.uint8,
    )
    path = Path(__file__).parent / "fi_trace_out" / f"{definition['name']}.json"
    assert json.loads(path.read_text()) == definition
