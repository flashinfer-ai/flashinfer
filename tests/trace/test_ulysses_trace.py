# Copyright (c) 2025 by FlashInfer team.
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

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flashinfer.comm import UlyssesCommunicator
from flashinfer.trace.templates.comm import (
    ulysses_gather_heads_trace,
    ulysses_scatter_heads_trace,
)
from flashinfer.trace_apply.apply import build_extractor_maps, extract_axes


FI_TRACE_OUT = Path(__file__).parent / "fi_trace_out"


@pytest.mark.parametrize(
    ("method", "name", "x_axes"),
    (
        (
            UlyssesCommunicator.scatter_heads,
            "ulysses_scatter_heads_ws1_d128",
            ["batch_size", "local_seq_len", "global_num_heads", "head_dim"],
        ),
        (
            UlyssesCommunicator.gather_heads,
            "ulysses_gather_heads_ws1_d128",
            ["batch_size", "global_seq_len", "local_num_heads", "head_dim"],
        ),
    ),
)
def test_ulysses_trace_schema_and_committed_example(tmp_path, method, name, x_axes):
    x = torch.zeros(1, 128, 8, 128, dtype=torch.bfloat16)
    communicator = SimpleNamespace(world_size=1)
    definition = method.fi_trace(save_dir=tmp_path, self=communicator, x=x)

    assert definition["name"] == name
    assert definition["axes"]["world_size"] == {
        "type": "const",
        "value": 1,
        "description": "fi_trace models the single-rank identity case only.",
    }
    assert definition["constraints"] == ["world_size == 1"]
    assert definition["inputs"]["x"] == {
        "shape": x_axes,
        "dtype": "bfloat16",
    }
    assert "out" not in definition["inputs"]
    assert definition["outputs"]["output"] == {
        "shape": x_axes,
        "dtype": "bfloat16",
        "param": "out",
    }

    generated = json.loads((tmp_path / f"{name}.json").read_text())
    committed = json.loads((FI_TRACE_OUT / f"{name}.json").read_text())
    assert generated == committed

    namespace = {}
    exec(committed["reference"], namespace)  # noqa: S102
    reference = namespace["_ulysses_single_rank_reference"]
    expected = reference(x)
    torch.testing.assert_close(expected, x, rtol=0, atol=0)
    assert expected.data_ptr() != x.data_ptr()

    out = torch.empty_like(x)
    assert reference(x, out=out) is out
    torch.testing.assert_close(out, x, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("method", "dispatch", "prefix"),
    (
        (
            UlyssesCommunicator.scatter_heads,
            ulysses_scatter_heads_trace,
            "ulysses_scatter_heads",
        ),
        (
            UlyssesCommunicator.gather_heads,
            ulysses_gather_heads_trace,
            "ulysses_gather_heads",
        ),
    ),
)
def test_ulysses_trace_never_routes_multi_rank_to_identity(method, dispatch, prefix):
    x = torch.zeros(1, 8, 8, 128, dtype=torch.bfloat16)
    communicator = SimpleNamespace(world_size=8)

    assert method.fi_trace(self=communicator, x=x) == {}
    template = dispatch.templates[0]
    axes = extract_axes(
        build_extractor_maps([template]),
        {"self": communicator, "x": x, "out": None},
    )
    # The world_size axis comes off the communicator, not off any tensor dim or
    # scalar argument, so a regressed extractor would fall back to the template's
    # fixed Const(value=1) and name this ws1 instead.
    assert template.definition_name(axes) == f"{prefix}_ws8_d128"
