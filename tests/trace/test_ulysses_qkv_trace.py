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

"""CPU schema tests; distributed numerical validation lives in tests/comm/."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flashinfer.comm import UlyssesCommunicator, UlyssesQKV
from flashinfer.trace.templates.comm import ulysses_scatter_qkv_trace_dispatch


@pytest.mark.parametrize("layout", ["sage2_sm90", "sage2_sm89_sm120"])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_ulysses_qkv_trace_metadata_without_collectives(
    monkeypatch, layout, head_dim, world_size
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Generating a trace must not prepare or run collectives")

    monkeypatch.setattr(torch.distributed, "init_process_group", forbidden)
    monkeypatch.setattr(torch.distributed, "all_gather", forbidden)
    monkeypatch.setattr(torch.distributed, "all_gather_object", forbidden)
    monkeypatch.setattr(torch.distributed, "all_to_all_single", forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_capability", forbidden)
    monkeypatch.setattr(UlyssesCommunicator, "prepare_qkv", forbidden)

    q = torch.empty(2, 65, 8, head_dim, dtype=torch.bfloat16)
    workspace = SimpleNamespace(
        layout=layout, world_size=world_size, used_sequence=world_size * 65 - 1
    )
    definition = UlyssesCommunicator.scatter_qkv.fi_trace(
        q=q, k=q, v=q, workspace=workspace
    )
    assert definition["axes"]["head_dim"]["value"] == head_dim
    assert definition["axes"]["world_size"]["value"] == world_size
    assert definition["axes"]["used_sequence"]["value"] == workspace.used_sequence
    assert f"layout:{layout}" in definition["tags"]
    assert "distributed:multi-rank" in definition["tags"]
    assert "init" not in definition
    assert "reference" not in definition
    assert list(definition["outputs"]) == list(UlyssesQKV._fields)
    assert definition["outputs"]["q"]["dtype"] == "int8"
    assert definition["outputs"]["k"]["dtype"] == "int8"
    assert definition["outputs"]["v"]["dtype"] == "float8_e4m3fn"
    assert definition["outputs"]["v_scale"]["shape"] == [
        "batch",
        "local_heads",
        "head_dim",
    ]

    sm90 = layout == "sage2_sm90"
    q_tile, k_group, alignment = (64, 128, 128) if sm90 else (128, 64, 64)
    axes = dict(
        batch=2,
        local_sequence=65,
        num_heads=8,
        head_dim=head_dim,
        world_size=world_size,
        used_sequence=workspace.used_sequence,
        logical_sequence=65 * world_size,
        padded_sequence=(65 * world_size + alignment - 1) // alignment * alignment,
        local_heads=8 // world_size,
        q_scale_width=(workspace.used_sequence + q_tile - 1) // q_tile * 4,
        k_scale_width=(workspace.used_sequence + k_group - 1) // k_group,
    )
    assert all(eval(expression, {}, axes) for expression in definition["constraints"])


def test_ulysses_qkv_trace_distinguishes_equal_shapes_with_different_live_prefixes():
    q = torch.empty(1, 128, 8, 64, dtype=torch.float16)
    definitions = [
        UlyssesCommunicator.scatter_qkv.fi_trace(
            q=q,
            k=q,
            v=q,
            workspace=SimpleNamespace(
                layout="sage2_sm89_sm120", world_size=2, used_sequence=used_sequence
            ),
        )
        for used_sequence in (129, 130)
    ]
    assert definitions[0]["outputs"] == definitions[1]["outputs"]
    assert definitions[0]["name"] != definitions[1]["name"]
    assert all(
        definition["inputs"]["q"]["dtype"] == "float16" for definition in definitions
    )


def test_ulysses_qkv_trace_requires_known_prepared_layout():
    with pytest.raises(ValueError, match="workspace"):
        UlyssesCommunicator.scatter_qkv.fi_trace()
    with pytest.raises(ValueError, match="Unsupported"):
        ulysses_scatter_qkv_trace_dispatch(workspace=SimpleNamespace(layout="unknown"))


@pytest.mark.parametrize("layout", ["sage2_sm90", "sage2_sm89_sm120"])
def test_ulysses_qkv_trace_checks_tensor_and_metadata_fields(layout):
    workspace = SimpleNamespace(layout=layout, world_size=2, used_sequence=129)
    template = ulysses_scatter_qkv_trace_dispatch(workspace=workspace)
    result = UlyssesQKV(
        torch.zeros(1, 256, 1, 64, dtype=torch.int8),
        torch.zeros(1, 256, 1, 64, dtype=torch.int8),
        torch.zeros(1, 64, 1, 256).to(torch.float8_e4m3fn),
        torch.ones(1, 1, 12 if layout == "sage2_sm90" else 8),
        torch.ones(1, 1, 2 if layout == "sage2_sm90" else 3),
        torch.ones(1, 1, 64),
        layout,
        256,
        129,
        torch.bfloat16,
    )
    copied = UlyssesQKV(
        *(
            value.clone() if isinstance(value, torch.Tensor) else value
            for value in result
        )
    )
    assert template.check(result, copied)
    assert not template.check(result, copied._replace(used_sequence=130))
    assert not template.check(result, copied._replace(input_dtype=torch.float16))
    assert not template.check(
        result, copied._replace(v_scale=torch.zeros_like(result.v_scale))
    )


@pytest.mark.parametrize("layout", ["sage2_sm90", "sage2_sm89_sm120"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_ulysses_qkv_committed_example_matches_schema(layout, head_dim):
    q = torch.empty(1, 65, 8, head_dim, dtype=torch.bfloat16)
    definition = UlyssesCommunicator.scatter_qkv.fi_trace(
        q=q,
        k=q,
        v=q,
        workspace=SimpleNamespace(layout=layout, world_size=8, used_sequence=513),
    )
    path = Path(__file__).parent / "fi_trace_out" / f"{definition['name']}.json"
    assert json.loads(path.read_text()) == definition
