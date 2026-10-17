"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json

import pytest
import torch


def _replay_clone(tmp_path, actual, expected):
    from flashinfer.api_logging import replay_from_dump

    torch.save({"arg_0": actual}, tmp_path / "inputs.pt")
    torch.save({"result": expected}, tmp_path / "outputs.pt")
    (tmp_path / "metadata.jsonl").write_text(
        json.dumps({"module": "torch", "function_name": "clone"}) + "\n"
    )
    result = replay_from_dump(
        str(tmp_path), compare_outputs=True, device="cpu", run=True
    )
    assert "execution_error" not in result
    torch.testing.assert_close(result["execution_result"], actual, rtol=0, atol=0)
    return result["comparison_match"]


@pytest.mark.parametrize(
    "dtype,value,different",
    [
        (torch.bool, True, False),
        (torch.uint8, 200, 201),
        (torch.int8, 100, 101),
        (torch.int16, 2000, 2001),
        (torch.int32, 100000, 100001),
        (torch.int64, 2**53, 2**53 + 1),
        (torch.uint16, 10000, 10001),
        (torch.uint32, 100000, 100001),
        (torch.uint64, 2**63, 2**63 + 1),
    ],
)
@pytest.mark.parametrize("matches", [True, False])
def test_replay_discrete_outputs_compare_exactly(
    tmp_path, dtype, value, different, matches
):
    actual = torch.tensor([value], dtype=dtype)
    expected = torch.tensor([value if matches else different], dtype=dtype)

    assert _replay_clone(tmp_path, actual, expected) is matches


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ],
)
@pytest.mark.parametrize(
    "expected_value,matches", [(0.125, True), (0.1255, True), (0.135, False)]
)
def test_replay_floating_outputs_preserve_tolerance(
    tmp_path, dtype, expected_value, matches
):
    actual = torch.tensor([0.125], dtype=dtype)
    expected = torch.tensor([expected_value], dtype=dtype)

    assert _replay_clone(tmp_path, actual, expected) is matches


@pytest.mark.parametrize(
    "expected",
    [torch.tensor([1, 1], dtype=torch.int32), torch.tensor([1], dtype=torch.int64)],
    ids=["shape-mismatch", "dtype-mismatch"],
)
def test_replay_output_shape_and_dtype_must_match(tmp_path, expected):
    actual = torch.tensor([1], dtype=torch.int32)

    assert _replay_clone(tmp_path, actual, expected) is False
