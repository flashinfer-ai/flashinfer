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

"""Correctness, fixed-address reuse and stream/graph tests for GB300 W4A8."""

import pytest
import torch

from flashinfer.fused_moe import prepare_fp4_block_scale_routed_moe
from flashinfer.experimental.cake_w4a8_received_tokens.cake_example import (
    assert_correct,
    make_inputs,
    reference,
)


def _supported():
    if not torch.cuda.is_available():
        return False
    p = torch.cuda.get_device_properties(0)
    return (p.major, p.minor, p.multi_processor_count) == (10, 3, 152)


pytestmark = pytest.mark.skipif(not _supported(), reason="requires a 152-SM GB300")


@pytest.mark.parametrize(
    "rows,pattern",
    [
        (121, "duplicates"),
        (223, "mixed"),
        (224, "mixed"),
        (384, "all_hot"),
        (639, "duplicates"),
        (640, "mixed"),
        (732, "duplicates"),
    ],
)
def test_live_inputs_stream_and_graph(rows, pattern):
    inputs = make_inputs(rows, pattern)
    output = torch.empty((rows, 3072), dtype=torch.bfloat16, device="cuda")
    prepared = prepare_fp4_block_scale_routed_moe(
        *inputs, local_expert_offset=0, output=output, backend="cake"
    )
    expected = reference(inputs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        assert prepared.run() is output
    torch.cuda.current_stream().wait_stream(stream)
    assert_correct(output, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        prepared.run()
    copies = []
    for _ in range(3):
        output.fill_(float("nan"))
        graph.replay()
        copies.append(output.clone())
    for value in copies:
        assert_correct(value, expected)
        assert torch.equal(value, copies[0])
    original = {i: inputs[i].clone() for i in (0, 6, 7)}
    pointers = [value.data_ptr() for value in inputs]
    inputs[0].copy_((-inputs[0].float()).to(torch.float8_e4m3fn))
    inputs[6].copy_(inputs[6].roll(1, -1))
    inputs[7].copy_(inputs[7].roll(2, -1))
    graph.replay()
    assert_correct(output, reference(inputs))
    assert pointers == [value.data_ptr() for value in inputs]
    for i, value in original.items():
        inputs[i].copy_(value)
    graph.replay()
    assert_correct(output, expected)
    before = torch.cuda.memory_stats()
    prepared.run()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert before["allocation.all.allocated"] == after["allocation.all.allocated"]
    assert before["allocation.all.freed"] == after["allocation.all.freed"]


def test_input_contract():
    inputs = make_inputs(121)
    output = torch.empty((121, 3072), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="backend"):
        prepare_fp4_block_scale_routed_moe(
            *inputs, local_expert_offset=0, output=output, backend="auto"
        )
    with pytest.raises(ValueError, match="local_expert_offset"):
        prepare_fp4_block_scale_routed_moe(
            *inputs, local_expert_offset=1, output=output
        )
    bad = list(inputs)
    bad[6] = bad[6].long()
    with pytest.raises(ValueError, match="dtype"):
        prepare_fp4_block_scale_routed_moe(*bad, local_expert_offset=0, output=output)
    bad = list(inputs)
    bad[0] = bad[0].T.contiguous().T
    with pytest.raises(ValueError, match="contiguous"):
        prepare_fp4_block_scale_routed_moe(*bad, local_expert_offset=0, output=output)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("rows", [121, 384, 732])
def test_run_with_different_current_device(rows):
    """Honor the bound device's stream while restoring the caller's device."""
    with torch.cuda.device(0):
        inputs = make_inputs(rows, "duplicates")
        output = torch.empty((rows, 3072), dtype=torch.bfloat16, device="cuda")
        prepared = prepare_fp4_block_scale_routed_moe(
            *inputs, local_expert_offset=0, output=output, backend="cake"
        )
        expected = reference(inputs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            output.fill_(float("nan"))
            with torch.cuda.device(1):
                assert prepared.run() is output
                assert torch.cuda.current_device() == 1
        torch.cuda.current_stream().wait_stream(stream)
        assert_correct(output, expected)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream), torch.cuda.device(1):
            assert prepared.run() is output
            assert torch.cuda.current_device() == 1
        for _ in range(3):
            output.fill_(float("nan"))
            graph.replay()
            assert_correct(output, expected)
