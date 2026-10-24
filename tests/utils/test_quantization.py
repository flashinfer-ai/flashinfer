"""
Copyright (c) 2024 by FlashInfer team.

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

from typing import Literal

import numpy
import pytest
import torch

import flashinfer


def numpy_packbits_ref(x_cpu: torch.Tensor, bitorder: Literal["big", "little"]):
    x_np = x_cpu.numpy()
    x_packed = numpy.packbits(x_np, bitorder=bitorder)
    return torch.tensor(x_packed)


@pytest.mark.parametrize("num_elements", [1, 10, 99, 128, 999, 5000, 131072, 999999])
@pytest.mark.parametrize("bitorder", ["big", "little"])
def test_packbits(num_elements, bitorder):
    torch.manual_seed(42)
    x_cpu = torch.rand(num_elements) < 0.5
    x_gpu = x_cpu.to(0)
    x_packed_ref = numpy_packbits_ref(x_cpu, bitorder)
    x_packed = flashinfer.quantization.packbits(x_gpu, bitorder)

    assert torch.equal(x_packed_ref.cpu(), x_packed.cpu())


@pytest.mark.parametrize("batch_size", [1, 10, 99, 128, 777, 999])
@pytest.mark.parametrize("bitorder", ["big", "little"])
def test_segment_packbits(batch_size, bitorder):
    torch.manual_seed(42)
    old_indptr = torch.cumsum(torch.arange(batch_size + 1), 0).to(0)
    num_elements = old_indptr[-1].item()
    x_cpu = torch.rand(num_elements) < 0.5
    x_gpu = x_cpu.to(0)

    y_gpu, new_indptr = flashinfer.quantization.segment_packbits(
        x_gpu, old_indptr, bitorder
    )

    for i in range(batch_size):
        x_segment_i = x_gpu[old_indptr[i] : old_indptr[i + 1]]
        y_segment_i_ref = flashinfer.packbits(x_segment_i, bitorder)
        assert torch.equal(y_gpu[new_indptr[i] : new_indptr[i + 1]], y_segment_i_ref)


@pytest.mark.parametrize(
    "lengths",
    [
        [0, 1, 7, 8, 9, 0, 17],
        [0, 2047, 2048, 2049, 4095, 4096, 4097],
        [131071, 65537, 0, 1048583],
        [0] * 17 + [1048579, 1, 7, 8, 0],
        [0, 1, 7, 8, 2049] * 128,
        [0, 0, 0],
        [],
    ],
)
@pytest.mark.parametrize("bitorder", ["big", "little"])
@pytest.mark.parametrize("input_offset", [0, 1, 3, 7])
@pytest.mark.parametrize("indptr_dtype", [torch.int32, torch.int64])
def test_segment_packbits_boundaries(lengths, bitorder, input_offset, indptr_dtype):
    torch.manual_seed(42)
    indptr_cpu = torch.tensor([0] + lengths, dtype=indptr_dtype).cumsum(0)
    x_cpu = torch.rand(sum(lengths) + input_offset) < 0.5
    x = x_cpu.to("cuda")[input_offset:]
    x_cpu = x_cpu[input_offset:]
    expected = numpy.concatenate(
        [
            numpy.packbits(x_cpu[start:end].numpy(), bitorder=bitorder)
            for start, end in zip(indptr_cpu[:-1], indptr_cpu[1:], strict=True)
        ]
        or [numpy.empty(0, dtype=numpy.uint8)]
    )
    expected_indptr = torch.tensor([0] + [(n + 7) // 8 for n in lengths]).cumsum(0)

    actual, actual_indptr = flashinfer.quantization.segment_packbits(
        x, indptr_cpu.to(device="cuda", dtype=indptr_dtype), bitorder
    )
    torch.testing.assert_close(actual.cpu(), torch.from_numpy(expected), rtol=0, atol=0)
    torch.testing.assert_close(
        actual_indptr.cpu().long(), expected_indptr, rtol=0, atol=0
    )


@pytest.mark.parametrize("bitorder", ["big", "little"])
def test_segment_packbits_graph_and_output_bounds(bitorder):
    from flashinfer.quantization.packbits import get_quantization_module

    torch.manual_seed(42)
    lengths = [1048583, 0, 65537, 7]
    indptr = torch.tensor([0] + lengths, device="cuda", dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    x = (torch.rand(sum(lengths) + 1, device="cuda") < 0.5)[1:]
    expected, out_indptr = flashinfer.quantization.segment_packbits(x, indptr, bitorder)
    storage = torch.full(
        (expected.numel() + 64,), 0xA5, device="cuda", dtype=torch.uint8
    )
    output = storage[32:-32]
    module = get_quantization_module()
    # Only the allocation-free kernel is graph-compatible; the public wrapper
    # reads the output size on the host before allocating its result.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        module.segment_packbits(x, indptr, out_indptr, bitorder, output)
    graph.replay()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert torch.all(storage[:32] == 0xA5)
    assert torch.all(storage[-32:] == 0xA5)


if __name__ == "__main__":
    test_packbits(999999, "big")
    test_segment_packbits(77, "little")
