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

import numpy
import torch
import pytest
import flashinfer


def numpy_packbits_ref(x_cpu: torch.Tensor, bitorder: str):
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
    x_packed = flashinfer.packbits(x_gpu, bitorder)

    assert torch.equal(x_packed_ref.cpu(), x_packed.cpu())


@pytest.mark.parametrize("batch_size", [1, 10, 99, 128, 777, 999])
@pytest.mark.parametrize("bitorder", ["big", "little"])
def test_segment_packbits(batch_size, bitorder):
    torch.manual_seed(42)
    old_indptr = torch.cumsum(torch.arange(batch_size + 1), 0).to(0)
    num_elements = old_indptr[-1].item()
    x_cpu = torch.rand(num_elements) < 0.5
    x_gpu = x_cpu.to(0)

    y_gpu, new_indptr = flashinfer.segment_packbits(x_gpu, old_indptr, bitorder)

    for i in range(batch_size):
        x_segment_i = x_gpu[old_indptr[i] : old_indptr[i + 1]]
        y_segment_i_ref = flashinfer.packbits(x_segment_i, bitorder)
        assert torch.equal(y_gpu[new_indptr[i] : new_indptr[i + 1]], y_segment_i_ref)


if __name__ == "__main__":
    test_packbits(999999, "big")
    test_segment_packbits(77, "little")
