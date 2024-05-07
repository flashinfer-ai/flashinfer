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


def llama_rms_norm(x, w, eps=1e-6):
    def _norm(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    output = _norm(x.float()).type_as(x)
    return output * w


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 4096, 8192])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_norm(batch_size, hidden_size, dtype):
    x = torch.randn(batch_size, hidden_size).to(0).to(dtype)
    w = torch.randn(hidden_size).to(0).to(dtype)

    y_ref = llama_rms_norm(x, w)
    y = flashinfer.norm.rmsnorm(x, w)

    numpy.testing.assert_allclose(
        y_ref.cpu().numpy(), y.cpu().numpy(), rtol=1e-3, atol=1e-3
    )


if __name__ == "__main__":
    test_norm(1, 111, torch.float16)
