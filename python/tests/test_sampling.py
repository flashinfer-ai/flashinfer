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


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 500, 32000, 128256])
def test_sampling(batch_size, vocab_size):
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    uniform_samples = torch.empty(batch_size, dtype=torch.float32).to(0)

    num_trails = 5000
    for _ in range(num_trails):
        uniform_samples.uniform_()
        samples = flashinfer.sampling.sampling_from_probs(
            normalized_prob, uniform_samples
        )
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 500, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_top_p_sampling(batch_size, vocab_size, p):
    torch.manual_seed(42)
    eps = 1e-4
    max_top_p_trails = 32
    pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.int32).to(0)
    mask.scatter_add_(1, indices, (cdf > (1 - p) - eps).int())
    uniform_samples = torch.empty(max_top_p_trails, batch_size, dtype=torch.float32).to(
        0
    )

    num_trails = 1000
    for _ in range(num_trails):
        uniform_samples.uniform_()
        samples, success = flashinfer.sampling.top_p_sampling_from_probs(
            normalized_prob, uniform_samples, p
        )
        assert torch.all(success)
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)
        assert torch.all(mask[torch.arange(batch_size), samples] == 1)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 500, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_sampling(batch_size, vocab_size, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    max_top_k_trails = 32
    pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask = (normalized_prob >= pivot.unsqueeze(-1)).int()
    uniform_samples = torch.empty(max_top_k_trails, batch_size, dtype=torch.float32).to(
        0
    )

    num_trails = 1000
    for _ in range(num_trails):
        uniform_samples.uniform_()
        samples, success = flashinfer.sampling.top_k_sampling_from_probs(
            normalized_prob, uniform_samples, k
        )
        assert torch.all(success)
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)
        assert torch.all(mask[torch.arange(batch_size), samples] == 1), normalized_prob[
            torch.arange(batch_size), samples
        ]


if __name__ == "__main__":
    test_sampling(1, 111)
    test_top_p_sampling(3, 111, 0.9)
    test_top_k_sampling(3, 111, 10)
