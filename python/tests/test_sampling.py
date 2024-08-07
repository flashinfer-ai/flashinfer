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


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 500, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5])
def test_top_k_top_p_sampling(batch_size, vocab_size, p):
    if p == 0.1:
        k = int(vocab_size * 0.5)
    elif p == 0.5:
        k = int(vocab_size * 0.1)
    else:
        raise ValueError("p not recognized")
    max_top_k_trails = 32
    eps = 1e-4
    pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    # top-p mask
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask_top_p = torch.zeros(batch_size, vocab_size, dtype=torch.int32).to(0)
    mask_top_p.scatter_add_(1, indices, (cdf > (1 - p) - eps).int())
    # top-k mask
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask_top_k = (normalized_prob >= pivot.unsqueeze(-1)).int()
    # overall mask
    mask = torch.minimum(mask_top_p, mask_top_k)
    uniform_samples = torch.empty(max_top_k_trails, batch_size, dtype=torch.float32).to(
        0
    )
    top_p_tensor = torch.full((batch_size,), p).to(0)
    top_k_tensor = torch.full((batch_size,), k).to(0)

    num_trails = 1000
    for _ in range(num_trails):
        uniform_samples.uniform_()
        samples, success = flashinfer.sampling.top_k_top_p_sampling_from_probs(
            normalized_prob, uniform_samples, top_k_tensor, top_p_tensor
        )
        assert torch.all(success)
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)
        assert torch.all(mask[torch.arange(batch_size), samples] == 1), normalized_prob[
            torch.arange(batch_size), samples
        ]


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 500, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_top_p_renorm_prob(batch_size, vocab_size, p):
    eps = 1e-6
    pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.int32).to(0)
    mask.scatter_add_(1, indices, (cdf >= (1 - p)).int())
    renorm_prob_ground_truth = normalized_prob
    renorm_prob_ground_truth[mask == 0] = 0
    renorm_prob_ground_truth = renorm_prob_ground_truth / renorm_prob_ground_truth.sum(
        dim=-1, keepdim=True
    )

    renorm_prob = flashinfer.sampling.top_p_renorm_prob(normalized_prob, p, eps=eps)
    numpy.testing.assert_allclose(
        renorm_prob_ground_truth.cpu().numpy(),
        renorm_prob.cpu().numpy(),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 500, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_renorm_prob(batch_size, vocab_size, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size).to(0)
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask = (normalized_prob >= pivot.unsqueeze(-1)).int()
    renorm_prob_ground_truth = normalized_prob
    renorm_prob_ground_truth[mask == 0] = 0
    renorm_prob_ground_truth = renorm_prob_ground_truth / renorm_prob_ground_truth.sum(
        dim=-1, keepdim=True
    )

    renorm_prob = flashinfer.sampling.top_k_renorm_prob(normalized_prob, k)
    numpy.testing.assert_allclose(
        renorm_prob_ground_truth.cpu().numpy(),
        renorm_prob.cpu().numpy(),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 500, 32000, 128256])
@pytest.mark.parametrize("num_speculate_tokens", [1, 3, 5, 7])
@pytest.mark.parametrize("onehot_target", [False, True])
def test_chain_speculative_sampling(
    batch_size,
    vocab_size,
    num_speculate_tokens,
    onehot_target,
):
    pre_norm_draft_prob = torch.rand(batch_size, num_speculate_tokens, vocab_size).to(0)
    normalized_draft_prob = pre_norm_draft_prob / pre_norm_draft_prob.sum(
        dim=-1, keepdim=True
    )
    draft_token_ids = torch.randint(vocab_size, (batch_size, num_speculate_tokens)).to(
        0
    )
    uniform_samples = torch.empty(batch_size, num_speculate_tokens + 1).to(0)
    if not onehot_target:
        pre_norm_target_prob = torch.rand(
            batch_size, num_speculate_tokens + 1, vocab_size
        ).to(0)
        target_onehot_prob = pre_norm_target_prob / pre_norm_target_prob.sum(
            dim=-1, keepdim=True
        )
    else:
        target_token_ids = torch.randint(
            vocab_size, (batch_size, num_speculate_tokens + 1)
        ).to(0)
        target_token_ids[..., :num_speculate_tokens] = draft_token_ids
        target_onehot_prob = torch.zeros(
            (batch_size, num_speculate_tokens + 1, vocab_size)
        ).to(0)
        target_onehot_prob.scatter_(2, target_token_ids.unsqueeze(-1), 1)

    # NOTE(Zihao): this is a very simple test that only checks whether output is valid or not.
    for trials in range(10):
        uniform_samples.uniform_()
        output_token_ids = flashinfer.sampling.chain_speculative_sampling(
            normalized_draft_prob,
            draft_token_ids,
            uniform_samples,
            target_onehot_prob,
        )
        if onehot_target:
            assert torch.all(output_token_ids == target_token_ids)
        else:
            assert torch.all(output_token_ids[output_token_ids >= 0] < vocab_size)
            assert output_token_ids.shape == (batch_size, num_speculate_tokens + 1)
            matches = output_token_ids[..., :-1] != draft_token_ids
            for row in range(batch_size):
                mismatch_idx = torch.nonzero(matches[row], as_tuple=True)[0]
                if len(mismatch_idx) > 0:
                    # mismatch_idx should be contiguous
                    assert torch.all(mismatch_idx[1:] == mismatch_idx[:-1] + 1)
                    # from the second mismatched token on, the output tokens should be -1
                    assert torch.all(output_token_ids[row, mismatch_idx[0] + 1 :] == -1)


if __name__ == "__main__":
    test_sampling(1, 111)
    test_top_p_sampling(3, 111, 0.9)
    test_top_k_sampling(3, 111, 10)
    test_top_p_renorm_prob(3, 111, 0.9)
    test_top_k_renorm_prob(3, 111, 10)
    test_chain_speculative_sampling(3, 111, 3, False)
    test_chain_speculative_sampling(3, 111, 3, True)
