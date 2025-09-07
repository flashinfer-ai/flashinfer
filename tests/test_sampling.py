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

import pytest
import torch

import flashinfer


def normal_distribution(std):
    def normal_noise(shape, device):
        return torch.randn(shape, device=device) * std

    normal_noise.__name__ = f"normal_distribution(std={std})"
    return normal_noise


def gumbel_distribution(beta):
    def gumbel_noise(shape, device):
        U = torch.rand(shape, device=device)
        eps = 1e-20
        return torch.log(-torch.log(U + eps) + eps) / beta

    gumbel_noise.__name__ = f"gumbel_distribution(beta={beta})"
    return gumbel_noise


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize(
    "distribution",
    [
        normal_distribution(1),
        normal_distribution(5),
        gumbel_distribution(0.1),
    ],
)
@pytest.mark.parametrize("temperature", [1.0, 0.5, 0.1])
@pytest.mark.parametrize("temperature_arr", [True, False])
@pytest.mark.parametrize("neg_inf_input", [True, False])
def test_softmax(
    batch_size, vocab_size, distribution, temperature, temperature_arr, neg_inf_input
):
    torch.manual_seed(42)
    logits = distribution((batch_size, vocab_size), "cuda:0")
    if neg_inf_input:
        # assign random logits to -inf
        num_inf = torch.randint(0, logits.numel() - 1, (), device=logits.device).item()
        inf_idx = torch.randperm(logits.numel(), device=logits.device)[:num_inf]
        logits.view(-1).index_fill_(0, inf_idx, float("-inf"))

    if temperature_arr:
        temperature_arr = torch.full((batch_size,), temperature, device="cuda:0")
        probs = flashinfer.sampling.softmax(logits, temperature=temperature_arr)
        logits_scaled = logits / temperature_arr.unsqueeze(-1)
    else:
        probs = flashinfer.sampling.softmax(logits, temperature=temperature)
        logits_scaled = logits / temperature

    probs_ref = torch.softmax(logits_scaled, dim=-1)

    assert torch.allclose(probs, probs_ref, atol=1e-5)


@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize(
    "distribution",
    [
        normal_distribution(1),
        normal_distribution(5),
        gumbel_distribution(0.1),
    ],
)
@pytest.mark.parametrize("zero_ratio", [0.0, 0.5, 0.9])
def test_sampling_freq(vocab_size, distribution, zero_ratio):
    torch.manual_seed(42)
    num_trials = 5000000
    logits = distribution((1, vocab_size), "cuda:0")
    zero_indices = torch.randperm(vocab_size)[: int(vocab_size * zero_ratio)]
    logits[:, zero_indices] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    counter = torch.zeros(vocab_size, dtype=torch.int32, device=logits.device)

    samples = flashinfer.sampling.sampling_from_probs(
        probs, indices=torch.zeros(num_trials, dtype=torch.int32, device=logits.device)
    )
    counter.scatter_add_(0, samples.long(), torch.ones_like(samples))
    freq = counter.float() / num_trials

    assert torch.all(counter[zero_indices] == 0)
    similarity = torch.cosine_similarity(freq, probs)
    assert similarity > 0.99, f"similarity: {similarity}"


@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize(
    "distribution",
    [
        normal_distribution(1),
        normal_distribution(5),
        gumbel_distribution(0.1),
    ],
)
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_top_p_sampling_freq(vocab_size, distribution, p):
    # use torch profiler to check the performance of the code
    torch.manual_seed(42)
    logits = distribution((1, vocab_size), "cuda:0")
    probs = torch.softmax(logits, dim=-1)
    sorted_prob, indices = torch.sort(probs, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros(1, vocab_size, dtype=torch.int32, device=logits.device)
    mask.scatter_add_(1, indices, (cdf > (1 - p)).int())

    renorm_probs = flashinfer.sampling.top_p_renorm_probs(probs, p)
    counter = torch.zeros(vocab_size, dtype=torch.int32, device=logits.device)
    num_trials = 5000000
    samples = flashinfer.sampling.top_p_sampling_from_probs(
        probs,
        p,
        indices=torch.zeros(num_trials, dtype=torch.int32, device=logits.device),
    )
    counter.scatter_add_(0, samples.long(), torch.ones_like(samples))
    freq = counter.float() / num_trials
    assert torch.all(mask[torch.arange(1), samples] == 1)
    similarity = torch.cosine_similarity(freq, renorm_probs)
    assert similarity > 0.99, f"similarity: {similarity}"


@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize(
    "distribution",
    [
        normal_distribution(1),
        normal_distribution(5),
        gumbel_distribution(0.1),
    ],
)
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_sampling_freq(vocab_size, distribution, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    logits = distribution((1, vocab_size), "cuda:0")
    probs = torch.softmax(logits, dim=-1)
    sorted_prob, _ = torch.sort(probs, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask = (probs >= pivot.unsqueeze(-1)).int()

    renorm_probs = flashinfer.sampling.top_k_renorm_probs(probs, k)
    counter = torch.zeros(vocab_size, dtype=torch.int32, device=logits.device)
    num_trials = 5000000
    samples = flashinfer.sampling.top_k_sampling_from_probs(
        probs,
        k,
        indices=torch.zeros(num_trials, dtype=torch.int32, device=logits.device),
    )
    counter.scatter_add_(0, samples.long(), torch.ones_like(samples))
    freq = counter.float() / num_trials
    assert torch.all(mask[torch.arange(1), samples] == 1)
    similarity = torch.cosine_similarity(freq, renorm_probs)
    assert similarity > 0.99, f"similarity: {similarity}"


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
def test_sampling(batch_size, vocab_size):
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    num_trails = 5000
    for _ in range(num_trails):
        samples = flashinfer.sampling.sampling_from_probs(normalized_prob)
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
def test_sampling_from_logits(batch_size, vocab_size):
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda:0")
    num_trails = 5000
    for _ in range(num_trails):
        samples = flashinfer.sampling.sampling_from_logits(logits)
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)


@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize(
    "distribution",
    [
        normal_distribution(1),
        normal_distribution(5),
        gumbel_distribution(0.1),
    ],
)
def test_sampling_from_logits_freq(vocab_size, distribution):
    torch.manual_seed(42)
    num_trials = 5000000
    logits = distribution((1, vocab_size), "cuda:0")
    probs = torch.softmax(logits, dim=-1)
    counter = torch.zeros(vocab_size, dtype=torch.int32, device=logits.device)
    samples = flashinfer.sampling.sampling_from_logits(
        logits, indices=torch.zeros(num_trials, dtype=torch.int32, device=logits.device)
    )
    counter.scatter_add_(0, samples.long(), torch.ones_like(samples))
    freq = counter.float() / num_trials
    similarity = torch.cosine_similarity(freq, probs)
    assert similarity > 0.99, f"similarity: {similarity}"


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_top_p_sampling(batch_size, vocab_size, p):
    torch.manual_seed(42)
    eps = 1e-4
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device="cuda:0")
    mask.scatter_add_(1, indices, (cdf > (1 - p) - eps).int())

    num_trails = 1000
    for _ in range(num_trails):
        samples = flashinfer.sampling.top_p_sampling_from_probs(normalized_prob, p)
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)
        assert torch.all(mask[torch.arange(batch_size), samples] == 1)


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_sampling(batch_size, vocab_size, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask = (normalized_prob >= pivot.unsqueeze(-1)).int()

    num_trails = 1000
    for _ in range(num_trails):
        samples = flashinfer.sampling.top_k_sampling_from_probs(normalized_prob, k)
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)
        assert torch.all(mask[torch.arange(batch_size), samples] == 1), normalized_prob[
            torch.arange(batch_size), samples
        ]


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_sampling_with_variable_k(batch_size, vocab_size, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    k = torch.randint(1, k + 1, (batch_size,), device="cuda:0")
    pivot = sorted_prob[torch.arange(batch_size), k - 1]
    mask = (normalized_prob >= pivot.unsqueeze(-1)).int()

    num_trails = 1000
    for _ in range(num_trails):
        samples = flashinfer.sampling.top_k_sampling_from_probs(normalized_prob, k)
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)
        assert torch.all(mask[torch.arange(batch_size), samples] == 1), normalized_prob[
            torch.arange(batch_size), samples
        ]


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.05, 0.1, 0.2, 0.7, 1])
def test_min_p_sampling(batch_size, vocab_size, p):
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    # scale min-p
    top_probs = sorted_prob[:, -1].unsqueeze(-1)
    scaled_p = p * top_probs
    # min-p mask
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device="cuda:0")
    mask.scatter_add_(1, indices, (sorted_prob >= scaled_p).int())
    min_p_tensor = torch.full((batch_size,), p, device="cuda:0")

    num_trails = 1000
    for _ in range(num_trails):
        samples = flashinfer.sampling.min_p_sampling_from_probs(
            normalized_prob,
            min_p_tensor,
        )

        assert torch.all(mask[torch.arange(batch_size), samples] == 1), samples[
            torch.nonzero(mask[torch.arange(batch_size), samples] == 0)
        ]


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5])
def test_top_k_top_p_joint_sampling_from_probs(batch_size, vocab_size, p):
    torch.manual_seed(42)
    if p == 0.1:
        k = int(vocab_size * 0.5)
    elif p == 0.5:
        k = int(vocab_size * 0.1)
    else:
        raise ValueError("p not recognized")
    eps = 1e-4
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    # top-p mask
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask_top_p = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device="cuda:0")
    mask_top_p.scatter_add_(1, indices, (cdf > (1 - p) - eps).int())
    # top-k mask
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask_top_k = (normalized_prob >= pivot.unsqueeze(-1)).int()
    # overall mask
    mask = torch.minimum(mask_top_p, mask_top_k)
    top_p_tensor = torch.full((batch_size,), p, device="cuda:0")
    top_k_tensor = torch.full((batch_size,), k, device="cuda:0")

    num_trails = 1000
    for _ in range(num_trails):
        samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(
            normalized_prob,
            top_k_tensor,
            top_p_tensor,
            filter_apply_order="joint",
        )
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)
        assert torch.all(mask[torch.arange(batch_size), samples] == 1), normalized_prob[
            torch.arange(batch_size), samples
        ]


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [100])
@pytest.mark.parametrize("p", [0.1, 0.5])
def test_top_k_top_p_sampling_from_probs_logits_alignment(batch_size, vocab_size, k, p):
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda:0") * 5
    generator_logits = torch.Generator("cuda:0")
    generator_probs = generator_logits.clone_state()
    samples = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        logits, k, p, filter_apply_order="top_k_first", generator=generator_logits
    )
    samples_ref = flashinfer.sampling.top_k_top_p_sampling_from_probs(
        torch.softmax(logits, dim=-1),
        k,
        p,
        filter_apply_order="top_k_first",
        generator=generator_probs,
    )
    assert torch.all(samples == samples_ref)


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5])
def test_top_k_top_p_joint_sampling_from_logits(batch_size, vocab_size, p):
    torch.manual_seed(42)
    logits = torch.rand(batch_size, vocab_size, device="cuda:0") * 5
    generator_logits = torch.Generator("cuda:0")
    generator_probs = generator_logits.clone_state()
    if p == 0.1:
        k = int(vocab_size * 0.5)
    elif p == 0.5:
        k = int(vocab_size * 0.1)
    else:
        raise ValueError("p not recognized")

    samples = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        logits, k, p, filter_apply_order="joint", generator=generator_logits
    )

    samples_ref = flashinfer.sampling.top_k_top_p_sampling_from_probs(
        torch.softmax(logits, dim=-1),
        k,
        p,
        filter_apply_order="joint",
        generator=generator_probs,
    )
    assert torch.all(samples == samples_ref)


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9, 1.0])
def test_top_p_renorm_probs(batch_size, vocab_size, p):
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device="cuda:0")
    mask.scatter_add_(1, indices, (cdf >= (1 - p)).int())
    renorm_prob_ground_truth = normalized_prob.clone()
    renorm_prob_ground_truth[mask == 0] = 0
    renorm_prob_ground_truth = renorm_prob_ground_truth / renorm_prob_ground_truth.sum(
        dim=-1, keepdim=True
    )

    renorm_prob = flashinfer.sampling.top_p_renorm_probs(normalized_prob, p)
    torch.testing.assert_close(
        renorm_prob_ground_truth,
        renorm_prob,
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_renorm_probs(batch_size, vocab_size, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask = (normalized_prob >= pivot.unsqueeze(-1)).int()
    renorm_prob_ground_truth = normalized_prob.clone()
    renorm_prob_ground_truth[mask == 0] = 0
    renorm_prob_ground_truth = renorm_prob_ground_truth / renorm_prob_ground_truth.sum(
        dim=-1, keepdim=True
    )

    renorm_prob = flashinfer.sampling.top_k_renorm_probs(normalized_prob, k)
    for i in range(batch_size):
        torch.testing.assert_close(
            renorm_prob_ground_truth[i],
            renorm_prob[i],
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
@pytest.mark.parametrize("neginf_input", [False, True])
def test_top_k_mask_logits(batch_size, vocab_size, k, neginf_input):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda:0") * 5
    if neginf_input:
        num_neginf = torch.randint(1, vocab_size * batch_size, (1,)).item()
        idxs = torch.randperm(batch_size * vocab_size, device="cuda:0")[:num_neginf]
        logits[idxs // vocab_size, idxs % vocab_size] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    masked_logits = flashinfer.sampling.top_k_mask_logits(logits, k)
    renormed_probs = torch.softmax(masked_logits, dim=-1)
    renormed_probs_ref = flashinfer.sampling.top_k_renorm_prob(probs, k)

    torch.testing.assert_close(
        renormed_probs,
        renormed_probs_ref,
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("num_speculate_tokens", [1, 3, 5, 7])
@pytest.mark.parametrize("onehot_target", [False, True])
def test_chain_speculative_sampling(
    batch_size,
    vocab_size,
    num_speculate_tokens,
    onehot_target,
):
    pre_norm_draft_prob = torch.rand(
        batch_size, num_speculate_tokens, vocab_size, device="cuda:0"
    )
    normalized_draft_prob = pre_norm_draft_prob / pre_norm_draft_prob.sum(
        dim=-1, keepdim=True
    )
    draft_token_ids = torch.randint(
        vocab_size, (batch_size, num_speculate_tokens), device="cuda:0"
    )
    if not onehot_target:
        pre_norm_target_prob = torch.rand(
            batch_size, num_speculate_tokens + 1, vocab_size, device="cuda:0"
        )
        target_onehot_prob = pre_norm_target_prob / pre_norm_target_prob.sum(
            dim=-1, keepdim=True
        )
    else:
        target_token_ids = torch.randint(
            vocab_size, (batch_size, num_speculate_tokens + 1), device="cuda:0"
        )
        target_token_ids[..., :num_speculate_tokens] = draft_token_ids
        target_onehot_prob = torch.zeros(
            (batch_size, num_speculate_tokens + 1, vocab_size), device="cuda:0"
        )
        target_onehot_prob.scatter_(2, target_token_ids.unsqueeze(-1), 1)

    # NOTE(Zihao): this is a very simple test that only checks whether output is valid or not.
    for trials in range(10):  # noqa: B007
        accepted_num = torch.zeros(batch_size, dtype=torch.int32, device="cuda:0")
        emitted_num = torch.zeros(batch_size, dtype=torch.int32, device="cuda:0")
        (
            output_token_ids,
            accepted_num,
            emitted_num,
        ) = flashinfer.sampling.chain_speculative_sampling(
            normalized_draft_prob,
            draft_token_ids,
            target_onehot_prob,
            accepted_num,
            emitted_num,
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

        assert torch.all(emitted_num + 1 == (output_token_ids != -1).sum(dim=1))


if __name__ == "__main__":
    # test_sampling_freq(128256, gumbel_distribution(0.1), 0.5)
    test_sampling_from_logits_freq(128256, gumbel_distribution(0.1))
    # test_top_p_sampling_freq(128256, gumbel_distribution(0.1), 0.5)
    # test_top_k_sampling_freq(1, 128256, 10)
    # test_sampling(19, 500)
    # test_sampling(1, 111)
    # test_top_p_sampling(3, 111, 0.9)
    # test_top_k_sampling(3, 111, 10)
    # test_top_p_renorm_probs(3, 111, 0.9)
    # test_top_k_renorm_probs(3, 111, 10)
    # test_top_k_mask_logits(99, 989, 10)
    # test_chain_speculative_sampling(3, 111, 3, False)
    # test_chain_speculative_sampling(3, 111, 3, True)
