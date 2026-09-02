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
        torch.cuda.synchronize()  # wait for the index_fill_ to finish because it can overlap with the softmax kernel

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

    num_matches = (samples == samples_ref).sum().item()
    match_rate = num_matches / samples.numel()

    # NOTE(Zihao): Applying softmax followed by top_k_renorm (softmax -> top_k_renorm)
    # does not guarantee bitwise-identical results compared to top_k_mask followed by softmax (top_k_mask -> softmax).
    # This may cause slight differences in subsequent top-p sampling.
    # Additionally, ties at the k-th position may be resolved differently.
    # We tolerate up to a 5% mismatch rate.
    assert match_rate >= 0.95, (
        f"Sample match rate {match_rate:.2%} is below threshold "
        f"({samples.numel() - num_matches}/{samples.numel()} mismatches, expected <=5%)"
    )


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


@pytest.mark.parametrize("batch_size", [4, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
def test_top_p_renorm_probs_per_request(batch_size, vocab_size):
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    # Per-request top_p values varying across the batch
    top_p_arr = torch.linspace(0.1, 0.9, batch_size, device="cuda:0")

    # Compute ground truth per row
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device="cuda:0")
    mask.scatter_add_(1, indices, (cdf >= (1 - top_p_arr.unsqueeze(1))).int())
    renorm_prob_ground_truth = normalized_prob.clone()
    renorm_prob_ground_truth[mask == 0] = 0
    renorm_prob_ground_truth = renorm_prob_ground_truth / renorm_prob_ground_truth.sum(
        dim=-1, keepdim=True
    )

    renorm_prob = flashinfer.sampling.top_p_renorm_probs(normalized_prob, top_p_arr)
    torch.testing.assert_close(
        renorm_prob_ground_truth,
        renorm_prob,
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
@pytest.mark.parametrize(
    "distribution",
    [
        normal_distribution(1),
        normal_distribution(5),
        gumbel_distribution(0.1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_top_k_renorm_probs(batch_size, vocab_size, k, distribution, dtype):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")

    torch.manual_seed(42)
    logits = distribution((batch_size, vocab_size), "cuda:0")
    normalized_prob_fp32 = torch.softmax(logits, dim=-1)
    normalized_prob = normalized_prob_fp32.to(dtype)

    renorm_prob = flashinfer.sampling.top_k_renorm_probs(normalized_prob, k)

    # Check output dtype matches input
    assert renorm_prob.dtype == dtype

    # Check that the output sums to 1
    sums = renorm_prob.float().sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-2, atol=1e-2)

    # Count non-zero elements in output
    nonzero_counts = (renorm_prob > 0).sum(dim=-1)

    # Find the pivot value (k-th largest) and count ties
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]

    # Count how many elements are strictly greater than pivot
    num_greater = (normalized_prob > pivot.unsqueeze(-1)).sum(dim=-1)
    # Count how many elements equal the pivot (ties)
    num_ties = (normalized_prob == pivot.unsqueeze(-1)).sum(dim=-1)

    # Valid range: [num_greater, num_greater + num_ties]
    # The kernel must keep all elements > pivot, and may keep some/all/none of the ties
    # But it must keep exactly k elements total (if there are enough)
    nonzero_input = (normalized_prob > 0).sum(dim=-1)
    expected_k = torch.minimum(
        torch.full_like(nonzero_input, k, dtype=torch.int64), nonzero_input
    )

    # Check: nonzero_counts should be in valid range considering ties
    max_valid = num_greater + num_ties

    # The actual count should be >= k (we keep at least k) and within tie range
    # Due to floating point, allow small tolerance
    assert torch.all(nonzero_counts >= torch.clamp(expected_k - 1, min=0)), (
        f"Some rows have fewer non-zero elements than expected. "
        f"nonzero_counts min: {nonzero_counts.min()}, expected_k min: {expected_k.min()}"
    )
    assert torch.all(nonzero_counts <= max_valid + 1), (
        f"Some rows have more non-zero elements than allowed by ties. "
        f"nonzero_counts max: {nonzero_counts.max()}, max_valid max: {max_valid.max()}"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_top_k_renorm_probs_mixed_k_persistent_loop(dtype):
    """Test top_k_renorm_probs with mixed k values in persistent loop (multi-CTA mode).

    This test catches a specific bug where:
    - Large batch size triggers the persistent loop (multiple iterations per CTA group)
    - Large vocab_size triggers multi-CTA mode (multiple CTAs per row)
    - Mixed k values: some rows have k >= vocab_size (skip radix select),
      others have k < vocab_size (use radix select)

    The bug was that k >= vocab_size iterations would skip radix select
    without clearing the histogram buffers, leaving stale data that corrupted
    subsequent k < vocab_size iterations.
    """
    batch_size = 1024  # Large batch to trigger persistent loop
    vocab_size = 128 * 1024  # Large vocab to trigger multi-CTA mode

    torch.manual_seed(42)
    generator = torch.Generator(device="cuda:0").manual_seed(42)

    # Generate random logits
    logits = torch.rand((batch_size, vocab_size), device="cuda:0", generator=generator)

    # Generate k values: mix of small k and k == vocab_size
    generator = torch.Generator(device="cuda:0").manual_seed(42)
    k_values = torch.randint(
        1, 1000, (batch_size,), device="cuda:0", generator=generator
    )

    # Randomly set some rows to k == vocab_size (about 50%)
    generator = torch.Generator(device="cuda:0").manual_seed(42)
    mask = torch.randint(
        0, 2, (batch_size,), generator=generator, dtype=torch.bool, device="cuda:0"
    )
    k_values.masked_fill_(mask, vocab_size)

    # Convert to probs
    probs = torch.softmax(logits, dim=-1).to(dtype)

    # Run FlashInfer top_k_renorm_probs
    renorm_probs = flashinfer.sampling.top_k_renorm_probs(probs, k_values)

    # Verify output dtype
    assert renorm_probs.dtype == dtype

    # Verify sum to 1
    sums = renorm_probs.float().sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-2, atol=1e-2)

    # Verify non-zero count matches k for each row
    nonzero_counts = (renorm_probs > 0).sum(dim=-1)

    # For rows with k >= vocab_size, all elements should be non-zero
    # For rows with k < vocab_size, non-zero count should be >= k (may be more due to ties)
    for i in range(batch_size):
        k = k_values[i].item()
        count = nonzero_counts[i].item()

        if k >= vocab_size:
            # All elements should be non-zero
            assert count == vocab_size, (
                f"Row {i}: k >= vocab_size but count={count} != {vocab_size}"
            )
        else:
            # Count should be at least k (may be more due to ties at the threshold)
            row_probs = probs[i].float()
            topk_vals, _ = torch.topk(row_probs, k, sorted=True)
            threshold = topk_vals[-1]
            expected_ge_threshold = (row_probs >= threshold).sum().item()

            # Allow small tolerance for floating point
            assert count >= k - 1, f"Row {i}: k={k} but only {count} non-zero elements"
            assert count <= expected_ge_threshold + 1, (
                f"Row {i}: k={k}, expected at most {expected_ge_threshold} but got {count}"
            )


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
@pytest.mark.parametrize(
    "distribution",
    [
        normal_distribution(1),
        normal_distribution(5),
        gumbel_distribution(0.1),
    ],
)
@pytest.mark.parametrize("neginf_input", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_top_k_mask_logits(
    batch_size, vocab_size, k, distribution, neginf_input, dtype
):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")

    torch.manual_seed(42)
    logits = distribution((batch_size, vocab_size), "cuda:0")
    if neginf_input:
        num_neginf = torch.randint(1, vocab_size * batch_size, (1,)).item()
        idxs = torch.randperm(batch_size * vocab_size, device="cuda:0")[:num_neginf]
        logits[idxs // vocab_size, idxs % vocab_size] = -float("inf")

    logits = logits.to(dtype)
    masked_logits = flashinfer.sampling.top_k_mask_logits(logits, k)

    # Check output dtype matches input
    assert masked_logits.dtype == dtype

    # Check that softmax of masked logits sums to 1
    probs = torch.softmax(masked_logits.float(), dim=-1)
    sums = probs.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-3, atol=1e-3)

    # Count finite elements in output
    finite_counts = torch.isfinite(masked_logits).sum(dim=-1)

    # Find the pivot value (k-th largest among finite values) and count ties
    # Replace -inf with a very small value for sorting
    logits_for_sort = logits.clone()
    logits_for_sort[~torch.isfinite(logits_for_sort)] = -float("inf")
    sorted_logits, _ = torch.sort(logits_for_sort, descending=True)

    # Count finite inputs per row
    finite_inputs = torch.isfinite(logits).sum(dim=-1)

    # For each row, find the pivot (k-th largest if enough finite values)
    effective_k = torch.minimum(
        torch.full_like(finite_inputs, k, dtype=torch.int64), finite_inputs
    )

    # Get pivot for each row (handle case where effective_k might be 0)
    pivot = torch.zeros(batch_size, dtype=dtype, device=logits.device)
    for i in range(batch_size):
        ek = effective_k[i].item()
        if ek > 0:
            pivot[i] = sorted_logits[i, ek - 1]
        else:
            pivot[i] = float("-inf")

    # Count how many elements are strictly greater than pivot
    num_greater = (logits > pivot.unsqueeze(-1)).sum(dim=-1)
    # Count how many elements equal the pivot (ties) - only among finite values
    num_ties = ((logits == pivot.unsqueeze(-1)) & torch.isfinite(logits)).sum(dim=-1)

    # Valid range considering ties
    max_valid = num_greater + num_ties

    # Check: finite_counts should be >= effective_k (we keep at least k finite values)
    # and <= max_valid (we don't keep more than all elements >= pivot)
    # Allow small tolerance for floating point issues
    assert torch.all(finite_counts >= torch.clamp(effective_k - 1, min=0)), (
        f"Some rows have fewer finite elements than expected. "
        f"finite_counts min: {finite_counts.min()}, effective_k min: {effective_k.min()}"
    )
    assert torch.all(finite_counts <= max_valid + 1), (
        f"Some rows have more finite elements than allowed by ties. "
        f"finite_counts max: {finite_counts.max()}, max_valid max: {max_valid.max()}"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_top_k_mask_logits_mixed_k_persistent_loop(dtype):
    """Test top_k_mask_logits with mixed k values in persistent loop (multi-CTA mode).

    This test catches the same bug as test_top_k_renorm_probs_mixed_k_persistent_loop
    but for the mask_logits variant.
    """
    batch_size = 1024  # Large batch to trigger persistent loop
    vocab_size = 128 * 1024  # Large vocab to trigger multi-CTA mode

    torch.manual_seed(42)
    generator = torch.Generator(device="cuda:0").manual_seed(42)

    # Generate random logits
    logits = torch.rand((batch_size, vocab_size), device="cuda:0", generator=generator)
    logits = logits.to(dtype)

    # Generate k values: mix of small k and k == vocab_size
    generator = torch.Generator(device="cuda:0").manual_seed(42)
    k_values = torch.randint(
        1, 1000, (batch_size,), device="cuda:0", generator=generator
    )

    # Randomly set some rows to k == vocab_size (about 50%)
    generator = torch.Generator(device="cuda:0").manual_seed(42)
    mask = torch.randint(
        0, 2, (batch_size,), generator=generator, dtype=torch.bool, device="cuda:0"
    )
    k_values.masked_fill_(mask, vocab_size)

    # Run FlashInfer top_k_mask_logits
    masked_logits = flashinfer.sampling.top_k_mask_logits(logits, k_values)

    # Verify output dtype
    assert masked_logits.dtype == dtype

    # Verify finite count matches k for each row
    finite_counts = torch.isfinite(masked_logits).sum(dim=-1)

    for i in range(batch_size):
        k = k_values[i].item()
        count = finite_counts[i].item()

        if k >= vocab_size:
            # All elements should be finite
            assert count == vocab_size, (
                f"Row {i}: k >= vocab_size but finite count={count} != {vocab_size}"
            )
        else:
            # Count should be at least k (may be more due to ties at the threshold)
            row_logits = logits[i].float()
            topk_vals, _ = torch.topk(row_logits, k, sorted=True)
            threshold = topk_vals[-1]
            expected_ge_threshold = (row_logits >= threshold).sum().item()

            # Allow small tolerance for floating point
            assert count >= k - 1, f"Row {i}: k={k} but only {count} finite elements"
            assert count <= expected_ge_threshold + 1, (
                f"Row {i}: k={k}, expected at most {expected_ge_threshold} but got {count}"
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


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.05, 0.1, 0.2, 0.7, 1])
def test_tensor_validation_min_p(batch_size, vocab_size, p):
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    # 1: Float p works and returns samples of shape (batch_size,).
    samples = flashinfer.sampling.min_p_sampling_from_probs(normalized_prob, p)
    assert samples.shape == (batch_size,)

    # 2: 2D tensor raises error.
    with pytest.raises(
        ValueError, match=r"Expected a 1D tensor or scalar.*got a 2D tensor"
    ):
        flashinfer.sampling.min_p_sampling_from_probs(
            normalized_prob,
            torch.tensor(
                [[p] * vocab_size] * batch_size, dtype=torch.float32, device="cuda:0"
            ),
        )

    # 3: 0D tensor raises error.
    with pytest.raises(
        ValueError,
        match=r"Expected a 1D tensor of shape \(batch_size,\) or scalar.*got a 0-dimensional tensor",
    ):
        flashinfer.sampling.min_p_sampling_from_probs(
            normalized_prob, torch.tensor(p, dtype=torch.float32, device="cuda:0")
        )

    # 4: 1D tensor with a broken batch size raises error (only when batch_size > 1).
    if batch_size > 1:
        with pytest.raises(
            ValueError, match="Sampling parameter tensor batch size mismatch"
        ):
            flashinfer.sampling.min_p_sampling_from_probs(
                normalized_prob, torch.tensor([p], dtype=torch.float32, device="cuda:0")
            )

    # 5: 1D tensor with the correct batch size works.
    samples = flashinfer.sampling.min_p_sampling_from_probs(
        normalized_prob,
        torch.tensor([p] * batch_size, dtype=torch.float32, device="cuda:0"),
    )
    assert samples.shape == (batch_size,)


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_check_tensor_param_top_p(batch_size, vocab_size, p):
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    # 1: Float p has the same shape as probs.
    samples = flashinfer.sampling.top_p_renorm_probs(normalized_prob, p)
    assert samples.shape == normalized_prob.shape

    # 2: 2D tensor raises error.
    with pytest.raises(
        ValueError, match=r"Expected a 1D tensor or scalar.*got a 2D tensor"
    ):
        flashinfer.sampling.top_p_renorm_probs(
            normalized_prob,
            torch.tensor(
                [[p] * vocab_size] * batch_size, dtype=torch.int, device="cuda:0"
            ),
        )

    # 3: 0D tensor raises error.
    with pytest.raises(
        ValueError,
        match=r"Expected a 1D tensor of shape \(batch_size,\) or scalar.*got a 0-dimensional tensor",
    ):
        flashinfer.sampling.top_p_renorm_probs(
            normalized_prob, torch.tensor(p, dtype=torch.int, device="cuda:0")
        )

    # 4: 1D tensor with a broken batch size raises error (only when batch_size > 1).
    if batch_size > 1:
        with pytest.raises(ValueError, match="Sampling parameter.*batch size mismatch"):
            flashinfer.sampling.top_p_renorm_probs(
                normalized_prob, torch.tensor([p], dtype=torch.int, device="cuda:0")
            )

    # 5: 1D tensor with the correct batch size works.
    samples = flashinfer.sampling.top_p_renorm_probs(
        normalized_prob,
        torch.tensor([p] * batch_size, dtype=torch.int, device="cuda:0"),
    )
    assert samples.shape == normalized_prob.shape


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_check_tensor_param_top_k(batch_size, vocab_size, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    # 1: Scalar k has the same shape as probs.
    samples = flashinfer.sampling.top_k_renorm_probs(normalized_prob, k)
    assert samples.shape == normalized_prob.shape

    # 2: 2D tensor raises error.
    with pytest.raises(
        ValueError, match=r"Expected a 1D tensor or scalar.*got a 2D tensor"
    ):
        flashinfer.sampling.top_k_renorm_probs(
            normalized_prob,
            torch.tensor(
                [[k] * vocab_size] * batch_size, dtype=torch.int, device="cuda:0"
            ),
        )

    # 3: 0D tensor raises error.
    with pytest.raises(
        ValueError,
        match=r"Expected a 1D tensor of shape \(batch_size,\) or scalar.*got a 0-dimensional tensor",
    ):
        flashinfer.sampling.top_k_renorm_probs(
            normalized_prob, torch.tensor(k, dtype=torch.int, device="cuda:0")
        )

    # 4: 1D tensor with a wrong shape raises error (only when batch_size > 1).
    if batch_size > 1:
        with pytest.raises(ValueError, match="Sampling parameter.*batch size mismatch"):
            flashinfer.sampling.top_k_renorm_probs(
                normalized_prob, torch.tensor([k], dtype=torch.int, device="cuda:0")
            )

    # 5: 1D tensor with the correct batch size works.
    samples = flashinfer.sampling.top_k_renorm_probs(
        normalized_prob,
        torch.tensor([k] * batch_size, dtype=torch.int, device="cuda:0"),
    )
    assert samples.shape == normalized_prob.shape


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
def test_sampling_from_probs_seed_offset_reproducibility(batch_size, vocab_size):
    """Test that explicit seed/offset produces reproducible results."""
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    seed, offset = 12345, 0

    samples1 = flashinfer.sampling.sampling_from_probs(
        normalized_prob, seed=seed, offset=offset
    )
    samples2 = flashinfer.sampling.sampling_from_probs(
        normalized_prob, seed=seed, offset=offset
    )

    assert torch.all(samples1 == samples2), (
        "Same seed/offset should produce identical samples"
    )


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
def test_sampling_from_logits_seed_offset_reproducibility(batch_size, vocab_size):
    """Test that explicit seed/offset produces reproducible results."""
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda:0")

    seed, offset = 12345, 0

    samples1 = flashinfer.sampling.sampling_from_logits(
        logits, seed=seed, offset=offset
    )
    samples2 = flashinfer.sampling.sampling_from_logits(
        logits, seed=seed, offset=offset
    )

    assert torch.all(samples1 == samples2), (
        "Same seed/offset should produce identical samples"
    )


@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
def test_sampling_different_seed_offset_produces_different_results(vocab_size):
    """Test that different seed/offset values produce different samples."""
    torch.manual_seed(42)
    batch_size = 1000
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    samples_seed1 = flashinfer.sampling.sampling_from_probs(
        normalized_prob, seed=12345, offset=0
    )
    samples_seed2 = flashinfer.sampling.sampling_from_probs(
        normalized_prob, seed=67890, offset=0
    )

    samples_offset1 = flashinfer.sampling.sampling_from_probs(
        normalized_prob, seed=12345, offset=0
    )
    samples_offset2 = flashinfer.sampling.sampling_from_probs(
        normalized_prob, seed=12345, offset=1000
    )

    seed_match_rate = (samples_seed1 == samples_seed2).float().mean().item()
    offset_match_rate = (samples_offset1 == samples_offset2).float().mean().item()

    assert seed_match_rate < 1, (
        f"Different seeds should produce mostly different samples, "
        f"got {seed_match_rate:.2%} match rate"
    )
    assert offset_match_rate < 1, (
        f"Different offsets should produce mostly different samples, "
        f"got {offset_match_rate:.2%} match rate"
    )


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000])
@pytest.mark.parametrize(
    "sampling_type",
    ["from_probs", "from_logits", "top_p", "top_k", "min_p", "top_k_top_p"],
)
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
def test_int64_indices_sampling(batch_size, vocab_size, sampling_type, indices_dtype):
    """Test that all sampling functions work with int64 indices."""
    torch.manual_seed(42)

    logits = torch.randn(batch_size, vocab_size, device="cuda:0")
    probs = torch.softmax(logits, dim=-1)
    indices = torch.arange(batch_size, dtype=indices_dtype, device="cuda:0")

    if sampling_type == "from_probs":
        samples = flashinfer.sampling.sampling_from_probs(probs, indices=indices)
    elif sampling_type == "from_logits":
        samples = flashinfer.sampling.sampling_from_logits(logits, indices=indices)
    elif sampling_type == "top_p":
        samples = flashinfer.sampling.top_p_sampling_from_probs(
            probs, 0.9, indices=indices
        )
    elif sampling_type == "top_k":
        k = min(100, vocab_size)
        samples = flashinfer.sampling.top_k_sampling_from_probs(
            probs, k, indices=indices
        )
    elif sampling_type == "min_p":
        samples = flashinfer.sampling.min_p_sampling_from_probs(
            probs, 0.1, indices=indices
        )
    elif sampling_type == "top_k_top_p":
        k = min(100, vocab_size)
        samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs, k, 0.9, indices=indices, filter_apply_order="joint"
        )

    assert samples.dtype == indices_dtype, (
        f"Output dtype {samples.dtype} doesn't match indices dtype {indices_dtype}"
    )
    assert samples.shape == (batch_size,)
    assert torch.all(samples < vocab_size) and torch.all(samples >= 0)


@pytest.mark.parametrize("batch_size", [1, 19, 99])
@pytest.mark.parametrize("vocab_size", [111, 32000])
def test_sampling_with_default_device_cuda(batch_size, vocab_size):
    """Test that sampling works correctly when torch.set_default_device("cuda") is set.

    This is a regression test for issue #2333 where generator.set_state() would fail
    with "RNG state must be a torch.ByteTensor" error when the default device is CUDA.
    """
    torch.manual_seed(42)
    original_device = torch.get_default_device()
    try:
        # Set default device to CUDA
        torch.set_default_device("cuda")

        # Create logits and test top_k_top_p_sampling_from_logits
        logits = torch.randn(batch_size, vocab_size, device="cuda:0")

        # This should not raise "RNG state must be a torch.ByteTensor" error
        samples = flashinfer.sampling.top_k_top_p_sampling_from_logits(
            logits, top_k=100, top_p=0.9
        )

        assert samples.shape == (batch_size,)
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)

        # Also test other sampling functions
        probs = torch.softmax(logits, dim=-1)

        samples = flashinfer.sampling.sampling_from_probs(probs)
        assert samples.shape == (batch_size,)
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)

        samples = flashinfer.sampling.top_p_sampling_from_probs(probs, 0.9)
        assert samples.shape == (batch_size,)
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)

    finally:
        # Restore original default device
        torch.set_default_device(original_device)


@pytest.mark.parametrize("batch_size", [1, 4, 19])
@pytest.mark.parametrize("vocab_size", [111, 32000])
def test_sampling_nan_input(batch_size, vocab_size):
    torch.manual_seed(42)
    probs = torch.rand(batch_size, vocab_size, device="cuda:0", dtype=torch.float32)
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # Set NaN at different positions: first, middle, last
    nan_indices = [0]
    if batch_size > 1:
        nan_indices.append(batch_size // 2)
    if batch_size > 2:
        nan_indices.append(batch_size - 1)

    for idx in nan_indices:
        probs[idx, :] = float("nan")

    valid_indices = [i for i in range(batch_size) if i not in nan_indices]

    def check_result(result, valid):
        # NaN rows should return 0 and valid=False
        for idx in nan_indices:
            assert result[idx].item() == 0 and not valid[idx].item()
        # Non-NaN rows should have valid=True and valid token index
        for idx in valid_indices:
            assert valid[idx].item()
            assert 0 <= result[idx].item() < vocab_size

    # sampling_from_probs
    result, valid = flashinfer.sampling.sampling_from_probs(probs, return_valid=True)
    check_result(result, valid)

    # top_k_sampling_from_probs
    result, valid = flashinfer.sampling.top_k_sampling_from_probs(
        probs, top_k=50, return_valid=True
    )
    check_result(result, valid)

    # top_p_sampling_from_probs
    result, valid = flashinfer.sampling.top_p_sampling_from_probs(
        probs, top_p=0.9, return_valid=True
    )
    check_result(result, valid)

    # min_p_sampling_from_probs
    result, valid = flashinfer.sampling.min_p_sampling_from_probs(
        probs, min_p=0.1, return_valid=True
    )
    check_result(result, valid)

    # top_k_top_p_sampling_from_probs (joint mode)
    result, valid = flashinfer.sampling.top_k_top_p_sampling_from_probs(
        probs, top_k=50, top_p=0.9, filter_apply_order="joint", return_valid=True
    )
    check_result(result, valid)


def test_fused_sampling_from_logits_hy3_portable_temperature():
    """The HY3 API must retain semantics when the SM100 backend is unavailable."""
    logits = torch.tensor([[0.0, 3.0, 1.0], [4.0, 1.0, 0.0]])
    noise = torch.zeros_like(logits, dtype=torch.float32)
    draft = torch.tensor([1, -1], dtype=torch.int64)
    output = flashinfer.sampling.fused_sampling_from_logits_hy3(
        logits,
        temperature=torch.tensor([1.0, 2.0], dtype=torch.float32),
        gumbel_noise=noise,
        draft_token_ids=draft,
    )
    assert torch.equal(output, torch.tensor([[2], [0]], dtype=torch.int32))


def test_fused_sampling_from_logits_hy3_portable_non_positive_temperature():
    """Non-positive per-row temperatures disable scaling without division by zero."""
    logits = torch.full((3, 16), -10.0)
    logits[:, 9] = 2.0
    logits[:, 3] = 1.0
    output = flashinfer.sampling.fused_sampling_from_logits_hy3(
        logits,
        temperature=torch.tensor([0.0, -1.0, 2.0]),
        gumbel_noise=torch.zeros_like(logits),
    )
    assert torch.equal(output, torch.tensor([[9], [9], [9]], dtype=torch.int32))


def test_fused_sampling_from_logits_hy3_generated_seed_uses_uint64_bits(monkeypatch):
    """A signed CUDA generator-state view must retain its uint64 seed bits."""
    unsigned_seed = (1 << 63) + 123
    signed_seed = unsigned_seed - (1 << 64)
    generated_offset = 64
    monkeypatch.setattr(
        flashinfer.sampling,
        "get_seed_and_offset",
        lambda increment, generator, device: (signed_seed, generated_offset),
    )
    logits = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])

    generated = flashinfer.sampling.fused_sampling_from_logits_hy3(
        logits,
        temperature=1.0,
    )
    explicit = flashinfer.sampling.fused_sampling_from_logits_hy3(
        logits,
        temperature=1.0,
        seed=unsigned_seed,
        offset=generated_offset,
    )

    assert torch.equal(generated, explicit)
    for invalid_seed in (0, -1):
        with pytest.raises(ValueError, match="seed must be > 0"):
            flashinfer.sampling.fused_sampling_from_logits_hy3(
                logits,
                temperature=1.0,
                seed=invalid_seed,
            )
    with pytest.raises(ValueError, match=r"seed must be less than 2\*\*64"):
        flashinfer.sampling.fused_sampling_from_logits_hy3(
            logits,
            temperature=1.0,
            seed=1 << 64,
        )


def test_fused_sampling_from_logits_hy3_uses_signed_ffi_seed_carrier(monkeypatch):
    """The FFI receives the signed carrier for a high-bit uint64 seed."""
    unsigned_seed = (1 << 63) + 123
    signed_seed = unsigned_seed - (1 << 64)
    captured = {}

    def capture_fused_sampler(*args):
        """Capture the signed seed passed to the custom-op boundary."""
        captured["seed"] = args[-3]

    monkeypatch.setattr(
        flashinfer.sampling,
        "_hy3_sampler_device_info",
        lambda device: (True, 1),
    )
    monkeypatch.setattr(
        flashinfer.sampling,
        "_fused_sampling_from_logits_hy3",
        capture_fused_sampler,
    )
    logits = torch.zeros((1, 120832))

    flashinfer.sampling.fused_sampling_from_logits_hy3(
        logits,
        workspace_buffer=torch.empty(1 << 20, dtype=torch.uint8),
        out=torch.empty((1, 1), dtype=torch.int32),
        temperature=1.0,
        seed=unsigned_seed,
        offset=0,
    )

    assert captured["seed"] == signed_seed


def test_fused_sampling_from_logits_hy3_remaps_generated_zero_seed(monkeypatch):
    """A generated zero is remapped without accepting caller-supplied zero."""
    generated_offset = 64
    monkeypatch.setattr(
        flashinfer.sampling,
        "get_seed_and_offset",
        lambda increment, generator, device: (0, generated_offset),
    )
    logits = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])

    generated = flashinfer.sampling.fused_sampling_from_logits_hy3(
        logits,
        temperature=1.0,
    )
    explicit = flashinfer.sampling.fused_sampling_from_logits_hy3(
        logits,
        temperature=1.0,
        seed=1,
        offset=generated_offset,
    )

    assert torch.equal(generated, explicit)


def test_fused_sampling_from_logits_hy3_portable_batched_penalty_writeback():
    """Fallback writeback preserves atomic-OR semantics for duplicate slots."""
    batch_size, vocab_size = 7, 16
    winners = torch.tensor([1, 2, 2, 9, 5, 6, 7])
    logits = torch.full((batch_size, vocab_size), -10.0)
    logits[torch.arange(batch_size), winners] = 10.0
    penalty_mask = torch.zeros((batch_size, 4), dtype=torch.uint8)
    penalty_mask[0, 0] = 1
    slot_id = torch.tensor([0, 0, 0, 1, -1, batch_size, 2], dtype=torch.int32)
    repetition_penalty = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])

    output = flashinfer.sampling.fused_sampling_from_logits_hy3(
        logits,
        penalty_mask=penalty_mask,
        slot_id=slot_id,
        repetition_penalty=repetition_penalty,
        temperature=1.0,
        gumbel_noise=torch.zeros_like(logits),
    )

    assert torch.equal(output[:, 0], winners.to(torch.int32))
    assert penalty_mask[0, 0].item() == 1 | (1 << 1) | (1 << 2)
    assert penalty_mask[1, 1].item() == 1 << (9 & 7)
    assert torch.count_nonzero(penalty_mask[2:]).item() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_sampling_from_logits_hy3_temperature(dtype):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 0):
        pytest.skip("requires an SM100 GPU")
    device = torch.device("cuda:0")
    batch_size, vocab_size = 2, 120832
    # Exercise a valid padded row stride as well as the deterministic external
    # Gumbel boundary used by the source-parity suite.
    storage = torch.full(
        (batch_size, vocab_size + 1), -10.0, dtype=dtype, device=device
    )
    logits = storage[:, :vocab_size]
    logits[0, 17] = 8
    logits[0, 5] = 7
    logits[1, 31] = 7
    logits[1, 6] = 6
    noise = torch.zeros((batch_size, vocab_size), dtype=torch.float32, device=device)
    workspace = torch.empty(1 << 20, dtype=torch.uint8, device=device)
    preallocated_output = torch.empty((batch_size, 1), dtype=torch.int32, device=device)
    output = flashinfer.sampling.fused_sampling_from_logits_hy3(
        logits,
        workspace_buffer=workspace,
        out=preallocated_output,
        temperature=0.75,
        gumbel_noise=noise,
    )
    assert output.data_ptr() == preallocated_output.data_ptr()
    assert torch.equal(
        output, torch.tensor([[17], [31]], dtype=torch.int32, device=device)
    )

    required = flashinfer.sampling._hy3_sampler_workspace_size(
        batch_size,
        torch.cuda.get_device_properties(device).multi_processor_count,
        True,
        flashinfer.sampling.HY3_SAMPLER_SOFTMAX_NONE,
    )
    exact_workspace = torch.empty(required, dtype=torch.uint8, device=device)
    exact = flashinfer.sampling.fused_sampling_from_logits_hy3(
        logits,
        workspace_buffer=exact_workspace,
        out=preallocated_output,
        temperature=0.75,
        gumbel_noise=noise,
    )
    assert torch.equal(exact, output)

    non_positive = flashinfer.sampling.fused_sampling_from_logits_hy3(
        logits,
        workspace_buffer=exact_workspace,
        out=preallocated_output,
        temperature=torch.tensor([0.0, -1.0], dtype=torch.float32, device=device),
        gumbel_noise=noise,
    )
    assert torch.equal(
        non_positive,
        torch.tensor([[17], [31]], dtype=torch.int32, device=device),
    )
    with pytest.raises(ValueError, match="workspace_buffer is too small"):
        flashinfer.sampling.fused_sampling_from_logits_hy3(
            logits,
            workspace_buffer=exact_workspace[:-1],
            out=preallocated_output,
            temperature=0.75,
            gumbel_noise=noise,
        )
    unaligned_storage = torch.empty(required + 1, dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match="aligned to four bytes"):
        flashinfer.sampling.fused_sampling_from_logits_hy3(
            logits,
            workspace_buffer=unaligned_storage[1:],
            out=preallocated_output,
            temperature=0.75,
            gumbel_noise=noise,
        )


def test_fused_sampling_from_logits_hy3_heavy_penalty_writeback():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 0):
        pytest.skip("requires an SM100 GPU")
    device = torch.device("cuda:0")
    batch_size, vocab_size = 2, 120832
    logits = torch.full(
        (batch_size, vocab_size), -10.0, dtype=torch.bfloat16, device=device
    )
    logits[0, 11], logits[0, 12] = 8, 5
    logits[1, 29], logits[1, 30] = 8, 5
    row_bytes = (vocab_size + 7) // 8
    penalty_mask = torch.zeros((2, row_bytes), dtype=torch.uint8, device=device)
    slot_id = torch.tensor([1, 0], dtype=torch.int32, device=device)
    penalty_mask[1, 11 >> 3] |= 1 << (11 & 7)
    penalty_mask[0, 29 >> 3] |= 1 << (29 & 7)
    noise = torch.zeros_like(logits, dtype=torch.float32)
    output = flashinfer.sampling.fused_sampling_from_logits_hy3(
        logits,
        penalty_mask=penalty_mask,
        slot_id=slot_id,
        repetition_penalty=2.0,
        temperature=1.0,
        softmax_policy=flashinfer.sampling.HY3_SAMPLER_SOFTMAX_AFTER_TOP_K,
        top_k=20,
        top_p=0.99,
        max_top_k=32,
        gumbel_noise=noise,
    )
    assert torch.equal(
        output, torch.tensor([[12], [30]], dtype=torch.int32, device=device)
    )
    assert (penalty_mask[1, 12 >> 3] & (1 << (12 & 7))) != 0
    assert (penalty_mask[0, 30 >> 3] & (1 << (30 & 7))) != 0


def _hy3_test_device():
    if not torch.cuda.is_available():
        pytest.skip("requires an SM100 GPU")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) != (10, 0):
        pytest.skip("requires an SM100 GPU")
    return device


def _reference_hy3_fused_sampler_candidates(
    logits,
    penalty_mask,
    slot_id,
    repetition_penalty,
    temperature,
    softmax_policy,
    top_k,
    max_top_k,
):
    """Build the sorted candidate state without calling a FlashInfer sampler."""
    _batch_size, vocab_size = logits.shape
    work = logits.float().clone()
    columns = torch.arange(vocab_size, device=logits.device)
    packed_mask = penalty_mask.index_select(0, slot_id.long())
    penalized = ((packed_mask[:, columns >> 3] >> (columns & 7)) & 1).bool()
    active_penalty = penalized & (repetition_penalty > 0)[:, None]
    safe_penalty = torch.where(
        repetition_penalty > 0,
        repetition_penalty,
        torch.ones_like(repetition_penalty),
    )
    adjusted = torch.where(
        work > 0,
        work / safe_penalty[:, None],
        work * safe_penalty[:, None],
    )
    work = torch.where(active_penalty, adjusted, work)
    work = work / temperature[:, None]

    if softmax_policy == flashinfer.sampling.HY3_SAMPLER_SOFTMAX_BEFORE_TOP_K:
        work = torch.softmax(work, dim=-1)

    values, tokens = torch.topk(work, max_top_k, dim=-1, sorted=True)
    requested_k = top_k.long()
    effective_k = torch.where(
        requested_k > 0,
        requested_k.clamp(max=max_top_k),
        torch.full_like(requested_k, max_top_k),
    )
    positions = torch.arange(max_top_k, device=logits.device)[None, :]
    candidate_valid = positions < effective_k[:, None]

    if softmax_policy == flashinfer.sampling.HY3_SAMPLER_SOFTMAX_AFTER_TOP_K:
        probabilities = torch.softmax(
            values.masked_fill(~candidate_valid, float("-inf")), dim=-1
        )
    else:
        probabilities = values
    sample_values = torch.where(
        probabilities > 0,
        probabilities.log(),
        torch.full_like(probabilities, float("-inf")),
    )
    return tokens, probabilities, sample_values, candidate_valid


def _reference_hy3_fused_sampler_pick(
    tokens, probabilities, sample_values, candidate_valid, top_p, gumbel_noise
):
    """Apply the fused sampler's exclusive-prefix Top-P and token tie break."""
    positions = torch.arange(tokens.size(1), device=tokens.device)[None, :]
    exclusive_probability = probabilities.cumsum(dim=-1) - probabilities
    keep = candidate_valid & (
        (top_p <= 0)[:, None]
        | (positions == 0)
        | (exclusive_probability < top_p[:, None])
    )
    scores = sample_values + gumbel_noise.gather(1, tokens)
    scores = scores.masked_fill(~keep, float("-inf"))
    maximum = scores.max(dim=-1, keepdim=True).values
    vocab_size = gumbel_noise.size(1)
    tied_tokens = torch.where(
        scores == maximum, tokens, torch.full_like(tokens, vocab_size)
    )
    output = tied_tokens.min(dim=-1).values
    output = torch.where(output < vocab_size, output, torch.zeros_like(output))
    return output.to(torch.int32).view(-1, 1), keep


@pytest.mark.parametrize(
    "softmax_policy",
    [
        flashinfer.sampling.HY3_SAMPLER_SOFTMAX_BEFORE_TOP_K,
        flashinfer.sampling.HY3_SAMPLER_SOFTMAX_AFTER_TOP_K,
    ],
)
@pytest.mark.parametrize("batch_size", [2, 8, 32])
@pytest.mark.parametrize("max_top_k", [32, 64])
def test_fused_sampling_from_logits_hy3_top_p_boundary_dispatch(
    max_top_k, batch_size, softmax_policy
):
    """Check both heavy dispatch shapes against a self-contained torch oracle."""
    device = _hy3_test_device()
    vocab_size = 120832
    candidate_tokens = torch.arange(64, device=device, dtype=torch.long) * 1733 + 29
    candidate_logits = torch.linspace(4.0, 0.0, 64, device=device)
    logits = torch.full(
        (batch_size, vocab_size), -4.0, dtype=torch.float32, device=device
    )
    logits[:, candidate_tokens] = candidate_logits

    row_pattern = torch.arange(batch_size, device=device) % 8
    top_k_values = torch.tensor(
        [64, 64, 2, 7, 16, 31, 48, 64], dtype=torch.int64, device=device
    )
    temperature_values = torch.tensor(
        [0.75, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75],
        dtype=torch.float32,
        device=device,
    )
    penalty_values = torch.tensor(
        [0.0, 1.0, 1.17, 1.43, 1.71, 1.91, 1.29, 2.13],
        dtype=torch.float32,
        device=device,
    )
    top_k = top_k_values.index_select(0, row_pattern)
    temperature = temperature_values.index_select(0, row_pattern)
    repetition_penalty = penalty_values.index_select(0, row_pattern)
    slot_id = torch.arange(batch_size - 1, -1, -1, dtype=torch.int32, device=device)

    penalty_row_bytes = (vocab_size + 7) // 8
    initial_penalty_mask = torch.zeros(
        (batch_size, penalty_row_bytes), dtype=torch.uint8
    )
    candidate_tokens_cpu = candidate_tokens.cpu()
    for row in range(batch_size):
        token = int(candidate_tokens_cpu[(row * 7) % 32])
        slot = batch_size - 1 - row
        initial_penalty_mask[slot, token >> 3] |= 1 << (token & 7)
    initial_penalty_mask = initial_penalty_mask.to(device)

    tokens, probabilities, sample_values, candidate_valid = (
        _reference_hy3_fused_sampler_candidates(
            logits,
            initial_penalty_mask,
            slot_id,
            repetition_penalty,
            temperature,
            softmax_policy,
            top_k,
            max_top_k,
        )
    )

    # Row 0 exercises top_p == 0 (filter disabled). Other rows put top_p in
    # the middle of one candidate's probability mass, avoiding numerical
    # ambiguity while checking the exclusive-prefix cutoff on both sides.
    desired_keep_pattern = [64, 1, 2, 3, 8, 16, 24, 47]
    top_p = torch.empty(batch_size, dtype=torch.float32, device=device)
    expected_keep_counts = []
    for row in range(batch_size):
        effective_k = min(int(top_k[row]), max_top_k)
        if row % 8 == 0:
            top_p[row] = 0.0
            expected_keep_counts.append(effective_k)
            continue
        keep_count = min(desired_keep_pattern[row % 8], effective_k)
        prefix = probabilities[row, : max(keep_count - 1, 0)].sum()
        top_p[row] = prefix + probabilities[row, keep_count - 1] * 0.5
        expected_keep_counts.append(keep_count)

    gumbel_noise = torch.full(
        (batch_size, vocab_size), -1000.0, dtype=torch.float32, device=device
    )
    gumbel_noise[:, candidate_tokens] = (
        torch.arange(64, dtype=torch.float32, device=device) * 0.5
    )
    expected, reference_keep = _reference_hy3_fused_sampler_pick(
        tokens,
        probabilities,
        sample_values,
        candidate_valid,
        top_p,
        gumbel_noise,
    )
    assert reference_keep.sum(dim=-1).cpu().tolist() == expected_keep_counts

    penalty_mask = initial_penalty_mask.clone()
    workspace = torch.empty(1 << 20, dtype=torch.uint8, device=device)
    preallocated_output = torch.full(
        (batch_size, 1), -1, dtype=torch.int32, device=device
    )
    output = flashinfer.sampling.fused_sampling_from_logits_hy3(
        logits,
        workspace_buffer=workspace,
        out=preallocated_output,
        penalty_mask=penalty_mask,
        slot_id=slot_id,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        softmax_policy=softmax_policy,
        top_k=top_k,
        top_p=top_p,
        max_top_k=max_top_k,
        gumbel_noise=gumbel_noise,
    )
    assert output.data_ptr() == preallocated_output.data_ptr()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)

    output_cpu = output.cpu()
    penalty_mask_cpu = penalty_mask.cpu()
    initial_penalty_mask_cpu = initial_penalty_mask.cpu()
    repetition_penalty_cpu = repetition_penalty.cpu()
    for row in range(batch_size):
        slot = batch_size - 1 - row
        if float(repetition_penalty_cpu[row]) == 0.0:
            assert torch.equal(penalty_mask_cpu[slot], initial_penalty_mask_cpu[slot])
            continue
        sampled = int(output_cpu[row, 0])
        assert penalty_mask_cpu[slot, sampled >> 3] & (1 << (sampled & 7))


@pytest.mark.parametrize(
    "softmax_policy",
    [
        flashinfer.sampling.HY3_SAMPLER_SOFTMAX_NONE,
        flashinfer.sampling.HY3_SAMPLER_SOFTMAX_BEFORE_TOP_K,
        flashinfer.sampling.HY3_SAMPLER_SOFTMAX_AFTER_TOP_K,
    ],
)
def test_fused_sampling_from_logits_hy3_partition_tie_order(softmax_policy):
    """Preserve the source kernel's stable Top-K order across block partitions."""
    device = _hy3_test_device()
    batch_size, vocab_size = 32, 120832
    logits = torch.full(
        (batch_size, vocab_size),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    early_partition = torch.arange(992, 1024, device=device)
    late_partition = torch.arange(15872, 15904, device=device)
    logits[:, early_partition] = 1.0
    logits[:, late_partition] = 1.0

    # The late-partition tokens have higher sample scores, but the source
    # 512-thread/8-block stage-1 partition excludes them at the equal-logit
    # Top-K cutoff.  A geometry change must not silently change that tie set.
    noise = torch.full_like(logits, -1000.0, dtype=torch.float32)
    noise[:, early_partition] = torch.arange(32, dtype=torch.float32, device=device)
    noise[:, late_partition] = 1000.0 + torch.arange(
        32, dtype=torch.float32, device=device
    )
    output = flashinfer.sampling.fused_sampling_from_logits_hy3(
        logits,
        temperature=1.0,
        softmax_policy=softmax_policy,
        top_k=32,
        top_p=0.99 if softmax_policy else 0.0,
        max_top_k=32,
        gumbel_noise=noise,
    )
    expected = torch.full((batch_size, 1), 1023, dtype=torch.int32, device=device)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_fused_sampling_from_logits_hy3_non_default_stream():
    """The public API must honor the current stream with caller-owned buffers."""
    device = _hy3_test_device()
    batch_size, vocab_size = 8, 120832
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        logits = torch.full(
            (batch_size, vocab_size), -10.0, dtype=torch.float32, device=device
        )
        winners = torch.arange(batch_size, dtype=torch.long, device=device) * 997 + 13
        logits[torch.arange(batch_size, device=device), winners] = 10.0
        noise = torch.zeros_like(logits, dtype=torch.float32)
        workspace = torch.empty(1 << 20, dtype=torch.uint8, device=device)
        preallocated_output = torch.full(
            (batch_size, 1), -1, dtype=torch.int32, device=device
        )
        output = flashinfer.sampling.fused_sampling_from_logits_hy3(
            logits,
            workspace_buffer=workspace,
            out=preallocated_output,
            temperature=torch.linspace(
                0.5, 1.5, batch_size, dtype=torch.float32, device=device
            ),
            gumbel_noise=noise,
        )
        completed = torch.cuda.Event()
        completed.record(stream)
    completed.synchronize()

    assert output.data_ptr() == preallocated_output.data_ptr()
    torch.testing.assert_close(output[:, 0], winners.to(torch.int32), rtol=0, atol=0)


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
