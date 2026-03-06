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
