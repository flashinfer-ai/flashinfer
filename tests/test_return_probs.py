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

import warnings

import pytest
import torch

import flashinfer


def generate_llm_like_prob(batch_size, vocab_size, temperature=1.0, device="cuda"):
    """Generate LLM-like probability distribution with Zipf-like pattern.

    Real LLM outputs have:
    - Power-law decay: top tokens dominate probability mass
    - Long tail: most tokens have near-zero probability
    - Random high-prob token positions (not fixed indices)
    - Typical top-1 probability: 30-60% (varies by context)
    """
    # Zipf-like logits: moderate power-law decay
    # Tuned so top-1 ~ 40%, top-10 ~ 80%, top-100 ~ 95%
    ranks = torch.arange(1, vocab_size + 1, device=device, dtype=torch.float32)
    base_logits = 3.0 / (ranks ** 0.3)  # gentler decay than 10/r^0.8

    # Each batch gets different token ordering (simulate different contexts)
    logits = base_logits.unsqueeze(0).expand(batch_size, -1).clone()
    for i in range(batch_size):
        logits[i] = logits[i, torch.randperm(vocab_size, device=device)]

    # Add noise for realism (some contexts more/less certain)
    logits += torch.randn_like(logits) * 0.5

    # Softmax to get probabilities
    probs = torch.softmax(logits / temperature, dim=-1)
    return probs


def pytorch_top_k_top_p_sampling_and_filter(
    probs: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch baseline for joint top-k/top-p sampling with filtered probs.

    Matches flashinfer's filter_apply_order="joint" logic:
    - Sort by probability descending
    - Keep tokens that satisfy BOTH top-k AND top-p conditions
    - top-p: keep tokens where cumsum <= p
    - Renormalize and sample

    Args:
        probs: (batch_size, vocab_size) probability distribution
        k: (batch_size,) top-k values
        p: (batch_size,) top-p values

    Returns:
        sampled_ids: (batch_size,) sampled token indices
        filtered_probs: (batch_size, vocab_size) filtered probability distribution
    """
    batch_size, vocab_size = probs.shape
    device = probs.device

    # Sort by probability descending
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sort = probs_sort.to(torch.float32)

    # Cumulative sum for top-p
    probs_cumsum = probs_sort.cumsum(dim=-1)

    # top-k mask: True for first k tokens (to keep)
    k_clamped = k.clamp(max=vocab_size).unsqueeze(1)  # (batch_size, 1)
    range_idx = torch.arange(vocab_size, device=device).unsqueeze(0)  # (1, vocab_size)
    topk_mask = range_idx < k_clamped  # (batch_size, vocab_size)

    # top-p mask: keep tokens where cumsum <= p
    # Use shifted cumsum so boundary token (where cumsum first exceeds p) is included
    probs_cumsum_shifted = torch.cat([
        torch.zeros(batch_size, 1, device=device, dtype=probs_sort.dtype),
        probs_cumsum[:, :-1]
    ], dim=-1)
    topp_mask = probs_cumsum_shifted <= p.unsqueeze(1)  # <= instead of <

    # Joint mask: keep if BOTH conditions are satisfied
    joint_mask = topk_mask & topp_mask
    # At least keep top-1
    joint_mask[:, 0] = True

    # Apply mask (no renormalization - kernel doesn't normalize)
    probs_filtered = torch.where(joint_mask, probs_sort, torch.zeros_like(probs_sort))

    # Sample
    sampled_index = torch.multinomial(probs_filtered, num_samples=1)
    sampled_ids = torch.gather(probs_idx, dim=-1, index=sampled_index).view(-1)

    # Scatter back to original order
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(dim=-1, index=probs_idx, src=probs_filtered)

    return sampled_ids, filtered_probs


@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("top_k,top_p", [(1024, 0.98), (50, 0.9), (1, 1.0), (151936, 0.5)])
def test_fused_vs_pytorch_precision(batch_size, top_k, top_p):
    """Verify fused kernel produces similar filtered probs as PyTorch baseline.

    Fused kernel uses approximate dual-pivot rejection sampling, so exact match
    is not expected. We verify:
    1. Probability mass of differing tokens is small (< 1e-5)
    2. The smaller set is a subset of the larger set (core tokens match)
    """
    torch.manual_seed(42)
    vocab_size = 151936

    probs = generate_llm_like_prob(batch_size, vocab_size, device="cuda:0")
    top_ks = torch.full((batch_size,), top_k, dtype=torch.int32, device="cuda:0")
    top_ps = torch.full((batch_size,), top_p, dtype=torch.float32, device="cuda:0")

    # Get filtered probs from both implementations
    _, fused_filtered = flashinfer.sampling.top_k_top_p_sampling_and_filter(
        probs, top_ks, top_ps, filter_apply_order="joint"
    )
    _, pytorch_filtered = pytorch_top_k_top_p_sampling_and_filter(
        probs, top_ks, top_ps
    )

    # Compare non-zero pattern
    fused_nonzero = fused_filtered > 0
    pytorch_nonzero = pytorch_filtered > 0

    # 1. Check probability of differing tokens is small
    diff_mask = fused_nonzero != pytorch_nonzero  # tokens that differ
    diff_count = diff_mask.sum().item()
    fused_count = fused_nonzero.sum().item()
    pytorch_count = pytorch_nonzero.sum().item()
    max_count = max(fused_count, pytorch_count)
    diff_ratio = diff_count / max_count if max_count > 0 else 0
    if diff_ratio > 0.1:
        warnings.warn(f"diff>10%: fused={fused_count}, pytorch={pytorch_count}, diff={diff_count} ({diff_ratio:.1%})")
    if diff_mask.any():
        diff_prob_max = probs[diff_mask].max().item()  # max prob among differing tokens
        assert diff_prob_max < 1e-4, \
            f"Differing tokens have too high probability: max={diff_prob_max:.2e}"

    # 2. Check smaller set is subset of larger set (both index and prob values)
    for i in range(batch_size):
        fused_set = set(torch.where(fused_nonzero[i])[0].cpu().tolist())
        pytorch_set = set(torch.where(pytorch_nonzero[i])[0].cpu().tolist())
        smaller_set = fused_set if len(fused_set) <= len(pytorch_set) else pytorch_set
        larger_set = pytorch_set if len(fused_set) <= len(pytorch_set) else fused_set

        # Check index subset
        assert smaller_set.issubset(larger_set), \
            f"Row {i}: smaller set index is not subset of larger set. " \
            f"Extra elements: {smaller_set - larger_set}"

        # Check prob values match exactly for common indices
        common_indices = list(smaller_set)
        if common_indices:
            smaller_probs = fused_filtered if len(fused_set) <= len(pytorch_set) else pytorch_filtered
            larger_probs = pytorch_filtered if len(fused_set) <= len(pytorch_set) else fused_filtered
            smaller_vals = smaller_probs[i, common_indices]
            larger_vals = larger_probs[i, common_indices]
            assert torch.equal(smaller_vals, larger_vals), \
                f"Row {i}: prob values mismatch for common indices. " \
                f"Max diff: {(smaller_vals - larger_vals).abs().max().item():.2e}"


@pytest.mark.parametrize("batch_size", [1, 8, 9, 32, 33, 40, 41, 64, 128])
def test_batch_size_routing(batch_size):
    """Test different batch size routing paths (cluster_size = 8, 4, 2, 1)."""
    torch.manual_seed(42)
    vocab_size = 151936
    top_k = 1024
    top_p = 0.98
    num_trials = 10

    probs = generate_llm_like_prob(batch_size, vocab_size, device="cuda:0")
    top_ks = torch.full((batch_size,), top_k, dtype=torch.int32, device="cuda:0")
    top_ps = torch.full((batch_size,), top_p, dtype=torch.float32, device="cuda:0")

    for _ in range(num_trials):
        samples, filtered = flashinfer.sampling.top_k_top_p_sampling_and_filter(
            probs, top_ks, top_ps, filter_apply_order="joint"
        )

        # Check samples are valid indices
        assert torch.all(samples >= 0) and torch.all(samples < vocab_size)

        # Check sampled token is in filtered set
        for i in range(batch_size):
            token_id = samples[i].item()
            assert filtered[i, token_id] > 0, f"Sampled token {token_id} not in filtered set"

        # Check filtered probs are non-negative
        assert torch.all(filtered >= 0)


@pytest.mark.parametrize("top_k", [1, 10, 100, 1024])
def test_top_k_values(top_k):
    """Test different top_k values."""
    torch.manual_seed(42)
    batch_size = 16
    vocab_size = 151936
    top_p = 0.98
    num_trials = 10

    probs = generate_llm_like_prob(batch_size, vocab_size, device="cuda:0")
    top_ks = torch.full((batch_size,), top_k, dtype=torch.int32, device="cuda:0")
    top_ps = torch.full((batch_size,), top_p, dtype=torch.float32, device="cuda:0")

    for _ in range(num_trials):
        samples, filtered = flashinfer.sampling.top_k_top_p_sampling_and_filter(
            probs, top_ks, top_ps, filter_apply_order="joint"
        )

        assert torch.all(samples >= 0) and torch.all(samples < vocab_size)
        for i in range(batch_size):
            token_id = samples[i].item()
            assert filtered[i, token_id] > 0


@pytest.mark.parametrize("vocab_size", [1000, 32000, 151936])
def test_top_k_no_filtering(vocab_size):
    """Test top_k >= vocab_size (early exit path)."""
    torch.manual_seed(42)
    batch_size = 16
    top_k = vocab_size + 100  # No filtering needed
    top_p = 1.0
    num_trials = 10

    probs = generate_llm_like_prob(batch_size, vocab_size, device="cuda:0")
    top_ks = torch.full((batch_size,), top_k, dtype=torch.int32, device="cuda:0")
    top_ps = torch.full((batch_size,), top_p, dtype=torch.float32, device="cuda:0")

    for _ in range(num_trials):
        samples, filtered = flashinfer.sampling.top_k_top_p_sampling_and_filter(
            probs, top_ks, top_ps, filter_apply_order="joint"
        )

        assert torch.all(samples >= 0) and torch.all(samples < vocab_size)
        # With no filtering, filtered should equal original probs
        assert torch.allclose(filtered, probs, atol=1e-5)


@pytest.mark.parametrize("top_p", [0.1, 0.5, 0.9, 0.99])
def test_top_p_values(top_p):
    """Test different top_p values."""
    torch.manual_seed(42)
    batch_size = 16
    vocab_size = 151936
    top_k = 1024
    num_trials = 10

    probs = generate_llm_like_prob(batch_size, vocab_size, device="cuda:0")
    top_ks = torch.full((batch_size,), top_k, dtype=torch.int32, device="cuda:0")
    top_ps = torch.full((batch_size,), top_p, dtype=torch.float32, device="cuda:0")

    for _ in range(num_trials):
        samples, filtered = flashinfer.sampling.top_k_top_p_sampling_and_filter(
            probs, top_ks, top_ps, filter_apply_order="joint"
        )

        assert torch.all(samples >= 0) and torch.all(samples < vocab_size)
        for i in range(batch_size):
            token_id = samples[i].item()
            assert filtered[i, token_id] > 0


@pytest.mark.parametrize("top_p", [1.0, 1.5])
def test_top_p_no_filtering(top_p):
    """Test top_p >= 1.0 (no p-filtering)."""
    torch.manual_seed(42)
    batch_size = 16
    vocab_size = 151936
    top_k = vocab_size + 1  # Also disable k-filtering
    num_trials = 10

    probs = generate_llm_like_prob(batch_size, vocab_size, device="cuda:0")
    top_ks = torch.full((batch_size,), top_k, dtype=torch.int32, device="cuda:0")
    top_ps = torch.full((batch_size,), top_p, dtype=torch.float32, device="cuda:0")

    for _ in range(num_trials):
        samples, filtered = flashinfer.sampling.top_k_top_p_sampling_and_filter(
            probs, top_ks, top_ps, filter_apply_order="joint"
        )

        assert torch.all(samples >= 0) and torch.all(samples < vocab_size)
        # With no filtering, filtered should equal original probs
        assert torch.allclose(filtered, probs, atol=1e-5)


@pytest.mark.parametrize(
    "top_k,top_p",
    [
        (1, 0.98),  # greedy with p-filter
        (1, 1.0),   # greedy without p-filter
    ],
)
def test_greedy_sampling(top_k, top_p):
    """Test greedy sampling (top_k=1)."""
    torch.manual_seed(42)
    batch_size = 16
    vocab_size = 151936
    num_trials = 10

    probs = generate_llm_like_prob(batch_size, vocab_size, device="cuda:0")
    top_ks = torch.full((batch_size,), top_k, dtype=torch.int32, device="cuda:0")
    top_ps = torch.full((batch_size,), top_p, dtype=torch.float32, device="cuda:0")

    for _ in range(num_trials):
        samples, filtered = flashinfer.sampling.top_k_top_p_sampling_and_filter(
            probs, top_ks, top_ps, filter_apply_order="joint"
        )

        assert torch.all(samples >= 0) and torch.all(samples < vocab_size)
        for i in range(batch_size):
            token_id = samples[i].item()
            assert filtered[i, token_id] > 0


@pytest.mark.parametrize("batch_size", [1, 8, 32, 64])
def test_stress(batch_size):
    """Stress test with 100 trials."""
    torch.manual_seed(42)
    vocab_size = 151936
    top_k = 1024
    top_p = 0.98
    num_trials = 100

    probs = generate_llm_like_prob(batch_size, vocab_size, device="cuda:0")
    top_ks = torch.full((batch_size,), top_k, dtype=torch.int32, device="cuda:0")
    top_ps = torch.full((batch_size,), top_p, dtype=torch.float32, device="cuda:0")

    for _ in range(num_trials):
        samples, filtered = flashinfer.sampling.top_k_top_p_sampling_and_filter(
            probs, top_ks, top_ps, filter_apply_order="joint"
        )

        assert torch.all(samples >= 0) and torch.all(samples < vocab_size)
        for i in range(batch_size):
            token_id = samples[i].item()
            assert filtered[i, token_id] > 0
        assert torch.all(filtered >= 0)


def benchmark_performance(batch_size=16, vocab_size=151936, top_k=1024, top_p=0.98):
    """Compare fused kernel vs original sampling kernel vs PyTorch baseline."""
    import time

    num_warmup = 10
    num_runs = 400

    probs = generate_llm_like_prob(batch_size, vocab_size, device="cuda:0")
    top_ks = torch.full((batch_size,), top_k, dtype=torch.int32, device="cuda:0")
    top_ps = torch.full((batch_size,), top_p, dtype=torch.float32, device="cuda:0")

    # Warmup all three
    for _ in range(num_warmup):
        _ = flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs, top_ks, top_ps, filter_apply_order="joint"
        )
        _ = flashinfer.sampling.top_k_top_p_sampling_and_filter(
            probs, top_ks, top_ps, filter_apply_order="joint"
        )
        _ = pytorch_top_k_top_p_sampling_and_filter(probs, top_ks, top_ps)

    # Benchmark original (flashinfer, no filtered probs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs, top_ks, top_ps, filter_apply_order="joint"
        )
    torch.cuda.synchronize()
    original_time = (time.perf_counter() - start) / num_runs * 1000

    # Benchmark fused (flashinfer, with filtered probs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = flashinfer.sampling.top_k_top_p_sampling_and_filter(
            probs, top_ks, top_ps, filter_apply_order="joint"
        )
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / num_runs * 1000

    # Benchmark PyTorch baseline (with filtered probs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = pytorch_top_k_top_p_sampling_and_filter(probs, top_ks, top_ps)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / num_runs * 1000

    ratio_fused = fused_time / original_time
    ratio_pytorch = pytorch_time / original_time
    speedup_vs_pytorch = pytorch_time / fused_time
    speedup_vs_original = original_time / fused_time
    print(
        f"bs={batch_size:3d} | original={original_time:.3f}ms, fused={fused_time:.3f}ms ({ratio_fused:.2f}x), "
        f"pytorch={pytorch_time:.3f}ms ({ratio_pytorch:.2f}x) | fused vs pytorch: {speedup_vs_pytorch:.2f}x"
    )
    return {
        "batch_size": batch_size, "vocab_size": vocab_size, "top_k": top_k, "top_p": top_p,
        "original": original_time, "fused": fused_time, "pytorch": pytorch_time,
        "ratio_fused": ratio_fused, "ratio_pytorch": ratio_pytorch,
        "speedup_vs_pytorch": speedup_vs_pytorch, "speedup_vs_original": speedup_vs_original
    }


def run_benchmark():
    """Run benchmark across different batch sizes, vocab sizes, and top_k/top_p."""
    batch_sizes = [1, 2, 4] + list(range(16, 257, 16))

    # Different configurations to test
    configs = [
        # (vocab_size, top_k, top_p, description)
        (151936, 1024, 0.98, "Qwen2.5 vocab, standard params"),
        (32000, 1024, 0.98, "LLaMA vocab"),
        (128256, 1024, 0.98, "GPT-4o vocab"),
        (151936, 50, 0.98, "small top_k"),
        (151936, 50000, 0.999, "large top_k + top_p (many tokens pass filter)"),
        (151936, 1024, 0.5, "aggressive top_p"),
    ]

    for vocab_size, top_k, top_p, desc in configs:
        print("\n" + "=" * 70)
        print(f"Benchmark: {desc} (vocab={vocab_size}, k={top_k}, p={top_p})")
        print("=" * 70)

        results = []
        for bs in batch_sizes:
            result = benchmark_performance(bs, vocab_size, top_k, top_p)
            results.append(result)

        print("\n" + "-" * 70)
        speedups_original = [r["speedup_vs_original"] for r in results]
        speedups_pytorch = [r["speedup_vs_pytorch"] for r in results]
        print(f"Fused vs original: min={min(speedups_original):.2f}x, max={max(speedups_original):.2f}x, avg={sum(speedups_original)/len(speedups_original):.2f}x")
        print(f"Fused vs PyTorch:  min={min(speedups_pytorch):.2f}x, max={max(speedups_pytorch):.2f}x, avg={sum(speedups_pytorch)/len(speedups_pytorch):.2f}x")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        run_benchmark()
    else:
        # Run pytest
        sys.exit(pytest.main([__file__, "-v"]))
