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

"""The top-k / top-p cases of ``tests/utils/test_sampling.py`` replayed on ``cake_sampling``.

Every test below is the corresponding ``flashinfer.sampling`` unit test with the sampler replaced
by :func:`flashinfer.cake_sampling.top_k_top_p_sampling_from_probs` and the mask written for the
``top_k_first`` order that function implements (top-k first, then top-p over the renormalized
top-k mass).  Parametrizations (batch sizes, vocabularies, ``k``, ``p``, trial counts, similarity
thresholds) are the upstream ones.  Cases the frozen kernels do not serve (``k >= vocab``) are
skipped like upstream skips ``k > vocab_size``; the wrapper's fallback route covers them.
"""

import pytest
import torch

import flashinfer
from flashinfer.cake_sampling import (
    cake_sampling_route,
    top_k_top_p_sampling_from_probs,
)
from flashinfer.jit.cake_sampling import supported_capability


def _require_supported_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if supported_capability(torch.cuda.get_device_capability()) is None:
        pytest.skip(
            "frozen radix sampling kernels need compute capability 9.0/10.0/10.3/10.7/11.0"
        )


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


def _top_k_mask(probs, k):
    """Upstream top-k mask: every entry at or above the k-th largest probability."""
    sorted_prob, _ = torch.sort(probs, descending=True)
    if isinstance(k, torch.Tensor):
        pivot = sorted_prob[torch.arange(probs.shape[0], device=probs.device), k - 1]
    else:
        pivot = sorted_prob[:, k - 1]
    return (probs >= pivot.unsqueeze(-1)).int()


def _top_k_first_mask(probs, k, p, eps=1e-6):
    """top_k_first support: the upstream top-p mask applied to the renormalized top-k mass.

    Upstream builds the top-p mask from an ascending cumsum, which puts an exactly tied entry on
    either side of the boundary; ``cake_sampling`` resolves ties by ascending index (the kept set
    is the shortest prefix of ``lexsort(-prob, index)`` whose exclusive mass reaches ``p``), so
    the mask is built in that order: an entry is kept iff the mass strictly before it is below
    ``p`` (``eps`` covers float rounding of the prefix sums).
    """
    mask_k = _top_k_mask(probs, k)
    kept = (probs * mask_k).double()
    kept = kept / kept.sum(dim=-1, keepdim=True)
    # stable descending sort: equal values stay in ascending index order
    sorted_prob, indices = torch.sort(kept, descending=True, stable=True)
    excl = torch.cumsum(sorted_prob, dim=-1) - sorted_prob
    mask_p = torch.zeros_like(mask_k)
    mask_p.scatter_(1, indices, (excl < p + eps).int())
    return torch.minimum(mask_k, mask_p)


def _pipeline_or_skip(probs, k):
    route = cake_sampling_route(probs, k)
    if route != "pipeline":
        pytest.skip(f"request is not served by the frozen kernels ({route})")


def _freq(probs_row, k, p, num_trials, rows=1000):
    """``num_trials`` draws from one row (upstream uses ``indices``; here the row is replicated)."""
    vocab = probs_row.shape[-1]
    probs = probs_row.expand(rows, vocab).contiguous()
    counter = torch.zeros(vocab, dtype=torch.int32, device=probs.device)
    out = torch.empty(rows, dtype=torch.int32, device=probs.device)
    for _ in range(num_trials // rows):
        top_k_top_p_sampling_from_probs(probs, k, p, out=out)
        counter.scatter_add_(0, out.long(), torch.ones_like(out))
    return counter.float() / (num_trials // rows * rows)


# --- test_top_k_sampling_freq -------------------------------------------------------------------


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
    _require_supported_device()
    if k >= vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    logits = distribution((1, vocab_size), "cuda:0")
    probs = torch.softmax(logits, dim=-1)
    _pipeline_or_skip(probs, k)
    mask = _top_k_mask(probs, k)

    renorm_probs = flashinfer.sampling.top_k_renorm_probs(probs, k)
    freq = _freq(probs, k, 1.0, num_trials=5000000)
    assert torch.all(mask[0][freq > 0] == 1)
    similarity = torch.cosine_similarity(freq, renorm_probs[0], dim=0)
    assert similarity > 0.99, f"similarity: {similarity}"


# --- test_top_p_sampling_freq (top_k_first: top-p over the top-k renormalized mass) -----------


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
def test_top_k_top_p_sampling_freq(vocab_size, distribution, p):
    _require_supported_device()
    k = 100
    if k >= vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    logits = distribution((1, vocab_size), "cuda:0")
    probs = torch.softmax(logits, dim=-1)
    _pipeline_or_skip(probs, k)
    mask = _top_k_first_mask(probs, k, p)

    renorm_probs = flashinfer.sampling.top_p_renorm_probs(
        flashinfer.sampling.top_k_renorm_probs(probs, k), p
    )
    freq = _freq(probs, k, p, num_trials=5000000)
    assert torch.all(mask[0][freq > 0] == 1)
    similarity = torch.cosine_similarity(freq, renorm_probs[0], dim=0)
    assert similarity > 0.99, f"similarity: {similarity}"


# --- test_top_k_sampling ------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_sampling(batch_size, vocab_size, k):
    _require_supported_device()
    if k >= vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    _pipeline_or_skip(normalized_prob, k)
    mask = _top_k_mask(normalized_prob, k)

    num_trails = 1000
    for _ in range(num_trails):
        samples = top_k_top_p_sampling_from_probs(normalized_prob, k, 1.0)
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)
        assert torch.all(mask[torch.arange(batch_size), samples] == 1), normalized_prob[
            torch.arange(batch_size), samples
        ]


# --- test_top_k_sampling_with_variable_k --------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_sampling_with_variable_k(batch_size, vocab_size, k):
    _require_supported_device()
    if k >= vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    k = torch.randint(1, k + 1, (batch_size,), device="cuda:0")  # int64 like upstream
    _pipeline_or_skip(normalized_prob, k)
    mask = _top_k_mask(normalized_prob, k)

    num_trails = 1000
    for _ in range(num_trails):
        samples = top_k_top_p_sampling_from_probs(normalized_prob, k, 1.0)
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)
        assert torch.all(mask[torch.arange(batch_size), samples] == 1), normalized_prob[
            torch.arange(batch_size), samples
        ]


# --- test_top_p_sampling / test_top_k_top_p_joint_sampling_from_probs (top_k_first order) ------


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_top_k_top_p_sampling_from_probs(batch_size, vocab_size, k, p):
    _require_supported_device()
    if k >= vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    _pipeline_or_skip(normalized_prob, k)
    mask = _top_k_first_mask(normalized_prob, k, p)
    top_p_tensor = torch.full((batch_size,), p, device="cuda:0")
    top_k_tensor = torch.full((batch_size,), k, device="cuda:0")

    num_trails = 1000
    for i in range(num_trails):
        if i % 2:
            samples = top_k_top_p_sampling_from_probs(
                normalized_prob, top_k_tensor, top_p_tensor
            )
        else:
            samples = top_k_top_p_sampling_from_probs(normalized_prob, k, p)
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)
        assert torch.all(mask[torch.arange(batch_size), samples] == 1), normalized_prob[
            torch.arange(batch_size), samples
        ]


# --- test_sampling_from_probs_seed_offset_reproducibility ---------------------------------------


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
def test_seed_offset_reproducibility(batch_size, vocab_size):
    """Explicit seed/offset produces reproducible results."""
    _require_supported_device()
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    k = min(100, vocab_size - 1)
    _pipeline_or_skip(normalized_prob, k)

    seed, offset = 12345, 0
    samples1 = top_k_top_p_sampling_from_probs(
        normalized_prob, k, 0.9, philox_seed=seed, philox_offset=offset
    )
    samples2 = top_k_top_p_sampling_from_probs(
        normalized_prob, k, 0.9, philox_seed=seed, philox_offset=offset
    )
    assert torch.all(samples1 == samples2), (
        "Same seed/offset should produce identical samples"
    )


# --- test_sampling_different_seed_offset_produces_different_results ----------------------------


@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
def test_different_seed_offset_produces_different_results(vocab_size):
    """Different seed/offset values produce different samples."""
    _require_supported_device()
    torch.manual_seed(42)
    batch_size = 1000
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    k = min(100, vocab_size - 1)
    _pipeline_or_skip(normalized_prob, k)

    def draw(seed, offset):
        return top_k_top_p_sampling_from_probs(
            normalized_prob, k, 0.9, philox_seed=seed, philox_offset=offset
        )

    samples_seed1 = draw(12345, 0)
    samples_seed2 = draw(67890, 0)
    samples_offset1 = draw(12345, 0)
    samples_offset2 = draw(12345, 1000)

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


# --- test_top_k_top_p_sampling_from_probs_logits_alignment ---------------------------------------


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [100])
@pytest.mark.parametrize("p", [0.1, 0.5])
def test_top_k_top_p_sampling_from_probs_logits_alignment(batch_size, vocab_size, k, p):
    """Upstream compares ``from_logits`` with ``from_probs`` on the ``top_k_first`` route and
    tolerates 5 % mismatches; the same softmax-vs-mask difference applies to this sampler, so
    the replay checks that every sample of both calls lies in the other's ``top_k_first``
    support (the draws themselves use a different inverse-CDF than the rejection sampler)."""
    _require_supported_device()
    if k >= vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda:0") * 5
    probs = torch.softmax(logits, dim=-1)
    _pipeline_or_skip(probs, k)
    generator = torch.Generator("cuda:0")
    samples_ref = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        logits, k, p, filter_apply_order="top_k_first", generator=generator
    )
    samples = top_k_top_p_sampling_from_probs(probs, k, p, generator=generator)

    mask = _top_k_first_mask(probs, k, p)
    rows = torch.arange(batch_size, device="cuda:0")
    assert torch.all(mask[rows, samples] == 1)
    ref_in_support = (mask[rows, samples_ref] == 1).float().mean().item()
    assert ref_in_support >= 0.95, (
        f"top_k_first reference outside the replayed support in "
        f"{1 - ref_in_support:.2%} of rows (expected <= 5%)"
    )


# --- test_sampling_nan_input -------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 4, 19])
@pytest.mark.parametrize("vocab_size", [111, 32000])
def test_sampling_nan_input(batch_size, vocab_size):
    """All-NaN rows sample index 0 (upstream: ``result == 0`` and ``valid == False``); other
    rows are unaffected and sample inside their top_k_first support."""
    _require_supported_device()
    torch.manual_seed(42)
    probs = torch.rand(batch_size, vocab_size, device="cuda:0", dtype=torch.float32)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    k, p = 50, 0.9
    _pipeline_or_skip(probs, k)

    nan_indices = [0]
    if batch_size > 1:
        nan_indices.append(batch_size // 2)
    if batch_size > 2:
        nan_indices.append(batch_size - 1)
    valid_indices = [i for i in range(batch_size) if i not in nan_indices]
    clean = probs.clone()
    for idx in nan_indices:
        probs[idx, :] = float("nan")

    mask = _top_k_first_mask(clean, k, p)
    for _ in range(100):
        result = top_k_top_p_sampling_from_probs(probs, k, p)
        for idx in nan_indices:
            assert result[idx].item() == 0
        for idx in valid_indices:
            assert 0 <= result[idx].item() < vocab_size
            assert mask[idx, result[idx]] == 1
