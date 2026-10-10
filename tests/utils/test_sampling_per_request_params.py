"""Per-request sampling parameters when `indices` remaps requests to probability rows.

The documented contract is that `probs` has one row per unique distribution, `indices` has one
entry per request, and a parameter tensor has one entry per request. A threshold therefore belongs
to the request, not to the row that request happens to read.
"""

import pytest
import torch

import flashinfer

# Cumulative mass is 0.5, 0.8, 1.0, so the thresholds below select known token sets.
SHARED_PROBS = [0.5, 0.3, 0.2]
TRIALS = 256


def _probs(device):
    return torch.tensor([SHARED_PROBS], dtype=torch.float32, device=device)


def _support_per_group(samples, num_groups):
    """Tokens actually sampled by each group of requests."""
    samples = samples.cpu().tolist()
    return [
        sorted({s for i, s in enumerate(samples) if i % num_groups == g})
        for g in range(num_groups)
    ]


@pytest.mark.parametrize("device", ["cuda:0"])
def test_top_k_per_request_with_indices(device):
    torch.manual_seed(42)
    probs = _probs(device)
    batch_size = 3 * TRIALS
    indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
    top_k = torch.tensor([1, 2, 3], dtype=torch.int32, device=device).repeat(TRIALS)

    samples = flashinfer.sampling.top_k_sampling_from_probs(
        probs, top_k, indices=indices
    )

    assert _support_per_group(samples, 3) == [[0], [0, 1], [0, 1, 2]]


@pytest.mark.parametrize("device", ["cuda:0"])
def test_top_p_per_request_with_indices(device):
    torch.manual_seed(42)
    probs = _probs(device)
    batch_size = 3 * TRIALS
    indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
    top_p = torch.tensor([0.4, 0.79, 1.0], dtype=torch.float32, device=device).repeat(
        TRIALS
    )

    samples = flashinfer.sampling.top_p_sampling_from_probs(
        probs, top_p, indices=indices
    )

    assert _support_per_group(samples, 3) == [[0], [0, 1], [0, 1, 2]]


@pytest.mark.parametrize("device", ["cuda:0"])
def test_min_p_per_request_with_indices(device):
    torch.manual_seed(42)
    probs = _probs(device)
    batch_size = 3 * TRIALS
    indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
    # min_p keeps tokens with probability >= min_p * max_prob, and the max is 0.5.
    min_p = torch.tensor([0.9, 0.5, 0.0], dtype=torch.float32, device=device).repeat(
        TRIALS
    )

    samples = flashinfer.sampling.min_p_sampling_from_probs(
        probs, min_p, indices=indices
    )

    assert _support_per_group(samples, 3) == [[0], [0, 1], [0, 1, 2]]


@pytest.mark.parametrize("device", ["cuda:0"])
def test_top_k_top_p_joint_per_request_with_indices(device):
    torch.manual_seed(42)
    probs = _probs(device)
    batch_size = 3 * TRIALS
    indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
    top_k = torch.tensor([1, 2, 3], dtype=torch.int32, device=device).repeat(TRIALS)
    top_p = torch.ones(batch_size, dtype=torch.float32, device=device)

    samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(
        probs, top_k, top_p, indices=indices, filter_apply_order="joint"
    )

    assert _support_per_group(samples, 3) == [[0], [0, 1], [0, 1, 2]]


@pytest.mark.parametrize("device", ["cuda:0"])
def test_thresholds_follow_the_request_not_the_row(device):
    """Two rows, two requests, reversed indices: each request keeps its own threshold."""
    torch.manual_seed(42)
    probs = torch.tensor([[0.6, 0.4], [0.4, 0.6]], dtype=torch.float32, device=device)
    # Request 0 reads row 1, request 1 reads row 0.
    indices = torch.tensor([1, 0], dtype=torch.int32, device=device).repeat(TRIALS)
    # Request 0 keeps only its row's top token; request 1 keeps both of its row's tokens.
    top_p = torch.tensor([0.5, 1.0], dtype=torch.float32, device=device).repeat(TRIALS)

    samples = flashinfer.sampling.top_p_sampling_from_probs(
        probs, top_p, indices=indices
    )

    support = _support_per_group(samples, 2)
    assert support[0] == [1], "request 0 sampled a token its own top_p excludes"
    assert support[1] == [0, 1], "request 1 was restricted by another request's top_p"


@pytest.mark.parametrize("device", ["cuda:0"])
def test_parameter_length_must_match_batch_size(device):
    probs = _probs(device)
    indices = torch.zeros(3, dtype=torch.int32, device=device)
    # One entry per row rather than per request is not enough for three requests.
    top_k = torch.tensor([1], dtype=torch.int32, device=device)

    with pytest.raises(Exception, match="batch size"):
        flashinfer.sampling.top_k_sampling_from_probs(probs, top_k, indices=indices)
