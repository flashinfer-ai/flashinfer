"""A top-p threshold of one retains every finite nonnegative entry."""

import pytest
import torch

from flashinfer.sampling import top_p_renorm_probs


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("vocabulary", [17, 65537])
def test_full_support_keeps_small_positive_tails_and_reuses_graph(
    deterministic, vocabulary
):
    torch.manual_seed(4161)
    probabilities = torch.rand((4, vocabulary), device="cuda", dtype=torch.float32)
    probabilities[0].zero_()
    probabilities[0, 0] = 1
    probabilities[0, 1] = 1e-12
    probabilities[1].zero_()
    probabilities[1, -1] = 1
    probabilities[2, ::3] = 0
    probabilities[2].div_(probabilities[2].sum())
    probabilities[3].fill_(1 / vocabulary)
    thresholds = torch.ones(4, device="cuda")
    output = top_p_renorm_probs(
        probabilities, thresholds, is_deterministic=deterministic
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = top_p_renorm_probs(
            probabilities, thresholds, is_deterministic=deterministic
        )
    for full_support in (True, False, True):
        thresholds[2] = 1 if full_support else 0.8
        captured.fill_(float("nan"))
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        expected = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-8)
        selected = torch.tensor([0, 1, 3], device="cuda")
        torch.testing.assert_close(
            captured[selected], expected[selected], rtol=1e-6, atol=0
        )
        assert captured[0, 1] > 0
        if full_support:
            torch.testing.assert_close(captured, expected, rtol=1e-6, atol=0)
        else:
            sorted_values = probabilities[2].sort(descending=True).values
            mass = sorted_values.cumsum(0)
            cutoff_index = (
                (mass < probabilities[2].sum() * 0.8).sum().clamp_max(vocabulary - 1)
            )
            kept = torch.where(
                probabilities[2] >= sorted_values[cutoff_index], probabilities[2], 0
            )
            torch.testing.assert_close(
                captured[2], kept / kept.sum(), rtol=1e-5, atol=1e-7
            )
    graph.reset()
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=0)
