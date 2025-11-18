import torch
import pytest
from flashinfer.dsv3_ops import NoAuxTc
import torch.nn.functional as F
from flashinfer.utils import get_compute_capability


def dsv3_ref_check(scores_in, bias, n_group, topk_group, top_k, routed_scaling_factor):
    scores = F.sigmoid(scores_in)
    scores_with_bias = scores + bias
    scores_shape = list(scores_with_bias.shape)
    group_scores = torch.sum(
        torch.topk(
            scores_with_bias.view(
                scores_shape[:-1] + [n_group, scores_shape[-1] // n_group]
            ),
            k=2,
            dim=-1,
            largest=True,
            sorted=True,
        )[0],
        dim=-1,
    )
    _, group_idx = torch.topk(
        group_scores, k=topk_group, dim=-1, largest=True, sorted=True
    )
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(-1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(scores_shape[:-1] + [n_group, scores_shape[-1] // n_group])
        .reshape(scores_shape)
    )
    scores_with_bias = scores_with_bias * score_mask
    _, topk_idx = torch.topk(
        scores_with_bias, k=top_k, dim=-1, largest=True, sorted=True
    )
    new_mask = torch.zeros_like(scores)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = scores * new_mask
    score_sum = torch.sum(scores, dim=-1, keepdim=True) + 1e-20
    scores = scores / score_sum * routed_scaling_factor
    topk_values, topk_indices = torch.topk(scores, k=top_k, dim=-1, largest=True)
    return topk_values, topk_indices


@pytest.mark.parametrize("num_tokens", [16])
@pytest.mark.parametrize("num_experts", [256])
def test_dsv3_fused_routing_op(num_tokens, num_experts):
    compute_capability = get_compute_capability(torch.device("cuda"))
    compute_capability_number = compute_capability[0] * 10 + compute_capability[1]
    if compute_capability_number != 100:
        pytest.skip("DSv3 Fused Routing is only supported on SM100")

    scores = torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.float32)
    bias = torch.randn(num_experts, device="cuda", dtype=torch.float32)
    n_group = 1
    topk_group = 1
    topk = 1
    routed_scaling_factor = 1.0
    topk_values = torch.randn(num_tokens, topk, device="cuda", dtype=torch.float32)
    topk_indices = torch.randn(num_tokens, topk, device="cuda").to(torch.int32)

    NoAuxTc(
        scores,
        bias,
        n_group,
        topk_group,
        topk,
        routed_scaling_factor,
        topk_values,
        topk_indices,
        launch_with_pdl=True,
    )

    topk_values_ref, topk_indices_ref = dsv3_ref_check(
        scores, bias, n_group, topk_group, topk, routed_scaling_factor
    )
    torch.testing.assert_close(topk_values_ref, topk_values, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(topk_indices_ref.int(), topk_indices.int())
