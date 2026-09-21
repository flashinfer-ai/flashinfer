"""Experimental fused BF16 routing GEMM, expert mapping and normalized top-k."""
from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def prepare_mega_gate(x, weight, num_topk=6, *, scoring_func="sqrtsoftplus", bias=None,
                      image_bias=None, image_token_mask=None, mask=None, to_physical_map=None,
                      logical_count=None, fix_routing_mask=None, force_random=None,
                      unmapped_topk_idx=None, use_shared_as_routed=False, num_shared_experts=1,
                      routed_scaling_factor=1.5, ep_rank=0, out=None, deterministic=False,
                      scratch=None, score_barriers=None, descriptor_workspace=None):
    """Prepare a reusable routing plan; plan.run() returns indices and weights.

    Inputs are BF16[M,K] and BF16[E,K]. Output is int64[M,topk+shared] plus
    FP32 weights. Plans preserve physical/logical expert mapping, unbiased
    scoring, deterministic routing, and the current PyTorch CUDA stream.
    Only the exported configurations are accepted. See MegaGatePlan for
    optional metadata and reusable workspace layouts.
    """
    from .experimental.deepgemm_mega_gate.mega_gate import MegaGatePlan
    return MegaGatePlan(x, weight, num_topk, scoring_func=scoring_func, bias=bias,
        image_bias=image_bias, image_token_mask=image_token_mask, mask=mask,
        to_physical_map=to_physical_map, logical_count=logical_count,
        fix_routing_mask=fix_routing_mask, force_random=force_random,
        unmapped_topk_idx=unmapped_topk_idx, use_shared_as_routed=use_shared_as_routed,
        num_shared_experts=num_shared_experts, routed_scaling_factor=routed_scaling_factor,
        ep_rank=ep_rank, out=out, deterministic=deterministic, scratch=scratch,
        score_barriers=score_barriers, descriptor_workspace=descriptor_workspace)
