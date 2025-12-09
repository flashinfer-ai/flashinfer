from flashinfer.jit import gen_dsv3_fused_routing_module
import functools
from types import SimpleNamespace
import torch
from flashinfer.utils import (
    register_custom_op,
    supported_compute_capability,
    backend_requirement,
)


@supported_compute_capability([89, 90, 100, 103, 120, 121])
def _check_dsv3_fused_routing_supported(
    scores,
    bias,
    n_group,
    topk_group,
    topk,
    routed_scaling_factor,
    topk_values,
    topk_indices,
    launch_with_pdl,
):
    """Validate configuration parameters for DSv3 fused routing kernel.

    Args:
        scores: Input routing scores tensor
        bias: Per-expert routing bias tensor
        n_group: Number of expert groups
        topk_group: Number of top groups to select
        topk: Number of top experts to select per token
        routed_scaling_factor: Scaling factor for normalized weights
        topk_values: Output tensor for normalized expert weights
        topk_indices: Output tensor for selected expert indices
        launch_with_pdl: Whether to use Persistent Device-side Launch

    Raises:
        ValueError: If configuration is invalid or exceeds kernel limits
    """
    # Extract number of experts from scores shape
    num_experts = scores.shape[1]

    # Check basic configuration constraints
    if topk_group * n_group < topk or topk_group > n_group:
        raise ValueError(
            f"Invalid configuration: topk_group * n_group ({topk_group * n_group}) must be >= topk ({topk}) "
            f"and topk_group ({topk_group}) must be <= n_group ({n_group})"
        )

    # Check kernel limits based on number of groups
    if n_group > 1:
        experts_per_group = num_experts / n_group
        max_experts_in_selected_groups = experts_per_group * topk_group

        if topk > 8:
            raise ValueError(
                f"Invalid configuration for n_group > 1: topk ({topk}) must be <= 8"
            )
        if experts_per_group > 32:
            raise ValueError(
                f"Invalid configuration for n_group > 1: num_experts / n_group "
                f"({experts_per_group}) must be <= 32"
            )
        if max_experts_in_selected_groups > 128:
            raise ValueError(
                f"Invalid configuration for n_group > 1: num_experts / n_group * topk_group "
                f"({max_experts_in_selected_groups}) must be <= 128"
            )
    else:  # n_group == 1
        if num_experts > 384:
            raise ValueError(
                f"Invalid configuration for n_group = 1: num_experts ({num_experts}) must be <= 384"
            )
        if topk > 8:
            raise ValueError(
                f"Invalid configuration for n_group = 1: topk ({topk}) must be <= 8"
            )

    return True


@functools.cache
def get_dsv3_fused_routing_module():
    module = gen_dsv3_fused_routing_module().build_and_load()

    @register_custom_op(
        "flashinfer::NoAuxTc",
        mutates_args=["topk_values", "topk_indices"],
    )
    def NoAuxTc(
        scores: torch.Tensor,
        bias: torch.Tensor,
        n_group: int,
        topk_group: int,
        topk: int,
        routed_scaling_factor: float,
        topk_values: torch.Tensor,
        topk_indices: torch.Tensor,
        launch_with_pdl: bool = True,
    ) -> None:
        module.NoAuxTc(
            scores,
            bias,
            n_group,
            topk_group,
            topk,
            routed_scaling_factor,
            topk_values,
            topk_indices,
            launch_with_pdl,
        )

    return SimpleNamespace(
        NoAuxTc=NoAuxTc,
    )


@backend_requirement({}, common_check=_check_dsv3_fused_routing_supported)
def fused_topk_deepseek(
    scores: torch.Tensor,
    bias: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    routed_scaling_factor: float,
    topk_values: torch.Tensor,
    topk_indices: torch.Tensor,
    launch_with_pdl: bool = True,
) -> None:
    """Fused expert routing with top-k selection for DeepSeek-V3.

    This function performs a highly optimized fused routing operation specifically
    designed for DeepSeek-V3's Mixture of Experts (MoE) architecture with grouped
    expert routing and no auxiliary loss. It combines score computation, expert
    selection, and normalization into a single kernel operation.

    The routing algorithm consists of the following steps:
    1. Compute biased scores: sigmoid(scores) + bias for each expert
    2. Group experts and compute group scores (sum of top-2 experts per group)
    3. Select top-k groups based on group scores
    4. From selected groups, select top-k experts based on biased scores
    5. Normalize selected expert weights: sigmoid_scores / sum(sigmoid_scores) * scale

    Args:
        scores (torch.Tensor): Input routing scores of shape (num_tokens, num_experts).
            The logits produced by the router network before activation. Supports
            bfloat16, float16, or float32.
        bias (torch.Tensor): Per-expert routing bias of shape (num_experts,). Added to
            sigmoid-activated scores to produce biased scores for expert selection.
            Must match the dtype of scores.
        n_group (int): Number of expert groups. Experts are divided into groups for
            hierarchical selection. Typical value is 8 for DeepSeek-V3 with 256 experts
            (32 experts per group).
        topk_group (int): Number of top groups to select. Must be <= n_group. Typical
            value is 4, meaning the top 4 groups are selected from 8 groups.
        topk (int): Number of top experts to select per token. Must be <= num_experts.
            Typical value is 8, meaning 8 experts are routed per token.
        routed_scaling_factor (float): Scaling factor applied to normalized expert
            weights. The final output weights are:
            sigmoid_scores / sum(sigmoid_scores) * routed_scaling_factor.
        topk_values (torch.Tensor): Pre-allocated output tensor of shape
            (num_tokens, topk) for the normalized expert weights. Must be float32.
            This tensor is mutated in-place.
        topk_indices (torch.Tensor): Pre-allocated output tensor of shape
            (num_tokens, topk) for the selected expert indices. Must be int32 or int64.
            This tensor is mutated in-place.
        launch_with_pdl (bool, optional): Whether to launch the kernel using Persistent
            Device-side Launch. Defaults to True.

    Returns:
        None: Results are written directly to `topk_values` and `topk_indices` tensors.

    Note:
        - The kernel uses float32 internally for all computations to ensure numerical
          precision, even when inputs are float16 or bfloat16.
        - This implementation is optimized for Hopper (compute capability 90, 100),
          Ada (compute capability 89), and Blackwell (compute capability 120, 121)
          architectures.
        - The "NoAux" prefix indicates this variant does not compute auxiliary losses
          (e.g., load balancing loss) during routing.
        - The "Tc" suffix indicates the use of Tensor Core optimizations in the
          underlying CUDA kernel.
    """
    get_dsv3_fused_routing_module().NoAuxTc(
        scores,
        bias,
        n_group,
        topk_group,
        topk,
        routed_scaling_factor,
        topk_values,
        topk_indices,
        launch_with_pdl,
    )
