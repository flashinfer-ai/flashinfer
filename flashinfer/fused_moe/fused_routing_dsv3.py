import functools
from types import SimpleNamespace
from typing import Optional

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.jit import gen_dsv3_fused_routing_module
from flashinfer.trace.templates.sampling import fused_topk_deepseek_trace
from flashinfer.utils import (
    backend_requirement,
    register_custom_op,
    supported_compute_capability,
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
    routing_replay_out=None,
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
    if routing_replay_out is not None:
        num_tokens = scores.shape[0]
        if routing_replay_out.dtype != torch.int16:
            raise ValueError(
                f"routing_replay_out must be int16, got {routing_replay_out.dtype}"
            )
        if (
            routing_replay_out.shape[0] < num_tokens
            or routing_replay_out.shape[1] != topk
        ):
            raise ValueError(
                f"routing_replay_out shape[0] must be >= {num_tokens} and shape[1] must be {topk}, "
                f"got {tuple(routing_replay_out.shape)}"
            )

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
        mutates_args=["topk_values", "topk_indices", "routing_replay_out"],
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
        routing_replay_out: Optional[torch.Tensor] = None,
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
            routing_replay_out,
        )

    return SimpleNamespace(
        NoAuxTc=NoAuxTc,
    )


@backend_requirement({}, common_check=_check_dsv3_fused_routing_supported)
@flashinfer_api(trace=fused_topk_deepseek_trace)
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
    routing_replay_out: Optional[torch.Tensor] = None,
) -> None:
    r"""Fused expert routing with top-k selection for DeepSeek-V3.

    Performs a highly optimized fused routing operation designed for
    DeepSeek-V3's Mixture-of-Experts architecture with grouped expert routing
    and no auxiliary loss.  Combines score computation, expert selection, and
    normalization into a single kernel:

    1. Compute biased scores ``sigmoid(scores) + bias``.
    2. Group experts and compute per-group scores (sum of top-2 experts per
       group).
    3. Select the top ``topk_group`` groups by group score.
    4. From the selected groups, pick the top ``topk`` experts by biased
       score.
    5. Normalize the selected expert weights:
       ``sigmoid_scores / sum(sigmoid_scores) * routed_scaling_factor``.

    Parameters
    ----------
    scores : torch.Tensor
        Router logits of shape ``(num_tokens, num_experts)``, before any
        activation.  ``bfloat16`` / ``float16`` / ``float32``.
    bias : torch.Tensor
        Per-expert routing bias of shape ``(num_experts,)``, same dtype as
        ``scores``.  Added to the sigmoid-activated scores before grouping.
    n_group : int
        Number of expert groups.  Must satisfy ``n_group <= 32`` and
        ``num_experts % n_group == 0``.  Typical value is 8 for DeepSeek-V3
        with 256 experts (32 experts per group).
    topk_group : int
        Number of top groups to select.  Must satisfy ``topk_group <=
        n_group`` and ``topk_group * n_group >= topk``.  Typical value is 4.
    topk : int
        Number of top experts to select per token.  Must be ``<= num_experts``.
        Hard cap ``topk <= 32``; in addition both branches of the kernel
        require ``topk <= 8``.  Typical value is 8.

        Further per-branch constraints:

        - When ``n_group > 1``: ``num_experts / n_group <= 32`` and
          ``(num_experts / n_group) * topk_group <= 128``.
        - When ``n_group == 1``: ``num_experts <= 384``.
    routed_scaling_factor : float
        Scaling factor applied to the normalized expert weights (see step 5
        in the algorithm summary above).
    topk_values : torch.Tensor
        Pre-allocated output tensor of shape ``(num_tokens, topk)``.  Must
        have the same dtype as ``scores`` (``bfloat16`` / ``float16`` /
        ``float32``); the normalized expert weights are written here in
        place.
    topk_indices : torch.Tensor
        Pre-allocated output tensor of shape ``(num_tokens, topk)``.  Must
        be ``int32``.  The selected expert indices are written here in
        place.
    launch_with_pdl : bool
        Whether to launch the kernel with Programmatic Dependent Launch.
        Defaults to ``True``.
    routing_replay_out : Optional[torch.Tensor]
        Pre-allocated ``int16`` tensor used to record the selected expert
        IDs.  Shape must satisfy ``shape[0] >= num_tokens`` and
        ``shape[1] == topk`` — the ``>=`` on ``shape[0]`` is intentional so
        the same buffer can be sized for the maximum batch and reused across
        steps with smaller ``num_tokens`` under CUDA graphs (the kernel only
        writes indices ``[0, num_tokens)``).  When ``None`` (default) the
        kernel skips this write (zero overhead).

    Returns
    -------
    None
        Results are written in place to ``topk_values`` and ``topk_indices``
        (and optionally ``routing_replay_out``).

    Notes
    -----
    The kernel uses ``float32`` internally for numerical precision regardless
    of the input dtype.  Supported on Ada (SM89), Hopper (SM90), and
    Blackwell (SM100/SM103/SM120/SM121).  In the underlying CUDA kernel name
    ``NoAuxTc``, the ``NoAux`` prefix indicates the absence of auxiliary
    load-balancing losses and the ``Tc`` suffix indicates Tensor-Core
    utilization.
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
        routing_replay_out,
    )
