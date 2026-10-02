import functools
from types import SimpleNamespace
from typing import Literal, Optional

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.jit import gen_dsv3_fused_routing_module
from flashinfer.trace.templates.sampling import fused_topk_deepseek_trace
from flashinfer.utils import (
    backend_requirement,
    register_custom_op,
    supported_compute_capability,
)


_CAKE_DTYPES = frozenset((torch.float16, torch.bfloat16, torch.float32))


def _is_cake_dsv3_fused_routing_supported(
    *,
    capability: tuple[int, int],
    num_tokens: int,
    num_experts: int,
    n_group: int,
    topk_group: int,
    topk: int,
    score_dtype: torch.dtype,
    bias_dtype: torch.dtype,
) -> bool:
    """Return whether the call is inside Cake's executable NoAuxTc contract."""

    if capability not in ((10, 0), (10, 3)):
        return False
    if score_dtype not in _CAKE_DTYPES or bias_dtype not in _CAKE_DTYPES:
        return False
    if num_tokens <= 0 or num_experts <= 0 or n_group <= 0:
        return False
    if num_experts % n_group != 0:
        return False
    if topk <= 0 or topk > 8 or topk > num_experts:
        return False
    if topk_group <= 0 or topk_group > n_group:
        return False

    if n_group == 1:
        # The single-group Cake schedules write a single winner per token, so
        # they only implement top-1 routing.
        return topk == 1 and num_experts <= 384

    experts_per_group = num_experts // n_group
    # The grouped Cake schedules select ``topk`` of the experts reachable from
    # the ``topk_group`` selected groups.
    if topk > topk_group * experts_per_group:
        return False

    return (
        n_group <= 8
        and topk_group <= 4
        and num_experts <= 256
        and 2 <= experts_per_group <= 32
        and experts_per_group * topk_group <= 128
    )


@supported_compute_capability([89, 90, 100, 103, 107, 120, 121])
def _check_default_dsv3_fused_routing_backend_supported(**_kwargs) -> bool:
    """Return whether the existing fused-routing backend is available."""

    return True


@supported_compute_capability([100, 103])
def _check_cake_dsv3_fused_routing_backend_supported(
    scores: torch.Tensor,
    bias: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    **_kwargs,
) -> bool:
    """Validate the explicit Cake backend's executable contract."""

    capability = torch.cuda.get_device_capability(scores.device)
    if not _is_cake_dsv3_fused_routing_supported(
        capability=capability,
        num_tokens=scores.shape[0],
        num_experts=scores.shape[1],
        n_group=n_group,
        topk_group=topk_group,
        topk=topk,
        score_dtype=scores.dtype,
        bias_dtype=bias.dtype,
    ):
        raise ValueError(
            "backend='cake' does not support this fused-routing configuration"
        )
    return True


@supported_compute_capability([89, 90, 100, 103, 107, 120, 121])
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
    backend="default",
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
        routing_replay_out: Optional tensor for recording selected expert IDs
        backend: Fused-routing implementation selected by the public API

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

    # ``n_group`` must be positive and divide the expert count so that the
    # per-group expert capacity is well defined.
    if n_group <= 0:
        raise ValueError(f"Invalid configuration: n_group ({n_group}) must be > 0")
    if num_experts % n_group != 0:
        raise ValueError(
            f"Invalid configuration: num_experts ({num_experts}) must be divisible by "
            f"n_group ({n_group})"
        )
    if topk_group <= 0 or topk_group > n_group:
        raise ValueError(
            f"Invalid configuration: topk_group ({topk_group}) must be in [1, n_group "
            f"({n_group})]"
        )
    if topk <= 0:
        raise ValueError(f"Invalid configuration: topk ({topk}) must be > 0")

    # The selected groups expose ``topk_group * num_experts / n_group`` experts,
    # and the kernel returns the top ``topk`` of exactly those candidates.
    reachable_experts = topk_group * (num_experts // n_group)
    if topk > reachable_experts:
        raise ValueError(
            f"Invalid configuration: topk ({topk}) must be <= the number of experts "
            f"reachable from the topk_group ({topk_group}) selected groups "
            f"({reachable_experts})"
        )

    # Check kernel limits based on number of groups
    if n_group > 1:
        experts_per_group = num_experts // n_group
        max_experts_in_selected_groups = experts_per_group * topk_group

        # ``deepseek_v3_topk_kernel`` launches ``NumDeepseekExperts / WARP_SIZE``
        # warps of the grouped schedule and keeps one group score per warp, so at
        # most 8 groups can be scored, and it tracks at most ``MaxNumTopGroups``
        # selected groups.
        if n_group > 8:
            raise ValueError(
                f"Invalid configuration for n_group > 1: n_group ({n_group}) must be <= 8"
            )
        if topk_group > 4:
            raise ValueError(
                f"Invalid configuration for n_group > 1: topk_group ({topk_group}) must be "
                f"<= 4"
            )
        # A group score is the sum of the two best experts of a group, which stays
        # finite only for at least two experts per group.
        if experts_per_group < 2:
            raise ValueError(
                f"Invalid configuration for n_group > 1: num_experts / n_group "
                f"({experts_per_group}) must be >= 2"
            )
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
def get_dsv3_fused_routing_module(backend: str = "default"):
    module = gen_dsv3_fused_routing_module(backend=backend).build_and_load()

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


@backend_requirement(
    {
        "default": _check_default_dsv3_fused_routing_backend_supported,
        "cake": _check_cake_dsv3_fused_routing_backend_supported,
    },
    common_check=_check_dsv3_fused_routing_supported,
)
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
    *,
    backend: Literal["default", "cake"] = "default",
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
        Number of expert groups.  Must satisfy ``1 <= n_group <= 8`` and
        ``num_experts % n_group == 0``.  The grouped kernel keeps one group
        score per warp of its launch, so at most 8 groups can be scored.
        Typical value is 8 for DeepSeek-V3 with 256 experts (32 experts per
        group).
    topk_group : int
        Number of top groups to select.  Must satisfy ``1 <= topk_group <=
        min(n_group, 4)`` and ``topk <= (num_experts / n_group) *
        topk_group``; the kernel tracks at most 4 selected groups.  Typical
        value is 4.
    topk : int
        Number of top experts to select per token.  Must satisfy ``1 <=
        topk <= 8`` and ``topk <= (num_experts / n_group) * topk_group``.
        Typical value is 8.

        Further per-branch constraints:

        - When ``n_group > 1``: ``2 <= num_experts / n_group <= 32`` (a
          group score sums the two best experts of the group) and
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
    backend : Literal["default", "cake"]
        Implementation backend. ``"default"`` preserves the existing
        FlashInfer implementation. ``"cake"`` explicitly selects the Cake
        kernel and raises when the call is outside its supported contract.
        For ``"cake"``, single-group routing requires ``topk == 1``.
        Defaults to ``"default"``.

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
    module = get_dsv3_fused_routing_module(backend=backend)
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
