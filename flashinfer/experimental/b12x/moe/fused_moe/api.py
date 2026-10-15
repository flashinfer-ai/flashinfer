"""Public surface for :mod:`b12x.moe.fused_moe`."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from ..._lib.gating import default_is_supported
from ..._lib.scratch import scratch_tensor
from ...preparation.types import FrozenMapping, Plan, require_prepared
from . import META
from ._compat import Caps
from ._impl import (
    TPMoEScratchPlan,
    TPMoEFP4Binding as Binding,
    TPMoERouteBinding as RouteBinding,
    TPMoESparseFP4Binding as SparseBinding,
    b12x_moe_fp4 as _run,
    build_tp_moe_route_binding,
    build_tp_moe_sparse_fp4_binding,
    clear_tp_moe_caches as clear_caches,
    plan_tp_moe_scratch as plan,
)
from .config import TrellisConfig
from ._routing_ops import route_topk_out
from ._tuning import MoeDecodeConfig, MoeDecodeQuery
from ._preparation import (
    FC2Invocation,
    RouteTopKInvocation,
    plan_fc2 as _plan_fc2,
    plan_route_topk as _plan_route_topk,
    variant_for,
)
from .execution import ExecutionCapacity, RoutingSpec, plan_execution as _plan_execution
from .planning import (
    ActivationMode,
    ActivationSpec,
    MoEGeometry,
    WeightPlan,
    WeightPlanConstraints,
    plan_weights as _plan_weights,
    prepare_weights as _prepare_weights,
)
from .source import (
    PackedSource,
    PackedSourceFormat,
    TrellisExtent,
    TrellisSource,
    W13Layout,
    WeightSource,
)
from .trellis_layout import TrellisStaging
from .weights import (
    PackedWeights,
    CsfBlockPlanes,
    Nvfp4CsfWeights,
    Mxfp4CsfWeights,
    IQ2XSWeights,
    BlockQuantWeights,
    PreparedExperts,
    PreparedWeightFormat,
    ScaleEncoding,
    ScaleFactors,
    TrellisWeights,
    WeightEncoding,
    WeightPacking,
)


def plan_weights(**kwargs):
    """Plan typed checkpoint sources or the tensor-based quantization contract."""
    if "quant_modes" in kwargs:
        if "source" in kwargs or "geometry" in kwargs:
            raise TypeError("quant_modes cannot be combined with source or geometry")
        from ._impl import plan_b12x_fp4_moe_weights

        return plan_b12x_fp4_moe_weights(**kwargs)
    return _canonical_plan_weights(**kwargs)


def prepare_weights(**kwargs):
    """Prepare typed sources or tensor arguments using their weight plan."""
    if isinstance(kwargs.get("plan"), WeightPlan):
        return _canonical_prepare_weights(**kwargs)
    from ._impl import prepare_b12x_fp4_moe_weights

    return prepare_b12x_fp4_moe_weights(**kwargs)


def plan_execution(**kwargs):
    """Declare session preparation or size a heuristic tensor-based invocation."""
    if "weight_plan" in kwargs:
        if "experts" in kwargs or "capacity" in kwargs:
            raise TypeError("weight_plan cannot be combined with experts or capacity")
        from ._compat import plan_execution as legacy_plan_execution

        return legacy_plan_execution(**kwargs)
    return _canonical_plan_execution(**kwargs)


def _canonical_plan_weights(
    *,
    source: WeightSource,
    activation: ActivationSpec,
    geometry: MoEGeometry,
    constraints: WeightPlanConstraints | None = None,
) -> WeightPlan:
    """Declare canonical checkpoint representation and preparation only."""
    return _plan_weights(
        source=source,
        activation=activation,
        geometry=geometry,
        constraints=constraints,
    )


def _canonical_prepare_weights(
    *,
    plan: WeightPlan,
    weights: PackedWeights
    | TrellisWeights
    | IQ2XSWeights
    | Mxfp4CsfWeights
    | Nvfp4CsfWeights,
    device: torch.device | str | None = None,
    staging: TrellisStaging | None = None,
    scale_scratch: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> PreparedExperts:
    """Prepare the canonical weight representation owned by this layer."""
    return _prepare_weights(
        plan=plan,
        weights=weights,
        device=device,
        staging=staging,
        scale_scratch=scale_scratch,
    )


def _canonical_plan_execution(
    *,
    experts: PreparedExperts,
    capacity: ExecutionCapacity,
    routing: RoutingSpec | None = None,
    invocation: FrozenMapping = FrozenMapping(),
    override: MoeDecodeConfig | None = None,
    scale_expansion=None,
):
    """Declare capacity variants; preparation publishes executable states."""
    return _plan_execution(
        experts=experts,
        capacity=capacity,
        routing=routing,
        invocation=invocation,
        override=override,
        scale_expansion=scale_expansion,
    )


def plan_scale_expansion(
    experts, *, routes, ids_dtype=torch.int32, invocation=FrozenMapping()
):
    """Declare native scale expansion; MoE requests depend on this prepared plan."""
    from b12x._lib.quant.csf_blocks_tuning import plan_csf_block_expansion

    decoder = experts._impl.nvfp4_csf
    if decoder is None:
        raise ValueError("Scale expansion requires compressed expert scales")
    return plan_csf_block_expansion(
        decoder.first,
        decoder.second,
        routes=routes,
        ids_dtype=ids_dtype,
        invocation=invocation,
    )


def plan_route_topk(
    invocation: RouteTopKInvocation,
    *,
    invocation_metadata: FrozenMapping = FrozenMapping(),
    override=None,
):
    """Declare a standalone native top-k route operation."""
    return _plan_route_topk(
        invocation, declaration_invocation=invocation_metadata, override=override
    )


def plan_fc2(
    *,
    experts: PreparedExperts,
    invocation: FC2Invocation,
    invocation_metadata: FrozenMapping = FrozenMapping(),
    override=None,
):
    """Declare standalone route-major W4A16 FC2 without a full MoE plan."""
    return _plan_fc2(
        experts,
        invocation,
        declaration_invocation=invocation_metadata,
        override=override,
    )


def bind(plan: Plan, **kwargs: Any) -> Binding:
    """Bind live tensors within a prepared or heuristic scratch capacity."""
    if isinstance(plan, TPMoEScratchPlan):
        return plan.bind(**kwargs)
    state = require_prepared(plan, "moe.decode")
    scratch = scratch_tensor(kwargs["scratch"], plan.scratch_specs(), owner="TP MoE")
    return replace(
        state.bind(**kwargs),
        plan=plan,
        compiler_scratch=scratch,
        compiler_a4_prefill=kwargs.get("a4_prefill"),
    )


def run(*, binding: Binding):
    """Run a binding through its prepared or heuristic native launch path."""
    plan = binding.plan
    if plan is not None:
        require_prepared(plan, "moe.decode", binding.a.device)
    if (
        plan is not None
        and binding.compiler_scratch is not None
        and binding.output is not None
    ):
        from torch.fx.experimental.proxy_tensor import get_proxy_mode

        if torch.compiler.is_compiling() or get_proxy_mode() is not None:
            from ._binding_ops import prepared_moe_out

            output = (
                binding.output_cast_target
                if binding.output_cast_target is not None
                else binding.output
            )
            prepared_moe_out(
                binding.a,
                binding.topk_weights,
                binding.topk_ids,
                output,
                binding.compiler_scratch,
                plan._handle,
                binding.input_scales_static,
                binding.fast_math,
                binding.activation_amax,
                binding.layer_idx,
                binding.route_expert_map,
                binding.output_expert_map,
                binding.scales_expanded,
                binding.compiler_a4_prefill,
            )
            return output
    return _run(binding=binding)


def expand_scales(experts: PreparedExperts) -> bool:
    """Expand every expert's NVFP4-CSF scales into its scratch on the experts' device stream.

    A later ``bind(..., scales_expanded=True)`` for these experts skips the
    per-call expansion. Callers that share one scratch across layers may run
    this for the next layer on a side stream while other work proceeds; they
    order it after the previous user of the scratch and before the bound call.
    Micro-kernel calls still expand scales to reset their synchronization state.
    Returns False, doing nothing, for experts without NVFP4-CSF scales.
    """
    impl = experts._impl
    decoder = getattr(impl, "nvfp4_csf", None)
    if decoder is None:
        return False
    with torch.cuda.device(experts.device):
        decoder.decode_all(impl.w1_blockscale, impl.w2_blockscale)
    return True


def uses_expanded_nvfp4_scales(
    plan: Plan,
    *,
    num_tokens: int,
    a4_prefill: bool | None = None,
    route_ids_dtype: torch.dtype = torch.int32,
    has_route_map: bool = False,
    collect_activation_amax: bool = False,
    scales_expanded: bool = False,
) -> bool:
    """Whether a prepared invocation consumes the shared NVFP4-CSF expansion.

    This metadata-only query uses the same variant, precision, scratch-fit,
    and scale-consumer selection as binding. It does not prepare a plan or
    launch GPU work. A caller may skip scale prefetch when it returns False.
    Route mappings and activation calibration must describe the later bind.
    ``scales_expanded=True`` queries a call whose scale expansion is already
    ordered before execution, or a prospective call for which the caller will
    arrange that ordering. Large eligible A4 calls can then consume the shared
    expansion instead of decoding scale tiles in shared memory.
    """
    if getattr(plan, "_prepared", None) is None:
        raise RuntimeError("scale-consumer queries require a prepared MoE plan")
    state = require_prepared(plan, "moe.decode")
    if hasattr(state, "for_tokens"):
        state = state.for_tokens(
            num_tokens,
            a4_prefill=a4_prefill,
            route_ids_dtype=route_ids_dtype,
            has_route_map=has_route_map,
            collect_activation_amax=collect_activation_amax,
            scales_expanded=scales_expanded,
        )
    return state.uses_expanded_nvfp4_scales(
        num_tokens=num_tokens,
        a4_prefill=a4_prefill,
        route_ids_dtype=route_ids_dtype,
        has_route_map=has_route_map,
        collect_activation_amax=collect_activation_amax,
        scales_expanded=scales_expanded,
    )


def _state_for(plan: Plan, hidden_states: torch.Tensor):
    root = require_prepared(plan, "moe.decode", hidden_states.device)
    if hasattr(root, "variants"):
        return variant_for(root.variants, hidden_states.shape[0])
    return root


def route_topk(
    plan: Plan,
    router_logits: torch.Tensor,
    topk_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    **kwargs: Any,
) -> None:
    """Run caller-owned top-k buffers through their retained route launcher."""
    state = require_prepared(plan, "moe.route_topk", router_logits.device)
    for name in ("renormalize", "score_func", "routed_scaling_factor"):
        expected = getattr(state.query, name)
        if kwargs.pop(name, expected) != expected:
            raise ValueError(f"{name} differs from the prepared route invocation")
    correction_bias = kwargs.pop("correction_bias", None)
    image_correction_bias = kwargs.pop("image_correction_bias", None)
    image_mask = kwargs.pop("image_mask", None)
    if kwargs:
        raise TypeError(f"unexpected route_topk arguments: {', '.join(kwargs)}")
    route_topk_out(
        router_logits,
        topk_logits,
        topk_ids,
        topk_weights,
        plan.handle,
        correction_bias,
        image_correction_bias,
        image_mask,
    )


def bind_route(
    plan: Plan, *, hidden_states: torch.Tensor, **kwargs: Any
) -> RouteBinding:
    """Bind native routing to an exact prepared MoE plan."""
    _state_for(plan, hidden_states)
    scratch = kwargs.pop("scratch")
    return replace(
        build_tp_moe_route_binding(
            scratch=scratch,
            hidden_states=hidden_states,
            **kwargs,
        ),
        plan=plan,
    )


def route(plan: Plan, *, binding: RouteBinding) -> object:
    """Run a route binding through the selected prepared route launcher."""
    if binding.plan is not plan:
        raise ValueError("route binding belongs to another prepared plan")
    return _state_for(plan, binding.hidden_states).route(binding)


def bind_sparse(
    plan: Plan, *, hidden_states: torch.Tensor, **kwargs: Any
) -> SparseBinding:
    """Bind sparse MoE math to an exact prepared native plan."""
    state = _state_for(plan, hidden_states)
    scratch = kwargs.pop("scratch")
    return replace(
        build_tp_moe_sparse_fp4_binding(
            scratch=scratch,
            hidden_states=hidden_states,
            experts=state.experts._impl,
            **kwargs,
        ),
        plan=plan,
    )


def run_sparse(plan: Plan, *, binding: SparseBinding):
    """Run sparse MoE through its prepared route and expert launchers."""
    if binding.plan is not plan:
        raise ValueError("sparse binding belongs to another prepared plan")
    return _state_for(plan, binding.hidden_states).run_sparse(binding)


def run_fc2(
    plan: Plan,
    intermediate: torch.Tensor,
    route_expert_ids: torch.Tensor,
    route_weights: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run standalone FC2 through its retained prepared native launcher."""
    state = require_prepared(plan, "moe.fc2_w4a16", intermediate.device)
    return state.run(intermediate, route_expert_ids, route_weights, output=output)


def is_supported(device=None) -> bool:
    """Return whether the active device satisfies the fused-MoE requirements."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "clear_caches",
    "Caps",
    "plan",
    "TrellisExtent",
    "TrellisSource",
    "TrellisStaging",
    "ActivationMode",
    "ActivationSpec",
    "Binding",
    "RouteBinding",
    "RouteTopKInvocation",
    "FC2Invocation",
    "SparseBinding",
    "MoEGeometry",
    "MoeDecodeConfig",
    "MoeDecodeQuery",
    "PackedSource",
    "PackedSourceFormat",
    "PackedWeights",
    "Mxfp4CsfWeights",
    "CsfBlockPlanes",
    "Nvfp4CsfWeights",
    "IQ2XSWeights",
    "BlockQuantWeights",
    "PreparedExperts",
    "PreparedWeightFormat",
    "RoutingSpec",
    "ScaleEncoding",
    "ScaleFactors",
    "TrellisConfig",
    "TrellisWeights",
    "W13Layout",
    "WeightEncoding",
    "WeightPacking",
    "WeightPlan",
    "WeightPlanConstraints",
    "WeightSource",
    "bind",
    "bind_route",
    "bind_sparse",
    "expand_scales",
    "uses_expanded_nvfp4_scales",
    "is_supported",
    "plan_execution",
    "plan_scale_expansion",
    "plan_route_topk",
    "plan_fc2",
    "plan_weights",
    "prepare_weights",
    "route_topk",
    "route",
    "run_fc2",
    "run_sparse",
    "run",
]
