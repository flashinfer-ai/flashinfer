"""Prepared declarations for replicated-input expert-parallel W4A16 MoE."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import FrozenMapping, MemoryRequirements, PersistentMemory, Plan
from b12x.preparation.types import _CompositePlan
from b12x.moe._shared.kernels.w4a16.host import select_route_block_size_m
from b12x.moe._shared.kernels.w4a16.kernel import compile_w4a16_fused_moe
from ..fused_moe._impl import B12XFP4ExpertWeights, plan_b12x_fp4_moe_weights
from ._impl import EPExpertMap, EPMoEFP4Binding, EPMoEScratchCaps, _materialize_layout
from ._tuning import EpMoeConfig, EpMoeQuery, TUNING


def _alignment(tensor: torch.Tensor) -> int:
    pointer = tensor.data_ptr()
    return min(16, pointer & -pointer) if pointer else 16


def invocation_from_tensors(*, a: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                            output: torch.Tensor, expert_map: EPExpertMap, **unused) -> FrozenMapping:
    """Capture immutable tensor layout/dtype metadata from the actual EP owners."""
    if not isinstance(expert_map, EPExpertMap):
        raise TypeError("expert_map must come from prepare_ep_expert_map")
    return FrozenMapping({
        "input_dtype": str(a.dtype).removeprefix("torch."),
        "route_ids_dtype": str(topk_ids.dtype).removeprefix("torch."),
        "route_weights_dtype": str(topk_weights.dtype).removeprefix("torch."),
        "output_dtype": str(output.dtype).removeprefix("torch."),
        "alignments": tuple(_alignment(tensor) for tensor in (a, topk_ids, topk_weights, output, expert_map.tensor)),
        "expert_map_ptr": expert_map.tensor.data_ptr(),
        "expert_map_version": expert_map._version,
    })


def _weight_payload(experts: B12XFP4ExpertWeights) -> dict[str, object]:
    plan = experts.plan
    return {
        "quant_modes": tuple(plan.quant_modes), "source_format": plan.source_format,
        "activation": plan.activation, "params_dtype": plan.io_dtype,
        "num_experts": plan.num_experts, "hidden_size": plan.hidden_size,
        "intermediate_size": plan.intermediate_size, "w13_layout": plan.w13_layout,
        "w4a16_layout": plan.w4a16_weight_layout, "trellis_bits": plan.trellis_bits,
        "trellis_tile_config": plan.trellis_tile_config, "intermediate_hadamard": plan.intermediate_hadamard,
        "trellis_codebook": plan.trellis_codebook, "trellis_rate_granularity": plan.trellis_rate_granularity,
        "trellis_pair_kinds": None if plan.trellis_pair_kinds is None else tuple(plan.trellis_pair_kinds),
        "intermediate_hadamard_blocks": plan.intermediate_hadamard_blocks,
    }


def _prepared_metadata(weight_payload):
    args = dict(weight_payload)
    args["params_dtype"] = getattr(torch, str(args["params_dtype"]).removeprefix("torch."))
    return plan_b12x_fp4_moe_weights(**args)


def _compile_arguments(query: EpMoeQuery, invocation: FrozenMapping, weight_payload, ordinal: int):
    prepared = _prepared_metadata(weight_payload)
    route_ids_dtype = getattr(torch, query.route_ids_dtype)
    direct = query.num_tokens <= 8 and route_ids_dtype in (torch.int32, torch.int64)
    block_size = select_route_block_size_m(query.num_tokens, query.top_k, query.num_experts)
    route_slots = query.num_tokens * query.top_k * block_size
    max_m_blocks = query.num_tokens * query.top_k if direct else (route_slots + block_size - 1) // block_size
    with torch.cuda.device(torch.device("cuda", ordinal)):
        props = torch.cuda.get_device_properties(ordinal)
        return compile_w4a16_fused_moe(
            size_m=query.num_tokens, hidden_size=query.hidden_size,
            intermediate_size=query.intermediate_size, num_experts=query.local_num_experts,
            top_k=query.top_k, activation=query.activation,
            apply_router_weight_on_input=query.apply_router_weight_on_input,
            zero_fc2_output=not direct, moe_block_size=block_size,
            max_m_blocks=max_m_blocks, element_dtype="bf16", fast_math=query.fast_math,
            sms=int(props.multi_processor_count), max_shared_mem=int(props.shared_memory_per_block_optin),
            swiglu_limit=query.swiglu_limit, swiglu_alpha=query.swiglu_alpha,
            swiglu_beta=query.swiglu_beta, weight_layout=getattr(prepared, "weight_layout", "packed"),
            scale_format=getattr(prepared, "scale_format", "e4m3_k16"),
            w13_layout=getattr(prepared, "w13_layout", "w13"),
            trellis_bits=int(getattr(prepared, "trellis_bits", 3)),
            trellis_codebook=str(getattr(prepared, "trellis_codebook", "lut_e4m3")),
            fc1_trellis_pair_kind=getattr(prepared, "fc1_trellis_pair_kind", None),
            fc2_trellis_pair_kind=getattr(prepared, "fc2_trellis_pair_kind", None),
            direct_topk_routes=direct, use_expert_map=direct,
            force_tile_config=getattr(prepared, "tile_config", None),
        )


@program_cache(scope="preparation")
def compile_ep_moe(query_payload, invocation_payload, weight_payload, ordinal):
    """Compile the real W4A16 EP route/GEMM/reduction launch from metadata only."""
    return _compile_arguments(EpMoeQuery(**dict(query_payload)), FrozenMapping(invocation_payload), weight_payload, ordinal)


@dataclass(frozen=True)
class _EPState:
    caps: EPMoEScratchCaps
    expert_map: EPExpertMap
    layout: object
    launch: object
    query: EpMoeQuery
    route_ids_workspace: bool

    def bind(self, **kwargs):
        plan = kwargs.pop("_plan", None)
        experts = kwargs.pop("experts", None)
        if not isinstance(experts, B12XFP4ExpertWeights):
            raise TypeError("experts must come from prepare_b12x_fp4_moe_weights")
        if experts.plan != self.caps.weight_plan:
            raise ValueError("experts do not match the prepared EP weight plan")
        supplied_map = kwargs.pop("expert_map", self.expert_map)
        if supplied_map is not self.expert_map:
            raise ValueError("prepared EP plan requires its declared expert_map owner")
        return self.layout.bind(
            experts=experts, expert_map=self.expert_map, _launch=self.launch,
            _route_ids_workspace=self.route_ids_workspace,
            _plan=plan, **kwargs,
        )

    def run(self, binding):
        from ._impl import _run_bound_ep_moe
        return _run_bound_ep_moe(binding)


@dataclass(frozen=True)
class _EPCapacityState:
    variants: MappingProxyType

    def bind(self, **kwargs):
        a = kwargs.get("a")
        if not isinstance(a, torch.Tensor) or a.ndim != 2:
            raise TypeError("EP MoE binding requires rank-two activations")
        try:
            return self.variants[int(a.shape[0])].bind(**kwargs)
        except KeyError:
            raise ValueError(f"token count {int(a.shape[0])} was not prepared") from None

    def run(self, binding):
        return binding.run()


def _query(caps: EPMoEScratchCaps, expert_map: EPExpertMap, tokens: int, invocation: FrozenMapping):
    allowed = {"input_dtype", "route_ids_dtype", "route_weights_dtype", "output_dtype", "alignments", "expert_map_ptr", "expert_map_version", "token_counts", "fast_math"}
    if set(invocation) - allowed:
        raise ValueError("unknown EP MoE invocation metadata")
    if invocation.get("input_dtype", "bfloat16") != "bfloat16" or invocation.get("output_dtype", "bfloat16") != "bfloat16":
        raise TypeError("replicated-input EP requires BF16 activations and output")
    route_dtype = str(invocation.get("route_ids_dtype", "int32"))
    if route_dtype not in {"int32", "int64"}:
        raise TypeError("EP topk_ids must have dtype torch.int32 or torch.int64")
    if invocation.get("route_weights_dtype", "float32") != "float32":
        raise TypeError("EP topk_weights must have dtype torch.float32")
    if invocation.get("expert_map_ptr", expert_map.tensor.data_ptr()) != expert_map.tensor.data_ptr() or invocation.get("expert_map_version", expert_map._version) != expert_map._version:
        raise ValueError("EP invocation expert_map identity differs from declared owner")
    return EpMoeQuery(
        max_tokens=caps.max_tokens, num_tokens=tokens, top_k=caps.num_topk,
        num_experts=caps.global_num_experts, local_num_experts=caps.local_num_experts,
        hidden_size=caps.weight_plan.hidden_size, intermediate_size=caps.weight_plan.intermediate_size,
        activation=caps.weight_plan.activation, apply_router_weight_on_input=caps.apply_router_weight_on_input,
        swiglu_limit=caps.swiglu_limit, swiglu_alpha=float(caps.swiglu_alpha), swiglu_beta=float(caps.swiglu_beta),
        route_ids_dtype=route_dtype, fast_math=bool(invocation.get("fast_math", True)),
    )


def plan(caps: EPMoEScratchCaps, *, expert_map: EPExpertMap, invocation: FrozenMapping = FrozenMapping(), override: EpMoeConfig | None = None):
    """Declare exact EP specializations; compilation and scratch allocation are session work."""
    if not isinstance(caps, EPMoEScratchCaps):
        raise TypeError("caps must be an EPMoEScratchCaps")
    if not isinstance(expert_map, EPExpertMap):
        raise TypeError("expert_map must come from prepare_ep_expert_map")
    expert_map.validate_static()
    if expert_map.global_num_experts != caps.global_num_experts or expert_map.local_num_experts != caps.local_num_experts or expert_map.device != caps.device:
        raise ValueError("expert_map does not match EP capacity/weight ownership")
    invocation = FrozenMapping(invocation)
    requested = invocation.get("token_counts", ())
    if not isinstance(requested, tuple):
        raise TypeError("EP token_counts invocation metadata must be a tuple")
    counts = tuple(sorted({caps.max_tokens, *(int(value) for value in requested)}))
    if any(count <= 0 or count > caps.max_tokens for count in counts):
        raise ValueError("EP token_counts must be positive and within capacity")
    weights = _weight_payload_placeholder(caps)

    def child(tokens):
        query = _query(caps, expert_map, tokens, invocation)
        def compile_jobs(config, device):
            return (CompileJob.create("b12x.moe.ep_moe._preparation:compile_ep_moe", TUNING.encode_query(query), invocation.to_dict(), weights, device.ordinal),)
        route_ids_workspace = (
            query.route_ids_dtype == "int64" and query.num_tokens <= 8
        )
        def memory(config, device):
            layout = _materialize_layout(
                EPMoEScratchCaps(**{**caps.__dict__, "max_tokens": tokens}),
                route_ids_workspace=route_ids_workspace,
            )
            return MemoryRequirements(scratch=layout.scratch_specs(), persistent=(PersistentMemory(key=("ep_expert_map", expert_map.tensor.data_ptr()), required_nbytes=expert_map.tensor.numel() * expert_map.tensor.element_size(), resident_nbytes=expert_map.tensor.numel() * expert_map.tensor.element_size()),))
        def materialize(selection, device):
            child_caps = EPMoEScratchCaps(**{**caps.__dict__, "max_tokens": tokens})
            layout = _materialize_layout(
                child_caps, route_ids_workspace=route_ids_workspace,
            )
            launch = _compile_arguments(query, invocation, weights, device.ordinal)
            return _EPState(
                child_caps, expert_map, layout, launch, query, route_ids_workspace,
            )
        return Plan(
            contract=TUNING, query=query, invocation=invocation, override=override,
            _compile_jobs=compile_jobs, _memory_requirements=memory,
            _materialize=materialize, _device=caps.device,
        )

    children = {tokens: child(tokens) for tokens in counts}
    if len(counts) == 1:
        return children[counts[0]]
    def assemble(states, device):
        del device
        return _EPCapacityState(MappingProxyType({tokens: states[tokens] for tokens in counts}))
    return _CompositePlan(component_id="moe.ep_moe", capacity_metadata=FrozenMapping({"caps": caps.max_tokens, "expert_map": expert_map.tensor.data_ptr(), "invocation": invocation}), variants=children, _assemble=assemble)


def _weight_payload_placeholder(caps):
    # The declaration holds only the immutable weight-plan metadata; tensors stay caller-owned.
    return {
        "quant_modes": tuple(caps.weight_plan.quant_modes), "source_format": caps.weight_plan.source_format,
        "activation": caps.weight_plan.activation, "params_dtype": caps.weight_plan.io_dtype,
        "num_experts": caps.weight_plan.num_experts, "hidden_size": caps.weight_plan.hidden_size,
        "intermediate_size": caps.weight_plan.intermediate_size, "w13_layout": caps.weight_plan.w13_layout,
        "w4a16_layout": caps.weight_plan.w4a16_weight_layout, "trellis_bits": caps.weight_plan.trellis_bits,
        "trellis_tile_config": caps.weight_plan.trellis_tile_config, "intermediate_hadamard": caps.weight_plan.intermediate_hadamard,
        "trellis_codebook": caps.weight_plan.trellis_codebook, "trellis_rate_granularity": caps.weight_plan.trellis_rate_granularity,
        "trellis_pair_kinds": None if caps.weight_plan.trellis_pair_kinds is None else tuple(caps.weight_plan.trellis_pair_kinds),
        "intermediate_hadamard_blocks": caps.weight_plan.intermediate_hadamard_blocks,
    }
