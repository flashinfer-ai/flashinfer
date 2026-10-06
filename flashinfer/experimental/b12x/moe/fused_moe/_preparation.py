"""Preparation declarations for canonical fused tensor-parallel MoE."""
from __future__ import annotations

from b12x._lib.quant.block_codec import BLOCK_CODECS

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import torch
from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation.types import FrozenMapping, MemoryRequirements, Plan, _CompositePlan

from ._impl import TPMoEScratchCaps, plan_b12x_fp4_moe_weights, plan_tp_moe_scratch
from ._tuning import FC2_TUNING, ROUTE_TUNING, MoeDecodeConfig, MoeDecodeQuery, TUNING
from .planning import ActivationMode, _packed_recipe
from .weights import PreparedExperts
from .execution import ExecutionCapacity, RoutingSpec


@dataclass(frozen=True, kw_only=True)
class RouteTopKInvocation:
    """Complete, immutable ABI for one top-k routing operation."""
    num_tokens: int
    num_experts: int
    top_k: int
    logits_dtype: str
    score_func: str = "softmax"
    renormalize: bool = True
    has_correction_bias: bool = False
    has_image_correction_bias: bool = False
    has_image_mask: bool = False
    routed_scaling_factor: float = 1.0
    logits_row_stride: int | None = None
    topk_row_stride: int | None = None
    ids_row_stride: int | None = None
    weights_row_stride: int | None = None

    def query(self):
        from ._tuning import MoeRouteQuery
        return MoeRouteQuery(
            num_tokens=self.num_tokens, num_experts=self.num_experts, top_k=self.top_k,
            logits_dtype=self.logits_dtype, score_func=self.score_func,
            renormalize=self.renormalize, has_correction_bias=self.has_correction_bias,
            has_image_correction_bias=self.has_image_correction_bias,
            has_image_mask=self.has_image_mask,
            routed_scaling_factor=self.routed_scaling_factor,
            logits_row_stride=self.logits_row_stride or self.num_experts,
            topk_row_stride=self.topk_row_stride or self.top_k,
            ids_row_stride=self.ids_row_stride or self.top_k,
            weights_row_stride=self.weights_row_stride or self.top_k,
        )


@dataclass(frozen=True, kw_only=True)
class FC2Invocation:
    """Complete, immutable ABI for standalone route-major W4A16 FC2."""
    max_routes: int
    route_ids_dtype: str = "int32"

def _codec_scalar(value):
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("MoE numerical controls cannot be NaN")
        if not math.isfinite(value):
            return "+inf" if value > 0 else "-inf"
    return value


def _decode_scalar(value):
    return {"+inf": float("inf"), "-inf": float("-inf")}.get(value, value)


def _quant_mode(experts: PreparedExperts, config: MoeDecodeConfig) -> str:
    modes = experts.plan._impl.quant_modes
    if experts.plan.activation.mode is ActivationMode.AUTO:
        return "w4a16" if config.backend == "w4a16" else "nvfp4"
    # A canonical weight plan may intentionally retain more than one recipe.
    # The selected backend determines the A16 route; all other retained recipes
    # use their declared concrete mode rather than assuming an arbitrary set order.
    if config.backend == "w4a16":
        if "w4a16" not in modes:
            raise ValueError("selected W4A16 backend is absent from the weight plan")
        return "w4a16"
    non_a16 = tuple(mode for mode in modes if mode != "w4a16")
    if len(non_a16) != 1:
        raise ValueError("non-W4A16 fused MoE selection requires one concrete quantization recipe")
    return non_a16[0]


def _control_snapshot() -> FrozenMapping:
    """Capture host controls once while declaring the immutable query."""
    from . import _impl
    from b12x.moe._shared.kernels.w4a16.host import (
        prefill_fused_sum_enabled, trellis_decode_table,
    )

    tile = _impl._dynamic_tile_mn_override()
    raw_materialized = _impl.os.environ.get(_impl._DYNAMIC_NVFP4_MATERIALIZED_ENV)
    return FrozenMapping({
        "trellis_decode_table": trellis_decode_table(),
        "w4a16_prefill_fused_sum": prefill_fused_sum_enabled(),
        "w4a16_skip_empty_m_blocks": _impl.os.environ.get(
            "B12X_W4A16_SKIP_EMPTY_M_BLOCKS", "1"
        ) == "1",
        "dynamic_nvfp4_materialized": (
            None if raw_materialized is None else raw_materialized not in ("", "0", "false", "False")
        ),
        "dynamic_down_scale": _impl._dynamic_down_scale_enabled(),
        "dynamic_swap_ab": _impl._DYNAMIC_SWAP_AB_OVERRIDE,
        "dynamic_tile_mn": None if tile is None else tuple(int(v) for v in tile),
        "dynamic_work_source": _impl._dynamic_work_source(),
        "dynamic_external_route_plan": _impl._env_flag(
            _impl._DYNAMIC_EXTERNAL_ROUTE_PLAN_ENV, default=False
        ),
        "dynamic_w4a8_repacked": _impl._env_flag(
            _impl._DYNAMIC_W4A8_REPACKED_ENV, default=False
        ),
        "dynamic_w4a8_share_input": _impl._env_flag(
            _impl._DYNAMIC_W4A8_SHARE_INPUT_ENV, default=False
        ),
        "dynamic_w4a8_materialized": _impl._env_flag(
            _impl._DYNAMIC_W4A8_MATERIALIZED_ENV, default=False
        ),
    })


def _query(
    experts: PreparedExperts,
    capacity: ExecutionCapacity,
    tokens: int,
    routing: RoutingSpec,
    controls: FrozenMapping,
    invocation: FrozenMapping,
) -> MoeDecodeQuery:
    plan = experts.plan._impl
    quant_modes = tuple(sorted(plan.quant_modes))
    quant_mode = (
        "nvfp4_auto" if experts.plan.activation.mode is ActivationMode.AUTO
        else (quant_modes[0] if len(quant_modes) == 1 else "multi")
    )
    cutoff = experts.plan.activation.a16_max_tokens
    if cutoff:
        if tokens <= cutoff:
            quant_mode, quant_modes = "w4a16", ("w4a16",)
        elif experts.plan.activation.mode is not ActivationMode.AUTO:
            quant_mode = _packed_recipe(experts.plan.source, experts.plan.activation.mode)
            quant_modes = (quant_mode,)
    deterministic_output = routing.deterministic_output
    if deterministic_output is None:
        from ._impl import _dynamic_deterministic_output_enabled

        # Candidate eligibility and the compiled kernel must see the same
        # reduction contract, including environment-selected determinism.
        deterministic_output = any(
            _dynamic_deterministic_output_enabled(quant_mode=mode, device=experts.device)
            for mode in quant_modes
        )
    return MoeDecodeQuery(
        quant_mode=quant_mode,
        quant_modes=quant_modes,
        source_format=plan.source_format,
        activation=plan.activation,
        io_dtype=plan.io_dtype,
        num_experts=experts.num_experts,
        hidden_size=experts.hidden_size,
        intermediate_size=experts.intermediate_size,
        top_k=capacity.top_k,
        num_tokens=tokens,
        routed_rows=tokens * capacity.top_k,
        route_num_experts=capacity.route_num_experts,
        route_logits_dtype=None if routing.logits_dtype is None else str(routing.logits_dtype).removeprefix("torch."),
        apply_router_weight_on_input=bool(routing.apply_router_weight_on_input),
        collect_activation_amax=bool(routing.collect_activation_amax),
        deterministic_output=deterministic_output,
        swiglu_limit=_codec_scalar(experts.plan.activation.swiglu_limit),
        swiglu_alpha=_codec_scalar(experts.plan.activation.swiglu_alpha),
        swiglu_beta=_codec_scalar(experts.plan.activation.swiglu_beta),
        w13_layout=plan.w13_layout,
        weight_layouts=tuple(sorted(layout.value for layout in plan.weight_layouts)),
        w4a16_weight_layout=plan.w4a16_weight_layout,
        w4a16_scale_format=plan.w4a16_scale_format,
        w4a16_block_size_m=invocation.get("w4a16_block_size_m"),
        fast_math=bool(invocation.get("fast_math", True)),
        numerical_recipe=invocation.get("numerical_recipe"),
        controls=controls,
        shared_input_scales=experts._impl.can_share_input(input_scales_static=True),
        nvfp4_inline_scales=plan.nvfp4_inline_scales,
    )


def _weight_payload(experts: PreparedExperts) -> dict[str, object]:
    """Serialize native planner inputs using its public layout enum values."""
    plan = experts.plan._impl
    w4a16_layout = (
        plan.required_weight_layout("w4a16")
        if "w4a16" in plan.quant_modes
        else None
    )
    return {
        "quant_modes": tuple(plan.quant_modes), "source_format": plan.source_format,
        "nvfp4_inline_scales": plan.nvfp4_inline_scales,
        "activation": plan.activation, "params_dtype": plan.io_dtype,
        "num_experts": plan.num_experts, "hidden_size": plan.hidden_size,
        "intermediate_size": plan.intermediate_size, "w13_layout": plan.w13_layout,
        "w4a16_layout": (
            None if w4a16_layout is None else w4a16_layout.value
        ),
        "trellis_bits": plan.trellis_bits,
        "trellis_tile_config": plan.trellis_tile_config,
        "intermediate_hadamard": plan.intermediate_hadamard,
        "trellis_codebook": plan.trellis_codebook,
        "trellis_rate_granularity": plan.trellis_rate_granularity,
        "trellis_pair_kinds": None if plan.trellis_pair_kinds is None else tuple(plan.trellis_pair_kinds),
        "intermediate_hadamard_blocks": plan.intermediate_hadamard_blocks,
    }


def _decode_query(payload) -> MoeDecodeQuery:
    values = dict(payload)
    values["controls"] = FrozenMapping(values["controls"])
    return MoeDecodeQuery(**values)



def _route_query_from_moe(query: MoeDecodeQuery, routing: RoutingSpec):
    from ._tuning import MoeRouteQuery
    return MoeRouteQuery(
        num_tokens=query.num_tokens,
        num_experts=query.route_num_experts or query.num_experts,
        top_k=query.top_k,
        logits_dtype=query.route_logits_dtype or "float32",
        score_func=routing.score_func, renormalize=routing.renormalize,
        has_correction_bias=routing.has_correction_bias,
        has_image_correction_bias=routing.has_image_correction_bias,
        has_image_mask=routing.has_image_mask,
        routed_scaling_factor=routing.routed_scaling_factor,
        logits_row_stride=query.route_num_experts or query.num_experts,
        topk_row_stride=query.top_k, ids_row_stride=query.top_k,
        weights_row_stride=query.top_k,
    )

def _lower_caps(
    query: MoeDecodeQuery,
    config: MoeDecodeConfig,
    weight_plan,
    device: torch.device,
) -> TPMoEScratchCaps:
    """One config-to-Caps lowering shared by compilation, sizing and serving."""
    mode = query.quant_mode
    if mode == "nvfp4_auto":
        mode = "w4a16" if config.backend == "w4a16" else "nvfp4"
    elif mode == "multi":
        if config.backend == "w4a16":
            mode = "w4a16"
        else:
            modes = tuple(item for item in weight_plan.quant_modes if item != "w4a16")
            if len(modes) != 1:
                raise ValueError("prepared multi-recipe MoE requires an explicit backend route")
            mode = modes[0]
    return TPMoEScratchCaps(
        max_tokens=query.num_tokens, num_topk=query.top_k, device=device,
        weight_plan=weight_plan, quant_mode=mode, decode_config=config,
        core_token_counts=(query.num_tokens,), route_num_experts=query.route_num_experts,
        route_logits_dtype=(None if query.route_logits_dtype is None else getattr(torch, query.route_logits_dtype)),
        apply_router_weight_on_input=query.apply_router_weight_on_input,
        collect_activation_amax=query.collect_activation_amax,
        deterministic_output=query.deterministic_output,
        swiglu_limit=_decode_scalar(query.swiglu_limit),
        swiglu_alpha=_decode_scalar(query.swiglu_alpha),
        w4a16_block_size_m=(config.w4a16_block_size_m
                            if config.w4a16_block_size_m is not None
                            else query.w4a16_block_size_m),
        w4a16_fast_math=query.fast_math,
        w4a16_prefill_fused_sum=bool(
            query.controls.get("w4a16_prefill_fused_sum", False)
        ),
        trellis_decode_table=str(query.controls.get("trellis_decode_table", "auto")),
        w4a16_skip_empty_m_blocks=bool(query.controls.get("w4a16_skip_empty_m_blocks", True)),
        swiglu_beta=_decode_scalar(query.swiglu_beta),
    )


@dataclass(frozen=True)
class _W4A16PrimaryLaunches:
    """Native W4A16 launches for a capacity and its exact direct route."""

    tokens: int
    route_mode: str
    packed: object
    packed_mapped: object
    direct: object | None
    direct_mapped: object | None
    topk_sum: object
    mapped_topk_sum: object
    route_pack: object | None

    def select(
        self,
        *,
        tokens: int,
        route_ids_dtype: torch.dtype,
        has_route_map: bool,
        activation_amax: object | None,
    ) -> tuple[object, object, object | None]:
        if not 0 < int(tokens) <= self.tokens:
            raise RuntimeError(
                "W4A16 execution exceeds its prepared capacity: "
                f"requested={int(tokens)}, prepared={self.tokens}"
            )
        native_direct = (
            int(tokens) == self.tokens
            and self.route_mode != "packed"
            and not has_route_map
            and activation_amax is None
            and any(
                launch.topk_ids_dtype == route_ids_dtype
                for launch in getattr(self.packed, "small_m_direct_launches", ())
            )
        )
        if native_direct:
            return self.packed, self.topk_sum, None
        direct = (
            int(tokens) == self.tokens
            and self.route_mode != "packed"
            and activation_amax is None
            and route_ids_dtype == torch.int32
            and (self.direct_mapped if has_route_map else self.direct) is not None
        )
        if direct:
            return (
                self.direct_mapped if has_route_map else self.direct,
                self.mapped_topk_sum if has_route_map else self.topk_sum,
                None,
            )
        if self.route_pack is None:
            raise RuntimeError("prepared W4A16 direct route has no packed-route fallback")
        return (
            self.packed_mapped if has_route_map else self.packed,
            self.topk_sum,
            self.route_pack,
        )

    def carriers(self) -> tuple[object, ...]:
        return tuple(
            launcher
            for launcher in (
                self.packed,
                self.packed_mapped,
                self.direct,
                self.direct_mapped,
                self.topk_sum,
                self.mapped_topk_sum,
                *(()
                  if self.route_pack is None
                  else self.route_pack.carriers()),
            )
            if launcher is not None
        )


def _w4a16_primary_launches(scratch, caps) -> _W4A16PrimaryLaunches:
    """Compile capacity launches and exact direct routing from metadata.

    This intentionally uses only immutable declaration metadata and compiler
    fake descriptors.  It must stay independent of caller scratch, routes, and
    bound expert tensors so compile admission never relies on a representative
    GPU invocation.
    """
    from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
    from b12x.moe._shared.kernels.w4a16.kernel import (
        _DEFAULT_MAX_SHARED_MEM,
        _MAX_DIRECT_TOPK_ROUTE_M,
        _W4A16_SMALL_M_DIRECT_MAX_M,
        compile_w4a16_fused_moe,
        compile_w4a16_topk_sum,
    )
    from b12x.moe._shared.kernels.w4a16.route_pack import (
        compile_w4a16_route_pack_launches,
    )

    core = scratch._core_workspace_plan
    if core.full_rotation or core.projection_mixed_trellis:
        raise RuntimeError("standard W4A16 compiler adapter received a Trellis plan")
    tokens = int(scratch.launch_plan.max_tokens_per_launch)
    weight_layout = caps.w4a16_weight_layout or "packed"
    scale_format = caps.w4a16_scale_format or "e4m3_k16"
    if weight_layout not in {"packed", "modelopt", "iq2_xs", "iq2_xxs", "q8_0"}:
        raise ValueError(f"unsupported standard W4A16 weight layout {weight_layout!r}")
    element_dtype = "bf16" if core.dtype == torch.bfloat16 else "fp16"
    w13_layout = caps.w13_layout if weight_layout == "modelopt" else "packed"
    route_block = core.route_block_size_m
    if route_block is None:
        from b12x.moe._shared.kernels.w4a16.host import select_route_block_size_m
        route_block = select_route_block_size_m(tokens, core.num_topk, core.route_E)
    _, _, packed_blocks = route_pack_capacity(
        tokens * core.num_topk, int(route_block), core.route_E,
        topk=core.num_topk,
    )
    with torch.cuda.device(core.device):
        props = torch.cuda.get_device_properties(core.device)
        compiler_args = dict(
            size_m=tokens, hidden_size=core.k, intermediate_size=core.n,
            direct_token_capacity=int(caps.max_tokens),
            num_experts=core.weight_E, top_k=core.num_topk,
            activation=core.activation,
            apply_router_weight_on_input=caps.apply_router_weight_on_input,
            moe_block_size=int(route_block), element_dtype=element_dtype,
            fast_math=caps.w4a16_fast_math,
            sms=int(props.multi_processor_count),
            max_shared_mem=int(getattr(
                props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM
            )),
            swiglu_limit=core.swiglu_limit, swiglu_alpha=core.swiglu_alpha,
            swiglu_beta=core.swiglu_beta, weight_layout=weight_layout,
            scale_format=scale_format, w13_layout=w13_layout,
            force_tile_config=caps.decode_config.w4a16_tile_config,
            pipeline_stages=caps.decode_config.w4a16_pipeline_stages,
            skip_empty_m_blocks=caps.w4a16_skip_empty_m_blocks,
        )
        packed = compile_w4a16_fused_moe(
            **compiler_args, zero_fc2_output=False, max_m_blocks=packed_blocks,
            prefill_fused_sum_fp32=core.prefill_fused_sum_fp32,
        )
        packed_mapped = compile_w4a16_fused_moe(
            **compiler_args, zero_fc2_output=True, max_m_blocks=packed_blocks,
            prefill_fused_sum_fp32=core.prefill_fused_sum_fp32,
        )
        direct = direct_mapped = None
        if weight_layout in BLOCK_CODECS:
            if (caps.decode_config.w4a16_route_mode != "packed"
                    and tokens <= 8 and core.activation in {"silu", "relu2"} and not caps.deterministic_output
                    and not caps.collect_activation_amax and not caps.apply_router_weight_on_input):
                decode_args = {**compiler_args, "moe_block_size": 8}
                direct = compile_w4a16_fused_moe(
                    **decode_args, zero_fc2_output=False, max_m_blocks=tokens * core.num_topk,
                    direct_topk_routes=True, tc_decode_fused_sum=True,
                )
                direct_mapped = compile_w4a16_fused_moe(
                    **decode_args, zero_fc2_output=False, max_m_blocks=tokens * core.num_topk,
                    direct_topk_routes=True, use_expert_map=True, tc_decode_fused_sum=True,
                )
        elif weight_layout == "packed":
            if tokens <= _MAX_DIRECT_TOPK_ROUTE_M:
                direct = compile_w4a16_fused_moe(
                    **compiler_args, zero_fc2_output=False,
                    max_m_blocks=tokens * core.num_topk, direct_topk_routes=True,
                )
            if tokens <= _W4A16_SMALL_M_DIRECT_MAX_M:
                direct_mapped = compile_w4a16_fused_moe(
                    **compiler_args, zero_fc2_output=False,
                    max_m_blocks=tokens * core.num_topk,
                    direct_topk_routes=True, use_expert_map=True,
                )
        topk_sum = compile_w4a16_topk_sum(
            m=tokens, topk=core.num_topk, hidden_size=core.k,
            element_dtype=element_dtype,
        )
        mapped_topk_sum = compile_w4a16_topk_sum(
            m=tokens, topk=core.num_topk, hidden_size=core.k,
            element_dtype=element_dtype, num_experts=core.weight_E,
            route_num_experts=caps.route_num_experts or core.weight_E,
            route_ids_dtype=torch.int32, use_expert_map=True,
        )
        route_pack = compile_w4a16_route_pack_launches(
            tokens=tokens, topk=core.num_topk, block_size=int(route_block),
            num_experts=core.route_E, ordinal=core.device.index,
        )
    # Direct routing requires exact M; packed routing accepts live M up to capacity.
    return _W4A16PrimaryLaunches(
        tokens=int(caps.max_tokens), route_mode=caps.decode_config.w4a16_route_mode or "auto",
        packed=packed, packed_mapped=packed_mapped, direct=direct,
        direct_mapped=direct_mapped, topk_sum=topk_sum,
        mapped_topk_sum=mapped_topk_sum, route_pack=route_pack,
    )



def _dynamic_program_arguments(plan, caps) -> dict[str, object]:
    """Derive the dynamic compiler ABI from the same native metadata as launch.

    The runtime does not use ``execution.weight_layout`` to decide whether
    W4A8 has its prepared operands: it receives the canonical prepared payload.
    A dynamic W4A8 declaration is valid only when the weight plan declares that
    payload, including the Trellis representation.  Keep this lowering beside
    the metadata compiler adapter so its program identity follows the real
    launch ABI rather than an execution-planner proxy.
    """
    from . import _impl

    quant_mode = plan.quant_mode
    prepared_w4a8 = (
        quant_mode == "w4a8_mx"
        and caps.weight_plan.required_weight_layout(quant_mode) is not None
    )
    logical_n = plan.n
    n = logical_n
    n64_repacked = bool(prepared_w4a8 and caps.weight_plan.source_format == "fp4_e8m0_k32" and int(n) % 128 == 64)
    if prepared_w4a8 and int(n) % 128 != 0 and not n64_repacked:
        n = _impl._dynamic_kernel_intermediate_size(n, quant_mode)
    tile_m = plan.execution.tile_m
    if tile_m is None:
        raise RuntimeError("dynamic MoE plan is missing its planned tile M")
    direct_routing = bool(
        (quant_mode != "w4a8_mx" or prepared_w4a8)
        and _impl._dynamic_direct_routing_selected(
            route_mode=plan.decode_config.dynamic_route_mode or "grouped",
            quant_mode=quant_mode,
            activation=plan.activation,
            routed_rows=plan.routed_rows,
            num_experts=plan.weight_E,
            n=n,
            deterministic_output=plan.deterministic_output,
            planned_tile_m=tile_m,
        )
    )
    external_route_plan = bool(
        _impl._dynamic_external_route_plan_supported(
            quant_mode=quant_mode,
            activation=plan.activation,
            routed_rows=plan.routed_rows,
            planned_tile_m=tile_m,
            dynamic_route_mode="direct" if direct_routing else "grouped",
            deterministic_output=plan.deterministic_output,
            w4a8_n64_repacked=n64_repacked,
        )
        and _impl._env_flag(
            _impl._DYNAMIC_EXTERNAL_ROUTE_PLAN_ENV,
            default=plan.decode_config.route_planner == "triton",
        )
    )
    share_input = plan.decode_config.nvfp4_share_input
    if prepared_w4a8:
        dense = _impl._w4a8_dynamic_dense_candidate(
            quant_mode=quant_mode,
            activation=plan.activation,
            routed_rows=plan.routed_rows,
            num_experts=plan.weight_E,
            k=plan.k,
            n=logical_n,
            deterministic_output=plan.deterministic_output,
            planned_tile_m=tile_m,
        )
        decode = _impl._w4a8_dynamic_decode_candidate(
            quant_mode=quant_mode,
            activation=plan.activation,
            routed_rows=plan.routed_rows,
            num_experts=plan.weight_E,
            n=logical_n,
            deterministic_output=plan.deterministic_output,
            planned_tile_m=tile_m,
        )
        share_input = _impl._env_flag(
            _impl._DYNAMIC_W4A8_SHARE_INPUT_ENV,
            default=dense or decode,
        )
    return {
        "n": n,
        "w4a8_repacked": prepared_w4a8,
        "w4a8_n64_repacked": n64_repacked,
        "direct_routing": direct_routing,
        "external_route_plan": external_route_plan,
        "share_input_across_experts": share_input,
        "planned_tile_m": tile_m,
        "nvfp4_inline_scales": bool(
            caps.weight_plan.nvfp4_inline_scales
            and quant_mode == "nvfp4"
            and plan.decode_config.nvfp4_inline_scales
        ),
    }


@dataclass(frozen=True)
class _CompactLaunches:
    kernels: object
    quantize: object
    topk_sum: object


def _compact_launches(plan, caps):
    if not (
        plan.implementation == "micro" and plan.quant_mode == "w4a8_mx"
        and caps.weight_plan.source_format == "fp4_e8m0_k32" and plan.n % 128 == 64
    ):
        return None
    from b12x._lib.quant.mxfp8_rows import _get_compiled_mxfp8_rows_quant, mxfp8_rows_quant_launch_options
    from b12x._lib.utils import get_num_sm
    from b12x.moe._shared.kernels.w4a8_compact_micro import _compiled_direct_w4a8_compact
    from b12x.moe._shared.kernels.w4a16.kernel import compile_w4a16_topk_sum

    m = plan.max_rows // plan.num_topk
    ordinal = plan.device.index
    sms = get_num_sm(plan.device)
    subgroup, threads = mxfp8_rows_quant_launch_options(m, "linear")
    quantize = _get_compiled_mxfp8_rows_quant(
        plan.k, torch.bfloat16, subgroup, threads, "linear", device_ordinal=ordinal, sm_count=sms,
    )
    kernels = {
        dtype: _compiled_direct_w4a8_compact(
            ordinal, m, plan.num_topk, plan.k, plan.n, plan.weight_E, dtype,
            plan.weight_E, plan.weight_E, plan.swiglu_limit, caps.w4a16_fast_math,
        ) for dtype in (torch.int32, torch.int64)
    }
    topk_sum = compile_w4a16_topk_sum(m=m, topk=plan.num_topk, hidden_size=plan.k)
    return attach_programs(_CompactLaunches(MappingProxyType(kernels), quantize, topk_sum),
                           tuple(kernels.values()), quantize, topk_sum)


def _program_carriers(
    scratch,
    caps,
    w4a16_launches: _W4A16PrimaryLaunches | None = None,
    *,
    compact_launches=None,
    input_scale_count: int,
    intermediate_scale_count: int,
):
    """Return every real launch carrier required by the selected native branch."""
    from . import _impl

    launches = [item[-1] for item in scratch._prewarmed_fused_launches]
    launches.extend(item[-1] for item in scratch._prewarmed_topk_sum_launches)
    launches.extend(item[-1] for item in scratch._mixed_trellis_launches)
    if scratch._prewarmed_route_pack_launches is not None:
        launches.extend(scratch._prewarmed_route_pack_launches.carriers())
    plan = scratch.launch_plan
    if plan.implementation == "w4a16" and not scratch.full_rotation:
        launches.extend(
            (w4a16_launches or _w4a16_primary_launches(scratch, caps)).carriers()
        )
    elif plan.implementation == "micro":
        compact = compact_launches
        if compact is not None:
            launches.append(compact)
            return tuple(launches)
        # Tiny W4A8-MX dispatch does not enter _get_micro_kernel: it always
        # normalizes IDs to int32 and resolves the paired tiny-decode phases.
        # Its layout is fixed by the same canonical weight plan that owns the
        # materialized payload, not by a guessed kernel-layout spelling.
        if (
            plan.quant_mode == "w4a8_mx"
            and caps.weight_plan.w4a8_weight_layout != "trellis_t256"
        ):
            launches.extend(_impl._get_tiny_decode_kernel(
                plan.weight_E, plan.max_tokens_per_launch, plan.k, plan.n,
                plan.num_topk, device=plan.device,
            ))
        else:
            weight_layout = (
                caps.weight_plan.w4a8_weight_layout
                if plan.quant_mode == "w4a8_mx"
                else caps.w4a16_weight_layout
            )
            single_token = plan.max_tokens_per_launch == 1
            share_expert_scales = (
                plan.activation in ("relu2", "silu")
                and input_scale_count == 1
                and intermediate_scale_count == 1
            )
            share_input_options = (False, True) if (
                plan.activation in ("relu2", "silu")
                and single_token
                and input_scale_count == 1
            ) else (False,)
            for dtype in (torch.int32, torch.int64):
                for share_input in share_input_options:
                    launch, _ = _impl._get_micro_kernel(
                        plan.weight_E, plan.max_tokens_per_launch, plan.k, plan.n,
                        plan.num_topk, topk_ids_dtype=dtype,
                        fast_math=caps.w4a16_fast_math,
                        share_input_across_experts=share_input,
                        share_expert_scales=share_expert_scales,
                        single_token=single_token,
                        activation=plan.activation, device=plan.device,
                        quant_mode=plan.quant_mode,
                        swiglu_limit=plan.swiglu_limit,
                        swiglu_alpha=plan.swiglu_alpha,
                        swiglu_beta=plan.swiglu_beta,
                        weight_layout=weight_layout,
                        trellis_bits=caps.weight_plan.trellis_bits or 0,
                        trellis_intermediate_hadamard=caps.weight_plan.intermediate_hadamard,
                    )
                    launches.append(launch)
    elif plan.implementation == "dynamic":
        exact_m = plan.routed_rows // plan.num_topk
        dynamic = _dynamic_program_arguments(plan, caps)
        for dtype in (torch.int32, torch.int64):
            launch, _ = _impl._get_dynamic_kernel(
                plan.weight_E, exact_m, plan.k, dynamic["n"], plan.num_topk,
                plan.max_rows, topk_ids_dtype=dtype, fast_math=caps.w4a16_fast_math,
                activation=plan.activation, quant_mode=plan.quant_mode,
                w4a8_repacked=dynamic["w4a8_repacked"],
                w4a8_n64_repacked=dynamic["w4a8_n64_repacked"],
                nvfp4_materialize_intermediate=plan.decode_config.nvfp4_materialize_intermediate,
                nvfp4_inline_scales=dynamic["nvfp4_inline_scales"],
                direct_routing=dynamic["direct_routing"],
                external_route_plan=dynamic["external_route_plan"],
                share_input_across_experts=dynamic["share_input_across_experts"],
                deterministic_output=plan.deterministic_output,
                swiglu_limit=plan.swiglu_limit, swiglu_alpha=plan.swiglu_alpha,
                swiglu_beta=plan.swiglu_beta, trellis_bits=caps.weight_plan.trellis_bits or 0,
                trellis_intermediate_hadamard=caps.weight_plan.intermediate_hadamard,
                planned_tile_m=dynamic["planned_tile_m"],
            )
            launches.append(launch)
        if plan.deterministic_output:
            from b12x.moe._shared.kernels.w4a16.kernel import compile_w4a16_topk_sum

            # The dynamic kernel writes one result per routed pair. Its
            # deterministic reduction is a separate retained launch.
            launches.append(compile_w4a16_topk_sum(
                m=exact_m, topk=plan.num_topk, hidden_size=plan.k,
                element_dtype=_impl._w4a16_element_dtype(plan.dtype),
            ))
    return tuple(launches)

@dataclass(frozen=True)
class _FusedMoePrograms:
    fused_launches: tuple
    topk_sum_launches: tuple
    mixed_trellis_launches: tuple
    route_pack_launches: object | None
    w4a16: _W4A16PrimaryLaunches | None
    compact: object | None
    launchers: tuple


def _x4t_scale_program_payload(experts):
    """Describe compressed-scale decoders without retaining weight tensors."""
    representation = experts._impl.representation
    weights = None if representation is None else representation.value
    first = getattr(weights, "x4t_w13_scale", None)
    if first is None:
        return ()
    if weights.x4t_packed_pair_programs is None:
        return ("tp12", int(weights.x4t_w13_row_rotation))
    return ("packed_pair", tuple(
        (plane.rows, plane.columns, plane.exception_task_rows,
         plane.exception_row_rotation)
        for plane in (first, weights.x4t_w2_scale)
    ))


@program_cache(scope="preparation")
def compile_x4t_scale_programs(payload, ordinal):
    """Declare every route ABI consumed by compressed-scale W4A16 weights."""
    with torch.cuda.device(ordinal):
        if payload[0] == "packed_pair":
            from b12x._lib.quant.x4t_packed_scales import _compiled_packed_scale_pair

            return tuple(
                _compiled_packed_scale_pair(*tuple(
                    (*plane, True, False, counts, ids64, sorted_ids)
                    for plane in payload[1]
                ))
                for counts, ids64, sorted_ids in (
                    (False, False, False), (True, False, False),
                    (False, True, False), (False, False, True),
                )
            )
        if payload[0] == "tp12":
            from b12x._lib.quant.x4t_scales import _compiled_x4t_tp12_w4a16

            return tuple(
                _compiled_x4t_tp12_w4a16(payload[1], mapped, unique)
                for mapped in (False, True) for unique in (False, True)
            )
        raise ValueError(f"unsupported X4T scale program layout: {payload[0]!r}")


@program_cache(scope="preparation")
def compile_nvfp4_csf_programs(payload, ordinal):
    """Declare both integer routing ABIs for native NVFP4 scale expansion."""
    from b12x._lib.quant.nvfp4_csf import compile_nvfp4_csf_pair
    from b12x._lib.quant.csf_routing import compile_csf_active_experts

    first, second, indexed = payload
    with torch.cuda.device(ordinal):
        return tuple(
            compile_nvfp4_csf_pair(first, second, ids64, use_index)
            for ids64 in (False, True)
            for use_index in ((False, True) if indexed else (False,))
        ) + tuple(compile_csf_active_experts(ids64) for ids64 in (False, True))


@program_cache(scope="preparation")
def compile_mxfp4_csf_programs(payload, ordinal):
    """Retain scale expansion and expert presence under frozen resolution."""
    from b12x._lib.quant.mxfp4_csf import compile_mxfp4_csf_pair
    from b12x._lib.quant.csf_routing import compile_csf_active_experts

    with torch.cuda.device(ordinal):
        return tuple(compile_mxfp4_csf_pair(*payload, ids64)
                     for ids64 in (False, True)) + tuple(
            compile_csf_active_experts(ids64) for ids64 in (False, True)
        )


@program_cache(scope="preparation")
def compile_fused_moe(
    query_payload, config_payload, weight_payload, scale_counts, ordinal
):
    """Compile and return actual primary native launch carriers from metadata."""
    query = _decode_query(query_payload)
    config = MoeDecodeConfig.from_config(FrozenMapping(config_payload))
    weight_args = dict(weight_payload)
    weight_args["params_dtype"] = getattr(torch, str(weight_args["params_dtype"]).removeprefix("torch."))
    weight_plan = plan_b12x_fp4_moe_weights(**weight_args)
    caps = _lower_caps(query, config, weight_plan, torch.device("cuda", ordinal))
    scratch = plan_tp_moe_scratch(caps, prewarm_launches=True)
    w4a16 = (
        _w4a16_primary_launches(scratch, caps)
        if scratch.launch_plan.implementation == "w4a16" and not scratch.full_rotation
        else None
    )
    compact = _compact_launches(scratch.launch_plan, caps)
    launchers = _program_carriers(
        scratch, caps, w4a16,
        compact_launches=compact,
        input_scale_count=int(scale_counts[0]), intermediate_scale_count=int(scale_counts[1]),
    )
    return attach_programs(
        _FusedMoePrograms(
            scratch._prewarmed_fused_launches, scratch._prewarmed_topk_sum_launches,
            scratch._mixed_trellis_launches, scratch._prewarmed_route_pack_launches,
            w4a16, compact, launchers,
        ),
        *launchers, compact,
    )


@dataclass(frozen=True)
class _FusedMoeState:
    experts: PreparedExperts
    scratch: object
    config: MoeDecodeConfig
    route_query: object
    launchers: object
    route_launcher: object
    w4a16_launches: _W4A16PrimaryLaunches | None = None
    compact_launches: _CompactLaunches | None = None

    def bind(self, **kwargs):
        experts = kwargs.pop("experts", self.experts)
        if experts is not self.experts:
            raise ValueError("prepared experts differ from the plan declaration")
        if self.config.nvfp4_share_input and not self.experts._impl.can_share_input(input_scales_static=True):
            raise ValueError("the input scales no longer satisfy the prepared shared-input contract")
        fast_math = self.scratch.caps.w4a16_fast_math
        if kwargs.get("fast_math") is not None and bool(kwargs["fast_math"]) != fast_math:
            raise ValueError("fast_math differs from the prepared numerical contract")
        kwargs["fast_math"] = fast_math
        kwargs["experts"] = self.experts._impl
        kwargs["unit_scale_contract"] = _quant_mode(self.experts, self.config) == "w4a16"
        if self.w4a16_launches is not None:
            kwargs["_w4a16_launches"] = self.w4a16_launches
        return replace(self.scratch.bind(**kwargs), compact_launches=self.compact_launches)

    def run(self, binding):
        return binding.run()

    def _check_route(self, route):
        if (
            route.score_func != self.route_query.score_func
            or route.renormalize != self.route_query.renormalize
            or route.routed_scaling_factor != self.route_query.routed_scaling_factor
            or (route.correction_bias is not None) != self.route_query.has_correction_bias
            or (route.image_correction_bias is not None) != self.route_query.has_image_correction_bias
            or (route.image_mask is not None) != self.route_query.has_image_mask
        ):
            raise ValueError("route binding differs from prepared routing invocation")

    def route(self, binding):
        from ._impl import b12x_route_experts_fast
        self._check_route(binding)
        return b12x_route_experts_fast(
            binding=binding, route_topk_launcher=self.route_launcher
        )

    def run_sparse(self, binding):
        from ._impl import b12x_sparse_moe_fp4
        if binding.routing is None:
            route = type("_Route", (), {
                "score_func": binding.score_func,
                "renormalize": binding.renormalize_topk,
                "routed_scaling_factor": binding.routed_scaling_factor,
                "correction_bias": binding.correction_bias,
                "image_correction_bias": binding.image_correction_bias,
                "image_mask": binding.image_mask,
            })()
            self._check_route(route)
        return b12x_sparse_moe_fp4(
            binding=binding, route_topk_launcher=self.route_launcher
        )


def variant_for(variants: Mapping[int, object], tokens: int):
    """Use an exact planned variant, otherwise the declared prefill capacity."""
    tokens = int(tokens)
    capacity = max(variants, default=0)
    if not 0 < tokens <= capacity:
        raise ValueError(
            f"token count {tokens} exceeds prepared MoE capacity {capacity}"
        )
    return variants[tokens] if tokens in variants else variants[capacity]


@dataclass(frozen=True)
class _FusedMoeCapacityState:
    variants: MappingProxyType
    a16_max_tokens: int = 0

    def bind(self, **kwargs):
        activations = kwargs.get("a")
        if not isinstance(activations, torch.Tensor) or activations.ndim != 2:
            raise TypeError("fused MoE binding requires a rank-two activation tensor")
        tokens = activations.shape[0]
        if 0 < tokens <= self.a16_max_tokens and tokens not in self.variants:
            return self.variants[self.a16_max_tokens].bind(**kwargs)
        return variant_for(self.variants, tokens).bind(**kwargs)

    def run(self, binding):
        return binding.run()


def plan(experts: PreparedExperts, *, capacity: ExecutionCapacity, routing: RoutingSpec | None = None,
         invocation: FrozenMapping = FrozenMapping(), override: MoeDecodeConfig | None = None):
    """Declare prefill capacity and exact planned variants without GPU work."""
    if not isinstance(experts, PreparedExperts):
        raise TypeError("experts must come from canonical prepare_weights")
    if not isinstance(capacity, ExecutionCapacity):
        raise TypeError("capacity must be an ExecutionCapacity")
    routing = routing or RoutingSpec()
    if not isinstance(routing, RoutingSpec):
        raise TypeError("routing must be a RoutingSpec")
    invocation = FrozenMapping(invocation)
    controls = _control_snapshot()
    cutoff = min(experts.plan.activation.a16_max_tokens, capacity.max_tokens)
    counts = tuple(sorted({capacity.max_tokens, *capacity.warmup_token_counts, *([cutoff] if cutoff else [])}))
    weight_payload = _weight_payload(experts)
    scale_counts = (
        int(experts._impl.a1_gscale.numel()),
        int(experts._impl.a2_gscale.numel()),
    )
    x4t_payload = _x4t_scale_program_payload(experts)
    lsc = experts._impl.nvfp4_csf
    lsc_payload = () if lsc is None else (
        lsc.first.geometry, lsc.second.geometry, lsc.indexed_programs is not None,
    )
    mx_csf = experts._impl.mxfp4_csf
    mx_payload = () if mx_csf is None else (mx_csf.first.geometry, mx_csf.second.geometry)

    def child(tokens):
        query = _query(experts, capacity, tokens, routing, controls, invocation)
        declaration_invocation = FrozenMapping({
            **invocation.to_dict(), "routing": FrozenMapping({
                "route_num_experts": query.route_num_experts,
                "route_logits_dtype": query.route_logits_dtype,
                "apply_router_weight_on_input": query.apply_router_weight_on_input,
                "collect_activation_amax": query.collect_activation_amax,
                "deterministic_output": query.deterministic_output,
            }), "controls": controls,
            **({"x4t_scale_programs": x4t_payload} if x4t_payload else {}),
            **({"nvfp4_csf_programs": lsc_payload} if lsc_payload else {}),
            **({"mxfp4_csf_programs": mx_payload} if mx_payload else {}),
            **({"nvfp4_inline_scales": True} if experts.plan._impl.nvfp4_inline_scales else {}),
        })

        def caps_for(config, device):
            return _lower_caps(query, config, experts.plan._impl, torch.device("cuda", device.ordinal))

        def compile_jobs(config, device):
            route_query = _route_query_from_moe(query, routing)
            caps = caps_for(config, device)
            launch_plan = plan_tp_moe_scratch(caps, prewarm_launches=False).launch_plan
            return (
                CompileJob.create(
                    "b12x.moe.fused_moe._preparation:compile_fused_moe",
                    TUNING.encode_query(query), TUNING.encode_config(config),
                    weight_payload, scale_counts, device.ordinal,
                ),
                CompileJob.create(
                    "b12x.moe.fused_moe._preparation:compile_route_topk",
                    ROUTE_TUNING.encode_query(route_query), device.ordinal,
                ),
                *((CompileJob.create(
                    "b12x.moe.fused_moe._preparation:compile_x4t_scale_programs",
                    x4t_payload, device.ordinal,
                ),) if x4t_payload else ()),
                *((CompileJob.create(
                    "b12x.moe.fused_moe._preparation:compile_nvfp4_csf_programs",
                    lsc_payload, device.ordinal,
                ),) if lsc_payload else ()),
                *((CompileJob.create(
                    "b12x.moe.fused_moe._preparation:compile_mxfp4_csf_programs",
                    mx_payload, device.ordinal,
                ),) if mx_payload else ()),
                *(
                    CompileJob.create(
                        "b12x.moe.fused_moe._preparation:compile_dynamic_route_plan",
                        payload, device.ordinal,
                    )
                    for payload in _dynamic_route_plan_payloads(launch_plan, caps)
                ),
            )

        def memory(config, device):
            return MemoryRequirements(scratch=plan_tp_moe_scratch(
                caps_for(config, device), prewarm_launches=False
            ).scratch_specs())
        def materialize(selection, device):
            caps = caps_for(selection.config, device)
            programs = compile_fused_moe(
                TUNING.encode_query(query), TUNING.encode_config(selection.config),
                weight_payload, scale_counts, device.ordinal,
            )
            scratch = replace(
                plan_tp_moe_scratch(caps, prewarm_launches=False),
                _prewarmed_fused_launches=programs.fused_launches,
                _prewarmed_topk_sum_launches=programs.topk_sum_launches,
                _mixed_trellis_launches=programs.mixed_trellis_launches,
                _prewarmed_route_pack_launches=programs.route_pack_launches,
            )
            launchers = list(programs.launchers)
            if x4t_payload:
                launchers.extend(compile_x4t_scale_programs(x4t_payload, device.ordinal))
            if lsc_payload:
                launchers.extend(compile_nvfp4_csf_programs(lsc_payload, device.ordinal))
            if mx_payload:
                launchers.extend(compile_mxfp4_csf_programs(mx_payload, device.ordinal))
            route_query = _route_query_from_moe(query, routing)
            route_launcher = compile_route_topk(
                ROUTE_TUNING.encode_query(route_query), device.ordinal
            )
            launchers.append(route_launcher)
            launchers.extend(
                compile_dynamic_route_plan(payload, device.ordinal)
                for payload in _dynamic_route_plan_payloads(scratch.launch_plan, caps)
            )
            launchers = tuple(launchers)
            carrier_tree = attach_programs(scratch, *launchers)
            load_programs(carrier_tree)
            return _FusedMoeState(
                experts, scratch, selection.config, route_query,
                launchers, route_launcher, programs.w4a16, programs.compact,
            )

        return Plan(
            contract=TUNING, query=query, invocation=declaration_invocation,
            override=override, _compile_jobs=compile_jobs,
            _memory_requirements=memory, _materialize=materialize,
            _device=experts.device,
        )

    children = {tokens: child(tokens) for tokens in counts}
    if len(counts) == 1:
        return children[counts[0]]

    def assemble(states, device):
        del device
        return _FusedMoeCapacityState(MappingProxyType({
            tokens: states[tokens] for tokens in counts
        }), a16_max_tokens=cutoff)

    return _CompositePlan(
        component_id="moe.decode",
        capacity_metadata=FrozenMapping({
            "token_counts": counts, "top_k": capacity.top_k,
            "controls": controls, "invocation": invocation,
            **({"a16_max_tokens": cutoff} if cutoff else {}),
        }),
        variants=children, _assemble=assemble,
    )

def _dynamic_route_plan_payloads(plan, caps):
    """Route-plan program identities a dynamic launch plan can reach.

    The kernel is specialized on the route-id dtype and on the tile the
    launch selects for the plan's routed rows; both are fixed by the
    declaration, so the race and the frozen serving path launch programs the
    compile jobs have already produced.
    """
    from . import _impl

    if plan.implementation != "dynamic":
        return ()
    dynamic = _dynamic_program_arguments(plan, caps)
    if not dynamic["external_route_plan"]:
        return ()
    planned = dynamic["planned_tile_m"]
    tiles = (
        (int(_impl._select_dynamic_tile_mn(
            plan.routed_rows, dynamic["n"], plan.quant_mode,
            num_experts=plan.weight_E, activation=plan.activation,
            planned_tile_m=planned,
        )[0]),)
        if planned is not None else (16, 32, 64, 128)
    )
    return tuple(
        {"num_experts": int(plan.weight_E), "tile_m": tile, "ids_dtype": ids_dtype}
        for tile in tiles for ids_dtype in ("int32", "int64")
    )


@program_cache(scope="preparation")
def compile_dynamic_route_plan(payload, ordinal):
    """Compile the Triton route-plan program for one dynamic MoE launch."""
    from triton.runtime.jit import MockTensor
    from . import _impl

    payload = dict(payload)
    num_experts = int(payload["num_experts"])
    tile_m = int(payload["tile_m"])
    ids_dtype = getattr(torch, payload["ids_dtype"])
    block_e = 1 << (num_experts - 1).bit_length()
    rows = _impl._DYNAMIC_EXTERNAL_ROUTE_PLAN_MAX_ROWS
    with torch.cuda.device(ordinal):
        return _impl._dynamic_route_plan_kernel.warmup(
            MockTensor(ids_dtype, (rows,)),
            MockTensor(torch.int32, (num_experts,)),
            MockTensor(torch.int32, (num_experts + 1,)),
            MockTensor(torch.int32, (1,)),
            MockTensor(torch.int32, (1,)),
            rows,
            NUM_EXPERTS=num_experts,
            TILE_M=tile_m,
            BLOCK_E=block_e,
            BLOCK_ROUTES=rows,
            num_warps=8,
            grid=(1,),
        )


@program_cache(scope="preparation")
def compile_route_topk(query_payload, ordinal):
    """Compile the exact Triton route program from immutable ABI metadata."""
    import triton
    from b12x.moe._shared.routing import _route_topk_kernel
    from ._tuning import MoeRouteQuery
    query = MoeRouteQuery(**dict(query_payload))
    ROUTE_TUNING.validate_query(query, None)
    block_e = triton.next_power_of_2(query.num_experts)
    topk_block = triton.next_power_of_2(query.top_k)
    with torch.cuda.device(ordinal):
        return _route_topk_kernel.warmup(
            getattr(torch, query.logits_dtype), torch.float32, torch.int32,
            torch.float32, torch.float32, torch.float32,
            torch.bool if query.has_image_mask else torch.float32,
            query.logits_row_stride, query.topk_row_stride, query.ids_row_stride,
            query.weights_row_stride, 1 if query.has_correction_bias else 0,
            1 if query.has_image_correction_bias else 0,
            1 if query.has_image_mask else 0, query.routed_scaling_factor,
            query.num_experts, BLOCK_E=block_e, TOP_K=query.top_k,
            TOPK_BLOCK=topk_block, RENORMALIZE=query.renormalize,
            SCORE_FUNC=query.score_func, HAS_CORRECTION_BIAS=query.has_correction_bias,
            HAS_IMAGE_BIAS=query.has_image_correction_bias,
            num_warps=4 if block_e <= 256 else 8, grid=(query.num_tokens,),
        )


@dataclass(frozen=True)
class _RouteTopKState:
    query: object
    launcher: object

    def run(self, router_logits, topk_logits, topk_ids, topk_weights, **kwargs):
        from b12x.moe._shared.routing import route_topk
        if (
            tuple(router_logits.shape) != (self.query.num_tokens, self.query.num_experts)
            or tuple(topk_logits.shape) != (self.query.num_tokens, self.query.top_k)
            or router_logits.stride(0) != self.query.logits_row_stride
            or topk_logits.stride(0) != self.query.topk_row_stride
            or topk_ids.stride(0) != self.query.ids_row_stride
            or topk_weights.stride(0) != self.query.weights_row_stride
        ):
            raise ValueError("routing tensors differ from the prepared top-k ABI")
        for name, expected in (
            ("renormalize", self.query.renormalize),
            ("score_func", self.query.score_func),
            ("routed_scaling_factor", self.query.routed_scaling_factor),
        ):
            actual = kwargs.pop(name, expected)
            if actual != expected:
                raise ValueError(f"{name} differs from the prepared route invocation")
        correction_bias = kwargs.pop("correction_bias", None)
        image_correction_bias = kwargs.pop("image_correction_bias", None)
        image_mask = kwargs.pop("image_mask", None)
        if (correction_bias is not None) != self.query.has_correction_bias:
            raise ValueError("correction bias differs from the prepared route invocation")
        if (image_correction_bias is not None) != self.query.has_image_correction_bias or (
            image_mask is not None
        ) != self.query.has_image_mask:
            raise ValueError("image routing controls differ from the prepared route invocation")
        route_topk(
            router_logits, topk_logits, topk_ids, topk_weights,
            renormalize=self.query.renormalize, score_func=self.query.score_func,
            correction_bias=correction_bias, image_correction_bias=image_correction_bias,
            image_mask=image_mask,
            routed_scaling_factor=self.query.routed_scaling_factor,
            _launcher=self.launcher, **kwargs,
        )


@dataclass(frozen=True)
class _FC2State:
    experts: PreparedExperts
    query: object
    launcher: object

    def run(self, intermediate, route_expert_ids, route_weights, *, output=None):
        from ._impl import run_w4a16_fc2_e8m0
        if int(intermediate.shape[0]) > self.query.max_routes:
            raise ValueError("FC2 route count exceeds prepared capacity")
        if str(route_expert_ids.dtype).removeprefix("torch.") != self.query.route_ids_dtype:
            raise TypeError("FC2 route-ID dtype differs from the prepared plan")
        return run_w4a16_fc2_e8m0(
            intermediate, self.experts._impl, route_expert_ids, route_weights,
            output=output, launcher=self.launcher,
        )


def plan_route_topk(invocation: RouteTopKInvocation, *, override=None,
                    declaration_invocation: FrozenMapping = FrozenMapping()) -> Plan:
    if not isinstance(invocation, RouteTopKInvocation):
        raise TypeError("route_topk preparation requires RouteTopKInvocation")
    query = invocation.query()
    declaration_invocation = FrozenMapping(declaration_invocation)
    def jobs(config, device):
        return (CompileJob.create(
            "b12x.moe.fused_moe._preparation:compile_route_topk",
            ROUTE_TUNING.encode_query(query), device.ordinal,
        ),)
    def materialize(selection, device):
        return _RouteTopKState(
            query, compile_route_topk(ROUTE_TUNING.encode_query(query), device.ordinal)
        )
    return Plan(
        contract=ROUTE_TUNING, query=query, invocation=declaration_invocation,
        override=override, _compile_jobs=jobs,
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=materialize,
    )


@program_cache(scope="preparation")
def compile_fc2(query_payload, ordinal):
    """Resolve the exact standalone FC2 launcher retained by its prepared plan."""
    from ._tuning import MoeFC2Query
    query = MoeFC2Query(**dict(query_payload))
    FC2_TUNING.validate_query(query, None)
    from b12x.moe._shared.kernels.w4a16.kernel import _compile_w4a16_fc2_direct
    return _compile_w4a16_fc2_direct(
        hidden_size=query.hidden_size, intermediate_size=query.intermediate_size,
        num_experts=query.num_experts,
        topk_ids_dtype=getattr(torch, query.route_ids_dtype),
        device=torch.device("cuda", ordinal),
    )


def plan_fc2(experts: PreparedExperts, invocation: FC2Invocation, *, override=None,
             declaration_invocation: FrozenMapping = FrozenMapping()) -> Plan:
    if not isinstance(experts, PreparedExperts):
        raise TypeError("FC2 preparation requires canonical PreparedExperts")
    if experts.plan._impl.source_format in BLOCK_CODECS:
        raise NotImplementedError("standalone IQ2_XS FC2 is unsupported; use fused MoE")
    if not isinstance(invocation, FC2Invocation):
        raise TypeError("FC2 preparation requires FC2Invocation")
    from ._tuning import MoeFC2Query
    query = MoeFC2Query(
        max_routes=invocation.max_routes, hidden_size=experts.hidden_size,
        intermediate_size=experts.intermediate_size, num_experts=experts.num_experts,
        route_ids_dtype=invocation.route_ids_dtype,
    )
    declaration_invocation = FrozenMapping(declaration_invocation)
    def jobs(config, device):
        return (CompileJob.create(
            "b12x.moe.fused_moe._preparation:compile_fc2",
            FC2_TUNING.encode_query(query), device.ordinal,
        ),)
    def materialize(selection, device):
        return _FC2State(experts, query, compile_fc2(
            FC2_TUNING.encode_query(query), device.ordinal
        ))
    return Plan(
        contract=FC2_TUNING, query=query, invocation=declaration_invocation,
        override=override, _compile_jobs=jobs,
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=materialize, _device=experts.device,
    )
