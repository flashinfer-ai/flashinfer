"""Capture orchestration for distribution-aware MoE dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from . import da_single_graph, da_state
from .da_config import DAConfig
from ...tllm_enums import DtypeTrtllmGen, Fp8QuantizationType, RoutingMethodType
from ..fused_routing_dsv3 import fused_topk_deepseek


@dataclass(frozen=True)
class DACaptureBackend:
    ffi_moe_op: Any
    prepare_routing_metadata: Callable[..., list[torch.Tensor]]
    prepare_routing_metadata_multi_tile: Callable[..., list[list[torch.Tensor]]]
    supported_tile_sizes: Callable[..., Sequence[int]]


ROUTING_METADATA_SCHEMA_VERSION = 1


def _routing_input_mode(name: str) -> int:
    """Read the core-owned launcher routing ABI without an import cycle."""

    from ..core import RoutingInputMode

    return int(getattr(RoutingInputMode, name))


@dataclass(frozen=True)
class RoutingMetadataBundle:
    tensors: Tuple[Any, ...]
    schema_version: int
    num_tokens: int
    top_k: int
    num_experts: int
    num_local_experts: int
    local_expert_offset: int
    tile_size: int
    routing_method_type: int
    routing_input_mode: int
    da_context: da_state.DAMoeContext
    device_type: str
    device_index: int

    def public_metadata(self) -> List[torch.Tensor]:
        return list(self.tensors)


@dataclass(frozen=True)
class DACaptureResources:
    da_context: da_state.DAMoeContext
    num_tokens: int
    num_tokens_bucket: int
    routing_method_type: int
    routing_input_mode: int
    topk_ids: torch.Tensor
    expert_weights_bf16: torch.Tensor
    routing_metadata_by_tile: Tuple[Tuple[int, RoutingMetadataBundle], ...]
    candidate_tile_sizes: Tuple[int, ...]
    per_body_tactics: Tuple[Tuple[int, int], ...]
    side_stream: torch.cuda.Stream
    routing_stream: torch.cuda.Stream
    pool_handle: Any


CAPTURE_RESOURCES: Dict[
    Tuple[da_state.DAMoeContext, int, int, int, int], DACaptureResources
] = {}


def _capture_resource_key(
    da_context: da_state.DAMoeContext,
    num_tokens_bucket: int,
    num_tokens: int,
    routing_method_type: int,
    routing_input_mode: int,
) -> Tuple[da_state.DAMoeContext, int, int, int, int]:
    return (
        da_context,
        int(num_tokens_bucket),
        int(num_tokens),
        int(routing_method_type),
        int(routing_input_mode),
    )


def store_capture_resources(resources: DACaptureResources) -> None:
    CAPTURE_RESOURCES[
        _capture_resource_key(
            resources.da_context,
            resources.num_tokens_bucket,
            resources.num_tokens,
            resources.routing_method_type,
            resources.routing_input_mode,
        )
    ] = resources


def prepare_capture_resources(
    *,
    da_context: da_state.DAMoeContext,
    num_tokens: int,
    num_tokens_bucket: int,
    routing_method_type: int,
    routing_input_mode: int,
    topk_ids: torch.Tensor,
    expert_weights_bf16: torch.Tensor,
    routing_metadata_by_tile: Sequence[Tuple[int, RoutingMetadataBundle]],
    candidate_tile_sizes: Sequence[int],
    per_body_tactics: Sequence[Sequence[int]],
) -> DACaptureResources:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("DA capture resources must be prepared before capture")
    with torch.cuda.device(
        torch.device(da_context.device_type, da_context.device_index)
    ):
        resources = DACaptureResources(
            da_context=da_context,
            num_tokens=int(num_tokens),
            num_tokens_bucket=int(num_tokens_bucket),
            routing_method_type=int(routing_method_type),
            routing_input_mode=int(routing_input_mode),
            topk_ids=topk_ids,
            expert_weights_bf16=expert_weights_bf16,
            routing_metadata_by_tile=tuple(routing_metadata_by_tile),
            candidate_tile_sizes=tuple(int(tile) for tile in candidate_tile_sizes),
            per_body_tactics=tuple(
                (int(tactic[0]), int(tactic[1])) for tactic in per_body_tactics
            ),
            side_stream=torch.cuda.Stream(),
            routing_stream=torch.cuda.Stream(),
            pool_handle=torch.cuda.graph_pool_handle(),
        )
    store_capture_resources(resources)
    da_state.retain_capture_tensor(da_context, resources.topk_ids)
    da_state.retain_capture_tensor(da_context, resources.expert_weights_bf16)
    for _, bundle in resources.routing_metadata_by_tile:
        for tensor in bundle.tensors:
            da_state.retain_capture_tensor(da_context, tensor)
    return resources


def lookup_capture_resources(
    da_context: da_state.DAMoeContext,
    num_tokens_bucket: int,
    *,
    num_tokens: int,
    routing_method_type: int,
    routing_input_mode: int,
) -> DACaptureResources:
    resources = CAPTURE_RESOURCES.get(
        _capture_resource_key(
            da_context,
            num_tokens_bucket,
            num_tokens,
            routing_method_type,
            routing_input_mode,
        )
    )
    if resources is None:
        raise RuntimeError(
            "DA capture resources are not prepared for "
            f"context={da_context}, bucket={num_tokens_bucket}, num_tokens={num_tokens}, "
            f"routing_method_type={routing_method_type}, "
            f"routing_input_mode={routing_input_mode}; run DA warmup before capture"
        )
    expected_topk_shape = (int(num_tokens), int(da_context.top_k))
    if tuple(resources.topk_ids.shape) != expected_topk_shape:
        raise RuntimeError(
            "DA capture resources are not prepared: "
            f"topk_ids shape={tuple(resources.topk_ids.shape)} expected={expected_topk_shape}"
        )
    if tuple(resources.expert_weights_bf16.shape) != expected_topk_shape:
        raise RuntimeError(
            "DA capture resources are not prepared: expert_weights_bf16 shape="
            f"{tuple(resources.expert_weights_bf16.shape)} expected={expected_topk_shape}"
        )
    for tile_size, bundle in resources.routing_metadata_by_tile:
        try:
            validate_routing_metadata_bundle(
                bundle,
                da_context=da_context,
                num_tokens=num_tokens,
                top_k=da_context.top_k,
                num_experts=da_context.num_experts,
                num_local_experts=da_context.num_local_experts,
                local_expert_offset=da_context.local_expert_offset,
                tile_size=tile_size,
                routing_method_type=routing_method_type,
                routing_input_mode=routing_input_mode,
            )
        except ValueError as error:
            raise RuntimeError(
                f"DA capture resources are not prepared: {error}"
            ) from error
        expanded_tokens = num_tokens * da_context.top_k
        filled_experts = min(da_context.num_experts, expanded_tokens)
        remaining_tokens = max(expanded_tokens - filled_experts, 0)
        max_num_ctas = filled_experts + remaining_tokens // int(tile_size)
        max_num_padded_tokens = max_num_ctas * int(tile_size)
        minimum_first_dims = (
            1,
            expanded_tokens,
            max_num_padded_tokens,
            num_tokens,
            max(da_context.num_experts * 2, 512),
            da_context.num_experts,
            max_num_ctas,
            max_num_ctas,
            1,
        )
        for index, (tensor, minimum) in enumerate(
            zip(bundle.tensors, minimum_first_dims, strict=True)
        ):
            if index == 3:
                shape_is_valid = tuple(tensor.shape) == expected_topk_shape
            else:
                shape_is_valid = tensor.ndim == 1 and int(tensor.shape[0]) >= int(
                    minimum
                )
            if not shape_is_valid:
                raise RuntimeError(
                    "DA capture resources are not prepared: routing metadata "
                    f"tensor {index} shape={tuple(tensor.shape)} requires dim0 >= {minimum}"
                )
    return resources


def make_routing_metadata_bundle(
    routing_metadata: Sequence[torch.Tensor],
    *,
    da_context: da_state.DAMoeContext,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    num_local_experts: int,
    local_expert_offset: int,
    tile_size: int,
    routing_method_type: int,
    routing_input_mode: int,
) -> RoutingMetadataBundle:
    return RoutingMetadataBundle(
        tensors=tuple(routing_metadata),
        schema_version=ROUTING_METADATA_SCHEMA_VERSION,
        num_tokens=int(num_tokens),
        top_k=int(top_k),
        num_experts=int(num_experts),
        num_local_experts=int(num_local_experts),
        local_expert_offset=int(local_expert_offset),
        tile_size=int(tile_size),
        routing_method_type=int(routing_method_type),
        routing_input_mode=int(routing_input_mode),
        da_context=da_context,
        device_type=da_context.device_type,
        device_index=int(da_context.device_index),
    )


def validate_routing_metadata_bundle(
    bundle: RoutingMetadataBundle,
    *,
    da_context: da_state.DAMoeContext,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    num_local_experts: int,
    local_expert_offset: int,
    tile_size: int,
    routing_method_type: int,
    routing_input_mode: int,
) -> None:
    expected = {
        "schema_version": ROUTING_METADATA_SCHEMA_VERSION,
        "num_tokens": int(num_tokens),
        "top_k": int(top_k),
        "num_experts": int(num_experts),
        "num_local_experts": int(num_local_experts),
        "local_expert_offset": int(local_expert_offset),
        "tile_size": int(tile_size),
        "routing_method_type": int(routing_method_type),
        "routing_input_mode": int(routing_input_mode),
        "da_context": da_context,
        "device_type": da_context.device_type,
        "device_index": int(da_context.device_index),
    }
    for field, expected_value in expected.items():
        actual_value = getattr(bundle, field)
        if actual_value != expected_value:
            raise ValueError(
                f"routing metadata {field} mismatch: "
                f"prepared={actual_value!r}, expected={expected_value!r}"
            )
    if len(bundle.tensors) != 9:
        raise ValueError(
            f"routing metadata must contain nine tensors, got {len(bundle.tensors)}"
        )
    expected_device = torch.device(bundle.device_type, bundle.device_index)
    for index, tensor in enumerate(bundle.tensors):
        if tensor.device != expected_device:
            raise ValueError(
                f"routing metadata tensor device mismatch at index {index}: "
                f"prepared={tensor.device}, expected={expected_device}"
            )


def run_from_routing_metadata_bundle(
    bundle: RoutingMetadataBundle,
    tactic: Sequence[int],
    run_from_routing_metadata: Callable[[RoutingMetadataBundle, Sequence[int]], None],
    **expected: Any,
) -> None:
    validate_routing_metadata_bundle(bundle, **expected)
    if len(tactic) != 2 or int(tactic[0]) != bundle.tile_size:
        raise ValueError(
            f"routing metadata tactic tile mismatch: tactic={tuple(tactic)}, "
            f"prepared tile_size={bundle.tile_size}"
        )
    run_from_routing_metadata(bundle, tactic)


def selector_expert_counts(
    routing_metadata_by_tile: Optional[Sequence[RoutingMetadataBundle]],
    *,
    enabled: bool,
) -> Optional[torch.Tensor]:
    if not enabled or not routing_metadata_by_tile:
        return None
    return routing_metadata_by_tile[0].tensors[4]


@dataclass
class CaptureRoutingPlan:
    precomputed_route_ready: bool
    expert_weights_bf16: Optional[torch.Tensor] = None
    anchor_tile: Optional[int] = None
    anchor_metadata: Optional[List[torch.Tensor]] = None


def pack_unpacked_routing_out(
    backend: DACaptureBackend,
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    packed_topk_ids_out: torch.Tensor,
) -> None:
    backend.ffi_moe_op.trtllm_moe_pack_unpacked_routing(
        expert_ids,
        expert_weights,
        packed_topk_ids_out,
    )


def prepare_capture_routing(
    *,
    backend: DACaptureBackend,
    routing_logits,
    routing_bias,
    topk_ids,
    num_experts,
    n_group,
    topk_group,
    top_k,
    local_expert_offset,
    num_local_experts,
    routed_scaling_factor,
    routing_method_type,
    anchor_tile,
    norm_topk_prob,
    use_routing_scales_on_input,
    pack_for_trtllm: bool = False,
) -> CaptureRoutingPlan:
    """Populate packed top-k IDs before the DA selector reads them.

    Grouped DeepSeekV3 keeps its routing-only kernel. Every other logits path
    runs the canonical TRT-LLM ``Routing::Runner`` for one anchor tile, which
    also yields model-exact BF16 weights and standard routing metadata. The
    packed IDs can be replayed for any remaining candidate tiles.
    """
    if routing_logits is None or topk_ids is None:
        return CaptureRoutingPlan(routing_logits is None and topk_ids is not None)

    _ng = int(n_group) if n_group is not None else 1
    _tg = int(topk_group) if topk_group is not None else 1
    use_dsv3_fast_path = (
        int(routing_method_type) == int(RoutingMethodType.DeepSeekV3)
        and _ng > 1
        and _tg * _ng >= int(top_k)
        and _tg <= _ng
    )

    if use_dsv3_fast_path:
        if pack_for_trtllm:
            # Use the same routing kernel (`routingMainKernel`) that NoDA's
            # body would have run. The packed topk_ids and bf16 expert
            # weights produced here are bit-exact to what the body's
            # `routing_runner.run` would write — so the SWITCH bodies can
            # consume them via routing_logits=None and the MoE finalize step
            # sees the same mixing weights NoDA does (modulo the SWITCH
            # picking a different gemm tile).
            #
            # This is the single-kernel pre-route; computing indices and
            # packing weights separately would introduce a bf16 quantisation
            # mismatch vs routingMainKernel and can flip argmax on long
            # generations.
            expert_weights_bf16 = torch.empty(
                topk_ids.shape,
                dtype=torch.bfloat16,
                device=topk_ids.device,
            )
            backend.ffi_moe_op.trtllm_deepseek_moe_compute_routing_packed_only(
                routing_logits,
                routing_bias,
                int(routing_logits.shape[1]),
                int(top_k),
                _ng,
                _tg,
                int(0),  # local_expert_offset (kNN selector applies its own)
                int(routing_logits.shape[1]),  # local_num_experts (full table)
                float(routed_scaling_factor)
                if routed_scaling_factor is not None
                else None,
                int(routing_method_type),
                topk_ids,
                expert_weights_bf16,
            )
            return CaptureRoutingPlan(True, expert_weights_bf16)

        # No pack requested -> populate raw topk_ids only (e.g. for the
        # DA selector's histogram when the caller intends the body to
        # re-route from logits anyway). Use the lighter fused_topk_deepseek.
        topk_values = torch.empty(
            topk_ids.shape[0],
            top_k,
            dtype=torch.float32,
            device=topk_ids.device,
        )
        routing_logits_fp32 = (
            routing_logits
            if routing_logits.dtype == torch.float32
            else routing_logits.to(torch.float32)
        )
        bias_fp32 = (
            routing_bias.to(torch.float32)
            if routing_bias is not None
            else torch.zeros(
                routing_logits.shape[1],
                dtype=torch.float32,
                device=routing_logits.device,
            )
        )
        fused_topk_deepseek(
            scores=routing_logits_fp32,
            bias=bias_fp32,
            n_group=_ng,
            topk_group=_tg,
            topk=int(top_k),
            routed_scaling_factor=float(routed_scaling_factor)
            if routed_scaling_factor is not None
            else 1.0,
            topk_values=topk_values,
            topk_indices=topk_ids,
        )
        return CaptureRoutingPlan(False)

    metadata = backend.prepare_routing_metadata(
        routing_logits,
        routing_bias,
        int(num_experts),
        int(top_k),
        n_group,
        topk_group,
        int(local_expert_offset),
        int(num_local_experts),
        routed_scaling_factor,
        int(routing_method_type),
        int(anchor_tile),
        bool(norm_topk_prob),
        bool(use_routing_scales_on_input),
        topk_ids,
    )
    return CaptureRoutingPlan(
        pack_for_trtllm,
        metadata[3],
        int(anchor_tile),
        metadata,
    )


def try_trtllm_capture_aware_da(
    *,
    backend: Union[DACaptureBackend, Callable[[], DACaptureBackend]],
    upload_bucket: Callable[[int, int], int],
    debug_log: Callable[[str], None],
    da_context: da_state.DAMoeContext,
    run_from_routing_metadata: Callable[[RoutingMetadataBundle, Sequence[int]], None],
    direct_body: Optional[Callable[[Sequence[int], Dict[str, Any]], None]] = None,
    precomputed_topk_ids_are_packed: bool = True,
    routing_input_mode: int,
    hidden_states,
    hidden_states_scale,
    routing_logits,
    topk_ids,
    expert_weights,
    routing_bias,
    gemm1_weights,
    gemm1_weights_scale,
    gemm1_bias,
    gemm1_alpha,
    gemm1_beta,
    gemm1_clamp_limit,
    gemm2_weights,
    gemm2_weights_scale,
    gemm2_bias,
    output1_scale_scalar,
    output1_scale_gate_scalar,
    output2_scale_scalar,
    output,
    num_experts: int,
    top_k: int,
    n_group,
    topk_group,
    intermediate_size: int,
    local_expert_offset: int,
    num_local_experts: int,
    routed_scaling_factor,
    routing_method_type: int,
    enable_pdl,
    activation_type: int,
    num_tokens: int,
    tune_max_num_tokens: int,
    dtype_act,
    norm_topk_prob: bool,
    use_routing_scales_on_input: bool = False,
    config: Optional[DAConfig] = None,
) -> Optional[List[torch.Tensor]]:
    """Return [output] if we handled this call via the inline-SWITCH path,
    or None to let the caller fall back to the non-capture autotuner path.

    Preconditions for the fast path:
      * current stream is in active CUDA graph capture
      * DAKNNv2 exemplars + per-body tactics are populated for this
        (num_tokens_bucket, top_k, num_local_experts, local_expert_offset)
      * topk_ids and expert_weights are caller-provided (non-None) so no
        torch.empty happens under capture
      * do_finalize is True (enforced by caller; we only return [output])
    """
    # Verbose diagnostic: set FLASHINFER_DA_VERBOSE=1 to learn which guard
    # (below) returned None for each call. Useful when vLLM integration
    # silently falls back to the D2H-laden path.
    config = DAConfig() if config is None else config
    _verbose = config.verbose

    def _skip(reason: str) -> Optional[List[torch.Tensor]]:
        debug_log(f"capture-aware SKIP {reason}")
        if _verbose:
            print(f"[DA capture-aware SKIP] {reason}", flush=True)
        return None

    # Use torch's native capture-detection — matches exactly what torch
    # itself uses to validate operations during capture, so we won't
    # disagree with torch on edge cases (e.g. capture mode transitions).
    is_capturing = torch.cuda.is_current_stream_capturing()
    unpacked_expert_ids = topk_ids
    unpacked_expert_weights = expert_weights

    # Look up per-tile tactics for this configuration. Runtime num_tokens
    # may be non-pow2 (vLLM's default capture list includes 640/768/896/…);
    # da_core.upload_bucket pads up to the next pow2 so the uploaded tile set
    # remains launcher-valid at the actual num_tokens. For pow2 num_tokens
    # (e.g. when the bench restricts capture sizes to {1,2,4,…,8192})
    # bucket == num_tokens and padding is a no-op.
    bucket = upload_bucket(num_tokens, tune_max_num_tokens)
    capture_resources = None
    if is_capturing:
        capture_resources = lookup_capture_resources(
            da_context,
            bucket,
            num_tokens=num_tokens,
            routing_method_type=routing_method_type,
            routing_input_mode=routing_input_mode,
        )
        if int(routing_input_mode) == _routing_input_mode("UnpackedPrecomputed"):
            if unpacked_expert_ids is None or unpacked_expert_weights is None:
                return _skip("unpacked routing requires both expert IDs and weights")
            if callable(backend):
                backend = backend()
            pack_unpacked_routing_out(
                backend,
                unpacked_expert_ids,
                unpacked_expert_weights,
                capture_resources.topk_ids,
            )
            precomputed_topk_ids_are_packed = True
        elif int(routing_input_mode) == _routing_input_mode("PackedPrecomputed"):
            if topk_ids is None:
                return _skip("packed routing requires packed top-k IDs")
            capture_resources.topk_ids.copy_(topk_ids)
        topk_ids = capture_resources.topk_ids
        expert_weights = capture_resources.expert_weights_bf16
    elif topk_ids is None or expert_weights is None:
        return _skip(
            f"topk_ids is None={topk_ids is None}, "
            f"expert_weights is None={expert_weights is None}"
        )

    if callable(backend):
        backend = backend()

    try:
        _ = backend.ffi_moe_op
    except Exception as e:
        return _skip(f"failed to get ffi_moe_op: {e}")

    config_key = da_state.cache_key(da_context, int(bucket))
    if capture_resources is not None:
        candidate_tile_sizes = capture_resources.candidate_tile_sizes
        per_tile_tactics = list(capture_resources.per_body_tactics)
    else:
        tile_map = da_state.PER_TILE_TACTICS.get(config_key)
        if not tile_map:
            avail = sorted(str(k) for k in da_state.PER_TILE_TACTICS.keys())
            return _skip(
                f"no per-tile tactics for config_key={config_key} "
                f"(num_tokens={num_tokens}, bucket={bucket}); "
                f"available configs: {avail}"
            )

        per_body_knn = da_state.PER_BODY_TACTICS.get(config_key)
        if per_body_knn is None:
            return _skip(f"no DAKNNv2 per-body tactics for config_key={config_key}")

        candidate_source_tile_sizes = backend.supported_tile_sizes(
            bucket,
            top_k=top_k,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            dtype_act=dtype_act,
            dtype_weights=DtypeTrtllmGen(da_context.dtype_weights),
            quantization_type=Fp8QuantizationType(da_context.quantization_type),
            da_context=da_context,
        )
        candidate_tile_sizes = tuple(
            t for t in candidate_source_tile_sizes if int(t) in tile_map
        )
        if len(candidate_tile_sizes) == 0:
            return _skip(
                f"no candidate tile sizes for bucket={bucket} num_tokens={num_tokens}; "
                f"tile_map keys={sorted(tile_map.keys())}"
            )

        try:
            static_knn_tile = int(
                backend.ffi_moe_op.da_get_static_knn_tile_with_handle(
                    da_state.selector_handle(da_context),
                    int(bucket),
                    int(top_k),
                    int(num_local_experts),
                    int(local_expert_offset),
                    [int(t) for t in candidate_tile_sizes],
                )
            )
        except Exception:
            static_knn_tile = -1
        if static_knn_tile > 0 and static_knn_tile in tile_map:
            candidate_tile_sizes = (int(static_knn_tile),)

        per_tile_tactics = list(per_body_knn)
    if _verbose:
        print(
            f"[DA capture-aware FIRE] num_tokens={num_tokens} bucket={bucket} "
            f"tiles={candidate_tile_sizes} tactics={per_tile_tactics} "
            f"config_key={config_key}",
            flush=True,
        )
    debug_log(
        "capture-aware FIRE "
        f"num_tokens={num_tokens} bucket={bucket} "
        f"tiles={candidate_tile_sizes} tactics={per_tile_tactics}"
    )

    base_kwargs = dict(
        routing_input_mode=int(routing_input_mode),
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        routing_logits=routing_logits,
        topk_ids=topk_ids,
        expert_weights=expert_weights,
        output=output,
        routing_bias=routing_bias,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm1_bias=gemm1_bias,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        gemm2_bias=gemm2_bias,
        output1_scale_scalar=output1_scale_scalar,
        output1_scale_gate_scalar=output1_scale_gate_scalar,
        output2_scale_scalar=output2_scale_scalar,
        num_experts=int(num_experts),
        top_k=int(top_k),
        n_group=n_group,
        topk_group=topk_group,
        intermediate_size=int(intermediate_size),
        local_expert_offset=int(local_expert_offset),
        num_local_experts=int(num_local_experts),
        routed_scaling_factor=routed_scaling_factor,
        routing_method_type=int(routing_method_type),
        do_finalize=True,
        enable_pdl=bool(enable_pdl) if enable_pdl is not None else False,
        activation_type=int(activation_type),
    )

    # Consume the pre-router's packed IDs and BF16 weights by default. Both the
    # generic runner and the DeepSeek fast path produce the same routing values
    # as the original body before tactic selection.
    use_precomputed_route = config.use_precomputed_route

    if capture_resources is None:
        if int(routing_input_mode) == _routing_input_mode("UnpackedPrecomputed"):
            if topk_ids is None or expert_weights is None:
                return _skip("unpacked routing requires both expert IDs and weights")
            packed_topk_ids = torch.empty_like(topk_ids)
            pack_unpacked_routing_out(
                backend, topk_ids, expert_weights, packed_topk_ids
            )
            topk_ids = packed_topk_ids
            precomputed_topk_ids_are_packed = True
        routing_plan = prepare_capture_routing(
            backend=backend,
            routing_logits=routing_logits,
            routing_bias=routing_bias,
            topk_ids=topk_ids,
            num_experts=num_experts,
            n_group=n_group,
            topk_group=topk_group,
            top_k=top_k,
            local_expert_offset=local_expert_offset,
            num_local_experts=num_local_experts,
            routed_scaling_factor=routed_scaling_factor,
            routing_method_type=routing_method_type,
            anchor_tile=int(per_tile_tactics[0][0]),
            norm_topk_prob=norm_topk_prob,
            use_routing_scales_on_input=use_routing_scales_on_input,
            pack_for_trtllm=use_precomputed_route,
        )
    else:
        prepared_by_tile = dict(capture_resources.routing_metadata_by_tile)
        first_tile = int(candidate_tile_sizes[0])
        first_bundle = prepared_by_tile[first_tile]
        ffi_moe_op = backend.ffi_moe_op
        if routing_logits is not None and (
            use_precomputed_route or len(per_tile_tactics) > 1
        ):
            # A multi-body selector always needs assignments for the current
            # logits, even when the selected body will rerun routing itself.
            ffi_moe_op.trtllm_moe_populate_routing_metadata_from_logits(
                routing_logits,
                routing_bias,
                int(num_experts),
                int(top_k),
                n_group,
                topk_group,
                int(local_expert_offset),
                int(num_local_experts),
                routed_scaling_factor,
                int(routing_method_type),
                first_tile,
                bool(norm_topk_prob),
                bool(use_routing_scales_on_input),
                topk_ids,
                first_bundle.public_metadata(),
            )
            remaining_tiles = (
                tuple(
                    int(tile)
                    for tile in candidate_tile_sizes
                    if int(tile) != first_tile
                )
                if use_precomputed_route
                else ()
            )
        elif use_precomputed_route:
            remaining_tiles = tuple(int(tile) for tile in candidate_tile_sizes)
        else:
            remaining_tiles = ()
        if remaining_tiles:
            ffi_moe_op.trtllm_moe_populate_routing_metadata_multi_tile(
                topk_ids,
                routing_bias,
                int(num_experts),
                int(top_k),
                n_group,
                topk_group,
                int(local_expert_offset),
                int(num_local_experts),
                routed_scaling_factor,
                int(routing_method_type),
                list(remaining_tiles),
                [
                    tensor
                    for tile in remaining_tiles
                    for tensor in prepared_by_tile[tile].tensors
                ],
            )
        routing_plan = CaptureRoutingPlan(
            use_precomputed_route,
            first_bundle.tensors[3] if use_precomputed_route else None,
            first_tile if use_precomputed_route else None,
            first_bundle.public_metadata() if use_precomputed_route else None,
        )

    def _bundle_routing_metadata(
        metadata: Sequence[torch.Tensor], tile_size: int
    ) -> RoutingMetadataBundle:
        return make_routing_metadata_bundle(
            metadata,
            da_context=da_context,
            num_tokens=num_tokens,
            top_k=top_k,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            tile_size=tile_size,
            routing_method_type=routing_method_type,
            routing_input_mode=routing_input_mode,
        )

    def _dispatch_routing_metadata(
        bundle: RoutingMetadataBundle, tactic: Sequence[int]
    ) -> None:
        run_from_routing_metadata_bundle(
            bundle,
            tactic,
            run_from_routing_metadata,
            da_context=da_context,
            num_tokens=num_tokens,
            top_k=top_k,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            tile_size=int(tactic[0]),
            routing_method_type=routing_method_type,
            routing_input_mode=routing_input_mode,
        )

    precomputed_route_ready = routing_plan.precomputed_route_ready
    expert_weights_bf16 = routing_plan.expert_weights_bf16
    if capture_resources is None:
        if expert_weights_bf16 is not None:
            da_state.retain_capture_tensor(da_context, expert_weights_bf16)
        if routing_plan.anchor_metadata is not None:
            for _tensor in routing_plan.anchor_metadata:
                da_state.retain_capture_tensor(da_context, _tensor)

    if (
        routing_logits is None
        and use_precomputed_route
        and precomputed_route_ready
        and not precomputed_topk_ids_are_packed
    ):
        if direct_body is None:
            return _skip(
                "precomputed topk_ids are unpacked; routing metadata replay requires "
                "TRT-LLM packed topk_ids"
            )
        precomputed_route_ready = False

    if not (use_precomputed_route and precomputed_route_ready) and direct_body is None:
        return _skip("precomputed routing metadata is unavailable for this precision")

    if not is_capturing:
        metadata_by_tile: Dict[int, RoutingMetadataBundle] = {}
        if (
            routing_plan.anchor_tile is not None
            and routing_plan.anchor_metadata is not None
        ):
            metadata_by_tile[int(routing_plan.anchor_tile)] = _bundle_routing_metadata(
                routing_plan.anchor_metadata,
                int(routing_plan.anchor_tile),
            )
        missing_tiles = tuple(
            int(tile)
            for tile in candidate_tile_sizes
            if int(tile) not in metadata_by_tile
        )
        if missing_tiles:
            missing_metadata = backend.prepare_routing_metadata_multi_tile(
                topk_ids,
                routing_bias,
                int(num_experts),
                int(top_k),
                n_group,
                topk_group,
                int(local_expert_offset),
                int(num_local_experts),
                routed_scaling_factor,
                int(routing_method_type),
                missing_tiles,
            )
            metadata_by_tile.update(
                (
                    tile,
                    _bundle_routing_metadata(metadata, tile),
                )
                for tile, metadata in zip(missing_tiles, missing_metadata, strict=True)
            )
        prepared_metadata = tuple(
            (int(tile), metadata_by_tile[int(tile)]) for tile in candidate_tile_sizes
        )
        prepared_expert_weights = (
            expert_weights_bf16
            if expert_weights_bf16 is not None
            else prepared_metadata[0][1].tensors[3]
        )
        prepare_capture_resources(
            da_context=da_context,
            num_tokens=num_tokens,
            num_tokens_bucket=bucket,
            routing_method_type=routing_method_type,
            routing_input_mode=routing_input_mode,
            topk_ids=topk_ids,
            expert_weights_bf16=prepared_expert_weights,
            routing_metadata_by_tile=prepared_metadata,
            candidate_tile_sizes=candidate_tile_sizes,
            per_body_tactics=per_tile_tactics,
        )
        return _skip("prepared DA capture resources during eager warmup")

    # Pruning or tactic deduplication can leave one body even when multiple
    # candidate tiles remain. Capture it directly so replay pays neither the
    # kNN selector kernel nor the CUDA conditional SWITCH node.
    if len(per_tile_tactics) == 1:
        if use_precomputed_route and precomputed_route_ready:
            body_tile = int(per_tile_tactics[0][0])
            if capture_resources is not None:
                prepared_by_tile = dict(capture_resources.routing_metadata_by_tile)
                routing_bundle = prepared_by_tile.get(body_tile)
                if routing_bundle is None:
                    raise RuntimeError(
                        "DA capture resources are not prepared for single-body "
                        f"tile {body_tile}"
                    )
            else:
                if routing_plan.anchor_tile == body_tile:
                    routing_metadata = routing_plan.anchor_metadata
                else:
                    routing_metadata = None
                if routing_metadata is None:
                    routing_metadata = backend.prepare_routing_metadata_multi_tile(
                        topk_ids,
                        routing_bias,
                        int(num_experts),
                        int(top_k),
                        n_group,
                        topk_group,
                        int(local_expert_offset),
                        int(num_local_experts),
                        routed_scaling_factor,
                        int(routing_method_type),
                        (body_tile,),
                    )[0]
                routing_bundle = _bundle_routing_metadata(routing_metadata, body_tile)
                for _tensor in routing_bundle.tensors:
                    da_state.retain_capture_tensor(da_context, _tensor)
            _dispatch_routing_metadata(routing_bundle, per_tile_tactics[0])
            globals().setdefault("_TRTLLM_DA_PRECOMPUTED_ROUTE_COUNT", 0)
            globals()["_TRTLLM_DA_PRECOMPUTED_ROUTE_COUNT"] += 1
        else:
            if direct_body is None:
                return _skip("direct capture body is unavailable for this precision")
            direct_body(per_tile_tactics[0], base_kwargs)
        globals().setdefault("_TRTLLM_DA_CAPTURE_DISPATCH_COUNT", 0)
        globals()["_TRTLLM_DA_CAPTURE_DISPATCH_COUNT"] += 1
        return [output]

    # Multi-tile SWITCH path. The router must populate packed top-k IDs before
    # the selector reads them. Body metadata is then derived from those packed
    # IDs, so logits-based top-k selection runs only once.

    switch_body_kwargs = base_kwargs
    if use_precomputed_route and precomputed_route_ready:
        switch_body_kwargs = dict(base_kwargs)
        switch_body_kwargs["routing_logits"] = None
        switch_body_kwargs["routing_input_mode"] = int(
            _routing_input_mode("PackedPrecomputed")
        )
        # The body's routing runner reads `mPtrTopKWeights` as bfloat16. With
        # routing_logits=None the runner skips the score+weight computation
        # block in routingMainKernel, so the buffer the body sees must already
        # contain valid bf16 weights at bf16 stride. Substitute the dedicated
        # bf16 buffer the pack kernel just populated.
        if expert_weights_bf16 is not None:
            switch_body_kwargs["expert_weights"] = expert_weights_bf16
        globals().setdefault("_TRTLLM_DA_PRECOMPUTED_ROUTE_COUNT", 0)
        globals()["_TRTLLM_DA_PRECOMPUTED_ROUTE_COUNT"] += 1

    overlap_routing_requested = config.overlap_routing
    overlap_routing = precomputed_route_ready and (
        overlap_routing_requested or direct_body is None
    )
    select_from_routing_counts = (
        config.select_from_routing_counts
        and use_precomputed_route
        and precomputed_route_ready
        and len(candidate_tile_sizes) > 1
    )

    def _prepare_candidate_routing_metadata(
        tile_sizes: Sequence[int], *, reuse_anchor: bool
    ) -> List[RoutingMetadataBundle]:
        metadata_by_tile: Dict[int, RoutingMetadataBundle] = {}
        if (
            reuse_anchor
            and routing_plan.anchor_tile is not None
            and routing_plan.anchor_metadata is not None
        ):
            metadata_by_tile[int(routing_plan.anchor_tile)] = _bundle_routing_metadata(
                routing_plan.anchor_metadata,
                int(routing_plan.anchor_tile),
            )

        missing_tiles = tuple(
            int(tile) for tile in tile_sizes if int(tile) not in metadata_by_tile
        )
        if missing_tiles:
            missing_metadata = backend.prepare_routing_metadata_multi_tile(
                topk_ids,
                routing_bias,
                int(num_experts),
                int(top_k),
                n_group,
                topk_group,
                int(local_expert_offset),
                int(num_local_experts),
                routed_scaling_factor,
                int(routing_method_type),
                missing_tiles,
            )
            metadata_by_tile.update(
                (
                    tile,
                    _bundle_routing_metadata(metadata, tile),
                )
                for tile, metadata in zip(missing_tiles, missing_metadata, strict=True)
            )
        return [metadata_by_tile[int(tile)] for tile in tile_sizes]

    routing_metadata_by_tile = (
        [bundle for _, bundle in capture_resources.routing_metadata_by_tile]
        if capture_resources is not None
        and use_precomputed_route
        and precomputed_route_ready
        else None
    )
    if select_from_routing_counts and routing_metadata_by_tile is None:
        routing_tile_sizes = tuple(int(t) for t in candidate_tile_sizes)
        routing_metadata_by_tile = _prepare_candidate_routing_metadata(
            routing_tile_sizes, reuse_anchor=False
        )
        for _metadata in routing_metadata_by_tile:
            for _tensor in _metadata.tensors:
                da_state.retain_capture_tensor(da_context, _tensor)

    injector = da_single_graph.DAInlineGraphInjector(backend.ffi_moe_op)
    with injector.inject(
        selector_handle=da_state.selector_handle(da_context),
        topk_ids=topk_ids,
        tile_sizes=candidate_tile_sizes,
        num_tokens_bucket=bucket,
        num_local_experts=int(num_local_experts),
        local_expert_offset=int(local_expert_offset),
        top_k=int(top_k),
        # Tensor 4 in the routing-metadata tuple is expert_count_histogram.
        # Its first num_experts entries are the per-expert counts already
        # produced by routingIndicesMultiTileClusterKernel.
        expert_counts=selector_expert_counts(
            routing_metadata_by_tile,
            enabled=select_from_routing_counts,
        ),
        side_stream=capture_resources.side_stream,
        routing_stream=capture_resources.routing_stream,
        pool_handle=capture_resources.pool_handle,
    ) as ctx:
        if overlap_routing and routing_metadata_by_tile is None:
            # Build routing metadata for every candidate tile on a graph branch
            # that starts from the same pre-SWITCH dependencies as the selector.
            # CUDA conditional body graphs cannot depend directly on parent-graph
            # routing nodes, so the C++ injector joins this branch at the SWITCH.
            # This still hides the kNN selector/conditional setup behind TRT-LLM's
            # routing kernels and lets body graphs start at FC1/FC2.
            routing_tile_sizes = tuple(int(t) for t in candidate_tile_sizes)
            with ctx.routing_branch():
                routing_metadata_by_tile = _prepare_candidate_routing_metadata(
                    routing_tile_sizes,
                    # Parent-graph metadata cannot be consumed directly by a
                    # conditional body graph. Replay packed IDs in this branch.
                    reuse_anchor=False,
                )
        tile_to_routing_idx = {
            int(t): idx for idx, t in enumerate(candidate_tile_sizes)
        }
        if ctx.num_bodies != len(per_tile_tactics):
            raise RuntimeError(
                f"[DA body/tactic mismatch] ctx.num_bodies={ctx.num_bodies} but "
                f"len(per_tile_tactics)={len(per_tile_tactics)}; "
                f"bucket={bucket} "
                f"config_key={config_key} "
                f"candidate_tile_sizes={candidate_tile_sizes} "
                f"per_tile_tactics={per_tile_tactics}"
            )
        for i in range(ctx.num_bodies):
            if _verbose:
                print(
                    f"[DA capture body {i}/{ctx.num_bodies}] "
                    f"tactic={per_tile_tactics[i]} "
                    f"tile_n={per_tile_tactics[i][0]} "
                    f"config_idx={per_tile_tactics[i][1]}",
                    flush=True,
                )
            with ctx.body(i):
                if routing_metadata_by_tile is None:
                    if direct_body is None:
                        raise RuntimeError(
                            "direct capture body is unavailable for this precision"
                        )
                    direct_body(per_tile_tactics[i], switch_body_kwargs)
                else:
                    ri = tile_to_routing_idx[per_tile_tactics[i][0]]
                    _dispatch_routing_metadata(
                        routing_metadata_by_tile[ri], per_tile_tactics[i]
                    )

    # Diagnostic counter (module-global) for integration tests to assert that
    # the capture-aware path actually fired rather than silently falling back.
    globals().setdefault("_TRTLLM_DA_CAPTURE_DISPATCH_COUNT", 0)
    globals()["_TRTLLM_DA_CAPTURE_DISPATCH_COUNT"] += 1
    return [output]


def reset_fast_path_stats() -> None:
    globals()["_TRTLLM_DA_CAPTURE_DISPATCH_COUNT"] = 0


def capture_dispatch_count() -> int:
    return int(globals().get("_TRTLLM_DA_CAPTURE_DISPATCH_COUNT", 0))
