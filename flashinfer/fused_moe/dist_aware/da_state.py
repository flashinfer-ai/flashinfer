"""Process-wide state for distribution-aware fused-MoE selection."""

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple, Union

import torch

from ...tllm_enums import DtypeTrtllmGen, Fp8QuantizationType, WeightLayout
from .da_config import DAConfig

# Increment when DAMoeContext fields or their cache-identity meaning changes;
# incompatible in-memory keys and persisted profile metadata must not mix.
CONTEXT_SCHEMA_VERSION = 1


def is_dist_aware_autotune(config: Optional[DAConfig] = None) -> bool:
    """Return the immutable DA policy for this runtime transaction."""
    return (DAConfig() if config is None else config).enabled


@dataclass(frozen=True)
class DAMoeContext:
    schema_version: int
    device_type: str
    device_index: int
    op_name: str
    dtype_act: int
    dtype_weights: int
    quantization_type: int
    top_k: int
    num_experts: int
    num_local_experts: int
    local_expert_offset: int
    hidden_size: int
    intermediate_size: int
    activation_type: int
    weight_layout: int
    use_shuffled_weight: bool
    use_per_token_scaling: bool
    has_gemm1_lora_delta: bool


def make_context(
    op_name: str,
    *,
    device: Union[str, torch.device],
    dtype_act: DtypeTrtllmGen,
    dtype_weights: DtypeTrtllmGen,
    quantization_type: Fp8QuantizationType,
    top_k: int,
    num_experts: int,
    num_local_experts: int,
    local_expert_offset: int,
    hidden_size: int,
    intermediate_size: int,
    activation_type: int,
    weight_layout: int = WeightLayout.MajorK,
    use_shuffled_weight: bool = False,
    use_per_token_scaling: bool = False,
    has_gemm1_lora_delta: bool = False,
) -> DAMoeContext:
    normalized_device = torch.device(device)
    if normalized_device.type != "cuda" or normalized_device.index is None:
        raise ValueError(
            f"DA MoE context requires an indexed CUDA device, got {device}"
        )
    return DAMoeContext(
        schema_version=CONTEXT_SCHEMA_VERSION,
        device_type=normalized_device.type,
        device_index=int(normalized_device.index),
        op_name=str(op_name),
        dtype_act=int(dtype_act),
        dtype_weights=int(dtype_weights),
        quantization_type=int(quantization_type),
        top_k=int(top_k),
        num_experts=int(num_experts),
        num_local_experts=int(num_local_experts),
        local_expert_offset=int(local_expert_offset),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        activation_type=int(activation_type),
        weight_layout=int(weight_layout),
        use_shuffled_weight=bool(use_shuffled_weight),
        use_per_token_scaling=bool(use_per_token_scaling),
        has_gemm1_lora_delta=bool(has_gemm1_lora_delta),
    )


def make_context_from_runner(
    op_name: str,
    runner: Any,
    *,
    device: Union[str, torch.device],
    num_experts: int,
    local_expert_offset: int,
    has_gemm1_lora_delta: bool = False,
) -> DAMoeContext:
    return make_context(
        op_name,
        device=device,
        dtype_act=runner.dtype_act,
        dtype_weights=runner.dtype_weights,
        quantization_type=runner.fp8_quantization_type,
        top_k=runner.top_k,
        num_experts=num_experts,
        num_local_experts=runner.num_local_experts,
        local_expert_offset=local_expert_offset,
        hidden_size=runner.hidden_size,
        intermediate_size=runner.intermediate_size,
        activation_type=int(runner.activation_type),
        weight_layout=int(runner.weight_layout),
        use_shuffled_weight=runner.use_shuffled_weight,
        use_per_token_scaling=runner.use_per_token_scaling,
        has_gemm1_lora_delta=has_gemm1_lora_delta,
    )


def cache_key(context: DAMoeContext, bucket: int) -> Tuple[int, DAMoeContext]:
    return (int(bucket), context)


def context_with_offset(
    context: DAMoeContext,
    local_expert_offset: int,
) -> DAMoeContext:
    return replace(context, local_expert_offset=int(local_expert_offset))


def context_supports_knn_capture(context: DAMoeContext) -> bool:
    return (
        context.op_name
        in {
            "flashinfer::trtllm_bf16_moe",
            "flashinfer::trtllm_fp8_per_tensor_scale_moe",
            "flashinfer::trtllm_fp8_block_scale_moe",
            "flashinfer::trtllm_fp4_block_scale_moe",
            "flashinfer::trtllm_mxint4_block_scale_moe",
        }
        and not context.has_gemm1_lora_delta
    )


BUNDLE_LOADED_CONTEXTS: set[DAMoeContext] = set()
BUNDLE_TACTIC_CONTEXTS: set[DAMoeContext] = set()
SELECTOR_HANDLES: Dict[DAMoeContext, int] = {}
_next_selector_handle = 1


def selector_handle(context: DAMoeContext) -> int:
    global _next_selector_handle
    handle = SELECTOR_HANDLES.get(context)
    if handle is None:
        handle = _next_selector_handle
        _next_selector_handle += 1
        SELECTOR_HANDLES[context] = handle
    return handle


PER_TILE_TACTICS: Dict[Tuple[int, DAMoeContext], Dict[int, Tuple[int, int]]] = {}
PER_BODY_TACTICS: Dict[Tuple[int, DAMoeContext], list[Tuple[int, int]]] = {}
CAPTURE_KEEPALIVE: Dict[DAMoeContext, Dict[int, torch.Tensor]] = {}
STATIC_FALLBACK_TACTICS: Dict[
    Tuple[str, int, int, DAMoeContext], Optional[Tuple[Tuple[int, int], float]]
] = {}
BASELINE_GUARD_DECISIONS: Dict[Tuple[int, DAMoeContext], Dict[str, Any]] = {}


def retain_capture_tensor(context: DAMoeContext, tensor: torch.Tensor) -> None:
    if (
        tensor.device.type != context.device_type
        or tensor.device.index != context.device_index
    ):
        raise ValueError(
            f"capture tensor device {tensor.device} does not match DA context "
            f"{context.device_type}:{context.device_index}"
        )
    CAPTURE_KEEPALIVE.setdefault(context, {})[int(tensor.data_ptr())] = tensor
