"""FlashInfer-TRTLLM cross-kernel oracle helpers.

Split from ``reference.py`` so the pure-torch references stay free of
flashinfer imports (the flashinfer.experimental port has a zero-outbound
-imports rule and simply skips this module).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .activations import (
    SITU,
    SWIGLUOAI_UNINTERLEAVE,
    is_gated_moe_activation,
    moe_activation_w1_rows,
    normalize_moe_activation,
)
from .reference import (
    _E8M0_K32_BF16_MAX_SCALE_BYTE,
    _e8m0_scale_bytes,
    _gated_row_slices,
    _validate_reference_inputs,
)


@dataclass(frozen=True)
class FlashInferTrtllmFP4E8M0K32Weights:
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor


def _block_scale_interleave_128x4_torch(unswizzled_sf: torch.Tensor) -> torch.Tensor:
    """Byte-preserving torch equivalent of FlashInfer/TRT-LLM block_scale_interleave."""
    if unswizzled_sf.dtype not in {torch.uint8, torch.bfloat16}:
        raise TypeError(f"expected uint8 or bfloat16 scale tensor, got {unswizzled_sf.dtype}")
    if unswizzled_sf.dim() == 2:
        sf = unswizzled_sf.reshape(1, unswizzled_sf.shape[0], unswizzled_sf.shape[1])
    elif unswizzled_sf.dim() == 3:
        sf = unswizzled_sf
    else:
        raise ValueError(f"expected 2D or 3D scale tensor, got shape {tuple(unswizzled_sf.shape)}")

    batches, rows, cols = sf.shape
    rows_padded = ((int(rows) + 127) // 128) * 128
    cols_padded = ((int(cols) + 3) // 4) * 4
    if rows_padded != rows or cols_padded != cols:
        padded = sf.new_zeros((batches, rows_padded, cols_padded))
        padded[:, :rows, :cols] = sf
        sf = padded
    else:
        sf = sf.contiguous()

    swizzled = sf.reshape(batches, rows_padded // 128, 4, 32, cols_padded // 4, 4)
    swizzled = swizzled.permute(0, 1, 4, 3, 2, 5).contiguous()
    return swizzled.reshape(-1)


def _flashinfer_block_scale_interleave(unswizzled_sf: torch.Tensor) -> torch.Tensor:
    try:
        from flashinfer.fp4_quantization import nvfp4_block_scale_interleave

        return nvfp4_block_scale_interleave(unswizzled_sf)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "nvcc" not in msg and "cuda_home" not in msg:
            raise
        return _block_scale_interleave_128x4_torch(unswizzled_sf)


def _interleave_flashinfer_w1_w3_rows(
    tensor: torch.Tensor,
    *,
    intermediate_size: int,
    activation: str,
) -> torch.Tensor:
    activation = normalize_moe_activation(activation)
    if not is_gated_moe_activation(activation):
        return tensor.contiguous()
    # This preparation helper has a fixed checkpoint-native source contract:
    # [w1/gate, w3/up].  Do not infer the source halves from the activation's
    # ordinary in-kernel W13 layout; FlashInfer needs those source halves
    # explicitly swapped into [up0, gate0, ...].
    gate_rows, up_rows = _gated_row_slices(
        activation,
        intermediate_size,
        w13_layout="w31",
    )
    gate = tensor[:, gate_rows]
    up = tensor[:, up_rows]
    return torch.stack([up, gate], dim=2).reshape(tensor.shape).contiguous()


def prepare_flashinfer_trtllm_fp4_e8m0_k32_weights(
    w13_fp4: torch.Tensor,
    w13_e8m0_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_e8m0_scale: torch.Tensor,
    K: int,
    I_tp: int,
    *,
    activation: str = "silu",
    scale_byte_clamp: int | None = None,
) -> FlashInferTrtllmFP4E8M0K32Weights:
    """Prepare b12x FP4/E8M0 K/32 source tensors for FlashInfer TRT-LLM MXFP4.

    The source W13 row contract is vLLM DeepSeek native loading order,
    [w1/gate, w3/up].  FlashInfer TRT-LLM expects the vLLM DeepSeek conversion
    style: [up0, gate0, up1, gate1, ...], then the TRT-LLM row permutation.
    Scale bytes stay E8M0 bytes; the final float8_e4m3fn dtype is only
    FlashInfer's ABI carrier for the interleaved byte storage.
    """
    activation = normalize_moe_activation(activation)
    if activation in {SITU, SWIGLUOAI_UNINTERLEAVE}:
        raise NotImplementedError(
            f"FlashInfer TRT-LLM FP4 preparation does not support {activation}"
        )
    _validate_reference_inputs(w13_fp4, I_tp, activation)
    if not w13_fp4.is_cuda or not w2_fp4.is_cuda:
        raise RuntimeError("FlashInfer TRT-LLM FP4 preparation requires CUDA tensors")
    if int(K) % 32 != 0 or int(I_tp) % 32 != 0:
        raise ValueError(f"FlashInfer MXFP4 prep requires K and I_tp divisible by 32, got K={K}, I_tp={I_tp}")
    rows_w13 = moe_activation_w1_rows(activation, I_tp)
    if tuple(w13_e8m0_scale.shape) != (int(w13_fp4.shape[0]), rows_w13, int(K) // 32):
        raise ValueError(
            f"w13_e8m0_scale must have shape {(int(w13_fp4.shape[0]), rows_w13, int(K) // 32)}, "
            f"got {tuple(w13_e8m0_scale.shape)}"
        )
    if tuple(w2_e8m0_scale.shape) != (int(w2_fp4.shape[0]), int(K), int(I_tp) // 32):
        raise ValueError(
            f"w2_e8m0_scale must have shape {(int(w2_fp4.shape[0]), int(K), int(I_tp) // 32)}, "
            f"got {tuple(w2_e8m0_scale.shape)}"
        )

    try:
        from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("FlashInfer TRT-LLM FP4 prep requires flashinfer") from exc

    w13_u8 = w13_fp4.view(torch.uint8).contiguous()
    w2_u8 = w2_fp4.view(torch.uint8).contiguous()
    w13_s_u8 = _e8m0_scale_bytes(w13_e8m0_scale, scale_byte_clamp=scale_byte_clamp)
    w2_s_u8 = _e8m0_scale_bytes(w2_e8m0_scale, scale_byte_clamp=scale_byte_clamp)

    w13_u8 = _interleave_flashinfer_w1_w3_rows(
        w13_u8,
        intermediate_size=int(I_tp),
        activation=activation,
    )
    w13_s_u8 = _interleave_flashinfer_w1_w3_rows(
        w13_s_u8,
        intermediate_size=int(I_tp),
        activation=activation,
    )

    cache: dict = {}
    epilogue_tile_m = 128
    w13_perm = get_w2_permute_indices_with_cache(
        cache,
        w13_u8[0],
        epilogue_tile_m,
    ).to(w13_u8.device)
    w13_out = w13_u8[:, w13_perm].contiguous()

    w13_sf_perm = get_w2_permute_indices_with_cache(
        cache,
        w13_s_u8[0],
        epilogue_tile_m,
        num_elts_per_sf=16,
    ).to(w13_s_u8.device)
    w13_s = w13_s_u8[:, w13_sf_perm].contiguous()
    E, N_s, K_s = w13_s.shape
    w13_scale_out = (
        _flashinfer_block_scale_interleave(w13_s.reshape(E * N_s, K_s))
        .reshape(E, rows_w13, int(K) // 32)
        .view(torch.float8_e4m3fn)
    )

    w2_perm = get_w2_permute_indices_with_cache(
        cache,
        w2_u8[0],
        epilogue_tile_m,
    ).to(w2_u8.device)
    w2_out = w2_u8[:, w2_perm].contiguous()

    w2_sf_perm = get_w2_permute_indices_with_cache(
        cache,
        w2_s_u8[0],
        epilogue_tile_m,
        num_elts_per_sf=16,
    ).to(w2_s_u8.device)
    w2_s = w2_s_u8[:, w2_sf_perm].contiguous()
    E2, N2_s, K2_s = w2_s.shape
    w2_scale_out = (
        _flashinfer_block_scale_interleave(w2_s.reshape(E2 * N2_s, K2_s))
        .reshape(E2, int(K), int(I_tp) // 32)
        .view(torch.float8_e4m3fn)
    )

    return FlashInferTrtllmFP4E8M0K32Weights(
        w13=w13_out,
        w13_scale=w13_scale_out,
        w2=w2_out,
        w2_scale=w2_scale_out,
    )


def _per_expert_float32(
    scale: torch.Tensor,
    *,
    num_experts: int,
    device: torch.device,
) -> torch.Tensor:
    scale = scale.to(device=device, dtype=torch.float32)
    if scale.numel() == 1:
        return scale.reshape(1).expand(num_experts).contiguous()
    if scale.numel() != num_experts:
        raise ValueError(f"expected scalar or {num_experts} per-expert scales, got {scale.numel()}")
    return scale.reshape(num_experts).contiguous()


def pack_flashinfer_trtllm_topk_ids_weights(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Pack top-k ids/weights exactly like vLLM/SGLang's TRT-LLM wrappers."""
    if topk_ids.shape != topk_weights.shape:
        raise ValueError(f"shape mismatch: topk_ids={tuple(topk_ids.shape)} topk_weights={tuple(topk_weights.shape)}")
    weight_bits = (
        topk_weights.contiguous().to(torch.bfloat16).view(torch.int16).to(torch.int32)
        & 0xFFFF
    )
    return (topk_ids.contiguous().to(torch.int32) << 16) | weight_bits


def moe_reference_w4a16_fp4_e8m0_k32_flashinfer(
    x: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_e8m0_scale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_e8m0_scale: torch.Tensor,
    w2_alphas: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    K: int,
    I_tp: int,
    *,
    activation: str = "silu",
    swiglu_limit: float | None = None,
    scale_byte_clamp: int | None = _E8M0_K32_BF16_MAX_SCALE_BYTE,
) -> torch.Tensor:
    activation = normalize_moe_activation(activation)
    if activation in {SITU, SWIGLUOAI_UNINTERLEAVE}:
        raise NotImplementedError(
            f"FlashInfer TRT-LLM FP4 oracle does not support {activation}"
        )
    _validate_reference_inputs(w1_fp4, I_tp, activation)
    if int(E) != int(w1_fp4.shape[0]) or int(E) != int(w2_fp4.shape[0]):
        raise ValueError("E must match the expert dimension of w1_fp4 and w2_fp4")
    prepared = prepare_flashinfer_trtllm_fp4_e8m0_k32_weights(
        w1_fp4,
        w1_e8m0_scale,
        w2_fp4,
        w2_e8m0_scale,
        K,
        I_tp,
        activation=activation,
        scale_byte_clamp=scale_byte_clamp,
    )
    return moe_reference_w4a16_fp4_e8m0_k32_flashinfer_prepared(
        x,
        prepared,
        w1_alphas,
        w2_alphas,
        topk_ids,
        topk_weights,
        E,
        K,
        I_tp,
        activation=activation,
        swiglu_limit=swiglu_limit,
    )


def moe_reference_w4a16_fp4_e8m0_k32_flashinfer_prepared(
    x: torch.Tensor,
    prepared: FlashInferTrtllmFP4E8M0K32Weights,
    w1_alphas: torch.Tensor,
    w2_alphas: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    K: int,
    I_tp: int,
    *,
    activation: str = "silu",
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    activation = normalize_moe_activation(activation)
    if activation == SWIGLUOAI_UNINTERLEAVE:
        raise NotImplementedError(
            "FlashInfer TRT-LLM FP4 oracle does not support swigluoai_uninterleave"
        )
    if x.dtype != torch.bfloat16:
        raise TypeError(f"FlashInfer W4A16 oracle expects BF16 activations, got {x.dtype}")
    if int(E) != int(prepared.w13.shape[0]) or int(E) != int(prepared.w2.shape[0]):
        raise ValueError("E must match the expert dimension of FlashInfer prepared weights")

    try:
        from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
        from flashinfer import ActivationType, RoutingMethodType
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("FlashInfer TRT-LLM FP4 oracle requires flashinfer") from exc

    packed_topk = pack_flashinfer_trtllm_topk_ids_weights(topk_ids, topk_weights)
    output = torch.empty(x.shape[0], K, dtype=torch.bfloat16, device=x.device)

    g1_scale = _per_expert_float32(w1_alphas, num_experts=int(E), device=x.device)
    g2_scale = _per_expert_float32(w2_alphas, num_experts=int(E), device=x.device)
    clamp_limit = None
    if swiglu_limit is not None:
        if activation != "silu":
            raise ValueError("swiglu_limit requires a gated W4A16 activation")
        clamp_limit = torch.full((int(E),), float(swiglu_limit), dtype=torch.float32, device=x.device)

    if activation == "silu":
        activation_type = int(ActivationType.Swiglu)
    elif activation == "relu2":
        activation_type = int(ActivationType.Relu2)
    else:
        raise ValueError(f"unsupported activation {activation!r}")

    result = trtllm_fp4_block_scale_routed_moe(
        topk_ids=packed_topk,
        routing_bias=None,
        hidden_states=x.contiguous(),
        hidden_states_scale=None,
        gemm1_weights=prepared.w13,
        gemm1_weights_scale=prepared.w13_scale,
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=clamp_limit,
        gemm2_weights=prepared.w2,
        gemm2_weights_scale=prepared.w2_scale,
        gemm2_bias=None,
        output1_scale_scalar=g1_scale,
        output1_scale_gate_scalar=g1_scale,
        output2_scale_scalar=g2_scale,
        num_experts=int(E),
        top_k=int(topk_ids.shape[1]),
        n_group=1,
        topk_group=1,
        intermediate_size=int(I_tp),
        local_expert_offset=0,
        local_num_experts=int(E),
        routed_scaling_factor=1.0,
        routing_method_type=int(RoutingMethodType.TopK),
        do_finalize=True,
        activation_type=activation_type,
        output=output,
        tune_max_num_tokens=max(16, int(x.shape[0])),
    )
    return result[0]
