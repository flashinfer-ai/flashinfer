from __future__ import annotations

import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
import torch


@triton.jit(do_not_specialize=["num_experts"])
def _route_topk_kernel(
    logits_ptr, topk_logits_ptr, topk_ids_ptr, topk_weights_ptr,
    correction_bias_ptr, image_correction_bias_ptr, image_mask_ptr,
    logits_row_stride, topk_row_stride, ids_row_stride, weights_row_stride,
    correction_bias_stride, image_correction_bias_stride, image_mask_stride,
    routed_scaling_factor, num_experts, BLOCK_E: tl.constexpr,
    TOP_K: tl.constexpr, TOPK_BLOCK: tl.constexpr, RENORMALIZE: tl.constexpr,
    SCORE_FUNC: tl.constexpr, HAS_CORRECTION_BIAS: tl.constexpr,
    HAS_IMAGE_BIAS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    expert_offsets = tl.arange(0, BLOCK_E)
    expert_offsets_i32 = expert_offsets.to(tl.int32)
    topk_offsets = tl.arange(0, TOPK_BLOCK)
    topk_mask = topk_offsets < TOP_K
    neg_inf = float("-inf")

    row_ptr = logits_ptr + pid * logits_row_stride
    logits = tl.load(row_ptr + expert_offsets, mask=expert_offsets < num_experts, other=neg_inf)
    logits = logits.to(tl.float32)
    if SCORE_FUNC == "sqrtsoftplus":
        scores = tl.sqrt(tl.maximum(logits, 0.0) + libdevice.log1p(tl.exp(-tl.abs(logits))))
    elif HAS_CORRECTION_BIAS or HAS_IMAGE_BIAS:
        exp_logits = tl.exp(logits - tl.max(logits, axis=0))
        scores = exp_logits / tl.sum(exp_logits, axis=0)
    else:
        scores = logits
    selection_scores = scores
    if HAS_CORRECTION_BIAS:
        selection_scores += tl.load(
            correction_bias_ptr + expert_offsets * correction_bias_stride,
            mask=expert_offsets < num_experts, other=0.0,
        ).to(tl.float32)
    if HAS_IMAGE_BIAS:
        is_image = tl.load(image_mask_ptr + pid * image_mask_stride)
        image_bias = tl.load(
            image_correction_bias_ptr + expert_offsets * image_correction_bias_stride,
            mask=expert_offsets < num_experts, other=0.0,
        ).to(tl.float32)
        selection_scores = tl.where(is_image, scores + image_bias, selection_scores)
    eligible = expert_offsets < num_experts
    remaining = tl.where(eligible, selection_scores, neg_inf)

    topk_ids = tl.zeros((TOPK_BLOCK,), dtype=tl.int32)
    for slot in range(TOP_K):
        best_logits = tl.max(remaining, axis=0)
        best_ids = tl.max(tl.where(eligible & (remaining == best_logits), expert_offsets_i32, -1), axis=0)
        topk_ids = tl.where(topk_offsets == slot, best_ids, topk_ids)
        eligible = eligible & (expert_offsets != best_ids)
        remaining = tl.where(eligible, remaining, neg_inf)

    topk_logits = tl.gather(logits, topk_ids, axis=0)
    if SCORE_FUNC == "sqrtsoftplus":
        topk_weights = tl.where(topk_mask, tl.gather(scores, topk_ids, axis=0), 0.0)
        if RENORMALIZE and TOP_K > 1:
            topk_weights /= tl.sum(topk_weights, axis=0) + 1e-20
    elif RENORMALIZE:
        masked_logits = tl.where(topk_mask, topk_logits, neg_inf)
        shifted = masked_logits - tl.max(masked_logits, axis=0)
        exp_logits = tl.where(topk_mask, tl.exp(shifted), 0.0)
        topk_weights = exp_logits / tl.sum(exp_logits, axis=0)
    else:
        topk_weights = topk_logits
    topk_weights *= routed_scaling_factor

    out_base = pid * topk_row_stride + topk_offsets
    tl.store(topk_logits_ptr + out_base, topk_logits, mask=topk_mask)
    tl.store(topk_ids_ptr + pid * ids_row_stride + topk_offsets, topk_ids, mask=topk_mask)
    tl.store(topk_weights_ptr + pid * weights_row_stride + topk_offsets, topk_weights, mask=topk_mask)


def route_topk(
    router_logits: torch.Tensor, topk_logits: torch.Tensor, topk_ids: torch.Tensor,
    topk_weights: torch.Tensor, *, renormalize: bool, score_func: str = "softmax",
    correction_bias: torch.Tensor | None = None,
    image_correction_bias: torch.Tensor | None = None,
    image_mask: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
    _launcher=None,
) -> None:
    """Select experts into caller-owned buffers, breaking ties by larger expert ID.

    Correction tensors steer selection but do not alter selected logits. The
    selected softmax or sqrt-softplus weights receive the scale last.
    """
    if router_logits.ndim != 2:
        raise ValueError(f"expected router_logits with rank 2, got shape {tuple(router_logits.shape)}")
    if topk_logits.ndim != 2 or topk_ids.ndim != 2 or topk_weights.ndim != 2:
        raise ValueError("top-k outputs must all have rank 2")
    if topk_logits.shape != topk_ids.shape or topk_logits.shape != topk_weights.shape:
        raise ValueError("top-k outputs must share a shape, got " f"{tuple(topk_logits.shape)}, {tuple(topk_ids.shape)}, {tuple(topk_weights.shape)}")
    if router_logits.shape[0] != topk_logits.shape[0]:
        raise ValueError(f"router_logits batch mismatch: expected {router_logits.shape[0]}, got {topk_logits.shape[0]}")
    if topk_ids.dtype != torch.int32:
        raise ValueError(f"expected topk_ids dtype int32, got {topk_ids.dtype}")
    if topk_logits.dtype != torch.float32:
        raise ValueError(f"expected topk_logits dtype float32, got {topk_logits.dtype}")
    if topk_weights.dtype != torch.float32:
        raise ValueError(f"expected topk_weights dtype float32, got {topk_weights.dtype}")
    if not router_logits.is_cuda:
        raise ValueError("route_topk requires CUDA tensors")
    for tensor in (topk_logits, topk_ids, topk_weights):
        if tensor.device != router_logits.device:
            raise ValueError("top-k outputs must be on the router_logits device")
    _validate_score_options(router_logits, score_func=score_func, correction_bias=correction_bias,
                            image_correction_bias=image_correction_bias, image_mask=image_mask)
    if router_logits.shape[1] > 1024:
        raise ValueError(f"route_topk currently supports up to 1024 experts, got {router_logits.shape[1]}")
    if router_logits.stride(-1) != 1:
        raise ValueError("router_logits must be contiguous in the expert dimension")
    if topk_logits.stride(-1) != 1 or topk_ids.stride(-1) != 1 or topk_weights.stride(-1) != 1:
        raise ValueError("top-k outputs must be contiguous in the top-k dimension")
    num_tokens, num_experts = router_logits.shape
    top_k = topk_logits.shape[1]
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_k > num_experts:
        raise ValueError(f"top_k={top_k} exceeds num_experts={num_experts}")
    if num_tokens == 0:
        return
    block_e = triton.next_power_of_2(num_experts)
    topk_block = triton.next_power_of_2(top_k)
    launcher = _route_topk_kernel if _launcher is None else _launcher
    launcher[(num_tokens,)](
        router_logits, topk_logits, topk_ids, topk_weights, correction_bias,
        image_correction_bias, image_mask, router_logits.stride(0), topk_logits.stride(0),
        topk_ids.stride(0), topk_weights.stride(0),
        correction_bias.stride(0) if correction_bias is not None else 0,
        image_correction_bias.stride(0) if image_correction_bias is not None else 0,
        image_mask.stride(0) if image_mask is not None else 0,
        float(routed_scaling_factor), num_experts, BLOCK_E=block_e, TOP_K=top_k,
        TOPK_BLOCK=topk_block, RENORMALIZE=renormalize, SCORE_FUNC=score_func,
        HAS_CORRECTION_BIAS=correction_bias is not None,
        HAS_IMAGE_BIAS=image_correction_bias is not None and image_mask is not None,
        num_warps=4 if block_e <= 256 else 8,
    )


def _validate_score_options(router_logits: torch.Tensor, *, score_func: str,
                            correction_bias: torch.Tensor | None,
                            image_correction_bias: torch.Tensor | None,
                            image_mask: torch.Tensor | None) -> None:
    if score_func not in ("softmax", "sqrtsoftplus"):
        raise ValueError(f"unsupported router score_func: {score_func!r}")
    for name, bias in (("correction_bias", correction_bias), ("image_correction_bias", image_correction_bias)):
        if bias is not None:
            if bias.ndim != 1 or bias.shape[0] != router_logits.shape[1]:
                raise ValueError(f"{name} must have shape (num_experts,)")
            if bias.dtype != torch.float32 or bias.device != router_logits.device:
                raise ValueError(f"{name} must be FP32 on the router_logits device")
    if image_mask is not None:
        if image_mask.ndim != 1 or image_mask.shape[0] != router_logits.shape[0]:
            raise ValueError("image_mask must have shape (num_tokens,)")
        if image_mask.dtype != torch.bool or image_mask.device != router_logits.device:
            raise ValueError("image_mask must be bool on the router_logits device")
