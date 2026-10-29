"""Caller metadata checks and execution scratch for SM120 SVDQuant auto."""

import torch

from . import gemm_svdquant as gemm


def _check_linear_gemm_inputs(inputs: list[torch.Tensor]) -> None:
    """Validate caller operands without materializing quantization/LoRA scratch."""
    x, weight, sf, alpha, _, l2t, l1, _, bias, _ = inputs
    _, k = x.shape
    n, k_packed = weight.shape
    rank = l2t.shape[1]
    if weight.dtype != torch.uint8 or not weight.is_contiguous():
        raise ValueError("weight_fp4 must be contiguous packed-e2m1 (uint8)")
    if k_packed * 2 != k or n % 32 or k % 32:
        raise ValueError("weight_fp4 must match x's K; N and K must be divisible by 32")
    if rank < 32 or rank % 32:
        raise ValueError("the LoRA rank must be a positive multiple of 32")
    if l1.shape != (n, rank) or l1.dtype != torch.bfloat16 or not l1.is_contiguous():
        raise ValueError(f"l1 must be contiguous bf16 with shape ({n}, {rank})")
    if (
        sf.dtype != torch.uint8
        or not sf.is_contiguous()
        or sf.numel() < gemm._swizzled_sf_size(n, k // 16)
    ):
        raise ValueError(
            "weight_sf must be a sufficiently large contiguous uint8 buffer"
        )
    if alpha.dtype != torch.float32 or alpha.numel() < 1:
        raise ValueError("alpha must be a non-empty float32 device tensor")
    if bias is not None and (bias.shape != (n,) or bias.dtype != torch.bfloat16):
        raise ValueError(f"bias must have shape ({n},) and dtype bf16")
    if any(
        t is not None and t.device != x.device for t in (weight, sf, alpha, l1, bias)
    ):
        raise ValueError("all SVDQuant tensors must be on the same device")


def _linear_gemm_inputs(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    x, weight, weight_sf, alpha, _, l2t, l1, _, bias, out = inputs
    m, k = x.shape
    xq = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    sf = torch.empty(
        gemm._swizzled_sf_size(m, k // 16), dtype=torch.uint8, device=x.device
    )
    down = torch.empty((m, l2t.shape[1]), dtype=torch.bfloat16, device=x.device)
    return [xq, weight, sf, weight_sf, alpha, down, l1, bias, out]
