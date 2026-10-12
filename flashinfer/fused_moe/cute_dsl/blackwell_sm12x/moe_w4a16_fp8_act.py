"""FC1-input FP8 activation quantization for the b12x W4A16 FP8-prefill mode.

Per-row (per-token) dynamic E4M3 quantization with an fp32 scale (amax / 448), written in
the sigma K-slot order the kernel's m16n8k32 A fragments expect (see
moe_w4a16_fp8_helpers.SIGMA16): inside every 16-wide K group, source element k lands in
slot  s = 4 * ((k % 8) // 2) + (k % 2) + 2 * (k // 8).

"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quant_rows_fp8_sigma_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    x = tl.load(x_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12)
    q = (x * (448.0 / amax)).to(tl.float8e4nv)
    k16 = offs % 16
    slot = 4 * ((k16 % 8) // 2) + (k16 % 2) + 2 * (k16 // 8)
    tl.store(q_ptr + row * K + (offs - k16) + slot, q, mask=mask)
    tl.store(s_ptr + row, amax / 448.0)


def quant_rows_fp8_sigma(
    x: torch.Tensor,
    out_q: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """x [M, K] bf16/fp16 -> (q [M, K] float8_e4m3fn in sigma order, scale [M] fp32)."""
    if x.ndim != 2 or not x.is_contiguous() or x.shape[1] == 0 or x.shape[1] % 16:
        raise ValueError("x must be contiguous [M, K] with positive K divisible by 16")
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("x must be a CUDA BF16 or FP16 tensor")
    m, k = x.shape
    q = (
        out_q
        if out_q is not None
        else torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    )
    s = (
        out_scale
        if out_scale is not None
        else torch.empty((m,), dtype=torch.float32, device=x.device)
    )
    for name, tensor, shape, dtype in (
        ("out_q", q, (m, k), torch.float8_e4m3fn),
        ("out_scale", s, (m,), torch.float32),
    ):
        if (
            tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != x.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} has an incompatible shape, dtype, device or stride"
            )
    if m:
        _quant_rows_fp8_sigma_kernel[(m,)](
            x, q, s, K=k, BLOCK=triton.next_power_of_2(k), num_warps=8
        )
    return q, s


def quant_rows_fp8_sigma_reference(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference of quant_rows_fp8_sigma (same rounding: RNE via .to(float8))."""
    xf = x.float()
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    q = (xf * (448.0 / amax)).to(torch.float8_e4m3fn)
    k = x.shape[-1]
    sigma = [2 * (s // 4) + (0, 1, 8, 9)[s % 4] for s in range(16)]
    idx = torch.tensor(sigma, device=x.device)
    idx = (torch.arange(0, k, 16, device=x.device)[:, None] + idx[None, :]).reshape(-1)
    return q.index_select(-1, idx).contiguous(), (amax / 448.0).squeeze(-1)


__all__ = ["quant_rows_fp8_sigma", "quant_rows_fp8_sigma_reference"]
