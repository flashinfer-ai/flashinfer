"""Per-row E4M3 quantization for the absorbed 576-dimensional MLA input."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_rows(
    X,
    Y,
    Scale,
    N: tl.constexpr,
    D: tl.constexpr,
    SX: tl.constexpr,
    SY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    x = tl.load(X + r * SX + d, d < D, 0).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x), 0), 1e-12) / 448.0
    tl.store(Y + r * SY + d, (x / s).to(Y.dtype.element_ty), d < D)
    tl.store(Scale + r, s)


def quantize_rows(x, out=None, scales=None):
    if not x.is_cuda or not x.is_contiguous() or x.shape[-1] != 576:
        raise ValueError("Expected contiguous CUDA input with last dimension 576.")
    if x.dtype != torch.bfloat16:
        raise ValueError("Expected BF16 input.")
    rows = x.numel() // 576
    if out is None:
        out = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    if scales is None:
        scales = torch.empty(x.shape[:-1], device=x.device, dtype=torch.float32)
    if (
        out.shape != x.shape
        or out.dtype != torch.float8_e4m3fn
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "Expected contiguous E4M3 output matching the input shape/device."
        )
    if (
        scales.shape != x.shape[:-1]
        or scales.dtype != torch.float32
        or scales.device != x.device
        or not scales.is_contiguous()
    ):
        raise ValueError(
            "Expected one contiguous FP32 scale per row on the input device."
        )
    if rows == 0:
        return out, scales
    _quantize_rows[(rows,)](x, out, scales, rows, 576, 576, 576, 1024)
    return out, scales
