"""FP6 operand construction shared by numerical tests and benchmarks."""

from __future__ import annotations

import torch
from b12x._lib.fp6 import quantize_grouped_mxfp6_torch


def _bf16_global_scale(amax: float) -> torch.Tensor:
    return torch.tensor(
        [float(torch.finfo(torch.float8_e4m3fn).max) * 28.0 / max(amax, 1e-6)],
        dtype=torch.float32,
        device="cuda",
    )


def _quantize_bf16_matrix(
    bf16: torch.Tensor, fmt: str = "e3m2"
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = bf16.shape
    row_counts = torch.tensor([m], dtype=torch.int32, device=bf16.device)
    gs = _bf16_global_scale(float(bf16.abs().max().item()))
    packed, scale_view = quantize_grouped_mxfp6_torch(
        bf16.unsqueeze(0),
        row_counts,
        gs,
        fmt=fmt,  # type: ignore[arg-type]
        bf16_round=True,
    )
    return packed[:, :, 0].contiguous(), scale_view
