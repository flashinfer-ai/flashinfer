"""Convert non-bf16 operands for GDN kernels that reinterpret via ``cutlass.BFloat16``."""

from typing import Optional

import torch


def as_bf16(*tensors: Optional[torch.Tensor]) -> tuple:
    """Return tensors as bf16 (``None`` and already-bf16 values unchanged)."""
    return tuple(
        t if t is None or t.dtype == torch.bfloat16 else t.to(torch.bfloat16)
        for t in tensors
    )
