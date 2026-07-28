"""Operand dtype coercion for the bf16-only GDN decode kernels.

The GDN decode kernels move ``q``/``k``/``v``/``a``/``b`` through fragments declared
``cutlass.BFloat16`` and store results with ``cutlass.BFloat16(...)``. A non-bf16
tensor handed to ``from_dlpack`` therefore reaches the kernel as *reinterpreted*
bits rather than a converted value, which produces silently wrong output. The public
``gdn_decode`` API documents fp16 q/k/v, and already converts the result back to the
caller's dtype, so these operands are converted here instead of rejected.

Tensors the kernels read through indexed scalar loads (``A_log``, ``dt_bias``, slot
indices) are genuinely polymorphic and must be keyed in the compile cache instead of
coerced.
"""

from typing import Optional

import torch


def as_bf16(*tensors: Optional[torch.Tensor]) -> tuple:
    """Convert bf16-only kernel operands to bf16, preserving argument order.

    ``None`` passes through unchanged so optional operands can be forwarded as-is.
    Already-bf16 tensors are returned untouched, so the common path adds no copy.
    """
    return tuple(
        t if t is None or t.dtype == torch.bfloat16 else t.to(torch.bfloat16)
        for t in tensors
    )
