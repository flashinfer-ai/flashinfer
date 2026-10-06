"""Public surface for ``gemm.tensor_fp8_linear``."""

from __future__ import annotations

import torch

from ..._lib.gating import default_is_supported
from ..blockscaled._linear import (
    TensorFP8LinearWeight as Weight,
)
from ..blockscaled.api import FixedBlockscaledQuery, mm, pack_weight, plan, query_from_call
from ._kernel import (
    is_tensor_fp8_linear_supported as _kernel_is_supported,
)
from . import META


def is_supported(device=None) -> bool:
    """Return whether the SM12x tensor-FP8 linear path is available."""
    kernel_supported, _ = _kernel_is_supported()
    return default_is_supported(device, requires=META.requires) and kernel_supported


@torch.inference_mode()
def prewarm(weight: Weight, token_counts, *, out_dtype=torch.bfloat16, stream=None) -> int:
    """Materialize heuristic plans before graph capture for serving capacities."""
    counts = sorted({int(count) for count in token_counts if int(count) > 0})
    for count in counts:
        source = torch.zeros((count, weight.in_features), device=weight.values.device,
                             dtype=torch.float8_e4m3fn)
        mm(source, weight, out_dtype=out_dtype, stream=stream)
    return len(counts)


__all__ = ["Weight", "FixedBlockscaledQuery", "plan", "query_from_call", "mm", "pack_weight", "prewarm", "is_supported"]
