"""Prepared public API for fused MLA query projection."""
from __future__ import annotations

from typing import Literal, Optional, TypeAlias

import torch

from b12x.preparation import Plan
from ..._lib.gating import default_is_supported
from . import META
from ._preparation import plan

Mxfp8Weight: TypeAlias = tuple[torch.Tensor, torch.Tensor]
MlaQueryWeight: TypeAlias = torch.Tensor | Mxfp8Weight


def run(q_nope: torch.Tensor, weight: MlaQueryWeight, q_pe: torch.Tensor,
        out: torch.Tensor, *, plan: Plan,
        q_scale: Optional[torch.Tensor] = None,
        stream: Optional[object] = None) -> torch.Tensor:
    """Assemble the admitted exact-M MLA query into caller-owned ``out``."""
    if isinstance(weight, torch.Tensor):
        from . import _bf16
        return _bf16.run(q_nope, weight, q_pe, out, plan=plan, q_scale=q_scale, stream=stream)
    from .._shared import mxfp8_bmm
    values, scales = mxfp8_bmm._rhs_tensors(weight)
    stream_int = None if stream is None else int(mxfp8_bmm._torch_stream(stream, q_nope.device).cuda_stream)
    torch.ops.b12x.mla_query_projection_mxfp8(
        q_nope, values, scales, q_pe, q_scale, out, 1, plan.handle, stream_int,
    )
    return out


def can_implement(*, num_heads: int, max_m: int, nope_dim: int, latent_dim: int,
                  output_dtype: torch.dtype, weight_format: Literal["bf16", "mxfp8"] = "mxfp8",
                  device=None) -> bool:
    if not is_supported(device):
        return False
    if weight_format == "bf16":
        from . import _bf16
        return _bf16.can_implement(num_heads=num_heads, max_m=max_m, nope_dim=nope_dim,
                                   latent_dim=latent_dim, output_dtype=output_dtype, device=device)
    if weight_format == "mxfp8":
        from .._shared import mxfp8_bmm
        return mxfp8_bmm.can_implement_mla_query_projection(
            batch=num_heads, max_m=max_m, n=latent_dim, k=nope_dim,
            output_dtype=output_dtype, b_major="n", sf_axis="n")
    return False


def is_supported(device=None) -> bool:
    return default_is_supported(device, requires=META.requires)


def clear_caches() -> None:
    from .._shared import mxfp8_bmm
    mxfp8_bmm.clear_mla_query_projection_caches()


__all__ = ["Mxfp8Weight", "MlaQueryWeight", "plan", "run", "can_implement", "is_supported"]
