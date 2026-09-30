"""Prepared dtype dispatch for :func:`b12x.gemm.bmm`."""
from __future__ import annotations

from typing import Literal, Optional
import torch
from b12x.preparation import Plan
from ..._lib.gating import default_is_supported
from . import META
from ._preparation import plan

_MXFP8_SPECIALIZATION = ("bfloat16", "float8_e4m3fn", "float8_e8m0fnu", "bfloat16", 32)


def _require_specialization(*, a_dtype, b_dtype, sf_dtype, c_dtype, sf_vec_size):
    if (a_dtype, b_dtype, sf_dtype, c_dtype, sf_vec_size) != _MXFP8_SPECIALIZATION:
        raise NotImplementedError("gemm.bmm supports only BF16 x rowwise-MXFP8 -> BF16")


def bmm(lhs: torch.Tensor, rhs: tuple[torch.Tensor, torch.Tensor], out: torch.Tensor, *,
        plan: Plan, a_dtype: str, b_dtype: str, sf_dtype: str,
        c_dtype: str, sf_vec_size: int, b_major: Literal["k", "n"] = "k",
        sf_axis: Literal["k", "n"] = "k", stream: Optional[object] = None) -> torch.Tensor:
    """Run the admitted exact-M BMM into caller-owned ``out``."""
    _require_specialization(a_dtype=a_dtype, b_dtype=b_dtype, sf_dtype=sf_dtype,
                            c_dtype=c_dtype, sf_vec_size=sf_vec_size)
    if b_major != sf_axis:
        raise ValueError("BMM requires matching weight and scale axes")
    from .._shared import mxfp8_bmm
    values, scales = mxfp8_bmm._rhs_tensors(rhs)
    stream_int = None if stream is None else int(mxfp8_bmm._torch_stream(stream, lhs.device).cuda_stream)
    torch.ops.b12x.bmm_mxfp8(lhs, values, scales, out,
                              int(mxfp8_bmm._coerce_b_major(b_major)), plan.handle, stream_int)
    return out


def can_implement_bmm(*, batch: int, max_m: int, n: int, k: int, a_dtype: str,
                      b_dtype: str, sf_dtype: str, c_dtype: str, sf_vec_size: int,
                      b_major: Literal["k", "n"] = "k", sf_axis: Literal["k", "n"] = "k", device=None) -> bool:
    from .._shared import mxfp8_bmm
    return ((a_dtype, b_dtype, sf_dtype, c_dtype, sf_vec_size) == _MXFP8_SPECIALIZATION and
            is_bmm_supported(device) and mxfp8_bmm.can_implement(batch=batch, max_m=max_m,
            n=n, k=k, b_major=b_major, sf_axis=sf_axis))


def is_bmm_supported(device=None) -> bool:
    return default_is_supported(device, requires=META.requires)


def clear_bmm_caches() -> None:
    from .._shared import mxfp8_bmm
    mxfp8_bmm.clear_caches()


__all__ = ["plan", "bmm", "can_implement_bmm", "is_bmm_supported"]
