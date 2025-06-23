from typing import Optional

import torch
import triton  # type: ignore[import]

from flashinfer.triton.kernels.norm import rms_norm_kernel


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    in_scale: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
) -> None:
    """RMS norm.

    Computes `out[i,j] = x[i,j] * weight[j] / sqrt(eps + sum(x[i]^2) / n)`.
    """

    b, n = x.shape

    block_size = triton.next_power_of_2(n)
    num_warps = max(8, min(32, block_size // 256))

    rms_norm_kernel[(b,)](
        n=n,
        b=b,
        x_ptr=x,
        x_stride=x.stride(0),
        x_scale_ptr=in_scale,
        r_ptr=None,
        r_stride=0,
        w_ptr=weight,
        o_ptr=out,
        o_stride=out.stride(0),
        o_scale_ptr=out_scale,
        EPS=eps,
        BLOCK_SIZE=block_size,
        HAS_IN_SCALE=in_scale is not None,
        HAS_OUT_SCALE=out_scale is not None,
        HAS_OUTPUT=True,
        HAS_RESIDUAL=False,
        num_warps=num_warps,
    )


def rms_norm_add_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    x_out: Optional[torch.Tensor] = None,
    x_in_scale: Optional[torch.Tensor] = None,
    x_out_scale: Optional[torch.Tensor] = None,
) -> None:
    """In-place RMS norm with fused residual addition.

    Computes `r = r + x`, followed by `x = rmsnorm(r)`.
    """

    b, n = x.shape

    assert x.shape == residual.shape
    assert x.stride(0) == residual.stride(0)

    block_size = triton.next_power_of_2(n)
    num_warps = min(32, triton.cdiv(block_size, 32))

    rms_norm_kernel[(b,)](
        n=n,
        b=b,
        x_ptr=x,
        x_stride=x.stride(0),
        x_scale_ptr=x_in_scale,
        r_ptr=residual,
        r_stride=residual.stride(0),
        w_ptr=weight,
        o_ptr=x_out,
        o_stride=x_out.stride(0) if x_out is not None else 0,
        o_scale_ptr=x_out_scale,
        EPS=eps,
        BLOCK_SIZE=block_size,
        HAS_IN_SCALE=x_in_scale is not None,
        HAS_OUT_SCALE=x_out_scale is not None,
        HAS_OUTPUT=x_out is not None,
        HAS_RESIDUAL=True,
        num_warps=num_warps,
    )
