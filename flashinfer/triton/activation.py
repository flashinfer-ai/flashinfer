from collections.abc import Mapping
from typing import Optional

import torch
import triton  # type: ignore[import]

from flashinfer.triton.kernels.activation import silu_and_mul_kernel


def silu_and_mul(
    x: torch.Tensor,
    x_scale: Optional[torch.Tensor] = None,
    o_scale: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Sigmoid Linear Unit and Multiplication

    Computes `silu(x[:,:d]) * x[:, d:]`, where `d = x.shape[-1] // 2.

    If the scale of `x` is `x_scale`, the scale applied to the output
    is the square of that, as the sigmoid function ranges in (0, 1).

    Args:
        x: The input tensor, of shape `(b, 2 * d)`.
        x_scale: An optional scale which was applied to `x`.
        o_scale: The scale to apply to the output.
        dtype: The desired output dtype.

    Returns:
        The output activation, of shape `(b, d)`.
    """

    b, n = x.shape

    assert n % 2 == 0
    d = n // 2

    o_dtype = dtype or x.dtype
    o = torch.empty((b, d), dtype=o_dtype, device=x.device)

    def grid(meta: Mapping[str, int]) -> tuple[int, int]:
        return (b, triton.cdiv(d, meta["BLOCK_SIZE"]))

    silu_and_mul_kernel[grid](
        o_ptr=o,
        o_stride=o.stride(0),
        o_scale_ptr=o_scale,
        x_ptr=x,
        x_stride=x.stride(0),
        x_scale_ptr=x_scale,
        d=d,
        BLOCK_SIZE=1024,
        HAS_X_SCALE=x_scale is not None,
        HAS_O_SCALE=o_scale is not None,
    )

    return o
