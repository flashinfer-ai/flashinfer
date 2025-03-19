import torch
from typing import Optional
import triton
import triton.language as tl

from flashinfer.triton.kernels.gemm_constraint_sm import batched_matmul_kernel

@triton.jit
def bmm_fp8_constraint_sm(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    sm_count: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""BMM FP8 with constraint on the number of SMs

    Parameters
    ----------
    A: torch.Tensor
        Input tensor, shape (b, m, k), fp8 e4m3 or fp8 e5m2.

    B: torch.Tensor
        Mat2 tensor, shape (b, k, n), should be column major, fp8 e4m3 or fp8 e5m2.

    A_scale: torch.Tensor
        Scale tensor for A, float.

    B_scale: torch.Tensor
        Scale tensor for B, float.

    dtype: torch.dtype
        out dtype, bf16 or fp16.
    
    sm_count: int
        Number of SMs to use.

    out: Optional[torch.Tensor]
        Out tensor, shape (b, m, n), bf16 or fp16, defaults to ``None``.

    Returns
    -------
    out: torch.Tensor
        Out tensor, shape (b, m, n), bf16 or fp16.

    Examples
    --------
    >>> import torch
    >>> import torch.nn.functional as F
    >>> import flashinfer
    >>> def to_float8(x, dtype=torch.float8_e4m3fn):
    ...     finfo = torch.finfo(dtype)
    ...     min_val, max_val = x.aminmax()
    ...     amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    ...     scale = finfo.max / amax
    ...     x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    ...     return x_scl_sat.to(dtype), scale.float().reciprocal()
    >>>
    >>> input = torch.randn([16, 48, 64], device="cuda", dtype=torch.bfloat16)
    >>> input_fp8, input_inv_s = to_float8(input, dtype=torch.float8_e4m3fn)
    >>> # column major weight
    >>> weight = torch.randn([16, 80, 64], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    >>> weight_fp8, weight_inv_s = to_float8(weight, dtype=torch.float8_e4m3fn)
    >>> out = flashinfer.bmm_fp8(input_fp8, weight_fp8, input_inv_s, weight_inv_s, torch.bfloat16)
    >>> out.shape
    torch.Size([16, 48, 80])
    >>> out.dtype
    torch.bfloat16
    """
    assert A.dtype in {torch.float8_e4m3fn, torch.float8_e5m2fn}, "A must be FP8"
    assert B.dtype in {torch.float8_e4m3fn, torch.float8_e5m2fn}, "B must be FP8"
    assert sm_count > 0, "sm_count must be greater than 0"

    b, m, k1 = A.shape
    _, k2, n = B.shape
    assert k1 == k2, "A and B must have the same number of columns"

    # TODO(yingyi): contiguous A and B and out???

    if out is None:
        out = torch.empty(
            (A.shape[0], A.shape[1], B.shape[2]),
            device=A.device,
            dtype=dtype,
        )
    else:
        # Ensure output tensor is properly shaped and contiguous
        assert out.shape == (b, m, n), f"Output shape mismatch: expected {(b, m, n)}, got {out.shape}"
        # out = out.contiguous()
    
    # Process each batch element
    grid = lambda META: (b, )
    
    batched_matmul_kernel[grid](
        A, B, out,
        A_scale, B_scale,
        b, m, n, k1,
        A.stride(0), B.stride(0), out.stride(0),
        NUM_SMS=sm_count,
    )
    
    
    return out