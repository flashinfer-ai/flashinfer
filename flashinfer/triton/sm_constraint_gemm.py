from typing import Optional

import torch
import triton

from .kernels.sm_constraint_gemm import (
    matmul_kernel_persistent,
)
from .utils import check_device, check_dim, check_input, check_shape


def matmul_persistent(a, b, c=None, alpha=1.0, beta=0.0, num_sms=None):
    # Check constraints.
    check_input(a)
    check_input(b)
    check_input(c)
    check_device([a, b, c])
    check_dim(2, a)
    check_dim(2, b)
    check_dim(2, c)

    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert a.shape[1] == c.shape[1], "Incompatible dimensions"

    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype) if c is None else c

    # Set num_sms to be 100% of the available SMs
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count if num_sms is None else num_sms
    
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        min(
            num_sms,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )

    matmul_kernel_persistent[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        alpha=alpha,
        beta=beta,
        NUM_SMS=num_sms,
    )
    return c
