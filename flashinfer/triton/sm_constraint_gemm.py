from typing import Optional

import torch

from .kernels.sm_constraint_gemm import (
    matmul_persistent,
)
from .utils import check_device, check_dim, check_input, check_shape

def matmul_persistent(a: torch.Tensor, b: torch.Tensor, c: Optional[torch.Tensor] = None, alpha=1.0, beta=0.0, NUM_SMS: Optional[int] = None) -> torch.Tensor:
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

    # if NUM_SMS is not provided, use 70% of SMs on the device
    if NUM_SMS is None:
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count * 0.7

    if c is None:
        c = torch.empty_like(a, dtype=a.dtype)
    
    matmul_persistent(a, b, c, NUM_SMS, alpha, beta)
    return c
