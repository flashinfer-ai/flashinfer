# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# torch <-> CuTe DSL glue shared by the KPage kernels and metadata builders.

from typing import Optional

import torch
import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


def to_cute_tensor(
    t: torch.Tensor,
    dynamic_modes: int | tuple[int, ...],
    assumed_align: int = 16,
    divisibility: int = 1,
) -> cute.Tensor:
    tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=True)
    # FI exposes LSE as [M, H], whose BHSD view has head stride 1.
    # Preserve that layout instead of assuming the final axis is contiguous.
    tensor = tensor.mark_layout_dynamic(leading_dim=t.dim_order()[-1])
    if isinstance(dynamic_modes, int):
        dynamic_modes = (dynamic_modes,)
    stride_order = t.dim_order()
    for mode in dynamic_modes:
        tensor = tensor.mark_compact_shape_dynamic(
            mode=mode, stride_order=stride_order, divisibility=divisibility
        )
    return tensor


def _tensor_key(t: torch.Tensor):
    # Rank, dtype, stride order and broadcast dims fix the compiled layout;
    # shape values do not (they are compiled dynamic).  The order is computed
    # from the strides directly: Tensor.dim_order() costs tens of microseconds
    # per call, which adds up over the dozen tensors of a launch.
    strides = t.stride()
    order = tuple(sorted(range(t.ndim), key=lambda i: (-strides[i], i)))
    return (t.dtype, t.ndim, order, tuple(s == 0 for s in strides))


def compile_key(
    namespace: str, config: tuple, tensors: tuple[Optional[torch.Tensor], ...]
):
    device = next(t.device for t in tensors if t is not None)
    return (
        namespace,
        device.index,
        torch.cuda.get_device_capability(device),
        *config,
        *(_tensor_key(t) if t is not None else None for t in tensors),
    )


def fake_stream():
    return cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False)


def current_stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)
