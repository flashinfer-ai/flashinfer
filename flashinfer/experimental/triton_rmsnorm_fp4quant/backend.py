"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math

import torch


def _check_output(tensor, shape, dtype, device, name):
    if tensor.shape != shape or tensor.dtype != dtype:
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous and on {device}")


def _overlap(a, b):
    return (
        a.numel() > 0
        and b.numel() > 0
        and a.data_ptr() < b.data_ptr() + b.numel() * b.element_size()
        and b.data_ptr() < a.data_ptr() + a.numel() * a.element_size()
    )


def run(
    input,
    weight,
    y_fp4,
    block_scale,
    global_scale,
    eps,
    block_size,
    scale_format,
    is_sf_swizzled_layout,
    enable_pdl,
):
    """Explicit SM120 backend; validation must not read device tensor values."""
    if input.ndim not in (2, 3) or weight.ndim != 1:
        raise ValueError("Expected 2D/3D input and 1D weight")
    if input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("The Triton backend supports BF16 input and weight only")
    if not input.is_cuda or weight.device != input.device:
        raise ValueError("input and weight must be on the same CUDA device")
    if not input.is_contiguous() or not weight.is_contiguous():
        raise ValueError("input and weight must be contiguous")
    k = input.shape[-1]
    if k < 64 or k > 8192 or k % 16 or weight.numel() != k:
        raise ValueError("Expected weight[K] and K divisible by 16 in [64, 8192]")
    if block_size != 16 or scale_format not in (None, "e4m3"):
        raise ValueError("The Triton backend supports NVFP4 (block_size=16, e4m3) only")
    if enable_pdl:
        raise ValueError("The Triton backend does not support enable_pdl=True")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if torch.cuda.get_device_capability(input.device) != (12, 0):
        raise ValueError("The Triton backend currently supports SM120 only")
    if global_scale is not None:
        _check_output(global_scale, (1,), torch.float32, input.device, "global_scale")

    from .kernel import _rmsnorm_nvfp4_kernel
    import triton

    m = input.numel() // k
    q_shape = (*input.shape[:-1], k // 2)
    sf_shape = (
        (triton.cdiv(m, 128) * triton.cdiv(k // 16, 4) * 512,)
        if is_sf_swizzled_layout
        else (*input.shape[:-1], k // 16)
    )
    if y_fp4 is None:
        y_fp4 = torch.empty(q_shape, device=input.device, dtype=torch.float4_e2m1fn_x2)
    else:
        _check_output(y_fp4, q_shape, torch.float4_e2m1fn_x2, input.device, "y_fp4")
    if block_scale is None:
        block_scale = torch.empty(
            sf_shape, device=input.device, dtype=torch.float8_e4m3fn
        )
    else:
        _check_output(
            block_scale, sf_shape, torch.float8_e4m3fn, input.device, "block_scale"
        )
    inputs = [input, weight] + ([global_scale] if global_scale is not None else [])
    if _overlap(y_fp4, block_scale) or any(
        _overlap(out, src) for out in (y_fp4, block_scale) for src in inputs
    ):
        raise ValueError("Output buffers must not overlap inputs or each other")
    if m:
        with torch.cuda.device(input.device):
            _rmsnorm_nvfp4_kernel[(m,)](
                input,
                weight,
                y_fp4.view(torch.uint8),
                block_scale.view(torch.uint8),
                global_scale,
                K=k,
                EPS=float(eps),
                BLOCK_K=triton.next_power_of_2(k),
                SWIZZLED=is_sf_swizzled_layout,
                HAS_GLOBAL_SCALE=global_scale is not None,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return y_fp4, block_scale
