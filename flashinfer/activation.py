"""
Copyright (c) 2024 by FlashInfer team.

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

import functools
from types import SimpleNamespace
from typing import Optional

import torch

from .jit import JitSpec
from .jit import gen_act_and_mul_module as gen_act_and_mul_module_impl
from .utils import (
    device_support_pdl,
    register_custom_op,
    register_fake_op,
    get_compute_capability,
)
from .fp4_quantization import get_fp4_quantization_module

silu_def_cu_str = r"""
__device__ __forceinline__ float silu(const float& val) {
  return val / (1.0f + __expf(-val));
}
"""

gelu_def_cu_str = r"""
__device__ __forceinline__ float gelu(const float& val) {
  constexpr float kAlpha = M_SQRT1_2;
  return val * 0.5f * (1.0f + ::erf(val * kAlpha));
}
"""

gelu_def_tanh_cu_str = r"""
__device__ __forceinline__ float gelu_tanh(const float& val) {
  const float cdf =
      0.5f * (1.0f + math::tanh((0.7978845608028654f * (val + 0.044715f * val * val * val))));
  return val * cdf;
}
"""

act_func_def_str = {
    "silu": silu_def_cu_str,
    "gelu": gelu_def_cu_str,
    "gelu_tanh": gelu_def_tanh_cu_str,
}


def gen_act_and_mul_module(act_func_name: str) -> JitSpec:
    return gen_act_and_mul_module_impl(act_func_name, act_func_def_str[act_func_name])


@functools.cache
def get_act_and_mul_module(act_func_name: str):
    module = gen_act_and_mul_module(act_func_name).build_and_load()

    # torch library for act_and_mul
    fname = f"{act_func_name}_and_mul"
    fn = getattr(module, fname)

    @register_custom_op(f"flashinfer::{fname}", mutates_args=("out",))
    def _act_and_mul(
        out: torch.Tensor, input: torch.Tensor, enable_pdl: Optional[bool] = None
    ) -> None:
        if enable_pdl is None:
            enable_pdl = device_support_pdl(input.device)
        fn(out, input, enable_pdl)

    @register_fake_op(f"flashinfer::{fname}")
    def _fake_act_and_mul(
        out: torch.Tensor, input: torch.Tensor, enable_pdl: Optional[bool] = None
    ) -> None:
        pass

    # Register the module
    return SimpleNamespace(**{fname: _act_and_mul})


def _check_shape(input: torch.Tensor, output: torch.Tensor) -> None:
    assert input.ndim == output.ndim, f"{input.ndim} != {output.ndim}"
    assert input.shape[:-1] == output.shape[:-1], (
        f"{input.shape[:-1]} != {output.shape[:-1]}"
    )
    assert input.shape[-1] == 2 * output.shape[-1], (
        f"{input.shape[-1]} != {2 * output.shape[-1]}"
    )


def silu_and_mul(
    input: torch.Tensor, out: torch.Tensor = None, enable_pdl: Optional[bool] = None
) -> torch.Tensor:
    r"""Fused SiLU and Mul operation.

    ``silu(input[..., :hidden_size]) * input[..., hidden_size:]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (..., 2 * hidden_size).

    out: Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor inplace.

    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    Returns
    -------
    output: torch.Tensor
        Output tensor, shape (..., hidden_size).
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")
    if out is not None:
        _check_shape(input, out)
    else:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )
    get_act_and_mul_module("silu").silu_and_mul(
        out,
        input,
        enable_pdl,
    )
    return out


def silu_and_mul_fp4_batched_quantize(
    a,
    mask,
    a_global_sf,
    sf_vec_size=16,
):
    """
    Silu and multiply and quantize batched input tensor to NVFP4 format with mask.

    Parameters:
        a (torch.Tensor): Input tensor of shape [B, M, K] with dtype fp16/bf16.
        a_global_sf (torch.Tensor): Global scale factor of shape [1] with dtype float32.
        mask (torch.Tensor): Mask tensor to apply before quantization.
        sf_vec_size (int, optional): Scale factor vector size. Defaults to 16.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Quantized tensor of shape [B, M, K/2] with dtype FLOAT4_E2M1X2
            - Scale factors tensor with shape determined by layout and sf_vec_size
    """
    major, minor = get_compute_capability(a.device)
    device_arch = f"{major * 10 + minor}"
    a_fp4, a_sf = get_fp4_quantization_module(
        device_arch
    ).silu_and_mul_fp4_batched_quantize_sm100(
        a,
        mask,
        a_global_sf,
        sf_vec_size,
    )
    return a_fp4, a_sf


def gelu_tanh_and_mul(
    input: torch.Tensor, out: torch.Tensor = None, enable_pdl: Optional[bool] = None
) -> torch.Tensor:
    r"""Fused GeLU Tanh and Mul operation.

    ``gelu(tanh(input[..., :hidden_size])) * input[..., hidden_size:]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (..., 2 * hidden_size).

    out: Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor inplace.

    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    Returns
    -------
    output: torch.Tensor
        Output tensor, shape (..., hidden_size).
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")
    if out is not None:
        _check_shape(input, out)
    else:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )
    get_act_and_mul_module("gelu_tanh").gelu_tanh_and_mul(out, input, enable_pdl)
    return out


def gelu_and_mul(
    input: torch.Tensor, out: torch.Tensor = None, enable_pdl: Optional[bool] = None
) -> torch.Tensor:
    r"""Fused GeLU and Mul operation.

    ``gelu(input[..., :hidden_size]) * input[..., hidden_size:]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (..., 2 * hidden_size).

    out: Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor inplace.

    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    Returns
    -------
    output: torch.Tensor
        Output tensor, shape (..., hidden_size).
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")
    if out is not None:
        _check_shape(input, out)
    else:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )
    get_act_and_mul_module("gelu").gelu_and_mul(out, input, enable_pdl)
    return out
