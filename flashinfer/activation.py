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

from types import SimpleNamespace

import torch

from .jit import gen_act_and_mul_module, has_prebuilt_ops, load_cuda_ops    # noqa: F401
from .utils import get_cuda_stream, register_custom_op, register_fake_op

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


_jit_modules = {}


def get_act_and_mul_module(act_func_name: str):
    global _jit_modules
    if act_func_name not in _jit_modules:
        if has_prebuilt_ops:
            from . import _kernels  # type: ignore[attr-defined]

            module = _kernels
        else:
            module = gen_act_and_mul_module(
                act_func_name, act_func_def_str[act_func_name]
            )

        # torch library for act_and_mul
        fname = f"{act_func_name}_and_mul"
        fn = getattr(module, fname)

        @register_custom_op(f"flashinfer::{fname}", mutates_args=("out",))
        def _act_and_mul(out: torch.Tensor, input: torch.Tensor) -> None:
            with input.device as device:  # device guard
                fn(out, input, get_cuda_stream(device))

        @register_fake_op(f"flashinfer::{fname}")
        def _fake_act_and_mul(out: torch.Tensor, input: torch.Tensor) -> None:
            pass

        # Register the module
        _jit_modules[act_func_name] = SimpleNamespace(**{fname: _act_and_mul})

    return _jit_modules[act_func_name]


def _check_shape(input: torch.Tensor, output: torch.Tensor) -> None:
    assert input.ndim == output.ndim, f"{input.ndim} != {output.ndim}"
    assert (
        input.shape[:-1] == output.shape[:-1]
    ), f"{input.shape[:-1]} != {output.shape[:-1]}"
    assert (
        input.shape[-1] == 2 * output.shape[-1]
    ), f"{input.shape[-1]} != {2 * output.shape[-1]}"


def silu_and_mul(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    r"""Fused SiLU and Mul operation.

    ``silu(input[..., :hidden_size]) * input[..., hidden_size:]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (..., 2 * hidden_size).

    out: Optional[torch.Tensor]
        The the output tensor, if specified, the kernel will update this tensor inplace.

    Returns
    -------
    output: torch.Tensor
        Output tensor, shape (..., hidden_size).
    """
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
    )
    return out


def gelu_tanh_and_mul(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    r"""Fused GeLU Tanh and Mul operation.

    ``gelu(tanh(input[..., :hidden_size])) * input[..., hidden_size:]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (..., 2 * hidden_size).

    out: Optional[torch.Tensor]
        The the output tensor, if specified, the kernel will update this tensor inplace.

    Returns
    -------
    output: torch.Tensor
        Output tensor, shape (..., hidden_size).
    """
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
    get_act_and_mul_module("gelu_tanh").gelu_tanh_and_mul(out, input)
    return out


def gelu_and_mul(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    r"""Fused GeLU and Mul operation.

    ``gelu(input[..., :hidden_size]) * input[..., hidden_size:]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (..., 2 * hidden_size).

    out: Optional[torch.Tensor]
        The the output tensor, if specified, the kernel will update this tensor inplace.

    Returns
    -------
    output: torch.Tensor
        Output tensor, shape (..., hidden_size).
    """
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
    get_act_and_mul_module("gelu").gelu_and_mul(out, input)
    return out
