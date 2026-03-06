"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Fused Activation + Multiply kernels using CuTe-DSL
===================================================

Implements silu_and_mul, gelu_and_mul, and gelu_tanh_and_mul operations
as CuTe-DSL kernels. This is an alternative backend to the CUDA JIT path,
with static vector size computation and pure Python kernel definitions.
"""

import functools
import math
from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from .fp4_common import get_sm_version

# 128-bit vectorized loads
COPY_BITS = 128

# Math constants
_M_SQRT1_2 = math.sqrt(0.5)
_GELU_TANH_COEFF = math.sqrt(2.0 / math.pi)


# =============================================================================
# Activation Functions
# =============================================================================


def silu_f32(x):
    """SiLU activation: x * sigmoid(x)"""
    return x * cute.arch.rcp_approx(1.0 + cute.math.exp(-x, fastmath=True))


def gelu_f32(x):
    """GeLU activation (exact): x * 0.5 * (1 + erf(x / sqrt(2)))"""
    return x * 0.5 * (1.0 + cute.math.erf(x * Float32(_M_SQRT1_2)))


def gelu_tanh_f32(x):
    """GeLU activation (tanh approximation):
    x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    inner = Float32(_GELU_TANH_COEFF) * (x + Float32(0.044715) * x * x * x)
    return x * 0.5 * (1.0 + cute.math.tanh(inner, fastmath=True))


# =============================================================================
# CuTe-DSL Kernel Class
# =============================================================================


class ActAndMulKernel:
    """
    Fused Activation + Multiply Kernel.

    Computes: out[i] = activation(input[..., i]) * input[..., d + i]
    where input has shape (num_tokens, 2 * d) and out has shape (num_tokens, d).

    The activation function is selected at compile time via act_func_name,
    and vector size is statically computed based on dtype and d.
    """

    _VALID_ACT_FUNCS: frozenset[str] = frozenset({"silu", "gelu", "gelu_tanh"})

    def __init__(
        self,
        dtype: cutlass.Numeric,
        d: int,
        act_func_name: str,
    ):
        """Initialize kernel parameters and compute vector size for 128-bit loads.

        Parameters
        ----------
        dtype : cutlass.Numeric
            Element type (Float16 or BFloat16).
        d : int
            Hidden dimension (half of the last input dimension).
        act_func_name : str
            Activation function: "silu", "gelu", or "gelu_tanh".
        """
        if act_func_name not in self._VALID_ACT_FUNCS:
            raise ValueError(
                f"Unknown activation: {act_func_name!r}. "
                f"Expected one of {self._VALID_ACT_FUNCS}"
            )
        if d <= 0:
            raise ValueError(f"d must be positive, got {d}")
        self.dtype = dtype
        self.d = d
        self.act_func_name = act_func_name

        # Compute optimal vector size for 128-bit loads.
        # Automatically handles non-power-of-2 dimensions by reducing vec_size
        # until d is evenly divisible.
        elem_bytes = dtype.width // 8
        vs = COPY_BITS // 8 // elem_bytes
        while vs > 1 and d % vs != 0:
            vs //= 2
        self.vec_size = vs

        self.num_threads = min(d // vs, 1024)
        self.num_iters = (d // vs + self.num_threads - 1) // self.num_threads

    @cute.jit
    def __call__(
        self,
        mOut: cute.Tensor,
        mInput: cute.Tensor,
        M: Int32,
        stream,
    ):
        """Host function to launch the kernel."""
        self.kernel(mOut, mInput, M).launch(
            grid=[M, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mOut: cute.Tensor,
        mInput: cute.Tensor,
        M: Int32,
    ):
        """Device kernel: fused activation + multiply.

        Each block processes one token. Threads iterate over d elements
        with a stride of num_threads * vec_size. The inner vec_size loop
        is unrolled at compile time.
        """
        token_idx = cute.arch.block_idx()[0]
        tidx = cute.arch.thread_idx()[0]
        d = self.d
        num_threads = self.num_threads
        vec_size = self.vec_size
        num_vecs = d // vec_size

        if token_idx >= M:
            return

        # Strided iteration: each thread processes elements
        # tidx, tidx + num_threads, tidx + 2*num_threads, ...
        for iter_idx in range(self.num_iters):
            vec_idx = tidx + iter_idx * num_threads
            if vec_idx < num_vecs:
                # Inner loop is unrolled (vec_size is a compile-time constant)
                for i in range(vec_size):
                    col = vec_idx * vec_size + i

                    # Load activation input and gate input, compute in float32
                    x_val = Float32(mInput[token_idx, col])
                    y_val = Float32(mInput[token_idx, d + col])

                    # Apply activation (compile-time dispatch)
                    if cutlass.const_expr(self.act_func_name == "silu"):
                        act_val = silu_f32(x_val)
                    elif cutlass.const_expr(self.act_func_name == "gelu"):
                        act_val = gelu_f32(x_val)
                    elif cutlass.const_expr(self.act_func_name == "gelu_tanh"):
                        act_val = gelu_tanh_f32(x_val)

                    mOut[token_idx, col] = act_val * y_val


# =============================================================================
# Compilation and Caching
# =============================================================================


@functools.lru_cache(maxsize=None)
def _get_compiled_kernel(
    act_func_name: str,
    d: int,
    is_fp16: bool,
    sm_version: int,  # noqa: ARG001 — used as cache key, not in body
) -> Callable:
    """
    Get a compiled kernel closure that takes torch.Tensor directly.

    Uses TVM-FFI for efficient tensor passing. Cached by
    (act_func_name, d, is_fp16, sm_version) — in practice, d and dtype
    are fixed per model, so this compiles once per activation function.
    """
    cutlass_dtype = cutlass.Float16 if is_fp16 else cutlass.BFloat16

    kernel_obj = ActAndMulKernel(cutlass_dtype, d, act_func_name)

    # Symbolic size for dynamic num_tokens dimension
    sym_m = cute.sym_int()

    # Fake tensors for compilation (TVM-FFI)
    input_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype, (sym_m, 2 * d), stride_order=(1, 0), assumed_align=128
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype, (sym_m, d), stride_order=(1, 0), assumed_align=128
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        out_fake,
        input_fake,
        Int32(1),  # Dummy M
        stream_fake,
        options="--enable-tvm-ffi",
    )

    def tensor_api(
        out: torch.Tensor,
        input: torch.Tensor,
        M: int,
    ) -> None:
        """Runtime API that passes torch tensors directly via TVM-FFI."""
        compiled_kernel(out, input, Int32(M))

    return tensor_api


# =============================================================================
# Public API
# =============================================================================


def act_and_mul(
    act_func_name: str,
    input: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Run fused activation + multiply using CuTe-DSL.

    Parameters
    ----------
    act_func_name : str
        Activation function name: "silu", "gelu", or "gelu_tanh".
    input : torch.Tensor
        Input tensor, shape (..., 2 * d). Must be float16 or bfloat16.
    out : torch.Tensor
        Output tensor, shape (..., d).
    """
    d = input.shape[-1] // 2
    if d == 0:
        return
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"act_and_mul: unsupported dtype {input.dtype}; "
            f"expected float16 or bfloat16"
        )
    if out.dtype != input.dtype:
        raise ValueError(
            f"act_and_mul: output dtype {out.dtype} must match "
            f"input dtype {input.dtype}"
        )
    is_fp16 = input.dtype == torch.float16
    sm_version = get_sm_version(input.device)

    input_2d = input.reshape(-1, 2 * d).contiguous()
    if not out.is_contiguous():
        raise ValueError("Output tensor must be contiguous for CuTe-DSL kernel")
    out_2d = out.reshape(-1, d)
    M = input_2d.shape[0]
    if M == 0:
        return

    tensor_api = _get_compiled_kernel(act_func_name, d, is_fp16, sm_version)
    tensor_api(out_2d, input_2d, M)
