"""W4A16 MoE activation kernels."""

from __future__ import annotations

from cutlass.cutlass_dsl import Int32
from b12x.moe._shared.kernels.activations import SITU
from b12x.moe._shared.kernels.activations import SITU_DEFAULT_BETA
from b12x.moe._shared.kernels.activations import SITU_DEFAULT_LINEAR_BETA
from b12x.moe._shared.kernels.activations import SWIGLUOAI_UNINTERLEAVE
import cuda.bindings.driver as cuda
import cutlass.cute as cute
import cutlass
from b12x.moe._shared.kernels.activations import normalize_moe_activation
from b12x.moe._shared.kernels.w4a16.host import validate_activation
from b12x.moe._shared.kernels.w4a16.kernel.config import (
    _covering_count,
    _normalize_activation_swiglu_params,
)


class W4A16ActivationKernel:
    def __init__(
        self,
        *,
        rows: int,
        intermediate_size: int,
        activation: str,
        element_dtype: str = "bf16",
        fast_math: bool = True,
        swiglu_limit: float | None = None,
        swiglu_clamp_after_silu: bool = False,
        swiglu_alpha: float | None = None,
        swiglu_beta: float | None = None,
    ):
        activation = normalize_moe_activation(activation)
        is_gated = validate_activation(activation)
        swiglu_limit, swiglu_alpha, swiglu_beta = _normalize_activation_swiglu_params(
            activation,
            swiglu_limit,
            swiglu_alpha,
            swiglu_beta,
        )
        if element_dtype not in {"bf16", "fp16"}:
            raise ValueError(f"unsupported element_dtype {element_dtype!r}")
        if rows <= 0 or intermediate_size <= 0:
            raise ValueError("rows and intermediate_size must be positive")
        self.rows = int(rows)
        self.intermediate_size = int(intermediate_size)
        self.activation = activation
        self.is_gated = is_gated
        self.is_situ = activation == SITU
        self.is_swigluoai = activation == SWIGLUOAI_UNINTERLEAVE
        self.swiglu_clamp_after_silu = bool(swiglu_clamp_after_silu)
        self.has_swiglu_limit = swiglu_limit is not None
        self.swiglu_limit = 0.0 if swiglu_limit is None else swiglu_limit
        self.swiglu_alpha = float(swiglu_alpha)
        self.swiglu_beta = float(swiglu_beta)
        self.element_dtype = element_dtype
        self.is_fp16 = element_dtype == "fp16"
        self.fast_math = bool(fast_math)
        self.cta_threads = 256

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (
            self.intermediate_size,
            self.activation,
            self.is_gated,
            self.is_situ,
            self.is_swigluoai,
            self.has_swiglu_limit,
            self.swiglu_limit,
            self.swiglu_clamp_after_silu,
            self.swiglu_alpha,
            self.swiglu_beta,
            self.element_dtype,
            self.fast_math,
            self.cta_threads,
        )

    @cute.jit
    def _cast_elem(self, x: cutlass.Float32):
        if cutlass.const_expr(self.is_fp16):
            return cutlass.Float16(x)
        return cutlass.BFloat16(x)

    @cute.jit
    def _clamp_swiglu_inputs(
        self,
        gate: cutlass.Float32,
        up: cutlass.Float32,
    ):
        if cutlass.const_expr(self.has_swiglu_limit):
            limit = cutlass.Float32(self.swiglu_limit)
            neg_limit = cutlass.Float32(-self.swiglu_limit)
            if cutlass.const_expr(not self.swiglu_clamp_after_silu):
                if gate > limit:
                    gate = limit
            if up > limit:
                up = limit
            if up < neg_limit:
                up = neg_limit
        return gate, up

    @cute.jit
    def __call__(
        self,
        fc1_flat: cute.Tensor,
        activated_flat: cute.Tensor,
        active_rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        total = active_rows * Int32(self.intermediate_size)
        grid = (_covering_count(total, self.cta_threads), 1, 1)
        self.kernel(fc1_flat, activated_flat, active_rows).launch(
            grid=grid,
            block=[self.cta_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        fc1_flat: cute.Tensor,
        activated_flat: cute.Tensor,
        active_rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        idx = Int32(bidx) * Int32(self.cta_threads) + Int32(tidx)
        total = Int32(active_rows) * Int32(self.intermediate_size)
        if idx < total:
            if cutlass.const_expr(self.is_gated):
                row = idx // Int32(self.intermediate_size)
                col = idx - row * Int32(self.intermediate_size)
                base = row * Int32(2 * self.intermediate_size)
                gate = fc1_flat[base + col].to(cutlass.Float32)
                up = fc1_flat[base + Int32(self.intermediate_size) + col].to(
                    cutlass.Float32
                )
                gate, up = self._clamp_swiglu_inputs(gate, up)
                sigmoid_arg = gate
                up_term = up
                if cutlass.const_expr(self.is_swigluoai):
                    sigmoid_arg = cutlass.Float32(self.swiglu_alpha) * gate
                    up_term = up + cutlass.Float32(self.swiglu_beta)
                if cutlass.const_expr(self.fast_math):
                    exp_neg_gate = cute.math.exp(-sigmoid_arg, fastmath=True)
                else:
                    exp_neg_gate = cute.math.exp(-sigmoid_arg, fastmath=False)
                sigmoid = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + exp_neg_gate)
                silu = gate * sigmoid
                if cutlass.const_expr(
                    self.has_swiglu_limit and self.swiglu_clamp_after_silu
                ):
                    if silu > cutlass.Float32(self.swiglu_limit):
                        silu = cutlass.Float32(self.swiglu_limit)
                if cutlass.const_expr(self.is_situ):
                    beta = cutlass.Float32(SITU_DEFAULT_BETA)
                    linear_beta = cutlass.Float32(SITU_DEFAULT_LINEAR_BETA)
                    situ_gate = (
                        beta
                        * cute.math.tanh(gate / beta, fastmath=self.fast_math)
                        * sigmoid
                    )
                    situ_up = linear_beta * cute.math.tanh(
                        up / linear_beta,
                        fastmath=self.fast_math,
                    )
                    activated_flat[idx] = self._cast_elem(situ_gate * situ_up)
                elif cutlass.const_expr(self.is_swigluoai):
                    activated_flat[idx] = self._cast_elem(silu * up_term)
                else:
                    activated_flat[idx] = self._cast_elem(
                        self._cast_elem(silu) * self._cast_elem(up_term)
                    )
            else:
                x = fc1_flat[idx].to(cutlass.Float32)
                if x < cutlass.Float32(0.0):
                    x = cutlass.Float32(0.0)
                activated_flat[idx] = self._cast_elem(x * x)
