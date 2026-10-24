"""W4A16 activation compilation and launch bindings."""

from __future__ import annotations

from cutlass.cutlass_dsl import Int32
from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.utils.device import current_cuda_stream
import cutlass.cute as cute
from b12x.moe._shared.kernels.activations import normalize_moe_activation
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.program_cache import register_program_cache
from dataclasses import replace
import torch
from b12x.moe._shared.kernels.w4a16.host import validate_activation
from b12x.moe._shared.kernels.w4a16.kernel.activation import W4A16ActivationKernel
from b12x.moe._shared.kernels.w4a16.kernel.config import (
    _cutlass_element_dtype,
    _fake_m_for_specialization,
    _normalize_activation_swiglu_params,
)
from b12x.moe._shared.kernels.w4a16.kernel.types import W4A16ActivationCompileResult


_ACTIVATION_CACHE: dict[tuple, W4A16ActivationCompileResult] = {}

register_program_cache(_ACTIVATION_CACHE)


def compile_w4a16_activation(
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
) -> W4A16ActivationCompileResult:
    cutlass_dtype = _cutlass_element_dtype(element_dtype)
    activation = normalize_moe_activation(activation)
    is_gated = validate_activation(activation)
    swiglu_limit, swiglu_alpha, swiglu_beta = _normalize_activation_swiglu_params(
        activation,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
    )
    kernel = W4A16ActivationKernel(
        rows=rows,
        intermediate_size=intermediate_size,
        activation=activation,
        element_dtype=element_dtype,
        fast_math=fast_math,
        swiglu_limit=swiglu_limit,
        swiglu_clamp_after_silu=swiglu_clamp_after_silu,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
    )
    cache_key = (
        "w4a16_activation",
        kernel.__cache_key__,
    )
    cached = _ACTIVATION_CACHE.get(cache_key)
    if cached is not None:
        return replace(cached, rows=rows)

    w13_shards = 2 if is_gated else 1
    compile_rows = _fake_m_for_specialization(rows)
    fc1_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (compile_rows * w13_shards * intermediate_size,),
        assumed_align=16,
    )
    activated_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (compile_rows * intermediate_size,),
        assumed_align=16,
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    compiled = b12x_compile(
        kernel,
        fc1_fake,
        activated_fake,
        Int32(compile_rows),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "moe.w4a16.activation",
            1,
            cache_key,
        ),
    )
    result = W4A16ActivationCompileResult(
        compiled=compiled,
        rows=rows,
        intermediate_size=intermediate_size,
        activation=activation,
        swiglu_limit=swiglu_limit,
        swiglu_clamp_after_silu=swiglu_clamp_after_silu,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
    )
    _ACTIVATION_CACHE[cache_key] = result
    return result


def _validate_activation_amax(
    activation_amax: torch.Tensor | None,
    *,
    layer_idx: int | None,
    num_experts: int,
    device: torch.device,
) -> int | None:
    if activation_amax is None:
        if layer_idx is not None:
            raise ValueError("layer_idx requires activation_amax")
        return None
    if layer_idx is None:
        raise ValueError("layer_idx is required when activation_amax is provided")
    if activation_amax.dtype != torch.float32:
        raise TypeError("activation_amax must be torch.float32")
    if activation_amax.device != device:
        raise ValueError("activation_amax must be on the same device as a_input")
    if not activation_amax.is_cuda:
        raise ValueError("activation_amax must be a CUDA tensor")
    if not activation_amax.is_contiguous():
        raise ValueError("activation_amax must be contiguous")
    if activation_amax.ndim != 3 or int(activation_amax.shape[2]) != 2:
        raise ValueError("activation_amax must have shape [num_layers, num_experts, 2]")
    if int(activation_amax.shape[1]) < int(num_experts):
        raise ValueError(
            "activation_amax expert dimension is smaller than the local expert count"
        )
    layer = int(layer_idx)
    if layer < 0 or layer >= int(activation_amax.shape[0]):
        raise ValueError(
            f"layer_idx {layer} is out of bounds for activation_amax with "
            f"{int(activation_amax.shape[0])} layers"
        )
    return layer
