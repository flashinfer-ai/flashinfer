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

import functools
from typing import List, Literal, Optional, Tuple

import torch

from ..api_logging import flashinfer_api
from ..autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ..fused_moe.utils import (
    get_hybrid_num_tokens_buckets,
    map_to_hybrid_bucket_uncapped,
)
from ..jit.gemm import gen_gemm_sm100_module_cutlass_nvfp4_svdquant
from ..trace.templates.gemm import (
    mm_nvfp4_svdquant_trace,
    nvfp4_quantize_smooth_trace,
    svdquant_linear_trace,
)
from ..utils import (
    _get_cache_buf,
    backend_requirement,
    device_support_pdl,
    supported_compute_capability,
)

DEFAULT_WORKSPACE_SIZE = 32 * 1024 * 1024

# The fused kernel accumulates the rank-r BF16 LoRA-up into the NVFP4 residual accumulator.
# The rank is inferred from the d/l1 shapes and must be a positive multiple of the collective's
# rank granularity (CollectiveMmaLoRA::LoRaK); ranks 32-128 are validated.
SVDQUANT_LORA_RANK_GRANULARITY = 32


def _pad_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def _swizzled_sf_size(rows: int, sf_cols: int) -> int:
    """Size of the 128x4-swizzled block-scale layout for a [rows, sf_cols] scale matrix."""
    return _pad_up(rows, 128) * _pad_up(sf_cols, 4)


@functools.cache
def get_nvfp4_svdquant_module():
    """JIT-build and load the SM100 CUTLASS NVFP4 SVDQuant module."""
    return gen_gemm_sm100_module_cutlass_nvfp4_svdquant().build_and_load()


def _nvfp4_svdquant_gemm_runner(enable_pdl: bool):
    module = get_nvfp4_svdquant_module()

    class Nvfp4SvdquantGemmRunner(TunableRunner):
        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            return list(range(module.nvfp4_svdquant_gemm_tactic_num()))

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ):
            (a, b, a_sf, b_sf, alpha, d, l1, bias, out, workspace_buffer) = inputs
            module.nvfp4_svdquant_gemm(
                a,
                b,
                a_sf,
                b_sf,
                alpha,
                d,
                l1,
                bias,
                out,
                workspace_buffer,
                tactic,
                enable_pdl,
            )
            return out

    return Nvfp4SvdquantGemmRunner()


_NVFP4_SVDQUANT_GEMM_TUNING_CONFIG = TuningConfig(
    use_cuda_graph=True,
    use_cold_l2_cache=True,
    dynamic_tensor_specs=(
        DynamicTensorSpec(
            (0,),  # a_tensor_index
            (0,),
            get_hybrid_num_tokens_buckets,
            map_to_hybrid_bucket_uncapped,
        ),
    ),
    constraint_specs=(
        ConstraintSpec(
            2,  # a_sf tensor index: 1-D 128x4-swizzled scale buffer sized by (m, k/16)
            0,
            lambda shapes: _swizzled_sf_size(shapes[0][0], shapes[0][1] * 2 // 16),
        ),
        ConstraintSpec(
            5,  # d tensor index: [m, r] LoRA-down output (r kept from the real input)
            0,
            lambda shapes: shapes[0][0],
        ),
        ConstraintSpec(
            8,  # out tensor index
            0,
            lambda shapes: shapes[0][0],
        ),
        ConstraintSpec(
            9,  # workspace_buffer index: scratch; exclude its (resizable) size from the
            0,  # cache key so a mid-tune resize never causes a silent cache miss.
            lambda shapes: shapes[9][0],
        ),
    ),
)


@supported_compute_capability([100, 103])
def _cutlass_nvfp4_svdquant_requirement(*args, **kwargs):
    return True


def _check_mm_nvfp4_svdquant_problem(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    d: torch.Tensor,
    l1: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    backend: Literal["cutlass"] = "cutlass",
    enable_pdl: Optional[bool] = None,
):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2-D packed-e2m1 (uint8) tensors")
    if a.dtype != torch.uint8 or b.dtype != torch.uint8:
        raise ValueError("a and b must be uint8 (two e2m1 values per byte)")
    m, k_packed = a.shape
    n = b.shape[0]
    if b.shape[1] != k_packed:
        raise ValueError(
            f"a and b inner dimensions mismatch: a {tuple(a.shape)} vs b {tuple(b.shape)}"
        )
    k = k_packed * 2
    if n % 32 != 0 or k % 32 != 0:
        raise ValueError(f"n and k must be divisible by 32, got n={n}, k={k}")
    if d.ndim != 2 or d.shape[0] != m:
        raise ValueError(
            f"d must have shape [m, r] (rank-r LoRA-down output), got {tuple(d.shape)}"
        )
    rank = d.shape[1]
    if rank < SVDQUANT_LORA_RANK_GRANULARITY or rank % SVDQUANT_LORA_RANK_GRANULARITY:
        raise ValueError(
            f"the LoRA rank (d.shape[1]) must be a positive multiple of "
            f"{SVDQUANT_LORA_RANK_GRANULARITY}, got {rank}"
        )
    if l1.ndim != 2 or l1.shape[0] != n or l1.shape[1] != rank:
        raise ValueError(
            f"l1 must have shape [n, {rank}] (rank-{rank} LoRA-up weight pre-divided "
            f"by alpha, same rank as d), got {tuple(l1.shape)}"
        )
    if d.dtype != torch.bfloat16 or l1.dtype != torch.bfloat16:
        raise ValueError("d and l1 must be bf16")
    return True


@backend_requirement(
    {"cutlass": _cutlass_nvfp4_svdquant_requirement},
    common_check=_check_mm_nvfp4_svdquant_problem,
)
@flashinfer_api(trace=mm_nvfp4_svdquant_trace)
def mm_nvfp4_svdquant(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    d: torch.Tensor,
    l1: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    backend: Literal["cutlass"] = "cutlass",
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    r"""SVDQuant fused NVFP4 GEMM (SM100): ``out = alpha * (a @ bᵀ) + d @ l1ᵀ [+ bias]``.

    The block-scaled NVFP4 residual GEMM is fused with the rank-r BF16 LoRA-up correction
    ``d @ l1ᵀ``, computed by a second BF16 tcgen05 MMA into the same accumulator after the
    NVFP4 K-loop, plus an optional fused per-column bias. The LoRA rank ``r`` is inferred
    from the ``d``/``l1`` shapes and must be a positive multiple of 32 (ranks 32-128 are
    validated). ``1/alpha`` must be folded into ``l1`` by the caller
    (``l1 = svdquant_lora_b / alpha``) so the epilogue ``out = alpha * acc + bias`` yields
    the correction unscaled.

    Parameters
    ----------
    a: torch.Tensor
        Quantized activation, shape ``(m, k // 2)`` uint8 (packed e2m1), row-major. Produce it
        with :func:`nvfp4_quantize_smooth` (which folds the SVDQuant ``pre_quant_scale`` into
        the quantization).
    b: torch.Tensor
        Quantized residual weight, shape ``(n, k // 2)`` uint8 (packed e2m1), row-major
        (i.e. the GEMM computes ``a @ bᵀ``).
    a_sf: torch.Tensor
        Activation block scales, uint8 (ue4m3) in the 128x4 swizzled layout,
        ``numel >= ceil(m / 128) * 128 * ceil(k / 16 / 4) * 4``.
    b_sf: torch.Tensor
        Weight block scales, same layout as ``a_sf`` with ``n`` rows.
    alpha: torch.Tensor
        Per-tensor residual dequantization scale, float32, device scalar (``numel >= 1``).
    d: torch.Tensor
        LoRA-down output ``x_hat @ L2ᵀ``, shape ``(m, r)`` bf16, contiguous and 16-byte
        aligned (TMA). Compute it as ``x @ (pre_quant_scale[:, None] * L2ᵀ)`` in bf16.
    l1: torch.Tensor
        LoRA-up weight pre-divided by alpha, shape ``(n, r)`` bf16 (same rank as ``d``).
    bias: Optional[torch.Tensor]
        Optional per-column bias, shape ``(n,)`` bf16, fused in the epilogue.
    out: Optional[torch.Tensor]
        Output tensor, shape ``(m, n)`` bf16; allocated when ``None``.
    backend: Literal["cutlass"]
        Only the CUTLASS backend exists.
    enable_pdl: Optional[bool]
        Whether to launch with Programmatic Dependent Launch. Defaults to the device default.

    Returns
    -------
    out: torch.Tensor
        Output tensor, shape ``(m, n)`` bf16.
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(a.device)
    if out is None:
        out = torch.empty(a.shape[0], b.shape[0], dtype=torch.bfloat16, device=a.device)
    workspace_buffer = _get_cache_buf(
        "nvfp4_svdquant_gemm_workspace", DEFAULT_WORKSPACE_SIZE, a.device
    )

    tuner = AutoTuner.get()
    runners = [_nvfp4_svdquant_gemm_runner(enable_pdl)]
    inputs = [a, b, a_sf, b_sf, alpha, d, l1, bias, out, workspace_buffer]
    runner, tactic = tuner.choose_one(
        "nvfp4_svdquant_gemm",
        runners,
        _NVFP4_SVDQUANT_GEMM_TUNING_CONFIG,
        inputs,
    )
    runner(inputs=inputs, tactic=tactic)
    return out


def _check_nvfp4_quantize_smooth_problem(
    x: torch.Tensor,
    pre_quant_scale: torch.Tensor,
    global_scale: torch.Tensor,
    enable_pdl: Optional[bool] = None,
    backend: Literal["cutlass"] = "cutlass",
):
    if x.ndim != 2:
        raise ValueError(f"x must be [m, n], got {tuple(x.shape)}")
    if x.dtype != torch.bfloat16 or pre_quant_scale.dtype != torch.bfloat16:
        raise ValueError("x and pre_quant_scale must be bf16")
    if x.shape[1] % 16 != 0:
        raise ValueError(
            f"n must be divisible by 16 (NVFP4 SF vector size), got {x.shape[1]}"
        )
    if pre_quant_scale.numel() != x.shape[1]:
        raise ValueError(
            f"pre_quant_scale must have n={x.shape[1]} elements, got {pre_quant_scale.numel()}"
        )
    return True


@backend_requirement(
    {"cutlass": _cutlass_nvfp4_svdquant_requirement},
    common_check=_check_nvfp4_quantize_smooth_problem,
)
@flashinfer_api(trace=nvfp4_quantize_smooth_trace)
def nvfp4_quantize_smooth(
    x: torch.Tensor,
    pre_quant_scale: torch.Tensor,
    global_scale: torch.Tensor,
    enable_pdl: Optional[bool] = None,
    backend: Literal["cutlass"] = "cutlass",
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Fused smooth + NVFP4 quantize: ``(xq, sf) = nvfp4-quantize(x * pre_quant_scale)``.

    Applies the SVDQuant per-input-channel smoothing scale and NVFP4-quantizes in one pass
    over the input; the result is byte-identical to quantizing ``x * pre_quant_scale`` with
    the stock NVFP4 quantizer (ue4m3 block scales, 128x4 swizzled layout, SF vector size 16).

    Parameters
    ----------
    x: torch.Tensor
        Input activation, shape ``(m, n)`` bf16.
    pre_quant_scale: torch.Tensor
        Per-input-channel smoothing scale, shape ``(n,)`` bf16.
    global_scale: torch.Tensor
        Global scale, float32 device scalar: ``(448 * 6) / (x * pre_quant_scale).abs().max()``.
    enable_pdl: Optional[bool]
        Whether to launch with Programmatic Dependent Launch. Defaults to the device default.
    backend: Literal["cutlass"]
        Only the CUDA backend exists.

    Returns
    -------
    xq: torch.Tensor
        Quantized tensor, shape ``(m, n // 2)`` uint8 (packed e2m1).
    sf: torch.Tensor
        Block scales, uint8 (ue4m3), 128x4 swizzled layout, 1-D of size
        ``ceil(m / 128) * 128 * ceil(n / 16 / 4) * 4``.
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(x.device)
    m, n = x.shape
    module = get_nvfp4_svdquant_module()
    xq = torch.empty(m, n // 2, dtype=torch.uint8, device=x.device)
    sf = torch.empty(_swizzled_sf_size(m, n // 16), dtype=torch.uint8, device=x.device)
    module.nvfp4_quantize_smooth(x, pre_quant_scale, global_scale, xq, sf, enable_pdl)
    return xq, sf


@flashinfer_api(trace=svdquant_linear_trace)
def svdquant_linear(
    x: torch.Tensor,
    weight_fp4: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    pre_quant_scale: torch.Tensor,
    l2t_smoothed: torch.Tensor,
    l1_scaled: torch.Tensor,
    global_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    r"""The full SVDQuant linear operator: ``y = x_hat @ (R + L1 @ L2)ᵀ [+ bias]`` where
    ``x_hat = x * pre_quant_scale`` and ``R`` is the NVFP4-quantized residual weight.

    Runs the three-step chain this library's kernels are designed for:

    1. ``xq, x_sf = nvfp4_quantize_smooth(x, pre_quant_scale, global_scale)``
    2. ``down = x @ l2t_smoothed``  (plain BF16 GEMM; ``l2t_smoothed = pre_quant_scale[:, None] * L2ᵀ``)
    3. ``mm_nvfp4_svdquant(xq, weight_fp4, x_sf, weight_sf, alpha, down, l1_scaled, bias)``

    The invariant per-layer transforms must be prepared offline by the caller:
    ``l2t_smoothed = (pre_quant_scale[:, None] * svdquant_lora_a.T).to(bf16)`` with shape
    ``(k, r)`` and ``l1_scaled = (svdquant_lora_b / alpha).to(bf16)`` with shape ``(n, r)``,
    where the LoRA rank ``r`` is a positive multiple of 32.

    Parameters
    ----------
    x: torch.Tensor
        Input activation, shape ``(m, k)`` bf16.
    weight_fp4: torch.Tensor
        NVFP4 residual weight, shape ``(n, k // 2)`` uint8 (packed e2m1).
    weight_sf: torch.Tensor
        Weight block scales, uint8 (ue4m3), 128x4 swizzled layout.
    alpha: torch.Tensor
        Per-tensor residual dequantization scale, float32 device scalar.
    pre_quant_scale: torch.Tensor
        Per-input-channel smoothing scale, shape ``(k,)`` bf16.
    l2t_smoothed: torch.Tensor
        ``pre_quant_scale[:, None] * L2ᵀ``, shape ``(k, r)`` bf16.
    l1_scaled: torch.Tensor
        ``L1 / alpha``, shape ``(n, r)`` bf16.
    global_scale: torch.Tensor
        Activation global scale, float32 device scalar.
    bias: Optional[torch.Tensor]
        Optional per-column bias, shape ``(n,)`` bf16.
    enable_pdl: Optional[bool]
        Whether to launch with Programmatic Dependent Launch. Defaults to the device default.

    Returns
    -------
    out: torch.Tensor
        Output tensor, shape ``(m, n)`` bf16.
    """
    xq, x_sf = nvfp4_quantize_smooth(
        x, pre_quant_scale, global_scale, enable_pdl=enable_pdl
    )
    down = torch.mm(x, l2t_smoothed)
    return mm_nvfp4_svdquant(
        xq,
        weight_fp4,
        x_sf,
        weight_sf,
        alpha,
        down,
        l1_scaled,
        bias=bias,
        enable_pdl=enable_pdl,
    )
