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

NVFP4 4over6 activation quantization for SM12x with BF16 input.

``nvfp4_4over6_quantize_sm120`` is a drop-in for
``fp4_quantize(x, global_scale, nvfp4_4over6=NVFP44Over6Config(err_mode="MSE", ...))``
on SM120/SM121: same packed values and the same swizzled scale buffer, bit for bit.
``nvfp4_4over6_dynamic_quantize_sm120`` adds the per-tensor amax reduction in the same launch.
"""

import functools
from typing import Dict, Optional, Tuple

import torch

from ..api_logging import flashinfer_api
from ..jit.nvfp4_4over6_sm120 import gen_nvfp4_4over6_quantize_sm120_module
from ..utils import (
    device_support_pdl,
    get_compute_capability,
    register_custom_op,
    register_fake_op,
    round_up,
)
from .nvfp4_quantization_utils import NVFP44Over6Config, NVFP44Over6ErrMode

#: The recipe these kernels implement: MSE candidate selection with exact fp32 error.
#: ``e4m3_max`` (256 or 448) only sets the global scale, so both are supported.
DEFAULT_NVFP4_4OVER6_SM120 = NVFP44Over6Config(e4m3_max=256, err_mode="MSE")

_scratch: Dict[torch.device, torch.Tensor] = {}
_lora_partials: Dict[torch.device, torch.Tensor] = {}


def _check_recipe(config: NVFP44Over6Config) -> None:
    if not isinstance(config, NVFP44Over6Config):
        raise TypeError(
            f"nvfp4_4over6 must be an NVFP44Over6Config, got {type(config)}"
        )
    if config.err_mode != NVFP44Over6ErrMode.MSE or config.err_use_fast_math:
        raise ValueError(
            "nvfp4_4over6_*_sm120 implements err_mode='MSE' with err_use_fast_math=False; "
            f"got {config!r}. Use fp4_quantize for other recipes."
        )


def _check_input(input: torch.Tensor) -> None:
    if not input.is_cuda:
        raise ValueError("input must be a CUDA tensor")
    if input.dtype != torch.bfloat16:
        raise ValueError(f"input must be bfloat16, got {input.dtype}")
    if input.shape[-1] % 16 != 0:
        raise ValueError(
            f"the last dimension must be a multiple of 16, got {input.shape[-1]}"
        )
    major, _ = get_compute_capability(input.device)
    if major != 12:
        raise RuntimeError(
            f"nvfp4_4over6_*_sm120 requires an SM12x GPU, got compute capability {major}.x"
        )


def _sf_numel(m: int, k: int) -> int:
    return round_up(m, 128) * round_up(k // 16, 4)


def _device_scratch(device: torch.device) -> torch.Tensor:
    scratch = _scratch.get(device)
    if scratch is None:
        # One float per cooperative block; every launch rewrites the slots it reads, so the buffer
        # needs no reset and is safe to reuse across CUDA-graph replays.
        scratch = torch.zeros(8192, dtype=torch.float32, device=device)
        _scratch[device] = scratch
    return scratch


def _default_rcp_quant_range(config: NVFP44Over6Config) -> float:
    q = torch.tensor(6.0 * config.e4m3_max, dtype=torch.float32)
    return float(torch.tensor(1.0, dtype=torch.float32) / q)


@functools.cache
def get_nvfp4_4over6_quantize_sm120_module():
    module = gen_nvfp4_4over6_quantize_sm120_module().build_and_load()

    @register_custom_op("flashinfer::nvfp4_4over6_quantize_sm120", mutates_args=())
    def quantize(
        input: torch.Tensor, global_scale: torch.Tensor, pdl: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m, k = input.shape
        out_fp4 = torch.empty((m, k // 2), dtype=torch.uint8, device=input.device)
        out_sf = torch.empty((_sf_numel(m, k),), dtype=torch.uint8, device=input.device)
        module.nvfp4_4over6_quantize_sm120(
            input, global_scale, out_fp4, out_sf, 0, 0, False, pdl, -1
        )
        return out_fp4, out_sf

    @register_fake_op("flashinfer::nvfp4_4over6_quantize_sm120")
    def _fake_quantize(
        input: torch.Tensor, global_scale: torch.Tensor, pdl: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m, k = input.shape
        return (
            input.new_empty((m, k // 2), dtype=torch.uint8),
            input.new_empty((_sf_numel(m, k),), dtype=torch.uint8),
        )

    @register_custom_op(
        "flashinfer::nvfp4_4over6_dynamic_quantize_sm120",
        mutates_args=("global_scale_out", "alpha_out"),
    )
    def dynamic_quantize(
        input: torch.Tensor,
        global_scale_out: torch.Tensor,
        alpha_out: torch.Tensor,
        weight_global_scale: torch.Tensor,
        rcp_quant_range: float,
        pdl: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m, k = input.shape
        out_fp4 = torch.empty((m, k // 2), dtype=torch.uint8, device=input.device)
        out_sf = torch.empty((_sf_numel(m, k),), dtype=torch.uint8, device=input.device)
        scratch = _device_scratch(input.device)
        module.nvfp4_4over6_dynamic_quantize_sm120(
            input,
            global_scale_out,
            alpha_out,
            weight_global_scale,
            rcp_quant_range,
            scratch,
            out_fp4,
            out_sf,
            0,
            0,
            0,
            False,
            pdl,
            -1,
        )
        return out_fp4, out_sf

    @register_fake_op("flashinfer::nvfp4_4over6_dynamic_quantize_sm120")
    def _fake_dynamic_quantize(
        input: torch.Tensor,
        global_scale_out: torch.Tensor,
        alpha_out: torch.Tensor,
        weight_global_scale: torch.Tensor,
        rcp_quant_range: float,
        pdl: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m, k = input.shape
        return (
            input.new_empty((m, k // 2), dtype=torch.uint8),
            input.new_empty((_sf_numel(m, k),), dtype=torch.uint8),
        )

    @register_custom_op(
        "flashinfer::nvfp4_4over6_dynamic_quantize_lora_down_sm120",
        mutates_args=("global_scale_out", "alpha_out"),
    )
    def dynamic_quantize_lora_down(
        input: torch.Tensor,
        lora_down: torch.Tensor,
        global_scale_out: torch.Tensor,
        alpha_out: torch.Tensor,
        weight_global_scale: torch.Tensor,
        rcp_quant_range: float,
        pdl: int,
        path: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m, k = input.shape
        rank = lora_down.shape[0]
        out_fp4 = torch.empty((m, k // 2), dtype=torch.uint8, device=input.device)
        out_sf = torch.empty((_sf_numel(m, k),), dtype=torch.uint8, device=input.device)
        down = torch.empty((m, rank), dtype=torch.bfloat16, device=input.device)
        scratch = _device_scratch(input.device)
        partials = _lora_partials.get(input.device)
        if partials is None:
            # rank-64 partial sums of the cooperative path: 64 slices x 64 rows x rank 64 (K <= 16384, M <= 64)
            partials = torch.empty(
                64 * 64 * 64, dtype=torch.float32, device=input.device
            )
            _lora_partials[input.device] = partials
        taken = module.nvfp4_4over6_dynamic_quantize_lora_down_sm120(
            input,
            global_scale_out,
            alpha_out,
            weight_global_scale,
            rcp_quant_range,
            scratch,
            lora_down,
            partials,
            out_fp4,
            out_sf,
            down,
            path,
            pdl,
        )
        if taken == 0:  # quantized only: LoRA-down after it, reading the alpha it wrote
            if module.nvfp4_lora_down_supported_sm120(m, k):
                module.nvfp4_lora_down_scaled_sm120(
                    input, lora_down, alpha_out, down, pdl != 0
                )
            else:
                torch.mm(input, lora_down.t(), out=down)
                module.nvfp4_div_by_alpha_sm120(down, alpha_out, pdl != 0)
        return out_fp4, out_sf, down

    @register_fake_op("flashinfer::nvfp4_4over6_dynamic_quantize_lora_down_sm120")
    def _fake_dynamic_quantize_lora_down(
        input: torch.Tensor,
        lora_down: torch.Tensor,
        global_scale_out: torch.Tensor,
        alpha_out: torch.Tensor,
        weight_global_scale: torch.Tensor,
        rcp_quant_range: float,
        pdl: int,
        path: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m, k = input.shape
        return (
            input.new_empty((m, k // 2), dtype=torch.uint8),
            input.new_empty((_sf_numel(m, k),), dtype=torch.uint8),
            input.new_empty((m, lora_down.shape[0]), dtype=torch.bfloat16),
        )

    return module, quantize, dynamic_quantize, dynamic_quantize_lora_down


@flashinfer_api
def nvfp4_4over6_quantize_sm120(
    input: torch.Tensor,
    global_scale: torch.Tensor,
    nvfp4_4over6: NVFP44Over6Config = DEFAULT_NVFP4_4OVER6_SM120,
    enable_pdl: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""NVFP4 4over6 quantization of BF16 activations on SM12x with a given global scale.

    Bit-identical to ``fp4_quantize(input, global_scale, nvfp4_4over6=nvfp4_4over6)`` (NVFP4, 128x4
    swizzled scales) and faster on SM120: one thread per 16-element block for medium and large inputs,
    the reference's two-lane layout for small ones, and a programmatic dependent launch.

    Parameters
    ----------
    input : torch.Tensor
        ``[..., K]`` bfloat16, ``K`` a multiple of 16, 32-byte aligned.
    global_scale : torch.Tensor
        ``[1]`` float32 encode scale, e.g. ``make_nvfp4_global_scale(input, False, nvfp4_4over6_config=...)``
        (``6 * e4m3_max / amax``).
    nvfp4_4over6 : NVFP44Over6Config
        The 4over6 recipe; ``err_mode`` must be ``"MSE"`` with ``err_use_fast_math=False``. Defaults to
        ``NVFP44Over6Config(e4m3_max=256, err_mode="MSE")``.
    enable_pdl : bool, optional
        Launch with programmatic dependent launch and let the next kernel launch once the blocks are
        quantized. Defaults to ``device_support_pdl(input.device)``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(x_q, sf)``: ``x_q`` is ``[..., K/2]`` uint8 (two e2m1 values per byte) and ``sf`` the uint8 UE4M3
        scale buffer of ``roundup(M, 128) * roundup(K/16, 4)`` bytes in the 128x4 swizzled layout, padding
        zeroed, as ``fp4_quantize`` returns them.
    """
    _check_recipe(nvfp4_4over6)
    _check_input(input)
    lead, k = input.shape[:-1], input.shape[-1]
    matrix = input.reshape(-1, k).contiguous()
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    global_scale = global_scale.to(device=input.device, dtype=torch.float32)
    _, quantize, _, _ = get_nvfp4_4over6_quantize_sm120_module()
    x_q, sf = quantize(matrix, global_scale, 1 if enable_pdl else 0)
    return x_q.reshape(*lead, k // 2), sf


@flashinfer_api
def nvfp4_4over6_dynamic_quantize_sm120(
    input: torch.Tensor,
    weight_global_scale: Optional[torch.Tensor] = None,
    nvfp4_4over6: NVFP44Over6Config = DEFAULT_NVFP4_4OVER6_SM120,
    rcp_quant_range: Optional[float] = None,
    enable_pdl: Optional[bool] = None,
    trigger_dependents: bool = False,
    global_scale_out: Optional[torch.Tensor] = None,
    alpha_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""NVFP4 4over6 quantization of BF16 activations on SM12x with a dynamic per-tensor global scale.

    One launch computes ``deq = max(amax(input), 1e-12) * rcp_quant_range``, ``global_scale = 1 / deq``
    (round to nearest) and ``alpha = deq * weight_global_scale`` (the FP4 GEMM's alpha), and quantizes
    ``input`` exactly as ``nvfp4_4over6_quantize_sm120(input, global_scale, nvfp4_4over6)`` /
    ``fp4_quantize`` would. Small inputs reduce the amax redundantly in every block; larger ones use one
    cooperative grid barrier and keep their blocks in registers across it.

    Parameters
    ----------
    input : torch.Tensor
        ``[..., K]`` bfloat16, ``K`` a multiple of 16, 32-byte aligned, at least one row.
    weight_global_scale : torch.Tensor, optional
        ``[1]`` float32 weight decode scale folded into ``alpha``; ``1`` when omitted.
    nvfp4_4over6 : NVFP44Over6Config
        The 4over6 recipe (MSE, exact error); ``6 * e4m3_max`` is the quantization range.
    rcp_quant_range : float, optional
        FP32 ``1 / (6 * e4m3_max)``. Pass the value a reference implementation multiplies by when its
        rounding of the reciprocal differs (the result is bit-identical only for the same factor).
    enable_pdl : bool, optional
        Programmatic dependent launch. Defaults to ``device_support_pdl(input.device)``.
    trigger_dependents : bool
        Let the next kernel launch before this one completes. Off by default: a dependent that reads
        ``alpha`` before its own ``griddepcontrol.wait`` (CUTLASS epilogues read the alpha pointer at
        kernel entry) could see a stale value.
    global_scale_out, alpha_out : torch.Tensor, optional
        ``[1]`` float32 outputs to write (e.g. buffers captured in a CUDA graph); allocated when omitted.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(x_q, sf, global_scale, alpha)``; ``x_q`` and ``sf`` as in :func:`nvfp4_4over6_quantize_sm120`.
    """
    _check_recipe(nvfp4_4over6)
    _check_input(input)
    lead, k = input.shape[:-1], input.shape[-1]
    matrix = input.reshape(-1, k).contiguous()
    if matrix.shape[0] == 0:
        raise ValueError("dynamic quantization needs at least one row")
    device = input.device
    if weight_global_scale is None:
        weight_global_scale = torch.ones(1, dtype=torch.float32, device=device)
    else:
        weight_global_scale = weight_global_scale.to(device=device, dtype=torch.float32)
    if rcp_quant_range is None:
        rcp_quant_range = _default_rcp_quant_range(nvfp4_4over6)
    if global_scale_out is None:
        global_scale_out = torch.empty(1, dtype=torch.float32, device=device)
    if alpha_out is None:
        alpha_out = torch.empty(1, dtype=torch.float32, device=device)
    if enable_pdl is None:
        enable_pdl = device_support_pdl(device)
    pdl = (1 if trigger_dependents else 2) if enable_pdl else 0
    _, _, dynamic_quantize, _ = get_nvfp4_4over6_quantize_sm120_module()
    x_q, sf = dynamic_quantize(
        matrix, global_scale_out, alpha_out, weight_global_scale, rcp_quant_range, pdl
    )
    return x_q.reshape(*lead, k // 2), sf, global_scale_out, alpha_out


@flashinfer_api
def nvfp4_4over6_dynamic_quantize_lora_down_sm120(
    input: torch.Tensor,
    lora_down: torch.Tensor,
    weight_global_scale: Optional[torch.Tensor] = None,
    nvfp4_4over6: NVFP44Over6Config = DEFAULT_NVFP4_4OVER6_SM120,
    rcp_quant_range: Optional[float] = None,
    enable_pdl: Optional[bool] = None,
    global_scale_out: Optional[torch.Tensor] = None,
    alpha_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Dynamic NVFP4 4over6 quantization plus the SVDQuant LoRA-down, for SM12x and BF16 input.

    Quantizes ``input`` exactly as :func:`nvfp4_4over6_dynamic_quantize_sm120` (same packed values, scale
    buffer, ``global_scale`` and ``alpha``) and also returns
    ``down = bf16((input @ lora_downᵀ) / alpha)``, the LoRA-down operand that
    :func:`flashinfer.gemm.mm_nvfp4_svdquant` turns into ``alpha * (x_q @ wᵀ + down @ lora_upᵀ)``. Dividing
    by the dynamic ``alpha`` here, instead of pre-dividing ``lora_up`` offline, keeps ``lora_up`` unchanged
    when the activation scale changes every call.

    ``down`` is computed in the quantization launch where that fits: up to 8 rows (at most 2048 scale
    blocks, K <= 16384) every block reduces the amax and the extra blocks compute one rank row each (FP32
    sums, one rounding); with a rank-64 ``lora_down``, up to 64 rows and K a multiple of 256 (at most
    16384), one cooperative launch computes it on tensor cores (FP32 sums in a fixed order, one rounding).
    Otherwise the quantization runs first and ``down`` follows it, from a one-launch kernel for
    decode-sized inputs or a BF16 GEMM and an in-place division (two BF16 roundings).

    Parameters
    ----------
    input : torch.Tensor
        ``[..., K]`` bfloat16, ``K`` a multiple of 16, 32-byte aligned, at least one row.
    lora_down : torch.Tensor
        ``[rank, K]`` bfloat16, 16-byte aligned: the SVDQuant down projection ``L2`` (``x @ L2ᵀ``).
    weight_global_scale, nvfp4_4over6, rcp_quant_range, enable_pdl, global_scale_out, alpha_out
        As in :func:`nvfp4_4over6_dynamic_quantize_sm120`.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(x_q, sf, down, global_scale, alpha)``; ``down`` is ``[..., rank]`` bfloat16.
    """
    _check_recipe(nvfp4_4over6)
    _check_input(input)
    lead, k = input.shape[:-1], input.shape[-1]
    matrix = input.reshape(-1, k).contiguous()
    if matrix.shape[0] == 0:
        raise ValueError("dynamic quantization needs at least one row")
    if (
        lora_down.dim() != 2
        or lora_down.shape[1] != k
        or lora_down.dtype != torch.bfloat16
    ):
        raise ValueError(
            f"lora_down must be a [rank, {k}] bfloat16 tensor, got {tuple(lora_down.shape)}"
        )
    lora_down = lora_down.contiguous()
    device = input.device
    if weight_global_scale is None:
        weight_global_scale = torch.ones(1, dtype=torch.float32, device=device)
    else:
        weight_global_scale = weight_global_scale.to(device=device, dtype=torch.float32)
    if rcp_quant_range is None:
        rcp_quant_range = _default_rcp_quant_range(nvfp4_4over6)
    if global_scale_out is None:
        global_scale_out = torch.empty(1, dtype=torch.float32, device=device)
    if alpha_out is None:
        alpha_out = torch.empty(1, dtype=torch.float32, device=device)
    if enable_pdl is None:
        enable_pdl = device_support_pdl(device)
    _, _, _, dynamic_quantize_lora_down = get_nvfp4_4over6_quantize_sm120_module()
    x_q, sf, down = dynamic_quantize_lora_down(
        matrix,
        lora_down,
        global_scale_out,
        alpha_out,
        weight_global_scale,
        rcp_quant_range,
        2 if enable_pdl else 0,
        -1,
    )
    rank = lora_down.shape[0]
    return (
        x_q.reshape(*lead, k // 2),
        sf,
        down.reshape(*lead, rank),
        global_scale_out,
        alpha_out,
    )
