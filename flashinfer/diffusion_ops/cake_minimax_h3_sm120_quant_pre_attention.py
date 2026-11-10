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
from typing import Any, NamedTuple, Optional, Tuple, Union

import torch
import tvm_ffi

from ..api_logging import flashinfer_api
from ..jit import cake_program
from ..jit.cake_minimax_h3_sm120_quant_pre_attention import (
    FP8_BF16,
    FP8_E4M3,
    FP8_MX_BF16,
    FP8_MX_E4M3,
    FP8_MX_NVFP4,
    FP8_NVFP4,
    NORM_FP8,
    NORM_NVFP4,
    NVFP4_BF16,
    NVFP4_E4M3,
    NVFP4_E4M3_G1,
    NVFP4_NVFP4,
)
from ..utils import (
    register_custom_op,
    register_fake_op,
    supported_compute_capability,
)
from .cake_minimax_h3_sm120_quant_gate import require_sm120

MINIMAX_H3_HIDDEN = 5376
MINIMAX_H3_NUM_HEADS = 56
MINIMAX_H3_HEAD_DIM = 128
MINIMAX_H3_QKV_WIDTH = MINIMAX_H3_NUM_HEADS * 3 * MINIMAX_H3_HEAD_DIM  # 21504
MINIMAX_H3_ROPE_DIM = 96
MINIMAX_H3_ADALN_ROWS = 9  # the fused norm/AdaLN kernels index a fixed 9-row table
MINIMAX_H3_SF_BLOCK = 16
MINIMAX_H3_DEFAULT_EPS = 1.0e-5
MINIMAX_H3_MAX_ROWS = 1 << 24
E4M3_MAX = 448.0
E2M1_MAX = 6.0

_OUT_MODES = {"bf16": 0, "e4m3": 1, "nvfp4": 2}
_HIDDEN_SF = MINIMAX_H3_HIDDEN // MINIMAX_H3_SF_BLOCK  # 336
_HEAD_SF = MINIMAX_H3_HEAD_DIM // MINIMAX_H3_SF_BLOCK  # 8
# The fused GEMM: persistent 256-row x 128-column tiles, one CTA per SM; each 128-column
# tile is one (head, kind) of the checkpoint row order head / kind / dim.
_GEMM_BLOCK_M = 256
_GEMM_N_TILES = MINIMAX_H3_QKV_WIDTH // MINIMAX_H3_HEAD_DIM  # 168
_NORM_CTAS_PER_SM = 8
# The weight scales as the NVFP4 GEMM's scale descriptor reads them: rows of 256 bytes, 168
# per 128-column tile (the 128x4 swizzled layout of fp4_quantize).
_SFB_ROW_BYTES = 256
_SFB_ROWS = MINIMAX_H3_QKV_WIDTH * _HIDDEN_SF // _SFB_ROW_BYTES  # 28224

# The fused GEMM program per (FP8 MMA form, out mode) and per out mode for NVFP4.  The
# unit-scaled block-scale FP8 MMA form gives bitwise the dense form's output; GeForce
# boards issue the dense FP8 MMA with FP32 accumulation at half the rate of that form.
_FP8_GEMM = {
    (False, 0): FP8_BF16,
    (False, 1): FP8_E4M3,
    (False, 2): FP8_NVFP4,
    (True, 0): FP8_MX_BF16,
    (True, 1): FP8_MX_E4M3,
    (True, 2): FP8_MX_NVFP4,
}
# The NVFP4 GEMM program per (GeForce board, out mode): on the GeForce board the
# E4M3-output program walks its tiles one M tile at a time (all 168 column tiles of
# an M tile in turn).
_NVFP4_GEMM = {
    (False, 0): NVFP4_BF16,
    (False, 1): NVFP4_E4M3,
    (False, 2): NVFP4_NVFP4,
    (True, 0): NVFP4_BF16,
    (True, 1): NVFP4_E4M3_G1,
    (True, 2): NVFP4_NVFP4,
}

Scalar = Union[float, torch.Tensor]


class MiniMaxH3PreAttentionOutput(NamedTuple):
    """Q, K, V ``[M, 56, 128]`` (BF16 / E4M3 / packed NVFP4 ``[M, 56, 64]``) and, for the NVFP4
    output format, their block-16 UE4M3 scales ``[M, 56, 8]``."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_sf: Optional[torch.Tensor] = None
    k_sf: Optional[torch.Tensor] = None
    v_sf: Optional[torch.Tensor] = None


@functools.cache
def _device_facts(index: int) -> Tuple[str, int, str]:
    """``(target, SM count, device name)`` of CUDA device ``index``."""
    properties = torch.cuda.get_device_properties(index)
    target = cake_program.CAPABILITY_TARGET.get(
        (int(properties.major), int(properties.minor))
    )
    if target is None or target not in NORM_FP8.targets:
        raise RuntimeError(
            "MiniMax-H3 SM120 quantized pre-attention requires compute capability 12.0, "
            f"got {properties.major}.{properties.minor}"
        )
    return target, int(properties.multi_processor_count), str(properties.name)


def _fp8_unit_scale_mma(index: int) -> bool:
    """Whether the FP8 GEMM on device ``index`` runs the unit-scaled block-scale MMA form."""
    return "GeForce" in _device_facts(index)[2]


def _nvfp4_one_m_tile_order(index: int) -> bool:
    """Whether the NVFP4 GEMM's E4M3-output program on device ``index`` walks its tiles
    one M tile at a time."""
    return "GeForce" in _device_facts(index)[2]


@functools.cache
def _entry(program: cake_program.CakeProgram, target: str) -> Any:
    return program.entry(target)


@functools.cache
def _placeholders(index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Operands a launch binds and never reads: the FP8 GEMM's two scale descriptors'
    sources and the scale outputs of the BF16 / E4M3 out modes."""
    device = torch.device("cuda", index)
    return (
        torch.zeros((_GEMM_BLOCK_M, 16), dtype=torch.uint8, device=device),
        torch.zeros((8, _SFB_ROW_BYTES), dtype=torch.uint8, device=device),
        torch.zeros((64,), dtype=torch.uint8, device=device),
    )


def _launch_plan(rows: int, num_sms: int) -> Tuple[int, int, int, int]:
    """``(stage grid, GEMM grid, M tiles, GEMM tiles)``: eight stage CTAs per SM over
    the rows, one persistent GEMM CTA per SM over the 256 x 128 tiles."""
    num_m_tiles = (rows + _GEMM_BLOCK_M - 1) // _GEMM_BLOCK_M
    total_tiles = num_m_tiles * _GEMM_N_TILES
    stage_grid = max(1, min(rows, _NORM_CTAS_PER_SM * num_sms))
    gemm_grid = max(1, min(total_tiles, num_sms))
    return stage_grid, gemm_grid, num_m_tiles, total_tiles


def _words(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.uint8).view(torch.uint32)


def _run_gemm(
    gemm,
    target: str,
    index: int,
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    act_scale: torch.Tensor,
    w_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_sf: Optional[torch.Tensor],
    k_sf: Optional[torch.Tensor],
    v_sf: Optional[torch.Tensor],
    rows: int,
    eps: float,
    alpha: float,
    out_scales: Tuple[float, float, float],
    sf_muls: Tuple[float, float, float],
    gemm_grid: int,
    num_m_tiles: int,
    total_tiles: int,
) -> None:
    unused_sf = _placeholders(index)[2]
    _entry(gemm, target)(
        a,
        b,
        sfa,
        sfb,
        act_scale,
        w_scale,
        q_norm_weight,
        k_norm_weight,
        rope_cos_sin,
        _words(q),
        _words(k),
        _words(v),
        unused_sf if q_sf is None else q_sf,
        unused_sf if k_sf is None else k_sf,
        unused_sf if v_sf is None else v_sf,
        rows,
        num_m_tiles,
        total_tiles,
        eps,
        alpha,
        *out_scales,
        *sf_muls,
        gemm_grid,
        1,
        1,
    )


def _require(
    name: str,
    tensor: torch.Tensor,
    dtype: torch.dtype,
    shape: Tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must be a CUDA tensor on {device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor


def _scalar(value: Optional[Scalar], name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required for this out_mode")
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a single value")
        return float(value.item())
    return float(value)


def _check_rows(x: torch.Tensor) -> int:
    if (
        not isinstance(x, torch.Tensor)
        or x.ndim != 2
        or x.shape[1] != MINIMAX_H3_HIDDEN
    ):
        raise ValueError(f"x must be [M, {MINIMAX_H3_HIDDEN}]")
    rows = int(x.shape[0])
    if not 1 <= rows <= MINIMAX_H3_MAX_ROWS:
        raise ValueError(f"M must lie in [1, {MINIMAX_H3_MAX_ROWS}], got {rows}")
    return rows


def _check_inputs(
    x,
    x_norm_weight,
    adaln_scale,
    adaln_shift,
    adaln_index,
    q_norm_weight,
    k_norm_weight,
    rope_cos_sin,
) -> Tuple[int, torch.device]:
    rows = _check_rows(x)
    device = x.device
    require_sm120(device, "pre-attention")
    _require("x", x, torch.bfloat16, (rows, MINIMAX_H3_HIDDEN), device)
    _require(
        "x_norm_weight", x_norm_weight, torch.bfloat16, (MINIMAX_H3_HIDDEN,), device
    )
    for name, table in (("adaln_scale", adaln_scale), ("adaln_shift", adaln_shift)):
        if (
            isinstance(table, torch.Tensor)
            and table.ndim == 2
            and table.shape[0] != MINIMAX_H3_ADALN_ROWS
        ):
            raise ValueError(
                f"{name} must have exactly {MINIMAX_H3_ADALN_ROWS} rows (the norm/AdaLN "
                f"kernels index a {MINIMAX_H3_ADALN_ROWS}-row table), got {table.shape[0]}"
            )
        _require(
            name,
            table,
            torch.bfloat16,
            (MINIMAX_H3_ADALN_ROWS, MINIMAX_H3_HIDDEN),
            device,
        )
    _require("adaln_index", adaln_index, torch.int32, (rows,), device)
    _require(
        "q_norm_weight", q_norm_weight, torch.bfloat16, (MINIMAX_H3_HEAD_DIM,), device
    )
    _require(
        "k_norm_weight", k_norm_weight, torch.bfloat16, (MINIMAX_H3_HEAD_DIM,), device
    )
    _require(
        "rope_cos_sin",
        rope_cos_sin,
        torch.bfloat16,
        (rows, MINIMAX_H3_ROPE_DIM),
        device,
    )
    return rows, device


def _check_common(
    x,
    x_norm_weight,
    adaln_scale,
    adaln_shift,
    adaln_index,
    q_norm_weight,
    k_norm_weight,
    rope_cos_sin,
) -> Tuple[int, torch.device]:
    """The public wrappers' check of the inputs both routes share."""
    return _check_inputs(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        q_norm_weight,
        k_norm_weight,
        rope_cos_sin,
    )


def _output_layout(rows: int, out_mode: int) -> Tuple[torch.dtype, Tuple[int, ...]]:
    """The dtype and shape of each of Q, K, V for ``out_mode`` (0, 1, 2)."""
    heads = (rows, MINIMAX_H3_NUM_HEADS, MINIMAX_H3_HEAD_DIM)
    if out_mode == _OUT_MODES["bf16"]:
        return torch.bfloat16, heads
    if out_mode == _OUT_MODES["e4m3"]:
        return torch.float8_e4m3fn, heads
    return torch.uint8, (rows, MINIMAX_H3_NUM_HEADS, MINIMAX_H3_HEAD_DIM // 2)


def _check_weight_sf(qkv_weight_sf, device: torch.device) -> None:
    if (
        not isinstance(qkv_weight_sf, torch.Tensor)
        or qkv_weight_sf.dtype != torch.uint8
        or qkv_weight_sf.numel() != MINIMAX_H3_QKV_WIDTH * _HIDDEN_SF
        or not qkv_weight_sf.is_cuda
        or qkv_weight_sf.device != device
        or not qkv_weight_sf.is_contiguous()
    ):
        raise ValueError(
            f"qkv_weight_sf must be a contiguous uint8 CUDA tensor with {MINIMAX_H3_QKV_WIDTH * _HIDDEN_SF} "
            "entries (FlashInfer 128x4 swizzled layout)"
        )


def _check_launch(
    x,
    x_norm_weight,
    adaln_scale,
    adaln_shift,
    adaln_index,
    q_norm_weight,
    k_norm_weight,
    rope_cos_sin,
    q,
    k,
    v,
    q_sf,
    k_sf,
    v_sf,
    out_mode: int,
) -> Tuple[int, torch.device]:
    """The custom ops' own check of the shared tensors their two launches index: the
    generated bindings check dtypes, devices and contiguity but not extents, and the ops
    are reachable without the public wrappers."""
    rows, device = _check_inputs(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        q_norm_weight,
        k_norm_weight,
        rope_cos_sin,
    )
    if out_mode not in _OUT_MODES.values():
        raise ValueError(
            f"out_mode must be 0 (bf16), 1 (e4m3) or 2 (nvfp4), got {out_mode}"
        )
    dtype, shape = _output_layout(rows, out_mode)
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        _require(name, tensor, dtype, shape, device)
    if out_mode == _OUT_MODES["nvfp4"]:
        sf_shape = (rows, MINIMAX_H3_NUM_HEADS, _HEAD_SF)
        for name, tensor in (("q_sf", q_sf), ("k_sf", k_sf), ("v_sf", v_sf)):
            if tensor is None:
                raise ValueError(
                    "q_sf, k_sf and v_sf are required for out_mode=2 (nvfp4)"
                )
            _require(name, tensor, torch.uint8, sf_shape, device)
    return rows, device


def _prepare_outputs(
    *,
    rows: int,
    device: torch.device,
    out_mode: str,
    q,
    k,
    v,
    q_sf,
    k_sf,
    v_sf,
    q_descale,
    k_descale,
    v_descale,
    q_global_scale,
    k_global_scale,
    v_global_scale,
):
    """Allocate or validate the Q/K/V outputs for ``out_mode`` and derive the epilogue scales."""

    if out_mode not in _OUT_MODES:
        raise ValueError(
            f"out_mode must be one of {sorted(_OUT_MODES)}, got {out_mode!r}"
        )
    dtype, shape = _output_layout(rows, _OUT_MODES[out_mode])
    outs = []
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor is None:
            tensor = torch.empty(shape, dtype=dtype, device=device)
        outs.append(_require(name, tensor, dtype, shape, device))
    sfs = [None, None, None]
    if out_mode == "nvfp4":
        sf_shape = (rows, MINIMAX_H3_NUM_HEADS, _HEAD_SF)
        for index, (name, tensor) in enumerate(
            (("q_sf", q_sf), ("k_sf", k_sf), ("v_sf", v_sf))
        ):
            if tensor is None:
                tensor = torch.empty(sf_shape, dtype=torch.uint8, device=device)
            sfs[index] = _require(name, tensor, torch.uint8, sf_shape, device)
        global_scales = (
            _scalar(q_global_scale, "q_global_scale"),
            _scalar(k_global_scale, "k_global_scale"),
            _scalar(v_global_scale, "v_global_scale"),
        )
        out_scales = global_scales
        sf_muls = tuple(g / E2M1_MAX for g in global_scales)
    elif out_mode == "e4m3":
        out_scales = (
            1.0 / _scalar(q_descale, "q_descale"),
            1.0 / _scalar(k_descale, "k_descale"),
            1.0 / _scalar(v_descale, "v_descale"),
        )
        sf_muls = (0.0, 0.0, 0.0)
    else:
        out_scales = (1.0, 1.0, 1.0)
        sf_muls = (0.0, 0.0, 0.0)
    return outs, sfs, out_scales, sf_muls


@register_custom_op(
    "flashinfer::minimax_h3_sm120_fp8_pre_attention",
    mutates_args=("act_q", "act_scale", "q", "k", "v", "q_sf", "k_sf", "v_sf"),
)
def _fp8_impl(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    qkv_weight_q: torch.Tensor,
    qkv_weight_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    act_q: torch.Tensor,
    act_scale: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_sf: Optional[torch.Tensor],
    k_sf: Optional[torch.Tensor],
    v_sf: Optional[torch.Tensor],
    out_mode: int,
    eps: float,
    out_scale_q: float,
    out_scale_k: float,
    out_scale_v: float,
    sf_mul_q: float,
    sf_mul_k: float,
    sf_mul_v: float,
) -> None:
    # Two launches on the current stream: norm / AdaLN / per-token E4M3 quantization,
    # then the persistent fused GEMM.  The bindings take the kernels' arguments in
    # signature order, then the grid; each encodes its tensor maps from the tensors
    # handed in its descriptor positions.
    rows, device = _check_launch(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        q_norm_weight,
        k_norm_weight,
        rope_cos_sin,
        q,
        k,
        v,
        q_sf,
        k_sf,
        v_sf,
        out_mode,
    )
    _require(
        "qkv_weight_q",
        qkv_weight_q,
        torch.float8_e4m3fn,
        (MINIMAX_H3_QKV_WIDTH, MINIMAX_H3_HIDDEN),
        device,
    )
    _require(
        "qkv_weight_scale",
        qkv_weight_scale,
        torch.float32,
        (MINIMAX_H3_QKV_WIDTH,),
        device,
    )
    _require("act_q", act_q, torch.float8_e4m3fn, (rows, MINIMAX_H3_HIDDEN), device)
    _require("act_scale", act_scale, torch.float32, (rows,), device)
    index = device.index
    target, num_sms, _ = _device_facts(index)
    stage_grid, gemm_grid, num_m_tiles, total_tiles = _launch_plan(rows, num_sms)
    sfa, sfb, _ = _placeholders(index)
    gemm = _FP8_GEMM[(_fp8_unit_scale_mma(index), out_mode)]
    with torch.cuda.device(x.device), tvm_ffi.use_torch_stream():
        _entry(NORM_FP8, target)(
            x,
            x_norm_weight,
            adaln_scale,
            adaln_shift,
            adaln_index,
            _words(act_q),
            act_scale,
            rows,
            eps,
            stage_grid,
            1,
            1,
        )
        _run_gemm(
            gemm,
            target,
            index,
            a=act_q.view(torch.uint8),
            b=qkv_weight_q.view(torch.uint8),
            sfa=sfa,
            sfb=sfb,
            act_scale=act_scale,
            w_scale=qkv_weight_scale,
            q_norm_weight=q_norm_weight,
            k_norm_weight=k_norm_weight,
            rope_cos_sin=rope_cos_sin,
            q=q,
            k=k,
            v=v,
            q_sf=q_sf,
            k_sf=k_sf,
            v_sf=v_sf,
            rows=rows,
            eps=eps,
            alpha=1.0,
            out_scales=(out_scale_q, out_scale_k, out_scale_v),
            sf_muls=(sf_mul_q, sf_mul_k, sf_mul_v),
            gemm_grid=gemm_grid,
            num_m_tiles=num_m_tiles,
            total_tiles=total_tiles,
        )


@register_fake_op("flashinfer::minimax_h3_sm120_fp8_pre_attention")
def _fp8_fake(
    x,
    x_norm_weight,
    adaln_scale,
    adaln_shift,
    adaln_index,
    qkv_weight_q,
    qkv_weight_scale,
    q_norm_weight,
    k_norm_weight,
    rope_cos_sin,
    act_q,
    act_scale,
    q,
    k,
    v,
    q_sf,
    k_sf,
    v_sf,
    out_mode,
    eps,
    out_scale_q,
    out_scale_k,
    out_scale_v,
    sf_mul_q,
    sf_mul_k,
    sf_mul_v,
) -> None:
    pass


@register_custom_op(
    "flashinfer::minimax_h3_sm120_nvfp4_pre_attention",
    mutates_args=("act_q", "act_sf", "q", "k", "v", "q_sf", "k_sf", "v_sf"),
)
def _nvfp4_impl(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    qkv_weight_q: torch.Tensor,
    qkv_weight_sf: torch.Tensor,
    act_global_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    act_q: torch.Tensor,
    act_sf: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_sf: Optional[torch.Tensor],
    k_sf: Optional[torch.Tensor],
    v_sf: Optional[torch.Tensor],
    out_mode: int,
    eps: float,
    alpha: float,
    out_scale_q: float,
    out_scale_k: float,
    out_scale_v: float,
    sf_mul_q: float,
    sf_mul_k: float,
    sf_mul_v: float,
) -> None:
    # Two launches on the current stream: norm / AdaLN / NVFP4 quantization, then the
    # persistent fused block-scaled GEMM (see the FP8 route).
    rows, device = _check_launch(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        q_norm_weight,
        k_norm_weight,
        rope_cos_sin,
        q,
        k,
        v,
        q_sf,
        k_sf,
        v_sf,
        out_mode,
    )
    _require(
        "qkv_weight_q",
        qkv_weight_q,
        torch.uint8,
        (MINIMAX_H3_QKV_WIDTH, MINIMAX_H3_HIDDEN // 2),
        device,
    )
    _check_weight_sf(qkv_weight_sf, device)
    _require("act_global_scale", act_global_scale, torch.float32, (1,), device)
    _require("act_q", act_q, torch.uint8, (rows, MINIMAX_H3_HIDDEN // 2), device)
    _require("act_sf", act_sf, torch.uint8, (rows, _HIDDEN_SF), device)
    index = device.index
    target, num_sms, _ = _device_facts(index)
    stage_grid, gemm_grid, num_m_tiles, total_tiles = _launch_plan(rows, num_sms)
    with torch.cuda.device(x.device), tvm_ffi.use_torch_stream():
        _entry(NORM_NVFP4, target)(
            x,
            x_norm_weight,
            adaln_scale,
            adaln_shift,
            adaln_index,
            _words(act_q),
            act_sf,
            act_global_scale,
            rows,
            eps,
            stage_grid,
            1,
            1,
        )
        # The per-token / per-channel dequant scales are FP8-only; the NVFP4 kernels bind
        # a device scalar in their place and never read it.
        _run_gemm(
            _NVFP4_GEMM[(_nvfp4_one_m_tile_order(index), out_mode)],
            target,
            index,
            a=act_q,
            b=qkv_weight_q,
            sfa=act_sf,
            sfb=qkv_weight_sf.reshape(_SFB_ROWS, _SFB_ROW_BYTES),
            act_scale=act_global_scale,
            w_scale=act_global_scale,
            q_norm_weight=q_norm_weight,
            k_norm_weight=k_norm_weight,
            rope_cos_sin=rope_cos_sin,
            q=q,
            k=k,
            v=v,
            q_sf=q_sf,
            k_sf=k_sf,
            v_sf=v_sf,
            rows=rows,
            eps=eps,
            alpha=alpha,
            out_scales=(out_scale_q, out_scale_k, out_scale_v),
            sf_muls=(sf_mul_q, sf_mul_k, sf_mul_v),
            gemm_grid=gemm_grid,
            num_m_tiles=num_m_tiles,
            total_tiles=total_tiles,
        )


@register_fake_op("flashinfer::minimax_h3_sm120_nvfp4_pre_attention")
def _nvfp4_fake(
    x,
    x_norm_weight,
    adaln_scale,
    adaln_shift,
    adaln_index,
    qkv_weight_q,
    qkv_weight_sf,
    act_global_scale,
    q_norm_weight,
    k_norm_weight,
    rope_cos_sin,
    act_q,
    act_sf,
    q,
    k,
    v,
    q_sf,
    k_sf,
    v_sf,
    out_mode,
    eps,
    alpha,
    out_scale_q,
    out_scale_k,
    out_scale_v,
    sf_mul_q,
    sf_mul_k,
    sf_mul_v,
) -> None:
    pass


def fp8_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    r"""``RN(amax / 448)`` with a true IEEE division (``tensor / 448.0`` multiplies by a reciprocal
    and differs in the last FP32 bit, which flips E4M3 rounding ties)."""

    return amax / torch.full((), E4M3_MAX, dtype=torch.float32, device=amax.device)


def nvfp4_global_scale_from_amax(amax: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""FlashInfer NVFP4 global scale ``448 * 6 / amax`` as an FP32 ``[1]`` CUDA tensor."""

    if isinstance(amax, torch.Tensor):
        return (E4M3_MAX * E2M1_MAX / amax.float()).reshape(1).contiguous()
    return torch.tensor(
        [E4M3_MAX * E2M1_MAX / float(amax)], dtype=torch.float32, device="cuda"
    )


def quantize_minimax_h3_qkv_weight_fp8(
    qkv_weight: torch.Tensor, chunk_rows: int = 2048
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Per-output-channel E4M3 quantization of the BF16 ``[21504, 5376]`` fused QKV weight.

    Returns ``(qkv_weight_q float8_e4m3fn [21504, 5376], qkv_weight_scale float32 [21504])`` with
    ``qkv_weight_scale[n] = RN(amax(row n) / 448)`` and ``qkv_weight_q = RN(row / scale)``.
    """

    if tuple(qkv_weight.shape) != (MINIMAX_H3_QKV_WIDTH, MINIMAX_H3_HIDDEN):
        raise ValueError(
            f"qkv_weight must be [{MINIMAX_H3_QKV_WIDTH}, {MINIMAX_H3_HIDDEN}]"
        )
    weight_q = torch.empty(
        qkv_weight.shape, dtype=torch.float8_e4m3fn, device=qkv_weight.device
    )
    scale = torch.empty(
        (MINIMAX_H3_QKV_WIDTH,), dtype=torch.float32, device=qkv_weight.device
    )
    for start in range(0, MINIMAX_H3_QKV_WIDTH, chunk_rows):
        stop = min(start + chunk_rows, MINIMAX_H3_QKV_WIDTH)
        rows = qkv_weight[start:stop].float()
        row_scale = fp8_scale_from_amax(rows.abs().amax(dim=1).clamp_min(1e-12))
        scale[start:stop] = row_scale
        weight_q[start:stop] = (
            (rows / row_scale[:, None])
            .clamp(-E4M3_MAX, E4M3_MAX)
            .to(torch.float8_e4m3fn)
        )
    return weight_q, scale


def quantize_minimax_h3_qkv_weight_nvfp4(
    qkv_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""FlashInfer NVFP4 quantization of the BF16 ``[21504, 5376]`` fused QKV weight.

    Returns ``(qkv_weight_q uint8 [21504, 2688], qkv_weight_sf uint8 (128x4 swizzled layout),
    qkv_weight_global_scale float32 [1])`` exactly as :func:`flashinfer.fp4_quantize` with
    ``sf_vec_size=16`` and ``is_sf_swizzled_layout=True`` produces them.
    """

    from ..quantization import fp4_quantize

    if tuple(qkv_weight.shape) != (MINIMAX_H3_QKV_WIDTH, MINIMAX_H3_HIDDEN):
        raise ValueError(
            f"qkv_weight must be [{MINIMAX_H3_QKV_WIDTH}, {MINIMAX_H3_HIDDEN}]"
        )
    global_scale = nvfp4_global_scale_from_amax(qkv_weight.float().abs().amax())
    weight_q, weight_sf = fp4_quantize(
        qkv_weight.contiguous(),
        global_scale,
        sf_vec_size=MINIMAX_H3_SF_BLOCK,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=True,
    )
    return weight_q.contiguous(), weight_sf.contiguous(), global_scale


@supported_compute_capability([120])
@flashinfer_api
def minimax_h3_fp8_pre_attention(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    qkv_weight_q: torch.Tensor,
    qkv_weight_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    *,
    eps: float = MINIMAX_H3_DEFAULT_EPS,
    out_mode: str = "bf16",
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    q_sf: Optional[torch.Tensor] = None,
    k_sf: Optional[torch.Tensor] = None,
    v_sf: Optional[torch.Tensor] = None,
    q_descale: Optional[Scalar] = None,
    k_descale: Optional[Scalar] = None,
    v_descale: Optional[Scalar] = None,
    q_global_scale: Optional[Scalar] = None,
    k_global_scale: Optional[Scalar] = None,
    v_global_scale: Optional[Scalar] = None,
    act_q: Optional[torch.Tensor] = None,
    act_scale: Optional[torch.Tensor] = None,
) -> MiniMaxH3PreAttentionOutput:
    r"""Fused FP8 (W8A8) MiniMax-H3 pre-attention for SM120 (RTX 5090 / RTX PRO 6000 Blackwell).

    Computes, for each of the ``M`` tokens (batch 1, no sequence parallelism)::

        n   = bf16(rmsnorm(x, eps) * x_norm_weight)
        a   = bf16(adaln_shift[i] + n * bf16(1 + adaln_scale[i]))    # i = adaln_index[t]
        a[i outside [0, 9)] = 0                                      # device-side guard
        a_q = e4m3(a / s_t),  s_t = amax_t(|a|) / 448                 # per-token activation scale
        y   = bf16(a_q @ qkv_weight_q^T * s_t * qkv_weight_scale)     # [M, 56 heads x (Q, K, V) x 128]
        q   = rope(bf16(rmsnorm(y_q, eps) * q_norm_weight)),  k likewise with k_norm_weight,  v = y_v

    RoPE rotates channel pairs ``(d, d + 48)`` of the first 96 channels with
    ``rope_cos_sin[t] = [cos(48), sin(48)]`` and passes channels 96..127 through.  Both launches
    (norm/AdaLN/quantization and the fused persistent GEMM) run on the current stream.

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[M, 5376]`` hidden states.
    x_norm_weight : torch.Tensor
        BF16 ``[5376]`` pre-norm weight.
    adaln_scale, adaln_shift : torch.Tensor
        BF16 ``[9, 5376]`` AdaLN modulation tables (``MINIMAX_H3_ADALN_ROWS`` rows: the fused
        kernels index a fixed 9-row table).
    adaln_index : torch.Tensor
        Int32 ``[M]`` indices selecting one AdaLN modulation row per token; a token whose index
        lies outside ``[0, 9)`` gets a zero modulated activation.
    qkv_weight_q, qkv_weight_scale : torch.Tensor
        ``float8_e4m3fn [21504, 5376]`` and FP32 ``[21504]`` from :func:`quantize_minimax_h3_qkv_weight_fp8`.
        Output column ``h * 384 + kind * 128 + d`` is channel ``d`` of head ``h`` for ``kind`` 0 = Q, 1 = K, 2 = V.
    q_norm_weight, k_norm_weight : torch.Tensor
        BF16 ``[128]`` per-head RMSNorm weights.
    rope_cos_sin : torch.Tensor
        BF16 ``[M, 96]`` = ``[cos(48), sin(48)]`` per token.
    eps : float
        RMSNorm epsilon for both the pre-norm and the Q/K norms.
    out_mode : str
        ``"bf16"``: Q/K/V BF16 ``[M, 56, 128]``.  ``"e4m3"``: ``float8_e4m3fn [M, 56, 128]`` storing
        ``RN(value / descale)`` with the caller's per-tensor ``q_descale``, ``k_descale``, ``v_descale``.
        ``"nvfp4"``: packed E2M1 ``uint8 [M, 56, 64]`` plus UE4M3 block-16 scales ``uint8 [M, 56, 8]``
        (row-major) with the caller's per-tensor ``*_global_scale`` (FlashInfer ``fp4_quantize`` semantics).
    q, k, v, q_sf, k_sf, v_sf : Optional[torch.Tensor]
        Optional pre-allocated outputs; allocated when omitted.
    q_descale, k_descale, v_descale : Optional[Scalar]
        Per-tensor E4M3 dequantization scales required when ``out_mode == "e4m3"``.
    q_global_scale, k_global_scale, v_global_scale : Optional[Scalar]
        NVFP4 global scales required when ``out_mode == "nvfp4"``.
    act_q, act_scale : Optional[torch.Tensor]
        Optional stage-1 workspaces (``float8_e4m3fn [M, 5376]``, FP32 ``[M]``); allocated when omitted.

    Returns
    -------
    MiniMaxH3PreAttentionOutput
        ``(q, k, v, q_sf, k_sf, v_sf)``; the scale entries are ``None`` unless ``out_mode == "nvfp4"``.
    """

    rows, device = _check_common(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        q_norm_weight,
        k_norm_weight,
        rope_cos_sin,
    )
    _require(
        "qkv_weight_q",
        qkv_weight_q,
        torch.float8_e4m3fn,
        (MINIMAX_H3_QKV_WIDTH, MINIMAX_H3_HIDDEN),
        device,
    )
    _require(
        "qkv_weight_scale",
        qkv_weight_scale,
        torch.float32,
        (MINIMAX_H3_QKV_WIDTH,),
        device,
    )
    if act_q is None:
        act_q = torch.empty(
            (rows, MINIMAX_H3_HIDDEN), dtype=torch.float8_e4m3fn, device=device
        )
    if act_scale is None:
        act_scale = torch.empty((rows,), dtype=torch.float32, device=device)
    _require("act_q", act_q, torch.float8_e4m3fn, (rows, MINIMAX_H3_HIDDEN), device)
    _require("act_scale", act_scale, torch.float32, (rows,), device)
    outs, sfs, out_scales, sf_muls = _prepare_outputs(
        rows=rows,
        device=device,
        out_mode=out_mode,
        q=q,
        k=k,
        v=v,
        q_sf=q_sf,
        k_sf=k_sf,
        v_sf=v_sf,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        q_global_scale=q_global_scale,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
    )
    _fp8_impl(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        qkv_weight_q,
        qkv_weight_scale,
        q_norm_weight,
        k_norm_weight,
        rope_cos_sin,
        act_q,
        act_scale,
        outs[0],
        outs[1],
        outs[2],
        sfs[0],
        sfs[1],
        sfs[2],
        _OUT_MODES[out_mode],
        float(eps),
        *out_scales,
        *sf_muls,
    )
    return MiniMaxH3PreAttentionOutput(
        outs[0], outs[1], outs[2], sfs[0], sfs[1], sfs[2]
    )


@supported_compute_capability([120])
@flashinfer_api
def minimax_h3_nvfp4_pre_attention(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    qkv_weight_q: torch.Tensor,
    qkv_weight_sf: torch.Tensor,
    qkv_weight_global_scale: Scalar,
    act_global_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    *,
    eps: float = MINIMAX_H3_DEFAULT_EPS,
    out_mode: str = "bf16",
    alpha: Optional[float] = None,
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    q_sf: Optional[torch.Tensor] = None,
    k_sf: Optional[torch.Tensor] = None,
    v_sf: Optional[torch.Tensor] = None,
    q_descale: Optional[Scalar] = None,
    k_descale: Optional[Scalar] = None,
    v_descale: Optional[Scalar] = None,
    q_global_scale: Optional[Scalar] = None,
    k_global_scale: Optional[Scalar] = None,
    v_global_scale: Optional[Scalar] = None,
    act_q: Optional[torch.Tensor] = None,
    act_sf: Optional[torch.Tensor] = None,
) -> MiniMaxH3PreAttentionOutput:
    r"""Fused NVFP4 MiniMax-H3 pre-attention for SM120 (RTX 5090 / RTX PRO 6000 Blackwell).

    Same pre-norm, AdaLN, GEMM epilogue and output formats as :func:`minimax_h3_fp8_pre_attention`,
    with NVFP4 operands in FlashInfer's conventions: the normalized activation ``a`` is quantized
    per token like ``fp4_quantize(a, act_global_scale, sf_vec_size=16, is_sf_swizzled_layout=False)``
    (E2M1 codes ``uint8 [M, 2688]`` + UE4M3 block scales ``uint8 [M, 336]``), the weight comes from
    :func:`quantize_minimax_h3_qkv_weight_nvfp4` (128x4 swizzled scales) and the accumulator is
    rescaled by ``alpha = 1 / (act_global_scale * qkv_weight_global_scale)``.

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[M, 5376]`` hidden states.
    x_norm_weight : torch.Tensor
        BF16 ``[5376]`` pre-norm weight.
    adaln_scale, adaln_shift : torch.Tensor
        BF16 ``[9, 5376]`` AdaLN modulation tables (``MINIMAX_H3_ADALN_ROWS`` rows: the fused
        kernels index a fixed 9-row table).
    adaln_index : torch.Tensor
        Int32 ``[M]`` indices selecting one AdaLN modulation row per token; a token whose index
        lies outside ``[0, 9)`` gets a zero modulated activation.
    qkv_weight_q, qkv_weight_sf : torch.Tensor
        Outputs of :func:`quantize_minimax_h3_qkv_weight_nvfp4`.
    qkv_weight_global_scale : Scalar
        NVFP4 weight global scale as a float or single-element tensor.
    act_global_scale : torch.Tensor
        FP32 ``[1]`` CUDA tensor ``448 * 6 / amax`` of the calibrated normalized activation
        (:func:`nvfp4_global_scale_from_amax`); read on the device, no host synchronization.
    q_norm_weight, k_norm_weight : torch.Tensor
        BF16 ``[128]`` per-head RMSNorm weights.
    rope_cos_sin : torch.Tensor
        BF16 ``[M, 96]`` tensor storing ``[cos(48), sin(48)]`` per token.
    eps : float
        RMSNorm epsilon for both the pre-norm and the Q/K norms.
    out_mode : str
        Output format: ``"bf16"``, ``"e4m3"``, or ``"nvfp4"``. Shapes and quantization
        conventions match :func:`minimax_h3_fp8_pre_attention`.
    alpha : Optional[float]
        ``1 / (act_global_scale * qkv_weight_global_scale)``.  Pass it explicitly to avoid the host
        synchronization needed to read the global scales; derived from them when omitted.
    q, k, v : Optional[torch.Tensor]
        Optional pre-allocated outputs; allocated when omitted.
    q_sf, k_sf, v_sf : Optional[torch.Tensor]
        Optional pre-allocated NVFP4 scale outputs; used only when
        ``out_mode == "nvfp4"`` and ignored for other output modes. Allocated
        when omitted in NVFP4 mode.
    q_descale, k_descale, v_descale : Optional[Scalar]
        Per-tensor E4M3 dequantization scales required when ``out_mode == "e4m3"``.
    q_global_scale, k_global_scale, v_global_scale : Optional[Scalar]
        NVFP4 global scales required when ``out_mode == "nvfp4"``.
    act_q, act_sf : Optional[torch.Tensor]
        Optional stage-1 workspaces (packed E2M1 ``uint8 [M, 2688]`` and UE4M3 scales
        ``uint8 [M, 336]``); allocated when omitted.

    Returns
    -------
    MiniMaxH3PreAttentionOutput
        ``(q, k, v, q_sf, k_sf, v_sf)``; the scale entries are ``None`` unless ``out_mode == "nvfp4"``.
    """

    rows, device = _check_common(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        q_norm_weight,
        k_norm_weight,
        rope_cos_sin,
    )
    _require(
        "qkv_weight_q",
        qkv_weight_q,
        torch.uint8,
        (MINIMAX_H3_QKV_WIDTH, MINIMAX_H3_HIDDEN // 2),
        device,
    )
    _check_weight_sf(qkv_weight_sf, device)
    if not isinstance(act_global_scale, torch.Tensor):
        raise ValueError("act_global_scale must be an FP32 [1] CUDA tensor")
    act_global_scale = act_global_scale.reshape(1)
    _require("act_global_scale", act_global_scale, torch.float32, (1,), device)
    if alpha is None:
        alpha = 1.0 / (
            _scalar(act_global_scale, "act_global_scale")
            * _scalar(qkv_weight_global_scale, "qkv_weight_global_scale")
        )
    if act_q is None:
        act_q = torch.empty(
            (rows, MINIMAX_H3_HIDDEN // 2), dtype=torch.uint8, device=device
        )
    if act_sf is None:
        act_sf = torch.empty((rows, _HIDDEN_SF), dtype=torch.uint8, device=device)
    _require("act_q", act_q, torch.uint8, (rows, MINIMAX_H3_HIDDEN // 2), device)
    _require("act_sf", act_sf, torch.uint8, (rows, _HIDDEN_SF), device)
    outs, sfs, out_scales, sf_muls = _prepare_outputs(
        rows=rows,
        device=device,
        out_mode=out_mode,
        q=q,
        k=k,
        v=v,
        q_sf=q_sf,
        k_sf=k_sf,
        v_sf=v_sf,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        q_global_scale=q_global_scale,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
    )
    _nvfp4_impl(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        qkv_weight_q,
        qkv_weight_sf,
        act_global_scale,
        q_norm_weight,
        k_norm_weight,
        rope_cos_sin,
        act_q,
        act_sf,
        outs[0],
        outs[1],
        outs[2],
        sfs[0],
        sfs[1],
        sfs[2],
        _OUT_MODES[out_mode],
        float(eps),
        float(alpha),
        *out_scales,
        *sf_muls,
    )
    return MiniMaxH3PreAttentionOutput(
        outs[0], outs[1], outs[2], sfs[0], sfs[1], sfs[2]
    )


__all__ = [
    "MINIMAX_H3_ADALN_ROWS",
    "MINIMAX_H3_DEFAULT_EPS",
    "MINIMAX_H3_HEAD_DIM",
    "MINIMAX_H3_HIDDEN",
    "MINIMAX_H3_NUM_HEADS",
    "MINIMAX_H3_QKV_WIDTH",
    "MINIMAX_H3_ROPE_DIM",
    "MINIMAX_H3_SF_BLOCK",
    "MiniMaxH3PreAttentionOutput",
    "fp8_scale_from_amax",
    "minimax_h3_fp8_pre_attention",
    "minimax_h3_nvfp4_pre_attention",
    "nvfp4_global_scale_from_amax",
    "quantize_minimax_h3_qkv_weight_fp8",
    "quantize_minimax_h3_qkv_weight_nvfp4",
]
