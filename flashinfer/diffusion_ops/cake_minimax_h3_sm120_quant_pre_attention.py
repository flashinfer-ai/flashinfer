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
from typing import NamedTuple, Optional, Tuple, Union

import torch

from ..api_logging import flashinfer_api
from ..jit.cake_minimax_h3_sm120_quant_pre_attention import (
    gen_minimax_h3_sm120_quant_pre_attention_module,
)
from ..utils import register_custom_op, register_fake_op, supported_compute_capability

MINIMAX_H3_HIDDEN = 5376
MINIMAX_H3_NUM_HEADS = 56
MINIMAX_H3_HEAD_DIM = 128
MINIMAX_H3_QKV_WIDTH = MINIMAX_H3_NUM_HEADS * 3 * MINIMAX_H3_HEAD_DIM  # 21504
MINIMAX_H3_ROPE_DIM = 96
MINIMAX_H3_SF_BLOCK = 16
MINIMAX_H3_DEFAULT_EPS = 1.0e-5
MINIMAX_H3_MAX_ROWS = 1 << 24
E4M3_MAX = 448.0
E2M1_MAX = 6.0

_OUT_MODES = {"bf16": 0, "e4m3": 1, "nvfp4": 2}
_HIDDEN_SF = MINIMAX_H3_HIDDEN // MINIMAX_H3_SF_BLOCK  # 336
_HEAD_SF = MINIMAX_H3_HEAD_DIM // MINIMAX_H3_SF_BLOCK  # 8

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
def _get_module():
    return gen_minimax_h3_sm120_quant_pre_attention_module().build_and_load()


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
    rows = _check_rows(x)
    device = x.device
    _require("x", x, torch.bfloat16, (rows, MINIMAX_H3_HIDDEN), device)
    _require(
        "x_norm_weight", x_norm_weight, torch.bfloat16, (MINIMAX_H3_HIDDEN,), device
    )
    if (
        adaln_scale.ndim != 2
        or adaln_scale.shape[1] != MINIMAX_H3_HIDDEN
        or adaln_scale.shape[0] < 1
    ):
        raise ValueError(f"adaln_scale must be [rows, {MINIMAX_H3_HIDDEN}]")
    adaln_rows = int(adaln_scale.shape[0])
    _require(
        "adaln_scale",
        adaln_scale,
        torch.bfloat16,
        (adaln_rows, MINIMAX_H3_HIDDEN),
        device,
    )
    _require(
        "adaln_shift",
        adaln_shift,
        torch.bfloat16,
        (adaln_rows, MINIMAX_H3_HIDDEN),
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
    heads = (rows, MINIMAX_H3_NUM_HEADS, MINIMAX_H3_HEAD_DIM)
    if out_mode == "bf16":
        dtype, shape = torch.bfloat16, heads
    elif out_mode == "e4m3":
        dtype, shape = torch.float8_e4m3fn, heads
    else:
        dtype, shape = (
            torch.uint8,
            (rows, MINIMAX_H3_NUM_HEADS, MINIMAX_H3_HEAD_DIM // 2),
        )
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
    _get_module().minimax_h3_sm120_fp8_pre_attention(
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
    _get_module().minimax_h3_sm120_nvfp4_pre_attention(
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
        BF16 ``[rows, 5376]`` AdaLN modulation tables; ``adaln_index`` (int32 ``[M]``) selects a row per token.
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
    qkv_weight_q, qkv_weight_sf, qkv_weight_global_scale
        Outputs of :func:`quantize_minimax_h3_qkv_weight_nvfp4`.
    act_global_scale : torch.Tensor
        FP32 ``[1]`` CUDA tensor ``448 * 6 / amax`` of the calibrated normalized activation
        (:func:`nvfp4_global_scale_from_amax`); read on the device, no host synchronization.
    alpha : Optional[float]
        ``1 / (act_global_scale * qkv_weight_global_scale)``.  Pass it explicitly to avoid the host
        synchronization needed to read the global scales; derived from them when omitted.

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
