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
from typing import Optional, Tuple

import torch

from .api_logging import flashinfer_api
from .jit.nvfp4_attention_sm120 import gen_nvfp4_attention_sm120_module
from .trace.templates.nvfp4_attention_sm120 import (
    nvfp4_attention_sm120_fwd_trace,
    nvfp4_attention_sm120_quantize_qkv_trace,
)
from .utils import supported_compute_capability


_TOKEN_BLOCK_SIZE = 128
_SUPPORTED_HEAD_DIMS = (64, 128)
_SUPPORTED_QKV_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_OUT_DTYPES = (torch.float16, torch.bfloat16)

_HND_LAYOUT = 1


@functools.cache
def get_nvfp4_attention_sm120_module():
    return gen_nvfp4_attention_sm120_module().build_and_load()


def _check_cuda_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor, got device={tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tensor.stride()}")


def _check_same_device(
    name: str,
    tensor: torch.Tensor,
    ref_name: str,
    ref: torch.Tensor,
) -> None:
    if tensor.device != ref.device:
        raise ValueError(
            f"{name} must be on the same device as {ref_name}, "
            f"got {tensor.device} and {ref.device}"
        )


def _pad_seq_len_to_128(x: torch.Tensor) -> torch.Tensor:
    pad_len = (-x.shape[2]) % _TOKEN_BLOCK_SIZE
    if pad_len == 0:
        return x.contiguous()
    return torch.nn.functional.pad(x, (0, 0, 0, pad_len), value=0).contiguous()


def _preprocess_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    per_block_mean: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        _check_cuda_contiguous(name, tensor)
        if tensor.dtype not in _SUPPORTED_QKV_DTYPES:
            raise ValueError(
                f"{name} must have dtype torch.float16 or torch.bfloat16, "
                f"got {tensor.dtype}"
            )
        if tensor.ndim != 4:
            raise ValueError(
                f"{name} must have shape [batch, num_heads, seq_len, head_dim], "
                f"got shape={tuple(tensor.shape)}"
            )
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            "q, k, and v must have the same shape, "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(
            "q, k, and v must have the same dtype, "
            f"got q={q.dtype}, k={k.dtype}, v={v.dtype}"
        )

    _check_same_device("k", k, "q", q)
    _check_same_device("v", v, "q", q)

    batch, num_heads, seq_len, head_dim = q.shape
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head_dim must be 64 or 128, got {head_dim}")

    k = k - k.mean(dim=-2, keepdim=True)
    q, k, v = map(_pad_seq_len_to_128, (q, k, v))
    seq_len = q.shape[2]

    if per_block_mean:
        num_groups = seq_len // _TOKEN_BLOCK_SIZE
        q_grouped = q.reshape(batch, num_heads, num_groups, _TOKEN_BLOCK_SIZE, head_dim)
        qm = q_grouped.mean(dim=3)
        q = (
            (q_grouped - qm.unsqueeze(3))
            .reshape(batch, num_heads, seq_len, head_dim)
            .contiguous()
        )
    else:
        qm = q.mean(dim=-2, keepdim=True)
        q = (q - qm).contiguous()

    # Compact layout: one correction row per 128-token Q block ([B, H,
    # seq_len / 128, seq_len]), or a single row when per_block_mean=False.
    # The kernel's TMA descriptor addresses the tensor this way and
    # broadcasts each row across the 128 rows of the Q tile in smem.
    # Multiply in fp32: a float16 matmul output would overflow at 65504
    # even though the accumulation itself runs in fp32.
    qk_correction = torch.matmul(
        qm.to(torch.float32), k.transpose(-2, -1).to(torch.float32)
    )
    qk_correction = qk_correction.contiguous()
    return q.contiguous(), k.contiguous(), v.contiguous(), qk_correction


@supported_compute_capability([120])
@flashinfer_api(trace=nvfp4_attention_sm120_quantize_qkv_trace)
def nvfp4_attention_sm120_quantize_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    per_block_mean: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    r"""Preprocess and quantize dense Q/K/V tensors for SM120 NVFP4 attention.

    The input layout is ``[batch, num_heads, seq_len, head_dim]``. Inputs must be
    contiguous CUDA tensors with the same shape, dtype, and device. The sequence
    dimension is padded to a multiple of 128 before Q/K/V are quantized.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Dense Q/K/V tensors with dtype ``torch.float16`` or ``torch.bfloat16``.
    per_block_mean : bool, optional
        Whether to center Q per 128-token block. When ``False``, Q is centered
        once across the full sequence.

    Returns
    -------
    Tuple[torch.Tensor, ...]
        ``q_fp4``, ``k_fp4``, transposed ``v_fp4_t``, scale tensors
        ``q_scale``, ``k_scale``, ``v_scale_t``, and the compact FP32 QK
        correction with shape ``[batch, num_heads, seq_len / 128, seq_len]``
        (``[batch, num_heads, 1, seq_len]`` when ``per_block_mean=False``).
    """
    q_proc, k_proc, v_proc, qk_correction = _preprocess_qkv(q, k, v, per_block_mean)
    batch, num_heads, seq_len, head_dim = q_proc.shape

    q_fp4 = torch.empty(
        (batch, num_heads, seq_len, head_dim // 2), device=q.device, dtype=torch.uint8
    )
    k_fp4 = torch.empty_like(q_fp4)
    v_fp4_t = torch.empty(
        (batch, num_heads, head_dim, seq_len // 2), device=q.device, dtype=torch.uint8
    )
    q_scale = torch.empty(
        (batch, num_heads, seq_len, head_dim // 16),
        device=q.device,
        dtype=torch.float8_e4m3fn,
    )
    k_scale = torch.empty_like(q_scale)
    v_scale_t = torch.empty(
        (batch, num_heads, head_dim, seq_len // 16),
        device=q.device,
        dtype=torch.float8_e4m3fn,
    )

    module = get_nvfp4_attention_sm120_module()
    module.scaled_fp4_quant(q_proc, q_fp4, q_scale, _HND_LAYOUT)
    module.scaled_fp4_quant_permute(k_proc, k_fp4, k_scale, _HND_LAYOUT)
    module.scaled_fp4_quant_trans(v_proc, v_fp4_t, v_scale_t, _HND_LAYOUT)

    return q_fp4, k_fp4, v_fp4_t, q_scale, k_scale, v_scale_t, qk_correction


def _check_inputs(
    q_fp4: torch.Tensor,
    k_fp4: torch.Tensor,
    v_fp4_t: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale_t: torch.Tensor,
    qk_correction: torch.Tensor,
    per_block_mean: bool,
) -> Tuple[int, int, int, int]:
    for name, tensor in (
        ("q_fp4", q_fp4),
        ("k_fp4", k_fp4),
        ("v_fp4_t", v_fp4_t),
        ("q_scale", q_scale),
        ("k_scale", k_scale),
        ("v_scale_t", v_scale_t),
        ("qk_correction", qk_correction),
    ):
        _check_cuda_contiguous(name, tensor)

    for name, tensor in (
        ("k_fp4", k_fp4),
        ("v_fp4_t", v_fp4_t),
        ("q_scale", q_scale),
        ("k_scale", k_scale),
        ("v_scale_t", v_scale_t),
        ("qk_correction", qk_correction),
    ):
        _check_same_device(name, tensor, "q_fp4", q_fp4)

    if (
        q_fp4.dtype != torch.uint8
        or k_fp4.dtype != torch.uint8
        or v_fp4_t.dtype != torch.uint8
    ):
        raise ValueError("q_fp4, k_fp4, and v_fp4_t must be uint8 packed FP4 tensors")
    if q_scale.dtype != torch.float8_e4m3fn or k_scale.dtype != torch.float8_e4m3fn:
        raise ValueError("q_scale and k_scale must be torch.float8_e4m3fn tensors")
    if v_scale_t.dtype != torch.float8_e4m3fn:
        raise ValueError("v_scale_t must be a torch.float8_e4m3fn tensor")
    if qk_correction.dtype != torch.float32:
        raise ValueError("qk_correction must be a torch.float32 tensor")

    if q_fp4.ndim != 4:
        raise ValueError(
            "q_fp4 must have shape [batch, num_heads, seq_len, head_dim / 2]"
        )
    if k_fp4.shape != q_fp4.shape:
        raise ValueError(
            f"k_fp4 shape {tuple(k_fp4.shape)} must match q_fp4 {tuple(q_fp4.shape)}"
        )

    batch, num_heads, seq_len, packed_head_dim = q_fp4.shape
    head_dim = packed_head_dim * 2
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head_dim must be 64 or 128, got {head_dim}")
    if seq_len % _TOKEN_BLOCK_SIZE != 0:
        raise ValueError(f"seq_len must be padded to a multiple of 128, got {seq_len}")
    if head_dim % 16 != 0:
        raise ValueError(f"head_dim must be divisible by 16, got {head_dim}")

    expected_v = (batch, num_heads, head_dim, seq_len // 2)
    if tuple(v_fp4_t.shape) != expected_v:
        raise ValueError(f"v_fp4_t shape {tuple(v_fp4_t.shape)} must be {expected_v}")

    expected_sf_qk = (batch, num_heads, seq_len, head_dim // 16)
    if tuple(q_scale.shape) != expected_sf_qk:
        raise ValueError(
            f"q_scale shape {tuple(q_scale.shape)} must be {expected_sf_qk}"
        )
    if tuple(k_scale.shape) != expected_sf_qk:
        raise ValueError(
            f"k_scale shape {tuple(k_scale.shape)} must be {expected_sf_qk}"
        )

    expected_v_scale = (batch, num_heads, head_dim, seq_len // 16)
    if tuple(v_scale_t.shape) != expected_v_scale:
        raise ValueError(
            f"v_scale_t shape {tuple(v_scale_t.shape)} must be {expected_v_scale}"
        )

    if (
        qk_correction.ndim != 4
        or qk_correction.shape[0] != batch
        or qk_correction.shape[1] != num_heads
    ):
        raise ValueError(
            "qk_correction must have shape [batch, num_heads, seq_len_s, seq_len]"
        )
    expected_delta_groups = seq_len // _TOKEN_BLOCK_SIZE if per_block_mean else 1
    if qk_correction.shape[2] != expected_delta_groups:
        raise ValueError(
            f"qk_correction must have one row per 128-token block "
            f"({expected_delta_groups}), got {qk_correction.shape[2]}"
        )
    if qk_correction.shape[-1] != seq_len:
        raise ValueError(
            f"qk_correction last dimension must be {seq_len}, got {qk_correction.shape[-1]}"
        )

    return batch, num_heads, seq_len, head_dim


@supported_compute_capability([120])
@flashinfer_api(trace=nvfp4_attention_sm120_fwd_trace)
def nvfp4_attention_sm120_fwd(
    q_fp4: torch.Tensor,
    k_fp4: torch.Tensor,
    v_fp4_t: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale_t: torch.Tensor,
    qk_correction: torch.Tensor,
    sm_scale: Optional[float] = None,
    causal: bool = False,
    per_block_mean: bool = True,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Run SM120 NVFP4 attention on pre-quantized Q/K/V tensors.

    The packed tensors should be produced by
    :func:`nvfp4_attention_sm120_quantize_qkv`. ``q_fp4`` and ``k_fp4`` use layout
    ``[batch, num_heads, seq_len, head_dim / 2]``; ``v_fp4_t`` and ``v_scale_t`` are
    stored transposed as ``[batch, num_heads, head_dim, packed_seq_len]``.

    Parameters
    ----------
    q_fp4, k_fp4, v_fp4_t : torch.Tensor
        Packed NVFP4 Q/K/V tensors.
    q_scale, k_scale, v_scale_t : torch.Tensor
        Per-vector FP8 scale factors for Q/K/V.
    qk_correction : torch.Tensor
        Compact FP32 correction term returned by
        :func:`nvfp4_attention_sm120_quantize_qkv`, one row per 128-token
        Q block.
    sm_scale : Optional[float], optional
        Scale applied to QK scores before softmax. Defaults to
        ``1 / sqrt(head_dim)`` when omitted.
    causal : bool, optional
        Whether to apply a causal mask.
    per_block_mean : bool, optional
        Must match the value used by ``nvfp4_attention_sm120_quantize_qkv``.
    out, lse : Optional[torch.Tensor], optional
        Optional output and log-sum-exp buffers.
    out_dtype : torch.dtype, optional
        Output dtype used when ``out`` is not provided.
    softmax_scale : Optional[float], optional
        Deprecated alias for ``sm_scale``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Attention output and log-sum-exp tensor.
    """
    per_block_mean = bool(per_block_mean)
    batch, num_heads, seq_len, head_dim = _check_inputs(
        q_fp4,
        k_fp4,
        v_fp4_t,
        q_scale,
        k_scale,
        v_scale_t,
        qk_correction,
        per_block_mean,
    )
    if sm_scale is not None and softmax_scale is not None:
        raise ValueError("Specify only one of sm_scale or softmax_scale")
    if sm_scale is None:
        sm_scale = head_dim**-0.5 if softmax_scale is None else softmax_scale

    if out is None:
        if out_dtype not in _SUPPORTED_OUT_DTYPES:
            raise ValueError(
                f"out_dtype must be torch.float16 or torch.bfloat16, got {out_dtype}"
            )
        out = torch.empty(
            (batch, num_heads, seq_len, head_dim),
            device=q_fp4.device,
            dtype=out_dtype,
        )
    else:
        _check_cuda_contiguous("out", out)
        _check_same_device("out", out, "q_fp4", q_fp4)
        if tuple(out.shape) != (batch, num_heads, seq_len, head_dim):
            raise ValueError(
                f"out shape {tuple(out.shape)} must be {(batch, num_heads, seq_len, head_dim)}"
            )
        if out.dtype not in _SUPPORTED_OUT_DTYPES:
            raise ValueError(
                f"out must have dtype torch.float16 or torch.bfloat16, got {out.dtype}"
            )

    if lse is None:
        lse = torch.empty(
            (batch, num_heads, seq_len), device=q_fp4.device, dtype=torch.float32
        )
    else:
        _check_cuda_contiguous("lse", lse)
        _check_same_device("lse", lse, "q_fp4", q_fp4)
        if tuple(lse.shape) != (batch, num_heads, seq_len):
            raise ValueError(
                f"lse shape {tuple(lse.shape)} must be {(batch, num_heads, seq_len)}"
            )
        if lse.dtype != torch.float32:
            raise ValueError(f"lse must have dtype torch.float32, got {lse.dtype}")

    get_nvfp4_attention_sm120_module().fwd(
        q_fp4,
        k_fp4,
        v_fp4_t,
        q_scale,
        k_scale,
        v_scale_t,
        qk_correction,
        out,
        lse,
        float(sm_scale),
        bool(causal),
        per_block_mean,
    )
    return out, lse
