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
from typing import Optional, Tuple, Union

import torch

from .api_logging import flashinfer_api
from .jit.qk_mxfp8_pv_nvfp4_attention_sm120 import (
    gen_qk_mxfp8_pv_nvfp4_attention_sm120_module,
)
from .trace.templates.qk_mxfp8_pv_nvfp4_attention_sm120 import (
    qk_mxfp8_pv_nvfp4_attention_sm120_fwd_trace,
    qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv_trace,
)
from .utils import supported_compute_capability


_TOKEN_BLOCK_SIZE = 128
_SUPPORTED_HEAD_DIMS = (128,)
_SUPPORTED_QKV_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_OUT_DTYPES = (torch.float16, torch.bfloat16)

_HND_LAYOUT = 1


@functools.cache
def get_qk_mxfp8_pv_nvfp4_attention_sm120_module():
    return gen_qk_mxfp8_pv_nvfp4_attention_sm120_module().build_and_load()


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        _check_cuda_contiguous(name, tensor)
        if tensor.dtype not in _SUPPORTED_QKV_DTYPES:
            raise ValueError(
                f"{name} must have dtype torch.float16 or torch.bfloat16, "
                f"got {tensor.dtype}"
            )
        if tensor.ndim != 4:
            raise ValueError(
                f"{name} must have shape [batch, heads, seq_len, head_dim], "
                f"got shape={tuple(tensor.shape)}"
            )
    if k.shape != v.shape:
        raise ValueError(
            "k and v must have the same shape, "
            f"got k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(
            "q, k, and v must have the same dtype, "
            f"got q={q.dtype}, k={k.dtype}, v={v.dtype}"
        )

    _check_same_device("k", k, "q", q)
    _check_same_device("v", v, "q", q)

    batch, num_qo_heads, _, head_dim = q.shape
    if q.shape[2] <= 0 or k.shape[2] <= 0:
        raise ValueError(
            "q and k/v sequence lengths must be positive, "
            f"got {q.shape[2]} and {k.shape[2]}"
        )
    if k.shape[0] != batch or k.shape[3] != head_dim:
        raise ValueError(
            "q, k, and v must have the same batch size and head dimension, "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    num_kv_heads = k.shape[1]
    if num_qo_heads <= 0 or num_kv_heads <= 0:
        raise ValueError(
            "num_qo_heads and num_kv_heads must be positive, "
            f"got {num_qo_heads} and {num_kv_heads}"
        )
    if num_qo_heads < num_kv_heads or num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            "num_qo_heads must be greater than or equal to and divisible by "
            f"num_kv_heads, got {num_qo_heads} and {num_kv_heads}"
        )
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head_dim must be 128, got {head_dim}")

    # K smoothing is exact for attention because subtracting one vector from
    # every K token shifts all logits in a row by the same scalar. Q remains
    # uncentered, so this recipe does not require a QK correction tensor.
    k = k - k.mean(dim=-2, keepdim=True)
    q, k, v = map(_pad_seq_len_to_128, (q, k, v))
    return q.contiguous(), k.contiguous(), v.contiguous()


@supported_compute_capability([120, 121])
@flashinfer_api(trace=qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv_trace)
def qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    r"""Preprocess and quantize dense Q/K/V tensors for SM120 QK MXFP8/PV NVFP4 attention.

    The input layout is ``[batch, num_qo_heads, M, head_dim]`` for Q and
    ``[batch, num_kv_heads, N, head_dim]`` for K/V. Inputs must be contiguous
    CUDA tensors with the same batch size, head dimension, dtype, and device.
    ``num_qo_heads`` must be greater than or equal to and divisible by
    ``num_kv_heads``. Q is padded independently to ``M_pad = round_up(M, 128)``;
    K/V are padded to ``N_pad = round_up(N, 128)``.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Dense Q/K/V tensors with dtype ``torch.float16`` or ``torch.bfloat16``.
    Returns
    -------
    Tuple[torch.Tensor, ...]
        ``q_fp8``, permuted ``k_fp8``, transposed ``v_fp4_t``, UE8M0
        ``q_scale`` and ``k_scale``, and UE4M3 ``v_scale_t``.
    """
    q_proc, k_proc, v_proc = _preprocess_qkv(q, k, v)
    batch, num_qo_heads, seq_len_q, head_dim = q_proc.shape
    num_kv_heads = k_proc.shape[1]
    seq_len_k = k_proc.shape[2]

    q_fp8 = torch.empty(
        (batch, num_qo_heads, seq_len_q, head_dim),
        device=q.device,
        dtype=torch.float8_e4m3fn,
    )
    k_fp8 = torch.empty(
        (batch, num_kv_heads, seq_len_k, head_dim),
        device=q.device,
        dtype=torch.float8_e4m3fn,
    )
    v_fp4_t = torch.empty(
        (batch, num_kv_heads, head_dim, seq_len_k // 2),
        device=q.device,
        dtype=torch.uint8,
    )
    q_scale = torch.empty(
        (batch, num_qo_heads, seq_len_q, head_dim // 32),
        device=q.device,
        dtype=torch.uint8,
    )
    k_scale = torch.empty(
        (batch, num_kv_heads, seq_len_k, head_dim // 32),
        device=q.device,
        dtype=torch.uint8,
    )
    v_scale_t = torch.empty(
        (batch, num_kv_heads, head_dim, seq_len_k // 16),
        device=q.device,
        dtype=torch.float8_e4m3fn,
    )

    module = get_qk_mxfp8_pv_nvfp4_attention_sm120_module()
    module.scaled_fp8_quant(q_proc, q_fp8, q_scale, _HND_LAYOUT)
    module.scaled_fp8_quant_permute(k_proc, k_fp8, k_scale, _HND_LAYOUT)
    module.scaled_fp4_quant_trans(v_proc, v_fp4_t, v_scale_t, _HND_LAYOUT)

    return q_fp8, k_fp8, v_fp4_t, q_scale, k_scale, v_scale_t


def _check_inputs(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    v_fp4_t: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale_t: torch.Tensor,
) -> Tuple[int, int, int, int, int, int]:
    for name, tensor in (
        ("q_fp8", q_fp8),
        ("k_fp8", k_fp8),
        ("v_fp4_t", v_fp4_t),
        ("q_scale", q_scale),
        ("k_scale", k_scale),
        ("v_scale_t", v_scale_t),
    ):
        _check_cuda_contiguous(name, tensor)

    for name, tensor in (
        ("k_fp8", k_fp8),
        ("v_fp4_t", v_fp4_t),
        ("q_scale", q_scale),
        ("k_scale", k_scale),
        ("v_scale_t", v_scale_t),
    ):
        _check_same_device(name, tensor, "q_fp8", q_fp8)

    if q_fp8.dtype != torch.float8_e4m3fn or k_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError("q_fp8 and k_fp8 must be torch.float8_e4m3fn tensors")
    if v_fp4_t.dtype != torch.uint8:
        raise ValueError("v_fp4_t must be a uint8 packed NVFP4 tensor")
    if q_scale.dtype != torch.uint8 or k_scale.dtype != torch.uint8:
        raise ValueError("q_scale and k_scale must be uint8 UE8M0 tensors")
    if v_scale_t.dtype != torch.float8_e4m3fn:
        raise ValueError("v_scale_t must be a torch.float8_e4m3fn tensor")
    if q_fp8.ndim != 4:
        raise ValueError(
            "q_fp8 must have shape [batch, num_qo_heads, seq_len, head_dim]"
        )
    if k_fp8.ndim != 4:
        raise ValueError(
            "k_fp8 must have shape [batch, num_kv_heads, seq_len, head_dim]"
        )

    batch, num_qo_heads, seq_len_q, head_dim = q_fp8.shape
    kv_batch, num_kv_heads, seq_len_k, kv_head_dim = k_fp8.shape
    if kv_batch != batch or kv_head_dim != head_dim:
        raise ValueError(
            "q_fp8 and k_fp8 must have the same batch size and head "
            "dimension, "
            f"got q_fp8={tuple(q_fp8.shape)} and k_fp8={tuple(k_fp8.shape)}"
        )
    if num_qo_heads <= 0 or num_kv_heads <= 0:
        raise ValueError(
            "num_qo_heads and num_kv_heads must be positive, "
            f"got {num_qo_heads} and {num_kv_heads}"
        )
    if num_qo_heads < num_kv_heads or num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            "num_qo_heads must be greater than or equal to and divisible by "
            f"num_kv_heads, got {num_qo_heads} and {num_kv_heads}"
        )

    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head_dim must be 128, got {head_dim}")
    if seq_len_q % _TOKEN_BLOCK_SIZE != 0:
        raise ValueError(
            f"Q sequence length must be padded to a multiple of 128, got {seq_len_q}"
        )
    if seq_len_k % _TOKEN_BLOCK_SIZE != 0:
        raise ValueError(
            f"K/V sequence length must be padded to a multiple of 128, got {seq_len_k}"
        )
    if head_dim % 16 != 0:
        raise ValueError(f"head_dim must be divisible by 16, got {head_dim}")

    expected_v = (batch, num_kv_heads, head_dim, seq_len_k // 2)
    if tuple(v_fp4_t.shape) != expected_v:
        raise ValueError(f"v_fp4_t shape {tuple(v_fp4_t.shape)} must be {expected_v}")

    expected_q_scale = (batch, num_qo_heads, seq_len_q, head_dim // 32)
    if tuple(q_scale.shape) != expected_q_scale:
        raise ValueError(
            f"q_scale shape {tuple(q_scale.shape)} must be {expected_q_scale}"
        )
    expected_k_scale = (batch, num_kv_heads, seq_len_k, head_dim // 32)
    if tuple(k_scale.shape) != expected_k_scale:
        raise ValueError(
            f"k_scale shape {tuple(k_scale.shape)} must be {expected_k_scale}"
        )

    expected_v_scale = (batch, num_kv_heads, head_dim, seq_len_k // 16)
    if tuple(v_scale_t.shape) != expected_v_scale:
        raise ValueError(
            f"v_scale_t shape {tuple(v_scale_t.shape)} must be {expected_v_scale}"
        )

    return batch, num_qo_heads, num_kv_heads, seq_len_q, seq_len_k, head_dim


@supported_compute_capability([120, 121])
@flashinfer_api(trace=qk_mxfp8_pv_nvfp4_attention_sm120_fwd_trace)
def qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    v_fp4_t: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale_t: torch.Tensor,
    sm_scale: Optional[float] = None,
    causal: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    unpadded_q_len: Optional[int] = None,
    unpadded_k_len: Optional[int] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Run SM120 QK MXFP8/PV NVFP4 attention on pre-quantized Q/K/V tensors.

    The packed tensors should be produced by
    :func:`qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv`. ``q_fp8`` uses layout
    ``[batch, num_qo_heads, M_pad, head_dim]`` and ``k_fp8`` uses
    ``[batch, num_kv_heads, N_pad, head_dim]``. ``v_fp4_t`` and
    ``v_scale_t`` are stored transposed with ``num_kv_heads``.

    Parameters
    ----------
    q_fp8, k_fp8, v_fp4_t : torch.Tensor
        MXFP8 Q/K and packed NVFP4 V tensors.
    q_scale, k_scale, v_scale_t : torch.Tensor
        UE8M0 Q/K scales and UE4M3 V scales.
    sm_scale : Optional[float], optional
        Scale applied to QK scores before softmax. Defaults to
        ``1 / sqrt(head_dim)`` when omitted.
    causal : bool, optional
        Whether to apply a causal mask.
    out, lse : Optional[torch.Tensor], optional
        Optional output and log-sum-exp buffers.
    out_dtype : torch.dtype, optional
        Output dtype used when ``out`` is not provided.
    softmax_scale : Optional[float], optional
        Deprecated alias for ``sm_scale``.
    return_lse : bool, optional
        Whether to compute and return the log-sum-exp tensor. Defaults to
        ``False`` because inference workloads generally only consume the
        attention output. The returned values correspond to the K-centered
        logits produced by the quantization helper; K centering preserves the
        attention output but shifts LSE relative to the original K tensor.
    unpadded_q_len : Optional[int], optional
        Logical query sequence length. This is required for correctly aligned
        causal masking when Q was padded by the quantization helper. Defaults
        to the physical ``M_pad`` extent.
    unpadded_k_len : Optional[int], optional
        Logical K/V sequence length. Values at or beyond this position are
        masked before softmax. Defaults to the physical ``N_pad`` extent for
        backward compatibility.

    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        The attention output when ``return_lse`` is ``False``; otherwise, the
        attention output and log-sum-exp tensor.
    """
    batch, num_qo_heads, _, seq_len_q, seq_len_k, head_dim = _check_inputs(
        q_fp8,
        k_fp8,
        v_fp4_t,
        q_scale,
        k_scale,
        v_scale_t,
    )
    if unpadded_q_len is None:
        unpadded_q_len = seq_len_q
    elif isinstance(unpadded_q_len, bool) or not isinstance(unpadded_q_len, int):
        raise ValueError(
            f"unpadded_q_len must be an integer or None, got {unpadded_q_len!r}"
        )
    if not 0 < unpadded_q_len <= seq_len_q:
        raise ValueError(
            f"unpadded_q_len must satisfy 0 < unpadded_q_len <= {seq_len_q}, "
            f"got {unpadded_q_len}"
        )

    if unpadded_k_len is None:
        unpadded_k_len = seq_len_k
    elif isinstance(unpadded_k_len, bool) or not isinstance(unpadded_k_len, int):
        raise ValueError(
            f"unpadded_k_len must be an integer or None, got {unpadded_k_len!r}"
        )
    if not 0 < unpadded_k_len <= seq_len_k:
        raise ValueError(
            f"unpadded_k_len must satisfy 0 < unpadded_k_len <= {seq_len_k}, "
            f"got {unpadded_k_len}"
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
            (batch, num_qo_heads, seq_len_q, head_dim),
            device=q_fp8.device,
            dtype=out_dtype,
        )
    else:
        _check_cuda_contiguous("out", out)
        _check_same_device("out", out, "q_fp8", q_fp8)
        if tuple(out.shape) != (batch, num_qo_heads, seq_len_q, head_dim):
            raise ValueError(
                f"out shape {tuple(out.shape)} must be "
                f"{(batch, num_qo_heads, seq_len_q, head_dim)}"
            )
        if out.dtype not in _SUPPORTED_OUT_DTYPES:
            raise ValueError(
                f"out must have dtype torch.float16 or torch.bfloat16, got {out.dtype}"
            )

    if not return_lse and lse is not None:
        raise ValueError("lse can only be provided when return_lse=True")
    if return_lse:
        if lse is None:
            lse = torch.empty(
                (batch, num_qo_heads, seq_len_q),
                device=q_fp8.device,
                dtype=torch.float32,
            )
        else:
            _check_cuda_contiguous("lse", lse)
            _check_same_device("lse", lse, "q_fp8", q_fp8)
            if tuple(lse.shape) != (batch, num_qo_heads, seq_len_q):
                raise ValueError(
                    f"lse shape {tuple(lse.shape)} must be "
                    f"{(batch, num_qo_heads, seq_len_q)}"
                )
            if lse.dtype != torch.float32:
                raise ValueError(f"lse must have dtype torch.float32, got {lse.dtype}")

    get_qk_mxfp8_pv_nvfp4_attention_sm120_module().fwd(
        q_fp8,
        k_fp8,
        v_fp4_t,
        q_scale,
        k_scale,
        v_scale_t,
        out,
        lse,
        float(sm_scale),
        bool(causal),
        unpadded_q_len,
        unpadded_k_len,
    )
    return (out, lse) if return_lse else out
