# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Internal Sparse-MLA paged attention implementation for SM120.

Auto-dispatches between decode (num_tokens <= 64) and prefill (larger). Both
DSv3.2 (d_qk=576) and DSv4 (d_qk=512) decode go through dedicated warp-spec
standalone kernels; prefill is dispatched through the shared orchestrator.

The user-facing sparse MLA entry points are
``flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4`` for DeepSeek V4 and
``flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(..., sparse_mla_top_k=...)``
for the DSv3.2 / GLM sparse top-k path. This module keeps only the SM120
implementation hooks used by those dispatchers and focused kernel
tests/benchmarks.
"""

from __future__ import annotations

import functools
import os
from types import SimpleNamespace
from typing import List, Optional

import torch

from ..autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ..jit.mla import gen_sparse_mla_sm120_module
from ..utils import (
    register_custom_op,
    register_fake_op,
    supported_compute_capability,
)

# Kernel-side constants. Mirrored from
# include/flashinfer/attention/sparse_mla_sm120/{arch,model}/*.cuh.
_D_V = 512  # value head dim (universal across DSV3_2 and DSV4)
_BI = 64  # KV partition tile size in candidates (BLOCK_SIZE_N)

# Decode/prefill cutoff: num_tokens > _DECODE_MAX_TOKENS routes to the
# prefill orchestrator; otherwise to the standalone decode kernels.
_DECODE_MAX_TOKENS = 64

# decode-dsv4 instantiation set. Shapes outside this table fall through to
# decode-dsv3_2 / prefill. NH=8 is the small-TP corner case; the kernel pads
# the head tile to HPB=16 with zero-Q rows and gates writes by NUM_HEADS.
_DECODE_DSV4_DISPATCH = frozenset(
    {
        (8, 128),
        (8, 512),
        (8, 1024),
        (16, 128),
        (16, 512),
        (16, 1024),
        (32, 128),
        (32, 512),
        (32, 1024),
        (64, 128),
        (64, 512),
        (64, 1024),
        (128, 128),
        (128, 512),
        (128, 1024),
    }
)
_DECODE_DSV4_PAGE_BLOCK_SIZE = 64

# decode-dsv3_2 instantiation set.
_DECODE_DSV3_2_DISPATCH = frozenset(
    {
        (8, 128),
        (8, 512),
        (8, 1024),
        (8, 2048),
        (16, 128),
        (16, 512),
        (16, 1024),
        (16, 2048),
        (32, 128),
        (32, 512),
        (32, 1024),
        (32, 2048),
        (64, 128),
        (64, 512),
        (64, 1024),
        (64, 2048),
        (128, 128),
        (128, 512),
        (128, 1024),
        (128, 2048),
    }
)
_DECODE_DSV3_2_PAGE_BLOCK_SIZE = 64

_MODEL_TYPE_DSV3_2 = 0
_MODEL_TYPE_DSV4 = 1
_MODEL_TYPE_GLM_NSA = 2
_KV_SCALE_FORMATS = frozenset({"auto", "pow2_fp32", "arbitrary_fp32"})
_BPT_DSV3_2 = 656
_BPT_DSV4 = 584


def _require_d_v_512(d_v: int) -> None:
    if int(d_v) != _D_V:
        raise ValueError(f"SM120 sparse-MLA requires d_v == {_D_V}, got {d_v}")


def _check_last_dim_512(tensor: torch.Tensor, name: str) -> None:
    if tensor.shape[-1] != _D_V:
        raise ValueError(
            f"{name} last dimension must be {_D_V}, got shape {tuple(tensor.shape)}"
        )


def _normalize_kv_scale_format(kv_scale_format: str) -> str:
    fmt = str(kv_scale_format).lower().replace("-", "_")
    if fmt not in _KV_SCALE_FORMATS:
        raise ValueError(
            "kv_scale_format must be one of "
            f"{sorted(_KV_SCALE_FORMATS)}, got {kv_scale_format!r}"
        )
    return fmt


def _resolve_model_type(d_qk: int, kv_scale_format: str) -> int:
    fmt = _normalize_kv_scale_format(kv_scale_format)
    if d_qk == 576:
        if fmt == "arbitrary_fp32":
            return _MODEL_TYPE_GLM_NSA
        return _MODEL_TYPE_DSV3_2
    if d_qk == 512:
        if fmt != "auto":
            raise ValueError(
                "kv_scale_format is only configurable for d_qk=576; "
                f"got d_qk=512 with kv_scale_format={kv_scale_format!r}"
            )
        return _MODEL_TYPE_DSV4
    raise ValueError(f"SM120 sparse-MLA supports d_qk=576 or d_qk=512, got d_qk={d_qk}")


def _bytes_per_token_for_model_type(model_type: int) -> int:
    if model_type in (_MODEL_TYPE_DSV3_2, _MODEL_TYPE_GLM_NSA):
        return _BPT_DSV3_2
    if model_type == _MODEL_TYPE_DSV4:
        return _BPT_DSV4
    raise ValueError(f"Unsupported SM120 sparse-MLA model_type={model_type}")


def _packed_kv_page_block_size(
    kv_cache: torch.Tensor,
    *,
    model_type: int,
    name: str,
) -> int:
    bytes_per_token = _bytes_per_token_for_model_type(model_type)
    if kv_cache.ndim == 2:
        block_bytes = int(kv_cache.shape[1])
        if block_bytes % bytes_per_token != 0:
            raise ValueError(
                f"{name} 2-D block width {block_bytes} is not divisible by "
                f"{bytes_per_token} bytes/token"
            )
        return block_bytes // bytes_per_token
    if kv_cache.ndim == 3:
        if kv_cache.shape[-1] != bytes_per_token:
            raise ValueError(
                f"{name} last dim must be {bytes_per_token}, got {kv_cache.shape[-1]}"
            )
        return int(kv_cache.shape[1])
    if kv_cache.ndim == 4:
        if kv_cache.shape[-1] != bytes_per_token:
            raise ValueError(
                f"{name} last dim must be {bytes_per_token}, got {kv_cache.shape[-1]}"
            )
        if kv_cache.shape[1] == 1:
            # HND: [num_pages, 1, page_block_size, bytes_per_token].
            return int(kv_cache.shape[2])
        if kv_cache.shape[2] == 1:
            # NHD: [num_pages, page_block_size, 1, bytes_per_token].
            return int(kv_cache.shape[1])
        raise ValueError(
            f"{name} must have a singleton KV-head axis in dim 1 (HND) or "
            f"dim 2 (NHD), got shape {tuple(kv_cache.shape)}"
        )
    raise ValueError(f"{name} must have ndim 2, 3, or 4, got {kv_cache.ndim}")


def _decode_dsv3_2_dispatchable(
    num_tokens: int, num_heads: int, topk: int, d_qk: int, page_block_size: int
) -> bool:
    """True iff decode-dsv3_2 supports this shape configuration."""
    return (
        num_tokens <= _DECODE_MAX_TOKENS
        and d_qk == 576
        and page_block_size == _DECODE_DSV3_2_PAGE_BLOCK_SIZE
        and (num_heads, topk) in _DECODE_DSV3_2_DISPATCH
    )


def _decode_dsv4_dispatchable(
    num_tokens: int,
    num_heads: int,
    topk: int,
    d_qk: int,
    page_block_size: int,
    extra_topk: int = 0,
) -> bool:
    """True iff decode-dsv4 supports this shape configuration.

    The split count only affects scratch size; the merge kernel stores per-split
    LSE in dynamic shared memory.
    """
    return (
        num_tokens <= _DECODE_MAX_TOKENS
        and d_qk == 512
        and page_block_size == _DECODE_DSV4_PAGE_BLOCK_SIZE
        and (num_heads, topk) in _DECODE_DSV4_DISPATCH
    )


def _decode_scratch_views(
    mid_out: Optional[torch.Tensor],
    mid_lse: Optional[torch.Tensor],
    num_tokens: int,
    num_heads: int,
    num_splits: int,
    d_v: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve caller-supplied scratch buffers for split-K decode kernels."""
    if mid_out is None or mid_lse is None:
        raise ValueError(
            "SM120 sparse-MLA decode requires caller-supplied mid_out and "
            "mid_lse scratch. Allocate shapes "
            f"[{num_tokens}, {num_heads}, {num_splits}, {d_v}] bf16 and "
            f"[{num_tokens}, {num_heads}, {num_splits}] fp32."
        )
    need_out = (num_tokens, num_heads, num_splits, d_v)
    need_lse = (num_tokens, num_heads, num_splits)
    if any(mid_out.size(d) < need_out[d] for d in range(4)):
        raise ValueError(
            f"mid_out shape {tuple(mid_out.shape)} too small for required "
            f"[num_tokens={num_tokens}, num_heads={num_heads}, "
            f"num_splits={num_splits}, d_v={d_v}]"
        )
    if any(mid_lse.size(d) < need_lse[d] for d in range(3)):
        raise ValueError(
            f"mid_lse shape {tuple(mid_lse.shape)} too small for required "
            f"[num_tokens={num_tokens}, num_heads={num_heads}, "
            f"num_splits={num_splits}]"
        )
    return (
        mid_out[:num_tokens, :num_heads, :num_splits, :d_v],
        mid_lse[:num_tokens, :num_heads, :num_splits],
    )


@functools.cache
def get_sparse_mla_sm120_module():
    """Build and cache the sparse-MLA SM120 module + bound custom op."""
    module = gen_sparse_mla_sm120_module().build_and_load()

    @register_custom_op(
        "flashinfer::sparse_mla_sm120_paged_attention",
        mutates_args=("output", "out_lse", "mid_out", "mid_lse"),
    )
    def _paged_attention(
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        output: torch.Tensor,
        out_lse: torch.Tensor,
        sm_scale: float,
        d_v: int,
        model_type: int,
        topk_length: Optional[torch.Tensor],
        attn_sink: Optional[torch.Tensor],
        extra_kv_cache: Optional[torch.Tensor],
        extra_indices: Optional[torch.Tensor],
        extra_topk_length: Optional[torch.Tensor],
        mid_out: Optional[torch.Tensor],
        mid_lse: Optional[torch.Tensor],
    ) -> None:
        num_tokens, num_heads, d_qk = q.shape
        topk = indices.shape[-1]
        _require_d_v_512(d_v)
        _check_last_dim_512(output, "output")

        kv_pbs = _packed_kv_page_block_size(
            kv_cache, model_type=model_type, name="kv_cache"
        )
        extra_topk = int(extra_indices.size(-1)) if extra_indices is not None else 0
        if (
            model_type == _MODEL_TYPE_DSV4
            and kv_pbs == _DECODE_DSV4_PAGE_BLOCK_SIZE
            and _decode_dsv4_dispatchable(
                num_tokens, num_heads, topk, d_qk, kv_pbs, extra_topk
            )
        ):
            num_splits_main = (topk + _BI - 1) // _BI
            num_splits_extra = (extra_topk + _BI - 1) // _BI
            num_splits = num_splits_main + num_splits_extra
            mid_out_view, mid_lse_view = _decode_scratch_views(
                mid_out, mid_lse, num_tokens, num_heads, num_splits, d_v
            )
            # FFI binding extracts the true block stride from kv_cache.stride(0),
            # so paged layouts with padded strides and microbench 2-D layouts
            # both work.
            sparse_mla_sm120_decode_dsv4(
                q,
                kv_cache,
                indices,
                mid_out_view,
                mid_lse_view,
                output,
                out_lse,
                sm_scale,
                topk_length=topk_length,
                attn_sink=attn_sink,
                extra_kv_cache=extra_kv_cache,
                extra_indices=extra_indices,
                extra_topk_length=extra_topk_length,
            )
            return

        if model_type in (
            _MODEL_TYPE_DSV3_2,
            _MODEL_TYPE_GLM_NSA,
        ) and _decode_dsv3_2_dispatchable(num_tokens, num_heads, topk, d_qk, kv_pbs):
            num_splits = (topk + _BI - 1) // _BI
            mid_out_view, mid_lse_view = _decode_scratch_views(
                mid_out, mid_lse, num_tokens, num_heads, num_splits, d_v
            )
            sparse_mla_sm120_decode_dsv3_2(
                q,
                kv_cache,
                indices,
                mid_out_view,
                mid_lse_view,
                output,
                out_lse,
                sm_scale,
                topk_length=topk_length,
                attn_sink=attn_sink,
                model_type=model_type,
            )
            return

        module.sparse_mla_sm120_paged_attention(
            q,
            kv_cache,
            indices,
            output,
            out_lse,
            sm_scale,
            model_type,
            topk_length,
            attn_sink,
            extra_kv_cache,
            extra_indices,
            extra_topk_length,
        )

    @register_fake_op("flashinfer::sparse_mla_sm120_paged_attention")
    def _fake_paged_attention(*_args, **_kwargs) -> None:
        return None

    return SimpleNamespace(paged_attention=_paged_attention)


@supported_compute_capability([120, 121])
def _sparse_mla_sm120_paged_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    out_lse: torch.Tensor,
    sm_scale: float,
    *,
    d_v: int = _D_V,
    kv_scale_format: str = "auto",
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    mid_out: Optional[torch.Tensor] = None,
    mid_lse: Optional[torch.Tensor] = None,
) -> None:
    r"""Internal Sparse-MLA paged attention on SM120.

    Auto-dispatches decode (``num_tokens <= 64``) vs prefill (larger).
    Mutates ``output`` and ``out_lse`` in place.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor, shape ``[num_tokens, num_heads, d_qk]``, dtype bf16.
        ``d_qk=576`` uses the V32-family inline-scale cache and
        ``d_qk=512`` uses the DSv4 footer-scale cache.
    kv_cache : torch.Tensor
        Byte-packed paged main KV cache. Accepted forms are 3D
        ``[num_blocks, page_block_size, bytes]``, HND
        ``[num_blocks, 1, page_block_size, bytes]``, or NHD
        ``[num_blocks, page_block_size, 1, bytes]``. The SM120 binding derives
        page size and block stride from the tensor metadata without
        materializing a layout conversion.
    indices : torch.Tensor
        Paged slot IDs per query token, shape ``[num_tokens, topk]`` or
        ``[num_tokens, 1, topk]``, dtype int32. ``-1`` marks invalid /
        out-of-window slots (kernel skips).
    output : torch.Tensor
        In-place output, shape ``[num_tokens, num_heads, d_v]``, dtype bf16.
    out_lse : torch.Tensor
        In-place log-sum-exp, shape ``[num_tokens, num_heads]``, dtype float32.
    sm_scale : float
        Softmax scale (typically ``1 / sqrt(d_qk)``).
    d_v : int
        Value head dim. ``512`` for both DSV3_2 and DSV4 today.
    kv_scale_format : str
        Scale semantics for ``d_qk=576``. ``"auto"`` and ``"pow2_fp32"``
        select DSv3.2 power-of-2 FP32 inline scales; ``"arbitrary_fp32"``
        selects GLM-style arbitrary FP32 inline scales.
    topk_length : Optional[torch.Tensor]
        Effective top-k length per query token, shape ``[num_tokens]``, dtype
        int32. Required for sliding-window MLA near sequence start; ``None``
        for uniform top-k.
    attn_sink : Optional[torch.Tensor]
        Per-head learnable bias added pre-softmax, shape ``[num_heads]``,
        dtype float32. FlashMLA V4 convention: ``output *= sigmoid(lse -
        sink)`` and ``lse' = log(exp(lse) + exp(sink))``.
    extra_kv_cache : Optional[torch.Tensor]
        Optional secondary KV cache (DSv4 C4A / C128A layers). When provided,
        ``extra_indices`` must also be passed. DSV4-only.
    extra_indices : Optional[torch.Tensor]
        Paged slot IDs for the secondary cache, shape
        ``[num_tokens, extra_topk]`` or ``[num_tokens, 1, extra_topk]``,
        dtype int32.
    extra_topk_length : Optional[torch.Tensor]
        Effective top-k length per query token for the secondary cache,
        shape ``[num_tokens]``, dtype int32.
    mid_out : Optional[torch.Tensor]
        Pre-allocated split-K partial-output scratch, shape
        ``[>=num_tokens, >=num_heads, >=num_splits, >=d_v]``, dtype bf16. Only
        consumed by the decode path; required when the call dispatches to a
        decode kernel.
    mid_lse : Optional[torch.Tensor]
        Pre-allocated split-K LSE scratch, shape
        ``[>=num_tokens, >=num_heads, >=num_splits]``, dtype float32. Pair with
        ``mid_out`` when the call dispatches to a decode kernel.

    Notes
    -----
    Requires SM120a / SM121a (block-scaled MXFP8 MMA + cp.async.bulk TMA).
    """
    _require_d_v_512(d_v)
    _check_last_dim_512(output, "output")
    model_type = _resolve_model_type(q.shape[-1], kv_scale_format)

    impl = get_sparse_mla_sm120_module()
    impl.paged_attention(
        q,
        kv_cache,
        indices,
        output,
        out_lse,
        sm_scale,
        d_v,
        model_type,
        topk_length,
        attn_sink,
        extra_kv_cache,
        extra_indices,
        extra_topk_length,
        mid_out,
        mid_lse,
    )


class _SparseMLAPagedAttentionRunner:
    """Sparse-MLA paged attention implementation runner for SM120.

    ``max_num_tokens`` and ``max_num_heads`` are optional upper bounds. When
    both are provided, the wrapper pre-allocates its LSE buffer. Otherwise, the
    buffer is allocated lazily and grown as needed. Decode split-K scratch may be
    supplied by the caller via ``run(mid_out=..., mid_lse=...)``; if omitted for
    a decode-sized call, this wrapper allocates temporary scratch.

    Parameters
    ----------
    max_num_tokens : Optional[int]
        Optional worst-case ``num_tokens`` the wrapper will accept. Used to size
        the pre-allocated ``out_lse`` buffer when paired with
        ``max_num_heads``.
    max_num_heads : Optional[int]
        Optional worst-case ``num_heads``.
    d_v : int
        Value head dim. ``512`` for DSV3_2 / DSV4.
    kv_scale_format : str
        Scale semantics for ``d_qk=576``. ``"auto"`` and ``"pow2_fp32"``
        select DSv3.2 power-of-2 FP32 inline scales; ``"arbitrary_fp32"``
        selects GLM-style arbitrary FP32 inline scales.
    device : Optional[torch.device]
        Allocation target. Defaults to the current CUDA device.

    Example
    -------
    >>> runner = _SparseMLAPagedAttentionRunner()
    >>> runner.run(q, kv_cache, indices, output, sm_scale=...)
    """

    @supported_compute_capability([120, 121])
    def __init__(
        self,
        max_num_tokens: Optional[int] = None,
        max_num_heads: Optional[int] = None,
        *,
        d_v: int = _D_V,
        kv_scale_format: str = "auto",
        device: Optional[torch.device] = None,
    ) -> None:
        if (max_num_tokens is None) != (max_num_heads is None):
            raise ValueError(
                "max_num_tokens and max_num_heads must be provided together"
            )
        if max_num_tokens is not None and max_num_tokens <= 0:
            raise ValueError(f"max_num_tokens must be > 0, got {max_num_tokens}")
        if max_num_heads is not None and (max_num_heads <= 0 or max_num_heads > 128):
            raise ValueError(f"max_num_heads must be in (0, 128], got {max_num_heads}")
        _require_d_v_512(d_v)
        self._kv_scale_format = _normalize_kv_scale_format(kv_scale_format)

        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self._device = torch.device(device)
        self._max_num_tokens = max_num_tokens
        self._max_num_heads = max_num_heads
        self._d_v = d_v

        self._out_lse: Optional[torch.Tensor] = None
        if max_num_tokens is not None and max_num_heads is not None:
            # Pre-allocated LSE buffer; sliced to actual shape on run(). Sized
            # for prefill worst case since prefill writes here too.
            self._out_lse = torch.empty(
                (max_num_tokens, max_num_heads),
                dtype=torch.float32,
                device=self._device,
            )

    def _get_out_lse(
        self,
        num_tokens: int,
        num_heads: int,
        out_lse: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if out_lse is not None:
            need = (num_tokens, num_heads)
            if out_lse.ndim != 2 or any(out_lse.size(d) < need[d] for d in range(2)):
                raise ValueError(
                    f"out_lse shape {tuple(out_lse.shape)} too small for required "
                    f"[num_tokens={num_tokens}, num_heads={num_heads}]"
                )
            if out_lse.dtype != torch.float32:
                raise ValueError(
                    f"out_lse must have dtype float32, got {out_lse.dtype}"
                )
            if out_lse.device != self._device:
                raise ValueError(
                    f"out_lse must be on device {self._device}, got {out_lse.device}"
                )
            return out_lse[:num_tokens, :num_heads]

        if self._out_lse is None or any(
            self._out_lse.size(d) < need
            for d, need in enumerate((num_tokens, num_heads))
        ):
            self._out_lse = torch.empty(
                (num_tokens, num_heads),
                dtype=torch.float32,
                device=self._device,
            )
        return self._out_lse[:num_tokens, :num_heads]

    def _maybe_allocate_decode_scratch(
        self,
        q: torch.Tensor,
        indices: torch.Tensor,
        extra_indices: Optional[torch.Tensor],
        mid_out: Optional[torch.Tensor],
        mid_lse: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if (mid_out is None) != (mid_lse is None):
            raise ValueError("mid_out and mid_lse must be passed together")
        if mid_out is not None:
            return mid_out, mid_lse

        num_tokens, num_heads, _ = q.shape
        if num_tokens > _DECODE_MAX_TOKENS:
            return None, None

        topk = indices.shape[-1]
        extra_topk = extra_indices.shape[-1] if extra_indices is not None else 0
        num_splits = (topk + _BI - 1) // _BI + (extra_topk + _BI - 1) // _BI
        mid_out = torch.empty(
            (num_tokens, num_heads, num_splits, self._d_v),
            dtype=torch.bfloat16,
            device=q.device,
        )
        mid_lse = torch.empty(
            (num_tokens, num_heads, num_splits),
            dtype=torch.float32,
            device=q.device,
        )
        return mid_out, mid_lse

    # The runner owns out_lse internally so no separate template is needed.
    def run(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        output: torch.Tensor,
        sm_scale: float,
        *,
        topk_length: Optional[torch.Tensor] = None,
        attn_sink: Optional[torch.Tensor] = None,
        extra_kv_cache: Optional[torch.Tensor] = None,
        extra_indices: Optional[torch.Tensor] = None,
        extra_topk_length: Optional[torch.Tensor] = None,
        out_lse: Optional[torch.Tensor] = None,
        mid_out: Optional[torch.Tensor] = None,
        mid_lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
    ) -> Optional[torch.Tensor]:
        """Run sparse-MLA paged attention.

        Mutates ``output`` and an LSE buffer in place. When ``out_lse`` is
        passed, that buffer is used; otherwise the wrapper uses an internal
        lazily-sized buffer. When ``return_lse=True``, returns a view into the
        LSE buffer sized to the actual ``num_tokens``; otherwise returns
        ``None``.

        Accepts ``q``/``output`` either as 3-D ``[num_tokens, num_heads, head_dim]``
        or as 4-D ``[num_tokens, 1, num_heads, head_dim]`` (some callers carry
        a singleton s_q dim); the 4-D form is squeezed in place. Calls that
        dispatch to a decode kernel must pass ``mid_out`` and ``mid_lse``.
        """
        if q.dim() == 4:
            if q.size(1) != 1:
                raise ValueError(
                    f"4-D q is only supported with s_q=1, got q.shape={tuple(q.shape)}"
                )
            q = q.squeeze(1)
        if output.dim() == 4:
            if output.size(1) != 1:
                raise ValueError(
                    f"4-D output is only supported with s_q=1, got "
                    f"output.shape={tuple(output.shape)}"
                )
            output = output.squeeze(1)
        num_tokens, num_heads, _ = q.shape
        if self._max_num_tokens is not None and num_tokens > self._max_num_tokens:
            raise ValueError(
                f"num_tokens ({num_tokens}) exceeds max_num_tokens "
                f"({self._max_num_tokens})"
            )
        if self._max_num_heads is not None and num_heads > self._max_num_heads:
            raise ValueError(
                f"num_heads ({num_heads}) exceeds max_num_heads ({self._max_num_heads})"
            )

        mid_out, mid_lse = self._maybe_allocate_decode_scratch(
            q, indices, extra_indices, mid_out, mid_lse
        )

        out_lse_view = self._get_out_lse(num_tokens, num_heads, out_lse)
        _sparse_mla_sm120_paged_attention(
            q,
            kv_cache,
            indices,
            output,
            out_lse_view,
            sm_scale,
            d_v=self._d_v,
            kv_scale_format=self._kv_scale_format,
            topk_length=topk_length,
            attn_sink=attn_sink,
            extra_kv_cache=extra_kv_cache,
            extra_indices=extra_indices,
            extra_topk_length=extra_topk_length,
            mid_out=mid_out,
            mid_lse=mid_lse,
        )
        return out_lse_view if return_lse else None


# Decode-DSv3.2 / DSv4: AutoTuner-driven chunks_per_block tuning. Optimal cpb
# is non-monotonic in (num_tokens, num_heads, topk), so expose it as a tactic
# and let AutoTuner cache the best value per (T_bucket, num_heads, topk).


class _SparseMlaDecodeDsv3Runner(TunableRunner):
    """Tactic = chunks_per_block ∈ [1, num_splits]; ≤0 → C++ heuristic."""

    def __init__(self, module, model_type: int) -> None:
        self.module = module
        self.model_type = int(model_type)

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.model_type))

    def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
        topk_length = inputs[6] if len(inputs) > 6 else None
        attn_sink = inputs[7] if len(inputs) > 7 else None
        return (
            self.model_type,
            topk_length is not None,
            attn_sink is not None,
        )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:
        indices = inputs[1]
        topk = indices.shape[-1]
        num_splits = (topk + _BI - 1) // _BI
        # tactic encodes chunks_per_block (1..num_splits).
        return list(range(1, num_splits + 1))

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
        do_preparation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        q, indices, mid_out, mid_lse, output, out_lse = inputs[:6]
        sm_scale = kwargs["sm_scale"]
        kv_cache = kwargs["kv_cache"]
        if len(inputs) > 6:
            topk_length, attn_sink = inputs[6:8]
        else:
            topk_length = kwargs.get("topk_length")
            attn_sink = kwargs.get("attn_sink")
        topk = indices.shape[-1]
        num_splits = (topk + _BI - 1) // _BI
        cpb_override = tactic if tactic > 0 else -1
        self.module.sparse_mla_sm120_decode_dsv3_2(
            q,
            kv_cache,
            indices,
            mid_out,
            mid_lse,
            output,
            out_lse,
            num_splits,
            sm_scale,
            topk_length,
            attn_sink,
            self.model_type,
            cpb_override,
        )
        return output


@functools.cache
def _get_sparse_mla_decode_dsv3_module(model_type: int):
    module = gen_sparse_mla_sm120_module().build_and_load()
    return SimpleNamespace(
        module=module,
        runner_cls=lambda: _SparseMlaDecodeDsv3Runner(module, model_type),
    )


@functools.cache
def _get_sparse_mla_decode_dsv4_module():
    module = gen_sparse_mla_sm120_module().build_and_load()

    class SparseMlaDecodeV3Runner(TunableRunner):
        """Tactic = chunks_per_block ∈ [1, num_splits]; ≤0 → C++ heuristic."""

        def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
            topk_length = inputs[6] if len(inputs) > 6 else None
            attn_sink = inputs[7] if len(inputs) > 7 else None
            extra_indices = inputs[8] if len(inputs) > 8 else None
            extra_topk_length = inputs[9] if len(inputs) > 9 else None
            extra_topk = extra_indices.shape[-1] if extra_indices is not None else 0
            return (
                topk_length is not None,
                attn_sink is not None,
                int(extra_topk),
                extra_topk_length is not None,
            )

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            indices = inputs[1]
            topk = indices.shape[-1]
            extra_indices = inputs[8] if len(inputs) > 8 else None
            extra_topk = extra_indices.shape[-1] if extra_indices is not None else 0
            num_splits = (topk + _BI - 1) // _BI + (extra_topk + _BI - 1) // _BI
            # tactic encodes chunks_per_block (1..num_splits).
            return list(range(1, num_splits + 1))

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            q, indices, mid_out, mid_lse, output, out_lse = inputs[:6]
            sm_scale = kwargs["sm_scale"]
            kv_cache = kwargs["kv_cache"]
            extra_kv_cache = kwargs.get("extra_kv_cache")
            if len(inputs) > 6:
                (
                    topk_length,
                    attn_sink,
                    extra_indices,
                    extra_topk_length,
                ) = inputs[6:10]
            else:
                topk_length = kwargs.get("topk_length")
                attn_sink = kwargs.get("attn_sink")
                extra_indices = kwargs.get("extra_indices")
                extra_topk_length = kwargs.get("extra_topk_length")
            topk = indices.shape[-1]  # 2D [T, topk] or 3D [T, 1, topk]
            extra_topk = extra_indices.shape[-1] if extra_indices is not None else 0
            num_splits = (topk + _BI - 1) // _BI + (extra_topk + _BI - 1) // _BI
            cpb_override = tactic if tactic > 0 else -1
            module.sparse_mla_sm120_decode_dsv4(
                q,
                kv_cache,
                indices,
                mid_out,
                mid_lse,
                output,
                out_lse,
                num_splits,
                sm_scale,
                topk_length,
                attn_sink,
                extra_kv_cache,
                extra_indices,
                extra_topk_length,
                cpb_override,
            )
            return output

    return SimpleNamespace(module=module, runner_cls=SparseMlaDecodeV3Runner)


def _decode_dsv4_num_token_buckets(*_args, **_kwargs):
    """Power-of-2-ish T buckets matching the contested decode shapes."""
    return (1, 4, 8, 16, 32, 64)


def _decode_dsv4_map_to_token_bucket(x):
    """Round T up to the next bucket boundary used by tuning."""
    buckets = (1, 4, 8, 16, 32, 64)
    for b in buckets:
        if x <= b:
            return b
    return buckets[-1]


def _decode_dsv4_init_q(shapes, dtype, device):
    """bf16 q ~ N(0, 0.1) clamped to [-1, 1] — matches the unit test distribution."""
    return (
        (torch.randn(shapes, device=device, dtype=torch.float32) / 10.0)
        .clamp(-1, 1)
        .to(dtype)
    )


def _decode_dsv4_init_indices(shapes, dtype, device):
    """int32 indices in a small safe range; assumes kv_cache has >=256 blocks.

    AutoTuner only profiles wall time, not correctness — random valid indices
    are sufficient. The cache built for the real call uses the ACTUAL indices.
    """
    return torch.randint(0, 256, shapes, dtype=dtype, device=device)


def _decode_dsv4_init_topk_length(shapes, dtype, device):
    """Placeholder full-active topk_length for autotune profiling."""
    return torch.full(shapes, 1 << 30, dtype=dtype, device=device)


def _decode_dsv4_inputs_pre_hook(inputs):
    """Pair (indices, topk_length) and (extra_indices, extra_topk_length)
    after per-tensor synthesis: cap lengths to their paired indices' top-k.
    """
    inputs = list(inputs)
    indices = inputs[1] if len(inputs) > 1 else None
    topk_length = inputs[6] if len(inputs) > 6 else None
    extra_indices = inputs[8] if len(inputs) > 8 else None
    extra_topk_length = inputs[9] if len(inputs) > 9 else None
    if topk_length is not None and indices is not None:
        inputs[6] = torch.full_like(topk_length, indices.shape[-1])
    if extra_topk_length is not None and extra_indices is not None:
        inputs[9] = torch.full_like(extra_topk_length, extra_indices.shape[-1])
    return inputs


@functools.cache
def _decode_dsv4_tuning_config() -> TuningConfig:
    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0, 1, 6, 8, 9),
                dim_idx=(0, 0, 0, 0, 0),
                gen_tuning_buckets=_decode_dsv4_num_token_buckets,
                map_to_tuning_buckets=_decode_dsv4_map_to_token_bucket,
                tensor_initializers=[
                    _decode_dsv4_init_q,
                    _decode_dsv4_init_indices,
                    _decode_dsv4_init_topk_length,
                    _decode_dsv4_init_indices,
                    _decode_dsv4_init_topk_length,
                ],
            ),
        ),
        inputs_pre_hook=_decode_dsv4_inputs_pre_hook,
        # Constrain T (dim 0) of all output/scratch tensors to q's T so the
        # autotuner's synthesised q propagates to mid_out (2), mid_lse (3),
        # output (4), out_lse (5). Without these constraints, the kernel
        # writes past the real tensors' T dim → IMA.
        constraint_specs=(
            ConstraintSpec(2, 0, lambda shapes: shapes[0][0]),  # mid_out
            ConstraintSpec(3, 0, lambda shapes: shapes[0][0]),  # mid_lse
            ConstraintSpec(4, 0, lambda shapes: shapes[0][0]),  # output
            ConstraintSpec(5, 0, lambda shapes: shapes[0][0]),  # out_lse
        ),
    )


@functools.cache
def _decode_dsv3_2_tuning_config() -> TuningConfig:
    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0, 1, 6),
                dim_idx=(0, 0, 0),
                gen_tuning_buckets=_decode_dsv4_num_token_buckets,
                map_to_tuning_buckets=_decode_dsv4_map_to_token_bucket,
                tensor_initializers=[
                    _decode_dsv4_init_q,
                    _decode_dsv4_init_indices,
                    _decode_dsv4_init_topk_length,
                ],
            ),
        ),
        inputs_pre_hook=_decode_dsv4_inputs_pre_hook,
        # Constrain T (dim 0) of all output/scratch tensors to q's T so the
        # autotuner's synthesised q propagates to mid_out (2), mid_lse (3),
        # output (4), out_lse (5).
        constraint_specs=(
            ConstraintSpec(2, 0, lambda shapes: shapes[0][0]),  # mid_out
            ConstraintSpec(3, 0, lambda shapes: shapes[0][0]),  # mid_lse
            ConstraintSpec(4, 0, lambda shapes: shapes[0][0]),  # output
            ConstraintSpec(5, 0, lambda shapes: shapes[0][0]),  # out_lse
        ),
    )


@functools.cache
def _decode_dsv3_2_runner_singleton(model_type: int):
    return _get_sparse_mla_decode_dsv3_module(model_type).runner_cls()


@functools.cache
def _decode_dsv4_runner_singleton():
    return _get_sparse_mla_decode_dsv4_module().runner_cls()


def _decode_dsv3_2_default_cache_path():
    """Default disk path for the decode-dsv3_2 AutoTuner cache."""
    import pathlib

    override = os.getenv("FLASHINFER_AUTOTUNE_DIR")
    if override:
        base = pathlib.Path(override)
    else:
        from ..jit.env import FLASHINFER_WORKSPACE_DIR

        base = FLASHINFER_WORKSPACE_DIR / "autotune"
    return base / "sparse_mla_sm120_decode_dsv3_2.json"


def _decode_dsv4_default_cache_path():
    """Default disk path for the decode-dsv4 AutoTuner cache.

    Override via ``FLASHINFER_AUTOTUNE_DIR`` env var or pass an explicit
    path to the generic :func:`flashinfer.autotune` context.
    """
    import pathlib

    override = os.getenv("FLASHINFER_AUTOTUNE_DIR")
    if override:
        base = pathlib.Path(override)
    else:
        from ..jit.env import FLASHINFER_WORKSPACE_DIR

        base = FLASHINFER_WORKSPACE_DIR / "autotune"
    return base / "sparse_mla_sm120_decode_dsv4.json"


_decode_dsv3_2_cache_mtime: float = -1.0
_decode_dsv4_cache_mtime: float = -1.0

# Per-process hot cache mapping shape signature → cpb tactic. Skips
# AutoTuner.choose_one on the steady-state path; entries are refreshed
# whenever a `with autotune(True):` session re-tunes the shape.
_decode_dsv3_2_hot_cache: dict = {}
_decode_dsv4_hot_cache: dict = {}


def _decode_dsv3_2_maybe_load_cache() -> None:
    """Mtime-gated lazy load of the default dsv3_2 decode AutoTuner cache."""
    global _decode_dsv3_2_cache_mtime
    path = _decode_dsv3_2_default_cache_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return
    if mtime <= _decode_dsv3_2_cache_mtime:
        return
    try:
        AutoTuner.get().load_configs(str(path))
        _decode_dsv3_2_cache_mtime = mtime
    except Exception:
        # Keep mtime unchanged so the next cold call retries.
        pass


def _decode_dsv4_maybe_load_cache() -> None:
    """Mtime-gated lazy load of the default disk cache.

    Silent on missing file or load failure (version mismatch, corrupt JSON):
    falls back to the C++ heuristic via AutoTuner's normal fallback path so
    a bad cache never blocks serving.
    """
    global _decode_dsv4_cache_mtime
    path = _decode_dsv4_default_cache_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return
    if mtime <= _decode_dsv4_cache_mtime:
        return
    try:
        AutoTuner.get().load_configs(str(path))
        _decode_dsv4_cache_mtime = mtime
    except Exception:
        # Keep mtime unchanged so the next cold call retries.
        pass


@supported_compute_capability([120, 121])
def sparse_mla_sm120_decode_dsv3_2(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    output: torch.Tensor,
    out_lse: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    model_type: int = _MODEL_TYPE_DSV3_2,
    chunks_per_block: Optional[int] = None,
) -> torch.Tensor:
    """Sparse-MLA paged decode (DSv3.2 / GLM-NSA kernel) on SM120.

    ``chunks_per_block`` follows the same contract as the DSv4 decode helper:
    explicit values bypass AutoTuner; otherwise a tuned/cache tactic is used
    when available, falling back to the C++ heuristic.
    """
    _check_last_dim_512(output, "output")
    _check_last_dim_512(mid_out, "mid_out")

    runner = _decode_dsv3_2_runner_singleton(int(model_type))
    inputs = [
        q,
        indices,
        mid_out,
        mid_lse,
        output,
        out_lse,
        topk_length,
        attn_sink,
    ]

    forward_kwargs = {
        "sm_scale": sm_scale,
        "kv_cache": kv_cache,
    }

    if chunks_per_block is not None:
        runner(
            inputs=inputs,
            tactic=int(chunks_per_block),
            **forward_kwargs,
        )
        return output

    tuner = AutoTuner.get()
    if not tuner.is_tuning_mode:
        T_bucket = _decode_dsv4_map_to_token_bucket(q.shape[0])
        num_splits = (indices.shape[-1] + _BI - 1) // _BI
        hot_key = (
            T_bucket,
            q.shape[1],
            indices.shape[-1],
            num_splits,
            runner.get_cache_key_extras(inputs),
        )
        cached_tactic = _decode_dsv3_2_hot_cache.get(hot_key)
        if cached_tactic is not None:
            runner(
                inputs=inputs,
                tactic=cached_tactic,
                **forward_kwargs,
            )
            return output

    _decode_dsv3_2_maybe_load_cache()
    chosen, tactic = tuner.choose_one(
        "sparse_mla_sm120_decode_dsv3_2",
        [runner],
        _decode_dsv3_2_tuning_config(),
        inputs,
        **forward_kwargs,
    )
    if int(tactic) > 0:
        T_bucket = _decode_dsv4_map_to_token_bucket(q.shape[0])
        num_splits = (indices.shape[-1] + _BI - 1) // _BI
        hot_key = (
            T_bucket,
            q.shape[1],
            indices.shape[-1],
            num_splits,
            runner.get_cache_key_extras(inputs),
        )
        _decode_dsv3_2_hot_cache[hot_key] = int(tactic)
    chosen(inputs=inputs, tactic=tactic, **forward_kwargs)
    return output


@supported_compute_capability([120, 121])
def sparse_mla_sm120_decode_dsv4(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    output: torch.Tensor,
    out_lse: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    chunks_per_block: Optional[int] = None,
) -> torch.Tensor:
    r"""Sparse-MLA paged decode (DSv4 standalone kernel) on SM120.

    The decode-dsv4 path is the split-K decode variant where each block handles
    ``chunks_per_block`` chunks of 64 candidates each. The wall-time-optimal
    ``chunks_per_block`` is shape-dependent and not well captured by a closed-
    form heuristic. This wrapper integrates flashinfer's :mod:`AutoTuner` to
    pick the per-shape best.

    Behaviour:

    - ``chunks_per_block`` explicitly given → use that value directly (no
      autotuning).
    - Otherwise, if a ``with autotune(...)`` context is active or a previous
      tuning run cached this shape → use the AutoTuner's choice.
    - Otherwise → fall back to the C++ closed-form heuristic.

    Parameters
    ----------
    q : torch.Tensor
        ``[T, num_heads, d_qk]`` bf16. ``d_qk == 512`` (DSV4 only).
    kv_cache : torch.Tensor
        Paged FP8 cache, shape ``[num_blocks, page_bytes]`` uint8.
    indices : torch.Tensor
        ``[T, topk]`` int32. ``topk`` must be one of {128, 512, 1024}; ``-1``
        marks invalid slots.
    mid_out : torch.Tensor
        Scratch, ``[T, num_heads, num_splits, d_v]`` bf16. ``num_splits =
        ceil(topk / 64) + ceil(extra_topk / 64)``.
    mid_lse : torch.Tensor
        Scratch, ``[T, num_heads, num_splits]`` float32.
    output : torch.Tensor
        In-place output, ``[T, num_heads, d_v]`` bf16.
    out_lse : torch.Tensor
        In-place log-sum-exp, ``[T, num_heads]`` float32.
    sm_scale : float
        Softmax scale.
    topk_length : Optional[torch.Tensor]
        Per-token effective top-k length, ``[T]`` int32.
    chunks_per_block : Optional[int]
        Explicit override. If ``None`` and no AutoTuner active, uses heuristic.

    Returns
    -------
    output : torch.Tensor
        The mutated output tensor (for chaining).
    """
    _check_last_dim_512(output, "output")
    _check_last_dim_512(mid_out, "mid_out")

    runner = _decode_dsv4_runner_singleton()
    inputs = [
        q,
        indices,
        mid_out,
        mid_lse,
        output,
        out_lse,
        topk_length,
        attn_sink,
        extra_indices,
        extra_topk_length,
    ]

    forward_kwargs = {
        "sm_scale": sm_scale,
        "kv_cache": kv_cache,
        "extra_kv_cache": extra_kv_cache,
    }

    if chunks_per_block is not None:
        # Explicit user override — skip AutoTuner entirely.
        runner(
            inputs=inputs,
            tactic=int(chunks_per_block),
            **forward_kwargs,
        )
        return output

    # Hot-cache fast path: skip AutoTuner.choose_one once a shape is resolved.
    # Tuning sessions always route through choose_one to collect data.
    tuner = AutoTuner.get()
    if not tuner.is_tuning_mode:
        T_bucket = _decode_dsv4_map_to_token_bucket(q.shape[0])
        extra_topk = extra_indices.shape[-1] if extra_indices is not None else 0
        num_splits = (indices.shape[-1] + _BI - 1) // _BI + (
            extra_topk + _BI - 1
        ) // _BI
        hot_key = (
            T_bucket,
            q.shape[1],
            indices.shape[-1],
            extra_topk,
            num_splits,
            runner.get_cache_key_extras(inputs),
        )
        cached_tactic = _decode_dsv4_hot_cache.get(hot_key)
        if cached_tactic is not None:
            runner(
                inputs=inputs,
                tactic=cached_tactic,
                **forward_kwargs,
            )
            return output

    # Cold path: lazy-load the disk cache once, then resolve via AutoTuner.
    _decode_dsv4_maybe_load_cache()
    chosen, tactic = tuner.choose_one(
        "sparse_mla_sm120_decode_dsv4",
        [runner],
        _decode_dsv4_tuning_config(),
        inputs,
        **forward_kwargs,
    )
    # Don't cache tactic=-1 (C++ heuristic fallback) so a later disk reload
    # can still take effect.
    if int(tactic) > 0:
        T_bucket = _decode_dsv4_map_to_token_bucket(q.shape[0])
        extra_topk = extra_indices.shape[-1] if extra_indices is not None else 0
        num_splits = (indices.shape[-1] + _BI - 1) // _BI + (
            extra_topk + _BI - 1
        ) // _BI
        hot_key = (
            T_bucket,
            q.shape[1],
            indices.shape[-1],
            extra_topk,
            num_splits,
            runner.get_cache_key_extras(inputs),
        )
        _decode_dsv4_hot_cache[hot_key] = int(tactic)
    chosen(inputs=inputs, tactic=tactic, **forward_kwargs)
    return output
