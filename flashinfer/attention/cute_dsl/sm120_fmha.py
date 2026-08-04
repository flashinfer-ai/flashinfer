# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Packed-contiguous PyTorch APIs for the SM120 FP8 FMHA kernel.

Public entry point
------------------
``sm120_fmha_fp8_ragged_prefill``: packed contiguous Q/K/V.
``sm120_fmha_fp8_paged_prefill``: packed Q with paged K/V pools.

Causal mask conventions
-----------------------
- Both paths use bottom-right alignment: query i attends to key j when
  ``j <= i + (kv_len - q_len)``.

Sequence lengths are runtime metadata. ``max_seqlen_q`` only sizes the launch
grid and is not part of the compiled kernel cache key.

Causal kernels use load-balanced scheduling by default. Set
``PRIMS_FMHA_DISABLE_BALANCED_SCHEDULING=1`` before planning or launching to
disable it.

Optional dependency
-------------------
Requires a compatible CuTe DSL package that provides ``cutlass.experimental``.
"""

import math
import os
from typing import Optional

import torch
from cutlass.cute.typing import Float32, Int32


def _cutlass_dtype(torch_dtype: torch.dtype):
    import cutlass

    return {
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
        torch.float8_e5m2: cutlass.Float8E5M2,
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
    }[torch_dtype]


def _check_sm120(device: torch.device) -> None:
    major, minor = torch.cuda.get_device_capability(device)
    if not (major == 12 and minor == 0):
        raise RuntimeError(
            f"SM120 FMHA kernel requires SM120 GPU (compute capability 12.0), "
            f"got {major}.{minor}"
        )


def _use_balanced_scheduler(is_causal: bool) -> bool:
    return is_causal and os.environ.get("PRIMS_FMHA_DISABLE_BALANCED_SCHEDULING") != "1"


def _validate_lse(q: torch.Tensor, lse: Optional[torch.Tensor]) -> None:
    """Validate the optional packed log2 LSE output tensor."""
    if lse is None:
        return
    expected_shape = (q.shape[0], q.shape[1])
    if tuple(lse.shape) != expected_shape:
        raise ValueError(
            f"lse must have shape {expected_shape}, got {tuple(lse.shape)}"
        )
    if lse.dtype != torch.float32:
        raise ValueError(f"lse must have dtype torch.float32, got {lse.dtype}")
    if lse.device != q.device:
        raise ValueError("lse and q must be on the same device")
    if not lse.is_contiguous():
        raise ValueError("lse must be contiguous")


def sm120_fmha_fp8_ragged_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    kv_tile: Optional[int] = None,
    q_tile: Optional[int] = None,
    lse: Optional[torch.Tensor] = None,
    v_scale: Optional[float] = None,
) -> None:
    """Run SM120 FP8 FMHA on packed contiguous ragged Q/K/V.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor, shape ``(total_q, Hq, D)``.
        dtype: ``float8_e4m3fn`` or ``float8_e5m2``.
    k : torch.Tensor
        Key tensor, shape ``(total_k, Hkv, D)``. Same dtype as ``q``.
    v : torch.Tensor
        Value tensor, shape ``(total_k, Hkv, D)``. Same dtype as ``q``.
    o : torch.Tensor
        Output tensor, shape ``(total_q, Hq, D)``, written in-place.
        dtype: ``float16`` or ``bfloat16``.
    cu_seqlens_q, cu_seqlens_k : torch.Tensor
        Runtime cumulative sequence offsets, both shape ``(B + 1,)`` int32.
    max_seqlen_q : int, optional
        Runtime launch-grid bound. Derived from ``cu_seqlens_q`` when omitted.
    is_causal : bool
        Bottom-right aligned causal mask. Causal kernels use balanced
        scheduling unless ``PRIMS_FMHA_DISABLE_BALANCED_SCHEDULING=1``.
    sm_scale : float, optional
        Softmax scale. Defaults to ``1 / sqrt(D)``.
    kv_tile : int, optional
        K/V tile size (64 or 128). Defaults to 128.
    q_tile : int, optional
        Q tile size (64 or 128). Defaults to 128.
    lse : torch.Tensor, optional
        Preallocated float32 log2 LSE output, shape ``(total_q, Hq)``.
    v_scale : float, optional
        Scalar multiplier folded into the normalized output. Defaults to 1.
    Raises
    ------
    RuntimeError
        If the GPU is not SM120 or the configuration is not supported.
    """
    from flashinfer.cute_dsl.attention.fmha.sm120 import (
        SM120FusedMultiHeadAttentionFP8ForwardTMA,
        compile_sm120_fmha_fp8_ragged_kernel,
    )

    assert q.ndim == 3, f"q must be (total_q, Hq, D), got {q.shape}"
    assert k.ndim == 3 and v.ndim == 3 and o.ndim == 3

    _check_sm120(q.device)
    _validate_lse(q, lse)

    _, Hq, D = q.shape
    _, Hkv, D_k = k.shape
    if D_k != D:
        raise ValueError(f"head_dim mismatch: q={D}, k={D_k}")
    if k.shape != v.shape:
        raise ValueError(
            f"k and v must have the same shape, got {k.shape} and {v.shape}"
        )
    # CUDA tensor maps cannot describe a zero-extent K/V tensor. Handle the
    # all-empty ragged batch before TMA descriptor construction; requests with
    # empty K/V inside a non-empty batch still use the kernel's empty-KV path.
    if k.shape[0] == 0:
        o.zero_()
        if lse is not None:
            lse.fill_(-float("inf"))
        return
    cu_seqlens_q_i32 = cu_seqlens_q.to(torch.int32)
    cu_seqlens_k_i32 = cu_seqlens_k.to(torch.int32)
    if max_seqlen_q is None:
        max_seqlen_q = int((cu_seqlens_q_i32[1:] - cu_seqlens_q_i32[:-1]).max().item())

    kv_tile = kv_tile or SM120FusedMultiHeadAttentionFP8ForwardTMA.SEQ_KV_TILES[0]
    q_tile = q_tile or SM120FusedMultiHeadAttentionFP8ForwardTMA.SEQ_Q_TILES[0]
    if (
        q_tile not in SM120FusedMultiHeadAttentionFP8ForwardTMA.SEQ_Q_TILES
        or kv_tile not in SM120FusedMultiHeadAttentionFP8ForwardTMA.SEQ_KV_TILES
        or D not in SM120FusedMultiHeadAttentionFP8ForwardTMA.SUPPORTED_HEAD_TILES
        or Hq % Hkv != 0
    ):
        raise RuntimeError(
            f"SM120 FP8 FMHA cannot implement config: "
            f"q={q.shape} k={k.shape} in={q.dtype} out={o.dtype} "
            f"kv_tile={kv_tile} q_tile={q_tile}"
        )

    kernel_fn = compile_sm120_fmha_fp8_ragged_kernel(
        in_dtype=q.dtype,
        out_dtype=o.dtype,
        num_qo_heads=Hq,
        num_kv_heads=Hkv,
        head_dim=D,
        is_causal=is_causal,
        kv_tile=kv_tile,
        q_tile=q_tile,
        device=q.device,
        with_lse=lse is not None,
        balanced_scheduler=_use_balanced_scheduler(is_causal),
    )

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    scale_log2 = Float32(sm_scale * math.log2(math.e))
    output_scale = Float32(1.0 if v_scale is None else float(v_scale))

    kernel_fn(
        q,
        k,
        v,
        o,
        lse,
        scale_log2,
        output_scale,
        None,
        cu_seqlens_q_i32,
        None,
        cu_seqlens_k_i32,
        Int32(max_seqlen_q),
    )


# =============================================================================
# Paged KV prefill — packed Q only
# =============================================================================


def sm120_fmha_fp8_paged_prefill(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    o: torch.Tensor,
    block_tables: torch.Tensor,
    seqlens_kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    max_seqlen_q: Optional[int] = None,
    kv_tile: Optional[int] = None,
    q_tile: Optional[int] = None,
    lse: Optional[torch.Tensor] = None,
    v_scale: Optional[float] = None,
) -> None:
    """Run SM120 FP8 FMHA prefill with paged K/V cache.

    Q/O use packed contiguous storage ``(total_q, Hq, D)``. K and V are
    stored in separate paged pools;
    ``block_tables`` maps each batch item's logical K/V pages to shared
    physical page IDs, matching the shared block-table paged-KV format.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor, shape ``(total_q, Hq, D)``.
        dtype: ``float8_e4m3fn`` or ``float8_e5m2``.
    k_pool : torch.Tensor
        NHD paged K pool, shape ``(num_pages, num_tokens_per_page, Hkv, D)``.
        Same dtype as ``q``. Every slot, including unused page padding, must
        contain a finite value.
    v_pool : torch.Tensor
        NHD paged V pool, shape ``(num_pages, num_tokens_per_page, Hkv, D)``.
        Same dtype and finite-value contract as ``k_pool``.
    o : torch.Tensor
        Output tensor, same shape as ``q``, written in-place.
        dtype: ``float16`` or ``bfloat16``.
    block_tables : torch.Tensor
        Shared K/V page index table, shape
        ``(B, max_num_pages_per_seq_kv)`` int32.
    seqlens_kv : torch.Tensor
        Actual K/V sequence length for each batch item, shape ``(B,)`` int32.
        Required for paged attention.
    cu_seqlens_q : torch.Tensor
        Runtime cumulative Q sequence offsets, shape ``(B+1,)`` int32.
    is_causal : bool
        Bottom-right aligned causal mask. Causal kernels use balanced
        scheduling unless ``PRIMS_FMHA_DISABLE_BALANCED_SCHEDULING=1``.
    sm_scale : float, optional
        Softmax scale. Defaults to ``1 / sqrt(D)``.
    max_seqlen_q : int, optional
        Runtime launch-grid bound. Derived from ``cu_seqlens_q`` if omitted.
    kv_tile : int, optional
        K/V tile size (64 or 128).  Auto-selected if ``None``.
    q_tile : int, optional
        Q tile size (64 or 128).  Auto-selected if ``None``.
    lse : torch.Tensor, optional
        Preallocated float32 log2 LSE output, shape ``(total_q, Hq)``.
    v_scale : float, optional
        Scalar multiplier folded into the normalized output. Defaults to 1.
    Raises
    ------
    RuntimeError
        If the GPU is not SM120 or the configuration is unsupported.
    ValueError
        If required arguments are missing or shapes are inconsistent.
    """
    from flashinfer.cute_dsl.attention.fmha.sm120 import (
        SM120FusedMultiHeadAttentionFP8ForwardTMA,
        compile_sm120_fmha_fp8_paged_kernel,
    )

    _check_sm120(q.device)
    _validate_lse(q, lse)

    assert q.ndim == 3, f"q must be packed (total_q, Hq, D), got {q.shape}"
    _, Hq, D = q.shape

    assert k_pool.ndim == 4, (
        f"k_pool must be (num_pages, page_size, Hkv, D), got {k_pool.shape}"
    )
    if tuple(v_pool.shape) != tuple(k_pool.shape):
        raise ValueError(
            f"v_pool must have the same NHD shape as k_pool, got "
            f"{v_pool.shape} and {k_pool.shape}"
        )
    if k_pool.dtype != q.dtype or v_pool.dtype != q.dtype:
        raise ValueError("q, k_pool, and v_pool must have the same FP8 dtype")
    if k_pool.device != q.device or v_pool.device != q.device:
        raise ValueError("q, k_pool, and v_pool must be on the same device")
    if not k_pool.is_contiguous() or not v_pool.is_contiguous():
        raise ValueError("NHD k_pool and v_pool must be contiguous")
    _, page_size, Hkv, D_k = k_pool.shape
    assert D_k == D, f"head_dim mismatch: q={D}, k_pool={D_k}"

    assert block_tables.ndim == 2, (
        f"block_tables must be (B, max_pages), got {block_tables.shape}"
    )
    B = block_tables.shape[0]

    in_ct = _cutlass_dtype(q.dtype)
    out_ct = _cutlass_dtype(o.dtype)

    seqlens_kv_i32 = seqlens_kv.to(torch.int32)
    cu_seqlens_q_i32 = cu_seqlens_q.to(torch.int32)
    if max_seqlen_q is None:
        max_seqlen_q = int((cu_seqlens_q_i32[1:] - cu_seqlens_q_i32[:-1]).max().item())
    kv_tile = kv_tile or SM120FusedMultiHeadAttentionFP8ForwardTMA.SEQ_KV_TILES[0]
    q_tile = q_tile or SM120FusedMultiHeadAttentionFP8ForwardTMA.SEQ_Q_TILES[0]

    # Only structural properties are checked here. Runtime lengths are bounded
    # by cu_seqlens_q, seqlens_kv, and block_tables capacity.
    if not SM120FusedMultiHeadAttentionFP8ForwardTMA.can_implement_paged(
        in_ct,
        out_ct,
        q_shape=(B, 1, Hq, D),
        k_shape=(B, 1, Hkv, D),
        num_tokens_per_page=page_size,
        kv_tile=kv_tile,
        q_tile=q_tile,
    ):
        raise RuntimeError(
            f"SM120 FP8 paged FMHA cannot implement config: "
            f"q={q.shape} k_pool={k_pool.shape} in={q.dtype} out={o.dtype} "
            f"page_size={page_size} kv_tile={kv_tile} q_tile={q_tile}"
        )

    kernel_fn = compile_sm120_fmha_fp8_paged_kernel(
        in_dtype=q.dtype,
        out_dtype=o.dtype,
        num_qo_heads=Hq,
        num_kv_heads=Hkv,
        head_dim=D,
        is_causal=is_causal,
        kv_tile=kv_tile,
        q_tile=q_tile,
        num_tokens_per_page=page_size,
        device=q.device,
        with_lse=lse is not None,
        balanced_scheduler=_use_balanced_scheduler(is_causal),
    )

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    scale_log2 = Float32(sm_scale * math.log2(math.e))
    output_scale = Float32(1.0 if v_scale is None else float(v_scale))

    block_tables_i32 = block_tables.to(torch.int32).contiguous()

    # TVM-FFI ABI (env stream):
    # kernel_fn(
    #     q, k_pool, v_pool, o, lse, scale_log2, output_scale,
    #     seqlens_kv, cu_seqlens_q, block_tables, max_seqlen_q
    # )
    kernel_fn(
        q,
        k_pool,
        v_pool,
        o,
        lse,
        scale_log2,
        output_scale,
        seqlens_kv_i32,
        cu_seqlens_q_i32,
        block_tables_i32,
        None,
        Int32(max_seqlen_q),
    )


class SM120PrimsBatchPrefillBackend:
    """Plan/run adapter used by the public batch-prefill wrappers.

    All host-derived metadata and kernel compilation happen in ``plan_*``.
    ``run_*`` only validates launch tensors and dispatches a cached kernel.
    """

    _FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
    _OUT_DTYPES = (torch.float16, torch.bfloat16)

    def __init__(self, device: torch.device) -> None:
        self.device = torch.device(device)
        self._mode: Optional[str] = None
        self._with_lse_compiled = False

    @staticmethod
    def _scalar_scale(name: str, value: Optional[float]) -> float:
        if value is None:
            return 1.0
        if isinstance(value, torch.Tensor):
            raise NotImplementedError(
                f"backend='cute-dsl-prims' only supports Python scalar {name}; "
                f"got tensor with shape {tuple(value.shape)}"
            )
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"backend='cute-dsl-prims' requires scalar {name}, got {value!r}"
            ) from exc

    def _validate_config(
        self,
        *,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        o_dtype: torch.dtype,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
    ) -> None:
        _check_sm120(self.device)
        if q_dtype not in self._FP8_DTYPES or kv_dtype != q_dtype:
            raise ValueError(
                "backend='cute-dsl-prims' requires Q/K/V to have the same "
                f"FP8 dtype (float8_e4m3fn or float8_e5m2); got q={q_dtype}, "
                f"kv={kv_dtype}"
            )
        if o_dtype not in self._OUT_DTYPES:
            raise ValueError(
                "backend='cute-dsl-prims' requires output dtype float16 or "
                f"bfloat16; got {o_dtype}"
            )
        if head_dim_qk != head_dim_vo or head_dim_qk not in (32, 64, 128, 256):
            raise ValueError(
                "backend='cute-dsl-prims' requires equal QK/VO head dimensions "
                f"in {{32, 64, 128, 256}}; got {head_dim_qk}/{head_dim_vo}"
            )
        if num_kv_heads <= 0 or num_qo_heads % num_kv_heads != 0:
            raise ValueError(
                "backend='cute-dsl-prims' requires num_qo_heads to be divisible "
                f"by num_kv_heads; got {num_qo_heads}/{num_kv_heads}"
            )

    def plan_ragged(
        self,
        *,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr_host: torch.Tensor,
        kv_indptr_host: torch.Tensor,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        o_dtype: torch.dtype,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
        causal: bool,
        sm_scale: Optional[float],
    ) -> None:
        from flashinfer.cute_dsl.attention.fmha.sm120 import (
            compile_sm120_fmha_fp8_ragged_kernel,
        )

        self._validate_config(
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            o_dtype=o_dtype,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
        )
        q_lens = qo_indptr_host[1:] - qo_indptr_host[:-1]
        kv_lens = kv_indptr_host[1:] - kv_indptr_host[:-1]
        if causal and bool(torch.any(q_lens > kv_lens)):
            raise ValueError(
                "backend='cute-dsl-prims' causal attention requires q_len <= "
                "kv_len for every request"
            )
        self._mode = "ragged"
        self._qo_indptr = qo_indptr
        self._kv_indptr = kv_indptr
        self._max_seqlen_q = int(q_lens.max().item())
        self._q_dtype = q_dtype
        self._kv_dtype = kv_dtype
        self._o_dtype = o_dtype
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim_qk
        self._causal = causal
        self._sm_scale = sm_scale
        self._page_size: Optional[int] = None
        compile_sm120_fmha_fp8_ragged_kernel(
            q_dtype,
            o_dtype,
            num_qo_heads,
            num_kv_heads,
            head_dim_qk,
            causal,
            128,
            128,
            self.device,
            False,
            _use_balanced_scheduler(causal),
        )
        self._with_lse_compiled = False

    def plan_paged(
        self,
        *,
        qo_indptr: torch.Tensor,
        qo_indptr_host: torch.Tensor,
        seqlens_kv: torch.Tensor,
        seqlens_kv_host: torch.Tensor,
        block_tables: torch.Tensor,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        o_dtype: torch.dtype,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
        page_size: int,
        causal: bool,
        sm_scale: Optional[float],
    ) -> None:
        from flashinfer.cute_dsl.attention.fmha.sm120 import (
            compile_sm120_fmha_fp8_paged_kernel,
        )

        self._validate_config(
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            o_dtype=o_dtype,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
        )
        if page_size not in (16, 32, 64, 128):
            raise ValueError(
                "backend='cute-dsl-prims' supports page_size in "
                f"{{16, 32, 64, 128}}; got {page_size}"
            )
        q_lens = qo_indptr_host[1:] - qo_indptr_host[:-1]
        if causal and bool(torch.any(q_lens > seqlens_kv_host)):
            raise ValueError(
                "backend='cute-dsl-prims' causal attention requires q_len <= "
                "kv_len for every request"
            )
        self._mode = "paged"
        self._qo_indptr = qo_indptr
        self._seqlens_kv = seqlens_kv
        self._block_tables = block_tables
        self._max_seqlen_q = int(q_lens.max().item())
        self._q_dtype = q_dtype
        self._kv_dtype = kv_dtype
        self._o_dtype = o_dtype
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim_qk
        self._causal = causal
        self._sm_scale = sm_scale
        self._page_size = page_size
        compile_sm120_fmha_fp8_paged_kernel(
            q_dtype,
            o_dtype,
            num_qo_heads,
            num_kv_heads,
            head_dim_qk,
            causal,
            128,
            128,
            page_size,
            self.device,
            False,
            _use_balanced_scheduler(causal),
        )
        self._with_lse_compiled = False

    def _validate_run(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor
    ) -> None:
        if self._mode is None:
            raise RuntimeError("backend='cute-dsl-prims' must be planned before run")
        for name, tensor, dtype in (
            ("q", q, self._q_dtype),
            ("k", k, self._kv_dtype),
            ("v", v, self._kv_dtype),
            ("out", out, self._o_dtype),
        ):
            if tensor.device != self.device or tensor.dtype != dtype:
                raise ValueError(
                    f"backend='cute-dsl-prims' expected {name} on {self.device} "
                    f"with dtype {dtype}; got {tensor.device}/{tensor.dtype}"
                )
            if not tensor.is_contiguous():
                raise ValueError(f"backend='cute-dsl-prims' requires contiguous {name}")

    def _ensure_lse_kernel(self) -> None:
        if self._with_lse_compiled:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "backend='cute-dsl-prims' LSE specialization was not compiled "
                "before CUDA Graph capture; call run_return_lse() once before capture"
            )
        from flashinfer.cute_dsl.attention.fmha.sm120 import (
            compile_sm120_fmha_fp8_paged_kernel,
            compile_sm120_fmha_fp8_ragged_kernel,
        )

        common = (
            self._q_dtype,
            self._o_dtype,
            self._num_qo_heads,
            self._num_kv_heads,
            self._head_dim,
            self._causal,
            128,
            128,
        )
        if self._mode == "ragged":
            compile_sm120_fmha_fp8_ragged_kernel(
                *common,
                self.device,
                True,
                _use_balanced_scheduler(self._causal),
            )
        else:
            compile_sm120_fmha_fp8_paged_kernel(
                *common,
                self._page_size,
                self.device,
                True,
                _use_balanced_scheduler(self._causal),
            )
        self._with_lse_compiled = True

    def run_ragged(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        *,
        lse: Optional[torch.Tensor],
        q_scale: Optional[float],
        k_scale: Optional[float],
        v_scale: Optional[float],
    ) -> None:
        self._validate_run(q, k, v, out)
        if self._mode != "ragged":
            raise RuntimeError("SM120 PRIMS backend was not planned for ragged KV")
        if lse is not None:
            self._ensure_lse_kernel()
        sm_scale = (
            self._sm_scale
            if self._sm_scale is not None
            else 1.0 / math.sqrt(self._head_dim)
        )
        sm_scale *= self._scalar_scale("q_scale", q_scale)
        sm_scale *= self._scalar_scale("k_scale", k_scale)
        scale_v = self._scalar_scale("v_scale", v_scale)
        sm120_fmha_fp8_ragged_prefill(
            q,
            k,
            v,
            out,
            self._qo_indptr,
            self._kv_indptr,
            max_seqlen_q=self._max_seqlen_q,
            is_causal=self._causal,
            sm_scale=sm_scale,
            v_scale=scale_v,
            lse=lse,
        )

    def run_paged(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        *,
        lse: Optional[torch.Tensor],
        q_scale: Optional[float],
        k_scale: Optional[float],
        v_scale: Optional[float],
    ) -> None:
        self._validate_run(q, k, v, out)
        if self._mode != "paged":
            raise RuntimeError("SM120 PRIMS backend was not planned for paged KV")
        if lse is not None:
            self._ensure_lse_kernel()
        sm_scale = (
            self._sm_scale
            if self._sm_scale is not None
            else 1.0 / math.sqrt(self._head_dim)
        )
        sm_scale *= self._scalar_scale("q_scale", q_scale)
        sm_scale *= self._scalar_scale("k_scale", k_scale)
        scale_v = self._scalar_scale("v_scale", v_scale)
        sm120_fmha_fp8_paged_prefill(
            q,
            k,
            v,
            out,
            self._block_tables,
            self._seqlens_kv,
            self._qo_indptr,
            is_causal=self._causal,
            sm_scale=sm_scale,
            v_scale=scale_v,
            max_seqlen_q=self._max_seqlen_q,
            lse=lse,
        )
