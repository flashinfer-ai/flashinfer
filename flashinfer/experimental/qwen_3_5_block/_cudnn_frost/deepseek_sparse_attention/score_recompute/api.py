# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""APIBase wrappers for the four DSA score-recompute operations.

Wraps the SM100 and SM90 CuTe-DSL score kernels. Each backend provides
indexer and attention score variants, giving four public classes.

Tile / SMEM dispatch logic and compile caching live in the backend interface
modules. This module adapts those entry points to the APIBase contract.
"""

from __future__ import annotations

from typing import Optional

import torch
import cuda.bindings.driver as cuda

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.runtime import (
    device_capability,
    maybe_contiguous,
    torch_stream_context as _torch_stream_context,
)

from flashinfer.experimental.qwen_3_5_block._cudnn_frost._torch_stream import contiguous_on_stream, copy_into_on_stream, record_streams
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.api_base import APIBase, TupleDict

from . import _interface_sm100 as _iface_sm100
from ._block_scaled import BLOCK_SCALED_PRECISION_NAMES

# ---------------------------------------------------------------------------
# Base helpers
# ---------------------------------------------------------------------------


def _check_score_arch(api: APIBase) -> None:
    major, _ = torch.cuda.get_device_capability()
    api._runtime_error_if(
        major != 9 and major < 10,
        f"{type(api).__name__} requires SM90 or SM100+ compute capability, found SM{major}",
    )


def _decline_unless_unit_innermost(api: APIBase, desc, name: str) -> None:
    """R5: the kernels read every operand natively; a non-unit innermost stride is declined, never copied."""
    api._not_implemented_error_if(
        desc.shape[-1] != 1 and desc.stride[-1] != 1,
        f"{type(api).__name__}: {name} must have a unit innermost stride (no .contiguous() copy on the execute path); got shape {desc.shape} strides {desc.stride}",
    )


def _decline_unless_contiguous(api: APIBase, desc, name: str) -> None:
    if desc is None:
        return
    api._not_implemented_error_if(
        not desc.is_contiguous(),
        f"{type(api).__name__}: {name} must be contiguous (no .contiguous() copy on the execute path); got shape {desc.shape} strides {desc.stride}",
    )


def _require_live_tensor(api: APIBase, t: Optional[torch.Tensor], desc, name: str, *, contiguous: bool) -> None:
    """Rule 1: execute re-validates the live tensor against its plan descriptor and raises, never converts."""
    cls = type(api).__name__
    api._value_error_if(t is None, f"{cls}: {name} is required at execute")
    api._value_error_if(t.dtype != desc.dtype, f"{cls}: {name} dtype {t.dtype} does not match the plan's {desc.dtype}")
    api._value_error_if(tuple(t.shape) != tuple(desc.shape), f"{cls}: {name} shape {tuple(t.shape)} does not match the plan's {tuple(desc.shape)}")
    api._value_error_if(t.device != desc.device, f"{cls}: {name} device {t.device} does not match the plan's {desc.device}")
    if contiguous:
        api._value_error_if(not t.is_contiguous(), f"{cls}: {name} must be contiguous, got strides {tuple(t.stride())}")
    else:
        api._value_error_if(t.shape[-1] != 1 and t.stride(-1) != 1, f"{cls}: {name} must have a unit innermost stride, got strides {tuple(t.stride())}")


def _require_live_optional(api: APIBase, t: Optional[torch.Tensor], desc, name: str) -> None:
    """An optional plan operand is present at execute exactly when it was declared (Rule 1)."""
    api._value_error_if(
        (t is None) != (desc is None),
        f"{type(api).__name__}: {name} {'is required by' if desc is not None else 'was not declared to'} this plan",
    )
    if desc is not None:
        _require_live_tensor(api, t, desc, name, contiguous=True)


def _require_int32_vector(api: APIBase, t: torch.Tensor, name: str, device) -> None:
    api._value_error_if(
        t.dtype != torch.int32 or t.ndim != 1 or not t.is_contiguous() or t.device != device,
        f"{type(api).__name__}: {name} must be a contiguous 1-D int32 tensor on {device}; got dtype {t.dtype} shape {tuple(t.shape)} strides {tuple(t.stride())} device {t.device}",
    )


def _wrapper_int32_contiguous(t: Optional[torch.Tensor], stream) -> Optional[torch.Tensor]:
    # Wrapper-surface convenience only; the classes decline/raise instead (R5). Call inside the
    # launch-stream context; the original is record_stream'ed there before the cast copies it (R1).
    if t is None or (t.dtype == torch.int32 and t.is_contiguous()):
        return t
    record_streams((t,), stream, t.device)
    return t.to(torch.int32).contiguous()


def _wrapper_stage_output(t: Optional[torch.Tensor], shape, dtype: torch.dtype, device) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Wrapper-surface: allocate an omitted output, or stage a non-contiguous caller output the class
    would decline; ``_wrapper_copy_back`` restores the caller's tensor identity. Call inside the launch-stream context."""
    if t is None:
        return torch.empty(shape, dtype=dtype, device=device), None
    if t.is_contiguous() or tuple(t.shape) != tuple(shape):
        return t, None  # bound directly, or left for the class to reject
    return torch.empty(shape, dtype=t.dtype, device=t.device), t


def _wrapper_copy_back(stream, staged: torch.Tensor, user: Optional[torch.Tensor]) -> torch.Tensor:
    if user is None:
        return staged
    copy_into_on_stream(user, staged, stream, user.device)  # R1 staging: the destination is recorded first
    return user


class _ScoreRecomputeBase(APIBase):
    """Common APIBase shell for score-recompute ops.

    ``_interface.py`` owns compile caches keyed per-kernel; ``check_support``
    and ``compile`` are thin markers that simply gate invocation. ``execute``
    delegates to the relevant ``_iface`` entry point.
    """

    def __init__(self):
        super().__init__()

    def compile(self) -> None:
        self._ensure_support_checked()
        # The score backends compile from the real execute-time tensors (their
        # plan-time-only compile caches key on dtypes/tile params/flags), so
        # this API validates eagerly and lets the backend compile on the first
        # execute(); compile() itself launches and allocates nothing.
        self._compiled_kernel = True


def _check_sparse_score_shapes(
    api: APIBase,
    q_desc,
    k_desc,
    aux_desc,
    topk_desc,
    out_desc,
    topk_length_desc,
    aux_name: str,
    qhead_per_kv_head: Optional[int],
) -> None:
    api._value_error_if(q_desc.ndim != 4, f"Q must be 4-D (B, S_q, H_q, D), got {q_desc.shape}")
    api._value_error_if(k_desc.ndim != 3, f"K must be 3-D (B, S_k, D) MQA, got {k_desc.shape}")
    api._value_error_if(aux_desc.ndim != 3, f"{aux_name} must be 3-D (B, S_q, H_q), got {aux_desc.shape}")
    api._value_error_if(topk_desc.ndim != 3, f"topk_indices must be 3-D (B, S_q, topk), got {topk_desc.shape}")
    api._value_error_if(out_desc.ndim != 3, f"out must be 3-D (B, S_q, topk), got {out_desc.shape}")

    b, s_q, h_q, d = q_desc.shape
    api._value_error_if(k_desc.shape[0] != b, f"K batch dim {k_desc.shape[0]} must match Q batch dim {b}")
    api._value_error_if(k_desc.shape[2] != d, f"K head dim {k_desc.shape[2]} must match Q head dim {d}")
    api._value_error_if(aux_desc.shape != (b, s_q, h_q), f"{aux_name} shape must be {(b, s_q, h_q)}, got {aux_desc.shape}")
    api._value_error_if(topk_desc.shape[:2] != (b, s_q), f"topk_indices leading dims must be {(b, s_q)}, got {topk_desc.shape[:2]}")
    api._value_error_if(out_desc.shape != topk_desc.shape, f"out shape must match topk_indices shape {topk_desc.shape}, got {out_desc.shape}")
    if qhead_per_kv_head is not None:
        api._value_error_if(qhead_per_kv_head != h_q, f"qhead_per_kv_head must equal H_q ({h_q}) for MQA sparse score, got {qhead_per_kv_head}")
    if topk_length_desc is not None:
        api._check_dtype(topk_length_desc, torch.int32, name="topk_length")
        api._value_error_if(topk_length_desc.shape != (b, s_q), f"topk_length must be shape {(b, s_q)}, got {topk_length_desc.shape}")
    for desc, name in ((q_desc, "Q"), (k_desc, "K"), (aux_desc, aux_name)):
        _decline_unless_unit_innermost(api, desc, name)
    for desc, name in ((topk_desc, "topk_indices"), (topk_length_desc, "topk_length"), (out_desc, "out")):
        _decline_unless_contiguous(api, desc, name)


def _check_dense_score_shapes(
    api: APIBase,
    q_desc,
    k_desc,
    aux_desc,
    out_desc,
    denom_desc,
    aux_name: str,
    is_thd: bool,
    qhead_per_kv_head: Optional[int],
) -> None:
    if is_thd:
        api._value_error_if(q_desc.ndim != 3, f"THD Q must be 3-D (total_q, H_q, D), got {q_desc.shape}")
        api._value_error_if(k_desc.ndim != 3, f"THD K must be 3-D (total_k, H_kv, D), got {k_desc.shape}")
        api._value_error_if(aux_desc.ndim != 2, f"THD {aux_name} must be 2-D (total_q, H_q), got {aux_desc.shape}")
        api._value_error_if(out_desc.ndim != 2, f"THD out must be 2-D (total_q, max_seqlen_k), got {out_desc.shape}")
        api._value_error_if(denom_desc.ndim != 1, f"THD denom_out must be 1-D (total_q,), got {denom_desc.shape}")
        total_q, h_q, d = q_desc.shape
        _, h_kv, d_k = k_desc.shape
        api._value_error_if(aux_desc.shape != (total_q, h_q), f"THD {aux_name} shape must be {(total_q, h_q)}, got {aux_desc.shape}")
        api._value_error_if(out_desc.shape[0] != total_q, f"THD out first dim must be total_q ({total_q}), got {out_desc.shape[0]}")
        api._value_error_if(denom_desc.shape != (total_q,), f"THD denom_out shape must be {(total_q,)}, got {denom_desc.shape}")
    else:
        api._value_error_if(q_desc.ndim != 4, f"BSHD Q must be 4-D (B, S_q, H_q, D), got {q_desc.shape}")
        api._value_error_if(k_desc.ndim != 4, f"BSHD K must be 4-D (B, S_k, H_kv, D), got {k_desc.shape}")
        api._value_error_if(aux_desc.ndim != 3, f"BSHD {aux_name} must be 3-D (B, S_q, H_q), got {aux_desc.shape}")
        api._value_error_if(out_desc.ndim != 3, f"BSHD out must be 3-D (B, S_q, S_k), got {out_desc.shape}")
        api._value_error_if(denom_desc.ndim != 2, f"BSHD denom_out must be 2-D (B, S_q), got {denom_desc.shape}")
        b, s_q, h_q, d = q_desc.shape
        b_k, s_k, h_kv, d_k = k_desc.shape
        api._value_error_if(b_k != b, f"K batch dim {b_k} must match Q batch dim {b}")
        api._value_error_if(aux_desc.shape != (b, s_q, h_q), f"{aux_name} shape must be {(b, s_q, h_q)}, got {aux_desc.shape}")
        api._value_error_if(out_desc.shape != (b, s_q, s_k), f"out shape must be {(b, s_q, s_k)}, got {out_desc.shape}")
        api._value_error_if(denom_desc.shape != (b, s_q), f"denom_out shape must be {(b, s_q)}, got {denom_desc.shape}")
    api._value_error_if(d_k != d, f"K head dim {d_k} must match Q head dim {d}")
    api._value_error_if(h_kv <= 0 or h_q % h_kv != 0, f"H_q ({h_q}) must be divisible by H_kv ({h_kv})")
    if qhead_per_kv_head is not None:
        api._value_error_if(h_q != h_kv * qhead_per_kv_head, f"H_q ({h_q}) must equal H_kv ({h_kv}) * qhead_per_kv_head ({qhead_per_kv_head})")
    for desc, name in ((q_desc, "Q"), (k_desc, "K"), (aux_desc, aux_name)):
        _decline_unless_unit_innermost(api, desc, name)
    for desc, name in ((out_desc, "out"), (denom_desc, "denom_out")):
        _decline_unless_contiguous(api, desc, name)


# ---------------------------------------------------------------------------
# Sparse indexer score
# ---------------------------------------------------------------------------


class SparseIndexerScoreRecompute(_ScoreRecomputeBase):
    """Sparse indexer score recompute.

    Computes per-query ``softmax( sum_h ReLU(Q_h · K_topk^T) * W_h )`` over
    the top-K KV positions given by ``topk_indices``.
    """

    def __init__(
        self,
        sample_q_indexer: torch.Tensor,  # (B, S_q, H_q, D) BF16
        sample_k_indexer: torch.Tensor,  # (B, S_k, D) BF16 (MQA)
        sample_weights: torch.Tensor,  # (B, S_q, H_q) BF16
        sample_topk_indices: torch.Tensor,  # (B, S_q, topk) INT32
        sample_out: torch.Tensor,  # (B, S_q, topk) FP32
        sample_topk_length: Optional[torch.Tensor] = None,  # (B, S_q) INT32
        qhead_per_kv_head: Optional[int] = None,
        topk_indices_global: bool = False,
    ):
        super().__init__()
        self.q_desc = self._make_tensor_desc(sample_q_indexer, name="sample_q_indexer")
        self.k_desc = self._make_tensor_desc(sample_k_indexer, name="sample_k_indexer")
        self.w_desc = self._make_tensor_desc(sample_weights, name="sample_weights")
        self.topk_desc = self._make_tensor_desc(sample_topk_indices, name="sample_topk_indices")
        self.out_desc = self._make_tensor_desc(sample_out, name="sample_out")
        self.topk_length_desc = self._make_tensor_desc(sample_topk_length, name="sample_topk_length")
        self.qhead_per_kv_head = qhead_per_kv_head
        self.topk_indices_global = bool(topk_indices_global)

    def check_support(self) -> bool:
        _check_score_arch(self)
        self._check_dtype(self.q_desc, torch.bfloat16, name="Q")
        self._check_dtype(self.k_desc, torch.bfloat16, name="K")
        self._check_dtype(self.w_desc, torch.bfloat16, name="W")
        self._check_dtype(self.topk_desc, torch.int32, name="topk_indices")
        self._check_dtype(self.out_desc, torch.float32, name="out")
        _check_sparse_score_shapes(
            self,
            self.q_desc,
            self.k_desc,
            self.w_desc,
            self.topk_desc,
            self.out_desc,
            self.topk_length_desc,
            "W",
            self.qhead_per_kv_head,
        )
        self._is_supported = True
        return True

    def execute(
        self,
        q_indexer: torch.Tensor,
        k_indexer: torch.Tensor,
        weights: torch.Tensor,
        topk_indices: torch.Tensor,
        out: torch.Tensor,
        topk_length: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> torch.Tensor:
        _require_live_tensor(self, q_indexer, self.q_desc, "q_indexer", contiguous=False)
        _require_live_tensor(self, k_indexer, self.k_desc, "k_indexer", contiguous=False)
        _require_live_tensor(self, weights, self.w_desc, "weights", contiguous=False)
        _require_live_tensor(self, topk_indices, self.topk_desc, "topk_indices", contiguous=True)
        _require_live_tensor(self, out, self.out_desc, "out", contiguous=True)
        _require_live_optional(self, topk_length, self.topk_length_desc, "topk_length")
        major, _ = device_capability()
        if major == 9:
            from . import _interface_sm90 as _iface_sm90

            return _iface_sm90.sparse_indexer_score_recompute(
                q_indexer,
                k_indexer,
                weights,
                topk_indices,
                out,
                topk_length=topk_length,
                topk_indices_global=self.topk_indices_global,
                current_stream=current_stream,
            )
        return _iface_sm100.sparse_indexer_score_recompute(
            q_indexer,
            k_indexer,
            weights,
            topk_indices,
            qhead_per_kv_head=self.qhead_per_kv_head,
            topk_indices_global=self.topk_indices_global,
            out=out,
            topk_length=topk_length,
            current_stream=current_stream,
        )


_cache_of_SparseIndexerScoreRecomputeObjects: dict = {}


def sparse_indexer_score_recompute_wrapper(
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights: torch.Tensor,
    topk_indices: torch.Tensor,
    qhead_per_kv_head: Optional[int] = None,
    topk_length: Optional[torch.Tensor] = None,
    topk_indices_global: bool = False,
    out: Optional[torch.Tensor] = None,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """High-level wrapper. Returns ``{'predict': (B, S_q, topk) FP32}``.

    ``topk_indices`` are per-batch local KV ids by default. Set
    ``topk_indices_global=True`` when passing ids encoded as
    ``batch_idx * S_k + local_idx``.

    Wrapper-surface convenience (the class declines instead): inputs with a
    non-unit innermost stride are copied contiguous, ``topk_indices`` /
    ``topk_length`` are cast to contiguous int32, and ``out`` is allocated when
    omitted. All of it runs on ``stream``.

    ``topk_length`` does not replace the ``-1`` padding: slots at and past
    ``topk_length[row]`` must still hold ``-1`` (docs/operations/sparse_attention/dsa.md,
    "Compact vs. non-compact sparse indices").
    """
    with _torch_stream_context(stream):
        q_indexer, k_indexer, weights = (maybe_contiguous(t, stream) for t in (q_indexer, k_indexer, weights))
        topk_indices = _wrapper_int32_contiguous(topk_indices, stream)
        topk_length = _wrapper_int32_contiguous(topk_length, stream)
        out, out_user = _wrapper_stage_output(out, (q_indexer.shape[0], q_indexer.shape[1], topk_indices.shape[-1]), torch.float32, q_indexer.device)
    key = (
        q_indexer.device,
        q_indexer.dtype,
        q_indexer.shape,
        k_indexer.shape,
        weights.shape,
        topk_indices.shape,
        q_indexer.stride(),
        k_indexer.stride(),
        weights.stride(),
        topk_indices.stride(),
        qhead_per_kv_head,
        topk_length is not None,
        bool(topk_indices_global),
    )
    obj = _cache_of_SparseIndexerScoreRecomputeObjects.get(key)
    if obj is None:
        obj = SparseIndexerScoreRecompute(
            sample_q_indexer=q_indexer,
            sample_k_indexer=k_indexer,
            sample_weights=weights,
            sample_topk_indices=topk_indices,
            sample_out=out,
            sample_topk_length=topk_length,
            qhead_per_kv_head=qhead_per_kv_head,
            topk_indices_global=topk_indices_global,
        )
        assert obj.check_support()
        obj.compile()
        _cache_of_SparseIndexerScoreRecomputeObjects[key] = obj

    predict = obj.execute(
        q_indexer,
        k_indexer,
        weights,
        topk_indices,
        out,
        topk_length=topk_length,
        current_stream=stream,
    )
    return TupleDict(predict=_wrapper_copy_back(stream, predict, out_user))


# ---------------------------------------------------------------------------
# Sparse attention score
# ---------------------------------------------------------------------------


class SparseAttnScoreRecompute(_ScoreRecomputeBase):
    """Sparse attention score recompute.

    Recovers per-head softmax from ``LSE``, sums across heads, and L1-normalizes
    over the top-K KV positions:
    ``target[b,q,i] = (sum_h exp(Q_h·K_topk^T·scale - LSE_h)) / sum_i(...)``.
    """

    def __init__(
        self,
        sample_q_attn: torch.Tensor,  # (B, S_q, H_q, D) BF16
        sample_k_attn: torch.Tensor,  # (B, S_k, D) BF16
        sample_lse: torch.Tensor,  # (B, S_q, H_q) FP32
        sample_topk_indices: torch.Tensor,  # (B, S_q, topk) INT32
        sample_out: torch.Tensor,  # (B, S_q, topk) FP32
        softmax_scale: float,
        sample_topk_length: Optional[torch.Tensor] = None,
        qhead_per_kv_head: Optional[int] = None,
        topk_indices_global: bool = False,
    ):
        super().__init__()
        self.q_desc = self._make_tensor_desc(sample_q_attn, name="sample_q_attn")
        self.k_desc = self._make_tensor_desc(sample_k_attn, name="sample_k_attn")
        self.lse_desc = self._make_tensor_desc(sample_lse, name="sample_lse")
        self.topk_desc = self._make_tensor_desc(sample_topk_indices, name="sample_topk_indices")
        self.out_desc = self._make_tensor_desc(sample_out, name="sample_out")
        self.topk_length_desc = self._make_tensor_desc(sample_topk_length, name="sample_topk_length")
        self.softmax_scale = float(softmax_scale)
        self.qhead_per_kv_head = qhead_per_kv_head
        self.topk_indices_global = bool(topk_indices_global)

    def check_support(self) -> bool:
        _check_score_arch(self)
        self._check_dtype(self.q_desc, torch.bfloat16, name="Q")
        self._check_dtype(self.k_desc, torch.bfloat16, name="K")
        self._check_dtype(self.lse_desc, torch.float32, name="LSE")
        self._check_dtype(self.topk_desc, torch.int32, name="topk_indices")
        self._check_dtype(self.out_desc, torch.float32, name="out")
        _check_sparse_score_shapes(
            self,
            self.q_desc,
            self.k_desc,
            self.lse_desc,
            self.topk_desc,
            self.out_desc,
            self.topk_length_desc,
            "LSE",
            self.qhead_per_kv_head,
        )
        self._is_supported = True
        return True

    def execute(
        self,
        q_attn: torch.Tensor,
        k_attn: torch.Tensor,
        lse: torch.Tensor,
        topk_indices: torch.Tensor,
        out: torch.Tensor,
        topk_length: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> torch.Tensor:
        _require_live_tensor(self, q_attn, self.q_desc, "q_attn", contiguous=False)
        _require_live_tensor(self, k_attn, self.k_desc, "k_attn", contiguous=False)
        _require_live_tensor(self, lse, self.lse_desc, "lse", contiguous=False)
        _require_live_tensor(self, topk_indices, self.topk_desc, "topk_indices", contiguous=True)
        _require_live_tensor(self, out, self.out_desc, "out", contiguous=True)
        _require_live_optional(self, topk_length, self.topk_length_desc, "topk_length")
        scale = self.softmax_scale if softmax_scale is None else float(softmax_scale)
        major, _ = device_capability()
        if major == 9:
            from . import _interface_sm90 as _iface_sm90

            return _iface_sm90.sparse_attn_score_recompute(
                q_attn,
                k_attn,
                lse,
                topk_indices,
                scale,
                out,
                topk_length=topk_length,
                topk_indices_global=self.topk_indices_global,
                current_stream=current_stream,
            )
        return _iface_sm100.sparse_attn_score_recompute(
            q_attn,
            k_attn,
            lse,
            topk_indices,
            scale,
            qhead_per_kv_head=self.qhead_per_kv_head,
            topk_indices_global=self.topk_indices_global,
            out=out,
            topk_length=topk_length,
            current_stream=current_stream,
        )


_cache_of_SparseAttnScoreRecomputeObjects: dict = {}


def sparse_attn_score_recompute_wrapper(
    q_attn: torch.Tensor,
    k_attn: torch.Tensor,
    lse: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    qhead_per_kv_head: Optional[int] = None,
    topk_length: Optional[torch.Tensor] = None,
    topk_indices_global: bool = False,
    out: Optional[torch.Tensor] = None,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """High-level wrapper. Returns ``{'target': (B, S_q, topk) FP32}``.

    ``topk_indices`` are per-batch local KV ids by default. Set
    ``topk_indices_global=True`` when passing ids encoded as
    ``batch_idx * S_k + local_idx``.

    Wrapper-surface convenience (the class declines instead): inputs with a
    non-unit innermost stride are copied contiguous, ``topk_indices`` /
    ``topk_length`` are cast to contiguous int32, and ``out`` is allocated when
    omitted. All of it runs on ``stream``.

    ``topk_length`` does not replace the ``-1`` padding: slots at and past
    ``topk_length[row]`` must still hold ``-1``, because the SM100 kernel may
    ignore ``topk_length`` and skip slots by index value (issue #1549; see
    docs/operations/sparse_attention/dsa.md, "Compact vs. non-compact sparse indices").
    """
    with _torch_stream_context(stream):
        q_attn, k_attn, lse = (maybe_contiguous(t, stream) for t in (q_attn, k_attn, lse))
        topk_indices = _wrapper_int32_contiguous(topk_indices, stream)
        topk_length = _wrapper_int32_contiguous(topk_length, stream)
        out, out_user = _wrapper_stage_output(out, (q_attn.shape[0], q_attn.shape[1], topk_indices.shape[-1]), torch.float32, q_attn.device)
    key = (
        q_attn.device,
        q_attn.dtype,
        q_attn.shape,
        k_attn.shape,
        lse.shape,
        topk_indices.shape,
        q_attn.stride(),
        k_attn.stride(),
        lse.stride(),
        topk_indices.stride(),
        qhead_per_kv_head,
        topk_length is not None,
        bool(topk_indices_global),
        float(softmax_scale),
    )
    obj = _cache_of_SparseAttnScoreRecomputeObjects.get(key)
    if obj is None:
        obj = SparseAttnScoreRecompute(
            sample_q_attn=q_attn,
            sample_k_attn=k_attn,
            sample_lse=lse,
            sample_topk_indices=topk_indices,
            sample_out=out,
            softmax_scale=softmax_scale,
            sample_topk_length=topk_length,
            qhead_per_kv_head=qhead_per_kv_head,
            topk_indices_global=topk_indices_global,
        )
        assert obj.check_support()
        obj.compile()
        _cache_of_SparseAttnScoreRecomputeObjects[key] = obj

    target = obj.execute(
        q_attn,
        k_attn,
        lse,
        topk_indices,
        out,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        current_stream=stream,
    )
    return TupleDict(target=_wrapper_copy_back(stream, target, out_user))


# ---------------------------------------------------------------------------
# Dense indexer score
# ---------------------------------------------------------------------------


def _uses_thd(cu_seqlens_q: Optional[torch.Tensor], cu_seqlens_k: Optional[torch.Tensor]) -> bool:
    if (cu_seqlens_q is None) != (cu_seqlens_k is None):
        raise ValueError("THD dense score requires both cu_seqlens_q and cu_seqlens_k")
    return cu_seqlens_q is not None


def _max_from_cu_seqlens(cu_seqlens: torch.Tensor, name: str) -> int:
    # Wrapper-surface convenience only (documented on the wrappers): one blocking
    # D2H read per call. The classes take the envelope as plan-time host ints.
    if cu_seqlens.ndim != 1:
        raise ValueError(f"{name} must be a 1D cumulative sequence length tensor")
    if cu_seqlens.numel() <= 1:
        return 0
    return int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())


def _dense_sample_shapes(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
) -> tuple[bool, int, int, tuple[int, ...], tuple[int, ...]]:
    is_thd = _uses_thd(cu_seqlens_q, cu_seqlens_k)
    if is_thd:
        if q.ndim != 3 or k.ndim != 3:
            raise ValueError("THD dense score expects q/k with shape (total, heads, dim)")
        max_q = int(max_seqlen_q) if max_seqlen_q is not None else _max_from_cu_seqlens(cu_seqlens_q, "cu_seqlens_q")
        max_k = int(max_seqlen_k) if max_seqlen_k is not None else _max_from_cu_seqlens(cu_seqlens_k, "cu_seqlens_k")
        return True, max_q, max_k, (q.shape[0], max_k), (q.shape[0],)

    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("BSHD dense score expects q/k with shape (B, S, H, D)")
    return False, q.shape[1], k.shape[1], (q.shape[0], q.shape[1], k.shape[1]), (q.shape[0], q.shape[1])


_SM90_THD_DECLINE = (
    "{name}: THD (cu_seqlens_q/cu_seqlens_k) dense score recompute is not plan-eligible on SM90 -- the SM90 kernel is "
    "BSHD-native and serving THD needs a host copy of cu_seqlens per call (Rule 3/8). "
    "Use BSHD on SM90, THD on SM100+, or the eager *_wrapper, which adapts per batch on SM90."
)


def _dense_thd_plan_envelope(is_thd: bool, max_seqlen_q: Optional[int], max_seqlen_k: Optional[int], out_desc) -> tuple[Optional[int], Optional[int]]:
    """Plan-time launch envelope as host ints; ``max_seqlen_k`` defaults to ``sample_out.shape[1]`` for THD."""
    q_env = None if max_seqlen_q is None else int(max_seqlen_q)
    k_env = None if max_seqlen_k is None else int(max_seqlen_k)
    if is_thd and k_env is None and out_desc is not None and out_desc.ndim == 2:
        k_env = int(out_desc.shape[1])
    return q_env, k_env


def _check_dense_thd_plan(api: APIBase) -> None:
    """THD plan checks shared by the two dense score classes.

    The envelope is a required host int, never read back from ``cu_seqlens``
    (Rule 3/8). SM90 is declined: its kernel is BSHD-native and the former
    adapter drove one launch per batch from a host copy of ``cu_seqlens``.
    """
    name = type(api).__name__
    if not api.is_thd:
        api._value_error_if(
            api.max_seqlen_q is not None or api.max_seqlen_k is not None,
            f"{name}: max_seqlen_q/max_seqlen_k are THD-only plan parameters; a BSHD plan takes its envelope from the sample shapes",
        )
        return
    api._value_error_if(
        api.max_seqlen_q is None,
        f"{name}: THD dense score requires max_seqlen_q at plan time (the launch envelope); pass it to __init__ -- "
        "it is not read back from cu_seqlens_q (Rule 3/8)",
    )
    out_k = int(api.out_desc.shape[1])
    api._value_error_if(
        api.max_seqlen_k != out_k,
        f"{name}: max_seqlen_k ({api.max_seqlen_k}) must equal sample_out.shape[1] ({out_k})",
    )
    major, _ = torch.cuda.get_device_capability()
    api._not_implemented_error_if(major == 9, _SM90_THD_DECLINE.format(name=name))


def _dense_thd_execute_envelope(
    api: APIBase,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
) -> tuple[Optional[int], Optional[int]]:
    """Resolve the execute-time envelope from the plan; never from a device read."""
    name = type(api).__name__
    has_cu = cu_seqlens_q is not None or cu_seqlens_k is not None
    api._value_error_if(has_cu and not api.is_thd, f"{name}: cu_seqlens_q/cu_seqlens_k passed to a plan built with is_thd=False")
    if not api.is_thd:
        return max_seqlen_q, max_seqlen_k
    api._value_error_if(cu_seqlens_q is None or cu_seqlens_k is None, f"{name}: a THD plan requires both cu_seqlens_q and cu_seqlens_k at execute")
    q_env = api.max_seqlen_q if max_seqlen_q is None else int(max_seqlen_q)
    k_env = api.max_seqlen_k if max_seqlen_k is None else int(max_seqlen_k)
    api._value_error_if(
        q_env != api.max_seqlen_q or k_env != api.max_seqlen_k,
        f"{name}: execute max_seqlen_q/max_seqlen_k ({q_env}, {k_env}) must match the plan's ({api.max_seqlen_q}, {api.max_seqlen_k}); "
        "build a new plan for a different envelope",
    )
    return q_env, k_env


def _require_dense_live(api: APIBase, q, k, aux, aux_desc, aux_name: str, out, denom_out, cu_seqlens_q, cu_seqlens_k, q_causal_offsets) -> None:
    _require_live_tensor(api, q, api.q_desc, "q", contiguous=False)
    _require_live_tensor(api, k, api.k_desc, "k", contiguous=False)
    _require_live_tensor(api, aux, aux_desc, aux_name, contiguous=False)
    _require_live_tensor(api, out, api.out_desc, "out", contiguous=True)
    _require_live_tensor(api, denom_out, api.denom_desc, "denom_out", contiguous=True)
    if api.is_thd:
        _require_int32_vector(api, cu_seqlens_q, "cu_seqlens_q", api.q_desc.device)
        _require_int32_vector(api, cu_seqlens_k, "cu_seqlens_k", api.q_desc.device)
        api._value_error_if(
            cu_seqlens_q.shape != cu_seqlens_k.shape,
            f"{type(api).__name__}: cu_seqlens_q/cu_seqlens_k shapes differ ({tuple(cu_seqlens_q.shape)} vs {tuple(cu_seqlens_k.shape)})",
        )
    if q_causal_offsets is not None:
        _require_int32_vector(api, q_causal_offsets, "q_causal_offsets", api.q_desc.device)


def _dense_wrapper_prepare(stream, q, k, aux, out, denom_out, out_shape, denom_shape, cu_seqlens_q, cu_seqlens_k, q_causal_offsets, q_scale, k_scale):
    """Wrapper-surface convenience copies/allocations, all on the launch stream (R1); the classes decline instead."""
    with _torch_stream_context(stream):
        q, k, aux, q_scale, k_scale = (maybe_contiguous(t, stream) for t in (q, k, aux, q_scale, k_scale))
        cu_seqlens_q, cu_seqlens_k = (_wrapper_int32_contiguous(t, stream) for t in (cu_seqlens_q, cu_seqlens_k))
        q_causal_offsets = contiguous_on_stream(q_causal_offsets, stream, q.device)
        out, out_user = _wrapper_stage_output(out, out_shape, torch.float32, q.device)
        denom_out, denom_user = _wrapper_stage_output(denom_out, denom_shape, torch.float32, q.device)
    return q, k, aux, out, denom_out, cu_seqlens_q, cu_seqlens_k, q_causal_offsets, q_scale, k_scale, out_user, denom_user


def _dense_wrapper_result(stream, out: torch.Tensor, denom: torch.Tensor, out_user, denom_user) -> TupleDict:
    return TupleDict(out=_wrapper_copy_back(stream, out, out_user), denom=_wrapper_copy_back(stream, denom, denom_user))


def _dense_sm90_thd_wrapper_path(is_thd: bool, device) -> bool:
    # Wrapper-only: the SM90 kernel is BSHD-native, so THD is adapted per batch from a
    # host copy of cu_seqlens (_interface_sm90._dense_score_recompute_varlen); the plan
    # classes decline it in check_support() (Rule 3/8).
    return is_thd and device_capability(device)[0] == 9


class DenseIndexerScoreRecompute(_ScoreRecomputeBase):
    """Dense indexer score recompute over full KV.

    ``S[b, q, t] = sm_scale * sum_h ReLU(Q_h . K_t^T) * W_h`` under the
    ratio-causal mask; returns ``(out, denom)`` with ``denom = logsumexp(S)``.

    Layouts: BSHD (``is_thd=False``) on SM90 and SM100+; THD packed
    (``is_thd=True``, ``cu_seqlens_q``/``cu_seqlens_k`` at execute) on SM100+
    only. SM90 declines THD in ``check_support()`` with ``NotImplementedError``:
    its kernel is BSHD-native and serving THD needs a host copy of
    ``cu_seqlens`` per call (the eager ``*_wrapper`` does that per batch).

    THD launch envelope: ``max_seqlen_q`` is required at ``__init__`` and
    ``max_seqlen_k`` defaults to (and must equal) ``sample_out.shape[1]``.
    Both are plan-time host ints; ``execute()`` never derives them from
    ``cu_seqlens`` and rejects caller-passed values that differ from the
    plan's. A BSHD plan rejects them.
    """

    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_weights: torch.Tensor,
        sample_out: torch.Tensor,
        sample_denom_out: torch.Tensor,
        qhead_per_kv_head: Optional[int] = None,
        sm_scale: float = 1.0,
        ratio: int = 1,
        is_thd: bool = False,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
    ):
        super().__init__()
        self.q_desc = self._make_tensor_desc(sample_q, name="sample_q")
        self.k_desc = self._make_tensor_desc(sample_k, name="sample_k")
        self.w_desc = self._make_tensor_desc(sample_weights, name="sample_weights")
        self.out_desc = self._make_tensor_desc(sample_out, name="sample_out")
        self.denom_desc = self._make_tensor_desc(sample_denom_out, name="sample_denom_out")
        self.qhead_per_kv_head = qhead_per_kv_head
        self.sm_scale = float(sm_scale)
        self.ratio = int(ratio)
        self.is_thd = bool(is_thd)
        self.max_seqlen_q, self.max_seqlen_k = _dense_thd_plan_envelope(self.is_thd, max_seqlen_q, max_seqlen_k, self.out_desc)

    def check_support(self) -> bool:
        _check_score_arch(self)
        self._check_dtype(self.q_desc, torch.bfloat16, name="Q")
        self._check_dtype(self.k_desc, torch.bfloat16, name="K")
        self._check_dtype(self.w_desc, torch.bfloat16, name="W")
        self._check_dtype(self.out_desc, torch.float32, name="out")
        self._check_dtype(self.denom_desc, torch.float32, name="denom_out")
        self._value_error_if(self.ratio < 1, f"ratio must be >= 1, got {self.ratio}")
        _check_dense_score_shapes(
            self,
            self.q_desc,
            self.k_desc,
            self.w_desc,
            self.out_desc,
            self.denom_desc,
            "W",
            self.is_thd,
            self.qhead_per_kv_head,
        )
        _check_dense_thd_plan(self)
        self._is_supported = True
        return True

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        out: torch.Tensor,
        denom_out: torch.Tensor,
        sm_scale: Optional[float] = None,
        ratio: Optional[int] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        q_causal_offsets: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ):
        scale = self.sm_scale if sm_scale is None else float(sm_scale)
        ratio_value = self.ratio if ratio is None else int(ratio)
        max_seqlen_q, max_seqlen_k = _dense_thd_execute_envelope(self, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
        _require_dense_live(self, q, k, weights, self.w_desc, "weights", out, denom_out, cu_seqlens_q, cu_seqlens_k, q_causal_offsets)
        major, _ = device_capability()
        if major == 9:
            from . import _interface_sm90 as _iface_sm90

            return _iface_sm90.dense_indexer_score_recompute(
                q,
                k,
                weights,
                out,
                denom_out,
                sm_scale=scale,
                ratio=ratio_value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                q_causal_offsets=q_causal_offsets,
                current_stream=current_stream,
            )
        return _iface_sm100.dense_indexer_score_recompute(
            q,
            k,
            weights,
            qhead_per_kv_head=self.qhead_per_kv_head,
            out=out,
            denom_out=denom_out,
            sm_scale=scale,
            ratio=ratio_value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            q_causal_offsets=q_causal_offsets,
            current_stream=current_stream,
        )


_cache_of_DenseIndexerScoreRecomputeObjects: dict = {}


def dense_indexer_score_recompute_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    qhead_per_kv_head: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    denom_out: Optional[torch.Tensor] = None,
    sm_scale: float = 1.0,
    ratio: int = 1,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    precision: str = "bf16",
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q_scale_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_scale_padded: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """High-level wrapper. Returns ``{'out': scores, 'denom': logsumexp}``.

    ``precision`` is ``"bf16"`` (SM90 and SM100+) or an SM100 block-scaled
    precision: ``"mxfp8"`` (E4M3 Q/K) or ``"mxfp4"``
    (``torch.float4_e2m1fn_x2`` Q/K with last dim ``head_dim // 2``), both with
    one E8M0 scale per 32 head_dim values (``head_dim // 32`` per row, for
    MXFP4 too). Block-scaled precisions take packed
    ``q_scale`` / ``k_scale`` (and, for THD, ``cu_seqlens_*_scale_padded``).

    THD (``cu_seqlens_q``/``cu_seqlens_k`` given) runs natively on SM100+; on
    SM90 the wrapper adapts it per batch from a host copy of ``cu_seqlens``
    (the class declines SM90 THD). Pass ``max_seqlen_q`` and ``max_seqlen_k``:
    when either is omitted the wrapper derives it as ``(cu[1:] - cu[:-1]).max()``,
    one blocking device-to-host read per call that is not CUDA-graph
    capturable (a wrapper-surface convenience; the class requires both at
    plan time). The memo key includes the resulting values.

    Wrapper-surface convenience (the class declines instead): inputs with a
    non-unit innermost stride are copied contiguous, ``cu_seqlens_*`` are cast
    to contiguous int32, ``out`` / ``denom_out`` are allocated when omitted and
    a non-contiguous caller ``out`` / ``denom_out`` is staged and copied back
    (identity kept). All of it runs on ``stream``.
    """
    is_thd, max_q, max_k, out_shape, denom_shape = _dense_sample_shapes(
        q,
        k,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    )
    q, k, weights, out, denom_out, cu_seqlens_q, cu_seqlens_k, q_causal_offsets, q_scale, k_scale, out_user, denom_user = _dense_wrapper_prepare(
        stream, q, k, weights, out, denom_out, out_shape, denom_shape, cu_seqlens_q, cu_seqlens_k, q_causal_offsets, q_scale, k_scale
    )
    precision = precision.lower()
    if precision != "bf16" or q_scale is not None or k_scale is not None or cu_seqlens_q_scale_padded is not None or cu_seqlens_k_scale_padded is not None:
        major, _ = device_capability()
        if major == 9:
            raise NotImplementedError(
                f"Dense indexer score with block scales (precision={BLOCK_SCALED_PRECISION_NAMES}) is SM100-only; SM90 supports precision='bf16'"
            )
        o, d = _iface_sm100.dense_indexer_score_recompute(
            q,
            k,
            weights,
            qhead_per_kv_head=qhead_per_kv_head,
            out=out,
            denom_out=denom_out,
            sm_scale=sm_scale,
            ratio=ratio,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_q if is_thd else max_seqlen_q,
            max_seqlen_k=max_k if is_thd else max_seqlen_k,
            q_causal_offsets=q_causal_offsets,
            precision=precision,
            q_scale=q_scale,
            k_scale=k_scale,
            cu_seqlens_q_scale_padded=cu_seqlens_q_scale_padded,
            cu_seqlens_k_scale_padded=cu_seqlens_k_scale_padded,
            sf_vec_size=sf_vec_size,
            current_stream=stream,
        )
        return _dense_wrapper_result(stream, o, d, out_user, denom_user)
    if _dense_sm90_thd_wrapper_path(is_thd, q.device):
        from . import _interface_sm90 as _iface_sm90

        o, d = _iface_sm90.dense_indexer_score_recompute(
            q,
            k,
            weights,
            out,
            denom_out,
            sm_scale=sm_scale,
            ratio=ratio,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            q_causal_offsets=q_causal_offsets,
            current_stream=stream,
        )
        return _dense_wrapper_result(stream, o, d, out_user, denom_user)
    key = (
        q.device,
        q.dtype,
        q.shape,
        k.shape,
        weights.shape,
        q.stride(),
        k.stride(),
        weights.stride(),
        qhead_per_kv_head,
        float(sm_scale),
        int(ratio),
        is_thd,
        max_q,
        max_k,
        tuple(cu_seqlens_q.shape) if cu_seqlens_q is not None else None,
        tuple(cu_seqlens_k.shape) if cu_seqlens_k is not None else None,
        q_causal_offsets is not None,
    )
    obj = _cache_of_DenseIndexerScoreRecomputeObjects.get(key)
    if obj is None:
        obj = DenseIndexerScoreRecompute(
            sample_q=q,
            sample_k=k,
            sample_weights=weights,
            sample_out=out,
            sample_denom_out=denom_out,
            qhead_per_kv_head=qhead_per_kv_head,
            sm_scale=sm_scale,
            ratio=ratio,
            is_thd=is_thd,
            max_seqlen_q=max_q if is_thd else None,
            max_seqlen_k=max_k if is_thd else None,
        )
        assert obj.check_support()
        obj.compile()
        _cache_of_DenseIndexerScoreRecomputeObjects[key] = obj

    o, d = obj.execute(
        q,
        k,
        weights,
        out,
        denom_out,
        sm_scale=sm_scale,
        ratio=ratio,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_q if is_thd else max_seqlen_q,
        max_seqlen_k=max_k if is_thd else max_seqlen_k,
        q_causal_offsets=q_causal_offsets,
        current_stream=stream,
    )
    return _dense_wrapper_result(stream, o, d, out_user, denom_user)


# ---------------------------------------------------------------------------
# Dense attention score
# ---------------------------------------------------------------------------


class DenseAttnScoreRecompute(_ScoreRecomputeBase):
    """Dense attention score recompute over full KV.

    ``P[b, q, h, t] = exp(Q_h . K_t^T * scale - LSE_h)``, ``out = sum_h P``
    under the ratio-causal mask; returns ``(out, denom)`` with
    ``denom = sum_t out`` (L1 norm).

    Layouts: BSHD (``is_thd=False``) on SM90 and SM100+; THD packed
    (``is_thd=True``, ``cu_seqlens_q``/``cu_seqlens_k`` at execute) on SM100+
    only. SM90 declines THD in ``check_support()`` with ``NotImplementedError``:
    its kernel is BSHD-native and serving THD needs a host copy of
    ``cu_seqlens`` per call (the eager ``*_wrapper`` does that per batch).

    THD launch envelope: ``max_seqlen_q`` is required at ``__init__`` and
    ``max_seqlen_k`` defaults to (and must equal) ``sample_out.shape[1]``.
    Both are plan-time host ints; ``execute()`` never derives them from
    ``cu_seqlens`` and rejects caller-passed values that differ from the
    plan's. A BSHD plan rejects them.
    """

    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_lse: torch.Tensor,
        sample_out: torch.Tensor,
        sample_denom_out: torch.Tensor,
        softmax_scale: float,
        qhead_per_kv_head: Optional[int] = None,
        ratio: int = 1,
        is_thd: bool = False,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
    ):
        super().__init__()
        self.q_desc = self._make_tensor_desc(sample_q, name="sample_q")
        self.k_desc = self._make_tensor_desc(sample_k, name="sample_k")
        self.lse_desc = self._make_tensor_desc(sample_lse, name="sample_lse")
        self.out_desc = self._make_tensor_desc(sample_out, name="sample_out")
        self.denom_desc = self._make_tensor_desc(sample_denom_out, name="sample_denom_out")
        self.softmax_scale = float(softmax_scale)
        self.qhead_per_kv_head = qhead_per_kv_head
        self.ratio = int(ratio)
        self.is_thd = bool(is_thd)
        self.max_seqlen_q, self.max_seqlen_k = _dense_thd_plan_envelope(self.is_thd, max_seqlen_q, max_seqlen_k, self.out_desc)

    def check_support(self) -> bool:
        _check_score_arch(self)
        self._check_dtype(self.q_desc, torch.bfloat16, name="Q")
        self._check_dtype(self.k_desc, torch.bfloat16, name="K")
        self._check_dtype(self.lse_desc, torch.float32, name="LSE")
        self._check_dtype(self.out_desc, torch.float32, name="out")
        self._check_dtype(self.denom_desc, torch.float32, name="denom_out")
        self._value_error_if(self.ratio < 1, f"ratio must be >= 1, got {self.ratio}")
        _check_dense_score_shapes(
            self,
            self.q_desc,
            self.k_desc,
            self.lse_desc,
            self.out_desc,
            self.denom_desc,
            "LSE",
            self.is_thd,
            self.qhead_per_kv_head,
        )
        _check_dense_thd_plan(self)
        self._is_supported = True
        return True

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        lse: torch.Tensor,
        out: torch.Tensor,
        denom_out: torch.Tensor,
        softmax_scale: Optional[float] = None,
        ratio: Optional[int] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        q_causal_offsets: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ):
        scale = self.softmax_scale if softmax_scale is None else float(softmax_scale)
        ratio_value = self.ratio if ratio is None else int(ratio)
        max_seqlen_q, max_seqlen_k = _dense_thd_execute_envelope(self, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
        _require_dense_live(self, q, k, lse, self.lse_desc, "lse", out, denom_out, cu_seqlens_q, cu_seqlens_k, q_causal_offsets)
        major, _ = device_capability()
        if major == 9:
            from . import _interface_sm90 as _iface_sm90

            return _iface_sm90.dense_attn_score_recompute(
                q,
                k,
                lse,
                scale,
                out,
                denom_out,
                ratio=ratio_value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                q_causal_offsets=q_causal_offsets,
                current_stream=current_stream,
            )
        return _iface_sm100.dense_attn_score_recompute(
            q,
            k,
            lse,
            scale,
            qhead_per_kv_head=self.qhead_per_kv_head,
            out=out,
            denom_out=denom_out,
            ratio=ratio_value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            q_causal_offsets=q_causal_offsets,
            current_stream=current_stream,
        )


_cache_of_DenseAttnScoreRecomputeObjects: dict = {}


def dense_attn_score_recompute_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    qhead_per_kv_head: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    denom_out: Optional[torch.Tensor] = None,
    ratio: int = 1,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    precision: str = "bf16",
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q_scale_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_scale_padded: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """High-level wrapper. Returns ``{'out': scores, 'denom': l1norm}``.

    THD (``cu_seqlens_q``/``cu_seqlens_k`` given) runs natively on SM100+; on
    SM90 the wrapper adapts it per batch from a host copy of ``cu_seqlens``
    (the class declines SM90 THD). Pass ``max_seqlen_q`` and ``max_seqlen_k``:
    when either is omitted the wrapper derives it as ``(cu[1:] - cu[:-1]).max()``,
    one blocking device-to-host read per call that is not CUDA-graph
    capturable (a wrapper-surface convenience; the class requires both at
    plan time). The memo key includes the resulting values.

    Wrapper-surface convenience (the class declines instead): inputs with a
    non-unit innermost stride are copied contiguous, ``cu_seqlens_*`` are cast
    to contiguous int32, ``out`` / ``denom_out`` are allocated when omitted and
    a non-contiguous caller ``out`` / ``denom_out`` is staged and copied back
    (identity kept). All of it runs on ``stream``.
    """
    is_thd, max_q, max_k, out_shape, denom_shape = _dense_sample_shapes(
        q,
        k,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    )
    q, k, lse, out, denom_out, cu_seqlens_q, cu_seqlens_k, q_causal_offsets, q_scale, k_scale, out_user, denom_user = _dense_wrapper_prepare(
        stream, q, k, lse, out, denom_out, out_shape, denom_shape, cu_seqlens_q, cu_seqlens_k, q_causal_offsets, q_scale, k_scale
    )
    precision = precision.lower()
    if precision != "bf16" or q_scale is not None or k_scale is not None or cu_seqlens_q_scale_padded is not None or cu_seqlens_k_scale_padded is not None:
        major, _ = device_capability()
        if major == 9:
            raise NotImplementedError("Dense attention score MXFP8 is SM100-only")
        o, d = _iface_sm100.dense_attn_score_recompute(
            q,
            k,
            lse,
            softmax_scale,
            qhead_per_kv_head=qhead_per_kv_head,
            out=out,
            denom_out=denom_out,
            ratio=ratio,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_q if is_thd else max_seqlen_q,
            max_seqlen_k=max_k if is_thd else max_seqlen_k,
            q_causal_offsets=q_causal_offsets,
            precision=precision,
            q_scale=q_scale,
            k_scale=k_scale,
            cu_seqlens_q_scale_padded=cu_seqlens_q_scale_padded,
            cu_seqlens_k_scale_padded=cu_seqlens_k_scale_padded,
            sf_vec_size=sf_vec_size,
            current_stream=stream,
        )
        return _dense_wrapper_result(stream, o, d, out_user, denom_user)
    if _dense_sm90_thd_wrapper_path(is_thd, q.device):
        from . import _interface_sm90 as _iface_sm90

        o, d = _iface_sm90.dense_attn_score_recompute(
            q,
            k,
            lse,
            softmax_scale,
            out,
            denom_out,
            ratio=ratio,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            q_causal_offsets=q_causal_offsets,
            current_stream=stream,
        )
        return _dense_wrapper_result(stream, o, d, out_user, denom_user)
    key = (
        q.device,
        q.dtype,
        q.shape,
        k.shape,
        lse.shape,
        q.stride(),
        k.stride(),
        lse.stride(),
        qhead_per_kv_head,
        float(softmax_scale),
        int(ratio),
        is_thd,
        max_q,
        max_k,
        tuple(cu_seqlens_q.shape) if cu_seqlens_q is not None else None,
        tuple(cu_seqlens_k.shape) if cu_seqlens_k is not None else None,
        q_causal_offsets is not None,
    )
    obj = _cache_of_DenseAttnScoreRecomputeObjects.get(key)
    if obj is None:
        obj = DenseAttnScoreRecompute(
            sample_q=q,
            sample_k=k,
            sample_lse=lse,
            sample_out=out,
            sample_denom_out=denom_out,
            softmax_scale=softmax_scale,
            qhead_per_kv_head=qhead_per_kv_head,
            ratio=ratio,
            is_thd=is_thd,
            max_seqlen_q=max_q if is_thd else None,
            max_seqlen_k=max_k if is_thd else None,
        )
        assert obj.check_support()
        obj.compile()
        _cache_of_DenseAttnScoreRecomputeObjects[key] = obj

    o, d = obj.execute(
        q,
        k,
        lse,
        out,
        denom_out,
        softmax_scale=softmax_scale,
        ratio=ratio,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_q if is_thd else max_seqlen_q,
        max_seqlen_k=max_k if is_thd else max_seqlen_k,
        q_causal_offsets=q_causal_offsets,
        current_stream=stream,
    )
    return _dense_wrapper_result(stream, o, d, out_user, denom_user)
