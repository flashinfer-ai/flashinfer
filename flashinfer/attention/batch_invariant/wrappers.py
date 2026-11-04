# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The delegate behind ``batch_invariant=True`` on ``BatchDecodeWithPagedKVCacheWrapper``, ``BatchPrefillWithPagedKVCacheWrapper`` and
``BatchPrefillWithRaggedKVCacheWrapper``: the wrappers' ``plan`` / ``run`` hand their arguments here, the class's typed checks raise
:class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError` for everything the class does not cover (nothing falls back), and the
engine (``engine.plan_attention`` / ``engine.run_attention``) runs the kernels. One class per head dim: a token's output bits depend on
its KV position, its head and the window alone, so the three wrappers compute the same bits for the same token.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from ...batch_invariant.support import check_device, unsupported
from . import engine
from .spec import HEAD_DIMS

CLASS = "attention.bf16"
#: The compute capabilities the class is stated and tested for: SM100 (B200), SM103 (B300), SM107 (Rubin).
ATTENTION_ARCHS = ((10, 0), (10, 3), (10, 7))
_MAX_PAGE = 128


def _dtype(x) -> torch.dtype:
    from ...utils import canonicalize_torch_dtype

    return canonicalize_torch_dtype(x) if x is not None else torch.bfloat16


class BatchInvariantAttention:
    """One instance per wrapper: ``kind`` is ``"decode"`` (paged KV, one or more query tokens per request), ``"prefill"`` (paged
    KV) or ``"ragged"`` (the K / V of every request packed along the token dimension)."""

    def __init__(self, float_workspace_buffer: torch.Tensor, kv_layout: str, kind: str, backend: str, jit_args) -> None:
        assert kind in ("decode", "prefill", "ragged"), kind
        if backend != "auto":
            raise unsupported(CLASS, f'backend="{backend}" with batch_invariant=True', 'backend="auto" (the mode selects its own kernels)')
        if jit_args is not None:
            raise unsupported(CLASS, "jit_args with batch_invariant=True", "no JIT customisation (the class has one dataflow)")
        if kv_layout not in ("NHD", "HND"):
            raise unsupported(CLASS, f"kv_layout {kv_layout!r}", '"NHD" or "HND"')
        check_device(float_workspace_buffer.device, CLASS, ATTENTION_ARCHS)
        if float_workspace_buffer.data_ptr() % 16:
            raise unsupported(CLASS, "a float workspace buffer that is not 16-byte aligned", "a 16-byte-aligned workspace")
        self.kind = kind
        self.kv_layout = kv_layout
        self.device = float_workspace_buffer.device
        self.workspace = float_workspace_buffer
        self._planned = None
        self._config = None
        self._kv_indptr: Optional[torch.Tensor] = None
        self._kv_indices: Optional[torch.Tensor] = None
        self._sm_scale: Optional[float] = None
        self._window_left = -1
        self._total_q = 0
        self._heads: Tuple[int, int, int] = (0, 0, 0)
        self._page = 1

    # ------------------------------------------------------------------ plan
    def plan(
        self,
        *,
        qo_indptr: Optional[torch.Tensor],
        kv_indptr: torch.Tensor,
        kv_indices: Optional[torch.Tensor],
        kv_last_page_len: Optional[torch.Tensor],
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: Optional[int],
        page_size: int,
        causal: bool,
        window_left: int,
        logits_soft_cap: Optional[float],
        pos_encoding_mode: str,
        q_data_type,
        kv_data_type,
        o_data_type,
        sm_scale: Optional[float],
        q_len_per_req: int = 1,
        non_blocking: bool = True,
        custom_mask: Optional[torch.Tensor] = None,
        packed_custom_mask: Optional[torch.Tensor] = None,
        use_fp16_qk_reduction: bool = False,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: bool = False,
        use_token_head_sf: bool = False,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> None:
        """The wrappers' ``plan`` arguments; every input outside the class raises before any work."""
        D = int(head_dim_qk)
        if D not in HEAD_DIMS:
            raise unsupported(CLASS, f"head_dim {D}", f"head dims {list(HEAD_DIMS)}")
        if head_dim_vo is not None and int(head_dim_vo) != D:
            raise unsupported(CLASS, f"head_dim_vo {head_dim_vo} with head_dim_qk {D}", "equal query and value head dims")
        if num_kv_heads <= 0 or num_qo_heads % num_kv_heads:
            raise unsupported(CLASS, f"{num_qo_heads} query heads over {num_kv_heads} KV heads", "num_qo_heads a multiple of num_kv_heads")
        g = num_qo_heads // num_kv_heads
        if g > 128:
            raise unsupported(CLASS, f"a GQA group of {g}", "at most 128 query heads per KV head")
        for name, dt in (("q_data_type", q_data_type), ("kv_data_type", kv_data_type), ("o_data_type", o_data_type)):
            if _dtype(dt) != torch.bfloat16:
                raise unsupported(CLASS, f"{name} {_dtype(dt)}", "bf16 queries, keys, values and output")
        if self.kind == "ragged":
            page_size = 1
        if page_size < 1 or page_size > _MAX_PAGE or page_size & (page_size - 1):
            raise unsupported(CLASS, f"page_size {page_size}", f"a power of two up to {_MAX_PAGE}")
        if self.kind != "decode" and not causal:
            raise unsupported(CLASS, "non-causal attention", "causal attention (bottom-right aligned)")
        if logits_soft_cap is not None and logits_soft_cap > 0:
            raise unsupported(CLASS, f"logits_soft_cap {logits_soft_cap}", "no logits soft cap")
        if pos_encoding_mode not in ("NONE", 0, None):
            raise unsupported(CLASS, f"pos_encoding_mode {pos_encoding_mode!r}", 'pos_encoding_mode "NONE" (apply RoPE to q / k beforehand)')
        if rope_scale is not None or rope_theta is not None:
            raise unsupported(CLASS, "rope_scale / rope_theta", "no fused RoPE")
        if custom_mask is not None or packed_custom_mask is not None:
            raise unsupported(CLASS, "a custom mask", "the causal mask with an optional sliding window")
        if use_fp16_qk_reduction:
            raise unsupported(CLASS, "use_fp16_qk_reduction", "the fp32 score accumulation of the class")
        if fixed_split_size is not None or disable_split_kv:
            raise unsupported(CLASS, "fixed_split_size / disable_split_kv", "the class's own split rule (a function of the SM count only)")
        if use_token_head_sf:
            raise unsupported(CLASS, "use_token_head_sf", "unscaled bf16 K / V")
        if window_left is None:
            window_left = -1
        if window_left < -1:
            raise unsupported(CLASS, f"window_left {window_left}", "window_left >= 0, or -1 for no window")
        if sm_scale is not None and not (float(sm_scale) > 0.0):
            raise unsupported(CLASS, f"sm_scale {sm_scale}", "a positive sm_scale")
        if self.kind == "decode":
            if q_len_per_req < 1:
                raise unsupported(CLASS, f"q_len_per_req {q_len_per_req}", "at least one query token per request")
            batch = int(kv_indptr.numel()) - 1
            qo_indptr = torch.arange(0, batch + 1, dtype=torch.int32, device=kv_indptr.device) * int(q_len_per_req)
        else:
            if qo_indptr is None:
                raise unsupported(CLASS, "a plan without qo_indptr", "the packed query offsets")
        kv_indptr_host = kv_indptr.to("cpu", non_blocking=False).to(torch.int64)
        qo_host = qo_indptr.to("cpu", non_blocking=False).to(torch.int64)
        if qo_host.numel() != kv_indptr_host.numel():
            raise unsupported(CLASS, f"{qo_host.numel() - 1} query requests over {kv_indptr_host.numel() - 1} KV requests", "one KV range per request")
        if self.kind == "ragged":
            last_host = torch.ones(max(0, kv_indptr_host.numel() - 1), dtype=torch.int64)
        else:
            if kv_last_page_len is None or kv_indices is None:
                raise unsupported(CLASS, "a paged plan without kv_indices / kv_last_page_len", "the paged KV CSR triple")
            last_host = kv_last_page_len.to("cpu", non_blocking=False).to(torch.int64)
        kv_lens = (torch.clamp(kv_indptr_host[1:] - kv_indptr_host[:-1] - 1, min=0) * page_size + last_host).tolist()
        q_lens = (qo_host[1:] - qo_host[:-1]).tolist()
        for i, (ql, kl) in enumerate(zip(q_lens, kv_lens)):
            if ql > kl:
                raise unsupported(CLASS, f"request {i} with {ql} query tokens over {kl} keys", "kv_len >= q_len per request (the causal class)")
        config = engine.default_config(D=D)
        if D == 64 and self.kind != "decode":
            from dataclasses import replace

            config = replace(config, softmax_wgs=2, q2=True)         # the paired placement (two Q tiles per CTA) at D = 64: same bits, faster prefill
        try:
            planned = engine.plan_attention(qo_indptr, kv_indptr, kv_last_page_len if self.kind != "ragged" else None, num_qo_heads, num_kv_heads, D,
                                            page_size, window_left, config, self.workspace, self.device, ragged=self.kind == "ragged")
        except engine.WorkspaceTooSmall as e:
            raise unsupported(CLASS, f"a float workspace buffer of {self.workspace.numel() * self.workspace.element_size()} bytes",
                              f"at least {e.needed} bytes for this batch's split-KV partials") from None
        self._planned = planned
        self._config = config
        self._kv_indptr = kv_indptr.contiguous()
        self._kv_indices = kv_indices.contiguous() if kv_indices is not None else torch.zeros(1, dtype=torch.int32, device=self.device)
        self._sm_scale = float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(D)
        self._window_left = int(window_left)
        self._total_q = int(qo_host[-1].item()) if qo_host.numel() else 0
        self._heads = (num_qo_heads, num_kv_heads, D)
        self._page = page_size

    # ------------------------------------------------------------------ run
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        out: Optional[torch.Tensor],
        return_lse: bool,
        lse: Optional[torch.Tensor],
        window_left: Optional[int],
        scales: Tuple = (),
        sinks=None,
        kv_cache_sf=None,
        skip_softmax_threshold_scale_factor=None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        if self._planned is None:
            raise RuntimeError("plan() must be called before run()")
        if return_lse or lse is not None:
            raise unsupported(CLASS, "return_lse", "the attention output alone (the class has no LSE member yet)")
        if any(s is not None for s in scales):
            raise unsupported(CLASS, "q_scale / k_scale / v_scale / o_scale", "unscaled bf16 operands")
        if sinks is not None:
            raise unsupported(CLASS, "attention sinks", "plain softmax")
        if kv_cache_sf is not None:
            raise unsupported(CLASS, "a scaled (fp8 / fp4) KV cache", "a bf16 KV cache")
        if skip_softmax_threshold_scale_factor is not None:
            raise unsupported(CLASS, "skip_softmax_threshold_scale_factor", "the full softmax")
        if window_left is not None and int(window_left) != self._window_left:
            raise unsupported(CLASS, f"window_left {window_left} at run() after planning {self._window_left}", "the planned window")
        Hq, Hkv, D = self._heads
        if q.dim() == 4:
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
        if q.dim() != 3 or q.shape[1] != Hq or q.shape[2] != D:
            raise unsupported(CLASS, f"q of shape {tuple(q.shape)}", f"(total_q, {Hq}, {D})")
        if q.shape[0] != self._total_q:
            raise unsupported(CLASS, f"{q.shape[0]} query rows", f"the planned {self._total_q}")
        if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
            raise unsupported(CLASS, f"dtypes ({q.dtype}, {k.dtype}, {v.dtype})", "bf16 q, k and v")
        if not q.is_contiguous():
            q = q.contiguous()
        if out is None:
            out = torch.empty_like(q)
        elif out.shape != q.shape or out.dtype != torch.bfloat16 or not out.is_contiguous():
            raise unsupported(CLASS, f"out {tuple(out.shape)} {out.dtype}", f"a contiguous bf16 ({q.shape[0]}, {Hq}, {D}) output")
        engine.run_attention(self._planned, q, k, v, self._kv_indptr, self._kv_indices, self._sm_scale, self._window_left, self.kv_layout, out,
                             self._config, ragged=self.kind == "ragged")
        return out
