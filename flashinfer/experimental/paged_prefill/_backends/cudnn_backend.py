"""cuDNN backend: cudnn_batch_prefill_with_kv_cache (post-#3921 tokens mode).

Dialect: token-unit indptr as batch offsets (units="tokens"), per-request
lens as (b,1,1,1).  Native LSE is base-2 padded (b, max_q, h) (natural-log
before #4663); this backend gathers it to packed (tokens, h).
"""

from __future__ import annotations

from typing import Optional

import torch

from .._contracts import PlanMetadata
from .._planning import Derived


class _CudnnBackend:
    name = "cudnn"

    def __init__(self, device, kv_layout, workspace):
        assert kv_layout == "HND"  # capability-gated upstream
        self._workspace = workspace.view(torch.int8)
        self._meta: Optional[PlanMetadata] = None
        self._derived: Optional[Derived] = None
        self._native_lse: Optional[torch.Tensor] = None
        self._batch_ids: Optional[torch.Tensor] = None
        self._pos: Optional[torch.Tensor] = None

    def plan(self, meta: PlanMetadata, derived: Derived) -> None:
        # The LSE-gather indices and the native stats buffer are static per
        # plan (qo_indptr and batch size are fixed here) — precompute them so
        # run() stays a single indexed lookup on the hot path.
        native_lse = batch_ids = pos = None
        if meta.return_lse:
            dev = meta.qo_indptr.device
            token = torch.arange(meta.total_q_tokens, device=dev, dtype=torch.int64)
            bounds = meta.qo_indptr[1:].to(torch.int64)
            batch_ids = torch.searchsorted(bounds, token, right=True)
            pos = token - meta.qo_indptr.to(torch.int64)[batch_ids]
            native_lse = torch.empty(
                meta.batch_size,
                meta.max_q_len,
                meta.num_qo_heads,
                device=dev,
                dtype=torch.float32,
            )
        # publish only after every allocation above succeeded
        self._meta, self._derived = meta, derived
        self._native_lse, self._batch_ids, self._pos = native_lse, batch_ids, pos

    def run(self, q, k_cache, v_cache, *, out=None, lse=None):
        from ....cudnn import cudnn_batch_prefill_with_kv_cache

        meta, derived = self._meta, self._derived
        assert meta is not None and derived is not None
        assert meta.block_tables is not None  # needs_dense contract
        b = meta.batch_size
        out_t, lse_t = cudnn_batch_prefill_with_kv_cache(
            q,
            k_cache,
            v_cache,
            meta.sm_scale,
            self._workspace,
            max_token_per_sequence=meta.max_q_len,
            max_sequence_kv=meta.max_kv_len,
            actual_seq_lens_q=derived.q_seq_lens.view(b, 1, 1, 1),
            actual_seq_lens_kv=meta.kv_seq_lens.view(b, 1, 1, 1),
            block_tables=meta.block_tables,
            causal=meta.causal,
            return_lse=meta.return_lse,
            batch_offsets_q=meta.qo_indptr,
            batch_offsets_units="tokens",
            out=out,
            lse=self._native_lse,
        )
        if not meta.return_lse:
            return out_t, None
        # padded (b, max_q, h) -> packed (tokens, h), using the plan-time
        # precomputed gather indices (zero sync). cuDNN returns base-2 LSE
        # since #4663, so no log-base fold here.
        packed = lse_t[self._batch_ids, self._pos, :]
        if lse is not None:
            lse.copy_(packed)
            packed = lse
        return out_t, packed


__all__ = ["_CudnnBackend"]
