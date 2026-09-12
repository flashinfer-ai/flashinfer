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
        # The cuDNN graph is built from k/v_cache.stride(), so NHD storage is
        # presented as a zero-copy permuted view with HND logical dim order.
        self._permute_kv = kv_layout == "NHD"
        self._workspace = workspace.view(torch.int8)
        self._meta: Optional[PlanMetadata] = None
        self._derived: Optional[Derived] = None
        self._native_lse: Optional[torch.Tensor] = None
        self._batch_ids: Optional[torch.Tensor] = None
        self._pos: Optional[torch.Tensor] = None
        self._device = device
        # cuDNN takes fp8 dequant scales as (1,1,1,1) GPU tensors; scales are
        # per-layer constants, so cache one tensor per distinct value (no H2D
        # on the hot path after the first call).
        self._scale_tensors: dict = {}

    def _scale_tensor(self, value):
        if value is None:
            return None
        t = self._scale_tensors.get(value)
        if t is None:
            t = torch.tensor([value], dtype=torch.float32, device=self._device).view(
                1, 1, 1, 1
            )
            self._scale_tensors[value] = t
        return t

    def plan(self, meta: PlanMetadata, derived: Derived) -> None:
        # The LSE-gather indices and the native stats buffer are static per
        # plan (qo_indptr and batch size are fixed here) — precompute them so
        # run() stays a single indexed lookup on the hot path.
        native_lse = batch_ids = pos = None
        if meta.need_lse:
            dev = meta.qo_indptr.device
            token = torch.arange(meta.total_q_tokens, device=dev, dtype=torch.int64)
            bounds = meta.qo_indptr[1:].to(torch.int64)
            new_batch_ids = torch.searchsorted(bounds, token, right=True)
            new_pos = token - meta.qo_indptr.to(torch.int64)[new_batch_ids]
            # Keep the storage stable when the shapes repeat (CUDA-graph
            # re-plan): refill in place instead of rebinding new tensors.
            batch_ids, pos, native_lse = self._batch_ids, self._pos, self._native_lse
            if batch_ids is None or batch_ids.shape != new_batch_ids.shape:
                batch_ids, pos = new_batch_ids, new_pos
            else:
                batch_ids.copy_(new_batch_ids)
                pos.copy_(new_pos)
            lse_shape = (meta.batch_size, meta.max_q_len, meta.num_qo_heads)
            if native_lse is None or tuple(native_lse.shape) != lse_shape:
                native_lse = torch.empty(*lse_shape, device=dev, dtype=torch.float32)
        # publish only after every allocation above succeeded
        self._meta, self._derived = meta, derived
        self._native_lse, self._batch_ids, self._pos = native_lse, batch_ids, pos

    def run(
        self,
        q,
        k_cache,
        v_cache,
        *,
        out=None,
        lse=None,
        sm_scale: float,
        k_scale=None,
        v_scale=None,
    ):
        from ....cudnn import cudnn_batch_prefill_with_kv_cache

        meta, derived = self._meta, self._derived
        assert meta is not None and derived is not None
        assert meta.block_tables is not None  # needs_dense contract
        b = meta.batch_size
        if self._permute_kv:
            k_cache = k_cache.permute(0, 2, 1, 3)
            v_cache = v_cache.permute(0, 2, 1, 3)
        out_t, lse_t = cudnn_batch_prefill_with_kv_cache(
            q,
            k_cache,
            v_cache,
            sm_scale,
            self._workspace,
            max_token_per_sequence=meta.max_q_len,
            max_sequence_kv=meta.max_kv_len,
            actual_seq_lens_q=derived.q_seq_lens.view(b, 1, 1, 1),
            actual_seq_lens_kv=meta.kv_seq_lens.view(b, 1, 1, 1),
            block_tables=meta.block_tables,
            causal=meta.causal,
            k_scale=self._scale_tensor(k_scale),
            v_scale=self._scale_tensor(v_scale),
            return_lse=meta.need_lse,
            # native stats are natural-log: basee costs nothing, base2 one fold
            lse_base="e" if meta.lse_mode == "basee" else "2",
            batch_offsets_q=meta.qo_indptr,
            batch_offsets_units="tokens",
            out=out,
            lse=self._native_lse,
        )
        if not meta.need_lse:
            return out_t, None
        # padded (b, max_q, h) -> packed (tokens, h), using the plan-time
        # precomputed gather indices (zero sync). The base was selected above.
        packed = lse_t[self._batch_ids, self._pos, :]
        if lse is not None:
            lse.copy_(packed)
            packed = lse
        return out_t, packed


__all__ = ["_CudnnBackend"]
