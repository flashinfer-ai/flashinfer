"""trtllm-gen backend: trtllm_batch_context_with_kv_cache.

Dialect: the unified form natively (this is where the canonical form came
from); the only derivations are cum_kv_seq_lens and the bmm scale fold
(bmm1 = sm_scale for unquantized, bmm2 = 1.0).  Workspace must be
zero-initialized (kernel counter semantics), so this backend owns a private
one instead of the shared scratch buffer.
"""

from __future__ import annotations

from typing import Optional

import torch

from .._contracts import LN2, PlanMetadata
from .._planning import Derived


class _TrtllmGenBackend:
    name = "trtllm-gen"

    def __init__(self, device, kv_layout, workspace):
        # deliberately NOT the shared workspace: trtllm-gen kernels rely on
        # zero-initialized counter semantics
        self._workspace = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self._kv_layout = kv_layout
        self._meta: Optional[PlanMetadata] = None
        self._derived: Optional[Derived] = None

    def plan(self, meta: PlanMetadata, derived: Derived) -> None:
        self._meta, self._derived = meta, derived

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
        from ....prefill import trtllm_batch_context_with_kv_cache

        meta, derived = self._meta, self._derived
        assert meta is not None and derived is not None
        assert meta.block_tables is not None  # needs_dense contract
        result = trtllm_batch_context_with_kv_cache(
            q,
            (k_cache, v_cache),
            self._workspace,
            meta.block_tables,
            meta.kv_seq_lens,
            meta.max_q_len,
            meta.max_kv_len,
            sm_scale
            * (k_scale if k_scale is not None else 1.0),  # bmm1 (k descale folds in)
            v_scale if v_scale is not None else 1.0,  # bmm2 (v descale)
            meta.batch_size,
            meta.qo_indptr,
            derived.cum_kv_seq_lens,
            window_left=meta.window_left,
            kv_layout=self._kv_layout,
            causal=meta.causal,
            out=out,
            lse=lse,
            return_lse=meta.need_lse,
        )
        if meta.need_lse:
            out_t, lse_t = result
            if meta.lse_mode == "basee":
                lse_t.mul_(LN2)  # trtllm-gen emits base-2; one fold
            return out_t, lse_t
        return result, None


__all__ = ["_TrtllmGenBackend"]
