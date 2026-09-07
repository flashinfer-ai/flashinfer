"""fa2/fa3 backend: the existing BatchPrefillWithPagedKVCacheWrapper.

Dialect: CSR page metadata + host arrays for the split-KV scheduler
(computed from the mirrors the contract layer guarantees — zero-sync).
The generated FA wrapper holds its own plan state, so this backend keeps one
wrapper instance as stable storage and re-plans it in place.
"""

from __future__ import annotations

import torch

from .._contracts import PlanMetadata
from .._planning import Derived


class _FaBackend:
    def __init__(self, device, kv_layout, workspace, backend: str = "fa2"):
        from ....prefill import BatchPrefillWithPagedKVCacheWrapper

        self.name = backend
        self._wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace, kv_layout, backend=backend
        )
        self._return_lse = False

    def plan(self, meta: PlanMetadata, derived: Derived) -> None:
        qo_host = meta.qo_indptr_cpu.to(torch.int32)
        kv_lens_host = meta.kv_seq_lens_cpu.to(torch.int32)
        page = meta.page_size
        pages_host = (kv_lens_host + page - 1) // page
        kv_indptr_host = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                torch.cumsum(pages_host, 0, dtype=torch.int32),
            ]
        )
        last_len_host = ((kv_lens_host - 1) % page + 1).to(torch.int32)
        self._wrapper.plan(
            qo_host,
            kv_indptr_host,
            derived.kv_page_indices,
            last_len_host,
            meta.num_qo_heads,
            meta.num_kv_heads,
            meta.head_dim_qk,
            page,
            head_dim_vo=meta.head_dim_vo,
            causal=meta.causal,
            window_left=meta.window_left,
            sm_scale=meta.sm_scale,
            q_data_type=meta.q_dtype,
            kv_data_type=meta.kv_dtype,
        )
        self._return_lse = meta.return_lse

    def run(self, q, k_cache, v_cache, *, out=None, lse=None):
        r = self._wrapper.run(
            q, (k_cache, v_cache), out=out, lse=lse, return_lse=self._return_lse
        )
        return r if self._return_lse else (r, None)


__all__ = ["_FaBackend"]
