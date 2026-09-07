"""fa2/fa3 backend: the existing BatchPrefillWithPagedKVCacheWrapper.

Dialect: CSR page metadata + host arrays for the split-KV scheduler
(computed from the mirrors the contract layer guarantees — zero-sync).
The generated FA wrapper holds its own plan state, so this backend keeps one
wrapper instance as stable storage and re-plans it in place.
"""

from __future__ import annotations

import torch

from .._contracts import LN2, PlanMetadata
from .._planning import Derived


class _FaBackend:
    def __init__(self, device, kv_layout, workspace, backend: str = "fa2"):
        from ....prefill import BatchPrefillWithPagedKVCacheWrapper

        self.name = backend
        self._wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace, kv_layout, backend=backend
        )
        self._lse_mode = "none"

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
            q_data_type=meta.q_dtype,
            kv_data_type=meta.kv_dtype,
        )
        self._lse_mode = meta.lse_mode

    def run(self, q, k_cache, v_cache, *, out=None, lse=None, sm_scale: float):
        # The generated-FA wrapper reads sm_scale from plan-time state and its
        # run() has no override, while the kernel takes it as a launch arg.
        # Setting it here keeps sm_scale a per-run (per-layer) value; this
        # poke goes away when the backend calls the FA module directly.
        self._wrapper._sm_scale = sm_scale
        need_lse = self._lse_mode != "none"
        r = self._wrapper.run(
            q, (k_cache, v_cache), out=out, lse=lse, return_lse=need_lse
        )
        if not need_lse:
            return r, None
        out_t, lse_t = r
        if self._lse_mode == "basee":
            lse_t.mul_(LN2)  # FA kernels emit base-2 (exp2 softmax); one fold
        return out_t, lse_t


__all__ = ["_FaBackend"]
