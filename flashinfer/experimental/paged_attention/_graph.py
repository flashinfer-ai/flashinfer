"""CUDA-graph re-plan protocol: reserved metadata storage + transactional staging.

Mirrors the reserved-buffer protocol of ``BatchPrefillWithPagedKVCacheWrapper``
(``use_cuda_graph=True``) and the transactional staging of the Batch MLA
backends (``flashinfer/mla/_batch_mla/_backends/_fa_common.py``).

A captured graph bakes in device pointers. In graph mode the controller
therefore never hands a backend the caller's tensors or a fresh derivation:
it owns one set of reserved buffers, sized by the FIRST plan (the capture
shapes), and every later ``plan()`` copies the new batch into them. Shapes a
captured kernel depends on — batch size, table width, host maxes, total query
tokens — must match the capture; a plan that would not fit is rejected before
anything is written, and a plan that fails midway restores every buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from ._contracts import PagedAttentionMetadata, _expect
from ._planning import Derived


@dataclass(frozen=True)
class GraphCapacity:
    """What the FIRST graph-mode plan fixed (the capture shapes)."""

    batch_size: int
    kv_input_form: str
    page_size: int
    max_q_len: int
    max_kv_len: int
    total_q_tokens: int
    # dense (b, W): the caller's width, or ceil(max_kv/page) when derived
    table_width: int
    flat_capacity: int  # flat page-id buffer length


class GraphBuffers:
    """Reserved device storage for one PagedAttention instance in graph mode."""

    def __init__(self, metadata: PagedAttentionMetadata, device: torch.device):
        b = metadata.batch_size
        if metadata.block_tables is not None:
            width = int(metadata.block_tables.shape[1])
            flat_cap = b * width
        else:
            width = (metadata.max_kv_len + metadata.page_size - 1) // metadata.page_size
            flat_cap = int(metadata.kv_page_indices.shape[0])
        self.capacity = GraphCapacity(
            batch_size=b,
            kv_input_form=metadata.kv_input_form,
            page_size=metadata.page_size,
            max_q_len=metadata.max_q_len,
            max_kv_len=metadata.max_kv_len,
            total_q_tokens=metadata.total_q_tokens,
            table_width=width,
            flat_capacity=flat_cap,
        )
        i32 = dict(dtype=torch.int32, device=device)
        # the caller-facing canonical metadata, mirrored into stable storage
        self.qo_indptr = torch.zeros(b + 1, **i32)
        self.kv_seq_lens = torch.zeros(b, **i32)
        self.block_tables = torch.zeros(b, width, **i32)  # given or derived dense
        self.kv_page_indices = torch.zeros(flat_cap, **i32)  # given (csr) or derived
        # backend-neutral derived forms
        self.q_seq_lens = torch.zeros(b, **i32)
        self.cum_kv_seq_lens = torch.zeros(b + 1, **i32)
        self.kv_page_indptr = torch.zeros(b + 1, **i32)

    # ---- checks ----
    def preflight(self, metadata: PagedAttentionMetadata) -> None:
        """Reject before writing anything a captured kernel would misread."""
        cap = self.capacity
        checks = (
            ("batch_size", metadata.batch_size, cap.batch_size),
            ("kv_input_form", metadata.kv_input_form, cap.kv_input_form),
            ("page_size", metadata.page_size, cap.page_size),
            ("max_q_len", metadata.max_q_len, cap.max_q_len),
            ("max_kv_len", metadata.max_kv_len, cap.max_kv_len),
            ("total_q_tokens", metadata.total_q_tokens, cap.total_q_tokens),
        )
        for name, got, want in checks:
            _expect(
                got == want,
                f"CUDA graph re-plan: {name} {got!r} differs from the captured "
                f"{want!r}; a captured graph cannot change it — use one "
                "PagedAttention(use_cuda_graph=True) instance per graph bucket",
            )
        if metadata.block_tables is not None:
            _expect(
                int(metadata.block_tables.shape[1]) == cap.table_width,
                f"CUDA graph re-plan: block_tables width {metadata.block_tables.shape[1]} "
                f"differs from the captured {cap.table_width}",
            )
        else:
            _expect(
                int(metadata.kv_page_indices.shape[0]) <= cap.flat_capacity,
                f"CUDA graph re-plan: kv_page_indices has "
                f"{metadata.kv_page_indices.shape[0]} entries, reserved capacity is "
                f"{cap.flat_capacity}",
            )

    # ---- staging ----
    def targets(
        self, metadata: PagedAttentionMetadata, fresh: Derived
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """(reserved destination, source) pairs for one re-plan."""
        pairs = [
            (self.qo_indptr, metadata.qo_indptr),
            (self.kv_seq_lens, metadata.kv_seq_lens),
            (self.q_seq_lens, fresh.q_seq_lens),
            (self.cum_kv_seq_lens, fresh.cum_kv_seq_lens),
            (self.kv_page_indptr, fresh.kv_page_indptr),
        ]
        if metadata.block_tables is not None:
            pairs.append((self.block_tables, metadata.block_tables))
            pairs.append((self.kv_page_indices, fresh.kv_page_indices))  # b*W exactly
        else:
            n = int(metadata.kv_page_indices.shape[0])
            pairs.append((self.kv_page_indices[:n], metadata.kv_page_indices))
            if fresh.block_tables is not None:
                pairs.append((self.block_tables, fresh.block_tables))
        return pairs

    def derived_view(self, *, needs_dense: bool) -> Derived:
        return Derived(
            q_seq_lens=self.q_seq_lens,
            cum_kv_seq_lens=self.cum_kv_seq_lens,
            kv_page_indptr=self.kv_page_indptr,
            kv_page_indices=self.kv_page_indices,
            block_tables=self.block_tables if needs_dense else None,
        )


class Transaction:
    """Snapshot/restore for a set of (destination, source) copies plus any
    follow-on work; ``commit()`` drops the snapshots, leaving the context
    without committing restores every destination."""

    def __init__(self, pairs: List[Tuple[torch.Tensor, torch.Tensor]]):
        self._pairs = pairs
        self._snapshots: Optional[List[torch.Tensor]] = None
        self._committed = False

    def __enter__(self) -> "Transaction":
        self._snapshots = [dst.clone() for dst, _ in self._pairs]
        for dst, src in self._pairs:
            dst.copy_(src, non_blocking=True)
        return self

    def commit(self) -> None:
        self._committed = True

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._committed and self._snapshots is not None:
            for (dst, _), snap in zip(self._pairs, self._snapshots, strict=True):
                dst.copy_(snap)
        self._snapshots = None


__all__ = ["GraphBuffers", "GraphCapacity", "Transaction"]
