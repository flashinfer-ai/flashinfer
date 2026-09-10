"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Experimental planning implementation behind
# ``flashinfer.RecurrentKDAPrefillWrapper``. The public API is the thin class in
# ``flashinfer/kda.py``; everything that decides buffer layout, graph-replay
# invariants and backend routing lives here.

from __future__ import annotations

import threading
from typing import Optional

import torch

from ... import kda_prefill as _kda_prefill

__all__ = ["RecurrentKDAPrefillPlanner"]


class RecurrentKDAPrefillPlanner:
    """Fixed-address plan state for packed CuTe DSL recurrent-KDA prefill."""

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("RecurrentKDAPrefillWrapper requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.workspace = _kda_prefill.RecurrentKDAPrefillWorkspace(self.device)
        self.cu_seqlens_buf: Optional[torch.Tensor] = None
        self.seq_order_buf: Optional[torch.Tensor] = None
        self.cu_chunks_buf: Optional[torch.Tensor] = None
        self.num_sequences: Optional[int] = None
        self.total_tokens: Optional[int] = None
        self.planned = False
        self._lock = threading.Lock()

    def plan(self, cu_seqlens: torch.Tensor, *, non_blocking: bool = True) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "RecurrentKDAPrefillWrapper.plan must run outside CUDA graph capture"
            )
        if not isinstance(cu_seqlens, torch.Tensor):
            raise TypeError("cu_seqlens must be a torch.Tensor")
        if (
            cu_seqlens.dtype not in (torch.int32, torch.int64)
            or cu_seqlens.ndim != 1
            or not cu_seqlens.is_contiguous()
            or cu_seqlens.numel() < 2
        ):
            raise ValueError(
                "cu_seqlens must be a contiguous int32 or int64 tensor with "
                "at least two entries"
            )
        if cu_seqlens.is_cuda and cu_seqlens.device != self.device:
            raise ValueError(
                f"cu_seqlens must be on {self.device} or CPU, got {cu_seqlens.device}"
            )

        num_sequences = cu_seqlens.numel() - 1

        with self._lock:
            if self.num_sequences is None:
                self.num_sequences = num_sequences
                self.cu_seqlens_buf = torch.empty(
                    num_sequences + 1, dtype=torch.int64, device=self.device
                )
                self.seq_order_buf = torch.empty(
                    num_sequences, dtype=torch.int32, device=self.device
                )
                self.cu_chunks_buf = torch.empty(
                    num_sequences + 1, dtype=torch.int32, device=self.device
                )
            elif num_sequences != self.num_sequences:
                raise ValueError(
                    "the number of sequences is fixed after the first plan call: "
                    f"expected {self.num_sequences}, got {num_sequences}"
                )
            assert self.cu_seqlens_buf is not None
            assert self.seq_order_buf is not None
            assert self.cu_chunks_buf is not None
            self.cu_seqlens_buf.copy_(cu_seqlens, non_blocking=non_blocking)
            self.workspace.__dict__["_cute_dsl_cu_chunks"] = self.cu_chunks_buf
            self.workspace.__dict__["_cute_dsl_generate_planned_metadata"] = True
            self.planned = True

    def run(self, **kwargs):
        with self._lock:
            if not self.planned:
                raise RuntimeError("call plan before run")
            q = kwargs["q"]
            token_count = q.shape[0] * q.shape[1] if q.ndim == 4 else None
            if self.total_tokens is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "warm RecurrentKDAPrefillWrapper.run once before CUDA "
                        "graph capture"
                    )
                if token_count is None:
                    raise ValueError("q must be a rank-4 tensor")
                self.total_tokens = token_count
            elif token_count != self.total_tokens:
                raise ValueError(
                    "q token count is fixed after the first run: "
                    f"expected {self.total_tokens}, got "
                    f"{token_count if token_count is not None else 'invalid rank'}"
                )
            cu_seqlens = self.cu_seqlens_buf
            seq_order = self.seq_order_buf
        assert cu_seqlens is not None
        assert seq_order is not None

        # Resolved per call so tests and tracing can patch the stable facade.
        from ... import kda as _kda

        return _kda.recurrent_kda(
            cu_seqlens=cu_seqlens,
            seq_order=seq_order,
            prefill_workspace=self.workspace,
            backend="cute-dsl",
            **kwargs,
        )
