# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Thin public entry point for experimental paged FP8 MQA TopK."""

from __future__ import annotations

from typing import Literal

import torch

from ..api_logging import flashinfer_experimental_api
from ._math_policy import _MathMode


def _backend_call(instance, method, *args, **kwargs):
    from ..experimental.attn_scores import _wrapper_type

    return getattr(_wrapper_type(), method)(instance, *args, **kwargs)


class FP8PagedMQATopKWrapper:
    """Experimental exact paged FP8 MQA TopK, with full or selective scoring.

    Both strategies use a capacity constructor, one ``plan`` per step and
    one ``run`` per layer. Calling the constructor is the explicit opt-in;
    no environment variable is required. The interface may change or be
    removed without deprecation. The ordinary paged scorer is unchanged.

    Construct after allocating the indexer KV cache and before graph capture.
    Keep the wrapper alive while caller graphs replay, and release wrappers
    outside capture. A capacity or layout change requires a new wrapper.
    See ``flashinfer/experimental/attn_scores/README.md`` for the contract,
    supported configurations, limitations and graduation requirements.
    """

    is_experimental = True
    _strategy: Literal["full", "selective"]
    _math_mode: _MathMode
    _compression_ratio: int

    def __new__(
        cls,
        *,
        strategy: Literal["full", "selective"],
        num_heads: int = 8,
        head_dim: int = 128,
        page_size: int = 128,
        top_k: int = 512,
        math_mode: str = "relu_strict",
        compression_ratio: int = 1,
        max_context_length: int | None = None,
        max_chunk_size: int | None = None,
        max_batch_size: int | None = None,
        kv_layout: torch.Tensor | None = None,
        num_pages: int | None = None,
        device: torch.device | str | int | None = None,
    ):
        if cls is FP8PagedMQATopKWrapper:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                if strategy == "selective":
                    raise RuntimeError(
                        "selective wrappers must be constructed outside capture"
                    )
                raise RuntimeError("construct the wrapper outside capture")
            from ..experimental.attn_scores import _wrapper_type

            return object.__new__(_wrapper_type())
        return object.__new__(cls)

    @flashinfer_experimental_api(feature="FP8PagedMQATopKWrapper")
    def __init__(
        self,
        *,
        strategy: Literal["full", "selective"],
        num_heads: int = 8,
        head_dim: int = 128,
        page_size: int = 128,
        top_k: int = 512,
        math_mode: str = "relu_strict",
        compression_ratio: int = 1,
        max_context_length: int | None = None,
        max_chunk_size: int | None = None,
        max_batch_size: int | None = None,
        kv_layout: torch.Tensor | None = None,
        num_pages: int | None = None,
        device: torch.device | str | int | None = None,
    ) -> None:
        """Reserve capacities and a KV layout outside capture.

        Supply positive ``max_context_length`` (original tokens),
        ``max_chunk_size`` (total query rows) and ``max_batch_size``. Give
        either an unretained ``kv_layout`` exemplar or ``num_pages`` plus
        ``device`` for a contiguous pool. Compilation remains lazy until
        ``prepare_pool`` or eager ``run``. Unsupported specializations fail
        at that first preparation. Both strategies are experimental.
        """
        _backend_call(
            self,
            "_initialize",
            strategy=strategy,
            num_heads=num_heads,
            head_dim=head_dim,
            page_size=page_size,
            top_k=top_k,
            math_mode=math_mode,
            compression_ratio=compression_ratio,
            max_context_length=max_context_length,
            max_chunk_size=max_chunk_size,
            max_batch_size=max_batch_size,
            kv_layout=kv_layout,
            num_pages=num_pages,
            device=device,
        )

    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        block_table: torch.Tensor,
        *,
        num_query_rows: int | None = None,
    ) -> None:
        """Publish prefixes and page maps once per step.

        Selective planning supports caller capture; full planning reads
        prefixes on the host and must run outside capture.
        """
        return _backend_call(
            self,
            "plan",
            qo_indptr,
            kv_indptr,
            block_table,
            num_query_rows=num_query_rows,
        )

    def run(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        kv_fused: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return request-local int32 TopK indices for one indexer layer.

        Normal instances dispatch directly to the backend method, without
        an additional experimental-warning or forwarding layer per run.
        """
        return _backend_call(self, "run", q, weights, kv_fused, out=out)

    def prepare_pool(self, kv_fused):
        """Optionally warm a selective layout; prepare each full-plan pool."""
        return _backend_call(self, "prepare_pool", kv_fused)

    def clear_pools(self):
        """Synchronize and release pool bindings outside caller capture."""
        return _backend_call(self, "clear_pools")

    def input_buffers(self) -> dict[str, torch.Tensor]:
        """Return fixed-address Q, weight, prefix and page-map staging views."""
        return _backend_call(self, "input_buffers")

    @property
    def strategy(self) -> Literal["full", "selective"]:
        """Return the constructor-owned scoring strategy."""
        return self._strategy

    @property
    def math_mode(self) -> _MathMode:
        """Return the immutable scoring arithmetic policy."""
        return self._math_mode

    @property
    def compression_ratio(self) -> int:
        """Return the immutable original-token to physical-key ratio."""
        return self._compression_ratio

    @property
    def error(self) -> torch.Tensor:
        """Return the device validation bitmask (extent/prefix/capacity/pages)."""
        from ..experimental.attn_scores import _wrapper_type

        return _wrapper_type().error.__get__(self, type(self))

    @property
    def repair_error(self) -> torch.Tensor:
        """Return the OR-accumulated repair flag, not a row count."""
        from ..experimental.attn_scores import _wrapper_type

        return _wrapper_type().repair_error.__get__(self, type(self))

    @property
    def active_shape(self) -> torch.Tensor:
        """Return the device active-row/request counts."""
        from ..experimental.attn_scores import _wrapper_type

        return _wrapper_type().active_shape.__get__(self, type(self))


__all__ = ["FP8PagedMQATopKWrapper"]
