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

    The API and backend are experimental. Construction is explicit opt-in and
    emits ``ExperimentalWarning`` once per process; no environment variable
    is required. The interface may change or disappear without deprecation.
    This feature does not participate in automatic routing, autotuning or
    trace-apply. Owner: TBD. Graduation requires a stable API and competitive
    decode performance. The ordinary paged scorers remain stable.

    Use this wrapper for paged FP8 MQA indexers in sparse attention. Both
    strategies require CuTe DSL and SM100 (CUDA >=12.8) or SM103 (CUDA >=12.9),
    FP8 E4M3 queries, fused paged FP8 KV and FP32 head weights. Supported page
    sizes are 32, 64 and 128 tokens, subject to specialization/resource checks.

    * ``full`` scores every key and runs GVRv2 using a shared score arena.
      TopK is 512, 1024 or 2048; compression ratio is 1.
    * ``selective`` samples thresholds, publishes bounded candidates and
      repairs failed rows exactly. TopK is 1..2048; ratio is 1 or 4. The caller
      supplies compressed KV for ratio 4 and receives compressed-slot indices.
      NaN scores rank as -inf; full retains its scorer's NaN behavior.

    Both use one ``plan`` per step and ``run`` per layer, returning request-local
    int32 indices with -1 padding. Preparation requires JIT compilation. Plans
    without host hints are device-only and caller-capturable; host hints are
    eager-only and only affect full-strategy sizing. The wrapper owns no parent
    graph. Under caller capture, selective exact repair uses a conditional
    node whose body is captured by the library. Multi-device execution and
    installed-cache-only operation are outside the qualification scope.

    Construct after allocating the indexer KV cache and before graph capture.
    Keep the wrapper alive while caller graphs replay, and release wrappers
    outside capture. A capacity or layout change requires a new wrapper.
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
        max_workspace_size: int | None = None,
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
        max_workspace_size: int | None = None,
        kv_layout: torch.Tensor | None = None,
        num_pages: int | None = None,
        device: torch.device | str | int | None = None,
    ) -> None:
        """Reserve capacities and a KV layout outside capture.

        Supply positive ``max_context_length`` (original tokens),
        ``max_chunk_size`` (total query rows) and ``max_batch_size``. Give
        either an unretained ``kv_layout`` exemplar or ``num_pages`` plus
        ``device`` for a contiguous pool. After ``plan``, either strategy can
        prepare a pool on its first eager ``run``. Optional ``prepare_pool``
        moves compilation and warmup out of that call. Unsupported selective
        specializations fail at preparation. Both strategies are experimental.

        ``max_workspace_size`` bounds only the full FP32 score arena, in
        bytes (default 512 MiB), shared by eager and caller-captured execution.
        It excludes GVRv2 workspace, metadata, inputs and outputs. Selective
        validates this positive integer but ignores it for scratch sizing.
        Full requires at least ``8 * max(8192, top_k,
        padded_seq_len(max_context_length))`` bytes for one query pair.
        Expanded page/hint tables and payload staging require additional
        capacity-dependent storage; see the backend README's memory formula.
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
            max_workspace_size=max_workspace_size,
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
        next_n: int | None = None,
        host_length_hints: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Publish prefixes and page maps once per step.

        Unhinted plans refresh device metadata without host reads or CUDA
        allocations and support caller capture after layout/pool preparation.
        Full uses the constructor's bounded shared score arena; in-capacity length
        and page-map changes preserve prepared bindings and buffer addresses.
        A full graph bucket is keyed by supplied Q rows, request count
        (``qo_indptr.numel() - 1``) and ``next_n``. Eagerly plan, then run or
        explicitly prepare every layer pool for that bucket before capture.
        The page map requires at least
        ``ceil((max_context_length // compression_ratio) / page_size)``
        columns. Every request or expanded row table must fit int32 indexing:
        ``rows * page_columns < 2**31``.
        ``next_n`` optionally declares exactly 1, 2 or 4
        query rows per request. A device-side mismatch invalidates the step
        and produces -1 output. Selective validates but does not use the hint
        to change its scoring layout; full uses direct uniform views.
        ``host_length_hints=(query_prefix_cpu, kv_bounds_cpu)`` optionally
        selects eager full stripes without packing. These contiguous CPU
        int32/int64 tensors give an exact query prefix and conservative KV
        length bounds. Hints are checked against device inputs and rejected
        during capture before CUDA work. Selective can bypass sampling eagerly
        when the validated physical bound fits the nonsampling capacity;
        otherwise its sample work follows device lengths. Hinted plans also refresh the
        capacity metadata used by existing caller graphs.
        A host-side planning error invalidates the step; publish a valid plan
        before calling ``run`` again.
        """
        return _backend_call(
            self,
            "plan",
            qo_indptr,
            kv_indptr,
            block_table,
            num_query_rows=num_query_rows,
            next_n=next_n,
            host_length_hints=host_length_hints,
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
        Either strategy prepares a missing pool binding on its first eager
        run; warmed bindings incur no preparation. Caller capture requires
        an earlier eager run or explicit ``prepare_pool`` for the binding.
        Full requires an unhinted active plan when recording ``run`` inside
        caller capture; a ``host_length_hints`` plan is eager-only.
        """
        return _backend_call(self, "run", q, weights, kv_fused, out=out)

    def prepare_pool(self, kv_fused):
        """Compile, warm and bind one layer pool outside capture.

        Optional for eager execution: either strategy also prepares lazily
        on its first eager run. Full needs an eager plan followed by either
        preparation or an eager run for every
        ``(supplied Q rows, request count, next_n)`` graph bucket and pool
        before capture.
        First-pool preparation warms every pitch bucket, even without host
        hints. Score/TopK scratch is shared across serialized layer pools;
        compilation and warmup are setup costs, paid here or on the first
        eager run, not on warmed runs.
        """
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
