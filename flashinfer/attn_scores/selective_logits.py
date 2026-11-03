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
"""Public entry points for continuous selective TopK and the paged Full baseline."""

from __future__ import annotations

from typing import Literal

import torch

from ..api_logging import flashinfer_experimental_api, flashinfer_api
from ._math_policy import _MathMode


def _backend_call(instance, method, *args, **kwargs):
    return getattr(type(instance), method)(instance, *args, **kwargs)


class FP8PagedMQATopKWrapper:
    """Paged adapter for Full or selective continuous TopK.

    Each layer gathers paged keys and scales into fixed contiguous buffers,
    then calls one :class:`FP8MQATopKWrapper`. Plan once per step and prepare
    every paged pool before caller capture. Calls on a wrapper are serialized.
    """

    is_experimental = True
    _strategy: Literal["full", "selective"]
    _math_mode: _MathMode
    _compression_ratio: int

    def __new__(
        cls,
        *,
        strategy: Literal["full", "selective"],
        backend: Literal["v1", "v2"] = "v1",
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
        qkv_dtype: str = "fp8",
        logits_dtype: torch.dtype = torch.float32,
        model_config: str | None = None,
    ):
        if cls is FP8PagedMQATopKWrapper:
            if strategy not in ("full", "selective"):
                raise ValueError("strategy must be 'full' or 'selective'")
            if backend not in ("v1", "v2"):
                raise ValueError("backend must be 'v1' or 'v2'")
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("construct the wrapper outside capture")
            from ..experimental.attn_scores import _wrapper_type

            return object.__new__(_wrapper_type())
        return object.__new__(cls)

    @flashinfer_experimental_api(feature="FP8PagedMQATopKWrapper")
    def __init__(
        self,
        *,
        strategy: Literal["full", "selective"],
        backend: Literal["v1", "v2"] = "v1",
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
        qkv_dtype: str = "fp8",
        logits_dtype: torch.dtype = torch.float32,
        model_config: str | None = None,
    ) -> None:
        """Reserve capacities and fixed gather storage outside capture.

        Context counts original tokens; chunk size bounds total query rows.
        Supply a contiguous uint8 ``kv_layout`` exemplar, or ``num_pages`` and
        ``device``. Each fused page stores all key bytes followed by four scale
        bytes per key. FP8 uses D key bytes; MXFP4 uses D/2.
        The exemplar is not retained. Prepared layer pools stay alive until
        ``clear_pools``; all calls on one wrapper must be serialized.

        ``max_workspace_size`` bounds only the Full FP32 score arena (default
        512 MiB). It excludes gather storage, GVRv2 workspace and metadata.
        Full needs at least eight times the padded physical context in bytes,
        with a minimum width of max(8192, top_k). Selective validates a positive
        budget but ignores it. Unsupported encodings/configurations fail clearly."""
        _backend_call(
            self,
            "_initialize",
            strategy=strategy,
            backend=backend,
            qkv_dtype=qkv_dtype,
            logits_dtype=logits_dtype,
            model_config=model_config,
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
        """Stage original-token prefixes and the page map once per step.

        The map must cover ceil(floor(max_context_length/compression_ratio)
        /page_size) columns. Prepared plans refresh device metadata and gather
        indices without host reads; addresses stay fixed during caller capture.
        ``next_n`` optionally declares 1, 2 or 4 query rows per request.
        Conservative CPU ``host_length_hints`` are checked outside capture and
        forwarded to the continuous wrapper. Host planning errors invalidate
        the step; publish a valid plan before the next run.
        Invalid device prefixes/extents/pages set error bits and yield -1."""
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
        q_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather paged K/scales and delegate to the continuous wrapper.

        Every strategy uses the gather. Prepare each external layer pool before
        caller capture; warmed captured calls allocate no CUDA storage, compile
        no kernels and make no host reads. Consume the shared output before the
        next serialized layer, or provide a separate nonoverlapping ``out``."""
        return _backend_call(
            self, "run", q, weights, kv_fused, out=out, q_scale=q_scale
        )

    def prepare_pool(self, kv_fused):
        """Warm gathering and bind the continuous workspace outside capture.

        Optional for eager execution: the first eager run prepares a pool
        lazily. Prepare every pool explicitly before capturing its run.
        Pools share fixed contiguous K/scales and score scratch. Discard caller
        graphs before clearing their pool bindings."""
        return _backend_call(self, "prepare_pool", kv_fused)

    def clear_pools(self):
        """Synchronize and release pool bindings outside caller capture."""
        return _backend_call(self, "clear_pools")

    def input_buffers(self) -> dict[str, torch.Tensor]:
        """Return fixed Q/weight, original-prefix and page-map staging views."""
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


def _continuous_trace_dispatch(**kwargs):
    from ..experimental.attn_scores.trace import continuous_topk_trace_dispatch

    return continuous_topk_trace_dispatch(**kwargs)


class FP8MQATopKWrapper:
    """Experimental Full or selective TopK on continuous K (backend v1/v2).

    K concatenates requests in prefix order. FP8 uses K[N,D] and FP32 scales[N];
    MXFP4 v2 uses packed uint8 K[N,D/2] and UE8M0 uint8 scales[N,4].
    ``num_keys`` describes storage, while plan's prefixes describe live tokens
    before compression. It defaults to B * floor(max_context_length / ratio).
    MXFP4 storage must be padded to a multiple of 256 rows.

    Construct outside capture, plan once per step, and run once per layer.
    Prepare every v2 K/scale binding before capture. Keep this wrapper and all
    capture inputs alive until the caller discards its graphs. Calls on a
    wrapper must be serialized. Output is shared int32[Q, top_k] storage with
    request-local indices and -1 padding; ties need not have stable indices.
    """

    is_experimental = True

    @flashinfer_experimental_api(feature="FP8MQATopKWrapper")
    def __init__(
        self,
        *,
        max_context_length: int,
        max_chunk_size: int,
        max_batch_size: int,
        device,
        strategy="selective",
        backend="v1",
        num_heads=8,
        head_dim=128,
        top_k=512,
        math_mode="relu_strict",
        compression_ratio=1,
        num_keys=None,
        qkv_dtype="fp8",
        logits_dtype=torch.float32,
        model_config=None,
        max_workspace_size=None,
    ):
        if strategy not in ("full", "selective"):
            raise ValueError("strategy must be 'full' or 'selective'")
        if backend not in ("v1", "v2"):
            raise ValueError("backend must be 'v1' or 'v2'")
        if strategy == "full" and (
            backend != "v1"
            or qkv_dtype != "fp8"
            or logits_dtype != torch.float32
            or model_config is not None
        ):
            raise ValueError(
                "full supports backend='v1', FP8 inputs and FP32 logits without model_config"
            )
        if backend == "v1" and (
            qkv_dtype != "fp8"
            or logits_dtype != torch.float32
            or model_config is not None
        ):
            raise ValueError(
                "v1 supports FP8 inputs and FP32 logits without model_config"
            )
        from ._math_policy import _validate_math_mode
        from ..experimental.attn_scores import _continuous_plan_type

        options = dict(
            max_context_length=max_context_length,
            max_chunk_size=max_chunk_size,
            max_batch_size=max_batch_size,
            device=device,
            num_heads=num_heads,
            head_dim=head_dim,
            top_k=top_k,
            compression_ratio=compression_ratio,
            math_mode=_validate_math_mode(math_mode),
            num_keys=num_keys,
        )
        if backend == "v2":
            options.update(
                qkv_dtype=qkv_dtype,
                logits_dtype=logits_dtype,
                model_config=model_config,
            )
        if strategy == "full":
            options["max_workspace_size"] = max_workspace_size
        elif max_workspace_size is not None and (
            type(max_workspace_size) is not int or max_workspace_size <= 0
        ):
            raise ValueError("max_workspace_size must be a positive integer")
        self._capacity_plan = _continuous_plan_type(
            "full" if strategy == "full" else backend
        )(**options)
        self._compression_ratio = compression_ratio
        self._backend = backend
        self._strategy = strategy

    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        *,
        num_query_rows: int | None = None,
        next_n: int | None = None,
        host_length_hints: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Stage original-token prefixes int32[B+1], with no page table.

        For query i of request b, visible keys end at
        floor((KV_b - Q_b + i + 1) / compression_ratio).
        K stores floor(KV_b / compression_ratio) keys per request.
        Invalid live prefixes fail closed (error bits 1/2/4); inspect error
        outside capture. Planning is capturable after eager preparation.
        """
        return self._capacity_plan.plan(
            qo_indptr,
            kv_indptr,
            num_query_rows=num_query_rows,
            next_n=next_n,
            host_length_hints=host_length_hints,
        )

    @flashinfer_api(trace=_continuous_trace_dispatch)
    def run(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        k: torch.Tensor,
        *,
        k_scale: torch.Tensor,
        out: torch.Tensor | None = None,
        q_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score continuous K directly; no KV packing, page lookup or gather."""
        return self._capacity_plan.run(
            q, weights, k, k_scale=k_scale, out=out, q_scale=q_scale
        )

    def prepare_pool(self, k, *, k_scale):
        """Compile/capture a K/scale binding outside caller graph capture."""
        return self._capacity_plan.prepare_pool(k, k_scale=k_scale)

    def clear_pools(self):
        return self._capacity_plan.clear_pools()

    def input_buffers(self):
        """Stable Q/weight/prefix buffers (plus q_scale for MXFP4)."""
        return self._capacity_plan.input_buffers()

    @property
    def strategy(self):
        return self._strategy

    @property
    def math_mode(self):
        return self._capacity_plan.math_mode

    @property
    def compression_ratio(self):
        return self._compression_ratio

    @property
    def error(self):
        return self._capacity_plan.error

    @property
    def repair_error(self):
        return self._capacity_plan.repair_error

    @property
    def active_shape(self):
        return self._capacity_plan.active_shape


__all__.append("FP8MQATopKWrapper")
