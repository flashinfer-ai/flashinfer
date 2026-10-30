# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Address-stable step inputs for experimental capacity-planned TopK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import weakref

import torch

from ._selective_contract import _SAMPLE_HISTOGRAM_BINS

if TYPE_CHECKING:
    from .selective_logits import _FP8PagedMQASelectiveTopKEngine


def _bind_metadata_module_lifetime(compiled):
    """Unload private metadata libraries when their callable owner is released.

    CuTe 4.7 can leave the JIT module in a reference cycle after its compiled
    callable is gone. Letting cyclic GC unload that library during an unrelated
    caller capture invalidates the capture. The finalizer owns only the module,
    not the callable or plan, and module.unload() is idempotent.
    """
    module = getattr(compiled, "jit_module", None)
    if module is not None:
        weakref.finalize(compiled, module.unload)
    return compiled


@dataclass(frozen=True)
class _SelectiveCapacity:
    """Immutable host-side storage bounds for one selective planning bucket.

    Context lengths count original tokens; physical lengths are computed by
    flooring each request separately at the selected compression ratio.
    ``max_chunk_size`` bounds supplied Q storage across the entire step, while
    ``max_batch_size`` bounds the number of requests. Fields are positive int32
    values, with page size 32/64/128 and compression ratio 1/4; bools are rejected.
    Scorer specialization and device resource checks occur when building the chain.
    """

    num_heads: int
    head_dim: int
    page_size: int
    compression_ratio: int
    max_context_length: int
    max_chunk_size: int
    max_batch_size: int

    def __post_init__(self) -> None:
        for name in (
            "num_heads",
            "head_dim",
            "max_context_length",
            "max_chunk_size",
            "max_batch_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value < 2**31:
                raise ValueError(f"{name} must be a positive int32 value")
        if self.head_dim % 32:
            raise ValueError("head_dim must be divisible by 32")
        if type(self.page_size) is not int or self.page_size not in (32, 64, 128):
            raise ValueError("page_size must be 32, 64 or 128")
        if type(self.compression_ratio) is not int or self.compression_ratio not in (
            1,
            4,
        ):
            raise ValueError("compression_ratio must be 1 or 4")


class _CapacityStepInputs:
    """Own stable payload/prefix buffers and sanitized per-step row metadata.

    ``stage_step`` and ``stage_payload`` check tensor metadata and aliases, then enqueue copies without
    reading device values or synchronizing. It is capturable: the external
    source addresses and supplied extents then belong to the caller's graph.
    ``input_buffers`` exposes Q/weights/prefix storage for zero-copy staging;
    callers still pass those views to plan/run
    to publish their row/batch extents. The supplied Q extent is an upper bound,
    not an equality constraint on the last query prefix; inactive rows remain -1
    in the owning chain's output.

    ``refresh`` validates prefixes and live page indices before publishing
    sanitized physical prefixes, row ends and owners. All metadata addresses
    remain stable. Launch refresh, optional sampling refresh and score consumers
    in order on one stream. Construction and sampling preparation are outside
    capture; repeated sampling preparation replaces its buffers and requires
    rebuilding consumers.

    ``error`` resets each step. Bit 1 means invalid staged extents and suppresses
    other checks. Prefix bit 2 and capacity bit 4 may combine. Page-index bit 8 is
    checked only when those earlier checks pass. Any error publishes zero active
    rows/prefixes, zero row ends and owner -1. Callers inspect this device status;
    the launch does not raise a synchronized Python error. This component itself
    launches no scorer and produces no TopK output.
    """

    def __init__(self, capacity: _SelectiveCapacity, *, device, num_pages):
        from .selective_logits import _require_cute

        _require_cute()
        from cutlass.cute.runtime import from_dlpack

        from .kernels.selective_logits_metadata import SelectiveCapacityRefreshKernel

        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("capacity device must be CUDA")
        if type(num_pages) is not int or not 0 <= num_pages < 2**31:
            raise ValueError("num_pages must be a nonnegative int32 value")
        columns = (
            capacity.max_context_length // capacity.compression_ratio
            + capacity.page_size
            - 1
        ) // capacity.page_size
        if columns * capacity.max_batch_size >= 2**31:
            raise ValueError("capacity page table must fit int32 indexing")
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("capacity buffers must be prepared outside capture")
            device = torch.device("cuda", torch.cuda.current_device())
        self.capacity = capacity
        self.num_pages = num_pages
        self.device = device
        self.block_table = torch.zeros(
            (capacity.max_batch_size, columns), dtype=torch.int32, device=device
        )
        self.q = torch.empty(
            (capacity.max_chunk_size, capacity.num_heads, capacity.head_dim),
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        self.weights = torch.empty(
            (capacity.max_chunk_size, capacity.num_heads),
            dtype=torch.float32,
            device=self.device,
        )
        prefix_shape = (capacity.max_batch_size + 1,)
        self.qo_indptr = torch.zeros(
            prefix_shape, dtype=torch.int32, device=self.device
        )
        self.kv_indptr = torch.zeros_like(self.qo_indptr)
        self.step_shape = torch.zeros(2, dtype=torch.int32, device=self.device)
        self._written_step_shape: tuple[int, int] | None = None
        self._skip_repeated_step_shape = True
        self.safe_qo_indptr = torch.zeros_like(self.qo_indptr)
        self.physical_kv_indptr = torch.zeros_like(self.qo_indptr)
        self.active_shape = torch.zeros_like(self.step_shape)
        self.row_ends = torch.zeros(
            capacity.max_chunk_size, dtype=torch.int32, device=self.device
        )
        self.row_requests = torch.full_like(self.row_ends, -1)
        self.error = torch.zeros(1, dtype=torch.int32, device=self.device)
        # One page-check flag per check CTA; every refresh overwrites them all.
        check_ctas = (
            2 * torch.cuda.get_device_properties(self.device).multi_processor_count
        )
        self.page_flags = torch.zeros(check_ctas, dtype=torch.int32, device=self.device)
        tensors = (
            self.qo_indptr,
            self.kv_indptr,
            self.step_shape,
            self.block_table,
            self.safe_qo_indptr,
            self.physical_kv_indptr,
            self.active_shape,
            self.row_ends,
            self.row_requests,
            self.error,
            self.page_flags,
        )
        self._args = tuple(from_dlpack(tensor, assumed_align=4) for tensor in tensors)
        self._refresh_kernel = SelectiveCapacityRefreshKernel(
            capacity.max_context_length,
            capacity.max_chunk_size,
            capacity.max_batch_size,
            capacity.page_size,
            capacity.compression_ratio,
            num_pages,
            check_ctas,
        )
        self._compiled = None
        self._next_n = 0
        from .host_length_hints import _HostHintValidation

        self._host_hint_validation = _HostHintValidation(capacity, self.device)

    def input_buffers(self) -> dict[str, torch.Tensor]:
        return dict(
            q=self.q,
            weights=self.weights,
            qo_indptr=self.qo_indptr,
            kv_indptr=self.kv_indptr,
            block_table=self.block_table,
        )

    def _payload_copies(self, q, weights):
        """Validate layer payload layouts without enqueuing any copies."""
        c = self.capacity
        if not all(isinstance(t, torch.Tensor) for t in (q, weights)):
            raise ValueError("step inputs must be tensors")
        if (
            q.device != self.device
            or q.dtype != self.q.dtype
            or q.ndim != 3
            or q.shape[1:] != self.q.shape[1:]
            or q.shape[0] > c.max_chunk_size
            or not q.is_contiguous()
            or weights.device != self.device
            or weights.dtype != torch.float32
            or weights.shape != (q.shape[0], c.num_heads)
            or not weights.is_contiguous()
        ):
            raise ValueError("Q and weights must fit the planned contiguous shapes")
        return (
            (self.q[: q.shape[0]], q),
            (self.weights[: q.shape[0]], weights),
        )

    def _prefix_copies(self, qo_indptr, kv_indptr):
        """Validate step prefix layouts without reading their device values."""
        if not all(isinstance(t, torch.Tensor) for t in (qo_indptr, kv_indptr)):
            raise ValueError("step inputs must be tensors")
        if (
            qo_indptr.device != self.device
            or kv_indptr.device != self.device
            or qo_indptr.dtype != torch.int32
            or kv_indptr.dtype != torch.int32
            or qo_indptr.ndim != 1
            or kv_indptr.shape != qo_indptr.shape
            or not 1 <= qo_indptr.numel() <= self.capacity.max_batch_size + 1
            or not qo_indptr.is_contiguous()
            or not kv_indptr.is_contiguous()
        ):
            raise ValueError("indptrs must be matching contiguous CUDA int32 [B+1]")
        return (
            (self.qo_indptr[: qo_indptr.numel()], qo_indptr),
            (self.kv_indptr[: kv_indptr.numel()], kv_indptr),
        )

    def _copy_inputs(self, copies) -> None:
        """Reject all cross-aliases before enqueuing the first staging copy."""
        from .selective_logits import _byte_range, _overlap

        # A staging copy cannot overwrite another source before it is consumed.
        for destination, source in copies:
            for owner in self.input_buffers().values():
                if _overlap(_byte_range(source), _byte_range(owner)) and not (
                    owner.data_ptr() == destination.data_ptr() == source.data_ptr()
                ):
                    raise ValueError("step inputs must not cross-alias staging buffers")
        # All host metadata/alias checks precede every copy.
        for destination, source in copies:
            if destination.data_ptr() != source.data_ptr():
                destination.copy_(source)

    def stage_payload(self, q, weights) -> None:
        """Stage one layer without changing step metadata."""
        self._copy_inputs(self._payload_copies(q, weights))

    def stage_step(
        self, qo_indptr, kv_indptr, block_table, *, num_query_rows, next_n=None
    ):
        """Stage shared prefixes/map and publish the supplied Q extent."""
        if next_n is not None and (type(next_n) is not int or next_n not in (1, 2, 4)):
            raise ValueError("next_n must be None, 1, 2 or 4")
        if type(num_query_rows) is not int or not (
            0 <= num_query_rows <= self.capacity.max_chunk_size
        ):
            raise ValueError("num_query_rows must fit max_chunk_size")
        copies = self._prefix_copies(qo_indptr, kv_indptr)
        if (
            not isinstance(block_table, torch.Tensor)
            or block_table.device != self.device
            or block_table.dtype != torch.int32
            or block_table.ndim != 2
            or block_table.shape[0] < qo_indptr.numel() - 1
            or block_table.shape[0] > self.capacity.max_batch_size
            or block_table.shape[1] < self.block_table.shape[1]
            or not block_table.is_contiguous()
        ):
            raise ValueError(
                "block_table must cover the planned page columns and fit the batch"
            )
        self._copy_inputs(
            (
                *copies,
                (
                    self.block_table[: block_table.shape[0]],
                    block_table[:, : self.block_table.shape[1]],
                ),
            )
        )
        step_shape = (num_query_rows, qo_indptr.numel() - 1)
        self._next_n = 0 if next_n is None else next_n
        # Replayed captures can write a shape without updating the host cache.
        # Once captured, always publish it, including on repeated eager steps.
        if torch.cuda.is_current_stream_capturing():
            self._skip_repeated_step_shape = False
        if not self._skip_repeated_step_shape or step_shape != self._written_step_shape:
            # CUDA is little-endian: one int64 fill publishes both int32 fields.
            self.step_shape.view(torch.int64).fill_(step_shape[0] | step_shape[1] << 32)
            self._written_step_shape = step_shape

    def refresh(self) -> None:
        from .selective_logits import _current_stream

        self.prepare()
        assert self._compiled is not None
        with torch.cuda.device(self.device):
            self._compiled(*self._args, self._next_n, _current_stream(self.device))
            self._host_hint_validation.check(self)

    def prepare_sampling(self, top_k: int) -> None:
        """Allocate the capacity-sized direct-page sample metadata."""
        from cutlass.cute.runtime import from_dlpack

        from .kernels.selective_logits_metadata import (
            SelectiveCapacitySampleMetadataKernel,
        )
        from .selective_logits import (
            _SAMPLE_CAPACITY,
            _candidate_target,
            _validate_selective_top_k,
        )

        _validate_selective_top_k(top_k)
        sample_pages = _SAMPLE_CAPACITY // self.capacity.page_size
        if self.capacity.max_batch_size * sample_pages >= 2**31:
            raise ValueError("capacity sample-page metadata must fit int32 indexing")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("sampling buffers must be prepared outside capture")
        self.sample_page_indices = torch.zeros(
            (self.capacity.max_batch_size, sample_pages),
            dtype=torch.int64,
            device=self.device,
        )
        self.sample_ends = torch.zeros_like(self.row_ends)
        self.sample_ranks = torch.zeros_like(self.row_ends)
        self._sample_args = tuple(
            from_dlpack(tensor, assumed_align=4)
            for tensor in (
                self.physical_kv_indptr,
                self.safe_qo_indptr,
                self.row_ends,
                self.row_requests,
                self.sample_page_indices,
                self.sample_ends,
                self.sample_ranks,
            )
        )
        self._sample_kernel = SelectiveCapacitySampleMetadataKernel(
            self.capacity.page_size, _candidate_target(top_k), top_k=top_k
        )
        self._sample_compiled = None

    def _set_sample_group_rows(self, group_rows: int, *, refresh: bool = True) -> None:
        """Bind group-consistent counts before compiling sparse SAMPLE consumers."""
        if self._sample_kernel.group_rows != group_rows:
            self._sample_kernel = type(self._sample_kernel)(
                self._sample_kernel.page_size,
                self._sample_kernel.target_candidates,
                group_rows,
                top_k=self._sample_kernel.top_k,
            )
            self._sample_compiled = None
            if refresh:
                self.refresh_sampling()

    @property
    def is_prepared(self) -> bool:
        return self._compiled is not None and (
            not hasattr(self, "_sample_kernel") or self._sample_compiled is not None
        )

    def prepare(self) -> None:
        """Compile metadata consumers outside capture without moving their buffers."""
        from .selective_logits import _compile_for_device, _current_stream

        if self.is_prepared:
            return
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("metadata must be prepared outside capture")
            compiled = self._compiled
            if compiled is None:
                compiled = _bind_metadata_module_lifetime(
                    _compile_for_device(
                        self.device,
                        self._refresh_kernel,
                        *self._args,
                        0,
                        _current_stream(self.device),
                    )
                )
            sample_compiled = getattr(self, "_sample_compiled", None)
            if hasattr(self, "_sample_kernel") and sample_compiled is None:
                sample_compiled = _bind_metadata_module_lifetime(
                    _compile_for_device(
                        self.device,
                        self._sample_kernel,
                        *self._sample_args,
                        _current_stream(self.device),
                    )
                )
        # Publish only after every required compilation succeeds, so a failed
        # first preparation can be retried with the same staging addresses.
        self._compiled = compiled
        self._sample_compiled = sample_compiled

    def refresh_sampling(self) -> None:
        from .selective_logits import _current_stream

        self.prepare()
        assert self._sample_compiled is not None
        with torch.cuda.device(self.device):
            self._sample_compiled(*self._sample_args, _current_stream(self.device))


class _CapacitySelectiveTopKPlan:
    """Capacity-sized selective chain with stable staging and pool bindings.

    Use one plan per concurrent stream; calls on one plan must be serialized.
    ``max_context_length`` counts original tokens.
    Sampling stays enabled when the physical capacity exceeds 6144, even on
    shorter runtime steps. Smaller physical capacities publish all eligible keys.
    """

    def __init__(
        self,
        *,
        max_context_length,
        max_chunk_size,
        max_batch_size,
        kv_layout=None,
        num_pages=None,
        device=None,
        num_heads: int = 8,
        head_dim: int = 128,
        page_size: int = 128,
        top_k: int = 512,
        compression_ratio: int = 1,
        math_mode: str = "relu_strict",
    ):
        """Allocate one workspace for fixed capacities and one KV pool layout.

        The layout comes from an unretained ``kv_layout`` exemplar, or from
        ``num_pages`` and ``device`` as a contiguous pool. Scorer
        allocation/compilation is completed by the first prepare_pool.
        Subsequent equal-layout pools add only launch bindings, not another
        score, candidate or repair workspace. Construct outside capture.
        """
        from ...attn_scores._math_policy import _validate_math_mode
        from .selective_logits import _SAMPLE_CAPACITY, _validate_selective_top_k

        capacity = _SelectiveCapacity(
            num_heads,
            head_dim,
            page_size,
            compression_ratio,
            max_context_length,
            max_chunk_size,
            max_batch_size,
        )
        _validate_selective_top_k(top_k)
        if capacity.max_batch_size * _SAMPLE_CAPACITY >= 2**31:
            raise ValueError("sample virtual prefixes must fit int32")
        self._top_k = top_k
        self._math_mode = _validate_math_mode(math_mode)
        layout, device, num_pages = self._resolve_layout(
            capacity, kv_layout, num_pages, device
        )
        self._pool_bindings: dict = {}
        self._protected_ranges: tuple[tuple[int, int], ...] = ()
        self._engine: _FP8PagedMQASelectiveTopKEngine | None = None
        self._step_prepared = False
        self._eager_skip_sampling = False
        self._state = _CapacityStepInputs(capacity, device=device, num_pages=num_pages)
        self._state.prepare_sampling(self.top_k)
        self._selected = torch.full(
            (capacity.max_chunk_size, self.top_k),
            -1,
            device=device,
            dtype=torch.int32,
        )
        self._repair_error = torch.zeros(1, device=device, dtype=torch.int32)
        self._pool_layout = layout
        self._capacity = capacity
        self._cache_protected_ranges()

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def math_mode(self) -> str:
        return self._math_mode

    @staticmethod
    def _layout(tensor):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("pool bindings must be tensors")
        return (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
        )

    @classmethod
    def _resolve_layout(cls, capacity, kv_layout, num_pages, device):
        """Return the pool layout, device and page count without retaining tensors.

        An explicit page count describes a contiguous pool with standard strides;
        pass an exemplar for pools whose singleton axes carry other strides.
        """
        from .selective_logits import _validate_selective_device

        page, dim = capacity.page_size, capacity.head_dim
        if kv_layout is not None:
            if num_pages is not None or device is not None:
                raise ValueError("pass either kv_layout or num_pages and device")
            layout = cls._layout(kv_layout)
            if (
                not kv_layout.is_cuda
                or kv_layout.dtype != torch.uint8
                or not kv_layout.is_contiguous()
                or kv_layout.ndim != 4
                or tuple(kv_layout.shape[1:]) != (page, 1, dim + 4)
                or not 0 <= kv_layout.shape[0] < 2**31
            ):
                raise ValueError("kv_layout must be contiguous CUDA uint8 paged KV")
            device, num_pages = kv_layout.device, kv_layout.shape[0]
        else:
            if num_pages is None or device is None:
                raise ValueError("selective requires kv_layout or num_pages and device")
            if type(num_pages) is not int or not 0 <= num_pages < 2**31:
                raise ValueError("num_pages must be a nonnegative int32 value")
            device = torch.device(device)
            if device.type != "cuda":
                raise ValueError("device must be CUDA")
            layout = None
        _validate_selective_device(device)
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "selective wrappers must be constructed outside capture"
                )
            device = torch.device("cuda", torch.cuda.current_device())
        if layout is None:
            row = dim + 4
            layout = (
                (num_pages, page, 1, row),
                (page * row, row, row, 1),
                torch.uint8,
                device,
            )
        return layout, device, num_pages

    def _build_engine(self, kv_fused) -> None:
        from dataclasses import replace
        from ...utils import get_device_sm_count

        from .selective_logits import (
            _SAMPLE_CAPACITY,
            _FP8PagedMQASelectiveTopKEngine,
            _prepare_fp8_mqa_selective_logits,
        )

        capacity, state = self._capacity, self._state
        block_table = state.block_table
        num_sms = get_device_sm_count(state.device)
        if (
            kv_fused.shape[0] == 0
            or capacity.max_context_length // capacity.compression_ratio == 0
        ):
            return
        # Construction geometry selects a reusable capacity family. It is never used
        # as runtime metadata, and does not assert that a bucket is all-Q1.
        rows = min(
            capacity.max_chunk_size,
            capacity.max_batch_size * capacity.max_context_length,
        )
        # Accepted prefixes require Q_i <= KV_i <= max_context_length, so
        # active rows cannot exceed this bound even with larger Q storage.
        batches = capacity.max_batch_size
        q_lengths = [
            rows // batches + (request < rows % batches) for request in range(batches)
        ]
        q_prefix, original_kv_prefix, physical_prefix = [0], [0], [0]
        row_ends: list[int] = []
        spare_tokens = 2**31 - 1 - rows
        for q_length in q_lengths:
            extra = min(capacity.max_context_length - q_length, spare_tokens)
            kv_length = q_length + extra
            spare_tokens -= extra
            q_prefix.append(q_prefix[-1] + q_length)
            original_kv_prefix.append(original_kv_prefix[-1] + kv_length)
            physical_prefix.append(
                physical_prefix[-1] + kv_length // capacity.compression_ratio
            )
            row_ends.extend(
                (kv_length - q_length + row + 1) // capacity.compression_ratio
                for row in range(q_length)
            )
        qo = torch.tensor(q_prefix, device=state.device, dtype=torch.int32)
        kv = torch.tensor(original_kv_prefix, device=state.device, dtype=torch.int32)
        self._q, self._weights = state.q[:rows], state.weights[:rows]
        engine = _FP8PagedMQASelectiveTopKEngine(math_mode=self.math_mode)
        engine._allocate_selective_plan(
            self._q,
            kv_fused,
            self._weights,
            qo,
            kv,
            block_table,
            q_prefix=q_prefix,
            kv_prefix=physical_prefix,
            logical_ends_host=row_ends,
            top_k=self.top_k,
            num_heads=capacity.num_heads,
            head_dim=capacity.head_dim,
            num_sms=num_sms,
            compression_ratio=capacity.compression_ratio,
            capacity_bounds=True,
        )
        self._engine = engine
        assert engine._exact_error is not None
        engine._cu_q, engine._cu_kv = state.safe_qo_indptr, state.kv_indptr
        engine._physical_cu_kv = state.physical_kv_indptr
        engine._row_ends = state.row_ends[:rows]
        engine._sample_ends = state.sample_ends[:rows]
        engine._sample_ranks = state.sample_ranks[:rows]
        engine._sample_page_indices = state.sample_page_indices
        engine._sample_gather_views = None
        engine._selected = self._selected[:rows]
        engine._needs_sampling = self._needs_sampling
        if not engine._needs_sampling:
            # This capacity cannot admit a row larger than the candidate arena.
            # No SAMPLE launch writes these policies, and candidate/repair only
            # read them. Their all-key policy therefore survives every layer.
            engine._thresholds.fill_(float("-inf"))
            engine._sample_ladder.fill_(float("-inf"))
            engine._selected_bins.fill_(_SAMPLE_HISTOGRAM_BINS - 1)
        engine._candidate_prepared = replace(
            engine._candidate_prepared, causal_bounds=False
        )
        self._sample_cu_kv = (
            torch.arange(batches + 1, device=state.device, dtype=torch.int32)
            * _SAMPLE_CAPACITY
        )
        engine._sample_block_table = torch.zeros(
            state.sample_page_indices.shape, device=state.device, dtype=torch.int32
        )
        engine._sample_prepared = _prepare_fp8_mqa_selective_logits(
            self._q,
            kv_fused,
            self._weights,
            state.safe_qo_indptr,
            self._sample_cu_kv,
            block_table,
            row_starts=engine._repair_starts,
            row_ends=engine._sample_ends,
            num_sms=num_sms,
            num_heads=capacity.num_heads,
            head_dim=capacity.head_dim,
            query_tile=engine._sample_prepared.query_tile,
            math_mode=self.math_mode,
            _capacity_bounds=True,
        )
        # Loader groups begin at request-local query prefixes. The count/rank
        # refresh and loader use the same complete-query-tile group size.
        from .selective_logits import (
            _select_sparse_sample_group_rows,
            _sparse_sample_gather4_serves,
        )

        if engine._needs_sampling and _sparse_sample_gather4_serves(
            kv_fused.shape[0], kv_fused.stride(0), capacity.page_size, capacity.head_dim
        ):
            state._set_sample_group_rows(
                _select_sparse_sample_group_rows(engine._sample_prepared.query_tile)
            )
            engine._sample_sparse_metadata = (
                engine._row_ends,
                engine._sample_ends,
                state.safe_qo_indptr,
                state.physical_kv_indptr,
            )
        # Bind auxiliary kernels only after all stable metadata/output owners
        # have their final addresses. No predecessor graph is captured here.
        engine._compile_auxiliary_launches()
        engine._prepare_repair_if(state.device)
        if engine._compact_repair:
            engine._compile_compact_repair_launches(self._q, self._weights)
        engine._precompile_score_launches(self._q, kv_fused, self._weights, block_table)
        # Materialize the remaining FFI and PyTorch kernels before the first
        # capture. Empty radix rows read no score payloads; in particular this
        # must not score the uninitialized query/weight staging buffers.
        empty_lengths = torch.zeros_like(engine._repair_lengths)
        empty_active = torch.zeros_like(
            engine._repair_active_count
            if engine._compact_repair
            else engine._repair_flags
        )
        engine._repair_topk_compiled(
            engine._repair_scores,
            engine._topk_row_states,
            empty_lengths,
            engine._repair_topk_positions_i32,
            None,
            empty_active,
        )
        torch._foreach_zero_(
            [
                engine._candidate_counts,
                engine._repair_counts,
                engine._repair_flags,
                engine._second_repair_flags,
                engine._repair_starts,
                engine._repair_ends,
                engine._exact_error,
            ]
        )
        self._repair_error.bitwise_or_(engine._exact_error)

    @property
    def _needs_sampling(self) -> bool:
        from .selective_logits import _CANDIDATE_CAPACITY

        capacity = self._capacity
        assert capacity is not None
        # Device validation bounds each original request by max_context_length.
        # Per-request flooring therefore bounds every causal physical row too.
        return (
            capacity.max_context_length // capacity.compression_ratio
            > _CANDIDATE_CAPACITY
        )

    def _refresh_schedules(self) -> None:
        from .selective_logits import _refresh_fp8_mqa_selective_logits

        state, engine = self._state, self._engine
        if engine is None:
            return
        engine._repair_starts.zero_()
        _refresh_fp8_mqa_selective_logits(
            engine._candidate_prepared,
            state.safe_qo_indptr,
            state.physical_kv_indptr,
            row_starts=engine._repair_starts,
            row_ends=engine._row_ends,
        )
        if engine._needs_sampling:
            _refresh_fp8_mqa_selective_logits(
                engine._sample_prepared,
                state.safe_qo_indptr,
                self._sample_cu_kv,
                row_starts=engine._repair_starts,
                row_ends=engine._sample_ends,
            )

    def plan(
        self,
        qo_indptr,
        kv_indptr,
        block_table,
        *,
        num_query_rows=None,
        next_n=None,
        host_length_hints=None,
    ):
        """Refresh prefixes, map and schedules once for all serialized layers.

        num_query_rows is the supplied Q storage extent, not a device value.
        It defaults to max_chunk_size. Each layer must supply that many rows;
        the final device query prefix may be smaller, leaving inactive -1 rows.
        Status is cleared here and survives every layer until the next step.
        """
        self._step_prepared = False
        if num_query_rows is None:
            num_query_rows = self._capacity.max_chunk_size
        from .host_length_hints import _validate_host_length_hints

        validated_hints = _validate_host_length_hints(
            host_length_hints, qo_indptr.numel() - 1, self._capacity, num_query_rows
        )
        from ._selective_contract import _CANDIDATE_CAPACITY

        self._eager_skip_sampling = validated_hints is not None and (
            max(validated_hints[1], default=0) // self._capacity.compression_ratio
            <= _CANDIDATE_CAPACITY
        )
        with torch.cuda.device(self._state.device):
            capturing = torch.cuda.is_current_stream_capturing()
            if capturing and (
                self._layout_needs_engine() or not self._state.is_prepared
            ):
                raise RuntimeError(
                    "prepare_pool or one eager run must precede capturing plan"
                )
            self._state._host_hint_validation.stage(host_length_hints)
            self._state.stage_step(
                qo_indptr,
                kv_indptr,
                block_table,
                num_query_rows=num_query_rows,
                next_n=next_n,
            )
            self._check_caller_capture(capturing, "plan input copies")
            self._repair_error.zero_()
            self._check_caller_capture(capturing, "plan repair-status reset")
            # Before the first pool is prepared, planning only stages inputs.
            # Pool preparation validates this staged step before any scoring.
            if self._state.is_prepared:
                self._state.refresh()
                self._check_caller_capture(capturing, "plan metadata refresh")
                if self._needs_sampling:
                    self._state.refresh_sampling()
                    self._check_caller_capture(capturing, "plan sampling refresh")
                self._refresh_schedules()
                self._check_caller_capture(capturing, "plan schedule refresh")
        self._supplied_rows = num_query_rows
        self._supplied_batch = qo_indptr.numel() - 1
        self._step_prepared = True

    def _check_caller_capture(self, capturing, step):
        """Name the first invalid host boundary without synchronizing the GPU."""
        if capturing:
            from .selective_logits import _check_capture_status

            _check_capture_status(torch.cuda.current_stream(self._state.device), step)

    def _execute_layer(self, kv_fused) -> None:
        """Score one pool using the shared step metadata and tensor workspace."""
        capturing = torch.cuda.is_current_stream_capturing()
        engine = self._engine
        if engine is None:
            self._selected.fill_(-1)
            return
        # Packed TopK and repair overwrite every scorer row, including invalid
        # steps. Clear only the uncovered tail: callers can mutate the shared
        # returned buffer, so the initial padding cannot protect later runs.
        covered_rows = self._q.shape[0]
        if covered_rows < self._selected.shape[0]:
            self._selected[covered_rows:].fill_(-1)
        # Every layer owns new populations and recovery decisions. These resets
        # do not clear the step validation status or accumulated repair status.
        torch._foreach_zero_(
            [
                engine._candidate_counts,
                engine._repair_counts,
                engine._repair_flags,
                engine._second_repair_flags,
                engine._repair_starts,
                engine._repair_ends,
                engine._exact_error,
            ]
        )
        self._check_caller_capture(capturing, "layer scratch resets")
        engine._run_impl(
            self._q,
            kv_fused,
            self._weights,
            self._state.block_table,
            _candidate_counts_precleared=True,
            _sampling_policy_initialized=not engine._needs_sampling,
            _skip_sampling=self._eager_skip_sampling and not capturing,
        )
        self._repair_error.bitwise_or_(engine._exact_error)
        self._check_caller_capture(capturing, "layer repair-status merge")

    def _layout_needs_engine(self) -> bool:
        """Whether scoring this layout still needs its first engine build."""
        return (
            self._engine is None
            and self._state.num_pages > 0
            and self._capacity.max_context_length // self._capacity.compression_ratio
            > 0
        )

    def _pool_key(self, kv_fused):
        # Validate on every call, including hits. Page count is a specialization
        # of the map validator, not just a bound on the pool's byte size.
        layout = self._layout(kv_fused)
        if layout != self._pool_layout or not kv_fused.is_contiguous():
            raise ValueError("KV pool layout must match the constructed layout")
        return kv_fused.data_ptr(), layout

    def _prepared_launches(self):
        if self._engine is None:
            return ()
        return tuple(
            (prepared, prepared._launches.copy())
            for prepared in (
                self._engine._sample_prepared,
                self._engine._candidate_prepared,
                self._engine._repair_prepared,
                self._engine._prepared,
            )
        )

    def _cache_protected_ranges(self) -> None:
        """Merge output-protected intervals only when workspace/bindings change."""
        from .selective_logits import _byte_range

        owners = [self._selected, self._repair_error, *self._state.__dict__.values()]
        if self._engine is not None:
            owners.extend(self._engine.__dict__.values())
        owners.extend(binding[0] for binding in self._pool_bindings.values())
        ranges = sorted(
            _byte_range(tensor)
            for tensor in owners
            if isinstance(tensor, torch.Tensor) and tensor.numel()
        )
        merged: list[tuple[int, int]] = []
        for begin, end in ranges:
            if merged and begin <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((begin, end))
        self._protected_ranges = tuple(merged)

    def clear_pools(self) -> None:
        """Synchronize and discard pool bindings without reallocating scratch.

        Discard caller-owned graphs using these pools before this call and
        recapture them before reuse. Pool bindings retain their tensors until
        cleared or wrapper destruction.
        """
        with torch.cuda.device(self._state.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("clear_pools must run outside CUDA graph capture")
            torch.cuda.synchronize(self._state.device)
            self._pool_bindings.clear()
            if self._engine is not None:
                for prepared, _ in self._prepared_launches():
                    prepared._launches.clear()
                self._engine._sample_fused = None
            self._cache_protected_ranges()

    def prepare_pool(self, kv_fused) -> None:
        """Compile and bind an equal-layout pool without capturing a graph.

        The first pool allocates and compiles the shared scoring workspace;
        that, or one eager run, must precede caller capture. Eager run calls
        this automatically, and caller capture binds later pools without it.
        Bindings retain pool/launch owners until clear_pools or destruction.
        """
        from .selective_logits import (
            _select_request_tiling,
            _select_sparse_sample_group_rows,
            _sparse_sample_gather4_serves,
        )

        key = self._pool_key(kv_fused)
        with torch.cuda.device(self._state.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare_pool must run outside CUDA graph capture")
            if not self._state.is_prepared:
                capacity = self._capacity
                if self._needs_sampling and _sparse_sample_gather4_serves(
                    kv_fused.shape[0],
                    kv_fused.stride(0),
                    capacity.page_size,
                    capacity.head_dim,
                ):
                    # Match the capacity-family geometry used by _build_engine
                    # before the first metadata compilation.
                    rows = min(
                        capacity.max_chunk_size,
                        capacity.max_batch_size * capacity.max_context_length,
                    )
                    batches = capacity.max_batch_size
                    lengths = [
                        rows // batches + (request < rows % batches)
                        for request in range(batches)
                    ]
                    tile = _select_request_tiling(
                        self._state.device,
                        capacity.num_heads,
                        capacity.head_dim,
                        lengths,
                    ).next_n
                    self._state._set_sample_group_rows(
                        _select_sparse_sample_group_rows(tile), refresh=False
                    )
                self._state.prepare()
                self._state.refresh()
                if self._needs_sampling:
                    self._state.refresh_sampling()
            if key in self._pool_bindings:
                return
            if self._engine is None:
                try:
                    self._build_engine(kv_fused)
                    self._refresh_schedules()
                except BaseException:
                    self._engine = None
                    raise
            engine = self._engine
            if engine is not None:
                # A binding owns only its pool's launch arguments, rather than
                # retaining every pool that happened to be prepared before it.
                for prepared, _ in self._prepared_launches():
                    prepared._launches.clear()
                engine._sample_fused = kv_fused
                engine._precompile_score_launches(
                    self._q, kv_fused, self._weights, self._state.block_table
                )
            # Keep pool and callable owners alive without owning a CUDA graph.
            self._pool_bindings[key] = (kv_fused, self._prepared_launches())
            self._cache_protected_ranges()

    def input_buffers(self) -> dict[str, torch.Tensor]:
        return self._state.input_buffers()

    @property
    def error(self) -> torch.Tensor:
        return self._state.error

    @property
    def repair_error(self) -> torch.Tensor:
        return self._repair_error

    @property
    def active_shape(self) -> torch.Tensor:
        return self._state.active_shape

    def run(self, q, weights, kv_fused, *, out=None):
        """Score a layer; consume/copy the shared output before the next layer."""
        from bisect import bisect_left
        from .selective_logits import _byte_range, _overlap

        if not self._step_prepared:
            raise RuntimeError("plan must be called before run")
        key = self._pool_key(kv_fused)
        if not isinstance(q, torch.Tensor) or q.ndim != 3:
            raise ValueError("Q must be a rank-three tensor")
        # Payload metadata checks precede allocation or staging. Output alias
        # checks below include the storage of a lazily prepared first pool.
        self._state._payload_copies(q, weights)
        if q.shape[0] != self._supplied_rows:
            raise ValueError("Q rows must match plan num_query_rows")
        selected = self._selected[: q.shape[0]]
        if out is not None:
            if (
                not isinstance(out, torch.Tensor)
                or out.device != self._state.device
                or out.dtype != torch.int32
                or out.shape != (q.shape[0], self.top_k)
                or not out.is_contiguous()
            ):
                raise ValueError("out must be contiguous CUDA int32 [Q, top_k]")
            if out.data_ptr() != selected.data_ptr():
                bounds = _byte_range(out)
                # Static intervals are sorted and disjoint; only the nearest
                # predecessor can overlap. Per-call inputs add three checks.
                index = bisect_left(self._protected_ranges, (bounds[1],))
                static_overlap = index > 0 and _overlap(
                    bounds, self._protected_ranges[index - 1]
                )
                inputs = tuple(
                    _byte_range(tensor)
                    for tensor in (q, weights, kv_fused)
                    if isinstance(tensor, torch.Tensor)
                )
                if static_overlap or any(_overlap(bounds, other) for other in inputs):
                    raise ValueError(
                        "out must not overlap inputs or plan-owned storage"
                    )
        with torch.cuda.device(self._state.device):
            capturing = torch.cuda.is_current_stream_capturing()
            if key in self._pool_bindings:
                _, launches = self._pool_bindings[key]
                if self._engine is not None:
                    for prepared, bindings in launches:
                        prepared._launches.clear()
                        prepared._launches.update(bindings)
            elif not capturing:
                self.prepare_pool(kv_fused)
            elif not self._state.is_prepared or self._layout_needs_engine():
                raise RuntimeError(
                    "prepare_pool or one eager run must precede caller capture"
                )
            elif self._engine is not None:
                # The layout is compiled, so an unseen equal-layout pool only
                # needs new launch arguments. They are rebuilt on the cache
                # miss below without compiling, host reads or allocation; no
                # graph is created for a capture-only pool.
                for prepared, _ in self._prepared_launches():
                    prepared._launches.clear()
            if self._engine is not None:
                self._engine._sample_fused = kv_fused
            self._state.stage_payload(q, weights)
            self._check_caller_capture(capturing, "layer payload copies")
            self._execute_layer(kv_fused)
            self._check_caller_capture(capturing, "layer execution")
            if out is not None and out.data_ptr() != selected.data_ptr():
                out.copy_(selected)
                self._check_caller_capture(capturing, "layer output copy")
        return selected if out is None else out
