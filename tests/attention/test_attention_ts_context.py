# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public-contract and correctness coverage for PrimTS context attention."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
import inspect
import itertools
import math
from types import SimpleNamespace
from typing import Optional, Sequence
import warnings

import pytest
import torch

pytest.importorskip(
    "cutlass",
    minversion="4.7.0",
    reason="PrimTS attention tests require nvidia-cutlass-dsl==4.7.0",
)

from cutlass import BFloat16, Float16, Float8E4M3FN
from cutlass.experimental.task_scheduling.enums import TileSchedulerType

import flashinfer.attention.prims_ts.context as context_module
from flashinfer.attention.prims_ts import (
    BatchPrefillPagedTSWrapper,
    BatchPrefillTSWrapper,
    batch_prefill,
    batch_prefill_with_paged_kv_cache,
)
from flashinfer.attention.prims_ts._tensor_aliasing import (
    _tensor_byte_span,
    _tensors_overlap,
)
from flashinfer.attention.prims_ts.kernels.fmha_context.fmha_kernel import (
    FmhaTs,
    build_fmha_task_manager,
)
from flashinfer.utils import is_sm100a_supported


_REQUIRES_CONTEXT_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")),
    reason="PrimTS context attention requires SM100 or SM103",
)

_HEAD_DIM = 128
_FP8 = torch.float8_e4m3fn


@dataclass(frozen=True)
class _ContextCase:
    """One fixed or packed-ragged context-attention problem."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    qo_indptr: Optional[torch.Tensor]
    kv_indptr: Optional[torch.Tensor]
    q_lengths: tuple[int, ...]
    k_lengths: tuple[int, ...]
    mask_type: str
    window_left: int
    sm_scale: float
    output_scale: float
    output_dtype: torch.dtype

    @property
    def packed(self) -> bool:
        return self.qo_indptr is not None


@dataclass(frozen=True)
class _PagedContextCase:
    """Paged storage plus its independent packed logical reference."""

    reference: _ContextCase
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    qo_indptr: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    page_size: int


@dataclass(frozen=True)
class _NativePagedMetadata:
    """Per-run fixed-table metadata accepted directly by the context kernel."""

    qo_indptr: torch.Tensor
    block_tables: torch.Tensor
    seq_lens_kv: torch.Tensor


def _cumulative(lengths: Sequence[int]) -> tuple[int, ...]:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + int(length))
    return tuple(offsets)


def _make_context_case(
    *,
    q_lengths: Sequence[int],
    k_lengths: Sequence[int],
    num_qo_heads: int,
    num_kv_heads: int,
    qkv_dtype: torch.dtype,
    packed: bool,
    mask_type: str,
    head_dim: int = _HEAD_DIM,
    window_left: int = -1,
    output_dtype: Optional[torch.dtype] = None,
    output_scale: float = 0.75,
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> _ContextCase:
    """Create deterministic compact BSHD or packed THD input storage."""

    q_lengths = tuple(int(length) for length in q_lengths)
    k_lengths = tuple(int(length) for length in k_lengths)
    if not q_lengths or len(q_lengths) != len(k_lengths):
        raise ValueError("Q and KV lengths must describe the same non-empty batch")
    if min(q_lengths) <= 0 or min(k_lengths) <= 0:
        raise ValueError("sequence lengths must be positive")
    if not packed and (len(set(q_lengths)) != 1 or len(set(k_lengths)) != 1):
        raise ValueError("fixed storage requires uniform Q and KV lengths")

    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    input_scale = 0.125 if qkv_dtype == _FP8 else 0.2
    if packed:
        q_shape = (sum(q_lengths), num_qo_heads, head_dim)
        kv_shape = (sum(k_lengths), num_kv_heads, head_dim)
    else:
        q_shape = (len(q_lengths), q_lengths[0], num_qo_heads, head_dim)
        kv_shape = (len(k_lengths), k_lengths[0], num_kv_heads, head_dim)

    q = (
        input_scale
        * torch.randn(q_shape, generator=generator, device=device, dtype=torch.float32)
    ).to(qkv_dtype)
    k = (
        input_scale
        * torch.randn(kv_shape, generator=generator, device=device, dtype=torch.float32)
    ).to(qkv_dtype)
    v = (
        input_scale
        * torch.randn(kv_shape, generator=generator, device=device, dtype=torch.float32)
    ).to(qkv_dtype)
    if packed:
        qo_indptr = torch.tensor(
            _cumulative(q_lengths), dtype=torch.int32, device=device
        )
        kv_indptr = torch.tensor(
            _cumulative(k_lengths), dtype=torch.int32, device=device
        )
    else:
        qo_indptr = None
        kv_indptr = None

    return _ContextCase(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        q_lengths=q_lengths,
        k_lengths=k_lengths,
        mask_type=mask_type,
        window_left=window_left,
        sm_scale=1.0 / math.sqrt(head_dim),
        output_scale=output_scale,
        output_dtype=qkv_dtype if output_dtype is None else output_dtype,
    )


def _make_paged_context_case(
    *,
    q_lengths: Sequence[int],
    k_lengths: Sequence[int],
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    qkv_dtype: torch.dtype,
    mask_type: str,
    page_size: int = 32,
    window_left: int = -1,
    output_dtype: Optional[torch.dtype] = None,
    output_scale: float = 0.75,
    seed: int = 0,
) -> _PagedContextCase:
    """Create nonidentity HND pages and the matching packed logical tensors."""

    q_lengths = tuple(int(length) for length in q_lengths)
    k_lengths = tuple(int(length) for length in k_lengths)
    if not q_lengths or len(q_lengths) != len(k_lengths):
        raise ValueError("Q and KV lengths must describe the same non-empty batch")
    if min(q_lengths) <= 0 or min(k_lengths) <= 0:
        raise ValueError("sequence lengths must be positive")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    input_scale = 0.125 if qkv_dtype == _FP8 else 0.2

    def random_tensor(shape: tuple[int, ...]) -> torch.Tensor:
        return (
            input_scale
            * torch.randn(
                shape, generator=generator, device=device, dtype=torch.float32
            )
        ).to(qkv_dtype)

    q = random_tensor((sum(q_lengths), num_qo_heads, head_dim)).contiguous()
    logical_k = random_tensor((sum(k_lengths), num_kv_heads, head_dim)).contiguous()
    logical_v = random_tensor((sum(k_lengths), num_kv_heads, head_dim)).contiguous()
    page_counts = tuple(math.ceil(length / page_size) for length in k_lengths)
    page_indptr = _cumulative(page_counts)
    num_used_pages = page_indptr[-1]
    num_physical_pages = num_used_pages + 2
    page_indices = tuple(reversed(range(1, num_used_pages + 1)))
    if page_indices == tuple(range(num_used_pages)):
        raise AssertionError("paged test requires a nonidentity page table")

    cache_shape = (num_physical_pages, num_kv_heads, page_size, head_dim)
    k_staging = torch.full(
        cache_shape, float("nan"), dtype=torch.float16, device=device
    )
    v_staging = torch.full_like(k_staging, float("nan"))
    logical_offset = 0
    for batch_idx, k_length in enumerate(k_lengths):
        for page_in_request in range(page_counts[batch_idx]):
            physical_page = page_indices[page_indptr[batch_idx] + page_in_request]
            page_begin = page_in_request * page_size
            page_extent = min(page_size, k_length - page_begin)
            k_staging[physical_page].zero_()
            v_staging[physical_page].zero_()
            logical_slice = slice(
                logical_offset + page_begin,
                logical_offset + page_begin + page_extent,
            )
            k_staging[physical_page, :, :page_extent].copy_(
                logical_k[logical_slice].permute(1, 0, 2).to(torch.float16)
            )
            v_staging[physical_page, :, :page_extent].copy_(
                logical_v[logical_slice].permute(1, 0, 2).to(torch.float16)
            )
        logical_offset += k_length

    qo_indptr = torch.tensor(_cumulative(q_lengths), dtype=torch.int32, device=device)
    paged_kv_indptr = torch.tensor(page_indptr, dtype=torch.int32, device=device)
    paged_kv_indices = torch.tensor(page_indices, dtype=torch.int32, device=device)
    paged_kv_last_page_len = torch.tensor(
        tuple((length - 1) % page_size + 1 for length in k_lengths),
        dtype=torch.int32,
        device=device,
    )
    reference = _ContextCase(
        q=q,
        k=logical_k,
        v=logical_v,
        qo_indptr=qo_indptr,
        kv_indptr=torch.tensor(
            _cumulative(k_lengths), dtype=torch.int32, device=device
        ),
        q_lengths=q_lengths,
        k_lengths=k_lengths,
        mask_type=mask_type,
        window_left=window_left,
        sm_scale=1.0 / math.sqrt(head_dim),
        output_scale=output_scale,
        output_dtype=qkv_dtype if output_dtype is None else output_dtype,
    )
    return _PagedContextCase(
        reference=reference,
        k_cache=k_staging.to(qkv_dtype),
        v_cache=v_staging.to(qkv_dtype),
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
        page_size=page_size,
    )


def _poison_invalid_paged_v_tails(case: _PagedContextCase) -> None:
    """Fill only unused final-page V rows with NaNs."""

    case.k_cache.nan_to_num_(nan=0.0)
    case.v_cache.nan_to_num_(nan=0.0)
    expected_nan = torch.zeros_like(case.v_cache, dtype=torch.bool)
    page_indptr = case.paged_kv_indptr.tolist()
    page_indices = case.paged_kv_indices.tolist()
    last_page_lens = case.paged_kv_last_page_len.tolist()
    for batch_idx, last_page_len in enumerate(last_page_lens):
        physical_page = page_indices[page_indptr[batch_idx + 1] - 1]
        expected_nan[physical_page, :, last_page_len:, :] = True

    case.v_cache.masked_fill_(expected_nan, float("nan"))
    assert torch.isfinite(case.k_cache).all()
    assert torch.equal(torch.isnan(case.v_cache), expected_nan)


def _request_slice(
    tensor: torch.Tensor,
    lengths: tuple[int, ...],
    batch_idx: int,
    *,
    packed: bool,
) -> torch.Tensor:
    if not packed:
        return tensor[batch_idx]
    begin = sum(lengths[:batch_idx])
    return tensor[begin : begin + lengths[batch_idx]]


def _visible_kv_bounds(
    *,
    q_length: int,
    k_length: int,
    query_idx: int,
    mask_type: str,
    window_left: int,
) -> tuple[int, int]:
    """Return FlashInfer's bottom-right visible interval ``[begin, end)``."""

    if mask_type == "dense":
        end = k_length
    elif mask_type == "causal":
        end = k_length - q_length + query_idx + 1
    else:
        raise ValueError("mask_type must be 'dense' or 'causal'")
    if end <= 0:
        raise ValueError("bottom-right causal attention requires Q length <= KV")
    begin = 0 if window_left < 0 else max(0, end - window_left - 1)
    return begin, end


@torch.no_grad()
def _context_reference(case: _ContextCase) -> torch.Tensor:
    """Independent FP32 MHA/GQA oracle for fixed and packed context inputs."""

    outputs = []
    for batch_idx, (q_length, k_length) in enumerate(
        zip(case.q_lengths, case.k_lengths, strict=True)
    ):
        q = _request_slice(case.q, case.q_lengths, batch_idx, packed=case.packed)
        k = _request_slice(case.k, case.k_lengths, batch_idx, packed=case.packed)
        v = _request_slice(case.v, case.k_lengths, batch_idx, packed=case.packed)
        q = q.float()
        k = k.float()
        v = v.float()
        if q.shape[1] % k.shape[1] != 0:
            raise ValueError("Q head count must be divisible by KV head count")
        head_ratio = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(head_ratio, dim=1)
        v = v.repeat_interleave(head_ratio, dim=1)

        request_output = []
        for query_idx in range(q_length):
            begin, end = _visible_kv_bounds(
                q_length=q_length,
                k_length=k_length,
                query_idx=query_idx,
                mask_type=case.mask_type,
                window_left=case.window_left,
            )
            scores = torch.einsum("hd,khd->hk", q[query_idx], k[begin:end])
            probabilities = torch.softmax(scores * case.sm_scale, dim=-1)
            request_output.append(
                torch.einsum("hk,khd->hd", probabilities, v[begin:end])
                * case.output_scale
            )
        outputs.append(torch.stack(request_output))
    return torch.cat(outputs) if case.packed else torch.stack(outputs)


@torch.no_grad()
def _variable_window_reference(
    case: _ContextCase,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    """Independent FP32 oracle for fixed per-Q inclusive window bounds."""

    if case.packed:
        raise ValueError("variable-window test reference requires fixed storage")
    outputs = []
    for batch_idx in range(len(case.q_lengths)):
        q = case.q[batch_idx].float()
        k = case.k[batch_idx].float()
        v = case.v[batch_idx].float()
        head_ratio = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(head_ratio, dim=1)
        v = v.repeat_interleave(head_ratio, dim=1)

        scores = torch.einsum("qhd,khd->hqk", q, k)
        key_positions = torch.arange(
            k.shape[0], dtype=torch.int32, device=k.device
        ).unsqueeze(0)
        valid = (key_positions >= starts[batch_idx].unsqueeze(1)) & (
            key_positions <= ends[batch_idx].unsqueeze(1)
        )
        probabilities = torch.softmax(
            scores.masked_fill(~valid.unsqueeze(0), -torch.inf) * case.sm_scale,
            dim=-1,
        )
        outputs.append(
            torch.einsum("hqk,khd->qhd", probabilities, v) * case.output_scale
        )
    return torch.stack(outputs)


def _assert_context_correct(
    actual: torch.Tensor,
    case: _ContextCase,
    *,
    expected: Optional[torch.Tensor] = None,
) -> None:
    if expected is None:
        expected = _context_reference(case)
    assert actual.shape == case.q.shape
    assert actual.dtype == case.output_dtype
    assert torch.isfinite(actual.float()).all()
    # Select by the least precise input/output type. FP8 includes the kernel's
    # E4M3 probability quantization as well as optional E4M3 output rounding.
    if case.q.dtype == _FP8 or case.output_dtype == _FP8:
        rtol, atol, max_relative_l2 = 5e-2, 1.3e-1, 1e-1
    elif case.q.dtype == torch.bfloat16 or case.output_dtype == torch.bfloat16:
        rtol, atol, max_relative_l2 = 2e-2, 1e-2, 2e-2
    else:
        rtol, atol, max_relative_l2 = 1e-2, 2e-3, 1e-2
    torch.testing.assert_close(actual.float(), expected, rtol=rtol, atol=atol)
    denominator = torch.linalg.vector_norm(expected).clamp_min(1e-6)
    relative_l2 = torch.linalg.vector_norm(actual.float() - expected) / denominator
    assert float(relative_l2) <= max_relative_l2


def _make_native_paged_metadata(
    case: _PagedContextCase,
    *,
    extra_page_columns: int = 0,
    row_stride_multiplier: int = 1,
) -> _NativePagedMetadata:
    """Convert canonical test CSR storage to the native fixed table."""

    device = case.reference.q.device
    page_offsets = tuple(int(value) for value in case.paged_kv_indptr.tolist())
    page_ids = tuple(int(value) for value in case.paged_kv_indices.tolist())
    page_counts = tuple(end - begin for begin, end in itertools.pairwise(page_offsets))
    page_columns = max(page_counts) + extra_page_columns
    row_stride = page_columns * row_stride_multiplier
    backing = torch.full(
        (len(page_counts), row_stride),
        -1,
        dtype=torch.int32,
        device=device,
    )
    block_tables = backing[:, :page_columns]
    for batch_idx, (begin, end) in enumerate(itertools.pairwise(page_offsets)):
        block_tables[batch_idx, : end - begin] = torch.tensor(
            page_ids[begin:end],
            dtype=torch.int32,
            device=device,
        )
    return _NativePagedMetadata(
        qo_indptr=case.qo_indptr,
        block_tables=block_tables,
        seq_lens_kv=torch.tensor(
            case.reference.k_lengths,
            dtype=torch.int32,
            device=device,
        ),
    )


def _plan_wrapper(wrapper: BatchPrefillTSWrapper, case: _ContextCase) -> None:
    wrapper.plan(
        device=case.q.device,
        batch_size=len(case.q_lengths),
        max_seq_len_q=max(case.q_lengths),
        max_kv_len=max(case.k_lengths),
        num_qo_heads=int(case.q.shape[-2]),
        num_kv_heads=int(case.k.shape[-2]),
        head_dim=int(case.q.shape[-1]),
        q_dtype=case.q.dtype,
        kv_dtype=case.k.dtype,
        packed=case.packed,
        mask_type=case.mask_type,
        window_left=case.window_left,
        sm_scale=case.sm_scale,
        output_scale=case.output_scale,
        out_dtype=case.output_dtype,
    )


def _run_wrapper(
    wrapper: BatchPrefillTSWrapper,
    case: _ContextCase,
    **kwargs,
) -> torch.Tensor:
    """Launch a contiguous plan with the request metadata owned by ``case``."""

    return wrapper.run(
        case.q,
        case.k,
        case.v,
        case.qo_indptr,
        case.kv_indptr,
        **kwargs,
    )


def _plan_paged_wrapper(
    wrapper: BatchPrefillPagedTSWrapper,
    case: _PagedContextCase,
    *,
    max_seq_len_q: Optional[int] = None,
    max_kv_len: Optional[int] = None,
    extra_page_columns: int = 0,
    row_stride_multiplier: int = 1,
) -> _NativePagedMetadata:
    """Compile a conservative paged plan and return its per-run metadata."""

    reference = case.reference
    max_seq_len_q = max(reference.q_lengths) if max_seq_len_q is None else max_seq_len_q
    max_kv_len = max(reference.k_lengths) if max_kv_len is None else max_kv_len
    metadata = _make_native_paged_metadata(
        case,
        extra_page_columns=extra_page_columns,
        row_stride_multiplier=row_stride_multiplier,
    )
    wrapper.plan(
        device=reference.q.device,
        batch_size=len(reference.q_lengths),
        max_seq_len_q=max_seq_len_q,
        max_kv_len=max_kv_len,
        num_qo_heads=int(reference.q.shape[1]),
        num_kv_heads=int(case.k_cache.shape[1]),
        head_dim=int(reference.q.shape[2]),
        q_dtype=reference.q.dtype,
        kv_dtype=case.k_cache.dtype,
        out_dtype=reference.output_dtype,
        page_size=case.page_size,
        mask_type=reference.mask_type,
        window_left=reference.window_left,
        sm_scale=reference.sm_scale,
        output_scale=reference.output_scale,
    )
    return metadata


def _run_paged_wrapper(
    wrapper: BatchPrefillPagedTSWrapper,
    case: _PagedContextCase,
    metadata: _NativePagedMetadata,
    *,
    out: Optional[torch.Tensor] = None,
    validate: bool = True,
) -> torch.Tensor:
    return wrapper.run(
        case.reference.q,
        case.k_cache,
        case.v_cache,
        metadata.qo_indptr,
        metadata.block_tables,
        metadata.seq_lens_kv,
        out=out,
        validate=validate,
    )


def _capture_context_graph(
    wrapper: BatchPrefillTSWrapper | BatchPrefillPagedTSWrapper,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    paged_metadata: Optional[_NativePagedMetadata] = None,
    *,
    qo_indptr: Optional[torch.Tensor] = None,
    kv_indptr: Optional[torch.Tensor] = None,
) -> torch.cuda.CUDAGraph:
    """Warm up and capture one wrapper run into caller-owned output."""

    def launch() -> torch.Tensor:
        if paged_metadata is None:
            return wrapper.run(
                q,
                k,
                v,
                qo_indptr,
                kv_indptr,
                out=out,
                validate=False,
            )
        return wrapper.run(
            q,
            k,
            v,
            paged_metadata.qo_indptr,
            paged_metadata.block_tables,
            paged_metadata.seq_lens_kv,
            out=out,
            validate=False,
        )

    assert launch() is out
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = launch()
    assert captured is out
    return graph


def _run_one_shot(case: _ContextCase, *, out: Optional[torch.Tensor] = None):
    return batch_prefill(
        case.q,
        case.k,
        case.v,
        qo_indptr=case.qo_indptr,
        kv_indptr=case.kv_indptr,
        mask_type=case.mask_type,
        window_left=case.window_left,
        sm_scale=case.sm_scale,
        output_scale=case.output_scale,
        out_dtype=case.output_dtype,
        out=out,
    )


# ---------------------------------------------------------------------------
# CPU-only oracle and public API contract
# ---------------------------------------------------------------------------


class _FakeCudaStream:
    """Minimal CUDA-stream stand-in for allocation-free CPU contract tests."""

    def __init__(self, *, handle: int = 17) -> None:
        self.device = torch.device("cpu")
        self.cuda_stream = handle
        self.waited_events: list[object] = []

    def wait_event(self, event: object) -> None:
        self.waited_events.append(event)


def _enable_context_cpu_lifecycle(monkeypatch):
    """Substitute stream/event bookkeeping while retaining wrapper behavior."""

    stream = _FakeCudaStream()
    ready_events = []
    recorded_tensor_calls = []

    def record_ready_event(actual_stream):
        assert actual_stream is stream
        event = object()
        ready_events.append(event)
        return event

    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device=None: stream)
    monkeypatch.setattr(
        context_module,
        "_resolve_context_plan_stream",
        lambda _device, *, api_name: stream,
    )
    monkeypatch.setattr(
        context_module, "_record_context_plan_ready_event", record_ready_event
    )
    monkeypatch.setattr(
        context_module,
        "_record_context_tensors",
        lambda *args: recorded_tensor_calls.append(args),
    )
    return stream, ready_events, recorded_tensor_calls


def test_attention_ts_context_plan_ready_event_is_external_and_recorded(
    monkeypatch,
) -> None:
    calls = {}

    class FakeEvent:
        def record(self, stream) -> None:
            calls["record_stream"] = stream

    event = FakeEvent()

    def make_event(**kwargs):
        calls["event_kwargs"] = kwargs
        return event

    monkeypatch.setattr(torch.cuda, "Event", make_event)
    stream = _FakeCudaStream()

    assert context_module._record_context_plan_ready_event(stream) is event
    assert calls == {
        "event_kwargs": {"external": True},
        "record_stream": stream,
    }


def test_attention_ts_context_plan_ready_wait_is_cross_stream_only() -> None:
    ready_event = object()
    state = SimpleNamespace(
        geometry=SimpleNamespace(device=torch.device("cpu")),
        ready_event=ready_event,
        ready_stream_handle=17,
    )
    plan_stream = _FakeCudaStream(handle=17)
    other_stream = _FakeCudaStream(handle=23)

    context_module._wait_for_context_plan(state, plan_stream)
    context_module._wait_for_context_plan(state, other_stream)

    assert plan_stream.waited_events == []
    assert other_stream.waited_events == [ready_event]


def test_attention_ts_context_storage_span_includes_stride_and_offset() -> None:
    storage = torch.empty(64, dtype=torch.bfloat16)
    tensor = storage.as_strided((2, 3), (10, 2), storage_offset=3)

    assert _tensor_byte_span(tensor) == (
        tensor.data_ptr(),
        tensor.data_ptr() + 15 * tensor.element_size(),
    )


def test_attention_ts_context_paged_views_are_conservatively_bounded() -> None:
    combined_cache = torch.empty((3, 2, 2, 4), dtype=torch.uint8)
    k_cache = combined_cache[:, 0]
    v_cache = combined_cache[:, 1]

    # The views select disjoint elements, but their outer-stride bounding spans
    # overlap. Treating them as overlapping is safer than under-bounding a
    # strided paged cache.
    assert _tensors_overlap(k_cache, v_cache)


def test_attention_ts_context_disjoint_storage_slices_do_not_overlap() -> None:
    storage = torch.empty(16, dtype=torch.float32)

    assert not _tensors_overlap(storage[:4], storage[8:12])


def test_attention_ts_context_alias_guard_covers_fixed_plan_storage(
    monkeypatch,
) -> None:
    """The contiguous wrapper checks runtime metadata and plan-owned scales."""

    run_stream, _, _ = _enable_context_cpu_lifecycle(monkeypatch)
    monkeypatch.setattr(
        context_module, "_validate_runtime_inputs", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        context_module,
        "_prepare_out",
        lambda out, *, q, output_dtype: out,
    )
    argument_names = (
        "k",
        "v",
        "qo_indptr",
        "kv_indptr",
    )
    plan_owned_names = (
        "scale_softmax_log2",
        "output_scale",
    )

    for aliased_name in (*argument_names, *plan_owned_names):
        out = torch.empty(8)
        q = torch.empty(8)
        arguments = {name: torch.empty(8) for name in argument_names}
        plan_owned = {name: torch.empty(8) for name in plan_owned_names}
        empty_i32 = torch.empty(1)

        if aliased_name in arguments:
            arguments[aliased_name] = out
        else:
            plan_owned[aliased_name] = out
        wrapper = BatchPrefillTSWrapper()
        wrapper._plan_state = context_module._ContextPlanState(
            geometry=SimpleNamespace(
                device=torch.device("cpu"),
                output_dtype=out.dtype,
                packed=True,
                mask_type="dense",
            ),
            scale_softmax_log2=plan_owned["scale_softmax_log2"],
            output_scale=plan_owned["output_scale"],
            empty_i32=empty_i32,
            variable_window_padded_starts=None,
            variable_window_cta_starts=empty_i32,
            compiled=lambda *_: None,
            policy=(),
            ready_event=object(),
            ready_stream_handle=run_stream.cuda_stream,
        )

        with pytest.raises(
            ValueError,
            match=rf"out must not overlap {aliased_name} storage",
        ):
            wrapper.run(
                q,
                arguments["k"],
                arguments["v"],
                arguments["qo_indptr"],
                arguments["kv_indptr"],
                out=out,
            )


def test_attention_ts_context_alias_guard_covers_paged_plan_storage(
    monkeypatch,
) -> None:
    """The paged wrapper checks every per-run and plan-owned allocation."""

    run_stream, _, _ = _enable_context_cpu_lifecycle(monkeypatch)
    monkeypatch.setattr(
        context_module, "_validate_paged_runtime_inputs", lambda *_: None
    )
    monkeypatch.setattr(
        context_module,
        "_validate_paged_runtime_metadata",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        context_module,
        "_prepare_out",
        lambda out, *, q, output_dtype: out,
    )
    argument_names = (
        "k_cache",
        "v_cache",
        "qo_indptr",
        "block_tables",
        "seq_lens_kv",
    )
    plan_owned_names = (
        "scale_softmax_log2",
        "output_scale",
    )

    for aliased_name in (*argument_names, *plan_owned_names):
        out = torch.empty(8)
        q = torch.empty(8)
        arguments = {name: torch.empty(8) for name in argument_names}
        plan_owned = {name: torch.empty(8) for name in plan_owned_names}

        if aliased_name in arguments:
            arguments[aliased_name] = out
        else:
            plan_owned[aliased_name] = out
        wrapper = BatchPrefillPagedTSWrapper()
        wrapper._plan_state = context_module._PagedContextPlanState(
            geometry=SimpleNamespace(
                device=torch.device("cpu"), output_dtype=out.dtype
            ),
            scale_softmax_log2=plan_owned["scale_softmax_log2"],
            output_scale=plan_owned["output_scale"],
            compiled=lambda *_: None,
            policy=(),
            ready_event=object(),
            ready_stream_handle=run_stream.cuda_stream,
        )

        with pytest.raises(
            ValueError,
            match=rf"out must not overlap {aliased_name} storage",
        ):
            wrapper.run(
                q,
                arguments["k_cache"],
                arguments["v_cache"],
                arguments["qo_indptr"],
                arguments["block_tables"],
                arguments["seq_lens_kv"],
                out=out,
            )


def test_attention_ts_context_public_surfaces_hide_internal_tuning() -> None:
    surfaces = (
        BatchPrefillTSWrapper.__init__,
        BatchPrefillTSWrapper.plan,
        BatchPrefillTSWrapper.run,
        batch_prefill,
        BatchPrefillPagedTSWrapper.__init__,
        BatchPrefillPagedTSWrapper.plan,
        BatchPrefillPagedTSWrapper.run,
        batch_prefill_with_paged_kv_cache,
    )
    forbidden_token_prefixes = {
        "autotun",
        "clc",
        "config",
        "cta",
        "impl",
        "inst",
        "kernel",
        "mma",
        "pdl",
        "persist",
        "profil",
        "reduc",
        "schedul",
        "split",
        "stag",
        "tile",
        "warp",
    }
    forbidden_token_sequences = (
        ("groups", "tokens", "heads"),
        ("single", "kv"),
        ("tensor", "cores"),
    )
    forbidden_exact_names = {
        "args",
        "backend",
        "enable_pdl",
        "groups_tokens_heads_q",
        "head_dim_per_cta_v",
        "implementation",
        "kernel",
        "mma_variant",
        "num_ctas_per_head_dim",
        "num_insts_kv",
        "separate_reducer_impl",
        "use_cluster_reduction",
        "use_cluster_smem_reduction",
        "use_tensor_cores",
    }
    violations = []
    for surface in surfaces:
        for parameter in inspect.signature(surface).parameters.values():
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                violations.append(f"{surface.__qualname__}.**{parameter.name}")
                continue
            tokens = tuple(parameter.name.split("_"))
            has_forbidden_sequence = any(
                tokens[index : index + len(sequence)] == sequence
                for sequence in forbidden_token_sequences
                for index in range(len(tokens) - len(sequence) + 1)
            )
            has_forbidden_token = any(
                token.startswith(prefix)
                for token in tokens
                for prefix in forbidden_token_prefixes
            )
            if (
                parameter.name in forbidden_exact_names
                or has_forbidden_token
                or has_forbidden_sequence
            ):
                violations.append(f"{surface.__qualname__}.{parameter.name}")

    assert violations == []


def test_attention_ts_context_paged_wrapper_has_no_workspace_api() -> None:
    constructor_parameters = inspect.signature(
        BatchPrefillPagedTSWrapper.__init__
    ).parameters

    assert "workspace_buffer" not in constructor_parameters
    assert not hasattr(BatchPrefillPagedTSWrapper, "reset_workspace_buffer")


def test_attention_ts_context_paged_wrapper_exposes_compile_oriented_contract() -> None:
    plan_parameters = inspect.signature(BatchPrefillPagedTSWrapper.plan).parameters
    run_parameters = inspect.signature(BatchPrefillPagedTSWrapper.run).parameters

    assert tuple(plan_parameters) == (
        "self",
        "device",
        "batch_size",
        "max_seq_len_q",
        "max_kv_len",
        "num_qo_heads",
        "num_kv_heads",
        "head_dim",
        "q_dtype",
        "kv_dtype",
        "out_dtype",
        "page_size",
        "mask_type",
        "window_left",
        "sm_scale",
        "output_scale",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in plan_parameters.items()
        if name != "self"
    )
    assert tuple(run_parameters) == (
        "self",
        "q",
        "k_cache",
        "v_cache",
        "qo_indptr",
        "block_tables",
        "seq_lens_kv",
        "out",
        "scale_softmax_log2",
        "output_scale",
        "validate",
    )
    assert run_parameters["validate"].kind is inspect.Parameter.KEYWORD_ONLY
    assert run_parameters["validate"].default is True
    assert not hasattr(BatchPrefillPagedTSWrapper, "plan_live")
    assert not hasattr(context_module, "PlanSpec")
    assert not hasattr(context_module, "PlanHints")
    assert context_module._PagedContextPlanState.__dataclass_params__.frozen is True


def test_attention_ts_context_contiguous_wrapper_exposes_compile_oriented_contract():
    plan_parameters = inspect.signature(BatchPrefillTSWrapper.plan).parameters
    run_parameters = inspect.signature(BatchPrefillTSWrapper.run).parameters

    assert tuple(plan_parameters) == (
        "self",
        "device",
        "batch_size",
        "max_seq_len_q",
        "max_kv_len",
        "num_qo_heads",
        "num_kv_heads",
        "head_dim",
        "q_dtype",
        "kv_dtype",
        "out_dtype",
        "packed",
        "mask_type",
        "window_left",
        "sm_scale",
        "output_scale",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in plan_parameters.items()
        if name != "self"
    )
    assert tuple(run_parameters) == (
        "self",
        "q",
        "k",
        "v",
        "qo_indptr",
        "kv_indptr",
        "variable_window_token_starts",
        "variable_window_token_ends",
        "out",
        "scale_softmax_log2",
        "output_scale",
        "validate",
    )
    assert run_parameters["qo_indptr"].default is None
    assert run_parameters["kv_indptr"].default is None
    assert run_parameters["validate"].kind is inspect.Parameter.KEYWORD_ONLY
    assert run_parameters["validate"].default is True
    assert context_module._ContextPlanState.__dataclass_params__.frozen is True


@pytest.mark.parametrize(
    ("wrapper_type", "compile_name"),
    (
        (BatchPrefillTSWrapper, "_get_compiled_context"),
        (BatchPrefillPagedTSWrapper, "_get_compiled_paged_context"),
    ),
    ids=("contiguous", "paged"),
)
def test_attention_ts_context_reusable_plan_rejects_capture_before_side_effects(
    monkeypatch,
    wrapper_type,
    compile_name: str,
) -> None:
    """Capture rejection precedes plan-owned allocations and kernel JIT."""

    wrapper = wrapper_type()
    stream = _FakeCudaStream()
    monkeypatch.setattr(
        context_module,
        "_resolve_cuda_device",
        lambda _device: (torch.device("cpu"), 0),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device=None: stream)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    def fail_after_capture_check(*_args, **_kwargs):
        pytest.fail("capture rejection occurred after a plan side effect")

    monkeypatch.setattr(torch, "tensor", fail_after_capture_check)
    monkeypatch.setattr(torch, "empty", fail_after_capture_check)
    monkeypatch.setattr(context_module, compile_name, fail_after_capture_check)

    with pytest.raises(
        RuntimeError,
        match=rf"{wrapper_type.__name__} planning is unsupported during CUDA Graph capture",
    ):
        wrapper.plan(
            device="cuda:0",
            batch_size=1,
            max_seq_len_q=1,
            max_kv_len=1,
            num_qo_heads=1,
            num_kv_heads=1,
            head_dim=128,
            q_dtype=torch.float16,
            kv_dtype=torch.float16,
        )


def test_attention_ts_context_paged_one_shot_exposes_fixed_table_contract() -> None:
    parameters = inspect.signature(batch_prefill_with_paged_kv_cache).parameters
    required_parameters = (
        "q",
        "k_cache",
        "v_cache",
        "qo_indptr",
        "block_tables",
        "seq_lens_kv",
    )
    keyword_only_defaults = {
        "page_size": 32,
        "kv_layout": "HND",
        "mask_type": "dense",
        "window_left": -1,
        "sm_scale": None,
        "output_scale": 1.0,
        "out_dtype": None,
        "out": None,
    }

    assert tuple(parameters) == (*required_parameters, *keyword_only_defaults)
    assert all(
        parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        and parameters[name].default is inspect.Parameter.empty
        for name in required_parameters
    )
    assert all(
        parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in keyword_only_defaults
    )
    assert {
        name: parameters[name].default for name in keyword_only_defaults
    } == keyword_only_defaults


def test_attention_ts_context_one_shot_apis_reject_cuda_graph_capture(
    monkeypatch,
) -> None:
    """One-shot planning must not perform host metadata reads during capture."""

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    q = torch.empty((1, 1, 2, 128), dtype=torch.float16)
    k = torch.empty((1, 1, 1, 128), dtype=torch.float16)
    v = torch.empty_like(k)
    with pytest.raises(RuntimeError, match="cannot derive host plan bounds"):
        batch_prefill(q, k, v)

    paged_q = q.reshape(1, 2, 128)
    k_cache = torch.empty((1, 1, 32, 128), dtype=torch.float16)
    v_cache = torch.empty_like(k_cache)
    qo_indptr = torch.tensor((0, 1), dtype=torch.int32)
    block_tables = torch.tensor(((0,),), dtype=torch.int32)
    seq_lens_kv = torch.tensor((1,), dtype=torch.int32)
    with pytest.raises(RuntimeError, match="cannot derive host plan bounds"):
        batch_prefill_with_paged_kv_cache(
            paged_q,
            k_cache,
            v_cache,
            qo_indptr,
            block_tables,
            seq_lens_kv,
        )


def test_attention_ts_context_paged_one_shot_forwards_fixed_table_to_wrapper(
    monkeypatch,
) -> None:
    """The convenience API preserves caller-owned fixed metadata tensors."""

    calls = {}

    class StubWrapper:
        def __init__(self, *, kv_layout):
            calls["kv_layout"] = kv_layout

        def plan(self, **kwargs):
            calls["plan"] = kwargs

        def run(self, *args, **kwargs):
            calls["run"] = (args, kwargs)
            return kwargs["out"]

    geometry = SimpleNamespace(
        device=torch.device("cpu"),
        batch_size=2,
        max_seq_len_q=8,
        max_kv_len=65,
        page_size=32,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=128,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        output_dtype=torch.float16,
    )

    def resolve(*args, **kwargs):
        calls["resolve"] = (args, kwargs)
        return geometry

    monkeypatch.setattr(
        context_module,
        "_resolve_paged_one_shot_inputs",
        resolve,
    )
    monkeypatch.setattr(context_module, "BatchPrefillPagedTSWrapper", StubWrapper)

    q = torch.empty((9, 4, 128), dtype=torch.float16)
    k_cache = torch.empty((8, 2, 32, 128), dtype=torch.float16)
    v_cache = torch.empty_like(k_cache)
    qo_indptr = torch.tensor((0, 4, 9), dtype=torch.int32)
    block_tables = torch.tensor(((3, 1, -1), (7, 0, 2)), dtype=torch.int32)
    seq_lens_kv = torch.tensor((33, 65), dtype=torch.int32)
    out = torch.empty_like(q)

    returned = batch_prefill_with_paged_kv_cache(
        q,
        k_cache,
        v_cache,
        qo_indptr,
        block_tables,
        seq_lens_kv,
        out=out,
    )

    assert returned is out
    resolve_args, resolve_kwargs = calls["resolve"]
    assert resolve_args == (q, k_cache, v_cache)
    assert resolve_kwargs["qo_indptr"] is qo_indptr
    assert resolve_kwargs["block_tables"] is block_tables
    assert resolve_kwargs["seq_lens_kv"] is seq_lens_kv
    assert calls["plan"]["max_kv_len"] == 65
    assert "max_num_pages_per_seq_kv" not in calls["plan"]
    run_args, run_kwargs = calls["run"]
    assert run_args[3] is qo_indptr
    assert run_args[4] is block_tables
    assert run_args[5] is seq_lens_kv
    assert tuple(run_kwargs) == ("out",)
    assert run_kwargs["out"] is out


def _enable_paged_one_shot_cpu_validation(monkeypatch) -> None:
    """Retain value validation while substituting CPU storage in unit tests."""

    monkeypatch.setattr(context_module, "_validate_base_tensors", lambda *_a: None)
    monkeypatch.setattr(context_module, "_validate_device", lambda _device: 0)
    monkeypatch.setattr(context_module, "_validate_tensor", lambda *_a, **_k: None)
    monkeypatch.setattr(
        context_module, "_validate_indptr_tensor", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        context_module, "_validate_paged_metadata_tensor", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        context_module,
        "_resolve_cuda_device",
        lambda _device: (torch.device("cpu"), 0),
    )
    monkeypatch.setattr(
        context_module, "_validate_paged_runtime_inputs", lambda *_a, **_k: None
    )


def test_attention_ts_context_paged_one_shot_validates_fixed_table(
    monkeypatch,
) -> None:
    """One-shot planning derives bounds from direct fixed-table metadata."""

    _enable_paged_one_shot_cpu_validation(monkeypatch)
    q = torch.empty((3, 4, 128), dtype=torch.bfloat16)
    k_cache = torch.empty((5, 2, 32, 128), dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    block_tables = torch.tensor(((4, -1, -1), (1, 3, 0)), dtype=torch.int32)
    seq_lens_kv = torch.tensor((17, 65), dtype=torch.int32)

    geometry = context_module._resolve_paged_one_shot_inputs(
        q,
        k_cache,
        v_cache,
        qo_indptr=torch.tensor((0, 1, 3), dtype=torch.int32),
        block_tables=block_tables,
        seq_lens_kv=seq_lens_kv,
        page_size=32,
        mask_type="dense",
        window_left=-1,
        output_dtype=torch.bfloat16,
    )

    assert geometry.max_kv_len == 65
    assert geometry.max_seq_len_q == 2
    assert geometry.batch_size == 2


@pytest.mark.parametrize(
    ("block_tables", "seq_lens_kv", "match"),
    (
        (((4, -1), (1, 3)), (17, 65), "at least ceil"),
        (((5, -1, -1), (1, 3, 0)), (17, 65), "active block_tables"),
        (((4, -1, -1), (1, 3, 0)), (0, 65), "entries must be positive"),
    ),
    ids=("short-row", "invalid-active-page", "nonpositive-kv-length"),
)
def test_attention_ts_context_paged_one_shot_rejects_invalid_fixed_metadata(
    monkeypatch,
    block_tables: tuple[tuple[int, ...], ...],
    seq_lens_kv: tuple[int, ...],
    match: str,
) -> None:
    _enable_paged_one_shot_cpu_validation(monkeypatch)
    q = torch.empty((3, 4, 128), dtype=torch.bfloat16)
    k_cache = torch.empty((5, 2, 32, 128), dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)

    with pytest.raises(ValueError, match=match):
        context_module._resolve_paged_one_shot_inputs(
            q,
            k_cache,
            v_cache,
            qo_indptr=torch.tensor((0, 1, 3), dtype=torch.int32),
            block_tables=torch.tensor(block_tables, dtype=torch.int32),
            seq_lens_kv=torch.tensor(seq_lens_kv, dtype=torch.int32),
            page_size=32,
            mask_type="dense",
            window_left=-1,
            output_dtype=torch.bfloat16,
        )


def test_attention_ts_context_contiguous_plan_reuses_dynamic_packed_requests(
    monkeypatch,
) -> None:
    compile_calls = []
    launch_calls = []
    run_stream, ready_events, _ = _enable_context_cpu_lifecycle(monkeypatch)

    def compiled(*args):
        launch_calls.append(args)

    def fake_compile(*key):
        compile_calls.append(key)
        return compiled, (("scheduler", "test"),)

    monkeypatch.setattr(
        context_module,
        "_resolve_cuda_device",
        lambda _device: (torch.device("cpu"), 0),
    )
    monkeypatch.setattr(context_module, "_get_compiled_context", fake_compile)

    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(
        device="cuda:0",
        batch_size=2,
        max_seq_len_q=8,
        max_kv_len=10,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=128,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        packed=True,
    )
    state = wrapper._plan_state
    assert state is not None
    assert state.ready_event is ready_events[0]
    assert state.ready_stream_handle == run_stream.cuda_stream

    first = (
        torch.empty((8, 4, 128), dtype=torch.float16),
        torch.empty((10, 2, 128), dtype=torch.float16),
        torch.empty((10, 2, 128), dtype=torch.float16),
        torch.tensor((0, 3, 8), dtype=torch.int32),
        torch.tensor((0, 4, 10), dtype=torch.int32),
    )
    second = (
        torch.empty((7, 4, 128), dtype=torch.float16),
        torch.empty((9, 2, 128), dtype=torch.float16),
        torch.empty((9, 2, 128), dtype=torch.float16),
        torch.tensor((0, 6, 7), dtype=torch.int32),
        torch.tensor((0, 2, 9), dtype=torch.int32),
    )
    first_out = torch.empty_like(first[0])
    second_out = torch.empty_like(second[0])
    scale_softmax_log2 = torch.tensor((0.25,), dtype=torch.float32)
    output_scale = torch.tensor((0.5,), dtype=torch.float32)

    wrapper.run(*first, out=first_out, validate=False)
    wrapper.run(
        *second,
        out=second_out,
        scale_softmax_log2=scale_softmax_log2,
        output_scale=output_scale,
        validate=False,
    )

    assert len(compile_calls) == 1
    assert len(launch_calls) == 2
    assert launch_calls[0][0].shape[0] == 8
    assert launch_calls[1][0].shape[0] == 7
    assert launch_calls[0][6] is first[3]
    assert launch_calls[0][7] is first[4]
    assert launch_calls[1][6] is second[3]
    assert launch_calls[1][7] is second[4]
    assert launch_calls[0][4] is state.scale_softmax_log2
    assert launch_calls[0][5] is state.output_scale
    assert launch_calls[1][4] is scale_softmax_log2
    assert launch_calls[1][5] is output_scale
    assert state.geometry.uniform_packed_lengths is False
    assert state.geometry.packed_dense_k_mask is True
    for request_name in (
        "q",
        "k",
        "v",
        "qo_indptr",
        "kv_indptr",
        "variable_window_token_starts",
        "variable_window_token_ends",
    ):
        assert not hasattr(state, request_name)


def test_attention_ts_context_variable_window_bounds_are_runtime_state(
    monkeypatch,
) -> None:
    launch_calls = []
    run_stream, _, recorded_tensor_calls = _enable_context_cpu_lifecycle(monkeypatch)

    def compiled(*args):
        assert len(recorded_tensor_calls) == 1
        launch_calls.append(args)

    monkeypatch.setattr(
        context_module,
        "_resolve_cuda_device",
        lambda _device: (torch.device("cpu"), 0),
    )
    monkeypatch.setattr(
        context_module,
        "_get_compiled_context",
        lambda *_key: (compiled, ()),
    )
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(
        device="cuda:0",
        batch_size=2,
        max_seq_len_q=3,
        max_kv_len=4,
        num_qo_heads=2,
        num_kv_heads=1,
        head_dim=128,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        mask_type="variable_window",
    )
    state = wrapper._plan_state
    assert state is not None
    refresh = context_module._refresh_variable_window_cta_starts

    def refresh_after_lifetime_record(*args, **kwargs):
        assert len(recorded_tensor_calls) == 1
        return refresh(*args, **kwargs)

    monkeypatch.setattr(
        context_module,
        "_refresh_variable_window_cta_starts",
        refresh_after_lifetime_record,
    )
    q = torch.empty((2, 3, 2, 128), dtype=torch.float16)
    k = torch.empty((2, 4, 1, 128), dtype=torch.float16)
    v = torch.empty_like(k)
    out = torch.empty_like(q)
    starts = torch.tensor(((0, 1, 1), (0, 0, 2)), dtype=torch.int32)
    ends = torch.tensor(((1, 2, 3), (0, 2, 3)), dtype=torch.int32)
    scale_softmax_log2 = torch.tensor((0.25,), dtype=torch.float32)
    output_scale = torch.tensor((0.5,), dtype=torch.float32)

    returned = wrapper.run(
        q,
        k,
        v,
        variable_window_token_starts=starts,
        variable_window_token_ends=ends,
        out=out,
        scale_softmax_log2=scale_softmax_log2,
        output_scale=output_scale,
        validate=False,
    )

    assert returned is out
    assert len(launch_calls) == 1
    launch = launch_calls[0]
    assert launch[4] is scale_softmax_log2
    assert launch[5] is output_scale
    assert launch[8].data_ptr() == starts.data_ptr()
    assert launch[9].data_ptr() == ends.data_ptr()
    assert launch[10] is state.variable_window_cta_starts
    assert state.variable_window_padded_starts is not None
    assert len(recorded_tensor_calls) == 1
    recorded_stream, *recorded_tensors = recorded_tensor_calls[0]
    assert recorded_stream is run_stream
    expected_recorded_tensors = (state.variable_window_padded_starts, *launch)
    assert all(
        actual is expected
        for actual, expected in zip(
            recorded_tensors, expected_recorded_tensors, strict=True
        )
    )
    assert not hasattr(state, "variable_window_token_starts")
    assert not hasattr(state, "variable_window_token_ends")


def test_attention_ts_context_paged_plan_compiles_once_for_dynamic_metadata(
    monkeypatch,
) -> None:
    compile_calls = []
    launch_calls = []
    run_stream, ready_events, _ = _enable_context_cpu_lifecycle(monkeypatch)

    def compiled(*args):
        launch_calls.append(args)

    def fake_compile(*key):
        compile_calls.append(key)
        return compiled, (("scheduler", "test"),)

    monkeypatch.setattr(
        context_module,
        "_resolve_cuda_device",
        lambda _device: (torch.device("cpu"), 0),
    )
    monkeypatch.setattr(context_module, "_get_compiled_paged_context", fake_compile)

    wrapper = BatchPrefillPagedTSWrapper()
    wrapper.plan(
        device="cuda:0",
        batch_size=2,
        max_seq_len_q=16,
        max_kv_len=64,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=128,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
    )
    first_state = wrapper._plan_state
    assert first_state is not None
    assert first_state.ready_event is ready_events[0]
    assert first_state.ready_stream_handle == run_stream.cuda_stream
    with pytest.raises(ValueError, match="max_kv_len must be positive"):
        wrapper.plan(
            device="cuda:0",
            batch_size=2,
            max_seq_len_q=16,
            max_kv_len=0,
            num_qo_heads=4,
            num_kv_heads=2,
            head_dim=128,
            q_dtype=torch.float16,
            kv_dtype=torch.float16,
        )
    assert wrapper._plan_state is first_state

    q = torch.empty((8, 4, 128), dtype=torch.float16)
    k_cache = torch.empty((6, 2, 32, 128), dtype=torch.float16)
    v_cache = torch.empty_like(k_cache)
    out = torch.empty_like(q)
    first_metadata = tuple(torch.empty(1) for _ in range(3))
    second_metadata = tuple(torch.empty(2) for _ in range(3))

    wrapper.run(
        q,
        k_cache,
        v_cache,
        *first_metadata,
        out=out,
        validate=False,
    )
    wrapper.run(
        q,
        k_cache,
        v_cache,
        *second_metadata,
        out=out,
        validate=False,
    )

    assert len(compile_calls) == 1
    assert len(launch_calls) == 2
    assert first_state is not None
    assert launch_calls[0][4] is first_state.scale_softmax_log2
    assert launch_calls[1][4] is first_state.scale_softmax_log2
    assert launch_calls[0][5] is first_state.output_scale
    assert launch_calls[1][5] is first_state.output_scale
    assert all(
        actual is expected
        for actual, expected in zip(launch_calls[0][6:9], first_metadata, strict=True)
    )
    assert all(
        actual is expected
        for actual, expected in zip(launch_calls[1][6:9], second_metadata, strict=True)
    )
    assert wrapper._plan_state is not None
    assert wrapper._plan_state.geometry.uniform_packed_lengths is False
    assert wrapper._plan_state.geometry.has_q_offset is False
    assert wrapper._plan_state.geometry.packed_dense_k_mask is True


def test_attention_ts_context_failed_paged_replan_retains_previous_state(
    monkeypatch,
) -> None:
    compile_count = 0
    _enable_context_cpu_lifecycle(monkeypatch)

    def fake_compile(*_key):
        nonlocal compile_count
        compile_count += 1
        if compile_count == 2:
            raise RuntimeError("synthetic compile failure")
        return (lambda *_: None), (("scheduler", "test"),)

    monkeypatch.setattr(
        context_module,
        "_resolve_cuda_device",
        lambda _device: (torch.device("cpu"), 0),
    )
    monkeypatch.setattr(context_module, "_get_compiled_paged_context", fake_compile)
    plan_kwargs = dict(
        device="cuda:0",
        batch_size=2,
        max_seq_len_q=16,
        max_kv_len=64,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=128,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
    )
    wrapper = BatchPrefillPagedTSWrapper()
    wrapper.plan(**plan_kwargs)
    previous_state = wrapper._plan_state

    with pytest.raises(RuntimeError, match="synthetic compile failure"):
        wrapper.plan(**plan_kwargs, output_scale=0.5)

    assert wrapper._plan_state is previous_state
    assert compile_count == 2


def test_attention_ts_context_failed_contiguous_replan_retains_previous_state(
    monkeypatch,
) -> None:
    compile_count = 0
    launch_count = 0
    _enable_context_cpu_lifecycle(monkeypatch)

    def fake_compile(*_key):
        nonlocal compile_count
        compile_count += 1
        if compile_count == 2:
            raise RuntimeError("synthetic compile failure")

        def launch(*_args):
            nonlocal launch_count
            launch_count += 1

        return launch, (("scheduler", "test"),)

    monkeypatch.setattr(
        context_module,
        "_resolve_cuda_device",
        lambda _device: (torch.device("cpu"), 0),
    )
    monkeypatch.setattr(context_module, "_get_compiled_context", fake_compile)
    plan_kwargs = dict(
        device="cuda:0",
        batch_size=2,
        max_seq_len_q=3,
        max_kv_len=4,
        num_qo_heads=2,
        num_kv_heads=1,
        head_dim=128,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
    )
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(**plan_kwargs)
    previous_state = wrapper._plan_state

    with pytest.raises(RuntimeError, match="synthetic compile failure"):
        wrapper.plan(**plan_kwargs, output_scale=0.5)

    assert wrapper._plan_state is previous_state
    q = torch.empty((2, 3, 2, 128), dtype=torch.float16)
    k = torch.empty((2, 4, 1, 128), dtype=torch.float16)
    wrapper.run(q, k, torch.empty_like(k), out=torch.empty_like(q), validate=False)
    assert compile_count == 2
    assert launch_count == 1


def test_attention_ts_context_paged_run_validate_false_bypasses_validators(
    monkeypatch,
) -> None:
    run_stream, _, _ = _enable_context_cpu_lifecycle(monkeypatch)

    def fail(*_args, **_kwargs):
        pytest.fail("validate=False reached an explicit runtime validator")

    monkeypatch.setattr(context_module, "_validate_paged_runtime_inputs", fail)
    monkeypatch.setattr(context_module, "_validate_paged_runtime_metadata", fail)
    monkeypatch.setattr(context_module, "_validate_runtime_scale_tensor", fail)
    monkeypatch.setattr(context_module, "_prepare_out", fail)
    monkeypatch.setattr(context_module, "_validate_out_does_not_overlap_inputs", fail)

    out = torch.empty(1)
    launched = []
    wrapper = BatchPrefillPagedTSWrapper()
    wrapper._plan_state = context_module._PagedContextPlanState(
        geometry=SimpleNamespace(device=torch.device("cpu"), output_dtype=out.dtype),
        scale_softmax_log2=torch.empty(1),
        output_scale=torch.empty(1),
        compiled=lambda *args: launched.append(args),
        policy=(),
        ready_event=object(),
        ready_stream_handle=run_stream.cuda_stream,
    )
    inputs = tuple(torch.empty(1) for _ in range(6))
    scale_softmax_log2 = torch.empty(1)
    output_scale = torch.empty(1)

    assert (
        wrapper.run(
            *inputs,
            out=out,
            scale_softmax_log2=scale_softmax_log2,
            output_scale=output_scale,
            validate=False,
        )
        is out
    )
    assert len(launched) == 1


@pytest.mark.parametrize("scale_name", ("scale_softmax_log2", "output_scale"))
@pytest.mark.parametrize(
    "invalid_value",
    (
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
        pytest.param(0.0, id="zero"),
        pytest.param(-1.0, id="negative"),
    ),
)
def test_attention_ts_context_paged_run_rejects_invalid_runtime_scale_values(
    monkeypatch,
    scale_name: str,
    invalid_value: float,
) -> None:
    """Value checks remain active for each validated per-run scale override."""

    run_stream, _, _ = _enable_context_cpu_lifecycle(monkeypatch)
    monkeypatch.setattr(
        context_module, "_validate_paged_runtime_inputs", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        context_module, "_validate_paged_runtime_metadata", lambda *_a, **_k: None
    )
    # Substitute CPU storage for the CUDA-only structure check; dtype, shape,
    # layout, alignment, device matching, and scalar-value checks remain real.
    monkeypatch.setattr(context_module, "_validate_tensor", lambda *_a, **_k: None)

    launched = []
    wrapper = BatchPrefillPagedTSWrapper()
    wrapper._plan_state = context_module._PagedContextPlanState(
        geometry=SimpleNamespace(
            device=torch.device("cpu"), output_dtype=torch.float32
        ),
        scale_softmax_log2=torch.tensor((0.25,), dtype=torch.float32),
        output_scale=torch.tensor((0.5,), dtype=torch.float32),
        compiled=lambda *args: launched.append(args),
        policy=(),
        ready_event=object(),
        ready_stream_handle=run_stream.cuda_stream,
    )
    inputs = tuple(torch.empty(1) for _ in range(6))
    invalid_scale = torch.tensor((invalid_value,), dtype=torch.float32)

    with pytest.raises(
        ValueError,
        match=rf"{scale_name} must be finite and positive",
    ):
        wrapper.run(
            *inputs,
            **{scale_name: invalid_scale},
            validate=True,
        )

    assert launched == []


def test_attention_ts_context_paged_run_forwards_valid_scales_and_lifecycle(
    monkeypatch,
) -> None:
    """Validated overrides are forwarded after the plan/event lifetime fence."""

    run_stream, _, recorded_tensor_calls = _enable_context_cpu_lifecycle(monkeypatch)
    monkeypatch.setattr(
        context_module, "_validate_paged_runtime_inputs", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        context_module, "_validate_paged_runtime_metadata", lambda *_a, **_k: None
    )
    monkeypatch.setattr(context_module, "_validate_tensor", lambda *_a, **_k: None)

    launched = []
    ready_event = object()

    def compiled(*args):
        assert run_stream.waited_events == [ready_event]
        assert len(recorded_tensor_calls) == 1
        launched.append(args)

    plan_scale_softmax_log2 = torch.tensor((0.125,), dtype=torch.float32)
    plan_output_scale = torch.tensor((0.75,), dtype=torch.float32)
    wrapper = BatchPrefillPagedTSWrapper()
    wrapper._plan_state = context_module._PagedContextPlanState(
        geometry=SimpleNamespace(
            device=torch.device("cpu"), output_dtype=torch.float32
        ),
        scale_softmax_log2=plan_scale_softmax_log2,
        output_scale=plan_output_scale,
        compiled=compiled,
        policy=(),
        ready_event=ready_event,
        ready_stream_handle=run_stream.cuda_stream + 1,
    )
    q, k_cache, v_cache, qo_indptr, block_tables, seq_lens_kv = (
        torch.empty(1) for _ in range(6)
    )
    out = torch.empty_like(q)
    scale_softmax_log2 = torch.tensor((0.25,), dtype=torch.float32)
    output_scale = torch.tensor((0.5,), dtype=torch.float32)

    returned = wrapper.run(
        q,
        k_cache,
        v_cache,
        qo_indptr,
        block_tables,
        seq_lens_kv,
        out=out,
        scale_softmax_log2=scale_softmax_log2,
        output_scale=output_scale,
        validate=True,
    )

    assert returned is out
    assert run_stream.waited_events == [ready_event]
    assert len(recorded_tensor_calls) == 1
    recorded_stream, *recorded_tensors = recorded_tensor_calls[0]
    assert recorded_stream is run_stream
    expected_recorded_tensors = (
        q,
        k_cache,
        v_cache,
        out,
        qo_indptr,
        block_tables,
        seq_lens_kv,
        scale_softmax_log2,
        output_scale,
    )
    assert all(
        actual is expected
        for actual, expected in zip(
            recorded_tensors, expected_recorded_tensors, strict=True
        )
    )
    assert len(launched) == 1
    expected_launch = (
        q,
        k_cache,
        v_cache,
        out,
        scale_softmax_log2,
        output_scale,
        qo_indptr,
        block_tables,
        seq_lens_kv,
    )
    assert all(
        actual is expected
        for actual, expected in zip(launched[0], expected_launch, strict=True)
    )


def test_attention_ts_context_paged_run_keeps_lifecycle_check_without_validation():
    wrapper = BatchPrefillPagedTSWrapper()
    placeholder = torch.empty(0)

    with pytest.raises(RuntimeError, match=r"plan\(\) must be called"):
        wrapper.run(
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            validate=False,
        )


def test_attention_ts_context_paged_run_rejects_non_bool_validate() -> None:
    wrapper = BatchPrefillPagedTSWrapper()
    wrapper._plan_state = context_module._PagedContextPlanState(
        geometry=SimpleNamespace(
            device=torch.device("cpu"), output_dtype=torch.float32
        ),
        scale_softmax_log2=torch.empty(1),
        output_scale=torch.empty(1),
        compiled=lambda *_: None,
        policy=(),
        ready_event=object(),
        ready_stream_handle=0,
    )
    placeholder = torch.empty(0)

    with pytest.raises(TypeError, match="validate must be a bool"):
        wrapper.run(
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            validate=1,
        )


def _validate_paged_metadata_on_cpu(
    monkeypatch,
    *,
    qo_indptr: tuple[int, ...] = (0, 2, 5),
    block_tables: tuple[tuple[int, ...], ...] = (
        (0, 1, -1),
        (2, 3, -1),
    ),
    seq_lens_kv: tuple[int, ...] = (33, 64),
    max_kv_len: int = 65,
    row_stride_extra: int = 0,
) -> None:
    """Exercise metadata value checks without requiring a CUDA device."""

    monkeypatch.setattr(context_module, "_validate_tensor", lambda *_a, **_k: None)
    geometry = SimpleNamespace(
        device=torch.device("cpu"),
        batch_size=2,
        max_seq_len_q=64,
        max_kv_len=max_kv_len,
        page_size=32,
        mask_type="causal",
    )
    compact_block_tables = torch.tensor(block_tables, dtype=torch.int32)
    if row_stride_extra:
        backing = torch.full(
            (
                compact_block_tables.shape[0],
                compact_block_tables.shape[1] + row_stride_extra,
            ),
            -123,
            dtype=torch.int32,
        )
        backing[:, : compact_block_tables.shape[1]].copy_(compact_block_tables)
        runtime_block_tables = backing[:, : compact_block_tables.shape[1]]
    else:
        runtime_block_tables = compact_block_tables
    context_module._validate_paged_runtime_metadata(
        torch.tensor(qo_indptr, dtype=torch.int32),
        runtime_block_tables,
        torch.tensor(seq_lens_kv, dtype=torch.int32),
        geometry,
        total_q=qo_indptr[-1],
        num_physical_pages=4,
    )


def test_attention_ts_context_paged_validation_allows_arbitrary_padding_ids(
    monkeypatch,
) -> None:
    """Only page IDs required by each logical K/V length are dereferenced."""

    _validate_paged_metadata_on_cpu(monkeypatch)


def test_attention_ts_context_paged_validation_allows_fixed_stride_extra_padding(
    monkeypatch,
) -> None:
    """Fixed table rows may have inactive columns and a larger storage stride."""

    _validate_paged_metadata_on_cpu(
        monkeypatch,
        block_tables=((0, 1, -1, 91), (2, -1, 92, -1)),
        seq_lens_kv=(33, 32),
        max_kv_len=33,
        row_stride_extra=4,
    )


def test_attention_ts_context_paged_validation_enforces_row_strided_table(
    monkeypatch,
) -> None:
    """The native table admits row padding but rejects overlapping rows."""

    monkeypatch.setattr(context_module, "_validate_tensor", lambda *_a, **_k: None)
    backing = torch.zeros((2, 2, 3), dtype=torch.int32)
    row_padded = backing[:, 0, :]
    context_module._validate_block_tables_tensor(
        row_padded,
        device=torch.device("cpu"),
        batch_size=2,
        required_page_columns=3,
    )
    assert row_padded.stride() == (6, 1)

    overlapping_rows = torch.as_strided(
        torch.zeros(5, dtype=torch.int32),
        size=(2, 3),
        stride=(2, 1),
    )
    with pytest.raises(ValueError, match=r"stride\(0\) >= C"):
        context_module._validate_block_tables_tensor(
            overlapping_rows,
            device=torch.device("cpu"),
            batch_size=2,
            required_page_columns=3,
        )

    nonunit_columns = torch.zeros((3, 2), dtype=torch.int32).transpose(0, 1)
    with pytest.raises(ValueError, match=r"stride\(1\) == 1"):
        context_module._validate_block_tables_tensor(
            nonunit_columns,
            device=torch.device("cpu"),
            batch_size=2,
            required_page_columns=3,
        )

    oversized_row_stride = torch.as_strided(
        torch.zeros(3, dtype=torch.int32),
        size=(1, 3),
        stride=(context_module._INT32_MAX + 1, 1),
    )
    with pytest.raises(NotImplementedError, match="row stride must fit"):
        context_module._validate_block_tables_tensor(
            oversized_row_stride,
            device=torch.device("cpu"),
            batch_size=1,
            required_page_columns=3,
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        pytest.param(
            {"block_tables": ((0, 1), (2, 3))},
            "at least ceil\\(max_kv_len / page_size\\)",
            id="short-fixed-table",
        ),
        pytest.param(
            {"seq_lens_kv": (66, 64)},
            "max_kv_len=65",
            id="kv-bound",
        ),
        pytest.param(
            {"qo_indptr": (0, 34, 35)},
            "Sq=34, Sk=33",
            id="causal-q-longer-than-kv",
        ),
        pytest.param(
            {"block_tables": ((4, 1, -1), (2, 3, -1))},
            "active block_tables",
            id="active-page-out-of-range",
        ),
    ),
)
def test_attention_ts_context_paged_validation_rejects_unsafe_fixed_values(
    monkeypatch,
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _validate_paged_metadata_on_cpu(monkeypatch, **overrides)


def test_attention_ts_context_fixed_oracle_is_bottom_right_causal():
    case = _make_context_case(
        q_lengths=(2,),
        k_lengths=(4,),
        num_qo_heads=2,
        num_kv_heads=1,
        qkv_dtype=torch.float32,
        packed=False,
        mask_type="causal",
        output_scale=1.0,
        device="cpu",
    )
    case.q.zero_()
    case.k.zero_()
    case.v.zero_()
    case.v[0, :, 0, 0] = torch.arange(1, 5, dtype=torch.float32)

    causal = _context_reference(case)
    dense = _context_reference(replace(case, mask_type="dense"))
    windowed = _context_reference(replace(case, window_left=1))
    torch.testing.assert_close(causal[0, :, 0, 0], torch.tensor((2.0, 2.5)))
    torch.testing.assert_close(dense[0, :, 0, 0], torch.tensor((2.5, 2.5)))
    # Sq=2, Sk=4 shifts causal Q by two positions. A one-token left window
    # therefore sees K[1:3] then K[2:4]. The old unshifted bound produced
    # (2.0, 2.5), identical to the unwindowed causal result above.
    torch.testing.assert_close(windowed[0, :, 0, 0], torch.tensor((2.5, 3.5)))
    assert not torch.equal(windowed, causal)
    assert not torch.equal(causal[:, 0], dense[:, 0])
    assert _visible_kv_bounds(
        q_length=2,
        k_length=4,
        query_idx=0,
        mask_type="causal",
        window_left=1,
    ) == (1, 3)


def test_attention_ts_context_packed_oracle_applies_left_window_per_row():
    case = _make_context_case(
        q_lengths=(3, 4),
        k_lengths=(3, 4),
        num_qo_heads=2,
        num_kv_heads=1,
        qkv_dtype=torch.float32,
        packed=True,
        mask_type="causal",
        window_left=1,
        output_scale=1.0,
        device="cpu",
    )
    case.q.zero_()
    case.k.zero_()
    case.v.zero_()
    case.v[:3, 0, 0] = torch.tensor((1.0, 2.0, 3.0))
    case.v[3:, 0, 0] = torch.tensor((11.0, 12.0, 13.0, 14.0))

    output = _context_reference(case)
    expected = torch.tensor((1.0, 1.5, 2.5, 11.0, 11.5, 12.5, 13.5))
    torch.testing.assert_close(output[:, 0, 0], expected)
    torch.testing.assert_close(output[:, 1, 0], expected)
    assert torch.count_nonzero(output[..., 1:]) == 0


@pytest.mark.parametrize(
    "wrapper_type",
    (BatchPrefillTSWrapper, BatchPrefillPagedTSWrapper),
    ids=("fixed", "paged"),
)
def test_attention_ts_context_run_requires_plan(wrapper_type):
    wrapper = wrapper_type()
    placeholder = torch.empty(0)
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called"):
        if wrapper_type is BatchPrefillPagedTSWrapper:
            wrapper.run(
                placeholder,
                placeholder,
                placeholder,
                placeholder,
                placeholder,
                placeholder,
            )
        else:
            wrapper.run(placeholder, placeholder, placeholder)


@pytest.mark.parametrize("use_paged_kv", (False, True), ids=("separate", "paged"))
@pytest.mark.parametrize(
    (
        "input_dtype",
        "output_dtype",
        "is_causal",
        "is_clc_dynamic",
    ),
    (
        pytest.param(
            Float8E4M3FN,
            Float16,
            True,
            True,
            id="fp8-causal-clc-fp16-o",
        ),
        pytest.param(
            Float8E4M3FN,
            Float8E4M3FN,
            False,
            False,
            id="fp8-dense-static-fp8-o",
        ),
        pytest.param(
            BFloat16,
            Float16,
            True,
            True,
            id="bf16-causal-clc-fp16-o",
        ),
        pytest.param(
            BFloat16,
            BFloat16,
            False,
            False,
            id="bf16-dense-static-bf16-o",
        ),
        pytest.param(
            Float16,
            BFloat16,
            False,
            True,
            id="fp16-dense-clc-bf16-o",
        ),
    ),
)
def test_attention_ts_context_d256_pipeline_policy_is_semantic_and_capacity_safe(
    use_paged_kv: bool,
    input_dtype,
    output_dtype,
    is_causal: bool,
    is_clc_dynamic: bool,
):
    """Every D256 topology provides enough stages for its K/V cadence."""

    kernel = FmhaTs(
        in_dtype=input_dtype,
        out_dtype=output_dtype,
        d=256,
        is_persistent=True,
        is_causal=is_causal,
        is_clc_dynamic=is_clc_dynamic,
        use_paged_kv=use_paged_kv,
    )

    cfg = kernel.cfg
    cadence_stages = cfg.num_head_dim_stages_k + cfg.num_head_dim_stages_v
    assert cfg.single_qkv_instance is True
    assert cfg.stage_kv_by_head_dim is True
    assert cfg.has_tmem_p_pipeline is True
    assert cfg.kv_stage >= cadence_stages
    # Persistent D256 must keep correction statistics in the compact SMEM
    # ring. Sharing its S/P TMEM columns is racy when producer latency varies
    # across work tiles (most visibly for causal paged KV).
    assert cfg.stats_via_smem is True
    assert cfg.mma_corr_stage == 1
    assert cfg.stages_page_offsets_in_smem is use_paged_kv

    if use_paged_kv:
        # The page-ID rings need one credit per K/V D stage plus one K-ahead
        # boundary credit, independent of the capacity-derived stage depth.
        assert sum(cfg.page_offset_pipeline_stage_counts) == cadence_stages + 1
        pages_per_tile = cfg.kv_tile_n // cfg.num_tokens_per_page
        assert cfg.page_table_window_entries >= pages_per_tile
        assert cfg.page_table_window_entries % pages_per_tile == 0
    else:
        assert cfg.page_offset_pipeline_stage_counts == ()


@pytest.mark.parametrize(
    ("input_dtype", "is_causal", "max_pages", "expected_entries"),
    (
        pytest.param(Float8E4M3FN, False, 128, 64, id="fp8-staged-window"),
        pytest.param(BFloat16, False, 128, 32, id="bf16-capacity-fallback"),
        pytest.param(Float16, False, 128, 32, id="fp16-capacity-fallback"),
        pytest.param(Float8E4M3FN, True, 128, 32, id="causal-natural-window"),
        pytest.param(Float8E4M3FN, False, 32, 32, id="short-natural-window"),
    ),
)
def test_attention_ts_context_page_window_fits_static_geometry_and_capacity(
    input_dtype,
    is_causal: bool,
    max_pages: int,
    expected_entries: int,
):
    """The wider page-ID handoff is selected only when its SMEM ring fits."""

    cfg = FmhaTs(
        in_dtype=input_dtype,
        out_dtype=BFloat16,
        d=256,
        is_persistent=True,
        is_causal=is_causal,
        use_paged_kv=True,
        max_kv_len=max_pages * 32,
    ).cfg

    assert cfg.page_table_window_entries == expected_entries
    cadence_stages = cfg.num_head_dim_stages_k + cfg.num_head_dim_stages_v
    assert cfg.kv_stage >= cadence_stages
    assert sum(cfg.page_offset_pipeline_stage_counts) == cadence_stages + 1
    assert (
        sum(cfg.page_offset_pipeline_stage_counts) * cfg.page_table_window_entries * 4
        == (cadence_stages + 1) * expected_entries * 4
    )


@pytest.mark.parametrize(
    ("mask_type", "window_left", "has_q_offset", "expected"),
    (
        pytest.param("causal", -1, False, True, id="triangular"),
        pytest.param("causal", -1, True, False, id="bottom-right-offset"),
        pytest.param("causal", 63, False, False, id="sliding-window"),
        pytest.param("dense", -1, False, False, id="dense"),
    ),
)
def test_attention_ts_context_heavy_first_static_raster_policy(
    mask_type: str,
    window_left: int,
    has_q_offset: bool,
    expected: bool,
):
    """Heavy-first order follows causal task-domain geometry, not shape."""

    assert (
        context_module._uses_heavy_first_static_causal_raster(
            mask_type=mask_type,
            window_left=window_left,
            has_q_offset=has_q_offset,
        )
        is expected
    )


@pytest.mark.parametrize(
    (
        "is_persistent",
        "single_qkv_instance",
        "is_causal",
        "uniform_packed_lengths",
        "expected",
    ),
    (
        pytest.param(True, True, True, True, False, id="fixed-causal-static"),
        pytest.param(True, True, True, False, True, id="live-ragged-clc"),
        pytest.param(True, False, True, True, True, id="paired-clc"),
        pytest.param(True, True, False, False, False, id="dense-static"),
        pytest.param(False, True, True, False, False, id="nonpersistent"),
    ),
)
def test_attention_ts_context_paged_clc_policy_is_structural(
    is_persistent: bool,
    single_qkv_instance: bool,
    is_causal: bool,
    uniform_packed_lengths: bool,
    expected: bool,
):
    """Paged CLC selection follows topology and live-domain requirements."""

    assert (
        context_module._paged_context_uses_clc_scheduler(
            is_persistent=is_persistent,
            single_qkv_instance=single_qkv_instance,
            is_causal=is_causal,
            uniform_packed_lengths=uniform_packed_lengths,
        )
        is expected
    )


@pytest.mark.parametrize(
    (
        "single_qkv_instance",
        "head_paired",
        "packed",
        "uniform_packed_lengths",
        "is_causal",
        "has_q_offset",
        "expected",
    ),
    (
        pytest.param(False, False, True, True, True, True, True, id="paired-clc"),
        pytest.param(True, True, False, False, True, True, True, id="head-paired-clc"),
        pytest.param(True, False, True, False, True, True, True, id="live-ragged-clc"),
        pytest.param(True, False, True, True, True, False, True, id="triangular-clc"),
        pytest.param(True, False, True, True, True, True, False, id="offset-static"),
        pytest.param(True, False, True, True, False, False, False, id="dense-static"),
    ),
)
def test_attention_ts_context_contiguous_clc_policy_is_structural(
    single_qkv_instance: bool,
    head_paired: bool,
    packed: bool,
    uniform_packed_lengths: bool,
    is_causal: bool,
    has_q_offset: bool,
    expected: bool,
):
    """Contiguous CLC follows topology and causal-domain structure."""

    assert (
        context_module._contiguous_context_uses_clc_scheduler(
            single_qkv_instance=single_qkv_instance,
            head_paired=head_paired,
            packed=packed,
            uniform_packed_lengths=uniform_packed_lengths,
            is_causal=is_causal,
            has_q_offset=has_q_offset,
        )
        is expected
    )


@pytest.mark.parametrize(
    (
        "single_qkv_instance",
        "head_paired",
        "packed",
        "uniform_packed_lengths",
        "logical_work_tiles",
        "is_causal",
        "has_q_offset",
        "expected",
    ),
    (
        pytest.param(False, False, True, True, 1, True, True, True, id="paired"),
        pytest.param(True, True, False, False, 1, True, True, True, id="head-paired"),
        pytest.param(True, False, True, False, 1, True, True, True, id="live-ragged"),
        pytest.param(True, False, True, True, 1, True, False, True, id="triangular"),
        pytest.param(True, False, True, True, 148, True, True, False, id="one-wave"),
        pytest.param(True, False, True, True, 149, True, True, True, id="multi-wave"),
    ),
)
def test_attention_ts_context_contiguous_persistence_follows_wave_count(
    single_qkv_instance: bool,
    head_paired: bool,
    packed: bool,
    uniform_packed_lengths: bool,
    logical_work_tiles: int,
    is_causal: bool,
    has_q_offset: bool,
    expected: bool,
):
    """Near-uniform immutable domains persist only beyond one CTA wave."""

    assert (
        context_module._contiguous_context_uses_persistent_scheduler(
            single_qkv_instance=single_qkv_instance,
            head_paired=head_paired,
            packed=packed,
            uniform_packed_lengths=uniform_packed_lengths,
            logical_work_tiles=logical_work_tiles,
            max_active_clusters=148,
            batch_size=1,
            num_qo_heads=32,
            is_causal=is_causal,
            has_q_offset=has_q_offset,
        )
        is expected
    )


@pytest.mark.parametrize(
    "input_dtype",
    (Float8E4M3FN, BFloat16),
    ids=("fp8", "bf16"),
)
def test_attention_ts_context_d128_paged_clc_task_graph_is_safe(
    input_dtype,
):
    """The paired D128 CLC graph is valid without a page-ID ring."""

    kernel = FmhaTs(
        in_dtype=input_dtype,
        out_dtype=Float16,
        d=128,
        is_persistent=True,
        is_causal=True,
        is_clc_dynamic=True,
        use_paged_kv=True,
        max_kv_len=256,
    )
    cfg = kernel.cfg
    cfg.has_varlen = True
    cfg.has_uniform_varlen = False
    cfg.has_q_offset = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, _, _, _, work_queue, clc_response_alloc = build_fmha_task_manager(
            cfg,
            tile_sched_params=None,
            tma_q_desc=None,
            tma_k_desc=None,
            tma_v_desc=None,
            tma_o_desc=None,
            cum_seqlen_q=None,
            cum_seqlen_k=None,
            num_kv_tiles=2,
            q_offset=128,
            g_block_tables=None,
            block_table_row_stride=0,
            g_seq_lens_kv=None,
            max_seq_len_kv=256,
            is_persistent=True,
            is_clc_dynamic=True,
            exhaustive_deadlock_race_check=True,
        )

    assert cfg.single_qkv_instance is False
    assert cfg.page_offset_pipeline_stage_counts == ()
    assert cfg.balance_causal_workload is True
    assert work_queue is not None
    assert (
        work_queue.tile_scheduler_config.tile_scheduler_type
        is TileSchedulerType.ClcDynamicPersistent
    )
    assert clc_response_alloc is not None


@pytest.mark.parametrize(
    "input_dtype",
    (Float8E4M3FN, BFloat16),
    ids=("fp8", "bf16"),
)
def test_attention_ts_context_d256_live_paged_clc_uses_distinct_auxiliary_warps(
    input_dtype,
):
    """The live-ragged D256 CLC and page-ID producers cannot overlap."""
    kernel = FmhaTs(
        in_dtype=input_dtype,
        out_dtype=Float16,
        d=256,
        is_persistent=True,
        is_causal=True,
        is_clc_dynamic=True,
        use_paged_kv=True,
        max_kv_len=256,
    )
    cfg = kernel.cfg
    cfg.has_varlen = True
    cfg.has_uniform_varlen = False
    cfg.has_q_offset = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        task_manager, _, _, _, work_queue, clc_response_alloc = build_fmha_task_manager(
            cfg,
            tile_sched_params=None,
            tma_q_desc=None,
            tma_k_desc=None,
            tma_v_desc=None,
            tma_o_desc=None,
            cum_seqlen_q=None,
            cum_seqlen_k=None,
            num_kv_tiles=2,
            q_offset=128,
            g_block_tables=None,
            block_table_row_stride=0,
            g_seq_lens_kv=None,
            max_seq_len_kv=256,
            is_persistent=True,
            is_clc_dynamic=True,
            exhaustive_deadlock_race_check=True,
        )

    tasks = {task.name: task for task in task_manager.tasks}
    scheduler_task = tasks["SchedulerTask"]
    page_offsets_task = tasks["PageTableTask"]
    scheduler_warps = set(
        range(
            scheduler_task.warp_idx, scheduler_task.warp_idx + scheduler_task.num_warps
        )
    )
    page_offsets_warps = set(
        range(
            page_offsets_task.warp_idx,
            page_offsets_task.warp_idx + page_offsets_task.num_warps,
        )
    )

    assert cfg.single_qkv_instance is True
    assert cfg.page_offset_pipeline_stage_counts
    assert scheduler_warps.isdisjoint(page_offsets_warps)
    assert work_queue is not None
    assert (
        work_queue.tile_scheduler_config.tile_scheduler_type
        is TileSchedulerType.ClcDynamicPersistent
    )
    assert clc_response_alloc is not None


@pytest.mark.parametrize(
    "input_dtype",
    (Float8E4M3FN, BFloat16),
    ids=("fp8", "bf16"),
)
@pytest.mark.parametrize(
    "has_q_offset",
    (False, True),
    ids=("zero-offset-heavy-first", "bottom-right-natural"),
)
def test_attention_ts_context_d256_uniform_paged_static_scheduler_is_safe(
    input_dtype,
    has_q_offset: bool,
):
    """Static D256 separates page/epilogue warps in both causal rasters."""

    balance_causal_workload = not has_q_offset
    kernel = FmhaTs(
        in_dtype=input_dtype,
        out_dtype=Float16,
        d=256,
        is_persistent=True,
        is_causal=True,
        balance_causal_workload=balance_causal_workload,
        is_clc_dynamic=False,
        use_paged_kv=True,
        max_kv_len=256,
    )
    cfg = kernel.cfg
    cfg.has_varlen = True
    cfg.has_uniform_varlen = True
    cfg.uniform_seq_len_q = 256 if not has_q_offset else 128
    cfg.uniform_seq_len_k = 256
    cfg.has_q_offset = has_q_offset
    cfg.num_seq_tiles = 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        task_manager, _, _, _, work_queue, clc_response_alloc = build_fmha_task_manager(
            cfg,
            tile_sched_params=None,
            tma_q_desc=None,
            tma_k_desc=None,
            tma_v_desc=None,
            tma_o_desc=None,
            cum_seqlen_q=None,
            cum_seqlen_k=None,
            num_kv_tiles=2,
            q_offset=128 if has_q_offset else 0,
            g_block_tables=None,
            block_table_row_stride=0,
            g_seq_lens_kv=None,
            max_seq_len_kv=256,
            is_persistent=True,
            is_clc_dynamic=False,
            exhaustive_deadlock_race_check=True,
        )

    tasks = {task.name: task for task in task_manager.tasks}
    epilogue_padding_task = tasks["EpiloguePaddingTask"]
    page_offsets_task = tasks["PageTableTask"]
    epilogue_warps = set(
        range(
            epilogue_padding_task.warp_idx,
            epilogue_padding_task.warp_idx + epilogue_padding_task.num_warps,
        )
    )
    page_offsets_warps = set(
        range(
            page_offsets_task.warp_idx,
            page_offsets_task.warp_idx + page_offsets_task.num_warps,
        )
    )
    assert epilogue_warps.isdisjoint(page_offsets_warps)
    assert work_queue is not None
    assert (
        work_queue.tile_scheduler_config.tile_scheduler_type
        is TileSchedulerType.StaticPersistent
    )
    assert clc_response_alloc is None
    assert cfg.uses_causal_reversed_head_batch_seq_tile_order is (
        balance_causal_workload
    )


# ---------------------------------------------------------------------------
# Public validation and bounded SM100/SM103 correctness matrix
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA tensors")
def test_attention_ts_context_public_plan_rejects_unsupported_arch(monkeypatch):
    """The public planner rejects unsupported CUDA architectures before JIT."""

    q = torch.empty((1, 8, 4, 128), dtype=torch.bfloat16, device="cuda")
    k = torch.empty((1, 8, 2, 128), dtype=torch.bfloat16, device="cuda")
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda *_args, **_kwargs: (9, 0)
    )
    with pytest.raises(
        NotImplementedError,
        match=r"requires an SM100a/B200.*GPU.*\(9, 0\)",
    ):
        BatchPrefillTSWrapper().plan(
            device=q.device,
            batch_size=1,
            max_seq_len_q=8,
            max_kv_len=8,
            num_qo_heads=4,
            num_kv_heads=2,
            head_dim=128,
            q_dtype=q.dtype,
            kv_dtype=k.dtype,
        )


@pytest.mark.parametrize(
    ("invalid_contract", "error_type", "message"),
    (
        (
            "dtype",
            NotImplementedError,
            "supports torch.float16, torch.bfloat16, and torch.float8_e4m3fn",
        ),
        ("head-dim", NotImplementedError, "supports head_dim"),
        ("head-ratio", ValueError, "Q head count must be divisible"),
        ("packed-offset", ValueError, "final qo_indptr offset"),
        (
            "dense-window",
            ValueError,
            "positive window_left requires mask_type='causal'",
        ),
    ),
    ids=("dtype", "head-dim", "head-ratio", "packed-offset", "mask-window"),
)
@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_plan_rejects_critical_public_contracts(
    invalid_contract,
    error_type,
    message,
):
    """Keep one bounded rejection matrix for geometry, metadata, and masking."""

    device = torch.device("cuda")
    dtype = torch.float32 if invalid_contract == "dtype" else torch.bfloat16
    head_dim = 64 if invalid_contract == "head-dim" else _HEAD_DIM
    num_qo_heads = 3 if invalid_contract == "head-ratio" else 4
    num_kv_heads = 2
    plan_kwargs = dict(
        device=device,
        batch_size=1,
        max_seq_len_q=8,
        max_kv_len=8,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_dtype=dtype,
        kv_dtype=dtype,
    )

    if invalid_contract == "packed-offset":
        q = torch.empty((2, num_qo_heads, head_dim), dtype=dtype, device=device)
        k = torch.empty((2, num_kv_heads, head_dim), dtype=dtype, device=device)
        v = torch.empty_like(k)
        # The terminal Q offset must cover both packed query rows.
        qo_indptr = torch.tensor((0, 1), dtype=torch.int32, device=device)
        kv_indptr = torch.tensor((0, 2), dtype=torch.int32, device=device)
        plan_kwargs.update(batch_size=1, max_seq_len_q=2, max_kv_len=2, packed=True)
    else:
        q = torch.empty((1, 8, num_qo_heads, head_dim), dtype=dtype, device=device)
        k = torch.empty((1, 8, num_kv_heads, head_dim), dtype=dtype, device=device)
        v = torch.empty_like(k)
    if invalid_contract == "dense-window":
        plan_kwargs["window_left"] = 1

    with pytest.raises(error_type, match=message):
        if invalid_contract == "packed-offset":
            batch_prefill(q, k, v, qo_indptr=qo_indptr, kv_indptr=kv_indptr)
        else:
            BatchPrefillTSWrapper().plan(**plan_kwargs)


@pytest.mark.parametrize("page_size", (16, 32, 64, 128))
def test_attention_ts_context_accepts_public_page_sizes(page_size: int):
    assert context_module._validate_page_size(page_size) == page_size


@pytest.mark.parametrize("page_size", (0, 8, 256))
def test_attention_ts_context_rejects_unsupported_page_sizes(page_size: int):
    with pytest.raises(NotImplementedError, match="supports page_size"):
        context_module._validate_page_size(page_size)


@pytest.mark.parametrize("page_size", (16, 32, 64, 128))
@pytest.mark.parametrize(
    ("head_dim", "qkv_dtype"),
    (
        pytest.param(128, torch.bfloat16, id="d128-bf16"),
        pytest.param(256, _FP8, id="d256-fp8"),
    ),
)
@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_paged_supported_page_sizes_accuracy(
    page_size: int,
    head_dim: int,
    qkv_dtype: torch.dtype,
):
    """Every public page size crosses page and K-tile boundaries accurately."""

    case = _make_paged_context_case(
        q_lengths=(33, 17),
        k_lengths=(129, 97),
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=head_dim,
        qkv_dtype=qkv_dtype,
        mask_type="causal",
        page_size=page_size,
        output_dtype=torch.bfloat16,
        output_scale=1.0,
        seed=2026072000 + page_size + head_dim,
    )
    wrapper = BatchPrefillPagedTSWrapper()
    metadata = _plan_paged_wrapper(wrapper, case)
    assert wrapper._plan_state is not None
    assert dict(wrapper._plan_state.policy)["page_size"] == page_size
    out = _run_paged_wrapper(wrapper, case, metadata)
    _assert_context_correct(out, case.reference)


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_paged_zero_fills_nan_v_tail():
    """Unused V rows must not poison PV after the score mask makes P zero."""

    case = _make_paged_context_case(
        q_lengths=(65, 37),
        k_lengths=(65, 37),
        num_qo_heads=28,
        num_kv_heads=4,
        head_dim=128,
        qkv_dtype=torch.bfloat16,
        mask_type="causal",
        page_size=32,
        output_dtype=torch.bfloat16,
        output_scale=1.0,
        seed=2026090301,
    )
    _poison_invalid_paged_v_tails(case)
    wrapper = BatchPrefillPagedTSWrapper()
    metadata = _plan_paged_wrapper(wrapper, case)
    assert wrapper._plan_state is not None
    assert dict(wrapper._plan_state.policy)["scheduler"] == "clc_dynamic_persistent"
    output = torch.full_like(case.reference.q, float("inf"))
    assert _run_paged_wrapper(wrapper, case, metadata, out=output) is output
    _assert_context_correct(output, case.reference)


@pytest.mark.parametrize("paged", (False, True), ids=("packed", "paged"))
@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_run_rejects_causal_q_longer_than_kv(paged: bool):
    """Bottom-right causal attention requires Sq <= Sk for every request."""

    if paged:
        case = _make_paged_context_case(
            q_lengths=(2, 3),
            k_lengths=(3, 2),
            num_qo_heads=4,
            num_kv_heads=4,
            head_dim=128,
            qkv_dtype=torch.bfloat16,
            mask_type="causal",
            seed=2026071930,
        )
    else:
        case = _make_context_case(
            q_lengths=(2, 3),
            k_lengths=(3, 2),
            num_qo_heads=4,
            num_kv_heads=4,
            qkv_dtype=torch.bfloat16,
            packed=True,
            mask_type="causal",
            device="cuda",
            seed=2026071930,
        )

    with pytest.raises(
        ValueError,
        match=r"batch 1: Sq=3, Sk=2",
    ):
        if paged:
            metadata = _make_native_paged_metadata(case)
            batch_prefill_with_paged_kv_cache(
                case.reference.q,
                case.k_cache,
                case.v_cache,
                case.qo_indptr,
                metadata.block_tables,
                metadata.seq_lens_kv,
                page_size=32,
                mask_type="causal",
            )
        else:
            wrapper = BatchPrefillTSWrapper()
            _plan_wrapper(wrapper, case)
            _run_wrapper(wrapper, case)


def test_attention_ts_context_paged_plan_uses_conservative_dynamic_facts(
    monkeypatch,
):
    """Metadata-free planning never infers uniform or zero-offset lengths."""

    monkeypatch.setattr(
        context_module,
        "_resolve_cuda_device",
        lambda _device: (torch.device("cpu"), 0),
    )
    common = dict(
        device="cuda:0",
        batch_size=2,
        max_seq_len_q=64,
        max_kv_len=128,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=128,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        page_size=32,
        window_left=-1,
        output_dtype=torch.bfloat16,
    )
    causal_geometry = context_module._resolve_paged_plan_geometry(
        **common,
        mask_type="causal",
    )
    dense_geometry = context_module._resolve_paged_plan_geometry(
        **common,
        mask_type="dense",
    )

    assert causal_geometry.uniform_packed_lengths is False
    assert causal_geometry.has_q_offset is True
    assert causal_geometry.packed_dense_k_mask is False
    assert dense_geometry.uniform_packed_lengths is False
    assert dense_geometry.has_q_offset is False
    assert dense_geometry.packed_dense_k_mask is True
    assert context_module._paged_context_compile_spec(causal_geometry) != (
        context_module._paged_context_compile_spec(dense_geometry)
    )


def test_attention_ts_context_paged_rejects_variable_window_before_compile(
    monkeypatch,
):
    """Paged context must not silently dispatch variable windows as dense."""

    with pytest.raises(
        NotImplementedError,
        match=r"variable-window masking.*not supported for paged context",
    ):
        FmhaTs(has_variable_window=True, use_paged_kv=True)

    def fail_compile(*args, **kwargs):
        pytest.fail("paged variable-window validation reached kernel compilation")

    monkeypatch.setattr(context_module, "_get_compiled_paged_context", fail_compile)
    wrapper = BatchPrefillPagedTSWrapper()
    with pytest.raises(
        NotImplementedError,
        match=r"variable_window.*not supported for paged context",
    ):
        wrapper.plan(
            device="cuda:0",
            batch_size=1,
            max_seq_len_q=1,
            max_kv_len=1,
            num_qo_heads=1,
            num_kv_heads=1,
            head_dim=128,
            q_dtype=torch.float16,
            kv_dtype=torch.float16,
            mask_type="variable_window",
            out_dtype=torch.float16,
        )


def test_attention_ts_context_paged_plan_ignores_aggregate_kv_capacity(monkeypatch):
    """Fixed-table plans bound each row, not aggregate batch K/V storage."""

    batch_size = 17
    monkeypatch.setattr(context_module, "_CONTEXT_PADDED_EXTENT_MAX", 128)
    monkeypatch.setattr(
        context_module,
        "_resolve_cuda_device",
        lambda _device: (torch.device("cpu"), 0),
    )
    geometry = context_module._resolve_paged_plan_geometry(
        device="cuda:0",
        batch_size=batch_size,
        max_seq_len_q=1,
        max_kv_len=65,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=128,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        page_size=32,
        mask_type="dense",
        window_left=-1,
        output_dtype=torch.bfloat16,
    )

    assert geometry.max_kv_len == 65
    assert not hasattr(geometry, "max_num_pages_per_seq_kv")
    assert (
        geometry.batch_size * geometry.max_kv_len
        > context_module._CONTEXT_PADDED_EXTENT_MAX
    )

    monkeypatch.setattr(context_module, "_validate_tensor", lambda *_a, **_k: None)
    runtime_indptr = torch.arange(batch_size + 1, dtype=torch.int32)
    context_module._validate_paged_runtime_metadata(
        runtime_indptr,
        torch.zeros((batch_size, 3), dtype=torch.int32),
        torch.ones(batch_size, dtype=torch.int32),
        geometry,
        total_q=batch_size,
        num_physical_pages=1,
    )


def test_attention_ts_context_reserves_int32_work_tile_padding():
    """Packed extents leave room for the largest padded Q work tile."""

    safe_max = 2**31 - 256
    assert safe_max == context_module._CONTEXT_PADDED_EXTENT_MAX
    assert context_module._validate_padded_data_extent(safe_max, "total_q") == safe_max

    with pytest.raises(
        NotImplementedError,
        match=(
            rf"total_q must be <= {safe_max} so padded context work-tile "
            r"coordinates fit in a signed int32"
        ),
    ):
        context_module._validate_padded_data_extent(safe_max + 1, "total_q")

    context_module._validate_query_work_tile_span(
        SimpleNamespace(q_tile_m=128, work_tile_q_seq_tiles=2)
    )
    with pytest.raises(RuntimeError, match=r"assumes at most 256 Q rows.*got 384"):
        context_module._validate_query_work_tile_span(
            SimpleNamespace(q_tile_m=128, work_tile_q_seq_tiles=3)
        )


_CONTEXT_SMOKE_CASES = (
    pytest.param(
        torch.float16,
        False,
        (33,),
        (65,),
        4,
        4,
        "dense",
        -1,
        torch.float16,
        False,
        id="fixed-fp16-dense-mha-one-shot",
    ),
    pytest.param(
        torch.bfloat16,
        False,
        (65,),
        (65,),
        8,
        2,
        "causal",
        -1,
        torch.bfloat16,
        True,
        id="fixed-bf16-causal-single-tile-wrapper",
    ),
    pytest.param(
        torch.bfloat16,
        True,
        (257, 1),
        (257, 257),
        8,
        2,
        "causal",
        -1,
        torch.bfloat16,
        False,
        id="packed-bf16-asymmetric-max-offset-one-shot",
    ),
    pytest.param(
        torch.float16,
        True,
        (33, 257),
        (65, 257),
        4,
        4,
        "dense",
        -1,
        torch.float16,
        False,
        id="packed-fp16-dense-mixed-k-lengths-one-shot",
    ),
    pytest.param(
        _FP8,
        False,
        (33,),
        (65,),
        8,
        1,
        "causal",
        31,
        _FP8,
        True,
        id="fixed-fp8-left-window-wrapper",
    ),
    pytest.param(
        torch.float16,
        False,
        (33,),
        (65,),
        8,
        2,
        "causal",
        31,
        _FP8,
        True,
        id="fixed-window-fp16-to-fp8-wrapper",
    ),
    pytest.param(
        torch.float16,
        False,
        (33,),
        (65,),
        8,
        2,
        "causal",
        31,
        torch.bfloat16,
        True,
        id="fixed-window-fp16-to-bf16-wrapper",
    ),
    pytest.param(
        torch.bfloat16,
        False,
        (33,),
        (65,),
        8,
        2,
        "causal",
        31,
        torch.float16,
        True,
        id="fixed-window-bf16-to-fp16-wrapper",
    ),
    pytest.param(
        torch.bfloat16,
        False,
        (33,),
        (65,),
        8,
        2,
        "causal",
        31,
        _FP8,
        True,
        id="fixed-window-bf16-to-fp8-wrapper",
    ),
    pytest.param(
        _FP8,
        False,
        (33,),
        (65,),
        8,
        2,
        "causal",
        31,
        torch.bfloat16,
        True,
        id="fixed-window-fp8-to-bf16-wrapper",
    ),
    pytest.param(
        _FP8,
        False,
        (33,),
        (65,),
        8,
        2,
        "causal",
        31,
        torch.float16,
        True,
        id="fixed-window-fp8-to-fp16-wrapper",
    ),
    pytest.param(
        torch.bfloat16,
        True,
        (128, 128, 128),
        (128, 192, 319),
        8,
        2,
        "causal",
        64,
        torch.bfloat16,
        False,
        id="packed-bf16-window-clipped-aligned-misaligned-one-shot",
    ),
)


@pytest.mark.parametrize(
    (
        "qkv_dtype",
        "packed",
        "q_lengths",
        "k_lengths",
        "num_qo_heads",
        "num_kv_heads",
        "mask_type",
        "window_left",
        "output_dtype",
        "use_wrapper",
    ),
    _CONTEXT_SMOKE_CASES,
)
@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_bounded_public_correctness_matrix(
    qkv_dtype: torch.dtype,
    packed: bool,
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    num_qo_heads: int,
    num_kv_heads: int,
    mask_type: str,
    window_left: int,
    output_dtype: torch.dtype,
    use_wrapper: bool,
):
    case = _make_context_case(
        q_lengths=q_lengths,
        k_lengths=k_lengths,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        qkv_dtype=qkv_dtype,
        packed=packed,
        mask_type=mask_type,
        window_left=window_left,
        output_dtype=output_dtype,
        device="cuda",
        seed=2026071410 + int(packed) + num_qo_heads,
    )
    if use_wrapper:
        wrapper = BatchPrefillTSWrapper()
        _plan_wrapper(wrapper, case)
        actual = _run_wrapper(wrapper, case)
    else:
        actual = _run_one_shot(case)
    _assert_context_correct(actual, case)


@pytest.mark.parametrize("head_dim", (128, 256), ids=("d128", "d256"))
@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_variable_window_t1_i1_t2_i2(
    head_dim: int,
):
    """Exercise the 2560-token text/image variable-window mask."""

    seq_len = 2560
    left_window = 128
    right_window = 128
    image_segments = ((256, 1280), (1536, 2560))
    case = _make_context_case(
        q_lengths=(seq_len,),
        k_lengths=(seq_len,),
        num_qo_heads=2,
        num_kv_heads=1,
        qkv_dtype=torch.float16,
        packed=False,
        mask_type="variable_window",
        head_dim=head_dim,
        output_dtype=torch.float16,
        output_scale=1.0,
        device="cuda",
        seed=2026073001 + head_dim,
    )

    query_positions = torch.arange(seq_len, dtype=torch.int32, device="cuda")
    starts = torch.clamp(query_positions - left_window, min=0)
    ends = query_positions.clone()
    for segment_start, segment_end in image_segments:
        in_segment = (query_positions >= segment_start) & (
            query_positions < segment_end
        )
        image_end = torch.clamp(
            query_positions + right_window,
            max=segment_end - 1,
        )
        ends = torch.where(in_segment, image_end, ends)
    starts = starts.unsqueeze(0).contiguous()
    ends = ends.unsqueeze(0).contiguous()

    # T1[0:256], I1[256:1280], T2[1280:1536], I2[1536:2560].
    assert starts[0, (0, 255, 256, 1279, 1280, 1535, 1536, 2559)].tolist() == [
        0,
        127,
        128,
        1151,
        1152,
        1407,
        1408,
        2431,
    ]
    end_checkpoints = (0, 255, 256, 1151, 1279, 1280, 1535, 1536, 2431, 2559)
    assert ends[0, end_checkpoints].tolist() == [
        0,
        255,
        384,
        1279,
        1279,
        1280,
        1535,
        1664,
        2559,
        2559,
    ]

    wrapper = BatchPrefillTSWrapper()
    _plan_wrapper(wrapper, case)
    actual = _run_wrapper(
        wrapper,
        case,
        variable_window_token_starts=starts,
        variable_window_token_ends=ends,
    )
    expected = _variable_window_reference(case, starts, ends)
    _assert_context_correct(actual, case, expected=expected)


@pytest.mark.parametrize("head_dim", (128, 256), ids=("d128", "d256"))
@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_variable_window_uses_cta_minimum_start(head_dim: int):
    """A later Q row may extend into a K tile earlier than the CTA's first row."""

    seq_len_q = 256
    seq_len_k = 256
    case = _make_context_case(
        q_lengths=(seq_len_q,),
        k_lengths=(seq_len_k,),
        num_qo_heads=2,
        num_kv_heads=1,
        qkv_dtype=torch.float16,
        packed=False,
        mask_type="variable_window",
        head_dim=head_dim,
        output_dtype=torch.float16,
        output_scale=1.0,
        device="cuda",
        seed=2026081901 + head_dim,
    )
    starts = torch.full((1, seq_len_q), 128, dtype=torch.int32, device="cuda")
    starts[0, 160] = 0
    ends = torch.full((1, seq_len_q), seq_len_k - 1, dtype=torch.int32, device="cuda")

    wrapper = BatchPrefillTSWrapper()
    _plan_wrapper(wrapper, case)
    actual = _run_wrapper(
        wrapper,
        case,
        variable_window_token_starts=starts,
        variable_window_token_ends=ends,
    )
    expected = _variable_window_reference(case, starts, ends)
    _assert_context_correct(actual, case, expected=expected)


@pytest.mark.parametrize("head_dim", (128, 256), ids=("d128", "d256"))
@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_variable_window_clamps_padded_q_rows(head_dim: int):
    """Padded Q rows must not read past flattened per-row window bounds."""

    seq_len = 33
    case = _make_context_case(
        q_lengths=(seq_len, seq_len),
        k_lengths=(seq_len, seq_len),
        num_qo_heads=2,
        num_kv_heads=2,
        qkv_dtype=torch.float16,
        packed=False,
        mask_type="variable_window",
        head_dim=head_dim,
        output_dtype=torch.float16,
        output_scale=1.0,
        device="cuda",
        seed=2026082701 + head_dim,
    )

    query_positions = torch.arange(seq_len, dtype=torch.int32, device="cuda")
    ends = query_positions.unsqueeze(0).expand(2, -1).contiguous()
    starts = torch.clamp(ends - 7, min=0)

    wrapper = BatchPrefillTSWrapper()
    _plan_wrapper(wrapper, case)
    actual = _run_wrapper(
        wrapper,
        case,
        variable_window_token_starts=starts,
        variable_window_token_ends=ends,
    )
    expected = _variable_window_reference(case, starts, ends)
    _assert_context_correct(actual, case, expected=expected)


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
@pytest.mark.parametrize("packed", (False, True), ids=("fixed", "packed"))
def test_attention_ts_context_reuses_compiled_topology_across_batch_sizes(
    packed: bool,
):
    """One resolved context topology accepts different batch extents."""

    wrappers = []
    for batch_size in (3, 4):
        case = _make_context_case(
            q_lengths=(33,) * batch_size,
            k_lengths=(65,) * batch_size,
            num_qo_heads=4,
            num_kv_heads=4,
            qkv_dtype=torch.float16,
            packed=packed,
            mask_type="dense",
            output_dtype=torch.float16,
            device="cuda",
            seed=2026071420 + batch_size,
        )
        wrapper = BatchPrefillTSWrapper()
        _plan_wrapper(wrapper, case)
        actual = _run_wrapper(wrapper, case)
        _assert_context_correct(actual, case)
        wrappers.append(wrapper)

    first_state = wrappers[0]._plan_state
    second_state = wrappers[1]._plan_state
    assert first_state is not None
    assert second_state is not None
    assert first_state.compiled is second_state.compiled


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_paged_context_reuses_compiled_topology_across_batch_sizes():
    """One paged-context topology accepts different batch extents."""

    wrappers = []
    for batch_size in (3, 4):
        case = _make_paged_context_case(
            q_lengths=(33,) * batch_size,
            k_lengths=(65,) * batch_size,
            num_qo_heads=4,
            num_kv_heads=4,
            head_dim=128,
            qkv_dtype=torch.float16,
            mask_type="dense",
            output_dtype=torch.float16,
            seed=2026071430 + batch_size,
        )
        wrapper = BatchPrefillPagedTSWrapper()
        metadata = _plan_paged_wrapper(wrapper, case)
        actual = _run_paged_wrapper(wrapper, case, metadata)
        _assert_context_correct(actual, case.reference)
        wrappers.append(wrapper)

    first_state = wrappers[0]._plan_state
    second_state = wrappers[1]._plan_state
    assert first_state is not None
    assert second_state is not None
    assert first_state.compiled is second_state.compiled


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_variable_window_graph_reloads_runtime_bounds():
    """Graph replay refreshes CTA minima from in-place-updated window bounds."""

    seq_len = 129
    case = _make_context_case(
        q_lengths=(seq_len,),
        k_lengths=(seq_len,),
        num_qo_heads=2,
        num_kv_heads=1,
        qkv_dtype=torch.float16,
        packed=False,
        mask_type="variable_window",
        head_dim=128,
        output_dtype=torch.float16,
        output_scale=1.0,
        device="cuda",
        seed=2026090401,
    )
    starts = torch.full((1, seq_len), 128, dtype=torch.int32, device="cuda")
    ends = torch.full((1, seq_len), seq_len - 1, dtype=torch.int32, device="cuda")
    wrapper = BatchPrefillTSWrapper()
    _plan_wrapper(wrapper, case)
    graph_out = torch.full_like(case.q, float("nan"), dtype=case.output_dtype)

    assert (
        _run_wrapper(
            wrapper,
            case,
            variable_window_token_starts=starts,
            variable_window_token_ends=ends,
            out=graph_out,
        )
        is graph_out
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_wrapper(
            wrapper,
            case,
            variable_window_token_starts=starts,
            variable_window_token_ends=ends,
            out=graph_out,
            validate=False,
        )
    assert captured is graph_out
    assert wrapper._plan_state is not None
    assert wrapper._plan_state.variable_window_cta_starts.tolist() == [128]

    starts.zero_()
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    assert wrapper._plan_state.variable_window_cta_starts.tolist() == [0]
    expected = _variable_window_reference(case, starts, ends)
    _assert_context_correct(graph_out, case, expected=expected)


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_fixed_dense_k_tail_excludes_tma_padding():
    # With zero Q/K, every real key has score zero. A missing right-edge mask
    # therefore dilutes the output by 65/128 because TMA zero-fills the rest
    # of the final K/V tile and softmax would count those lanes as real keys.
    case = _make_context_case(
        q_lengths=(33,),
        k_lengths=(65,),
        num_qo_heads=4,
        num_kv_heads=4,
        qkv_dtype=torch.float16,
        packed=False,
        mask_type="dense",
        output_dtype=torch.float16,
        output_scale=0.75,
        device="cuda",
        seed=2026071422,
    )
    case.q.zero_()
    case.k.zero_()
    case.v.fill_(1.0)

    actual = _run_one_shot(case)
    _assert_context_correct(actual, case)
    torch.testing.assert_close(
        actual.float(),
        torch.full_like(actual.float(), case.output_scale),
        rtol=0.0,
        atol=1e-3,
    )


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_packed_dense_k_bounds_exclude_peer_requests():
    # The first request owns only 65 K/V rows while the shared domain has
    # three 128-row tiles. Give the following request a distinct V marker so
    # both final-tile crossing and wholly-OOB tiles are observable.
    case = _make_context_case(
        q_lengths=(33, 257),
        k_lengths=(65, 257),
        num_qo_heads=4,
        num_kv_heads=4,
        qkv_dtype=torch.float16,
        packed=True,
        mask_type="dense",
        output_dtype=torch.float16,
        output_scale=0.75,
        device="cuda",
        seed=2026071423,
    )
    case.q.zero_()
    case.k.zero_()
    case.v[:65].fill_(1.0)
    case.v[65:].fill_(2.0)

    actual = _run_one_shot(case)
    _assert_context_correct(actual, case)
    torch.testing.assert_close(
        actual[:33].float(),
        torch.full_like(actual[:33].float(), 0.75),
        rtol=0.0,
        atol=1e-3,
    )
    torch.testing.assert_close(
        actual[33:].float(),
        torch.full_like(actual[33:].float(), 1.5),
        rtol=0.0,
        atol=1e-3,
    )


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_uniform_aligned_packed_dense_accuracy():
    case = _make_context_case(
        q_lengths=(33, 257),
        k_lengths=(128, 128),
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=256,
        qkv_dtype=_FP8,
        packed=True,
        mask_type="dense",
        output_dtype=torch.bfloat16,
        device="cuda",
        seed=2026071510,
    )
    wrapper = BatchPrefillTSWrapper()
    _plan_wrapper(wrapper, case)
    assert wrapper._plan_state is not None
    assert dict(wrapper._plan_state.policy)["uniform_packed_lengths"] is False
    _assert_context_correct(_run_wrapper(wrapper, case), case)


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_uniform_packed_offsets_accuracy():
    case = _make_context_case(
        q_lengths=(65, 65),
        k_lengths=(128, 128),
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=128,
        qkv_dtype=_FP8,
        packed=True,
        mask_type="dense",
        output_dtype=torch.bfloat16,
        device="cuda",
        seed=2026071810,
    )
    wrapper = BatchPrefillTSWrapper()
    _plan_wrapper(wrapper, case)
    assert wrapper._plan_state is not None
    assert dict(wrapper._plan_state.policy)["uniform_packed_lengths"] is False
    _assert_context_correct(_run_wrapper(wrapper, case), case)


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_uniform_packed_window_offsets_accuracy():
    case = _make_context_case(
        q_lengths=(33, 33),
        k_lengths=(65, 65),
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=128,
        qkv_dtype=torch.bfloat16,
        packed=True,
        mask_type="causal",
        window_left=31,
        output_dtype=torch.bfloat16,
        device="cuda",
        seed=2026071811,
    )
    wrapper = BatchPrefillTSWrapper()
    _plan_wrapper(wrapper, case)
    assert wrapper._plan_state is not None
    assert dict(wrapper._plan_state.policy)["uniform_packed_lengths"] is False
    _assert_context_correct(_run_wrapper(wrapper, case), case)


@pytest.mark.parametrize("head_dim", (128, 256), ids=("d128", "d256"))
@pytest.mark.parametrize(
    "k_lengths",
    (
        pytest.param((1152, 1152), id="uniform-aligned-long"),
        pytest.param((1025, 1153), id="mixed-partial-long"),
        pytest.param((65, 257), id="mixed-partial-short"),
    ),
)
@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_paged_dense_k_mask_accuracy(
    head_dim: int,
    k_lengths: tuple[int, int],
):
    case = _make_paged_context_case(
        q_lengths=(33, 257),
        k_lengths=k_lengths,
        num_qo_heads=8,
        num_kv_heads=4,
        head_dim=head_dim,
        qkv_dtype=_FP8,
        mask_type="dense",
        seed=2026071511 + head_dim + sum(k_lengths),
    )
    case = replace(
        case,
        reference=replace(case.reference, output_dtype=torch.bfloat16),
    )
    wrapper = BatchPrefillPagedTSWrapper()
    metadata = _plan_paged_wrapper(wrapper, case)
    output = _run_paged_wrapper(wrapper, case, metadata)
    _assert_context_correct(output, case.reference)


@pytest.mark.parametrize(
    ("head_dim", "stages_page_offsets"),
    (
        pytest.param(128, False, id="d128-direct-page-ids"),
        pytest.param(256, True, id="d256-staged-page-ids"),
    ),
)
@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_paged_invalid_padding_ids_are_not_dereferenced(
    head_dim: int,
    stages_page_offsets: bool,
) -> None:
    """Both lookup paths ignore poisoned columns in a noncompact table."""

    case = _make_paged_context_case(
        q_lengths=(17, 17),
        k_lengths=(65, 33),
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=head_dim,
        qkv_dtype=torch.bfloat16,
        mask_type="causal",
        output_scale=1.0,
        seed=2026090303 + head_dim,
    )
    invalid_page_id = int(case.k_cache.shape[0]) + 17
    metadata = _make_native_paged_metadata(
        case,
        extra_page_columns=2,
        row_stride_multiplier=2,
    )
    page_counts = tuple(
        math.ceil(length / case.page_size) for length in case.reference.k_lengths
    )
    for batch_idx, page_count in enumerate(page_counts):
        padding_count = metadata.block_tables.shape[1] - page_count
        metadata.block_tables[batch_idx, page_count:] = torch.tensor(
            tuple((-1, invalid_page_id)[offset % 2] for offset in range(padding_count)),
            dtype=torch.int32,
            device="cuda",
        )
    assert metadata.block_tables.stride(0) == 2 * metadata.block_tables.shape[1]
    assert metadata.block_tables.stride(1) == 1
    wrapper = BatchPrefillPagedTSWrapper()
    _plan_paged_wrapper(wrapper, case)

    cfg = FmhaTs(
        in_dtype=BFloat16,
        out_dtype=BFloat16,
        d=head_dim,
        is_persistent=True,
        is_causal=True,
        is_clc_dynamic=True,
        use_paged_kv=True,
        num_tokens_per_page=case.page_size,
        max_kv_len=max(case.reference.k_lengths),
    ).cfg
    assert cfg.stages_page_offsets_in_smem is stages_page_offsets
    assert wrapper._plan_state is not None
    assert dict(wrapper._plan_state.policy)["scheduler"] == "clc_dynamic_persistent"

    output = _run_paged_wrapper(wrapper, case, metadata)
    _assert_context_correct(output, case.reference)


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
@pytest.mark.parametrize(
    "qkv_dtype",
    (_FP8, torch.bfloat16),
    ids=("fp8", "bf16"),
)
def test_attention_ts_context_d128_paged_s16k_runtime(qkv_dtype: torch.dtype):
    case = _make_paged_context_case(
        q_lengths=(1, 1),
        k_lengths=(16384, 16384),
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=128,
        qkv_dtype=qkv_dtype,
        mask_type="dense",
        seed=2026071522,
    )
    if qkv_dtype == torch.bfloat16:
        # Both requests intentionally alias the same physical pages. Preserve
        # the matching logical reference while guarding repeated page IDs in
        # the public paged-KV path without adding another compiled shape.
        pages_per_request = 16384 // 32
        first_pages = case.paged_kv_indices[:pages_per_request]
        second_pages = case.paged_kv_indices[pages_per_request:]
        second_pages.copy_(first_pages)
        case.reference.k[16384:].copy_(case.reference.k[:16384])
        case.reference.v[16384:].copy_(case.reference.v[:16384])
        assert (
            torch.unique(case.paged_kv_indices).numel() < case.paged_kv_indices.numel()
        )
    case = replace(
        case,
        reference=replace(case.reference, output_dtype=torch.bfloat16),
    )
    wrapper = BatchPrefillPagedTSWrapper()
    metadata = _plan_paged_wrapper(wrapper, case)
    assert wrapper._plan_state is not None
    assert dict(wrapper._plan_state.policy)["scheduler"] == "nonpersistent"
    output = _run_paged_wrapper(wrapper, case, metadata)
    _assert_context_correct(output, case.reference)


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_fixed_window_tail_excludes_left_marker():
    # This shape has exactly one K/V tile in the head-paired window domain.
    # Every query's visible interval starts after K=0, so a marker at K=0
    # detects a tail path that applies only the causal right bound.
    case = _make_context_case(
        q_lengths=(33,),
        k_lengths=(65,),
        num_qo_heads=8,
        num_kv_heads=2,
        qkv_dtype=torch.bfloat16,
        packed=False,
        mask_type="causal",
        window_left=31,
        output_dtype=torch.bfloat16,
        output_scale=1.0,
        device="cuda",
        seed=2026071419,
    )
    case.q.zero_()
    case.k.zero_()
    case.v.zero_()
    case.v[0, 0, :, 0] = 64.0

    wrapper = BatchPrefillTSWrapper()
    _plan_wrapper(wrapper, case)
    actual = _run_wrapper(wrapper, case)
    _assert_context_correct(actual, case)
    assert torch.count_nonzero(actual) == 0


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_fixed_window_loop_excludes_right_marker():
    # The non-tile-aligned bottom-right offset makes K tile 0 a LOOP tile even
    # though its upper lanes are to the right of early query rows. A tail-only
    # right mask therefore leaks this marker into Q row zero.
    case = _make_context_case(
        q_lengths=(64,),
        k_lengths=(129,),
        num_qo_heads=8,
        num_kv_heads=2,
        qkv_dtype=torch.float16,
        packed=False,
        mask_type="causal",
        window_left=63,
        output_dtype=torch.float16,
        output_scale=1.0,
        device="cuda",
        seed=2026071424,
    )
    case.q.zero_()
    case.k.zero_()
    case.v.zero_()
    case.v[0, 66:, :, 0] = 64.0

    actual = _run_one_shot(case)
    _assert_context_correct(actual, case)
    assert torch.count_nonzero(actual[0, 0]) == 0


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_d256_fixed_head_paired_window_runtime():
    case = _make_context_case(
        q_lengths=(64,),
        k_lengths=(129,),
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=256,
        qkv_dtype=torch.bfloat16,
        packed=False,
        mask_type="causal",
        window_left=63,
        output_dtype=torch.bfloat16,
        output_scale=1.0,
        device="cuda",
        seed=2026071623,
    )
    case.q.zero_()
    case.k.zero_()
    case.v.zero_()
    case.v[0, 66:, :, 0] = 64.0

    wrapper = BatchPrefillTSWrapper()
    _plan_wrapper(wrapper, case)
    output = _run_wrapper(wrapper, case)
    _assert_context_correct(output, case)
    assert torch.count_nonzero(output[0, 0]) == 0


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_d256_bf16_fixed_dense_runtime():
    # BF16 D256 is the largest fixed-input footprint and therefore guards the
    # B200 dynamic-SMEM launch limit independently of automatic scheduling.
    case = _make_context_case(
        q_lengths=(129,),
        k_lengths=(257,),
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=256,
        qkv_dtype=torch.bfloat16,
        packed=False,
        mask_type="dense",
        output_dtype=torch.bfloat16,
        device="cuda",
        seed=2026071625,
    )
    wrapper = BatchPrefillTSWrapper()
    _plan_wrapper(wrapper, case)
    assert wrapper._plan_state is not None
    assert dict(wrapper._plan_state.policy)["scheduler"] == "static_persistent"

    for _ in range(2):
        output = _run_wrapper(wrapper, case)
        _assert_context_correct(output, case)


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
@pytest.mark.parametrize(
    ("qkv_dtype", "output_dtype", "kv_length"),
    (
        pytest.param(torch.bfloat16, torch.bfloat16, 256, id="bf16-full-ring"),
        pytest.param(
            torch.bfloat16,
            torch.bfloat16,
            993,
            id="bf16-split-rings-partial-tail",
        ),
        pytest.param(_FP8, torch.bfloat16, 256, id="fp8-full-ring"),
        pytest.param(_FP8, torch.bfloat16, 1024, id="fp8-split-rings"),
    ),
)
def test_attention_ts_context_d256_paged_dense_persistent_capacity_runtime(
    qkv_dtype: torch.dtype,
    output_dtype: torch.dtype,
    kv_length: int,
):
    # Use more than three resident CTA waves across distinct batch page tables
    # while keeping Q compact. Eight pages retain the full page-ID ring and 32
    # pages select split K/V rings, covering the tight BF16 footprint and the
    # capacity-derived FP8 stage counts without a tuned staging cap.
    case = _make_paged_context_case(
        q_lengths=(1, 1, 1, 1),
        k_lengths=(kv_length,) * 4,
        num_qo_heads=128,
        num_kv_heads=32,
        head_dim=256,
        qkv_dtype=qkv_dtype,
        mask_type="dense",
        seed=2026071826,
    )
    case = replace(case, reference=replace(case.reference, output_dtype=output_dtype))
    if kv_length % case.page_size:
        # A 993-token domain still occupies 32 pages, selecting the reused
        # page-window schedule while leaving the final V page tail invalid.
        _poison_invalid_paged_v_tails(case)
    wrapper = BatchPrefillPagedTSWrapper()
    metadata = _plan_paged_wrapper(wrapper, case)
    assert wrapper._plan_state is not None
    assert dict(wrapper._plan_state.policy)["scheduler"] == "static_persistent"
    if kv_length == 993:
        assert wrapper._plan_state.geometry.max_kv_len == 993
        cfg = FmhaTs(
            in_dtype=BFloat16,
            out_dtype=BFloat16,
            d=256,
            is_persistent=True,
            is_causal=False,
            use_paged_kv=True,
            num_tokens_per_page=case.page_size,
            max_kv_len=993,
        ).cfg
        assert cfg.reuses_page_table_windows
    output = _run_paged_wrapper(wrapper, case, metadata)
    _assert_context_correct(output, case.reference)


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_d256_fp8_paged_dense_crosses_page_windows():
    """A persistent launch consumes two complete 64-ID windows per request."""

    case = _make_paged_context_case(
        q_lengths=(1, 1, 1, 1),
        k_lengths=(4096, 4096, 4096, 4096),
        num_qo_heads=64,
        num_kv_heads=4,
        head_dim=256,
        qkv_dtype=_FP8,
        mask_type="dense",
        seed=2026071935,
    )
    case = replace(
        case,
        reference=replace(case.reference, output_dtype=torch.bfloat16),
    )
    wrapper = BatchPrefillPagedTSWrapper()
    metadata = _plan_paged_wrapper(wrapper, case)

    assert wrapper._plan_state is not None
    assert dict(wrapper._plan_state.policy)["scheduler"] == "static_persistent"
    output = _run_paged_wrapper(wrapper, case, metadata)
    _assert_context_correct(output, case.reference)


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_d256_paged_head_paired_window_runtime():
    case = _make_paged_context_case(
        q_lengths=(64,),
        k_lengths=(129,),
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=256,
        qkv_dtype=torch.bfloat16,
        mask_type="causal",
        window_left=63,
        output_scale=1.0,
        seed=2026071624,
    )
    wrapper = BatchPrefillPagedTSWrapper()
    metadata = _plan_paged_wrapper(wrapper, case)
    assert wrapper._plan_state is not None
    assert dict(wrapper._plan_state.policy)["scheduler"] == "clc_dynamic_persistent"
    output = _run_paged_wrapper(wrapper, case, metadata)
    _assert_context_correct(output, case.reference)


@pytest.mark.parametrize(
    ("q_lengths", "k_lengths", "expected_q_offset"),
    (
        pytest.param((257, 257), (257, 257), False, id="zero-offset-heavy-first"),
        pytest.param(
            (257, 257),
            (513, 513),
            True,
            id="bottom-right-natural",
        ),
    ),
)
@pytest.mark.parametrize(
    "qkv_dtype",
    (torch.bfloat16, _FP8),
    ids=("bf16", "fp8"),
)
@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_d256_paged_dynamic_causal_runtime(
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    expected_q_offset: bool,
    qkv_dtype: torch.dtype,
):
    """Conservative causal planning handles both runtime offset geometries."""

    case = _make_paged_context_case(
        q_lengths=q_lengths,
        k_lengths=k_lengths,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=256,
        qkv_dtype=qkv_dtype,
        mask_type="causal",
        output_scale=1.0,
        seed=2026071933 + int(expected_q_offset) + int(qkv_dtype == _FP8) * 100,
    )
    wrapper = BatchPrefillPagedTSWrapper()
    metadata = _plan_paged_wrapper(wrapper, case)

    assert wrapper._plan_state is not None
    assert wrapper._plan_state.geometry.uniform_packed_lengths is False
    assert wrapper._plan_state.geometry.has_q_offset is True
    assert dict(wrapper._plan_state.policy)["scheduler"] == "clc_dynamic_persistent"
    assert (q_lengths != k_lengths) is expected_q_offset

    direct_out = torch.full_like(
        case.reference.q,
        float("nan"),
        dtype=case.reference.output_dtype,
    )
    _run_paged_wrapper(wrapper, case, metadata, out=direct_out)
    graph_out = torch.full_like(direct_out, float("nan"))
    graph = _capture_context_graph(
        wrapper,
        case.reference.q,
        case.k_cache,
        case.v_cache,
        graph_out,
        metadata,
    )
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    _assert_context_correct(direct_out, case.reference)
    _assert_context_correct(graph_out, case.reference)


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
@pytest.mark.parametrize(
    ("head_dim", "plan_k_lengths"),
    (
        pytest.param(128, (65, 33), id="d128-direct-page-ids"),
        pytest.param(256, (1057, 1025), id="d256-staged-page-window"),
    ),
)
def test_attention_ts_context_paged_graph_replay_reads_updated_fixed_metadata(
    head_dim: int,
    plan_k_lengths: tuple[int, int],
) -> None:
    """Captured runs reload noncompact page-table rows and lengths in place."""

    case = _make_paged_context_case(
        q_lengths=(17, 17),
        k_lengths=plan_k_lengths,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=head_dim,
        qkv_dtype=torch.bfloat16,
        mask_type="causal",
        output_scale=1.0,
        seed=2026090302 + head_dim,
    )
    wrapper = BatchPrefillPagedTSWrapper()
    metadata = _plan_paged_wrapper(
        wrapper,
        case,
        extra_page_columns=1,
        row_stride_multiplier=2,
    )
    assert wrapper._plan_state is not None
    assert dict(wrapper._plan_state.policy)["scheduler"] == "clc_dynamic_persistent"

    cfg = FmhaTs(
        in_dtype=BFloat16,
        out_dtype=BFloat16,
        d=head_dim,
        is_persistent=True,
        is_causal=True,
        is_clc_dynamic=True,
        use_paged_kv=True,
        num_tokens_per_page=case.page_size,
        max_kv_len=max(plan_k_lengths),
    ).cfg
    assert cfg.stages_page_offsets_in_smem is (head_dim == 256)

    warmup_out = torch.empty_like(case.reference.q)
    _run_paged_wrapper(wrapper, case, metadata, out=warmup_out)

    graph_out = torch.full_like(case.reference.q, float("nan"))
    graph = _capture_context_graph(
        wrapper,
        case.reference.q,
        case.k_cache,
        case.v_cache,
        graph_out,
        metadata,
    )
    metadata_ptrs = tuple(
        tensor.data_ptr()
        for tensor in (
            metadata.block_tables,
            metadata.seq_lens_kv,
        )
    )
    metadata_stride = metadata.block_tables.stride()

    runtime_k_lengths = tuple(reversed(plan_k_lengths))
    runtime_page_counts = tuple(
        math.ceil(length / case.page_size) for length in runtime_k_lengths
    )
    runtime_page_indptr = _cumulative(runtime_page_counts)
    runtime_page_indices = tuple(range(1, runtime_page_indptr[-1] + 1))
    runtime_block_tables = torch.full_like(metadata.block_tables, -911)
    for batch_idx, (begin, end) in enumerate(itertools.pairwise(runtime_page_indptr)):
        runtime_block_tables[batch_idx, : end - begin] = torch.tensor(
            runtime_page_indices[begin:end], dtype=torch.int32, device="cuda"
        )
    if head_dim == 256:
        assert max(runtime_page_counts) > cfg.page_table_window_entries
    metadata.block_tables.copy_(runtime_block_tables)
    metadata.seq_lens_kv.copy_(
        torch.tensor(runtime_k_lengths, dtype=torch.int32, device="cuda")
    )
    assert metadata_ptrs == tuple(
        tensor.data_ptr()
        for tensor in (
            metadata.block_tables,
            metadata.seq_lens_kv,
        )
    )
    assert metadata.block_tables.stride() == metadata_stride

    def gather_logical_cache(cache: torch.Tensor) -> torch.Tensor:
        requests = []
        for batch_idx, k_length in enumerate(runtime_k_lengths):
            page_begin = runtime_page_indptr[batch_idx]
            page_end = runtime_page_indptr[batch_idx + 1]
            page_ids = runtime_page_indices[page_begin:page_end]
            requests.append(
                cache[list(page_ids)]
                .permute(0, 2, 1, 3)
                .reshape(-1, cache.shape[1], cache.shape[3])[:k_length]
            )
        return torch.cat(requests)

    runtime_reference = replace(
        case.reference,
        k=gather_logical_cache(case.k_cache),
        v=gather_logical_cache(case.v_cache),
        kv_indptr=torch.tensor(
            _cumulative(runtime_k_lengths), dtype=torch.int32, device="cuda"
        ),
        k_lengths=runtime_k_lengths,
    )
    expected = _context_reference(runtime_reference)

    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    one_shot_out = batch_prefill_with_paged_kv_cache(
        case.reference.q,
        case.k_cache,
        case.v_cache,
        metadata.qo_indptr,
        metadata.block_tables,
        metadata.seq_lens_kv,
        page_size=case.page_size,
        mask_type=case.reference.mask_type,
        sm_scale=case.reference.sm_scale,
        output_scale=case.reference.output_scale,
        out_dtype=case.reference.output_dtype,
    )

    _assert_context_correct(graph_out, runtime_reference, expected=expected)
    _assert_context_correct(one_shot_out, runtime_reference, expected=expected)


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_d256_fixed_causal_single_tile_runtime():
    case = _make_context_case(
        q_lengths=(65,),
        k_lengths=(65,),
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=256,
        qkv_dtype=torch.bfloat16,
        packed=False,
        mask_type="causal",
        output_dtype=torch.bfloat16,
        device="cuda",
        seed=2026071519,
    )
    wrapper = BatchPrefillTSWrapper()
    _plan_wrapper(wrapper, case)
    assert wrapper._plan_state is not None
    assert dict(wrapper._plan_state.policy)["scheduler"] == "clc_dynamic_persistent"
    output = _run_wrapper(wrapper, case)
    _assert_context_correct(output, case)


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_paged_one_shot_causal_partial_tail_d256():
    case = _make_paged_context_case(
        q_lengths=(17, 65),
        # Cross the 128-token KV tile boundary so poisoned V tails also
        # exercise nonzero logical tile/page coordinates.
        k_lengths=(177, 193),
        num_qo_heads=8,
        num_kv_heads=4,
        head_dim=256,
        qkv_dtype=torch.float16,
        mask_type="causal",
        seed=2026071520,
    )
    _poison_invalid_paged_v_tails(case)
    metadata = _make_native_paged_metadata(case)
    output = torch.full_like(case.reference.q, float("inf"))
    returned = batch_prefill_with_paged_kv_cache(
        case.reference.q,
        case.k_cache,
        case.v_cache,
        case.qo_indptr,
        metadata.block_tables,
        metadata.seq_lens_kv,
        page_size=32,
        mask_type="causal",
        sm_scale=case.reference.sm_scale,
        output_scale=case.reference.output_scale,
        out_dtype=case.reference.output_dtype,
        out=output,
    )
    assert returned is output
    assert case.paged_kv_last_page_len.tolist() == [17, 1]
    assert case.paged_kv_indices.tolist() != list(range(case.paged_kv_indices.numel()))
    _assert_context_correct(output, case.reference)


@pytest.mark.parametrize("head_dim", (128, 256), ids=("d128", "d256"))
@pytest.mark.parametrize("paged", (False, True), ids=("packed", "paged"))
@pytest.mark.parametrize(
    "qkv_dtype",
    (torch.bfloat16, _FP8),
    ids=("bf16", "fp8"),
)
@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_live_q_offsets_expand_causal_domain_on_graph_replay(
    paged: bool,
    head_dim: int,
    qkv_dtype: torch.dtype,
):
    """Live Q offsets must drive the causal K-tile count for each request.

    Both plans have total_q=564 and max(Sq)=500. Redistributing the same Q
    storage from (500, 64) to (64, 500) raises request zero's bottom-right
    offset from 500 to 936. A domain frozen at the plan-time maximum omits the
    K/V tail [768:1000], which the deterministic V marker makes observable.
    """

    plan_q_lengths = (500, 64)
    replay_q_lengths = (64, 500)
    k_lengths = (1000, 500)
    assert replay_q_lengths[0] < k_lengths[0]
    assert replay_q_lengths[1] == k_lengths[1]
    runtime_qo_indptr = torch.tensor(
        _cumulative(replay_q_lengths), dtype=torch.int32, device="cuda"
    )
    paged_metadata = None

    if paged:
        paged_case = _make_paged_context_case(
            q_lengths=plan_q_lengths,
            k_lengths=k_lengths,
            num_qo_heads=4,
            num_kv_heads=4,
            head_dim=head_dim,
            qkv_dtype=qkv_dtype,
            mask_type="causal",
            output_scale=1.0,
            seed=2026071931 + head_dim + (1 if qkv_dtype == _FP8 else 0),
        )
        paged_case = replace(
            paged_case,
            reference=replace(paged_case.reference, output_dtype=torch.bfloat16),
        )
        reference = paged_case.reference
        reference.q.zero_()
        reference.k.zero_()
        reference.v.zero_()
        reference.v[768:1000].fill_(1.0)
        paged_case.k_cache.zero_()
        paged_case.v_cache.zero_()
        page_ids = paged_case.paged_kv_indices.tolist()
        page_offsets = paged_case.paged_kv_indptr.tolist()
        request_zero_page_begin = page_offsets[0]
        for logical_page in range(768 // 32, math.ceil(1000 / 32)):
            physical_page = page_ids[request_zero_page_begin + logical_page]
            page_extent = min(32, 1000 - logical_page * 32)
            paged_case.v_cache[physical_page, :, :page_extent].fill_(1.0)

        # Reconstruct logical V from the nonidentity page table so the marker
        # cannot accidentally spill into request one or page-tail padding.
        logical_v_pages = []
        for batch_idx, k_length in enumerate(k_lengths):
            for logical_page in range(math.ceil(k_length / 32)):
                physical_page = page_ids[page_offsets[batch_idx] + logical_page]
                page_extent = min(32, k_length - logical_page * 32)
                logical_v_pages.append(
                    paged_case.v_cache[physical_page, :, :page_extent].permute(1, 0, 2)
                )
        torch.testing.assert_close(
            torch.cat(logical_v_pages), reference.v, rtol=0.0, atol=0.0
        )

        wrapper = BatchPrefillPagedTSWrapper()
        paged_metadata = _plan_paged_wrapper(wrapper, paged_case)
        q = reference.q
        k = paged_case.k_cache
        v = paged_case.v_cache
        qo_indptr = paged_case.qo_indptr
    else:
        reference = _make_context_case(
            q_lengths=plan_q_lengths,
            k_lengths=k_lengths,
            num_qo_heads=4,
            num_kv_heads=4,
            head_dim=head_dim,
            qkv_dtype=qkv_dtype,
            packed=True,
            mask_type="causal",
            output_dtype=torch.bfloat16,
            output_scale=1.0,
            device="cuda",
            seed=2026071931 + head_dim + (1 if qkv_dtype == _FP8 else 0),
        )
        reference.q.zero_()
        reference.k.zero_()
        reference.v.zero_()
        reference.v[768:1000].fill_(1.0)

        wrapper = BatchPrefillTSWrapper()
        _plan_wrapper(wrapper, reference)
        q = reference.q
        k = reference.k
        v = reference.v
        assert reference.qo_indptr is not None
        qo_indptr = reference.qo_indptr

    assert wrapper._plan_state is not None
    policy = wrapper._plan_state.policy
    assert dict(policy)["scheduler"] == "clc_dynamic_persistent"
    graph_out = torch.full_like(q, float("nan"), dtype=reference.output_dtype)
    graph = _capture_context_graph(
        wrapper,
        q,
        k,
        v,
        graph_out,
        paged_metadata,
        qo_indptr=qo_indptr,
        kv_indptr=reference.kv_indptr,
    )

    qo_indptr.copy_(runtime_qo_indptr)
    runtime_reference = replace(
        reference,
        qo_indptr=qo_indptr,
        q_lengths=replay_q_lengths,
    )
    expected = _context_reference(runtime_reference)

    direct_out = torch.full_like(q, float("nan"), dtype=reference.output_dtype)
    if paged_metadata is None:
        wrapper.run(
            q,
            k,
            v,
            qo_indptr,
            reference.kv_indptr,
            out=direct_out,
        )
    else:
        wrapper.run(
            q,
            k,
            v,
            paged_metadata.qo_indptr,
            paged_metadata.block_tables,
            paged_metadata.seq_lens_kv,
            out=direct_out,
        )
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    # Check graph replay first so this specifically locks down live metadata
    # behavior under capture; the direct launch must obey the same contract.
    _assert_context_correct(graph_out, runtime_reference, expected=expected)
    _assert_context_correct(direct_out, runtime_reference, expected=expected)


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_live_zero_offset_qk_redistribution_graph_replay():
    """Live Q/K redistribution preserves D128 zero-offset paired semantics."""

    plan_lengths = (500, 64)
    replay_lengths = (300, 264)
    assert all(math.ceil(length / 128) == 3 for length in replay_lengths)
    case = _make_context_case(
        q_lengths=plan_lengths,
        k_lengths=plan_lengths,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=128,
        qkv_dtype=torch.bfloat16,
        packed=True,
        mask_type="causal",
        output_dtype=torch.bfloat16,
        output_scale=1.0,
        device="cuda",
        seed=2026071932,
    )
    case.q.zero_()
    case.k.zero_()
    runtime_v_rows = torch.cat(
        tuple(
            torch.arange(1, length + 1, dtype=torch.float32, device="cuda") / length
            for length in replay_lengths
        )
    ).to(torch.bfloat16)
    case.v.copy_(runtime_v_rows[:, None, None].expand_as(case.v))

    wrapper = BatchPrefillTSWrapper()
    _plan_wrapper(wrapper, case)
    assert wrapper._plan_state is not None
    assert wrapper._plan_state.geometry.has_q_offset is True
    assert dict(wrapper._plan_state.policy)["pairing"] == "query"
    assert dict(wrapper._plan_state.policy)["uniform_packed_lengths"] is False

    graph_out = torch.full_like(case.q, float("nan"), dtype=case.output_dtype)
    graph = _capture_context_graph(
        wrapper,
        case.q,
        case.k,
        case.v,
        graph_out,
        qo_indptr=case.qo_indptr,
        kv_indptr=case.kv_indptr,
    )

    runtime_indptr = torch.tensor(
        _cumulative(replay_lengths), dtype=torch.int32, device="cuda"
    )
    assert case.qo_indptr is not None and case.kv_indptr is not None
    case.qo_indptr.copy_(runtime_indptr)
    case.kv_indptr.copy_(runtime_indptr)
    runtime_reference = replace(
        case,
        q_lengths=replay_lengths,
        k_lengths=replay_lengths,
    )
    expected = _context_reference(runtime_reference)

    direct_out = torch.full_like(case.q, float("nan"), dtype=case.output_dtype)
    _run_wrapper(wrapper, case, out=direct_out)
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    _assert_context_correct(graph_out, runtime_reference, expected=expected)
    _assert_context_correct(direct_out, runtime_reference, expected=expected)


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_live_k_redistribution_graph_replay():
    """Live K offsets independently update causal domains and right bounds."""

    q_lengths = (64, 500)
    plan_k_lengths = (1000, 500)
    replay_k_lengths = (500, 1000)
    case = _make_context_case(
        q_lengths=q_lengths,
        k_lengths=plan_k_lengths,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=128,
        qkv_dtype=torch.bfloat16,
        packed=True,
        mask_type="causal",
        output_dtype=torch.bfloat16,
        output_scale=1.0,
        device="cuda",
        seed=2026071937,
    )
    case.q.zero_()
    case.k.zero_()
    marker = (
        torch.arange(
            1,
            case.v.shape[0] + 1,
            dtype=torch.float32,
            device=case.v.device,
        )
        / case.v.shape[0]
    )
    case.v.copy_(marker[:, None, None].expand_as(case.v).to(case.v.dtype))

    wrapper = BatchPrefillTSWrapper()
    _plan_wrapper(wrapper, case)
    graph_out = torch.full_like(case.q, float("nan"), dtype=case.output_dtype)
    graph = _capture_context_graph(
        wrapper,
        case.q,
        case.k,
        case.v,
        graph_out,
        qo_indptr=case.qo_indptr,
        kv_indptr=case.kv_indptr,
    )

    assert case.kv_indptr is not None
    case.kv_indptr.copy_(
        torch.tensor(_cumulative(replay_k_lengths), dtype=torch.int32, device="cuda")
    )
    runtime_reference = replace(case, k_lengths=replay_k_lengths)
    expected = _context_reference(runtime_reference)

    direct_out = torch.full_like(case.q, float("nan"), dtype=case.output_dtype)
    _run_wrapper(wrapper, case, out=direct_out)
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    _assert_context_correct(graph_out, runtime_reference, expected=expected)
    _assert_context_correct(direct_out, runtime_reference, expected=expected)


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_paged_window_live_q_redistribution_graph_replay():
    """Finite-window start and right masks follow live packed Q offsets."""

    plan_q_lengths = (48, 16)
    replay_q_lengths = (16, 48)
    case = _make_paged_context_case(
        q_lengths=plan_q_lengths,
        k_lengths=(65, 65),
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=128,
        qkv_dtype=torch.bfloat16,
        mask_type="causal",
        window_left=31,
        output_scale=1.0,
        seed=2026071938,
    )
    wrapper = BatchPrefillPagedTSWrapper()
    metadata = _plan_paged_wrapper(wrapper, case)
    graph_out = torch.full_like(
        case.reference.q, float("nan"), dtype=case.reference.output_dtype
    )
    graph = _capture_context_graph(
        wrapper,
        case.reference.q,
        case.k_cache,
        case.v_cache,
        graph_out,
        metadata,
    )

    case.qo_indptr.copy_(
        torch.tensor(_cumulative(replay_q_lengths), dtype=torch.int32, device="cuda")
    )
    runtime_reference = replace(case.reference, q_lengths=replay_q_lengths)
    expected = _context_reference(runtime_reference)

    direct_out = torch.full_like(graph_out, float("nan"))
    _run_paged_wrapper(wrapper, case, metadata, out=direct_out)
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    _assert_context_correct(graph_out, runtime_reference, expected=expected)
    _assert_context_correct(direct_out, runtime_reference, expected=expected)


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_paged_window_graph_replay_writes_fresh_output():
    case = _make_paged_context_case(
        q_lengths=(33,),
        k_lengths=(65,),
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=128,
        qkv_dtype=_FP8,
        mask_type="causal",
        window_left=31,
        seed=2026071521,
    )
    wrapper = BatchPrefillPagedTSWrapper()
    metadata = _plan_paged_wrapper(wrapper, case)
    assert wrapper._plan_state is not None
    assert dict(wrapper._plan_state.policy)["scheduler"] == "clc_dynamic_persistent"
    output = torch.full_like(case.reference.q, float("nan"))
    assert _run_paged_wrapper(wrapper, case, metadata, out=output) is output
    _assert_context_correct(output, case.reference)

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_paged_wrapper(
            wrapper, case, metadata, out=output, validate=False
        )
    assert captured is output
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_context_correct(output, case.reference)


@pytest.mark.parametrize("paged", (False, True), ids=("contiguous", "paged"))
@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_plan_ready_event_orders_cross_stream_run(
    monkeypatch,
    paged: bool,
) -> None:
    """A run stream consumes plan state without explicitly waiting on its stream."""

    if paged:
        paged_case = _make_paged_context_case(
            q_lengths=(33, 17),
            k_lengths=(129, 97),
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=128,
            qkv_dtype=torch.bfloat16,
            mask_type="causal",
            page_size=32,
            output_dtype=torch.bfloat16,
            output_scale=1.0,
            seed=2026090402,
        )
        reference = paged_case.reference
        metadata = _make_native_paged_metadata(paged_case)
        wrapper_type = BatchPrefillPagedTSWrapper
        compile_name = "_get_compiled_paged_context"
        plan_kwargs = dict(
            device=reference.q.device,
            batch_size=len(reference.q_lengths),
            max_seq_len_q=max(reference.q_lengths),
            max_kv_len=max(reference.k_lengths),
            num_qo_heads=int(reference.q.shape[1]),
            num_kv_heads=int(paged_case.k_cache.shape[1]),
            head_dim=int(reference.q.shape[2]),
            q_dtype=reference.q.dtype,
            kv_dtype=paged_case.k_cache.dtype,
            out_dtype=reference.output_dtype,
            page_size=paged_case.page_size,
            mask_type=reference.mask_type,
            window_left=reference.window_left,
            sm_scale=reference.sm_scale,
            output_scale=reference.output_scale,
        )
        run_args = (
            reference.q,
            paged_case.k_cache,
            paged_case.v_cache,
            metadata.qo_indptr,
            metadata.block_tables,
            metadata.seq_lens_kv,
        )
    else:
        reference = _make_context_case(
            q_lengths=(65,),
            k_lengths=(65,),
            num_qo_heads=8,
            num_kv_heads=2,
            qkv_dtype=torch.bfloat16,
            packed=False,
            mask_type="causal",
            output_dtype=torch.bfloat16,
            output_scale=1.0,
            device="cuda",
            seed=2026090403,
        )
        wrapper_type = BatchPrefillTSWrapper
        compile_name = "_get_compiled_context"
        plan_kwargs = dict(
            device=reference.q.device,
            batch_size=len(reference.q_lengths),
            max_seq_len_q=max(reference.q_lengths),
            max_kv_len=max(reference.k_lengths),
            num_qo_heads=int(reference.q.shape[-2]),
            num_kv_heads=int(reference.k.shape[-2]),
            head_dim=int(reference.q.shape[-1]),
            q_dtype=reference.q.dtype,
            kv_dtype=reference.k.dtype,
            out_dtype=reference.output_dtype,
            packed=False,
            mask_type=reference.mask_type,
            window_left=reference.window_left,
            sm_scale=reference.sm_scale,
            output_scale=reference.output_scale,
        )
        run_args = (
            reference.q,
            reference.k,
            reference.v,
            reference.qo_indptr,
            reference.kv_indptr,
        )

    # Compile the exact specialization before introducing the plan-stream
    # delay, so the delayed plan performs no JIT work that could synchronize.
    warmup_wrapper = wrapper_type()
    warmup_wrapper.plan(**plan_kwargs)
    output = torch.full_like(reference.q, float("nan"), dtype=reference.output_dtype)
    torch.cuda.synchronize()

    get_compiled = getattr(context_module, compile_name)

    def get_compiled_after_plan_stream_delay(*args):
        compiled = get_compiled(*args)
        torch.cuda._sleep(1_000_000_000)
        return compiled

    monkeypatch.setattr(
        context_module, compile_name, get_compiled_after_plan_stream_delay
    )
    plan_stream = torch.cuda.Stream()
    run_stream = torch.cuda.Stream()
    wrapper = wrapper_type()

    with torch.cuda.stream(plan_stream):
        wrapper.plan(**plan_kwargs)
    state = wrapper._plan_state
    assert state is not None
    assert state.ready_stream_handle == plan_stream.cuda_stream
    assert state.ready_stream_handle != run_stream.cuda_stream
    assert not state.ready_event.query()

    complete = torch.cuda.Event()
    with torch.cuda.stream(run_stream):
        returned = wrapper.run(*run_args, out=output, validate=False)
        complete.record()

    assert returned is output
    complete.synchronize()
    # No run_stream.wait_stream(plan_stream) was issued above. Completion of
    # the run therefore proves that the wrapper consumed its plan-ready event.
    assert state.ready_event.query()
    _assert_context_correct(output, reference)


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_supplied_out_stream_and_cuda_graph():
    first = _make_context_case(
        q_lengths=(65,),
        k_lengths=(65,),
        num_qo_heads=8,
        num_kv_heads=2,
        qkv_dtype=torch.bfloat16,
        packed=False,
        mask_type="causal",
        output_dtype=torch.bfloat16,
        device="cuda",
        seed=2026071420,
    )
    second = _make_context_case(
        q_lengths=first.q_lengths,
        k_lengths=first.k_lengths,
        num_qo_heads=8,
        num_kv_heads=2,
        qkv_dtype=torch.bfloat16,
        packed=False,
        mask_type="causal",
        output_dtype=torch.bfloat16,
        device="cuda",
        seed=2026071421,
    )

    wrapper = BatchPrefillTSWrapper()
    _plan_wrapper(wrapper, first)
    with pytest.raises(ValueError, match="out must not overlap q storage"):
        _run_wrapper(wrapper, first, out=first.q)

    shared_out = torch.full_like(first.q, float("nan"), dtype=first.output_dtype)
    returned = _run_wrapper(wrapper, first, out=shared_out)
    assert returned is shared_out
    _assert_context_correct(shared_out, first)
    first_result = shared_out.clone()

    worker_stream = torch.cuda.Stream()
    complete = torch.cuda.Event()
    worker_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(worker_stream):
        shared_out.fill_(float("nan"))
        returned = _run_wrapper(wrapper, second, out=shared_out)
        assert returned is shared_out
        complete.record()
    torch.cuda.current_stream().wait_event(complete)
    _assert_context_correct(shared_out, second)
    assert not torch.equal(shared_out, first_result)

    graph_out = torch.full_like(second.q, float("nan"), dtype=second.output_dtype)
    _run_wrapper(wrapper, second, out=graph_out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_wrapper(wrapper, second, out=graph_out, validate=False)
    assert captured is graph_out
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_context_correct(graph_out, second)
