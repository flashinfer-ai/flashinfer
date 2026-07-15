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

"""Validate and benchmark the 48-case PrimTS FP8 context matrix.

The matrix is deliberately defined in one place and validated before any CUDA
work starts::

    D       = 128, 256
    Hq      = 32
    Hkv     = 32, 4
    batch   = 1, 4
    max Sq  = max Skv = 1024, 4096, 16384
    layout  = separate_qkv (packed ragged), paged_kv (page size 32)

That Cartesian product contains exactly 48 cases.  ``dtype`` is FP8 E4M3 for
Q, K, V, output, and the attention mode is dense because masking was not an
axis of the requested matrix.  Every one of the ``B`` requests has exactly
``Sq=Skv=S``.  ``separate_qkv (ragged)`` describes the packed THD ABI with
explicit Q and KV cumulative offsets; it does not perturb the timed shapes.

Performance is steady-state CUDA-graph replay.  Planning/compilation,
correctness, graph capture, and warmup all happen outside timed regions.  Each
case is accuracy checked with deterministic exact FP32 attention samples over
the complete logical KV sequence; this avoids an O(S^2) reference allocation
at 16K while still checking every matrix row.  Paged cases use a shuffled,
nonidentity physical page mapping and derive the reference through that same
mapping.

Examples
--------
List the matrix without importing a PrimTS kernel or requiring a GPU::

    python benchmarks/bench_attention_ts_context.py --list

Run the complete signoff matrix and save an incrementally updated report::

    python benchmarks/bench_attention_ts_context.py \
        --output context_ts_48.json

Run a development subset::

    python benchmarks/bench_attention_ts_context.py \
        --output context_ts_d128.json --head-dim 128 --layout separate_qkv
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fnmatch
import gc
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import traceback
from typing import Callable, Optional, Sequence

import torch


_FP8 = torch.float8_e4m3fn
_PAGE_SIZE = 32
_NUM_QO_HEADS = 32
_INPUT_STD = 1.0
_ACCURACY_SAMPLES = 8
_ACCURACY_RTOL = 5e-2
_ACCURACY_ATOL = 1.3e-1
_ACCURACY_RELATIVE_L2 = 1e-1
_FINITE_CHUNK_ELEMENTS = 8 * 1024 * 1024
_SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))


@dataclass(frozen=True)
class ContextPerfCase:
    """One member of the exact context performance matrix."""

    matrix_index: int
    dtype: str
    head_dim: int
    num_heads_q: int
    num_heads_kv: int
    batch_size: int
    max_seq_len_q: int
    max_seq_len_kv: int
    qkv_layout: str
    page_size: Optional[int]
    mask_type: str

    @property
    def case_id(self) -> str:
        page = f"_p{self.page_size}" if self.page_size is not None else ""
        return (
            f"ctx_fp8_{self.mask_type}_{self.qkv_layout}{page}"
            f"_b{self.batch_size}_s{self.max_seq_len_q}"
            f"_hq{self.num_heads_q}_hkv{self.num_heads_kv}"
            f"_d{self.head_dim}"
        )

    def to_json(self) -> dict[str, object]:
        result = asdict(self)
        result["case_id"] = self.case_id
        return result


def context_perf_cases() -> tuple[ContextPerfCase, ...]:
    """Return and structurally validate the exact 48-case matrix."""

    cases = []
    axes = itertools.product(
        (128, 256),
        (32, 4),
        (1, 4),
        (1024, 4096, 16384),
        ("separate_qkv", "paged_kv"),
    )
    for index, (head_dim, num_heads_kv, batch_size, seq_len, layout) in enumerate(axes):
        cases.append(
            ContextPerfCase(
                matrix_index=index,
                dtype="float8_e4m3fn",
                head_dim=head_dim,
                num_heads_q=_NUM_QO_HEADS,
                num_heads_kv=num_heads_kv,
                batch_size=batch_size,
                max_seq_len_q=seq_len,
                max_seq_len_kv=seq_len,
                qkv_layout=layout,
                page_size=_PAGE_SIZE if layout == "paged_kv" else None,
                mask_type="dense",
            )
        )

    expected_keys = set(
        itertools.product(
            (128, 256),
            (32, 4),
            (1, 4),
            (1024, 4096, 16384),
            ("separate_qkv", "paged_kv"),
        )
    )
    observed_keys = {
        (
            case.head_dim,
            case.num_heads_kv,
            case.batch_size,
            case.max_seq_len_q,
            case.qkv_layout,
        )
        for case in cases
    }
    case_ids = {case.case_id for case in cases}
    if len(cases) != 48:
        raise AssertionError(f"context matrix must contain 48 cases, got {len(cases)}")
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise AssertionError(
            f"context matrix mismatch: missing={missing}, extra={extra}"
        )
    if len(case_ids) != len(cases):
        raise AssertionError("context matrix case IDs must be unique")
    return tuple(cases)


def _sequence_lengths(case: ContextPerfCase) -> tuple[int, ...]:
    """Return the exact timed shape: all B requests have Sq=Skv=S."""

    if case.batch_size not in (1, 4):
        raise AssertionError(f"unexpected matrix batch size {case.batch_size}")
    return (case.max_seq_len_q,) * case.batch_size


def _case_seed(case: ContextPerfCase, base_seed: int) -> int:
    """Use identical logical Q/K/V values for each separate/paged pair."""

    if case.matrix_index < 0 or case.matrix_index >= 48:
        raise ValueError(f"matrix index is out of range: {case.matrix_index}")
    return base_seed + case.matrix_index // 2


def _cumulative(values: Sequence[int]) -> tuple[int, ...]:
    offsets = [0]
    for value in values:
        offsets.append(offsets[-1] + int(value))
    return tuple(offsets)


def _permuted_page_ids(
    num_used_pages: int, num_physical_pages: int, seed: int
) -> tuple[int, ...]:
    """Create deterministic unique physical IDs, never an identity page table."""

    if num_used_pages <= 1 or num_physical_pages < num_used_pages:
        raise ValueError("paged context cases require at least two usable pages")
    physical_ids = list(range(num_physical_pages))
    random.Random(seed).shuffle(physical_ids)
    selected = physical_ids[:num_used_pages]
    if selected == list(range(num_used_pages)):
        selected = selected[1:] + selected[:1]
    if len(set(selected)) != num_used_pages:
        raise AssertionError("physical page IDs must be unique")
    if selected == list(range(num_used_pages)):
        raise AssertionError("physical page IDs must be nonidentity")
    return tuple(selected)


@dataclass
class _ContextInputs:
    case: ContextPerfCase
    lengths: tuple[int, ...]
    q_offsets_host: tuple[int, ...]
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    qo_indptr: torch.Tensor
    kv_indptr: Optional[torch.Tensor]
    paged_kv_indptr: Optional[torch.Tensor]
    paged_kv_indices: Optional[torch.Tensor]
    paged_kv_last_page_len: Optional[torch.Tensor]
    paged_kv_indptr_host: Optional[tuple[int, ...]]
    paged_kv_indices_host: Optional[tuple[int, ...]]


def _fp8_randn(
    shape: tuple[int, ...], *, generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    # Generate in FP16 because torch.randn does not directly implement FP8.
    source = torch.randn(shape, dtype=torch.float16, generator=generator, device=device)
    source.mul_(_INPUT_STD)
    return source.to(_FP8)


@torch.inference_mode()
def _make_inputs(
    case: ContextPerfCase, *, device: torch.device, seed: int
) -> _ContextInputs:
    lengths = _sequence_lengths(case)
    if case.max_seq_len_q != case.max_seq_len_kv:
        raise AssertionError("this matrix requires equal maximum Q and KV lengths")
    q_offsets_host = _cumulative(lengths)
    qo_indptr = torch.tensor(q_offsets_host, dtype=torch.int32, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    q = _fp8_randn(
        (q_offsets_host[-1], case.num_heads_q, case.head_dim),
        generator=generator,
        device=device,
    )
    out = torch.empty_like(q)

    if case.qkv_layout == "separate_qkv":
        kv_indptr = torch.tensor(q_offsets_host, dtype=torch.int32, device=device)
        kv_shape = (q_offsets_host[-1], case.num_heads_kv, case.head_dim)
        k = _fp8_randn(kv_shape, generator=generator, device=device)
        v = _fp8_randn(kv_shape, generator=generator, device=device)
        return _ContextInputs(
            case=case,
            lengths=lengths,
            q_offsets_host=q_offsets_host,
            q=q,
            k=k,
            v=v,
            out=out,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            paged_kv_indptr=None,
            paged_kv_indices=None,
            paged_kv_last_page_len=None,
            paged_kv_indptr_host=None,
            paged_kv_indices_host=None,
        )

    if case.qkv_layout != "paged_kv" or case.page_size != _PAGE_SIZE:
        raise AssertionError(f"unexpected paged case {case}")
    page_counts = tuple(math.ceil(length / _PAGE_SIZE) for length in lengths)
    page_indptr_host = _cumulative(page_counts)
    num_used_pages = page_indptr_host[-1]
    # Unreferenced guard pages ensure a valid page table cannot accidentally be
    # replaced by an implicit contiguous [0, N) interpretation.
    num_physical_pages = num_used_pages + max(2, case.batch_size)
    page_indices_host = _permuted_page_ids(
        num_used_pages, num_physical_pages, seed + 17041
    )
    page_shape = (
        num_physical_pages,
        case.num_heads_kv,
        _PAGE_SIZE,
        case.head_dim,
    )
    logical_shape = (q_offsets_host[-1], case.num_heads_kv, case.head_dim)
    k_logical = _fp8_randn(logical_shape, generator=generator, device=device)
    v_logical = _fp8_randn(logical_shape, generator=generator, device=device)
    if q_offsets_host[-1] != num_used_pages * _PAGE_SIZE:
        raise AssertionError("the exact performance matrix must use full KV pages")
    # Poison unreferenced guard pages so an implicit or out-of-range page-table
    # access becomes an accuracy failure instead of reading plausible data.
    # CUDA does not implement index_copy_ for FP8. Scatter the already-FP8-
    # quantized logical values through FP16 staging, then convert the complete
    # physical cache back to FP8. FP8 -> FP16 -> FP8 is value preserving.
    k_staging = torch.full(page_shape, float("nan"), dtype=torch.float16, device=device)
    v_staging = torch.full_like(k_staging, float("nan"))
    physical_ids = torch.tensor(page_indices_host, dtype=torch.int64, device=device)
    logical_page_shape = (
        num_used_pages,
        _PAGE_SIZE,
        case.num_heads_kv,
        case.head_dim,
    )
    k_staging.index_copy_(
        0,
        physical_ids,
        k_logical.reshape(logical_page_shape).permute(0, 2, 1, 3).to(torch.float16),
    )
    v_staging.index_copy_(
        0,
        physical_ids,
        v_logical.reshape(logical_page_shape).permute(0, 2, 1, 3).to(torch.float16),
    )
    k = k_staging.to(_FP8)
    v = v_staging.to(_FP8)
    del k_logical, v_logical, k_staging, v_staging, physical_ids
    paged_kv_indptr = torch.tensor(page_indptr_host, dtype=torch.int32, device=device)
    paged_kv_indices = torch.tensor(page_indices_host, dtype=torch.int32, device=device)
    last_page_len = tuple((length - 1) % _PAGE_SIZE + 1 for length in lengths)
    paged_kv_last_page_len = torch.tensor(
        last_page_len, dtype=torch.int32, device=device
    )
    if torch.equal(
        paged_kv_indices,
        torch.arange(num_used_pages, dtype=torch.int32, device=device),
    ):
        raise AssertionError("paged test data must use nonidentity physical indices")
    return _ContextInputs(
        case=case,
        lengths=lengths,
        q_offsets_host=q_offsets_host,
        q=q,
        k=k,
        v=v,
        out=out,
        qo_indptr=qo_indptr,
        kv_indptr=None,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
        paged_kv_indptr_host=page_indptr_host,
        paged_kv_indices_host=page_indices_host,
    )


class _PlannedRunner:
    """Own a planned wrapper and its allocation-free launch closure."""

    def __init__(self, wrapper: object, launch: Callable[[], torch.Tensor]) -> None:
        self.wrapper = wrapper
        self.launch = launch


def _plan_separate_runner(inputs: _ContextInputs) -> _PlannedRunner:
    from flashinfer.attention.prims_ts import BatchPrefillTSWrapper

    if inputs.kv_indptr is None:
        raise AssertionError("separate_qkv requires kv_indptr")
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(
        inputs.q,
        inputs.k,
        inputs.v,
        qo_indptr=inputs.qo_indptr,
        kv_indptr=inputs.kv_indptr,
        mask_type="dense",
        out_dtype=_FP8,
    )

    def launch() -> torch.Tensor:
        return wrapper.run(inputs.q, inputs.k, inputs.v, out=inputs.out)

    return _PlannedRunner(wrapper, launch)


def _plan_paged_runner(inputs: _ContextInputs) -> _PlannedRunner:
    """Adapt the benchmark to the dedicated PrimTS paged-context wrapper.

    Keep this adapter narrow: if the experimental API changes, only this
    function should need an update.  It intentionally does not fall back to a
    different FlashInfer backend because that would silently benchmark the
    wrong kernel.
    """

    import flashinfer.attention.prims_ts as prims_ts

    try:
        wrapper_cls = prims_ts.BatchPrefillPagedTSWrapper
    except AttributeError as error:
        raise RuntimeError(
            "paged_kv matrix rows require "
            "flashinfer.attention.prims_ts.BatchPrefillPagedTSWrapper; "
            "the paged context migration has not exposed that API yet"
        ) from error
    if (
        inputs.paged_kv_indptr is None
        or inputs.paged_kv_indices is None
        or inputs.paged_kv_last_page_len is None
    ):
        raise AssertionError("paged_kv metadata is incomplete")

    wrapper = wrapper_cls()
    try:
        wrapper.plan(
            inputs.q,
            inputs.k,
            inputs.v,
            qo_indptr=inputs.qo_indptr,
            paged_kv_indptr=inputs.paged_kv_indptr,
            paged_kv_indices=inputs.paged_kv_indices,
            paged_kv_last_page_len=inputs.paged_kv_last_page_len,
            page_size=_PAGE_SIZE,
            mask_type="dense",
            out_dtype=_FP8,
        )
    except TypeError as error:
        raise TypeError(
            "BatchPrefillPagedTSWrapper.plan is expected to accept "
            "plan(q, k_pages, v_pages, qo_indptr=..., "
            "paged_kv_indptr=..., paged_kv_indices=..., "
            "paged_kv_last_page_len=..., page_size=32, mask_type='dense', "
            "out_dtype=torch.float8_e4m3fn)"
        ) from error

    def launch() -> torch.Tensor:
        return wrapper.run(inputs.q, inputs.k, inputs.v, out=inputs.out)

    return _PlannedRunner(wrapper, launch)


def _plan_runner(inputs: _ContextInputs) -> _PlannedRunner:
    if inputs.case.qkv_layout == "separate_qkv":
        return _plan_separate_runner(inputs)
    return _plan_paged_runner(inputs)


def _logical_kv_for_request(
    inputs: _ContextInputs, batch_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return logical [Skv, Hkv, D] K/V, following the physical page table."""

    length = inputs.lengths[batch_idx]
    if inputs.case.qkv_layout == "separate_qkv":
        begin = inputs.q_offsets_host[batch_idx]
        return inputs.k[begin : begin + length], inputs.v[begin : begin + length]

    if inputs.paged_kv_indptr_host is None or inputs.paged_kv_indices is None:
        raise AssertionError("paged reference metadata is incomplete")
    page_begin = inputs.paged_kv_indptr_host[batch_idx]
    page_end = inputs.paged_kv_indptr_host[batch_idx + 1]
    physical_ids = inputs.paged_kv_indices[page_begin:page_end].to(torch.int64)
    k_pages = inputs.k.index_select(0, physical_ids)
    v_pages = inputs.v.index_select(0, physical_ids)
    k_logical = k_pages.permute(0, 2, 1, 3).reshape(
        -1, inputs.case.num_heads_kv, inputs.case.head_dim
    )
    v_logical = v_pages.permute(0, 2, 1, 3).reshape(
        -1, inputs.case.num_heads_kv, inputs.case.head_dim
    )
    return k_logical[:length], v_logical[:length]


def _accuracy_sample_points(inputs: _ContextInputs) -> tuple[tuple[int, int, int], ...]:
    """Choose deterministic samples covering requests, query rows, and heads."""

    last_batch = len(inputs.lengths) - 1
    lower_middle_batch = len(inputs.lengths) // 4
    upper_middle_batch = len(inputs.lengths) // 2
    last_head = inputs.case.num_heads_q - 1
    quarter_head = inputs.case.num_heads_q // 4
    middle_head = inputs.case.num_heads_q // 2
    templates = (
        (0, 0.0, 0),
        (last_batch, 1.0, last_head),
        (lower_middle_batch, 0.5, middle_head),
        (upper_middle_batch, 1.0, 0),
        (last_batch, 0.0, middle_head),
        (lower_middle_batch, 1.0 / 3.0, quarter_head),
        (0, 2.0 / 3.0, last_head),
        (upper_middle_batch, 0.5, 0),
    )
    points = []
    for batch_idx, fraction, query_head in templates:
        length = inputs.lengths[batch_idx]
        query_idx = int(round(fraction * (length - 1)))
        point = (batch_idx, query_idx, query_head)
        if point in points:
            raise AssertionError(f"accuracy sample templates collide at {point}")
        points.append(point)
    if len(points) != _ACCURACY_SAMPLES:
        raise AssertionError("accuracy sample count does not match the protocol")
    return tuple(points)


@torch.inference_mode()
def _check_accuracy(inputs: _ContextInputs) -> dict[str, object]:
    """Check full-output finiteness and exact sampled FP32 dense attention."""

    flat_out = inputs.out.reshape(-1)
    finite_chunks = 0
    for begin in range(0, flat_out.numel(), _FINITE_CHUNK_ELEMENTS):
        chunk = flat_out[begin : begin + _FINITE_CHUNK_ELEMENTS]
        if not bool(torch.isfinite(chunk.float()).all().item()):
            raise AssertionError(
                f"{inputs.case.case_id}: output contains nonfinite values"
            )
        finite_chunks += 1

    points = _accuracy_sample_points(inputs)
    points_by_batch: dict[int, list[tuple[int, int]]] = {}
    for batch_idx, query_idx, query_head in points:
        points_by_batch.setdefault(batch_idx, []).append((query_idx, query_head))

    sample_results = []
    total_error_sq = 0.0
    total_expected_sq = 0.0
    max_abs_error = 0.0
    sm_scale = 1.0 / math.sqrt(inputs.case.head_dim)
    head_ratio = inputs.case.num_heads_q // inputs.case.num_heads_kv
    for batch_idx, batch_points in points_by_batch.items():
        k_logical, v_logical = _logical_kv_for_request(inputs, batch_idx)
        q_begin = inputs.q_offsets_host[batch_idx]
        for query_idx, query_head in batch_points:
            kv_head = query_head // head_ratio
            q_vector = inputs.q[q_begin + query_idx, query_head].float()
            k_matrix = k_logical[:, kv_head].float()
            v_matrix = v_logical[:, kv_head].float()
            scores = torch.mv(k_matrix, q_vector) * sm_scale
            probabilities = torch.softmax(scores, dim=0)
            expected = torch.matmul(probabilities, v_matrix)
            actual = inputs.out[q_begin + query_idx, query_head].float()
            difference = actual - expected
            sample_max_abs = float(difference.abs().max().item())
            max_abs_error = max(max_abs_error, sample_max_abs)
            allowed = _ACCURACY_ATOL + _ACCURACY_RTOL * float(
                expected.abs().max().item()
            )
            if sample_max_abs > allowed:
                raise AssertionError(
                    f"{inputs.case.case_id}: sampled FP32 reference mismatch at "
                    f"batch={batch_idx}, query={query_idx}, head={query_head}: "
                    f"max_abs={sample_max_abs:.6g} > allowed={allowed:.6g}"
                )
            total_error_sq += float(torch.sum(difference * difference).item())
            total_expected_sq += float(torch.sum(expected * expected).item())
            sample_results.append(
                {
                    "batch": batch_idx,
                    "query": query_idx,
                    "query_head": query_head,
                    "kv_head": kv_head,
                    "key_range": [0, int(k_logical.shape[0])],
                    "max_abs_error": sample_max_abs,
                    "max_allowed_error": allowed,
                }
            )
        del k_logical, v_logical

    relative_l2 = math.sqrt(total_error_sq) / max(math.sqrt(total_expected_sq), 1e-6)
    if relative_l2 > _ACCURACY_RELATIVE_L2:
        raise AssertionError(
            f"{inputs.case.case_id}: sampled relative L2 {relative_l2:.6g} "
            f"> {_ACCURACY_RELATIVE_L2:.6g}"
        )
    return {
        "method": "sampled_exact_fp32_full_kv",
        "sample_count": len(sample_results),
        "samples": sample_results,
        "full_output_finite_chunks": finite_chunks,
        "finite_chunk_elements": _FINITE_CHUNK_ELEMENTS,
        "rtol": _ACCURACY_RTOL,
        "atol": _ACCURACY_ATOL,
        "relative_l2": relative_l2,
        "relative_l2_limit": _ACCURACY_RELATIVE_L2,
        "max_abs_error": max_abs_error,
    }


@torch.inference_mode()
def _benchmark_cuda_graph(
    runner: _PlannedRunner,
    *,
    output: torch.Tensor,
    device: torch.device,
    rounds: int,
    iterations: int,
    warmup: int,
) -> dict[str, object]:
    """Time only steady-state graph replay using CUDA events."""

    if rounds <= 0 or iterations <= 0 or warmup < 0:
        raise ValueError("rounds/iterations must be positive and warmup nonnegative")
    with torch.cuda.device(device):
        current_stream = torch.cuda.current_stream(device)
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(current_stream)
        # Warm the compiled launch and all lazy runtime state on a side stream.
        with torch.cuda.stream(capture_stream):
            for _ in range(max(3, warmup)):
                runner.launch()
        capture_stream.synchronize()
        current_stream.wait_stream(capture_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            runner.launch()
        torch.cuda.synchronize(device)
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize(device)

        round_ms = []
        for _ in range(rounds):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(current_stream)
            for _ in range(iterations):
                graph.replay()
            end.record(current_stream)
            end.synchronize()
            round_ms.append(float(start.elapsed_time(end)) / iterations)

        # Prove replay writes fresh output: a stale/no-op graph cannot inherit
        # the valid tensor produced before capture or during timed replays.
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)

    return {
        "timing": "CUDA event around steady-state CUDA-graph replay",
        "compile_plan_timed": False,
        "graph_capture_timed": False,
        "warmup_timed": False,
        "post_timing_output_poisoned_and_replayed": True,
        "cache_state": "natural steady-state (no explicit L2 flush)",
        "rounds": rounds,
        "iterations_per_round": iterations,
        "warmup_replays": warmup,
        "round_ms": round_ms,
        "min_ms": min(round_ms),
        "max_ms": max(round_ms),
        "mean_ms": statistics.fmean(round_ms),
        "median_ms": statistics.median(round_ms),
        "stdev_ms": statistics.stdev(round_ms) if len(round_ms) > 1 else 0.0,
    }


def _dense_flops(case: ContextPerfCase, lengths: Sequence[int]) -> int:
    # QK and PV each contribute one multiply and one add.
    return int(
        4
        * sum(length * length for length in lengths)
        * case.num_heads_q
        * case.head_dim
    )


def _git_value(repo: Path, *args: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            ("git", *args), cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cutlass_environment() -> dict[str, object]:
    """Record enough package provenance to reproduce the selected wheel."""

    result: dict[str, object] = {
        "cuda_home": os.environ.get("CUDA_HOME"),
        "cute_dsl_libs": os.environ.get("CUTE_DSL_LIBS"),
        "wheel_validation_target": os.environ.get("FLASHINFER_WHEEL_VALIDATION_TARGET"),
        "wheel_validation_cutlass_python": os.environ.get(
            "FLASHINFER_WHEEL_VALIDATION_CUTLASS_PY"
        ),
        "wheel_validation_runtime": os.environ.get(
            "FLASHINFER_WHEEL_VALIDATION_RUNTIME"
        ),
    }
    try:
        import cutlass

        result["cutlass_module_file"] = str(Path(cutlass.__file__).resolve())
    except Exception as error:  # pragma: no cover - diagnostic only
        result["cutlass_import_error"] = repr(error)
    try:
        distribution = importlib.metadata.distribution("nvidia-cutlass-dsl")
        result["distribution_version"] = distribution.version
        result["distribution_root"] = str(Path(distribution.locate_file("")).resolve())
    except importlib.metadata.PackageNotFoundError as error:  # pragma: no cover
        result["distribution_error"] = repr(error)

    target_value = result["wheel_validation_target"]
    if isinstance(target_value, str):
        wheelhouse = Path(target_value).resolve().parent / "wheelhouse"
        wheel_candidates = sorted(wheelhouse.glob("nvidia_cutlass_dsl-*.whl"))
        result["wheel_candidates"] = [str(path) for path in wheel_candidates]
        if len(wheel_candidates) == 1:
            wheel = wheel_candidates[0]
            result["wheel_artifact"] = str(wheel)
            result["wheel_sha256"] = _sha256_file(wheel)
    return result


def _source_provenance(repo: Path) -> dict[str, object]:
    relative_paths = (
        "benchmarks/bench_attention_ts_context.py",
        "flashinfer/attention/prims_ts/__init__.py",
        "flashinfer/attention/prims_ts/context.py",
        "flashinfer/attention/prims_ts/kernels/fmha_context/fmha_kernel.py",
        "flashinfer/attention/prims_ts/kernels/fmha_context/fmha_resources.py",
        "flashinfer/attention/prims_ts/kernels/fmha_context/fmha_tasks.py",
        "flashinfer/attention/prims_ts/kernels/fmha_context/helpers_paged.py",
    )
    return {
        "argv": list(sys.argv),
        "files_sha256": {
            relative_path: _sha256_file(repo / relative_path)
            for relative_path in relative_paths
        },
    }


def _environment(repo: Path, device: Optional[torch.device]) -> dict[str, object]:
    result: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "git_commit": _git_value(repo, "rev-parse", "HEAD"),
        "git_branch": _git_value(repo, "branch", "--show-current"),
        "git_status_porcelain": _git_value(repo, "status", "--porcelain"),
        "cutlass_dsl": _cutlass_environment(),
        "source": _source_provenance(repo),
    }
    try:
        import flashinfer

        result["flashinfer"] = getattr(flashinfer, "__version__", None)
    except Exception as error:  # pragma: no cover - diagnostic only
        result["flashinfer_import_error"] = repr(error)
    if device is not None:
        properties = torch.cuda.get_device_properties(device)
        result["gpu"] = {
            "index": device.index,
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": properties.total_memory,
            "multi_processor_count": properties.multi_processor_count,
            "l2_cache_size_bytes": getattr(properties, "L2_cache_size", None),
        }
    return result


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _selected_cases(
    cases: Sequence[ContextPerfCase], args: argparse.Namespace
) -> tuple[ContextPerfCase, ...]:
    selected = []
    for case in cases:
        if not fnmatch.fnmatch(case.case_id, args.case_glob):
            continue
        if args.head_dim and case.head_dim not in args.head_dim:
            continue
        if args.num_kv_heads and case.num_heads_kv not in args.num_kv_heads:
            continue
        if args.batch_size and case.batch_size not in args.batch_size:
            continue
        if args.seq_len and case.max_seq_len_q not in args.seq_len:
            continue
        if args.layout and case.qkv_layout not in args.layout:
            continue
        selected.append(case)
    return tuple(selected)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="incrementally updated JSON report (required unless --list)",
    )
    parser.add_argument(
        "--list", action="store_true", help="print the exact matrix as JSON and exit"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--case-glob", default="*")
    parser.add_argument("--head-dim", type=int, action="append", choices=(128, 256))
    parser.add_argument("--num-kv-heads", type=int, action="append", choices=(4, 32))
    parser.add_argument("--batch-size", type=int, action="append", choices=(1, 4))
    parser.add_argument(
        "--seq-len", type=int, action="append", choices=(1024, 4096, 16384)
    )
    parser.add_argument(
        "--layout", action="append", choices=("separate_qkv", "paged_kv")
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="retain successful rows already present in --output",
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="stop after the first failed row"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    cases = context_perf_cases()
    selected = _selected_cases(cases, args)
    if args.list:
        print(
            json.dumps(
                {
                    "count": len(cases),
                    "selected_count": len(selected),
                    "cases": [case.to_json() for case in selected],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.output is None:
        raise SystemExit("--output is required unless --list is used")
    if not selected:
        raise SystemExit("filters selected no matrix cases")
    if not torch.cuda.is_available():
        raise SystemExit("PrimTS context benchmarking requires CUDA")

    device = torch.device("cuda", args.device)
    capability = torch.cuda.get_device_capability(device)
    if capability not in _SUPPORTED_CAPABILITIES:
        raise SystemExit(
            "PrimTS context benchmarking requires SM100 or SM103; "
            f"cuda:{args.device} has capability {capability}"
        )
    repo = Path(__file__).resolve().parents[1]
    protocol = {
        "matrix_expected_count": 48,
        "matrix_cardinality_formula": "2 head dims * 2 Hkv * 2 batches * 3 sequence lengths * 2 layouts = 48",
        "matrix_axes": {
            "head_dim": [128, 256],
            "num_heads_q": [32],
            "num_heads_kv": [32, 4],
            "batch_size": [1, 4],
            "seq_len_q_equals_seq_len_kv": [1024, 4096, 16384],
            "qkv_layout": ["separate_qkv", "paged_kv"],
            "dtype": ["float8_e4m3fn"],
        },
        "selected_count": len(selected),
        "dtype": "torch.float8_e4m3fn for Q/K/V/O",
        "mask_type": "dense",
        "q_storage": "packed [total_q, Hq, D] with qo_indptr",
        "separate_kv_storage": "packed [total_kv, Hkv, D] with kv_indptr",
        "paged_kv_storage": "separate HND K/V [pages, Hkv, 32, D]",
        "logical_lengths": "every one of B requests has Sq=Skv=S",
        "ragged_abi": "separate_qkv always uses packed THD plus Q/KV indptr",
        "paged_indices": "deterministic shuffled nonidentity unique physical IDs",
        "layout_pair_inputs": "same seed and identical logical Q/K/V per separate/paged shape pair",
        "paged_guard_pages": "unreferenced physical pages filled with NaN",
        "correctness": "8 exact FP32 samples over full logical KV plus full finiteness",
        "performance": "CUDA-graph steady-state; compile/capture/warmup excluded",
        "graph_correctness": "poison output after timing, replay once, then repeat the FP32 sample and finiteness checks",
        "rounds": args.rounds,
        "iterations_per_round": args.iterations,
        "warmup_replays": args.warmup,
        "seed": args.seed,
        "input_distribution": f"normal mean=0 std={_INPUT_STD}",
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "matrix": [case.to_json() for case in cases],
        "protocol": protocol,
        "environment": _environment(repo, device),
        "results": [],
        "summary": {"status": "running", "completed": 0, "failed": 0},
    }
    existing_by_id: dict[str, dict[str, object]] = {}
    if args.resume and args.output.exists():
        existing = json.loads(args.output.read_text())
        if existing.get("protocol") != protocol:
            raise SystemExit(
                "cannot resume: report protocol differs from this invocation"
            )
        existing_environment = existing.get("environment", {})
        current_environment = report["environment"]
        existing_identity = {
            "cutlass_dsl": existing_environment.get("cutlass_dsl"),
            "gpu": existing_environment.get("gpu"),
            "torch": existing_environment.get("torch"),
            "torch_cuda": existing_environment.get("torch_cuda"),
            "source_files_sha256": existing_environment.get("source", {}).get(
                "files_sha256"
            ),
        }
        current_identity = {
            "cutlass_dsl": current_environment.get("cutlass_dsl"),
            "gpu": current_environment.get("gpu"),
            "torch": current_environment.get("torch"),
            "torch_cuda": current_environment.get("torch_cuda"),
            "source_files_sha256": current_environment.get("source", {}).get(
                "files_sha256"
            ),
        }
        if existing_identity != current_identity:
            raise SystemExit(
                "cannot resume: source, wheel, runtime, or GPU provenance differs"
            )
        existing_by_id = {
            row["case_id"]: row
            for row in existing.get("results", [])
            if row.get("status") == "ok"
        }
        report["results"] = list(existing_by_id.values())

    results: list[dict[str, object]] = report["results"]  # type: ignore[assignment]
    failures = 0
    for case in selected:
        if case.case_id in existing_by_id:
            print(f"[resume] {case.case_id}", flush=True)
            continue
        print(f"[run] {case.case_id}", flush=True)
        inputs: Optional[_ContextInputs] = None
        runner: Optional[_PlannedRunner] = None
        row: dict[str, object] = {
            "case_id": case.case_id,
            "case": case.to_json(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            case_seed = _case_seed(case, args.seed)
            row["input_seed"] = case_seed
            inputs = _make_inputs(case, device=device, seed=case_seed)
            row["lengths_q"] = list(inputs.lengths)
            row["lengths_kv"] = list(inputs.lengths)
            if inputs.paged_kv_indices_host is not None:
                row["paged_kv_indices_nonidentity"] = (
                    inputs.paged_kv_indices_host
                    != tuple(range(len(inputs.paged_kv_indices_host)))
                )
                row["paged_kv_indices_unique"] = len(
                    set(inputs.paged_kv_indices_host)
                ) == len(inputs.paged_kv_indices_host)

            torch.cuda.synchronize(device)
            plan_begin = datetime.now(timezone.utc)
            runner = _plan_runner(inputs)
            torch.cuda.synchronize(device)
            row["plan_compile_seconds"] = (
                datetime.now(timezone.utc) - plan_begin
            ).total_seconds()

            runner.launch()
            torch.cuda.synchronize(device)
            row["accuracy"] = _check_accuracy(inputs)
            timing = _benchmark_cuda_graph(
                runner,
                output=inputs.out,
                device=device,
                rounds=args.rounds,
                iterations=args.iterations,
                warmup=args.warmup,
            )
            flops = _dense_flops(case, inputs.lengths)
            timing["dense_flops"] = flops
            timing["tflops"] = flops / (float(timing["median_ms"]) * 1e9)
            row["performance"] = timing
            row["graph_replay_accuracy"] = _check_accuracy(inputs)
            row["status"] = "ok"
            print(
                f"[ok] {case.case_id}: {timing['median_ms']:.6f} ms, "
                f"{timing['tflops']:.3f} TFLOP/s",
                flush=True,
            )
        except Exception as error:
            failures += 1
            row["status"] = "error"
            row["error"] = repr(error)
            row["traceback"] = traceback.format_exc()
            print(f"[error] {case.case_id}: {error}", file=sys.stderr, flush=True)
        finally:
            row["finished_at"] = datetime.now(timezone.utc).isoformat()
            results[:] = [
                result for result in results if result["case_id"] != case.case_id
            ]
            results.append(row)
            results.sort(
                key=lambda result: next(
                    case.matrix_index
                    for case in cases
                    if case.case_id == result["case_id"]
                )
            )
            completed = sum(result.get("status") == "ok" for result in results)
            failed = sum(result.get("status") == "error" for result in results)
            report["summary"] = {
                "status": "running",
                "selected": len(selected),
                "completed": completed,
                "failed": failed,
            }
            _write_report(args.output, report)
            del runner, inputs
            gc.collect()
            torch.cuda.empty_cache()
        if row["status"] == "error" and args.fail_fast:
            break

    selected_ids = {case.case_id for case in selected}
    selected_results = [row for row in results if row["case_id"] in selected_ids]
    completed = sum(row.get("status") == "ok" for row in selected_results)
    failed = sum(row.get("status") == "error" for row in selected_results)
    report["summary"] = {
        "status": "ok" if completed == len(selected) and failed == 0 else "failed",
        "matrix_count": len(cases),
        "selected": len(selected),
        "completed": completed,
        "failed": failed,
        "all_48_completed": len(selected) == 48 and completed == 48 and failed == 0,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_report(args.output, report)
    return 0 if report["summary"]["status"] == "ok" else 1  # type: ignore[index]


if __name__ == "__main__":
    raise SystemExit(main())
