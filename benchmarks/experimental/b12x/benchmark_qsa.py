#!/usr/bin/env python3
"""Benchmark prepared Qwen3.8 Flash Next QSA decode and prefill transactions.

The harness declares each static Caps case, prepares it through the shared
session, then binds caller-owned cache/state tensors to its prepared plan.
The timed operation is the bound ``qsa.run`` transaction: selector query
preparation, streaming representative compression, stable top-k selection,
position expansion, and sparse paged GQA. Each request owns disjoint main K/V
pages plus independent selector and recurrent state, as it does in serving.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from b12x.attention import qsa
from b12x.preparation import PreparedCall, PreparationResult, PreparationSession
from b12x.preparation.types import require_prepared
from b12x.attention.qsa.reference import (
    gemma_rmsnorm_reference,
    packed_stream_compress_reference,
    paged_store_compressed_reference,
    score_select_reference,
    sparse_paged_gqa_reference,
)
from benchmarks.experimental.b12x.common import (
    bench_cuda_graph,
    capture_cuda_graph,
    make_l2_flush_fn,
    require_sm120,
    resolve_l2_flush_bytes,
)


_RESULT_KIND = "b12x_qwen38_flash_next_qsa_benchmark"
_RESULT_SCHEMA_VERSION = 1

MODEL_MAX_CONTEXT = 262_144
MAIN_PAGE_SIZE = 16
COMPRESS_RATIO = 4
BUDGET = 2048
INDEX_HEADS = 4
INDEX_KV_HEADS = 1
INDEX_HEAD_DIM = 128
INDEX_ROTARY_DIM = 64
HEAD_DIM = 256
POSITION_AXES = 3
MROPE_SECTIONS = (11, 11, 10)
SELECTED_POSITION_POISON = torch.iinfo(torch.int32).min

DEFAULT_ROWS = (1, 4, 16, 64)
DEFAULT_CONTEXTS = (2048, 8192)
FULL_CONTEXTS = (2048, 8192, 32_768, 131_072, MODEL_MAX_CONTEXT)


class BenchmarkFailure(RuntimeError):
    """A correctness or benchmark-contract failure."""


@dataclass(frozen=True)
class ParallelProfile:
    name: str
    tensor_parallel_size: int
    q_heads: int
    kv_heads: int


PROFILES = {
    profile.name: profile
    for profile in (
        ParallelProfile("tp1", tensor_parallel_size=1, q_heads=24, kv_heads=2),
        ParallelProfile("tp2", tensor_parallel_size=2, q_heads=12, kv_heads=1),
        ParallelProfile("tp4", tensor_parallel_size=4, q_heads=6, kv_heads=1),
    )
}


@dataclass(frozen=True)
class BenchmarkCase:
    profile: ParallelProfile
    rows: int
    context: int
    kind: str = "throughput"
    tail_length: int = 0
    preceding_accepted_tokens: int = 1
    main_page_size: int = MAIN_PAGE_SIZE
    planned_max_batch: int | None = None
    planned_max_q_rows: int | None = None
    planned_max_speculative_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"throughput", "prefill", "stream_phase", "speculative"}:
            raise ValueError(f"unknown QSA benchmark case kind {self.kind!r}")
        if self.context % COMPRESS_RATIO:
            raise ValueError("QSA benchmark capacity must be compression aligned")
        if self.main_page_size <= 0 or self.main_page_size % COMPRESS_RATIO:
            raise ValueError(
                "QSA benchmark main page size must be positive and compression aligned"
            )
        if self.planned_batch < self.request_count:
            raise ValueError("planned QSA batch capacity must cover active requests")
        if self.planned_q_rows < self.rows:
            raise ValueError("planned QSA row capacity must cover active rows")
        if self.planned_speculative_tokens < self.max_speculative_tokens:
            raise ValueError(
                "planned QSA speculative capacity must cover the active transaction"
            )
        if self.kind == "stream_phase":
            if self.rows != 1 or self.tail_length not in range(COMPRESS_RATIO):
                raise ValueError("stream-phase cases require one row and tail 0..3")
        elif self.tail_length:
            raise ValueError("tail_length is only valid for stream-phase cases")
        if self.kind == "speculative":
            if self.rows != 4 or not 1 <= self.preceding_accepted_tokens <= 4:
                raise ValueError(
                    "speculative cases require four packed rows and acceptance 1..4"
                )
        elif self.preceding_accepted_tokens != 1:
            raise ValueError(
                "preceding_accepted_tokens is only configurable for speculative cases"
            )
        if self.kind == "prefill" and self.rows > self.context:
            raise ValueError("prefill rows cannot exceed the active context")

    @property
    def name(self) -> str:
        page_suffix = (
            "" if self.main_page_size == MAIN_PAGE_SIZE else f"-p{self.main_page_size}"
        )
        if self.kind == "stream_phase":
            return (
                f"{self.profile.name}-r1-cap{self.context}-tail{self.tail_length}"
                f"{page_suffix}"
            )
        if self.kind == "speculative":
            return (
                f"{self.profile.name}-r4-cap{self.context}-spec-"
                f"accept{self.preceding_accepted_tokens}{page_suffix}"
            )
        if self.kind == "prefill":
            return (
                f"{self.profile.name}-prefill-r{self.rows}-c{self.context}{page_suffix}"
            )
        return f"{self.profile.name}-r{self.rows}-c{self.context}{page_suffix}"

    @property
    def request_count(self) -> int:
        return 1 if self.kind in {"prefill", "speculative"} else self.rows

    @property
    def planned_batch(self) -> int:
        return (
            self.request_count
            if self.planned_max_batch is None
            else int(self.planned_max_batch)
        )

    @property
    def planned_q_rows(self) -> int:
        return (
            self.rows
            if self.planned_max_q_rows is None
            else int(self.planned_max_q_rows)
        )

    @property
    def planned_speculative_tokens(self) -> int:
        return (
            self.max_speculative_tokens
            if self.planned_max_speculative_tokens is None
            else int(self.planned_max_speculative_tokens)
        )

    @property
    def max_speculative_tokens(self) -> int:
        return 3 if self.kind == "speculative" else 0

    @property
    def positions(self) -> tuple[int, ...]:
        if self.kind == "stream_phase":
            position = (
                self.context - 1
                if self.tail_length == 0
                else self.context - COMPRESS_RATIO + self.tail_length - 1
            )
            return (position,)
        if self.kind == "speculative":
            setup_start = self.context - 2 * COMPRESS_RATIO
            start = setup_start + self.preceding_accepted_tokens
            return tuple(range(start, start + self.rows))
        if self.kind == "prefill":
            return tuple(range(self.context - self.rows, self.context))
        return (self.context - 1,) * self.rows

    @property
    def setup_positions(self) -> tuple[int, ...] | None:
        if self.kind != "speculative":
            return None
        start = self.context - 2 * COMPRESS_RATIO
        return tuple(range(start, start + self.rows))

    @property
    def active_sequence_length(self) -> int:
        return max(self.positions) + 1

    @property
    def groups(self) -> int:
        return self.context // COMPRESS_RATIO

    @property
    def rank_prefix_groups(self) -> int:
        if self.kind == "speculative":
            return (self.context - 2 * COMPRESS_RATIO) // COMPRESS_RATIO
        if self.kind == "prefill":
            return (self.context - self.rows) // COMPRESS_RATIO
        return self.groups - 1

    @property
    def main_pages_per_request(self) -> int:
        return (self.context + self.main_page_size - 1) // self.main_page_size

    @property
    def main_pages_total(self) -> int:
        return self.request_count * self.main_pages_per_request

    @property
    def main_pages_capacity(self) -> int:
        return self.planned_batch * self.main_pages_per_request

    @property
    def compressed_page_size(self) -> int:
        return self.main_page_size // COMPRESS_RATIO

    @property
    def compressed_pages_per_request(self) -> int:
        return (
            self.groups + self.compressed_page_size - 1
        ) // self.compressed_page_size

    @property
    def compressed_pages_capacity(self) -> int:
        return self.planned_batch * self.compressed_pages_per_request


@dataclass
class MutableStateRestore:
    compressed_rows: torch.Tensor
    compressed_rows_initial: torch.Tensor
    raw_key_rows: torch.Tensor
    raw_key_rows_initial: torch.Tensor
    raw_tags: torch.Tensor
    raw_tags_initial: torch.Tensor
    raw_rope_rows: torch.Tensor
    raw_rope_rows_initial: torch.Tensor
    interval_starts: torch.Tensor
    interval_starts_initial: torch.Tensor

    def restore(self) -> None:
        self.compressed_rows.copy_(self.compressed_rows_initial)
        self.raw_key_rows.copy_(self.raw_key_rows_initial)
        self.raw_tags.copy_(self.raw_tags_initial)
        self.raw_rope_rows.copy_(self.raw_rope_rows_initial)
        self.interval_starts.copy_(self.interval_starts_initial)

    def assert_restored(self) -> None:
        pairs = (
            (
                "compressed representative",
                self.compressed_rows,
                self.compressed_rows_initial,
            ),
            ("raw key slot", self.raw_key_rows, self.raw_key_rows_initial),
            ("raw logical tag", self.raw_tags, self.raw_tags_initial),
            ("raw RoPE slot", self.raw_rope_rows, self.raw_rope_rows_initial),
            ("interval anchor", self.interval_starts, self.interval_starts_initial),
        )
        for name, actual, expected in pairs:
            if not torch.equal(actual, expected):
                raise BenchmarkFailure(f"QSA state restore changed {name}")


@dataclass(frozen=True)
class PersistentStateSnapshot:
    main_k_cache: torch.Tensor
    main_v_cache: torch.Tensor
    compressed_rows: torch.Tensor
    raw_key_rows: torch.Tensor
    raw_tags: torch.Tensor
    raw_rope_rows: torch.Tensor
    interval_starts: torch.Tensor

    @classmethod
    def capture(
        cls,
        state: MutableStateRestore,
        *,
        main_k_cache: torch.Tensor,
        main_v_cache: torch.Tensor,
    ) -> PersistentStateSnapshot:
        return cls(
            main_k_cache=main_k_cache,
            main_v_cache=main_v_cache,
            compressed_rows=state.compressed_rows.clone(),
            raw_key_rows=state.raw_key_rows.clone(),
            raw_tags=state.raw_tags.clone(),
            raw_rope_rows=state.raw_rope_rows.clone(),
            interval_starts=state.interval_starts.clone(),
        )

    def assert_matches(
        self,
        state: MutableStateRestore,
        *,
        main_k_cache: torch.Tensor,
        main_v_cache: torch.Tensor,
        label: str,
    ) -> None:
        pairs = (
            ("main K cache", main_k_cache, self.main_k_cache),
            ("main V cache", main_v_cache, self.main_v_cache),
            ("compressed representative", state.compressed_rows, self.compressed_rows),
            ("raw key slot", state.raw_key_rows, self.raw_key_rows),
            ("raw logical tag", state.raw_tags, self.raw_tags),
            ("raw RoPE slot", state.raw_rope_rows, self.raw_rope_rows),
            ("interval anchor", state.interval_starts, self.interval_starts),
        )
        for name, actual, expected in pairs:
            if not torch.equal(actual, expected):
                raise BenchmarkFailure(f"{label} changed {name}")


@dataclass
class PreparedCase:
    case: BenchmarkCase
    binding: qsa.Binding
    dynamic: dict[str, torch.Tensor]
    state_restore: MutableStateRestore
    setup_metadata: dict[str, object]
    session: PreparationSession
    result: PreparationResult

    def close(self) -> None:
        self.result.close()
        self.session.close()

    def run(self) -> torch.Tensor:
        return qsa.run(self.binding, **self.dynamic)


def _parse_positive_csv(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(
            "expected a non-empty list of positive integers"
        )
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("values must be unique")
    return parsed


def _parse_context_csv(value: str) -> tuple[int, ...]:
    parts = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one context length")
    contexts: list[int] = []
    for part in parts:
        if part == "full":
            context = MODEL_MAX_CONTEXT
        else:
            try:
                context = int(part)
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    "contexts must be integers or 'full'"
                ) from error
        if context < COMPRESS_RATIO or context % COMPRESS_RATIO:
            raise argparse.ArgumentTypeError(
                f"contexts must be multiples of {COMPRESS_RATIO} and at least "
                f"{COMPRESS_RATIO}"
            )
        if context > MODEL_MAX_CONTEXT:
            raise argparse.ArgumentTypeError(
                f"contexts cannot exceed the model limit {MODEL_MAX_CONTEXT}"
            )
        contexts.append(context)
    if len(set(contexts)) != len(contexts):
        raise argparse.ArgumentTypeError("contexts must be unique")
    return tuple(contexts)


def _parse_profiles(value: str) -> tuple[str, ...]:
    names = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if names == ("all",):
        return tuple(PROFILES)
    if not names:
        raise argparse.ArgumentTypeError("expected at least one TP profile")
    unknown = sorted(set(names) - PROFILES.keys())
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown profiles {unknown}; choose from {sorted(PROFILES)} or all"
        )
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("profiles must be unique")
    return names


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        type=_parse_profiles,
        default=("tp2",),
        help="comma-separated TP profiles: tp1,tp2,tp4, or all (default: tp2)",
    )
    parser.add_argument(
        "--rows",
        type=_parse_positive_csv,
        default=DEFAULT_ROWS,
        help="comma-separated packed decode row counts",
    )
    parser.add_argument(
        "--prefill-rows",
        type=_parse_positive_csv,
        default=(),
        help=(
            "comma-separated packed single-request prefill row counts; cases "
            "whose row count exceeds the selected context are omitted"
        ),
    )
    parser.add_argument(
        "--contexts",
        type=_parse_context_csv,
        default=DEFAULT_CONTEXTS,
        help=(
            "comma-separated context lengths; 'full' means 262144 (default: 2048,8192)"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--main-cache-layout",
        choices=("interleaved", "separate"),
        default="interleaved",
        help=(
            "physical K/V allocation layout; interleaved matches the vLLM "
            "BLHNC per-layer cache view (default: interleaved)"
        ),
    )
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("bf16", "fp8_e4m3"),
        default="bf16",
        help="main K/V cache storage dtype (default: bf16)",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--eager-replays", type=int, default=20)
    parser.add_argument("--graph-replays", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--contract-cases",
        action="store_true",
        help=(
            "append fixed-capacity streaming-tail phases and a two-transaction "
            "packed accepted-prefix rollback case"
        ),
    )
    parser.add_argument(
        "--flush-l2", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--print-raw-samples",
        action="store_true",
        help="print each complete case record as JSON after its summary",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.warmup < 1 or args.eager_replays < 1 or args.graph_replays < 1:
        raise BenchmarkFailure("warmup and replay counts must be positive")
    if args.l2_flush_bytes < 0:
        raise BenchmarkFailure("L2 flush bytes must be non-negative")
    if any(rows > 8192 for rows in (*args.rows, *args.prefill_rows)):
        raise BenchmarkFailure(
            "row counts above 8192 require an explicit harness change"
        )


def _resolve_cases(args: argparse.Namespace) -> tuple[BenchmarkCase, ...]:
    cases = [
        BenchmarkCase(PROFILES[profile], rows, context)
        for profile in args.profiles
        for rows in args.rows
        for context in args.contexts
    ]
    cases.extend(
        BenchmarkCase(PROFILES[profile], rows, context, kind="prefill")
        for profile in args.profiles
        for rows in args.prefill_rows
        for context in args.contexts
        if rows <= context
    )
    if args.contract_cases:
        profile = PROFILES["tp2"]
        cases.extend(
            BenchmarkCase(
                profile,
                1,
                8192,
                kind="stream_phase",
                tail_length=tail,
            )
            for tail in range(COMPRESS_RATIO)
        )
        cases.append(
            BenchmarkCase(
                profile,
                4,
                8192,
                kind="speculative",
                preceding_accepted_tokens=2,
            )
        )
    return tuple(cases)


def _cache_capacity_bytes(
    case: BenchmarkCase,
    *,
    kv_cache_dtype: str = "bf16",
) -> dict[str, int]:
    element_bytes = torch.bfloat16.itemsize
    kv_element_bytes = (
        torch.float8_e4m3fn.itemsize
        if kv_cache_dtype == "fp8_e4m3"
        else torch.bfloat16.itemsize
    )
    main_kv = (
        2
        * case.main_pages_capacity
        * case.main_page_size
        * case.profile.kv_heads
        * HEAD_DIM
        * kv_element_bytes
    )
    compressed = (
        case.planned_batch
        * case.compressed_pages_per_request
        * case.compressed_page_size
        * INDEX_HEAD_DIM
        * element_bytes
    )
    raw_ring_capacity = COMPRESS_RATIO * (
        (COMPRESS_RATIO + case.max_speculative_tokens + COMPRESS_RATIO - 1)
        // COMPRESS_RATIO
    )
    raw = case.planned_batch * (
        raw_ring_capacity * INDEX_HEAD_DIM * element_bytes
        + raw_ring_capacity * torch.int64.itemsize
        + raw_ring_capacity * POSITION_AXES * torch.int64.itemsize
        + torch.int64.itemsize
    )
    return {
        "main_kv": main_kv,
        "compressed": compressed,
        "raw_state": raw,
        "total": main_kv + compressed + raw,
    }


def _disjoint_page_table(
    request_count: int,
    pages_per_request: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return dense request tables with pairwise-disjoint physical page IDs."""
    return torch.arange(
        int(request_count) * int(pages_per_request),
        dtype=torch.int32,
        device=device,
    ).view(int(request_count), int(pages_per_request))


def _random_bf16(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
    scale: float = 0.25,
) -> torch.Tensor:
    return (
        torch.randn(
            shape,
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        * scale
    )


def _rank_safe_raw_keys(
    prepared_query: torch.Tensor,
    key_norm_weight: torch.Tensor,
    count: int,
    *,
    generator: torch.Generator,
    rms_norm_eps: float,
) -> torch.Tensor:
    """Build distinct keys whose compressed score is strictly negative."""
    if prepared_query.ndim != 2 or int(prepared_query.shape[1]) != INDEX_HEAD_DIM:
        raise ValueError("prepared_query must have shape [index_heads, index_head_dim]")
    if tuple(key_norm_weight.shape) != (INDEX_HEAD_DIM,):
        raise ValueError("key_norm_weight must have shape [index_head_dim]")
    if count <= 0:
        raise ValueError("rank-safe raw-key count must be positive")

    query = prepared_query.float()
    gram = query @ query.T
    identity = torch.eye(int(query.shape[0]), dtype=torch.float32, device=query.device)
    gram = gram + identity * 1e-4
    separating = -4.0 * (
        torch.linalg.solve(
            gram,
            torch.ones(
                (int(query.shape[0]),), dtype=torch.float32, device=query.device
            ),
        )
        @ query
    )

    noise = torch.randn(
        (int(count), INDEX_HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
        device=query.device,
    )
    projection = torch.linalg.solve(gram, query @ noise.T).T @ query
    null_component = noise - projection
    null_component = null_component / null_component.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-6)
    transformed = separating[None, :] + (null_component * separating.norm() * 0.75)
    divisor = 1.0 + key_norm_weight.float()
    if bool(torch.any(divisor.abs() < 0.25)):
        raise BenchmarkFailure("index-key norm weight is unsuitable for rank-safe keys")
    raw = (transformed / divisor[None, :]).to(torch.bfloat16)
    representatives = gemma_rmsnorm_reference(
        raw,
        key_norm_weight,
        rms_norm_eps,
    )
    scores = torch.einsum("hd,nd->hn", query, representatives.float())
    if not bool(torch.all(scores < -0.25)):
        raise BenchmarkFailure("rank-safe raw keys do not have a negative score margin")
    if int(torch.unique(raw, dim=0).shape[0]) != int(count):
        raise BenchmarkFailure("rank-safe raw keys must be distinct")
    return raw


def _identity_rope(value: torch.Tensor, _positions: torch.Tensor) -> torch.Tensor:
    return value


def _selector_rank_coefficients(
    count: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return BF16 score markers separated beyond reduction-order noise."""
    if count < 0:
        raise ValueError("selector rank count must be non-negative")
    if count == 0:
        return torch.empty((0,), dtype=torch.bfloat16, device=device)
    coefficients = torch.logspace(
        1.0,
        -7.0,
        count,
        base=2.0,
        dtype=torch.float32,
        device=device,
    ).to(torch.bfloat16)
    if count > 1 and not bool(torch.all(coefficients[:-1] > coefficients[1:])):
        raise BenchmarkFailure("selector rank coefficients are not distinct")
    return coefficients


def _selector_rank_group_ids(
    groups: int,
    count: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Spread rank markers across completed groups, excluding the in-flight tail."""
    available = max(int(groups) - 1, 0)
    if count < 0 or count > available:
        raise ValueError("selector rank count exceeds completed groups")
    if count == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    if count == 1:
        return torch.zeros((1,), dtype=torch.int64, device=device)
    ranks = torch.arange(count, dtype=torch.int64, device=device)
    return torch.div(
        ranks * (available - 1),
        count - 1,
        rounding_mode="floor",
    )


def _initialize_selector_dataset(
    *,
    prepared_query_by_request: torch.Tensor,
    compressed_by_request: torch.Tensor,
    raw_ring: torch.Tensor,
    raw_index_key: torch.Tensor,
    key_norm_weight: torch.Tensor,
    rank_prefix_groups: int,
    group_budget: int,
    generator: torch.Generator,
    rms_norm_eps: float,
) -> None:
    """Build rank-separated representatives for an exact selector oracle."""
    requests = int(prepared_query_by_request.shape[0])
    flattened = compressed_by_request.reshape(requests, -1, INDEX_HEAD_DIM)
    flattened.zero_()

    for request in range(requests):
        raw_ring[request].copy_(
            _rank_safe_raw_keys(
                prepared_query_by_request[request],
                key_norm_weight,
                int(raw_ring.shape[1]),
                generator=generator,
                rms_norm_eps=rms_norm_eps,
            )
        )
    if int(raw_index_key.shape[0]) == requests:
        for request in range(requests):
            raw_index_key[request].copy_(
                _rank_safe_raw_keys(
                    prepared_query_by_request[request],
                    key_norm_weight,
                    1,
                    generator=generator,
                    rms_norm_eps=rms_norm_eps,
                )[0]
            )
    elif requests == 1:
        raw_index_key.copy_(
            _rank_safe_raw_keys(
                prepared_query_by_request[0],
                key_norm_weight,
                int(raw_index_key.shape[0]),
                generator=generator,
                rms_norm_eps=rms_norm_eps,
            )
        )
    else:
        raise BenchmarkFailure(
            "rank-safe raw-key construction requires one row per request or one "
            "packed request"
        )

    # Rank markers stay strictly before any group the timed transaction may
    # replace. Rank-safe raw inputs leave mutable groups below every marker.
    ranked_groups = min(int(group_budget), max(int(rank_prefix_groups), 0))
    if ranked_groups == 0:
        return
    coefficients = _selector_rank_coefficients(
        ranked_groups,
        device=prepared_query_by_request.device,
    )
    ranked_group_ids = _selector_rank_group_ids(
        int(rank_prefix_groups) + 1,
        ranked_groups,
        device=prepared_query_by_request.device,
    )
    peak_dimensions = (
        prepared_query_by_request[:, 0].float().abs().argmax(dim=-1, keepdim=True)
    )
    peak_signs = torch.gather(
        prepared_query_by_request[:, 0], 1, peak_dimensions
    ).sign()
    peak_signs = torch.where(
        peak_signs == 0,
        torch.ones_like(peak_signs),
        peak_signs,
    )
    rank_direction = torch.zeros_like(prepared_query_by_request[:, 0])
    rank_direction.scatter_(1, peak_dimensions, peak_signs)
    flattened[:, ranked_group_ids, :] = (
        rank_direction[:, None, :] * coefficients[None, :, None]
    )


def _make_caps(
    case: BenchmarkCase,
    device: torch.device,
    *,
    kv_dtype: torch.dtype,
) -> qsa.Caps:
    return qsa.Caps(
        device=device,
        max_batch=case.planned_batch,
        max_raw_state_slots=case.planned_batch,
        max_q_rows=case.planned_q_rows,
        max_seq_len=case.context,
        num_main_cache_pages=case.main_pages_capacity,
        num_compressed_cache_pages=(case.compressed_pages_capacity),
        main_page_size=case.main_page_size,
        compressed_page_size=case.compressed_page_size,
        max_speculative_tokens=case.planned_speculative_tokens,
        q_heads=case.profile.q_heads,
        kv_heads=case.profile.kv_heads,
        head_dim=HEAD_DIM,
        index_heads=INDEX_HEADS,
        index_kv_heads=INDEX_KV_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        index_rotary_dim=INDEX_ROTARY_DIM,
        compress_ratio=COMPRESS_RATIO,
        budget=BUDGET,
        position_axes=POSITION_AXES,
        mrope_sections=MROPE_SECTIONS,
        mrope_interleaved=True,
        kv_dtype=kv_dtype,
    )


def _prepare_case(
    case: BenchmarkCase,
    *,
    device: torch.device,
    seed: int,
    main_cache_layout: str,
    kv_cache_dtype: str,
) -> PreparedCase:
    kv_dtype = torch.float8_e4m3fn if kv_cache_dtype == "fp8_e4m3" else torch.bfloat16
    caps = _make_caps(case, device, kv_dtype=kv_dtype)
    declaration = qsa.plan(caps)
    generator = torch.Generator(device=device).manual_seed(seed)

    main_cache_shape = (
        case.main_pages_capacity,
        case.main_page_size,
        case.profile.kv_heads,
        HEAD_DIM,
    )
    if main_cache_layout == "interleaved":
        main_kv_source = _random_bf16(
            (case.main_pages_capacity, 2, *main_cache_shape[1:]),
            generator=generator,
            device=device,
        )
        main_kv = torch.empty_like(main_kv_source, dtype=kv_dtype)
        main_k, main_v = main_kv.unbind(1)
        source_k, source_v = main_kv_source.unbind(1)
    elif main_cache_layout == "separate":
        source_k = _random_bf16(
            main_cache_shape,
            generator=generator,
            device=device,
        )
        source_v = _random_bf16(
            main_cache_shape,
            generator=generator,
            device=device,
        )
        main_k = torch.empty_like(source_k, dtype=kv_dtype)
        main_v = torch.empty_like(source_v, dtype=kv_dtype)
    else:
        raise BenchmarkFailure(f"unknown main-cache layout {main_cache_layout!r}")
    k_descale = None
    v_descale = None
    if kv_dtype == torch.float8_e4m3fn:
        k_descale = torch.tensor([0.0125], dtype=torch.float32, device=device)
        v_descale = torch.tensor([0.01], dtype=torch.float32, device=device)
        main_k.copy_((source_k.float() / k_descale).clamp(-448.0, 448.0))
        main_v.copy_((source_v.float() / v_descale).clamp(-448.0, 448.0))
    else:
        main_k.copy_(source_k)
        main_v.copy_(source_v)
    main_table = _disjoint_page_table(
        case.planned_batch,
        case.main_pages_per_request,
        device=device,
    )

    compressed = _random_bf16(
        (
            case.compressed_pages_capacity,
            case.compressed_page_size,
            INDEX_HEAD_DIM,
        ),
        generator=generator,
        device=device,
    )
    compressed_by_request = compressed.view(
        case.planned_batch,
        case.compressed_pages_per_request,
        case.compressed_page_size,
        INDEX_HEAD_DIM,
    )
    compressed_table = _disjoint_page_table(
        case.planned_batch,
        case.compressed_pages_per_request,
        device=device,
    )

    raw_ring = _random_bf16(
        (case.planned_batch, caps.raw_ring_capacity, INDEX_HEAD_DIM),
        generator=generator,
        device=device,
    )
    raw_tags = torch.full(
        (case.planned_batch, caps.raw_ring_capacity),
        -1,
        dtype=torch.int64,
        device=device,
    )
    raw_rope = torch.full(
        (case.planned_batch, caps.raw_ring_capacity, POSITION_AXES),
        -1,
        dtype=torch.int64,
        device=device,
    )
    interval_starts = torch.full(
        (case.planned_batch,), -1, dtype=torch.int64, device=device
    )
    state_slot_ids = torch.arange(case.planned_batch, dtype=torch.int64, device=device)

    q_weight = (
        torch.randn(
            (INDEX_HEAD_DIM,), generator=generator, dtype=torch.float32, device=device
        )
        * 0.05
    )
    k_weight = (
        torch.randn(
            (INDEX_HEAD_DIM,), generator=generator, dtype=torch.float32, device=device
        )
        * 0.05
    )
    rope_cos = torch.ones(
        (case.context, INDEX_ROTARY_DIM // 2),
        dtype=torch.bfloat16,
        device=device,
    )
    rope_sin = torch.zeros_like(rope_cos)
    output = torch.empty(
        (case.planned_q_rows, case.profile.q_heads, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    selected = torch.empty(
        (case.planned_q_rows, caps.selection_width),
        dtype=torch.int32,
        device=device,
    )

    binding_args = dict(
        main_k_cache=main_k,
        main_v_cache=main_v,
        k_descale=k_descale,
        v_descale=v_descale,
        main_block_table=main_table,
        compressed_k_cache=compressed,
        compressed_block_table=compressed_table,
        raw_k_ring=raw_ring,
        raw_logical_positions=raw_tags,
        raw_rope_positions=raw_rope,
        raw_interval_start_positions=interval_starts,
        raw_state_slot_ids=state_slot_ids,
        index_q_norm_weight=q_weight,
        index_k_norm_weight=k_weight,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        output=output,
        selected_positions=selected,
    )

    query = _random_bf16(
        (case.rows, case.profile.q_heads, HEAD_DIM),
        generator=generator,
        device=device,
    )
    index_query = _random_bf16(
        (case.rows, INDEX_HEADS, INDEX_HEAD_DIM),
        generator=generator,
        device=device,
    )
    if case.kind in {"prefill", "speculative"}:
        index_query[1:].copy_(index_query[:1].expand(case.rows - 1, -1, -1))
    raw_key = _random_bf16(
        (case.rows, INDEX_HEAD_DIM), generator=generator, device=device
    )
    query_positions = torch.tensor(case.positions, dtype=torch.int64, device=device)
    if case.kind in {"prefill", "speculative"}:
        request_ids = torch.zeros((case.rows,), dtype=torch.int64, device=device)
        query_start_loc = torch.tensor([0, case.rows], dtype=torch.int32, device=device)
    else:
        request_ids = torch.arange(case.rows, dtype=torch.int64, device=device)
        query_start_loc = torch.arange(case.rows + 1, dtype=torch.int32, device=device)
    sequence_lengths = torch.zeros(
        (case.planned_batch,), dtype=torch.int32, device=device
    )
    sequence_lengths[: case.request_count] = case.active_sequence_length
    padded_query_start_loc = torch.full(
        (case.planned_batch + 1,),
        case.rows,
        dtype=torch.int32,
        device=device,
    )
    padded_query_start_loc[: case.request_count + 1].copy_(query_start_loc)
    accepted = torch.zeros((case.planned_batch,), dtype=torch.int32, device=device)
    accepted[: case.request_count] = 1
    if case.kind == "speculative":
        accepted[0] = case.preceding_accepted_tokens
    dynamic = {
        "query": query,
        "index_query": index_query,
        "raw_index_key": raw_key,
        "request_ids": request_ids,
        "query_positions": query_positions,
        "rope_positions": query_positions[:, None]
        .expand(case.rows, POSITION_AXES)
        .contiguous(),
        "sequence_lengths": sequence_lengths,
        "query_start_loc": padded_query_start_loc,
        "num_accepted_tokens": accepted,
        "is_prefilling": torch.cat(
            (
                torch.full(
                    (case.request_count,),
                    case.kind == "prefill",
                    dtype=torch.bool,
                    device=device,
                ),
                torch.zeros(
                    (case.planned_batch - case.request_count,),
                    dtype=torch.bool,
                    device=device,
                ),
            )
        ),
    }

    prepared_selector_query = gemma_rmsnorm_reference(
        index_query,
        q_weight,
        caps.rms_norm_eps,
    )
    prepared_query_by_request = (
        prepared_selector_query
        if case.request_count == case.rows
        else prepared_selector_query[:1]
    )
    _initialize_selector_dataset(
        prepared_query_by_request=prepared_query_by_request,
        compressed_by_request=compressed_by_request[: case.request_count],
        raw_ring=raw_ring[: case.request_count],
        raw_index_key=raw_key,
        key_norm_weight=k_weight,
        rank_prefix_groups=case.rank_prefix_groups,
        group_budget=caps.group_budget,
        generator=generator,
        rms_norm_eps=caps.rms_norm_eps,
    )

    initial_positions = case.setup_positions or case.positions
    initial_acceptance = (
        1 if case.setup_positions is not None else case.preceding_accepted_tokens
    )
    first_positions = (
        initial_positions
        if case.request_count == case.rows
        else (initial_positions[0],)
    )
    for request, first_position in enumerate(first_positions):
        interval_starts[request] = first_position - initial_acceptance
        for prior in range(
            max(0, first_position - caps.raw_ring_capacity), first_position
        ):
            slot = prior % caps.raw_ring_capacity
            raw_tags[request, slot] = prior
            raw_rope[request, slot, :] = prior
    def prepare_call(state):
        (spec,) = state.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        binding = state.bind_for_preparation(scratch=scratch, **binding_args)
        mutable = (
            compressed,
            raw_ring,
            raw_tags,
            raw_rope,
            interval_starts,
            output,
            selected,
        )
        initial = tuple(value.clone() for value in mutable)

        def restore():
            for value, original in zip(mutable, initial, strict=True):
                value.copy_(original)

        return PreparedCall(
            run=lambda: qsa.run(binding, **dynamic),
            reset=restore,
            restore=restore,
            owners=(binding,),
        )

    session = PreparationSession(device=device)
    request = declaration.request(
        name=case.name,
        prepare_call=prepare_call,
    )
    result = session.prepare((request,))
    plan = declaration
    (scratch_spec,) = require_prepared(plan, "attention.qsa").scratch_specs()
    scratch = torch.empty(
        scratch_spec.shape,
        dtype=scratch_spec.dtype,
        device=scratch_spec.device,
    )
    binding = qsa.bind(plan, scratch=scratch, **binding_args)

    setup_metadata: dict[str, object] = {
        "transaction": (
            "packed_prefill_interval"
            if case.kind == "prefill"
            else "single_decode_interval"
        ),
        "setup_transaction_executed": False,
    }
    if case.setup_positions is not None:
        setup_query_positions = torch.tensor(
            case.setup_positions, dtype=torch.int64, device=device
        )
        setup_raw_index_key = _rank_safe_raw_keys(
            prepared_query_by_request[0],
            k_weight,
            case.rows,
            generator=generator,
            rms_norm_eps=caps.rms_norm_eps,
        )
        if torch.equal(setup_raw_index_key, raw_key):
            raise BenchmarkFailure(
                f"{case.name}: setup and replacement raw keys must be distinct"
            )
        setup_dynamic = {
            "query": _random_bf16(
                (case.rows, case.profile.q_heads, HEAD_DIM),
                generator=generator,
                device=device,
            ),
            "index_query": _random_bf16(
                (case.rows, INDEX_HEADS, INDEX_HEAD_DIM),
                generator=generator,
                device=device,
            ),
            "raw_index_key": setup_raw_index_key,
            "request_ids": torch.zeros((case.rows,), dtype=torch.int64, device=device),
            "query_positions": setup_query_positions,
            "rope_positions": setup_query_positions[:, None]
            .expand(case.rows, POSITION_AXES)
            .contiguous(),
            "sequence_lengths": torch.tensor(
                [case.setup_positions[-1] + 1, *([0] * (case.planned_batch - 1))],
                dtype=torch.int32,
                device=device,
            ),
            "query_start_loc": torch.tensor(
                [0, *([case.rows] * case.planned_batch)],
                dtype=torch.int32,
                device=device,
            ),
            "num_accepted_tokens": torch.tensor(
                [1, *([0] * (case.planned_batch - 1))],
                dtype=torch.int32,
                device=device,
            ),
            "is_prefilling": torch.zeros(
                (case.planned_batch,), dtype=torch.bool, device=device
            ),
        }
        setup_oracle_compressed = compressed.clone()
        setup_oracle_raw_ring = raw_ring[0].clone()
        setup_oracle_raw_tags = raw_tags[0].clone()
        setup_oracle_raw_rope = raw_rope[0].clone()
        setup_main_k_before = main_k.clone()
        setup_main_v_before = main_v.clone()
        setup_group_ids, setup_representatives, expected_anchor = (
            packed_stream_compress_reference(
                setup_dynamic["raw_index_key"],
                setup_dynamic["query_positions"],
                setup_dynamic["rope_positions"],
                setup_oracle_raw_ring,
                setup_oracle_raw_tags,
                setup_oracle_raw_rope,
                prior_interval_start_position=int(interval_starts[0]),
                num_accepted_tokens=1,
                compress_ratio=COMPRESS_RATIO,
                key_norm_weight=k_weight,
                eps=caps.rms_norm_eps,
                rope=_identity_rope,
            )
        )
        paged_store_compressed_reference(
            setup_oracle_compressed,
            compressed_table,
            0,
            setup_group_ids,
            setup_representatives,
        )
        qsa.run(binding, **setup_dynamic)
        torch.cuda.synchronize(device)
        if int(interval_starts[0]) != expected_anchor:
            raise BenchmarkFailure(
                f"{case.name}: setup committed anchor {int(interval_starts[0])}, "
                f"expected {expected_anchor}"
            )
        torch.testing.assert_close(
            compressed,
            setup_oracle_compressed,
            rtol=0.0,
            atol=2e-2,
        )
        if not torch.equal(raw_ring[0], setup_oracle_raw_ring):
            raise BenchmarkFailure(f"{case.name}: setup raw-ring payload mismatch")
        if not torch.equal(raw_tags[0], setup_oracle_raw_tags):
            raise BenchmarkFailure(f"{case.name}: setup raw logical-tag mismatch")
        if not torch.equal(raw_rope[0], setup_oracle_raw_rope):
            raise BenchmarkFailure(f"{case.name}: setup raw RoPE-tag mismatch")
        if not torch.equal(main_k, setup_main_k_before) or not torch.equal(
            main_v, setup_main_v_before
        ):
            raise BenchmarkFailure(f"{case.name}: setup mutated main K/V")
        setup_metadata = {
            "transaction": "replacement_interval",
            "setup_transaction_executed": True,
            "setup_positions": list(case.setup_positions),
            "setup_preceding_interval_accepted_tokens": 1,
            "replacement_positions": list(case.positions),
            "timed_setup_interval_accepted_tokens": case.preceding_accepted_tokens,
            "post_setup_anchor": expected_anchor,
            "setup_persistent_state_reference_checked": True,
            "setup_main_kv_read_only": True,
        }

    if case.kind in {"prefill", "speculative"}:
        compressed_rows = compressed
    else:
        completed_groups = {
            position // COMPRESS_RATIO
            for position in case.positions
            if (position + 1) % COMPRESS_RATIO == 0
        }
        if completed_groups:
            (completed_group,) = completed_groups
            group_page, group_offset = divmod(
                completed_group, case.compressed_page_size
            )
            compressed_rows = compressed_by_request[:, group_page, group_offset]
        else:
            compressed_rows = compressed[:0]
    state_restore = MutableStateRestore(
        compressed_rows=compressed_rows,
        compressed_rows_initial=compressed_rows.clone(),
        raw_key_rows=raw_ring,
        raw_key_rows_initial=raw_ring.clone(),
        raw_tags=raw_tags,
        raw_tags_initial=raw_tags.clone(),
        raw_rope_rows=raw_rope,
        raw_rope_rows_initial=raw_rope.clone(),
        interval_starts=interval_starts,
        interval_starts_initial=interval_starts.clone(),
    )
    return PreparedCase(
        case=case,
        binding=binding,
        dynamic=dynamic,
        state_restore=state_restore,
        setup_metadata=setup_metadata,
        session=session,
        result=result,
    )


def _validate_correctness(
    prepared: PreparedCase,
) -> tuple[torch.Tensor, dict[str, object], PersistentStateSnapshot]:
    case = prepared.case
    binding = prepared.binding
    prepared.state_restore.restore()
    main_k_before = binding.main_k_cache.clone()
    main_v_before = binding.main_v_cache.clone()

    oracle_compressed = binding.compressed_k_cache.clone()
    stale_speculative_compressed = (
        oracle_compressed.clone() if case.kind == "speculative" else None
    )
    oracle_raw_ring = prepared.state_restore.raw_key_rows_initial.clone()
    oracle_raw_tags = prepared.state_restore.raw_tags_initial.clone()
    oracle_raw_rope = prepared.state_restore.raw_rope_rows_initial.clone()
    expected_anchors = prepared.state_restore.interval_starts_initial.clone()
    for request in range(case.request_count):
        start = int(prepared.dynamic["query_start_loc"][request])
        end = int(prepared.dynamic["query_start_loc"][request + 1])
        state_slot = int(binding.raw_state_slot_ids[request])
        group_ids, representatives, expected_anchor = packed_stream_compress_reference(
            prepared.dynamic["raw_index_key"][start:end],
            prepared.dynamic["query_positions"][start:end],
            prepared.dynamic["rope_positions"][start:end],
            oracle_raw_ring[state_slot],
            oracle_raw_tags[state_slot],
            oracle_raw_rope[state_slot],
            prior_interval_start_position=int(expected_anchors[state_slot]),
            num_accepted_tokens=int(prepared.dynamic["num_accepted_tokens"][request]),
            is_prefilling=bool(prepared.dynamic["is_prefilling"][request]),
            compress_ratio=COMPRESS_RATIO,
            key_norm_weight=binding.index_k_norm_weight,
            eps=binding.plan.caps.rms_norm_eps,
            rope=_identity_rope,
        )
        paged_store_compressed_reference(
            oracle_compressed,
            binding.compressed_block_table,
            request,
            group_ids,
            representatives,
        )
        expected_anchors[state_slot] = expected_anchor

    speculative_replacement_delta: float | None = None
    if stale_speculative_compressed is not None:
        speculative_replacement_delta = float(
            (oracle_compressed.float() - stale_speculative_compressed.float())
            .abs()
            .max()
        )
        if speculative_replacement_delta <= 0.1:
            raise BenchmarkFailure(
                f"{case.name}: replacement representative is not distinct from "
                "the stale setup state"
            )

    actual = prepared.run().clone()
    selected = binding.selected_positions[: case.rows].clone()
    torch.cuda.synchronize(binding.plan.caps.device)

    if not bool(torch.all(torch.isfinite(actual))):
        raise BenchmarkFailure(f"{case.name}: eager output is non-finite")
    nonzero = int(torch.count_nonzero(actual).item())
    if nonzero == 0:
        raise BenchmarkFailure(f"{case.name}: eager output is all zero")
    if not torch.equal(binding.main_k_cache, main_k_before) or not torch.equal(
        binding.main_v_cache, main_v_before
    ):
        raise BenchmarkFailure(f"{case.name}: QSA mutated the read-only main K/V cache")
    torch.testing.assert_close(
        binding.compressed_k_cache,
        oracle_compressed,
        rtol=0.0,
        atol=2e-2,
    )
    if not torch.equal(binding.raw_k_ring, oracle_raw_ring):
        raise BenchmarkFailure(f"{case.name}: raw-ring payload mismatch")
    if not torch.equal(binding.raw_logical_positions, oracle_raw_tags):
        raise BenchmarkFailure(f"{case.name}: raw-ring logical-tag mismatch")
    if not torch.equal(binding.raw_rope_positions, oracle_raw_rope):
        raise BenchmarkFailure(f"{case.name}: raw-ring RoPE-tag mismatch")
    if not torch.equal(binding.raw_interval_start_positions, expected_anchors):
        raise BenchmarkFailure(f"{case.name}: committed interval anchor mismatch")

    prepared_query = gemma_rmsnorm_reference(
        prepared.dynamic["index_query"],
        binding.index_q_norm_weight,
        binding.plan.caps.rms_norm_eps,
    )
    final_workspace_rows = case.rows % binding.plan.workspace_q_rows
    if final_workspace_rows == 0:
        final_workspace_rows = min(case.rows, binding.plan.workspace_q_rows)
    torch.testing.assert_close(
        binding.prepared_index_query[:final_workspace_rows],
        prepared_query[-final_workspace_rows:],
        rtol=0.0,
        atol=2e-2,
    )

    compressed_groups = oracle_compressed.view(
        case.planned_batch,
        case.compressed_pages_per_request,
        case.compressed_page_size,
        INDEX_HEAD_DIM,
    ).reshape(case.planned_batch, -1, INDEX_HEAD_DIM)[
        : case.request_count, : case.groups
    ]
    reference_k_cache = binding.main_k_cache.float()
    reference_v_cache = binding.main_v_cache.float()
    sparse_gqa_atol = 2e-2
    if binding.k_descale is not None:
        assert binding.v_descale is not None
        reference_k_cache *= binding.k_descale
        reference_v_cache *= binding.v_descale
        sparse_gqa_atol = 4e-2
    max_abs = 0.0
    tail_lengths: list[int] = []
    if case.rows <= 64:
        reference_rows = tuple(range(case.rows))
    else:
        reference_rows = tuple(
            sorted(
                {
                    0,
                    1,
                    COMPRESS_RATIO - 1,
                    COMPRESS_RATIO,
                    case.rows // 2,
                    case.rows - 2,
                    case.rows - 1,
                }
            )
        )
    for row in reference_rows:
        request = int(prepared.dynamic["request_ids"][row])
        sequence_length = int(prepared.dynamic["sequence_lengths"][request])
        _, expected_selected = score_select_reference(
            prepared_query[row : row + 1],
            compressed_groups[request],
            prepared.dynamic["query_positions"][row : row + 1],
            sequence_length,
            COMPRESS_RATIO,
            BUDGET,
        )
        if not torch.equal(selected[row : row + 1], expected_selected):
            raise BenchmarkFailure(f"{case.name}: selector mismatch at row {row}")
        expected_output = sparse_paged_gqa_reference(
            prepared.dynamic["query"][row : row + 1],
            reference_k_cache,
            reference_v_cache,
            binding.main_block_table,
            prepared.dynamic["request_ids"][row : row + 1],
            expected_selected,
            prepared.dynamic["query_positions"][row : row + 1],
        )
        try:
            torch.testing.assert_close(
                actual[row : row + 1],
                expected_output,
                rtol=0.0,
                atol=sparse_gqa_atol,
            )
        except AssertionError as error:
            raise BenchmarkFailure(
                f"{case.name}: sparse GQA reference mismatch at row {row}: {error}"
            ) from error
        max_abs = max(
            max_abs,
            float(
                (actual[row : row + 1].float() - expected_output.float()).abs().max()
            ),
        )

        position = int(prepared.dynamic["query_positions"][row])
        eligible = min(
            (position + 1) // COMPRESS_RATIO,
            sequence_length // COMPRESS_RATIO,
            case.groups,
        )
        expanded_count = min(eligible, binding.plan.caps.group_budget) * COMPRESS_RATIO
        tail_start = ((position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
        expected_tail = torch.arange(
            tail_start,
            position + 1,
            dtype=torch.int32,
            device=selected.device,
        )
        tail_length = int(expected_tail.numel())
        tail_lengths.append(tail_length)
        if not torch.equal(
            selected[row, expanded_count : expanded_count + tail_length],
            expected_tail,
        ):
            raise BenchmarkFailure(f"{case.name}: tail mismatch at row {row}")
        if bool(torch.any(selected[row, expanded_count + tail_length :] != -1)):
            raise BenchmarkFailure(
                f"{case.name}: selector padding mismatch at row {row}"
            )

    eager_persistent_state = PersistentStateSnapshot.capture(
        prepared.state_restore,
        main_k_cache=main_k_before,
        main_v_cache=main_v_before,
    )
    prepared.state_restore.restore()
    torch.cuda.synchronize(binding.plan.caps.device)
    prepared.state_restore.assert_restored()
    return (
        actual,
        {
            "reference": (
                "packed_stream_compress_reference + score_select_reference + "
                "sparse_paged_gqa_reference"
            ),
            "reference_rows": list(reference_rows),
            "selector_exact": True,
            "sparse_gqa_rtol": 0.0,
            "sparse_gqa_atol": sparse_gqa_atol,
            "sparse_gqa_max_abs": max_abs,
            "finite": True,
            "nonzero_elements": nonzero,
            "state_restore_exact": True,
            "main_kv_read_only": True,
            "main_page_tables_disjoint": int(
                torch.unique(binding.main_block_table).numel()
            )
            == int(binding.main_block_table.numel()),
            "compression_reference_checked": True,
            "compressed_representatives_reference": True,
            "raw_ring_payload_exact": True,
            "raw_logical_tags_exact": True,
            "raw_rope_tags_exact": True,
            "interval_anchor_exact": True,
            "tail_lengths": tail_lengths,
            "speculative_replacement_representative_max_abs_delta": (
                speculative_replacement_delta
            ),
        },
        eager_persistent_state,
    )


def _time_eager(run, restore, *, warmup, replays, l2_flush):
    from b12x.testing.benchmark import measure_call
    return measure_call(run, warmup=warmup, samples=replays, reset=restore,
                        eviction=l2_flush or (lambda: None)).raw_samples("workload")


def _summary(samples: Sequence[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("cannot summarize an empty sample set")
    ordered = sorted(float(sample) for sample in samples)
    return {
        "median_us": statistics.median(ordered),
        "p10_us": ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        "p90_us": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def _binding_storage_bytes(prepared: PreparedCase) -> int:
    storages: dict[tuple[str, int], int] = {}
    for field in fields(prepared.binding):
        value = getattr(prepared.binding, field.name)
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), int(storage.data_ptr()))
            storages[key] = int(storage.nbytes())
    for value in prepared.dynamic.values():
        storage = value.untyped_storage()
        key = (str(value.device), int(storage.data_ptr()))
        storages[key] = int(storage.nbytes())
    return sum(storages.values())


def _run_case(
    case: BenchmarkCase,
    *,
    args: argparse.Namespace,
    device: torch.device,
    l2_flush: Callable[[], None] | None,
    case_index: int,
) -> dict[str, object]:
    prepared = _prepare_case(
        case,
        device=device,
        seed=args.seed + 1009 * case_index,
        main_cache_layout=args.main_cache_layout,
        kv_cache_dtype=args.kv_cache_dtype,
    )
    eager_output, correctness, eager_persistent_state = _validate_correctness(prepared)
    eager_selected = prepared.binding.selected_positions[: case.rows].clone()

    with prepared.session.capture():
        graph = capture_cuda_graph(
            prepared.run,
            warmup=args.warmup,
            prepare=prepared.state_restore.restore,
        )
    graph_addresses = {
        "output": prepared.binding.output.data_ptr(),
        "selected_positions": prepared.binding.selected_positions.data_ptr(),
        "scratch": prepared.binding.scratch.data_ptr(),
    }
    prepared.state_restore.restore()
    prepared.binding.scratch.fill_(0xFF)
    prepared.binding.output[: case.rows].fill_(float("nan"))
    prepared.binding.selected_positions[: case.rows].fill_(SELECTED_POSITION_POISON)
    torch.cuda.synchronize(device)
    allocated_before_replay = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after_replay = torch.cuda.memory_allocated(device)
    replay_allocation_delta = allocated_after_replay - allocated_before_replay
    if replay_allocation_delta != 0:
        raise BenchmarkFailure(
            f"{case.name}: CUDA graph replay allocated {replay_allocation_delta} bytes"
        )
    if graph_addresses != {
        "output": prepared.binding.output.data_ptr(),
        "selected_positions": prepared.binding.selected_positions.data_ptr(),
        "scratch": prepared.binding.scratch.data_ptr(),
    }:
        raise BenchmarkFailure(
            f"{case.name}: CUDA graph replay changed a bound tensor address"
        )
    graph_output = prepared.binding.output[: case.rows].clone()
    graph_selected = prepared.binding.selected_positions[: case.rows].clone()
    torch.cuda.synchronize(device)
    if not torch.equal(graph_output, eager_output):
        raise BenchmarkFailure(f"{case.name}: CUDA graph output differs from eager")
    if not torch.equal(graph_selected, eager_selected):
        raise BenchmarkFailure(
            f"{case.name}: CUDA graph selector result differs from eager"
        )
    eager_persistent_state.assert_matches(
        prepared.state_restore,
        main_k_cache=prepared.binding.main_k_cache,
        main_v_cache=prepared.binding.main_v_cache,
        label=f"{case.name}: CUDA graph persistent state",
    )
    del eager_persistent_state
    correctness["eager_graph_exact"] = True
    correctness["graph_persistent_state_exact"] = True
    correctness["graph_main_kv_read_only"] = True
    correctness["graph_replay_after_output_selector_scratch_poison"] = True
    correctness["graph_finite"] = bool(torch.all(torch.isfinite(graph_output)))
    correctness["graph_nonzero_elements"] = int(torch.count_nonzero(graph_output))
    if not correctness["graph_finite"] or not correctness["graph_nonzero_elements"]:
        raise BenchmarkFailure(f"{case.name}: invalid CUDA graph output")

    eager_samples = _time_eager(
        prepared.run,
        prepared.state_restore.restore,
        warmup=args.warmup,
        replays=args.eager_replays,
        l2_flush=l2_flush,
    )
    graph_samples = bench_cuda_graph(
        graph,
        replays=args.graph_replays,
        prepare=prepared.state_restore.restore,
        l2_flush=l2_flush,
    )
    prepared.state_restore.restore()
    torch.cuda.synchronize(device)
    prepared.state_restore.assert_restored()

    record = {
        "name": case.name,
        "profile": {
            "name": case.profile.name,
            "tensor_parallel_size": case.profile.tensor_parallel_size,
            "q_heads": case.profile.q_heads,
            "kv_heads": case.profile.kv_heads,
        },
        "rows": case.rows,
        "requests": case.request_count,
        "context": case.context,
        "active_sequence_length": case.active_sequence_length,
        "case_kind": case.kind,
        "setup": prepared.setup_metadata,
        "geometry": {
            "main_cache_layout": args.main_cache_layout,
            "kv_cache_dtype": args.kv_cache_dtype,
            "head_dim": HEAD_DIM,
            "main_page_size": case.main_page_size,
            "compressed_page_size": case.compressed_page_size,
            "index_heads": INDEX_HEADS,
            "index_kv_heads": INDEX_KV_HEADS,
            "index_head_dim": INDEX_HEAD_DIM,
            "index_rotary_dim": INDEX_ROTARY_DIM,
            "compress_ratio": COMPRESS_RATIO,
            "budget": BUDGET,
            "selection_width": BUDGET + COMPRESS_RATIO - 1,
            "position_axes": POSITION_AXES,
            "mrope_sections": list(MROPE_SECTIONS),
            "mrope_interleaved": True,
        },
        "capacity": {
            "main_pages_per_request": case.main_pages_per_request,
            "main_pages_active": case.main_pages_total,
            "main_pages_capacity": case.main_pages_capacity,
            "compressed_pages_per_request": case.compressed_pages_per_request,
            "compressed_pages_active": (
                case.request_count * case.compressed_pages_per_request
            ),
            "compressed_pages_capacity": case.compressed_pages_capacity,
            "cache_bytes": _cache_capacity_bytes(
                case,
                kv_cache_dtype=args.kv_cache_dtype,
            ),
            "allocated_binding_and_dynamic_storage_bytes": _binding_storage_bytes(
                prepared
            ),
        },
        "correctness": correctness,
        "timing": {
            "state_restore_outside_timed_run": True,
            "eager": {
                "samples_us": eager_samples,
                "summary": _summary(eager_samples),
            },
            "cuda_graph": {
                "replay_samples_us": graph_samples["replay_us"],
                "restore_samples_us": graph_samples["metadata_us"],
                "step_samples_us": graph_samples["step_us"],
                "replay_summary": _summary(graph_samples["replay_us"]),
                "restore_summary": _summary(graph_samples["metadata_us"]),
                "step_summary": _summary(graph_samples["step_us"]),
            },
        },
        "graph_contract": {
            "stable_bound_addresses": True,
            "addresses": graph_addresses,
            "replay_allocation_delta_bytes": replay_allocation_delta,
            "replay_after_output_selector_scratch_poison": True,
            "selected_position_poison": SELECTED_POSITION_POISON,
            "persistent_state_exact": True,
            "main_kv_read_only": True,
        },
    }
    prepared.close()
    return record


def _git_value(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[3],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _write_new_result(path: Path, result: dict[str, object]) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    temporary = target.parent / f".{target.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            raise FileExistsError(f"refusing to overwrite benchmark result: {target}")
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _validate_args(args)
        if args.output is not None and args.output.expanduser().exists():
            raise BenchmarkFailure(
                f"refusing to overwrite benchmark result: {args.output.expanduser()}"
            )
        device = torch.device(args.device) if args.device != "cuda" else require_sm120()
        if device.type != "cuda":
            raise BenchmarkFailure("QSA benchmarking requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        cases = _resolve_cases(args)
        l2_flush = make_l2_flush_fn(args.flush_l2, bytes_hint=args.l2_flush_bytes)
        properties = torch.cuda.get_device_properties(device)
        capability = torch.cuda.get_device_capability(device)
        result: dict[str, object] = {
            "kind": _RESULT_KIND,
            "schema_version": _RESULT_SCHEMA_VERSION,
            "complete": False,
            "provenance": {
                "command": shlex.join([sys.executable, *sys.argv]),
                "commit": _git_value("rev-parse", "HEAD"),
                "branch": _git_value("branch", "--show-current"),
                "git_dirty": bool(_git_value("status", "--porcelain")),
                "worktree": str(Path(__file__).resolve().parents[3]),
                "benchmark_source_sha256": _source_sha256(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "software": {
                "python": sys.version,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "triton": _package_version("triton"),
                "cutlass_dsl": _package_version("nvidia-cutlass-dsl"),
            },
            "hardware": {
                "device": str(device),
                "sm": f"{capability[0]}{capability[1]}",
                "total_memory_bytes": properties.total_memory,
            },
            "contract": {
                "model": "Qwen3.8 Flash Next",
                "api": ["Caps", "declaration", "PreparationSession", "Plan", "bind", "run"],
                "timed_operation": "bound qsa.run transaction",
                "setup_operation": "Caps -> declaration -> prepare -> bind",
                "main_kv_pool": (
                    "disjoint read-only physical pages per request with "
                    f"{args.main_cache_layout} K/V storage"
                ),
                "selector_state": "independent per request",
                "selector_dataset": (
                    "context-spanning rank-separated exact-oracle representatives"
                ),
                "default_profile": "tp2",
                "available_profiles": list(PROFILES),
                "model_max_context": MODEL_MAX_CONTEXT,
            },
            "parameters": {
                "profiles": list(args.profiles),
                "rows": list(args.rows),
                "prefill_rows": list(args.prefill_rows),
                "contexts": list(args.contexts),
                "main_cache_layout": args.main_cache_layout,
                "kv_cache_dtype": args.kv_cache_dtype,
                "contract_cases": args.contract_cases,
                "warmup": args.warmup,
                "eager_replays": args.eager_replays,
                "graph_replays": args.graph_replays,
                "seed": args.seed,
                "flush_l2": args.flush_l2,
                "l2_flush_bytes": (
                    resolve_l2_flush_bytes(args.l2_flush_bytes) if args.flush_l2 else 0
                ),
                "l2_flush_order": "flush, restore mutable state, timed QSA run",
            },
            "cases": [],
        }

        print(
            f"device={properties.name} sm={capability[0]}{capability[1]} "
            f"cases={len(cases)} page={MAIN_PAGE_SIZE} "
            f"selector={INDEX_HEADS}x{INDEX_HEAD_DIM} "
            f"compress={COMPRESS_RATIO} budget={BUDGET}"
        )
        print("case                         eager_us  graph_us  graph_step_us  max_abs")
        for case_index, case in enumerate(cases):
            record = _run_case(
                case,
                args=args,
                device=device,
                l2_flush=l2_flush,
                case_index=case_index,
            )
            result["cases"].append(record)
            eager_us = record["timing"]["eager"]["summary"]["median_us"]
            graph_us = record["timing"]["cuda_graph"]["replay_summary"]["median_us"]
            step_us = record["timing"]["cuda_graph"]["step_summary"]["median_us"]
            max_abs = record["correctness"]["sparse_gqa_max_abs"]
            print(
                f"{case.name:<28} {eager_us:>9.3f}  {graph_us:>8.3f}  "
                f"{step_us:>13.3f}  {max_abs:.6f}"
            )
            if args.print_raw_samples:
                print(json.dumps(record, sort_keys=True))
            del record
            gc.collect()
            torch.cuda.empty_cache()

        result["complete"] = True
        if args.output is not None:
            _write_new_result(args.output, result)
            print(f"wrote {args.output.expanduser().resolve()}")
        return 0
    except (BenchmarkFailure, FileExistsError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    raise SystemExit(main())
