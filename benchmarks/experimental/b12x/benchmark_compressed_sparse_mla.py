#!/usr/bin/env python3
"""Benchmark native DeepSeek-V4 compressed sparse MLA layouts."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import math
import pathlib
import statistics
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch

from b12x.attention._shared.mla.compressed_reference import (
    COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN,
    COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE,
    COMPRESSED_SPARSE_MLA_C4_PAGE_SIZE,
    COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE,
    COMPRESSED_SPARSE_MLA_HEAD_DIM,
    COMPRESSED_SPARSE_MLA_NOPE_DIM,
    COMPRESSED_SPARSE_MLA_ROPE_DIM,
    COMPRESSED_SPARSE_MLA_SWA_TOKENS,
    compressed_sparse_mla_page_nbytes,
    compressed_sparse_mla_reference,
    pack_compressed_sparse_mla_kv_cache_reference,
)
from b12x.attention._shared.mla.api import clear_mla_caches
from b12x.attention import compressed_sparse_mla as mla
from b12x.preparation import PreparationSession
from benchmarks.experimental.b12x.attention_preparation import prepare_compressed
from b12x.attention._shared.mla.compressed_config import (
    compressed_sparse_mla_split_chunks_for_contract,
)
from b12x.attention.compressed_sparse_mla._scratch import (
    B12XCompressedSparseMLAScratchCaps,
)

from benchmarks.experimental.b12x.common import (
    bench_cuda_graph,
    capture_cuda_graph,
    make_l2_flush_fn,
    require_sm120,
    resolve_l2_flush_bytes,
)


_SM_SCALE = 1.0 / math.sqrt(COMPRESSED_SPARSE_MLA_HEAD_DIM)
_ALGORITHM_COS_TOL = 0.995
_DECODE_TARGET_US = 25.0
_PREFILL4096_TARGET_US = 2_000.0
_PAGE_INDEX_ALIGNMENT = 64
_DEFAULT_NUM_Q_HEADS = 32
_DEFAULT_INDEX_TOPK = 512
_DECODE_SPLIT_TILE = 64
_C128_COMPRESSION_RATIO = 128

DEFAULT_DSV4_HF_REPO_ID = "deepseek-ai/DeepSeek-V4-Flash-DSpark"
VLLM_DSV4_TRACE_PRESET = "vllm-dsv4-trace"
VLLM_DSV4_TRACE_SWA_PAGE_SIZE = 64
# The captured vLLM run packs the DSV4 cache groups into one allocation.  Its
# 22 x 37,440B, 21 x 8,640B, and 20 x 1,728B slots make each physical block
# 1,039,680B.  The trace allocation contains 7,792 blocks; consequently a
# 37,440B layer view spans exactly 8,100,184,320B, matching the Cute tensor
# extent encoded in the captured B12X kernel name.
VLLM_DSV4_TRACE_CACHE_PAGE_STRIDE_BYTES = 1_039_680
VLLM_DSV4_TRACE_CACHE_NUM_PAGES = 7_792
VLLM_DSV4_TRACE_CACHE_VIEW_SPAN_BYTES = 8_100_184_320


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    rows: int
    swa_width: int
    indexed_width: int
    indexed_page_size: int | None

    @property
    def topk(self) -> int:
        return self.swa_width + self.indexed_width


@dataclass(frozen=True)
class Sanity:
    max_abs: float
    rmse: float
    cos: float


@dataclass(frozen=True)
class CaseReport:
    case: BenchmarkCase
    replay_us: float
    p90_replay_us: float
    sanity_algorithm: Sanity | None
    replay_samples_us: tuple[float, ...] = ()
    split_chunks: int | None = None
    swa_valid: int | None = None
    indexed_valid: int | None = None



@dataclass(frozen=True)
class TargetSummary:
    rows1_geo_us: float
    rows4096_geo_us: float
    rows1_target_ratio: float
    rows4096_target_ratio: float
    avg_target_ratio: float


@dataclass(frozen=True)
class DSV4CompressedSparseMLAProfile:
    swa_width: int
    c4_indexed_width: int
    c128_indexed_width: int
    selected_widths: tuple[int, ...]


@dataclass(frozen=True)
class CacheViews:
    b12x: torch.Tensor


@dataclass(frozen=True)
class TraceWeightedSummary:
    b12x_total_us: float
    b12x_avg_us: float
    layer_count: int


class BenchmarkFailure(RuntimeError):
    pass


def _align_up(value: int, alignment: int = _PAGE_INDEX_ALIGNMENT) -> int:
    if value < 0:
        raise ValueError(f"value must be non-negative, got {value}")
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((value + alignment - 1) // alignment) * alignment


def _load_model_config(path: pathlib.Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"model config must be a JSON object: {path}")
    return loaded


def _resolve_cached_hf_config(
    repo_id: str = DEFAULT_DSV4_HF_REPO_ID,
    *,
    cache_root: pathlib.Path | None = None,
) -> pathlib.Path:
    """Resolve config.json from the local HF cache without downloading it."""

    from huggingface_hub import try_to_load_from_cache

    cached_config = try_to_load_from_cache(
        repo_id=repo_id,
        filename="config.json",
        cache_dir=cache_root,
        revision="main",
    )
    if isinstance(cached_config, str):
        return pathlib.Path(cached_config)
    cache_desc = (
        "the configured Hugging Face cache" if cache_root is None else str(cache_root)
    )
    raise SystemExit(
        f"cached Hugging Face config not found for {repo_id!r} in {cache_desc}; "
        "populate the cache or pass --model-config /path/to/config.json"
    )




def _trace_layer_weights(config: dict[str, object]) -> dict[str, int]:
    """Map the real model layers to the three compressed attention regimes."""

    num_layers = int(config["num_hidden_layers"])
    ratios = tuple(int(value) for value in config["compress_ratios"])  # type: ignore[index]
    if len(ratios) < num_layers:
        raise ValueError(
            f"compress_ratios has {len(ratios)} entries for {num_layers} model layers"
        )
    counts = Counter(ratios[:num_layers])
    unexpected = sorted(ratio for ratio in counts if ratio not in (0, 4, 128))
    if unexpected:
        raise ValueError(
            f"unsupported DSV4 compression ratios in model layers: {unexpected}"
        )
    return {
        "swa": counts[0],
        "swa-c4": counts[4],
        "swa-c128": counts[128],
    }


def _derive_dsv4_compressed_sparse_mla_profile(
    config: dict[str, object],
    *,
    full_token_capacity: int | None = None,
    c128_pool_size: int | None = None,
) -> DSV4CompressedSparseMLAProfile:
    sliding_window = int(config.get("sliding_window", COMPRESSED_SPARSE_MLA_SWA_TOKENS))
    c4_topk = int(config.get("index_topk", _DEFAULT_INDEX_TOPK))
    max_positions = int(config.get("max_position_embeddings", 0))
    compress_ratios_raw = config.get("compress_ratios", ())
    if compress_ratios_raw is None:
        compress_ratios_raw = ()
    compress_ratios = tuple(int(value) for value in compress_ratios_raw)  # type: ignore[arg-type]

    uses_c4 = 4 in compress_ratios
    uses_c128 = 128 in compress_ratios
    swa_width = _align_up(sliding_window)
    c4_indexed_width = _align_up(c4_topk) if uses_c4 else 0

    c128_source_tokens = full_token_capacity
    if c128_source_tokens is None:
        if c128_pool_size is not None:
            c128_source_tokens = c128_pool_size * COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE
        else:
            c128_source_tokens = max_positions

    c128_width = 0
    if uses_c128 and c128_source_tokens:
        # Selected C128 slots are source-token positions compressed by 128.
        # The resulting cache then packs two compressed slots per physical
        # page (256 / 128); dividing source capacity by that page size would
        # overstate the cache by 64x.
        c128_width = (
            int(c128_source_tokens) + _C128_COMPRESSION_RATIO - 1
        ) // _C128_COMPRESSION_RATIO
        if c128_pool_size is not None:
            c128_width = min(c128_width, int(c128_pool_size))
    c128_indexed_width = _align_up(c128_width) if c128_width else 0

    selected_widths = {swa_width}
    if c4_indexed_width:
        selected_widths.add(swa_width + c4_indexed_width)
    if c128_indexed_width:
        selected_widths.add(swa_width + c128_indexed_width)

    return DSV4CompressedSparseMLAProfile(
        swa_width=swa_width,
        c4_indexed_width=c4_indexed_width,
        c128_indexed_width=c128_indexed_width,
        selected_widths=tuple(sorted(selected_widths)),
    )


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"all values must be positive, got {raw!r}")
    return values


def _parse_cases(
    raw: str,
    rows: list[int],
    *,
    c4_indexed_width: int = _DEFAULT_INDEX_TOPK,
    c128_indexed_width: int = _DEFAULT_INDEX_TOPK,
) -> list[BenchmarkCase]:
    names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not names or names == ["all"]:
        names = ["swa", "c4", "c128", "swa-c4", "swa-c128"]
    elif names == ["model"]:
        names = ["swa", "swa-c4", "swa-c128"]

    cases: list[BenchmarkCase] = []
    for row_count in rows:
        for name in names:
            if name == "swa":
                cases.append(
                    BenchmarkCase(
                        name=name,
                        rows=row_count,
                        swa_width=COMPRESSED_SPARSE_MLA_SWA_TOKENS,
                        indexed_width=0,
                        indexed_page_size=None,
                    )
                )
            elif name == "c4":
                cases.append(
                    BenchmarkCase(
                        name=name,
                        rows=row_count,
                        swa_width=0,
                        indexed_width=c4_indexed_width,
                        indexed_page_size=COMPRESSED_SPARSE_MLA_C4_PAGE_SIZE,
                    )
                )
            elif name == "c128":
                cases.append(
                    BenchmarkCase(
                        name=name,
                        rows=row_count,
                        swa_width=0,
                        indexed_width=c128_indexed_width,
                        indexed_page_size=COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE,
                    )
                )
            elif name == "swa-c4":
                cases.append(
                    BenchmarkCase(
                        name=name,
                        rows=row_count,
                        swa_width=COMPRESSED_SPARSE_MLA_SWA_TOKENS,
                        indexed_width=c4_indexed_width,
                        indexed_page_size=COMPRESSED_SPARSE_MLA_C4_PAGE_SIZE,
                    )
                )
            elif name == "swa-c128":
                cases.append(
                    BenchmarkCase(
                        name=name,
                        rows=row_count,
                        swa_width=COMPRESSED_SPARSE_MLA_SWA_TOKENS,
                        indexed_width=c128_indexed_width,
                        indexed_page_size=COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE,
                    )
                )
            else:
                raise argparse.ArgumentTypeError(
                    "cases must be one of all,model,swa,c4,c128,swa-c4,swa-c128; "
                    f"got {name!r}"
                )
    return cases


def _resolve_case_widths(args: argparse.Namespace) -> tuple[int, int]:
    profile: DSV4CompressedSparseMLAProfile | None = None
    if args.model_config is not None:
        profile = _derive_dsv4_compressed_sparse_mla_profile(
            _load_model_config(args.model_config),
            full_token_capacity=args.full_token_capacity,
            c128_pool_size=args.c128_pool_size,
        )

    c4_indexed_width = args.c4_indexed_width
    if c4_indexed_width is None:
        c4_indexed_width = (
            profile.c4_indexed_width
            if profile is not None and profile.c4_indexed_width
            else _DEFAULT_INDEX_TOPK
        )

    c128_indexed_width = args.c128_indexed_width
    if c128_indexed_width is None:
        c128_indexed_width = (
            profile.c128_indexed_width
            if profile is not None and profile.c128_indexed_width
            else _DEFAULT_INDEX_TOPK
        )

    if c4_indexed_width <= 0 or c128_indexed_width <= 0:
        raise ValueError(
            "--c4-indexed-width and --c128-indexed-width must be positive after model derivation"
        )
    return int(c4_indexed_width), int(c128_indexed_width)


def _runtime_valid_widths(
    case: BenchmarkCase,
    *,
    context_length: int | None,
) -> tuple[int, int]:
    """Return replay-time SWA/indexed lengths for a captured-capacity case."""

    if context_length is None:
        return case.swa_width, case.indexed_width
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")

    swa_valid = min(case.swa_width, context_length)
    indexed_valid = 0
    if case.indexed_width:
        if case.indexed_page_size is None:
            raise ValueError("indexed_width requires indexed_page_size")
        if COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE % case.indexed_page_size:
            raise ValueError(
                "indexed page size must divide the DSV4 page size: "
                f"{case.indexed_page_size} vs {COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE}"
            )
        compression_ratio = (
            COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE // case.indexed_page_size
        )
        indexed_valid = min(case.indexed_width, context_length // compression_ratio)
    return swa_valid, indexed_valid


def _planned_split_chunks(
    case: BenchmarkCase,
    *,
    production_decode_cap: bool = False,
) -> int:
    max_chunks = None
    if production_decode_cap:
        max_chunks = max(1, math.ceil(case.topk / _DECODE_SPLIT_TILE))
    return compressed_sparse_mla_split_chunks_for_contract(
        rows=case.rows,
        width=max(1, case.topk),
        max_chunks=max_chunks,
    )


def _make_q(
    *, rows: int, num_q_heads: int, seed: int, device: torch.device
) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    q = torch.randn(
        (rows, num_q_heads, COMPRESSED_SPARSE_MLA_HEAD_DIM),
        generator=gen,
        dtype=torch.float32,
        device=device,
    )
    return (q * 0.04).to(dtype=torch.bfloat16)


def _make_compressed_cache(
    *,
    tokens: int,
    page_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    k_nope = (
        torch.randn(
            (tokens, COMPRESSED_SPARSE_MLA_NOPE_DIM),
            generator=gen,
            dtype=torch.float32,
            device=device,
        )
        * 0.05
    )
    k_rope = (
        torch.randn(
            (tokens, COMPRESSED_SPARSE_MLA_ROPE_DIM),
            generator=gen,
            dtype=torch.float32,
            device=device,
        )
        * 0.05
    )
    return pack_compressed_sparse_mla_kv_cache_reference(
        k_nope,
        k_rope.to(dtype=torch.bfloat16),
        page_size=page_size,
    )


def _make_cache_views(
    packed_cache: torch.Tensor,
    *,
    page_size: int,
    block_stride_bytes: int,
    num_pages: int | None = None,
    backing: torch.Tensor | None = None,
    byte_offset: int = 0,
) -> CacheViews:
    """Expose packed KV storage with the serving cache's physical page stride.

    KV cache groups retain the aggregate all-layer byte stride between pages.
    """

    if packed_cache.dtype != torch.uint8 or packed_cache.ndim != 2:
        raise ValueError("packed compressed cache must be rank-2 uint8")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if block_stride_bytes < 0:
        raise ValueError(
            f"block_stride_bytes must be non-negative, got {block_stride_bytes}"
        )

    active_pages, page_nbytes = (int(value) for value in packed_cache.shape)
    pages = active_pages if num_pages is None else int(num_pages)
    if pages <= 0:
        raise ValueError(f"num_pages must be positive, got {pages}")
    if active_pages > pages:
        raise ValueError(
            f"active cache has {active_pages} pages but pool only has {pages}"
        )
    if byte_offset < 0:
        raise ValueError(f"byte_offset must be non-negative, got {byte_offset}")
    nominal_page_bytes = int(page_size) * COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN
    if page_nbytes < nominal_page_bytes:
        raise ValueError(
            f"packed page has {page_nbytes} bytes, smaller than nominal "
            f"{nominal_page_bytes} bytes"
        )
    page_stride = page_nbytes if block_stride_bytes == 0 else int(block_stride_bytes)
    if page_stride < page_nbytes:
        raise ValueError(
            f"block stride {page_stride} is smaller than packed page {page_nbytes}"
        )

    if (
        backing is None
        and byte_offset == 0
        and pages == active_pages
        and page_stride == page_nbytes
    ):
        b12x_view = packed_cache
    else:
        required_nbytes = byte_offset + (pages - 1) * page_stride + page_nbytes
        if backing is None:
            backing = torch.empty(
                required_nbytes,
                dtype=torch.uint8,
                device=packed_cache.device,
            )
        elif (
            backing.dtype != torch.uint8
            or backing.ndim != 1
            or backing.device != packed_cache.device
        ):
            raise ValueError(
                "cache backing must be a rank-1 uint8 tensor on the cache device"
            )
        if int(backing.numel()) < required_nbytes:
            raise ValueError(
                f"cache backing has {backing.numel()} bytes, needs {required_nbytes}"
            )
        b12x_view = torch.as_strided(
            backing,
            size=(pages, page_nbytes),
            stride=(page_stride, 1),
            storage_offset=byte_offset,
        )
        b12x_view[:active_pages].copy_(packed_cache)

    return CacheViews(b12x=b12x_view, )


def _make_indices(
    *,
    rows: int,
    width: int,
    tokens: int,
    device: torch.device,
) -> torch.Tensor:
    if width == 0:
        return torch.empty((rows, 0), dtype=torch.int32, device=device)
    if tokens < width:
        raise ValueError(f"tokens {tokens} must be at least width {width}")
    stride = max(1, tokens // max(1, rows))
    offsets = (torch.arange(rows, dtype=torch.int64, device=device) * stride)[:, None]
    cols = torch.arange(width, dtype=torch.int64, device=device)[None, :]
    return ((offsets + cols) % tokens).to(torch.int32)


def _indexed_cache_tokens(
    case: BenchmarkCase,
    *,
    shared_indexed_cache: bool,
    reuse_cache_pool: bool = False,
    cache_num_pages: int = 0,
) -> int:
    """Size the indexed cache for independent rows or one shared prefill pool."""

    if not case.indexed_width:
        return 0
    if shared_indexed_cache:
        return case.indexed_width
    if reuse_cache_pool:
        assert case.indexed_page_size is not None
        return max(case.indexed_width, cache_num_pages * case.indexed_page_size)
    return case.indexed_width * max(case.rows, 1)


def _swa_cache_tokens(
    case: BenchmarkCase,
    *,
    shared_indexed_cache: bool,
    reuse_cache_pool: bool,
    cache_num_pages: int,
    swa_page_size: int,
) -> int:
    if shared_indexed_cache:
        return max(case.swa_width, 1)
    if reuse_cache_pool:
        return max(case.swa_width, cache_num_pages * swa_page_size, 1)
    return max(case.swa_width * max(case.rows, 1), 1)


def _benchmark_workspace_mode(*, shared_indexed_cache: bool) -> str:
    """Select the serving workspace contract represented by the benchmark."""

    return "extend" if shared_indexed_cache else "decode"


def _sanity(actual: torch.Tensor, expected: torch.Tensor) -> Sanity:
    diff = actual.float() - expected.float()
    flat_actual = actual.float().reshape(-1)
    flat_expected = expected.float().reshape(-1)
    return Sanity(
        max_abs=diff.abs().max().item(),
        rmse=torch.sqrt(torch.mean(diff * diff)).item(),
        cos=torch.nn.functional.cosine_similarity(
            flat_actual, flat_expected, dim=0
        ).item(),
    )


def _check_algorithm_sanity(case: BenchmarkCase, sanity: Sanity) -> None:
    if not math.isfinite(sanity.cos) or sanity.cos < _ALGORITHM_COS_TOL:
        raise BenchmarkFailure(
            "compressed MLA algorithm cosine below threshold for "
            f"case={case.name} rows={case.rows}: "
            f"max_abs={sanity.max_abs:.6f} rmse={sanity.rmse:.6f} "
            f"cos={sanity.cos:.6f} threshold={_ALGORITHM_COS_TOL:.6f}"
        )


def _geomean(values: list[float]) -> float:
    if not values:
        raise ValueError("geomean requires at least one value")
    if any(value <= 0.0 for value in values):
        raise ValueError(f"geomean values must be positive, got {values}")
    return math.exp(statistics.mean(math.log(value) for value in values))


def _compute_target_summary(reports: list[CaseReport]) -> TargetSummary:
    by_rows: dict[int, list[float]] = {}
    for report in reports:
        by_rows.setdefault(report.case.rows, []).append(report.replay_us)

    missing = [rows for rows in (1, 4096) if rows not in by_rows]
    if missing:
        raise BenchmarkFailure(
            "compressed MLA target scoring requires rows=1 and rows=4096; "
            f"missing rows={','.join(str(row) for row in missing)}"
        )

    rows1_geo = _geomean(by_rows[1])
    rows4096_geo = _geomean(by_rows[4096])
    rows1_ratio = rows1_geo / _DECODE_TARGET_US
    rows4096_ratio = rows4096_geo / _PREFILL4096_TARGET_US
    return TargetSummary(
        rows1_geo_us=rows1_geo,
        rows4096_geo_us=rows4096_geo,
        rows1_target_ratio=rows1_ratio,
        rows4096_target_ratio=rows4096_ratio,
        avg_target_ratio=(rows1_ratio + rows4096_ratio) / 2.0,
    )


def _benchmark_case(
    case: BenchmarkCase,
    *,
    device: torch.device,
    seed: int,
    warmup: int,
    replays: int,
    l2_flush,
    verify: bool,
    num_q_heads: int,
    swa_page_size: int,
    cache_page_stride_bytes: int,
    cache_num_pages: int,
    production_decode_cap: bool,
    use_attn_sink: bool,
    context_length: int | None,
    shared_indexed_cache: bool,
    reuse_cache_pool: bool,
) -> CaseReport:
    clear_mla_caches()
    q = _make_q(rows=case.rows, num_q_heads=num_q_heads, seed=seed, device=device)
    attn_sink = None
    if use_attn_sink:
        attn_sink = torch.linspace(
            -0.25,
            0.25,
            num_q_heads,
            dtype=torch.float32,
            device=device,
        )

    swa_tokens = _swa_cache_tokens(
        case,
        shared_indexed_cache=shared_indexed_cache,
        reuse_cache_pool=reuse_cache_pool,
        cache_num_pages=cache_num_pages,
        swa_page_size=swa_page_size,
    )
    swa_packed = _make_compressed_cache(
        tokens=swa_tokens,
        page_size=swa_page_size,
        seed=seed + 1,
        device=device,
    )
    indexed_packed: torch.Tensor | None = None
    if case.indexed_width:
        assert case.indexed_page_size is not None
        indexed_tokens = _indexed_cache_tokens(
            case,
            shared_indexed_cache=shared_indexed_cache,
            reuse_cache_pool=reuse_cache_pool,
            cache_num_pages=cache_num_pages,
        )
        indexed_packed = _make_compressed_cache(
            tokens=indexed_tokens,
            page_size=case.indexed_page_size,
            seed=seed + 2,
            device=device,
        )

    cache_backing: torch.Tensor | None = None
    indexed_byte_offset = 0
    if cache_num_pages:
        if cache_page_stride_bytes <= 0:
            raise ValueError("cache_num_pages requires a positive packed-cache stride")
        indexed_byte_offset = compressed_sparse_mla_page_nbytes(swa_page_size)
        largest_payload_end = indexed_byte_offset
        if case.indexed_page_size is not None:
            largest_payload_end += compressed_sparse_mla_page_nbytes(
                case.indexed_page_size
            )
        if largest_payload_end > cache_page_stride_bytes:
            raise ValueError(
                "SWA and indexed cache payloads do not fit in one packed block: "
                f"need {largest_payload_end}, stride is {cache_page_stride_bytes}"
            )
        cache_backing = torch.empty(
            cache_num_pages * cache_page_stride_bytes,
            dtype=torch.uint8,
            device=device,
        )

    swa_cache = _make_cache_views(
        swa_packed,
        page_size=swa_page_size,
        block_stride_bytes=cache_page_stride_bytes,
        num_pages=cache_num_pages or None,
        backing=cache_backing,
    )
    swa_indices = _make_indices(
        rows=case.rows, width=case.swa_width, tokens=swa_tokens, device=device
    )
    swa_valid, indexed_valid = _runtime_valid_widths(
        case, context_length=context_length
    )
    swa_lengths = torch.full((case.rows,), swa_valid, dtype=torch.int32, device=device)

    indexed_cache: CacheViews | None = None
    indexed_indices: torch.Tensor | None = None
    indexed_lengths: torch.Tensor | None = None
    if case.indexed_width:
        assert case.indexed_page_size is not None
        indexed_tokens = _indexed_cache_tokens(
            case,
            shared_indexed_cache=shared_indexed_cache,
            reuse_cache_pool=reuse_cache_pool,
            cache_num_pages=cache_num_pages,
        )
        assert indexed_packed is not None
        indexed_cache = _make_cache_views(
            indexed_packed,
            page_size=case.indexed_page_size,
            block_stride_bytes=cache_page_stride_bytes,
            num_pages=cache_num_pages or None,
            backing=cache_backing,
            byte_offset=indexed_byte_offset,
        )
        indexed_indices = _make_indices(
            rows=case.rows,
            width=case.indexed_width,
            tokens=indexed_tokens,
            device=device,
        )
        indexed_lengths = torch.full(
            (case.rows,), indexed_valid, dtype=torch.int32, device=device
        )

    split_chunks = _planned_split_chunks(case, production_decode_cap=production_decode_cap)
    output = torch.empty_like(q)
    run_args = dict(
        swa_k_cache=swa_cache.b12x, swa_page_size=swa_page_size,
        indexed_k_cache=indexed_cache.b12x if indexed_cache is not None else None,
        indexed_page_size=case.indexed_page_size, attn_sink=attn_sink,
        sm_scale=_SM_SCALE, expected_num_q_heads=num_q_heads, out=output,
    )
    session = PreparationSession(device=device, autotune=False, compile_workers=2)
    plan, binding = prepare_compressed(
        session,
        B12XCompressedSparseMLAScratchCaps(
            device=device, num_q_heads=num_q_heads, max_q_rows=case.rows,
            max_width=max(1, case.topk), head_dim=COMPRESSED_SPARSE_MLA_HEAD_DIM,
            v_head_dim=COMPRESSED_SPARSE_MLA_HEAD_DIM, max_batch=case.rows,
            page_size=swa_page_size, swa_width=case.swa_width,
            indexed_width=case.indexed_width, indexed_page_size=case.indexed_page_size,
            max_chunks_per_row=split_chunks,
            mode=_benchmark_workspace_mode(shared_indexed_cache=shared_indexed_cache),
            use_cuda_graph=True,
        ),
        bind_args=dict(q=q, swa_indices=swa_indices, swa_lengths=swa_lengths,
                       indexed_indices=indexed_indices, indexed_lengths=indexed_lengths),
        run_args=run_args,
    )

    def run() -> torch.Tensor:
        return mla.run(binding=binding, **run_args)

    expected_algorithm: torch.Tensor | None = None
    if verify:
        expected_algorithm = compressed_sparse_mla_reference(
            q,
            swa_cache.b12x,
            swa_indices,
            swa_lengths,
            sm_scale=_SM_SCALE,
            attn_sink=attn_sink,
            extra_k_cache=indexed_cache.b12x if indexed_cache is not None else None,
            extra_indices=indexed_indices,
            extra_topk_lengths=indexed_lengths,
            swa_page_size=swa_page_size,
            extra_page_size=case.indexed_page_size,
        )

    session.freeze()
    graph = torch.cuda.CUDAGraph()
    try:
        graph = capture_cuda_graph(run, warmup=warmup)
        stats = bench_cuda_graph(graph, replays=replays, l2_flush=l2_flush)
        if output is None:
            raise RuntimeError("benchmark graph did not produce an output tensor")
    finally:
        torch.cuda.synchronize(device)
        graph.reset()
        session.close()

    replay_us = stats["replay_us"]
    sanity_algorithm: Sanity | None = None
    if expected_algorithm is not None:
        sanity_algorithm = _sanity(output, expected_algorithm)
        _check_algorithm_sanity(case, sanity_algorithm)
    if not bool(torch.isfinite(output.float()).all().item()):
        raise BenchmarkFailure(f"non-finite B12X output for case={case.name}")
    if not bool(torch.count_nonzero(output).item()):
        raise BenchmarkFailure(f"all-zero B12X output for case={case.name}")


    gc.collect()
    torch.cuda.empty_cache()
    return CaseReport(
        case=case,
        replay_us=statistics.median(replay_us),
        p90_replay_us=statistics.quantiles(replay_us, n=10)[8]
        if len(replay_us) >= 10
        else max(replay_us),
        sanity_algorithm=sanity_algorithm,
        replay_samples_us=tuple(replay_us),





        split_chunks=split_chunks,
        swa_valid=swa_valid,
        indexed_valid=indexed_valid,
    )


def collect_case_reports(
    args: argparse.Namespace, *, device: torch.device | None = None
) -> list[CaseReport]:
    if device is None:
        device = require_sm120()
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)
    l2_flush = make_l2_flush_fn(args.flush_l2, l2_flush_bytes)
    c4_indexed_width, c128_indexed_width = _resolve_case_widths(args)

    reports: list[CaseReport] = []
    for case_idx, case in enumerate(
        _parse_cases(
            args.cases,
            args.rows,
            c4_indexed_width=c4_indexed_width,
            c128_indexed_width=c128_indexed_width,
        )
    ):
        reports.append(
            _benchmark_case(
                case,
                device=device,
                seed=args.seed + case_idx * 17,
                warmup=args.warmup,
                replays=args.replays,
                l2_flush=l2_flush,
                verify=not args.skip_verify,
                num_q_heads=args.num_q_heads,
                swa_page_size=args.swa_page_size,
                cache_page_stride_bytes=args.cache_page_stride_bytes,
                cache_num_pages=args.cache_num_pages,
                production_decode_cap=args.production_decode_cap,
                use_attn_sink=args.attn_sink,
                    context_length=args.context_length,
                shared_indexed_cache=args.shared_indexed_cache,
                reuse_cache_pool=args.reuse_cache_pool,
            )
        )
    return reports


def _render_report(report: CaseReport) -> str:
    indexed_page = (
        report.case.indexed_page_size
        if report.case.indexed_page_size is not None
        else 0
    )
    parts = [
        f"compressed-mla-native case={report.case.name:8s}",
        f"rows={report.case.rows:2d}",
        f"swa={report.case.swa_width:3d}",
        f"indexed={report.case.indexed_width:3d}",
        f"indexed_page={indexed_page:3d}",
        f"topk={report.case.topk:3d}",
        f"valid={int(report.swa_valid or 0):3d}+{int(report.indexed_valid or 0):4d}",
        f"chunks={(report.split_chunks if report.split_chunks is not None else _planned_split_chunks(report.case)):3d}",
        f"replay={report.replay_us:8.2f} us",
        f"p90={report.p90_replay_us:8.2f} us",
    ]
    if report.sanity_algorithm is not None:
        parts.append(
            "algorithm="
            f"max_abs:{report.sanity_algorithm.max_abs:.4f},"
            f"rmse:{report.sanity_algorithm.rmse:.5f},"
            f"cos:{report.sanity_algorithm.cos:.6f}"
        )
    return " | ".join(parts)


def _compute_trace_weighted_summary(
    reports: list[CaseReport],
    layer_weights: dict[str, int],
) -> TraceWeightedSummary:
    by_name = {report.case.name: report for report in reports}
    missing = sorted(
        name for name, weight in layer_weights.items() if weight and name not in by_name
    )
    if missing:
        raise BenchmarkFailure(f"missing trace cases for weighted summary: {missing}")

    layer_count = sum(layer_weights.values())
    b12x_total = sum(
        layer_weights[name] * by_name[name].replay_us for name in layer_weights
    )
    return TraceWeightedSummary(
        b12x_total_us=b12x_total,

        b12x_avg_us=b12x_total / layer_count,

        layer_count=layer_count,
    )


def _render_trace_weighted_summary(
    summary: TraceWeightedSummary,
    layer_weights: dict[str, int],
) -> str:
    return " | ".join(
        [
            "Trace-weighted",
            f"layers={summary.layer_count}",
            "weights="
            + ",".join(f"{name}:{weight}" for name, weight in layer_weights.items()),
            f"b12x_total={summary.b12x_total_us:.2f} us",
            f"b12x_avg={summary.b12x_avg_us:.2f} us",
        ]
    )


def _render_summary(reports: list[CaseReport], summary: TargetSummary) -> str:
    return " | ".join(
        [
            f"Summary | cases={len(reports)}",
            f"rows1_geo={summary.rows1_geo_us:.2f} us",
            f"rows1_target_ratio={summary.rows1_target_ratio:.4f}",
            f"rows4096_geo={summary.rows4096_geo_us:.2f} us",
            f"rows4096_target_ratio={summary.rows4096_target_ratio:.4f}",
            f"avg_target_ratio={summary.avg_target_ratio:.4f}",
        ]
    )


def _apply_benchmark_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset == "none":
        return args
    if args.preset != VLLM_DSV4_TRACE_PRESET:
        raise ValueError(f"unknown preset {args.preset!r}")

    args.cases = "model"
    args.rows = [1]
    args.num_q_heads = _DEFAULT_NUM_Q_HEADS
    args.swa_page_size = VLLM_DSV4_TRACE_SWA_PAGE_SIZE
    args.cache_page_stride_bytes = VLLM_DSV4_TRACE_CACHE_PAGE_STRIDE_BYTES
    args.cache_num_pages = VLLM_DSV4_TRACE_CACHE_NUM_PAGES
    args.production_decode_cap = True
    args.attn_sink = True
    return args


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("none", VLLM_DSV4_TRACE_PRESET),
        default="none",
        help=(
            "vllm-dsv4-trace reproduces the captured TP2 C1, C4, and C128 "
            "decode contracts, packed cache stride, sink, split caps, and "
            "b12x reference checks"
        ),
    )
    parser.add_argument(
        "--cases",
        default="all",
        help="comma-separated cases: all,model,swa,c4,c128,swa-c4,swa-c128",
    )
    parser.add_argument(
        "--rows", type=_parse_csv_ints, default=_parse_csv_ints("1,4096")
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--seed", type=int, default=91_000)
    parser.add_argument(
        "--model-config",
        type=pathlib.Path,
        default=None,
        help=(
            "DeepSeek V4 config.json; with --cases model this derives the real "
            "SWA, SWA+C4, and SWA+C128 selected widths"
        ),
    )
    parser.add_argument(
        "--full-token-capacity",
        type=int,
        default=None,
        help=(
            "runtime full-token KV capacity used to derive C128 indexed width; "
            "matches the SGLang DSV4 pool log full=..."
        ),
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help=(
            "replay-time source-token context length; keeps captured widths at "
            "capacity while setting SWA/C4/C128 valid lengths as vLLM does"
        ),
    )
    parser.add_argument(
        "--shared-indexed-cache",
        action="store_true",
        help=(
            "share one indexed cache pool across all query rows, matching "
            "single-sequence chunked prefill and select the extend workspace "
            "contract instead of modeling each row as an independent decode "
            "sequence"
        ),
    )
    parser.add_argument(
        "--reuse-cache-pool",
        action="store_true",
        help=(
            "address the finite packed cache pool across decode rows while "
            "preserving the decode workspace contract"
        ),
    )
    parser.add_argument(
        "--c128-pool-size",
        type=int,
        default=None,
        help="optional runtime C128 pool size cap used by SGLang when deriving the C128 width",
    )
    parser.add_argument(
        "--c4-indexed-width",
        type=int,
        default=None,
        help="override indexed-token width for C4 cases; default comes from config or model top-k",
    )
    parser.add_argument(
        "--c128-indexed-width",
        type=int,
        default=None,
        help=(
            "override indexed-token width for C128 cases; by default this comes from "
            "config plus the runtime full-token/C128 pool capacity"
        ),
    )
    parser.add_argument("--flush-l2", action="store_true", default=True)
    parser.add_argument("--no-flush-l2", action="store_false", dest="flush_l2")
    parser.add_argument(
        "--l2-flush-bytes",
        type=int,
        default=0,
        help="L2 eviction size in bytes; default is 2x detected L2 capacity.",
    )
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument(
        "--verify-algorithm",
        action="store_true",
        help="deprecated; compressed-layout algorithm verification is the default unless --skip-verify is set",
    )
    parser.add_argument(
        "--num-q-heads",
        type=int,
        default=_DEFAULT_NUM_Q_HEADS,
        help="local query-head count to benchmark; default is the synthetic 32-head profile",
    )
    parser.add_argument(
        "--swa-page-size",
        type=int,
        default=COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE,
        help="physical SWA page size; vLLM's DeepSeek-V4 cache uses 64",
    )
    parser.add_argument(
        "--cache-page-stride-bytes",
        type=int,
        default=0,
        help=(
            "physical byte stride between pages in a packed vLLM cache view; "
            "0 uses contiguous per-layer pages"
        ),
    )
    parser.add_argument(
        "--cache-num-pages",
        type=int,
        default=0,
        help=(
            "physical pages in the packed cache pool; 0 allocates only pages "
            "addressed by the synthetic case"
        ),
    )
    parser.add_argument(
        "--production-decode-cap",
        action="store_true",
        help="cap split scratch at ceil(total_width/64), matching the vLLM backend",
    )
    parser.add_argument(
        "--attn-sink",
        action="store_true",
        help="include the per-head FP32 attention sink used by the traced model",
    )
    parser.add_argument(
        "--print-raw-samples",
        action="store_true",
        help="print every b12x timing sample",
    )
    return _apply_benchmark_preset(parser.parse_args(argv))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.warmup <= 0 or args.replays <= 0:
        raise SystemExit("--warmup and --replays must be positive")
    if args.full_token_capacity is not None and args.full_token_capacity <= 0:
        raise SystemExit("--full-token-capacity must be positive")
    if args.context_length is not None and args.context_length <= 0:
        raise SystemExit("--context-length must be positive")
    if args.c128_pool_size is not None and args.c128_pool_size <= 0:
        raise SystemExit("--c128-pool-size must be positive")
    if args.c4_indexed_width is not None and args.c4_indexed_width <= 0:
        raise SystemExit("--c4-indexed-width must be positive")
    if args.c128_indexed_width is not None and args.c128_indexed_width <= 0:
        raise SystemExit("--c128-indexed-width must be positive")
    if args.num_q_heads <= 0:
        raise SystemExit("--num-q-heads must be positive")
    if args.swa_page_size <= 0:
        raise SystemExit("--swa-page-size must be positive")
    if args.cache_page_stride_bytes < 0:
        raise SystemExit("--cache-page-stride-bytes must be non-negative")
    if args.cache_num_pages < 0:
        raise SystemExit("--cache-num-pages must be non-negative")

    if args.preset == VLLM_DSV4_TRACE_PRESET:
        if args.model_config is None:
            args.model_config = _resolve_cached_hf_config()
        trace_config = _load_model_config(args.model_config)
        args.full_token_capacity = int(trace_config["max_position_embeddings"])
        layer_weights = _trace_layer_weights(trace_config)
    else:
        trace_config = None
        layer_weights = None

    try:
        _resolve_case_widths(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    try:
        reports = collect_case_reports(args)
    except BenchmarkFailure as exc:
        print(str(exc), file=sys.stderr)
        return 1

    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)
    flush_desc = (
        f"on ({l2_flush_bytes / (1 << 20):.1f} MiB per replay)"
        if args.flush_l2
        else "off"
    )
    print(f"L2 flush: {flush_desc}")
    if trace_config is not None:
        print(
            "DSV4 trace contract: "
            f"config={args.model_config} rows=1 local_heads={args.num_q_heads} "
            f"swa_page={args.swa_page_size} "
            f"cache_page_stride={args.cache_page_stride_bytes} "
            f"cache_num_pages={args.cache_num_pages} "
            f"cache_allocation={args.cache_num_pages * args.cache_page_stride_bytes} "
            f"max_positions={trace_config['max_position_embeddings']} "
            f"context_length={args.context_length or 'capacity'}"
        )
    for report in reports:
        print(_render_report(report))
        if args.print_raw_samples:
            print(
                f"raw case={report.case.name} backend=b12x us="
                + ",".join(f"{sample:.3f}" for sample in report.replay_samples_us)
            )
    if layer_weights is not None:
        weighted = _compute_trace_weighted_summary(reports, layer_weights)
        print(_render_trace_weighted_summary(weighted, layer_weights))
        return 0
    try:
        summary = _compute_target_summary(reports)
    except BenchmarkFailure as exc:
        print(f"Summary skipped: {exc}")
    else:
        print(_render_summary(reports, summary))
    return 0


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    raise SystemExit(main())
