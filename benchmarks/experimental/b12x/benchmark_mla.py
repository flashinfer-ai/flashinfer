#!/usr/bin/env python3
"""Benchmark GLM MLA and traced DeepSeek-V4 compressed-MLA serving shapes."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch
import triton
import triton.language as tl

from b12x.attention._shared.workspace import (
    default_sparse_mla_split_decode_config_for_width,
)
from b12x.attention.dsa_indexer.reference import (
    contiguous_logits_reference,
    pack_index_k_cache_reference,
    paged_decode_logits_reference,
)
from b12x.attention._shared.mla.reference import (
    dense_mla_reference,
    pack_mla_kv_cache_reference,
)
from b12x.attention._shared.mla.api import (
    MLASparseDecodeMetadata,
    MLASparseExtendMetadata,
    clear_mla_caches,
    sparse_mla_decode_forward,
    sparse_mla_extend_forward,
)
from b12x.attention.sparse_mla._scratch import (
    B12XSparseMLAScratchCaps,
    plan_sparse_mla_scratch,
)
from b12x.attention.dsa_indexer.contiguous_kernel import (
    _PREFILL512_BLOCK_K,
    _PREFILL512_BLOCK_Q,
    _PREFILL_BLOCK_Q,
)
from b12x.attention.dsa_indexer.tiled_topk import run_tiled_supertile_topk
from b12x.attention.dsa_indexer.persistent_topk import (
    run_persistent_topk2048,
    supports_persistent_topk2048,
)
from b12x.attention.dsa_indexer._impl import (
    IndexerContiguousMetadata,
    IndexerPagedDecodeMetadata,
    build_paged_mqa_schedule_metadata,
    clear_indexer_caches,
    contiguous_logits,
    contiguous_tiled_topk,
    paged_decode_logits,
    uses_paged_mqa_schedule,
)
from b12x.attention.dsa_indexer.contiguous_kernel import (
    resolve_contiguous_prefill_block_k,
)
from b12x.attention.dsa_indexer.paged import (
    index_topk_fp8,
    prepare_paged_indexer_metadata,
)
from b12x.attention.dsa_indexer.scratch import (
    INDEXER_SOURCE_LAYOUT_PAGED,
    B12XIndexerScratchCaps,
    plan_indexer_scratch,
)

from benchmarks.experimental.b12x.common import (
    capture_cuda_graph,
    make_dense_candidate_page_table,
    make_dense_real_page_table,
    make_l2_flush_fn,
    make_sparse_pool_locs,
    require_sm120,
    resolve_l2_flush_bytes,
    scatter_rows_into_pool,
)

try:
    from sgl_kernel.top_k import (
        fast_topk_transform_fused as _sgl_fast_topk_transform_fused,
        fast_topk_transform_ragged_fused as _sgl_fast_topk_transform_ragged_fused,
    )
except Exception:  # pragma: no cover - optional dependency
    _sgl_fast_topk_transform_fused = None
    _sgl_fast_topk_transform_ragged_fused = None


DEFAULT_GLM52_HF_REPO_ID = "lukealonso/GLM-5.2-NVFP4"
DEFAULT_BATCH_SIZES = (1, 2, 4, 8)
DEFAULT_CACHE_LENS = (1024, 32768, 65536, 131072)
DEFAULT_PREFILL_Q_LENS = (2048, 16384)
DEFAULT_DECODE_ROW_PATTERN = "uniform"
DEFAULT_TP_SIZE = 8
DEFAULT_TP_RANK = 0
DEFAULT_POOL_FACTOR = 6
DEFAULT_GRAPH_WIDTH = 8192
TARGET_PREFILL64K_BS1_PRESET = "target-prefill64k-bs1"
TARGET_GLM52_PREFILL4K_CTX16K_PRESET = "target-glm52-prefill4k-ctx16k"
TARGET_DSV4_TRACE_PRESET = "target-dsv4-trace"
MLA_MAX_ABS_TOL = 0.10
MLA_RMSE_TOL = 0.005
MLA_COS_TOL = 0.9995
_RAGGED_TOPK_CHUNK = 4096
_DSA_PREFILL_BLOCK_K_ENV = "B12X_DSA_CONTIGUOUS_PREFILL_BLOCK_K"
_DSA_DECODE_TOPK_BACKEND_ENV = "B12X_DSA_DECODE_TOPK_BACKEND"


def _align_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _align_down(value: int, multiple: int) -> int:
    return (value // multiple) * multiple


@triton.jit
def _remap_logical_topk_to_physical_kernel(
    logical_indices,
    real_page_table,
    physical_indices,
    numel,
    page_table_width,
    page_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < numel
    logical = tl.load(logical_indices + offsets, mask=valid, other=-1)
    page_col = logical // page_size
    valid_logical = valid & (logical >= 0) & (page_col < page_table_width)
    page_id = tl.load(
        real_page_table + page_col,
        mask=valid_logical,
        other=-1,
    )
    physical = page_id * page_size + logical % page_size
    physical = tl.where(valid_logical & (page_id >= 0), physical, -1)
    tl.store(physical_indices + offsets, physical, mask=valid)


def _remap_logical_topk_to_physical(
    *,
    logical_indices: torch.Tensor,
    real_page_table: torch.Tensor,
    physical_indices: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    if physical_indices.shape != logical_indices.shape:
        raise ValueError("physical and logical top-k buffers must have the same shape")
    numel = logical_indices.numel()
    _remap_logical_topk_to_physical_kernel[(triton.cdiv(numel, 256),)](
        logical_indices,
        real_page_table,
        physical_indices,
        numel,
        int(real_page_table.shape[1]),
        page_size=int(page_size),
        BLOCK=256,
        num_warps=4,
    )
    return physical_indices


def _make_vllm_packed_cache_views(
    *,
    index_k_cache: torch.Tensor,
    mla_kv_cache: torch.Tensor,
    page_size: int,
    block_stride_bytes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy one index/MLA layer pair into vLLM-style packed block views."""
    if block_stride_bytes == 0:
        return index_k_cache, mla_kv_cache
    if index_k_cache.dtype != torch.uint8 or mla_kv_cache.dtype != torch.uint8:
        raise ValueError("packed vLLM cache views require uint8 caches")
    if index_k_cache.ndim != 2 or mla_kv_cache.ndim != 3:
        raise ValueError("unexpected index/MLA cache ranks for packed vLLM views")
    num_pages = int(index_k_cache.shape[0])
    if int(mla_kv_cache.shape[0]) % int(page_size) != 0:
        raise ValueError("MLA cache token count must be divisible by page_size")
    if int(mla_kv_cache.shape[0]) // int(page_size) != num_pages:
        raise ValueError("index and MLA caches must contain the same page count")

    index_page_bytes = int(index_k_cache.shape[1])
    mla_record_bytes = int(mla_kv_cache.shape[2])
    mla_page_bytes = int(page_size) * mla_record_bytes
    occupied_bytes = index_page_bytes + mla_page_bytes
    if block_stride_bytes < occupied_bytes:
        raise ValueError(
            f"packed block stride {block_stride_bytes} is smaller than the "
            f"index+MLA page bytes {occupied_bytes}"
        )

    backing_bytes = (num_pages - 1) * block_stride_bytes + occupied_bytes
    backing = torch.empty(
        backing_bytes,
        dtype=torch.uint8,
        device=index_k_cache.device,
    )
    strided_index = torch.as_strided(
        backing,
        size=(num_pages, index_page_bytes),
        stride=(block_stride_bytes, 1),
    )
    strided_mla = torch.as_strided(
        backing[index_page_bytes:],
        size=(num_pages, int(page_size), mla_record_bytes),
        stride=(block_stride_bytes, mla_record_bytes, 1),
    )
    strided_index.copy_(index_k_cache)
    strided_mla.copy_(mla_kv_cache.view(num_pages, int(page_size), mla_record_bytes))
    return strided_index, strided_mla


@dataclass(frozen=True)
class GLMDecodeContractConfig:
    num_hidden_layers: int
    num_attention_heads: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    kv_lora_rank: int
    tp_size: int
    tp_rank: int
    page_size: int = 64

    @property
    def num_local_heads(self) -> int:
        return self.num_attention_heads // self.tp_size

    @property
    def q_head_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def sm_scale(self) -> float:
        return (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5

    @property
    def index_cache_page_bytes(self) -> int:
        scale_bytes = (self.index_head_dim // 128) * 4
        return self.page_size * (self.index_head_dim + scale_bytes)

    @property
    def mla_cache_page_bytes(self) -> int:
        scale_bytes = (self.kv_lora_rank // 128) * 4
        rope_bytes = self.qk_rope_head_dim * 2
        return self.page_size * (self.kv_lora_rank + scale_bytes + rope_bytes)

    @property
    def all_layer_cache_block_bytes(self) -> int:
        # Useful as a cross-layer packed-layout stress case. Normal GLM-5.2
        # serving does not use this stride: vLLM groups its full-MLA main and
        # index caches into one UniformTypeKVCacheSpecs group, then allocates
        # one contiguous tensor per cache layer. Multi-group layouts such as
        # DeepSeek V4 can instead expose cross-layer block-strided views.
        return self.num_hidden_layers * (
            self.index_cache_page_bytes + self.mla_cache_page_bytes
        )


@dataclass(frozen=True)
class DecodeCase:
    mode: str
    batch_size: int
    cache_len: int
    topk: int
    q_len: int = 1
    row_cache_lens: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.row_cache_lens is None:
            return
        if self.mode != "decode":
            raise ValueError("row_cache_lens is only supported for decode cases")
        if len(self.row_cache_lens) != self.batch_size:
            raise ValueError(
                "row_cache_lens length must match batch_size, got "
                f"{len(self.row_cache_lens)} vs {self.batch_size}"
            )
        if any(cache_len <= 0 for cache_len in self.row_cache_lens):
            raise ValueError(
                f"row_cache_lens must be positive, got {self.row_cache_lens}"
            )
        if max(self.row_cache_lens) != self.cache_len:
            raise ValueError(
                f"row_cache_lens max must equal cache_len {self.cache_len}, got {self.row_cache_lens}"
            )

    @property
    def total_q(self) -> int:
        return self.batch_size * self.q_len

    @property
    def decode_row_cache_lens(self) -> tuple[int, ...]:
        if self.row_cache_lens is not None:
            return self.row_cache_lens
        return (self.cache_len,) * self.batch_size

    @property
    def is_heterogeneous_decode(self) -> bool:
        return len(set(self.decode_row_cache_lens)) > 1


@dataclass(frozen=True)
class SanityMetrics:
    max_abs: float
    rmse: float
    cos: float


@dataclass(frozen=True)
class CaseReport:
    case: DecodeCase
    graph_width: int = 0
    cache_page_stride_bytes: int = 0
    step_samples_us: tuple[float, ...] = ()
    mla_samples_us: tuple[float, ...] = ()
    metadata_us: float = 0.0
    replay_us: float = 0.0
    indexer_us: float = 0.0
    indexer_logits_us: float = 0.0
    indexer_topk_us: float = 0.0
    mla_us: float = 0.0
    mla_forward_us: float = 0.0
    mla_merge_us: float = 0.0
    split_enabled: bool = False
    chunk_size: int = 0
    num_chunks: int = 0
    indexer_logits_fill: bool = True
    indexer_tiled_topk: bool = False
    indexer_topk_path: str = ""
    indexer_prefill_block_k: int | None = None
    mla_sanity: SanityMetrics = field(
        default_factory=lambda: SanityMetrics(max_abs=0.0, rmse=0.0, cos=1.0)
    )

    @property
    def total_us(self) -> float:
        if (
            self.metadata_us == 0.0
            and self.replay_us == 0.0
            and (self.indexer_us > 0.0 or self.mla_us > 0.0)
        ):
            return self.indexer_us + self.mla_us
        return self.metadata_us + self.replay_us



class BenchmarkFailure(RuntimeError):
    def __init__(self, case: DecodeCase, message: str):
        super().__init__(
            f"bs={case.batch_size} ctx={case.cache_len} topk={case.topk}: {message}"
        )
        self.case = case


def _resolve_cached_hf_config(
    repo_id: str = DEFAULT_GLM52_HF_REPO_ID,
    *,
    cache_root: pathlib.Path | None = None,
) -> pathlib.Path:
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


def _load_glm_contract_config(
    *,
    tp_size: int,
    tp_rank: int,
    model_config: pathlib.Path | None = None,
) -> GLMDecodeContractConfig:
    if model_config is None:
        model_config = _resolve_cached_hf_config()
    model_config = model_config.expanduser()
    if not model_config.is_file():
        raise SystemExit(f"model config not found at {model_config}")
    config = json.loads(model_config.read_text())
    num_attention_heads = int(config["num_attention_heads"])
    if num_attention_heads % tp_size != 0:
        raise SystemExit(
            f"num_attention_heads={num_attention_heads} is not divisible by tp_size={tp_size}"
        )
    if tp_rank < 0 or tp_rank >= tp_size:
        raise SystemExit(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")
    return GLMDecodeContractConfig(
        num_hidden_layers=int(config["num_hidden_layers"]),
        num_attention_heads=num_attention_heads,
        index_n_heads=int(config["index_n_heads"]),
        index_head_dim=int(config["index_head_dim"]),
        index_topk=int(config["index_topk"]),
        qk_nope_head_dim=int(config["qk_nope_head_dim"]),
        qk_rope_head_dim=int(config["qk_rope_head_dim"]),
        kv_lora_rank=int(config["kv_lora_rank"]),
        tp_size=tp_size,
        tp_rank=tp_rank,
    )


def _resolve_cache_page_stride_bytes(
    value: int,
    cfg: GLMDecodeContractConfig,
) -> int:
    value = int(value)
    if value == -1:
        return cfg.all_layer_cache_block_bytes
    if value < 0:
        raise ValueError(
            f"cache_page_stride_bytes must be -1, 0, or positive, got {value}"
        )
    minimum = cfg.index_cache_page_bytes + cfg.mla_cache_page_bytes
    if value != 0 and value < minimum:
        raise ValueError(
            f"cache_page_stride_bytes={value} is smaller than one index+MLA "
            f"page pair ({minimum} bytes)"
        )
    return value


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _format_csv_ints(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def _apply_benchmark_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset == "none":
        pass
    elif args.preset == TARGET_PREFILL64K_BS1_PRESET:
        args.modes = "prefill"
        args.batch_sizes = "1"
        args.cache_lens = "65536"
        args.verify_q_lens = "2048"
        args.topk_cap = 2048
        args.graph_width = 65536
    elif args.preset == TARGET_GLM52_PREFILL4K_CTX16K_PRESET:
        args.modes = "prefill"
        args.batch_sizes = "1"
        args.cache_lens = "16384"
        args.verify_q_lens = "4096"
        args.topk_cap = 2048
        args.graph_width = 16384
        args.use_tiled_topk = True
        args.prefill_indexer_layout = "paged"
        # Current vLLM's one-group UniformTypeKVCacheSpecs branch allocates
        # every normal GLM main/index cache as its own contiguous tensor.
        if args.cache_page_stride_bytes is None:
            args.cache_page_stride_bytes = 0
    elif args.preset == TARGET_DSV4_TRACE_PRESET:
        # main() delegates this distinct compressed-MLA contract to the native
        # DSV4 benchmark before any GLM config or tensor setup occurs.
        pass
    else:
        raise ValueError(f"unknown preset {args.preset!r}")
    if args.cache_page_stride_bytes is None:
        args.cache_page_stride_bytes = 0
    return args


def _resolve_topk(*, cache_len: int, topk_cap: int) -> int:
    if cache_len <= 0:
        raise ValueError(f"cache_len must be positive, got {cache_len}")
    if topk_cap <= 0:
        raise ValueError(f"topk_cap must be positive, got {topk_cap}")
    return min(cache_len, topk_cap)


def _build_decode_row_cache_lens(
    *,
    batch_size: int,
    cache_len: int,
    page_size: int,
    pattern: str,
) -> tuple[int, ...] | None:
    allowed_patterns = {"uniform", "staggered"}
    if pattern not in allowed_patterns:
        raise ValueError(
            f"unsupported decode row pattern {pattern!r}, expected one of {sorted(allowed_patterns)}"
        )
    if pattern == "uniform" or batch_size <= 1:
        return None
    row_cache_lens = []
    for row_idx in range(batch_size):
        row_len = cache_len * (batch_size - row_idx) // batch_size
        row_len = max(_align_down(row_len, page_size), page_size)
        row_cache_lens.append(min(row_len, cache_len))
    row_cache_lens[0] = cache_len
    return tuple(row_cache_lens)


def _build_decode_cases(
    *,
    modes: list[str],
    batch_sizes: list[int],
    cache_lens: list[int],
    verify_q_lens: list[int],
    topk_cap: int,
    decode_row_pattern: str,
    page_size: int,
) -> list[DecodeCase]:
    cases: list[DecodeCase] = []
    allowed_modes = {"decode", "prefill", "verify"}
    for mode in modes:
        if mode not in allowed_modes:
            raise ValueError(
                f"unsupported mode {mode!r}, expected one of {sorted(allowed_modes)}"
            )
    for batch_size in batch_sizes:
        if batch_size <= 0:
            raise ValueError(f"batch sizes must be positive, got {batch_size}")
        for cache_len in cache_lens:
            topk = _resolve_topk(cache_len=cache_len, topk_cap=topk_cap)
            if "decode" in modes:
                cases.append(
                    DecodeCase(
                        mode="decode",
                        batch_size=batch_size,
                        cache_len=cache_len,
                        topk=topk,
                        q_len=1,
                        row_cache_lens=_build_decode_row_cache_lens(
                            batch_size=batch_size,
                            cache_len=cache_len,
                            page_size=page_size,
                            pattern=decode_row_pattern,
                        ),
                    )
                )
            for prefill_mode in ("prefill", "verify"):
                if prefill_mode not in modes:
                    continue
                for q_len in verify_q_lens:
                    if q_len <= 0:
                        raise ValueError(f"prefill q_len must be positive, got {q_len}")
                    if q_len > cache_len:
                        continue
                    cases.append(
                        DecodeCase(
                            mode=prefill_mode,
                            batch_size=batch_size,
                            cache_len=cache_len,
                            topk=topk,
                            q_len=q_len,
                        )
                    )
    return cases


def _geomean(values: list[float]) -> float:
    if not values:
        raise ValueError("geomean requires at least one value")
    if any(value <= 0.0 for value in values):
        if all(value >= 0.0 for value in values):
            return 0.0
        raise ValueError("geomean requires non-negative values")
    return statistics.geometric_mean(values)


def _compare(a: torch.Tensor, b: torch.Tensor) -> SanityMetrics:
    diff = (a - b).to(torch.float32)
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()
    return SanityMetrics(
        max_abs=diff.abs().max().item(),
        rmse=torch.sqrt(diff.square().mean()).item(),
        cos=cos,
    )


def _check_mla_sanity(
    *,
    case: DecodeCase,
    label: str,
    metrics: SanityMetrics,
) -> None:
    if metrics.max_abs > MLA_MAX_ABS_TOL:
        raise BenchmarkFailure(
            case,
            f"{label} max_abs {metrics.max_abs:.6f} exceeded {MLA_MAX_ABS_TOL:.6f}",
        )
    if metrics.rmse > MLA_RMSE_TOL:
        raise BenchmarkFailure(
            case,
            f"{label} rmse {metrics.rmse:.6f} exceeded {MLA_RMSE_TOL:.6f}",
        )
    if metrics.cos < MLA_COS_TOL:
        raise BenchmarkFailure(
            case,
            f"{label} cos {metrics.cos:.6f} fell below {MLA_COS_TOL:.6f}",
        )


def _resolve_graph_width(*, cache_len: int, graph_width: int) -> int:
    if graph_width <= 0:
        raise ValueError(f"graph_width must be positive, got {graph_width}")
    return max(cache_len, graph_width)


def _assert_decode_contract_match(
    *,
    case: DecodeCase,
    actual: torch.Tensor,
    expected: torch.Tensor,
    page_table_1: torch.Tensor,
    seqlens: torch.Tensor,
    topk: int,
) -> None:
    del page_table_1, seqlens, topk
    sort_pad = torch.iinfo(actual.dtype).max
    actual_canon = torch.sort(torch.where(actual >= 0, actual, sort_pad), dim=1).values
    expected_canon = torch.sort(
        torch.where(expected >= 0, expected, sort_pad), dim=1
    ).values
    if not torch.equal(actual_canon, expected_canon):
        mismatch = int((actual_canon != expected_canon).sum().item())
        raise BenchmarkFailure(case, f"topk mismatch: {mismatch} differing entries")


def _select_paged_topk_from_logits(
    *,
    logits: torch.Tensor,
    page_table_1: torch.Tensor,
    seqlens: torch.Tensor,
    topk: int,
    cu_seqlens_q: torch.Tensor | None = None,
    query_row_to_batch: torch.Tensor | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    backend = backend.replace("_", "-")
    if backend not in {"auto", "sgl", "torch", "cute-persistent"}:
        raise ValueError(f"unsupported decode topk backend {backend!r}")

    if (
        backend == "cute-persistent"
        and query_row_to_batch is None
        and supports_persistent_topk2048(
            logits,
            seqlens.reshape(-1),
            topk=topk,
            page_table_1=page_table_1,
        )
    ):
        return run_persistent_topk2048(
            logits,
            seqlens.reshape(-1),
            page_table_1=page_table_1,
            max_seq_len=logits.shape[1],
        )

    if (
        backend in {"auto", "sgl"}
        and _sgl_fast_topk_transform_fused is not None
        and logits.is_cuda
        and topk == 2048
        and cu_seqlens_q is not None
    ):
        try:
            return _sgl_fast_topk_transform_fused(
                score=logits,
                lengths=seqlens,
                page_table_size_1=page_table_1,
                cu_seqlens_q=cu_seqlens_q,
                topk=topk,
            )
        except Exception:
            if backend == "sgl":
                raise

    if backend == "sgl":
        raise RuntimeError(
            "SGL fused topk backend is unavailable for this decode topk call"
        )

    rows = logits.shape[0]
    output = torch.full((rows, topk), -1, dtype=torch.int32, device=logits.device)
    gather_k = min(topk, logits.shape[1], page_table_1.shape[1])
    if gather_k == 0:
        return output
    topk_values, topk_pos = torch.topk(
        logits, k=gather_k, dim=1, largest=True, sorted=False
    )
    if query_row_to_batch is None:
        gathered = torch.gather(page_table_1, 1, topk_pos.to(torch.long))
    else:
        gathered = page_table_1[
            query_row_to_batch.to(torch.long).unsqueeze(1),
            topk_pos.to(torch.long),
        ]
    output[:, :gather_k] = torch.where(
        torch.isfinite(topk_values),
        gathered,
        torch.full_like(gathered, -1),
    )
    return output


def _rank_topk_candidates(
    *,
    values: torch.Tensor,
    positions: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape != positions.shape:
        raise ValueError("values and positions must have the same shape")
    if values.ndim != 2:
        raise ValueError(f"values must be rank-2, got {tuple(values.shape)}")
    gather_k = min(topk, values.shape[1])
    if gather_k == 0:
        empty_values = values[:, :0]
        empty_positions = positions[:, :0]
        return empty_values, empty_positions
    pos_order = torch.argsort(positions, dim=1, descending=False, stable=True)
    positions = torch.gather(positions, 1, pos_order)
    values = torch.gather(values, 1, pos_order)
    value_order = torch.argsort(values, dim=1, descending=True, stable=True)[
        :, :gather_k
    ]
    return (
        torch.gather(values, 1, value_order),
        torch.gather(positions, 1, value_order),
    )


def _select_ragged_topk_from_logits_chunked(
    *,
    logits: torch.Tensor,
    k_start: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    rows = logits.shape[0]
    output = torch.full((rows, topk), -1, dtype=torch.int32, device=logits.device)
    gather_k = min(topk, logits.shape[1])
    if gather_k == 0:
        return output

    row_start = k_start.unsqueeze(1)
    row_end = row_start + lengths.unsqueeze(1)
    best_values = torch.full(
        (rows, gather_k),
        float("-inf"),
        dtype=logits.dtype,
        device=logits.device,
    )
    best_pos = torch.full((rows, gather_k), -1, dtype=torch.int32, device=logits.device)

    for chunk_start in range(0, logits.shape[1], _RAGGED_TOPK_CHUNK):
        chunk_end = min(chunk_start + _RAGGED_TOPK_CHUNK, logits.shape[1])
        local_k = min(gather_k, chunk_end - chunk_start)
        if local_k == 0:
            continue
        chunk_logits = logits[:, chunk_start:chunk_end]
        positions = torch.arange(
            chunk_start,
            chunk_end,
            dtype=torch.int32,
            device=logits.device,
        ).unsqueeze(0)
        valid = (positions >= row_start) & (positions < row_end)
        masked_logits = torch.where(
            valid, chunk_logits, torch.full_like(chunk_logits, float("-inf"))
        )
        chunk_values, chunk_pos = torch.topk(
            masked_logits,
            k=local_k,
            dim=1,
            largest=True,
            sorted=False,
        )
        chunk_pos = chunk_pos.to(torch.int32) + chunk_start
        chunk_values, chunk_pos = _rank_topk_candidates(
            values=chunk_values,
            positions=chunk_pos,
            topk=local_k,
        )
        merged_values = torch.cat([best_values, chunk_values], dim=1)
        merged_pos = torch.cat([best_pos, chunk_pos], dim=1)
        best_values, best_pos = _rank_topk_candidates(
            values=merged_values,
            positions=merged_pos,
            topk=gather_k,
        )

    output[:, :gather_k] = torch.where(
        torch.isfinite(best_values),
        best_pos,
        torch.full_like(best_pos, -1),
    )
    return output


def _select_ragged_topk_from_logits(
    *,
    logits: torch.Tensor,
    k_start: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    if (
        _sgl_fast_topk_transform_ragged_fused is not None
        and logits.is_cuda
        and topk == 2048
    ):
        try:
            return _sgl_fast_topk_transform_ragged_fused(
                score=logits,
                lengths=lengths,
                topk_indices_offset=k_start,
                topk=topk,
                row_starts=k_start,
            )
        except Exception:
            pass

    return _select_ragged_topk_from_logits_chunked(
        logits=logits,
        k_start=k_start,
        lengths=lengths,
        topk=topk,
    )


def _capture_and_bench_cuda_graph(fn, *, warmup, replays, prepare=None, l2_flush=None):
    from b12x.testing.benchmark import transaction_samples
    graph = capture_cuda_graph(fn, warmup=warmup, prepare=prepare)
    try:
        return transaction_samples(fn, samples=replays, prepare=prepare, l2_flush=l2_flush)
    finally:
        del graph


def _make_decode_graph_prepare(
    *,
    live_page_table_1: torch.Tensor,
    live_real_page_table: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    nsa_cache_seqlens_int32: torch.Tensor,
    graph_page_table_1: torch.Tensor,
    graph_real_page_table: torch.Tensor,
    graph_cache_seqlens_int32: torch.Tensor,
    graph_nsa_cache_seqlens_int32: torch.Tensor,
    graph_paged_mqa_schedule_metadata: torch.Tensor | None = None,
    schedule_block_kv: int | None = None,
):
    live_width = live_page_table_1.shape[1]
    live_block_width = live_real_page_table.shape[1]

    def prepare() -> None:
        graph_page_table_1[:, :live_width].copy_(live_page_table_1)
        graph_real_page_table[:, :live_block_width].copy_(live_real_page_table)
        graph_cache_seqlens_int32.copy_(cache_seqlens_int32)
        graph_nsa_cache_seqlens_int32.copy_(nsa_cache_seqlens_int32)
        if graph_paged_mqa_schedule_metadata is not None:
            if schedule_block_kv is None:
                raise ValueError(
                    "schedule_block_kv must be provided when graph schedule metadata is set"
                )
            build_paged_mqa_schedule_metadata(
                graph_cache_seqlens_int32,
                schedule_block_kv,
                out=graph_paged_mqa_schedule_metadata,
            )

    return prepare


def _make_indexer_inputs(
    *,
    q_rows: int,
    cache_len: int,
    cfg: GLMDecodeContractConfig,
    seed: int,
    device: torch.device,
    pool_locs: torch.Tensor,
    pool_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del seed
    q_fp8 = torch.full(
        (q_rows, cfg.index_n_heads, cfg.index_head_dim),
        0.5,
        dtype=torch.float32,
        device=device,
    ).to(torch.float8_e4m3fn)
    weights = torch.ones(
        (q_rows, cfg.index_n_heads, 1),
        dtype=torch.float32,
        device=device,
    )
    token_scores = torch.linspace(
        0.25,
        1.25,
        cache_len,
        dtype=torch.float32,
        device=device,
    )
    k = token_scores.unsqueeze(1).expand(-1, cfg.index_head_dim).contiguous()
    k_pool = scatter_rows_into_pool(k, pool_locs=pool_locs, pool_tokens=pool_tokens)
    return q_fp8, weights, pack_index_k_cache_reference(k_pool, page_size=cfg.page_size)


def _make_mla_inputs(
    *,
    q_rows: int,
    cache_len: int,
    cfg: GLMDecodeContractConfig,
    seed: int,
    device: torch.device,
    pool_locs: torch.Tensor,
    pool_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    q_all = (
        torch.randn(
            (q_rows, cfg.num_local_heads, cfg.q_head_dim),
            generator=gen,
            dtype=torch.float32,
        )
        .to(device=device)
        .div_(4.0)
        .to(torch.bfloat16)
    )
    k_nope = (
        torch.randn(
            (cache_len, 1, cfg.kv_lora_rank),
            generator=gen,
            dtype=torch.float32,
        )
        .to(device=device)
        .div_(4.0)
        .to(torch.bfloat16)
    )
    k_rope = (
        torch.randn(
            (cache_len, 1, cfg.qk_rope_head_dim),
            generator=gen,
            dtype=torch.float32,
        )
        .to(device=device)
        .div_(4.0)
        .to(torch.bfloat16)
    )
    k_nope_pool = scatter_rows_into_pool(
        k_nope, pool_locs=pool_locs, pool_tokens=pool_tokens
    )
    k_rope_pool = scatter_rows_into_pool(
        k_rope, pool_locs=pool_locs, pool_tokens=pool_tokens
    )
    kv_cache = pack_mla_kv_cache_reference(k_nope_pool, k_rope_pool)
    return q_all, k_nope_pool, k_rope_pool, kv_cache








def _make_mla_binding(
    *,
    mode: str,
    cfg: GLMDecodeContractConfig,
    device: torch.device,
    topk: int,
    max_total_q: int,
    max_batch: int,
    q_all: torch.Tensor,
    selected_indices: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    nsa_cache_seqlens_int32: torch.Tensor,
):
    plan = plan_sparse_mla_scratch(
        B12XSparseMLAScratchCaps(
            softmax_scale=1.0,
            mode=mode,
            device=device,
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            num_q_heads=cfg.num_local_heads,
            head_dim=cfg.q_head_dim,
            v_head_dim=cfg.kv_lora_rank,
            max_width=topk,
            max_q_rows=max_total_q,
            max_batch=max_batch,
            page_size=cfg.page_size,
        )
    )
    (spec,) = plan.scratch_specs()
    scratch_storage = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    return plan.bind(
        scratch=scratch_storage,
        q=q_all,
        selected_indices=selected_indices,
        cache_seqlens_int32=cache_seqlens_int32,
        nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
    )


def _remap_selected_indices_to_local_offsets(
    *,
    selected_indices: torch.Tensor,
    physical_to_local: torch.Tensor,
) -> torch.Tensor:
    local_offsets = physical_to_local.index_select(
        0,
        selected_indices.clamp_min(0).reshape(-1).to(torch.long),
    ).view_as(selected_indices)
    local_offsets.masked_fill_(selected_indices < 0, -1)
    return local_offsets


def _make_extend_kv_fp8(
    *,
    index_k_cache: torch.Tensor,
    real_page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    data_bytes = page_size * 128
    total_rows = int(seq_lens.sum().item())
    k_bytes = torch.empty(
        (total_rows, 128), dtype=torch.uint8, device=index_k_cache.device
    )
    scale_bytes = torch.empty(
        (total_rows, 4), dtype=torch.uint8, device=index_k_cache.device
    )
    write_row = 0
    for batch_row in range(real_page_table.shape[0]):
        seq_len = int(seq_lens[batch_row].item())
        for token_pos in range(seq_len):
            page_col = token_pos // page_size
            slot = token_pos % page_size
            page_id = int(real_page_table[batch_row, page_col].item())
            k_bytes[write_row] = index_k_cache[page_id, slot * 128 : (slot + 1) * 128]
            scale_bytes[write_row] = index_k_cache[
                page_id,
                data_bytes + slot * 4 : data_bytes + (slot + 1) * 4,
            ]
            write_row += 1
    return k_bytes.view(torch.float8_e4m3fn), scale_bytes.view(torch.float32).squeeze(
        -1
    )


def _run_decode_case(
    *,
    case: DecodeCase,
    cfg: GLMDecodeContractConfig,
    warmup: int,
    replays: int,
    seed: int,
    device: torch.device,
    pool_factor: int,
    graph_width: int,
    cache_page_stride_bytes: int,
    l2_flush,
    skip_indexer_logits_fill: bool,
    decode_topk_backend: str,
) -> CaseReport:
    if pool_factor <= 0:
        raise ValueError(f"pool_factor must be positive, got {pool_factor}")
    graph_width = _resolve_graph_width(
        cache_len=case.cache_len, graph_width=graph_width
    )
    aligned_graph_width = _align_up(graph_width, cfg.page_size)
    pool_tokens = _align_up(
        max(case.cache_len, case.cache_len * pool_factor), cfg.page_size
    )
    pool_locs = make_sparse_pool_locs(
        active_tokens=case.cache_len,
        pool_tokens=pool_tokens,
        seed=seed + 2,
        device=device,
        page_size=cfg.page_size,
    )
    q_fp8, weights, index_k_cache = _make_indexer_inputs(
        q_rows=case.total_q,
        cache_len=case.cache_len,
        cfg=cfg,
        seed=seed,
        device=device,
        pool_locs=pool_locs,
        pool_tokens=pool_tokens,
    )
    q_all, k_nope, k_rope, kv_cache = _make_mla_inputs(
        q_rows=case.total_q,
        cache_len=case.cache_len,
        cfg=cfg,
        seed=seed + 1,
        device=device,
        pool_locs=pool_locs,
        pool_tokens=pool_tokens,
    )
    index_k_cache, kv_cache = _make_vllm_packed_cache_views(
        index_k_cache=index_k_cache,
        mla_kv_cache=kv_cache,
        page_size=cfg.page_size,
        block_stride_bytes=cache_page_stride_bytes,
    )
    live_candidate_page_table = make_dense_candidate_page_table(
        batch_size=case.batch_size,
        token_locs=pool_locs,
        width=case.cache_len,
        fill_value=-1,
    )
    live_real_page_table = make_dense_real_page_table(
        batch_size=case.batch_size,
        token_locs=pool_locs,
        width_blocks=aligned_graph_width // cfg.page_size,
        page_size=cfg.page_size,
    )
    full_cache_seqlens = torch.tensor(
        case.decode_row_cache_lens,
        dtype=torch.int32,
        device=device,
    )
    nsa_cache_seqlens = torch.minimum(
        full_cache_seqlens,
        torch.full((case.batch_size,), case.topk, dtype=torch.int32, device=device),
    )
    graph_candidate_page_table = torch.full(
        (case.batch_size, aligned_graph_width),
        -1,
        dtype=torch.int32,
        device=device,
    )
    graph_real_page_table = torch.full(
        (case.batch_size, aligned_graph_width // cfg.page_size),
        -1,
        dtype=torch.int32,
        device=device,
    )
    graph_cache_seqlens = torch.empty_like(full_cache_seqlens)
    graph_nsa_cache_seqlens = torch.empty_like(nsa_cache_seqlens)
    graph_active_width_override = None
    if (
        case.batch_size == 1
        and case.q_len == 1
        and case.cache_len == aligned_graph_width
    ):
        graph_active_width_override = torch.tensor(
            [aligned_graph_width],
            dtype=torch.int32,
            device=device,
        )
    use_graph_schedule_metadata = uses_paged_mqa_schedule(
        q_rows=case.batch_size,
        max_pages=graph_real_page_table.shape[1],
    )
    graph_schedule_metadata = (
        torch.empty(
            (torch.cuda.get_device_properties(device).multi_processor_count + 1, 2),
            dtype=torch.int32,
            device=device,
        )
        if use_graph_schedule_metadata
        else None
    )
    prepare_decode_graph = _make_decode_graph_prepare(
        live_page_table_1=live_candidate_page_table,
        live_real_page_table=live_real_page_table,
        cache_seqlens_int32=full_cache_seqlens,
        nsa_cache_seqlens_int32=nsa_cache_seqlens,
        graph_page_table_1=graph_candidate_page_table,
        graph_real_page_table=graph_real_page_table,
        graph_cache_seqlens_int32=graph_cache_seqlens,
        graph_nsa_cache_seqlens_int32=graph_nsa_cache_seqlens,
        graph_paged_mqa_schedule_metadata=graph_schedule_metadata,
        schedule_block_kv=cfg.page_size,
    )
    prepare_decode_graph()
    indexer_metadata = IndexerPagedDecodeMetadata(
        real_page_table=graph_real_page_table,
        cache_seqlens_int32=graph_cache_seqlens,
        paged_mqa_schedule_metadata=graph_schedule_metadata,
    )
    preinitialize_indexer_logits = not (
        skip_indexer_logits_fill
        and case.batch_size == 1
        and case.q_len == 1
        and case.cache_len == aligned_graph_width
    )

    def run_indexer():
        logits = paged_decode_logits(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=indexer_metadata,
            page_size=cfg.page_size,
            preinitialize_invalid_logits=preinitialize_indexer_logits,
            active_width_override=graph_active_width_override,
        )
        return _select_paged_topk_from_logits(
            logits=logits,
            page_table_1=graph_candidate_page_table,
            seqlens=graph_cache_seqlens,
            topk=case.topk,
            backend=decode_topk_backend,
        )

    def run_indexer_logits():
        return paged_decode_logits(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=indexer_metadata,
            page_size=cfg.page_size,
            preinitialize_invalid_logits=preinitialize_indexer_logits,
            active_width_override=graph_active_width_override,
        )

    clear_indexer_caches()
    actual_topk = run_indexer()
    logits_for_topk = run_indexer_logits()
    expected_logits = paged_decode_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        query_row_to_batch=torch.arange(
            case.batch_size, dtype=torch.int32, device=device
        ),
        seqlens_per_query=graph_cache_seqlens,
        page_size=cfg.page_size,
    )
    expected_topk = _select_paged_topk_from_logits(
        logits=expected_logits,
        page_table_1=graph_candidate_page_table,
        seqlens=graph_cache_seqlens,
        topk=case.topk,
        backend="torch",
    )
    torch.cuda.synchronize()
    _assert_decode_contract_match(
        case=case,
        actual=actual_topk,
        expected=expected_topk,
        page_table_1=graph_candidate_page_table,
        seqlens=graph_cache_seqlens,
        topk=case.topk,
    )
    del expected_logits
    del expected_topk

    mla_metadata = MLASparseDecodeMetadata(
        page_table_1=actual_topk,
        cache_seqlens_int32=graph_cache_seqlens,
        nsa_cache_seqlens_int32=graph_nsa_cache_seqlens,
        max_seq_len_k=aligned_graph_width,
    )
    mla_binding = _make_mla_binding(
        mode="decode",
        cfg=cfg,
        device=device,
        topk=case.topk,
        max_total_q=case.total_q,
        max_batch=case.batch_size,
        q_all=q_all,
        selected_indices=mla_metadata.page_table_1,
        cache_seqlens_int32=mla_metadata.cache_seqlens_int32,
        nsa_cache_seqlens_int32=mla_metadata.nsa_cache_seqlens_int32,
    )
    mla_workspace = mla_binding.scratch
    split_cfg = default_sparse_mla_split_decode_config_for_width(
        int(actual_topk.shape[1]),
        max_chunks=int(mla_workspace.max_chunks_per_row),
    )

    def run_mla():
        return sparse_mla_decode_forward(
            kv_cache=kv_cache,
            binding=mla_binding,
            sm_scale=cfg.sm_scale,
            v_head_dim=cfg.kv_lora_rank,
        )


    def run_step():
        topk_indices = _select_paged_topk_from_logits(
            logits=paged_decode_logits(
                q_fp8=q_fp8,
                weights=weights,
                index_k_cache=index_k_cache,
                metadata=indexer_metadata,
                page_size=cfg.page_size,
                preinitialize_invalid_logits=preinitialize_indexer_logits,
                active_width_override=graph_active_width_override,
            ),
            page_table_1=graph_candidate_page_table,
            seqlens=graph_cache_seqlens,
            topk=case.topk,
            backend=decode_topk_backend,
        )
        step_binding = mla_workspace.bind(
            q=q_all,
            selected_indices=topk_indices,
            cache_seqlens_int32=graph_cache_seqlens,
            nsa_cache_seqlens_int32=graph_nsa_cache_seqlens,
        )
        return sparse_mla_decode_forward(
            kv_cache=kv_cache,
            binding=step_binding,
            sm_scale=cfg.sm_scale,
            v_head_dim=cfg.kv_lora_rank,
        )

    clear_mla_caches()
    actual_output = run_mla()
    expected_output = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=actual_topk,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize()
    mla_sanity = _compare(actual_output, expected_output)
    _check_mla_sanity(case=case, label="MLA", metrics=mla_sanity)
    del actual_output
    del expected_output

    clear_indexer_caches()
    indexer_stats = _capture_and_bench_cuda_graph(
        run_indexer,
        warmup=warmup,
        replays=replays,
        prepare=prepare_decode_graph,
        l2_flush=l2_flush,
    )
    indexer_us = statistics.median(indexer_stats["replay_us"])

    clear_indexer_caches()
    indexer_logits_stats = _capture_and_bench_cuda_graph(
        run_indexer_logits,
        warmup=warmup,
        replays=replays,
        prepare=prepare_decode_graph,
        l2_flush=l2_flush,
    )
    indexer_logits_us = statistics.median(indexer_logits_stats["replay_us"])

    def run_indexer_topk():
        return _select_paged_topk_from_logits(
            logits=logits_for_topk,
            page_table_1=graph_candidate_page_table,
            seqlens=graph_cache_seqlens,
            topk=case.topk,
            backend=decode_topk_backend,
        )

    indexer_topk_stats = _capture_and_bench_cuda_graph(
        run_indexer_topk,
        warmup=warmup,
        replays=replays,
        prepare=prepare_decode_graph,
        l2_flush=l2_flush,
    )
    indexer_topk_us = statistics.median(indexer_topk_stats["replay_us"])

    clear_mla_caches()
    prepare_decode_graph()
    mla_stats = _capture_and_bench_cuda_graph(
        run_mla,
        warmup=warmup,
        replays=replays,
        l2_flush=l2_flush,
    )
    mla_us = statistics.median(mla_stats["replay_us"])


    clear_indexer_caches()
    clear_mla_caches()
    step_stats = _capture_and_bench_cuda_graph(
        run_step,
        warmup=warmup,
        replays=replays,
        prepare=prepare_decode_graph,
        l2_flush=l2_flush,
    )
    return CaseReport(
        case=case,
        graph_width=graph_width,
        cache_page_stride_bytes=cache_page_stride_bytes,
        step_samples_us=tuple(
            metadata_us + replay_us
            for metadata_us, replay_us in zip(
                step_stats["metadata_us"],
                step_stats["replay_us"],
                strict=True,
            )
        ),
        mla_samples_us=tuple(mla_stats["replay_us"]),

        metadata_us=statistics.median(step_stats["metadata_us"]),
        replay_us=statistics.median(step_stats["replay_us"]),
        indexer_us=indexer_us,
        indexer_logits_us=indexer_logits_us,
        indexer_topk_us=indexer_topk_us,
        mla_us=mla_us,

        split_enabled=split_cfg is not None,
        chunk_size=0 if split_cfg is None else split_cfg.chunk_size,
        num_chunks=0 if split_cfg is None else split_cfg.num_chunks,
        indexer_logits_fill=preinitialize_indexer_logits,
        indexer_topk_path=decode_topk_backend.replace("_", "-"),
        mla_sanity=mla_sanity,


    )


def _run_prefill_or_verify_case(
    *,
    case: DecodeCase,
    cfg: GLMDecodeContractConfig,
    warmup: int,
    replays: int,
    seed: int,
    device: torch.device,
    pool_factor: int,
    graph_width: int,
    cache_page_stride_bytes: int,
    l2_flush,
    skip_indexer_logits_fill: bool,
    use_tiled_topk: bool,
    prefill_indexer_layout: str,
) -> CaseReport:
    if pool_factor <= 0:
        raise ValueError(f"pool_factor must be positive, got {pool_factor}")
    if case.q_len <= 1:
        raise ValueError(f"prefill q_len must be > 1, got {case.q_len}")
    use_paged_prefill = case.mode == "prefill" and prefill_indexer_layout == "paged"
    if use_paged_prefill and case.batch_size != 1:
        raise ValueError("paged prefill indexer benchmark requires batch_size=1")
    graph_width = _resolve_graph_width(
        cache_len=case.cache_len, graph_width=graph_width
    )
    aligned_graph_width = _align_up(graph_width, cfg.page_size)
    pool_tokens = _align_up(
        max(case.cache_len, case.cache_len * pool_factor), cfg.page_size
    )
    pool_locs = make_sparse_pool_locs(
        active_tokens=case.cache_len,
        pool_tokens=pool_tokens,
        seed=seed + 2,
        device=device,
        page_size=cfg.page_size,
    )
    q_fp8, weights, index_k_cache = _make_indexer_inputs(
        q_rows=case.total_q,
        cache_len=case.cache_len,
        cfg=cfg,
        seed=seed,
        device=device,
        pool_locs=pool_locs,
        pool_tokens=pool_tokens,
    )
    q_all, k_nope, k_rope, kv_cache = _make_mla_inputs(
        q_rows=case.total_q,
        cache_len=case.cache_len,
        cfg=cfg,
        seed=seed + 1,
        device=device,
        pool_locs=pool_locs,
        pool_tokens=pool_tokens,
    )
    index_k_cache, kv_cache = _make_vllm_packed_cache_views(
        index_k_cache=index_k_cache,
        mla_kv_cache=kv_cache,
        page_size=cfg.page_size,
        block_stride_bytes=cache_page_stride_bytes,
    )
    base_real_page_table = make_dense_real_page_table(
        batch_size=case.batch_size,
        token_locs=pool_locs,
        width_blocks=aligned_graph_width // cfg.page_size,
        page_size=cfg.page_size,
    )
    query_row_to_batch = torch.arange(
        case.batch_size,
        dtype=torch.int32,
        device=device,
    ).repeat_interleave(case.q_len)
    if use_paged_prefill:
        # vLLM expands the single request's block table across all query rows;
        # preserving stride(0)==0 is part of the production b12x contract.
        live_real_page_table = base_real_page_table[:1].expand(case.total_q, -1)
    else:
        live_real_page_table = base_real_page_table.index_select(
            0,
            query_row_to_batch.to(torch.long),
        ).contiguous()
    batch_cache_seqlens = torch.full(
        (case.batch_size,),
        case.cache_len,
        dtype=torch.int32,
        device=device,
    )
    expanded_cache_seqlens = torch.repeat_interleave(
        batch_cache_seqlens, repeats=case.q_len
    )
    nsa_cache_seqlens = torch.full(
        (case.total_q,),
        case.topk,
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_q = torch.arange(
        0,
        case.total_q + 1,
        step=case.q_len,
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_k = torch.arange(
        0,
        case.total_q * case.topk + 1,
        step=case.topk,
        dtype=torch.int32,
        device=device,
    )
    if use_paged_prefill:
        graph_real_page_table_storage = torch.full(
            (1, aligned_graph_width // cfg.page_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        graph_real_page_table = graph_real_page_table_storage.expand(case.total_q, -1)
    else:
        graph_real_page_table_storage = None
        graph_real_page_table = torch.full(
            (case.total_q, aligned_graph_width // cfg.page_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
    graph_batch_cache_seqlens = torch.empty_like(batch_cache_seqlens)
    graph_expanded_cache_seqlens = torch.empty_like(expanded_cache_seqlens)
    graph_nsa_cache_seqlens = torch.empty_like(nsa_cache_seqlens)
    use_graph_schedule_metadata = not use_paged_prefill and uses_paged_mqa_schedule(
        q_rows=case.total_q,
        max_pages=graph_real_page_table.shape[1],
    )
    graph_schedule_metadata = (
        torch.empty(
            (torch.cuda.get_device_properties(device).multi_processor_count + 1, 2),
            dtype=torch.int32,
            device=device,
        )
        if use_graph_schedule_metadata
        else None
    )
    live_contiguous_lengths = torch.arange(
        case.cache_len - case.q_len + 1,
        case.cache_len + 1,
        dtype=torch.int32,
        device=device,
    ).repeat(case.batch_size)
    live_contiguous_k_start = torch.repeat_interleave(
        torch.arange(case.batch_size, dtype=torch.int32, device=device)
        * case.cache_len,
        case.q_len,
    )
    graph_contiguous_k_start = torch.empty_like(live_contiguous_k_start)
    graph_contiguous_lengths = torch.empty_like(live_contiguous_lengths)

    def prepare_verify_graph() -> None:
        if use_paged_prefill:
            assert graph_real_page_table_storage is not None
            graph_real_page_table_storage[0, : live_real_page_table.shape[1]].copy_(
                live_real_page_table[0]
            )
        else:
            graph_real_page_table[:, : live_real_page_table.shape[1]].copy_(
                live_real_page_table
            )
        graph_batch_cache_seqlens.copy_(batch_cache_seqlens)
        graph_expanded_cache_seqlens.copy_(
            live_contiguous_lengths if use_paged_prefill else expanded_cache_seqlens
        )
        graph_nsa_cache_seqlens.copy_(nsa_cache_seqlens)
        if graph_schedule_metadata is not None:
            build_paged_mqa_schedule_metadata(
                graph_expanded_cache_seqlens,
                cfg.page_size,
                out=graph_schedule_metadata,
            )
        graph_contiguous_k_start.copy_(live_contiguous_k_start)
        graph_contiguous_lengths.copy_(live_contiguous_lengths)

    prepare_verify_graph()
    contiguous_k_nope = k_nope[pool_locs.to(torch.long)]
    contiguous_k_rope = k_rope[pool_locs.to(torch.long)]
    extend_kv_cache = pack_mla_kv_cache_reference(contiguous_k_nope, contiguous_k_rope)
    extend_kv_fp8 = (
        None
        if use_paged_prefill
        else _make_extend_kv_fp8(
            index_k_cache=index_k_cache,
            real_page_table=base_real_page_table,
            seq_lens=batch_cache_seqlens,
            page_size=cfg.page_size,
        )
    )
    use_runtime_ragged_topk = (
        device.type == "cuda"
        and case.topk == 2048
        and _sgl_fast_topk_transform_ragged_fused is not None
    )
    paged_prefill_plan = None

    def map_indexer_topk(topk_indices: torch.Tensor) -> torch.Tensor:
        return topk_indices

    if case.mode == "verify":
        base_candidate_page_table = make_dense_candidate_page_table(
            batch_size=case.batch_size,
            token_locs=pool_locs,
            width=case.cache_len,
            fill_value=-1,
        )
        indexer_metadata = IndexerPagedDecodeMetadata(
            real_page_table=graph_real_page_table,
            cache_seqlens_int32=graph_expanded_cache_seqlens,
            paged_mqa_schedule_metadata=graph_schedule_metadata,
        )

        def run_indexer_logits():
            return paged_decode_logits(
                q_fp8=q_fp8,
                weights=weights,
                index_k_cache=index_k_cache,
                metadata=indexer_metadata,
                page_size=cfg.page_size,
            )

        def run_indexer():
            logits = run_indexer_logits()
            return _select_paged_topk_from_logits(
                logits=logits,
                page_table_1=base_candidate_page_table,
                seqlens=graph_expanded_cache_seqlens,
                topk=case.topk,
                cu_seqlens_q=cu_seqlens_q,
                query_row_to_batch=query_row_to_batch,
            )

        clear_indexer_caches()
        actual_topk = run_indexer()
        logits_for_topk = run_indexer_logits()
        expected_logits = paged_decode_logits_reference(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            real_page_table=graph_real_page_table,
            query_row_to_batch=query_row_to_batch,
            seqlens_per_query=graph_expanded_cache_seqlens,
            page_size=cfg.page_size,
        )
        expected_topk = _select_paged_topk_from_logits(
            logits=expected_logits,
            page_table_1=base_candidate_page_table,
            seqlens=graph_expanded_cache_seqlens,
            topk=case.topk,
            cu_seqlens_q=cu_seqlens_q,
            query_row_to_batch=query_row_to_batch,
        )
        torch.cuda.synchronize()
        _assert_decode_contract_match(
            case=case,
            actual=actual_topk,
            expected=expected_topk,
            page_table_1=base_candidate_page_table,
            seqlens=graph_expanded_cache_seqlens,
            topk=case.topk,
        )
        del expected_logits
        del expected_topk
        mla_selected_indices = actual_topk
        mla_kv_cache = kv_cache
        mla_k_nope = k_nope
        mla_k_rope = k_rope
        mla_metadata_mode = "target_verify"
        mla_workspace_mode = "verify"
    elif use_paged_prefill:
        paged_prefill_plan = plan_indexer_scratch(
            B12XIndexerScratchCaps(
                device=device,
                source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
                num_q_heads=cfg.index_n_heads,
                max_q_rows=case.total_q,
                max_page_table_width=graph_real_page_table.shape[1],
                topk=case.topk,
                page_size=cfg.page_size,
                mode="prefill",
                shared_page_table=True,
            )
        )
        paged_scratch = [
            torch.empty(shape, dtype=dtype, device=device)
            for shape, dtype in paged_prefill_plan.shapes_and_dtypes()
        ]
        paged_metadata = prepare_paged_indexer_metadata(
            real_page_table=graph_real_page_table,
            cache_seqlens_int32=graph_expanded_cache_seqlens,
            expected_num_q_heads=cfg.index_n_heads,
            build_schedule=False,
            shared_page_table=True,
        )
        paged_binding = paged_prefill_plan.bind(
            scratch=paged_scratch,
            real_page_table=paged_metadata.real_page_table,
            cache_seqlens_int32=paged_metadata.cache_seqlens_int32,
            expected_num_q_heads=cfg.index_n_heads,
            shared_page_table=True,
            output_physical_slots=True,
        )
        paged_topk_indices = torch.empty(
            (case.total_q, case.topk), dtype=torch.int32, device=device
        )
        paged_topk_scores = torch.empty(
            (case.total_q, case.topk), dtype=torch.float32, device=device
        )

        def run_indexer():
            return index_topk_fp8(
                q_fp8=q_fp8,
                weights=weights.squeeze(-1),
                index_k_cache=index_k_cache,
                binding=paged_binding,
                topk=case.topk,
                expected_num_q_heads=cfg.index_n_heads,
                out_indices=paged_topk_indices,
                out_scores=paged_topk_scores,
            )

        clear_indexer_caches()
        actual_topk = run_indexer()
        torch.cuda.synchronize()
        expected_logical_topk = (
            graph_expanded_cache_seqlens[:, None]
            - case.topk
            + torch.arange(case.topk, dtype=torch.int32, device=device)
        )
        expected_topk = _remap_logical_topk_to_physical(
            logical_indices=expected_logical_topk,
            real_page_table=graph_real_page_table,
            physical_indices=torch.empty_like(expected_logical_topk),
            page_size=cfg.page_size,
        )
        actual_topk_sorted = torch.sort(actual_topk, dim=1).values
        expected_topk_sorted = torch.sort(expected_topk, dim=1).values
        if not torch.equal(actual_topk_sorted, expected_topk_sorted):
            mismatch = int((actual_topk_sorted != expected_topk_sorted).sum().item())
            raise BenchmarkFailure(
                case,
                f"paged prefill topk mismatch: {mismatch} differing entries",
            )
        if not torch.isfinite(paged_topk_scores).all().item():
            raise BenchmarkFailure(case, "paged prefill topk scores are non-finite")
        mla_selected_indices = actual_topk
        mla_kv_cache = kv_cache
        mla_k_nope = k_nope
        mla_k_rope = k_rope
        mla_metadata_mode = "extend"
        mla_workspace_mode = "extend"
        preinitialize_indexer_logits = False
        _use_tiled_output = True
    else:
        assert extend_kv_fp8 is not None
        extend_indexer_metadata = IndexerContiguousMetadata(
            k_start=graph_contiguous_k_start,
            k_end=graph_contiguous_k_start + graph_contiguous_lengths,
        )
        preinitialize_indexer_logits = not skip_indexer_logits_fill

        _prefill_block_k = resolve_contiguous_prefill_block_k(
            valid_q_rows=case.total_q,
            k_rows=int(extend_kv_fp8[0].shape[0]),
            num_heads=cfg.index_n_heads,
        )
        _use_tiled_output = use_tiled_topk
        if _use_tiled_output and _prefill_block_k is None:
            raise BenchmarkFailure(case, "tiled topk requires the prefill indexer path")
        _block_q = (
            _PREFILL512_BLOCK_Q
            if _prefill_block_k == _PREFILL512_BLOCK_K
            else _PREFILL_BLOCK_Q
        )

        _tiled_tile_logits = [None]  # mutable holder for the tiled output
        if _use_tiled_output:
            num_q_tiles = (case.total_q + _block_q - 1) // _block_q
            num_k_tiles = (
                int(extend_kv_fp8[0].shape[0]) + _prefill_block_k - 1
            ) // _prefill_block_k
            tile_size = _block_q * _prefill_block_k
            _tiled_tile_logits[0] = torch.empty(
                (num_q_tiles * num_k_tiles * tile_size,),
                dtype=torch.float32,
                device=device,
            )

        def run_indexer_logits():
            result = contiguous_logits(
                q_fp8=q_fp8,
                weights=weights,
                kv_fp8=extend_kv_fp8,
                metadata=extend_indexer_metadata,
                preinitialize_invalid_logits=preinitialize_indexer_logits,
                tile_logits=_tiled_tile_logits[0] if _use_tiled_output else None,
            )
            if _use_tiled_output:
                _tiled_tile_logits[0] = result
            return result

        def _run_tiled_topk(tile_logits_result=None):
            tl = (
                tile_logits_result
                if tile_logits_result is not None
                else _tiled_tile_logits[0]
            )
            topk_val = min(case.topk, case.cache_len)
            _, topk_indices = run_tiled_supertile_topk(
                tile_logits=tl,
                k_start=graph_contiguous_k_start,
                k_end=graph_contiguous_k_start + graph_contiguous_lengths,
                topk=topk_val,
                block_q=_block_q,
                block_k=_prefill_block_k,
            )
            return topk_indices

        def run_indexer_topk():
            if _use_tiled_output:
                return _run_tiled_topk(logits_for_topk)
            else:
                return _select_ragged_topk_from_logits(
                    logits=logits_for_topk,
                    k_start=graph_contiguous_k_start,
                    lengths=graph_contiguous_lengths,
                    topk=case.topk,
                )

        def run_indexer():
            if _use_tiled_output:
                return contiguous_tiled_topk(
                    q_fp8=q_fp8,
                    weights=weights,
                    kv_fp8=extend_kv_fp8,
                    metadata=extend_indexer_metadata,
                    topk=case.topk,
                )
            result = run_indexer_logits()
            return _select_ragged_topk_from_logits(
                logits=result,
                k_start=graph_contiguous_k_start,
                lengths=graph_contiguous_lengths,
                topk=case.topk,
            )

        clear_indexer_caches()
        actual_topk = run_indexer()
        logits_for_topk = run_indexer_logits()
        if skip_indexer_logits_fill and not _use_tiled_output:
            # In tiled mode, there's no -inf scatter matrix to validate
            torch.cuda.synchronize()
            sample_rows = sorted(
                {0, logits_for_topk.shape[0] // 2, logits_for_topk.shape[0] - 1}
            )
            for sample_row in sample_rows:
                row_start = int(graph_contiguous_k_start[sample_row].item())
                row_end = int(
                    (
                        graph_contiguous_k_start[sample_row]
                        + graph_contiguous_lengths[sample_row]
                    ).item()
                )
                if (
                    row_start > 0
                    and not torch.isneginf(
                        logits_for_topk[sample_row, :row_start]
                    ).all()
                ):
                    raise BenchmarkFailure(
                        case, f"no-fill logits prefix was not -inf for row {sample_row}"
                    )
                if (
                    row_end < logits_for_topk.shape[1]
                    and not torch.isneginf(logits_for_topk[sample_row, row_end:]).all()
                ):
                    raise BenchmarkFailure(
                        case, f"no-fill logits suffix was not -inf for row {sample_row}"
                    )
        if not use_runtime_ragged_topk:
            expected_logits = contiguous_logits_reference(
                q_fp8=q_fp8,
                weights=weights,
                kv_fp8=extend_kv_fp8,
                k_start=graph_contiguous_k_start,
                k_end=graph_contiguous_k_start + graph_contiguous_lengths,
            )
            expected_topk = _select_ragged_topk_from_logits(
                logits=expected_logits,
                k_start=graph_contiguous_k_start,
                lengths=graph_contiguous_lengths,
                topk=case.topk,
            )
            torch.cuda.synchronize()
            _assert_decode_contract_match(
                case=case,
                actual=actual_topk,
                expected=expected_topk,
                page_table_1=actual_topk,
                seqlens=graph_expanded_cache_seqlens,
                topk=case.topk,
            )
            del expected_logits
            del expected_topk
        mla_selected_indices = actual_topk
        mla_kv_cache = extend_kv_cache
        mla_k_nope = contiguous_k_nope
        mla_k_rope = contiguous_k_rope
        mla_metadata_mode = "extend"
        mla_workspace_mode = "extend"

    mla_metadata = MLASparseExtendMetadata(
        selected_token_offsets=mla_selected_indices,
        cache_seqlens_int32=graph_batch_cache_seqlens,
        nsa_cache_seqlens_int32=graph_nsa_cache_seqlens,
        nsa_cu_seqlens_q=cu_seqlens_q,
        nsa_cu_seqlens_k=cu_seqlens_k,
        max_seq_len_q=case.q_len,
        max_seq_len_k=aligned_graph_width,
        mode=mla_metadata_mode,
    )
    mla_binding = _make_mla_binding(
        mode=mla_workspace_mode,
        cfg=cfg,
        device=device,
        topk=case.topk,
        max_total_q=case.total_q,
        max_batch=case.batch_size,
        q_all=q_all,
        selected_indices=mla_metadata.selected_token_offsets,
        cache_seqlens_int32=mla_metadata.cache_seqlens_int32,
        nsa_cache_seqlens_int32=mla_metadata.nsa_cache_seqlens_int32,
    )
    mla_workspace = mla_binding.scratch
    # The integration planner routes every extend/verify binding through the
    # single-pass SM120 prefill kernel. Keep the benchmark report aligned with
    # that planner-owned serving route instead of reviving the legacy split
    # path solely for component timing.
    split_cfg = None

    def run_mla():
        return sparse_mla_extend_forward(
            kv_cache=mla_kv_cache,
            binding=mla_binding,
            sm_scale=cfg.sm_scale,
            v_head_dim=cfg.kv_lora_rank,
        )


    def run_step():
        topk_indices = map_indexer_topk(run_indexer())
        step_binding = mla_workspace.bind(
            q=q_all,
            selected_indices=topk_indices,
            cache_seqlens_int32=graph_batch_cache_seqlens,
            nsa_cache_seqlens_int32=graph_nsa_cache_seqlens,
        )
        return sparse_mla_extend_forward(
            kv_cache=mla_kv_cache,
            binding=step_binding,
            sm_scale=cfg.sm_scale,
            v_head_dim=cfg.kv_lora_rank,
        )

    clear_mla_caches()
    actual_output = run_mla()
    expected_output = dense_mla_reference(
        q_all=q_all,
        k_nope=mla_k_nope,
        k_rope=mla_k_rope,
        page_table_1=mla_selected_indices,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize()
    mla_sanity = _compare(actual_output, expected_output)
    _check_mla_sanity(case=case, label="MLA", metrics=mla_sanity)
    del actual_output
    del expected_output

    clear_indexer_caches()
    indexer_stats = _capture_and_bench_cuda_graph(
        run_indexer,
        warmup=warmup,
        replays=replays,
        prepare=prepare_verify_graph,
        l2_flush=l2_flush,
    )
    indexer_us = statistics.median(indexer_stats["replay_us"])

    if use_paged_prefill:
        # The production paged path is a streaming gather+score+top-k contract;
        # its stages are intentionally not exposed as standalone benchmark APIs.
        indexer_logits_us = 0.0
        indexer_topk_us = 0.0
    else:
        clear_indexer_caches()
        indexer_logits_stats = _capture_and_bench_cuda_graph(
            run_indexer_logits,
            warmup=warmup,
            replays=replays,
            prepare=prepare_verify_graph,
            l2_flush=l2_flush,
        )
        indexer_logits_us = statistics.median(indexer_logits_stats["replay_us"])

        if case.mode == "verify":

            def run_indexer_topk():
                return _select_paged_topk_from_logits(
                    logits=logits_for_topk,
                    page_table_1=base_candidate_page_table,
                    seqlens=graph_expanded_cache_seqlens,
                    topk=case.topk,
                    cu_seqlens_q=cu_seqlens_q,
                    query_row_to_batch=query_row_to_batch,
                )
        elif _use_tiled_output:

            def run_indexer_topk():
                return _run_tiled_topk()
        else:

            def run_indexer_topk():
                return _select_ragged_topk_from_logits(
                    logits=logits_for_topk,
                    k_start=graph_contiguous_k_start,
                    lengths=graph_contiguous_lengths,
                    topk=case.topk,
                )

        indexer_topk_stats = _capture_and_bench_cuda_graph(
            run_indexer_topk,
            warmup=warmup,
            replays=replays,
            prepare=prepare_verify_graph,
            l2_flush=l2_flush,
        )
        indexer_topk_us = statistics.median(indexer_topk_stats["replay_us"])

    clear_mla_caches()
    prepare_verify_graph()
    mla_stats = _capture_and_bench_cuda_graph(
        run_mla,
        warmup=warmup,
        replays=replays,
        l2_flush=l2_flush,
    )
    mla_us = statistics.median(mla_stats["replay_us"])
    mla_forward_us = 0.0
    mla_merge_us = 0.0

    clear_indexer_caches()
    clear_mla_caches()
    step_stats = _capture_and_bench_cuda_graph(
        run_step,
        warmup=warmup,
        replays=replays,
        prepare=prepare_verify_graph,
        l2_flush=l2_flush,
    )
    indexer_prefill_block_k = None
    if use_paged_prefill:
        assert paged_prefill_plan is not None
        indexer_prefill_block_k = paged_prefill_plan.layout.prefill_block_k
    elif case.mode == "prefill":
        assert extend_kv_fp8 is not None
        indexer_prefill_block_k = resolve_contiguous_prefill_block_k(
            valid_q_rows=case.total_q,
            k_rows=int(extend_kv_fp8[0].shape[0]),
            num_heads=cfg.index_n_heads,
        )

    return CaseReport(
        case=case,
        graph_width=graph_width,
        cache_page_stride_bytes=cache_page_stride_bytes,
        step_samples_us=tuple(
            metadata_us + replay_us
            for metadata_us, replay_us in zip(
                step_stats["metadata_us"],
                step_stats["replay_us"],
                strict=True,
            )
        ),
        mla_samples_us=tuple(mla_stats["replay_us"]),

        metadata_us=statistics.median(step_stats["metadata_us"]),
        replay_us=statistics.median(step_stats["replay_us"]),
        indexer_us=indexer_us,
        indexer_logits_us=indexer_logits_us,
        indexer_topk_us=indexer_topk_us,
        mla_us=mla_us,

        mla_forward_us=mla_forward_us,
        mla_merge_us=mla_merge_us,
        split_enabled=split_cfg is not None,
        chunk_size=0 if split_cfg is None else split_cfg.chunk_size,
        num_chunks=0 if split_cfg is None else split_cfg.num_chunks,
        indexer_logits_fill=(
            True
            if case.mode == "verify"
            else False
            if use_paged_prefill
            else not skip_indexer_logits_fill
        ),
        indexer_tiled_topk=_use_tiled_output if case.mode == "prefill" else False,
        indexer_topk_path=(
            "paged-streaming"
            if use_paged_prefill
            else "tiled"
            if _use_tiled_output and case.mode == "prefill"
            else "scatter"
        ),
        indexer_prefill_block_k=indexer_prefill_block_k,
        mla_sanity=mla_sanity,


    )


def collect_case_reports(
    args: argparse.Namespace,
    *,
    device: torch.device | None = None,
) -> list[CaseReport]:
    if getattr(args, "dsa_prefill_block_k", None) is not None:
        os.environ[_DSA_PREFILL_BLOCK_K_ENV] = args.dsa_prefill_block_k
    cfg = _load_glm_contract_config(
        tp_size=args.tp_size,
        tp_rank=args.tp_rank,
        model_config=args.model_config,
    )
    cache_page_stride_bytes = _resolve_cache_page_stride_bytes(
        args.cache_page_stride_bytes,
        cfg,
    )
    device = require_sm120() if device is None else device
    l2_flush = make_l2_flush_fn(args.flush_l2, args.l2_flush_bytes)
    cases = _build_decode_cases(
        modes=[mode for mode in args.modes.split(",") if mode],
        batch_sizes=_parse_csv_ints(args.batch_sizes),
        cache_lens=_parse_csv_ints(args.cache_lens),
        verify_q_lens=_parse_csv_ints(args.verify_q_lens),
        topk_cap=min(args.topk_cap, cfg.index_topk),
        decode_row_pattern=args.decode_row_pattern,
        page_size=cfg.page_size,
    )
    reports: list[CaseReport] = []
    case_seed = args.seed
    for case in cases:
        reports.append(
            _run_decode_case(
                case=case,
                cfg=cfg,
                warmup=args.warmup,
                replays=args.replays,
                seed=case_seed,
                device=device,
                pool_factor=args.pool_factor,
                graph_width=args.graph_width,
                cache_page_stride_bytes=cache_page_stride_bytes,
                l2_flush=l2_flush,
                skip_indexer_logits_fill=args.skip_indexer_logits_fill,
                decode_topk_backend=args.decode_topk_backend,
                )
            if case.mode == "decode"
            else _run_prefill_or_verify_case(
                case=case,
                cfg=cfg,
                warmup=args.warmup,
                replays=args.replays,
                seed=case_seed,
                device=device,
                pool_factor=args.pool_factor,
                graph_width=args.graph_width,
                cache_page_stride_bytes=cache_page_stride_bytes,
                l2_flush=l2_flush,
                skip_indexer_logits_fill=args.skip_indexer_logits_fill,
                use_tiled_topk=args.use_tiled_topk,
                prefill_indexer_layout=args.prefill_indexer_layout,
                )
        )
        case_seed += 17
    return reports


def _render_case_line(report: CaseReport) -> str:
    split_flag = "on" if report.split_enabled else "off"
    fill_flag = "fill" if report.indexer_logits_fill else "skipfill"
    topk_path = report.indexer_topk_path or (
        "tiled" if report.indexer_tiled_topk else "scatter"
    )
    if report.indexer_prefill_block_k is not None:
        idx_bk = str(report.indexer_prefill_block_k)
    else:
        idx_bk = "decode" if report.case.mode == "prefill" else "paged"
    row_ctx_desc = ""
    if report.case.mode == "decode" and report.case.is_heterogeneous_decode:
        row_ctx_desc = f" rowctx={min(report.case.decode_row_cache_lens):6d}-{report.case.cache_len:6d}"
    sanity_desc = ""
    if report.mla_sanity is not None:
        sanity_desc = (
            f" mla_max_abs={report.mla_sanity.max_abs:.6g}"
            f" mla_rmse={report.mla_sanity.rmse:.6g}"
            f" mla_cos={report.mla_sanity.cos:.7f}"
        )
    return (
        f"glm52-{report.case.mode:6s} tp8 bs={report.case.batch_size:2d} "
        f"q={report.case.q_len:2d} ctx={report.case.cache_len:6d}{row_ctx_desc} "
        f"graphw={report.graph_width:6d} cache_stride={report.cache_page_stride_bytes:d} "
        f"topk={report.case.topk:4d} split={split_flag:>3s} "
        f"chunk={report.chunk_size:3d} nchunks={report.num_chunks:d} | "
        f"step={report.total_us:8.2f} us | "
        f"total={report.total_us:8.2f} us | "
        f"meta={report.metadata_us:8.2f} us | "
        f"replay={report.replay_us:8.2f} us | "
        f"indexer={report.indexer_us:8.2f} us | "
        f"idx_logits={report.indexer_logits_us:8.2f} us | "
        f"idx_topk={report.indexer_topk_us:8.2f} us | "
        f"mla={report.mla_us:8.2f} us | "
        f"mla_fwd={report.mla_forward_us:8.2f} us | "
        f"mla_merge={report.mla_merge_us:8.2f} us | "
        f"idx_bk={idx_bk} "
        f"idx_init={fill_flag} "
        f"idx_topk_path={topk_path}"
        f"{sanity_desc}"
    )


def _render_summary_lines(reports: list[CaseReport]) -> list[str]:
    total_geo = _geomean([report.total_us for report in reports])
    metadata_geo = _geomean([report.metadata_us for report in reports])
    replay_geo = _geomean([report.replay_us for report in reports])
    indexer_geo = _geomean([report.indexer_us for report in reports])
    indexer_logits_geo = _geomean([report.indexer_logits_us for report in reports])
    indexer_topk_geo = _geomean([report.indexer_topk_us for report in reports])
    mla_geo = _geomean([report.mla_us for report in reports])
    mla_forward_geo = _geomean([report.mla_forward_us for report in reports])
    mla_merge_geo = _geomean([report.mla_merge_us for report in reports])
    lines = [
        "Summary",
        f"  cases: {len(reports)}",
        f"  total geo:   {total_geo:.2f} us",
        f"  indexer geo: {indexer_geo:.2f} us",
        f"  mla geo:     {mla_geo:.2f} us",
        f"  step geo:    {total_geo:.2f} us",
        f"  meta geo:    {metadata_geo:.2f} us",
        f"  replay geo:  {replay_geo:.2f} us",
        f"  idx logits:  {indexer_logits_geo:.2f} us",
        f"  idx topk:    {indexer_topk_geo:.2f} us",
        f"  mla fwd:     {mla_forward_geo:.2f} us",
        f"  mla merge:   {mla_merge_geo:.2f} us",
    ]
    return lines


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=(
            "none",
            TARGET_PREFILL64K_BS1_PRESET,
            TARGET_GLM52_PREFILL4K_CTX16K_PRESET,
            TARGET_DSV4_TRACE_PRESET,
        ),
        default="none",
        help=(
            "shape preset; target-glm52-prefill4k-ctx16k maps to a vLLM-oriented "
            "TP8 4096-token chunk at 16k context with top-k 2048 and tiled top-k; "
            "target-dsv4-trace runs the TP2 C1/C4/C128 compressed-MLA trace race"
        ),
    )
    parser.add_argument(
        "--model-config",
        type=pathlib.Path,
        default=None,
        help=(
            "Hugging Face config.json override; by default resolve "
            f"{DEFAULT_GLM52_HF_REPO_ID} from the local Hugging Face cache"
        ),
    )
    parser.add_argument(
        "--modes",
        default="decode,prefill",
        help="benchmark modes to run: decode, prefill, verify, or a csv mix (default: decode,prefill)",
    )
    parser.add_argument(
        "--batch-sizes",
        default=_format_csv_ints(DEFAULT_BATCH_SIZES),
        help=f"decode batch sizes, default {','.join(str(v) for v in DEFAULT_BATCH_SIZES)}",
    )
    parser.add_argument(
        "--cache-lens",
        default="1024,32768,131072",
        help="decode cache lengths, default 1024,32768,131072",
    )
    parser.add_argument(
        "--decode-row-pattern",
        default=DEFAULT_DECODE_ROW_PATTERN,
        help=(
            "decode-only per-row context pattern: uniform or staggered "
            "(staggered uses row contexts [ctx, ctx*(bs-1)/bs, ..., ctx/bs])"
        ),
    )
    parser.add_argument(
        "--verify-q-lens",
        "--prefill-q-lens",
        dest="verify_q_lens",
        default="16384",
        help="prefill/verify chunk q lengths, default 16384",
    )
    parser.add_argument("--topk-cap", type=int, default=2048)
    parser.add_argument(
        "--decode-topk-backend",
        choices=("auto", "sgl", "torch", "cute-persistent"),
        default=os.getenv(_DSA_DECODE_TOPK_BACKEND_ENV, "auto").replace("_", "-"),
        help=(
            "decode topk selector backend; cute-persistent enables the experimental "
            "large-N CuTe persistent TopK=2048 path"
        ),
    )
    parser.add_argument("--tp-size", type=int, default=DEFAULT_TP_SIZE)
    parser.add_argument("--tp-rank", type=int, default=DEFAULT_TP_RANK)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument(
        "--print-raw-samples",
        action="store_true",
        help="print every step, b12x MLA, and MLA sample",
    )
    parser.add_argument("--seed", type=int, default=70_000)
    parser.add_argument("--pool-factor", type=int, default=DEFAULT_POOL_FACTOR)
    parser.add_argument(
        "--graph-width",
        type=int,
        default=DEFAULT_GRAPH_WIDTH,
        help="decode graph candidate-table width; actual width is max(cache_len, graph_width)",
    )
    parser.add_argument("--flush-l2", action="store_true", default=True)
    parser.add_argument("--no-flush-l2", action="store_false", dest="flush_l2")
    parser.add_argument(
        "--l2-flush-bytes",
        type=int,
        default=0,
        help="L2 eviction size in bytes; default is 2x detected L2 capacity.",
    )
    parser.add_argument(
        "--skip-indexer-logits-fill",
        action="store_true",
        default=True,
        help=(
            "skip the prefill DSA logits -inf initialization when the logits are consumed "
            "only by ragged topk; the benchmark validates sampled invalid logits regions"
        ),
    )
    parser.add_argument(
        "--keep-indexer-logits-fill",
        action="store_false",
        dest="skip_indexer_logits_fill",
        help="preserve the prefill DSA logits -inf initialization.",
    )
    parser.add_argument(
        "--use-tiled-topk",
        action="store_true",
        help="benchmark prefill with tiled indexer output consumed directly by the CuTe topk kernel.",
    )
    parser.add_argument(
        "--prefill-indexer-layout",
        choices=("contiguous", "paged"),
        default="contiguous",
        help=(
            "prefill indexer source layout; paged matches vLLM's shared-page-table "
            "production route"
        ),
    )
    parser.add_argument(
        "--cache-page-stride-bytes",
        type=int,
        default=None,
        help=(
            "physical stride between pages of one vLLM cache view; 0 keeps "
            "the normal per-layer contiguous GLM layout, while -1 derives an "
            "aggregate all-layer stride for packed-layout regression testing"
        ),
    )
    parser.add_argument(
        "--dsa-prefill-block-k",
        choices=("auto", "256", "512"),
        default=None,
        help=(
            "override B12X_DSA_CONTIGUOUS_PREFILL_BLOCK_K for contiguous/prefill DSA logits; "
            "default leaves the environment unchanged"
        ),
    )
    return _apply_benchmark_preset(parser.parse_args(argv))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.preset == TARGET_DSV4_TRACE_PRESET:
        from benchmarks.experimental.b12x.benchmark_compressed_sparse_mla import (
            main as compressed_sparse_mla_main,
        )

        forwarded = [
            "--preset",
            "vllm-dsv4-trace",
            "--warmup",
            str(args.warmup),
            "--replays",
            str(args.replays),
            "--seed",
            str(args.seed),
            "--l2-flush-bytes",
            str(args.l2_flush_bytes),
        ]
        if not args.flush_l2:
            forwarded.append("--no-flush-l2")
        if args.print_raw_samples:
            forwarded.append("--print-raw-samples")
        if args.model_config is not None:
            forwarded.extend(("--model-config", str(args.model_config)))
        return compressed_sparse_mla_main(forwarded)

    device = require_sm120()
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)
    flush_desc = (
        f"on ({l2_flush_bytes / (1 << 20):.1f} MiB per launch)"
        if args.flush_l2
        else "off"
    )
    print(f"L2 flush: {flush_desc}")
    try:
        reports = collect_case_reports(args, device=device)
    except BenchmarkFailure as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for report in reports:
        print(_render_case_line(report))
        if args.print_raw_samples:
            raw_us = ",".join(f"{sample:.2f}" for sample in report.step_samples_us)
            print(f"  step_raw_us=[{raw_us}]")
            mla_raw_us = ",".join(f"{sample:.2f}" for sample in report.mla_samples_us)
            print(f"  mla_raw_us=[{mla_raw_us}]")
    for line in _render_summary_lines(reports):
        print(line)
    return 0


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    raise SystemExit(main())
