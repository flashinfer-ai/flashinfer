#!/usr/bin/env python3
"""Benchmark realistic SGLang-like DSA decode plus eager-prefill chunks."""

from __future__ import annotations

import argparse
import functools
import json
import pathlib
import statistics
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch

from b12x.attention.dsa_indexer.reference import (
    contiguous_logits_reference,
    pack_index_k_cache_reference,
    paged_decode_logits_reference,
)
from b12x.attention.dsa_indexer._impl import (
    build_paged_mqa_schedule_metadata,
    clear_indexer_caches,
    contiguous_tiled_topk,
    paged_decode_logits,
    uses_paged_mqa_schedule,
)
from b12x.attention.dsa_indexer.scratch import (
    INDEXER_SOURCE_LAYOUT_CONTIGUOUS,
    INDEXER_SOURCE_LAYOUT_PAGED,
    B12XIndexerScratchCaps,
    plan_indexer_scratch,
)

from benchmarks.experimental.b12x.common import (
    bench_cuda_graph,
    capture_cuda_graph,
    make_dense_candidate_page_table,
    make_dense_real_page_table,
    make_l2_flush_fn,
    make_sparse_pool_locs,
    require_sm120,
    resolve_l2_flush_bytes,
    scatter_rows_into_pool,
)


MODEL_PATH = pathlib.Path("/data/models/GLM-5.1-NVFP4")
DEFAULT_POOL_FACTOR = 6
DEFAULT_GRAPH_WIDTH = 8192
DEFAULT_TOPK = 2048
# K rows per contiguous extend logits tile (the contiguous scratch default).
DEFAULT_EXTEND_SUPERTILE_K = 32_768
DEFAULT_EXTEND_Q_LENS = (16384,)


@dataclass(frozen=True)
class GLMNSAConfig:
    num_heads: int
    head_dim: int = 128
    page_size: int = 64


def _align_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _allocate_plan_scratch(plan):
    return [
        torch.empty(shape, dtype=dtype, device=plan.caps.device)
        for shape, dtype in plan.shapes_and_dtypes()
    ]


@functools.lru_cache(maxsize=1)
def _load_glm_config() -> GLMNSAConfig:
    config_path = MODEL_PATH / "config.json"
    if not config_path.exists():
        raise SystemExit(f"GLM-5.1 config not found at {config_path}")
    config = json.loads(config_path.read_text())
    return GLMNSAConfig(num_heads=int(config["index_n_heads"]))


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _make_q_and_weights(
    *,
    rows: int,
    cfg: GLMNSAConfig,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    del seed
    q_fp8 = torch.full(
        (rows, cfg.num_heads, cfg.head_dim),
        0.5,
        dtype=torch.float32,
        device=device,
    ).to(torch.float8_e4m3fn)
    weights = torch.ones((rows, cfg.num_heads, 1), dtype=torch.float32, device=device)
    return q_fp8, weights


def _make_index_k_cache(
    *,
    active_tokens: int,
    pool_locs: torch.Tensor,
    pool_tokens: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    del seed
    token_scores = torch.linspace(
        0.25,
        1.25,
        active_tokens,
        dtype=torch.float32,
        device=device,
    )
    k = token_scores.unsqueeze(1).expand(-1, 128).contiguous()
    k_pool = scatter_rows_into_pool(k, pool_locs=pool_locs, pool_tokens=pool_tokens)
    return pack_index_k_cache_reference(k_pool)


def _make_page_table(
    *,
    rows: int,
    width: int,
    valid_per_row: int,
    token_locs: torch.Tensor,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if width <= 0:
        raise ValueError("width must be positive")
    if valid_per_row <= 0 or valid_per_row > width:
        raise ValueError("valid_per_row must be in [1, width]")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    token_locs_cpu = token_locs.to("cpu")
    num_tokens = int(token_locs_cpu.numel())
    out = torch.full((rows, width), -1, dtype=torch.int32)
    for row in range(rows):
        perm = torch.randperm(num_tokens, generator=gen, dtype=torch.int64)[
            :valid_per_row
        ]
        ids = token_locs_cpu[perm].to(torch.int32)
        out[row, :valid_per_row] = ids
    return out.to(device=device)


def _assert_exact_match(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if torch.equal(actual, expected):
        return
    mismatch = int((actual != expected).sum().item())
    raise AssertionError(
        f"DSA indexer correctness mismatch: {mismatch} differing entries, "
        f"actual[0]={actual[0].tolist()} expected[0]={expected[0].tolist()}"
    )


def _assert_topk_set_match(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_sorted = torch.sort(actual, dim=1).values
    expected_sorted = torch.sort(expected, dim=1).values
    _assert_exact_match(actual_sorted, expected_sorted)


def _select_paged_topk_from_logits(
    *,
    logits: torch.Tensor,
    page_table_1: torch.Tensor,
    seqlens: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    rows = logits.shape[0]
    output = torch.full((rows, topk), -1, dtype=torch.int32, device=logits.device)
    gather_k = min(topk, logits.shape[1], page_table_1.shape[1])
    if gather_k == 0:
        return output
    topk_pos = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :gather_k]
    topk_values = torch.gather(logits, 1, topk_pos)
    gathered = torch.gather(page_table_1, 1, topk_pos.to(torch.long))
    output[:, :gather_k] = torch.where(
        torch.isfinite(topk_values),
        gathered,
        torch.full_like(gathered, -1),
    )
    return output


def _select_ragged_topk_from_logits(
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
    positions = torch.arange(
        logits.shape[1], device=logits.device, dtype=torch.int32
    ).unsqueeze(0)
    row_start = k_start.unsqueeze(1)
    row_end = row_start + lengths.unsqueeze(1)
    valid = (positions >= row_start) & (positions < row_end)
    masked_logits = torch.where(valid, logits, torch.full_like(logits, float("-inf")))
    topk_pos = torch.argsort(masked_logits, dim=1, descending=True, stable=True)[
        :, :gather_k
    ]
    topk_values = torch.gather(masked_logits, 1, topk_pos)
    output[:, :gather_k] = torch.where(
        torch.isfinite(topk_values),
        topk_pos.to(torch.int32),
        torch.full_like(topk_pos, -1, dtype=torch.int32),
    )
    return output


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


def _resolve_graph_width(*, cache_len: int, graph_width: int) -> int:
    if graph_width <= 0:
        raise ValueError(f"graph_width must be positive, got {graph_width}")
    return max(cache_len, graph_width)


def _run_decode_case(
    *,
    cfg: GLMNSAConfig,
    q_rows: int,
    cache_len: int,
    width: int,
    topk: int,
    warmup: int,
    replays: int,
    seed: int,
    device: torch.device,
    pool_factor: int,
    l2_flush,
) -> None:
    graph_width = _resolve_graph_width(cache_len=cache_len, graph_width=width)
    pool_tokens = _align_up(max(cache_len, cache_len * pool_factor), cfg.page_size)
    pool_locs = make_sparse_pool_locs(
        active_tokens=cache_len,
        pool_tokens=pool_tokens,
        seed=seed + 10,
        device=device,
        page_size=cfg.page_size,
    )
    q_fp8, weights = _make_q_and_weights(rows=q_rows, cfg=cfg, seed=seed, device=device)
    index_k_cache = _make_index_k_cache(
        active_tokens=cache_len,
        pool_locs=pool_locs,
        pool_tokens=pool_tokens,
        seed=seed + 1,
        device=device,
    )
    live_page_table_1 = make_dense_candidate_page_table(
        batch_size=q_rows,
        token_locs=pool_locs,
        width=cache_len,
        fill_value=-1,
    )
    live_real_page_table = make_dense_real_page_table(
        batch_size=q_rows,
        token_locs=pool_locs,
        width_blocks=_align_up(graph_width, cfg.page_size) // cfg.page_size,
        page_size=cfg.page_size,
    )
    seqlens = torch.full((q_rows,), cache_len, dtype=torch.int32, device=device)
    graph_page_table_1 = torch.full(
        (q_rows, _align_up(graph_width, cfg.page_size)),
        -1,
        dtype=torch.int32,
        device=device,
    )
    graph_real_page_table = torch.full(
        (q_rows, _align_up(graph_width, cfg.page_size) // cfg.page_size),
        -1,
        dtype=torch.int32,
        device=device,
    )
    graph_seqlens = torch.empty_like(seqlens)
    use_graph_schedule_metadata = uses_paged_mqa_schedule(
        q_rows=q_rows,
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

    def prepare_decode_graph() -> None:
        graph_page_table_1[:, :cache_len].copy_(live_page_table_1)
        graph_real_page_table[:, : live_real_page_table.shape[1]].copy_(
            live_real_page_table
        )
        graph_seqlens.copy_(seqlens)
        if graph_schedule_metadata is not None:
            build_paged_mqa_schedule_metadata(
                graph_seqlens,
                cfg.page_size,
                out=graph_schedule_metadata,
            )

    prepare_decode_graph()
    decode_plan = plan_indexer_scratch(
        B12XIndexerScratchCaps(
            device=device,
            source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
            num_q_heads=cfg.num_heads,
            max_q_rows=q_rows,
            max_page_table_width=int(graph_real_page_table.shape[1]),
            topk=topk,
            page_size=cfg.page_size,
            mode="decode",
        )
    )
    decode_binding = decode_plan.bind(
        scratch=_allocate_plan_scratch(decode_plan),
        real_page_table=graph_real_page_table,
        cache_seqlens_int32=graph_seqlens,
        schedule_metadata=graph_schedule_metadata,
        expected_num_q_heads=cfg.num_heads,
    )

    def run():
        logits = paged_decode_logits(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            binding=decode_binding,
            page_size=cfg.page_size,
        )
        return _select_paged_topk_from_logits(
            logits=logits,
            page_table_1=graph_page_table_1,
            seqlens=graph_seqlens,
            topk=topk,
        )

    clear_indexer_caches()
    actual = run()
    expected_logits = paged_decode_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=graph_real_page_table,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens_per_query=graph_seqlens,
        page_size=cfg.page_size,
    )
    expected = _select_paged_topk_from_logits(
        logits=expected_logits,
        page_table_1=graph_page_table_1,
        seqlens=graph_seqlens,
        topk=topk,
    )
    torch.cuda.synchronize()
    _assert_exact_match(actual, expected)

    graph = capture_cuda_graph(
        run,
        warmup=warmup,
        prepare=prepare_decode_graph,
    )
    stats = bench_cuda_graph(
        graph,
        replays=replays,
        prepare=prepare_decode_graph,
        l2_flush=l2_flush,
    )
    print(
        json.dumps(
            {
                "contract": "sglang_decode_graph",
                "mode": "decode",
                "q_rows": q_rows,
                "cache_len": cache_len,
                "graph_width": graph_width,
                "graph_width_blocks": graph_real_page_table.shape[1],
                "topk": topk,
                "pool_tokens": pool_tokens,
                "metadata_median_us": statistics.median(stats["metadata_us"]),
                "replay_median_us": statistics.median(stats["replay_us"]),
                "step_median_us": statistics.median(stats["step_us"]),
                "replay_mean_us": statistics.fmean(stats["replay_us"]),
                "replay_min_us": min(stats["replay_us"]),
                "replay_max_us": max(stats["replay_us"]),
                "replays": replays,
                "l2_flush_enabled": l2_flush is not None,
            }
        )
    )


def _run_extend_case(
    *,
    cfg: GLMNSAConfig,
    batch: int,
    q_len: int,
    cache_len: int,
    width: int,
    topk: int,
    warmup: int,
    replays: int,
    seed: int,
    device: torch.device,
    pool_factor: int,
    l2_flush,
    supertile_k: int = DEFAULT_EXTEND_SUPERTILE_K,
) -> None:
    """Time the contiguous extend path for ``batch`` requests of ``q_len`` tokens.

    ``supertile_k`` bounds the K rows scored per logits tile; the tile holds
    ``batch * q_len`` rows by ``supertile_k`` scores, so an unbounded tile
    over every request's K rows would not fit a serving GPU at large batches.
    """
    total_q = batch * q_len
    if q_len > cache_len:
        raise ValueError(f"extend q_len {q_len} must not exceed cache_len {cache_len}")
    pool_tokens = _align_up(max(cache_len, cache_len * pool_factor), cfg.page_size)
    pool_locs = make_sparse_pool_locs(
        active_tokens=cache_len,
        pool_tokens=pool_tokens,
        seed=seed + 10,
        device=device,
        page_size=cfg.page_size,
    )
    q_fp8, weights = _make_q_and_weights(
        rows=total_q, cfg=cfg, seed=seed, device=device
    )
    index_k_cache = _make_index_k_cache(
        active_tokens=cache_len,
        pool_locs=pool_locs,
        pool_tokens=pool_tokens,
        seed=seed + 1,
        device=device,
    )
    valid_per_row = min(width, cache_len)
    _page_table_1 = make_dense_candidate_page_table(
        batch_size=batch,
        token_locs=pool_locs[:valid_per_row],
        width=valid_per_row,
        fill_value=-1,
    )
    real_page_table = make_dense_real_page_table(
        batch_size=batch,
        token_locs=pool_locs[:valid_per_row],
        width_blocks=_align_up(valid_per_row, cfg.page_size) // cfg.page_size,
        page_size=cfg.page_size,
    )
    seq_lens = torch.full((batch,), valid_per_row, dtype=torch.int32, device=device)
    kv_fp8 = _make_extend_kv_fp8(
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        seq_lens=seq_lens,
        page_size=cfg.page_size,
    )
    _contiguous_lengths = [q_len] * batch
    batch_offsets = (
        torch.arange(batch, dtype=torch.int32, device=device) * valid_per_row
    )
    k_start = torch.repeat_interleave(batch_offsets, q_len)
    per_request_ke = torch.arange(
        valid_per_row - q_len + 1,
        valid_per_row + 1,
        dtype=torch.int32,
        device=device,
    )
    seqlens_expanded = per_request_ke.repeat(batch)
    contiguous_plan = plan_indexer_scratch(
        B12XIndexerScratchCaps(
            device=device,
            source_layout=INDEXER_SOURCE_LAYOUT_CONTIGUOUS,
            num_q_heads=cfg.num_heads,
            max_q_rows=total_q,
            max_k_rows=int(kv_fp8[0].shape[0]),
            topk=topk,
            supertile_k=int(supertile_k),
        )
    )
    contiguous_binding = contiguous_plan.bind(
        scratch=_allocate_plan_scratch(contiguous_plan),
        k_start=k_start,
        k_end=k_start + seqlens_expanded,
        gather_rows=int(kv_fp8[0].shape[0]),
        topk=topk,
    )
    contiguous_scratch = contiguous_binding.scratch
    contiguous_scratch.k_quant[: int(kv_fp8[0].shape[0])].copy_(kv_fp8[0])
    contiguous_scratch.k_scale[: int(kv_fp8[1].shape[0])].copy_(kv_fp8[1])
    bound_kv_fp8 = (
        contiguous_scratch.k_quant[: int(kv_fp8[0].shape[0])],
        contiguous_scratch.k_scale[: int(kv_fp8[1].shape[0])],
    )

    def run():
        return contiguous_tiled_topk(
            q_fp8=q_fp8,
            weights=weights,
            kv_fp8=bound_kv_fp8,
            binding=contiguous_binding,
        )

    clear_indexer_caches()
    actual = run()
    # Reference per request: a request's rows see only its own K range, so
    # the dense [total_q, all K rows] logits matrix is never materialized.
    expected_chunks = []
    for request in range(batch):
        rows = slice(request * q_len, (request + 1) * q_len)
        k_base = int(batch_offsets[request])
        k_rows = slice(k_base, k_base + valid_per_row)
        local_k_start = k_start[rows] - k_base
        request_logits = contiguous_logits_reference(
            q_fp8=q_fp8[rows],
            weights=weights[rows],
            kv_fp8=(kv_fp8[0][k_rows], kv_fp8[1][k_rows]),
            k_start=local_k_start,
            k_end=local_k_start + seqlens_expanded[rows],
        )
        selected = _select_ragged_topk_from_logits(
            logits=request_logits,
            k_start=local_k_start,
            lengths=seqlens_expanded[rows],
            topk=topk,
        )
        expected_chunks.append(torch.where(selected >= 0, selected + k_base, selected))
    expected = torch.cat(expected_chunks, dim=0)
    torch.cuda.synchronize()
    _assert_topk_set_match(actual[: expected.shape[0]], expected)

    from b12x.testing.benchmark import measure_call

    measured = measure_call(
        run, warmup=warmup, samples=replays, eviction=l2_flush or (lambda: None)
    )
    replay_us = measured.raw_samples("workload")
    print(
        json.dumps(
            {
                "contract": "sglang_extend_eager",
                "mode": "extend",
                "batch": batch,
                "q_len": q_len,
                "shape": "eager_prefill_chunk",
                "cache_len": cache_len,
                "width": width,
                "topk": topk,
                "pool_tokens": pool_tokens,
                "median_us": statistics.median(replay_us),
                "mean_us": statistics.fmean(replay_us),
                "min_us": min(replay_us),
                "max_us": max(replay_us),
                "replays": replays,
                "l2_flush_enabled": l2_flush is not None,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("decode", "extend", "both"), default="decode"
    )
    parser.add_argument("--decode-rows", default="1,16")
    parser.add_argument("--extend-batches", default="8")
    parser.add_argument(
        "--extend-q-lens",
        default="16384",
        help=f"eager-prefill chunk q lengths, default {','.join(str(v) for v in DEFAULT_EXTEND_Q_LENS)}",
    )
    parser.add_argument("--cache-lens", default="1024,32768,131072")
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_GRAPH_WIDTH,
        help="decode graph candidate-table width; actual width is max(cache_len, width)",
    )
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument(
        "--extend-supertile-k",
        type=int,
        default=DEFAULT_EXTEND_SUPERTILE_K,
        help="K rows per contiguous extend logits tile",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--seed", type=int, default=88_000)
    parser.add_argument("--pool-factor", type=int, default=DEFAULT_POOL_FACTOR)
    parser.add_argument("--flush-l2", action="store_true", default=True)
    parser.add_argument("--no-flush-l2", action="store_false", dest="flush_l2")
    parser.add_argument(
        "--l2-flush-bytes",
        type=int,
        default=0,
        help="L2 eviction size in bytes; default is 2x detected L2 capacity.",
    )
    args = parser.parse_args()

    device = require_sm120()
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)
    l2_flush = make_l2_flush_fn(args.flush_l2, args.l2_flush_bytes)
    flush_desc = (
        f"on ({l2_flush_bytes / (1 << 20):.1f} MiB per launch)" if l2_flush else "off"
    )
    print(f"L2 flush: {flush_desc}", file=sys.stderr)
    cfg = _load_glm_config()
    cache_lens = _parse_csv_ints(args.cache_lens)
    decode_rows = _parse_csv_ints(args.decode_rows)
    extend_batches = _parse_csv_ints(args.extend_batches)
    contiguous_q_lens = _parse_csv_ints(args.extend_q_lens)

    case_seed = args.seed
    if args.mode in ("decode", "both"):
        for cache_len in cache_lens:
            for q_rows in decode_rows:
                _run_decode_case(
                    cfg=cfg,
                    q_rows=q_rows,
                    cache_len=cache_len,
                    width=args.width,
                    topk=args.topk,
                    warmup=args.warmup,
                    replays=args.replays,
                    seed=case_seed,
                    device=device,
                    pool_factor=args.pool_factor,
                    l2_flush=l2_flush,
                )
                case_seed += 17
    if args.mode in ("extend", "both"):
        for cache_len in cache_lens:
            for batch in extend_batches:
                for q_len in contiguous_q_lens:
                    if q_len > cache_len:
                        print(
                            f"skip extend batch={batch} q_len={q_len} cache_len={cache_len}: "
                            "one eager prefill chunk cannot exceed the active KV span",
                            file=sys.stderr,
                        )
                        continue
                    _run_extend_case(
                        cfg=cfg,
                        batch=batch,
                        q_len=q_len,
                        cache_len=cache_len,
                        width=args.width,
                        topk=args.topk,
                        warmup=args.warmup,
                        replays=args.replays,
                        seed=case_seed,
                        device=device,
                        pool_factor=args.pool_factor,
                        l2_flush=l2_flush,
                        supertile_k=args.extend_supertile_k,
                    )
                    case_seed += 17


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
