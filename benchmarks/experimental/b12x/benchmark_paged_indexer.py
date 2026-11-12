#!/usr/bin/env python3
"""Benchmark the vLLM-shaped paged-indexer FP8 top-k path."""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch

from benchmarks.experimental.b12x.common import make_l2_flush_fn
import b12x.attention.dsa_indexer.tiled_topk as tiled_topk_module
from b12x.attention.dsa_indexer._impl import uses_paged_mqa_schedule
from b12x.attention.dsa_indexer.kernel import (
    run_paged_supertile_logits_kernel,
)
from b12x.attention.dsa_indexer._impl import clear_indexer_caches
from b12x.attention.dsa_indexer.paged import (
    index_topk_fp8,
    pack_paged_index_k_cache_reference,
    prepare_paged_indexer_metadata,
    resolve_replicated_num_q_heads,
)
from b12x.attention.dsa_indexer.scratch import (
    INDEXER_SOURCE_LAYOUT_PAGED,
    B12XIndexerScratchCaps,
    plan_indexer_scratch,
)


def _make_page_table(
    *,
    rows: int,
    page_table_width: int,
    seq_len: int,
    page_stride: int,
    device: torch.device,
) -> torch.Tensor:
    pages_per_row = min((int(seq_len) + 63) // 64, int(page_table_width))
    if int(page_stride) == 0:
        # vLLM's single-request prefill path expands one row without packing;
        # b12x receives a stride-0 row-shared page table.
        table = torch.full((1, page_table_width), -1, dtype=torch.int32, device=device)
        table[0, :pages_per_row] = torch.arange(
            pages_per_row, dtype=torch.int32, device=device
        )
        return table.expand(rows, -1)

    table = torch.full((rows, page_table_width), -1, dtype=torch.int32, device=device)
    for row in range(rows):
        start = row * int(page_stride)
        table[row, :pages_per_row] = torch.arange(
            start,
            start + pages_per_row,
            dtype=torch.int32,
            device=device,
        )
    return table.contiguous()


def _physical_slots(
    page_ids: torch.Tensor,
    page_offsets: torch.Tensor,
) -> torch.Tensor:
    """Return page-based token slots without 32-bit offset overflow."""
    return page_ids.to(torch.int64) * 64 + page_offsets.to(torch.int64)


def _validate_analytic_topk(
    *,
    indices: torch.Tensor,
    scores: torch.Tensor | None,
    seqlens: torch.Tensor,
    topk: int,
    analytic_scores: torch.Tensor,
    real_page_table: torch.Tensor,
    output_physical_slots: bool,
) -> float | None:
    """Validate every target-shape row against a cheap analytic top-k oracle."""
    columns = torch.arange(topk, dtype=torch.int32, device=indices.device)
    valid_counts = torch.minimum(seqlens, torch.full_like(seqlens, int(topk)))
    valid_mask = indices >= 0
    actual_counts = valid_mask.sum(dim=1)
    bad_counts = actual_counts != valid_counts
    if bool(bad_counts.any().item()):
        row = int(torch.nonzero(bad_counts, as_tuple=False)[0, 0].item())
        raise AssertionError(
            "analytic top-k valid-count mismatch: "
            f"row={row} actual={int(actual_counts[row].item())} "
            f"expected={int(valid_counts[row].item())}"
        )
    if bool((indices[~valid_mask] != -1).any().item()):
        raise AssertionError("analytic top-k invalid entries must be -1")

    expected_valid = columns.unsqueeze(0) < valid_counts.unsqueeze(1)
    expected_logical = (
        seqlens.unsqueeze(1) - valid_counts.unsqueeze(1) + columns.unsqueeze(0)
    )
    expected_page_cols = torch.div(expected_logical, 64, rounding_mode="floor")
    expected_page_offsets = torch.remainder(expected_logical, 64)
    page_table_width = int(real_page_table.shape[1])
    if page_table_width <= 0:
        raise ValueError("analytic top-k validation requires a non-empty page table")
    safe_page_cols = expected_page_cols.clamp(min=0, max=page_table_width - 1).to(
        torch.int64
    )
    expected_page_ids = torch.gather(real_page_table, 1, safe_page_cols)
    expected_physical = _physical_slots(expected_page_ids, expected_page_offsets)
    expected_indices = expected_physical if output_physical_slots else expected_logical
    expected_indices = torch.where(
        expected_valid, expected_indices, torch.full_like(expected_indices, -1)
    )
    actual_sorted = torch.sort(indices, dim=1).values
    expected_sorted = torch.sort(expected_indices, dim=1).values
    bad_rows = (actual_sorted != expected_sorted).any(dim=1)
    if bool(bad_rows.any().item()):
        row = int(torch.nonzero(bad_rows, as_tuple=False)[0, 0].item())
        actual_valid_indices = indices[row][valid_mask[row]]
        expected_valid_indices = expected_indices[row][expected_valid[row]]
        actual_set = set(actual_valid_indices.tolist())
        expected_set = set(expected_valid_indices.tolist())
        missing = sorted(expected_set - actual_set)[:8]
        extra = sorted(actual_set - expected_set)[:8]
        raise AssertionError(
            "analytic top-k index oracle mismatch: "
            f"row={row} missing={missing} extra={extra} "
            f"actual_unique={len(actual_set)} expected_unique={len(expected_set)}"
        )

    if scores is None:
        return None

    safe_indices = indices.clamp(min=0)
    if output_physical_slots:
        score_indices = safe_indices
    else:
        actual_page_cols = torch.div(safe_indices, 64, rounding_mode="floor")
        actual_page_offsets = torch.remainder(safe_indices, 64)
        actual_page_ids = torch.gather(
            real_page_table, 1, actual_page_cols.to(torch.int64)
        )
        score_indices = _physical_slots(actual_page_ids, actual_page_offsets)
    expected_scores = analytic_scores[score_indices.to(torch.int64)]
    actual_valid_scores = scores[valid_mask]
    expected_valid_scores = expected_scores[valid_mask]
    if not bool(torch.isfinite(actual_valid_scores).all().item()):
        raise AssertionError("top-k valid scores are non-finite")
    max_abs = float((actual_valid_scores - expected_valid_scores).abs().max().item())
    torch.testing.assert_close(
        actual_valid_scores, expected_valid_scores, rtol=2e-3, atol=2e-3
    )
    return max_abs


def _event_time_us(fn, *, warmup: int, iters: int, l2_flush=None) -> list[float]:
    from b12x.testing.benchmark import measure_call

    result = measure_call(
        fn, warmup=warmup, samples=iters, eviction=l2_flush or (lambda: None)
    )
    return result.raw_samples("workload")


def _graph_time_us(
    fn,
    *,
    warmup: int,
    iters: int,
    l2_flush=None,
    nsys_capture: bool = False,
    torch_profile: bool = False,
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    for _ in range(warmup):
        if l2_flush is not None:
            l2_flush()
        graph.replay()
    torch.cuda.synchronize()

    if nsys_capture:
        if l2_flush is not None:
            l2_flush()
            torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        graph.replay()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

    if torch_profile:
        if l2_flush is not None:
            l2_flush()
            torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            graph.replay()
            torch.cuda.synchronize()
        print(
            profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
        )

    return _event_time_us(fn, warmup=0, iters=iters, l2_flush=l2_flush)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--global-heads", type=int, default=64)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--page-table-width", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=2304)
    parser.add_argument(
        "--page-stride",
        type=int,
        default=0,
        help="physical page-id stride between rows; 0 shares pages across rows",
    )
    parser.add_argument(
        "--cache-page-stride-bytes",
        type=int,
        default=0,
        help=(
            "physical byte stride between index-cache pages; 0 uses the packed "
            "page width. Some vLLM multi-group packed-KV allocations use a "
            "larger per-block stride shared by multiple cache layers"
        ),
    )
    parser.add_argument(
        "--cache-num-pages",
        type=int,
        default=0,
        help=(
            "physical cache allocation in pages; 0 allocates the full page-table "
            "capacity. A smaller value can model a large graph-static block table "
            "with only the live pages physically allocated"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("supertile-logits", "supertile-topk", "fused-topk"),
        default="supertile-topk",
    )
    parser.add_argument(
        "--route",
        choices=("auto", "packed-contiguous", "paged-tiled"),
        default="auto",
        help="force the paged prefill route; auto uses production policy",
    )
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument(
        "--topk-candidate-capacity",
        type=int,
        choices=(1024, 8192),
        help=("benchmark-only override for the compile-time shared candidate capacity"),
    )
    parser.add_argument(
        "--output-index-space",
        choices=("logical", "physical"),
        default="logical",
        help=(
            "index space emitted by top-k; physical exercises the closed-system "
            "indexer-to-MLA contract without a post-selection adapter"
        ),
    )
    parser.add_argument(
        "--fused-ctas",
        type=int,
        default=0,
        help="fused-topk: override ctas_per_group (0 = auto heuristic)",
    )
    parser.add_argument(
        "--fused-merge-threshold",
        type=int,
        default=-1,
        help="fused-topk cross-CTA merge auto-switch: seq_len<=thr uses last-CTA "
        "reduction, else cooperative radix. -1=topk-aware auto (default), "
        "0=force coop, large=force last-CTA.",
    )
    parser.add_argument(
        "--supertile-k",
        type=int,
        default=0,
        help="supertile width in K rows; 0 uses the production capacity-aware default",
    )
    parser.add_argument(
        "--persistent-ctas",
        type=int,
        default=0,
        help="benchmark-only override for paged scorer persistent CTAs",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--eager",
        action="store_true",
        help="time eager launches instead of graph replay",
    )
    parser.add_argument(
        "--no-l2-flush",
        action="store_true",
        help="disable the 2x-L2 flush between timed replays (default: flush, "
        "for HBM-bound serving-realistic timings instead of L2-hot)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="use analytic inputs and validate every output row before timing",
    )
    parser.add_argument(
        "--input-pattern",
        choices=("random", "low-contrast", "equal"),
        default="random",
        help=(
            "score-distribution stress case used when --check is absent; "
            "low-contrast and equal inputs exercise crowded radix buckets"
        ),
    )
    parser.add_argument(
        "--indices-only",
        action="store_true",
        help="omit final top-k score output, matching sparse-attention serving",
    )
    parser.add_argument(
        "--nsys-capture",
        action="store_true",
        help="bracket one warmed graph replay with cudaProfilerStart/Stop",
    )
    parser.add_argument(
        "--torch-profile",
        action="store_true",
        help="print per-kernel timings for one warmed graph replay",
    )
    parser.add_argument("--seed", type=int, default=91_100)
    args = parser.parse_args()

    if (
        args.topk_candidate_capacity is not None
        and args.topk_candidate_capacity < args.topk
    ):
        raise ValueError(
            "top-k candidate capacity must cover the requested top-k width, got "
            f"{args.topk_candidate_capacity} < {args.topk}"
        )
    if args.topk_candidate_capacity is not None and args.mode != "supertile-topk":
        raise ValueError(
            "--topk-candidate-capacity applies only to --mode supertile-topk"
        )
    if args.topk_candidate_capacity is not None:
        candidate_capacity = int(args.topk_candidate_capacity)

        def resolve_candidate_capacity(*, topk: int) -> int:
            if candidate_capacity < int(topk):
                raise ValueError(
                    "top-k candidate capacity must cover the requested width, got "
                    f"{candidate_capacity} < {topk}"
                )
            return candidate_capacity

        tiled_topk_module._resolve_smem_candidate_capacity = resolve_candidate_capacity

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    num_heads = resolve_replicated_num_q_heads(
        global_num_q_heads=args.global_heads,
        tensor_parallel_size=args.tp_size,
    )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)

    rows = int(args.rows)
    page_table_width = int(args.page_table_width)
    seq_len = int(args.seq_len)
    page_stride = int(args.page_stride)
    if page_stride < 0:
        raise ValueError(f"page_stride must be non-negative, got {page_stride}")
    cache_page_stride_bytes = int(args.cache_page_stride_bytes)
    logical_cache_page_bytes = 64 * 132
    if cache_page_stride_bytes == 0:
        cache_page_stride_bytes = logical_cache_page_bytes
    if cache_page_stride_bytes < logical_cache_page_bytes:
        raise ValueError(
            "cache_page_stride_bytes must be zero or at least one logical page "
            f"({logical_cache_page_bytes}), got {cache_page_stride_bytes}"
        )
    if page_stride == 0:
        max_pages_needed = page_table_width
    else:
        max_pages_needed = (rows - 1) * page_stride + page_table_width
    active_pages_per_row = min((seq_len + 63) // 64, page_table_width)
    min_cache_pages = (
        active_pages_per_row
        if page_stride == 0
        else (rows - 1) * page_stride + active_pages_per_row
    )
    cache_num_pages = int(args.cache_num_pages) or max_pages_needed
    if cache_num_pages < min_cache_pages:
        raise ValueError(
            "cache_num_pages is too small for the live page table: "
            f"need at least {min_cache_pages}, got {cache_num_pages}"
        )
    cache_tokens = cache_num_pages * 64
    analytic_scores = None
    if args.check:
        if str(args.mode) not in ("supertile-topk", "fused-topk"):
            raise ValueError("--check validates --mode supertile-topk or fused-topk")
        if page_stride != 0 and page_stride < page_table_width:
            raise ValueError(
                "--check with distinct page tables requires page_stride >= "
                "page_table_width"
            )
        q_source = torch.zeros(
            (rows, num_heads, 128), dtype=torch.float32, device=device
        )
        q_source[..., 0] = 1.0
        q_fp8 = q_source.to(torch.float8_e4m3fn)
        weights = torch.full(
            (rows, num_heads), 1.0 / num_heads, dtype=torch.float32, device=device
        )
        # Repeat a request-local monotonic ramp for each disjoint physical page
        # span. Kernel outputs are logical token indices, so every row must have
        # the same logical score oracle even when its physical pages are offset.
        analytic_span = cache_tokens if page_stride == 0 else page_stride * 64
        # Give the entire causal tail that can enter any row's top-k a strict,
        # unit-spaced ordering, while leaving older tokens tied safely below it.
        # A globally linear 266K ramp either exceeds FP16_MAX or packs too many
        # candidates into one high-byte radix bin; this bounded tail remains an
        # exact index oracle without violating the selector's refinement capacity.
        tail_span = rows + int(args.topk) + 1024
        tail_start = max(int(seq_len) - tail_span, 0)
        analytic_base = 1.0 + (
            torch.arange(analytic_span, dtype=torch.float32, device=device)
            - float(tail_start)
        ).clamp_(min=0.0)
        analytic_scores = analytic_base.repeat(
            (cache_tokens + analytic_span - 1) // analytic_span
        )[:cache_tokens]
        # Construct the planar FP8+scale cache directly.  Every K row has one
        # positive component: quantized value 448 in column zero and scale
        # analytic_score/448.  This is exactly what the reference packer would
        # produce, without materializing an O(cache_tokens * 128) FP32 tensor.
        packed_index_k_cache = torch.zeros(
            (cache_num_pages, logical_cache_page_bytes),
            dtype=torch.uint8,
            device=device,
        )
        quant_plane = packed_index_k_cache[:, : 64 * 128].view(cache_num_pages, 64, 128)
        fp8_max_byte = int(
            torch.tensor(448.0, dtype=torch.float8_e4m3fn).view(torch.uint8).item()
        )
        quant_plane[..., 0].fill_(fp8_max_byte)
        packed_index_k_cache[:, 64 * 128 :].view(torch.float32).copy_(
            (analytic_scores / 448.0).view(cache_num_pages, 64)
        )
    else:
        q_fp8 = (
            torch.randn((rows, num_heads, 128), generator=gen, dtype=torch.float32).to(
                device
            )
            / 2
        ).to(torch.float8_e4m3fn)
        weights = torch.randn((rows, num_heads), generator=gen, dtype=torch.float32).to(
            device
        )
        if args.input_pattern == "random":
            k_source = (
                torch.randn((cache_tokens, 128), generator=gen, dtype=torch.float32).to(
                    device
                )
                / 3
            )
        else:
            k_source = torch.full(
                (cache_tokens, 128),
                0.25,
                dtype=torch.float32,
                device=device,
            )
            if args.input_pattern == "low-contrast":
                k_source.add_(
                    torch.randn(
                        (cache_tokens, 128), generator=gen, dtype=torch.float32
                    ).to(device)
                    / 4096
                )
        packed_index_k_cache = pack_paged_index_k_cache_reference(k_source)
    # Match vLLM's rank-3 allocation, whose page bytes are planar
    # [64*128 quant][64*4 scales], then reproduce its zero-copy flattening for
    # b12x. The apparent [64, 132] shape is an allocation contract, not an
    # interleaved per-token byte layout.
    if cache_page_stride_bytes == logical_cache_page_bytes:
        vllm_kv_cache = packed_index_k_cache.view(cache_num_pages, 64, 132)
        index_k_cache = vllm_kv_cache.as_strided(
            (cache_num_pages, logical_cache_page_bytes),
            (int(vllm_kv_cache.stride(0)), 1),
        )
    else:
        # Packed multi-group allocators can give each layer a view into one
        # per-block allocation. Consecutive logical pages for that layer are
        # then separated by the aggregate bytes for every cache slot.
        backing_bytes = (
            cache_num_pages - 1
        ) * cache_page_stride_bytes + logical_cache_page_bytes
        packed_backing = torch.empty(backing_bytes, dtype=torch.uint8, device=device)
        index_k_cache = torch.as_strided(
            packed_backing,
            size=(cache_num_pages, logical_cache_page_bytes),
            stride=(cache_page_stride_bytes, 1),
        )
        index_k_cache.copy_(packed_index_k_cache)
    page_table = _make_page_table(
        rows=rows,
        page_table_width=page_table_width,
        seq_len=seq_len,
        page_stride=page_stride,
        device=device,
    )
    bench_mode = str(args.mode)
    requested_route = str(args.route).replace("-", "_")
    topk = int(args.topk)
    output_physical_slots = str(args.output_index_space) == "physical"
    requested_supertile_k = int(args.supertile_k)
    shared_page_table = page_stride == 0
    max_seq_len = min(seq_len, page_table_width * 64)
    if shared_page_table:
        # A vLLM chunked-prefill request exposes one shared page table and one
        # causal length per query token. For a 4k chunk ending at 16k these are
        # 12289..16384, rather than 4096 copies of the final length.
        seqlens = (
            max_seq_len
            - rows
            + 1
            + torch.arange(rows, dtype=torch.int32, device=device)
        ).clamp_(min=1, max=max_seq_len)
    else:
        seqlens = torch.full((rows,), max_seq_len, dtype=torch.int32, device=device)
    plan = plan_indexer_scratch(
        B12XIndexerScratchCaps(
            device=device,
            source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
            num_q_heads=num_heads,
            max_q_rows=rows,
            max_page_table_width=page_table_width,
            topk=topk,
            page_size=64,
            reserve_paged_logits=False,
            supertile_k=requested_supertile_k,
            mode="prefill" if shared_page_table else "decode",
            shared_page_table=shared_page_table,
            route=(
                "paged_tiled"
                if bench_mode == "supertile-logits" and requested_route == "auto"
                else requested_route
            ),
        ),
    )
    supertile_k = int(plan.layout.supertile_tokens)
    if int(args.persistent_ctas) > 0:
        raise ValueError(
            "--persistent-ctas was removed with the workspace-backed indexer path"
        )
    schedule_out = None
    build_schedule = None
    if bench_mode == "supertile-topk":
        build_schedule = False
    else:
        build_schedule = uses_paged_mqa_schedule(
            q_rows=rows,
            max_pages=page_table_width,
        )
    metadata = prepare_paged_indexer_metadata(
        real_page_table=page_table,
        cache_seqlens_int32=seqlens,
        expected_num_q_heads=num_heads,
        schedule_out=schedule_out,
        build_schedule=build_schedule,
        shared_page_table=shared_page_table,
    )
    scratch_specs = plan.shapes_and_dtypes()
    scratch = [
        torch.empty(shape, dtype=dtype, device=device) for shape, dtype in scratch_specs
    ]
    binding = plan.bind(
        scratch=scratch,
        real_page_table=metadata.real_page_table,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
        schedule_metadata=metadata.schedule_metadata,
        expected_num_q_heads=num_heads,
        shared_page_table=shared_page_table,
        output_physical_slots=output_physical_slots,
    )

    out_indices = torch.empty((rows, topk), dtype=torch.int32, device=device)
    out_scores = torch.empty((rows, topk), dtype=torch.float32, device=device)
    indices_only_scratch_scores = None
    untouched_scratch_scores = None
    if args.check and args.indices_only and bench_mode == "supertile-topk":
        # index_topk_fp8 receives no output score tensor in indices-only mode.
        # Its terminal fold instead targets the binding-owned workspace slice,
        # so validate that exact slice rather than unrelated caller storage.
        scratch_scores, _ = binding.scratch.get_indexer_contiguous_topk_buffers(
            row_count=rows
        )
        indices_only_scratch_scores = scratch_scores[:, :topk]
        indices_only_scratch_scores.copy_(
            torch.arange(rows * topk, dtype=torch.float32, device=device).view(
                rows, topk
            )
        )
        untouched_scratch_scores = indices_only_scratch_scores.clone()
    fused_ctas = int(args.fused_ctas) if int(args.fused_ctas) > 0 else None
    fused_cache = None
    if bench_mode == "fused-topk":
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        planned_ctas = int(
            fused_ctas or max(1, min(page_table_width, num_sms // max(rows, 1)))
        )
        try:
            fused_cache = binding.scratch.get_fused_indexer_scratch(topk=topk)
            pack_need = rows * planned_ctas * topk
            if min(fused_cache[0].numel(), fused_cache[1].numel()) < pack_need:
                fused_cache = None
        except RuntimeError:
            fused_cache = None
        if fused_cache is None:
            # The benchmark can force the primitive outside the production row
            # routing gate. Keep that comparison graph-realistic with one fixed,
            # preinitialized workspace rather than per-replay allocations.
            from b12x.attention.dsa_indexer.fused_indexer import (
                fused_indexer_scratch_capacity,
            )

            scratch_ctas = max(num_sms, rows * planned_ctas)
            pack_elems, state_words = fused_indexer_scratch_capacity(
                rows,
                topk,
                scratch_ctas,
            )
            fused_cache = (
                torch.empty(pack_elems, dtype=torch.float32, device=device),
                torch.empty(pack_elems, dtype=torch.int32, device=device),
                torch.zeros(state_words, dtype=torch.int32, device=device),
            )

    clear_indexer_caches()

    def run() -> torch.Tensor:
        if bench_mode == "supertile-logits":
            tile_logits = binding.scratch.get_indexer_contiguous_tile_logits()
            return run_paged_supertile_logits_kernel(
                q_fp8=q_fp8,
                weights=weights,
                index_k_cache=index_k_cache,
                real_page_table=metadata.real_page_table,
                seqlens_per_query=metadata.cache_seqlens_int32,
                active_width=binding.active_width,
                tile_logits=tile_logits,
                source_page_offset=0,
                output_width_tokens=supertile_k,
                preinitialize_tile_logits=False,
            )
        if bench_mode == "fused-topk":
            from b12x.attention.dsa_indexer.kernel import (
                _split_index_k_cache_runtime_views,
            )
            from b12x.attention.dsa_indexer.fused_indexer import run_fused_paged_indexer

            assert fused_cache is not None
            quant, scales = _split_index_k_cache_runtime_views(index_k_cache)
            return run_fused_paged_indexer(
                q_bytes=q_fp8.view(torch.uint8),
                weights=weights,
                k_quant_bytes=quant,
                k_scales=scales,
                real_page_table=metadata.real_page_table,
                seqlens=metadata.cache_seqlens_int32,
                num_heads=num_heads,
                topk=topk,
                out_indices=out_indices,
                out_values=None if args.indices_only else out_scores,
                ctas_per_group=fused_ctas,
                merge_threshold=(
                    None
                    if args.fused_merge_threshold < 0
                    else int(args.fused_merge_threshold)
                ),
                pack_values=fused_cache[0],
                pack_indices=fused_cache[1],
                merge_state=fused_cache[2],
                merge_state_preinitialized=True,
                output_physical_slots=output_physical_slots,
            )[0]
        return index_topk_fp8(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            binding=binding,
            topk=topk,
            expected_num_q_heads=num_heads,
            out_indices=out_indices,
            out_scores=None if args.indices_only else out_scores,
            supertile_k=supertile_k,
        )

    # First call compiles the CuTe DSL kernel before timing or capture.
    out = run()
    torch.cuda.synchronize()
    oracle_state = "not_run"
    if args.check:
        assert analytic_scores is not None
        oracle_max_abs = _validate_analytic_topk(
            indices=out_indices,
            scores=None if args.indices_only else out_scores,
            seqlens=seqlens,
            topk=topk,
            analytic_scores=analytic_scores,
            real_page_table=metadata.real_page_table,
            output_physical_slots=output_physical_slots,
        )
        if args.indices_only:
            if indices_only_scratch_scores is not None:
                assert untouched_scratch_scores is not None
                if not torch.equal(
                    indices_only_scratch_scores, untouched_scratch_scores
                ):
                    raise AssertionError(
                        "indices-only selector must leave its terminal workspace "
                        "score slice untouched"
                    )
                oracle_state = "analytic_indices_pass(workspace_scores_untouched)"
            else:
                oracle_state = "analytic_indices_pass"
        else:
            assert oracle_max_abs is not None
            oracle_state = f"analytic_pass(max_abs={oracle_max_abs:.3g})"
    l2_flush = make_l2_flush_fn(not args.no_l2_flush)
    if args.eager:
        samples_us = _event_time_us(
            run, warmup=args.warmup, iters=args.iters, l2_flush=l2_flush
        )
        timing_mode = "eager"
    else:
        samples_us = _graph_time_us(
            run,
            warmup=args.warmup,
            iters=args.iters,
            l2_flush=l2_flush,
            nsys_capture=bool(args.nsys_capture),
            torch_profile=bool(args.torch_profile),
        )
        timing_mode = "graph"

    median_us = statistics.median(samples_us)
    min_us = min(samples_us)
    raw_us = ",".join(f"{sample:.2f}" for sample in samples_us)

    print(
        "paged_indexer "
        f"mode={bench_mode} timing={timing_mode} rows={rows} indexer_heads={num_heads} "
        f"page_table_width={page_table_width} seq_len={seq_len} "
        f"seqlen_range={int(seqlens.min().item())}-{int(seqlens.max().item())} "
        f"page_stride={page_stride} cache_page_stride_bytes={cache_page_stride_bytes} "
        f"cache_num_pages={cache_num_pages} "
        f"cache_span_mib={((cache_num_pages - 1) * cache_page_stride_bytes + logical_cache_page_bytes) / (1024 * 1024):.2f} "
        f"topk={topk} "
        f"topk_candidate_capacity={args.topk_candidate_capacity or 'policy'} "
        f"supertile_k={supertile_k} "
        f"output_index_space={args.output_index_space} "
        f"requested_supertile_k={requested_supertile_k} "
        f"route={plan.layout.route} prefill_block_k={plan.layout.prefill_block_k} "
        f"input_pattern={'analytic' if args.check else args.input_pattern} "
        f"indices_only={bool(args.indices_only)} "
        f"scratch_mib={plan.layout.nbytes / (1024 * 1024):.2f} "
        f"output_shape={tuple(out.shape)} correctness={oracle_state} "
        f"median_us={median_us:.2f} min_us={min_us:.2f}"
        f" raw_us=[{raw_us}]"
    )


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
