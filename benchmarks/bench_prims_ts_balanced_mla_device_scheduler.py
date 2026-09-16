"""Benchmark graph-captured balanced MLA device scheduling on realistic traffic.

The scheduler-only graph isolates planner latency. The attention-only and
scheduler-plus-attention graphs share one balanced plan and descriptor ABI, so
their difference measures the production critical-path cost of scheduling once
before a compatible MLA layer group.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
from typing import Callable

import torch

from flashinfer.attention.prims_ts import BatchMLADecodePagedTSWrapper


RL_SEQ = (100, 1000, 2000, 3000, 10000, 30000, 50000, 100000, 110000)
RL_WEIGHTS = (7, 18, 200, 200, 400, 780, 600, 80, 65)
PROD_SEQ = (4096, 8192, 16384, 32768, 65536, 131072)
PROD_WEIGHTS = (35, 20, 15, 12, 10, 8)
DISTRIBUTIONS = {
    "rl": (RL_SEQ, RL_WEIGHTS, 20000),
    "prod": (PROD_SEQ, PROD_WEIGHTS, 10000),
}

NUM_HEADS = 128
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
PAGE_SIZE = 64
MAX_SEQ_LEN = 131072
POOL_PAGES = 8192


def sample_lengths(distribution: str, batch_size: int, sample_idx: int) -> list[int]:
    """Return the historical deterministic RL/Prod draw, aligned to 128 tokens."""

    values, weights, base_seed = DISTRIBUTIONS[distribution]
    rng = random.Random(base_seed + batch_size * 100 + sample_idx)
    return [
        ((value + 127) // 128) * 128
        for value in rng.choices(values, weights=weights, k=batch_size)
    ]


def capture_graph(fn: Callable[[], None]) -> torch.cuda.CUDAGraph:
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def time_graph(
    graph: torch.cuda.CUDAGraph, *, warmups: int, iterations: int, trials: int
) -> float:
    for _ in range(warmups):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(trials):
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return statistics.median(samples)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = math.ceil(probability * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def make_runtime(
    batch_size: int,
    initial_lengths: list[int],
    device: torch.device,
    *,
    qkv_dtype: torch.dtype,
    scheduler: str,
    forced_target_piece_tiles: int,
):
    query = torch.zeros(
        (batch_size, 1, NUM_HEADS, QK_HEAD_DIM),
        dtype=qkv_dtype,
        device=device,
    )
    kv_cache = torch.zeros(
        (POOL_PAGES, PAGE_SIZE, QK_HEAD_DIM),
        dtype=qkv_dtype,
        device=device,
    )
    max_pages = (MAX_SEQ_LEN + PAGE_SIZE - 1) // PAGE_SIZE
    block_tables = (
        torch.arange(batch_size * max_pages, dtype=torch.int32, device=device)
        .reshape(batch_size, max_pages)
        .remainder_(POOL_PAGES)
    )
    seq_lens = torch.tensor(initial_lengths, dtype=torch.int32, device=device)
    output = torch.empty(
        (batch_size, 1, NUM_HEADS, KV_LORA_RANK),
        dtype=torch.bfloat16,
        device=device,
    )
    wrapper = BatchMLADecodePagedTSWrapper()
    wrapper.plan_balanced(
        device,
        batch_size,
        NUM_HEADS,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        PAGE_SIZE,
        MAX_SEQ_LEN,
        max_seq_len_q=1,
        packed_query=False,
        q_data_type=qkv_dtype,
        kv_data_type=qkv_dtype,
        o_data_type=torch.bfloat16,
        seq_lens=initial_lengths,
    )

    def schedule() -> None:
        if forced_target_piece_tiles:
            wrapper._plan_state.balanced_plan.schedule_device(
                seq_lens,
                validate=False,
                scheduler=scheduler,
                forced_target_piece_tiles=forced_target_piece_tiles,
            )
            return
        wrapper.schedule(
            seq_lens,
            validate=False,
            scheduler=scheduler,
        )

    def attention() -> None:
        wrapper.run(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            bmm1_scale=1.0 / math.sqrt(QK_HEAD_DIM),
            bmm2_scale=1.0,
            out=output,
            validate=False,
        )

    def schedule_and_attention() -> None:
        schedule()
        attention()

    # Warm compilation and validate the initial live metadata before capture.
    if forced_target_piece_tiles:
        wrapper._plan_state.balanced_plan.schedule_device(
            seq_lens,
            scheduler=scheduler,
            forced_target_piece_tiles=forced_target_piece_tiles,
        )
    else:
        wrapper.schedule(seq_lens, scheduler=scheduler)
    wrapper.run(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        bmm1_scale=1.0 / math.sqrt(QK_HEAD_DIM),
        bmm2_scale=1.0,
        out=output,
        validate=not forced_target_piece_tiles,
    )
    torch.cuda.synchronize()
    return (
        wrapper,
        seq_lens,
        capture_graph(schedule),
        capture_graph(attention),
        capture_graph(schedule_and_attention),
        attention,
    )


def run_policy_comparison(args, device: torch.device, qkv_dtype: torch.dtype) -> None:
    """Time exact and optimized policies back-to-back on each deterministic draw."""

    records = []
    print(
        "distribution,batch,samples,exact_scheduler_p50_us,optimized_scheduler_p50_us,"
        "exact_attention_p50_us,optimized_attention_p50_us,"
        "exact_combined_p50_us,optimized_combined_p50_us",
        flush=True,
    )
    for distribution in args.distributions:
        for batch_size in args.batches:
            draws = [
                sample_lengths(distribution, batch_size, sample_idx)
                for sample_idx in range(
                    args.sample_start, args.sample_start + args.samples
                )
            ]
            (
                wrapper,
                live_lengths,
                exact_scheduler_graph,
                attention_graph,
                exact_combined_graph,
                attention,
            ) = make_runtime(
                batch_size,
                draws[0],
                device,
                qkv_dtype=qkv_dtype,
                scheduler="exact",
                forced_target_piece_tiles=args.forced_target_piece_tiles,
            )

            def optimized_schedule() -> None:
                wrapper.schedule(
                    live_lengths,
                    validate=False,
                    scheduler="optimized",
                )

            def optimized_schedule_and_attention() -> None:
                optimized_schedule()
                attention()

            optimized_scheduler_graph = capture_graph(optimized_schedule)
            optimized_combined_graph = capture_graph(optimized_schedule_and_attention)
            measurements = {policy: [] for policy in ("exact", "optimized")}

            def measure_policy(policy: str):
                if policy == "exact":
                    scheduler_graph = exact_scheduler_graph
                    combined_graph = exact_combined_graph
                else:
                    scheduler_graph = optimized_scheduler_graph
                    combined_graph = optimized_combined_graph
                # Publish this policy before timing the shared attention graph.
                scheduler_graph.replay()
                torch.cuda.synchronize()
                return {
                    "scheduler_us": time_graph(
                        scheduler_graph,
                        warmups=args.warmups,
                        iterations=args.iterations,
                        trials=args.trials,
                    ),
                    "attention_us": time_graph(
                        attention_graph,
                        warmups=args.warmups,
                        iterations=args.iterations,
                        trials=args.trials,
                    ),
                    "combined_us": time_graph(
                        combined_graph,
                        warmups=args.warmups,
                        iterations=args.iterations,
                        trials=args.trials,
                    ),
                }

            for local_sample_idx, lengths in enumerate(draws):
                sample_idx = args.sample_start + local_sample_idx
                live_lengths.copy_(
                    torch.tensor(
                        lengths,
                        dtype=torch.int32,
                        device=live_lengths.device,
                    )
                )
                order = (
                    ("exact", "optimized")
                    if local_sample_idx % 2 == 0
                    else ("optimized", "exact")
                )
                current = {policy: measure_policy(policy) for policy in order}
                for policy in ("exact", "optimized"):
                    measurements[policy].append(current[policy])
                records.append(
                    {
                        "distribution": distribution,
                        "dtype": args.dtype,
                        "batch_size": batch_size,
                        "sample": sample_idx,
                        "seed": DISTRIBUTIONS[distribution][2]
                        + batch_size * 100
                        + sample_idx,
                        **{
                            f"{policy}_{metric}": value
                            for policy, values in current.items()
                            for metric, value in values.items()
                        },
                    }
                )

            def policy_median(policy: str, metric: str) -> float:
                return statistics.median(
                    value[metric] for value in measurements[policy]
                )

            print(
                f"{distribution},{batch_size},{args.samples},"
                f"{policy_median('exact', 'scheduler_us'):.3f},"
                f"{policy_median('optimized', 'scheduler_us'):.3f},"
                f"{policy_median('exact', 'attention_us'):.3f},"
                f"{policy_median('optimized', 'attention_us'):.3f},"
                f"{policy_median('exact', 'combined_us'):.3f},"
                f"{policy_median('optimized', 'combined_us'):.3f}",
                flush=True,
            )
            torch.cuda.empty_cache()

    if args.jsonl is not None:
        with args.jsonl.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--batches", type=int, nargs="+", default=(32, 64, 128))
    parser.add_argument(
        "--distributions",
        choices=tuple(DISTRIBUTIONS),
        nargs="+",
        default=("rl", "prod"),
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument(
        "--scheduler",
        choices=("exact", "optimized"),
        default="optimized",
        help="CUDA scheduler policy (default: optimized)",
    )
    parser.add_argument(
        "--compare-policies",
        action="store_true",
        help="interleave exact and optimized policies on every sampled draw",
    )
    parser.add_argument("--forced-target-piece-tiles", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    qkv_dtype = {
        "bf16": torch.bfloat16,
        "fp8": torch.float8_e4m3fn,
    }[args.dtype]
    if args.compare_policies:
        if args.forced_target_piece_tiles:
            parser.error("--compare-policies does not support a forced target")
        run_policy_comparison(args, device, qkv_dtype)
        return
    records = []
    print(
        "distribution,batch,samples,scheduler_p50_us,scheduler_p95_us,"
        "attention_p50_us,combined_p50_us,combined_minus_attention_p50_us",
        flush=True,
    )
    for distribution in args.distributions:
        for batch_size in args.batches:
            draws = [
                sample_lengths(distribution, batch_size, sample_idx)
                for sample_idx in range(
                    args.sample_start, args.sample_start + args.samples
                )
            ]
            (
                wrapper,
                live_lengths,
                scheduler_graph,
                attention_graph,
                combined_graph,
                _,
            ) = make_runtime(
                batch_size,
                draws[0],
                device,
                qkv_dtype=qkv_dtype,
                scheduler=args.scheduler,
                forced_target_piece_tiles=args.forced_target_piece_tiles,
            )
            scheduler_times = []
            attention_times = []
            combined_times = []
            deltas = []
            for local_sample_idx, lengths in enumerate(draws):
                sample_idx = args.sample_start + local_sample_idx
                live_lengths.copy_(
                    torch.tensor(lengths, dtype=torch.int32, device=live_lengths.device)
                )
                # Publish the selected plan used by the attention-only timing.
                scheduler_graph.replay()
                torch.cuda.synchronize()
                attention_us = time_graph(
                    attention_graph,
                    warmups=args.warmups,
                    iterations=args.iterations,
                    trials=args.trials,
                )
                scheduler_us = time_graph(
                    scheduler_graph,
                    warmups=args.warmups,
                    iterations=args.iterations,
                    trials=args.trials,
                )
                combined_us = time_graph(
                    combined_graph,
                    warmups=args.warmups,
                    iterations=args.iterations,
                    trials=args.trials,
                )
                scheduler_times.append(scheduler_us)
                attention_times.append(attention_us)
                combined_times.append(combined_us)
                deltas.append(combined_us - attention_us)
                records.append(
                    {
                        "distribution": distribution,
                        "dtype": args.dtype,
                        "batch_size": batch_size,
                        "sample": sample_idx,
                        "seed": DISTRIBUTIONS[distribution][2]
                        + batch_size * 100
                        + sample_idx,
                        "scheduler": args.scheduler,
                        "scheduler_us": scheduler_us,
                        "attention_us": attention_us,
                        "combined_us": combined_us,
                        "combined_minus_attention_us": combined_us - attention_us,
                    }
                )
            print(
                f"{distribution},{batch_size},{args.samples},"
                f"{statistics.median(scheduler_times):.3f},"
                f"{percentile(scheduler_times, 0.95):.3f},"
                f"{statistics.median(attention_times):.3f},"
                f"{statistics.median(combined_times):.3f},"
                f"{statistics.median(deltas):.3f}",
                flush=True,
            )
            del wrapper, live_lengths, scheduler_graph, attention_graph, combined_graph
            torch.cuda.empty_cache()

    if args.jsonl is not None:
        with args.jsonl.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
