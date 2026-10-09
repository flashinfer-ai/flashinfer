# Copyright (c) 2026 by NVIDIA CORPORATION.
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

"""Ratio-aware testbench for the SM100 union_gqa sparse-prefill kernel.

``sharing_ratio`` is the fraction of top-k KV blocks shared by adjacent
tokens within each 8-token query stage. All measured queries follow a fixed 8192-token KV history. Every selected
block is visible and every adjacent pair has exactly the requested sharing,
including query-stage boundaries. The history queries are not scheduled.

The default SM100 grid uses topk=16, head dimension 128, one KV head,
a 1-second continuous kernel preheat, 10 warmup iterations, and three
consecutive rounds of 100 CUDA Event timing iterations.
Latency is the median of round means. ``--cold-l2-cache`` measures a separate
cache mode. Reference checks, metadata creation and JIT are outside timing.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import numpy as np
import torch

from flashinfer.testing.utils import bench_gpu_time
from benchmarks.union_gqa_telemetry import GPUTelemetry
from benchmarks.union_gqa_workload import (
    HISTORY_TOKENS,
    make_history_indices,
    make_union_runner,
    validate_history_indices,
)

BLOCK_SIZE = 64
CTA_Q_TOKENS = 16
Q_STAGE_TOKENS = 8
GRID_SEQ_LENS = [2048, 4096, 8192, 16384, 32768]
GRID_SHARING_RATIOS = [step / 16 for step in range(17)]


@dataclass(frozen=True)
class Result:
    device: str
    pattern_period: int
    cold_l2_cache: bool
    seq_len: int
    num_kv_heads: int
    num_qo_heads: int
    topk_blocks: int
    selected_tokens: int
    sharing_ratio: float
    effective_sharing_ratio: float
    actual_sharing_ratio: float
    stride_blocks: int
    max_union_blocks_per_qstage: int
    latency_us: float
    sparse_tflops: float
    rounds: int
    preheat_ms: float
    mode_repeat: int
    unified_timer: bool
    warm_gap_cycles: int
    history_tokens: int
    workload: str


def _int_list(value: str) -> list[int]:
    result = [int(item) for item in value.split(",")]
    if not result:
        raise argparse.ArgumentTypeError("list must not be empty")
    return result


def _float_list(value: str) -> list[float]:
    result = [float(item) for item in value.split(",")]
    if not result or any(item < 0.0 or item > 1.0 for item in result):
        raise argparse.ArgumentTypeError("sharing ratios must be in [0, 1]")
    return result


def _check_reference(q, k, v, indices, out):
    """Check causal boundaries against independently computed FP32 attention."""
    positions = {0, 63, 64, 1023, len(q) // 2, len(q) - 1}
    for pos in sorted(p for p in positions if p < len(q)):
        for head in {0, k.shape[1] - 1}:
            keys = [
                token
                for block in indices[pos, head]
                for token in range(
                    BLOCK_SIZE * int(block),
                    min(BLOCK_SIZE * (int(block) + 1), HISTORY_TOKENS + pos + 1),
                )
            ]
            key_idx = torch.tensor(keys, device=q.device)
            heads = slice(head * 16, (head + 1) * 16)
            scores = q[pos, heads].float() @ k[key_idx, head].float().T / math.sqrt(128)
            expected = torch.softmax(scores, dim=-1) @ v[key_idx, head].float()
            torch.testing.assert_close(
                out[pos, heads].float(), expected, atol=2e-3, rtol=2e-2
            )


def _preheat(run, duration_ms: float) -> None:
    """Run the measured backend continuously before collecting timed rounds.

    Iteration-only warmups are too short to remove the power/clock history of
    a different backend on long-sequence SM100 workloads.
    """
    torch.cuda.synchronize()
    deadline = time.perf_counter() + duration_ms / 1000
    while time.perf_counter() < deadline:
        for _ in range(10):
            run()
        torch.cuda.synchronize()


def _bench_event_mean(run, warmup: int, iters: int, before_each_run=None) -> float:
    """Return the mean CUDA Event latency in milliseconds."""
    for _ in range(warmup):
        if before_each_run is not None:
            before_each_run()
        run()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if before_each_run is not None:
            before_each_run()
        starts[i].record()
        run()
        stops[i].record()
    torch.cuda.synchronize()
    return (
        sum(start.elapsed_time(stop) for start, stop in zip(starts, stops, strict=True))
        / iters
    )


def _run_case(
    seq_len: int,
    num_kv_heads: int,
    topk_blocks: int,
    sharing_ratio: float,
    dry_run_iters: int,
    repeat_iters: int,
    cold_l2_cache: bool,
    seed: int,
    rounds: int,
    preheat_ms: float,
    telemetry=None,
    mode_repeat: int = 0,
    unified_timer: bool = False,
    warm_gap_cycles: int = 0,
) -> Result:
    torch.manual_seed(seed)
    num_qo_heads = 16 * num_kv_heads
    full_len = HISTORY_TOKENS + seq_len
    q = (
        torch.randn(full_len, num_qo_heads, 128, dtype=torch.bfloat16, device="cuda")
        * 0.3
    )
    k = (
        torch.randn(full_len, num_kv_heads, 128, dtype=torch.bfloat16, device="cuda")
        * 0.3
    )
    v = torch.randn_like(k) * 0.3
    indices_cpu = make_history_indices(seq_len, sharing_ratio, topk_blocks)
    stats = validate_history_indices(indices_cpu, sharing_ratio, topk_blocks)
    stride = round(topk_blocks * (1.0 - sharing_ratio))
    union_blocks = stats["max_union_blocks"]
    actual_sharing = stats["actual_sharing_ratio"]
    out = torch.empty_like(q)
    run = make_union_runner(q, k, v, indices_cpu, out)
    run()
    torch.cuda.synchronize()
    _check_reference(
        q[HISTORY_TOKENS:],
        k,
        v,
        np.broadcast_to(indices_cpu[:, None, :], (seq_len, num_kv_heads, topk_blocks)),
        out[HISTORY_TOKENS:],
    )

    def phase(name, **extra):
        return (
            telemetry.phase(
                phase=name,
                seq_len=seq_len,
                sharing_ratio=sharing_ratio,
                cold_l2_cache=cold_l2_cache,
                mode_repeat=mode_repeat,
                unified_timer=unified_timer,
                warm_gap_cycles=warm_gap_cycles,
                history_tokens=HISTORY_TOKENS,
                workload="strict-fixed-history",
                **extra,
            )
            if telemetry is not None
            else nullcontext({})
        )

    with phase("preheat"):
        _preheat(run, preheat_ms)
    round_means = []
    for round_index in range(rounds):
        with phase("timing", round_index=round_index) as marker:
            if cold_l2_cache or unified_timer:
                times_ms = bench_gpu_time(
                    run,
                    dry_run_iters=dry_run_iters,
                    repeat_iters=repeat_iters,
                    cold_l2_cache=cold_l2_cache,
                    use_cuda_graph=False,
                    enable_cupti=False,
                )
                round_means.append(statistics.mean(times_ms))
            else:
                gap = (
                    (lambda: torch.cuda._sleep(warm_gap_cycles))
                    if warm_gap_cycles
                    else None
                )
                round_means.append(
                    _bench_event_mean(run, dry_run_iters, repeat_iters, gap)
                )
            marker["latency_us"] = round_means[-1] * 1000
    latency_ms = statistics.median(round_means)
    selected_tokens = topk_blocks * BLOCK_SIZE
    # All selected tokens are fully visible; the history prefix is not timed.
    sparse_flops = 4.0 * seq_len * selected_tokens * 128 * num_qo_heads
    sparse_tflops = sparse_flops / (latency_ms * 1e-3) / 1e12
    return Result(
        device=torch.cuda.get_device_name(q.device),
        pattern_period=Q_STAGE_TOKENS,
        cold_l2_cache=cold_l2_cache,
        seq_len=seq_len,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_qo_heads,
        topk_blocks=topk_blocks,
        selected_tokens=selected_tokens,
        sharing_ratio=sharing_ratio,
        effective_sharing_ratio=1.0 - stride / topk_blocks,
        actual_sharing_ratio=actual_sharing,
        stride_blocks=stride,
        max_union_blocks_per_qstage=union_blocks,
        latency_us=latency_ms * 1e3,
        sparse_tflops=sparse_tflops,
        rounds=rounds,
        preheat_ms=preheat_ms,
        mode_repeat=mode_repeat,
        unified_timer=unified_timer,
        warm_gap_cycles=warm_gap_cycles,
        history_tokens=HISTORY_TOKENS,
        workload="strict-fixed-history",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", type=_int_list, default=GRID_SEQ_LENS)
    parser.add_argument("--topk-blocks", type=_int_list, default=[16])
    parser.add_argument(
        "--sharing-ratios",
        type=_float_list,
        default=GRID_SHARING_RATIOS,
        help="fractions of top-k KV blocks shared by adjacent tokens",
    )
    parser.add_argument("--num-kv-heads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run-iters", type=int, default=10)
    parser.add_argument("--repeat-iters", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--preheat-ms",
        type=float,
        default=1000,
        help="continuous kernel preheat duration per case, outside timing",
    )
    parser.add_argument(
        "--cold-l2-cache",
        action="store_true",
        help="flush L2 before each measurement; default uses warm L2",
    )
    parser.add_argument(
        "--csv", action="store_true", help="write machine-readable CSV to stdout"
    )
    parser.add_argument(
        "--telemetry-prefix",
        help="write .samples.csv, .phases.jsonl, .summary.json and .stderr files",
    )
    parser.add_argument("--telemetry-interval-ms", type=int, default=20)
    parser.add_argument(
        "--cache-modes",
        choices=["warm,cold", "cold,warm"],
        help="measure both modes in one process, reversing order each mode repeat",
    )
    parser.add_argument("--mode-repeats", type=int, default=1)
    parser.add_argument(
        "--unified-timer",
        action="store_true",
        help="use the same bench_gpu_time CUDA Event path for warm and cold",
    )
    parser.add_argument(
        "--warm-gap-cycles",
        type=int,
        default=0,
        help="diagnostic GPU sleep before each warm iteration, outside timing; no L2 flush",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("bench_union_gqa.py requires exact SM100")
    if args.num_kv_heads <= 0:
        parser.error("--num-kv-heads must be positive")
    if any(value <= 0 for value in args.seq_lens):
        parser.error("--seq-lens values must be positive")
    if any(value <= 0 or value > 16 for value in args.topk_blocks):
        parser.error("--topk-blocks values must be in [1, 16]")
    if min(args.dry_run_iters, args.repeat_iters, args.rounds) <= 0:
        parser.error("--dry-run-iters, --repeat-iters and --rounds must be positive")
    if not math.isfinite(args.preheat_ms) or args.preheat_ms < 0:
        parser.error("--preheat-ms must be finite and nonnegative")
    if args.mode_repeats <= 0 or args.telemetry_interval_ms <= 0:
        parser.error("mode repeats and telemetry interval must be positive")
    if args.cache_modes and args.cold_l2_cache:
        parser.error("--cache-modes cannot be combined with --cold-l2-cache")
    if args.warm_gap_cycles < 0 or (args.warm_gap_cycles and args.unified_timer):
        parser.error("warm gap must be nonnegative and cannot use --unified-timer")

    telemetry = (
        GPUTelemetry(
            args.telemetry_prefix,
            str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
            args.telemetry_interval_ms,
        )
        if args.telemetry_prefix
        else None
    )
    modes = (
        [mode == "cold" for mode in args.cache_modes.split(",")]
        if args.cache_modes
        else [args.cold_l2_cache]
    )
    results = []
    try:
        for seq_len in args.seq_lens:
            for topk_blocks in args.topk_blocks:
                for ratio in args.sharing_ratios:
                    for repeat in range(args.mode_repeats):
                        for cold in modes if repeat % 2 == 0 else modes[::-1]:
                            results.append(
                                _run_case(
                                    seq_len,
                                    args.num_kv_heads,
                                    topk_blocks,
                                    ratio,
                                    args.dry_run_iters,
                                    args.repeat_iters,
                                    cold,
                                    args.seed,
                                    args.rounds,
                                    args.preheat_ms,
                                    telemetry,
                                    repeat,
                                    args.unified_timer,
                                    args.warm_gap_cycles,
                                )
                            )
    finally:
        if telemetry is not None:
            telemetry.close()

    if args.csv:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
        return

    print(
        f"device={torch.cuda.get_device_name()} block_size={BLOCK_SIZE} "
        f"pattern_period={Q_STAGE_TOKENS} timing=median_of_event_round_means "
        f"cold_l2_cache={args.cold_l2_cache}"
    )
    for result in results:
        print(
            f"s={result.seq_len:>7} hkv={result.num_kv_heads} "
            f"cold_l2_cache={result.cold_l2_cache} repeat={result.mode_repeat} "
            f"topk_blocks={result.topk_blocks:>2} "
            f"ratio={result.sharing_ratio:>5.3f} "
            f"effective_ratio={result.effective_sharing_ratio:>5.3f} "
            f"actual_ratio={result.actual_sharing_ratio:>5.3f} "
            f"union/qstage={result.max_union_blocks_per_qstage:>3} "
            f"latency={result.latency_us:>9.2f} us "
            f"sparse={result.sparse_tflops:>7.1f} TFLOP/s"
        )


if __name__ == "__main__":
    main()
