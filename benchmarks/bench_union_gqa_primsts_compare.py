# Copyright (c) 2026 by NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Compare exact causal token-sparse GQA16 on SM100, including primsts prepare.

BSR conversion, planning and compilation are outside timing. Each measurement
uses CUDA Events with optional cold L2. Each backend runs all its rounds consecutively
after a time-based preheat. Select one backend for independent-process timing.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext, redirect_stdout
import importlib.util
import math
from pathlib import Path
import statistics
import sys
import tempfile

import torch

from benchmarks.bench_union_gqa import (
    GRID_SEQ_LENS,
    GRID_SHARING_RATIOS,
    _bench_event_mean,
    _preheat,
    make_history_indices,
)
from benchmarks.union_gqa_workload import (
    HISTORY_TOKENS,
    make_union_runner,
    validate_history_indices,
)
from benchmarks.union_gqa_telemetry import GPUTelemetry
from flashinfer.attention.prims_ts import BlockSparseTSWrapper
from flashinfer.testing import bench_gpu_time


def local_runtime_compat():
    """Adapt CUTLASS 4.8 task calls and old Torch events for eager timing only.

    Load an isolated source copy; do not change repository or installed code.
    This changes API plumbing, not task schedules or attention arithmetic.
    """
    import flashinfer.attention.prims_ts.kernels.fmha_decode as package
    from flashinfer.attention.prims_ts._block_sparse import plan

    name = package.__name__ + ".fmha_decode_tasks"
    source = Path(package.__file__).with_name("fmha_decode_tasks.py").read_text()
    replacements = {
        "Task._run_task_body_persistent(self, context)": "Task._run_task_body_persistent(self)",
        "            context=context,\n": "",
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise RuntimeError("Compatibility adapter no longer matches source")
        source = source.replace(before, after)
    directory = tempfile.TemporaryDirectory(prefix="primsts_bench_compat_")
    patched = Path(directory.name) / "fmha_decode_tasks.py"
    patched.write_text(source)
    spec = importlib.util.spec_from_file_location(name, patched)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    def record_ready(stream):
        event = torch.cuda.Event()
        event.record(stream)
        return event

    plan._record_block_sparse_plan_ready_event = record_ready
    return directory


def check_reference(q, k, v, indices, outputs):
    """Check both kernels independently against FP32 at causal boundaries."""
    positions = {0, 1, 63, 64, 65, 1023, 1024, len(q) // 2, len(q) - 1}
    for pos in sorted(p for p in positions if p < len(q)):
        keys = [
            token
            for block in indices[pos]
            for token in range(
                64 * int(block), min(64 * (int(block) + 1), HISTORY_TOKENS + pos + 1)
            )
        ]
        key_idx = torch.tensor(keys, device=q.device)
        scores = q[pos].float() @ k[key_idx, 0].float().T / math.sqrt(128)
        expected = torch.softmax(scores, dim=-1) @ v[key_idx, 0].float()
        for output in outputs:
            torch.testing.assert_close(
                output[pos].float(), expected, atol=2e-3, rtol=2e-2
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", default=",".join(map(str, GRID_SEQ_LENS)))
    parser.add_argument(
        "--sharing-ratios", default=",".join(map(str, GRID_SHARING_RATIOS))
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--preheat-ms", type=float, default=1000)
    parser.add_argument("--cold-l2-cache", action="store_true")
    parser.add_argument("--telemetry-prefix", type=Path)
    parser.add_argument("--telemetry-interval-ms", type=int, default=10)
    parser.add_argument(
        "--backend", choices=("primsts", "union", "both"), default="both"
    )
    parser.add_argument(
        "--local-runtime-compat",
        action="store_true",
        help="eager-only adapters for CUTLASS 4.8 and old Torch",
    )
    args = parser.parse_args()
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    ratios = [float(x) for x in args.sharing_ratios.split(",")]
    if min(args.warmup, args.iters, args.rounds, args.telemetry_interval_ms) < 1:
        parser.error("warmup, iters and rounds must be positive")
    if not math.isfinite(args.preheat_ms) or args.preheat_ms < 0:
        parser.error("preheat-ms must be finite and nonnegative")
    if any(s < 1024 or s % 8 for s in seq_lens):
        parser.error("sequence lengths must be >=1024 and divisible by 8")
    if any(not 0 <= r <= 1 for r in ratios):
        parser.error("sharing ratios must lie in [0, 1]")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("This comparison requires SM100")
    compat_directory = local_runtime_compat() if args.local_runtime_compat else None
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "gpu",
            "seq_len",
            "requested_sharing",
            "actual_selected_block_sharing",
            "effective_pairs_per_head",
            "primsts_us",
            "union_us",
            "primsts_over_union",
            "max_abs_diff",
            "warmup",
            "iters",
            "rounds",
            "primsts_prepare_included",
            "pattern",
            "max_union_blocks",
            "timed_backend",
            "preheat_ms",
            "history_tokens",
            "cold_l2_cache",
        ]
    )
    sys.stdout.flush()
    telemetry = (
        GPUTelemetry(
            args.telemetry_prefix,
            str(torch.cuda.get_device_properties(0).uuid),
            args.telemetry_interval_ms,
        )
        if args.telemetry_prefix
        else None
    )
    try:
        run_grid(args, seq_lens, ratios, writer, telemetry)
    finally:
        if telemetry is not None:
            telemetry.close()
        if compat_directory is not None:
            compat_directory.cleanup()


def run_grid(args, seq_lens, ratios, writer, telemetry):
    for seq_len in seq_lens:
        torch.manual_seed(42)
        full_len = HISTORY_TOKENS + seq_len
        q_full = (
            torch.randn(full_len, 16, 128, dtype=torch.bfloat16, device="cuda") * 0.3
        )
        q = q_full[HISTORY_TOKENS:]
        k = torch.randn(full_len, 1, 128, dtype=torch.bfloat16, device="cuda") * 0.3
        v = torch.randn_like(k) * 0.3
        out_full, out_primsts = torch.empty_like(q_full), torch.empty_like(q)
        out_union = out_full[HISTORY_TOKENS:]
        wrapper = BlockSparseTSWrapper()
        wrapper.plan(
            batch_size=1,
            seq_len_q=seq_len,
            seq_len_kv=full_len,
            num_qo_heads=16,
            num_kv_heads=1,
            head_dim=128,
            q_block_size=1,
            kv_block_size=64,
            device="cuda",
            max_blocks_per_row=16,
            use_kv_valid_bits=False,
            mask_type="causal",
            q_data_type=torch.bfloat16,
        )
        q_b, k_b, v_b, out_b = (x.unsqueeze(0) for x in (q, k, v, out_primsts))
        indptr = (
            torch.arange(seq_len + 1, dtype=torch.int32, device="cuda") * 16
        ).view(1, 1, -1)
        for ratio in ratios:
            indices = make_history_indices(seq_len, ratio)
            stats = validate_history_indices(indices, ratio)
            columns = torch.from_numpy(indices.reshape(-1)).to("cuda")
            pairs = stats["effective_pairs_per_head"]
            max_union = stats["max_union_blocks"]
            run_union = make_union_runner(q_full, k, v, indices, out_full)

            def run_primsts():
                wrapper.run(q_b, k_b, v_b, indptr, columns, out=out_b)

            # Compilation diagnostics belong on stderr, preserving valid CSV output.
            with redirect_stdout(sys.stderr):
                run_primsts()
                run_union()
            torch.cuda.synchronize()
            torch.testing.assert_close(out_primsts, out_union, atol=2e-3, rtol=2e-2)
            check_reference(q, k, v, indices, (out_primsts, out_union))
            max_diff = (out_primsts.float() - out_union.float()).abs().max().item()
            times = {"primsts": [], "union": []}
            for name, run in (("primsts", run_primsts), ("union", run_union)):
                if args.backend not in (name, "both"):
                    continue

                def phase(kind, **extra):
                    return (
                        telemetry.phase(
                            phase=kind,
                            backend=name,
                            seq_len=seq_len,
                            sharing_ratio=ratio,
                            cold_l2_cache=args.cold_l2_cache,
                            **extra,
                        )
                        if telemetry is not None
                        else nullcontext({})
                    )

                with phase("preheat"):
                    _preheat(run, args.preheat_ms)
                for round_index in range(args.rounds):
                    with phase("timing", round_index=round_index) as marker:
                        if args.cold_l2_cache:
                            samples = bench_gpu_time(
                                run,
                                dry_run_iters=args.warmup,
                                repeat_iters=args.iters,
                                cold_l2_cache=True,
                                use_cuda_graph=False,
                                enable_cupti=False,
                            )
                            latency_us = 1000 * statistics.mean(samples)
                        else:
                            latency_us = 1000 * _bench_event_mean(
                                run, args.warmup, args.iters
                            )
                        marker["latency_us"] = latency_us
                        times[name].append(latency_us)
            primsts_us, union_us = (
                statistics.median(times[n]) if times[n] else None
                for n in ("primsts", "union")
            )
            writer.writerow(
                [
                    torch.cuda.get_device_name(),
                    seq_len,
                    ratio,
                    stats["actual_sharing_ratio"],
                    pairs,
                    round(primsts_us, 3) if primsts_us is not None else "",
                    round(union_us, 3) if union_us is not None else "",
                    round(primsts_us / union_us, 3) if args.backend == "both" else "",
                    max_diff,
                    args.warmup,
                    args.iters,
                    args.rounds,
                    True,
                    "strict-fixed-history",
                    max_union,
                    args.backend,
                    args.preheat_ms,
                    HISTORY_TOKENS,
                    args.cold_l2_cache,
                ]
            )
            sys.stdout.flush()


if __name__ == "__main__":
    main()
