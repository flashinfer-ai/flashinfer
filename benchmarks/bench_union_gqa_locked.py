# Copyright (c) 2026 by NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Manually run a locked-clock Union-GQA control and verify actual clocks.

Only nvidia-smi configuration commands use sudo; Python/CUDA runs as the caller.
Start with default GPU clocks: cleanup resets GPU clocks to their defaults.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys


def summarize(source, destination, target_mhz, tolerance_mhz):
    """Write grouped timing metrics and return clock validity with summary rows."""
    phases = json.loads(source.read_text())
    timing = [phase for phase in phases if phase["phase"] == "timing"]
    if not timing:
        raise RuntimeError("No timing phases recorded")
    valid = all(
        phase["samples"] > 0
        and phase.get("sm_mhz_min", float("-inf")) >= target_mhz - tolerance_mhz
        and phase.get("sm_mhz_max", float("inf")) <= target_mhz + tolerance_mhz
        for phase in timing
    )
    groups = {}
    for phase in timing:
        key = (
            phase["seq_len"],
            phase["sharing_ratio"],
            phase["cold_l2_cache"],
        )
        groups.setdefault(key, []).append(phase)
    rows = []
    for (seq_len, sharing, cold), group in sorted(groups.items()):
        row = dict(
            seq_len=seq_len,
            sharing_ratio=sharing,
            cold_l2_cache=cold,
            latency_us=statistics.median(phase["latency_us"] for phase in group),
            sm_mhz_min=min(phase.get("sm_mhz_min", float("nan")) for phase in group),
            sm_mhz_max=max(phase.get("sm_mhz_max", float("nan")) for phase in group),
            rounds=len(group),
            samples=sum(phase["samples"] for phase in group),
        )
        for metric in (
            "sm_mhz_mean",
            "memory_mhz_mean",
            "power_w_mean",
            "temperature_c_mean",
            "sw_power_cap_fraction",
            "sw_thermal_slowdown_fraction",
            "hw_thermal_slowdown_fraction",
        ):
            values = [phase[metric] for phase in group if phase.get(metric) is not None]
            row[metric] = statistics.mean(values) if values else None
        rows.append(row)
    with destination.open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return valid, rows


def stop_benchmark(process):
    """Terminate the benchmark process group, escalating after a timeout."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    """Run warm and cold benchmarks with locked SM clocks and validate telemetry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0", help="nvidia-smi index or GPU UUID")
    parser.add_argument("--sm-clock", type=int, default=1500)
    parser.add_argument("--clock-mode", type=int, choices=(0, 1), default=0)
    parser.add_argument("--clock-tolerance", type=int, default=15)
    parser.add_argument("--seq-lens", default="16384,32768")
    parser.add_argument("--sharing-ratios", default="0,0.5,1")
    parser.add_argument("--mode-repeats", type=int, default=3)
    parser.add_argument("--repeat-iters", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--dry-run", action="store_true", help="print plan; no GPU access or writes"
    )
    args = parser.parse_args()
    if min(args.sm_clock, args.mode_repeats, args.repeat_iters, args.timeout) <= 0:
        parser.error("clock, repeats, iterations and timeout must be positive")
    if args.clock_tolerance < 0:
        parser.error("clock tolerance must be nonnegative")
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = (
        args.output_dir or root / "benchmarks/results" / f"union_gqa_locked_{stamp}"
    ).resolve()
    lock_args = [
        f"--lock-gpu-clocks={args.sm_clock},{args.sm_clock}",
        f"--mode={args.clock_mode}",
    ]
    benchmark = [
        sys.executable,
        str(root / "benchmarks/bench_union_gqa.py"),
        "--seq-lens",
        args.seq_lens,
        "--sharing-ratios",
        args.sharing_ratios,
        "--cache-modes",
        "warm,cold",
        "--mode-repeats",
        str(args.mode_repeats),
        "--repeat-iters",
        str(args.repeat_iters),
        "--unified-timer",
        "--telemetry-prefix",
        str(destination / "locked"),
        "--csv",
    ]
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    gpu=args.gpu,
                    lock=lock_args,
                    benchmark=benchmark,
                    restore="--reset-gpu-clocks",
                    output=str(destination),
                ),
                indent=2,
            )
        )
        return 0
    destination.mkdir(parents=True, exist_ok=False)
    records = []

    def control(arguments, privileged=False):
        """Run a GPU control command, save its result, and raise on failure."""
        prefix = ["sudo", "-n"] if privileged and os.geteuid() != 0 else []
        command = prefix + ["nvidia-smi", f"--id={args.gpu}"] + arguments
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        records.append(
            dict(
                command=command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )
        (destination / "controls.json").write_text(json.dumps(records, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"GPU command failed: {result.stdout}{result.stderr}")
        return result.stdout

    uuid = control(["--query-gpu=uuid", "--format=csv,noheader,nounits"]).strip()
    if not uuid.startswith("GPU-") or "\n" in uuid:
        raise RuntimeError("Select exactly one GPU by index or UUID")
    args.gpu = uuid
    control(
        [
            "--query-gpu=name,driver_version,clocks.current.sm,clocks.current.memory,power.limit",
            "--format=csv",
        ]
    )
    processes = control(
        ["--query-compute-apps=pid,process_name", "--format=csv,noheader"]
    )
    if processes.strip():
        raise RuntimeError(
            f"GPU is occupied; stop other GPU work before running:\n{processes}"
        )
    control(["-q", "-d", "CLOCK,POWER,PERFORMANCE"])
    metadata = dict(
        gpu_uuid=uuid,
        requested_sm_mhz=args.sm_clock,
        clock_mode=args.clock_mode,
        tolerance_mhz=args.clock_tolerance,
        benchmark=benchmark,
        python=sys.version,
        source_sha256={},
        restore_policy="Start with default clocks; reset GPU clocks on exit; memory clocks and power limit unchanged",
    )
    for name in (
        "benchmarks/bench_union_gqa_locked.py",
        "benchmarks/bench_union_gqa.py",
        "benchmarks/union_gqa_telemetry.py",
        "flashinfer/experimental/union_gqa/csrc/sm100a/union_gqa_prefill.cu",
    ):
        metadata["source_sha256"][name] = hashlib.sha256(
            (root / name).read_bytes()
        ).hexdigest()
    (destination / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = uuid
    environment["PYTHONPATH"] = (
        str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    environment.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/union_gqa_telemetry_jit")

    def interrupted(signum, frame):
        """Raise KeyboardInterrupt so signal handling reaches benchmark cleanup."""
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    # Include the lock attempt in cleanup: restore even if setting it partially fails.
    try:
        control(lock_args, privileged=True)
        print(
            f"Requested {args.sm_clock} MHz, mode {args.clock_mode}; results: {destination}",
            flush=True,
        )
        with (
            (destination / "latencies.csv").open("w") as output,
            (destination / "benchmark.stderr").open("w") as errors,
        ):
            process = subprocess.Popen(
                benchmark,
                cwd=root,
                env=environment,
                stdout=output,
                stderr=errors,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=args.timeout)
                if code:
                    raise RuntimeError(
                        f"Benchmark exited {code}; inspect benchmark.stderr"
                    )
            finally:
                stop_benchmark(process)
    finally:
        # A second Ctrl+C must not interrupt the restoration operation.
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        control(["--reset-gpu-clocks"], privileged=True)
        control(
            [
                "--query-gpu=clocks.current.sm,clocks.current.memory,power.limit",
                "--format=csv",
            ]
        )
        print("GPU clocks reset to defaults.", flush=True)

    valid, rows = summarize(
        destination / "locked.summary.json",
        destination / "comparison.csv",
        args.sm_clock,
        args.clock_tolerance,
    )
    status = dict(
        valid_fixed_clock_control=valid,
        requested_sm_mhz=args.sm_clock,
        tolerance_mhz=args.clock_tolerance,
        actual_sm_mhz_min=min(row["sm_mhz_min"] for row in rows),
        actual_sm_mhz_max=max(row["sm_mhz_max"] for row in rows),
    )
    (destination / "validation.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    print(f"Summary: {destination / 'comparison.csv'}")
    if not valid:
        print(
            "INVALID fixed-clock control: actual clocks missed the target or samples were absent."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
