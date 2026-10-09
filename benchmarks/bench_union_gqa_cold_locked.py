# Copyright (c) 2026 by NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Run cold-L2 Union-GQA/primsts with verified SM and memory clock locks.

Run sudo -v first. Only nvidia-smi controls use sudo; CUDA runs as the caller.
Start with default clocks; both clock locks are reset on exit. Publish the
README table only after every case has passed actual-clock validation.
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


def validate_clocks(path, sm_clock, memory_clock, tolerance, expected_cases):
    phases = json.loads(path.read_text())
    groups = {}
    for phase in phases:
        if phase["phase"] == "timing":
            key = (phase["seq_len"], phase["sharing_ratio"])
            groups.setdefault(key, []).append(phase)
    if set(groups) != expected_cases:
        raise RuntimeError(f"Missing or unexpected timing cases in {path}")
    for key, group in groups.items():
        if sum(p["samples"] for p in group) == 0:
            raise RuntimeError(f"No actual-clock samples for {key} in {path}")
        for phase in group:
            if not phase["cold_l2_cache"]:
                raise RuntimeError(f"Non-cold measurement at {key}")
            if phase["samples"] == 0:
                continue
            for metric, target in (("sm_mhz", sm_clock), ("memory_mhz", memory_clock)):
                low = phase.get(metric + "_min")
                high = phase.get(metric + "_max")
                if (
                    low is None
                    or high is None
                    or not target - tolerance <= low <= high <= target + tolerance
                ):
                    raise RuntimeError(
                        f"Clock lock did not hold: {key}, {metric}={low}..{high}, target={target}"
                    )
    return groups


def read_measurements(path):
    """Keep known compiler diagnostics out of CSV parsing; retain raw input.

    Native compiler/subprocess output can bypass Python redirect_stdout.
    Skip only this known diagnostic prefix, never malformed measurement rows.
    """
    with path.open() as source:
        lines = []
        diagnostics = []
        for line in source:
            if line.startswith("[head-loop-overlap] WARNING:"):
                diagnostics.append(line)
            else:
                lines.append(line)
    if diagnostics:
        path.with_suffix(".diagnostics.log").write_text("".join(diagnostics))
    return list(csv.DictReader(lines))


def make_report(destination, expected_cases, sm_clock, memory_clock, tolerance):
    measurements = {}
    for backend in ("union", "primsts"):
        validate_clocks(
            destination / f"{backend}.summary.json",
            sm_clock,
            memory_clock,
            tolerance,
            expected_cases,
        )
        rows = read_measurements(destination / f"{backend}.csv")
        data = {(int(r["seq_len"]), float(r["requested_sharing"])): r for r in rows}
        if len(rows) != len(expected_cases) or set(data) != expected_cases:
            raise RuntimeError(f"Incomplete {backend} CSV")
        for (n, ratio), row in data.items():
            if (
                row["cold_l2_cache"] != "True"
                or float(row["actual_selected_block_sharing"]) != ratio
            ):
                raise RuntimeError("Wrong cache mode or actual sharing")
            if (
                int(row["max_union_blocks"]) != 16 + 7 * round(16 * (1 - ratio))
                or int(row["effective_pairs_per_head"]) != n * 1024
            ):
                raise RuntimeError("Wrong union size or selected-key count")
        measurements[backend] = data
    comparison = []
    for n, ratio in sorted(expected_cases):
        u = measurements["union"][n, ratio]
        p = measurements["primsts"][n, ratio]
        union_us, primsts_us = float(u["union_us"]), float(p["primsts_us"])
        comparison.append(
            dict(
                seq_len=n,
                sharing_ratio=ratio,
                union_cold_us=union_us,
                primsts_cold_us=primsts_us,
                primsts_over_union=primsts_us / union_us,
                sm_clock_mhz=sm_clock,
                memory_clock_mhz=memory_clock,
                max_abs_diff=max(float(u["max_abs_diff"]), float(p["max_abs_diff"])),
            )
        )
    with (destination / "comparison.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    lengths = sorted({n for n, _ in expected_cases})
    ratios = sorted({r for _, r in expected_cases})
    lookup = {(r["seq_len"], r["sharing_ratio"]): r for r in comparison}
    report = f"Both backends use cold L2; verified SM {sm_clock} MHz and memory {memory_clock} MHz (tolerance ±{tolerance} MHz).\n\n"
    report += "Each cell shows Union-GQA standalone cold-L2 latency in microseconds (speedup over primsts). Speedup is primsts / standalone latency at the same length and ratio; values above 1 mean standalone is faster. The primsts row is the arithmetic mean across the measured ratios, with individual values retained in comparison.csv.\n\n"
    report += "| Actual sharing | " + " | ".join(f"{n:,}" for n in lengths) + " |\n"
    report += "| ---: | " + " | ".join("---:" for _ in lengths) + " |\n"
    for ratio in ratios:
        report += (
            f"| {ratio * 100:g}% | "
            + " | ".join(
                f"{lookup[n, ratio]['union_cold_us']:.2f} ({lookup[n, ratio]['primsts_over_union']:.2f}x)"
                for n in lengths
            )
            + " |\n"
        )
    report += (
        "| **primsts cold L2 (mean)** | "
        + " | ".join(
            f"**{statistics.mean(lookup[n, r]['primsts_cold_us'] for r in ratios):.2f}**"
            for n in lengths
        )
        + " |\n"
    )
    (destination / "table.md").write_text(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--sm-clock", type=int, default=1500)
    parser.add_argument("--memory-clock", type=int, default=3996)
    parser.add_argument("--clock-tolerance", type=int, default=15)
    parser.add_argument("--seq-lens", default="2048,4096,8192,16384,32768")
    parser.add_argument(
        "--sharing-ratios", default=",".join(str(r / 16) for r in range(17))
    )
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    lengths = [int(x) for x in args.seq_lens.split(",")]
    ratios = [float(x) for x in args.sharing_ratios.split(",")]
    if (
        min(args.sm_clock, args.memory_clock, args.iters, args.rounds) <= 0
        or args.clock_tolerance < 0
    ):
        parser.error(
            "Clocks, iterations and rounds must be positive; tolerance nonnegative"
        )
    if any(n < 1024 or n % 16 for n in lengths) or any(
        not 0 <= r <= 1 or abs(16 * r - round(16 * r)) > 1e-9 for r in ratios
    ):
        parser.error(
            "Query lengths must be >=1024 and divisible by 16; ratios must be k/16"
        )
    if args.update_readme and (
        set(lengths) != {2048, 4096, 8192, 16384, 32768}
        or set(ratios) != {r / 16 for r in range(17)}
    ):
        parser.error("README update requires the complete 85-case grid")
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = (
        args.output_dir
        or root / "benchmarks/results" / f"union_gqa_cold_locked_{stamp}"
    ).resolve()
    common = [
        sys.executable,
        str(root / "benchmarks/bench_union_gqa_primsts_compare.py"),
        "--local-runtime-compat",
        "--cold-l2-cache",
        "--seq-lens",
        args.seq_lens,
        "--sharing-ratios",
        args.sharing_ratios,
        "--iters",
        str(args.iters),
        "--rounds",
        str(args.rounds),
    ]
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    output=str(destination),
                    sm_clock=args.sm_clock,
                    memory_clock=args.memory_clock,
                    benchmark=common,
                    backends=["union", "primsts"],
                    cleanup="Reset SM and memory clocks",
                    update_readme=args.update_readme,
                ),
                indent=2,
            )
        )
        return
    destination.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        requested_sm_mhz=args.sm_clock,
        requested_memory_mhz=args.memory_clock,
        tolerance_mhz=args.clock_tolerance,
        verified=False,
        commands=[],
        jobs={},
        source_sha256={},
    )
    for name in (
        "benchmarks/bench_union_gqa_cold_locked.py",
        "benchmarks/bench_union_gqa_primsts_compare.py",
        "benchmarks/union_gqa_workload.py",
        "benchmarks/union_gqa_telemetry.py",
        "flashinfer/experimental/union_gqa/csrc/sm100a/union_gqa_prefill.cu",
        "flashinfer/experimental/union_gqa/jit.py",
    ):
        metadata["source_sha256"][name] = hashlib.sha256(
            (root / name).read_bytes()
        ).hexdigest()

    def save():
        (destination / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

    def control(arguments, privileged=False):
        prefix = ["sudo", "-n"] if privileged and os.geteuid() != 0 else []
        command = prefix + ["nvidia-smi", f"--id={args.gpu}", *arguments]
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        metadata["commands"].append(
            dict(
                command=command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )
        save()
        if result.returncode:
            raise RuntimeError(f"GPU control failed: {result.stdout}{result.stderr}")
        return result.stdout

    uuid = control(["--query-gpu=uuid", "--format=csv,noheader,nounits"]).strip()
    if not uuid.startswith("GPU-") or "\n" in uuid:
        raise RuntimeError("Select exactly one GPU")
    args.gpu = uuid
    metadata["gpu_uuid"] = uuid
    control(["-q", "-d", "CLOCK,POWER,PERFORMANCE"])
    if control(
        ["--query-compute-apps=pid,process_name", "--format=csv,noheader"]
    ).strip():
        raise RuntimeError("GPU is occupied; stop other GPU jobs first")
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=uuid,
        PYTHONPATH=str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    )
    environment.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/union_gqa_strict_jit")
    expected = {(n, r) for n in lengths for r in ratios}
    process = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    try:
        control(
            [f"--lock-gpu-clocks={args.sm_clock},{args.sm_clock}", "--mode=0"], True
        )
        control([f"--lock-memory-clocks={args.memory_clock},{args.memory_clock}"], True)
        for backend in ("union", "primsts"):
            command = common + [
                "--backend",
                backend,
                "--telemetry-prefix",
                str(destination / backend),
            ]
            print(f"Measuring {backend}: cold L2; output {destination}", flush=True)
            with (
                (destination / f"{backend}.csv").open("w") as output,
                (destination / f"{backend}.stderr").open("w") as errors,
            ):
                process = subprocess.Popen(
                    command,
                    cwd=root,
                    env=environment,
                    stdout=output,
                    stderr=errors,
                    start_new_session=True,
                )
                code = process.wait(timeout=1800)
            metadata["jobs"][backend] = dict(command=command, returncode=code)
            save()
            if code:
                raise RuntimeError(
                    f"{backend} failed; inspect {destination / (backend + '.stderr')}"
                )
            validate_clocks(
                destination / f"{backend}.summary.json",
                args.sm_clock,
                args.memory_clock,
                args.clock_tolerance,
                expected,
            )
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        restore_errors = []
        for command in (["--reset-memory-clocks"], ["--reset-gpu-clocks"]):
            try:
                control(command, True)
            except Exception as error:
                restore_errors.append(str(error))
        if restore_errors:
            raise RuntimeError("Clock cleanup failed: " + "; ".join(restore_errors))
    report = make_report(
        destination, expected, args.sm_clock, args.memory_clock, args.clock_tolerance
    )
    metadata["verified"] = True
    save()
    if args.update_readme:
        readme = root / "flashinfer/experimental/union_gqa/README.md"
        text = readme.read_text()
        start, end = (
            "<!-- COLD_LOCKED_RESULTS_START -->",
            "<!-- COLD_LOCKED_RESULTS_END -->",
        )
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        artifact = os.path.relpath(destination / "comparison.csv", readme.parent)
        readme.write_text(
            before
            + start
            + "\n\n"
            + report
            + f"\n[Individual measurements]({artifact}).\n\n"
            + end
            + after
        )
    print(f"Verified clocks; results: {destination / 'table.md'}", flush=True)


if __name__ == "__main__":
    main()
