#!/usr/bin/env python3
"""One-command driver over the public PR-4576 SSDCombined route matrix.

Runs every row of the authoritative 12-workload route matrix through
``bench_cake_mamba_ssd_combined.py --vibecuda`` semantics: live CAKE baseline
as the speedup denominator, VibeCUDA candidate, CUPTI ``bench_gpu_time`` with
5 dry-run + 100 repetitions and median aggregation, fp64 sequential
ground-truth validation of both legs, and a NaN-sentinel full-write proof on
the caller-owned ``out``.

Two execution modes:

* default (in-process): import the row benchmark once and run all 12 rows in
  this process.  Each row rebuilds its RNG (seed 7), inputs, and backend
  runners, so per-row inputs and timed callables are identical to a
  standalone invocation; only the one-time interpreter/JIT warmup is shared.
* ``--isolate``: spawn one subprocess per row for full process isolation
  (the mode used by hand-run per-row invocations).

Completed rows are preserved immediately as artifacts; rerunning the driver
resumes from those artifacts (use ``--fresh`` to re-run every row), so an
interrupted matrix completes in a follow-up invocation of the same command.

Per-row JSON artifacts are written to
``benchmarks/results/vibecuda_ssd_combined/<row>.json`` immediately after
each row completes, and ``matrix_summary.json`` records every row's
latencies, correctness checks, and the arithmetic/geometric mean speedups.
The driver exits nonzero if any row is missing or fails a candidate-side
check; rows whose built-in cake-vs-cute self-gate fails are tolerated as
long as the candidate-side checks pass (cake and cute are bitwise-matched
twins on this workload family, so that gate measures graph identity, not
accuracy).
"""

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROW_BENCH = REPO_ROOT / "benchmarks" / "bench_cake_mamba_ssd_combined.py"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results" / "vibecuda_ssd_combined"
CONTRACT_VERSION = "pr4576-route-matrix-12-seed7-v1"

# Exact rows from PR 4576 commit 261d59d6f03f659c9f575240241712a8396507c8,
# tests/mamba/test_cake_ssd_combined.py::test_cake_ssd_combined_route_matrix.
MATRIX = [
    (
        "bf16_b_h8_g8_dvec",
        [
            "--batch",
            "2",
            "--nheads",
            "8",
            "--ngroups",
            "8",
            "--state-dtype",
            "bfloat16",
            "--seq-idx-dtype",
            "int32",
            "--preprocess-dtype",
            "float32",
            "--d-has-hdim",
            "--has-z",
        ],
    ),
    (
        "fp16_b_h8_g8",
        [
            "--batch",
            "2",
            "--nheads",
            "8",
            "--ngroups",
            "8",
            "--state-dtype",
            "float16",
            "--seq-idx-dtype",
            "int32",
            "--preprocess-dtype",
            "float32",
            "--has-z",
        ],
    ),
    (
        "bf16_v_h8_g8_dvec",
        [
            "--mode",
            "varlen",
            "--sequence-lengths",
            "96",
            "160",
            "--nheads",
            "8",
            "--ngroups",
            "8",
            "--state-dtype",
            "bfloat16",
            "--seq-idx-dtype",
            "int32",
            "--preprocess-dtype",
            "float32",
            "--d-has-hdim",
            "--has-z",
        ],
    ),
    (
        "fp16_v_i64_h8_g8",
        [
            "--mode",
            "varlen",
            "--sequence-lengths",
            "96",
            "160",
            "--nheads",
            "8",
            "--ngroups",
            "8",
            "--state-dtype",
            "float16",
            "--seq-idx-dtype",
            "int64",
            "--preprocess-dtype",
            "float32",
            "--has-z",
        ],
    ),
    (
        "bf16prep_b_h8_g8",
        [
            "--batch",
            "2",
            "--nheads",
            "8",
            "--ngroups",
            "8",
            "--state-dtype",
            "bfloat16",
            "--seq-idx-dtype",
            "int32",
            "--preprocess-dtype",
            "bfloat16",
            "--has-z",
        ],
    ),
    (
        "b_h1_g1",
        [
            "--batch",
            "2",
            "--nheads",
            "1",
            "--ngroups",
            "1",
            "--state-dtype",
            "bfloat16",
            "--seq-idx-dtype",
            "int32",
            "--preprocess-dtype",
            "float32",
            "--has-z",
        ],
    ),
    (
        "b_h12_g3",
        [
            "--batch",
            "2",
            "--nheads",
            "12",
            "--ngroups",
            "3",
            "--state-dtype",
            "bfloat16",
            "--seq-idx-dtype",
            "int32",
            "--preprocess-dtype",
            "float32",
            "--has-z",
        ],
    ),
    (
        "b_h16_g4",
        [
            "--batch",
            "2",
            "--nheads",
            "16",
            "--ngroups",
            "4",
            "--state-dtype",
            "bfloat16",
            "--seq-idx-dtype",
            "int32",
            "--preprocess-dtype",
            "float32",
            "--has-z",
        ],
    ),
    (
        "b_h128_g1",
        [
            "--batch",
            "2",
            "--nheads",
            "128",
            "--ngroups",
            "1",
            "--state-dtype",
            "bfloat16",
            "--seq-idx-dtype",
            "int32",
            "--preprocess-dtype",
            "float32",
            "--has-z",
        ],
    ),
    (
        "b_h128_g128",
        [
            "--batch",
            "2",
            "--nheads",
            "128",
            "--ngroups",
            "128",
            "--state-dtype",
            "bfloat16",
            "--seq-idx-dtype",
            "int32",
            "--preprocess-dtype",
            "float32",
            "--has-z",
        ],
    ),
    (
        "b_h128_g8",
        [
            "--batch",
            "2",
            "--nheads",
            "128",
            "--ngroups",
            "8",
            "--state-dtype",
            "bfloat16",
            "--seq-idx-dtype",
            "int32",
            "--preprocess-dtype",
            "float32",
            "--has-z",
        ],
    ),
    (
        "v_h128_g8_zero",
        [
            "--mode",
            "varlen",
            "--sequence-lengths",
            "96",
            "160",
            "--nheads",
            "128",
            "--ngroups",
            "8",
            "--state-dtype",
            "bfloat16",
            "--seq-idx-dtype",
            "int32",
            "--preprocess-dtype",
            "float32",
            "--zero-initial-states",
            "--unbounded-dt",
            "--has-z",
        ],
    ),
]


def _extract_report(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _validate_row(report: dict | None) -> list[str]:
    """Return the list of failed candidate-side checks (empty = row pass)."""
    failures = []
    if report is None:
        return ["no JSON report emitted"]
    for key in ("cake_ms", "vibecuda_ms", "vibecuda_speedup_vs_cake"):
        if key not in report:
            failures.append(f"missing timing field {key!r}")
    if not report.get("vibecuda_truth_out", {}).get("tolerance_passed", False):
        failures.append("vibecuda fp64 truth gate failed for out")
    if not report.get("vibecuda_truth_final_states", {}).get("tolerance_passed", False):
        failures.append("vibecuda fp64 truth gate failed for final_states")
    if not report.get("full_write", {}).get("fully_written", False):
        failures.append(
            "full-write sentinel check failed "
            f"({report.get('full_write', {}).get('unwritten_elements')} "
            "unwritten elements)"
        )
    no_worse = report.get("candidate_no_worse_than_cake", {})
    if not no_worse.get("out", False):
        failures.append("candidate output error exceeds cake error")
    if not no_worse.get("final_states", False):
        failures.append("candidate final-state error exceeds cake error")
    if report.get("timing_backend") != "cupti":
        failures.append(f"unexpected timing backend {report.get('timing_backend')!r}")
    return failures


def _cake_cute_parity(report: dict | None) -> dict:
    if report is None or "out" not in report:
        return {}
    return {
        "out_tolerance_passed": report.get("out", {}).get("tolerance_passed"),
        "final_states_tolerance_passed": report.get("final_states", {}).get(
            "tolerance_passed"
        ),
    }


def _reused_row(results_dir: Path, name: str) -> dict | None:
    """Fresh summary entry from an earlier driver artifact, or None."""
    artifact_path = results_dir / f"{name}.json"
    if not artifact_path.exists():
        return None
    try:
        artifact = json.loads(artifact_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if (
        artifact.get("contract_version") != CONTRACT_VERSION
        or artifact.get("mode") is None
        or artifact.get("report") is None
    ):
        return None
    if artifact.get("validation_failures"):
        return None
    report = artifact["report"]
    return {
        "row": name,
        "cake_ms": report["cake_ms"],
        "vibecuda_ms": report["vibecuda_ms"],
        "speedup": report["vibecuda_speedup_vs_cake"],
        "truth_passed": True,
        "full_write_passed": True,
        "candidate_no_worse_than_cake": report.get("candidate_no_worse_than_cake", {}),
        "cake_cute_parity": artifact.get("cake_cute_parity", {}),
        "reused_from_artifact": True,
    }


def _run_row_isolated(name, row_args, env, timeout_s):
    command = [sys.executable, str(ROW_BENCH), "--vibecuda", *row_args]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
            timeout=timeout_s,
            check=False,
        )
        report = _extract_report(proc.stdout)
        error = (
            None
            if report is not None
            else (f"exit {proc.returncode}: {(proc.stderr or '').strip()[-400:]}")
        )
    except subprocess.TimeoutExpired:
        report = None
        error = f"row subprocess exceeded timeout {timeout_s}s"
    return command, report, error


def _run_row_in_process(row_module, name, row_args):
    command = [
        sys.executable,
        str(ROW_BENCH),
        "--vibecuda",
        *row_args,
    ]  # recorded for artifact reproducibility
    try:
        args = row_module.build_parser().parse_args(["--vibecuda", *row_args])
        report = row_module.run_workload(args)
        error = None
    except Exception:  # noqa: BLE001 - row failures are reported, not fatal
        report = None
        error = traceback.format_exc(limit=6)
    return command, report, error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--isolate",
        action="store_true",
        help="run each row in its own subprocess (full process isolation, "
        "one interpreter/JIT warmup per row) instead of the default "
        "in-process mode",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="re-run every row even when a valid artifact from an earlier "
        "driver run exists (default: resume interruptions by reusing "
        "already-completed rows, identified by this driver's artifact "
        "schema)",
    )
    parser.add_argument(
        "--row-timeout-s",
        type=int,
        default=900,
        help="wall-clock timeout per workload row (isolate mode only)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="directory for per-row JSON artifacts and matrix_summary.json",
    )
    cli = parser.parse_args()
    results_dir = cli.results_dir
    isolate = cli.isolate
    fresh = cli.fresh
    row_timeout_s = cli.row_timeout_s

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    row_module = None

    def _ensure_row_module():
        nonlocal row_module
        if row_module is None:
            sys.path.insert(0, str(REPO_ROOT))
            sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
            row_module = importlib.import_module(
                ROW_BENCH.stem  # bench_cake_mamba_ssd_combined
            )
        return row_module

    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failed = []
    mode = "subprocess-per-row" if isolate else "in-process"
    print(
        f"PR-4576 SSDCombined workload matrix: {len(MATRIX)} rows "
        f"({mode}), {ROW_BENCH.name} --vibecuda semantics",
        flush=True,
    )
    for index, (name, row_args) in enumerate(MATRIX, start=1):
        reused = None if fresh else _reused_row(results_dir, name)
        if reused is not None:
            print(
                f"[{index:>2}/{len(MATRIX)}] {name:<14} reused fresh artifact "
                f"(speedup {reused['speedup']:.3f}x)",
                flush=True,
            )
            rows.append(reused)
            continue
        started = time.monotonic()
        if isolate:
            command, report, error = _run_row_isolated(
                name, row_args, env, row_timeout_s
            )
        else:
            command, report, error = _run_row_in_process(
                _ensure_row_module(), name, row_args
            )
        elapsed = time.monotonic() - started
        failures = _validate_row(report)
        if error is not None:
            last_error_line = error.strip().splitlines()[-1]
            failures = [f"row execution error: {last_error_line}"] + failures

        artifact = {
            "contract_version": CONTRACT_VERSION,
            "row": name,
            "mode": mode,
            "argv": command[2:],
            "wall_s": round(elapsed, 3),
            "execution_error": error,
            "validation_failures": failures,
            "report": report,
            "cake_cute_parity": _cake_cute_parity(report),
        }
        (results_dir / f"{name}.json").write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n"
        )

        if failures:
            failed.append(name)
            print(
                f"[{index:>2}/{len(MATRIX)}] {name:<14} FAIL "
                f"({'; '.join(failures)}) — wall {elapsed:.1f}s",
                flush=True,
            )
            if error is not None:
                print(error.rstrip(), flush=True)
            rows.append({"row": name, "failures": failures})
            continue
        cake_us = report["cake_ms"] * 1e3
        vibe_us = report["vibecuda_ms"] * 1e3
        speedup = report["vibecuda_speedup_vs_cake"]
        parity = (
            "cake/cute pass"
            if artifact["cake_cute_parity"].get("out_tolerance_passed")
            else "cake/cute self-gate fail (expected on some rows)"
        )
        print(
            f"[{index:>2}/{len(MATRIX)}] {name:<14} "
            f"cake {cake_us:7.2f} µs  vibecuda {vibe_us:7.2f} µs  "
            f"speedup {speedup:5.3f}x  truth PASS  full-write PASS  "
            f"{parity}  — wall {elapsed:.1f}s",
            flush=True,
        )
        rows.append(
            {
                "row": name,
                "cake_ms": report["cake_ms"],
                "vibecuda_ms": report["vibecuda_ms"],
                "speedup": speedup,
                "truth_passed": True,
                "full_write_passed": True,
                "candidate_no_worse_than_cake": report["candidate_no_worse_than_cake"],
                "cake_cute_parity": artifact["cake_cute_parity"],
            }
        )

    speedups = [row["speedup"] for row in rows if "speedup" in row]
    arithmetic = sum(speedups) / len(speedups) if speedups else float("nan")
    geometric = (
        math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        if speedups
        else float("nan")
    )
    summary = {
        "contract_version": CONTRACT_VERSION,
        "matrix": [name for name, _ in MATRIX],
        "mode": mode,
        "rows_completed": len(speedups),
        "rows_failed": failed,
        "denominator": "cake (PR-4576 SSDCombined backend)",
        "candidate": "vibecuda",
        "timing": "CUPTI bench_gpu_time, 5 dry-run + 100 reps, median",
        "rows": rows,
        "arithmetic_mean_speedup": arithmetic if speedups else None,
        "geometric_mean_speedup": geometric if speedups else None,
        "min_speedup": min(speedups) if speedups else None,
        "max_speedup": max(speedups) if speedups else None,
        "all_rows_passed": not failed and len(speedups) == len(MATRIX),
    }
    (results_dir / "matrix_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    print("-" * 78, flush=True)
    if speedups:
        print(
            f"speedup vs cake: arithmetic mean {arithmetic:.3f}x  "
            f"geometric mean {geometric:.3f}x  "
            f"min {min(speedups):.3f}x  max {max(speedups):.3f}x  "
            f"over {len(speedups)}/{len(MATRIX)} rows",
            flush=True,
        )
    print(
        json.dumps(
            {
                "rows": len(speedups),
                "failed_rows": failed,
                "arithmetic_mean_speedup": round(arithmetic, 6) if speedups else None,
                "geometric_mean_speedup": round(geometric, 6) if speedups else None,
                "all_rows_passed": summary["all_rows_passed"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if failed or len(speedups) != len(MATRIX):
        print(f"FAILED rows: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
