#!/usr/bin/env python3
"""Run the fixed Hopper FP8 P02/P03 tokens-per-rank benchmark sweep.

The default run covers every public non-swap and swap-AB tile, both FP8
scale modes, and tokens-per-rank from 512 through 32768 in powers of two.
Each CSV owns one rank/scale/order/tile configuration and contains one row
per attempt so failed runs and forced reruns remain auditable.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import importlib.util
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PERF_SCRIPT = SCRIPT_DIR / "run_perf_test.sh"
PLOT_SCRIPT = SCRIPT_DIR / "plot_token_sweep.py"
SUMMARY_SCRIPT = SCRIPT_DIR / "summarize_token_sweep.py"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "benchmark_data"
BENCHMARK_REQUIREMENTS = SCRIPT_DIR / "benchmark_requirements.txt"

TOKEN_SIZES = (512, 1024, 2048, 4096, 8192, 16384, 32768)
TOPK = 6
TOTAL_EXPERTS = 384
EP_SIZE = 4
LOCAL_EXPERTS = TOTAL_EXPERTS // EP_SIZE
HIDDEN = 7168
INTERMEDIATE_DOWNPROJ = 3072
INTERMEDIATE_GATEUP = INTERMEDIATE_DOWNPROJ * 2
GATE_UP_CLAMP = "10.0"
WARMUP = 3
ITERS = 20
TIMEOUT_SECONDS = 3600
TILE_K = 128

SCALE_MODES = ("per_tensor", "blockwise")
RANK_MODES = (
    ("singlerank", "P02", 1),
    ("multirank", "P03", EP_SIZE),
)
RANK_MODE_CHOICES = ("singlerank", "multirank", "both")

DISPLAY_ENV_KEYS = (
    "PYTHON",
    "PYTHONUNBUFFERED",
    "DSV4_TOKENS_PER_RANK",
    "DSV4_TOPK",
    "DSV4_TOTAL_EXPERTS",
    "DSV4_LOCAL_EXPERTS",
    "DSV4_SINGLE_MEGA_TOTAL_EXPERTS",
    "DSV4_HIDDEN",
    "DSV4_INTERMEDIATE_DOWNPROJ",
    "DSV4_INTERMEDIATE_GATEUP",
    "DSV4_ROUTE_ROWS",
    "DSV4_GATE_UP_CLAMP",
    "FP8_ACCUM_MODE",
    "FP8_NON_SWAP_M",
    "FP8_NON_SWAP_N",
    "FP8_SWAP_AB_M",
    "FP8_SWAP_AB_N",
    "PERF_WARMUP",
    "PERF_ITERS",
    "TIMEOUT_SECONDS",
    "MEGA_NPROC",
    "NCCL_NVLS_ENABLE",
    "NVSHMEM_DISABLE_NVLS",
)

CSV_FIELDS = (
    "run_date",
    "timestamp_utc",
    "attempt",
    "case",
    "rank_mode",
    "scale_mode",
    "operand_order",
    "tile_m",
    "tile_n",
    "tile_k",
    "tokens_per_rank",
    "topk",
    "routed_tokens_per_rank",
    "world_size",
    "total_experts",
    "local_experts",
    "hidden",
    "intermediate_downproj",
    "intermediate_gateup",
    "warmup",
    "iters",
    "status",
    "return_code",
    "wall_time_s",
    "min_rank",
    "min_mega_us",
    "max_mega_us",
    "mean_mega_us",
    "rank_0_mega_us",
    "rank_1_mega_us",
    "rank_2_mega_us",
    "rank_3_mega_us",
    "reported_min_rank",
    "reported_min_mega_us",
    "reported_min_topk_us",
    "reported_min_total_us",
    "fc1_flops_per_rank",
    "fc2_flops_per_rank",
    "total_flops_per_rank",
    "critical_tflops_per_rank",
    "git_commit",
    "gpu_names",
    "log_file",
    "command",
)

_MIN_LINE_RE = re.compile(
    r"min_rank_by_mega=rank\s+(?P<rank>-?\d+):\s+"
    r"mega=(?P<mega>n/a|[0-9]+(?:\.[0-9]+)?\s+us)\s+"
    r"topk_reduce=(?P<topk>n/a|[0-9]+(?:\.[0-9]+)?\s+us)\s+"
    r"total=(?P<total>n/a|[0-9]+(?:\.[0-9]+)?\s+us)"
)
_RANK_LINE_RE = re.compile(
    r"^rank_(?P<rank>\d+):\s+(?P<time>n/a|[0-9]+(?:\.[0-9]+)?\s+us)$"
)


@dataclass(frozen=True)
class BenchmarkCase:
    rank_mode: str
    perf_case: str
    world_size: int
    scale_mode: str
    operand_order: str
    tile_m: int
    tile_n: int

    @property
    def scale_tag(self) -> str:
        return "pertensor" if self.scale_mode == "per_tensor" else "blockwise"

    @property
    def order_tag(self) -> str:
        return "swapab" if self.operand_order == "swap_ab" else "nonswapab"

    def stem(self, run_date: str) -> str:
        return (
            f"{run_date}_{self.rank_mode}_{self.scale_tag}_{self.order_tag}_"
            f"TileM{self.tile_m}_TileN{self.tile_n}"
        )

    def csv_path(self, output_dir: Path, run_date: str) -> Path:
        return output_dir / run_date / f"{self.stem(run_date)}.csv"


@dataclass(frozen=True)
class ParsedTiming:
    rank_times_us: dict[int, float]
    reported_min_rank: int | None
    reported_min_mega_us: float | None
    reported_min_topk_us: float | None
    reported_min_total_us: float | None


def compute_gemm_flops_per_rank(
    tokens_per_rank: int,
    topk: int = TOPK,
    hidden: int = HIDDEN,
    intermediate_downproj: int = INTERMEDIATE_DOWNPROJ,
) -> tuple[int, int, int]:
    """Return FC1, FC2, and total GEMM FLOPs executed by one rank."""
    routed_tokens = tokens_per_rank * topk
    intermediate_gateup = intermediate_downproj * 2
    fc1_flops = 2 * routed_tokens * hidden * intermediate_gateup
    fc2_flops = 2 * routed_tokens * hidden * intermediate_downproj
    return fc1_flops, fc2_flops, fc1_flops + fc2_flops


def effective_tflops(flops: int, time_us: float) -> float:
    """Convert GEMM FLOPs and elapsed microseconds to effective TFLOPS."""
    if time_us <= 0.0:
        raise ValueError("time_us must be positive")
    return flops / time_us / 1_000_000.0


def _parse_time_us(value: str) -> float | None:
    if value == "n/a":
        return None
    return float(value.removesuffix(" us"))


def parse_profiler_output(lines: Iterable[str]) -> ParsedTiming:
    """Parse the rank-level mega CUDA times printed by mega_runner.py."""
    rank_times: dict[int, float] = {}
    section: str | None = None
    reported_min_rank: int | None = None
    reported_min_mega_us: float | None = None
    reported_min_topk_us: float | None = None
    reported_min_total_us: float | None = None

    for raw_line in lines:
        line = raw_line.strip()
        min_match = _MIN_LINE_RE.search(line)
        if min_match:
            reported_min_rank = int(min_match.group("rank"))
            reported_min_mega_us = _parse_time_us(min_match.group("mega"))
            reported_min_topk_us = _parse_time_us(min_match.group("topk"))
            reported_min_total_us = _parse_time_us(min_match.group("total"))
            continue
        if line == "mega:":
            section = "mega"
            continue
        if line == "topk:":
            section = "topk"
            continue
        rank_match = _RANK_LINE_RE.match(line)
        if rank_match and section == "mega":
            value = _parse_time_us(rank_match.group("time"))
            if value is not None:
                rank_times[int(rank_match.group("rank"))] = value

    return ParsedTiming(
        rank_times_us=rank_times,
        reported_min_rank=reported_min_rank,
        reported_min_mega_us=reported_min_mega_us,
        reported_min_topk_us=reported_min_topk_us,
        reported_min_total_us=reported_min_total_us,
    )


def _read_tuple_constant(path: Path, name: str) -> tuple[int, ...]:
    """Read a literal tuple constant without importing the CuTe modules."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, tuple) or not all(isinstance(item, int) for item in value):
            raise ValueError(f"{path}:{name} must be a literal tuple of integers")
        return value
    raise ValueError(f"Unable to find {name} in {path}")


def supported_tiles() -> dict[str, tuple[tuple[int, int], ...]]:
    non_swap_file = SCRIPT_DIR / "epilogue_fp8.py"
    swap_file = SCRIPT_DIR / "epilogue_fp8_swapab.py"
    non_swap_m = _read_tuple_constant(non_swap_file, "NonSwapTileMChoices")
    non_swap_n = _read_tuple_constant(non_swap_file, "NonSwapTileNChoices")
    swap_m = _read_tuple_constant(swap_file, "SwapABTileMChoices")
    swap_n = _read_tuple_constant(swap_file, "SwapABTokenTileNChoices")
    return {
        "non_swap_ab": tuple((m, n) for m in non_swap_m for n in non_swap_n),
        "swap_ab": tuple((m, n) for m in swap_m for n in swap_n),
    }


def build_cases(
    smoke: bool = False, rank_mode: str = "both"
) -> tuple[BenchmarkCase, ...]:
    if rank_mode not in RANK_MODE_CHOICES:
        raise ValueError(
            f"rank_mode must be one of {','.join(RANK_MODE_CHOICES)}"
        )
    tiles = supported_tiles()
    selected_rank_modes = (
        RANK_MODES
        if rank_mode == "both"
        else tuple(mode for mode in RANK_MODES if mode[0] == rank_mode)
    )
    cases = tuple(
        BenchmarkCase(
            case_rank_mode, perf_case, world_size, scale_mode, order, m, n
        )
        for case_rank_mode, perf_case, world_size in selected_rank_modes
        for scale_mode in SCALE_MODES
        for order in ("non_swap_ab", "swap_ab")
        for m, n in tiles[order]
    )
    if not smoke:
        return cases
    return tuple(
        case
        for case in cases
        if case.scale_mode == "per_tensor"
        and case.operand_order == "non_swap_ab"
        and (case.tile_m, case.tile_n) == tiles["non_swap_ab"][0]
    )


def _case_environment(case: BenchmarkCase, tokens_per_rank: int) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("FP8_SCALE_MODES", None)
    env.pop("FP8_SCALE_MODE", None)
    env.pop("FP8_SWAP_AB", None)
    env.pop("MEGA_NO_DIST", None)
    env.update(
        {
            "PYTHON": sys.executable,
            "PYTHONUNBUFFERED": "1",
            "DSV4_TOKENS_PER_RANK": str(tokens_per_rank),
            "DSV4_TOPK": str(TOPK),
            "DSV4_TOTAL_EXPERTS": str(TOTAL_EXPERTS),
            "DSV4_LOCAL_EXPERTS": str(LOCAL_EXPERTS),
            "DSV4_SINGLE_MEGA_TOTAL_EXPERTS": str(LOCAL_EXPERTS),
            "DSV4_HIDDEN": str(HIDDEN),
            "DSV4_INTERMEDIATE_DOWNPROJ": str(INTERMEDIATE_DOWNPROJ),
            "DSV4_INTERMEDIATE_GATEUP": str(INTERMEDIATE_GATEUP),
            "DSV4_ROUTE_ROWS": str(tokens_per_rank * TOPK),
            "DSV4_GATE_UP_CLAMP": GATE_UP_CLAMP,
            "FP8_ACCUM_MODE": "1xacc",
            "PERF_WARMUP": str(WARMUP),
            "PERF_ITERS": str(ITERS),
            "TIMEOUT_SECONDS": str(TIMEOUT_SECONDS),
            "MEGA_NPROC": str(EP_SIZE),
            "NCCL_NVLS_ENABLE": "0",
            "NVSHMEM_DISABLE_NVLS": "1",
        }
    )
    if case.operand_order == "swap_ab":
        env.update(
            {
                "FP8_SWAP_AB_M": str(case.tile_m),
                "FP8_SWAP_AB_N": str(case.tile_n),
            }
        )
        env.pop("FP8_NON_SWAP_M", None)
        env.pop("FP8_NON_SWAP_N", None)
    else:
        env.update(
            {
                "FP8_NON_SWAP_M": str(case.tile_m),
                "FP8_NON_SWAP_N": str(case.tile_n),
            }
        )
        env.pop("FP8_SWAP_AB_M", None)
        env.pop("FP8_SWAP_AB_N", None)
    return env


def _case_cli_args(case: BenchmarkCase) -> list[str]:
    scale_mode = "per-tensor" if case.scale_mode == "per_tensor" else "blockwise"
    args = ["--scale-mode", scale_mode]
    if case.operand_order == "swap_ab":
        args.append("--swapab")
    return args


def _display_command(case: BenchmarkCase, tokens_per_rank: int) -> str:
    env = _case_environment(case, tokens_per_rank)
    assignments = [f"{key}={env[key]}" for key in DISPLAY_ENV_KEYS if key in env]
    command = ["bash", str(PERF_SCRIPT), *_case_cli_args(case), case.perf_case]
    return shlex.join(["env", *assignments, *command])


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _successful_tokens(path: Path) -> set[int]:
    return {
        int(row["tokens_per_rank"])
        for row in _read_csv_rows(path)
        if row.get("status") == "pass"
    }


def _next_attempt(path: Path, tokens_per_rank: int) -> int:
    attempts = [
        int(row.get("attempt", "0") or 0)
        for row in _read_csv_rows(path)
        if int(row["tokens_per_rank"]) == tokens_per_rank
    ]
    return max(attempts, default=0) + 1


def _append_csv_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _run_and_tee(
    command: Sequence[str], env: dict[str, str], log_path: Path
) -> tuple[int, list[str]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=SCRIPT_DIR.parent,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
                log_handle.flush()
                lines.append(line)
        except KeyboardInterrupt:
            process.terminate()
            process.wait(timeout=30)
            raise
        return process.wait(), lines


def _run_case(
    case: BenchmarkCase,
    tokens_per_rank: int,
    output_dir: Path,
    run_date: str,
    git_commit: str,
    gpu_names: str,
) -> tuple[str, Path]:
    csv_path = case.csv_path(output_dir, run_date)
    attempt = _next_attempt(csv_path, tokens_per_rank)
    log_name = f"{case.stem(run_date)}_Tokens{tokens_per_rank}_Attempt{attempt}.log"
    log_path = csv_path.parent / log_name
    env = _case_environment(case, tokens_per_rank)
    command = ["bash", str(PERF_SCRIPT), *_case_cli_args(case), case.perf_case]
    display_command = _display_command(case, tokens_per_rank)

    print("=" * 79)
    print(
        f"[SWEEP] {case.rank_mode} {case.scale_tag} {case.order_tag} "
        f"M{case.tile_m}N{case.tile_n} tokens_per_rank={tokens_per_rank} "
        f"attempt={attempt}"
    )
    print(f"[CMD] {display_command}")
    print(f"[LOG] {log_path}")
    start = time.monotonic()
    return_code, output_lines = _run_and_tee(command, env, log_path)
    wall_time_s = time.monotonic() - start
    timing = parse_profiler_output(output_lines)

    expected_ranks = set(range(case.world_size))
    parsed_ranks = set(timing.rank_times_us)
    if return_code != 0:
        status = "failed"
    elif parsed_ranks != expected_ranks:
        status = "parse_error"
    else:
        status = "pass"

    rank_values = [timing.rank_times_us[rank] for rank in sorted(parsed_ranks)]
    if rank_values:
        min_rank = min(timing.rank_times_us, key=timing.rank_times_us.__getitem__)
        min_mega_us = min(rank_values)
        max_mega_us = max(rank_values)
        mean_mega_us = statistics.fmean(rank_values)
    else:
        min_rank = ""
        min_mega_us = ""
        max_mega_us = ""
        mean_mega_us = ""

    fc1_flops, fc2_flops, total_flops = compute_gemm_flops_per_rank(
        tokens_per_rank
    )
    critical_tflops = (
        effective_tflops(total_flops, max_mega_us)
        if status == "pass" and isinstance(max_mega_us, float)
        else ""
    )

    row: dict[str, object] = {
        "run_date": run_date,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "attempt": attempt,
        "case": case.perf_case,
        "rank_mode": case.rank_mode,
        "scale_mode": case.scale_mode,
        "operand_order": case.operand_order,
        "tile_m": case.tile_m,
        "tile_n": case.tile_n,
        "tile_k": TILE_K,
        "tokens_per_rank": tokens_per_rank,
        "topk": TOPK,
        "routed_tokens_per_rank": tokens_per_rank * TOPK,
        "world_size": case.world_size,
        "total_experts": LOCAL_EXPERTS if case.world_size == 1 else TOTAL_EXPERTS,
        "local_experts": LOCAL_EXPERTS,
        "hidden": HIDDEN,
        "intermediate_downproj": INTERMEDIATE_DOWNPROJ,
        "intermediate_gateup": INTERMEDIATE_GATEUP,
        "warmup": WARMUP,
        "iters": ITERS,
        "status": status,
        "return_code": return_code,
        "wall_time_s": f"{wall_time_s:.3f}",
        "min_rank": min_rank,
        "min_mega_us": min_mega_us,
        "max_mega_us": max_mega_us,
        "mean_mega_us": mean_mega_us,
        "reported_min_rank": timing.reported_min_rank,
        "reported_min_mega_us": timing.reported_min_mega_us,
        "reported_min_topk_us": timing.reported_min_topk_us,
        "reported_min_total_us": timing.reported_min_total_us,
        "fc1_flops_per_rank": fc1_flops,
        "fc2_flops_per_rank": fc2_flops,
        "total_flops_per_rank": total_flops,
        "critical_tflops_per_rank": critical_tflops,
        "git_commit": git_commit,
        "gpu_names": gpu_names,
        "log_file": log_path.relative_to(output_dir).as_posix(),
        "command": display_command,
    }
    for rank in range(EP_SIZE):
        row[f"rank_{rank}_mega_us"] = timing.rank_times_us.get(rank, "")
    _append_csv_row(csv_path, row)
    print(
        f"[RECORDED] status={status} csv={csv_path} "
        f"rank_times_us={timing.rank_times_us}"
    )
    return status, csv_path


def _run_command_text(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _gpu_names() -> str:
    output = _run_command_text(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    )
    if output == "unknown":
        return output
    return ";".join(line.strip() for line in output.splitlines() if line.strip())


def _require_plot_dependency() -> None:
    if importlib.util.find_spec("matplotlib") is not None:
        return
    raise RuntimeError(
        "Plotting requires matplotlib. Install the fixed benchmark dependencies "
        f"with: {sys.executable} -m pip install -r {BENCHMARK_REQUIREMENTS}"
    )


def _validate_run_date(value: str) -> str:
    if not re.fullmatch(r"\d{8}", value):
        raise argparse.ArgumentTypeError("date must use YYYYMMDD")
    try:
        dt.datetime.strptime(value, "%Y%m%d")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Benchmark-data root; artifacts are stored under ROOT/YYYYMMDD "
            f"(default: {DEFAULT_OUTPUT_DIR})"
        ),
    )
    parser.add_argument(
        "--date",
        type=_validate_run_date,
        default=dt.datetime.now().strftime("%Y%m%d"),
        help="Run date used in filenames, formatted YYYYMMDD.",
    )
    parser.add_argument(
        "--rank-mode",
        choices=RANK_MODE_CHOICES,
        default="both",
        help="Run singlerank (P02), multirank (P03), or both (default: both).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Append a new attempt even when this token already passed.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed or unparseable case.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not invoke plot_token_sweep.py after the benchmark.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the fixed CSV configurations without running them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all fixed commands without creating files or running kernels.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Validation-only subset for the selected rank mode at token=512.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = _build_parser().parse_args(argv)
    if not PERF_SCRIPT.is_file():
        raise FileNotFoundError(PERF_SCRIPT)
    if not PLOT_SCRIPT.is_file():
        raise FileNotFoundError(PLOT_SCRIPT)
    if not SUMMARY_SCRIPT.is_file():
        raise FileNotFoundError(SUMMARY_SCRIPT)

    cases = build_cases(smoke=args.smoke, rank_mode=args.rank_mode)
    token_sizes = TOKEN_SIZES[:1] if args.smoke else TOKEN_SIZES
    expected_runs = len(cases) * len(token_sizes)

    if args.list:
        for case in cases:
            csv_path = case.csv_path(Path("."), args.date)
            print(f"{csv_path} tokens={','.join(map(str, token_sizes))}")
        print(f"CONFIGS={len(cases)} RUNS={expected_runs}")
        return 0

    if args.dry_run:
        for case in cases:
            for tokens_per_rank in token_sizes:
                print(_display_command(case, tokens_per_rank))
        print(f"RUNS={expected_runs}")
        return 0

    if not args.no_plot:
        _require_plot_dependency()

    output_dir = args.output_dir.resolve()
    run_dir = output_dir / args.date
    run_dir.mkdir(parents=True, exist_ok=True)
    git_commit = _run_command_text(["git", "rev-parse", "HEAD"])
    gpu_names = _gpu_names()

    print("=" * 79)
    print("Hopper FP8 fixed P02/P03 token sweep")
    print(f"  output_root      : {output_dir}")
    print(f"  run_dir          : {run_dir}")
    print(f"  run_date         : {args.date}")
    print(f"  rank_mode        : {args.rank_mode}")
    print(f"  token_sizes      : {token_sizes}")
    print(f"  configurations   : {len(cases)}")
    print(f"  planned runs     : {expected_runs}")
    print(f"  topk             : {TOPK}")
    print(f"  warmup / iters   : {WARMUP} / {ITERS}")
    print(f"  git_commit       : {git_commit}")
    print(f"  gpu_names        : {gpu_names}")
    print(f"  resume           : {'disabled (--force)' if args.force else 'enabled'}")
    print("=" * 79)

    passed = 0
    failed = 0
    skipped = 0
    stop = False
    for case in cases:
        csv_path = case.csv_path(output_dir, args.date)
        successful = _successful_tokens(csv_path)
        for tokens_per_rank in token_sizes:
            if not args.force and tokens_per_rank in successful:
                print(f"[SKIP passed] {csv_path.name} tokens_per_rank={tokens_per_rank}")
                skipped += 1
                continue
            status, _ = _run_case(
                case,
                tokens_per_rank,
                output_dir,
                args.date,
                git_commit,
                gpu_names,
            )
            if status == "pass":
                passed += 1
            else:
                failed += 1
                if args.fail_fast:
                    stop = True
                    break
        if stop:
            break

    summary_command = [
        sys.executable,
        str(SUMMARY_SCRIPT),
        "--input-dir",
        str(output_dir),
        "--date",
        args.date,
    ]
    print(f"[SUMMARY] {shlex.join(summary_command)}")
    summary_rc = subprocess.run(
        summary_command, cwd=SCRIPT_DIR.parent, check=False
    ).returncode

    plot_rc = 0
    if not args.no_plot:
        plot_command = [
            sys.executable,
            str(PLOT_SCRIPT),
            "--input-dir",
            str(output_dir),
            "--date",
            args.date,
            "--rank-mode",
            args.rank_mode,
        ]
        print(f"[PLOT] {shlex.join(plot_command)}")
        plot_rc = subprocess.run(plot_command, cwd=SCRIPT_DIR.parent, check=False).returncode

    print("=" * 79)
    print(
        f"SUMMARY: passed={passed} failed={failed} skipped={skipped} "
        f"planned={expected_runs} summary_rc={summary_rc} plot_rc={plot_rc}"
    )
    print("=" * 79)
    return 1 if failed or summary_rc or plot_rc else 0


if __name__ == "__main__":
    raise SystemExit(main())
