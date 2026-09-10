#!/usr/bin/env python3
"""Run the Hopper FP8 P03 tokens-per-rank performance sweep.

By default, each scale/token pair uses the launch configuration selected by
``heuristic_config.py``. Pass ``--no-heuristic`` to cover every valid
scale/order/tile/CGA/ping-pong combination. Tokens-per-rank range from 8
through 32768 in powers of two. Each CSV owns one resolved configuration and
contains one row per attempt so failed runs and forced reruns remain auditable.
P02 remains available as an explicit compatibility mode, but the default and
heuristic target are P03.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import importlib.util
import os
import re
import selectors
import signal
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    from .heuristic_config import select_heuristic_config
except ImportError:
    from heuristic_config import select_heuristic_config

SCRIPT_DIR = Path(__file__).resolve().parent
PERF_SCRIPT = SCRIPT_DIR / "run_perf_test.sh"
PLOT_SCRIPT = SCRIPT_DIR / "plot_token_sweep.py"
SUMMARY_SCRIPT = SCRIPT_DIR / "summarize_token_sweep.py"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "benchmark_data"
BENCHMARK_REQUIREMENTS = SCRIPT_DIR / "benchmark_requirements.txt"

TOKEN_SIZES = tuple(1 << power for power in range(3, 16))
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
SILENCE_TIMEOUT_SECONDS = 300
TILE_K = 128

SCALE_MODES = ("per_tensor", "blockwise")
OPERAND_ORDERS = ("non_swap_ab", "swap_ab")
CLUSTER_SHAPES = ((1, 1, 1), (1, 2, 1), (2, 1, 1), (2, 2, 1))
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
    "FP8_CLUSTER_SHAPE",
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
    "pingpong",
    "tile_m",
    "tile_n",
    "tile_k",
    "cluster_m",
    "cluster_n",
    "cluster_k",
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
    "max_rank",
    "min_mega_us",
    "max_mega_us",
    "mean_mega_us",
    "min_rank_tflops_per_rank",
    "max_rank_tflops_per_rank",
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
    "gpu_clocks_mhz",
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
    pingpong: bool
    tile_m: int
    tile_n: int
    cluster_shape_mnk: tuple[int, int, int]

    @property
    def scale_tag(self) -> str:
        return "pertensor" if self.scale_mode == "per_tensor" else "blockwise"

    @property
    def order_tag(self) -> str:
        return "swapab" if self.operand_order == "swap_ab" else "nonswapab"

    @property
    def schedule_tag(self) -> str:
        return "pingpong" if self.pingpong else "legacy"

    @property
    def cluster_tag(self) -> str:
        m, n, _ = self.cluster_shape_mnk
        return f"CGA{m}x{n}"

    def stem(self, run_date: str) -> str:
        return (
            f"{run_date}_{self.rank_mode}_{self.scale_tag}_{self.order_tag}_"
            f"{self.schedule_tag}_{self.cluster_tag}_"
            f"TileM{self.tile_m}_TileN{self.tile_n}"
        )

    def csv_path(self, output_dir: Path, run_date: str) -> Path:
        return output_dir / run_date / f"{self.stem(run_date)}.csv"


@dataclass(frozen=True)
class BenchmarkJob:
    case: BenchmarkCase
    tokens_per_rank: int
    use_heuristic: bool


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
    """Parse the rank-level mega+topk CUDA times printed by mega_runner.py.

    ``rank_times_us`` is the per-rank TOTAL (mega + standalone topk reduce)
    so the sweep TFLOPS account for the full compute pipeline, matching the
    FlashInfer benchmark's compute series; a missing ``topk:`` section (in-
    kernel reduce) contributes zero.
    """
    rank_times: dict[int, float] = {}
    rank_topk_times: dict[int, float] = {}
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
        elif rank_match and section == "topk":
            value = _parse_time_us(rank_match.group("time"))
            if value is not None:
                rank_topk_times[int(rank_match.group("rank"))] = value

    for rank, topk_us in rank_topk_times.items():
        if rank in rank_times:
            rank_times[rank] += topk_us

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
        if not any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, tuple) or not all(
            isinstance(item, int) for item in value
        ):
            raise ValueError(f"{path}:{name} must be a literal tuple of integers")
        return value
    raise ValueError(f"Unable to find {name} in {path}")


def supported_tiles(operand_order: str, pingpong: bool) -> tuple[tuple[int, int], ...]:
    non_swap_file = SCRIPT_DIR / "epilogue_fp8.py"
    swap_file = SCRIPT_DIR / "epilogue_fp8_swapab.py"
    non_swap_m = _read_tuple_constant(non_swap_file, "NonSwapTileMChoices")
    non_swap_n = _read_tuple_constant(non_swap_file, "NonSwapTileNChoices")
    swap_m = _read_tuple_constant(swap_file, "SwapABTileMChoices")
    swap_n = _read_tuple_constant(swap_file, "SwapABTokenTileNChoices")
    tiles = {
        "non_swap_ab": tuple((m, n) for m in non_swap_m for n in non_swap_n),
        "swap_ab": tuple((m, n) for m in swap_m for n in swap_n),
    }
    selected = tiles[operand_order]
    if not pingpong:
        return selected
    if operand_order == "non_swap_ab":
        return tuple((m, n) for m, n in selected if n == 128)
    return tuple((m, n) for m, n in selected if m == 128)


def build_cases(rank_mode: str = "multirank") -> tuple[BenchmarkCase, ...]:
    if rank_mode not in RANK_MODE_CHOICES:
        raise ValueError(f"rank_mode must be one of {','.join(RANK_MODE_CHOICES)}")
    selected_rank_modes = (
        RANK_MODES
        if rank_mode == "both"
        else tuple(mode for mode in RANK_MODES if mode[0] == rank_mode)
    )
    cases = tuple(
        BenchmarkCase(
            case_rank_mode,
            perf_case,
            world_size,
            scale_mode,
            order,
            pingpong,
            m,
            n,
            cluster_shape,
        )
        for case_rank_mode, perf_case, world_size in selected_rank_modes
        for scale_mode in SCALE_MODES
        for order in OPERAND_ORDERS
        for pingpong in (False, True)
        for m, n in supported_tiles(order, pingpong)
        for cluster_shape in CLUSTER_SHAPES
    )
    return cases


def build_heuristic_jobs(
    rank_mode: str,
    scale_mode: str,
    token_sizes: Sequence[int],
) -> tuple[BenchmarkJob, ...]:
    selected_rank_modes = (
        RANK_MODES
        if rank_mode == "both"
        else tuple(mode for mode in RANK_MODES if mode[0] == rank_mode)
    )
    normalized_scale_mode = scale_mode.replace("-", "_")
    scale_modes = tuple(
        mode
        for mode in SCALE_MODES
        if normalized_scale_mode == "both" or mode == normalized_scale_mode
    )
    jobs = []
    for case_rank_mode, perf_case, world_size in selected_rank_modes:
        for selected_scale_mode in scale_modes:
            for tokens_per_rank in token_sizes:
                config = select_heuristic_config(
                    selected_scale_mode, tokens_per_rank
                ).config
                tile_m, tile_n, tile_k = config.mma_tiler_mnk
                if tile_k != TILE_K:
                    raise ValueError(f"Heuristic tile K must be {TILE_K}, got {tile_k}")
                case = BenchmarkCase(
                    case_rank_mode,
                    perf_case,
                    world_size,
                    selected_scale_mode,
                    "swap_ab" if config.swap_ab else "non_swap_ab",
                    config.pingpong,
                    tile_m,
                    tile_n,
                    config.cluster_shape_mnk,
                )
                jobs.append(BenchmarkJob(case, tokens_per_rank, True))
    return tuple(jobs)


def _case_environment(
    case: BenchmarkCase,
    tokens_per_rank: int,
    use_heuristic: bool = False,
) -> dict[str, str]:
    env = os.environ.copy()
    python_bin = str(Path(sys.executable).resolve().parent)
    env["PATH"] = os.pathsep.join((python_bin, env.get("PATH", "")))
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
    if use_heuristic:
        for key in (
            "FP8_CLUSTER_SHAPE",
            "FP8_NON_SWAP_M",
            "FP8_NON_SWAP_N",
            "FP8_SWAP_AB_M",
            "FP8_SWAP_AB_N",
        ):
            env.pop(key, None)
    elif case.operand_order == "swap_ab":
        env.update(
            {
                "FP8_CLUSTER_SHAPE": ",".join(
                    str(value) for value in case.cluster_shape_mnk
                ),
                "FP8_SWAP_AB_M": str(case.tile_m),
                "FP8_SWAP_AB_N": str(case.tile_n),
            }
        )
        env.pop("FP8_NON_SWAP_M", None)
        env.pop("FP8_NON_SWAP_N", None)
    else:
        env.update(
            {
                "FP8_CLUSTER_SHAPE": ",".join(
                    str(value) for value in case.cluster_shape_mnk
                ),
                "FP8_NON_SWAP_M": str(case.tile_m),
                "FP8_NON_SWAP_N": str(case.tile_n),
            }
        )
        env.pop("FP8_SWAP_AB_M", None)
        env.pop("FP8_SWAP_AB_N", None)
    return env


def _case_cli_args(
    case: BenchmarkCase,
    use_heuristic: bool = False,
) -> list[str]:
    scale_mode = "per-tensor" if case.scale_mode == "per_tensor" else "blockwise"
    args = ["--scale-mode", scale_mode]
    if use_heuristic:
        args.append("--heuristic")
    else:
        args.append("--no-heuristic")
        if case.operand_order == "swap_ab":
            args.append("--swapab")
        if case.pingpong:
            args.append("--pingpong")
    return args


def _display_command(
    case: BenchmarkCase,
    tokens_per_rank: int,
    use_heuristic: bool = False,
) -> str:
    env = _case_environment(case, tokens_per_rank, use_heuristic)
    assignments = [f"{key}={env[key]}" for key in DISPLAY_ENV_KEYS if key in env]
    command = [
        "bash",
        str(PERF_SCRIPT),
        *_case_cli_args(case, use_heuristic),
        case.perf_case,
    ]
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
    if path.exists() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_fields = tuple(reader.fieldnames or ())
            existing_rows = list(reader)
        if existing_fields != CSV_FIELDS:
            unknown_fields = set(existing_fields) - set(CSV_FIELDS)
            if unknown_fields:
                raise ValueError(
                    f"Cannot migrate {path}; unknown fields: {sorted(unknown_fields)}"
                )
            temporary_path = path.with_suffix(".csv.tmp")
            with temporary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(
                    {field: existing.get(field, "") for field in CSV_FIELDS}
                    for existing in existing_rows
                )
            os.replace(temporary_path, path)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _run_and_tee(
    command: Sequence[str],
    env: dict[str, str],
    log_path: Path,
    silence_timeout_s: int,
) -> tuple[int, list[str], bool]:
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
            start_new_session=True,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        last_output = time.monotonic()
        timed_out = False
        try:
            while process.poll() is None:
                events = selector.select(timeout=1.0)
                if events:
                    line = process.stdout.readline()
                    if line:
                        print(line, end="", flush=True)
                        log_handle.write(line)
                        log_handle.flush()
                        lines.append(line)
                        last_output = time.monotonic()
                    continue
                silent_for = time.monotonic() - last_output
                if silent_for < silence_timeout_s:
                    continue
                message = (
                    f"[TIMEOUT] no output for {silent_for:.1f}s; "
                    "terminating the process group\n"
                )
                print(message, end="", flush=True)
                log_handle.write(message)
                log_handle.flush()
                lines.append(message)
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                break

            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
                lines.append(line)
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            raise
        finally:
            selector.close()
        return process.wait(), lines, timed_out


def _run_case(
    case: BenchmarkCase,
    tokens_per_rank: int,
    use_heuristic: bool,
    output_dir: Path,
    run_date: str,
    git_commit: str,
    gpu_names: str,
    gpu_clocks_mhz: str,
    silence_timeout_s: int,
) -> tuple[str, Path]:
    csv_path = case.csv_path(output_dir, run_date)
    attempt = _next_attempt(csv_path, tokens_per_rank)
    log_name = f"{case.stem(run_date)}_Tokens{tokens_per_rank}_Attempt{attempt}.log"
    log_path = csv_path.parent / log_name
    env = _case_environment(case, tokens_per_rank, use_heuristic)
    command = [
        "bash",
        str(PERF_SCRIPT),
        *_case_cli_args(case, use_heuristic),
        case.perf_case,
    ]
    display_command = _display_command(case, tokens_per_rank, use_heuristic)

    print("=" * 79)
    print(
        f"[SWEEP] {'heuristic' if use_heuristic else 'exhaustive'} "
        f"{case.rank_mode} {case.scale_tag} {case.order_tag} "
        f"{case.schedule_tag} {case.cluster_tag} "
        f"M{case.tile_m}N{case.tile_n} tokens_per_rank={tokens_per_rank} "
        f"attempt={attempt}"
    )
    print(f"[CMD] {display_command}")
    print(f"[LOG] {log_path}")
    start = time.monotonic()
    return_code, output_lines, timed_out = _run_and_tee(
        command, env, log_path, silence_timeout_s
    )
    wall_time_s = time.monotonic() - start
    timing = parse_profiler_output(output_lines)

    expected_ranks = set(range(case.world_size))
    parsed_ranks = set(timing.rank_times_us)
    if timed_out:
        status = "timeout"
    elif return_code != 0:
        status = "failed"
    elif parsed_ranks != expected_ranks:
        status = "parse_error"
    else:
        status = "pass"

    rank_values = [timing.rank_times_us[rank] for rank in sorted(parsed_ranks)]
    if rank_values:
        min_rank = min(timing.rank_times_us, key=timing.rank_times_us.__getitem__)
        max_rank = max(timing.rank_times_us, key=timing.rank_times_us.__getitem__)
        min_mega_us = min(rank_values)
        max_mega_us = max(rank_values)
        mean_mega_us = statistics.fmean(rank_values)
    else:
        min_rank = ""
        max_rank = ""
        min_mega_us = ""
        max_mega_us = ""
        mean_mega_us = ""

    fc1_flops, fc2_flops, total_flops = compute_gemm_flops_per_rank(tokens_per_rank)
    critical_tflops = (
        effective_tflops(total_flops, max_mega_us)
        if status == "pass" and isinstance(max_mega_us, float)
        else ""
    )
    min_rank_tflops = (
        effective_tflops(total_flops, min_mega_us)
        if status == "pass" and isinstance(min_mega_us, float)
        else ""
    )
    max_rank_tflops = critical_tflops
    cluster_m, cluster_n, cluster_k = case.cluster_shape_mnk

    row: dict[str, object] = {
        "run_date": run_date,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "attempt": attempt,
        "case": case.perf_case,
        "rank_mode": case.rank_mode,
        "scale_mode": case.scale_mode,
        "operand_order": case.operand_order,
        "pingpong": int(case.pingpong),
        "tile_m": case.tile_m,
        "tile_n": case.tile_n,
        "tile_k": TILE_K,
        "cluster_m": cluster_m,
        "cluster_n": cluster_n,
        "cluster_k": cluster_k,
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
        "max_rank": max_rank,
        "min_mega_us": min_mega_us,
        "max_mega_us": max_mega_us,
        "mean_mega_us": mean_mega_us,
        "min_rank_tflops_per_rank": min_rank_tflops,
        "max_rank_tflops_per_rank": max_rank_tflops,
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
        "gpu_clocks_mhz": gpu_clocks_mhz,
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
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _gpu_names() -> str:
    output = _run_command_text(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    )
    if output == "unknown":
        return output
    return ";".join(line.strip() for line in output.splitlines() if line.strip())


def _gpu_clocks_mhz() -> str:
    output = _run_command_text(
        [
            "nvidia-smi",
            "--query-gpu=clocks.mem,clocks.sm",
            "--format=csv,noheader,nounits",
        ]
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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parse_tokens(value: str) -> tuple[int, ...]:
    try:
        tokens = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "tokens must be comma-separated integers"
        ) from error
    if not tokens or any(token <= 0 for token in tokens):
        raise argparse.ArgumentTypeError("tokens must all be positive")
    return tokens


def _parse_cluster_shape(value: str) -> tuple[int, int, int]:
    try:
        shape = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "cluster shape must be M,N or M,N,K"
        ) from error
    if len(shape) == 2:
        shape = (*shape, 1)
    if shape not in CLUSTER_SHAPES:
        choices = ", ".join("x".join(map(str, item[:2])) for item in CLUSTER_SHAPES)
        raise argparse.ArgumentTypeError(f"cluster shape must be one of {choices}")
    return shape


def _select_cases(
    cases: Sequence[BenchmarkCase],
    scale_mode: str,
    operand_order: str,
    schedule: str,
    cluster_shapes: Sequence[tuple[int, int, int]] | None,
) -> tuple[BenchmarkCase, ...]:
    scale_filter = scale_mode.replace("-", "_")
    order_filter = operand_order.replace("-", "_")
    selected = tuple(
        case
        for case in cases
        if (scale_filter == "both" or case.scale_mode == scale_filter)
        and (order_filter == "both" or case.operand_order == order_filter)
        and (schedule == "both" or case.pingpong == (schedule == "pingpong"))
        and (cluster_shapes is None or case.cluster_shape_mnk in cluster_shapes)
    )
    return selected


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
        default="multirank",
        help="Run singlerank (P02), multirank (P03), or both (default: multirank).",
    )
    parser.add_argument(
        "--tokens",
        type=_parse_tokens,
        default=TOKEN_SIZES,
        help="Comma-separated tokens per rank (default: 8,16,...,32768).",
    )
    heuristic_group = parser.add_mutually_exclusive_group()
    heuristic_group.add_argument(
        "--heuristic",
        dest="use_heuristic",
        action="store_true",
        help="Use token/scale launch heuristics (default).",
    )
    heuristic_group.add_argument(
        "--no-heuristic",
        dest="use_heuristic",
        action="store_false",
        help="Disable heuristics and run the complete launch-configuration sweep.",
    )
    parser.set_defaults(use_heuristic=True)
    parser.add_argument(
        "--scale-mode",
        choices=("both", "per-tensor", "blockwise"),
        default="both",
        help="Select one scale mode or both (default: both).",
    )
    parser.add_argument(
        "--operand-order",
        choices=("both", "non-swap-ab", "swap-ab"),
        default="both",
        help="Select non-swapAB, swapAB, or both (default: both).",
    )
    parser.add_argument(
        "--schedule",
        choices=("both", "legacy", "pingpong"),
        default="both",
        help="Select legacy, ping-pong, or both schedules (default: both).",
    )
    parser.add_argument(
        "--cluster-shape",
        type=_parse_cluster_shape,
        action="append",
        default=None,
        help="Select a CGA as M,N; repeat to select multiple shapes.",
    )
    parser.add_argument(
        "--shard-count",
        type=_positive_int,
        default=1,
        help="Partition configurations into this many deterministic shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index (default: 0).",
    )
    parser.add_argument(
        "--silence-timeout",
        type=_positive_int,
        default=SILENCE_TIMEOUT_SECONDS,
        help="Kill a run after this many seconds without output (default: 300).",
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
        "--no-finalize",
        action="store_true",
        help="Do not generate summary or plots after running the selected shard.",
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
        help="Run only the first selected configuration and token.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not PERF_SCRIPT.is_file():
        raise FileNotFoundError(PERF_SCRIPT)
    if not PLOT_SCRIPT.is_file():
        raise FileNotFoundError(PLOT_SCRIPT)
    if not SUMMARY_SCRIPT.is_file():
        raise FileNotFoundError(SUMMARY_SCRIPT)

    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must be in [0, --shard-count)")
    token_sizes = tuple(args.tokens)
    if args.use_heuristic:
        if (
            args.operand_order != "both"
            or args.schedule != "both"
            or args.cluster_shape is not None
        ):
            parser.error(
                "--operand-order, --schedule, and --cluster-shape require "
                "--no-heuristic"
            )
        all_jobs = build_heuristic_jobs(
            args.rank_mode,
            args.scale_mode,
            token_sizes,
        )
        all_cases = tuple(dict.fromkeys(job.case for job in all_jobs))
        cases = tuple(
            case
            for index, case in enumerate(all_cases)
            if index % args.shard_count == args.shard_index
        )
        selected_cases = set(cases)
        jobs = tuple(job for job in all_jobs if job.case in selected_cases)
    else:
        cases = _select_cases(
            build_cases(rank_mode=args.rank_mode),
            args.scale_mode,
            args.operand_order,
            args.schedule,
            args.cluster_shape,
        )
        cases = tuple(
            case
            for index, case in enumerate(cases)
            if index % args.shard_count == args.shard_index
        )
        jobs = tuple(
            BenchmarkJob(case, tokens_per_rank, False)
            for case in cases
            for tokens_per_rank in token_sizes
        )
    if args.smoke:
        jobs = jobs[:1]
        cases = tuple(dict.fromkeys(job.case for job in jobs))
    if not jobs:
        raise ValueError("No benchmark jobs match the selected filters")
    expected_runs = len(jobs)
    selected_token_sizes = tuple(dict.fromkeys(job.tokens_per_rank for job in jobs))

    if args.list:
        if args.use_heuristic:
            for job in jobs:
                csv_path = job.case.csv_path(Path("."), args.date)
                print(f"{csv_path} tokens={job.tokens_per_rank} mode=heuristic")
        else:
            for case in cases:
                csv_path = case.csv_path(Path("."), args.date)
                print(f"{csv_path} tokens={','.join(map(str, token_sizes))}")
        print(f"CONFIGS={len(cases)} RUNS={expected_runs}")
        return 0

    if args.dry_run:
        for job in jobs:
            print(
                _display_command(
                    job.case,
                    job.tokens_per_rank,
                    job.use_heuristic,
                )
            )
        print(f"RUNS={expected_runs}")
        return 0

    finalize = not args.no_finalize and args.shard_count == 1
    if finalize and not args.no_plot:
        _require_plot_dependency()

    output_dir = args.output_dir.resolve()
    run_dir = output_dir / args.date
    run_dir.mkdir(parents=True, exist_ok=True)
    git_commit = _run_command_text(["git", "rev-parse", "HEAD"])
    gpu_names = _gpu_names()
    gpu_clocks_mhz = _gpu_clocks_mhz()

    print("=" * 79)
    print("Hopper FP8 P03 tile/CGA/ping-pong token sweep")
    print(f"  output_root      : {output_dir}")
    print(f"  run_dir          : {run_dir}")
    print(f"  run_date         : {args.date}")
    print(f"  rank_mode        : {args.rank_mode}")
    print(f"  config_mode      : {'heuristic' if args.use_heuristic else 'exhaustive'}")
    print(f"  token_sizes      : {selected_token_sizes}")
    print(f"  configurations   : {len(cases)}")
    print(f"  shard             : {args.shard_index}/{args.shard_count}")
    print(f"  planned runs     : {expected_runs}")
    print(f"  topk             : {TOPK}")
    print(f"  warmup / iters   : {WARMUP} / {ITERS}")
    print(f"  git_commit       : {git_commit}")
    print(f"  gpu_names        : {gpu_names}")
    print(f"  gpu_clocks_mhz   : {gpu_clocks_mhz}")
    print(f"  resume           : {'disabled (--force)' if args.force else 'enabled'}")
    print("=" * 79)

    passed = 0
    failed = 0
    skipped = 0
    successful_by_csv: dict[Path, set[int]] = {}
    for job in jobs:
        case = job.case
        tokens_per_rank = job.tokens_per_rank
        csv_path = case.csv_path(output_dir, args.date)
        if csv_path not in successful_by_csv:
            successful_by_csv[csv_path] = _successful_tokens(csv_path)
        successful = successful_by_csv[csv_path]
        if not args.force and tokens_per_rank in successful:
            print(f"[SKIP passed] {csv_path.name} tokens_per_rank={tokens_per_rank}")
            skipped += 1
            continue
        status, _ = _run_case(
            case,
            tokens_per_rank,
            job.use_heuristic,
            output_dir,
            args.date,
            git_commit,
            gpu_names,
            gpu_clocks_mhz,
            args.silence_timeout,
        )
        if status == "pass":
            passed += 1
            successful.add(tokens_per_rank)
        else:
            failed += 1
            if args.fail_fast:
                break

    summary_rc = 0
    plot_rc = 0
    if not finalize:
        print("[FINALIZE] skipped for this shard")
    summary_command = [
        sys.executable,
        str(SUMMARY_SCRIPT),
        "--input-dir",
        str(output_dir),
        "--date",
        args.date,
    ]
    if finalize:
        print(f"[SUMMARY] {shlex.join(summary_command)}")
        summary_rc = subprocess.run(
            summary_command, cwd=SCRIPT_DIR.parent, check=False
        ).returncode

    if finalize and not args.no_plot:
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
        plot_rc = subprocess.run(
            plot_command, cwd=SCRIPT_DIR.parent, check=False
        ).returncode

    print("=" * 79)
    print(
        f"SUMMARY: passed={passed} failed={failed} skipped={skipped} "
        f"planned={expected_runs} summary_rc={summary_rc} plot_rc={plot_rc}"
    )
    print("=" * 79)
    return 1 if failed or summary_rc or plot_rc else 0


if __name__ == "__main__":
    raise SystemExit(main())
