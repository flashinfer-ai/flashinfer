#!/usr/bin/env python3
"""Summarize peak Hopper FP8 token-sweep throughput for each mode."""

from __future__ import annotations

import argparse
import csv
import math
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "benchmark_data"
DATE_DIR_RE = re.compile(r"^\d{8}$")
RAW_CSV_NAME_RE = re.compile(
    r"^(?P<date>\d{8})_"
    r"(?P<rank>singlerank|multirank)_"
    r"(?P<scale>pertensor|blockwise)_"
    r"(?P<order>swapab|nonswapab)_"
    r"TileM(?P<m>\d+)_TileN(?P<n>\d+)\.csv$"
)

SUMMARY_FIELDS = (
    "run_date",
    "rank_mode",
    "case",
    "scale_mode",
    "operand_order",
    "peak_tflops_per_rank",
    "accum_mode",
    "tile_m",
    "tile_n",
    "tile_k",
    "tokens_per_rank",
    "routed_tokens_per_rank",
    "world_size",
    "topk",
    "total_experts",
    "local_experts",
    "hidden",
    "intermediate_downproj",
    "intermediate_gateup",
    "critical_latency_us",
    "min_rank",
    "min_mega_us",
    "max_mega_us",
    "mean_mega_us",
    "rank_0_mega_us",
    "rank_1_mega_us",
    "rank_2_mega_us",
    "rank_3_mega_us",
    "warmup",
    "iters",
    "attempt",
    "timestamp_utc",
    "git_commit",
    "gpu_names",
    "source_csv",
    "log_file",
    "command",
)

_RANK_ORDER = {"singlerank": 0, "multirank": 1}
_SCALE_ORDER = {"per_tensor": 0, "blockwise": 1}
_OPERAND_ORDER = {"non_swap_ab": 0, "swap_ab": 1}


@dataclass(frozen=True)
class SummaryGroup:
    run_date: str
    rank_mode: str
    scale_mode: str
    operand_order: str

    def sort_key(self) -> tuple[int, int, int]:
        return (
            _RANK_ORDER[self.rank_mode],
            _SCALE_ORDER[self.scale_mode],
            _OPERAND_ORDER[self.operand_order],
        )


@dataclass(frozen=True)
class PeakRecord:
    group: SummaryGroup
    tflops: float
    source_csv: Path
    row: dict[str, str]

    def selection_key(self) -> tuple[float, str, int, int, str]:
        return (
            self.tflops,
            self.row.get("timestamp_utc", ""),
            _int_or_zero(self.row.get("attempt", "")),
            _int_or_zero(self.row.get("tokens_per_rank", "")),
            self.source_csv.name,
        )


@dataclass(frozen=True)
class SummaryResult:
    output_path: Path
    source_files: int
    successful_rows: int
    invalid_rows: int
    groups: int
    missing_groups: tuple[SummaryGroup, ...]


def _int_or_zero(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _accum_mode(row: dict[str, str]) -> str:
    recorded = row.get("accum_mode", "").strip()
    if recorded:
        return recorded
    try:
        command_parts = shlex.split(row.get("command", ""))
    except ValueError:
        return "unknown"
    prefix = "FP8_ACCUM_MODE="
    for part in command_parts:
        if part.startswith(prefix):
            value = part[len(prefix) :]
            return value or "unknown"
    return "unknown"


def _group_from_match(match: re.Match[str]) -> SummaryGroup:
    return SummaryGroup(
        run_date=match.group("date"),
        rank_mode=match.group("rank"),
        scale_mode=(
            "per_tensor" if match.group("scale") == "pertensor" else "blockwise"
        ),
        operand_order=(
            "swap_ab" if match.group("order") == "swapab" else "non_swap_ab"
        ),
    )


def _raw_csv_files(date_dir: Path) -> Iterable[tuple[Path, SummaryGroup]]:
    for path in sorted(date_dir.glob("*.csv")):
        match = RAW_CSV_NAME_RE.fullmatch(path.name)
        if match is None or match.group("date") != date_dir.name:
            continue
        yield path, _group_from_match(match)


def _read_peaks(
    date_dir: Path,
) -> tuple[dict[SummaryGroup, PeakRecord], int, int, int]:
    peaks: dict[SummaryGroup, PeakRecord] = {}
    source_files = 0
    successful_rows = 0
    invalid_rows = 0
    for path, group in _raw_csv_files(date_dir):
        source_files += 1
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "pass":
                    continue
                try:
                    tflops = float(row.get("critical_tflops_per_rank", ""))
                except (TypeError, ValueError):
                    invalid_rows += 1
                    continue
                if not math.isfinite(tflops) or tflops <= 0.0:
                    invalid_rows += 1
                    continue
                successful_rows += 1
                record = PeakRecord(group, tflops, path, row)
                current = peaks.get(group)
                if current is None or record.selection_key() > current.selection_key():
                    peaks[group] = record
    return peaks, source_files, successful_rows, invalid_rows


def _expected_groups(run_date: str) -> set[SummaryGroup]:
    return {
        SummaryGroup(run_date, rank_mode, scale_mode, operand_order)
        for rank_mode in _RANK_ORDER
        for scale_mode in _SCALE_ORDER
        for operand_order in _OPERAND_ORDER
    }


def _summary_row(record: PeakRecord) -> dict[str, str]:
    source = record.row
    row = {field: source.get(field, "") for field in SUMMARY_FIELDS}
    row.update(
        {
            "run_date": record.group.run_date,
            "rank_mode": record.group.rank_mode,
            "scale_mode": record.group.scale_mode,
            "operand_order": record.group.operand_order,
            "peak_tflops_per_rank": f"{record.tflops:.6f}",
            "accum_mode": _accum_mode(source),
            "critical_latency_us": source.get("max_mega_us", ""),
            "source_csv": record.source_csv.name,
        }
    )
    return row


def write_peak_summary(date_dir: Path, output_dir: Path) -> SummaryResult:
    peaks, source_files, successful_rows, invalid_rows = _read_peaks(date_dir)
    if not peaks:
        raise ValueError(f"No valid successful token-sweep rows in {date_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{date_dir.name}_token_sweep_peak_summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for group in sorted(peaks, key=SummaryGroup.sort_key):
            writer.writerow(_summary_row(peaks[group]))

    missing_groups = tuple(
        sorted(
            _expected_groups(date_dir.name) - peaks.keys(),
            key=SummaryGroup.sort_key,
        )
    )
    return SummaryResult(
        output_path=output_path,
        source_files=source_files,
        successful_rows=successful_rows,
        invalid_rows=invalid_rows,
        groups=len(peaks),
        missing_groups=missing_groups,
    )


def _date_dirs(input_dir: Path, run_date: str | None) -> list[Path]:
    if DATE_DIR_RE.fullmatch(input_dir.name):
        if run_date is not None and run_date != input_dir.name:
            raise FileNotFoundError(
                f"Input directory {input_dir} does not match date {run_date}"
            )
        candidates = [input_dir]
    elif run_date is not None:
        candidates = [input_dir / run_date]
    else:
        candidates = sorted(
            path
            for path in input_dir.iterdir()
            if path.is_dir() and DATE_DIR_RE.fullmatch(path.name)
        )
    date_dirs = [path for path in candidates if path.is_dir()]
    if not date_dirs:
        selected = run_date or "any date"
        raise FileNotFoundError(f"No token-sweep data for {selected} in {input_dir}")
    return date_dirs


def _date_output_dir(output_dir: Path, run_date: str) -> Path:
    return output_dir if output_dir.name == run_date else output_dir / run_date


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Benchmark-data root or one YYYYMMDD directory "
            f"(default: {DEFAULT_INPUT_DIR})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Benchmark-data root for summary output; defaults to --input-dir.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="YYYYMMDD to summarize; defaults to all date directories found.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.date is not None and not DATE_DIR_RE.fullmatch(args.date):
        raise ValueError("--date must use YYYYMMDD")
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()

    results: list[SummaryResult] = []
    for date_dir in _date_dirs(input_dir, args.date):
        result = write_peak_summary(
            date_dir, _date_output_dir(output_dir, date_dir.name)
        )
        results.append(result)
        print(
            f"[WROTE] {result.output_path} groups={result.groups} "
            f"source_files={result.source_files} "
            f"successful_rows={result.successful_rows} "
            f"invalid_rows={result.invalid_rows}"
        )
        if result.missing_groups:
            missing = ", ".join(
                f"{group.rank_mode}/{group.scale_mode}/{group.operand_order}"
                for group in result.missing_groups
            )
            print(f"[WARN] missing groups: {missing}")
    print(f"SUMMARIES={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
