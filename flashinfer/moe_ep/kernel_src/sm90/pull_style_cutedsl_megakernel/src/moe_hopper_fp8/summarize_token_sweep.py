#!/usr/bin/env python3
"""Consolidate Hopper FP8 token sweeps and derive per-token heuristics."""

from __future__ import annotations

import argparse
import csv
import math
import re
import shlex
import statistics
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
    r"(?P<schedule>legacy|pingpong)_"
    r"CGA(?P<cm>\d+)x(?P<cn>\d+)_"
    r"TileM(?P<m>\d+)_TileN(?P<n>\d+)(?P<genc>_genc)?\.csv$"  # _genc: generate_c sweep
)

HEURISTIC_FIELDS = (
    "run_date",
    "rank_mode",
    "case",
    "scale_mode",
    "generate_c",
    "tokens_per_rank",
    "routed_tokens_per_rank",
    "operand_order",
    "pingpong",
    "accum_mode",
    "cluster_m",
    "cluster_n",
    "cluster_k",
    "tile_m",
    "tile_n",
    "tile_k",
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
    "world_size",
    "topk",
    "total_experts",
    "local_experts",
    "hidden",
    "intermediate_downproj",
    "intermediate_gateup",
    "warmup",
    "iters",
    "attempt",
    "timestamp_utc",
    "git_commit",
    "gpu_names",
    "gpu_clocks_mhz",
    "source_csv",
    "log_file",
    "command",
)

PEAK_SUMMARY_FIELDS = (
    "rank_mode",
    "case",
    "scale_mode",
    "operand_order",
    "peak_tflops_per_rank(min_rank)",
    "peak_tflops_per_rank(max_rank)",
    "accum_mode",
    "cga",
    "ping-pong",
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
)

TOKEN_TFLOPS_BASE_FIELDS = (
    "rank_mode",
    "case",
    "scale_mode",
    "rank_metric",
)

RANK_TFLOPS_METRICS = (
    ("min_rank", "min_rank_tflops_per_rank"),
    ("max_rank", "max_rank_tflops_per_rank"),
)


@dataclass(frozen=True)
class SourceRow:
    source_csv: Path
    row: dict[str, str]

    @property
    def token(self) -> int:
        return int(self.row["tokens_per_rank"])

    @property
    def critical_tflops(self) -> float:
        raw = self.row.get("max_rank_tflops_per_rank", "") or self.row.get(
            "critical_tflops_per_rank", ""
        )
        return float(raw)

    def recency_key(self) -> tuple[str, int]:
        return (
            self.row.get("timestamp_utc", ""),
            int(self.row.get("attempt", "0") or 0),
        )

    def selection_key(self) -> tuple[float, str, int, str]:
        return (
            self.critical_tflops,
            self.row.get("timestamp_utc", ""),
            int(self.row.get("attempt", "0") or 0),
            self.source_csv.name,
        )


def _raw_csv_files(date_dir: Path) -> Iterable[Path]:
    for path in sorted(date_dir.glob("*.csv")):
        match = RAW_CSV_NAME_RE.fullmatch(path.name)
        if match is not None and match.group("date") == date_dir.name:
            yield path


def _read_all_rows(date_dir: Path) -> list[SourceRow]:
    rows: list[SourceRow] = []
    for path in _raw_csv_files(date_dir):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                # Older CSVs record this flag only in their filename.
                row["generate_c"] = row.get("generate_c") or str(int("_genc.csv" in path.name))
                rows.append(SourceRow(path, row))
    if len({item.row["generate_c"] for item in rows}) > 1:
        raise ValueError("Keep inference and generate_c sweeps in separate directories")
    return rows


def _latest_by_config_token(rows: Sequence[SourceRow]) -> list[SourceRow]:
    latest: dict[tuple[str, int], SourceRow] = {}
    for source_row in rows:
        key = (source_row.source_csv.name, source_row.token)
        current = latest.get(key)
        if current is None or source_row.recency_key() > current.recency_key():
            latest[key] = source_row
    return list(latest.values())


def _valid_success(source_row: SourceRow) -> bool:
    if source_row.row.get("status") != "pass":
        return False
    try:
        value = source_row.critical_tflops
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0.0


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
            return part[len(prefix) :] or "unknown"
    return "unknown"


def _export_row(source_row: SourceRow) -> dict[str, str]:
    row = {field: source_row.row.get(field, "") for field in HEURISTIC_FIELDS}
    row["source_csv"] = source_row.source_csv.name
    row["accum_mode"] = _accum_mode(source_row.row)
    return row


def _export_peak_summary_row(source_row: SourceRow) -> dict[str, str]:
    row = source_row.row
    return {
        "rank_mode": row.get("rank_mode", ""),
        "case": row.get("case", ""),
        "scale_mode": row.get("scale_mode", ""),
        "operand_order": row.get("operand_order", ""),
        "peak_tflops_per_rank(min_rank)": row.get("min_rank_tflops_per_rank", ""),
        "peak_tflops_per_rank(max_rank)": row.get("max_rank_tflops_per_rank", ""),
        "accum_mode": _accum_mode(row),
        "cga": "x".join(
            (
                row.get("cluster_m", ""),
                row.get("cluster_n", ""),
                row.get("cluster_k", ""),
            )
        ),
        "ping-pong": row.get("pingpong", ""),
        "tile_m": row.get("tile_m", ""),
        "tile_n": row.get("tile_n", ""),
        "tile_k": row.get("tile_k", ""),
        "tokens_per_rank": row.get("tokens_per_rank", ""),
        "routed_tokens_per_rank": row.get("routed_tokens_per_rank", ""),
        "world_size": row.get("world_size", ""),
        "topk": row.get("topk", ""),
        "total_experts": row.get("total_experts", ""),
        "local_experts": row.get("local_experts", ""),
        "hidden": row.get("hidden", ""),
        "intermediate_downproj": row.get("intermediate_downproj", ""),
        "intermediate_gateup": row.get("intermediate_gateup", ""),
    }


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[WROTE] {path}")


def _write_all_results(date_dir: Path, rows: Sequence[SourceRow]) -> Path:
    fieldnames: list[str] = ["source_csv"]
    for source_row in rows:
        for field in source_row.row:
            if field not in fieldnames:
                fieldnames.append(field)
    output = date_dir / f"{date_dir.name}_token_sweep_all_results.csv"
    exported = (
        {"source_csv": source_row.source_csv.name, **source_row.row}
        for source_row in rows
    )
    _write_csv(output, fieldnames, exported)
    return output


def _write_failures(date_dir: Path, latest: Sequence[SourceRow]) -> Path:
    output = date_dir / f"{date_dir.name}_token_sweep_failures.csv"
    failures = sorted(
        (source_row for source_row in latest if not _valid_success(source_row)),
        key=lambda item: (item.source_csv.name, item.token),
    )
    fieldnames = ("source_csv", "status", "return_code", "tokens_per_rank", "log_file")
    _write_csv(
        output,
        fieldnames,
        (
            {
                "source_csv": item.source_csv.name,
                "status": item.row.get("status", ""),
                "return_code": item.row.get("return_code", ""),
                "tokens_per_rank": item.row.get("tokens_per_rank", ""),
                "log_file": item.row.get("log_file", ""),
            }
            for item in failures
        ),
    )
    return output


def _best_per_scale_token(latest: Sequence[SourceRow]) -> list[SourceRow]:
    best: dict[tuple[str, str, str, int], SourceRow] = {}
    for source_row in latest:
        if not _valid_success(source_row):
            continue
        key = (
            source_row.row["rank_mode"],
            source_row.row["case"],
            source_row.row["scale_mode"],
            source_row.token,
        )
        current = best.get(key)
        if current is None or source_row.selection_key() > current.selection_key():
            best[key] = source_row
    return sorted(
        best.values(),
        key=lambda item: (
            item.row["rank_mode"],
            item.row["case"],
            item.row["scale_mode"],
            item.token,
        ),
    )


def _write_heuristic(date_dir: Path, best: Sequence[SourceRow]) -> Path:
    output = date_dir / f"{date_dir.name}_token_sweep_heuristic.csv"
    _write_csv(output, HEURISTIC_FIELDS, (_export_row(item) for item in best))
    return output


def _write_peak_summary(date_dir: Path, latest: Sequence[SourceRow]) -> Path:
    best: dict[tuple[str, str, str], SourceRow] = {}
    for source_row in latest:
        if not _valid_success(source_row):
            continue
        key = (
            source_row.row["rank_mode"],
            source_row.row["scale_mode"],
            source_row.row["operand_order"],
        )
        current = best.get(key)
        if current is None or source_row.selection_key() > current.selection_key():
            best[key] = source_row
    output = date_dir / f"{date_dir.name}_token_sweep_peak_summary.csv"
    records = sorted(best.values(), key=lambda item: item.source_csv.name)
    _write_csv(
        output,
        PEAK_SUMMARY_FIELDS,
        (_export_peak_summary_row(item) for item in records),
    )
    return output


def _write_optimal_tflops_by_token(date_dir: Path, best: Sequence[SourceRow]) -> Path:
    lookup: dict[tuple[str, str, str, int], SourceRow] = {}
    for source_row in best:
        key = (
            source_row.row["rank_mode"],
            source_row.row["case"],
            source_row.row["scale_mode"],
            source_row.token,
        )
        lookup[key] = source_row

    tokens = sorted({source_row.token for source_row in best})
    contexts = sorted(
        {(source_row.row["rank_mode"], source_row.row["case"]) for source_row in best}
    )
    rows: list[dict[str, str]] = []
    for rank_mode, case in contexts:
        scale_modes = sorted(
            {
                source_row.row["scale_mode"]
                for source_row in best
                if source_row.row["rank_mode"] == rank_mode
                and source_row.row["case"] == case
            },
            key=lambda scale_mode: (scale_mode != "per_tensor", scale_mode),
        )
        for scale_mode in scale_modes:
            for rank_metric, source_field in RANK_TFLOPS_METRICS:
                row = {
                    "rank_mode": rank_mode,
                    "case": case,
                    "scale_mode": scale_mode,
                    "rank_metric": rank_metric,
                }
                for token in tokens:
                    source_row = lookup.get((rank_mode, case, scale_mode, token))
                    row[str(token)] = (
                        source_row.row.get(source_field, "")
                        if source_row is not None
                        else ""
                    )
                rows.append(row)

    output = date_dir / f"{date_dir.name}_token_sweep_optimal_tflops_by_token.csv"
    fieldnames = (*TOKEN_TFLOPS_BASE_FIELDS, *(str(token) for token in tokens))
    _write_csv(output, fieldnames, rows)
    return output


def summarize(date_dir: Path) -> tuple[int, int, int]:
    rows = _read_all_rows(date_dir)
    if not rows:
        raise ValueError(f"No raw token-sweep CSV rows in {date_dir}")
    latest = _latest_by_config_token(rows)
    best = _best_per_scale_token(latest)
    _write_all_results(date_dir, rows)
    _write_failures(date_dir, latest)
    _write_heuristic(date_dir, best)
    _write_peak_summary(date_dir, latest)
    _write_optimal_tflops_by_token(date_dir, best)
    failures = sum(not _valid_success(item) for item in latest)
    print(
        f"[SUMMARY] attempts={len(rows)} latest={len(latest)} "
        f"heuristic_rows={len(best)} failures={failures}"
    )
    return len(rows), len(best), failures


def _date_dirs(input_dir: Path, run_date: str | None) -> list[Path]:
    if DATE_DIR_RE.fullmatch(input_dir.name):
        candidates = [input_dir]
    elif run_date is not None:
        candidates = [input_dir / run_date]
    else:
        candidates = sorted(
            path
            for path in input_dir.iterdir()
            if path.is_dir() and DATE_DIR_RE.fullmatch(path.name)
        )
    result = [path for path in candidates if path.is_dir()]
    if not result:
        raise FileNotFoundError(f"No token-sweep data in {input_dir}")
    return result


COMPARISON_KEYS = ("rank_mode", "case", "scale_mode", "tokens_per_rank")
COMPARISON_CONFIG_FIELDS = (
    "operand_order",
    "pingpong",
    "accum_mode",
    "cluster_m",
    "cluster_n",
    "cluster_k",
    "tile_m",
    "tile_n",
    "tile_k",
    "world_size",
    "topk",
    "total_experts",
    "local_experts",
    "hidden",
    "intermediate_downproj",
    "intermediate_gateup",
    "warmup",
    "iters",
)
COMPARISON_METRICS = (
    "min_mega_us",
    "max_mega_us",
    "min_rank_tflops_per_rank",
    "max_rank_tflops_per_rank",
)


def _heuristic_points(sweep_root: Path, run_date: str | None) -> dict[tuple, dict]:
    date_dir = _date_dirs(sweep_root, run_date)[-1]
    path = date_dir / f"{date_dir.name}_token_sweep_heuristic.csv"
    points = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = tuple(row[field] for field in COMPARISON_KEYS)
            if key in points:
                raise ValueError(f"Duplicate comparison point {key} in {path}")
            for metric in COMPARISON_METRICS:
                value = float(row[metric])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid {metric} for {key} in {path}")
            points[key] = row
    return points


def compare_heuristic_sweeps(
    dir_a: Path,
    label_a: str,
    dir_b: Path,
    label_b: str,
    out_prefix: Path,
    run_date: str | None = None,
) -> tuple[Path, Path]:
    """Compare B/A-1 at each token and scale mode; positive latency = B slower.

    The legacy min/max_mega_us fields include Mega plus separate TopkReduce.
    Both rank extrema come from the same winning configuration, selected by
    slowest-rank throughput. Missing points or different configs get no ratio.
    """
    if not label_a or not label_b or label_a == label_b:
        raise ValueError("Comparison labels must be nonempty and distinct")
    a_points = _heuristic_points(dir_a.resolve(), run_date)
    b_points = _heuristic_points(dir_b.resolve(), run_date)
    keys = sorted(
        set(a_points) | set(b_points), key=lambda key: (*key[:3], int(key[3]))
    )
    if not keys:
        raise ValueError("No successful points to compare")
    value_fields = (
        *COMPARISON_CONFIG_FIELDS,
        "generate_c",
        "git_commit",
        "gpu_names",
        "source_csv",
        "log_file",
        *COMPARISON_METRICS,
    )
    delta_fields = {
        metric: f"{label_b}_vs_{label_a}_{metric}_pct" for metric in COMPARISON_METRICS
    }
    fields = (
        *COMPARISON_KEYS,
        "comparison_status",
        "same_config",
        *(f"{label}_{field}" for label in (label_a, label_b) for field in value_fields),
        *delta_fields.values(),
    )
    rows = []
    for key in keys:
        a, b = a_points.get(key), b_points.get(key)
        same_config = (
            a is not None
            and b is not None
            and all(
                a.get(field, "") == b.get(field, "")
                for field in COMPARISON_CONFIG_FIELDS
            )
        )
        row = dict(zip(COMPARISON_KEYS, key))
        row["same_config"] = str(int(same_config))
        row["comparison_status"] = (
            "matched"
            if same_config
            else "different_config" if a and b else "missing_point"
        )
        for label, point in ((label_a, a), (label_b, b)):
            for field in value_fields:
                row[f"{label}_{field}"] = point.get(field, "") if point else ""
        for metric, delta_field in delta_fields.items():
            row[delta_field] = (
                f"{(float(b[metric]) / float(a[metric]) - 1.0) * 100.0:.6f}"
                if same_config
                else ""
            )
        rows.append(row)
    row_path = Path(f"{out_prefix}.csv")
    _write_csv(row_path, fields, rows)

    tokens = sorted({key[3] for key in keys}, key=int)
    contexts = sorted({key[:3] for key in keys})
    transposed = []
    metrics = (
        "comparison_status",
        "same_config",
        *(
            f"{label}_{metric}"
            for label in (label_a, label_b)
            for metric in COMPARISON_METRICS
        ),
        *delta_fields.values(),
    )
    lookup = {tuple(row[field] for field in COMPARISON_KEYS): row for row in rows}
    for context in contexts:
        for metric in metrics:
            row = dict(zip(COMPARISON_KEYS[:3], context))
            row["metric"] = metric
            row.update(
                {
                    token: lookup.get((*context, token), {}).get(metric, "")
                    for token in tokens
                }
            )
            transposed.append(row)
        deltas = [
            float(row[delta_fields["max_mega_us"]])
            for row in rows
            if tuple(row[field] for field in COMPARISON_KEYS[:3]) == context
            and row[delta_fields["max_mega_us"]]
        ]
        if deltas:
            print(
                f"[COMPARE] {'/'.join(context)}: slowest-rank latency "
                f"median {statistics.median(deltas):+.2f}%, "
                f"range {min(deltas):+.2f}% .. {max(deltas):+.2f}% "
                f"(positive = {label_b} slower), matched={len(deltas)}"
            )
    transposed_path = Path(f"{out_prefix}_transposed.csv")
    _write_csv(transposed_path, (*COMPARISON_KEYS[:3], "metric", *tokens), transposed)
    return row_path, transposed_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--date", default=None)
    parser.add_argument(
        "--compare",
        nargs=4,
        metavar=("DIR_A", "LABEL_A", "DIR_B", "LABEL_B"),
        help="Compare two heuristic summaries (B/A-1); --date selects the run date.",
    )
    parser.add_argument(
        "--compare-out",
        type=Path,
        help="Write <prefix>.csv and <prefix>_transposed.csv.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.date is not None and not DATE_DIR_RE.fullmatch(args.date):
        raise ValueError("--date must use YYYYMMDD")
    if args.compare is not None:
        if args.compare_out is None:
            parser.error("--compare requires --compare-out")
        dir_a, label_a, dir_b, label_b = args.compare
        compare_heuristic_sweeps(
            Path(dir_a), label_a, Path(dir_b), label_b, args.compare_out, args.date
        )
        return 0
    count = 0
    for date_dir in _date_dirs(args.input_dir.resolve(), args.date):
        summarize(date_dir)
        count += 1
    print(f"SUMMARIES={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
