#!/usr/bin/env python3
"""Consolidate Hopper BF16 token sweeps and derive per-token heuristics."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "benchmark_data"
DATE_DIR_RE = re.compile(r"^\d{8}$")
RAW_CSV_NAME_RE = re.compile(
    r"^(?P<date>\d{8})_"
    r"(?P<rank>singlerank|multirank)_"
    r"(?P<order>swapab|nonswapab)_"
    r"(?P<schedule>legacy|pingpong)_"
    r"CGA(?P<cm>\d+)x(?P<cn>\d+)_"
    r"TileM(?P<m>\d+)_TileN(?P<n>\d+)(?:_TileK(?P<k>\d+))?\.csv$"
)

HEURISTIC_FIELDS = (
    "run_date",
    "rank_mode",
    "case",
    "tokens_per_rank",
    "routed_tokens_per_rank",
    "operand_order",
    "pingpong",
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
    "operand_order",
    "peak_tflops_per_rank(min_rank)",
    "peak_tflops_per_rank(max_rank)",
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
            rows.extend(SourceRow(path, row) for row in csv.DictReader(handle))
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


def _export_row(source_row: SourceRow) -> dict[str, str]:
    row = {field: source_row.row.get(field, "") for field in HEURISTIC_FIELDS}
    row["source_csv"] = source_row.source_csv.name
    return row


def _export_peak_summary_row(source_row: SourceRow) -> dict[str, str]:
    row = source_row.row
    return {
        "rank_mode": row.get("rank_mode", ""),
        "case": row.get("case", ""),
        "operand_order": row.get("operand_order", ""),
        "peak_tflops_per_rank(min_rank)": row.get("min_rank_tflops_per_rank", ""),
        "peak_tflops_per_rank(max_rank)": row.get("max_rank_tflops_per_rank", ""),
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


def _best_per_token(latest: Sequence[SourceRow]) -> list[SourceRow]:
    best: dict[tuple[str, str, int], SourceRow] = {}
    for source_row in latest:
        if not _valid_success(source_row):
            continue
        key = (
            source_row.row["rank_mode"],
            source_row.row["case"],
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
            item.token,
        ),
    )


def _write_heuristic(date_dir: Path, best: Sequence[SourceRow]) -> Path:
    output = date_dir / f"{date_dir.name}_token_sweep_heuristic.csv"
    _write_csv(output, HEURISTIC_FIELDS, (_export_row(item) for item in best))
    return output


def _write_peak_summary(date_dir: Path, latest: Sequence[SourceRow]) -> Path:
    best: dict[tuple[str, str], SourceRow] = {}
    for source_row in latest:
        if not _valid_success(source_row):
            continue
        key = (
            source_row.row["rank_mode"],
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
    lookup: dict[tuple[str, str, int], SourceRow] = {}
    for source_row in best:
        key = (
            source_row.row["rank_mode"],
            source_row.row["case"],
            source_row.token,
        )
        lookup[key] = source_row

    tokens = sorted({source_row.token for source_row in best})
    contexts = sorted(
        {(source_row.row["rank_mode"], source_row.row["case"]) for source_row in best}
    )
    rows: list[dict[str, str]] = []
    for rank_mode, case in contexts:
        for rank_metric, source_field in RANK_TFLOPS_METRICS:
            row = {
                "rank_mode": rank_mode,
                "case": case,
                "rank_metric": rank_metric,
            }
            for token in tokens:
                source_row = lookup.get((rank_mode, case, token))
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
    best = _best_per_token(latest)
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


# ---------------------------------------------------------------------------
# Two-sweep comparison (e.g. inference vs generate_c training forward)
# ---------------------------------------------------------------------------


def _heuristic_points(sweep_root: Path) -> dict[int, dict[str, str]]:
    """Read the newest ``<date>/<date>_token_sweep_heuristic.csv`` under ``sweep_root``.

    Returns ``{tokens_per_rank: row}`` where ``row`` carries a ``config`` label
    plus the raw ``max_mega_us`` / ``max_rank_tflops_per_rank`` strings.
    """
    date_dirs = _date_dirs(sweep_root, None)
    path = date_dirs[-1] / f"{date_dirs[-1].name}_token_sweep_heuristic.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    points: dict[int, dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            order = "swap" if row["operand_order"] == "swap_ab" else "non-swap"
            sched = "pp" if row["pingpong"] == "1" else "legacy"
            points[int(row["tokens_per_rank"])] = {
                "config": (
                    f"{order} {sched} CGA{row['cluster_m']}x{row['cluster_n']} "
                    f"M{row['tile_m']}N{row['tile_n']}K{row.get('tile_k', '128')}"
                ),
                "max_mega_us": f"{float(row['max_mega_us']):.2f}",
                "tflops": f"{float(row['max_rank_tflops_per_rank']):.2f}",
            }
    return points


def compare_heuristic_sweeps(
    dir_a: Path, label_a: str, dir_b: Path, label_b: str, out_prefix: Path
) -> tuple[Path, Path]:
    """Compare two heuristic sweeps token by token and write two CSV files.

    ``<out_prefix>.csv`` has one row per tokens-per-rank; ``<out_prefix>_transposed.csv``
    has metrics as rows and tokens-per-rank as columns.  The latency delta is
    ``B / A - 1`` in percent (positive = B slower) on the slowest-rank Mega time.
    """
    points_a = _heuristic_points(dir_a.resolve())
    points_b = _heuristic_points(dir_b.resolve())
    tokens = sorted(set(points_a) | set(points_b))
    delta_key = f"{label_b}_vs_{label_a}_latency_pct"
    rows: list[dict[str, str]] = []
    for t in tokens:
        a = points_a.get(t)
        b = points_b.get(t)
        delta = ""
        if a and b:
            delta = f"{(float(b['max_mega_us']) / float(a['max_mega_us']) - 1.0) * 100.0:+.2f}"
        rows.append(
            {
                "tokens_per_rank": str(t),
                f"{label_a}_config": a["config"] if a else "",
                f"{label_a}_max_mega_us": a["max_mega_us"] if a else "",
                f"{label_a}_tflops": a["tflops"] if a else "",
                f"{label_b}_config": b["config"] if b else "",
                f"{label_b}_max_mega_us": b["max_mega_us"] if b else "",
                f"{label_b}_tflops": b["tflops"] if b else "",
                "same_config": "1" if (a and b and a["config"] == b["config"]) else "0",
                delta_key: delta,
            }
        )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    row_path = Path(f"{out_prefix}.csv")
    with row_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    metrics = [
        (f"{label_a} config", f"{label_a}_config"),
        (f"{label_b} config", f"{label_b}_config"),
        (f"{label_a} max_mega_us", f"{label_a}_max_mega_us"),
        (f"{label_b} max_mega_us", f"{label_b}_max_mega_us"),
        (f"{label_a} tflops_per_rank", f"{label_a}_tflops"),
        (f"{label_b} tflops_per_rank", f"{label_b}_tflops"),
        (f"{label_b} vs {label_a} latency_pct", delta_key),
        ("same_config", "same_config"),
    ]
    transposed_path = Path(f"{out_prefix}_transposed.csv")
    with transposed_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", *[row["tokens_per_rank"] for row in rows]])
        for title, key in metrics:
            writer.writerow([title, *[row[key] or "-" for row in rows]])
    deltas = [float(row[delta_key]) for row in rows if row[delta_key]]
    if deltas:
        deltas_sorted = sorted(deltas)
        median = deltas_sorted[len(deltas_sorted) // 2]
        print(
            f"[COMPARE] {label_b} vs {label_a}: median {median:+.2f}%, "
            f"range {min(deltas):+.2f}% .. {max(deltas):+.2f}% "
            f"(positive = {label_b} slower) over {len(deltas)} tokens"
        )
    print(f"[COMPARE] wrote {row_path} and {transposed_path}")
    return row_path, transposed_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--date", default=None)
    parser.add_argument(
        "--compare",
        nargs=4,
        metavar=("DIR_A", "LABEL_A", "DIR_B", "LABEL_B"),
        default=None,
        help=(
            "Instead of summarizing, compare the heuristic CSVs of two sweep "
            "roots token by token (B relative to A) and write CSVs to --compare-out."
        ),
    )
    parser.add_argument(
        "--compare-out",
        type=Path,
        default=None,
        help="Output path prefix for --compare (writes <prefix>.csv and <prefix>_transposed.csv).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.compare is not None:
        if args.compare_out is None:
            parser.error("--compare requires --compare-out")
        dir_a, label_a, dir_b, label_b = args.compare
        compare_heuristic_sweeps(Path(dir_a), label_a, Path(dir_b), label_b, args.compare_out)
        return 0
    if args.date is not None and not DATE_DIR_RE.fullmatch(args.date):
        raise ValueError("--date must use YYYYMMDD")
    count = 0
    for date_dir in _date_dirs(args.input_dir.resolve(), args.date):
        summarize(date_dir)
        count += 1
    print(f"SUMMARIES={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
