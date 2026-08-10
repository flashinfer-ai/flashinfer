#!/usr/bin/env python3
"""Plot Hopper FP8 token-sweep CSV files produced by the benchmark driver."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "benchmark_data"
LEGACY_TOPK = 6
LEGACY_HIDDEN = 7168
LEGACY_INTERMEDIATE_DOWNPROJ = 3072
RANK_MODE_CHOICES = ("singlerank", "multirank", "both")
CSV_NAME_RE = re.compile(
    r"^(?P<date>\d{8})_"
    r"(?P<rank>singlerank|multirank)_"
    r"(?P<scale>pertensor|blockwise)_"
    r"(?P<order>swapab|nonswapab)_"
    r"TileM(?P<m>\d+)_TileN(?P<n>\d+)\.csv$"
)
DATE_DIR_RE = re.compile(r"^\d{8}$")


@dataclass(frozen=True, order=True)
class PlotGroup:
    run_date: str
    rank_mode: str
    scale_tag: str
    order_tag: str

    def output_name(self) -> str:
        return (
            f"{self.run_date}_{self.rank_mode}_{self.scale_tag}_"
            f"{self.order_tag}.jpg"
        )


@dataclass(frozen=True, order=True)
class TileSeries:
    tile_m: int
    tile_n: int
    csv_path: Path


def _load_pyplot() -> Any:
    try:
        import matplotlib
    except ModuleNotFoundError as error:
        requirements = SCRIPT_DIR / "benchmark_requirements.txt"
        raise RuntimeError(
            "Plotting requires matplotlib. Install it with: "
            f"python -m pip install -r {requirements}"
        ) from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _latest_success_by_token(path: Path) -> dict[int, dict[str, str]]:
    latest: dict[int, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "pass":
                continue
            latest[int(row["tokens_per_rank"])] = row
    return latest


def _total_gemm_flops_per_rank(row: dict[str, str]) -> int:
    recorded = row.get("total_flops_per_rank", "")
    if recorded:
        return int(recorded)

    tokens_per_rank = int(row["tokens_per_rank"])
    topk = int(row.get("topk", "") or LEGACY_TOPK)
    routed_tokens = int(
        row.get("routed_tokens_per_rank", "") or tokens_per_rank * topk
    )
    hidden = int(row.get("hidden", "") or LEGACY_HIDDEN)
    intermediate_downproj = int(
        row.get("intermediate_downproj", "") or LEGACY_INTERMEDIATE_DOWNPROJ
    )
    intermediate_gateup = int(
        row.get("intermediate_gateup", "") or intermediate_downproj * 2
    )
    fc1_flops = 2 * routed_tokens * hidden * intermediate_gateup
    fc2_flops = 2 * routed_tokens * hidden * intermediate_downproj
    return fc1_flops + fc2_flops


def _effective_tflops_per_rank(row: dict[str, str], time_us: float) -> float:
    if time_us <= 0.0:
        raise ValueError("CUDA time must be positive")
    return _total_gemm_flops_per_rank(row) / time_us / 1_000_000.0


def _discover(
    input_dir: Path, run_date: str | None, rank_mode: str = "both"
) -> dict[PlotGroup, list[TileSeries]]:
    if rank_mode not in RANK_MODE_CHOICES:
        raise ValueError(
            f"rank_mode must be one of {','.join(RANK_MODE_CHOICES)}"
        )
    if DATE_DIR_RE.fullmatch(input_dir.name):
        if run_date is not None and run_date != input_dir.name:
            raise FileNotFoundError(
                f"Input directory {input_dir} does not match date {run_date}"
            )
        candidate_dirs = [input_dir]
    elif run_date is not None:
        candidate_dirs = [input_dir / run_date]
    else:
        candidate_dirs = sorted(
            (
                path
                for path in input_dir.iterdir()
                if path.is_dir() and DATE_DIR_RE.fullmatch(path.name)
            ),
            reverse=True,
        )

    for date_dir in candidate_dirs:
        groups: dict[PlotGroup, list[TileSeries]] = defaultdict(list)
        for path in sorted(date_dir.glob("*.csv")):
            match = CSV_NAME_RE.match(path.name)
            if match is None or match.group("date") != date_dir.name:
                continue
            if rank_mode != "both" and match.group("rank") != rank_mode:
                continue
            group = PlotGroup(
                date_dir.name,
                match.group("rank"),
                match.group("scale"),
                match.group("order"),
            )
            groups[group].append(
                TileSeries(int(match.group("m")), int(match.group("n")), path)
            )
        if groups:
            return groups

    selected = run_date or "the latest date"
    raise FileNotFoundError(
        f"No {rank_mode} token-sweep CSV files for {selected} in {input_dir}"
    )


def _date_output_dir(output_dir: Path, run_date: str) -> Path:
    return output_dir if output_dir.name == run_date else output_dir / run_date


def _title(group: PlotGroup) -> str:
    rank = "P02 single-rank" if group.rank_mode == "singlerank" else "P03 4-rank"
    scale = "per-tensor" if group.scale_tag == "pertensor" else "blockwise"
    order = "swap A/B" if group.order_tag == "swapab" else "non-swap A/B"
    return f"Hopper FP8 {rank} | {scale} | {order}"


def _plot_group(group: PlotGroup, series_list: list[TileSeries], output_dir: Path) -> Path | None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    all_tokens: set[int] = set()
    plotted = 0

    for series in sorted(series_list):
        rows = _latest_success_by_token(series.csv_path)
        points: list[tuple[int, float]] = []
        for tokens, row in sorted(rows.items()):
            metric_name = (
                "rank_0_mega_us"
                if group.rank_mode == "singlerank"
                else "max_mega_us"
            )
            raw_value = row.get(metric_name, "")
            if not raw_value:
                continue
            value_us = float(raw_value)
            if math.isfinite(value_us) and value_us > 0.0:
                points.append(
                    (tokens, _effective_tflops_per_rank(row, value_us))
                )
        if not points:
            print(f"[SKIP plot] no successful rows in {series.csv_path}")
            continue
        x_values, y_values = zip(*points)
        all_tokens.update(x_values)
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=f"Tile M{series.tile_m} x N{series.tile_n}",
        )
        plotted += 1

    if not plotted:
        plt.close(fig)
        return None

    ax.set_xscale("log", base=2)
    ordered_tokens = sorted(all_tokens)
    ax.set_xticks(ordered_tokens)
    ax.set_xticklabels([str(value) for value in ordered_tokens])
    ax.set_xlabel("Tokens per rank before top-k")
    ax.set_ylabel("Effective GEMM throughput per rank (TFLOPS)")
    ax.set_title(_title(group))
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.legend(title="MMA tile", ncols=2, fontsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / group.output_name()
    fig.savefig(output_path, dpi=180, format="jpg")
    plt.close(fig)
    print(f"[WROTE] {output_path}")
    return output_path


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
        help="Benchmark-data root for JPG output; defaults to --input-dir.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="YYYYMMDD to plot; defaults to the newest date found.",
    )
    parser.add_argument(
        "--rank-mode",
        choices=RANK_MODE_CHOICES,
        default="both",
        help="Plot singlerank, multirank, or both groups (default: both).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()
    if args.date is not None and not re.fullmatch(r"\d{8}", args.date):
        raise ValueError("--date must use YYYYMMDD")

    groups = _discover(input_dir, args.date, args.rank_mode)
    written = 0
    for group, series_list in sorted(groups.items()):
        group_output_dir = _date_output_dir(output_dir, group.run_date)
        if _plot_group(group, series_list, group_output_dir) is not None:
            written += 1
    print(f"PLOTS={written}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
