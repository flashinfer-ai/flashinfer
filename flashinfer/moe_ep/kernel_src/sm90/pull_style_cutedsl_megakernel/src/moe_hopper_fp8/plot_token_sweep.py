#!/usr/bin/env python3
"""Plot Hopper FP8 token-sweep throughput as four separate figures."""

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
DATE_DIR_RE = re.compile(r"^\d{8}$")
RANK_MODE_CHOICES = ("singlerank", "multirank", "both")
CSV_NAME_RE = re.compile(
    r"^(?P<date>\d{8})_"
    r"(?P<rank>singlerank|multirank)_"
    r"(?P<scale>pertensor|blockwise)_"
    r"(?P<order>swapab|nonswapab)_"
    r"(?P<schedule>legacy|pingpong)_"
    r"CGA(?P<cm>\d+)x(?P<cn>\d+)_"
    r"TileM(?P<m>\d+)_TileN(?P<n>\d+)\.csv$"
)


@dataclass(frozen=True, order=True)
class Series:
    scale_tag: str
    order_tag: str
    schedule: str
    cluster_m: int
    cluster_n: int
    tile_m: int
    tile_n: int
    csv_path: Path

    @property
    def label(self) -> str:
        schedule = "PP" if self.schedule == "pingpong" else "L"
        return (
            f"{schedule} M{self.tile_m}N{self.tile_n} "
            f"C{self.cluster_m}x{self.cluster_n}"
        )


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


def _candidate_dirs(input_dir: Path, run_date: str | None) -> list[Path]:
    if DATE_DIR_RE.fullmatch(input_dir.name):
        if run_date is not None and run_date != input_dir.name:
            raise FileNotFoundError(
                f"Input directory {input_dir} does not match date {run_date}"
            )
        return [input_dir]
    if run_date is not None:
        return [input_dir / run_date]
    return sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_dir() and DATE_DIR_RE.fullmatch(path.name)
        ),
        reverse=True,
    )


def _discover(
    input_dir: Path, run_date: str | None, rank_mode: str
) -> tuple[Path, dict[str, list[Series]]]:
    for date_dir in _candidate_dirs(input_dir, run_date):
        by_rank: dict[str, list[Series]] = defaultdict(list)
        for path in sorted(date_dir.glob("*.csv")):
            match = CSV_NAME_RE.fullmatch(path.name)
            if match is None or match.group("date") != date_dir.name:
                continue
            rank = match.group("rank")
            if rank_mode != "both" and rank != rank_mode:
                continue
            by_rank[rank].append(
                Series(
                    match.group("scale"),
                    match.group("order"),
                    match.group("schedule"),
                    int(match.group("cm")),
                    int(match.group("cn")),
                    int(match.group("m")),
                    int(match.group("n")),
                    path,
                )
            )
        if by_rank:
            return date_dir, by_rank
    selected = run_date or "the latest date"
    raise FileNotFoundError(
        f"No {rank_mode} token-sweep CSV files for {selected} in {input_dir}"
    )


def _latest_success_by_token(path: Path) -> dict[int, dict[str, str]]:
    latest: dict[int, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "pass":
                latest[int(row["tokens_per_rank"])] = row
    return latest


def _critical_tflops(row: dict[str, str]) -> float | None:
    raw = row.get("max_rank_tflops_per_rank", "") or row.get(
        "critical_tflops_per_rank", ""
    )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def _plot_rank(
    run_date: str,
    rank_mode: str,
    series_list: Sequence[Series],
    output_dir: Path,
) -> list[Path]:
    plt = _load_pyplot()
    panel_keys = (
        ("pertensor", "nonswapab"),
        ("pertensor", "swapab"),
        ("blockwise", "nonswapab"),
        ("blockwise", "swapab"),
    )
    output_paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for scale_tag, order_tag in panel_keys:
        fig, ax = plt.subplots(figsize=(15, 10), constrained_layout=True)
        all_tokens: set[int] = set()
        line_count = 0
        panel_series = sorted(
            series
            for series in series_list
            if series.scale_tag == scale_tag and series.order_tag == order_tag
        )
        for series in panel_series:
            points: list[tuple[int, float]] = []
            for tokens, row in sorted(
                _latest_success_by_token(series.csv_path).items()
            ):
                tflops = _critical_tflops(row)
                if tflops is not None:
                    points.append((tokens, tflops))
            if not points:
                continue
            x_values, y_values = zip(*points)
            all_tokens.update(x_values)
            linestyle = "--" if series.schedule == "pingpong" else "-"
            marker = "s" if series.schedule == "pingpong" else "o"
            ax.plot(
                x_values,
                y_values,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.25,
                markersize=3.5,
                label=series.label,
            )
            line_count += 1

        scale = "Per-tensor" if scale_tag == "pertensor" else "Blockwise"
        order = "Swap A/B" if order_tag == "swapab" else "Non-swap A/B"
        rank_title = "P03 4-rank" if rank_mode == "multirank" else "P02 single-rank"
        ax.set_title(f"Hopper FP8 {rank_title} | {scale} | {order}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Tokens per rank before top-k")
        ax.set_ylabel("Slowest-rank effective throughput (TFLOPS/rank)")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(ncols=3, fontsize=5.5, columnspacing=0.8, handlelength=2.0)
        if not line_count:
            plt.close(fig)
            raise ValueError(
                f"No successful rows for {rank_mode} {scale_tag} {order_tag}"
            )
        ordered_tokens = sorted(all_tokens)
        ax.set_xticks(ordered_tokens)
        ax.set_xticklabels([str(value) for value in ordered_tokens], rotation=45)
        output_path = output_dir / f"{run_date}_{rank_mode}_{scale_tag}_{order_tag}.jpg"
        fig.savefig(output_path, dpi=180, format="jpg")
        plt.close(fig)
        output_paths.append(output_path)
        print(f"[WROTE] {output_path} lines={line_count}")
    return output_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument(
        "--rank-mode",
        choices=RANK_MODE_CHOICES,
        default="multirank",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.date is not None and not DATE_DIR_RE.fullmatch(args.date):
        raise ValueError("--date must use YYYYMMDD")
    input_dir = args.input_dir.resolve()
    output_root = (args.output_dir or input_dir).resolve()
    date_dir, by_rank = _discover(input_dir, args.date, args.rank_mode)
    output_dir = (
        output_root
        if output_root.name == date_dir.name
        else output_root / date_dir.name
    )
    output_paths = []
    for rank_mode, series_list in sorted(by_rank.items()):
        output_paths.extend(
            _plot_rank(date_dir.name, rank_mode, series_list, output_dir)
        )
    print(f"PLOTS={len(output_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
