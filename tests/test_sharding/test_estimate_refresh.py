from __future__ import annotations

import csv
import gzip
from pathlib import Path

from scripts.test_sharding.observations import (
    EstimateRefresh,
    ObservedCase,
    adjust_first_case_warmup,
    refresh_estimates,
)


def _observation(
    nodeid: str,
    seconds: float,
    *,
    outcome: str = "passed",
    first: bool = False,
) -> ObservedCase:
    return ObservedCase(
        profile="profile",
        nodeid=nodeid,
        source_file="tests/test_sample.py",
        base_function="tests/test_sample.py::test_case",
        outcome=outcome,
        seconds=seconds,
        adjusted_seconds=seconds,
        synthetic=False,
        run_id="run",
        batch_id="batch",
        first_in_batch=first,
    )


def test_first_case_outlier_is_reclassified_as_source_warmup() -> None:
    observations = [
        _observation("tests/test_sample.py::test_case[slow]", 20, first=True)
    ]
    observations.extend(
        _observation(f"tests/test_sample.py::test_case[{index}]", 1)
        for index in range(10)
    )

    adjusted, warmup = adjust_first_case_warmup(observations)

    slow = next(item for item in adjusted if item.first_in_batch)
    assert slow.adjusted_seconds == 1
    assert warmup[("run", "batch")] == 19


def test_refresh_is_byte_reproducible_and_decreases_gradually(tmp_path: Path) -> None:
    duration_file = tmp_path / "estimates.csv.gz"
    with gzip.open(duration_file, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["profile", "nodeid", "estimated_seconds", "sample_count"])
        writer.writerow(
            ["profile", "tests/test_sample.py::test_case[pass]", "100", "2"]
        )
        writer.writerow(
            ["profile", "tests/test_sample.py::test_case[fail]", "100", "2"]
        )
    overhead_file = tmp_path / "overhead.csv"
    overhead_file.write_text(
        "profile,source_file,process_startup_seconds,source_warmup_seconds,sample_count\n",
        encoding="utf-8",
    )
    summary_file = tmp_path / "summary.csv"
    observations = [
        _observation("tests/test_sample.py::test_case[pass]", 50),
        _observation("tests/test_sample.py::test_case[fail]", 200, outcome="failed"),
        _observation("tests/test_sample.py::test_case[new]", 10),
    ]

    refresh_estimates(
        observations,
        [],
        EstimateRefresh(
            duration_file=duration_file,
            overhead_file=overhead_file,
            summary_file=summary_file,
        ),
    )
    first_bytes = duration_file.read_bytes()
    with gzip.open(duration_file, "rt", newline="", encoding="utf-8") as stream:
        rows = {row["nodeid"]: row for row in csv.DictReader(stream)}

    assert (
        float(rows["tests/test_sample.py::test_case[pass]"]["estimated_seconds"]) == 90
    )
    assert (
        float(rows["tests/test_sample.py::test_case[fail]"]["estimated_seconds"]) == 240
    )
    assert (
        float(rows["tests/test_sample.py::test_case[new]"]["estimated_seconds"]) == 12
    )

    # Refreshing an identical starting file with identical observations is
    # byte-for-byte stable, including the gzip header timestamp.
    second_file = tmp_path / "second.csv.gz"
    with gzip.open(second_file, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["profile", "nodeid", "estimated_seconds", "sample_count"])
        writer.writerow(
            ["profile", "tests/test_sample.py::test_case[pass]", "100", "2"]
        )
        writer.writerow(
            ["profile", "tests/test_sample.py::test_case[fail]", "100", "2"]
        )
    refresh_estimates(
        observations,
        [],
        EstimateRefresh(
            duration_file=second_file,
            overhead_file=tmp_path / "second-overhead.csv",
            summary_file=tmp_path / "second-summary.csv",
        ),
    )
    assert second_file.read_bytes() == first_bytes
