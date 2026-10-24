import json
import sys
import tarfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.collect_ninja_diagnostics import (
    collect,
    latest_invocation,
    parse_ninja_log,
)


def test_collects_completed_edge_timings_and_raw_state(tmp_path: Path) -> None:
    build_root = tmp_path / "aot-providers"
    cached_ops = build_root / "sm90a" / "cached_ops"
    module_dir = cached_ops / "attention"
    tmp_dir = cached_ops / "tmp"
    module_dir.mkdir(parents=True)
    tmp_dir.mkdir()
    ninja_log = cached_ops / ".ninja_log"
    ninja_log.write_text(
        "# ninja log v5\n"
        "0\t100\t1000\tattention/a.cuda.o\thash-a\n"
        "0\t300\t1001\tattention/b.cuda.o\thash-b\n"
        "301\t350\t1002\tattention/attention.so\thash-c\n"
    )
    (module_dir / "build.ninja").write_text("rule cuda_compile\n")
    (tmp_dir / "flashinfer_jit.ninja").write_text("subninja ../attention/build.ninja\n")
    output_dir = tmp_path / "diagnostics"

    report = collect(build_root, output_dir)

    log = report["logs"][0]
    assert log["records_in_latest_invocation"] == 3
    assert log["duration_ms"] == {
        "maximum": 300,
        "p50": 100,
        "p90": 300,
        "p95": 300,
        "p99": 300,
    }
    assert log["maximum_gap_between_completions_ms"] == 200
    assert log["maximum_parallel_completed_edges"] == 2
    assert log["longest_completed_edges"][0]["output"] == "attention/b.cuda.o"
    assert (output_dir / "ninja-log.tsv").read_text() == ninja_log.read_text()
    assert json.loads((output_dir / "ninja-summary.json").read_text()) == report
    assert "Longest completed edges:" in (output_dir / "ninja-summary.txt").read_text()
    with tarfile.open(output_dir / "ninja-state.tar.gz") as archive:
        assert set(archive.getnames()) == {
            "sm90a/cached_ops/.ninja_log",
            "sm90a/cached_ops/attention/build.ninja",
            "sm90a/cached_ops/tmp/flashinfer_jit.ninja",
        }


def test_uses_latest_invocation_from_appended_log(tmp_path: Path) -> None:
    log_path = tmp_path / ".ninja_log"
    log_path.write_text(
        "# ninja log v5\n"
        "0\t100\t1000\told-a.o\thash-a\n"
        "20\t200\t1001\told-b.o\thash-b\n"
        "0\t40\t1002\tnew-a.o\thash-c\n"
        "10\t60\t1003\tnew-b.o\thash-d\n"
    )

    version, entries = parse_ninja_log(log_path)
    latest = latest_invocation(entries)

    assert version == "v5"
    assert [entry.output for entry in latest] == ["new-a.o", "new-b.o"]


def test_writes_empty_report_when_ninja_never_started(tmp_path: Path) -> None:
    output_dir = tmp_path / "diagnostics"

    report = collect(tmp_path / "missing-build-root", output_dir)

    assert report["logs"] == []
    assert report["archived_state_files"] == 0
    assert not (output_dir / "ninja-state.tar.gz").exists()
    assert "ninja_logs_found: 0" in (output_dir / "ninja-summary.txt").read_text()
