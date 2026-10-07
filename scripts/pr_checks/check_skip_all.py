#!/usr/bin/env python3
"""Detect PR test files where every collected node was skipped on all lanes.

This catches the scenario where a PR adds or modifies test files targeting
hardware that no CI lane provides (e.g. SM100 tests when only H100/SM90,
A10G/SM86, and T4/SM75 lanes exist).  Pytest correctly skips those tests,
and the CI reports success — but no test in the changed file ever executed.

Usage (CI)::

    python3 scripts/pr_checks/check_skip_all.py \\
        --base-sha "$BASE_SHA" --head-sha "$HEAD_SHA" \\
        --summary junit/run-summary.json \\
        [--github-actions] [--fail]

Usage (standalone / selftest)::

    python3 scripts/pr_checks/check_skip_all.py --selftest

Inputs:

* ``--base-sha`` / ``--head-sha``: the PR diff range, used to find changed or
  added test files (``tests/**/*.py``).
* ``--summary``: one or more ``run-summary.json`` files (one per lane that
  produced results).  The format is the ``RunSummary`` written by
  ``scripts/test_sharding/summary.py``.
* ``--changed-files``: alternative to git diff — a file listing changed paths,
  one per line (for environments where git history is unavailable).

Outputs:

* Exit 0 if every changed test file had at least one non-skipped node on at
  least one lane, OR if no test files were changed.
* Exit 1 if ``--fail`` is set and any changed test file was skip-all on every
  lane.
* Exit 0 with warnings (and GitHub annotations if ``--github-actions``) if
  ``--fail`` is not set.
* A Markdown summary on stdout suitable for ``$GITHUB_STEP_SUMMARY``.

Design notes:

* The check is *per-file*, not per-node: a file with 200 skipped nodes and 1
  passed node is fine.  The signal is "nothing in this file ran anywhere."
* Source files in the summary use pytest-root-relative paths (e.g.
  ``tests/gdn/test_foo.py``).  Changed files from git diff are repo-relative.
  Both are normalized to forward-slash POSIX paths for matching.
* Only ``tests/**/*.py`` files are considered; changes to conftest, fixtures,
  or non-test modules are ignored.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FileResult:
    """Aggregated test results for one source file across all lanes."""
    path: str
    lanes: dict[str, LaneResult] = field(default_factory=dict)

    @property
    def executed_anywhere(self) -> bool:
        return any(lr.passed + lr.failed > 0 for lr in self.lanes.values())

    @property
    def collected_anywhere(self) -> bool:
        return any(lr.total > 0 for lr in self.lanes.values())

    @property
    def total_skipped(self) -> int:
        return sum(lr.skipped for lr in self.lanes.values())

    @property
    def total_passed(self) -> int:
        return sum(lr.passed for lr in self.lanes.values())

    @property
    def total_failed(self) -> int:
        return sum(lr.failed for lr in self.lanes.values())


@dataclass
class LaneResult:
    """Results for one source file on one lane."""
    lane: str
    planned: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    unknown: int = 0

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.skipped + self.unknown


@dataclass
class SkipAllFinding:
    """A changed test file that was skip-all on every lane."""
    path: str
    result: FileResult
    skip_reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Skip reason extraction from JUnit XML
# ---------------------------------------------------------------------------

def _extract_skip_reasons_from_xml(xml_path: Path, source_file: str) -> list[str]:
    """Extract distinct skip-reason messages for *source_file* from a JUnit XML.

    Pytest writes ``<skipped message="...">long text</skipped>`` inside each
    ``<testcase>`` element.  The *message* attribute is the short reason (what
    ``pytest.skip(reason=...)`` or ``skipIf(reason=...)`` supplied); the element
    text is the full representation including the file location.

    We key on the ``pytest_nodeid`` property to match nodes to their source
    file, then collect the *message* attribute (or, if absent, the element
    text) and de-duplicate.
    """
    reasons: set[str] = set()
    try:
        root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError):
        return []
    # Walk all <testcase> elements regardless of nesting.
    for testcase in root.iter("testcase"):
        # Match by source file via the pytest_nodeid property.
        nodeid = ""
        props_el = testcase.find("./properties")
        if props_el is not None:
            for prop in props_el.findall("property"):
                if prop.attrib.get("name") == "pytest_nodeid":
                    nodeid = prop.attrib.get("value", "")
                    break
        if not nodeid:
            # Fall back to classname-based matching.
            classname = testcase.attrib.get("classname", "")
            # pytest classnames use dots; source files use slashes.
            nodeid = classname.replace(".", "/") + ".py"
        # Check whether this testcase belongs to the source file we care about.
        if not nodeid.startswith(source_file.rstrip(".py").replace(".py", "")):
            # More robust: just check the source_file prefix.
            if not (
                nodeid.startswith(source_file)
                or nodeid.startswith(source_file.split("::")[0])
            ):
                continue
        skipped_el = testcase.find("./skipped")
        if skipped_el is None:
            continue
        msg = skipped_el.attrib.get("message", "").strip()
        if not msg:
            msg = (skipped_el.text or "").strip()
        if not msg:
            msg = "(no reason given)"
        # Truncate very long reasons (e.g. full tracebacks).
        if len(msg) > 200:
            msg = msg[:197] + "..."
        reasons.add(msg)
    return sorted(reasons)


def _collect_skip_reasons(
    source_file: str,
    junit_dirs: list[Path],
) -> list[str]:
    """Scan all JUnit XML files under *junit_dirs* for skip reasons.

    The JUnit directory structure written by the test runner is:
      ``junit/shards/shard-NNNN/units/<unit_id>/batches/<batch_id>.xml``
    We walk the tree and check every XML file.
    """
    all_reasons: set[str] = set()
    for junit_dir in junit_dirs:
        if not junit_dir.is_dir():
            continue
        for xml_path in junit_dir.rglob("*.xml"):
            reasons = _extract_skip_reasons_from_xml(xml_path, source_file)
            all_reasons.update(reasons)
    return sorted(all_reasons)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

_TEST_FILE_RE = re.compile(r"^tests/.*\.py$")


def _changed_test_files(base_sha: str, head_sha: str) -> list[str]:
    """Return repo-relative paths of changed/added test files in the PR."""
    try:
        output = subprocess.check_output(
            ["git", "diff", "--name-only", "--diff-filter=ACMR",
             f"{base_sha}...{head_sha}"],
            text=True, stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"git diff failed: {exc.stderr.strip() if exc.stderr else exc}"
        ) from exc
    return sorted(set(
        line.strip() for line in output.splitlines()
        if line.strip() and _TEST_FILE_RE.match(line.strip())
    ))


def _changed_test_files_from_list(path: str) -> list[str]:
    """Read changed file paths from a newline-separated file."""
    with open(path, encoding="utf-8") as fh:
        return sorted(set(
            line.strip() for line in fh
            if line.strip() and _TEST_FILE_RE.match(line.strip())
        ))


# ---------------------------------------------------------------------------
# Summary parsing
# ---------------------------------------------------------------------------

def _load_summary(path: str) -> tuple[str, list[dict]]:
    """Load a run-summary.json and return (lane_name, sources).

    The lane name is inferred from the path or defaults to "unknown".
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    # Infer lane name from parent directory structure or filename.
    # Typical: junit/run-summary.json (H100), junit-a10g/run-summary.json, etc.
    parts = PurePosixPath(path).parts
    lane = "unknown"
    for part in parts:
        lower = part.lower()
        if any(gpu in lower for gpu in ("h100", "a10g", "t4", "sm90", "sm86", "sm75")):
            lane = part
            break
    sources = data.get("sources", [])
    return lane, sources


def _aggregate_results(
    changed_files: list[str],
    summary_paths: list[str],
) -> dict[str, FileResult]:
    """Build per-file, per-lane result aggregation for changed test files."""
    results: dict[str, FileResult] = {f: FileResult(path=f) for f in changed_files}

    for summary_path in summary_paths:
        lane, sources = _load_summary(summary_path)
        for source in sources:
            source_file = source.get("source_file", "")
            if source_file not in results:
                continue
            fr = results[source_file]
            lr = fr.lanes.get(lane, LaneResult(lane=lane))
            lr.planned += source.get("planned_nodes", 0)
            lr.passed += source.get("passed", 0)
            lr.failed += source.get("failed", 0)
            lr.skipped += source.get("skipped", 0)
            lr.unknown += source.get("unknown", 0)
            fr.lanes[lane] = lr

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def find_skip_all_files(
    changed_files: list[str],
    summary_paths: list[str],
    junit_dirs: list[Path] | None = None,
) -> tuple[list[SkipAllFinding], dict[str, FileResult]]:
    """Return findings for changed test files that were skip-all on every lane.

    Also returns the full results dict for reporting purposes.
    When *junit_dirs* are provided, skip reasons are extracted from JUnit XML.
    """
    results = _aggregate_results(changed_files, summary_paths)
    findings: list[SkipAllFinding] = []

    for path in changed_files:
        fr = results[path]
        if not fr.collected_anywhere:
            continue
        if not fr.executed_anywhere:
            reasons: list[str] = []
            if junit_dirs:
                reasons = _collect_skip_reasons(path, junit_dirs)
            findings.append(SkipAllFinding(
                path=path, result=fr, skip_reasons=reasons,
            ))

    return findings, results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(
    findings: list[SkipAllFinding],
    results: dict[str, FileResult],
    changed_files: list[str],
) -> list[str]:
    """Format a Markdown report suitable for $GITHUB_STEP_SUMMARY."""
    lines: list[str] = []

    if not changed_files:
        lines.append("No test files changed in this PR.")
        return lines

    # Summary table: all changed test files and their counts.
    lines.append("### Changed Test File Results")
    lines.append("")
    lines.append("| File | Passed | Failed | Skipped | Executed | Status |")
    lines.append("|------|--------|--------|---------|----------|--------|")

    for path in changed_files:
        fr = results.get(path, FileResult(path=path))
        passed = fr.total_passed
        failed = fr.total_failed
        skipped = fr.total_skipped
        executed = passed + failed
        if not fr.collected_anywhere:
            status = "⚪ not collected"
        elif not fr.executed_anywhere:
            status = "⚠️ **skip-all**"
        elif failed:
            status = "❌ failures"
        else:
            status = "✅ ok"
        lines.append(f"| `{path}` | {passed} | {failed} | {skipped} | {executed} | {status} |")

    lines.append("")

    if findings:
        lines.append("### ⚠️ Skip-All Test Files")
        lines.append("")
        lines.append(
            "The following changed test files had **zero executed tests** across "
            "all CI lanes. Every collected node was skipped."
        )
        lines.append("")
        for finding in findings:
            fr = finding.result
            lines.append(f"**`{finding.path}`**")
            for lane_name, lr in sorted(fr.lanes.items()):
                lines.append(
                    f"  - {lane_name}: {lr.planned} planned, "
                    f"{lr.skipped} skipped, 0 executed"
                )
            if finding.skip_reasons:
                lines.append("  - Skip reasons:")
                for reason in finding.skip_reasons:
                    lines.append(f"    - `{reason}`")
            lines.append("")
        lines.append(
            "> This usually means the tests require hardware not available in CI "
            "(e.g. SM100 tests when only SM90/SM86/SM75 lanes exist). "
            "The tests are syntactically valid and correctly skip, but they have "
            "**never actually run**."
        )
        lines.append("")

    # Files not collected on any lane.
    uncollected = [
        path for path in changed_files
        if not results.get(path, FileResult(path=path)).collected_anywhere
        and path in results
    ]
    if uncollected:
        lines.append("### ⚪ Uncollected Test Files")
        lines.append("")
        lines.append(
            "These changed test files were not collected by any lane "
            "(not in any shard's scope):"
        )
        lines.append("")
        for path in uncollected:
            lines.append(f"- `{path}`")
        lines.append("")

    return lines


def emit_annotations(
    findings: list[SkipAllFinding],
    results: dict[str, FileResult],
    changed_files: list[str],
) -> None:
    """Emit GitHub Actions annotations for skip-all files."""
    for finding in findings:
        fr = finding.result
        lanes_desc = ", ".join(
            f"{name}: {lr.skipped} skipped"
            for name, lr in sorted(fr.lanes.items())
        )
        reasons_desc = ""
        if finding.skip_reasons:
            reasons_desc = " Reasons: " + "; ".join(finding.skip_reasons)
        msg = (
            f"All {fr.total_skipped} test nodes in this file were skipped "
            f"across all CI lanes ({lanes_desc}). No test actually executed.{reasons_desc}"
        )
        print(f"::warning file={finding.path},line=1::{msg}")

    uncollected = [
        path for path in changed_files
        if not results.get(path, FileResult(path=path)).collected_anywhere
        and path in results
    ]
    for path in uncollected:
        print(
            f"::notice file={path},line=1::"
            "This test file was not collected by any CI lane."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _selftest() -> int:
    """Minimal self-test with synthetic data."""
    # Simulate a summary where tests/gdn/test_foo.py has 10 skipped, 0 passed.
    summary = {
        "schema_version": 3,
        "complete": True,
        "sources": [
            {
                "source_file": "tests/gdn/test_foo.py",
                "shard_index": 0,
                "planned_nodes": 10,
                "finalized_nodes": 10,
                "pending_nodes": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 10,
                "unknown": 0,
                "synthetic": 0,
                "process_seconds": 1.0,
                "max_host_rss_mib": 0.0,
                "max_gpu_memory_mib": 0.0,
                "memory_samples": 0,
                "partial_resources": False,
                "status": "passed",
            },
            {
                "source_file": "tests/attention/test_ok.py",
                "shard_index": 0,
                "planned_nodes": 5,
                "finalized_nodes": 5,
                "pending_nodes": 0,
                "passed": 5,
                "failed": 0,
                "skipped": 0,
                "unknown": 0,
                "synthetic": 0,
                "process_seconds": 2.0,
                "max_host_rss_mib": 0.0,
                "max_gpu_memory_mib": 0.0,
                "memory_samples": 0,
                "partial_resources": False,
                "status": "passed",
            },
        ],
    }
    import tempfile
    import os

    failures = 0
    with tempfile.TemporaryDirectory() as td:
        summary_path = os.path.join(td, "run-summary.json")
        with open(summary_path, "w") as fh:
            json.dump(summary, fh)

        # Create a JUnit XML with skip reasons for the GDN file.
        junit_dir = os.path.join(td, "junit", "shards", "shard-0000",
                                 "units", "u0", "batches")
        os.makedirs(junit_dir)
        xml_path = os.path.join(junit_dir, "b0.xml")
        root_el = ET.Element("testsuites")
        suite_el = ET.SubElement(root_el, "testsuite",
                                 name="tests/gdn/test_foo.py",
                                 tests="10", skipped="10")
        for i in range(10):
            tc = ET.SubElement(suite_el, "testcase",
                               classname="tests.gdn.test_foo",
                               name=f"test_something[{i}]", time="0")
            props = ET.SubElement(tc, "properties")
            ET.SubElement(props, "property", name="pytest_nodeid",
                          value=f"tests/gdn/test_foo.py::test_something[{i}]")
            skip_el = ET.SubElement(tc, "skipped",
                                    message="ReplaySSM requires SM100/SM103")
        tree = ET.ElementTree(root_el)
        tree.write(xml_path, xml_declaration=True)

        changed = ["tests/gdn/test_foo.py", "tests/attention/test_ok.py"]
        findings, results = find_skip_all_files(
            changed, [summary_path],
            junit_dirs=[Path(td) / "junit"],
        )

        if len(findings) != 1:
            print(f"FAIL: expected 1 finding, got {len(findings)}", file=sys.stderr)
            failures += 1
        elif findings[0].path != "tests/gdn/test_foo.py":
            print(f"FAIL: wrong file: {findings[0].path}", file=sys.stderr)
            failures += 1
        else:
            # Verify skip reason was extracted.
            if not findings[0].skip_reasons:
                print("FAIL: skip_reasons is empty", file=sys.stderr)
                failures += 1
            elif "ReplaySSM requires SM100/SM103" not in findings[0].skip_reasons:
                print(f"FAIL: unexpected reasons: {findings[0].skip_reasons}",
                      file=sys.stderr)
                failures += 1

        fr_ok = results["tests/attention/test_ok.py"]
        if not fr_ok.executed_anywhere:
            print("FAIL: test_ok.py should show as executed", file=sys.stderr)
            failures += 1

        # Test with no changed test files.
        findings2, _ = find_skip_all_files([], [summary_path], junit_dirs=None)
        if findings2:
            print("FAIL: no changed files should produce no findings", file=sys.stderr)
            failures += 1

        # Test with a file not in any summary (uncollected).
        findings3, results3 = find_skip_all_files(
            ["tests/new/test_brand_new.py"], [summary_path], junit_dirs=None,
        )
        if findings3:
            print("FAIL: uncollected file should not be a skip-all finding",
                  file=sys.stderr)
            failures += 1
        if results3["tests/new/test_brand_new.py"].collected_anywhere:
            print("FAIL: brand new file should not be collected", file=sys.stderr)
            failures += 1

    print("selftest: FAILED" if failures else "selftest: all cases pass")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Detect changed test files where all nodes were skipped."
    )
    ap.add_argument("--base-sha", help="Base commit SHA for PR diff")
    ap.add_argument("--head-sha", help="Head commit SHA for PR diff")
    ap.add_argument(
        "--changed-files",
        help="File listing changed paths (alternative to --base-sha/--head-sha)",
    )
    ap.add_argument(
        "--summary", action="append", default=[],
        help="Path to run-summary.json (repeatable, one per lane)",
    )
    ap.add_argument(
        "--junit-dir", action="append", default=[],
        help="Path to JUnit output directory for skip reason extraction "
             "(repeatable, one per lane)",
    )
    ap.add_argument(
        "--github-actions", action="store_true",
        help="Emit GitHub Actions annotations",
    )
    ap.add_argument(
        "--fail", action="store_true",
        help="Exit 1 if any changed test file is skip-all on every lane",
    )
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    # Resolve changed test files.
    if args.changed_files:
        changed = _changed_test_files_from_list(args.changed_files)
    elif args.base_sha and args.head_sha:
        changed = _changed_test_files(args.base_sha, args.head_sha)
    else:
        ap.error("provide --base-sha + --head-sha, or --changed-files")
        return 1  # unreachable

    if not args.summary:
        ap.error("at least one --summary is required")
        return 1

    junit_dirs = [Path(d) for d in args.junit_dir] if args.junit_dir else None
    findings, results = find_skip_all_files(
        changed, args.summary, junit_dirs=junit_dirs,
    )

    # Report.
    report = format_report(findings, results, changed)
    for line in report:
        print(line)

    if args.github_actions:
        emit_annotations(findings, results, changed)

    if findings and args.fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
