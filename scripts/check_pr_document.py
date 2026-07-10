#!/usr/bin/env python3
"""Report documentation-check findings introduced by one pull request.

This is a PR adapter for the complete static rule set from
``flashinfer_document_check``.  Each rule runs on both commits, then only
findings present at the PR head but absent from its base are reported.  This
avoids turning pre-existing repository debt into contributor-facing warnings.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from pr_checks.register_checks import API_RST_MISSING, API_RST_STALE
from pr_checks.write_reports import PrFinding, emit_finding, write_report


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

CHECKER_MODULES = (
    "pr_checks.check_api_docs",
    "pr_checks.check_docstrings",
    "pr_checks.check_cross_sources",
)
CHECKER_REPORT_PATTERNS = {
    "pr_checks.check_api_docs": "api_rst_*.json",
    "pr_checks.check_docstrings": "docstring_checks.json",
    "pr_checks.check_cross_sources": "flashinfer_cross_source_check.json",
}


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], stderr=subprocess.PIPE)


def archive_revision(revision: str, destination: Path) -> None:
    data = git("archive", "--format=tar", revision)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        if hasattr(tarfile, "data_filter"):
            archive.extractall(destination, filter="data")
        else:
            archive.extractall(destination)


def run_checker(source: Path, output: Path, label: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ | {
        "FLASHINFER_SRC": str(source),
        "DOC_CHECK_OUT": str(output),
        "DOC_CHECK_VERSION": label,
    }
    for module in CHECKER_MODULES:
        result = subprocess.run(
            [sys.executable, "-m", module],
            cwd=SCRIPTS_DIR,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        # Exit 1 means the checker found drift. Exit 2 is reserved by the
        # checker entrypoints for execution failures; any other non-zero value
        # is likewise an execution failure.
        if result.returncode not in (0, 1):
            raise RuntimeError(f"{module} failed:\n{result.stdout}\n{result.stderr}")
        expected_pattern = CHECKER_REPORT_PATTERNS[module]
        if not any(output.glob(expected_pattern)):
            raise RuntimeError(
                f"{module} exited {result.returncode} without writing "
                f"its expected report ({expected_pattern}).\n"
                f"{result.stdout}\n{result.stderr}"
            )


def location_parts(location: str) -> tuple[str, int]:
    path, separator, line = location.rpartition(":")
    if separator and line.isdigit():
        return path, int(line)
    return location or "docs", 1


def load_findings(output: Path) -> set[PrFinding]:
    findings: set[PrFinding] = set()

    for report in output.glob("api_rst_*.json"):
        payload = json.loads(report.read_text())
        for module in payload.get("modules", []):
            name = module["module"]
            for symbol in module.get("missing", []):
                findings.add(
                    PrFinding(
                        API_RST_MISSING,
                        "docs/api",
                        1,
                        f"{name}.{symbol} is absent from docs/api/*.rst",
                    )
                )
            for symbol in module.get("stale", []):
                findings.add(
                    PrFinding(
                        API_RST_STALE,
                        "docs/api",
                        1,
                        f"{name}.{symbol} is documented but no longer public",
                    )
                )

    for filename in (
        "docstring_checks.json",
        "flashinfer_cross_source_check.json",
    ):
        report = output / filename
        payload = json.loads(report.read_text())
        for item in payload.get("findings", []):
            path = item.get("file", "")
            line = int(item.get("line", 0) or 0)
            if not path:
                path, line = location_parts(item.get("location", ""))
            subject = ".".join(
                part
                for part in (item.get("module", ""), item.get("symbol", ""))
                if part
            )
            message = f"{subject}: {item['message']}" if subject else item["message"]
            findings.add(PrFinding(item["check"], path, line or 1, message))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base commit SHA")
    parser.add_argument("--head", required=True, help="Head commit SHA")
    parser.add_argument(
        "--github-actions", action="store_true", help="Emit workflow annotations"
    )
    parser.add_argument(
        "--strict", action="store_true", help="Fail when this PR introduces findings"
    )
    parser.add_argument("--report-json", type=Path, help="Write findings as JSON")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="flashinfer-pr-doc-check-") as temp:
        temp_root = Path(temp)
        base_src, head_src = temp_root / "base", temp_root / "head"
        archive_revision(args.base, base_src)
        archive_revision(args.head, head_src)
        base_out, head_out = temp_root / "base-out", temp_root / "head-out"
        run_checker(base_src, base_out, "pr-base")
        run_checker(head_src, head_out, "pr-head")
        new_findings = sorted(load_findings(head_out) - load_findings(base_out))

    print(f"Static documentation checks: {len(new_findings)} new finding(s)")
    for finding in new_findings:
        emit_finding(finding, args.github_actions)
    if not new_findings:
        print("No new static documentation findings introduced by this PR.")
    if args.report_json:
        write_report(
            args.report_json,
            "static_documentation",
            args.base,
            args.head,
            new_findings,
        )
    return 1 if args.strict and new_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
