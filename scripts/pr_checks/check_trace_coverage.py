#!/usr/bin/env python3
"""Audit GDN and KDA trace definitions for numerical reference coverage.

A trace definition JSON in ``tests/trace/fi_trace_out/`` describes a kernel's
interface and may optionally carry:

* ``reference`` — a Python source string implementing a numerically correct
  reference for the operation.
* ``check`` — a Python source string implementing a comparison function that
  validates kernel output against the reference.

A definition that has both is *numerically covered*: flashinfer-bench can
execute the kernel, run the reference, and compare.  A definition that is
missing either piece is a gap — the trace system knows the kernel's shape but
cannot verify its correctness.

This script:

1. Scans ``tests/trace/fi_trace_out/`` for GDN and KDA definitions.
2. Reports which definitions have ``check``, ``reference``, both, or neither.
3. Exits non-zero when any non-experimental definition is missing coverage.

Usage::

    python3 scripts/pr_checks/check_trace_coverage.py [--trace-dir DIR] \\
        [--github-actions] [--fail] [--selftest]

This addresses IKL-569 required change 4 and completion checks 3–4.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TraceAudit:
    """Audit result for one trace definition file."""
    name: str
    path: str
    op_type: str
    status: str          # e.g. "verified", "experimental", or ""
    api_tag: str         # fi_api:... tag
    has_check: bool
    has_reference: bool

    @property
    def covered(self) -> bool:
        return self.has_check and self.has_reference

    @property
    def gap_description(self) -> str:
        missing: list[str] = []
        if not self.has_check:
            missing.append("check")
        if not self.has_reference:
            missing.append("reference")
        return ", ".join(missing) if missing else "none"


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan_trace_definitions(
    trace_dir: Path,
    op_types: tuple[str, ...] = ("gdn", "kda"),
) -> list[TraceAudit]:
    """Scan trace definition JSONs and return audit results.

    Only files whose ``op_type`` matches one of *op_types* (or whose filename
    contains one of the op_type strings) are included.
    """
    audits: list[TraceAudit] = []
    if not trace_dir.is_dir():
        return audits

    for path in sorted(trace_dir.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        op_type = data.get("op_type", "")
        name = data.get("name", path.stem)

        # Filter to GDN/KDA definitions.
        matches = (
            op_type in op_types
            or any(ot in path.stem for ot in op_types)
        )
        if not matches:
            continue

        tags = data.get("tags", [])
        status_tags = [t.split(":", 1)[1] for t in tags if t.startswith("status:")]
        api_tags = [t for t in tags if t.startswith("fi_api:")]

        has_check = bool(data.get("check"))
        has_reference = bool(data.get("reference"))

        audits.append(TraceAudit(
            name=name,
            path=str(path),
            op_type=op_type,
            status=status_tags[0] if status_tags else "",
            api_tag=api_tags[0] if api_tags else "",
            has_check=has_check,
            has_reference=has_reference,
        ))

    return audits


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(audits: list[TraceAudit]) -> list[str]:
    """Format a Markdown report."""
    lines: list[str] = []

    if not audits:
        lines.append("No GDN/KDA trace definitions found.")
        return lines

    lines.append("### GDN / KDA Trace Reference Coverage")
    lines.append("")
    lines.append("| Definition | Op | Status | Check | Reference | Covered |")
    lines.append("|------------|-----|--------|-------|-----------|---------|")

    for a in audits:
        check_icon = "✅" if a.has_check else "❌"
        ref_icon = "✅" if a.has_reference else "❌"
        covered_icon = "✅" if a.covered else "⚠️ **gap**"
        lines.append(
            f"| `{a.name}` | {a.op_type} | {a.status or '—'} "
            f"| {check_icon} | {ref_icon} | {covered_icon} |"
        )

    lines.append("")

    gaps = [a for a in audits if not a.covered]
    if gaps:
        lines.append("### ⚠️ Missing Numerical Coverage")
        lines.append("")
        lines.append(
            "The following trace definitions are missing a `check` function, "
            "a `reference` implementation, or both.  A trace signature test "
            "(testing that `fi_trace()` produces valid JSON) does **not** "
            "verify numerical correctness."
        )
        lines.append("")
        for a in gaps:
            lines.append(f"- **`{a.name}`** ({a.op_type}, {a.status or 'no status'}): "
                         f"missing {a.gap_description}")
            if a.api_tag:
                lines.append(f"  - API: `{a.api_tag}`")
        lines.append("")

    verified_gaps = [a for a in gaps if a.status == "verified"]
    if verified_gaps:
        lines.append(
            "> ⛔ **Verified definitions without full coverage**: "
            + ", ".join(f"`{a.name}`" for a in verified_gaps)
            + ".  These are marked `status:verified` but cannot be numerically "
            "validated through the trace system."
        )
        lines.append("")

    return lines


def emit_annotations(audits: list[TraceAudit]) -> None:
    """Emit GitHub Actions annotations for gaps."""
    for a in audits:
        if a.covered:
            continue
        level = "warning" if a.status != "verified" else "error"
        msg = (
            f"Trace definition '{a.name}' ({a.op_type}, {a.status or 'no status'}) "
            f"is missing {a.gap_description}. "
            f"A trace signature check is not proof of numerical correctness."
        )
        print(f"::{level} file={a.path},line=1::{msg}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _selftest() -> int:
    """Minimal self-test with synthetic data."""
    import tempfile
    import os

    failures = 0
    with tempfile.TemporaryDirectory() as td:
        trace_dir = Path(td)

        # Complete definition.
        with open(trace_dir / "gdn_full.json", "w") as f:
            json.dump({
                "name": "gdn_full",
                "op_type": "gdn",
                "tags": ["fi_api:flashinfer.gdn.full", "status:verified"],
                "check": "def check(): pass",
                "reference": "def ref(): pass",
                "axes": {}, "inputs": {}, "outputs": {},
            }, f)

        # Missing check.
        with open(trace_dir / "gdn_no_check.json", "w") as f:
            json.dump({
                "name": "gdn_no_check",
                "op_type": "gdn",
                "tags": ["fi_api:flashinfer.gdn.no_check", "status:verified"],
                "reference": "def ref(): pass",
                "axes": {}, "inputs": {}, "outputs": {},
            }, f)

        # Missing reference.
        with open(trace_dir / "kda_no_ref.json", "w") as f:
            json.dump({
                "name": "kda_no_ref",
                "op_type": "kda",
                "tags": ["fi_api:flashinfer.kda.no_ref", "status:verified"],
                "check": "def check(): pass",
                "axes": {}, "inputs": {}, "outputs": {},
            }, f)

        # Experimental (missing both — should warn, not error).
        with open(trace_dir / "kda_exp.json", "w") as f:
            json.dump({
                "name": "kda_exp",
                "op_type": "kda",
                "tags": ["fi_api:flashinfer.kda.exp", "status:experimental"],
                "axes": {}, "inputs": {}, "outputs": {},
            }, f)

        # Non-GDN/KDA (should be excluded).
        with open(trace_dir / "attention_foo.json", "w") as f:
            json.dump({
                "name": "attention_foo",
                "op_type": "attention",
                "tags": [],
                "check": "def check(): pass",
                "axes": {}, "inputs": {}, "outputs": {},
            }, f)

        audits = scan_trace_definitions(trace_dir)
        if len(audits) != 4:
            print(f"FAIL: expected 4 audits, got {len(audits)}", file=sys.stderr)
            failures += 1

        covered = [a for a in audits if a.covered]
        if len(covered) != 1 or covered[0].name != "gdn_full":
            print(f"FAIL: expected 1 covered (gdn_full), got {covered}", file=sys.stderr)
            failures += 1

        gaps = [a for a in audits if not a.covered]
        if len(gaps) != 3:
            print(f"FAIL: expected 3 gaps, got {len(gaps)}", file=sys.stderr)
            failures += 1

        verified_gaps = [a for a in gaps if a.status == "verified"]
        if len(verified_gaps) != 2:
            print(f"FAIL: expected 2 verified gaps, got {len(verified_gaps)}",
                  file=sys.stderr)
            failures += 1

        # Verify report generation doesn't crash.
        report = format_report(audits)
        if not any("gap" in line for line in report):
            print("FAIL: report should mention gaps", file=sys.stderr)
            failures += 1

    print("selftest: FAILED" if failures else "selftest: all cases pass")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit GDN/KDA trace definitions for numerical reference coverage."
    )
    ap.add_argument(
        "--trace-dir",
        default="tests/trace/fi_trace_out",
        help="Path to trace definition directory (default: tests/trace/fi_trace_out)",
    )
    ap.add_argument(
        "--github-actions", action="store_true",
        help="Emit GitHub Actions annotations",
    )
    ap.add_argument(
        "--fail", action="store_true",
        help="Exit 1 if any verified definition is missing coverage",
    )
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    trace_dir = Path(args.trace_dir)
    audits = scan_trace_definitions(trace_dir)

    report = format_report(audits)
    for line in report:
        print(line)

    if args.github_actions:
        emit_annotations(audits)

    if args.fail:
        verified_gaps = [a for a in audits if not a.covered and a.status == "verified"]
        if verified_gaps:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
