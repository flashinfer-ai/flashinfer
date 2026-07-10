"""Shared pull-request finding and report helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class PrFinding:
    """A normalized finding emitted by a PR-level checker."""

    check: str
    path: str
    line: int
    message: str
    level: str = "warning"


def emit_finding(finding: PrFinding, github_actions: bool) -> None:
    """Print one finding, optionally as a GitHub Actions annotation."""
    if github_actions:
        level = "error" if finding.level == "error" else "warning"
        message = (
            finding.message.replace("%", "%25").replace("\n", " ").replace("\r", "")
        )
        print(f"::{level} file={finding.path},line={finding.line}::{message}")
    print(f"[{finding.level.upper()}] {finding.path}:{finding.line} {finding.message}")


def write_report(
    path: Path,
    check: str,
    base: str,
    head: str,
    findings: list[PrFinding],
) -> None:
    """Write the stable report artifact consumed by the comment workflow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "check": check,
                "base": base,
                "head": head,
                "findings": [asdict(finding) for finding in findings],
            },
            indent=2,
        )
        + "\n"
    )
