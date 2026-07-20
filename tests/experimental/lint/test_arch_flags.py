# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Arch lint: experimental kernels may not target flagship datacenter archs.

Proposal rule §1: experimental ops are for client GPU SKUs (SM12x).  This
scan forbids SM90/SM100/SM103/SM110 arch spellings, gencode flags, capability
helpers, and capability tuples anywhere under flashinfer/experimental/.

Escape hatch for a legitimately needed mention (e.g. a comment explaining a
ported heuristic): end the line with ``# arch-lint: allow``.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "flashinfer" / "experimental"

ALLOW_MARKER = "arch-lint: allow"

FORBIDDEN = [
    re.compile(pattern)
    for pattern in (
        r"\bsm_?90a?\b",
        r"\bsm_?100[af]?\b",
        r"\bsm_?103a?\b",
        r"\bsm_?110a?\b",
        r"\bcompute_(90|100|103|110)\b",
        r"is_sm(90|100|103|110)a?_supported",
        r"\(\s*9\s*,\s*0\s*\)",
        r"\(\s*10\s*,\s*[03]\s*\)",
    )
]


def test_no_datacenter_arch_targets():
    violations = []
    for suffix in ("*.py", "*.cu", "*.cuh", "*.cpp", "*.h"):
        for path in sorted(EXP.rglob(suffix)):
            if "__pycache__" in path.parts:
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if ALLOW_MARKER in line:
                    continue
                for pattern in FORBIDDEN:
                    if pattern.search(line):
                        violations.append(
                            f"  {path.relative_to(REPO)}:{lineno}: {line.strip()!r} "
                            f"(matched {pattern.pattern!r})"
                        )
    assert not violations, (
        "datacenter arch (SM90/SM100/SM103/SM110) references are forbidden in "
        "flashinfer/experimental (rule §1); append '# arch-lint: allow' only "
        "for justified mentions:\n" + "\n".join(violations)
    )
