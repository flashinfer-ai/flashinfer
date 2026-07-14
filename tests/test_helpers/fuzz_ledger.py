"""Shared known-failure / quarantine ledger for the quality fuzzers (gh #3605).

Every fuzz-style suite (unified MoE, scaled GEMM/BMM, and future sampling /
norm-RoPE / attention testers) needs the same two escape hatches for
already-filed bugs, with opposite semantics:

* ``quarantine=False`` (a *tolerated wrong answer*): the case still RUNS; a
  failing invariant is converted to xfail so the suite stays green on a
  tracked bug, and an unexpected PASS is flagged loudly ("fixed? remove the
  entry") so entries cannot silently outlive their bug.

* ``quarantine=True`` (the *crash class*: device-side assert / IMA /
  device-state corruption): the case is xfailed UP FRONT and the kernel is
  never launched. One such config poisons the CUDA context for every later
  test in the process (see gh #3957, gh #3604), so "run and tolerate" is not
  an option. Because a quarantined case never runs, it cannot announce its
  own fix -- which is why ``reason`` MUST name a tracking issue: the issue,
  not an xpass, is the re-enable signal.

Governance rules (enforced here so every fuzzer inherits them):

1. Every entry's ``reason`` must reference a tracker (``#NNNN``) --
   construction fails otherwise. No untracked waivers.
2. Entries are declared next to the fuzzer that owns them (context lives with
   the input model), but always through this module, so the project's entire
   waived surface is one grep away: ``grep -rn "FuzzLedger(" tests/``.
3. Quarantine scope should be as narrow as the current knowledge allows
   (backend-scoped once the culprit backend is known; config-predicate
   otherwise), and each entry should say how to probe fixed-ness.
"""

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

import pytest

_ISSUE_RE = re.compile(r"#\d+")

# Instances register here so future report tooling (e.g. the weekly xfails
# report) can enumerate the full waived surface without importing each fuzzer.
_REGISTRY: List["FuzzLedger"] = []


@dataclass(frozen=True)
class Finding:
    """One filed-and-tracked bug affecting a slice of a fuzzer's config space.

    Attributes:
        match: predicate over the owning fuzzer's config object.
        reason: human-readable description; MUST reference the tracking issue
            (``#NNNN``) and should say how to probe whether it is fixed.
        quarantine: True -> xfail up front, never launch (crash class);
            False -> run, tolerate a wrong answer, flag an unexpected pass.
        backend: optionally scope the entry to one backend key (None = any).
    """

    match: Callable[[Any], bool]
    reason: str
    quarantine: bool = False
    backend: Optional[str] = None


@dataclass
class FuzzLedger:
    """Per-fuzzer view over its :class:`Finding` entries (validated)."""

    op: str  # e.g. "unified-moe", "scaled-gemm" -- for reports/messages
    findings: Sequence[Finding] = field(default_factory=tuple)

    def __post_init__(self):
        for f in self.findings:
            if not _ISSUE_RE.search(f.reason):
                raise ValueError(
                    f"FuzzLedger({self.op!r}): every entry must reference a "
                    f"tracking issue ('#NNNN'); offending reason: {f.reason!r}"
                )
        _REGISTRY.append(self)

    def find(self, cfg, backend: Optional[str] = None) -> Optional[Finding]:
        """First entry matching this config (and backend, if scoped)."""
        for f in self.findings:
            if f.backend is not None and backend is not None and f.backend != backend:
                continue
            if f.match(cfg):
                return f
        return None

    def xfail_if_quarantined(self, cfg, backend: Optional[str] = None) -> None:
        """Call at the top of a fuzz case, BEFORE any kernel launch."""
        f = self.find(cfg, backend)
        if f is not None and f.quarantine:
            pytest.xfail(f"[{self.op}] quarantined (never run): {f.reason}")

    def flag_xpass(self, finding: Finding, tag: str) -> None:
        """Standardized loud flag when a tolerated-wrong-answer entry passes."""
        warnings.warn(
            f"[{self.op}] {tag}: KNOWN-FAILURE unexpectedly PASSED -- fixed? "
            f"remove its ledger entry ({finding.reason})",
            stacklevel=2,
        )
