"""Narrow a targeted CI scope to what a lane's bare run covers.

A targeted run on a lane never tests more than that lane's bare run. The A10G and
T4 bare runs are the fixed shard lists in ``scripts/task_jit_run_tests_part*.sh``,
so their targeted scope is the requested paths intersected with the ``pytest``
targets of those scripts. Reading the scripts keeps them the single source of
truth: adding a file to a shard covers it on every bare run and makes it eligible
for targeted runs on that lane. (H100's bare run covers ``tests/``, so it needs no
narrowing and does not use this tool.)

Only ``pytest ... tests/...`` lines count. Wrapper commands such as
``bash tests/moe_ep/run_tests.sh unit`` run a curated subset with their own
flags, so they do not make their directory eligible.

Usage::

    targeted_lane_scope.py --scripts scripts/task_jit_run_tests_part{1..5}.sh \\
        --targets "tests/attention/ tests/utils/test_topk.py"

Prints the narrowed scope on one line; an empty line means nothing requested is
covered and the lane should not be scheduled. Targets are files or directories
under ``tests/`` (``::`` selectors are rejected upstream by
``experimental_test_scope.py``). A shard script with no parseable ``pytest`` line
is a parser-drift bug, not an empty scope, and exits non-zero; ``pr-test.yml``
runs ``--coverage-only`` on every setup so such drift fails the PR that causes it.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path, PurePosixPath

_PYTEST_LINE = re.compile(r"^\s*pytest\b(?:\s+-\S+)*\s+(tests/\S+)")


def _norm(path: str) -> str:
    """``./tests/x/`` -> ``tests/x``."""
    path = path.strip()
    return PurePosixPath(path).as_posix() if path else ""


def _is_test_file(path: str) -> bool:
    # Targets are not existence-checked, so classify by shape.
    return PurePosixPath(path).suffix == ".py"


def _within(path: str, directory: str) -> bool:
    # Component-wise: tests/att does not contain tests/attention/x.py.
    return PurePosixPath(path).is_relative_to(PurePosixPath(directory))


def coverage(scripts: list[Path]) -> list[str]:
    """Test files the shard scripts run, in script order, without duplicates."""
    files: dict[str, None] = {}  # insertion-ordered set
    unparsed: list[str] = []
    for script in scripts:
        matches = filter(None, map(_PYTEST_LINE.match, script.read_text().splitlines()))
        targets = [_norm(m.group(1)) for m in matches]
        if not targets:
            unparsed.append(str(script))
        files.update(dict.fromkeys(targets))
    # Checked per script: with five A10G shards, one drifting to an unparsed
    # invocation would leave the total non-empty and silently drop its files.
    if unparsed:
        raise ValueError(
            f"no pytest targets parsed from {', '.join(unparsed)} -- parser drift?"
        )
    return list(files)


def intersect(targets: list[str], covered: list[str]) -> list[str]:
    """Requested files kept if covered; requested directories become their covered files."""
    selected: dict[str, None] = {}  # insertion-ordered set
    for target in filter(None, map(_norm, targets)):
        if _is_test_file(target):
            if target in covered:
                selected[target] = None
        else:
            selected.update(dict.fromkeys(f for f in covered if _within(f, target)))
    return list(selected)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument(
        "--scripts",
        nargs="+",
        type=Path,
        help="shard scripts defining the lane's coverage",
    )
    ap.add_argument(
        "--targets",
        default="",
        help="requested scope: space-separated paths under tests/",
    )
    ap.add_argument(
        "--coverage-only",
        action="store_true",
        help="print the coverage instead of the intersection",
    )
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if not args.scripts:
        ap.error("--scripts is required (or use --selftest)")
    missing = [str(s) for s in args.scripts if not s.is_file()]
    if missing:
        sys.exit(f"targeted_lane_scope: shard script not found: {', '.join(missing)}")
    try:
        covered = coverage(args.scripts)
    except ValueError as exc:
        sys.exit(f"targeted_lane_scope: {exc}")
    print(
        " ".join(
            covered if args.coverage_only else intersect(args.targets.split(), covered)
        )
    )
    return 0


def _selftest() -> int:
    import tempfile

    failures: list[str] = []

    def check(name: str, got: object, want: object) -> None:
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    def raises(name: str, exc_type: type[BaseException], fn) -> BaseException | None:
        try:
            fn()
        except exc_type as exc:
            return exc
        failures.append(f"{name}: expected {exc_type.__name__}")
        return None

    with tempfile.TemporaryDirectory() as td:
        part = Path(td) / "part.sh"
        part.write_text(
            "#!/bin/bash\n"
            "set -x\n"
            "# pytest -s tests/commented_out.py\n"
            "bash tests/moe_ep/run_tests.sh unit\n"
            "pytest -s tests/attention/test_a.py\n"
            "  pytest -s -x tests/utils/test_b.py\n"
            "pytest tests/gemm/test_c.py\n"
        )
        cov = coverage([part])
        a, b, c = (
            "tests/attention/test_a.py",
            "tests/utils/test_b.py",
            "tests/gemm/test_c.py",
        )
        check(
            "coverage: pytest lines only, comments and wrappers ignored", cov, [a, b, c]
        )
        check("file in coverage", intersect([b], cov), [b])
        check("file not in coverage", intersect(["tests/kda/test_x.py"], cov), [])
        check(
            "directory expands to covered files",
            intersect(["tests/attention/"], cov),
            [a],
        )
        check(
            "directory without trailing slash", intersect(["tests/attention"], cov), [a]
        )
        check("tests/ expands to everything", intersect(["tests/"], cov), cov)
        check(
            "wrapper-run directory is not covered",
            intersect(["tests/moe_ep/"], cov),
            [],
        )
        check(
            "order kept, uncovered dropped, deduplicated",
            intersect(["tests/gemm/", "tests/kda/", c], cov),
            [c],
        )
        check("./ prefix normalised", intersect(["./" + b], cov), [b])
        check("path prefix is not containment", intersect(["tests/att"], cov), [])

        drifted = Path(td) / "drifted.sh"
        drifted.write_text("#!/bin/bash\npython -m pytest tests/attention/test_a.py\n")
        exc = raises("unparsed shard rejected", ValueError, lambda: coverage([drifted]))
        exc = raises(
            "one unparsed shard among parsed ones rejected",
            ValueError,
            lambda: coverage([part, drifted]),
        )
        if exc:
            check("error names the unparsed script", str(drifted) in str(exc), True)
            check("error does not name the parsed script", str(part) in str(exc), False)
        exc = raises(
            "main: unparsed shard exits non-zero",
            SystemExit,
            lambda: main(["--scripts", str(drifted)]),
        )
        if exc:
            check("main: unparsed shard exit code is an error", bool(exc.code), True)
        check(
            "main: empty intersection exits zero",
            main(["--scripts", str(part), "--targets", "tests/kda/x.py"]),
            0,
        )

    for failure in failures:
        print(failure)
    print("selftest: FAILED" if failures else "selftest: all cases pass")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
