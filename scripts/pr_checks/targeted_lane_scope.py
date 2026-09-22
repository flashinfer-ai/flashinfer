"""Narrow a targeted CI scope to what a lane's bare run covers.

Rule: a targeted run on a given lane never tests more than the bare run tests on
that lane. The H100 bare run covers ``tests/`` so its targeted scope is the request
unchanged; the A10G and T4 bare runs are the fixed shard lists in
``scripts/task_jit_run_tests_part*.sh``, so their targeted scope is the request
intersected with those lists. Reading the shard scripts keeps them the single
source of truth: adding a file to a shard both covers it on every bare run and
makes it eligible for targeted runs on that lane.

Coverage is the set of ``pytest ... tests/...`` invocations in the given shard
scripts. Other commands (e.g. ``bash tests/moe_ep/run_tests.sh unit``) run a
curated subset with their own flags, so they are not treated as covering their
directory for ``task_run_unit_tests.sh``; counting them would let a targeted run
test more than the bare run does.

Usage::

    targeted_lane_scope.py --scripts scripts/task_jit_run_tests_part{1..5}.sh \
        --targets "tests/attention/ tests/utils/test_topk.py"

Prints the narrowed scope, space-separated, on one line (an empty line when no
target is covered). Targets are paths under ``tests/``, files or directories, as
validated by ``experimental_test_scope.py``; ``::`` selectors are not accepted.

An empty *intersection* is normal (the lane is not scheduled). Empty *coverage* --
the shard scripts parsing to zero ``pytest`` lines -- is always parser drift (a
script switched to ``python -m pytest``, a loop over a list variable, ...) and
would otherwise silently turn every later targeted run into "lane not scheduled",
so it exits non-zero. ``pr-test.yml`` also runs ``--coverage-only`` on every
setup so drift fails the PR that introduces it.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path, PurePosixPath

_PYTEST_LINE = re.compile(r"^\s*pytest\b(?:\s+-\S+)*\s+(tests/\S+)")


def _norm(path: str) -> str:
    return PurePosixPath(path.strip().lstrip("./")).as_posix() if path.strip() else ""


def _is_dir(path: str) -> bool:
    return not path.endswith(".py")


def _under(path: str, directory: str) -> bool:
    return path == directory or path.startswith(directory + "/")


def coverage(scripts: list[Path]) -> list[str]:
    """Test files a set of shard scripts run, in stable order."""
    seen: dict[str, None] = {}
    for script in scripts:
        for line in script.read_text().splitlines():
            match = _PYTEST_LINE.match(line)
            if match:
                seen.setdefault(_norm(match.group(1)), None)
    return list(seen)


def intersect(targets: list[str], covered: list[str]) -> list[str]:
    """Requested targets narrowed to covered files.

    A requested file survives if a shard runs it; a requested directory becomes
    the covered files under it. Output preserves the request order, then
    coverage order, without duplicates.
    """
    out: dict[str, None] = {}
    for target in (_norm(t) for t in targets):
        if not target:
            continue
        if _is_dir(target):
            for file in covered:
                if _under(file, target):
                    out.setdefault(file, None)
        elif target in covered:
            out.setdefault(target, None)
    return list(out)


def _selftest() -> int:
    import tempfile

    failures: list[str] = []

    def check(name: str, got: object, want: object) -> None:
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")

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
        check(
            "coverage parses pytest lines only",
            cov,
            [
                "tests/attention/test_a.py",
                "tests/utils/test_b.py",
                "tests/gemm/test_c.py",
            ],
        )
        check(
            "file in coverage",
            intersect(["tests/utils/test_b.py"], cov),
            ["tests/utils/test_b.py"],
        )
        check("file not in coverage", intersect(["tests/kda/test_x.py"], cov), [])
        check(
            "directory expands to covered files",
            intersect(["tests/attention/"], cov),
            ["tests/attention/test_a.py"],
        )
        check(
            "directory without trailing slash",
            intersect(["tests/attention"], cov),
            ["tests/attention/test_a.py"],
        )
        check("tests/ root expands to everything", intersect(["tests/"], cov), cov)
        check(
            "uncovered runner dir is not coverage",
            intersect(["tests/moe_ep/"], cov),
            [],
        )
        check(
            "mixed request keeps order, drops uncovered, dedups",
            intersect(["tests/gemm/", "tests/kda/", "tests/gemm/test_c.py"], cov),
            ["tests/gemm/test_c.py"],
        )
        check(
            "./ prefix normalised",
            intersect(["./tests/utils/test_b.py"], cov),
            ["tests/utils/test_b.py"],
        )
        check("prefix is not containment", intersect(["tests/att"], cov), [])

        drifted = Path(td) / "drifted.sh"
        drifted.write_text("#!/bin/bash\npython -m pytest tests/attention/test_a.py\n")
        argv = sys.argv
        try:
            sys.argv = [
                "x",
                "--scripts",
                str(drifted),
                "--targets",
                "tests/attention/test_a.py",
            ]
            try:
                main()
                check("empty coverage exits non-zero", "returned", "SystemExit")
            except SystemExit as exc:
                check("empty coverage exit is an error", bool(exc.code), True)
            sys.argv = ["x", "--scripts", str(part), "--targets", "tests/kda/test_x.py"]
            check("empty intersection exits zero", main(), 0)
        finally:
            sys.argv = argv

    for failure in failures:
        print(failure)
    print("selftest: FAILED" if failures else "selftest: all cases pass")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument(
        "--scripts",
        nargs="+",
        type=Path,
        help="shard scripts defining the lane's bare-run coverage",
    )
    ap.add_argument(
        "--targets",
        default="",
        help="requested scope, space-separated paths under tests/",
    )
    ap.add_argument(
        "--coverage-only",
        action="store_true",
        help="print the lane's coverage instead of the intersection",
    )
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if not args.scripts:
        ap.error("--scripts is required (or use --selftest)")
    missing = [s for s in args.scripts if not s.is_file()]
    if missing:
        print(
            f"error: shard script not found: {', '.join(map(str, missing))}",
            file=sys.stderr,
        )
        return 2
    covered = coverage(args.scripts)
    if not covered:
        names = ", ".join(map(str, args.scripts))
        sys.exit(
            f"targeted_lane_scope: no pytest targets parsed from {names} -- parser drift?"
        )
    result = covered if args.coverage_only else intersect(args.targets.split(), covered)
    print(" ".join(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
