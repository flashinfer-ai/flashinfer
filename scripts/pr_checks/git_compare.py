"""Shared Git helpers for PR base/head comparisons."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def resolve_merge_base(base: str, head: str) -> str:
    """Return the common ancestor of the supplied target and PR head commits."""
    try:
        resolved = subprocess.check_output(
            ["git", "merge-base", base, head],
            text=True,
            errors="replace",
            stderr=subprocess.PIPE,
        ).strip()
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() if exc.stderr else str(exc)
        raise RuntimeError(
            f"Could not resolve merge base for {base} and {head}: {detail}"
        ) from exc
    if not resolved:
        raise RuntimeError(f"Git returned an empty merge base for {base} and {head}")
    return resolved


def resolve_pr_comparison(merge: str, head: str) -> tuple[str, str]:
    """Use the checked-out PR test merge's target parent, validating its head.

    An event's base SHA can predate main commits already merged into the PR.
    Taking the target from the same test merge avoids attributing those commits
    to the PR; merge-base still excludes target-only changes for unsynced PRs.
    """
    try:
        head_sha = subprocess.check_output(
            ["git", "rev-parse", "--verify", "--end-of-options", f"{head}^{{commit}}"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        merge_sha = subprocess.check_output(
            ["git", "rev-parse", "--verify", "--end-of-options", f"{merge}^{{commit}}"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        commits = subprocess.check_output(
            ["git", "rev-list", "--parents", "-n", "1", merge_sha],
            text=True,
            stderr=subprocess.PIPE,
        ).split()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Could not resolve PR test merge: {exc.stderr.strip()}"
        ) from exc
    if len(commits) != 3 or commits[2] != head_sha:
        raise RuntimeError(
            "Checkout must be a two-parent PR test merge with the expected "
            f"head {head_sha} as its second parent; got {merge_sha}."
        )
    return commits[1], head_sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge", default="HEAD", help="Checked-out PR test merge")
    parser.add_argument("--head", required=True, help="Expected PR head commit")
    parser.add_argument("--github-output", required=True, type=Path)
    args = parser.parse_args()
    try:
        base, head = resolve_pr_comparison(args.merge, args.head)
    except RuntimeError as exc:
        parser.error(str(exc))
    with args.github_output.open("a") as output:
        output.write(f"base={base}\nhead={head}\n")
    print(f"PR comparison: target {base}, head {head}")


if __name__ == "__main__":
    main()
