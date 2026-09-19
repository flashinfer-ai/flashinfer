# SPDX-License-Identifier: Apache-2.0
"""Explicitly check/apply the experimental FA4 patch to a pinned clean checkout.

Default is read-only. Use --apply only on a dedicated checkout.
"""

import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.checkout.resolve(strict=True)
    patch = (
        Path(__file__).resolve().parents[1]
        / "flashinfer/experimental/ulysses/patches/flash-attention-sm100-distributed-qo.patch"
    )

    def git(*parts, check=True):
        return subprocess.run(
            ["git", "-C", str(root), *parts],
            check=check,
            capture_output=True,
            text=True,
        )

    expected = "c2006099f3ff03de187f4e1b27e756fe6df482ba"
    if git("rev-parse", "HEAD").stdout.strip() != expected:
        parser.error(f"requires pinned FlashAttention commit {expected}")
    if (
        git(
            "apply", "--unidiff-zero", "--reverse", "--check", str(patch), check=False
        ).returncode
        == 0
    ):
        print("Patch already present; no changes made.")
        return
    if git("status", "--porcelain").stdout.strip():
        parser.error("checkout is not clean; refusing to modify user changes")
    git("apply", "--unidiff-zero", "--check", str(patch))
    if args.apply:
        git("apply", "--unidiff-zero", str(patch))
        print("Applied distributed FA4 patch.")
    else:
        print("Patch check passed; use --apply to modify this checkout.")


if __name__ == "__main__":
    main()
