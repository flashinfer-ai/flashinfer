#!/usr/bin/env python3
"""Warn in Slack before the flashinfer-python PyPI files reach the size limit.

PyPI rejects any file larger than the project's per-file limit, and a release
that hits it fails at upload time. `.github/workflows/pypi-package-size.yml`
builds the files `release.yml` uploads (the sdist and the wheel built with
FLASHINFER_WHEEL_EXCLUDE_EXPERIMENTAL_SOURCES=1) from main every night and runs
this script on them. When the largest file reaches the warning threshold it
posts to Slack, every night until the package shrinks again, so the size can
be reduced before a release is blocked.

Sizes are MiB (2**20 bytes), the unit PyPI file size limits are set in.

Usage:
    python scripts/infra/check_pypi_package_size.py --dist-dir dist \
        --warn-mib 250 --limit-mib 300 --slack-channel C09AY3CNFEK

SLACK_BOT_TOKEN must hold a bot token with chat:write that is a member of the
channel. Without --slack-channel, or with --dry-run, the message is only
printed. A failed Slack post exits non-zero so a broken alert is visible.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

MIB = 1024 * 1024
SLACK_POST_URL = "https://slack.com/api/chat.postMessage"
# The files release.yml uploads to PyPI: the flashinfer-python sdist and wheel.
PACKAGE_GLOBS = ("flashinfer_python-*.whl", "flashinfer_python-*.tar.gz")


@dataclass(frozen=True)
class PackageFile:
    name: str
    size: int

    @property
    def mib(self) -> float:
        return self.size / MIB


def find_package_files(dist_dir: pathlib.Path) -> list[PackageFile]:
    paths = sorted({p for pattern in PACKAGE_GLOBS for p in dist_dir.glob(pattern)})
    return [PackageFile(p.name, p.stat().st_size) for p in paths]


def status(largest: PackageFile, warn_mib: float, limit_mib: float) -> str:
    if largest.mib >= limit_mib:
        return "over_limit"
    if largest.mib >= warn_mib:
        return "warning"
    return "ok"


def format_report(
    files: list[PackageFile],
    warn_mib: float,
    limit_mib: float,
    run_url: Optional[str] = None,
    commit: Optional[str] = None,
) -> str:
    """Slack mrkdwn text; also readable as Markdown in the job summary."""
    largest = max(files, key=lambda f: f.size)
    state = status(largest, warn_mib, limit_mib)
    if state == "over_limit":
        headline = (
            f":rotating_light: *flashinfer-python is over the {limit_mib:g} MiB "
            f"PyPI file limit*: `{largest.name}` is {largest.mib:.1f} MiB. "
            "The next release upload will be rejected."
        )
    elif state == "warning":
        headline = (
            f":warning: *flashinfer-python reached the {warn_mib:g} MiB warning "
            f"threshold*: `{largest.name}` is {largest.mib:.1f} MiB, "
            f"{limit_mib - largest.mib:.1f} MiB below the {limit_mib:g} MiB "
            "PyPI file limit. Please prioritize reducing the package size."
        )
    else:
        headline = (
            f":white_check_mark: flashinfer-python is within budget: largest file "
            f"`{largest.name}` is {largest.mib:.1f} MiB (warning at "
            f"{warn_mib:g} MiB, PyPI limit {limit_mib:g} MiB)."
        )
    lines = [headline, ""]
    for f in sorted(files, key=lambda f: f.size, reverse=True):
        lines.append(f"• `{f.name}`: {f.mib:.1f} MiB ({f.size:,} bytes)")
    context = []
    if commit:
        context.append(f"main @ `{commit[:12]}`")
    if run_url:
        context.append(f"<{run_url}|workflow run>")
    if context:
        lines += ["", " | ".join(context)]
    return "\n".join(lines)


def post_to_slack(
    token: str,
    channel: str,
    text: str,
    urlopen: Callable = urllib.request.urlopen,
) -> None:
    payload = {"channel": channel, "text": text, "unfurl_links": False}
    request = urllib.request.Request(
        SLACK_POST_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        result = json.load(response)
    # chat.postMessage answers HTTP 200 with ok=false on errors such as
    # not_in_channel or invalid_auth.
    if not result.get("ok"):
        raise RuntimeError(f"Slack chat.postMessage failed: {result.get('error')}")


def default_run_url() -> Optional[str]:
    server = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if server and repo and run_id:
        return f"{server}/{repo}/actions/runs/{run_id}"
    return None


def main(
    argv: Optional[list[str]] = None, urlopen: Callable = urllib.request.urlopen
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dist-dir", type=pathlib.Path, default=pathlib.Path("dist")
    )
    parser.add_argument("--warn-mib", type=float, default=250.0)
    parser.add_argument("--limit-mib", type=float, default=300.0)
    parser.add_argument(
        "--slack-channel",
        help="Slack channel ID to alert; omit to only print the report",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the alert instead of posting it",
    )
    args = parser.parse_args(argv)
    if args.warn_mib > args.limit_mib:
        parser.error("--warn-mib must not exceed --limit-mib")

    files = find_package_files(args.dist_dir)
    if not files:
        print(
            f"error: no flashinfer-python sdist or wheel in {args.dist_dir}",
            file=sys.stderr,
        )
        return 1

    report = format_report(
        files,
        args.warn_mib,
        args.limit_mib,
        run_url=default_run_url(),
        commit=os.environ.get("GITHUB_SHA"),
    )
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(report + "\n")

    largest = max(files, key=lambda f: f.size)
    if status(largest, args.warn_mib, args.limit_mib) == "ok":
        return 0
    if not args.slack_channel or args.dry_run:
        print("Threshold reached; Slack alert not sent (dry run or no channel).")
        return 0
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token:
        print(
            "error: threshold reached but SLACK_BOT_TOKEN is not set",
            file=sys.stderr,
        )
        return 1
    post_to_slack(token, args.slack_channel, report, urlopen=urlopen)
    print(f"Posted size alert to Slack channel {args.slack_channel}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
