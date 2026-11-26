#!/usr/bin/env python3
"""Reuse CI only for a recorded head/base/tree; wait conservatively on API errors."""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path
from urllib.parse import quote

POLL_SECONDS = 300
WORKFLOWS = {".github/workflows/pr-test.yml", ".github/workflows/pr-test-dispatch.yml"}


class ReadError(Exception):
    """The API has not supplied a reliable result yet."""


def gh_api(path: str, *, binary: bool = False):
    try:
        result = subprocess.run(
            ["gh", "api", path], capture_output=True, timeout=60, check=True
        ).stdout
        return result if binary else json.loads(result)
    except (subprocess.SubprocessError, ValueError) as error:
        raise ReadError("GitHub API read unavailable") from error


class History:
    def __init__(self, repository: str):
        self.root = f"repos/{repository}"
        self.sources = {}

    def pages(self, path, key):
        # Explicit pagination keeps a later burst of label events from hiding a pass.
        page = 1
        while True:
            rows = gh_api(f"{path}&per_page=100&page={page}")[key]
            yield from rows
            if len(rows) < 100:
                return
            page += 1

    def base_sha(self, ref):
        return gh_api(f"{self.root}/git/ref/heads/{quote(ref, safe='/')}")["object"][
            "sha"
        ]

    def runs(self, head):
        return [
            run
            for run in self.pages(
                f"{self.root}/actions/runs?event=pull_request&head_sha={head}",
                "workflow_runs",
            )
            if run["path"].split("@")[0] in WORKFLOWS
            and run["head_sha"] == head
            and run["event"] == "pull_request"
        ]

    def source(self, run):
        key = (run["id"], run["run_attempt"])
        if key in self.sources:
            return self.sources[key]
        artifacts = []
        for artifact in self.pages(
            f"{self.root}/actions/runs/{run['id']}/artifacts?", "artifacts"
        ):
            match = re.fullmatch(r"pr-test-source-([1-9][0-9]*)", artifact["name"])
            if match and int(match[1]) <= run["run_attempt"]:
                artifacts.append((int(match[1]), artifact))
        source = None
        if artifacts:
            attempt, artifact = max(artifacts, key=lambda item: item[0])
            # Partial reruns reuse a successful authorize dependency's record.
            # While active, a rerun of authorize may still publish a newer one.
            if attempt < run["run_attempt"] and run["status"] != "completed":
                return None
            if not artifact["expired"] and artifact["size_in_bytes"] <= 16384:
                data = gh_api(
                    f"{self.root}/actions/artifacts/{artifact['id']}/zip", binary=True
                )
                try:
                    with zipfile.ZipFile(io.BytesIO(data)) as archive:
                        record = archive.getinfo("pr-test-source.json")
                        value = (
                            json.loads(archive.read(record))
                            if record.file_size <= 4096
                            else None
                        )
                    if isinstance(value, dict) and all(
                        isinstance(value.get(field), str)
                        and re.fullmatch(r"[0-9a-f]{40}", value[field])
                        for field in ("head", "base", "tree")
                    ):
                        source = value
                except (ValueError, KeyError, zipfile.BadZipFile):
                    pass
        # Missing artifacts can still arrive while authorization is in progress.
        if source is not None or run["status"] == "completed":
            self.sources[key] = source
        return source


def required_source(merge: str, head: str):
    """Use the event merge's actual parents, rather than a stale PR payload base."""
    if not all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (merge, head)):
        return None
    result = subprocess.check_output(
        ["git", "--no-replace-objects", "show", "-s", "--format=%T%n%P", merge],
        text=True,
        timeout=30,
    ).splitlines()
    if len(result) != 2:
        return None
    parents = result[1].split()
    if len(parents) != 2 or parents[1] != head:
        return None
    return dict(head=head, base=parents[0], tree=result[0])


def wait_for_result(history, required, ref, current_run, *, sleep=time.sleep):
    """A failed read is unknown; only a reliable snapshot can decide the check."""
    delay = POLL_SECONDS
    while True:
        try:
            if history.base_sha(ref) != required["base"]:
                return False
            runs = history.runs(required["head"])
            active = False
            passed = False
            for run in runs:
                # Older runs only: summary jobs must never wait on each other.
                if run["id"] >= current_run and run["status"] != "completed":
                    continue
                if run["status"] == "completed" and run["conclusion"] != "success":
                    continue
                source = history.source(run)
                if source == required and run["status"] == "completed":
                    passed = True
                    break
                if run["status"] != "completed" and source in (None, required):
                    active = True
            if passed:
                # The target may have advanced while history/artifacts were read.
                return history.base_sha(ref) == required["base"]
            if not active:
                return False
            print(
                "::notice::Matching CI may still be running; checking again in 5 minutes",
                flush=True,
            )
            delay = POLL_SECONDS
        except ReadError:
            print(
                "::notice::CI history is unknown (API read failed); retrying",
                flush=True,
            )
            delay = min(delay * 2, 3600)
        sleep(delay)


def main():
    required = required_source(os.environ["GITHUB_SHA"], os.environ["HEAD_SHA"])
    passed = required is not None and wait_for_result(
        History(os.environ["GITHUB_REPOSITORY"]),
        required,
        os.environ["BASE_REF"],
        int(os.environ["GITHUB_RUN_ID"]),
    )
    message = (
        "CI passed for this PR head, current base and merge tree"
        if passed
        else "No verified pass for the current source. Request fresh CI with `@flashinfer-bot run`."
    )
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as summary:
        summary.write(message + "\n")
    print(message)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
