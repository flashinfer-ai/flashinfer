#!/usr/bin/env python3
"""Replay scope preparation from saved GitHub JSON and local Git objects.

No network or inference requests are made. A fake model requests the supplied
--read-path files, then returns needs_maintainer; this measures request sizes,
not recommendation quality. PR source is read as evidence, never imported.
"""

import argparse
import asyncio
import importlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import AsyncMock, Mock


def read_pages(path):
    """Accept one JSON array or concatenated arrays from gh api --paginate."""
    text = Path(path).read_text()
    items = []
    while text.strip():
        page, end = json.JSONDecoder().raw_decode(text.lstrip())
        items.extend(page)
        text = text.lstrip()[end:]
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True, type=Path)
    parser.add_argument("--pr-json", required=True, type=Path)
    parser.add_argument("--files-json", required=True, type=Path)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--read-path", action="append", default=[])
    parser.add_argument(
        "--internal-bot-root",
        type=Path,
        help="Replay ci-bot instead, reading CI files from --ci-repo",
    )
    parser.add_argument("--ci-repo", type=Path)
    parser.add_argument("--ci-ref", default="HEAD")
    args = parser.parse_args()
    if args.internal_bot_root and not args.ci_repo:
        parser.error("--internal-bot-root requires --ci-repo")
    if args.internal_bot_root:
        sys.path.insert(0, str(args.internal_bot_root.resolve() / "src"))
    selector = importlib.import_module(
        "pytest_targets" if args.internal_bot_root else "ai_test_scope"
    )
    from merged_source import MergedSource

    class CachedSource(MergedSource):
        def git(self, *command, **kwargs):
            if command[0] == "fetch":
                raise ValueError(
                    "Offline replay needs both commits and complete history in --source-repo"
                )
            return super().git(*command, **kwargs)

    pr = json.loads(args.pr_json.read_text())
    changed = read_pages(args.files_json)
    captured = []

    def complete(context, task):
        chars = len(json.dumps(context))
        assert chars <= selector.MAX_CONTEXT_CHARS, (
            "Outgoing request exceeds the context cap"
        )
        captured.append(
            {
                "stage": "evidence" if not captured else "selection",
                "chars": chars,
                "sections": {
                    key: len(json.dumps(value)) for key, value in context.items()
                },
                "omitted": context.get("omitted_evidence", {}),
            }
        )
        if len(captured) == 1:
            return {"read_paths": args.read_path}
        return {
            "outcome": "needs_maintainer",
            "targets": [],
            "reason": "Offline size replay; no inference or coverage assessment was performed.",
        }

    def read_ci(path):
        repo = args.ci_repo if args.internal_bot_root else args.source_repo
        ref = args.ci_ref if args.internal_bot_root else args.base_sha
        return subprocess.check_output(
            ["git", "-C", str(repo), "show", ref + ":" + path], text=True
        )

    with_source = CachedSource(
        pr["base"]["repo"]["full_name"],
        args.base_sha,
        pr["head"]["sha"],
        args.source_repo.resolve(),
        timeout=240,
    )
    try:
        github = AsyncMock() if args.internal_bot_root else Mock()
        github.get_current_base_sha.return_value = args.base_sha
        github.get_pr_merge_commit.return_value = with_source.commit
        github.get_repository_tree.side_effect = lambda repo, sha: with_source.tree(sha)
        github.get_repository_blob.side_effect = (
            lambda repo, sha, limit: with_source.blob(sha, limit)
        )
        github.get_pr_files.return_value = changed
        github.get_pr_details.return_value = pr
        inference = Mock(api_key="offline-replay-not-a-key")
        inference.complete = (
            AsyncMock(side_effect=complete)
            if args.internal_bot_root
            else Mock(side_effect=complete)
        )
        routing = AsyncMock(side_effect=read_ci) if args.internal_bot_root else read_ci
        engine = selector.PytestTargetSelector(github, routing, inference)
        result = (
            asyncio.run(engine.select(pr))
            if args.internal_bot_root
            else engine.select(pr)
        )
        print(json.dumps({"outcome": result.outcome, "requests": captured}, indent=2))
    except selector.TargetSelectionError as error:
        print(json.dumps({"error": str(error), "requests": captured}, indent=2))
        return 1
    finally:
        with_source.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
