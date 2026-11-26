#!/usr/bin/env python3
"""Publish a response-only internal /bot run gate without posting PR comments."""

import argparse
import copy
from dataclasses import dataclass
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from ai_test_scope import (
    EVIDENCE_PROMPT,
    GitHub,
    InferenceHub,
    PytestTargetSelector,
    TargetSelectionError,
    _public_prose,
    _remaining,
)

CHECK_NAME = "Bot Run Request Check"
BOT_ID = 236526959  # Stable GitHub account ID; never rely on copied display text.
PUBLIC_NOTE = "<!-- flashinfer-bot:ci-bot-commands-note -->"
VERSION = "bot-run-request-v1"


class PinnedPRSource:
    """Bind source analysis to one immutable base/head/file-list snapshot."""

    def __init__(self, client, pr, base_sha, changed):
        self.client = client
        self.number = pr["number"]
        self.base_sha = base_sha
        self.changed = copy.deepcopy(changed)

    def __getattr__(self, name):
        return getattr(self.client, name)

    @property
    def deadline(self):
        return self.client.deadline

    @deadline.setter
    def deadline(self, value):
        self.client.deadline = value

    def get_current_base_sha(self, pr):
        # run_pr checks the real current base separately before publication.
        return self.base_sha

    def get_pr_files(self, number):
        if number != self.number:
            raise TargetSelectionError("Unexpected PR for pinned file list")
        return copy.deepcopy(self.changed)


class RetryingImpactInference:
    """Reuse validated file requests only while their pinned context is identical."""

    def __init__(self, model):
        self.model = model
        self.api_key = model.api_key
        self.deadline = None
        self.retry = False
        self.evidence = None
        self.evidence_context = None
        self.confirmed = False

    def complete(self, context, task):
        if task == EVIDENCE_PROMPT:
            key = json.dumps(context, sort_keys=True)
            if self.retry and self.confirmed and key == self.evidence_context:
                return copy.deepcopy(self.evidence)
            self.confirmed = False
        else:
            # Selection is reached only after the selector validates read_paths.
            self.confirmed = True
        self.model.deadline = self.deadline
        result = self.model.complete(context, task)
        if task == EVIDENCE_PROMPT:
            self.evidence = copy.deepcopy(result)
            self.evidence_context = key
        return result


def analyze_gpu_impact(client, pr, read_ci_file, inference=None):
    model = inference or InferenceHub()
    wrapped = RetryingImpactInference(model)
    deadline = client.deadline
    attempts = []
    secrets = (model.api_key, os.getenv("GH_TOKEN"), os.getenv("GITHUB_TOKEN"))
    for index, effort in enumerate((None, "minimal")):
        model.reasoning_effort = effort
        wrapped.retry = index > 0
        try:
            # Share the outer budget between attempts and leave publication time.
            available = _remaining(deadline, 540) - 60
            if available <= 0:
                raise TargetSelectionError("Insufficient time for analysis retry")
            selector = PytestTargetSelector(client, read_ci_file, inference=wrapped)
            selector.timeout = min(selector.timeout, available / (2 - index))
            selection = selector.select(pr, include_gpu_impact=True)
            impact = {**selection.gpu_impact, "targets": selection.paths}
            if impact["classification"] == "unknown":
                raise TargetSelectionError(impact["reason"])
        except (TargetSelectionError, ValueError) as error:
            attempts.append(
                {
                    "effort": effort or "default",
                    "error": _public_prose(str(error), secrets),
                }
            )
        else:
            attempts.append({"effort": effort or "default", "success": True})
            return {**impact, "analysis_attempts": attempts}
        finally:
            client.deadline = deadline
    return {
        "classification": "unknown",
        "reason": attempts[-1]["error"],
        "evidence_paths": [],
        "targets": [],
        "case_gap": "",
        "analysis_attempts": attempts,
    }


def is_run_command(body):
    """Match a command at the start, as the internal bot does; ignore examples."""
    return bool(re.match(r"\A[ \t]*/bot[ \t]+run(?:[ \t\r\n]|$)", body.strip(), re.I))


def is_bot_response(item):
    user = item.get("user", {})
    if user.get("id") != BOT_ID or user.get("login", "").lower() != "flashinfer-bot":
        return False
    body = item.get("body", "").strip()
    if (
        not body
        or body.startswith((">", "`", "~"))
        or PUBLIC_NOTE in body
        or re.search(r"@+flashinfer-bot\b", body, re.I)
    ):
        return False
    # Internal acknowledgements may follow scope/cost or stacked-PR notes.
    if not body.startswith("##") and "\n\n---\n\n" in body:
        body = body.rsplit("\n\n---\n\n", 1)[1].strip()
    if body.startswith("**Stacked PR**") and "\n\n" in body:
        body = body.split("\n\n", 1)[1].strip()
    # These are the internal service's replies, including failures/advice. Public
    # Actions suggestions carry a different marker and must not satisfy this gate.
    patterns = (
        r"GitLab MR \[!\d+\].*?(?:pipeline|trigger failed)",
        r"\*\*AI[- ](?:found|suggested|recommends|could not).*?No CI pipeline was started",
        r"\*\*Automatic target selection unavailable\.\*\*",
        r"The PR changed during target selection\. No CI pipeline was started",
        r"Invalid `/bot run` path\(s\):",
        r"Too many test paths \(",
        r"@\S+ is not authorized to trigger this CI job\.",
        r"\*\*Mirroring Failed\*\*",
        r"\*\*Warning\*\*: Failed to sync latest changes\.",
        r"\*\*Error\*\*: Failed to update branch\.",
        r"The PR head or base changed before CI launch\. No pipeline was started",
        r"\[(?:SUCCESS|FAILED|FAILURE|CANCELED)\] Pipeline \[#\d+\]",
    )
    return any(re.match(pattern, body, re.S | re.I) for pattern in patterns)


@dataclass(frozen=True)
class Result:
    conclusion: str
    reason: str
    title: str
    summary: str
    analysis: dict = None


def evidence_link(item, label):
    # Construct the link from validated IDs, not a comment-supplied URL/body.
    url = item.get("html_url", "")
    if re.fullmatch(
        r"https://github\.com/[\w.-]+/[\w.-]+/pull/\d+#issuecomment-\d+", url
    ):
        return f"[{label}]({url})"
    return f"{label} (comment #{int(item['id'])})"


def evaluate(comments, analyze):
    reply = next((item for item in reversed(comments) if is_bot_response(item)), None)
    if reply:
        return Result(
            "success",
            "bot_response_found",
            "Bot response found",
            evidence_link(reply, "Internal bot response")
            + "\n\nSource and LLM analysis skipped. Pipeline creation, result and commit correspondence are not evaluated.",
        )
    command = next(
        (item for item in reversed(comments) if is_run_command(item.get("body", ""))),
        None,
    )
    try:
        impact = analyze()
    except (TargetSelectionError, ValueError) as error:
        impact = {
            "classification": "unknown",
            "reason": str(error),
            "evidence_paths": [],
            "targets": [],
            "case_gap": "",
        }
    kind = impact["classification"]
    details = _public_prose(
        impact["reason"],
        (os.getenv("GH_TOKEN"), os.getenv("INFERENCE_API_KEY")),
        limit=1500,
    )
    if impact.get("evidence_paths"):
        details += "\n\nEvidence: " + ", ".join(
            html.escape(path) for path in impact["evidence_paths"]
        )
    if impact.get("targets"):
        details += "\n\nSuggested tests: " + ", ".join(
            html.escape(path) for path in impact["targets"]
        )
    if impact.get("missing_patch_paths"):
        details += "\n\nPatch content unavailable (informational): " + ", ".join(
            html.escape(path) for path in impact["missing_patch_paths"]
        )
        count = impact["missing_patch_count"]
        if count > len(impact["missing_patch_paths"]):
            details += f" (showing the first 30 of {count} files)"
    if impact.get("case_gap"):
        details += "\n\nTest selection gap: " + _public_prose(
            impact["case_gap"], limit=1000
        )
    if len(impact.get("analysis_attempts", [])) > 1:
        details += "\n\nAnalysis attempts:\n" + "\n".join(
            "- " + item["effort"] + ": " + item.get("error", "succeeded")
            for item in impact["analysis_attempts"]
        )
    if kind in {"cpu_only", "docs_only", "experimental_only"}:
        return Result("success", kind, "Bot run not required", details, impact)
    if kind == "unknown":
        title, reason = "Unable to determine GPU impact", "unknown"
        details += "\n\nNo CPU-only exemption could be established. Comment `/bot run` and wait for the internal bot response."
    elif command:
        title, reason = "Waiting for flashinfer-bot", "waiting_for_bot"
    else:
        title, reason = "Bot run required", "command_missing"
        details += "\n\nGPU-related changes: comment `/bot run` and wait for `flashinfer-bot` to reply."
    if command:
        details += (
            "\n\nDetected "
            + evidence_link(command, "/bot run command")
            + "; waiting for the bot response."
        )
    return Result("failure", reason, title, details, impact)


class Client(GitHub):
    def get_pr_files_at(self, pr, base_sha):
        head_sha = pr["head"]["sha"]
        if not all(
            isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{40}", sha)
            for sha in (base_sha, head_sha)
        ):
            raise TargetSelectionError("Invalid pinned comparison commits")
        comparison = self.request(
            f"repos/{self.repository}/compare/{base_sha}...{head_sha}?per_page=1"
        )
        files = comparison.get("files") if isinstance(comparison, dict) else None
        if (
            not isinstance(files, list)
            or len(files) != pr.get("changed_files")
            or any(
                not isinstance(item, dict) or not isinstance(item.get("filename"), str)
                for item in files
            )
            or not isinstance(comparison.get("base_commit"), dict)
            or comparison.get("base_commit", {}).get("sha") != base_sha
        ):
            # GitHub's immutable compare response contains at most 300 files.
            # Never use a partial list to grant a documentation/CPU exemption.
            raise TargetSelectionError("Pinned changed-file comparison is incomplete")
        return files

    def request(self, endpoint, method="GET", data=None):
        args = ["gh", "api", "--method", method, endpoint]
        if data is not None:
            args += ["--input", "-"]
        try:
            response = subprocess.run(
                args,
                input=json.dumps(data) if data is not None else None,
                capture_output=True,
                text=True,
                check=True,
                timeout=_remaining(self.deadline, 45),
            )
            if len(response.stdout) > 10_000_000:
                raise TargetSelectionError(
                    "GitHub response exceeds the evidence size limit"
                )
            return json.loads(response.stdout) if response.stdout.strip() else None
        except (subprocess.SubprocessError, OSError, ValueError) as error:
            raise TargetSelectionError(
                f"GitHub {method} request failed; check permissions or rerun the check"
            ) from error

    def pages(self, endpoint, field=None):
        items = []
        for page in range(1, 101):
            data = self.request(
                f"{endpoint}{'&' if '?' in endpoint else '?'}per_page=100&page={page}"
            )
            batch = data[field] if field else data
            items.extend(batch)
            if len(batch) < 100:
                return items
        raise TargetSelectionError(
            "GitHub comment/check history exceeds 100 pages; evidence is incomplete"
        )

    def comments(self, number):
        return self.pages(f"repos/{self.repository}/issues/{number}/comments")

    def checks(self, sha):
        return self.pages(
            f"repos/{self.repository}/commits/{sha}/check-runs?filter=all", "check_runs"
        )

    def begin(self, pr, check_id=None):
        data = {
            "name": CHECK_NAME,
            "status": "in_progress",
            "external_id": f"{VERSION}:{pr['number']}",
            "output": {
                "title": "Checking bot run request",
                "summary": "Reading existing bot responses and exemption evidence.",
            },
        }
        if check_id:
            return self.request(
                f"repos/{self.repository}/check-runs/{int(check_id)}", "PATCH", data
            )["id"]
        data["head_sha"] = pr["head"]["sha"]
        return self.request(f"repos/{self.repository}/check-runs", "POST", data)["id"]

    def publish(self, pr, result, check_id=None, base_sha=None):
        data = {
            "name": CHECK_NAME,
            "status": "completed",
            "conclusion": result.conclusion,
            "external_id": f"{VERSION}:{pr['number']}",
            "output": {
                "title": result.title,
                "summary": result.summary,
                "text": json.dumps(
                    {
                        "reason": result.reason,
                        "analysis": result.analysis,
                        "base_sha": base_sha,
                        "version": analysis_version(),
                    }
                ),
            },
        }
        run = os.getenv("GITHUB_RUN_ID")
        if run:
            data["details_url"] = (
                f"https://github.com/{self.repository}/actions/runs/{int(run)}"
            )
        if check_id:
            return self.request(
                f"repos/{self.repository}/check-runs/{int(check_id)}", "PATCH", data
            )
        data["head_sha"] = pr["head"]["sha"]
        return self.request(f"repos/{self.repository}/check-runs", "POST", data)


def analysis_version():
    import hashlib

    digest = hashlib.sha256()
    for name in (
        "ai_test_scope.py",
        "gpu_impact.py",
        "scope_evidence.py",
        "check_bot_run_request.py",
    ):
        digest.update(Path(__file__).with_name(name).read_bytes())
    digest.update(os.getenv("CI_BOT_TARGET_MODEL", "default").encode())
    return digest.hexdigest()


def cached_analysis(checks, pr, base_sha):
    for item in sorted(checks, key=lambda check: check["id"], reverse=True):
        if (
            item.get("app", {}).get("slug") != "github-actions"
            or item.get("external_id") != f"{VERSION}:{pr['number']}"
        ):
            continue
        try:
            stored = json.loads(item["output"]["text"])
            if (
                stored["base_sha"] == base_sha
                and stored["version"] == analysis_version()
                and isinstance(stored.get("analysis"), dict)
                and stored["analysis"].get("classification")
                in {"docs_only", "cpu_only", "gpu_required", "experimental_only"}
            ):
                return stored["analysis"]
        except (KeyError, TypeError, ValueError):
            continue
    return None


def docs_only(changed, expected_count):
    def is_doc(path):
        return isinstance(path, str) and path.endswith((".md", ".rst"))

    return (
        bool(changed)
        and len(changed) == expected_count
        and all(
            is_doc(item.get("filename"))
            and (not item.get("previous_filename") or is_doc(item["previous_filename"]))
            for item in changed
        )
    )


def experimental_only(changed, expected_count):
    def is_experimental(path):
        return isinstance(path, str) and path.startswith(
            ("flashinfer/experimental/", "tests/experimental/")
        )

    return (
        bool(changed)
        and len(changed) == expected_count
        and all(
            is_experimental(item.get("filename"))
            and (
                not item.get("previous_filename")
                or is_experimental(item["previous_filename"])
            )
            for item in changed
        )
    )


def run_pr(client, number):
    for attempt in range(2):
        pr = copy.deepcopy(client.get_pr_details(number))
        if pr["state"] != "open" or pr["base"]["ref"] != "main":
            return None
        checks = client.checks(pr["head"]["sha"])
        owned = [
            item
            for item in checks
            if item.get("app", {}).get("slug") == "github-actions"
            and item.get("external_id") == f"{VERSION}:{number}"
            and item.get("name") == CHECK_NAME
        ]
        existing_id = max((item["id"] for item in owned), default=None)
        existing_id = client.begin(pr, existing_id)
        base_sha = None
        retry = False
        try:
            base_sha = client.get_current_base_sha(pr)
            comments = client.comments(number)
            saved = cached_analysis(owned, pr, base_sha)

            def analyze():
                if saved:
                    return saved
                changed = client.get_pr_files_at(pr, base_sha)
                kind = None
                if docs_only(changed, pr.get("changed_files")):
                    kind = "docs_only"
                elif experimental_only(changed, pr.get("changed_files")):
                    kind = "experimental_only"
                if kind:
                    return {
                        "classification": kind,
                        "reason": (
                            "Only Markdown/RST documentation changed."
                            if kind == "docs_only"
                            else "Experimental-only changes are excluded from stable checks by the containment policy."
                        ),
                        "evidence_paths": [item["filename"] for item in changed],
                        "case_gap": "",
                        "targets": [],
                    }
                root = Path(__file__).resolve().parents[3]
                return analyze_gpu_impact(
                    PinnedPRSource(client, pr, base_sha, changed),
                    pr,
                    lambda path: (root / path).read_text(),
                )

            result = evaluate(comments, analyze)
            latest = client.get_pr_details(number)
            fresh = (
                latest["head"]["sha"] == pr["head"]["sha"]
                and latest["base"]["ref"] == pr["base"]["ref"]
                and latest["state"] == "open"
            )
            if not fresh or client.get_current_base_sha(latest) != base_sha:
                retry = latest["state"] == "open" and latest["base"]["ref"] == "main"
                result = Result(
                    "failure",
                    "stale_evaluation",
                    "PR changed during evaluation",
                    "The captured head/base changed. No stale success was published. "
                    + (
                        "Re-evaluating the current snapshot once."
                        if attempt == 0 and retry
                        else "Rerun this check against the current snapshot."
                    ),
                )
        except TargetSelectionError as error:
            result = Result(
                "failure",
                "evidence_error",
                "Unable to read bot request evidence",
                _public_prose(str(error)),
            )
        # A read can exhaust the evidence deadline. Give completion its own
        # bounded budget so an already-started Check is not left in progress.
        deadline = client.deadline
        client.deadline = time.monotonic() + 45
        try:
            client.publish(pr, result, existing_id, base_sha)
        finally:
            client.deadline = deadline
        summary = f"## {CHECK_NAME} — PR #{number}\n\n**{result.title}**\n\n{result.summary}\n"
        if os.getenv("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
                stream.write(summary)
        print(f"PR #{number}: {result.reason} ({result.conclusion})")
        if not retry or attempt == 1:
            return result
        try:
            _remaining(client.deadline, 1)
        except TargetSelectionError:
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", type=int, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo) or args.pr <= 0:
        parser.error("Invalid repository or PR")
    result = run_pr(Client(args.repo, timeout=540), args.pr)
    return 1 if result and result.conclusion == "failure" else 0


if __name__ == "__main__":
    sys.exit(main())
