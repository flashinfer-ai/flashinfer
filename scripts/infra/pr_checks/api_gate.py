#!/usr/bin/env python3
"""Trusted API gate and comment commands. Never load code from the PR head."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Direct workflow invocation must import helpers from this trusted checkout.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pr_checks.api_report import publish_report
from pr_checks.api_codeowners import owners_for_paths
from pr_checks.github_api import GitHub, GitHubError, warning as emit_warning
from pr_checks.git_compare import resolve_pr_comparison

GATE = "API compatibility"


def parse_command(body: str) -> str | None:
    match = re.fullmatch(
        r"@flashinfer-bot[ \t]+API[ \t]+BREAK[ \t]+(?:(expected)(?:[ \t]*,[ \t]*reason:[ \t]*(\S[^\r\n]*))?|(revoke))",
        str(body).strip(),
        re.IGNORECASE,
    )
    return (match[1] or match[3]).lower() if match else None


def can_approve(role: str, actor: str, author: str, kind: str) -> bool:
    return (
        role in ("admin", "maintain")
        and kind == "User"
        and actor.lower() != author.lower()
    )


def gate_state(report: dict, expected: bool) -> str:
    if (
        not isinstance(report, dict)
        or not isinstance(report.get("findings"), list)
        or report.get("execution_status") not in ("COMPLETE", "FAILED")
    ):
        raise ValueError("Invalid API report")
    if report["execution_status"] != "COMPLETE":
        return "error"
    levels = {
        "BREAKING": "error",
        "LIKELY_BREAKING": "warning",
        "NON_BREAKING": "notice",
    }
    for finding in report["findings"]:
        if (
            finding.get("classification") not in levels
            or finding.get("level") != levels[finding["classification"]]
        ):
            raise ValueError("Invalid API classification")
    return (
        "failure"
        if not expected
        and any(f["classification"] == "BREAKING" for f in report["findings"])
        else "success"
    )


def command_id(record: dict) -> int:
    match = re.fullmatch(
        r"(?:expected|revoke):(\d+)(?::[a-f0-9]{24})?", record.get("description", "")
    )
    return int(match[1]) if match else 0


def approved(gate: Gate, pr: dict, records: list[dict]) -> bool:
    items = sorted(
        (s for s in records if s["context"] == gate.exception_context(pr)),
        key=command_id,
        reverse=True,
    )
    for item in items:
        if gate.provenance(item, "issue_comment", "ci-bot-commands.yml"):
            # Invalidating the latest command must never revive an older one.
            return gate.valid_exception(pr, item)
    return False


@dataclass(frozen=True)
class Context:
    repository: str
    run_id: str
    payload: dict
    server_url: str = "https://github.com"

    @property
    def root(self) -> str:
        return f"repos/{self.repository}"

    @property
    def run_url(self) -> str:
        return f"{self.server_url}/{self.repository}/actions/runs/{self.run_id}"

    @classmethod
    def from_env(cls) -> Context:
        return cls(
            os.environ["GITHUB_REPOSITORY"],
            os.environ["GITHUB_RUN_ID"],
            json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()),
            os.environ.get("GITHUB_SERVER_URL", "https://github.com"),
        )


class Gate:
    def __init__(self, api, context: Context, *, warning=emit_warning):
        self.api, self.context, self.warning = api, context, warning
        self.root = context.root

    def pull(self, number: int) -> dict:
        return self.api.request("GET", f"{self.root}/pulls/{number}")

    @staticmethod
    def scan_context(pr: dict) -> str:
        return f"api-scan/pr-{pr['number']}"

    @staticmethod
    def exception_context(pr: dict) -> str:
        return f"api-break-exception/pr-{pr['number']}"

    def status(self, pr: dict, name: str, state: str, description: str) -> None:
        self.api.request(
            "POST",
            f"{self.root}/statuses/{pr['head']['sha']}",
            {
                "context": name,
                "state": state,
                "description": description,
                "target_url": self.context.run_url,
            },
        )

    def statuses(self, pr: dict) -> list[dict]:
        return self.api.paginate(f"{self.root}/commits/{pr['head']['sha']}/statuses")

    def current_head(self, pr: dict) -> bool:
        latest = self.pull(pr["number"])
        return (
            latest["state"] == "open"
            and latest["head"]["sha"] == pr["head"]["sha"]
            and latest["base"]["sha"] == pr["base"]["sha"]
            and (
                "_api_generation" not in pr
                or self.generation(latest) == pr["_api_generation"]
            )
        )

    def generation(self, pr: dict) -> str:
        # Commit statuses survive force-pushes. Trusted timeline event IDs
        # distinguish returning to an old SHA from rescanning the same head.
        events = self.api.paginate(f"{self.root}/issues/{pr['number']}/timeline")
        updates = sorted(
            (e["event"], e["id"])
            for e in events
            if e.get("event")
            in {
                "head_ref_force_pushed",
                "head_ref_deleted",
                "head_ref_restored",
                "base_ref_changed",
                "base_ref_force_pushed",
            }
        )
        event = json.dumps(updates, separators=(",", ":")) if updates else "0"
        stamp = f"{pr['head']['sha']}:{pr['base']['sha']}:{event}"
        return hashlib.sha256(stamp.encode()).hexdigest()[:24]

    def bind(self, pr: dict) -> dict:
        return {**pr, "_api_generation": self.generation(pr)}

    def revision(self, pr: dict) -> str:
        return pr.get("_api_generation") or self.generation(pr)

    def scan_description(self, pr: dict) -> str:
        return f"complete:{self.revision(pr)}"

    def decision_description(self, pr: dict, command: str, id: int) -> str:
        return f"{command}:{id}:{self.revision(pr)}"

    def shared_head(self, pr: dict) -> bool:
        return any(
            other["number"] != pr["number"]
            and other["head"]["sha"] == pr["head"]["sha"]
            for other in self.api.paginate(f"{self.root}/pulls?state=open")
        )

    def provenance(self, item: dict, event: str, workflow: str) -> bool:
        if item.get("creator", {}).get("login") != "github-actions[bot]":
            return False
        prefix = f"{self.context.server_url}/{self.context.repository}/actions/runs/"
        target = item.get("target_url") or ""
        if not target.startswith(prefix) or not re.fullmatch(
            r"[1-9][0-9]*", target[len(prefix) :]
        ):
            return False
        run = self.api.request(
            "GET", f"{self.root}/actions/runs/{target[len(prefix) :]}"
        )
        return (
            run.get("event") == event
            and (run.get("path") or "").split("@")[0] == f".github/workflows/{workflow}"
            and run.get("repository", {}).get("full_name") == self.context.repository
        )

    def trusted_scan(self, record: dict) -> bool:
        return self.provenance(
            record, "pull_request_target", "pr-api-compatibility.yml"
        ) or self.provenance(record, "issue_comment", "ci-bot-commands.yml")

    def command_permission(self, pr: dict, comment: dict) -> bool:
        user = comment["user"]
        permission = self.api.request(
            "GET", f"{self.root}/collaborators/{user['login']}/permission"
        )
        return can_approve(
            permission.get("role_name"),
            user["login"],
            pr["user"]["login"],
            user["type"],
        )

    def maintainer_hints(self, pr: dict, source: str, paths: list[str]) -> list[str]:
        result = []
        author = pr["user"]["login"]
        for owner in owners_for_paths(source, paths):
            if "/" in owner or owner.lower() == author.lower():
                continue
            try:
                permission = self.api.request(
                    "GET", f"{self.root}/collaborators/{owner}/permission"
                )
                if can_approve(
                    permission.get("role_name"),
                    owner,
                    author,
                    (permission.get("user") or {}).get("type"),
                ):
                    result.append(owner)
            except Exception as error:
                self.warning(f"Omitting unverified code owner @{owner}: {error}")
        return result

    def exception_comment(self, pr: dict, item: dict) -> dict | None:
        if (
            item.get("context") != self.exception_context(pr)
            or not re.fullmatch(
                r"(?:expected|revoke):\d+:[a-f0-9]{24}", item.get("description", "")
            )
            or not self.provenance(item, "issue_comment", "ci-bot-commands.yml")
        ):
            return None
        if item["description"].rsplit(":", 1)[-1] != self.revision(pr):
            return None
        id = command_id(item)
        try:
            comment = self.api.request("GET", f"{self.root}/issues/comments/{id}")
        except GitHubError as error:
            if error.status == 404:
                return None
            raise
        url = f"{self.context.server_url}/{self.context.repository}/pull/{pr['number']}#issuecomment-{id}"
        if (
            comment["html_url"] != url
            or comment["updated_at"] != comment["created_at"]
            or parse_command(comment["body"]) != item["description"].split(":")[0]
            or not self.command_permission(pr, comment)
        ):
            return None
        return comment

    def valid_exception(self, pr: dict, item: dict) -> bool:
        return (
            item.get("state") == "success"
            and item.get("description", "").startswith("expected:")
            and self.exception_comment(pr, item) is not None
        )

    @staticmethod
    def write_report(report: dict) -> None:
        Path("reports").mkdir(exist_ok=True)
        Path("reports/public-api.json").write_text(json.dumps(report, indent=2) + "\n")

    def scan(self, pr: dict) -> dict:
        if any(
            not re.fullmatch(r"[a-f0-9]{40}", pr[side]["sha"])
            for side in ("base", "head")
        ):
            raise ValueError("Invalid PR revisions")
        number = pr["number"]
        if type(number) is not int or number <= 0:
            raise ValueError("Invalid PR number")
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        # Inspect GitHub's test merge as data. The checkout stays trusted.
        subprocess.run(
            [
                "git",
                "fetch",
                "--no-tags",
                "origin",
                f"refs/pull/{number}/merge",
            ],
            env=env,
            check=True,
            timeout=180,
            capture_output=True,
        )
        base, head = resolve_pr_comparison("FETCH_HEAD", pr["head"]["sha"], env=env)
        # The executable path belongs to the trusted checkout, never a PR tree.
        scanner = Path(__file__).resolve().parents[1] / "check_pr_api_diff.py"
        with tempfile.TemporaryDirectory(
            prefix="api-scan-", dir=os.environ.get("RUNNER_TEMP")
        ) as directory:
            output = Path(directory) / "public-api.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(scanner),
                    "--base",
                    base,
                    "--head",
                    head,
                    "--fail-on",
                    "error",
                    "--report-json",
                    str(output),
                ],
                env=env,
                timeout=180,
                capture_output=True,
                text=True,
            )
            if result.returncode not in (0, 1, 2) or not output.exists():
                raise RuntimeError(
                    f"Trusted API scanner failed ({result.returncode}): {result.stderr[-2000:]}"
                )
            report = json.loads(output.read_text())
        if report["head"] != pr["head"]["sha"]:
            raise ValueError("API report head mismatch")
        self.write_report(report)
        gate_state(report, False)
        return report

    def publish(self, pr: dict, report: dict, **options):
        owner, repo = self.context.repository.split("/")
        return publish_report(
            self.api,
            pr,
            report,
            owner=owner,
            repo=repo,
            run_url=self.context.run_url,
            warning=self.warning,
            **options,
        )

    def evaluate(self, pr: dict, report: dict) -> bool:
        expected = approved(self, pr, self.statuses(pr))
        state = gate_state(report, expected)
        blocked_reason = ""
        if state == "success" and self.shared_head(pr):
            state = "failure"
            blocked_reason = "Head commit is shared by multiple open PRs; push a distinct commit before retrying."
        if not self.current_head(pr):
            return False
        description = blocked_reason or (
            "API scan incomplete"
            if state == "error"
            else "API breaks accepted for this head"
            if expected
            else "Breaking API changes require maintainer confirmation"
            if state == "failure"
            else "No blocking API changes"
        )
        self.status(pr, GATE, state, description)
        # Repair a success if a peer appeared between the preflight and write.
        # GitHub offers no atomic PR-scoped commit-status transaction.
        if state == "success" and self.shared_head(pr):
            state = "failure"
            blocked_reason = "Head commit is shared by multiple open PRs; push a distinct commit before retrying."
            self.status(pr, GATE, state, blocked_reason)
        if state == "success" and not self.current_head(pr):
            latest = self.pull(pr["number"])
            if latest["head"]["sha"] != pr["head"]["sha"]:
                return False
            state = "failure"
            blocked_reason = "PR revision changed during gate publication; a fresh API scan is required."
            self.status(pr, GATE, state, blocked_reason)
        maintainers = []
        try:
            revision = pr["base"]["sha"]
            if not re.fullmatch(r"[a-f0-9]{40}", revision):
                raise ValueError("Invalid base revision")
            owners = subprocess.run(
                ["git", "show", f"{revision}:.github/CODEOWNERS"],
                cwd=Path(__file__).resolve().parents[3],
                env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
                capture_output=True,
                text=True,
                timeout=30,
            )
            if owners.returncode == 0:
                paths = [
                    f["path"]
                    for f in report["findings"]
                    if f["classification"] == "BREAKING"
                ]
                maintainers = self.maintainer_hints(pr, owners.stdout, paths)
        except Exception as error:
            self.warning(f"Code owners could not be resolved: {error}")
        try:
            report["comment_delivery"] = self.publish(
                pr,
                report,
                expected=expected and state == "success",
                blocked_reason=blocked_reason,
                maintainers=maintainers,
            )
            self.write_report(report)
        except Exception as error:
            self.warning(f"API comments could not be updated: {error}")
        return state == "success" and self.current_head(pr)

    def evaluate_safely(self, pr: dict, report: dict) -> bool:
        try:
            return self.evaluate(pr, report)
        except Exception as error:
            self.warning(f"API gate evaluation failed: {error}")
            # A reporting/API failure does not erase completed scan evidence.
            if self.current_head(pr):
                self.status(
                    pr, GATE, "error", "API gate evaluation failed; retry the workflow"
                )
            return False

    def command_scan(self, pr: dict, scan) -> dict | None:
        self.status(pr, GATE, "pending", "Re-evaluating API compatibility")
        try:
            report = scan(pr)
            if gate_state(report, True) == "error":
                raise RuntimeError("API rescan did not complete")
        except Exception as error:
            self.warning(str(error))
            if not self.current_head(pr):
                return None
            self.status(pr, self.scan_context(pr), "error", "Trusted API rescan failed")
            self.status(pr, GATE, "error", "API rescan failed; inspect workflow logs")
            failed = {
                "execution_status": "FAILED",
                "findings": [],
                "error": str(error),
                "base": pr["base"]["sha"],
                "head": pr["head"]["sha"],
            }
            self.write_report(failed)
            try:
                self.publish(pr, failed)
            except Exception as publication_error:
                self.warning(
                    f"API failure summary could not be updated: {publication_error}"
                )
            return None
        if not self.current_head(pr):
            return None
        self.status(pr, self.scan_context(pr), "success", self.scan_description(pr))
        return report

    def run_scan(self) -> None:
        pr = self.bind(self.pull(self.context.payload["pull_request"]["number"]))
        if not self.current_head(pr):
            return
        self.status(pr, GATE, "pending", "Scanning public API compatibility")
        try:
            report = self.scan(pr)
            if not self.current_head(pr):
                return
            complete = report["execution_status"] == "COMPLETE"
            self.status(
                pr,
                self.scan_context(pr),
                "success" if complete else "error",
                self.scan_description(pr)
                if complete
                else "Trusted API scan incomplete",
            )
        except Exception as error:
            self.warning(str(error))
            if self.current_head(pr):
                self.status(
                    pr, self.scan_context(pr), "error", "Trusted API scan failed"
                )
                self.status(pr, GATE, "error", "API scan failed; inspect workflow logs")
            raise
        self.evaluate_safely(pr, report)

    def changed_approval(self, comment: dict, scan) -> None:
        pr = self.pull(self.context.payload["issue"]["number"])
        if pr["state"] != "open":
            return
        records = sorted(
            (
                s
                for s in self.statuses(pr)
                if s["context"] == self.exception_context(pr)
            ),
            key=command_id,
            reverse=True,
        )
        latest = next(
            (
                s
                for s in records
                if self.provenance(s, "issue_comment", "ci-bot-commands.yml")
            ),
            None,
        )
        if latest is None or command_id(latest) != comment["id"]:
            return
        pr = self.bind(pr)
        report = self.command_scan(pr, scan)
        if report is not None:
            self.evaluate_safely(pr, report)

    def handle_command(self, *, scan=None) -> None:
        comment = self.context.payload["comment"]
        scan = scan or self.scan
        if self.context.payload.get("action", "created") in ("edited", "deleted"):
            self.changed_approval(comment, scan)
            return
        command = parse_command(comment["body"])
        if not command:
            return

        def react(content):
            self.api.request(
                "POST",
                f"{self.root}/issues/comments/{comment['id']}/reactions",
                {"content": content},
            )

        react("eyes")
        pr = self.pull(self.context.payload["issue"]["number"])
        if (
            not self.command_permission(pr, comment)
            or pr["state"] != "open"
            or comment["updated_at"] != comment["created_at"]
        ):
            return
        pr = self.bind(pr)
        records = self.statuses(pr)
        scans = sorted(
            (s for s in records if s["context"] == self.scan_context(pr)),
            key=lambda s: s["id"],
            reverse=True,
        )
        healthy = bool(
            scans
            and scans[0]["state"] == "success"
            and scans[0].get("description") == self.scan_description(pr)
            and self.trusted_scan(scans[0])
        )
        resumed = (
            any(
                s["context"] == self.exception_context(pr)
                and command_id(s) == comment["id"]
                and self.valid_exception(pr, s)
                for s in records
            )
            if command == "expected"
            else False
        )

        def predates(record):
            return datetime.fromisoformat(
                record["created_at"].replace("Z", "+00:00")
            ) <= datetime.fromisoformat(comment["created_at"].replace("Z", "+00:00"))

        if command == "expected" and (
            not healthy
            or not (
                resumed
                or any(
                    s["state"] == "success"
                    and s.get("description") == self.scan_description(pr)
                    and predates(s)
                    and self.trusted_scan(s)
                    for s in scans
                )
            )
        ):
            return
        live = self.api.request("GET", f"{self.root}/issues/comments/{comment['id']}")
        if (
            live["body"] != comment["body"]
            or live["updated_at"] != comment["updated_at"]
            or not self.current_head(pr)
        ):
            return
        for record in records:
            if (
                record["context"] == self.exception_context(pr)
                and command_id(record) > comment["id"]
                and self.provenance(record, "issue_comment", "ci-bot-commands.yml")
            ):
                return
        if command == "revoke":
            self.status(
                pr,
                self.exception_context(pr),
                "failure",
                self.decision_description(pr, command, comment["id"]),
            )
        report = self.command_scan(pr, scan)
        if report is None:
            return
        if command == "expected" and not resumed:
            self.status(
                pr,
                self.exception_context(pr),
                "success",
                self.decision_description(pr, command, comment["id"]),
            )
        if self.evaluate_safely(pr, report) and command == "expected":
            react("+1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("scan", "command"))
    mode = parser.parse_args().mode
    context = Context.from_env()
    api = GitHub(
        os.environ["GH_TOKEN"],
        os.environ.get("GITHUB_API_URL", "https://api.github.com"),
    )
    gate = Gate(api, context)
    if mode == "scan":
        gate.run_scan()
    else:
        gate.handle_command()


if __name__ == "__main__":
    main()
