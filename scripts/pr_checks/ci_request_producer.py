"""Trusted default-branch adapter for exact CI request receipts.

Only this adapter performs provider writes. Tests inject an API; imports do not
contact GitHub. Persist an attempt before dispatch, and recover ambiguous sends
by an immutable provider request key instead of reposting.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from urllib.parse import quote
from pathlib import Path

import ci_request_protocol as protocol

WORKFLOW = "pr-test.yml"
WORKFLOW_PATH = ".github/workflows/" + WORKFLOW
JOURNAL_MARKER = "watcher-ci-dispatch-journal-v1"
REQUEST_MARKER = "watcher-ci-request-v1"
BOT = "flashinfer-bot"


class API:
    def __init__(self, repository, token=None):
        self.repository = repository
        self.token = token

    def call(self, method, path, data=None):
        if not path.startswith("repos/" + self.repository + "/"):
            raise ValueError("API request must stay in the configured repository")
        command = [
            "gh",
            "api",
            "--method",
            method,
            "-H",
            "X-GitHub-Api-Version: 2022-11-28",
            path,
        ]
        if data is not None:
            command += ["--input", "-"]
        result = subprocess.run(
            command,
            input=json.dumps(data) if data is not None else None,
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
            env={**os.environ, "GH_TOKEN": self.token} if self.token else None,
        )
        return json.loads(result.stdout)

    def get(self, path):
        return self.call("GET", path)

    def post(self, path, data):
        return self.call("POST", path, data)

    def patch(self, path, data):
        return self.call("PATCH", path, data)


def decode_marker(body, marker):
    matches = re.findall(
        r"<!--\s*" + re.escape(marker) + r"\s+(\{.*?\})\s*-->", body or "", re.S
    )
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("Ambiguous producer marker")
    value = json.loads(matches[0])
    if not isinstance(value, dict):
        raise ValueError("Producer marker must be an object")
    return value


def _scope_metadata(event, scope):
    metadata = decode_marker(event["comment"]["body"], REQUEST_MARKER)
    if metadata is None:
        if "watcher-ci-request" in event["comment"]["body"]:
            raise ValueError("Malformed watcher request marker")
        return None
    if (
        set(metadata) != {"requestId", "headSha", "provider", "scope"}
        or metadata["provider"] != "github"
        or not re.fullmatch("[0-9a-f]{64}", metadata["requestId"])
        or not re.fullmatch("[0-9a-f]{40}", metadata["headSha"])
        or protocol.normalize_scope(metadata["scope"])
        != protocol.normalize_scope(scope)
    ):
        raise ValueError("Watcher request metadata does not match parsed command")
    return metadata


def matching_checks(api, repository, request):
    found = []
    for page in range(1, 11):
        value = api.get(
            f"repos/{repository}/commits/{request['headSha']}/check-runs?filter=all&per_page=100&page={page}"
        )
        found.extend(value["check_runs"])
        if len(found) == value["total_count"]:
            return found
        if not value["check_runs"]:
            break
    raise ValueError("Incomplete current-head checks inventory")


def summary_check(api, repository, request, conclusion=None, existing_only=False):
    external = f"fi-ci/{request['triggerCommentId']}/summary"
    matches = [
        item
        for item in matching_checks(api, repository, request)
        if item.get("external_id") == external
        and (item.get("app") or {}).get("slug") == "github-actions"
    ]
    if len(matches) > 1:
        raise ValueError("Duplicate required Summary identity")
    if not matches and existing_only:
        return
    data = {
        "name": "Test Results Summary",
        "external_id": external,
        "status": "completed" if conclusion else "in_progress",
        "details_url": request["triggerCommentUrl"],
        "output": {
            "title": "Requested CI " + ("not submitted" if conclusion else "pending"),
            "summary": "Request-bound provider lifecycle; no PR logs are interpreted.",
        },
    }
    if conclusion:
        data["conclusion"] = conclusion
    if matches:
        return api.patch(f"repos/{repository}/check-runs/{matches[0]['id']}", data)
    return api.post(
        f"repos/{repository}/check-runs", {**data, "head_sha": request["headSha"]}
    )


class Producer:
    def __init__(
        self,
        api,
        repository,
        pr_number,
        comment_id,
        comment_url,
        scope,
        status_api=None,
    ):
        self.api, self.repository = api, repository.lower()
        self.pr_number, self.comment_id, self.comment_url = (
            pr_number,
            comment_id,
            comment_url,
        )
        self.scope = protocol.normalize_scope(scope)
        self.journal_id = None
        self.status_api = status_api
        self.request_metadata = None

    @property
    def issue(self):
        return f"repos/{self.repository}/issues/{self.pr_number}"

    def find(self):
        found = []
        for page in range(1, 11):
            rows = self.api.get(self.issue + f"/comments?per_page=100&page={page}")
            if not isinstance(rows, list):
                raise ValueError("Incomplete journal inventory")
            for row in rows:
                if (row.get("user") or {}).get("login") != BOT:
                    continue
                journal = decode_marker(row.get("body"), JOURNAL_MARKER)
                if (
                    journal is None
                    or (journal.get("request") or {}).get("triggerCommentId")
                    != self.comment_id
                ):
                    continue
                journal = protocol._journal(journal)
                request = journal["request"]
                if (
                    request["repository"],
                    request["prNumber"],
                    request["triggerCommentUrl"],
                    request["scope"],
                ) != (
                    self.repository,
                    self.pr_number,
                    self.comment_url.lower(),
                    self.scope,
                ):
                    raise ValueError(
                        "Existing journal does not match the original request"
                    )
                found.append((row["id"], journal))
            if len(rows) < 100:
                break
        else:
            raise ValueError("Journal inventory exceeds bounded pagination")
        if len(found) > 1:
            raise ValueError("More than one journal for this command; do not dispatch")
        if found:
            self.journal_id, journal = found[0]
            return journal
        return None

    def save(self, journal):
        journal = protocol._journal(journal)
        body = (
            "CI request: "
            + journal["state"]
            + ".\n\n<!-- "
            + JOURNAL_MARKER
            + " "
            + json.dumps(journal, sort_keys=True, separators=(",", ":"))
            + " -->"
        )
        if journal["state"] == "no-send" and self.request_metadata is not None:
            request = journal["request"]
            if (
                request["headSha"] != self.request_metadata["headSha"]
                or journal["dispatchAttempted"]
            ):
                raise ValueError("No-send proof does not match the frozen request")
            receipt = {
                "schemaVersion": 1,
                "requestId": self.request_metadata["requestId"],
                "repository": request["repository"],
                "prNumber": request["prNumber"],
                "triggerCommentId": request["triggerCommentId"],
                "triggerCommentUrl": request["triggerCommentUrl"],
                "requestedHeadSha": request["headSha"],
                "provider": "github",
                "reason": "pre_dispatch_rejected",
                "submissionAttempted": False,
                "rootRunIds": [],
            }
            body += (
                "\n\n<!-- watcher-ci-no-send-v1 "
                + json.dumps(receipt, sort_keys=True, separators=(",", ":"))
                + " -->"
            )
        if journal["state"] == "completed":
            body += "\n\n" + protocol.receipt_body(protocol.acknowledgment(journal))
            body += "\n\n" + "\n".join(
                f"https://github.com/{self.repository}/actions/runs/{s['rootRunId']}"
                for s in journal["slots"]
            )
        if self.journal_id is None:
            result = self.api.post(self.issue + "/comments", {"body": body})
            if type(result.get("id")) is not int:
                raise ValueError("Journal write did not return an exact comment ID")
            self.journal_id = result["id"]
        else:
            result = self.api.patch(
                f"repos/{self.repository}/issues/comments/{self.journal_id}",
                {"body": body},
            )
        if (result.get("user") or {}).get("login") != BOT or result.get("body") != body:
            raise ValueError("Journal write identity or readback differs")
        if journal["state"] == "no-send" and self.status_api is not None:
            # A pending check may have been accepted before another pre-send
            # operation failed. Retry cleanup by exact external ID, not by name.
            summary_check(
                self.status_api,
                self.repository,
                journal["request"],
                "failure",
                existing_only=True,
            )
        return journal

    def exact_roots(self, request):
        # Time narrows inventory only; the full immutable key is still the proof.
        # Without this server timestamp, mature repos exceed the API search cap.
        original = self.api.get(
            f"repos/{self.repository}/issues/comments/{self.comment_id}"
        )
        since = original.get("created_at", "")
        if (
            original.get("id") != self.comment_id
            or original.get("html_url") != self.comment_url
            or not re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", since
            )
        ):
            raise ValueError("Original provider comment timestamp is unavailable")
        created = quote(">=" + since, safe="")
        roots = []
        total = None
        seen = 0
        for page in range(1, 11):
            value = self.api.get(
                f"repos/{self.repository}/actions/workflows/{WORKFLOW}/runs?event=workflow_dispatch&per_page=100&page={page}&created={created}"
            )
            if type(value.get("total_count")) is not int or value["total_count"] > 1000:
                raise ValueError("Recovery inventory is incomplete")
            if total is not None and total != value["total_count"]:
                raise ValueError("Recovery inventory changed")
            total = value["total_count"]
            rows = value["workflow_runs"]
            seen += len(rows)
            for run in rows:
                if run.get("display_title") != request["requestKey"]:
                    continue
                if (
                    run.get("event") != "workflow_dispatch"
                    or run.get("path", "").split("@", 1)[0] != WORKFLOW_PATH
                    or (run.get("repository") or {}).get("full_name", "").lower()
                    != self.repository
                    or (run.get("actor") or {}).get("login") != BOT
                ):
                    raise ValueError("Recovery run has a different producer identity")
                roots.append(
                    {
                        "workflow_run_id": run["id"],
                        "run_url": run["url"],
                        "html_url": run["html_url"],
                    }
                )
            if seen == total:
                break
            if not rows:
                raise ValueError("Incomplete recovery pagination")
        else:
            raise ValueError("Recovery inventory exceeds bounded pagination")
        return roots

    def recover(self, journal):
        if journal["state"] in ("completed", "no-send"):
            return self.save(journal)
        if journal["state"] not in ("attempt_recorded", "uncertain"):
            return journal
        roots = self.exact_roots(journal["request"])
        if len(roots) != 1:
            # Missing or duplicate roots remain ambiguous, never a reason to POST.
            return self.save(
                protocol.mark_uncertain(
                    journal, "Exact provider root is absent or ambiguous"
                )
            )
        return self.save(
            protocol.recover(journal, journal["request"]["requestKey"], roots[0])
        )

    def dispatch(self, expected_head=None):
        journal = self.find()
        if (
            journal is not None
            and expected_head is not None
            and journal["request"]["headSha"] != expected_head
        ):
            raise ValueError("Retry changed the requested head")
        if journal is not None and journal["state"] != "planned":
            return self.recover(journal)
        if journal is None:
            pr = self.api.get(f"repos/{self.repository}/pulls/{self.pr_number}")
            head, base = pr["head"]["sha"], pr["base"]["sha"]
            requested = expected_head or head
            checkout = pr.get("merge_commit_sha") or requested
            request = protocol.build_request(
                self.repository,
                self.pr_number,
                self.comment_id,
                self.comment_url,
                requested,
                base,
                checkout,
                self.scope,
            )
            journal = protocol.plan(request)
            if (
                pr.get("state") != "open"
                or pr.get("draft")
                or head != requested
                or pr.get("mergeable") is not True
            ):
                return self.save(
                    protocol.no_send(
                        journal, "PR is not open, current, ready and mergeable"
                    )
                )
            merge = self.api.get(f"repos/{self.repository}/commits/{checkout}")
            if len(merge.get("parents", [])) != 2 or set(
                p["sha"] for p in merge.get("parents", [])
            ) != {head, base}:
                return self.save(
                    protocol.no_send(
                        journal,
                        "Immutable merge checkout does not match requested head and base",
                    )
                )
            self.save(journal)
        request = journal["request"]
        # A saved plan is immutable. Recheck the same source identity rather than
        # silently rebasing/rescoping a retry of the original command.
        if expected_head is not None and request["headSha"] != expected_head:
            raise ValueError("Retry changed the requested head")
        pr = self.api.get(f"repos/{self.repository}/pulls/{self.pr_number}")
        if (
            pr["head"]["sha"] != request["headSha"]
            or pr.get("state") != "open"
            or pr.get("draft")
        ):
            return self.save(protocol.no_send(journal, "PR changed before dispatch"))
        repo = self.api.get(f"repos/{self.repository}/")
        payload = protocol.dispatch_payload(request, repo["default_branch"])
        if self.status_api is not None:
            try:
                # The existing required Summary check must stop being green
                # before a new same-head request can start. No branch-rule change.
                summary_check(self.status_api, self.repository, request)
                self.status_api.post(
                    f"repos/{self.repository}/statuses/{request['headSha']}",
                    {
                        "context": "ci/watcher-request",
                        "state": "pending",
                        "description": f"request:{self.comment_id}: pending",
                        "target_url": self.comment_url,
                    },
                )
            except Exception:
                return self.save(
                    protocol.no_send(
                        journal, "Pending check could not be published before dispatch"
                    )
                )
        journal = self.save(
            protocol.record_attempt(journal)
        )  # Immediately before the POST.
        try:
            response = self.api.post(
                f"repos/{self.repository}/actions/workflows/{WORKFLOW}/dispatches",
                payload,
            )
            journal = protocol.record_response(journal, response)
        except Exception:
            journal = protocol.mark_uncertain(
                journal,
                "Dispatch response unavailable or not exact; no automatic resend",
            )
            self.save(journal)
            return self.recover(journal)
        return self.save(journal)


def request_from_inputs(repository, inputs):
    request = protocol.build_request(
        repository,
        int(inputs["watcher_pr_number"]),
        int(inputs["watcher_trigger_comment_id"]),
        inputs["watcher_trigger_comment_url"],
        inputs["watcher_head_sha"],
        inputs["watcher_base_sha"],
        inputs["watcher_checkout_sha"],
        inputs["watcher_scope"],
    )
    if any(inputs.get(k) != value for k, value in request["dispatchInputs"].items()):
        raise ValueError("Dispatch input identity changed")
    if any(
        inputs.get(k) not in (False, "false", "", None)
        for k in ("skip_aot", "skip_gpu")
    ):
        raise ValueError("Correlated requests cannot omit authorized CI lanes")
    return request


def validate_run(api, repository, inputs, run_id):
    request = request_from_inputs(repository, inputs)
    producer = Producer(
        api,
        repository,
        request["prNumber"],
        request["triggerCommentId"],
        request["triggerCommentUrl"],
        request["scope"],
    )
    journal = producer.find()
    if (
        journal is None
        or journal["request"] != request
        or journal["state"] not in ("attempt_recorded", "uncertain", "completed")
    ):
        raise ValueError("No matching durable dispatch attempt")
    if journal["state"] == "completed" and run_id not in [
        slot["rootRunId"] for slot in journal["slots"]
    ]:
        raise ValueError("Workflow run is not a returned root for this request")
    run = api.get(f"repos/{repository}/actions/runs/{run_id}")
    if (
        run.get("id") != run_id
        or run.get("display_title") != request["requestKey"]
        or run.get("event") != "workflow_dispatch"
        or run.get("path", "").split("@", 1)[0] != WORKFLOW_PATH
        or (run.get("repository") or {}).get("full_name", "").lower()
        != repository.lower()
        or (run.get("actor") or {}).get("login") != BOT
    ):
        raise ValueError(
            "Workflow run does not carry the exact trusted producer identity"
        )
    pr = api.get(f"repos/{repository}/pulls/{request['prNumber']}")
    if (
        pr.get("state") != "open"
        or pr.get("draft")
        or pr["head"]["sha"] != request["headSha"]
    ):
        raise ValueError("PR changed before executing requested source")
    merge = api.get(f"repos/{repository}/commits/{request['checkoutSha']}")
    if len(merge.get("parents", [])) != 2 or set(
        p["sha"] for p in merge.get("parents", [])
    ) != {request["headSha"], request["baseSha"]}:
        raise ValueError("Requested checkout ancestry is not verified")
    request = dict(request)
    request["isFork"] = (pr["head"].get("repo") or {}).get(
        "full_name", ""
    ).lower() != repository.lower()
    return request


def identify_run(event):
    run = event["workflow_run"]
    match = re.fullmatch(
        r"fi-ci/([1-9][0-9]*)/([1-9][0-9]*)/([0-9a-f]{40})/([0-9a-f]{64})",
        run.get("display_title", ""),
    )
    if (
        not match
        or run.get("event") != "workflow_dispatch"
        or run.get("path", "").split("@", 1)[0] != WORKFLOW_PATH
    ):
        raise ValueError("Not a correlated PR Test run")
    return int(match[1]), int(match[2])


def recover_event(api, repository, event):
    pr_number, comment_id = identify_run(event)
    comment = api.get(f"repos/{repository}/issues/comments/{comment_id}")
    # Scope comes from the original durable request, not mutable statuses or time.
    for page in range(1, 11):
        rows = api.get(
            f"repos/{repository}/issues/{pr_number}/comments?per_page=100&page={page}"
        )
        for row in rows:
            if (row.get("user") or {}).get("login") != BOT:
                continue
            journal = decode_marker(row.get("body"), JOURNAL_MARKER)
            if (
                journal
                and (journal.get("request") or {}).get("triggerCommentId") == comment_id
            ):
                request = protocol._journal(journal)["request"]
                producer = Producer(
                    api,
                    repository,
                    pr_number,
                    comment_id,
                    comment["html_url"],
                    request["scope"],
                )
                current = producer.find()
                if event["workflow_run"]["display_title"] != request["requestKey"]:
                    raise ValueError("Completion event has another request identity")
                return producer.recover(current)
        if len(rows) < 100:
            break
    raise ValueError("No exact producer journal for completion event")


def report_completed(api, repository, event):
    pr_number, comment_id = identify_run(event)
    comment = api.get(f"repos/{repository}/issues/comments/{comment_id}")
    journal = None
    for page in range(1, 11):
        rows = api.get(
            f"repos/{repository}/issues/{pr_number}/comments?per_page=100&page={page}"
        )
        for row in rows:
            if (row.get("user") or {}).get("login") != BOT:
                continue
            value = decode_marker(row.get("body"), JOURNAL_MARKER)
            if (
                value
                and (value.get("request") or {}).get("triggerCommentId") == comment_id
            ):
                if journal is not None:
                    raise ValueError("Ambiguous request journal")
                journal = protocol._journal(value)
        if len(rows) < 100:
            break
    else:
        raise ValueError("Incomplete journal inventory")
    if journal is None or journal["state"] != "completed":
        raise ValueError("Only a correlated completed submission can report PR checks")
    request = journal["request"]
    if (
        comment.get("id") != comment_id
        or comment.get("html_url") != request["triggerCommentUrl"]
    ):
        raise ValueError("Original comment identity changed")
    run_id = event["workflow_run"]["id"]
    if event["workflow_run"]["display_title"] != request[
        "requestKey"
    ] or run_id not in [slot["rootRunId"] for slot in journal["slots"]]:
        raise ValueError("Completion event belongs to another request")
    run = api.get(f"repos/{repository}/actions/runs/{run_id}")
    if (
        run.get("id") != run_id
        or run.get("event") != "workflow_dispatch"
        or (run.get("repository") or {}).get("full_name", "").lower()
        != repository.lower()
        or run.get("status") != "completed"
        or run.get("display_title") != request["requestKey"]
        or run.get("path", "").split("@", 1)[0] != WORKFLOW_PATH
        or (run.get("actor") or {}).get("login") != BOT
    ):
        raise ValueError("Provider run is not an exact terminal request")
    jobs = []
    for page in range(1, 11):
        result = api.get(
            f"repos/{repository}/actions/runs/{run_id}/jobs?filter=latest&per_page=100&page={page}"
        )
        jobs.extend(result["jobs"])
        if len(jobs) == result["total_count"]:
            break
        if not result["jobs"]:
            raise ValueError("Incomplete provider job inventory")
    else:
        raise ValueError("Provider job inventory exceeds bounded pagination")
    if any(job.get("status") != "completed" for job in jobs):
        raise ValueError("Cannot report nonterminal jobs as final PR checks")
    after = api.get(f"repos/{repository}/actions/runs/{run_id}")
    if any(
        run.get(key) != after.get(key)
        for key in ("status", "conclusion", "run_attempt", "updated_at")
    ):
        raise ValueError("Run changed during check reporting")
    status = api.get(f"repos/{repository}/commits/{request['headSha']}/status")
    current = next(
        (
            item
            for item in status.get("statuses", [])
            if item.get("context") == "ci/watcher-request"
        ),
        None,
    )
    if not current or not current.get("description", "").startswith(
        f"request:{comment_id}:"
    ):
        return {
            "runId": run_id,
            "headSha": request["headSha"],
            "reportedJobs": 0,
            "superseded": True,
        }
    summary_success = any(
        job.get("name") == "Test Results Summary" and job.get("conclusion") == "success"
        for job in jobs
    )
    # Cancellation before Setup must still publish a failing required summary.
    # This is a terminal root outcome, never a synthetic success.
    if not any(job.get("name") == "Test Results Summary" for job in jobs):
        jobs.append(
            {
                "id": "root-summary",
                "name": "Test Results Summary",
                "status": "completed",
                "conclusion": "failure",
                "html_url": run["html_url"],
            }
        )
    existing = []
    for page in range(1, 11):
        value = api.get(
            f"repos/{repository}/commits/{request['headSha']}/check-runs?filter=all&per_page=100&page={page}"
        )
        existing.extend(value["check_runs"])
        if len(existing) == value["total_count"]:
            break
        if not value["check_runs"]:
            raise ValueError("Incomplete existing check inventory")
    else:
        raise ValueError("Existing check inventory exceeds bounded pagination")
    # All real job names (including required matrix checks) are mirrored. Their
    # results come from GitHub, never PR-produced logs or an agent's assertion.
    for job in jobs:
        conclusion = job.get("conclusion")
        if job["name"] == "Test Results Summary" and (
            run.get("conclusion") != "success" or conclusion != "success"
        ):
            conclusion = (
                "cancelled" if run.get("conclusion") == "cancelled" else "failure"
            )
        if conclusion not in (
            "success",
            "failure",
            "neutral",
            "cancelled",
            "skipped",
            "timed_out",
            "action_required",
            "stale",
        ):
            raise ValueError("Unknown provider job conclusion")
        if conclusion == "stale":
            conclusion = "failure"  # Only GitHub itself may write stale conclusions.
        payload = {
            "name": job["name"],
            "head_sha": request["headSha"],
            "external_id": f"fi-ci/{comment_id}/summary"
            if job["name"] == "Test Results Summary"
            else f"fi-ci/{comment_id}/{run_id}/{run['run_attempt']}/{job['id']}",
            "status": "completed",
            "conclusion": conclusion,
            "details_url": job["html_url"],
            "output": {
                "title": job["name"],
                "summary": "Result copied from the linked GitHub job for the frozen request checkout; no PR logs are interpreted.",
            },
        }
        for timestamp in ("started_at", "completed_at"):
            if job.get(timestamp):
                payload[timestamp] = job[timestamp]
        matches = [
            item
            for item in existing
            if item.get("external_id") == payload["external_id"]
            and (item.get("app") or {}).get("slug") == "github-actions"
        ]
        if len(matches) > 1:
            raise ValueError("Duplicate correlated check identity")
        if matches:
            update = dict(payload)
            update.pop("head_sha")
            api.patch(f"repos/{repository}/check-runs/{matches[0]['id']}", update)
        else:
            api.post(f"repos/{repository}/check-runs", payload)
    status = api.get(f"repos/{repository}/commits/{request['headSha']}/status")
    current = next(
        (
            item
            for item in status.get("statuses", [])
            if item.get("context") == "ci/watcher-request"
        ),
        None,
    )
    # Producer/reporter jobs share PR-level concurrency. An older cancelled run
    # must not overwrite the pending status of a newer command on the same head.
    if current and current.get("description", "").startswith(f"request:{comment_id}:"):
        api.post(
            f"repos/{repository}/statuses/{request['headSha']}",
            {
                "context": "ci/watcher-request",
                "state": "success"
                if run["conclusion"] == "success" and summary_success
                else "failure",
                "description": f"request:{comment_id}: {run['conclusion']}",
                "target_url": run["html_url"],
            },
        )
    return {"runId": run_id, "headSha": request["headSha"], "reportedJobs": len(jobs)}


def report_progress(api, repository, event):
    pr_number, comment_id = identify_run(event)
    rows = []
    for page in range(1, 11):
        batch = api.get(
            f"repos/{repository}/issues/{pr_number}/comments?per_page=100&page={page}"
        )
        rows.extend(batch)
        if len(batch) < 100:
            break
    else:
        raise ValueError("Incomplete request journal inventory")
    found = []
    for row in rows:
        if (row.get("user") or {}).get("login") != BOT:
            continue
        journal = decode_marker(row.get("body"), JOURNAL_MARKER)
        if (
            journal
            and (journal.get("request") or {}).get("triggerCommentId") == comment_id
        ):
            found.append(protocol._journal(journal))
    if len(found) != 1 or found[0]["state"] != "completed":
        raise ValueError("Progress requires one complete exact-root journal")
    journal = found[0]
    request = journal["request"]
    run_id = event["workflow_run"]["id"]
    run = api.get(f"repos/{repository}/actions/runs/{run_id}")
    if (
        run_id not in [slot["rootRunId"] for slot in journal["slots"]]
        or run.get("id") != run_id
        or run.get("display_title") != request["requestKey"]
        or run.get("event") != "workflow_dispatch"
        or run.get("path", "").split("@", 1)[0] != WORKFLOW_PATH
        or (run.get("actor") or {}).get("login") != BOT
        or (run.get("repository") or {}).get("full_name", "").lower()
        != repository.lower()
    ):
        raise ValueError("Progress belongs to another provider request")
    if (
        event.get("action") not in ("prepare-rerun", "rerun-submit-failed")
        and run.get("status") == "completed"
    ):
        return {"ignored": "Run already completed"}
    status = api.get(f"repos/{repository}/commits/{request['headSha']}/status")
    current = next(
        (
            item
            for item in status.get("statuses", [])
            if item.get("context") == "ci/watcher-request"
        ),
        None,
    )
    if not current or not current.get("description", "").startswith(
        f"request:{comment_id}:"
    ):
        return {"superseded": True}
    failed_send = event.get("action") == "rerun-submit-failed"
    summary_check(api, repository, request, "failure" if failed_send else None)
    api.post(
        f"repos/{repository}/statuses/{request['headSha']}",
        {
            "context": "ci/watcher-request",
            "state": "failure" if failed_send else "pending",
            "description": f"request:{comment_id}: "
            + ("rerun response unavailable" if failed_send else "pending"),
            "target_url": run["html_url"],
        },
    )
    return {"runId": run_id, "pending": not failed_send, "uncertainRerun": failed_send}


def correlated_run_ids(api, repository, pr_number, status=None):
    """Read-only extension of existing rerun/stop discovery for the current head."""
    pr = api.get(f"repos/{repository}/pulls/{pr_number}")
    head = pr["head"]["sha"]
    roots = {}
    for page in range(1, 11):
        rows = api.get(
            f"repos/{repository}/issues/{pr_number}/comments?per_page=100&page={page}"
        )
        for row in rows:
            if (row.get("user") or {}).get("login") != BOT:
                continue
            raw = decode_marker(row.get("body"), JOURNAL_MARKER)
            if raw is None:
                continue
            journal = protocol._journal(raw)
            request = journal["request"]
            if (request["repository"], request["prNumber"], request["headSha"]) != (
                repository.lower(),
                pr_number,
                head,
            ):
                continue
            if journal["state"] not in ("completed", "attempt_recorded", "uncertain"):
                continue
            handler = Producer(
                api,
                repository,
                pr_number,
                request["triggerCommentId"],
                request["triggerCommentUrl"],
                request["scope"],
            )
            if journal["state"] == "completed":
                run_ids = [slot["rootRunId"] for slot in journal["slots"]]
            else:
                run_ids = [
                    item["workflow_run_id"] for item in handler.exact_roots(request)
                ]
            for run_id in run_ids:
                run = api.get(f"repos/{repository}/actions/runs/{run_id}")
                if (
                    run.get("id") != run_id
                    or run.get("display_title") != request["requestKey"]
                    or run.get("event") != "workflow_dispatch"
                    or run.get("path", "").split("@", 1)[0] != WORKFLOW_PATH
                    or (run.get("repository") or {}).get("full_name", "").lower()
                    != repository.lower()
                    or (run.get("actor") or {}).get("login") != BOT
                ):
                    raise ValueError(
                        "Current-head command discovered an unrelated provider run"
                    )
                if status == "active" and run.get("status") not in (
                    "in_progress",
                    "queued",
                ):
                    continue
                if (
                    status in ("failure", "cancelled")
                    and run.get("conclusion") != status
                ):
                    continue
                roots[run_id] = True
        if len(rows) < 100:
            return sorted(roots)
    raise ValueError("Current-head command journal inventory is incomplete")


def output(values):
    target = os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a") as stream:
            for key, value in values.items():
                if "\n" in str(value) or "\r" in str(value):
                    raise ValueError("Invalid multiline workflow output")
                stream.write(f"{key}={value}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "route",
            "dispatch",
            "validate-run",
            "identify",
            "recover",
            "report-completed",
            "report-progress",
            "prepare-rerun",
            "rerun-submit-failed",
            "run-ids",
        ),
    )
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--status", choices=("active", "failure", "cancelled"))
    args = parser.parse_args()
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    repository = os.environ["GITHUB_REPOSITORY"].lower()
    if event["repository"]["full_name"].lower() != repository:
        raise ValueError("Event repository mismatch")
    api = API(repository)
    if args.command in ("prepare-rerun", "rerun-submit-failed"):
        run = api.get(f"repos/{repository}/actions/runs/{args.run_id}")
        if run.get("event") == "workflow_dispatch" and run.get(
            "display_title", ""
        ).startswith("fi-ci/"):
            print(
                json.dumps(
                    report_progress(
                        API(repository, os.environ.get("STATUS_TOKEN")),
                        repository,
                        {"action": args.command, "workflow_run": run},
                    )
                )
            )
    elif args.command == "report-progress":
        print(json.dumps(report_progress(api, repository, event)))
    elif args.command == "run-ids":
        print(
            "\n".join(
                str(value)
                for value in correlated_run_ids(
                    api, repository, event["issue"]["number"], args.status
                )
            )
        )
    elif args.command == "route":
        metadata = _scope_metadata(event, os.environ.get("TEST_SCOPE", ""))
        print("watcher" if metadata is not None else "legacy")
    elif args.command == "dispatch":
        scope = os.environ.get("TEST_SCOPE", "")
        metadata = _scope_metadata(event, scope)
        if metadata is None:
            raise ValueError("Correlated dispatch requires explicit watcher metadata")
        producer = Producer(
            api,
            repository,
            event["issue"]["number"],
            event["comment"]["id"],
            event["comment"]["html_url"],
            scope,
        )
        producer.status_api = API(repository, os.environ.get("STATUS_TOKEN"))
        producer.request_metadata = metadata
        result = producer.dispatch(metadata["headSha"] if metadata else None)
        print(
            json.dumps(
                {
                    "state": result["state"],
                    "dispatchAttempted": result["dispatchAttempted"],
                }
            )
        )
    elif args.command == "identify":
        pr, comment = identify_run(event)
        output({"pr_number": pr, "comment_id": comment})
    elif args.command == "report-completed":
        print(json.dumps(report_completed(api, repository, event)))
    elif args.command == "recover":
        result = recover_event(api, repository, event)
        print(json.dumps({"state": result["state"]}))
    else:
        inputs = event.get("inputs") or {}
        if not any(
            value not in (None, "")
            for key, value in inputs.items()
            if key.startswith("watcher_")
        ):
            print(json.dumps({"legacyManualDispatch": True}))
            return
        run_id = int(os.environ["GITHUB_RUN_ID"])
        request = validate_run(api, repository, inputs, run_id)
        output(
            {
                "checkout_sha": request["checkoutSha"],
                "head_sha": request["headSha"],
                "base_sha": request["baseSha"],
                "test_scope": " ".join(request["scope"]),
                "is_fork": str(request["isFork"]).lower(),
            }
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Provider bodies and credentials must not enter workflow logs.
        print(
            json.dumps({"state": "error", "reason": type(error).__name__}),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
