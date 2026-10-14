"""Pure request/receipt protocol for CI dispatch; this module never sends requests.

The adapter must durably save ``record_attempt`` before making the POST. An
attempt without an exact response remains uncertain and cannot be redispatched.
Recovery requires the provider's exact request key, never head/time proximity.
"""

import copy
import hashlib
import json
import re
from pathlib import PurePosixPath

SHA = re.compile(r"[0-9a-f]{40}")
REPOSITORY = re.compile(r"[a-z0-9][a-z0-9_.-]*/[a-z0-9][a-z0-9_.-]*")
TARGET = re.compile(r"tests/[A-Za-z0-9_./-]+(?:::[A-Za-z0-9_.-]+)*")
MARKER = "watcher-ci-receipt-v1"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _positive(value, name):
    if type(value) is not int or not 0 < value < 2**63:
        raise ValueError(name + " must be a positive 64-bit integer")
    return value


def normalize_scope(scope):
    if isinstance(scope, str):
        scope = [] if scope.strip() in ("", "full") else scope.split()
    if not isinstance(scope, list) or len(scope) > 64:
        raise ValueError("Scope must be full or at most 64 test targets")
    for target in scope:
        if not isinstance(target, str) or not TARGET.fullmatch(target):
            raise ValueError("Invalid test target")
        path = target.split("::", 1)[0]
        if ".." in PurePosixPath(path).parts or "//" in path:
            raise ValueError("Test targets must stay under tests/")
    return sorted(set(scope))


def build_request(
    repository,
    pr_number,
    comment_id,
    comment_url,
    head_sha,
    base_sha,
    checkout_sha,
    scope,
):
    if not isinstance(repository, str):
        raise ValueError("Repository is required")
    repository = repository.lower()
    if not REPOSITORY.fullmatch(repository):
        raise ValueError("Repository must be owner/name")
    _positive(pr_number, "PR number")
    _positive(comment_id, "Trigger comment ID")
    expected_url = (
        f"https://github.com/{repository}/pull/{pr_number}#issuecomment-{comment_id}"
    )
    if not isinstance(comment_url, str) or comment_url.lower() != expected_url:
        raise ValueError("Trigger comment URL must match repository, PR and ID")
    shas = []
    for value in (head_sha, base_sha, checkout_sha):
        if not isinstance(value, str) or not SHA.fullmatch(value.lower()):
            raise ValueError("Head, base and checkout must be full commit SHAs")
        shas.append(value.lower())
    request = {
        "repository": repository,
        "prNumber": pr_number,
        "triggerCommentId": comment_id,
        "triggerCommentUrl": expected_url,
        "headSha": shas[0],
        "baseSha": shas[1],
        "checkoutSha": shas[2],
        "scope": normalize_scope(scope),
    }
    request["scopeDigest"] = _digest(request["scope"])
    request["requestKey"] = (
        f"fi-ci/{pr_number}/{comment_id}/{shas[0]}/{request['scopeDigest']}"
    )
    request["dispatchInputs"] = {
        "watcher_request_key": request["requestKey"],
        "watcher_pr_number": str(pr_number),
        "watcher_trigger_comment_id": str(comment_id),
        "watcher_trigger_comment_url": expected_url,
        "watcher_head_sha": shas[0],
        "watcher_base_sha": shas[1],
        "watcher_checkout_sha": shas[2],
        "watcher_scope": " ".join(request["scope"]),
        "watcher_scope_digest": request["scopeDigest"],
    }
    return request


def _request(request):
    try:
        rebuilt = build_request(
            request["repository"],
            request["prNumber"],
            request["triggerCommentId"],
            request["triggerCommentUrl"],
            request["headSha"],
            request["baseSha"],
            request["checkoutSha"],
            request["scope"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid normalized request") from error
    if rebuilt != request:
        raise ValueError("Request identity or dispatch inputs changed")
    return rebuilt


def dispatch_payload(request, ref):
    request = _request(request)
    if (
        not isinstance(ref, str)
        or not ref
        or len(ref) > 255
        or not re.fullmatch(r"[A-Za-z0-9_./-]+", ref)
        or ".." in ref
        or ref.startswith("-")
        or ref.endswith("/")
    ):
        raise ValueError("A trusted default-branch ref is required")
    return {
        "ref": ref,
        "inputs": copy.deepcopy(request["dispatchInputs"]),
        "return_run_details": True,
    }


def returned_root(request, response):
    request = _request(request)
    if not isinstance(response, dict):
        raise ValueError("Dispatch must return the exact run details")
    run_id = _positive(response.get("workflow_run_id"), "Workflow run ID")
    repository = request["repository"]
    if (
        response.get("run_url")
        != f"https://api.github.com/repos/{repository}/actions/runs/{run_id}"
        or response.get("html_url")
        != f"https://github.com/{repository}/actions/runs/{run_id}"
    ):
        raise ValueError("Returned run URLs do not match the requested repository")
    # Cancellation/failure before Setup does not remove the exact returned root.
    return run_id


def completed_receipt(request, root_ids):
    request = _request(request)
    if not isinstance(root_ids, list) or not root_ids:
        raise ValueError("At least one exact root is required")
    for run_id in root_ids:
        _positive(run_id, "Root run ID")
    if len(set(root_ids)) != len(root_ids):
        raise ValueError("Root run IDs must be unique")
    return {
        "schemaVersion": 1,
        "repository": request["repository"],
        "prNumber": request["prNumber"],
        "triggerCommentId": request["triggerCommentId"],
        "triggerCommentUrl": request["triggerCommentUrl"],
        "requestedHeadSha": request["headSha"],
        "provider": "github",
        "rootRunIds": sorted(root_ids),
        "rootSubmissionComplete": True,
    }


def receipt_body(receipt):
    expected = {
        "schemaVersion",
        "repository",
        "prNumber",
        "triggerCommentId",
        "triggerCommentUrl",
        "requestedHeadSha",
        "provider",
        "rootRunIds",
        "rootSubmissionComplete",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != expected
        or type(receipt.get("schemaVersion")) is not int
        or receipt.get("schemaVersion") != 1
        or receipt.get("rootSubmissionComplete") is not True
    ):
        raise ValueError("Unexpected receipt schema")
    request = build_request(
        receipt["repository"],
        receipt["prNumber"],
        receipt["triggerCommentId"],
        receipt["triggerCommentUrl"],
        receipt["requestedHeadSha"],
        receipt["requestedHeadSha"],
        receipt["requestedHeadSha"],
        [],
    )
    if receipt != completed_receipt(request, receipt["rootRunIds"]):
        raise ValueError("Only a complete exact-root receipt may be published")
    return "<!-- " + MARKER + " " + _canonical(receipt) + " -->"


def plan(request, root_count=1):
    request = _request(request)
    if type(root_count) is not int or not 1 <= root_count <= 32:
        raise ValueError("Expected root count must be 1..32")
    return {
        "schemaVersion": 1,
        "request": request,
        "state": "planned",
        "dispatchAttempted": False,
        "slots": [{"state": "planned", "rootRunId": None} for _ in range(root_count)],
    }


def _journal(journal, slot=None):
    value = copy.deepcopy(journal)
    if type(value.get("schemaVersion")) is not int or value["schemaVersion"] != 1:
        raise ValueError("Invalid journal version")
    _request(value["request"])
    slots = value.get("slots")
    if not isinstance(slots, list) or not 1 <= len(slots) <= 32:
        raise ValueError("Invalid submission slots")
    if slot is not None and (type(slot) is not int or not 0 <= slot < len(slots)):
        raise ValueError("Unknown submission slot")
    roots = []
    for entry in slots:
        if entry.get("state") not in (
            "planned",
            "attempt_recorded",
            "uncertain",
            "completed",
        ):
            raise ValueError("Invalid submission state")
        if entry["state"] == "completed":
            roots.append(_positive(entry.get("rootRunId"), "Root run ID"))
        elif entry.get("rootRunId") is not None:
            raise ValueError("Unconfirmed slot must not claim a root")
    if len(set(roots)) != len(roots):
        raise ValueError("Duplicate roots across submission slots")
    attempted = any(entry["state"] != "planned" for entry in slots)
    if (
        type(value.get("dispatchAttempted")) is not bool
        or value["dispatchAttempted"] != attempted
    ):
        raise ValueError("Journal attempt flag contradicts recorded slots")
    if value.get("state") == "no-send":
        if (
            attempted
            or not isinstance(value.get("reason"), str)
            or not value["reason"].strip()
        ):
            raise ValueError("Invalid no-send journal")
    elif value.get("state") != _state(copy.deepcopy(value))["state"]:
        raise ValueError("Journal state contradicts recorded slots")
    return value


def _state(value):
    states = {entry["state"] for entry in value["slots"]}
    value["state"] = (
        "completed"
        if states == {"completed"}
        else "uncertain"
        if "uncertain" in states
        else "attempt_recorded"
        if "attempt_recorded" in states
        else "partial"
        if "completed" in states
        else "planned"
    )
    return value


def record_attempt(journal, slot=0):
    value = _journal(journal, slot)
    if value.get("state") == "no-send" or value["slots"][slot]["state"] != "planned":
        raise ValueError("Already attempted; recover exact roots, never redispatch")
    value["slots"][slot]["state"] = "attempt_recorded"
    value["dispatchAttempted"] = True
    return _state(value)


def record_response(journal, response, slot=0):
    value = _journal(journal, slot)
    if value["slots"][slot]["state"] != "attempt_recorded":
        raise ValueError("Persist an attempt before accepting a response")
    root = returned_root(value["request"], response)
    if any(entry["rootRunId"] == root for entry in value["slots"]):
        raise ValueError("Root is already bound to another submission slot")
    value["slots"][slot] = {"state": "completed", "rootRunId": root}
    return _state(value)


def mark_uncertain(journal, reason, slot=0):
    value = _journal(journal, slot)
    if value["slots"][slot]["state"] not in ("attempt_recorded", "uncertain"):
        raise ValueError("Only an attempted submission can be uncertain")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
        raise ValueError("A bounded uncertainty reason is required")
    value["slots"][slot].update(state="uncertain", reason=reason)
    return _state(value)


def recover(journal, request_key, response, slot=0):
    """request_key must come from exact provider metadata verified by the adapter."""
    value = _journal(journal, slot)
    if request_key != value["request"]["requestKey"]:
        raise ValueError("Recovery requires exact request-key equality")
    if value["slots"][slot]["state"] not in ("attempt_recorded", "uncertain"):
        raise ValueError("Only an unacknowledged attempt can be recovered")
    value["slots"][slot] = {"state": "attempt_recorded", "rootRunId": None}
    return record_response(_state(value), response, slot)


def no_send(journal, reason):
    value = _journal(journal)
    if value.get("dispatchAttempted") or any(
        entry["state"] != "planned" for entry in value["slots"]
    ):
        raise ValueError("Attempted requests can never become no-send")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
        raise ValueError("A bounded no-send reason is required")
    value.update(state="no-send", reason=reason, dispatchAttempted=False)
    return value


def acknowledgment(journal):
    value = _journal(journal)
    if value.get("state") != "completed" or any(
        entry["state"] != "completed" for entry in value["slots"]
    ):
        raise ValueError("Partial or uncertain submission cannot be complete")
    return completed_receipt(
        value["request"], [entry["rootRunId"] for entry in value["slots"]]
    )
