"""Compact API reports and idempotent GitHub inline delivery."""

import hashlib
import html
from urllib.parse import quote

from .api_anchors import select_anchor
from .github_api import anchor_validation_error, warning as emit_warning

MARKER = "<!-- flashinfer-api-compatibility -->"
INLINE = "<!-- flashinfer-api-inline -->"
RANK = {"NON_BREAKING": 0, "LIKELY_BREAKING": 1, "BREAKING": 2}


def code_text(value) -> str:
    return (
        str(value if value is not None else "")
        .replace("@", "@\u200b")
        .replace("`", "ˋ")
        .replace("\r", " ")
        .replace("\n", " ")[:2500]
    )


def text(value) -> str:
    return code_text(html.escape(str(value if value is not None else ""), quote=False))


def heading(finding: dict) -> str:
    return (
        "🚨 API BREAK 🚨"
        if finding["classification"] == "BREAKING"
        else "⚠️ API LIKELY BREAK"
    )


def group_findings(findings: list[dict]) -> list[dict]:
    groups = {}
    for finding in findings:
        key = finding.get("api") or f"{finding.get('path')}:{finding.get('check')}"
        previous = groups.get(key)
        if previous is None:
            groups[key] = dict(finding)
            continue
        higher = (
            finding
            if RANK[finding["classification"]] > RANK[previous["classification"]]
            else previous
        )
        anchors = []
        for anchor in [*previous.get("anchors", []), *finding.get("anchors", [])]:
            if anchor not in anchors:
                anchors.append(anchor)
        groups[key] = {
            **higher,
            "message": " ".join(
                dict.fromkeys([previous["message"], finding["message"]])
            ),
            "anchors": anchors,
        }
    return list(groups.values())


def diff(finding: dict) -> str:
    before, after = str(finding.get("before") or ""), str(finding.get("after") or "")
    prefix = suffix = 0
    if max(len(before), len(after)) > 220 and before != after:
        while prefix < min(len(before), len(after)) and before[prefix] == after[prefix]:
            prefix += 1
        while (
            suffix < min(len(before), len(after)) - prefix
            and before[len(before) - suffix - 1] == after[len(after) - suffix - 1]
        ):
            suffix += 1
    start = prefix - 70 if prefix > 120 else 0
    trim_end = suffix - 70 if suffix > 120 else 0

    def fragment(value):
        return (
            ("…" if start else "")
            + code_text(value[start : len(value) - trim_end])
            + ("…" if trim_end else "")
        )

    return "\n".join(
        ["```diff", f"- {fragment(before)}", f"+ {fragment(after)}", "```"]
    )


def inline_key(finding: dict) -> str:
    key = finding.get("api") or f"{finding.get('path')}:{finding.get('check')}"
    return hashlib.sha256(key.encode()).hexdigest()


def render_inline(finding: dict) -> str:
    return f"{INLINE}<!-- key:{inline_key(finding)} -->\n**{heading(finding)}**\n\n{diff(finding)}\n\n{text(finding.get('message'))}"


def render_summary(
    report: dict,
    *,
    repo: str,
    head: str,
    expected: bool = False,
    maintainers: list[str] | None = None,
    run_url: str = "",
    blocked_reason: str = "",
) -> str | None:
    findings = group_findings(report.get("findings", []))
    counts = {kind: sum(f["classification"] == kind for f in findings) for kind in RANK}
    if (
        report.get("execution_status") == "COMPLETE"
        and not counts["BREAKING"]
        and not counts["LIKELY_BREAKING"]
        and not blocked_reason
    ):
        return None
    if blocked_reason:
        state = "BLOCKED"
    elif report.get("execution_status") != "COMPLETE":
        state = "INCOMPLETE"
    elif counts["BREAKING"]:
        state = "PASS (expected)" if expected else "BLOCKED"
    else:
        state = "PASS"
    lines = [
        MARKER,
        f"**API check: {state} · {counts['BREAKING']} breaking · {counts['LIKELY_BREAKING']} likely breaking**",
    ]
    if blocked_reason:
        lines.extend(["", text(blocked_reason)])
    if report.get("execution_status") != "COMPLETE":
        lines.extend(
            [
                "",
                f"Scan did not complete: {text(report.get('error') or 'See the check run.')}",
            ]
        )
    lines.extend(
        ["", f"Analyzed head: [`{head[:7]}`](https://github.com/{repo}/commit/{head})"]
    )
    remaining = 50000
    for kind in ("BREAKING", "LIKELY_BREAKING"):
        items = [f for f in findings if f["classification"] == kind]
        if not items:
            continue
        lines.extend(["", f"**{heading(items[0])} · {len(items)}**"])
        omitted = 0
        for item in items:
            anchors = item.get("anchors") or []
            source = next(
                (a for a in anchors if a[1] == "RIGHT"), anchors[0] if anchors else None
            )
            path = source[0] if source else item["path"]
            line = (source[2] if source else item.get("line")) or 1
            revision = (
                (report.get("base") or head) if source and source[1] == "LEFT" else head
            )
            encoded_path = quote(path, safe="/~()*!.'-")
            url = f"https://github.com/{repo}/blob/{revision}/{encoded_path}#L{line}"
            entry = "\n".join(
                [
                    "",
                    f"**`{text(item.get('api'))}`** · [{text(path)}:{line}]({url})",
                    "",
                    diff(item),
                ]
            )
            if len(entry) > remaining:
                omitted += 1
                continue
            remaining -= len(entry)
            lines.append(entry)
        if omitted:
            lines.extend(
                [
                    "",
                    f"{omitted} API changes omitted here; see the full report artifact in the check run.",
                ]
            )
    if state == "BLOCKED" and not blocked_reason:
        if maintainers:
            lines.extend(
                [
                    "",
                    "Code owners: "
                    + ", ".join(f"@{name}" for name in maintainers)
                    + ".",
                ]
            )
        lines.extend(
            [
                "",
                "Maintainers with maintain/admin permission (other than the PR author) can comment `@flashinfer-bot API BREAK expected` as the entire comment to unblock all API BREAK findings for the current head. Reason: … (provide separately in the PR description or discussion).",
            ]
        )
    if run_url:
        lines.extend(["", f"[Full report and check run]({run_url})"])
    return "\n".join(lines)


def publish_report(
    api,
    pr: dict,
    report: dict,
    *,
    owner: str,
    repo: str,
    expected: bool = False,
    maintainers: list[str] | None = None,
    run_url: str = "",
    blocked_reason: str = "",
    warning=emit_warning,
) -> dict | None:
    root = f"repos/{owner}/{repo}"
    pull = f"{root}/pulls/{pr['number']}"
    delivery = dict.fromkeys(
        ("total", "line", "file", "posted", "updated", "unplaced", "failed"), 0
    )

    def current():
        latest = api.request("GET", pull)
        return latest["head"]["sha"] == pr["head"]["sha"] and latest.get(
            "base", {}
        ).get("sha") == pr.get("base", {}).get("sha")

    if not current():
        return None
    comments = api.paginate(f"{root}/issues/{pr['number']}/comments")
    if not current():
        return None
    body = render_summary(
        report,
        repo=f"{owner}/{repo}",
        head=pr["head"]["sha"],
        expected=expected,
        maintainers=maintainers,
        run_url=run_url,
        blocked_reason=blocked_reason,
    )
    previous = next(
        (
            c
            for c in comments
            if c.get("user", {}).get("login") == "github-actions[bot]"
            and (c.get("body") or "").startswith(MARKER)
        ),
        None,
    )
    if body is None:
        if previous:
            api.request("DELETE", f"{root}/issues/comments/{previous['id']}")
        return delivery
    if previous:
        api.request("PATCH", f"{root}/issues/comments/{previous['id']}", {"body": body})
    else:
        api.request("POST", f"{root}/issues/{pr['number']}/comments", {"body": body})
    files = api.paginate(pull + "/files")
    existing = api.paginate(pull + "/comments")
    if not current():
        return None
    pending = []
    for finding in group_findings(report.get("findings", [])):
        if finding["classification"] == "NON_BREAKING":
            continue
        if not current():
            return None
        delivery["total"] += 1
        anchor = select_anchor(finding, files)
        if not anchor:
            delivery["unplaced"] += 1
            warning(f"No changed file for API review comment: {finding.get('api')}")
            continue
        key = f"<!-- key:{inline_key(finding)} -->"
        file_level = anchor.get("subject_type") == "file"
        prior = next(
            (
                c
                for c in existing
                if c.get("user", {}).get("login") == "github-actions[bot]"
                and (c.get("body") or "").startswith(INLINE)
                and key in c["body"]
                and c.get("path") == anchor["path"]
                and (
                    (c.get("subject_type") == "file")
                    if file_level
                    else (
                        c.get("line") == anchor["line"]
                        and (c.get("side") or "RIGHT") == anchor["side"]
                    )
                )
            ),
            None,
        )
        body = render_inline(finding)
        try:
            if prior:
                api.request(
                    "PATCH", f"{root}/pulls/comments/{prior['id']}", {"body": body}
                )
                delivery["updated"] += 1
            elif not file_level:
                pending.append({**anchor, "body": body})
                continue
            else:
                api.request(
                    "POST",
                    pull + "/comments",
                    {"commit_id": pr["head"]["sha"], **anchor, "body": body},
                )
                delivery["posted"] += 1
            delivery["file" if file_level else "line"] += 1
        except Exception as error:
            delivery["failed"] += 1
            warning(f"API review comment failed for {finding.get('api')}: {error}")
    if pending:
        if not current():
            return None
        try:
            api.request(
                "POST",
                pull + "/reviews",
                {
                    "commit_id": pr["head"]["sha"],
                    "event": "COMMENT",
                    "body": "<!-- flashinfer-api-inline-review -->",
                    "comments": pending,
                },
            )
            delivery["posted"] += len(pending)
            delivery["line"] += len(pending)
        except Exception as error:
            # Validation errors get per-line fallback; never amplify rate limits.
            if anchor_validation_error(error):
                for index, comment in enumerate(pending):
                    if not current():
                        return None
                    try:
                        api.request(
                            "POST",
                            pull + "/comments",
                            {"commit_id": pr["head"]["sha"], **comment},
                        )
                        delivery["posted"] += 1
                        delivery["line"] += 1
                    except Exception as individual:
                        delivery["failed"] += 1
                        warning(f"API inline comment failed: {individual}")
                        if not anchor_validation_error(individual):
                            delivery["failed"] += len(pending) - index - 1
                            break
            else:
                delivery["failed"] += len(pending)
                warning(f"API inline review failed: {error}")
    return delivery
