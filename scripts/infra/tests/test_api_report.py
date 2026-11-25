"""Report contracts migrated from Node; golden output captured before migration."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pr_checks.api_report import (
    group_findings,
    inline_key,
    publish_report,
    render_inline,
    render_summary,
)
from pr_checks.github_api import GitHubError
from api_test_support import FakeGitHub

ROOT = "repos/org/repo"


def make(api="foo", classification="BREAKING"):
    return {
        "api": api,
        "classification": classification,
        "check": "public_api_signature_changed",
        "path": "flashinfer/example.py",
        "line": 12,
        "before": f"def {api}(x)",
        "after": f"def {api}(x, y)",
        "message": "Old calls omit the required parameter.",
        "level": {
            "BREAKING": "error",
            "LIKELY_BREAKING": "warning",
            "NON_BREAKING": "notice",
        }[classification],
    }


def fixture(findings=None):
    pr = {"number": 3, "head": {"sha": "a" * 40}, "base": {"sha": "b" * 40}}
    report = {
        "execution_status": "COMPLETE",
        "findings": findings if findings is not None else [make()],
    }
    api = FakeGitHub()
    api.routes.update(
        {
            ("GET", ROOT + "/pulls/3"): lambda _: json.loads(json.dumps(pr)),
            ("GET", ROOT + "/issues/3/comments"): [],
            ("POST", ROOT + "/issues/3/comments"): {},
            ("GET", ROOT + "/pulls/3/files"): [
                {
                    "filename": "flashinfer/example.py",
                    "patch": "@@ -12 +12 @@\n-old\n+new",
                }
            ],
            ("GET", ROOT + "/pulls/3/comments"): [],
            ("POST", ROOT + "/pulls/3/reviews"): {},
        }
    )
    return api, pr, report


class ReportTest(unittest.TestCase):
    def publish(self, api, pr, report):
        return publish_report(
            api, pr, report, owner="org", repo="repo", warning=lambda _: None
        )

    def test_mixed_summary_lists_only_changes(self):
        report = {
            "execution_status": "COMPLETE",
            "findings": [
                make("foo"),
                make("bar"),
                make("baz", "LIKELY_BREAKING"),
                make("qux", "LIKELY_BREAKING"),
                make("safe", "NON_BREAKING"),
            ],
        }
        body = render_summary(
            report, repo="org/repo", head="a" * 40, maintainers=["alice"]
        )
        for value in (
            "BLOCKED · 2 breaking · 2 likely breaking",
            "🚨 API BREAK 🚨 · 2",
            "⚠️ API LIKELY BREAK · 2",
            "Maintainers @alice",
            "Reason: …",
            "/blob/" + "a" * 40,
        ):
            self.assertIn(value, body)
        self.assertNotIn("Old calls omit", body)
        self.assertNotIn("non-breaking", body)
        self.assertEqual(body.count("```diff"), 4)

    def test_inline_explains_impact_with_no_global_controls(self):
        body = render_inline(make("foo", "LIKELY_BREAKING"))
        self.assertIn("⚠️ API LIKELY BREAK", body)
        self.assertIn("Old calls omit", body)
        for value in ("🚨", "expected", "Maintainers"):
            self.assertNotIn(value, body)

    def test_group_highest_severity_and_evidence(self):
        grouped = group_findings([make("foo", "LIKELY_BREAKING"), make()])
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0]["classification"], "BREAKING")

    def test_untrusted_source_cannot_inject_mentions_or_fences(self):
        body = render_inline({**make(), "before": "```\n@someone <script>"})
        self.assertNotIn("@someone", body)
        self.assertEqual(body.count("```"), 2)

    def test_reuse_active_thread_after_unrelated_push(self):
        api, pr, report = fixture()
        api.routes[("GET", ROOT + "/pulls/3/comments")] = [
            {
                "id": 7,
                "user": {"login": "github-actions[bot]"},
                "body": f"<!-- flashinfer-api-inline --><!-- key:{inline_key(make())} -->",
                "path": "flashinfer/example.py",
                "line": 12,
                "commit_id": "c" * 40,
            }
        ]
        api.routes[("PATCH", ROOT + "/pulls/comments/7")] = {}
        result = self.publish(api, pr, report)
        self.assertEqual(result["updated"], 1)
        self.assertFalse(
            any(c[0] == "POST" and c[1].endswith("/reviews") for c in api.calls)
        )

    def test_long_signatures_show_changed_region(self):
        unchanged = ", ".join(f"arg_{i}" for i in range(60))
        body = render_inline(
            {
                **make(),
                "before": f"def foo({unchanged}, y)",
                "after": f"def foo({unchanged}, *, y)",
            }
        )
        self.assertIn("*, y", body)
        self.assertIn("…", body)
        self.assertNotIn("arg_0, arg_1", body)

    def test_native_return_arrows(self):
        body = render_inline(
            {**make(), "before": "def foo(x) -> Tensor", "after": "def foo(x) -> None"}
        )
        self.assertIn("-> Tensor", body)
        self.assertNotIn("&gt;", body)

    def test_batch_new_lines_into_one_submitted_review(self):
        api, pr, report = fixture([make(), make("bar", "LIKELY_BREAKING")])
        result = self.publish(api, pr, report)
        reviews = [
            c[2] for c in api.calls if c[0] == "POST" and c[1].endswith("/reviews")
        ]
        self.assertEqual(len(reviews), 1)
        self.assertEqual(len(reviews[0]["comments"]), 2)
        self.assertEqual(reviews[0]["event"], "COMMENT")
        self.assertEqual(
            (result["posted"], result["line"], result["failed"]), (2, 2, 0)
        )

    def test_clean_summary_is_silent(self):
        for findings in ([], [make("safe", "NON_BREAKING")]):
            self.assertIsNone(
                render_summary(
                    {"execution_status": "COMPLETE", "findings": findings},
                    repo="org/repo",
                    head="a" * 40,
                )
            )

    def test_warnings_expected_and_incomplete_remain_visible(self):
        for report, options, phrase in [
            (
                {
                    "execution_status": "COMPLETE",
                    "findings": [make("risk", "LIKELY_BREAKING")],
                },
                {},
                "API LIKELY BREAK",
            ),
            (
                {"execution_status": "COMPLETE", "findings": [make()]},
                {"expected": True},
                "PASS (expected)",
            ),
            ({"execution_status": "FAILED", "findings": []}, {}, "INCOMPLETE"),
        ]:
            self.assertIn(
                phrase,
                render_summary(report, repo="org/repo", head="a" * 40, **options),
            )

    def test_clean_publication_deletes_only_own_summary(self):
        for stale in (False, True):
            api, pr, report = fixture([])
            comments = [
                {
                    "id": 1,
                    "user": {"login": "someone"},
                    "body": "<!-- flashinfer-api-compatibility -->",
                }
            ]
            if stale:
                comments.append(
                    {
                        "id": 2,
                        "user": {"login": "github-actions[bot]"},
                        "body": "<!-- flashinfer-api-compatibility -->old",
                    }
                )
            api.routes[("GET", ROOT + "/issues/3/comments")] = comments
            api.routes[("DELETE", ROOT + "/issues/comments/2")] = None
            result = self.publish(api, pr, report)
            self.assertEqual(
                [(c[0], c[1]) for c in api.calls if c[0] != "GET"],
                [("DELETE", ROOT + "/issues/comments/2")] if stale else [],
            )
            self.assertEqual(result["total"], 0)
            self.assertFalse(any(c[1].endswith("/files") for c in api.calls))

    def test_removed_source_links_use_base(self):
        item = {**make(), "anchors": [["flashinfer/old.py", "LEFT", 8, 10]]}
        body = render_summary(
            {"execution_status": "COMPLETE", "base": "b" * 40, "findings": [item]},
            repo="org/repo",
            head="a" * 40,
        )
        self.assertIn("/blob/" + "b" * 40 + "/flashinfer/old.py#L8", body)

    def test_analyzed_head_and_standalone_command(self):
        body = render_summary(
            {"execution_status": "COMPLETE", "findings": [make()]},
            repo="org/repo",
            head="a" * 40,
        )
        self.assertIn("Analyzed head: [`aaaaaaa`]", body)
        self.assertIn("entire comment", body)

    def test_bounded_summary_preserves_counts_and_artifact_link(self):
        report = {
            "execution_status": "COMPLETE",
            "findings": [
                {**make(f"api_{i}"), "before": "x" * 2500, "after": "y" * 2500}
                for i in range(300)
            ],
        }
        body = render_summary(
            report, repo="org/repo", head="a" * 40, run_url="https://example.com/run"
        )
        self.assertLess(len(body), 60000)
        self.assertIn("300 breaking", body)
        self.assertIn("omitted", body)
        self.assertIn("https://example.com/run", body)

    def test_head_races_do_not_publish_stale_findings(self):
        for phase in ("comments", "diff"):
            api, pr, report = fixture()
            snapshot = json.loads(json.dumps(pr))

            def advance(_):
                pr["head"]["sha"] = "c" * 40
                return []

            api.routes[
                (
                    "GET",
                    ROOT
                    + (
                        "/issues/3/comments"
                        if phase == "comments"
                        else "/pulls/3/files"
                    ),
                )
            ] = advance
            self.publish(api, snapshot, report)
            self.assertFalse(
                any(c[0] == "POST" and c[1].endswith("/reviews") for c in api.calls)
            )
            if phase == "comments":
                self.assertFalse(any(c[0] != "GET" for c in api.calls))

    def test_exact_rendering_matches_captured_javascript_contract(self):
        for case in json.loads(
            Path(__file__).with_name("api_render_contract.json").read_text()
        ):
            options = dict(case["options"])
            if "runUrl" in options:
                options["run_url"] = options.pop("runUrl")
            self.assertEqual(render_summary(case["report"], **options), case["summary"])
            self.assertEqual(
                [render_inline(f) for f in case["report"]["findings"]], case["inline"]
            )

    def test_only_validation_failures_retry_individual_comments(self):
        for status in (422, 403, 0):
            api, pr, report = fixture([make(), make("bar")])
            api.routes[("POST", ROOT + "/pulls/3/reviews")] = (
                GitHubError(status, "review failed")
                if status
                else TimeoutError("timeout")
            )
            if status == 422:
                api.routes[("POST", ROOT + "/pulls/3/reviews")].errors = [
                    {
                        "resource": "PullRequestReviewComment",
                        "field": "line",
                        "code": "invalid",
                    }
                ]
            api.routes[("POST", ROOT + "/pulls/3/comments")] = {}
            result = self.publish(api, pr, report)
            individual = [
                c
                for c in api.calls
                if c[0] == "POST" and c[1].endswith("/pulls/3/comments")
            ]
            self.assertEqual(len(individual), 2 if status == 422 else 0)
            self.assertEqual(result["failed"], 0 if status == 422 else 2)

    def test_retarget_during_inline_updates_stops_later_writes(self):
        from copy import deepcopy

        api, pr, report = fixture([make(), make("bar")])
        snapshot = deepcopy(pr)
        api.routes[("GET", ROOT + "/pulls/3/comments")] = [
            {
                "id": i,
                "user": {"login": "github-actions[bot]"},
                "body": f"<!-- flashinfer-api-inline --><!-- key:{inline_key(f)} -->",
                "path": f["path"],
                "line": 12,
            }
            for i, f in zip([7, 8], report["findings"], strict=True)
        ]

        def advance(_):
            pr["base"]["sha"] = "c" * 40
            return {}

        api.routes[("PATCH", ROOT + "/pulls/comments/7")] = advance
        api.routes[("PATCH", ROOT + "/pulls/comments/8")] = {}
        self.publish(api, snapshot, report)
        self.assertFalse(
            any(c[0] == "PATCH" and c[1].endswith("/8") for c in api.calls)
        )

    def test_retarget_after_last_anchor_prevents_batch_review(self):
        from copy import deepcopy
        from unittest.mock import patch
        from pr_checks.api_anchors import select_anchor

        api, pr, report = fixture()
        snapshot = deepcopy(pr)

        def advance(finding, files):
            result = select_anchor(finding, files)
            pr["base"]["sha"] = "c" * 40
            return result

        with patch("pr_checks.api_report.select_anchor", side_effect=advance):
            self.publish(api, snapshot, report)
        self.assertFalse(
            any(c[0] == "POST" and c[1].endswith("/reviews") for c in api.calls)
        )

    def test_spam_rejection_does_not_fan_out(self):
        api, pr, report = fixture([make(), make("bar")])
        error = GitHubError(422, "spam rejection")
        error.errors = [
            {
                "resource": "PullRequestReview",
                "code": "custom",
                "message": "You have submitted too quickly",
            }
        ]
        api.routes[("POST", ROOT + "/pulls/3/reviews")] = error
        api.routes[("POST", ROOT + "/pulls/3/comments")] = {}
        result = self.publish(api, pr, report)
        self.assertEqual(result["failed"], 2)
        self.assertFalse(
            any(
                c[0] == "POST" and c[1].endswith("/pulls/3/comments") for c in api.calls
            )
        )

    def test_individual_rejection_stops_further_fallback(self):
        api, pr, report = fixture([make(), make("bar")])
        error = GitHubError(422, "invalid line")
        error.errors = [
            {"resource": "PullRequestReviewComment", "field": "line", "code": "invalid"}
        ]
        api.routes[("POST", ROOT + "/pulls/3/reviews")] = error
        api.routes[("POST", ROOT + "/pulls/3/comments")] = GitHubError(
            403, "rate limited"
        )
        result = self.publish(api, pr, report)
        self.assertEqual(result["failed"], 2)
        self.assertEqual(
            sum(
                c[0] == "POST" and c[1].endswith("/pulls/3/comments") for c in api.calls
            ),
            1,
        )
