"""Mock-only tests for the trusted adapter; never executes gh or contacts GitHub."""

import copy
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/pr_checks"))
import ci_request_protocol as protocol
import ci_request_producer as producer


class FakeAPI:
    def __init__(self):
        self.calls = []
        self.comments = []
        self.next_comment_id = 1000
        self.pr = {
            "state": "open",
            "draft": False,
            "mergeable": True,
            "head": {"sha": "a" * 40},
            "base": {"sha": "b" * 40},
            "merge_commit_sha": "c" * 40,
        }
        self.pr_reads = 0
        self.changed_head_on_second_read = False
        self.parents = [{"sha": "a" * 40}, {"sha": "b" * 40}]
        self.dispatch_error = None
        self.dispatch_result = self.response(456)
        self.runs = []
        self.run = self.run_record(456)
        self.run_reads = 0
        self.change_run_on_second_read = False
        self.status_comment = 123
        self.checks = []
        self.comment_created_at = "2026-10-14T02:00:00Z"
        self.jobs = [
            {
                "id": 1,
                "name": "Setup",
                "status": "completed",
                "conclusion": "success",
                "html_url": "https://github.com/owner/repo/actions/runs/456/job/1",
            },
            {
                "id": 2,
                "name": "Test Results Summary",
                "status": "completed",
                "conclusion": "success",
                "html_url": "https://github.com/owner/repo/actions/runs/456/job/2",
            },
        ]

    @staticmethod
    def response(number):
        return {
            "workflow_run_id": number,
            "run_url": f"https://api.github.com/repos/owner/repo/actions/runs/{number}",
            "html_url": f"https://github.com/owner/repo/actions/runs/{number}",
        }

    def request(self, comment=123):
        return protocol.build_request(
            "owner/repo",
            7,
            comment,
            f"https://github.com/owner/repo/pull/7#issuecomment-{comment}",
            "a" * 40,
            "b" * 40,
            "c" * 40,
            "tests/utils",
        )

    def run_record(self, number, comment=123):
        response = self.response(number)
        return {
            "id": number,
            "url": response["run_url"],
            "html_url": response["html_url"],
            "display_title": self.request(comment)["requestKey"],
            "event": "workflow_dispatch",
            "path": producer.WORKFLOW_PATH,
            "repository": {"full_name": "owner/repo"},
            "triggering_actor": {"login": producer.BOT},
            "actor": {"login": producer.BOT},
            "status": "completed",
            "conclusion": "cancelled",
            "run_attempt": 1,
            "updated_at": "2026-10-14T03:00:00Z",
        }

    def seed(self, journal):
        body = "<!-- " + producer.JOURNAL_MARKER + " " + json.dumps(journal) + " -->"
        row = {
            "id": self.next_comment_id,
            "body": body,
            "user": {"login": producer.BOT},
        }
        self.next_comment_id += 1
        self.comments.append(row)

    def get(self, path):
        self.calls.append(("GET", path, None))
        if path.startswith("repos/owner/repo/issues/7/comments?"):
            return copy.deepcopy(self.comments)
        if path == "repos/owner/repo/pulls/7":
            self.pr_reads += 1
            value = copy.deepcopy(self.pr)
            if self.changed_head_on_second_read and self.pr_reads >= 2:
                value["head"]["sha"] = "d" * 40
            return value
        if path == "repos/owner/repo/commits/" + "c" * 40:
            return {"parents": copy.deepcopy(self.parents)}
        if path == "repos/owner/repo/":
            return {"default_branch": "main"}
        if path.startswith("repos/owner/repo/actions/workflows/pr-test.yml/runs?"):
            return {
                "total_count": len(self.runs),
                "workflow_runs": copy.deepcopy(self.runs),
            }
        if path == "repos/owner/repo/actions/runs/456":
            self.run_reads += 1
            value = copy.deepcopy(self.run)
            if self.change_run_on_second_read and self.run_reads >= 2:
                value["run_attempt"] += 1
            return value
        if path.startswith("repos/owner/repo/actions/runs/456/jobs?"):
            return {"total_count": len(self.jobs), "jobs": copy.deepcopy(self.jobs)}
        if path.startswith("repos/owner/repo/issues/comments/"):
            number = int(path.rsplit("/", 1)[1])
            return {
                "id": number,
                "html_url": f"https://github.com/owner/repo/pull/7#issuecomment-{number}",
                "created_at": self.comment_created_at,
            }
        if path == "repos/owner/repo/commits/" + "a" * 40 + "/status":
            return {
                "statuses": [
                    {
                        "context": "ci/watcher-request",
                        "description": f"request:{self.status_comment}: pending",
                    }
                ]
            }
        if path.startswith("repos/owner/repo/commits/" + "a" * 40 + "/check-runs?"):
            return {
                "total_count": len(self.checks),
                "check_runs": copy.deepcopy(self.checks),
            }
        raise AssertionError("Unexpected mock GET: " + path)

    def post(self, path, data):
        self.calls.append(("POST", path, copy.deepcopy(data)))
        if path == "repos/owner/repo/issues/7/comments":
            row = {
                "id": self.next_comment_id,
                "body": data["body"],
                "user": {"login": producer.BOT},
            }
            self.next_comment_id += 1
            self.comments.append(row)
            return copy.deepcopy(row)
        if path == "repos/owner/repo/actions/workflows/pr-test.yml/dispatches":
            if self.dispatch_error:
                raise self.dispatch_error
            return copy.deepcopy(self.dispatch_result)
        if path == "repos/owner/repo/check-runs":
            value = {
                "id": 900 + len(self.checks),
                "app": {"slug": "github-actions"},
                **copy.deepcopy(data),
            }
            self.checks.append(value)
            return copy.deepcopy(value)
        if path.startswith("repos/owner/repo/statuses/"):
            return {"id": 950, **copy.deepcopy(data)}
        raise AssertionError("Unexpected mock POST: " + path)

    def patch(self, path, data):
        self.calls.append(("PATCH", path, copy.deepcopy(data)))
        for row in self.comments:
            if path == f"repos/owner/repo/issues/comments/{row['id']}":
                row["body"] = data["body"]
                return copy.deepcopy(row)
        for row in self.checks:
            if path == f"repos/owner/repo/check-runs/{row['id']}":
                row.update(copy.deepcopy(data))
                return copy.deepcopy(row)
        raise AssertionError("Unexpected mock PATCH: " + path)

    def sends(self):
        return [
            call
            for call in self.calls
            if call[0] == "POST" and call[1].endswith("/dispatches")
        ]


class ProducerTests(unittest.TestCase):
    def setUp(self):
        self.api = FakeAPI()

    def handler(self, comment=123):
        return producer.Producer(
            self.api,
            "owner/repo",
            7,
            comment,
            f"https://github.com/owner/repo/pull/7#issuecomment-{comment}",
            "tests/utils",
        )

    def uncertain(self, comment=123):
        return protocol.mark_uncertain(
            protocol.record_attempt(protocol.plan(self.api.request(comment))),
            "lost_response",
        )

    def test_durable_attempt_patch_precedes_dispatch_post(self):
        result = self.handler().dispatch()
        self.assertEqual(result["state"], "completed")
        send = next(
            i
            for i, call in enumerate(self.api.calls)
            if call[1].endswith("/dispatches")
        )
        attempts = [
            (i, producer.decode_marker(call[2]["body"], producer.JOURNAL_MARKER))
            for i, call in enumerate(self.api.calls)
            if call[0] == "PATCH"
        ]
        self.assertTrue(
            any(i < send and row["state"] == "attempt_recorded" for i, row in attempts)
        )
        self.assertEqual(len(self.api.sends()), 1)
        self.assertTrue(self.api.sends()[0][2]["return_run_details"])

    def test_unverified_durable_attempt_write_cannot_dispatch(self):
        original_patch = self.api.patch

        def altered_readback(path, data):
            result = original_patch(path, data)
            result["body"] += " altered"
            return result

        self.api.patch = altered_readback
        with self.assertRaisesRegex(ValueError, "readback differs"):
            self.handler().dispatch()
        self.assertEqual(self.api.sends(), [])
        self.api.patch = original_patch
        self.assertEqual(self.handler().dispatch()["state"], "uncertain")
        self.assertEqual(self.api.sends(), [])

    def test_network_failure_and_malformed_response_never_redispatch(self):
        for failure in (TimeoutError("mock"), None):
            with self.subTest(failure=failure):
                self.api = FakeAPI()
                self.api.dispatch_error = failure
                self.api.dispatch_result = {}
                first = self.handler().dispatch()
                second = self.handler().dispatch()
                self.assertEqual(first["state"], "uncertain")
                self.assertEqual(second["state"], "uncertain")
                self.assertTrue(second["dispatchAttempted"])
                self.assertEqual(len(self.api.sends()), 1)

    def test_exact_recovery_of_cancelled_root_before_setup(self):
        self.api.seed(self.uncertain())
        self.api.runs = [self.api.run_record(456)]
        result = self.handler().dispatch()
        self.assertEqual(protocol.acknowledgment(result)["rootRunIds"], [456])
        self.assertEqual(self.api.sends(), [])

    def test_recovery_requires_exact_key_repository_workflow_and_actor(self):
        variants = [
            {"display_title": "nearby-head-not-an-exact-request-key"},
            {"repository": {"full_name": "other/repo"}},
            {"path": ".github/workflows/other.yml"},
            {"actor": {"login": "someone-else"}},
            {"event": "pull_request"},
        ]
        for change in variants:
            with self.subTest(change=change):
                self.api = FakeAPI()
                self.api.seed(self.uncertain())
                self.api.runs = [{**self.api.run_record(456), **change}]
                try:
                    result = self.handler().dispatch()
                except ValueError:
                    pass
                else:
                    self.assertNotEqual(result["state"], "completed")
                self.assertEqual(self.api.sends(), [])

    def test_interleaved_comment_requests_have_separate_journals(self):
        first = self.handler().dispatch()
        self.api.dispatch_result = self.api.response(789)
        second = self.handler(124).dispatch()
        replay = self.handler().dispatch()
        self.assertEqual(len(self.api.comments), 2)
        self.assertEqual(len(self.api.sends()), 2)
        self.assertEqual(protocol.acknowledgment(first)["rootRunIds"], [456])
        self.assertEqual(protocol.acknowledgment(second)["rootRunIds"], [789])
        self.assertEqual(protocol.acknowledgment(replay)["triggerCommentId"], 123)

    def test_head_change_before_dispatch_records_no_send(self):
        self.api.changed_head_on_second_read = True
        result = self.handler().dispatch()
        self.assertEqual(result["state"], "no-send")
        self.assertFalse(result["dispatchAttempted"])
        self.assertEqual(self.api.sends(), [])
        self.assertIn("changed", result["reason"])
        self.assertEqual(self.handler().dispatch()["state"], "no-send")

    def test_duplicate_journals_fail_closed(self):
        self.api.seed(self.uncertain())
        self.api.seed(self.uncertain())
        with self.assertRaisesRegex(ValueError, "More than one journal"):
            self.handler().dispatch()
        self.assertEqual(self.api.sends(), [])

    def test_two_exact_recovery_roots_remain_uncertain(self):
        self.api.seed(self.uncertain())
        self.api.runs = [self.api.run_record(456), self.api.run_record(789)]
        self.assertEqual(self.handler().dispatch()["state"], "uncertain")
        self.assertEqual(self.api.sends(), [])

    def test_merge_checkout_requires_exactly_two_matching_parents(self):
        for parents in (
            [{"sha": "a" * 40}],
            [{"sha": "a" * 40}, {"sha": "d" * 40}],
            [{"sha": "a" * 40}, {"sha": "b" * 40}, {"sha": "a" * 40}],
        ):
            with self.subTest(parents=parents):
                self.api = FakeAPI()
                self.api.parents = parents
                result = self.handler().dispatch()
                self.assertEqual(result["state"], "no-send")
                self.assertEqual(self.api.sends(), [])

    def test_validate_run_requires_frozen_journal_head_scope_and_no_skips(self):
        journal = protocol.record_response(
            protocol.record_attempt(protocol.plan(self.api.request())),
            self.api.response(456),
        )
        self.api.seed(journal)
        inputs = self.api.request()["dispatchInputs"]
        self.assertEqual(
            producer.validate_run(self.api, "owner/repo", inputs, 456),
            {**self.api.request(), "isFork": True},
        )
        variants = [
            {**inputs, "watcher_head_sha": "d" * 40},
            {**inputs, "watcher_base_sha": "d" * 40},
            {**inputs, "watcher_scope": "tests/other"},
            {**inputs, "skip_aot": True},
            {**inputs, "skip_gpu": "true"},
        ]
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                producer.validate_run(self.api, "owner/repo", value, 456)
        self.api.pr["head"]["sha"] = "d" * 40
        with self.assertRaises(ValueError):
            producer.validate_run(self.api, "owner/repo", inputs, 456)

    def test_fork_flag_requires_positive_internal_repository_identity(self):
        for repository, expected in (
            ("owner/repo", False),
            ("OWNER/REPO", False),
            ("contributor/repo", True),
            (None, True),
        ):
            with self.subTest(repository=repository):
                self.api = FakeAPI()
                self.api.seed(self.uncertain())
                if repository is not None:
                    self.api.pr["head"]["repo"] = {"full_name": repository}
                result = producer.validate_run(
                    self.api, "owner/repo", self.api.request()["dispatchInputs"], 456
                )
                self.assertIs(result["isFork"], expected)

    def test_retry_with_changed_expected_head_never_recovers_or_sends(self):
        for completed in (False, True):
            with self.subTest(completed=completed):
                self.api = FakeAPI()
                journal = self.uncertain()
                if completed:
                    journal = protocol.recover(
                        journal,
                        self.api.request()["requestKey"],
                        self.api.response(456),
                    )
                self.api.seed(journal)
                with self.assertRaisesRegex(ValueError, "Retry changed"):
                    self.handler().dispatch(expected_head="d" * 40)
                self.assertEqual(self.api.sends(), [])
                self.assertFalse(
                    any(method == "PATCH" for method, _, _ in self.api.calls)
                )

    def test_route_requires_explicit_valid_marker_and_preserves_plain_commands(self):
        valid = {
            "requestId": "f" * 64,
            "headSha": "a" * 40,
            "provider": "github",
            "scope": [],
        }
        ordinary = {"comment": {"body": "@flashinfer-bot run"}}
        self.assertIsNone(producer._scope_metadata(ordinary, []))
        marked = {
            "comment": {
                "body": "@flashinfer-bot run\n\n<!-- watcher-ci-request-v1 "
                + json.dumps(valid)
                + " -->"
            }
        }
        self.assertEqual(producer._scope_metadata(marked, []), valid)
        for body in (
            "@flashinfer-bot run\n<!-- watcher-ci-request-v1 broken -->",
            marked["comment"]["body"] + "\n" + marked["comment"]["body"],
        ):
            with self.subTest(body=body), self.assertRaises(ValueError):
                producer._scope_metadata({"comment": {"body": body}}, [])

    def test_partial_watcher_inputs_fail_without_any_provider_request(self):
        with tempfile.TemporaryDirectory() as directory:
            event = Path(directory) / "event.json"
            event.write_text(
                json.dumps(
                    {
                        "repository": {"full_name": "owner/repo"},
                        "inputs": {"watcher_pr_number": "7"},
                    }
                )
            )
            environment = {
                "GITHUB_EVENT_PATH": str(event),
                "GITHUB_REPOSITORY": "owner/repo",
                "GITHUB_RUN_ID": "456",
            }
            with (
                patch.dict(producer.os.environ, environment, clear=True),
                patch.object(producer.sys, "argv", ["producer", "validate-run"]),
                patch.object(producer, "API", return_value=self.api),
                self.assertRaises((KeyError, ValueError)),
            ):
                producer.main()
            self.assertEqual(self.api.calls, [])

    def prepare_report(self):
        journal = protocol.recover(
            self.uncertain(), self.api.request()["requestKey"], self.api.response(456)
        )
        self.api.seed(journal)
        return {"workflow_run": copy.deepcopy(self.api.run)}

    def test_report_mirrors_actual_terminal_jobs_and_run_failure(self):
        self.api.run["conclusion"] = "failure"
        self.api.jobs[0]["conclusion"] = "failure"
        result = producer.report_completed(
            self.api, "owner/repo", self.prepare_report()
        )
        checks = [
            data
            for method, path, data in self.api.calls
            if method == "POST" and path.endswith("/check-runs")
        ]
        statuses = [
            data
            for method, path, data in self.api.calls
            if method == "POST" and "/statuses/" in path
        ]
        self.assertEqual(result["reportedJobs"], 2)
        self.assertEqual(
            [item["conclusion"] for item in checks], ["failure", "failure"]
        )
        self.assertTrue(all(item["head_sha"] == "a" * 40 for item in checks))
        self.assertEqual(statuses[0]["state"], "failure")

    def test_report_requires_terminal_stable_provider_state(self):
        for mode in ("running_root", "running_job", "changed_attempt"):
            with self.subTest(mode=mode):
                self.api = FakeAPI()
                if mode == "running_root":
                    self.api.run["status"] = "in_progress"
                elif mode == "running_job":
                    self.api.jobs[0]["status"] = "in_progress"
                else:
                    self.api.change_run_on_second_read = True
                event = self.prepare_report()
                with self.assertRaises(ValueError):
                    producer.report_completed(self.api, "owner/repo", event)
                self.assertFalse(
                    any(method == "POST" for method, _, _ in self.api.calls)
                )

    def test_superseded_request_cannot_overwrite_newer_request_status(self):
        self.api.status_comment = 124
        result = producer.report_completed(
            self.api, "owner/repo", self.prepare_report()
        )
        self.assertTrue(result["superseded"])
        self.assertEqual(result["reportedJobs"], 0)
        self.assertFalse(any(method == "POST" for method, _, _ in self.api.calls))

    def test_human_rerun_keeps_trusted_original_bot_actor(self):
        self.api.seed(self.uncertain())
        self.api.run["triggering_actor"] = {"login": "human-maintainer"}
        self.api.runs = [copy.deepcopy(self.api.run)]
        result = self.handler().dispatch()
        self.assertEqual(result["state"], "completed")
        request = producer.validate_run(
            self.api, "owner/repo", self.api.request()["dispatchInputs"], 456
        )
        self.assertEqual(request["headSha"], "a" * 40)

    def test_cancelled_before_jobs_reports_blocking_summary(self):
        self.api.jobs = []
        result = producer.report_completed(
            self.api, "owner/repo", self.prepare_report()
        )
        self.assertEqual(result["reportedJobs"], 1)
        self.assertEqual(self.api.checks[0]["name"], "Test Results Summary")
        self.assertIn(self.api.checks[0]["conclusion"], ("failure", "cancelled"))
        states = [
            value["state"]
            for method, path, value in self.api.calls
            if method == "POST" and "/statuses/" in path
        ]
        self.assertEqual(states, ["failure"])

    def test_repeated_report_updates_same_check_identities(self):
        self.api.run["conclusion"] = "success"
        event = self.prepare_report()
        producer.report_completed(self.api, "owner/repo", event)
        producer.report_completed(self.api, "owner/repo", event)
        self.assertEqual(len(self.api.checks), 2)
        posts = [
            call
            for call in self.api.calls
            if call[0] == "POST" and call[1].endswith("/check-runs")
        ]
        updates = [
            call
            for call in self.api.calls
            if call[0] == "PATCH" and "/check-runs/" in call[1]
        ]
        self.assertEqual(len(posts), 2)
        self.assertEqual(len(updates), 2)

    def test_missing_summary_never_publishes_overall_success(self):
        self.api.run["conclusion"] = "success"
        self.api.jobs = [self.api.jobs[0]]
        producer.report_completed(self.api, "owner/repo", self.prepare_report())
        summary = next(
            check
            for check in self.api.checks
            if check["name"] == "Test Results Summary"
        )
        self.assertEqual(summary["conclusion"], "failure")
        states = [
            value["state"]
            for method, path, value in self.api.calls
            if method == "POST" and "/statuses/" in path
        ]
        self.assertEqual(states, ["failure"])

    def metadata(self):
        value = {
            "requestId": "e" * 64,
            "headSha": "a" * 40,
            "provider": "github",
            "scope": ["tests/utils"],
        }
        event = {
            "comment": {
                "body": "@flashinfer-bot run tests/utils\n<!-- "
                + producer.REQUEST_MARKER
                + " "
                + json.dumps(value)
                + " -->"
            }
        }
        return producer._scope_metadata(event, "tests/utils")

    def test_matching_watcher_request_emits_explicit_no_send_proof(self):
        self.api.changed_head_on_second_read = True
        handler = self.handler()
        handler.request_metadata = self.metadata()
        result = handler.dispatch(expected_head="a" * 40)
        self.assertEqual(result["state"], "no-send")
        receipt = producer.decode_marker(
            self.api.comments[0]["body"], "watcher-ci-no-send-v1"
        )
        self.assertEqual(
            receipt,
            {
                "schemaVersion": 1,
                "requestId": "e" * 64,
                "repository": "owner/repo",
                "prNumber": 7,
                "triggerCommentId": 123,
                "triggerCommentUrl": "https://github.com/owner/repo/pull/7#issuecomment-123",
                "requestedHeadSha": "a" * 40,
                "provider": "github",
                "reason": "pre_dispatch_rejected",
                "submissionAttempted": False,
                "rootRunIds": [],
            },
        )
        self.assertEqual(self.api.sends(), [])

    def test_attempted_uncertain_or_completed_requests_never_emit_no_send_proof(self):
        for failure in (True, False):
            with self.subTest(failure=failure):
                self.api = FakeAPI()
                if failure:
                    self.api.dispatch_error = TimeoutError("mock")
                handler = self.handler()
                handler.request_metadata = self.metadata()
                result = handler.dispatch(expected_head="a" * 40)
                self.assertTrue(result["dispatchAttempted"])
                for method, _path, value in self.api.calls:
                    if method in ("POST", "PATCH") and value and "body" in value:
                        self.assertIsNone(
                            producer.decode_marker(
                                value["body"], "watcher-ci-no-send-v1"
                            )
                        )

    def test_no_send_proof_cannot_be_rebound_to_another_head(self):
        handler = self.handler()
        handler.request_metadata = {**self.metadata(), "headSha": "d" * 40}
        journal = protocol.no_send(protocol.plan(self.api.request()), "not_ready")
        with self.assertRaisesRegex(ValueError, "No-send proof"):
            handler.save(journal)
        self.assertEqual(self.api.calls, [])

    def test_required_pending_check_is_published_before_attempt_and_dispatch(self):
        handler = self.handler()
        handler.status_api = self.api
        self.assertEqual(handler.dispatch()["state"], "completed")
        check_index = next(
            i
            for i, (method, path, _) in enumerate(self.api.calls)
            if method == "POST" and path.endswith("/check-runs")
        )
        attempt_index = next(
            i
            for i, (method, path, value) in enumerate(self.api.calls)
            if method == "PATCH"
            and producer.decode_marker(value.get("body"), producer.JOURNAL_MARKER)[
                "state"
            ]
            == "attempt_recorded"
        )
        dispatch_index = next(
            i
            for i, (_, path, _) in enumerate(self.api.calls)
            if path.endswith("/dispatches")
        )
        self.assertLess(check_index, attempt_index)
        self.assertLess(attempt_index, dispatch_index)
        check = self.api.checks[0]
        self.assertEqual(check["name"], "Test Results Summary")
        self.assertEqual(check["status"], "in_progress")
        self.assertEqual(check["head_sha"], "a" * 40)
        self.assertEqual(check["external_id"], "fi-ci/123/summary")

    def test_current_head_discovery_excludes_old_journals_and_keeps_human_reruns(self):
        request = self.api.request()
        old = protocol.build_request(
            "owner/repo",
            7,
            124,
            "https://github.com/owner/repo/pull/7#issuecomment-124",
            "d" * 40,
            "b" * 40,
            "c" * 40,
            "tests/utils",
        )
        self.api.seed(
            protocol.record_response(
                protocol.record_attempt(protocol.plan(old)), self.api.response(789)
            )
        )
        self.api.seed(
            protocol.record_response(
                protocol.record_attempt(protocol.plan(request)), self.api.response(456)
            )
        )
        self.api.run["triggering_actor"] = {"login": "human-maintainer"}
        self.assertEqual(producer.correlated_run_ids(self.api, "owner/repo", 7), [456])
        self.assertEqual(
            producer.correlated_run_ids(self.api, "owner/repo", 7, "cancelled"), [456]
        )
        self.assertEqual(
            producer.correlated_run_ids(self.api, "owner/repo", 7, "active"), []
        )
        self.assertFalse(any(method != "GET" for method, _, _ in self.api.calls))

    def test_current_head_discovery_rejects_unrelated_provider_root(self):
        self.api.seed(
            protocol.record_response(
                protocol.record_attempt(protocol.plan(self.api.request())),
                self.api.response(456),
            )
        )
        self.api.run["repository"] = {"full_name": "other/repo"}
        with self.assertRaisesRegex(ValueError, "unrelated provider run"):
            producer.correlated_run_ids(self.api, "owner/repo", 7)
        self.assertFalse(any(method != "GET" for method, _, _ in self.api.calls))

    def test_no_send_after_pending_status_failure_does_not_orphan_required_check(self):
        real_post = self.api.post

        def failing_status(path, data):
            if "/statuses/" in path:
                raise TimeoutError("status publication failed")
            return real_post(path, data)

        self.api.post = failing_status
        handler = self.handler()
        handler.status_api = self.api
        result = handler.dispatch()
        self.assertEqual(result["state"], "no-send")
        self.assertEqual(self.api.sends(), [])
        self.assertTrue(
            all(
                check["status"] == "completed"
                and check.get("conclusion") in ("failure", "cancelled")
                for check in self.api.checks
            )
        )

    def test_planned_handler_retry_upserts_existing_required_summary(self):
        self.api.seed(protocol.plan(self.api.request()))
        producer.summary_check(self.api, "owner/repo", self.api.request())
        check_id = self.api.checks[0]["id"]
        handler = self.handler()
        handler.status_api = self.api
        self.assertEqual(handler.dispatch()["state"], "completed")
        self.assertEqual(len(self.api.checks), 1)
        self.assertEqual(self.api.checks[0]["id"], check_id)
        self.assertEqual(len(self.api.sends()), 1)

    def test_ui_rerun_resets_same_required_check_to_pending(self):
        event = self.prepare_report()
        self.api.run["conclusion"] = "success"
        producer.report_completed(self.api, "owner/repo", event)
        summary = next(
            c for c in self.api.checks if c["name"] == "Test Results Summary"
        )
        summary_id = summary["id"]
        self.api.run.update(
            status="in_progress",
            conclusion=None,
            run_attempt=2,
            triggering_actor={"login": "human-maintainer"},
        )
        result = producer.report_progress(
            self.api,
            "owner/repo",
            {"action": "in_progress", "workflow_run": copy.deepcopy(self.api.run)},
        )
        self.assertTrue(result["pending"])
        current = next(c for c in self.api.checks if c["id"] == summary_id)
        self.assertEqual(current["status"], "in_progress")
        self.assertEqual(
            len([c for c in self.api.checks if c["name"] == "Test Results Summary"]), 1
        )
        statuses = [
            data
            for method, path, data in self.api.calls
            if method == "POST" and "/statuses/" in path
        ]
        self.assertEqual(statuses[-1]["state"], "pending")

    def test_superseded_progress_cannot_reset_newer_request(self):
        self.prepare_report()
        self.api.status_comment = 124
        self.api.run["status"] = "in_progress"
        result = producer.report_progress(
            self.api,
            "owner/repo",
            {"action": "in_progress", "workflow_run": copy.deepcopy(self.api.run)},
        )
        self.assertTrue(result["superseded"])
        self.assertFalse(
            any(method in ("POST", "PATCH") for method, _, _ in self.api.calls)
        )

    def test_late_progress_event_cannot_reset_completed_root(self):
        event = self.prepare_report()
        result = producer.report_progress(
            self.api, "owner/repo", {**event, "action": "in_progress"}
        )
        self.assertIn("ignored", result)
        self.assertFalse(
            any(method in ("POST", "PATCH") for method, _, _ in self.api.calls)
        )

    def test_synthetic_summary_omits_null_timestamps_and_skipped_is_failure(self):
        for jobs in (
            [],
            [
                {
                    "id": 2,
                    "name": "Test Results Summary",
                    "status": "completed",
                    "conclusion": "skipped",
                    "html_url": self.api.run["html_url"],
                }
            ],
        ):
            with self.subTest(jobs=jobs):
                self.api = FakeAPI()
                self.api.run["conclusion"] = "success"
                self.api.jobs = jobs
                producer.report_completed(self.api, "owner/repo", self.prepare_report())
                summary = next(
                    c for c in self.api.checks if c["name"] == "Test Results Summary"
                )
                self.assertEqual(summary["conclusion"], "failure")
                for method, path, data in self.api.calls:
                    if method in ("POST", "PATCH") and "/check-runs" in path:
                        self.assertNotIn("started_at", data)
                        self.assertNotIn("completed_at", data)

    def test_recovery_search_is_bounded_by_original_comment_not_matched_by_time(self):
        self.api.seed(self.uncertain())
        self.api.runs = [self.api.run_record(456)]
        self.assertEqual(self.handler().dispatch()["state"], "completed")
        queries = [
            path
            for method, path, _ in self.api.calls
            if method == "GET" and "/workflows/pr-test.yml/runs?" in path
        ]
        self.assertEqual(len(queries), 1)
        self.assertIn("created=%3E%3D2026-10-14T02%3A00%3A00Z", queries[0])

    def test_missing_or_malformed_original_timestamp_cannot_recover_or_send(self):
        for timestamp in (None, "", "not-a-timestamp", "2026-10-14"):
            with self.subTest(timestamp=timestamp):
                self.api = FakeAPI()
                self.api.comment_created_at = timestamp
                self.api.seed(self.uncertain())
                self.api.runs = [self.api.run_record(456)]
                with self.assertRaises((ValueError, TypeError)):
                    self.handler().dispatch()
                self.assertFalse(
                    any(method in ("POST", "PATCH") for method, _, _ in self.api.calls)
                )

    def test_failed_rerun_response_clears_pending_without_claiming_old_success(self):
        event = self.prepare_report()
        self.api.run["conclusion"] = "success"
        producer.report_progress(
            self.api, "owner/repo", {**event, "action": "prepare-rerun"}
        )
        result = producer.report_progress(
            self.api, "owner/repo", {**event, "action": "rerun-submit-failed"}
        )
        self.assertFalse(result["pending"])
        self.assertTrue(result["uncertainRerun"])
        summary = self.api.checks[0]
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["conclusion"], "failure")
        statuses = [
            data
            for method, path, data in self.api.calls
            if method == "POST" and "/statuses/" in path
        ]
        self.assertEqual(statuses[-1]["state"], "failure")
        self.assertIn("unavailable", statuses[-1]["description"])
        self.assertEqual(self.api.sends(), [])

    def test_progress_cli_invokes_real_progress_path(self):
        self.prepare_report()
        self.api.run["status"] = "in_progress"
        with tempfile.TemporaryDirectory() as directory:
            event = Path(directory) / "event.json"
            event.write_text(
                json.dumps(
                    {
                        "repository": {"full_name": "owner/repo"},
                        "action": "in_progress",
                        "workflow_run": self.api.run,
                    }
                )
            )
            environment = {
                "GITHUB_EVENT_PATH": str(event),
                "GITHUB_REPOSITORY": "owner/repo",
            }
            output = io.StringIO()
            with (
                patch.dict(producer.os.environ, environment, clear=True),
                patch.object(producer.sys, "argv", ["producer", "report-progress"]),
                patch.object(producer, "API", return_value=self.api),
                redirect_stdout(output),
            ):
                producer.main()
            self.assertTrue(json.loads(output.getvalue())["pending"])
            self.assertEqual(self.api.checks[0]["status"], "in_progress")

    def test_validate_run_during_uncertain_send_rejects_unrelated_provider_run(self):
        variants = [
            {"display_title": "another-request"},
            {"repository": {"full_name": "other/repo"}},
            {"path": ".github/workflows/other.yml"},
            {"actor": {"login": "someone-else"}},
            {"event": "pull_request"},
        ]
        for change in variants:
            with self.subTest(change=change):
                self.api = FakeAPI()
                self.api.seed(self.uncertain())
                self.api.run.update(change)
                with self.assertRaises(ValueError):
                    producer.validate_run(
                        self.api,
                        "owner/repo",
                        self.api.request()["dispatchInputs"],
                        456,
                    )
                self.assertEqual(self.api.sends(), [])


if __name__ == "__main__":
    unittest.main()
