"""No-network tests for exact CI request correlation and durable-send ordering."""

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ci_request_protocol", ROOT / "scripts/pr_checks/ci_request_protocol.py"
)
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


class ProtocolTests(unittest.TestCase):
    def request(self, comment=123, scope="tests/utils"):
        return p.build_request(
            "owner/repo",
            7,
            comment,
            f"https://github.com/owner/repo/pull/7#issuecomment-{comment}",
            "a" * 40,
            "b" * 40,
            "c" * 40,
            scope,
        )

    def response(self, number=456):
        return {
            "workflow_run_id": number,
            "run_url": f"https://api.github.com/repos/owner/repo/actions/runs/{number}",
            "html_url": f"https://github.com/owner/repo/actions/runs/{number}",
        }

    def test_normalized_key_includes_comment_head_and_scope(self):
        request = self.request(scope=["tests/z", "tests/a", "tests/z"])
        self.assertEqual(request, self.request(scope="tests/a tests/z"))
        self.assertNotEqual(request["requestKey"], self.request(124)["requestKey"])
        other = p.build_request(
            "owner/repo",
            7,
            123,
            "https://github.com/owner/repo/pull/7#issuecomment-123",
            "d" * 40,
            "b" * 40,
            "c" * 40,
            "tests/a tests/z",
        )
        self.assertNotEqual(request["requestKey"], other["requestKey"])

    def test_request_key_is_bounded_and_carries_full_scope_and_head(self):
        request = self.request()
        self.assertLess(len(request["requestKey"]), 200)
        self.assertIn("a" * 40, request["requestKey"])
        self.assertTrue(request["requestKey"].endswith(request["scopeDigest"]))

    def test_inconsistent_persisted_journal_and_boolean_receipt_rejected(self):
        journal = p.record_attempt(p.plan(self.request()))
        journal["dispatchAttempted"] = False
        with self.assertRaises(ValueError):
            p.recover(journal, self.request()["requestKey"], self.response())
        receipt = p.completed_receipt(self.request(), [1])
        receipt["schemaVersion"] = True
        with self.assertRaises(ValueError):
            p.receipt_body(receipt)

    def test_payload_uses_pinned_inputs_and_returned_run_details(self):
        request = self.request()
        payload = p.dispatch_payload(request, "main")
        self.assertEqual(payload["ref"], "main")
        self.assertIs(payload["return_run_details"], True)
        self.assertEqual(payload["inputs"]["watcher_checkout_sha"], "c" * 40)
        payload["inputs"]["watcher_head_sha"] = "changed"
        self.assertEqual(request["dispatchInputs"]["watcher_head_sha"], "a" * 40)

    def test_complete_path_and_exact_receipt_schema(self):
        planned = p.plan(self.request())
        attempted = p.record_attempt(planned)
        self.assertFalse(planned["dispatchAttempted"])
        self.assertEqual(attempted["state"], "attempt_recorded")
        completed = p.record_response(attempted, self.response())
        receipt = p.acknowledgment(completed)
        self.assertEqual(receipt["rootRunIds"], [456])
        self.assertIs(receipt["rootSubmissionComplete"], True)
        self.assertEqual(len(receipt), 9)
        self.assertTrue(
            p.receipt_body(receipt).startswith("<!-- watcher-ci-receipt-v1 {")
        )
        self.assertEqual(receipt["requestedHeadSha"], "a" * 40)

    def test_response_requires_prior_attempt(self):
        with self.assertRaises(ValueError):
            p.record_response(p.plan(self.request()), self.response())

    def test_ambiguous_send_is_never_retried_or_changed_to_no_send(self):
        journal = p.mark_uncertain(
            p.record_attempt(p.plan(self.request())), "transport_error"
        )
        self.assertEqual(journal["state"], "uncertain")
        for fn in (
            lambda: p.record_attempt(journal),
            lambda: p.no_send(journal, "timeout"),
            lambda: p.acknowledgment(journal),
        ):
            with self.assertRaises(ValueError):
                fn()

    def test_exact_key_recovery_only(self):
        journal = p.mark_uncertain(
            p.record_attempt(p.plan(self.request())), "lost_response"
        )
        with self.assertRaises(ValueError):
            p.recover(journal, self.request(124)["requestKey"], self.response())
        fixed = p.recover(journal, self.request()["requestKey"], self.response())
        self.assertEqual(p.acknowledgment(fixed)["rootRunIds"], [456])

    def test_partial_multi_root_submission_is_not_complete(self):
        journal = p.plan(self.request(), root_count=2)
        journal = p.record_response(p.record_attempt(journal, 0), self.response(1), 0)
        self.assertEqual(journal["state"], "partial")
        with self.assertRaises(ValueError):
            p.acknowledgment(journal)
        journal = p.record_response(p.record_attempt(journal, 1), self.response(2), 1)
        self.assertEqual(p.acknowledgment(journal)["rootRunIds"], [1, 2])

    def test_duplicate_roots_cannot_fill_multiple_slots(self):
        journal = p.plan(self.request(), root_count=2)
        journal = p.record_response(p.record_attempt(journal), self.response())
        attempted = p.record_attempt(journal, 1)
        with self.assertRaises(ValueError):
            p.record_response(attempted, self.response(), 1)
        self.assertEqual(attempted["slots"][1]["state"], "attempt_recorded")

    def test_interleaved_requests_keep_distinct_identity(self):
        first = p.record_attempt(p.plan(self.request()))
        second = p.record_attempt(p.plan(self.request(124)))
        first = p.record_response(first, self.response(1))
        second = p.record_response(second, self.response(2))
        self.assertEqual(p.acknowledgment(first)["triggerCommentId"], 123)
        self.assertEqual(p.acknowledgment(second)["rootRunIds"], [2])

    def test_cancelled_root_before_setup_remains_exact_root(self):
        response = {**self.response(), "status": "completed", "conclusion": "cancelled"}
        journal = p.record_response(p.record_attempt(p.plan(self.request())), response)
        self.assertEqual(p.acknowledgment(journal)["rootRunIds"], [456])

    def test_repeated_handler_attempt_is_rejected_after_acceptance(self):
        journal = p.record_response(
            p.record_attempt(p.plan(self.request())), self.response()
        )
        with self.assertRaises(ValueError):
            p.record_attempt(journal)
        with self.assertRaises(ValueError):
            p.recover(journal, self.request()["requestKey"], self.response())

    def test_no_send_only_before_attempt(self):
        journal = p.no_send(p.plan(self.request()), "head_changed_before_dispatch")
        self.assertIs(journal["dispatchAttempted"], False)
        with self.assertRaises(ValueError):
            p.record_attempt(journal)
        with self.assertRaises(ValueError):
            p.no_send(p.record_attempt(p.plan(self.request())), "not_sent")

    def test_malformed_response_leaves_durable_attempt_unresolved(self):
        attempted = p.record_attempt(p.plan(self.request()))
        for response in (
            {},
            None,
            {**self.response(), "workflow_run_id": True},
            {**self.response(), "html_url": "https://evil.test/"},
        ):
            with self.subTest(response=response), self.assertRaises(ValueError):
                p.record_response(attempted, response)
            self.assertEqual(attempted["state"], "attempt_recorded")
            with self.assertRaises(ValueError):
                p.record_attempt(attempted)

    def test_invalid_scope_identity_and_forged_receipt_fail(self):
        for scope in ("tests/../secret", "tests/a;echo", "docs/file", [True]):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                self.request(scope=scope)
        request = self.request()
        request["dispatchInputs"]["watcher_head_sha"] = "d" * 40
        with self.assertRaises(ValueError):
            p.dispatch_payload(request, "main")
        receipt = p.completed_receipt(self.request(), [1])
        receipt["rootSubmissionComplete"] = False
        with self.assertRaises(ValueError):
            p.receipt_body(receipt)


if __name__ == "__main__":
    unittest.main()
