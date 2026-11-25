"""CPU-only tests for the bot-response merge gate; never execute PR code."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/infra/pr_checks"))
sys.path.insert(0, str(ROOT / "tests/infra/test_sharding"))
import check_bot_run_request as gate
import test_ai_test_scope as fixtures
from test_ai_test_scope import target


def comment(body, bot=False, ident=1):
    return {
        "id": ident,
        "body": body,
        "user": {
            "login": "flashinfer-bot" if bot else "contributor",
            "id": 236526959 if bot else 12,
        },
        "html_url": f"https://github.com/flashinfer-ai/flashinfer/pull/42#issuecomment-{ident}",
    }


ACK = "GitLab MR [!3051](https://nv/flashinfer/-/merge_requests/3051) has been created, and the CI pipeline [#84885886](https://nv/flashinfer-ci/-/pipelines/84885886) is currently running. I'll report back once the pipeline job completes."
ADVICE = "**AI found no pytest-applicable behavior change** (PR head `3ac003a01b20`):\n\nNo CI pipeline was started. Review the AI decision and test coverage.\nChoose test paths with `/bot run tests/<dir-or-file>` after review."


def analysis(kind="gpu_required", targets=None):
    return {
        "classification": kind,
        "reason": "Changed backend dispatch"
        if kind == "gpu_required"
        else "Pure CPU helper",
        "evidence_paths": ["flashinfer/rope.py"],
        "case_gap": "No corresponding case selected",
        "targets": targets or [],
    }


class GateTests(unittest.TestCase):
    def test_command_whitespace_and_arbitrary_args(self):
        for body in [
            "/bot run",
            "/bot  run xxx test",
            "/BOT\trun tests/a.py tests/b.py",
            "/bot run all",
        ]:
            self.assertTrue(gate.is_run_command(body), body)
        for body in [
            "/bot runner",
            "Use /bot run",
            "> /bot run",
            "```\n/bot run\n```",
            "`/bot run`",
            "@flashinfer-bot run",
        ]:
            self.assertFalse(gate.is_run_command(body), body)

    def test_existing_replies_skip_analysis_including_old_sha(self):
        for reply in [
            "**Warning**: Failed to sync latest changes. Please try again.",
            "**Error**: Failed to update branch. Please check logs.",
            "The PR head or base changed before CI launch. No pipeline was started. Retry `/bot run`.",
            ACK,
            ADVICE,
            "Invalid `/bot run` path(s): xxx",
            "@user is not authorized to trigger this CI job.",
            "**Mirroring Failed**\n\nFailed to mirror PR to GitLab. Check logs for details.",
        ]:
            analyze = Mock(side_effect=AssertionError("Should skip the LLM"))
            result = gate.evaluate(
                [comment("/bot run"), comment(reply, True, 2)], analyze
            )
            self.assertEqual("success", result.conclusion, reply)
            self.assertEqual("bot_response_found", result.reason)
            self.assertIn("issuecomment-2", result.summary)
            analyze.assert_not_called()

    def test_scope_notes_and_stacked_pr_advice_are_valid_responses(self):
        for body in [
            "**Test scope**\nSelected paths and cost\n\n---\n\n" + ACK,
            "**Stacked PR** (2 of 3)\n\n" + ADVICE,
            ADVICE.replace("AI found", "AI-suggested"),
        ]:
            self.assertTrue(gate.is_bot_response(comment(body, True)), body)

    def test_unrelated_forged_and_public_notes_do_not_bypass(self):
        forged = comment(ACK, True)
        forged["user"]["id"] = 12
        for reply in [
            comment(ACK),
            forged,
            comment("Documentation check complete", True),
            comment(ADVICE.replace("/bot run", "@flashinfer-bot run"), True),
            comment("> " + ACK, True),
            comment("```\n" + ACK + "\n```", True),
            comment(ADVICE + "\n<!-- flashinfer-bot:ci-bot-commands-note -->", True),
        ]:
            result = gate.evaluate([comment("/bot run"), reply], lambda: analysis())
            self.assertEqual("failure", result.conclusion)

    def test_gpu_with_no_selected_case_still_fails(self):
        result = gate.evaluate([], lambda: analysis())
        self.assertEqual("command_missing", result.reason)
        self.assertIn("No corresponding case selected", result.summary)

    def test_waiting_message_changes_after_command(self):
        result = gate.evaluate([comment("/bot  run xxx test")], lambda: analysis())
        self.assertEqual("failure", result.conclusion)
        self.assertEqual("waiting_for_bot", result.reason)
        self.assertIn("issuecomment-1", result.summary)
        self.assertIn("Waiting", result.title)

    def test_cpu_and_docs_pass_without_reply(self):
        for kind in ["cpu_only", "docs_only"]:
            self.assertEqual(
                "success", gate.evaluate([], lambda: analysis(kind)).conclusion
            )

    def test_unknown_and_api_model_errors_fail_without_claiming_gpu(self):
        result = gate.evaluate([], lambda: analysis("unknown"))
        self.assertEqual("failure", result.conclusion)
        self.assertIn("Unable", result.title)
        result = gate.evaluate(
            [],
            Mock(side_effect=gate.TargetSelectionError("Inference request timed out")),
        )
        self.assertEqual("failure", result.conclusion)
        self.assertIn("timed out", result.summary)

    def test_deleted_reply_recomputes_and_new_head_cpu_pass_is_not_permanent(self):
        self.assertEqual(
            "success",
            gate.evaluate(
                [comment("/bot run"), comment(ACK, True, 2)], Mock()
            ).conclusion,
        )
        self.assertEqual(
            "failure",
            gate.evaluate([comment("/bot run")], lambda: analysis()).conclusion,
        )
        self.assertEqual(
            "success", gate.evaluate([], lambda: analysis("cpu_only")).conclusion
        )
        self.assertEqual("failure", gate.evaluate([], lambda: analysis()).conclusion)


class ImpactTests(unittest.TestCase):
    def setUp(self):
        # Reuse the existing selection fixture to verify the opt-in analysis
        # inspects exactly the same pinned sources and added tests.
        self.fixture = fixtures.PytestTargetSelectorTests()
        self.fixture.setUp()

    def select(self, impact, outcome="needs_maintainer", targets=None):
        f = self.fixture
        f.inference.outputs[1] = {
            "outcome": outcome,
            "targets": targets or [],
            "reason": "No corresponding case found",
            "gpu_impact": impact,
        }
        from test_ai_test_scope import PR

        return f.selector.select(PR, include_gpu_impact=True)

    def test_gpu_is_independent_from_missing_selection(self):
        result = self.select(analysis())
        self.assertEqual([], result.paths)
        self.assertEqual("gpu_required", result.gpu_impact["classification"])

    def test_cpu_and_docs_impacts_require_complete_diff(self):
        for kind in ["cpu_only", "docs_only"]:
            self.setUp()
            self.assertEqual(
                kind, self.select(analysis(kind)).gpu_impact["classification"]
            )
        self.setUp()
        self.fixture.github.get_pr_files.return_value[0]["patch"] = None
        self.assertEqual(
            "unknown", self.select(analysis("cpu_only")).gpu_impact["classification"]
        )

    def test_impact_rejects_invented_evidence_and_missing_schema(self):
        for value in [
            {},
            {**analysis(), "evidence_paths": ["invented.py"]},
            {**analysis(), "classification": "safe"},
        ]:
            self.setUp()
            with self.assertRaises(gate.TargetSelectionError):
                self.select(value)

    def test_selected_cpu_tests_do_not_override_gpu_impact(self):
        result = self.select(analysis(), "selected", [target()])
        self.assertEqual("gpu_required", result.gpu_impact["classification"])
        self.assertEqual(1, len(result.paths))

    def test_added_gpu_test_is_inspected_before_the_model_request(self):
        f = self.fixture
        path = "tests/new/test_gpu.py"
        f.github.get_repository_tree.return_value["tree"].append(
            {"path": path, "sha": "new-test", "mode": "100644", "type": "blob"}
        )
        previous = f.github.get_repository_blob.side_effect
        f.github.get_repository_blob.side_effect = (
            lambda repo, sha, limit: "def test_cuda():\n assert launch_cuda()\n"
            if sha == "new-test"
            else previous(repo, sha, limit)
        )
        f.github.get_pr_files.return_value.append(
            {
                "filename": path,
                "status": "added",
                "patch": "+def test_cuda():\n+ assert launch_cuda()",
            }
        )
        result = self.select({**analysis(), "evidence_paths": [path]})
        self.assertEqual("gpu_required", result.gpu_impact["classification"])
        self.assertIn(path, f.inference.calls[0][0]["source_files"])
        self.assertIn("GPU", f.inference.calls[1][1])
        self.assertIn("independent", f.inference.calls[1][1])


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.pr = {
            "number": 42,
            "state": "open",
            "base": {"ref": "main"},
            "head": {"sha": "a" * 40},
            "changed_files": 1,
        }
        self.client = Mock()
        self.client.get_pr_details.side_effect = lambda number: __import__(
            "copy"
        ).deepcopy(self.pr)
        self.client.get_current_base_sha.return_value = "b" * 40
        self.client.checks.return_value = []
        self.client.comments.return_value = []
        self.client.get_pr_files.return_value = [
            {"filename": "README.md", "status": "modified"}
        ]

    def test_docs_fast_path_and_no_conversation_writes(self):
        from unittest.mock import patch

        with patch.object(
            gate, "PytestTargetSelector", side_effect=AssertionError("docs need no LLM")
        ):
            result = gate.run_pr(self.client, 42)
        self.assertEqual("docs_only", result.reason)
        self.client.publish.assert_called_once()
        self.client.add_comment.assert_not_called()

    def test_historical_reply_skips_all_diff_and_model_calls(self):
        self.client.comments.return_value = [comment(ADVICE, True)]
        result = gate.run_pr(self.client, 42)
        self.assertEqual("bot_response_found", result.reason)
        self.client.get_pr_files.assert_not_called()

    def test_read_failure_updates_check_as_failure_not_command_missing(self):
        self.client.comments.side_effect = gate.TargetSelectionError(
            "Comment API failed"
        )
        result = gate.run_pr(self.client, 42)
        self.assertEqual("evidence_error", result.reason)
        self.assertEqual("failure", result.conclusion)
        self.client.publish.assert_called_once()

    def test_push_during_analysis_does_not_publish_old_verdict(self):
        self.client.get_pr_details.side_effect = [
            self.pr,
            {**self.pr, "head": {"sha": "c" * 40}},
        ]
        self.assertIsNone(gate.run_pr(self.client, 42))
        self.client.publish.assert_not_called()

    def test_base_advance_during_analysis_does_not_publish_stale_exemption(self):
        self.client.get_current_base_sha.side_effect = ["b" * 40, "c" * 40]
        self.assertIsNone(gate.run_pr(self.client, 42))
        self.client.publish.assert_not_called()

    def test_analysis_timeout_restores_publication_budget(self):
        from unittest.mock import patch

        self.client.deadline = 12345
        self.client.get_pr_files.return_value = [{"filename": "kernel.cu"}]

        def timeout(*args, **kwargs):
            self.client.deadline = 0
            raise gate.TargetSelectionError("Inference deadline exceeded")

        selector = Mock()
        selector.select.side_effect = timeout
        with patch.object(gate, "PytestTargetSelector", return_value=selector):
            result = gate.run_pr(self.client, 42)
        self.assertEqual(12345, self.client.deadline)
        self.assertEqual("failure", result.conclusion)
        self.client.publish.assert_called_once()

    def test_check_publication_targets_pr_head_and_updates_existing(self):
        client = gate.Client("upstream/flashinfer")
        client.request = Mock(return_value={"id": 23})
        result = gate.Result(
            "failure",
            "waiting_for_bot",
            "Waiting for flashinfer-bot",
            "Command detected",
        )
        client.publish(self.pr, result, base_sha="b" * 40)
        args = client.request.call_args.args
        self.assertEqual("a" * 40, args[2]["head_sha"])
        self.assertEqual(gate.CHECK_NAME, args[2]["name"])
        client.publish(self.pr, result, check_id=23, base_sha="b" * 40)
        self.assertEqual("PATCH", client.request.call_args.args[1])
        self.assertNotIn("head_sha", client.request.call_args.args[2])

    def test_comment_pagination_and_no_partial_success(self):
        client = gate.Client("upstream/flashinfer")
        client.request = Mock(
            side_effect=[[comment("unrelated")] * 100, [comment(ACK, True)]]
        )
        self.assertEqual(101, len(client.comments(42)))
        self.assertIn("page=2", client.request.call_args.args[0])
        client.request = Mock(side_effect=gate.TargetSelectionError("API failure"))
        with self.assertRaises(gate.TargetSelectionError):
            client.comments(42)

    def test_cached_classification_is_bound_to_base_and_analysis_version(self):
        import json

        checks = [
            {
                "id": 10,
                "name": gate.CHECK_NAME,
                "external_id": gate.VERSION + ":42",
                "app": {"slug": "github-actions"},
                "output": {
                    "text": json.dumps(
                        {
                            "base_sha": "b" * 40,
                            "version": gate.analysis_version(),
                            "analysis": analysis("cpu_only"),
                        }
                    )
                },
            }
        ]
        self.assertEqual(
            "cpu_only",
            gate.cached_analysis(checks, self.pr, "b" * 40)["classification"],
        )
        self.assertIsNone(gate.cached_analysis(checks, self.pr, "c" * 40))
        checks[0]["external_id"] = gate.VERSION + ":43"
        self.assertIsNone(gate.cached_analysis(checks, self.pr, "b" * 40))

    def test_response_success_without_analysis_is_not_a_classification_cache(self):
        import json

        checks = [
            {
                "id": 11,
                "name": gate.CHECK_NAME,
                "external_id": gate.VERSION + ":42",
                "app": {"slug": "github-actions"},
                "output": {
                    "text": json.dumps(
                        {
                            "base_sha": "b" * 40,
                            "version": gate.analysis_version(),
                            "reason": "bot_response_found",
                            "analysis": None,
                        }
                    )
                },
            }
        ]
        self.assertIsNone(gate.cached_analysis(checks, self.pr, "b" * 40))
        self.client.checks.return_value = checks
        self.client.comments.return_value = [comment(ACK, True)]
        self.assertEqual("success", gate.run_pr(self.client, 42).conclusion)

    def test_documentation_rename_from_gpu_source_is_not_exempt(self):
        self.assertFalse(
            gate.docs_only(
                [{"filename": "docs/new.md", "previous_filename": "kernel.cu"}], 1
            )
        )
        self.assertFalse(gate.docs_only([{"filename": "docs/new.md"}], 2))
        self.assertFalse(gate.docs_only([{"filename": "docs/setup.py"}], 1))

    def test_workflow_trust_and_event_contract(self):
        source = (ROOT / ".github/workflows/bot-run-request-check.yml").read_text()
        self.assertIn("pull_request_target:", source)
        self.assertIn("types: [created, edited, deleted]", source)
        self.assertIn("ref: ${{ github.event.repository.default_branch }}", source)
        self.assertIn("persist-credentials: false", source)
        self.assertIn("checks: write", source)
        self.assertIn("pull-requests: read", source)
        self.assertNotIn("pull-requests: write", source)
        self.assertNotIn("github.event.pull_request.head", source)


if __name__ == "__main__":
    unittest.main()
