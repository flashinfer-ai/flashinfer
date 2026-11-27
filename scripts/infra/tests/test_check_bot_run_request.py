"""CPU-only tests for the bot-response merge gate; never execute PR code."""

import sys
import time
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

    def test_actual_public_scope_note_is_not_an_internal_response(self):
        from ai_test_scope import TargetSelection

        public_note = TargetSelection(
            [], outcome="no_pytest_change", reason="Only documentation changed"
        ).note("a" * 40)
        self.assertIn("@flashinfer-bot run all", public_note)
        self.assertFalse(gate.is_bot_response(comment(public_note, True)))
        # Internal src/pytest_targets.py uses /bot run (verified against service source).
        self.assertTrue(
            gate.is_bot_response(
                comment(ADVICE + "\nUse `/bot run all` for the full suite.", True)
            )
        )

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

    def test_impact_prompt_distinguishes_host_build_from_device_behavior(self):
        self.select(analysis("cpu_only"))
        prompt = " ".join(self.fixture.inference.calls[1][1].split())
        self.assertIn("CUDA compilation can run entirely on the host", prompt)
        self.assertIn("host-only CI orchestration", prompt)
        self.assertIn("a CPU runner does not justify an exemption", prompt)

    def test_cpu_and_docs_impacts_allow_missing_patch_with_note(self):
        for kind in ["cpu_only", "docs_only"]:
            self.setUp()
            self.assertEqual(
                kind, self.select(analysis(kind)).gpu_impact["classification"]
            )
        self.setUp()
        self.fixture.github.get_pr_files.return_value[0]["patch"] = None
        self.assertEqual(
            "cpu_only", self.select(analysis("cpu_only")).gpu_impact["classification"]
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

    def test_experimental_opt_in_outside_experimental_tree(self):
        result = self.select(
            {
                **analysis("experimental_only"),
                "reason": "Adds an explicit opt-in entry point; default dispatch is unchanged.",
            }
        )
        self.assertEqual("experimental_only", result.gpu_impact["classification"])
        verdict = gate.evaluate([], lambda: result.gpu_impact)
        self.assertEqual(
            ("success", "experimental_only"), (verdict.conclusion, verdict.reason)
        )

    def test_experimental_opt_in_needs_inspected_core_diff(self):
        self.fixture.github.get_pr_files.return_value[0]["patch"] = None
        result = self.select(analysis("experimental_only"))
        self.assertEqual("unknown", result.gpu_impact["classification"])
        self.assertIn("flashinfer/rope.py", result.gpu_impact["reason"])

    def test_experimental_opt_in_does_not_require_every_outside_diff_as_a_citation(
        self,
    ):
        f = self.fixture
        f.github.get_pr_files.return_value.append(
            {
                "filename": "examples/try_opt_in.py",
                "status": "added",
                "patch": "+from flashinfer.rope import experimental_rope",
            }
        )
        result = self.select(analysis("experimental_only"))
        self.assertEqual("experimental_only", result.gpu_impact["classification"])
        self.assertEqual(["flashinfer/rope.py"], result.gpu_impact["evidence_paths"])

    def test_experimental_opt_in_can_cite_full_source_of_added_file(self):
        changed = self.fixture.github.get_pr_files.return_value[0]
        changed.update(status="added", patch=None)
        result = self.select(analysis("experimental_only"))
        self.assertIn(
            changed["filename"], self.fixture.inference.calls[0][0]["source_files"]
        )
        first_blob = self.fixture.github.get_repository_blob.call_args_list[0].args[1]
        self.assertEqual("blob:" + changed["filename"], first_blob)
        self.assertEqual("experimental_only", result.gpu_impact["classification"])

    def test_selected_cpu_tests_do_not_override_gpu_impact(self):
        result = self.select(analysis(), "selected", [target()])
        self.assertEqual("gpu_required", result.gpu_impact["classification"])
        self.assertEqual(1, len(result.paths))

    def test_gate_keeps_valid_targets_when_other_targets_are_invalid(self):
        result = self.select(
            analysis(), "selected", [target("tests/missing.py"), target()]
        )
        self.assertEqual([target()["path"]], result.paths)
        self.assertEqual("gpu_required", result.gpu_impact["classification"])
        summary = gate.evaluate([], lambda: result.gpu_impact).summary
        self.assertIn("Warnings", summary)
        self.assertIn("tests/missing.py", summary)

    def test_gate_keeps_impact_when_no_target_survives(self):
        result = self.select(analysis(), "selected", [target("tests/missing.py")])
        self.assertEqual([], result.paths)
        self.assertEqual("gpu_required", result.gpu_impact["classification"])
        self.assertTrue(result.gpu_impact["evidence_warnings"])

    def test_gate_can_classify_when_ci_routing_file_is_unavailable(self):
        self.fixture.selector.read_ci_file = Mock(side_effect=FileNotFoundError)
        result = self.select(analysis())
        self.assertEqual("gpu_required", result.gpu_impact["classification"])
        self.assertIn(
            ".github/workflows/pr-test.yml",
            " ".join(result.gpu_impact["evidence_warnings"]),
        )

    def test_gate_can_classify_when_dispatcher_is_unavailable(self):
        from ai_test_scope import TargetSelectionError

        read_ci = self.fixture.selector.read_ci_file
        path = ".github/workflows/pr-test-dispatch.yml"
        self.fixture.selector.read_ci_file = lambda name: (
            "" if name == path else read_ci(name)
        )
        with self.assertRaises(TargetSelectionError):
            self.fixture.selector.select(fixtures.PR)
        result = self.select(analysis())
        self.assertEqual("gpu_required", result.gpu_impact["classification"])
        self.assertIn(path, " ".join(result.gpu_impact["evidence_warnings"]))

    def test_gate_can_classify_when_supplemental_blob_cannot_be_read(self):
        f = self.fixture
        path = "scripts/another_runner.sh"
        f.github.get_repository_tree.return_value["tree"].append(
            {"path": path, "sha": "runner", "mode": "100644", "type": "blob"}
        )
        read_blob = f.github.get_repository_blob.side_effect

        def read(repo, sha, limit):
            if sha == "runner":
                raise ValueError("Blob unavailable")
            return read_blob(repo, sha, limit)

        f.github.get_repository_blob.side_effect = read
        f.inference.outputs[0] = {"read_paths": [path]}
        result = self.select(analysis())
        self.assertEqual("gpu_required", result.gpu_impact["classification"])
        self.assertIn(path, " ".join(result.gpu_impact["evidence_warnings"]))
        self.assertEqual(
            "file could not be read",
            f.inference.calls[1][0]["unavailable_evidence"][path],
        )

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
        self.client.deadline = time.monotonic() + 540
        self.client.get_pr_details.side_effect = lambda number: __import__(
            "copy"
        ).deepcopy(self.pr)
        self.client.get_current_base_sha.return_value = "b" * 40
        self.client.checks.return_value = []
        self.client.begin.return_value = 23
        self.client.comments.return_value = []
        self.client.get_pr_files.return_value = [
            {"filename": "README.md", "status": "modified"}
        ]
        self.client.get_pr_files_at.side_effect = (
            lambda pr, base: self.client.get_pr_files(pr["number"])
        )
        self.client.python_sources_at.return_value = {}
        self.client.python_sources_at.side_effect = lambda *_: (
            item for item in self.client.python_sources_at.return_value.items()
        )

        self.publisher = gate.Client("upstream/flashinfer")
        self.publisher.get_current_base_sha = self.client.get_current_base_sha
        self.publisher.request = Mock(return_value={"id": 23})
        self.client.publish.side_effect = self.publisher.publish

    def test_publication_base_read_error_completes_failed_check(self):
        self.client.get_current_base_sha.side_effect = [
            "b" * 40,
            gate.TargetSelectionError("Final base API failed"),
        ]
        result = gate.run_pr(self.client, 42)
        self.assertEqual(
            ("failure", "evidence_error"), (result.conclusion, result.reason)
        )
        self.assertIn("Final base API failed", result.summary)
        self.publisher.request.assert_called_once()
        self.assertEqual(
            "failure", self.publisher.request.call_args.args[2]["conclusion"]
        )

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

    def test_historical_reply_passes_when_base_lookup_is_unavailable(self):
        self.client.comments.return_value = [comment(ADVICE, True)]
        self.client.get_current_base_sha.side_effect = gate.TargetSelectionError(
            "Base API unavailable"
        )
        result = gate.run_pr(self.client, 42)
        self.assertEqual("bot_response_found", result.reason)
        self.client.get_current_base_sha.assert_not_called()
        self.assertIsNone(self.client.publish.call_args.args[3])

    def test_publication_rechecks_base_after_preparing_payload(self):
        from unittest.mock import patch

        client = gate.Client("upstream/flashinfer")
        current_base = "b" * 40
        client.get_current_base_sha = Mock(side_effect=lambda pr: current_base)
        client.request = Mock()
        result = gate.Result(
            "success", "docs_only", "Bot run not required", "Docs only"
        )

        def version():
            nonlocal current_base
            current_base = "c" * 40
            return "version"

        with (
            patch.object(gate, "analysis_version", side_effect=version),
            self.assertRaises(gate.PublicationBaseError),
        ):
            client.publish(self.pr, result, check_id=23, base_sha="b" * 40)
        client.request.assert_not_called()

    def test_read_failure_updates_check_as_failure_not_command_missing(self):
        self.client.comments.side_effect = gate.TargetSelectionError(
            "Comment API failed"
        )
        result = gate.run_pr(self.client, 42)
        self.assertEqual("evidence_error", result.reason)
        self.assertEqual("failure", result.conclusion)
        self.client.publish.assert_called_once()

    def test_push_during_analysis_never_publishes_old_success(self):
        latest = {**self.pr, "head": {"sha": "c" * 40}}
        self.client.get_pr_details.side_effect = [self.pr, latest, latest, latest]
        result = gate.run_pr(self.client, 42)
        self.assertEqual("success", result.conclusion)
        first, last = self.client.publish.call_args_list
        self.assertEqual("failure", first.args[1].conclusion)
        self.assertEqual("a" * 40, first.args[0]["head"]["sha"])
        self.assertEqual("c" * 40, last.args[0]["head"]["sha"])

    def test_base_advance_retries_current_base_and_completes(self):
        import json

        self.client.get_current_base_sha.side_effect = [
            "b" * 40,
            "c" * 40,
            "c" * 40,
            "c" * 40,
        ]
        result = gate.run_pr(self.client, 42)
        self.assertEqual("success", result.conclusion)
        first, last = (call.args[2] for call in self.publisher.request.call_args_list)
        self.assertEqual("failure", first["conclusion"])
        self.assertEqual("success", last["conclusion"])
        self.assertEqual("c" * 40, json.loads(last["output"]["text"])["base_sha"])

    def test_repeated_base_advance_finishes_failed_instead_of_hanging(self):
        self.client.get_current_base_sha.side_effect = [
            "b" * 40,
            "c" * 40,
            "c" * 40,
            "d" * 40,
        ]
        result = gate.run_pr(self.client, 42)
        self.assertEqual("failure", result.conclusion)
        self.assertEqual("stale_evaluation", result.reason)
        self.assertEqual(2, self.client.begin.call_count)
        self.assertEqual(2, self.publisher.request.call_count)

    def test_base_read_failure_after_begin_completes_failure(self):
        self.client.get_current_base_sha.side_effect = gate.TargetSelectionError(
            "Base API failed"
        )
        result = gate.run_pr(self.client, 42)
        self.assertEqual("evidence_error", result.reason)
        self.client.publish.assert_called_once()

    def test_failure_completion_has_time_after_evidence_deadline_expires(self):
        def expired(pr):
            self.client.deadline = time.monotonic() - 1
            raise gate.TargetSelectionError("Evidence deadline exceeded")

        self.client.get_current_base_sha.side_effect = expired

        def publish(*args):
            self.assertGreater(self.client.deadline, time.monotonic())

        self.client.publish.side_effect = publish
        self.assertEqual("failure", gate.run_pr(self.client, 42).conclusion)
        self.assertLess(self.client.deadline, time.monotonic())

    def test_final_read_failure_after_begin_completes_failure(self):
        self.client.get_pr_details.side_effect = [
            self.pr,
            gate.TargetSelectionError("Final PR API failed"),
        ]
        result = gate.run_pr(self.client, 42)
        self.assertEqual("evidence_error", result.reason)
        self.client.publish.assert_called_once()

    def test_docs_fastpath_uses_captured_commit_files_even_after_head_aba(self):
        from unittest.mock import patch

        self.client.get_pr_files_at.side_effect = None
        self.client.get_pr_files_at.return_value = [
            {"filename": "kernel.cu", "status": "modified"}
        ]
        # The mutable endpoint shows docs, while final PR metadata is back at A.
        with patch.object(
            gate, "analyze_gpu_impact", return_value=analysis()
        ) as analyze:
            result = gate.run_pr(self.client, 42)
        self.assertEqual("command_missing", result.reason)
        self.client.get_pr_files_at.assert_called_once_with(self.pr, "b" * 40)
        self.client.get_pr_files.assert_not_called()
        pinned = analyze.call_args.args[0]
        self.client.get_current_base_sha.return_value = "c" * 40
        self.assertEqual("b" * 40, pinned.get_current_base_sha(self.pr))
        changed = pinned.get_pr_files(42)
        changed[0]["filename"] = "README.md"
        self.assertEqual("kernel.cu", pinned.get_pr_files(42)[0]["filename"])

    def test_commit_file_comparison_is_immutable_and_count_checked(self):
        client = gate.Client("upstream/flashinfer")
        client.request = Mock(
            return_value={
                "base_commit": {"sha": "b" * 40},
                "files": [{"filename": "README.md"}],
            }
        )
        self.assertEqual(
            "README.md", client.get_pr_files_at(self.pr, "b" * 40)[0]["filename"]
        )
        self.assertIn("b" * 40 + "..." + "a" * 40, client.request.call_args.args[0])
        client.request.return_value["base_commit"] = None
        with self.assertRaises(gate.TargetSelectionError):
            client.get_pr_files_at(self.pr, "b" * 40)
        client.request.return_value["base_commit"] = {"sha": "b" * 40}
        client.request.return_value["files"] = []
        with self.assertRaises(gate.TargetSelectionError):
            client.get_pr_files_at(self.pr, "b" * 40)

    def test_experimental_only_passes_and_shows_targeted_command(self):
        from unittest.mock import patch

        paths = (
            "flashinfer/experimental/feature/kernel.cu",
            "tests/experimental/test_new_gpu_kernel.py",
        )
        for path in paths:
            with self.subTest(path=path):
                self.client.get_pr_files.return_value = [{"filename": path}]
                with patch.object(gate, "analyze_gpu_impact") as analyze:
                    result = gate.run_pr(self.client, 42)
                self.assertEqual(
                    ("success", "experimental_only"), (result.conclusion, result.reason)
                )
                self.assertIn("/bot run tests/experimental/", result.summary)
                self.assertIn("run all", result.summary)
                self.assertNotIn("Declare the test scope", result.summary)
                analyze.assert_not_called()

    def test_python_documentation_takes_priority_over_experimental_paths(self):
        from unittest.mock import patch

        path = "flashinfer/experimental/a/feature.py"
        self.pr["changed_files"] = 2
        self.client.get_pr_files.return_value = [
            {"filename": path, "status": "modified"},
            {"filename": "README.md", "status": "modified"},
        ]
        self.client.python_sources_at.return_value = {
            path: (
                'def kernel():\n    """Old documentation."""\n    return launch()\n',
                'def kernel():\n    """New documentation."""\n    # Explain the launch.\n    return launch()\n',
            ),
        }
        with patch.object(gate, "analyze_gpu_impact") as analyze:
            result = gate.run_pr(self.client, 42)
        self.assertEqual(("success", "docs_only"), (result.conclusion, result.reason))
        analyze.assert_not_called()

    def test_python_code_or_directive_changes_are_not_documentation(self):
        from unittest.mock import patch

        path = "flashinfer/experimental/a/feature.py"
        self.client.get_pr_files.return_value = [
            {"filename": path, "status": "modified"},
        ]
        for after in ("launch(2)\n", "launch(1)  # type: ignore\n"):
            with self.subTest(after=after), patch.object(gate, "analyze_gpu_impact"):
                self.client.python_sources_at.return_value = {
                    path: ("launch(1)\n", after),
                }
                result = gate.run_pr(self.client, 42)
                self.assertEqual("experimental_only", result.reason)

    def test_experimental_with_documentation_also_passes_without_inference(self):
        from unittest.mock import patch

        self.pr["changed_files"] = 2
        self.client.get_pr_files.return_value = [
            {"filename": "README.md", "status": "modified"},
            {
                "filename": "flashinfer/experimental/feature/kernel.cu",
                "status": "added",
            },
        ]
        with patch.object(gate, "analyze_gpu_impact") as analyze:
            result = gate.run_pr(self.client, 42)
        self.assertEqual(
            ("success", "experimental_only"), (result.conclusion, result.reason)
        )
        analyze.assert_not_called()

    def test_rename_from_stable_into_experimental_requires_analysis(self):
        from unittest.mock import patch

        self.client.get_pr_files.return_value = [
            {
                "filename": "flashinfer/experimental/backend.py",
                "previous_filename": "flashinfer/backend.py",
                "status": "renamed",
            }
        ]
        with patch.object(
            gate, "analyze_gpu_impact", return_value=analysis()
        ) as analyze:
            result = gate.run_pr(self.client, 42)
        self.assertEqual(
            ("failure", "command_missing"), (result.conclusion, result.reason)
        )
        analyze.assert_called_once()

    def test_experimental_gpu_unknown_is_not_an_exemption(self):
        from unittest.mock import patch

        self.client.get_pr_files.return_value = [
            {"filename": "flashinfer/feature/backend.py"}
        ]
        with patch.object(gate, "analyze_gpu_impact", return_value=analysis("unknown")):
            result = gate.run_pr(self.client, 42)
        self.assertEqual(("failure", "unknown"), (result.conclusion, result.reason))

    def test_stable_gpu_with_experimental_change_still_requires_bot(self):
        from unittest.mock import patch

        self.pr["changed_files"] = 2
        self.client.get_pr_files.return_value = [
            {"filename": "flashinfer/experimental/new.py"},
            {"filename": "csrc/stable_kernel.cu"},
        ]
        with patch.object(gate, "analyze_gpu_impact", return_value=analysis()):
            result = gate.run_pr(self.client, 42)
        self.assertEqual("command_missing", result.reason)

    def test_experimental_exemption_cache_checks_analysis_version(self):
        import json

        checks = [
            {
                "id": 10,
                "external_id": gate.VERSION + ":42",
                "app": {"slug": "github-actions"},
                "output": {
                    "text": json.dumps(
                        {
                            "base_sha": "b" * 40,
                            "version": gate.analysis_version(),
                            "analysis": analysis("experimental_only"),
                        }
                    )
                },
            }
        ]
        self.assertEqual(
            "experimental_only",
            gate.cached_analysis(checks, self.pr, "b" * 40)["classification"],
        )
        self.assertIsNone(gate.cached_analysis(checks, self.pr, "c" * 40))

    def test_analysis_timeout_restores_publication_budget(self):
        from unittest.mock import patch

        import time

        deadline = time.monotonic() + 540
        self.client.deadline = deadline
        self.client.get_pr_files.return_value = [{"filename": "kernel.cu"}]

        def timeout(*args, **kwargs):
            self.client.deadline = 0
            raise gate.TargetSelectionError("Inference deadline exceeded")

        selector = Mock()
        selector.timeout = 240
        selector.select.side_effect = timeout
        with patch.object(gate, "PytestTargetSelector", return_value=selector):
            result = gate.run_pr(self.client, 42)
        self.assertEqual(deadline, self.client.deadline)
        self.assertEqual("failure", result.conclusion)
        self.client.publish.assert_called_once()

    def test_check_publication_targets_pr_head_and_updates_existing(self):
        client = gate.Client("upstream/flashinfer")
        client.request = Mock(return_value={"id": 23})
        client.get_current_base_sha = Mock(return_value="b" * 40)
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
                [{"filename": "docs/new.md", "previous_filename": "kernel.cu"}],
                1,
                Mock(side_effect=AssertionError("Unexpected source lookup")),
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


class RetryAnalysisTests(unittest.TestCase):
    def setUp(self):
        import time

        self.fixture = fixtures.PytestTargetSelectorTests()
        self.fixture.setUp()
        self.fixture.github.deadline = time.monotonic() + 540

    def run_outputs(self, outputs):
        class Model(fixtures.FakeInference):
            reasoning_effort = None

            def __init__(self):
                super().__init__(outputs)
                self.efforts = []

            def complete(self, context, task):
                self.efforts.append(self.reasoning_effort)
                value = super().complete(context, task)
                if isinstance(value, Exception):
                    raise value
                return value

        model = Model()
        f = self.fixture
        result = gate.analyze_gpu_impact(f.github, fixtures.PR, f.read_ci, model)
        return result, model

    def selection(self, kind):
        return {
            "outcome": "needs_maintainer",
            "targets": [],
            "reason": "Scope advice",
            "gpu_impact": analysis(kind),
        }

    def test_known_cpu_or_gpu_does_not_retry(self):
        for kind in ("cpu_only", "gpu_required"):
            impact, model = self.run_outputs([{"read_paths": []}, self.selection(kind)])
            self.assertEqual(kind, impact["classification"])
            self.assertEqual([None, None], model.efforts)

    def test_failed_final_request_retries_minimal_with_same_evidence(self):
        impact, model = self.run_outputs(
            [
                {"read_paths": []},
                gate.TargetSelectionError("Incomplete JSON"),
                self.selection("gpu_required"),
            ]
        )
        self.assertEqual("gpu_required", impact["classification"])
        self.assertEqual([None, None, "minimal"], model.efforts)
        self.assertEqual(model.calls[1][0], model.calls[2][0])
        self.assertEqual("Incomplete JSON", impact["analysis_attempts"][0]["error"])

    def test_invalid_optional_evidence_does_not_interrupt_classification(self):
        impact, model = self.run_outputs(
            [
                {"read_paths": ["invented.py", None]},
                self.selection("cpu_only"),
            ]
        )
        self.assertEqual("cpu_only", impact["classification"])
        self.assertEqual([None, None], model.efforts)
        self.assertEqual(2, len(impact["evidence_warnings"]))

    def test_unknown_retries_once_and_requires_bot(self):
        impact, model = self.run_outputs(
            [{"read_paths": []}, self.selection("unknown"), self.selection("unknown")]
        )
        self.assertEqual("unknown", impact["classification"])
        self.assertEqual(2, len(impact["analysis_attempts"]))
        result = gate.evaluate([], lambda: impact)
        self.assertEqual("failure", result.conclusion)
        self.assertIn("/bot run", result.summary)
        self.assertNotIn("Rerun this check", result.summary)

    def test_missing_patches_are_informational_for_any_filename(self):
        from gpu_impact import validate_impact

        context = {
            "source_files": {"utils.py": "host code"},
            "changed_files": [{"filename": "utils.py"}],
            "ci_files": {},
            "unavailable_evidence": {},
            "omitted_evidence": {"changed_files": 0},
        }
        impact = validate_impact(
            {**analysis("cpu_only"), "evidence_paths": ["utils.py"]}, context
        )
        self.assertEqual("cpu_only", impact["classification"])
        self.assertEqual(["utils.py"], impact["missing_patch_paths"])
        self.assertIn(
            "Patch content unavailable", gate.evaluate([], lambda: impact).summary
        )
        context["omitted_evidence"]["changed_files"] = 1
        self.assertEqual(
            "unknown",
            validate_impact(
                {**analysis("cpu_only"), "evidence_paths": ["utils.py"]}, context
            )["classification"],
        )

    def test_budget_omitted_file_can_supply_filename_evidence(self):
        from gpu_impact import validate_impact

        # This validates the agreed fallback policy, not path-based ground truth.
        for path in ("docs/large.md", "tests/cpu/host_utils.py"):
            context = {
                "source_files": {},
                "changed_files": [{"filename": path, "patch_omitted": True}],
                "ci_files": {},
                "unavailable_evidence": {},
                "omitted_evidence": {"changed_files": 0},
            }
            impact = validate_impact(
                {**analysis("cpu_only"), "evidence_paths": [path]}, context
            )
            self.assertEqual("cpu_only", impact["classification"])
            self.assertEqual([path], impact["filename_only_evidence"])

    def test_missing_patch_note_is_bounded_for_large_changes(self):
        from gpu_impact import validate_impact

        context = {
            "source_files": {"utils.py": "host code"},
            "changed_files": [
                {"filename": f"utils_{index}.py"} for index in range(100)
            ],
            "ci_files": {},
            "unavailable_evidence": {},
            "omitted_evidence": {"changed_files": 0},
        }
        impact = validate_impact(
            {**analysis("cpu_only"), "evidence_paths": ["utils.py"]}, context
        )
        self.assertEqual(30, len(impact["missing_patch_paths"]))
        self.assertEqual(100, impact["missing_patch_count"])
        self.assertIn("first 30 of 100", gate.evaluate([], lambda: impact).summary)

    def test_available_body_is_not_labeled_filename_only(self):
        from gpu_impact import validate_impact

        context = {
            "source_files": {"helper.py": "def host_only(): pass"},
            "changed_files": [{"filename": "helper.py", "patch_omitted": True}],
            "ci_files": {},
            "unavailable_evidence": {},
            "omitted_evidence": {"changed_files": 0},
        }
        impact = validate_impact(
            {**analysis("cpu_only"), "evidence_paths": ["helper.py"]}, context
        )
        self.assertEqual("cpu_only", impact["classification"])
        self.assertNotIn("filename_only_evidence", impact)
        self.assertEqual(["helper.py"], impact["missing_patch_paths"])

    def test_evidence_cache_is_not_reused_for_changed_context(self):
        model = fixtures.FakeInference(
            [
                {"read_paths": ["first.py"]},
                self.selection("gpu_required"),
                {"read_paths": ["second.py"]},
            ]
        )
        wrapped = gate.RetryingImpactInference(model)
        from ai_test_scope import EVIDENCE_PROMPT

        wrapped.complete({"head": "first"}, EVIDENCE_PROMPT)
        wrapped.complete({"head": "first"}, "selection")
        wrapped.retry = True
        self.assertEqual(
            {"read_paths": ["second.py"]},
            wrapped.complete({"head": "second"}, EVIDENCE_PROMPT),
        )
        self.assertEqual(3, len(model.calls))


if __name__ == "__main__":
    unittest.main()
