"""Offline coverage of the public bot's API transport and CLI boundaries."""

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/infra/pr_checks"))
import ai_test_scope as scope
import test_ai_test_scope as fixtures


def completion(content='{"read_paths": []}', finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": content}}]}


def response(data):
    result = Mock()
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    result.read.return_value = json.dumps(data).encode()
    return result


class InferenceTransportTests(unittest.TestCase):
    def test_reasoning_effort_is_only_sent_when_explicitly_requested(self):
        for effort in (None, "minimal"):
            with patch.object(
                scope.urllib.request, "urlopen", return_value=response(completion())
            ) as call:
                scope.InferenceHub("secret", reasoning_effort=effort).complete(
                    {}, "JSON"
                )
            payload = json.loads(call.call_args.args[0].data)
            if effort is None:
                self.assertNotIn("reasoning_effort", payload)
            else:
                self.assertEqual("minimal", payload["reasoning_effort"])

    def test_transient_http_and_network_errors_retry_then_succeed(self):
        for error in (
            urllib.error.HTTPError("https://hub.example", 429, "busy", {}, None),
            urllib.error.HTTPError("https://hub.example", 503, "busy", {}, None),
            urllib.error.URLError("private diagnostics"),
            TimeoutError("private diagnostics"),
        ):
            with (
                self.subTest(error=type(error).__name__),
                patch.object(
                    scope.urllib.request,
                    "urlopen",
                    side_effect=[error, response(completion())],
                ) as call,
                patch.object(scope.time, "sleep"),
            ):
                self.assertEqual(
                    {"read_paths": []},
                    scope.InferenceHub("secret").complete({}, "JSON"),
                )
                self.assertEqual(2, call.call_count)

    def test_retry_exhaustion_and_bad_responses_fail_closed(self):
        for error in (
            urllib.error.HTTPError("https://hub.example", 503, "secret", {}, None),
            urllib.error.URLError("secret"),
            TimeoutError("secret"),
        ):
            with (
                self.subTest(error=type(error).__name__),
                patch.object(
                    scope.urllib.request, "urlopen", side_effect=error
                ) as call,
                patch.object(scope.time, "sleep"),
                self.assertRaises(scope.TargetSelectionError) as caught,
            ):
                scope.InferenceHub("secret").complete({}, "JSON")
            self.assertEqual(3, call.call_count)
            self.assertNotIn("secret", str(caught.exception))
        for value in (
            completion("not JSON"),
            completion("[]"),
            completion("{}", "length"),
            completion(None),
            {"choices": []},
        ):
            with (
                self.subTest(value=value),
                patch.object(
                    scope.urllib.request, "urlopen", return_value=response(value)
                ),
                self.assertRaises(scope.TargetSelectionError),
            ):
                scope.InferenceHub("secret").complete({}, "JSON")

    def test_missing_key_and_invalid_endpoint_never_send_a_request(self):
        with patch.object(scope.urllib.request, "urlopen") as call:
            with self.assertRaises(scope.TargetSelectionError):
                scope.InferenceHub("").complete({}, "JSON")
            for endpoint in (
                "http://hub.example",
                "https://secret@hub.example",
                "https://hub.example?token=secret",
            ):
                with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                    scope.InferenceHub("secret", endpoint)
            call.assert_not_called()


class GitHubReadTests(unittest.TestCase):
    def test_get_pagination_without_a_shell(self):
        github = scope.GitHub("owner/repo")
        with patch.object(
            scope.subprocess,
            "run",
            return_value=Mock(stdout='[{"filename":"tests/x.py"}]'),
        ) as call:
            self.assertEqual([{"filename": "tests/x.py"}], github.get_pr_files(42))
            self.assertEqual(
                ["gh", "api", "--method", "GET"], call.call_args.args[0][:4]
            )
            self.assertIn("per_page=100&page=1", call.call_args.args[0][-1])
            self.assertNotIn("shell", call.call_args.kwargs)

    def test_read_errors_oversized_responses_and_deadlines_fail_closed(self):
        github = scope.GitHub("owner/repo")
        for error in (
            subprocess.CalledProcessError(1, ["gh"]),
            subprocess.TimeoutExpired(["gh"], 1),
            OSError("secret"),
        ):
            with (
                patch.object(scope.subprocess, "run", side_effect=error),
                self.assertRaises(scope.TargetSelectionError),
            ):
                github.get_pr_details(42)
        for stdout in ("not JSON", "x" * 10_000_001):
            with (
                patch.object(scope.subprocess, "run", return_value=Mock(stdout=stdout)),
                self.assertRaises(scope.TargetSelectionError),
            ):
                github.get_pr_details(42)
        github.deadline = scope.time.monotonic() - 1
        with (
            patch.object(scope.subprocess, "run") as call,
            self.assertRaises(scope.TargetSelectionError),
        ):
            github.get_pr_details(42)
        call.assert_not_called()


class ScopeCliTests(unittest.TestCase):
    def test_cli_modes_write_exact_scope_and_failures_remove_stale_output(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            command, output, note = (
                directory / name for name in ("command", "output", "note")
            )
            args = [
                "--repo",
                "owner/repo",
                "--pr",
                "42",
                "--command-file",
                str(command),
                "--output",
                str(output),
                "--note",
                str(note),
            ]
            github = Mock()
            github.get_current_base_sha.side_effect = lambda pr: pr["base"]["sha"]
            github.get_pr_details.return_value = {
                "number": 42,
                "head": {"sha": "head-A"},
                "base": {"sha": "base-A"},
            }
            selection = scope.TargetSelection(
                [
                    scope.PytestTarget(
                        "tests/example/test_example.py",
                        "Tests changed behavior",
                        "CPU",
                        "unit-test lane",
                    )
                ],
                snapshot={
                    "head_sha": "head-A",
                    "base_sha": "base-A",
                    "merge_sha": "merge-A",
                    "tree_sha": "tree-A",
                },
            )
            for text, mode, paths in (
                ("@flashinfer-bot run all", "full", []),
                ("@flashinfer-bot run tests/example", "manual", ["tests/example"]),
                ("@flashinfer-bot run", "advisory", selection.paths),
            ):
                command.write_text(text)
                with (
                    patch.object(scope, "GitHub", return_value=github),
                    patch.object(
                        scope.PytestTargetSelector, "select", return_value=selection
                    ),
                ):
                    self.assertEqual(0, scope.main(args))
                data = json.loads(output.read_text())
                self.assertEqual((mode, paths), (data["mode"], data["targets"]))
            command.write_text("@flashinfer-bot run tests/../../etc")
            with (
                redirect_stderr(io.StringIO()),
                patch.object(scope, "GitHub", return_value=github),
            ):
                self.assertEqual(1, scope.main(args))
            self.assertFalse(output.exists())
            self.assertIn("No CI pipeline was started", note.read_text())


class MergeReadTransportTests(unittest.TestCase):
    def test_resolves_current_target_tip_without_using_recorded_base(self):
        github = scope.GitHub("owner/repo")
        pr = {
            "base": {
                "sha": "old",
                "ref": "release/next",
                "repo": {"full_name": "owner/repo"},
            }
        }
        with patch.object(
            github, "api", return_value={"object": {"sha": "current"}}
        ) as api:
            self.assertEqual("current", github.get_current_base_sha(pr))
        api.assert_called_once_with("repos/owner/repo/git/ref/heads/release%2Fnext")


class EvidenceInventoryTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.PytestTargetSelectorTests()
        fixture.setUp()
        self.github = fixture.github
        self.inference = fixture.inference
        self.selector = fixture.selector

    def test_regular_file_outside_trimmed_inventory_is_gate_only(self):
        path = "include/flashinfer/nested/private_kernel.cuh"
        self.github.get_repository_tree.return_value["tree"].append(
            {"path": path, "sha": "kernel", "mode": "100644", "type": "blob", "size": 7}
        )
        original = self.github.get_repository_blob.side_effect
        self.github.get_repository_blob.side_effect = (
            lambda repo, sha, limit: "// CUDA"
            if sha == "kernel"
            else original(repo, sha, limit)
        )
        self.inference.outputs[0]["read_paths"].append(path)
        with self.assertRaises(scope.TargetSelectionError):
            self.selector.select(fixtures.PR)
        self.setUp()
        self.github.get_repository_tree.return_value["tree"].append(
            {"path": path, "sha": "kernel", "mode": "100644", "type": "blob", "size": 7}
        )
        self.github.get_repository_blob.side_effect = (
            lambda repo, sha, limit: "// CUDA"
            if sha == "kernel"
            else original(repo, sha, limit)
        )
        self.inference.outputs[0]["read_paths"].append(path)
        self.inference.outputs[1]["gpu_impact"] = {
            "classification": "gpu_required",
            "reason": "Kernel behavior changed",
            "evidence_paths": [path],
            "case_gap": "",
        }
        self.selector.select(fixtures.PR, include_gpu_impact=True)
        self.assertNotIn(path, self.inference.calls[0][0]["inventory_at_merged_tree"])
        self.assertIn(path, self.inference.calls[1][0]["source_files"])

    def test_symlinks_and_submodules_are_not_readable_evidence(self):
        for mode, kind in (("120000", "blob"), ("160000", "commit")):
            self.setUp()
            path = "flashinfer/unsafe.py"
            self.github.get_repository_tree.return_value["tree"].append(
                {"path": path, "sha": "unsafe", "mode": mode, "type": kind}
            )
            self.inference.outputs[0] = {"read_paths": [path]}
            with self.assertRaises(scope.TargetSelectionError):
                self.selector.select(fixtures.PR)
