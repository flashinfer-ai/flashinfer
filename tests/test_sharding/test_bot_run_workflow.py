"""CPU-only tests for trusted bot scope preparation and label handoff."""

import io
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
CHECKS = ROOT / "scripts" / "pr_checks"
sys.path.insert(0, str(CHECKS))
import ai_test_scope as scope

WORKFLOW_TEXT = (ROOT / ".github/workflows/ci-bot-commands.yml").read_text()
PR_WORKFLOW_TEXT = (ROOT / ".github/workflows/pr-test.yml").read_text()


def step_block(name, workflow_text=WORKFLOW_TEXT):
    return workflow_text.split("      - name: " + name + "\n", 1)[1].split(
        "\n      - name: ", 1
    )[0]


def step(name, workflow_text=WORKFLOW_TEXT):
    # The runner consumes the actual checked-in Bash block. No YAML dependency
    # is needed by this CPU-only test suite.
    block = step_block(name, workflow_text).split("        run: |\n", 1)[1]
    lines = []
    for line in block.splitlines():
        if line and not line.startswith("          "):
            break
        lines.append(line[10:] if line.startswith("          ") else "")
    return "\n".join(lines)


class RunCommandTests(unittest.TestCase):
    def test_modes_and_manual_syntax_remain_distinct(self):
        self.assertEqual(
            ("automatic", []), scope.parse_run_command("@flashinfer-bot run")
        )
        self.assertEqual(
            ("full", []), scope.parse_run_command("@FlashInfer-bot RUN FULL")
        )
        self.assertEqual(
            ("manual", ["tests/moe", "tests/attention/test_rope.py"]),
            scope.parse_run_command(
                "@flashinfer-bot run tests/moe tests/attention/test_rope.py"
            ),
        )
        for command in (
            "@flashinfer-bot run full tests/moe",
            "@flashinfer-bot run --full",
            "@flashinfer-bot run tests/a.py;echo",
            "@flashinfer-bot run ../../etc",
            "@flashinfer-bot run tests/a.py::test_x",
            "prose @flashinfer-bot run",
        ):
            with (
                self.subTest(command=command),
                self.assertRaises((ValueError, scope.TargetSelectionError)),
            ):
                scope.parse_run_command(command)

    def test_manual_modes_do_not_create_an_inference_client(self):
        github = Mock()
        github.get_pr_details.return_value = {"head": {"sha": "head-A"}}
        with patch.object(
            scope, "InferenceHub", side_effect=AssertionError("AI must not run")
        ):
            full = scope.prepare_scope("@flashinfer-bot run full", 42, github, Mock())
            manual = scope.prepare_scope(
                "@flashinfer-bot run tests/moe", 42, github, Mock()
            )
        self.assertEqual([], full["chunks"])
        self.assertEqual(["tests/moe"], manual["chunks"])

    def test_cli_posts_actionable_manual_validation_errors_but_hides_api_errors(self):
        cases = (
            ("tests/a.py::test_x", "::selectors"),
            (" ".join(f"tests/test_{i}.py" for i in range(17)), "maximum is 16"),
            ("tests/" + "a" * 140, "140-character chunk limit"),
            ("tests/../../outside", "targets outside tests/"),
            ("tests/a.py;@FLASHINFER-bot", "characters not allowed"),
            ("tests/secret-value;", "characters not allowed"),
        )
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            command = directory / "command"
            note = directory / "note"
            output = directory / "result"
            argv = [
                "--command-file",
                str(command),
                "--repo",
                "owner/repo",
                "--pr",
                "42",
                "--output",
                str(output),
                "--note",
                str(note),
            ]
            github = Mock()
            github.get_pr_details.return_value = {"head": {"sha": "head-A"}}
            with (
                patch.object(scope, "GitHub", return_value=github),
                patch.dict(os.environ, {"INFERENCE_API_KEY": "secret-value"}),
                patch("sys.stderr", new_callable=io.StringIO),
            ):
                for targets, reason in cases:
                    with self.subTest(targets=targets):
                        command.write_text("@flashinfer-bot run " + targets)
                        output.write_text("stale result")
                        self.assertEqual(1, scope.main(argv))
                        text = note.read_text()
                        self.assertIn(reason, text)
                        self.assertIn("run full", text)
                        self.assertEqual(1, len(text.splitlines()))
                        self.assertNotIn("@flashinfer-bot", text.lower())
                        self.assertNotIn("secret-value", text)
                        self.assertFalse(output.exists())
                command.write_text("@flashinfer-bot run tests/a.py")
                github.get_pr_details.side_effect = scope.TargetSelectionError(
                    "private API response"
                )
                self.assertEqual(1, scope.main(argv))
                self.assertNotIn("private API response", note.read_text())

    def test_preparation_rejects_a_changed_head_and_uncertain_automatic_selection(self):
        github = Mock()
        github.get_pr_details.side_effect = [
            {"head": {"sha": "A"}},
            {"head": {"sha": "B"}},
        ]
        with self.assertRaises(scope.TargetSelectionError):
            scope.prepare_scope("@flashinfer-bot run full", 42, github, Mock())
        github.get_pr_details.side_effect = None
        github.get_pr_details.return_value = {"head": {"sha": "A"}}
        with (
            patch.object(
                scope.PytestTargetSelector,
                "select",
                return_value=scope.TargetSelection([], "needs_maintainer"),
            ),
            self.assertRaises(scope.TargetSelectionError),
        ):
            scope.prepare_scope("@flashinfer-bot run", 42, github, Mock())

    def test_model_call_uses_bearer_and_structured_messages_and_temperature_fallback(
        self,
    ):
        data = {
            "choices": [
                {"finish_reason": "stop", "message": {"content": '{"read_paths": []}'}}
            ]
        }
        response = Mock()
        response.__enter__ = Mock(return_value=io.BytesIO(json.dumps(data).encode()))
        response.__exit__ = Mock(return_value=False)
        rejection = urllib.error.HTTPError(
            "https://hub.example/v1",
            400,
            "bad",
            {},
            io.BytesIO(b"temperature unsupported"),
        )
        hub = scope.InferenceHub("secret", "https://hub.example/v1", "model")
        with patch.object(
            scope.urllib.request, "urlopen", side_effect=[rejection, response]
        ) as urlopen:
            self.assertEqual(
                {"read_paths": []}, hub.complete({"pr": "untrusted"}, "Return JSON")
            )
        first = urlopen.call_args_list[0].args[0]
        last = urlopen.call_args_list[1].args[0]
        self.assertEqual("Bearer secret", first.get_header("Authorization"))
        self.assertIn("temperature", json.loads(first.data))
        self.assertNotIn("temperature", json.loads(last.data))
        self.assertIn("untrusted DATA", json.loads(last.data)["messages"][0]["content"])
        self.assertLessEqual(urlopen.call_args.kwargs["timeout"], 60)

    def test_auth_error_never_echoes_key_and_is_not_retried(self):
        error = urllib.error.HTTPError(
            "https://hub.example", 403, "bad", {}, io.BytesIO(b"secret")
        )
        with (
            patch.object(scope.urllib.request, "urlopen", side_effect=error) as call,
            self.assertRaisesRegex(scope.TargetSelectionError, "HTTP 403") as caught,
        ):
            scope.InferenceHub("secret").complete({}, "JSON")
        self.assertNotIn("secret", str(caught.exception))
        call.assert_called_once()


FAKE_GH = r"""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if '--body-file' in args:
    args += ['body=' + Path(args[args.index('--body-file') + 1]).read_text()]
with open(os.environ['CALL_LOG'], 'a') as output:
    output.write(json.dumps(args) + '\n')
if args[:2] == ['pr', 'view']:
    print(('head-B' if os.getenv('HEAD_MOVED') else 'head-A') if 'headRefOid' in args else 'true')
elif args[:1] == ['api'] and any('/pulls/' in arg for arg in args):
    print('main')
elif args[:1] == ['api'] and any('/git/ref/' in arg for arg in args):
    print(('e' if os.getenv('BASE_MOVED') else 'b') * 40)
elif args[:1] == ['api'] and any('/commits/' in arg for arg in args):
    if os.getenv('READ_FAIL'): sys.exit(1)
    print('3')
elif args[:1] == ['api'] and any('/statuses/' in arg for arg in args):
    if os.getenv('PUBLISH_FAIL') or (os.getenv('SNAPSHOT_FAIL') and 'context=ci/test-snapshot' in args): sys.exit(1)
"""
FAKE_PYTHON = r"""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
def arg(name): return args[args.index(name) + 1]
command = Path(arg('--command-file')).read_text()
Path(os.environ['SEEN_COMMAND']).write_text(command)
if os.getenv('SELECT_FAIL'):
    if os.getenv('FAIL_NOTE'): Path(arg('--note')).write_text(os.environ['FAIL_NOTE'])
    sys.exit(1)
mode = os.getenv('SCOPE_MODE', 'automatic')
Path(arg('--output')).write_text(json.dumps({'head_sha':'head-A','mode':mode,
    'chunks':[] if mode == 'full' else ['tests/attention/test_rope.py'],
    'snapshot':None if mode != 'automatic' or os.getenv('BAD_SNAPSHOT') else {
        'head_sha':'head-A','base_sha':'b'*40,'merge_sha':'c'*40,'tree_sha':'d'*40}}))
Path(arg('--note')).write_text('AI-selected test scope')
"""


class BotWorkflowTests(unittest.TestCase):
    def test_permission_gate_handles_roles_team_membership_and_missing_credentials(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            gh = directory / "gh"
            gh.write_text(
                "#!" + sys.executable + "\n"
                "import os, sys\n"
                'print(os.environ["TEST_ROLE"] if any("/collaborators/" in arg for arg in sys.argv) else os.environ["TEST_MEMBERS"])\n'
            )
            gh.chmod(0o755)
            for role, members, token, authorized in (
                ("admin", "", "token", True),
                ("maintain", "", "token", True),
                ("write", "", "token", True),
                ("triage", "", "token", True),
                ("read", "developer", "token", True),
                ("read", "another-user", "token", False),
                ("read", "developer-suffix", "token", False),
                ("admin", "", "", False),
            ):
                with self.subTest(role=role, members=members, has_token=bool(token)):
                    output = directory / "output"
                    output.unlink(missing_ok=True)
                    env = dict(
                        os.environ,
                        PATH=str(directory) + ":" + os.environ["PATH"],
                        GH_TOKEN=token,
                        ORG="owner",
                        TEAM="ci-users",
                        ACTOR="developer",
                        REPO="owner/repo",
                        RUNNER_TEMP=str(directory),
                        GITHUB_OUTPUT=str(output),
                        TEST_ROLE=role,
                        TEST_MEMBERS=members,
                    )
                    result = subprocess.run(
                        ["bash", "-euo", "pipefail", "-c", step("Check permission")],
                        env=env,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(0, result.returncode, result.stderr)
                    self.assertEqual(
                        f"authorized={str(authorized).lower()}\n", output.read_text()
                    )

    def test_parser_ignores_quoted_inline_fenced_and_prose_mentions(self):
        for body in (
            "> @flashinfer-bot run",
            "`@flashinfer-bot run`",
            "prose @flashinfer-bot run",
            "```\n~~~\n@flashinfer-bot run\n```",
            "~~~\n```\n@flashinfer-bot run\n~~~",
            "@flashinfer-bot running",
        ):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                output = directory / "output"
                result = subprocess.run(
                    ["bash", "-euo", "pipefail", "-c", step("Parse command")],
                    env=dict(
                        os.environ,
                        COMMENT_BODY=body,
                        RUNNER_TEMP=str(directory),
                        GITHUB_OUTPUT=str(output),
                    ),
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertEqual("command=unknown\n", output.read_text())
                self.assertFalse((directory / "ci-bot-run-command.txt").exists())

    def run_workflow(self, command="@flashinfer-bot run", **settings):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for name, content in [("gh", FAKE_GH), ("python3", FAKE_PYTHON)]:
                path = directory / name
                # Absolute interpreter avoids recursion through the fake python3.
                path.write_text(
                    "#!" + sys.executable + "\n" + content.split("\n", 1)[1]
                )
                path.chmod(0o755)
            env = dict(
                os.environ,
                PATH=str(directory) + ":" + os.environ["PATH"],
                RUNNER_TEMP=str(directory),
                GITHUB_OUTPUT=str(directory / "output"),
                GITHUB_REPOSITORY="owner/repo",
                PR_NUMBER="42",
                COMMENT_ID="100",
                COMMENT_URL="https://github.com/owner/repo/pull/42#issuecomment-100",
                GH_TOKEN="bot-token",
                STATUS_TOKEN="status-token",
                COMMENT_BODY=command,
                CALL_LOG=str(directory / "calls"),
                SEEN_COMMAND=str(directory / "command"),
                **settings,
            )
            parsed = subprocess.run(
                ["bash", "-euo", "pipefail", "-c", step("Parse command")],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, parsed.returncode, parsed.stderr)
            body = subprocess.run(
                ["bash", "-euo", "pipefail", "-c", step("Handle @flashinfer-bot run")],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, body.returncode, body.stderr)
            calls = [
                json.loads(line)
                for line in (directory / "calls").read_text().splitlines()
            ]
            return calls, (directory / "command").read_text()

    def test_automatic_run_publishes_scope_then_clears_stale_chunks_and_labels(self):
        calls, command = self.run_workflow()
        self.assertEqual("@flashinfer-bot run", command)
        published = [args for args in calls if any("/statuses/" in arg for arg in args)]
        self.assertEqual(4, len(published))
        self.assertIn("description=tests/attention/test_rope.py", published[0])
        self.assertIn("description=", published[1])
        self.assertIn("context=ci/test-snapshot", published[-1])
        self.assertIn(
            "description="
            + " ".join(char * 40 for char in "bcd")
            + " "
            + hashlib.sha256(b"tests/attention/test_rope.py").hexdigest()[:16],
            published[-1],
        )
        label = next(index for index, args in enumerate(calls) if "--add-label" in args)
        self.assertEqual("run-ci-focused", calls[label][-1])
        self.assertTrue(
            all(index < label for index, args in enumerate(calls) if args in published)
        )

    def test_full_run_clears_all_old_scope_chunks(self):
        calls, _ = self.run_workflow("@flashinfer-bot run full", SCOPE_MODE="full")
        published = [args for args in calls if any("/statuses/" in arg for arg in args)]
        self.assertEqual(4, len(published))
        self.assertTrue(all("description=" in args for args in published))
        self.assertTrue(any("--add-label" in args for args in calls))

    def test_manual_scope_clears_an_earlier_automatic_snapshot(self):
        calls, _ = self.run_workflow(
            "@flashinfer-bot run tests/moe", SCOPE_MODE="manual"
        )
        snapshot = [args for args in calls if "context=ci/test-snapshot" in args]
        self.assertEqual(1, len(snapshot))
        self.assertIn("description=", snapshot[0])
        self.assertTrue(any("--add-label" in args for args in calls))

    def test_failures_and_head_changes_never_apply_ci_label(self):
        for setting in (
            "SELECT_FAIL",
            "READ_FAIL",
            "PUBLISH_FAIL",
            "HEAD_MOVED",
            "BASE_MOVED",
            "SNAPSHOT_FAIL",
            "BAD_SNAPSHOT",
        ):
            with self.subTest(setting=setting):
                calls, _ = self.run_workflow(**{setting: "1"})
                self.assertFalse(any("--add-label" in args for args in calls))
                self.assertTrue(any("content=confused" in args for args in calls))

    def test_rejection_preserves_helper_feedback_and_has_a_fallback(self):
        for note in ("Rejected: pytest ::selectors are not supported.", ""):
            with self.subTest(note=note):
                calls, _ = self.run_workflow(SELECT_FAIL="1", FAIL_NOTE=note)
                comment = next(args for args in calls if args[:2] == ["pr", "comment"])
                body = next(
                    arg.removeprefix("body=")
                    for arg in comment
                    if arg.startswith("body=")
                )
                self.assertIn(
                    note or "test targets could not be selected or validated", body
                )
                self.assertFalse(any("--add-label" in args for args in calls))

    def test_only_the_first_unfenced_command_is_used_for_scope_selection(self):
        calls, command = self.run_workflow(
            "```\n@flashinfer-bot run full\n```\n@flashinfer-bot run\nprose @flashinfer-bot run tests/evil"
        )
        self.assertEqual("@flashinfer-bot run", command)
        self.assertTrue(any("--add-label" in args for args in calls))

    def test_key_is_only_exposed_to_the_authorized_run_step(self):
        handler = step_block("Handle @flashinfer-bot run")
        self.assertIn("authorized == 'true'", handler)
        self.assertIn("INFERENCE_API_KEY: ${{ secrets.INFERENCE_API_KEY }}", handler)
        self.assertEqual(1, WORKFLOW_TEXT.count("INFERENCE_API_KEY:"))
        checkout = WORKFLOW_TEXT.split("uses: actions/checkout@", 1)[1].split(
            "      - name:", 1
        )[0]
        self.assertNotIn("ref:", checkout)
        self.assertIn("persist-credentials: false", checkout)


class MergeSnapshotWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        cls.repo = cls.root / "repo"
        cls.repo.mkdir()

        def git(*args):
            return subprocess.check_output(
                ["git", "-C", str(cls.repo), *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()

        git("init", "--initial-branch=main")
        git("config", "user.name", "CI Snapshot Test")
        git("config", "user.email", "snapshot@example.invalid")
        (cls.repo / "base.txt").write_text("base\n")
        git("add", ".")
        git("commit", "-m", "base")
        git("checkout", "-b", "pr")
        (cls.repo / "pr.txt").write_text("PR source\n")
        git("add", ".")
        git("commit", "-m", "PR")
        cls.head = git("rev-parse", "HEAD")
        git("checkout", "main")
        (cls.repo / "base.txt").write_text("advanced base\n")
        git("commit", "-am", "advance base")
        cls.base = git("rev-parse", "HEAD")
        git("merge", "--no-ff", "--no-edit", "pr")
        cls.merge = git("rev-parse", "HEAD")
        cls.tree = git("rev-parse", "HEAD^{tree}")
        cls.bin = cls.root / "bin"
        cls.bin.mkdir()
        gh = cls.bin / "gh"
        gh.write_text('#!/bin/sh\ncat "$SIMULATED_STATUSES"\n')
        gh.chmod(0o755)

    def verify(
        self,
        *,
        snapshot=None,
        scope="tests/gdn",
        base=None,
        head=None,
        event=None,
        required=False,
    ):
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            directory = Path(directory)
            statuses = [{"context": "ci/test-scope-1", "description": scope}]
            if snapshot is not None:
                statuses.append(
                    {"context": "ci/test-snapshot", "description": snapshot}
                )
            state = directory / "statuses.json"
            state.write_text(json.dumps({"statuses": statuses}))
            output = directory / "output"
            env = dict(
                os.environ,
                PATH=str(self.bin) + ":" + os.environ["PATH"],
                SIMULATED_STATUSES=str(state),
                GITHUB_OUTPUT=str(output),
                GITHUB_REPOSITORY="owner/repo",
                GH_TOKEN="test-token",
                BASE_SHA=base or self.base,
                HEAD_SHA=head or self.head,
                EVENT_SHA=event or self.merge,
                REQUIRE_SNAPSHOT=str(required).lower(),
            )
            result = subprocess.run(
                [
                    "bash",
                    "-euo",
                    "pipefail",
                    "-c",
                    step("Read requested test scope", PR_WORKFLOW_TEXT),
                ],
                cwd=self.repo,
                env=env,
                text=True,
                capture_output=True,
            )
            return result, output.read_text() if output.exists() else ""

    def test_matching_merged_commit_parents_and_tree_are_accepted(self):
        result, output = self.verify(
            snapshot=self.snapshot_description(), required=True
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("test_path=tests/gdn\n", output)
        self.assertIn("checkout_sha=" + self.head, output)

    def test_stale_github_event_does_not_replace_the_selected_merge(self):
        result, output = self.verify(
            snapshot=self.snapshot_description(),
            base="0" * 40,
            event="0" * 40,
            required=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("snapshot_tree=" + self.tree, output)
        self.assertIn("checkout_sha=" + self.head, output)

    def test_gpu_source_preparation_recovers_shallow_head_ancestry(self):
        with tempfile.TemporaryDirectory() as directory:
            clone = Path(directory) / "checkout"
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth=1",
                    "--branch=pr",
                    self.repo.as_uri(),
                    str(clone),
                ],
                check=True,
                capture_output=True,
            )
            result = subprocess.run(
                [
                    "bash",
                    "-euo",
                    "pipefail",
                    "-c",
                    step("Prepare selected source and submodules", PR_WORKFLOW_TEXT),
                ],
                cwd=clone,
                env=dict(os.environ, SELECTED_BASE=self.base, SELECTED_TREE=self.tree),
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            tree = subprocess.check_output(
                ["git", "rev-parse", "HEAD^{tree}"], cwd=clone, text=True
            ).strip()
            self.assertEqual(self.tree, tree)
            self.assertTrue((clone / "pr.txt").exists())

    def test_stale_or_malformed_automatic_snapshots_fail_before_scheduling(self):
        correct = self.snapshot_description()
        digest = hashlib.sha256(b"tests/gdn").hexdigest()[:16]
        for settings in (
            {"snapshot": "malformed"},
            {"snapshot": " ".join((self.base, self.merge, "0" * 40, digest))},
            {"snapshot": correct, "head": "0" * 40},
            {"snapshot": correct, "scope": ""},
            {"snapshot": correct, "scope": "tests/a-different-command"},
        ):
            with self.subTest(settings=settings):
                result, output = self.verify(**settings)
                self.assertNotEqual(0, result.returncode)
                self.assertNotIn("test_path=", output)

    def snapshot_description(self):
        digest = hashlib.sha256(b"tests/gdn").hexdigest()[:16]
        return " ".join((self.base, self.merge, self.tree, digest))

    def test_focused_label_never_falls_back_if_another_command_clears_metadata(self):
        for snapshot in (None, ""):
            with self.subTest(snapshot=snapshot):
                result, output = self.verify(snapshot=snapshot, scope="", required=True)
                self.assertNotEqual(0, result.returncode)
                self.assertNotIn("test_path=", output)

    def test_only_focused_and_existing_run_labels_authorize_pr_ci(self):
        for label, authorized in (
            ("run-ci-focused", True),
            ("run-ci", True),
            ("unrelated", False),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "output"
                body = step("Check authorization", PR_WORKFLOW_TEXT).replace(
                    "${{ github.event_name }}", "pull_request"
                )
                result = subprocess.run(
                    ["bash", "-euo", "pipefail", "-c", body],
                    env=dict(
                        os.environ,
                        GITHUB_OUTPUT=str(output),
                        EVENT_ACTION="labeled",
                        LABEL_NAME=label,
                    ),
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertEqual(
                    f"authorized={str(authorized).lower()}\n", output.read_text()
                )
        self.assertIn(
            "REQUIRE_SNAPSHOT: ${{ github.event.label.name == 'run-ci-focused' }}",
            PR_WORKFLOW_TEXT,
        )

    def test_manual_full_and_legacy_requests_do_not_require_snapshot_metadata(self):
        for snapshot in (None, ""):
            for requested_scope in ("", "tests/gdn"):
                with self.subTest(snapshot=snapshot, scope=requested_scope):
                    result, output = self.verify(
                        snapshot=snapshot, scope=requested_scope
                    )
                    self.assertEqual(0, result.returncode, result.stderr)
                    self.assertIn("test_path=" + requested_scope + "\n", output)

    def test_setup_and_test_jobs_checkout_immutable_commits(self):
        self.assertIn("ref: ${{ github.sha }}", PR_WORKFLOW_TEXT)
        self.assertIn("ref: ${{ needs.setup.outputs.checkout_sha }}", PR_WORKFLOW_TEXT)


if __name__ == "__main__":
    unittest.main()
