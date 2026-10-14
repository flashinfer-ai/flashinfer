"""CPU-only execution evidence fixtures; no provider calls or repository tests."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
import yaml
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ci_execution_manifest", ROOT / "scripts/pr_checks/ci_execution_manifest.py"
)
manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "checkout"
        self.workspace.mkdir()
        self.git_env = {
            "PATH": os.environ["PATH"],
            "HOME": str(self.root),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_AUTHOR_NAME": "Fixture",
            "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
            "GIT_COMMITTER_NAME": "Fixture",
            "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
        }
        self.git("init", "-q")
        self.git("-c", "commit.gpgsign=false", "commit", "--allow-empty", "-qm", "base")
        self.base = self.git("rev-parse", "HEAD")
        self.git("-c", "commit.gpgsign=false", "commit", "--allow-empty", "-qm", "head")
        self.head = self.git("rev-parse", "HEAD")
        tree = self.git("rev-parse", "HEAD^{tree}")
        self.merge = self.git(
            "commit-tree", tree, "-p", self.base, "-p", self.head, "-m", "merge"
        )
        self.git("reset", "--hard", self.merge)
        self.paths = ["tests/jit"]
        self.inputs = {
            "watcher_pr_number": "42",
            "watcher_trigger_comment_id": "123",
            "watcher_trigger_comment_url": "https://github.com/org/repo/pull/42#issuecomment-123",
            "watcher_head_sha": self.head,
            "watcher_base_sha": self.base,
            "watcher_checkout_sha": self.merge,
            "watcher_scope": " ".join(self.paths),
            "watcher_scope_digest": manifest.value_digest(self.paths),
        }
        self.inputs["watcher_request_key"] = (
            f"fi-ci/42/123/{self.head}/{self.inputs['watcher_scope_digest']}"
        )
        self.event = {"inputs": self.inputs}
        self.env = {
            **self.git_env,
            "GITHUB_REPOSITORY": "org/repo",
            "GITHUB_RUN_ID": "456",
            "GITHUB_RUN_ATTEMPT": "2",
            "GITHUB_WORKFLOW_SHA": "f" * 40,
            "GITHUB_EVENT_PATH": str(self.root / "event.json"),
            "GITHUB_OUTPUT": str(self.root / "outputs"),
        }
        Path(self.env["GITHUB_EVENT_PATH"]).write_text(json.dumps(self.event))
        self.lane = {
            "name": "Targeted Unittest (H100)",
            "kind": "h100",
            "runner": ["self-hosted", "h100"],
            "cuda": "cu129",
            "script": "task_run_unit_tests.sh",
            "test_path": "tests/jit",
            "timeout": 480,
        }
        self.matrix = {"include": [self.lane]}
        self.plan = manifest.create_plan(
            self.env, self.event, self.workspace, self.matrix, False
        )
        self.plan_path = self.root / "plan.json"
        manifest.write_json(self.plan_path, self.plan)
        self.output = self.root / "result"

    def git(self, *args):
        return (
            subprocess.check_output(
                ["git", *args],
                cwd=self.workspace,
                env=self.git_env,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )

    def initialize(self):
        return manifest.initialize(self.plan, self.lane, self.workspace, self.output)

    def test_plan_binds_real_merge_parents_matrix_and_attempt(self):
        self.assertEqual(self.plan["testedCheckoutSha"], self.merge)
        self.assertEqual(self.plan["checkoutParents"], [self.base, self.head])
        self.assertEqual(self.plan["requestedHeadSha"], self.head)
        self.assertEqual(self.plan["matrix"], self.matrix)
        self.assertEqual(self.plan["matrixDigest"], manifest.value_digest(self.matrix))
        self.assertEqual(self.plan["runAttempt"], 2)
        self.assertEqual(
            manifest.checked_plan(self.plan_path, self.env, self.event, self.workspace),
            self.plan,
        )
        with self.assertRaises(ValueError):
            manifest.checked_plan(
                self.plan_path,
                {**self.env, "GITHUB_RUN_ATTEMPT": "1"},
                self.event,
                self.workspace,
            )

    def test_rerun_inherits_original_setup_plan_without_fabricating_setup_attempt(self):
        later = {**self.env, "GITHUB_RUN_ATTEMPT": "3"}
        plan = manifest.checked_plan(self.plan_path, later, self.event, self.workspace)
        self.assertEqual(plan["runAttempt"], 2)
        result = manifest.initialize(
            plan, self.lane, self.workspace, self.output, run_attempt=3
        )
        self.assertEqual(result["runAttempt"], 3)
        self.assertEqual(result["planRunAttempt"], 2)
        self.assertEqual(result["matrixDigest"], self.plan["matrixDigest"])

    def test_checkout_ignores_inherited_commit_hook_git_overrides(self):
        foreign = self.root / "foreign-git"
        shutil.copytree(self.workspace / ".git", foreign)
        (foreign / "HEAD").write_text(self.base + "\n")
        with patch.dict(
            os.environ,
            {
                "GIT_DIR": str(foreign),
                "GIT_WORK_TREE": str(self.root),
                "GIT_INDEX_FILE": str(foreign / "index"),
            },
        ):
            self.assertEqual(
                manifest.checkout(self.workspace), (self.merge, [self.base, self.head])
            )

    def test_shallow_checkout_keeps_actual_commit_parents(self):
        (self.workspace / ".git/shallow").write_text(self.merge + "\n")
        self.assertEqual(
            manifest.checkout(self.workspace), (self.merge, [self.base, self.head])
        )

    def test_wrong_head_parent_matrix_or_request_key_rejected(self):
        for field, value in [
            ("watcher_head_sha", "a" * 40),
            ("watcher_scope_digest", "b" * 64),
            ("watcher_request_key", "wrong"),
        ]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                manifest.create_plan(
                    self.env,
                    {"inputs": {**self.inputs, field: value}},
                    self.workspace,
                    self.matrix,
                    False,
                )
        tampered = {**self.plan, "matrix": {"include": []}}
        manifest.write_json(self.plan_path, tampered)
        with self.assertRaises(ValueError):
            manifest.checked_plan(self.plan_path, self.env, self.event, self.workspace)

    def test_zero_matrix_and_skipped_build_do_not_invent_lanes(self):
        plan = manifest.create_plan(
            self.env, self.event, self.workspace, {"include": []}, True
        )
        self.assertEqual(plan["matrix"], {"include": []})
        self.assertNotIn("passed", manifest.canonical(plan))
        with self.assertRaises(ValueError):
            manifest.initialize(plan, self.lane, self.workspace, self.output)

    def test_initialization_records_no_command_or_hardware_assumption(self):
        result = self.initialize()
        self.assertFalse(result["commandStarted"])
        self.assertIsNone(result["exitCode"])
        self.assertEqual(result["laneKey"], manifest.value_digest(self.lane))
        self.assertEqual(result["hardware"], {"status": "not_recorded"})
        self.assertEqual(result["ranks"], {"status": "not_recorded"})
        self.assertEqual(manifest.finalize(self.output, self.workspace)["evidence"], [])

    def test_actual_failed_command_retains_exit_and_original_xml_metadata(self):
        self.initialize()
        xml = '<testsuite tests="1" failures="1"><testcase classname="tests.jit.test_x" name="test_x"><failure/></testcase></testsuite>'
        code = (
            "from pathlib import Path; import json; p=Path('junit'); (p/'batch.xml').write_text("
            + repr(xml)
            + "); (p/'batch.results.json').write_text('{\"nodeids\":[\"tests/jit/test_x.py::test_x\"]}'); raise SystemExit(3)"
        )
        self.assertEqual(
            manifest.run_command(
                self.output, self.workspace, [sys.executable, "-c", code]
            ),
            3,
        )
        result = manifest.finalize(self.output, self.workspace)
        self.assertTrue(result["commandStarted"])
        self.assertEqual(result["exitCode"], 3)
        self.assertEqual(
            {x["kind"] for x in result["evidence"]}, {"junit", "execution-results"}
        )
        for entry in result["evidence"]:
            data = (self.output / entry["path"]).read_bytes()
            self.assertEqual(manifest.digest(data), entry["sha256"])
            self.assertEqual(len(data), entry["size"])
        with self.assertRaises(ValueError):
            manifest.run_command(
                self.output,
                self.workspace,
                [sys.executable, "-c", "raise SystemExit(0)"],
            )

    def test_zero_cases_skips_and_no_xml_never_become_a_pass(self):
        self.initialize()
        code = 'from pathlib import Path; Path(\'junit/skipped.xml\').write_text(\'<testsuite tests="1" skipped="1"><testcase classname="x" name="y"><skipped/></testcase></testsuite>\')'
        self.assertEqual(
            manifest.run_command(
                self.output, self.workspace, [sys.executable, "-c", code]
            ),
            0,
        )
        result = manifest.finalize(self.output, self.workspace)
        self.assertNotIn("verdict", result)
        self.assertNotIn("passed", result)
        self.assertEqual(result["evidence"][0]["kind"], "junit")

    def test_spawn_failure_is_not_a_started_command(self):
        self.initialize()
        self.assertEqual(
            manifest.run_command(
                self.output, self.workspace, [str(self.root / "missing")]
            ),
            127,
        )
        result = manifest.json_value((self.output / "result.json").read_bytes())
        self.assertFalse(result["commandStarted"])
        self.assertEqual(result["exitCode"], 127)

    def test_preexisting_or_linked_junit_and_linked_artifacts_fail_closed(self):
        (self.workspace / "junit").mkdir()
        with self.assertRaises(ValueError):
            self.initialize()
        (self.workspace / "junit").rmdir()
        self.output.rmdir()
        self.initialize()
        private = self.root / "private.xml"
        private.write_text("do not publish")
        (self.workspace / "junit/link.xml").symlink_to(private)
        result = manifest.finalize(self.output, self.workspace)
        self.assertTrue(result["artifactErrors"])
        self.assertEqual(result["evidence"], [])
        self.assertFalse((self.output / "evidence/link.xml").exists())

    def test_size_and_ancestor_link_bounds(self):
        self.initialize()
        (self.workspace / "junit/oversize.xml").write_bytes(b"x" * 100)
        with patch.object(manifest, "MAX_FILE", 32):
            result = manifest.finalize(self.output, self.workspace)
        self.assertTrue(result["artifactErrors"])
        other = self.root / "other"
        other.mkdir()
        (other / "data.xml").write_text("private")
        (self.workspace / "junit/dir").symlink_to(other)
        with self.assertRaises(OSError):
            manifest.read_regular(
                self.workspace / "junit/dir/data.xml", root=self.workspace / "junit"
            )

    def test_lane_scope_normalizes_actual_intersection_order(self):
        lane = {**self.lane, "test_path": "tests/z.py tests/a.py"}
        plan = {**self.plan, "matrix": {"include": [lane]}}
        result = manifest.initialize(plan, lane, self.workspace, self.output)
        self.assertEqual(result["testTargets"], ["tests/a.py", "tests/z.py"])

    def test_checkout_drift_refuses_execution(self):
        self.initialize()
        self.git("reset", "--hard", self.head)
        with self.assertRaises(ValueError):
            manifest.run_command(
                self.output,
                self.workspace,
                [sys.executable, "-c", "raise SystemExit(0)"],
            )
        self.assertTrue(
            manifest.finalize(self.output, self.workspace)["artifactErrors"]
        )

    def actual_workflow_lane(self, watcher, kind, target, exit_code):
        workflow = yaml.safe_load((ROOT / ".github/workflows/pr-test.yml").read_text())
        condition = "matrix.kind == 'aot'" if kind == "aot" else "matrix.kind != 'aot'"
        step = next(
            s
            for s in workflow["jobs"]["test"]["steps"]
            if s.get("name") == "Run Test" and s["if"] == condition
        )
        ci = self.workspace / "ci"
        ci.mkdir()
        fake = ci / "bash.sh"
        fake.write_text(
            '#!/bin/bash\nprintf \'%s\\n\' "$@" > "$CALLS"\nexit "$FIXTURE_EXIT"\n'
        )
        tool_dir = self.root / "watcher-ci-tool-456-2"
        tool_dir.mkdir()
        shutil.copy(
            ROOT / "scripts/pr_checks/ci_execution_manifest.py", tool_dir / "writer.py"
        )
        if watcher:
            self.output = self.root / "watcher-ci-result-456-2"
            self.initialize()
        environment = {
            **self.env,
            "RUNNER_TEMP": str(self.root),
            "GITHUB_WORKSPACE": str(self.workspace),
            "WATCHER_REQUEST_KEY": self.inputs["watcher_request_key"]
            if watcher
            else "",
            "WATCHER_CHECKOUT_SHA": self.merge,
            "DOCKER_IMAGE": "fixture-image",
            "TEST_SCRIPT": "task_test_jit_cache_package_build_import.sh"
            if kind == "aot"
            else "task_run_unit_tests.sh",
            "TEST_PATH_ARG": target,
            "CALLS": str(self.root / "calls"),
            "FIXTURE_EXIT": str(exit_code),
            "SCCACHE_BUCKET": "",
            "SCCACHE_REGION": "",
        }
        result = subprocess.run(
            ["/bin/bash", "-e", "-c", step["run"]],
            cwd=self.workspace,
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, exit_code, result.stderr)
        argv = (self.root / "calls").read_text().splitlines()
        if watcher:
            self.assertEqual(argv[1:4], ["-e", "SOURCE_GIT_SHA", self.merge])
            recorded = manifest.json_value((self.output / "result.json").read_bytes())
            self.assertTrue(recorded["commandStarted"])
            self.assertEqual(recorded["exitCode"], exit_code)
        else:
            self.assertNotIn("SOURCE_GIT_SHA", argv)
            self.assertFalse(self.output.exists())
        return argv

    def test_real_watcher_gpu_wrapper_preserves_scope_and_failure(self):
        argv = self.actual_workflow_lane(True, "gpu", "tests/jit tests/utils", 17)
        self.assertEqual(
            argv[-4:],
            [
                "./scripts/task_run_unit_tests.sh",
                "--test-path",
                "tests/jit",
                "tests/utils",
            ],
        )

    def test_real_legacy_gpu_wrapper_unchanged(self):
        argv = self.actual_workflow_lane(False, "gpu", "tests/jit", 0)
        self.assertEqual(
            argv,
            [
                "fixture-image",
                "./scripts/task_run_unit_tests.sh",
                "--test-path",
                "tests/jit",
            ],
        )

    def test_real_watcher_aot_wrapper_preserves_failure_and_no_gpu(self):
        argv = self.actual_workflow_lane(True, "aot", "", 9)
        self.assertIn("--no-gpu", argv)
        self.assertEqual(
            argv[-1], "./scripts/task_test_jit_cache_package_build_import.sh"
        )

    def test_real_cli_plan_initialize_run_finalize(self):
        matrix_path = self.root / "matrix.json"
        matrix_path.write_text(json.dumps(self.matrix))
        tool = str(ROOT / "scripts/pr_checks/ci_execution_manifest.py")

        def cli(mode, *args, expected=0):
            command = [
                sys.executable,
                "-I",
                tool,
                mode,
                "--workspace",
                str(self.workspace),
                *args,
            ]
            result = subprocess.run(
                command, env=self.env, capture_output=True, text=True, timeout=15
            )
            self.assertEqual(result.returncode, expected, result.stderr)

        cli("plan", "--matrix", str(matrix_path), "--output", str(self.plan_path))
        cli(
            "initialize",
            "--plan",
            str(self.plan_path),
            "--lane",
            json.dumps(self.lane),
            "--output",
            str(self.output),
        )
        cli(
            "run",
            "--output",
            str(self.output),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(7)",
            expected=7,
        )
        cli("finalize", "--output", str(self.output))
        self.assertIn(
            "lane_key=" + manifest.value_digest(self.lane),
            Path(self.env["GITHUB_OUTPUT"]).read_text(),
        )
        self.assertEqual(
            manifest.json_value((self.output / "result.json").read_bytes())["exitCode"],
            7,
        )


if __name__ == "__main__":
    unittest.main()
