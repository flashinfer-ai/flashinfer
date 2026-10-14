"""Workflow boundaries plus execution of the real scope/matrix shell snippets.

No GitHub requests, runner allocations or repository scripts are executed.
"""

import json
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]


def workflow(name):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


def step_named(job, name):
    return next(step for step in job["steps"] if step.get("name") == name)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.test = workflow("pr-test.yml")
        self.commands = workflow("ci-bot-commands.yml")
        self.receipts = workflow("ci-request-receipts.yml")

    def test_privileged_command_and_receipt_jobs_never_checkout_requested_code(self):
        jobs = [
            self.commands["jobs"]["handle-command"],
            *self.receipts["jobs"].values(),
        ]
        for job in jobs:
            for step in job["steps"]:
                if step.get("uses", "").startswith("actions/checkout@"):
                    settings = step.get("with", {})
                    self.assertIs(settings["persist-credentials"], False)
                    # issue_comment defaults to the default branch; workflow_run
                    # helpers explicitly pin their own trusted workflow revision.
                    self.assertIn(
                        settings.get("ref"), (None, "${{ github.workflow_sha }}")
                    )
                    self.assertNotIn("repository", settings)
                if "run" in step:
                    self.assertNotIn("watcher_checkout_sha", step["run"])

    def test_requested_source_jobs_have_readonly_token_and_validate_before_gate(self):
        self.assertTrue(
            all(
                value in ("read", "none") for value in self.test["permissions"].values()
            )
        )
        for job in self.test["jobs"].values():
            self.assertTrue(
                all(
                    value in ("read", "none")
                    for value in job.get("permissions", {}).values()
                )
            )
        gate = self.test["jobs"]["gate"]
        names = [s.get("name") for s in gate["steps"]]
        self.assertLess(
            names.index("Validate correlated request"),
            names.index("Check authorization"),
        )
        validator = step_named(gate, "Validate correlated request")
        self.assertEqual(validator["if"], "github.event_name == 'workflow_dispatch'")
        self.assertNotIn("continue-on-error", validator)
        self.assertEqual(
            gate["outputs"]["request_is_fork"], "${{ steps.request.outputs.is_fork }}"
        )
        for name in ("setup", "test"):
            job = self.test["jobs"][name]
            self.assertIn("gate", job["needs"])
            self.assertIn("needs.gate.outputs.authorized == 'true'", job["if"])
            checkouts = [
                s
                for s in job["steps"]
                if s.get("uses", "").startswith("actions/checkout@")
            ]
            self.assertTrue(checkouts)
            for checkout in checkouts:
                if checkout.get("name") == "Checkout trusted execution evidence writer":
                    self.assertEqual(
                        checkout["with"]["ref"], "${{ github.workflow_sha }}"
                    )
                    self.assertEqual(checkout["if"], "inputs.watcher_request_key != ''")
                    self.assertEqual(checkout["with"]["path"], ".ci-evidence-trusted")
                else:
                    self.assertEqual(
                        checkout["with"]["ref"],
                        "${{ inputs.watcher_checkout_sha || github.sha }}",
                    )
                self.assertIs(checkout["with"]["persist-credentials"], False)

    def test_requested_source_secret_access_requires_verified_internal_route(self):
        found = []
        for job in self.test["jobs"].values():
            for step in job["steps"]:
                for key, value in step.get("env", {}).items():
                    if (
                        isinstance(value, str)
                        and "secrets." in value
                        and "secrets.GITHUB_TOKEN" not in value
                    ):
                        found.append((key, value))
                        # Missing/fork proof must evaluate false when a request
                        # key exists. Legacy pull_request secret restrictions are
                        # retained by GitHub rather than expanded by this route.
                        self.assertIn(
                            "needs.setup.outputs.request_is_fork == 'false'", value
                        )
                        self.assertIn("!inputs.watcher_request_key", value)
                        self.assertTrue(value.endswith("|| '' }}"))
        self.assertEqual(
            {key for key, _ in found}, {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}
        )
        self.assertEqual(
            self.test["jobs"]["setup"]["outputs"]["request_is_fork"],
            "${{ needs.gate.outputs.request_is_fork }}",
        )

    def test_execution_manifests_are_watcher_only_and_keep_attempt_history(self):
        setup = self.test["jobs"]["setup"]
        test = self.test["jobs"]["test"]
        plan = step_named(setup, "Upload watcher CI plan")
        lane = step_named(test, "Upload watcher lane execution evidence")
        for step in [plan, lane]:
            self.assertIn("inputs.watcher_request_key != ''", step["if"])
            self.assertIn(
                "${{ github.run_id }}-${{ github.run_attempt }}", step["with"]["name"]
            )
            self.assertNotIn("overwrite", step["with"])
        self.assertIn("lane_key", lane["with"]["name"])
        download = step_named(test, "Download watcher CI plan")
        self.assertEqual(
            download["with"]["name"], "${{ needs.setup.outputs.watcher_plan_artifact }}"
        )
        self.assertEqual(
            setup["outputs"]["watcher_plan_artifact"],
            "${{ steps.watcher-plan.outputs.artifact_name }}",
        )
        finish = step_named(test, "Finalize watcher lane evidence")
        self.assertIn("always()", finish["if"])
        self.assertIn("steps.watcher-lane.outcome", finish["if"])
        for step in test["steps"]:
            if step.get("name") == "Run Test":
                self.assertIn('writer.py" run', step["run"])
                self.assertIn('-e SOURCE_GIT_SHA "$WATCHER_CHECKOUT_SHA"', step["run"])
                self.assertNotIn("continue-on-error", step)
                self.assertIn("/watcher-ci-evidence:ro", step["run"])
                self.assertIn(
                    "/watcher-ci-evidence/hardware.py run --kind", step["run"]
                )

    def test_dispatch_does_not_also_trigger_a_label_run(self):
        job = self.commands["jobs"]["handle-command"]
        dispatch = step_named(job, "Handle @flashinfer-bot run")
        self.assertIn("ci_request_producer.py dispatch", dispatch["run"])
        route = (
            dispatch["run"][dispatch["run"].index("          ROUTE=") :]
            if "          ROUTE=" in dispatch["run"]
            else dispatch["run"][dispatch["run"].index("ROUTE=") :]
        )
        watcher = route[: route.index("# There is no DELETE")]
        self.assertIn('if [ "$ROUTE" = watcher ]', watcher)
        self.assertIn("exit 0", watcher)
        self.assertNotRegex(watcher, r"--(?:add|remove)-label")
        self.assertIn(
            '--add-label "run-ci"', route[route.index("# There is no DELETE") :]
        )
        self.assertIn(
            "steps.check-permission.outputs.authorized == 'true'", dispatch["if"]
        )
        self.assertIs(job["concurrency"]["cancel-in-progress"], False)
        self.assertEqual(job["concurrency"]["queue"], "max")
        report = self.receipts["jobs"]["report"]
        self.assertIs(report["concurrency"]["cancel-in-progress"], False)
        self.assertEqual(report["concurrency"]["queue"], "max")
        self.assertTrue(job["concurrency"]["group"].startswith("watcher-ci-pr-"))
        self.assertTrue(report["concurrency"]["group"].startswith("watcher-ci-pr-"))

    def test_actual_route_exits_before_legacy_label_for_watcher_only(self):
        body = step_named(
            self.commands["jobs"]["handle-command"], "Handle @flashinfer-bot run"
        )["run"]
        route = body[body.index("ROUTE=") : body.index("# There is no DELETE")]
        route = re.sub(r"\$\{\{.*?\}\}", "fixture", route)
        for mode in ("watcher", "legacy"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                fake = directory / "python3"
                fake.write_text(
                    '#!/bin/sh\necho "$2" >> "$CALLS"\nif [ "$2" = route ]; then echo "$ROUTE_MODE"; else echo \'{"state":"completed"}\'; fi\n'
                )
                fake.chmod(0o700)
                gh = directory / "gh"
                gh.write_text("#!/bin/sh\nexit 0\n")
                gh.chmod(0o700)
                jq = shutil.which("jq")
                env = {
                    "PATH": str(directory)
                    + ":"
                    + str(Path(jq).parent)
                    + ":/usr/bin:/bin",
                    "ROUTE_MODE": mode,
                    "CALLS": str(directory / "calls"),
                }
                result = subprocess.run(
                    ["/bin/bash", "-e", "-c", route + "\necho LEGACY_LABEL_ROUTE"],
                    env=env,
                    cwd=directory,
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = (directory / "calls").read_text().splitlines()
                self.assertEqual(
                    calls, ["route", "dispatch"] if mode == "watcher" else ["route"]
                )
                self.assertEqual(
                    "LEGACY_LABEL_ROUTE" in result.stdout, mode == "legacy"
                )

    def test_recovery_uses_established_bot_secret_and_reports_on_terminal_event(self):
        event = self.receipts.get("on", self.receipts.get(True))["workflow_run"]
        self.assertEqual(event["workflows"], ["PR Test"])
        self.assertEqual(event["types"], ["requested", "in_progress", "completed"])
        report = self.receipts["jobs"]["report"]
        progress = step_named(
            report, "Mark correlated reruns pending on the requested PR head"
        )
        terminal = step_named(
            report, "Report real provider checks on the requested PR head"
        )
        self.assertEqual(progress["if"], "github.event.action != 'completed'")
        self.assertEqual(terminal["if"], "github.event.action == 'completed'")
        dispatch = step_named(
            self.commands["jobs"]["handle-command"], "Handle @flashinfer-bot run"
        )
        recovery = step_named(
            self.receipts["jobs"]["report"], "Recover exact root and complete receipt"
        )
        self.assertEqual(recovery["env"]["GH_TOKEN"], dispatch["env"]["GH_TOKEN"])

    def test_actual_rerun_failure_branches_report_uncertainty_and_exit(self):
        job = self.commands["jobs"]["handle-command"]
        for name in (
            "Handle @flashinfer-bot rerun",
            "Handle @flashinfer-bot rerun failed",
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                source = step_named(job, name)["run"]
                lines = source[
                    source.index(
                        "python3 scripts/pr_checks/ci_request_producer.py prepare-rerun"
                    ) :
                ].splitlines()
                end = next(i for i, line in enumerate(lines) if line.strip() == "fi")
                branch = re.sub(
                    r"\$\{\{.*?\}\}", "owner/repo", "\n".join(lines[: end + 1])
                )
                for binary, text in (
                    ("gh", '#!/bin/sh\necho rerun >> "$CALLS"\nexit 42\n'),
                    ("python3", '#!/bin/sh\necho "$2" >> "$ACTIONS"\nexit 0\n'),
                ):
                    path = directory / binary
                    path.write_text(text)
                    path.chmod(0o700)
                result = subprocess.run(
                    [
                        "/bin/bash",
                        "-e",
                        "-c",
                        "run_id=456\n" + branch + "\necho THUMBSUP",
                    ],
                    env={
                        "PATH": str(directory) + ":/usr/bin:/bin",
                        "CALLS": str(directory / "calls"),
                        "ACTIONS": str(directory / "actions"),
                    },
                    cwd=directory,
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("THUMBSUP", result.stdout)
                self.assertEqual(
                    (directory / "calls").read_text().splitlines(), ["rerun"]
                )
                self.assertEqual(
                    (directory / "actions").read_text().splitlines(),
                    ["prepare-rerun", "rerun-submit-failed"],
                )

    def run_shell(self, source, environment):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fake = directory / "gh"
            fake.write_text("#!/bin/sh\necho unexpected-provider-call >&2\nexit 97\n")
            fake.chmod(0o700)
            # Use the current trusted helper/shard definitions; never execute shards.
            (directory / "scripts/pr_checks").mkdir(parents=True)
            shutil.copy(
                ROOT / "scripts/pr_checks/targeted_lane_scope.py",
                directory / "scripts/pr_checks",
            )
            for shard_path in (ROOT / "scripts").glob("task_jit_run_tests_*.sh"):
                shutil.copy(shard_path, directory / "scripts")
            (directory / "ci").mkdir()
            shutil.copy(ROOT / "ci/docker-tags.yml", directory / "ci/docker-tags.yml")
            output = directory / "outputs"
            jq = shutil.which("jq")
            binary_path = (str(Path(jq).parent) + ":") if jq else ""
            env = {
                "PATH": str(directory) + ":" + binary_path + "/usr/bin:/bin",
                "GITHUB_OUTPUT": str(output),
                "GITHUB_REPOSITORY": "owner/repo",
                **environment,
            }
            self.assertIsInstance(source, str)
            result = subprocess.run(
                ["/bin/bash", "-e", "-c", source],
                cwd=directory,
                env=env,
                text=True,
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return output.read_text(), json.loads(
                (directory / "matrix.json").read_text()
            ) if (directory / "matrix.json").exists() else None

    def test_real_scope_script_uses_exact_inputs_without_status_lookup(self):
        script = step_named(self.test["jobs"]["setup"], "Read requested test scope")[
            "run"
        ]
        for scope in ("", "tests/utils tests/attention/test_x.py"):
            with self.subTest(scope=scope):
                output, _ = self.run_shell(
                    script,
                    {
                        "REQUEST_KEY": "fi-ci/exact",
                        "EXPLICIT_SCOPE": scope,
                        "HEAD_SHA": "a" * 40,
                    },
                )
                self.assertEqual(output, "test_path=" + scope + "\n")

    @unittest.skipUnless(
        shutil.which("jq"), "jq required to execute the workflow's real matrix builder"
    )
    def test_real_matrix_preserves_current_lane_intersections(self):
        script = step_named(self.test["jobs"]["setup"], "Build Test Matrix")["run"]
        output, matrix = self.run_shell(
            script,
            {
                "SKIP_AOT": "false",
                "SKIP_GPU": "false",
                "RUNTIME_CUDA_MATRIX": '[{"label":"cu129"},{"label":"cu130"}]',
                "REQUESTED_TEST_PATH": "tests/utils",
            },
        )
        aot = [job for job in matrix["include"] if job["kind"] == "aot"]
        gpu = [job for job in matrix["include"] if job["kind"] != "aot"]
        self.assertEqual(len(aot), 4)
        self.assertEqual(len(gpu), 3)
        for job in gpu:
            paths = job["test_path"].split()
            self.assertTrue(paths)
            self.assertTrue(
                all(
                    path == "tests/utils" or path.startswith("tests/utils/")
                    for path in paths
                )
            )
            self.assertEqual(job["script"], "task_run_unit_tests.sh")
        self.assertIn("count=7", output)

    @unittest.skipUnless(shutil.which("jq"), "jq required for actual matrix builder")
    def test_moe_ep_scope_does_not_manufacture_blackwell_multirank_coverage(self):
        script = step_named(self.test["jobs"]["setup"], "Build Test Matrix")["run"]
        _, matrix = self.run_shell(
            script,
            {
                "SKIP_AOT": "false",
                "SKIP_GPU": "false",
                "RUNTIME_CUDA_MATRIX": '[{"label":"cu129"},{"label":"cu130"}]',
                "REQUESTED_TEST_PATH": "tests/moe_ep",
            },
        )
        gpu = [job for job in matrix["include"] if job["kind"] != "aot"]
        self.assertEqual([job["kind"] for job in gpu], ["h100"])
        self.assertEqual(gpu[0]["test_path"], "tests/moe_ep")
        self.assertFalse(any("sm100" in job.get("runner", []) for job in gpu))


if __name__ == "__main__":
    unittest.main()
