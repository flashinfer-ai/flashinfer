"""Exercise workflow safety and reporting with external commands stubbed."""

import copy
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
WORKFLOWS = Path(__file__).resolve().parents[3] / ".github/workflows"
REPO = "flashinfer-ai/flashinfer"
HEAD, MERGE, BASE, TREE = (character * 40 for character in "1234")


def workflow(name):
    return yaml.safe_load((WORKFLOWS / name).read_text())


def step_script(filename, job, name):
    return next(
        s["run"]
        for s in workflow(filename)["jobs"][job]["steps"]
        if s.get("name") == name
    )


def render(script, values):
    return re.sub(r"\$\{\{\s*(.*?)\s*\}\}", lambda match: values[match[1]], script)


def bash(script, tmp_path, **env):
    return subprocess.run(
        ["bash", "-eu", "-c", script],
        env={**os.environ, "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"), **env},
        capture_output=True,
        text=True,
        timeout=5,
    )


def test_runner_jobs_depend_on_trusted_admission_and_source():
    dispatcher = workflow("pr-test-dispatch.yml")["jobs"]
    assert dispatcher["test"]["uses"] == f"{REPO}/.github/workflows/pr-test.yml@main"
    assert "test" in dispatcher["test-results-summary"]["needs"]
    report = next(
        s
        for s in dispatcher["test-results-summary"]["steps"]
        if s.get("name") == "Report unauthorized PR status"
    )
    assert "python3 scripts/infra/pr_checks/pr_test_history.py" in report["run"]
    runners = workflow("pr-test.yml")["jobs"]
    assert runners["authorize"]["runs-on"] == "ubuntu-latest"
    for name in ("test", "spec-lint"):
        job = runners[name]
        needs = job["needs"]
        assert "authorize" in ([needs] if isinstance(needs, str) else needs)
        checkout = next(
            s for s in job["steps"] if s.get("uses", "").startswith("actions/checkout@")
        )
        assert checkout["with"]["ref"] == "${{ needs.authorize.outputs.checkout_sha }}"
        prepare = next(
            s
            for s in job["steps"]
            if s.get("name") == "Prepare selected source and submodules"
        )
        assert (
            prepare["env"]["SELECTED_BASE"]
            == "${{ needs.authorize.outputs.snapshot_base }}"
        )
        assert (
            prepare["env"]["SELECTED_TREE"]
            == "${{ needs.authorize.outputs.snapshot_tree }}"
        )


@pytest.mark.parametrize("job", ["test", "spec-lint"])
def test_cleanup_removes_only_containers_mounted_to_this_workspace(tmp_path, job):
    workspace = tmp_path / "checkout"
    containers = {
        "own-running": (str(workspace), True),
        "own-stopped": (str(workspace), False),
        "peer": (str(workspace) + "-peer", True),
        "other": (str(tmp_path / "other-checkout"), True),
        "peer-stopped": (str(tmp_path / "other-checkout"), False),
    }
    log = tmp_path / "docker-calls"
    stub = tmp_path / "docker"
    stub.write_text(
        "#!"
        + sys.executable
        + "\n"
        + r"""
import json, os, sys
args = sys.argv[1:]
containers = json.loads(os.environ["CONTAINERS"])
if args[0] == "ps":
    print("\n".join(cid for cid, (_, running) in containers.items() if running or "a" in args[1]))
elif args[0] == "inspect":
    print(containers[args[1]][0])
elif args[:2] == ["container", "prune"]:
    with open(os.environ["DOCKER_CALLS"], "a") as log:
        log.write(json.dumps(["prune"] + [cid for cid, (_, running) in containers.items() if not running]) + "\n")
elif args[0] in {"rm", "stop"}:
    with open(os.environ["DOCKER_CALLS"], "a") as log:
        log.write(json.dumps(args) + "\n")
"""
    )
    stub.chmod(0o755)
    for command in ("sudo", "nvidia-smi"):
        stub = tmp_path / command
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)
    cleanup_steps = [
        step
        for step in workflow("pr-test.yml")["jobs"][job]["steps"]
        if step.get("name") == "Cleanup"
    ]
    for step in cleanup_steps:
        result = bash(
            render(step["run"], {"github.workspace": str(workspace)}),
            tmp_path,
            PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
            HOME=str(tmp_path / "home"),
            CI_JOB_WORKSPACE=str(workspace),
            CONTAINERS=json.dumps(containers),
            DOCKER_CALLS=str(log),
            CUDA_VISIBLE_DEVICES="0,1",
        )
        assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert sorted(calls) == [["rm", "-f", "own-running"], ["rm", "-f", "own-stopped"]]


@pytest.fixture
def require_jq():
    if shutil.which("jq") is None:
        pytest.skip("jq is required to execute the workflow shell code")


@pytest.fixture
def check_results(tmp_path, require_jq):
    """Run the actual verdict and diagnostics code against a saved job response."""

    def check(test_lanes, *, api_available=True, aggregate="failure"):
        jobs = [
            dict(
                name="Self-hosted checks / " + name,
                run_id=42,
                run_attempt=1,
                status="completed",
                conclusion=outcome,
                html_url=f"https://example.com/{outcome}",
            )
            for name, outcome in test_lanes
            + [("Provider Spec Lint (cu130)", "failure")]
        ]
        stub = tmp_path / "gh"
        stub.write_text(
            "#!"
            + sys.executable
            + "\n"
            + r"""
import os, subprocess, sys
if os.environ["API_AVAILABLE"] == "false":
    raise SystemExit(1)
print(subprocess.check_output(["jq", "-c", sys.argv[sys.argv.index("--jq") + 1]], input=os.environ["JOBS"], text=True).strip())
"""
        )
        stub.chmod(0o755)
        script = render(
            step_script(
                "pr-test-dispatch.yml", "test-results-summary", "Check Results"
            ),
            {
                "needs.setup.result": "success",
                "needs.setup.outputs.skip_build": "false",
                "needs.setup.outputs.test_count": str(len(test_lanes)),
                "needs.setup.outputs.spec_lint_count": "1",
                "needs.test.result": aggregate,
            },
        )
        result = bash(
            script,
            tmp_path,
            PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
            RUNNER_TEMP=str(tmp_path),
            GH_REPO=REPO,
            RUN_ID="42",
            RUN_ATTEMPT="1",
            TEST_MATRIX=json.dumps(
                {"include": [{"name": name} for name, _ in test_lanes]}
            ),
            SPEC_LINT_MATRIX=json.dumps({"include": [{"cuda": "cu130"}]}),
            JOBS=json.dumps({"jobs": jobs}),
            API_AVAILABLE=str(api_available).lower(),
            H100_DISABLED="false",
            SKIP_ALL_FOUND="false",
            TRACE_GAPS="false",
        )
        assert result.returncode == (0 if aggregate == "success" else 1), result.stderr
        summary = (tmp_path / "summary").read_text()
        assert f"Self-hosted checks: {aggregate}" in summary
        assert (
            "Result: Tests Passed" if aggregate == "success" else "Result: Tests Failed"
        ) in summary
        return summary

    return check


def test_summary_identifies_lint_failure_when_tests_pass(check_results):
    summary = check_results(
        [("CPU Unittest", "success"), ("Targeted Unittest (H100)", "success")]
    )
    assert "Test matrix: 2 success" in summary
    assert "Provider spec lint: 1 failure" in summary
    assert "All planned test matrix lanes completed successfully" in summary
    for name, outcome in [
        ("CPU Unittest", "success"),
        ("Targeted Unittest (H100)", "success"),
        ("Provider Spec Lint (cu130)", "failure"),
    ]:
        assert re.search(
            re.escape(name)
            + r" \| "
            + outcome
            + r" \| .*"
            + re.escape(f"https://example.com/{outcome}"),
            summary,
        )


def test_summary_reports_lint_without_test_lanes(check_results):
    assert "Provider spec lint: 1 failure" in check_results([])


@pytest.mark.parametrize("aggregate", ["success", "failure"])
def test_diagnostics_unavailable_does_not_change_required_verdict(
    check_results, aggregate
):
    assert "Self-hosted lane outcomes unavailable" in check_results(
        [], api_available=False, aggregate=aggregate
    )


@pytest.mark.usefixtures("require_jq")
@pytest.mark.parametrize(
    "lane", ["AOT Build Import (x64, cu130)", "JIT Unittest (H100)"]
)
def test_cleanup_keeps_real_runs_even_when_setup_is_skipped(lane):
    script = workflow("pr-test-run-cleanup.yml")["jobs"]["discard"]["steps"][0]["run"]
    query = re.search(r"--jq '\n(.*?)' 2>", script, re.S).group(1)
    result = subprocess.run(
        ["jq", "-r", query],
        input=json.dumps(
            {
                "jobs": [
                    {"name": "Setup", "conclusion": "skipped"},
                    {"name": "Self-hosted checks / " + lane},
                ]
            }
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "tests"


@pytest.fixture
def admit(tmp_path, monkeypatch):
    """Run the trusted gate with an event and optional published status pages."""
    event = {
        "action": "labeled",
        "label": {"name": "run-ci"},
        "pull_request": {
            "head": {"sha": HEAD},
            "base": {"ref": "main", "repo": {"full_name": REPO}},
        },
    }
    script = step_script("pr-test.yml", "authorize", "Check trusted triggering event")

    def check(*, event_update=None, statuses=None, parents=None, **env):
        payload = copy.deepcopy(event)
        if event_update:
            event_update(payload)
        event_path, output = tmp_path / "event.json", tmp_path / "output"
        event_path.write_text(json.dumps(payload))

        def read_status(command, **kwargs):
            if "/git/commits/" in command[-1]:
                return json.dumps(
                    dict(
                        parents=[
                            dict(sha=sha)
                            for sha in (
                                parents if parents is not None else [BASE, HEAD]
                            )
                        ],
                        tree=dict(sha=TREE),
                    )
                )
            assert statuses is not None
            assert f"/commits/{HEAD}/status" in command[-1]
            return "\n".join(json.dumps({"statuses": page}) for page in statuses)

        with monkeypatch.context() as patch:
            for key, value in {
                "GITHUB_EVENT_PATH": str(event_path),
                "GITHUB_OUTPUT": str(output),
                "RUNNER_TEMP": str(tmp_path),
                "GITHUB_EVENT_NAME": "pull_request",
                "GITHUB_REPOSITORY": REPO,
                "GITHUB_REF": "refs/pull/42/merge",
                "GITHUB_SHA": MERGE,
                "INPUT_CHECKOUT": MERGE,
                "INPUT_BASE": "",
                "INPUT_TREE": "",
                **env,
            }.items():
                patch.setenv(key, value)
            patch.setattr(subprocess, "check_output", read_status)
            exec(compile(script, "pr-test.yml:authorize", "exec"), {})
        outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
        assert outputs.pop("authorized") == "true"
        if env.get("GITHUB_EVENT_NAME", "pull_request") == "pull_request":
            assert json.loads((tmp_path / "pr-test-source.json").read_text()) == dict(
                head=HEAD, base=BASE, tree=TREE
            )
        return outputs

    return check


@pytest.mark.parametrize(
    "event,ref",
    [
        ("pull_request", "refs/pull/42/merge"),
        ("schedule", "refs/heads/main"),
        ("workflow_dispatch", "refs/heads/main"),
    ],
)
def test_admission_accepts_approved_no_snapshot_events(admit, event, ref):
    assert admit(GITHUB_EVENT_NAME=event, GITHUB_REF=ref)["checkout_sha"] == MERGE


@pytest.mark.parametrize("parents", [[BASE], [BASE, "f" * 40]])
def test_ordinary_run_rejects_merge_not_containing_event_head(admit, parents):
    with pytest.raises(SystemExit, match="merge does not match"):
        admit(parents=parents)


@pytest.mark.parametrize(
    "event_update,env",
    [
        (lambda event: event.update(action="synchronize"), {}),
        (lambda event: event["label"].update(name="unrelated"), {}),
        (lambda event: event["pull_request"]["base"].update(ref="other"), {}),
        (
            lambda event: event["pull_request"]["base"]["repo"].update(
                full_name="outsider/repo"
            ),
            {},
        ),
        (None, {"GITHUB_REPOSITORY": "outsider/repo"}),
        (None, {"GITHUB_EVENT_NAME": "issue_comment"}),
        (
            None,
            {
                "GITHUB_EVENT_NAME": "workflow_dispatch",
                "GITHUB_REF": "refs/heads/feature",
            },
        ),
    ],
    ids=[
        "push-with-label-present",
        "wrong-label",
        "wrong-base",
        "wrong-base-repo",
        "wrong-repo",
        "wrong-event",
        "manual-feature-branch",
    ],
)
def test_admission_rejects_unapproved_events(admit, event_update, env):
    with pytest.raises(SystemExit):
        admit(event_update=event_update, **env)


@pytest.mark.parametrize(
    "label,paginated", [("run-ci-focused", True), ("run-ci", False)]
)
def test_snapshot_admission_returns_verified_source(admit, label, paginated):
    statuses = published_snapshot()
    pages = [[status] for status in statuses] if paginated else [statuses]
    outputs = admit(
        event_update=lambda event: event["label"].update(name=label),
        statuses=pages,
        INPUT_CHECKOUT=HEAD,
        INPUT_BASE=BASE,
        INPUT_TREE=TREE,
    )
    assert {
        key: outputs[key] for key in ("checkout_sha", "snapshot_base", "snapshot_tree")
    } == dict(checkout_sha=HEAD, snapshot_base=BASE, snapshot_tree=TREE)


def published_snapshot():
    scope = "tests/attention/test_rope.py"
    digest = hashlib.sha256(scope.encode()).hexdigest()[:16]
    return [
        {
            "context": "ci/test-snapshot",
            "state": "success",
            "description": f"{BASE} {'5' * 40} {TREE} {digest}",
        },
        {"context": "ci/test-scope-1", "state": "success", "description": scope},
    ]


@pytest.mark.parametrize(
    "change",
    [
        "missing-base",
        "missing-tree",
        "wrong-checkout",
        "wrong-base",
        "wrong-tree",
        "missing-status",
        "failed-status",
        "wrong-digest",
    ],
)
def test_focused_admission_rejects_unverified_source(admit, change):
    inputs = dict(INPUT_CHECKOUT=HEAD, INPUT_BASE=BASE, INPUT_TREE=TREE)
    statuses = published_snapshot()
    if change.startswith("missing-") and change != "missing-status":
        inputs["INPUT_" + change.removeprefix("missing-").upper()] = ""
    elif change.startswith("wrong-") and change != "wrong-digest":
        inputs["INPUT_" + change.removeprefix("wrong-").upper()] = "6" * 40
    elif change == "missing-status":
        statuses.pop(0)
    elif change == "failed-status":
        statuses[0]["state"] = "failure"
    else:
        statuses[1]["description"] = "tests/other.py"
    with pytest.raises(SystemExit):
        admit(
            event_update=lambda event: event["label"].update(name="run-ci-focused"),
            statuses=[statuses],
            **inputs,
        )


# Import the stdlib-only history helper without importing the GPU package.
_spec = importlib.util.spec_from_file_location(
    "pr_test_history",
    WORKFLOWS.parents[1] / "scripts/infra/pr_checks/pr_test_history.py",
)
history_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(history_module)
SOURCE = dict(head=HEAD, base=BASE, tree=TREE)


def run_record(run_id=10, result="queued", attempt=1):
    return dict(
        id=run_id,
        run_attempt=attempt,
        status="completed" if result in {"success", "failure"} else result,
        conclusion=result if result in {"success", "failure"} else None,
    )


class Scenario:
    """An API snapshot changes only when the waiter sleeps."""

    def __init__(self, snapshots):
        self.snapshots = iter(snapshots)
        self.current = next(self.snapshots)
        self.delays = []

    def base_sha(self, ref):
        return self.current.get("base", BASE)

    def runs(self, head):
        if "error" in self.current:
            raise history_module.ReadError()
        return self.current["runs"]

    def source(self, run):
        return self.current.get("source", SOURCE)

    def sleep(self, delay):
        self.delays.append(delay)
        self.current = next(self.snapshots)

    def wait(self):
        return history_module.wait_for_result(
            self, SOURCE, "main", 100, sleep=self.sleep
        )


@pytest.mark.parametrize("field", ["head", "base", "tree"])
def test_prior_pass_requires_exact_source(field):
    history = Scenario(
        [dict(runs=[run_record(result="success")], source={**SOURCE, field: "f" * 40})]
    )
    assert not history.wait()


def test_base_advance_rejects_even_matching_record():
    assert not Scenario(
        [dict(base="f" * 40, runs=[run_record(result="success")])]
    ).wait()


def test_legacy_pass_without_source_requires_fresh_ci():
    assert not Scenario([dict(runs=[run_record(result="success")], source=None)]).wait()


@pytest.mark.parametrize("first_result", ["failure", "queued"])
def test_another_matching_pass_unblocks_waiter(first_result):
    history = Scenario(
        [
            dict(runs=[run_record(), run_record(20)]),
            dict(runs=[run_record(result=first_result), run_record(20, "success")]),
        ]
    )
    assert history.wait()
    assert history.delays == [300]


@pytest.mark.parametrize("result,expected", [("queued", False), ("success", True)])
def test_newer_runs_can_satisfy_check_but_never_be_waited_on(result, expected):
    assert Scenario([dict(runs=[run_record(200, result)])]).wait() == expected


def test_rerun_attempt_is_observed_before_deciding():
    history = Scenario(
        [
            dict(runs=[run_record()]),
            dict(runs=[run_record(result="queued", attempt=2)]),
            dict(runs=[run_record(result="success", attempt=2)]),
        ]
    )
    assert history.wait()


def test_read_errors_do_not_become_failed_or_empty_history():
    history = Scenario(
        [
            dict(runs=[run_record()]),
            *[dict(error=True) for _ in range(12)],
            dict(runs=[run_record(result="success")]),
        ]
    )
    assert history.wait()
    assert min(history.delays) >= 300
    assert history.delays[-1] > history.delays[0]


@pytest.mark.parametrize("path", sorted(history_module.WORKFLOWS))
def test_history_pagination_and_attempt_scoped_artifacts(monkeypatch, path):
    calls = []
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("pr-test-source.json", json.dumps(SOURCE))

    def api(url, **kwargs):
        calls.append(url)
        if "/artifacts?" in url:
            return dict(
                artifacts=[
                    dict(
                        id=42, name="pr-test-source-2", expired=False, size_in_bytes=500
                    )
                ]
            )
        if url.endswith("/zip"):
            return archive.getvalue()
        row = {
            **run_record(result="success", attempt=2),
            "path": path,
            "head_sha": HEAD,
            "event": "pull_request",
        }
        if "&page=1" in url:
            return dict(
                workflow_runs=[{**row, "path": ".github/workflows/other.yml"}] * 100
            )
        return dict(workflow_runs=[row])

    monkeypatch.setattr(history_module, "gh_api", api)
    history = history_module.History(REPO)
    runs = history.runs(HEAD)
    assert len(runs) == 1
    assert history.source(runs[0]) == SOURCE
    assert history.source(runs[0]) == SOURCE
    assert len([url for url in calls if "/artifacts?" in url]) == 1


@pytest.mark.parametrize("latest", [1, 2])
def test_partial_rerun_uses_latest_authorized_source_without_caching_while_active(
    monkeypatch, latest
):
    def api(url, **kwargs):
        if "/artifacts?" in url:
            return dict(
                artifacts=[
                    dict(
                        id=attempt,
                        name=f"pr-test-source-{attempt}",
                        expired=False,
                        size_in_bytes=500,
                    )
                    for attempt in range(1, latest + 1)
                ]
            )
        archive = io.BytesIO()
        value = (
            SOURCE if latest == 1 or "/2/zip" in url else {**SOURCE, "tree": "f" * 40}
        )
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("pr-test-source.json", json.dumps(value))
        return archive.getvalue()

    monkeypatch.setattr(history_module, "gh_api", api)
    history = history_module.History(REPO)
    # Attempt 2 can still publish a record when all jobs are rerun.
    if latest == 1:
        assert history.source(run_record(attempt=2)) is None
    assert history.source(run_record(result="success", attempt=2)) == SOURCE


@pytest.mark.parametrize(
    "ref,expected",
    [("refs/heads/main", 0), ("refs/heads/feature", 1), ("refs/tags/main", 1)],
)
def test_manual_dispatch_rejected_before_setup(tmp_path, ref, expected):
    output = tmp_path / "output"
    result = bash(
        step_script("pr-test-dispatch.yml", "gate", "Check authorization"),
        tmp_path,
        EVENT_NAME="workflow_dispatch",
        EVENT_REF=ref,
        EVENT_ACTION="",
        LABEL_NAME="",
        GITHUB_OUTPUT=str(output),
    )
    assert result.returncode == expected
    assert output.read_text().strip() == f"authorized={str(expected == 0).lower()}"
    if expected:
        assert "Select main" in result.stdout


def test_required_source_reads_actual_merge_parents(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(tmp_path), *args], text=True
        ).strip()

    git("init", "-q")
    git(
        "-c",
        "user.name=CI",
        "-c",
        "user.email=ci@example.com",
        "commit",
        "--allow-empty",
        "-qm",
        "base",
    )
    base = git("rev-parse", "HEAD")
    (tmp_path / "source").write_text("PR change")
    git("add", "source")
    git(
        "-c", "user.name=CI", "-c", "user.email=ci@example.com", "commit", "-qm", "head"
    )
    head = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    merge = subprocess.check_output(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=CI",
            "-c",
            "user.email=ci@example.com",
            "commit-tree",
            tree,
            "-p",
            base,
            "-p",
            head,
        ],
        input="merge\n",
        text=True,
    ).strip()
    monkeypatch.chdir(tmp_path)
    assert history_module.required_source(merge, head) == dict(
        head=head, base=base, tree=tree
    )
    assert history_module.required_source(merge, base) is None
    assert history_module.required_source(head, head) is None


@pytest.mark.parametrize(
    "reply", [subprocess.CalledProcessError(1, ["gh", "api"]), b"not JSON"]
)
def test_failed_api_reads_are_unknown(monkeypatch, reply):
    def run(*args, **kwargs):
        if isinstance(reply, Exception):
            raise reply
        return subprocess.CompletedProcess(args, 0, stdout=reply)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(history_module.ReadError):
        history_module.gh_api("repos/example/repo/actions/runs")
