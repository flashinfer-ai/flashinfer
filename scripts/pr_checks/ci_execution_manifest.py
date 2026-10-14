"""Record watcher CI execution facts; never infer test coverage or publish receipts.

The workflow stages this stdlib-only writer from github.workflow_sha outside the
Docker workspace. Test code gets no provider write token. Existing JUnit/sharding
outputs remain assertions produced by tests; the consumer must verify them and
provider job state. A started lane entrypoint is not proof that pytest ran.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

MAX_JSON = 2 * 1024 * 1024
MAX_FILE = 8 * 1024 * 1024
MAX_TOTAL = 32 * 1024 * 1024
MAX_FILES = 256
MAX_ENTRIES = 2048
SHA = re.compile(r"[0-9a-f]{40}")
IDENTITY = (
    "repository",
    "prNumber",
    "requestedHeadSha",
    "baseSha",
    "testedCheckoutSha",
    "checkoutParents",
    "provider",
    "runId",
    "runAttempt",
    "workflowRevision",
    "requestKey",
    "scopeDigest",
    "scope",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def value_digest(value):
    return digest(canonical(value).encode())


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def json_value(data):
    return json.loads(data, object_pairs_hook=unique_object)


def read_regular(path, limit=MAX_JSON, root=None):
    if root is None:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    else:
        relative = Path(path).relative_to(root)
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in relative.parts[:-1]:
                following = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
                )
                os.close(directory)
                directory = following
            fd = os.open(
                relative.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory,
            )
        finally:
            os.close(directory)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > limit
        ):
            raise ValueError("Artifact is not a bounded regular file")
        data = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
        if len(data) > limit or (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("Artifact changed or exceeded bound")
        return data


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (canonical(value) + "\n").encode()
    if len(data) > MAX_JSON:
        raise ValueError("Manifest exceeds bound")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def positive(value):
    if (
        not isinstance(value, (str, int))
        or isinstance(value, bool)
        or not re.fullmatch(r"[1-9][0-9]{0,18}", str(value))
    ):
        raise ValueError("Positive provider identifier required")
    return int(value)


def scope(value, require_canonical=True):
    if not isinstance(value, str) or len(value) > 8192:
        raise ValueError("Bounded scope string required")
    paths = value.split()
    if len(paths) > 64 or (require_canonical and paths != sorted(set(paths))):
        raise ValueError("Canonical scope required")
    for path in paths:
        if (
            not re.fullmatch(r"tests/[A-Za-z0-9_./-]+(?:::[A-Za-z0-9_.-]+)*", path)
            or ".." in path.split("::", 1)[0].split("/")
            or "//" in path
        ):
            raise ValueError("Invalid test scope")
    return sorted(set(paths))


def checkout(workspace):
    """Read commit headers, including parents missing from a shallow checkout."""
    env = {**os.environ, "GIT_NO_REPLACE_OBJECTS": "1", "GIT_CONFIG_NOSYSTEM": "1"}
    sha = (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=workspace, env=env, timeout=10
        )
        .decode()
        .strip()
    )
    if not SHA.fullmatch(sha):
        raise ValueError("Invalid checkout SHA")
    # A commit message may be arbitrarily large: stream only bounded headers.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            ["git", "cat-file", "commit", sha],
            cwd=workspace,
            env=env,
            stdout=subprocess.PIPE,
            stderr=errors,
        )
        parents = []
        try:
            for _ in range(128):
                line = process.stdout.readline(4097)
                if len(line) > 4096:
                    raise ValueError("Commit header exceeds bound")
                if line == b"\n":
                    break
                if not line:
                    raise ValueError("Incomplete commit headers")
                if line.startswith(b"parent "):
                    parent = line[7:].decode().strip()
                    if not SHA.fullmatch(parent) or len(parents) >= 2:
                        raise ValueError("Unexpected merge ancestry")
                    parents.append(parent)
            else:
                raise ValueError("Commit header count exceeds bound")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    return sha, parents


def request_identity(environment, event, workspace):
    inputs = event.get("inputs", {})
    repo = environment["GITHUB_REPOSITORY"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*/[a-z0-9][a-z0-9_.-]*", repo):
        raise ValueError("Canonical repository required")
    number = positive(inputs["watcher_pr_number"])
    comment = positive(inputs["watcher_trigger_comment_id"])
    if (
        inputs["watcher_trigger_comment_url"]
        != f"https://github.com/{repo}/pull/{number}#issuecomment-{comment}"
    ):
        raise ValueError("Original comment binding mismatch")
    shas = [
        inputs[k]
        for k in ("watcher_head_sha", "watcher_base_sha", "watcher_checkout_sha")
    ]
    revision = environment["GITHUB_WORKFLOW_SHA"]
    if any(
        not isinstance(value, str) or not SHA.fullmatch(value)
        for value in [*shas, revision]
    ):
        raise ValueError("Immutable request and workflow commits required")
    paths = scope(inputs["watcher_scope"])
    scope_digest = value_digest(paths)
    key = f"fi-ci/{number}/{comment}/{shas[0]}/{scope_digest}"
    if (
        inputs["watcher_scope_digest"] != scope_digest
        or inputs["watcher_request_key"] != key
    ):
        raise ValueError("Request key/scope binding mismatch")
    actual, parents = checkout(workspace)
    if actual != shas[2] or len(parents) != 2 or set(parents) != set(shas[:2]):
        raise ValueError(
            "Actual checkout does not contain the requested head/base parents"
        )
    return {
        "repository": repo,
        "prNumber": number,
        "requestedHeadSha": shas[0],
        "baseSha": shas[1],
        "testedCheckoutSha": actual,
        "checkoutParents": parents,
        "provider": "github",
        "runId": positive(environment["GITHUB_RUN_ID"]),
        "runAttempt": positive(environment["GITHUB_RUN_ATTEMPT"]),
        "workflowRevision": revision,
        "requestKey": key,
        "scopeDigest": scope_digest,
        "scope": paths,
    }


def validate_matrix(matrix):
    if (
        not isinstance(matrix, dict)
        or set(matrix) != {"include"}
        or not isinstance(matrix["include"], list)
        or len(matrix["include"]) > 100
    ):
        raise ValueError("Bounded include matrix required")
    names = []
    for lane in matrix["include"]:
        name = lane.get("name") if isinstance(lane, dict) else None
        if (
            not isinstance(name, str)
            or not 1 <= len(name) <= 200
            or any(ord(c) < 32 for c in name)
        ):
            raise ValueError("Stable lane name required")
        if lane.get("kind") not in ("aot", "gpu", "h100"):
            raise ValueError("Unsupported existing lane kind")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("Lane job names must be unique")
    canonical(matrix)
    return matrix


def create_plan(environment, event, workspace, matrix, skip_build):
    return {
        "schemaVersion": 1,
        "kind": "watcher-ci-plan",
        **request_identity(environment, event, workspace),
        "matrix": validate_matrix(matrix),
        "matrixDigest": value_digest(matrix),
        "skipBuild": skip_build,
    }


def checked_plan(path, environment, event, workspace):
    plan = json_value(read_regular(path))
    identity = request_identity(environment, event, workspace)
    if (
        plan.get("schemaVersion") != 1
        or plan.get("kind") != "watcher-ci-plan"
        or any(
            plan.get(key) != identity[key] for key in IDENTITY if key != "runAttempt"
        )
        or type(plan.get("runAttempt")) is not int
        or not 1 <= plan["runAttempt"] <= identity["runAttempt"]
        or plan.get("matrixDigest") != value_digest(validate_matrix(plan.get("matrix")))
        or type(plan.get("skipBuild")) is not bool
    ):
        raise ValueError("Plan does not match actual execution/run/attempt")
    return plan


def initialize(plan, lane, workspace, output, run_attempt=None, hardware_context=None):
    if plan["skipBuild"] or lane not in plan["matrix"]["include"]:
        raise ValueError("Lane was not scheduled in this plan")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    junit = Path(workspace) / "junit"
    if junit.exists() or junit.is_symlink():
        raise ValueError(
            "Pre-existing JUnit directory cannot establish fresh execution"
        )
    junit.mkdir()
    current_attempt = (
        plan["runAttempt"] if run_attempt is None else positive(run_attempt)
    )
    if current_attempt < plan["runAttempt"]:
        raise ValueError("A future setup plan cannot bind this execution")
    result = {
        "schemaVersion": 1,
        "kind": "watcher-ci-result",
        **{k: plan[k] for k in IDENTITY},
        "runAttempt": current_attempt,
        "planRunAttempt": plan["runAttempt"],
        "matrixDigest": plan["matrixDigest"],
        "laneKey": value_digest(lane),
        "lane": lane,
        "jobName": lane["name"],
        "evidenceKind": "build-import" if lane["kind"] == "aot" else "test",
        "commandKind": "lane-entrypoint",
        "commandStarted": False,
        "exitCode": None,
        "testTargets": scope(lane.get("test_path", ""), require_canonical=False),
        "evidence": [],
        "hardware": {"status": "not_recorded"},
        "ranks": {"status": "not_recorded"},
        "artifactErrors": [],
    }
    if hardware_context is not None:
        probe = Path(hardware_context).parent / "hardware.py"
        context = {
            key: result[key]
            for key in (
                *IDENTITY,
                "matrixDigest",
                "laneKey",
                "jobName",
                "planRunAttempt",
            )
        }
        context.update(
            captureNonce=os.urandom(24).hex(), probeSha256=digest(read_regular(probe))
        )
        result["hardwareContext"] = context
        write_json(hardware_context, context)
    write_json(output / "result.json", result)
    return result


class HardwareFrame:
    """Accept only the first pre-test frame; later PR stdout cannot update it."""

    MAGIC = b"WATCHER_CI_HARDWARE_V1:"
    MAX_LINE = 131072

    def __init__(self, context):
        self.context = context
        self.pending = b""
        self.discard = False
        self.seen = False
        self.document = None

    def feed(self, chunk):
        if self.context is None or self.seen:
            return
        self.pending += chunk
        while b"\n" in self.pending:
            line, self.pending = self.pending.split(b"\n", 1)
            if self.discard:
                self.discard = False
                continue
            if line.startswith(self.MAGIC):
                self.seen = True
                try:
                    if len(line) > self.MAX_LINE:
                        raise ValueError("Hardware frame exceeds bound")
                    data = base64.b64decode(line[len(self.MAGIC) :], validate=True)
                    if len(data) > 65536:
                        raise ValueError("Hardware payload exceeds bound")
                    document = json_value(data)
                    if (
                        document.get("schemaVersion") != 1
                        or document.get("kind") != "watcher-ci-hardware-observation"
                        or document.get("context") != self.context
                    ):
                        raise ValueError("Hardware frame identity mismatch")
                    validate_hardware(document["hardware"])
                    self.document = document
                except (ValueError, TypeError, KeyError, AttributeError):
                    pass
                self.pending = b""
                return
        if len(self.pending) > self.MAX_LINE:
            if self.pending.startswith(self.MAGIC):
                self.seen = True
            self.pending = b""
            self.discard = True


def validate_hardware(value):
    if (
        not isinstance(value, dict)
        or value.get("status") not in ("observed", "unavailable")
        or value.get("gpuUseProven") is not False
        or value.get("rankParticipation") != "not_recorded"
    ):
        raise ValueError(
            "Hardware observation must not imply GPU use or rank participation"
        )
    if value["status"] == "unavailable":
        if not isinstance(value.get("reason"), str) or len(value["reason"]) > 200:
            raise ValueError("Bounded missing-inventory reason required")
        return
    if value.get("source") == "container-platform":
        if value.get("devices") != [] or value.get("gpuCount") is not None:
            raise ValueError("CPU inventory cannot assert GPU observations")
        return
    devices = value.get("devices")
    if (
        value.get("source") != "container-cuda-driver"
        or value.get("enumeration") != "cuda-visible"
        or not isinstance(devices, list)
        or not 1 <= len(devices) <= 16
        or type(value.get("gpuCount")) is not int
        or value["gpuCount"] != len(devices)
    ):
        raise ValueError("Actual CUDA-visible device inventory required")
    for ordinal, device in enumerate(devices):
        capability = (
            device.get("computeCapability") if isinstance(device, dict) else None
        )
        if (
            not isinstance(capability, str)
            or not re.fullmatch(r"[1-9][0-9]?\.[0-9]", capability)
            or type(device.get("ordinal")) is not int
            or device["ordinal"] != ordinal
            or device.get("hardwareTarget") != "sm" + capability.replace(".", "")
            or not isinstance(device.get("name"), str)
            or not 1 <= len(device["name"]) <= 200
            or any(ord(c) < 32 for c in device["name"])
        ):
            raise ValueError("Malformed observed GPU property")


def run_command(output, workspace, argv):
    if not argv:
        raise ValueError("Actual lane command required")
    path = Path(output) / "result.json"
    result = json_value(read_regular(path))
    if result.get("commandStarted") or result.get("exitCode") is not None:
        raise ValueError("Lane command cannot be replayed")
    actual, parents = checkout(workspace)
    if actual != result["testedCheckoutSha"] or parents != result["checkoutParents"]:
        raise ValueError("Checkout changed before lane execution")
    process = None
    hardware = HardwareFrame(result.get("hardwareContext"))
    try:
        process = subprocess.Popen(
            argv, cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        result.update(
            commandStarted=True, startedAt=datetime.now(timezone.utc).isoformat()
        )
        write_json(path, result)
        while True:
            chunk = process.stdout.read(65536)
            if not chunk:
                break
            hardware.feed(chunk)
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        process.stdout.close()
        code = process.wait()
        if code < 0:
            code = 128 - code
    except OSError:
        code = 127
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    if result.get("hardwareContext"):
        if hardware.document is not None:
            hardware_path = Path(output) / "evidence/hardware.json"
            write_json(hardware_path, hardware.document)
            data = read_regular(hardware_path)
            result["hardware"] = hardware.document["hardware"]
            result["evidence"].append(
                {
                    "kind": "hardware-observation",
                    "path": "evidence/hardware.json",
                    "sha256": digest(data),
                    "size": len(data),
                }
            )
        else:
            result["hardware"] = {
                "status": "unavailable",
                "source": "container-hardware-probe",
                "reason": "Actual container hardware frame missing or failed identity validation",
                "gpuUseProven": False,
                "rankParticipation": "not_recorded",
            }
    result.update(exitCode=code, finishedAt=datetime.now(timezone.utc).isoformat())
    write_json(path, result)
    return code


def artifact_kind(path):
    if path.name == "manifest.json":
        return "execution-plan"
    for suffix, kind in (
        (".xml", "junit"),
        (".results.json", "execution-results"),
        (".meta.json", "execution-metadata"),
        (".telemetry.json", "telemetry"),
    ):
        if path.name.endswith(suffix):
            return kind
    return None


def finalize(output, workspace):
    output = Path(output)
    result = json_value(read_regular(output / "result.json"))
    try:
        actual, parents = checkout(workspace)
        if (
            actual != result["testedCheckoutSha"]
            or parents != result["checkoutParents"]
        ):
            result["artifactErrors"].append("Checkout changed after lane execution")
        root = Path(workspace) / "junit"
        if root.is_symlink() or not root.is_dir():
            raise ValueError("JUnit output directory unavailable")
        total = entries = 0
        evidence = list(result.get("evidence", []))
        for directory, directories, files in os.walk(root, followlinks=False):
            if len(Path(directory).relative_to(root).parts) > 16:
                raise ValueError("Artifact directory depth exceeds bound")
            entries += len(directories) + len(files)
            if entries > MAX_ENTRIES:
                raise ValueError("Artifact directory inventory exceeds bound")
            for name in list(directories):
                if (Path(directory) / name).is_symlink():
                    raise ValueError("Linked artifact directory rejected")
            for name in sorted(files):
                source = Path(directory) / name
                kind = artifact_kind(source)
                if not kind:
                    continue
                if len(evidence) >= MAX_FILES:
                    raise ValueError("Artifact file count exceeds bound")
                data = read_regular(source, MAX_FILE, root=root)
                total += len(data)
                if total > MAX_TOTAL:
                    raise ValueError("Artifact total size exceeds bound")
                relative = Path("evidence") / source.relative_to(root)
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(data)
                evidence.append(
                    {
                        "kind": kind,
                        "path": relative.as_posix(),
                        "sha256": digest(data),
                        "size": len(data),
                    }
                )
        result["evidence"] = evidence
    except (OSError, ValueError) as error:
        # No private paths, raw provider output, or test exception bodies.
        result["artifactErrors"].append(
            type(error).__name__ + ": execution artifact capture incomplete"
        )
    write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", "initialize", "run", "finalize"))
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--lane")
    parser.add_argument("--hardware-context", type=Path)
    parser.add_argument("--skip-build", choices=("true", "false"), default="false")
    arguments = sys.argv[1:]
    separator = arguments.index("--") if "--" in arguments else len(arguments)
    command = arguments[separator + 1 :]
    args = parser.parse_args(arguments[:separator])
    if args.mode == "run":
        return run_command(args.output, args.workspace, command)
    if command:
        parser.error("Only run accepts an external command")
    if args.mode == "finalize":
        finalize(args.output, args.workspace)
        return 0
    event = json_value(read_regular(os.environ["GITHUB_EVENT_PATH"]))
    if args.mode == "plan":
        plan = create_plan(
            os.environ,
            event,
            args.workspace,
            json_value(read_regular(args.matrix)),
            args.skip_build == "true",
        )
        write_json(args.output, plan)
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write(
                f"artifact_name=watcher-ci-plan-{plan['runId']}-{plan['runAttempt']}\n"
            )
    else:
        plan = checked_plan(args.plan, os.environ, event, args.workspace)
        result = initialize(
            plan,
            json_value(args.lane),
            args.workspace,
            args.output,
            positive(os.environ["GITHUB_RUN_ATTEMPT"]),
            args.hardware_context,
        )
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write("lane_key=" + result["laneKey"] + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
