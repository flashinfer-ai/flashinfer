"""Execute the trusted pre-checkout status bootstrap with a fake GitHub API."""

import copy
import io
import json
import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch


class WorkflowBootstrapTest(unittest.TestCase):
    def test_stale_head_or_base_never_overwrites_status(self):
        workflow = (
            Path(__file__).resolve().parents[3]
            / ".github/workflows/pr-api-compatibility.yml"
        ).read_text()
        code = workflow.split("run: &api-gate-status |\n", 1)[1].split(
            "      # Never checkout", 1
        )[0]
        code = (
            textwrap.dedent(code)
            .split("python3 - <<'PYTHON'\n", 1)[1]
            .rsplit("PYTHON", 1)[0]
        )
        for state in ("pending", "error"):
            for changed in (None, "head", "base"):
                with (
                    self.subTest(state=state, changed=changed),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    snapshot = {
                        "number": 3,
                        "state": "open",
                        "head": {"sha": "a" * 40},
                        "base": {"sha": "b" * 40},
                    }
                    latest = copy.deepcopy(snapshot)
                    if changed:
                        latest[changed]["sha"] = "c" * 40
                    event = Path(directory) / "event.json"
                    event.write_text(json.dumps({"pull_request": snapshot}))
                    writes = []

                    def open_request(request, **kwargs):
                        if request.get_method() == "GET":
                            return io.BytesIO(json.dumps(latest).encode())
                        writes.append(json.loads(request.data))
                        return io.BytesIO(b"{}")

                    env = {
                        "GITHUB_EVENT_PATH": str(event),
                        "GITHUB_API_URL": "https://api.github.com",
                        "GITHUB_REPOSITORY": "org/repo",
                        "GITHUB_SERVER_URL": "https://github.com",
                        "GITHUB_RUN_ID": "2",
                        "GH_TOKEN": "test-token",
                        "API_GATE_STATE": state,
                        "API_GATE_DESCRIPTION": "test",
                    }
                    with (
                        patch.dict(os.environ, env),
                        patch("urllib.request.urlopen", side_effect=open_request),
                    ):
                        exec(compile(code, "workflow-bootstrap", "exec"), {})
                    self.assertEqual(len(writes), 0 if changed else 2)
