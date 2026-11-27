"""Tests for the nightly PyPI package size alert.

`scripts/infra/check_pypi_package_size.py` runs in
`.github/workflows/pypi-package-size.yml` on the sdist and wheel release.yml
uploads. It must post to Slack once the largest file reaches the warning
threshold, stay quiet below it, and fail when an alert cannot be delivered.
"""

import importlib.util
import io
import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "infra" / "check_pypi_package_size.py"
MIB = 1024 * 1024
WHEEL = "flashinfer_python-0.7.5-py3-none-any.whl"
SDIST = "flashinfer_python-0.7.5.tar.gz"


@pytest.fixture(scope="module")
def size_check():
    spec = importlib.util.spec_from_file_location("check_pypi_package_size", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sparse_file(path: pathlib.Path, size: int) -> None:
    with open(path, "wb") as fh:
        fh.truncate(size)


@pytest.fixture
def dist(tmp_path):
    def make(wheel_mib: float, sdist_mib: float = 50) -> pathlib.Path:
        _sparse_file(tmp_path / WHEEL, int(wheel_mib * MIB))
        _sparse_file(tmp_path / SDIST, int(sdist_mib * MIB))
        # Not uploaded to PyPI by release.yml; must not be measured.
        _sparse_file(tmp_path / "flashinfer_cubin-0.7.5-py3-none-any.whl", 900 * MIB)
        return tmp_path

    return make


class FakeSlack:
    def __init__(self, response=None):
        self.requests = []
        self.response = response or {"ok": True}

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        return io.BytesIO(json.dumps(self.response).encode())


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "GITHUB_STEP_SUMMARY",
        "GITHUB_SHA",
        "GITHUB_RUN_ID",
        "SLACK_BOT_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)


def _run(size_check, dist_dir, slack, *extra):
    argv = ["--dist-dir", str(dist_dir), "--slack-channel", "C123", *extra]
    return size_check.main(argv, urlopen=slack)


def test_below_threshold_does_not_post(size_check, dist, monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    slack = FakeSlack()
    assert _run(size_check, dist(249.9), slack) == 0
    assert slack.requests == []


@pytest.mark.parametrize("wheel_mib", [250, 299.9])
def test_at_or_above_threshold_posts_warning(size_check, dist, monkeypatch, wheel_mib):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    slack = FakeSlack()
    assert _run(size_check, dist(wheel_mib), slack) == 0
    (request,) = slack.requests
    assert request.full_url == size_check.SLACK_POST_URL
    assert request.get_header("Authorization") == "Bearer xoxb-test"
    payload = json.loads(request.data)
    assert payload["channel"] == "C123"
    assert "250 MiB warning threshold" in payload["text"]
    assert WHEEL in payload["text"]
    assert "flashinfer_cubin" not in payload["text"]


def test_over_limit_message(size_check, dist, monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    slack = FakeSlack()
    assert _run(size_check, dist(50, sdist_mib=301), slack) == 0
    text = json.loads(slack.requests[0].data)["text"]
    assert "over the 300 MiB PyPI file limit" in text
    assert f"`{SDIST}` is 301.0 MiB" in text


def test_dry_run_does_not_post(size_check, dist, monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    slack = FakeSlack()
    assert _run(size_check, dist(260), slack, "--dry-run") == 0
    assert slack.requests == []


def test_missing_token_fails_when_alert_is_due(size_check, dist):
    assert _run(size_check, dist(260), FakeSlack()) == 1


def test_slack_error_fails(size_check, dist, monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    slack = FakeSlack({"ok": False, "error": "not_in_channel"})
    with pytest.raises(RuntimeError, match="not_in_channel"):
        _run(size_check, dist(260), slack)


def test_no_package_files_fails(size_check, tmp_path):
    assert _run(size_check, tmp_path, FakeSlack()) == 1


def test_report_goes_to_step_summary(size_check, dist, monkeypatch, tmp_path):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("GITHUB_SHA", "0123456789abcdef")
    assert _run(size_check, dist(120), FakeSlack()) == 0
    text = summary.read_text()
    assert "within budget" in text
    assert "main @ `0123456789ab`" in text
