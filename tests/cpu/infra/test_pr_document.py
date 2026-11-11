import pytest

import json
import subprocess
import sys
from pathlib import Path


CHECKER = Path(__file__).parents[3] / "scripts" / "check_pr_document.py"
LEGACY_REDIRECT = (
    "This repository’s agent instructions live in [CLAUDE.md](./CLAUDE.md).\n\n"
    "Before making changes, read `CLAUDE.md` and follow its guidance for build, "
    "test, style, and contribution workflows.\n"
)
ARCH_DOC = "FlashInfer supports NVIDIA SM89.\n"
DOCUMENTED_ENV = "`FLASHINFER_DOCUMENTED` controls the example.\n"


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            "user.name=Documentation test",
            "-c",
            "user.email=docs@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        cwd=repo,
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


def commit_docs(repo: Path, agents: str | None, legacy: str | None) -> str:
    for name, content in (("AGENTS.md", agents), ("CLAUDE.md", legacy)):
        path = repo / name
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(content, encoding="utf-8")
    git(repo, "add", "--all")
    git(repo, "commit", "--quiet", "--allow-empty", "-m", "Documentation snapshot")
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "--quiet")
    package = tmp_path / "flashinfer"
    package.mkdir()
    (package / "example.py").write_text(
        'import os\nos.getenv("FLASHINFER_DOCUMENTED")\n'
        'os.getenv("FLASHINFER_EXISTING_DEBT")\n',
        encoding="utf-8",
    )
    return tmp_path


def check_docs(repo: Path, base: str, head: str) -> tuple[int, list[dict]]:
    report = repo / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--base",
            base,
            "--head",
            head,
            "--strict",
            "--report-json",
            str(report),
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert result.returncode in (0, 1), result.stdout + result.stderr
    return result.returncode, json.loads(report.read_text())["findings"]


def missing_env_finding() -> dict:
    return {
        "check": "env_vars_consistency",
        "path": "flashinfer/example.py",
        "line": 2,
        "message": "env var read in code but not documented: FLASHINFER_DOCUMENTED",
        "level": "warning",
    }


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "base_agents", [LEGACY_REDIRECT, None], ids=["redirect", "absent"]
)
@pytest.mark.parametrize("preserve_env", [False, True], ids=["omitted", "preserved"])
def test_legacy_migration_reports_only_new_documentation_debt(
    repo: Path, base_agents: str | None, preserve_env: bool
) -> None:
    base = commit_docs(repo, base_agents, ARCH_DOC + DOCUMENTED_ENV)
    head = commit_docs(repo, ARCH_DOC + (DOCUMENTED_ENV if preserve_env else ""), None)

    status, findings = check_docs(repo, base, head)

    assert findings == ([] if preserve_env else [missing_env_finding()])
    assert status == (0 if preserve_env else 1)


@pytest.mark.cpu_only
def test_substantive_agents_takes_precedence_over_stale_legacy(repo: Path) -> None:
    base = commit_docs(repo, ARCH_DOC + DOCUMENTED_ENV, ARCH_DOC)
    head = commit_docs(repo, ARCH_DOC, None)

    status, findings = check_docs(repo, base, head)

    assert findings == [missing_env_finding()]
    assert status == 1


@pytest.mark.cpu_only
def test_head_redirect_does_not_hide_missing_canonical_documentation(
    repo: Path,
) -> None:
    base = commit_docs(repo, ARCH_DOC + DOCUMENTED_ENV, None)
    head = commit_docs(repo, LEGACY_REDIRECT, ARCH_DOC + DOCUMENTED_ENV)

    status, findings = check_docs(repo, base, head)

    assert findings == [
        missing_env_finding(),
        {
            "check": "supported_arch_consistency",
            "path": "AGENTS.md",
            "line": 1,
            "message": "Could not find 'FlashInfer supports NVIDIA ...' arch line",
            "level": "warning",
        },
    ]
    assert status == 1
