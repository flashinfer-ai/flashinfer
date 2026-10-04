"""The qualification gate must reject both collected and collection-time skips."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "outcome", ["passed", "runtime_skip", "collection_skip", "empty"]
)
def test_qualification_rejects_incomplete_pytest_runs(tmp_path, outcome):
    if outcome != "empty":
        (tmp_path / "test_pass.py").write_text("def test_pass():\n    assert True\n")
    if outcome == "runtime_skip":
        (tmp_path / "test_skip.py").write_text(
            "import pytest\ndef test_skip():\n    pytest.skip('missing device')\n"
        )
    elif outcome == "collection_skip":
        (tmp_path / "test_skip.py").write_text(
            "import pytest\npytest.skip('missing dependency', allow_module_level=True)\n"
        )
    runner = Path(__file__).with_name("qualify_sm107.py")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys, pytest; "
            "gate = runpy.run_path(sys.argv[1])['_NoSkips'](); "
            "sys.exit(int(pytest.main([sys.argv[2], '-q', '-p', 'no:cacheprovider'], plugins=[gate])))",
            str(runner),
            str(tmp_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == (0 if outcome == "passed" else 1), (
        result.stdout + result.stderr
    )
    if outcome != "passed":
        assert "SM107 qualification FAILED" in result.stdout
