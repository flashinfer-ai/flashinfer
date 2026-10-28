"""Exercise the GPU CI guard through real pytest session startup."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "lane,available,exit_code,outcome",
    [
        (
            "H100",
            False,
            pytest.ExitCode.USAGE_ERROR,
            "requires an available CUDA device",
        ),
        ("H100", True, pytest.ExitCode.OK, "1 passed"),
        (None, False, pytest.ExitCode.OK, "1 skipped"),
    ],
)
def test_gpu_ci_boundary(tmp_path, lane, available, exit_code, outcome):
    """A declared GPU lane fails without CUDA while an ordinary CPU run can skip."""
    (tmp_path / "torch.py").write_text(
        "from types import SimpleNamespace\n"
        f"cuda = SimpleNamespace(is_available=lambda: {available!r})\n"
    )
    target = tmp_path / "test_optional_cuda.py"
    target.write_text(
        "import pytest, torch\n"
        "def test_optional_cuda():\n"
        "    if not torch.cuda.is_available():\n"
        "        pytest.skip('intentional CPU-only run')\n"
    )
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join([str(tmp_path), str(ROOT)]),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_ADDOPTS="",
    )
    if lane is None:
        env.pop("CI_GPU_NAME", None)
    else:
        env["CI_GPU_NAME"] = lane
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "tests.test_helpers.gpu_ci",
            "-p",
            "no:cacheprovider",
            "--confcutdir",
            str(tmp_path),
            "-c",
            os.devnull,
            str(target),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout + result.stderr
    assert result.returncode == exit_code, output
    assert outcome in output, output
