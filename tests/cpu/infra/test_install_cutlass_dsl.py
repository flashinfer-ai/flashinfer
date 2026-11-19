import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.cpu_only
@pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None, reason="requires POSIX bash"
)
def test_install_cutlass_dsl_from_outside_checkout(tmp_path):
    modules = tmp_path / "site-packages"
    modules.mkdir()
    # Reproduce the installed helper without its source-only ci/ directory.
    shutil.copyfile(REPO_ROOT / "build_utils.py", modules / "build_utils.py")
    (modules / "torch.py").write_text(
        "from types import SimpleNamespace\nversion = SimpleNamespace(cuda='13.0')\n"
    )
    (modules / "cutlass.py").write_text("__version__ = 'test'\n")
    # Exercise the real shell script and resolver without installing packages.
    (modules / "pip.py").write_text(
        "import json, sys\nprint(json.dumps(sys.argv[1:]))\n"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to(sys.executable)
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
        "PYTHONPATH": str(modules),
        "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "infra" / "install_cutlass_dsl.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    policy = json.loads((REPO_ROOT / "ci" / "cuda-versions.json").read_text())[
        "dependency_policy"
    ]["nvidia-cutlass-dsl"]
    extras = policy.get("cuda_major_extras", {}).get("13", [])
    package = "nvidia-cutlass-dsl" + (f"[{','.join(extras)}]" if extras else "")
    assert json.loads(result.stdout.splitlines()[0]) == [
        "install",
        package + policy["cuda_extra_specifier"],
    ]
    assert "CuTe DSL import verified: test" in result.stdout
