"""SM120 production import checks for source and installed package layouts."""

import ast
from importlib import metadata
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.cpu_only

_PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "flashinfer"
if not _PACKAGE_ROOT.is_dir():
    # Nightly stages tests without the source tree beside them.
    _PACKAGE_ROOT = Path(
        metadata.distribution("flashinfer-python").locate_file("flashinfer")
    )


def test_production_modules_do_not_import_mega_runner() -> None:
    package = _PACKAGE_ROOT / "moe_ep"
    roots = (
        package / "backends" / "mega" / "kernel" / "sm120" / "mxfp4_mxfp8_bf16_cutedsl",
        package / "shims" / "sm120" / "split_cutedsl",
    )
    offenders: list[str] = []
    for root in roots:
        assert root.is_dir(), root
        for source in root.rglob("*.py"):
            tree = ast.parse(source.read_text())
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if any("mega_runner" in name for name in names):
                    offenders.append(str(source.relative_to(package)))
    assert not offenders, offenders


@pytest.mark.parametrize("layout", ["checkout", "installed"])
def test_production_import_check_in_isolated_layout(tmp_path, layout):
    isolated = tmp_path / "nightly"
    test_path = isolated / "tests/cpu/moe_ep" / Path(__file__).name
    test_path.parent.mkdir(parents=True)
    shutil.copyfile(__file__, test_path)

    installed = tmp_path / "site-packages"
    package = (isolated if layout == "checkout" else installed) / "flashinfer"
    subdirs = (
        "moe_ep/backends/mega/kernel/sm120/mxfp4_mxfp8_bf16_cutedsl",
        "moe_ep/shims/sm120/split_cutedsl",
    )
    for subdir in subdirs:
        shutil.copytree(_PACKAGE_ROOT / subdir, package / subdir)

    if layout == "installed":
        dist_info = installed / "flashinfer_python-0.0.0.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text("Name: flashinfer-python\nVersion: 0.0.0\n")

    script = r"""
from pathlib import Path
import re
import runpy
import sys

import pytest

package, test_path = map(Path, sys.argv[1:])
sys.path.insert(0, str(package.parent))
checks = runpy.run_path(str(test_path))
assert checks["_PACKAGE_ROOT"] == package
check = checks["test_production_modules_do_not_import_mega_runner"]
check()
assert "flashinfer" not in sys.modules

# Both trees must be scanned, and missing directories must still fail.
roots = (
    package / "moe_ep/backends/mega/kernel/sm120/mxfp4_mxfp8_bf16_cutedsl",
    package / "moe_ep/shims/sm120/split_cutedsl",
)
for root, statement in zip(roots, (
    "import mega_runner\n",
    "from package.mega_runner import main\n",
)):
    offender = root / "forbidden_import.py"
    offender.write_text(statement)
    with pytest.raises(AssertionError, match="forbidden_import"):
        check()
    offender.unlink()
    missing = root.with_name(root.name + ".missing")
    root.rename(missing)
    with pytest.raises(AssertionError, match=re.escape(str(root))):
        check()
    missing.rename(root)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(package), str(test_path)],
        cwd=isolated,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
