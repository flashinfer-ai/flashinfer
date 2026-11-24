"""Provider integrity checks must also work outside the source checkout."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.cpu_only
def test_provider_integrity_checks_in_isolated_layout(tmp_path):
    root = Path(__file__).resolve().parents[3]
    isolated = tmp_path / "nightly"
    test_path = isolated / "tests/moe_ep/test_provider_package.py"
    test_path.parent.mkdir(parents=True)
    shutil.copyfile(root / "tests/moe_ep/test_provider_package.py", test_path)
    benchmark = Path("benchmarks/bench_cake_megamoe_topk_reduce.py")
    (isolated / benchmark).parent.mkdir(parents=True)
    shutil.copyfile(root / benchmark, isolated / benchmark)

    # Use real vendored bytes and distribution metadata, without importing the
    # provider or requiring CUDA. The test directory has no flashinfer/ tree.
    installed = tmp_path / "site-packages"
    provider = Path("flashinfer/moe_ep/kernels/cutedsl_megamoe")
    shutil.copytree(root / provider, installed / provider)
    dist_info = installed / "flashinfer_python-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Name: flashinfer-python\nVersion: 0.0.0\n")

    script = """
from pathlib import Path
import runpy
import sys

installed, test_path = map(Path, sys.argv[1:])
sys.path.insert(0, str(installed))
checks = runpy.run_path(str(test_path))
checks["test_export_manifest_matches_sources"]()
checks["test_reducer_benchmark_pins_match_vendored_sources"]()
assert checks["_PACKAGE_ROOT"] == installed / "flashinfer"
assert "flashinfer" not in sys.modules

# Missing package data must still fail rather than being silently skipped.
manifest = installed / "flashinfer/moe_ep/kernels/cutedsl_megamoe/export_manifest.json"
manifest.unlink()
try:
    checks["test_export_manifest_matches_sources"]()
except FileNotFoundError:
    pass
else:
    raise AssertionError("missing installed manifest was not detected")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(installed), str(test_path)],
        cwd=isolated,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
