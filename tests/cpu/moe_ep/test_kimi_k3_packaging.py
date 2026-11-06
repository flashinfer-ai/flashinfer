"""Import isolation and packaged Kimi source checks."""

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

pytestmark = pytest.mark.cpu_only
ROOT = Path(__file__).resolve().parents[3]
KIMI = ROOT / "flashinfer/moe_ep/kernel_src/sm100/kimi-k3-megamoe"
NAMESPACE = "flashinfer.moe_ep.kernel_src.sm100._kimi_k3_megamoe"

IMPORT_CHECK = """
import importlib
from pathlib import Path
import sys

root, order = sys.argv[1:]
sys.path.insert(0, root)
import flashinfer.moe_ep
other = 'flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe'
kimi = 'flashinfer.moe_ep.kernel_src.sm100._kimi_k3_megamoe'
if order == 'after':
    importlib.import_module(other)
paths = list(sys.path)
modules = dict(sys.modules)
provider = importlib.import_module(kimi)
assert sys.path == paths
assert all(sys.modules.get(k) is v for k, v in modules.items())
new_modules = sys.modules.keys() - modules.keys()
assert not any(
    name.startswith(('tensorrt_llm', 'kernel_src.', 'common.', other, 'cutlass'))
    or name in ('kernel_src', 'common') for name in new_modules
), new_modules
assert Path(provider.__file__).parent.name == 'kimi-k3-megamoe'
assert Path(provider.__file__).is_relative_to(Path(root))
assert provider.__path__ == [str(Path(provider.__file__).parent)]
if order == 'before':
    importlib.import_module(other)
    assert importlib.import_module(kimi) is provider
assert callable(provider.preprocess_weights)
assert callable(provider.create_workspace)
import cutlass.cute as cute
missing = object()
major_mode = getattr(cute.nvgpu, 'OperandMajorMode', missing)
paths = list(sys.path)
device_modules = set(sys.modules)
provider.get_combine_format('bf16')
assert getattr(cute.nvgpu, 'OperandMajorMode', missing) is major_mode
assert sys.path == paths
assert not any(name == 'tensorrt_llm' or name.startswith('tensorrt_llm.')
               for name in sys.modules.keys() - device_modules)
"""


@pytest.mark.parametrize("order", ["before", "after"])
def test_import_order_does_not_change_other_provider(order, tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", "-c", IMPORT_CHECK, str(ROOT), order],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_kimi_adapters_do_not_import_other_providers():
    paths = list((KIMI / "shim").rglob("*.py"))
    paths += list(
        (ROOT / "flashinfer/moe_ep/backends/mega/kernel/sm100/kimi_k3").glob("*.py")
    )
    forbidden = (
        "next_cutedsl_megamoe",
        "standard_cutedsl",
        "automatic",
        "tensorrt_llm",
    )
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(
                part in name.split(".") for name in names for part in forbidden
            ), path


def test_export_manifest_matches_vendored_source():
    manifest = json.loads((KIMI / "export_manifest.json").read_text())
    assert manifest["kimi_commit"] == "6e8cfe94e9d6c92b184fb46700c72e9f66ccb884"
    assert set(manifest["files"]) == {
        path.relative_to(KIMI / "src").as_posix()
        for path in (KIMI / "src").rglob("*.py")
    }
    for name, digest in manifest["files"].items():
        path = KIMI / "src" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
    exporter = ROOT / "scripts/export_kimi_k3_megamoe.py"
    assert (
        hashlib.sha256(exporter.read_bytes()).hexdigest() == manifest["exporter_sha256"]
    )


def test_exporter_rejects_a_changed_upstream_manifest(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "kimi_exporter_test", ROOT / "scripts/export_kimi_k3_megamoe.py"
    )
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    overlay = tmp_path / "overlay"
    (overlay / "install").mkdir(parents=True)
    (overlay / "install/base_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="pinned revision"):
        exporter.export(
            tmp_path / "missing-base", overlay, tmp_path / "out", check=True
        )


@pytest.mark.parametrize("order", ["before", "after"])
def test_installed_wheel_imports_without_source_checkout(order, tmp_path):
    wheel_name = os.environ.get("FLASHINFER_KIMI_TEST_WHEEL")
    if not wheel_name:
        pytest.skip("set FLASHINFER_KIMI_TEST_WHEEL to a built wheel")
    wheel = Path(wheel_name).resolve()
    destination = tmp_path / "installed"
    prefix = "flashinfer/moe_ep/kernel_src/sm100/"
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        for source in KIMI.rglob("*"):
            if source.is_file() and "__pycache__" not in source.parts:
                assert (
                    prefix + "kimi-k3-megamoe/" + str(source.relative_to(KIMI)) in names
                )
        assert prefix + "_kimi_k3_megamoe.py" in names
        archive.extractall(destination)
    result = subprocess.run(
        [sys.executable, "-I", "-c", IMPORT_CHECK, str(destination), order],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
