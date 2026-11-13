from __future__ import annotations


from pathlib import Path
import re
import shutil
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[3]
PCIE_SOURCE = ROOT / "flashinfer/experimental/b12x" / "comm" / "pcie"
COMPAT_INIT = ROOT / "flashinfer/experimental/b12x/_compat/b12x/__init__.py"


def test_pcie_collectives_are_python_only() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package_data = config["tool"]["setuptools"].get("package-data", {})

    assert "b12x.comm.pcie" not in package_data
    assert not list(PCIE_SOURCE.glob("*.cu"))
    assert not list(PCIE_SOURCE.glob("*.cuh"))
    assert not list(PCIE_SOURCE.glob("*.cpp"))
    for source in PCIE_SOURCE.glob("*.py"):
        text = source.read_text()
        assert "torch.utils.cpp_extension" not in text
        assert "cpp_extension.load" not in text
        assert "load_inline(" not in text


def _import_in_site(site: Path, module: str) -> str:
    """Import ``module`` in a fresh interpreter whose ``b12x`` is the shim."""
    code = (
        "import importlib, sys\n"
        f"sys.path[:0] = [{str(site)!r}, {str(ROOT)!r}]\n"
        "try:\n"
        f"    importlib.import_module({module!r})\n"
        "except ModuleNotFoundError as exc:\n"
        "    print(exc.name, exc, sep='|')\n"
        "else:\n"
        "    print('ok')\n"
    )
    run = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    )
    return run.stdout.strip()


def test_compat_package_explains_standalone_b12x_conflict(tmp_path: Path) -> None:
    # Layout after installing standalone b12x, then flashinfer-python: the shim
    # owns b12x/__init__.py while the standalone modules and metadata remain.
    package = tmp_path / "b12x"
    plugin = package / "integration" / "vllm"
    plugin.mkdir(parents=True)
    shutil.copy(COMPAT_INIT, package / "__init__.py")
    for init in (package / "integration", plugin):
        (init / "__init__.py").touch()
    (plugin / "plugin.py").write_text("def register_b12x_fp6():\n    pass\n")

    # Without standalone metadata, a missing module keeps the default error.
    assert _import_in_site(tmp_path, "b12x.integration.vllm.plugin") == (
        "b12x.integration|No module named 'b12x.integration'"
    )

    dist_info = tmp_path / "b12x-1.5.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: b12x\nVersion: 1.5.1\n"
    )
    (dist_info / "RECORD").write_text(
        "".join(
            f"{path},,\n"
            for path in (
                "b12x/__init__.py",
                "b12x/integration/__init__.py",
                "b12x/integration/vllm/__init__.py",
                "b12x/integration/vllm/plugin.py",
            )
        )
    )

    name, message = _import_in_site(tmp_path, "b12x.integration.vllm.plugin").split("|")
    assert name == "b12x.integration"
    assert "standalone b12x 1.5.1 distribution" in message
    assert "reinstall flashinfer-python" in message
    # Modules the standalone wheel does not ship keep the default error, and
    # the embedded implementation still resolves first.
    assert _import_in_site(tmp_path, "b12x.not_a_module") == (
        "b12x.not_a_module|No module named 'b12x.not_a_module'"
    )
    assert _import_in_site(tmp_path, "b12x._lib.meta") == "ok"
