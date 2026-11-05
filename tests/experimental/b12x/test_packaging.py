from __future__ import annotations


from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[3]
PCIE_SOURCE = ROOT / "flashinfer/experimental/b12x" / "comm" / "pcie"


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
