"""Packaging checks for the cake DSA training and indexer kernels (CPU only).

Both backends compile their generated CUDA sources on first use from the
installed package's ``csrc`` directory. ``include-package-data`` is off, so
each package needs its own ``[tool.setuptools.package-data]`` entry; without it
a built wheel ships none of the sources and the first launch fails, while
source-checkout runs still pass.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PACKAGES = (
    "flashinfer.experimental.cake_dsa_train",
    "flashinfer.experimental.cake_dsa_indexer",
)


def _registered_sources(package: str) -> list[str]:
    """Every translation unit the package's JIT registry compiles, relative to csrc."""
    if package.endswith(".cake_dsa_train"):
        from flashinfer.experimental.cake_dsa_train import cake_jit as train_jit

        record = train_jit.record()
        return [
            source
            for stage in train_jit.registered_stages()
            for source in record[stage]["sources"]
        ]
    from flashinfer.experimental.cake_dsa_indexer import cake_jit as indexer_jit

    return [
        source
        for program in indexer_jit.PROGRAMS.values()
        for source in program["sources"]
    ]


@pytest.mark.parametrize("package", _PACKAGES)
def test_package_data_declares_csrc(package: str) -> None:
    pyproject_path = _PROJECT_ROOT / "pyproject.toml"
    if not pyproject_path.is_file():
        pytest.skip("pyproject.toml is only available in source-tree test runs")
    config = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    package_data = config["tool"]["setuptools"]["package-data"]
    assert package_data.get(package) == ["csrc/**"]


@pytest.mark.parametrize("package", _PACKAGES)
def test_registered_sources_are_package_resources(package: str) -> None:
    sources = _registered_sources(package)
    assert sources
    missing = []
    for source in sources:
        resource = resources.files(package).joinpath("csrc")
        for part in source.split("/"):
            resource = resource.joinpath(part)
        if not resource.is_file():
            missing.append(source)
    assert not missing, f"{package} does not ship these JIT sources: {missing}"
