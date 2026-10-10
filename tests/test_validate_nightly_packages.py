from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


VALIDATOR_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "validate_nightly_packages.py"
)


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_nightly_packages_under_test", VALIDATOR_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_packages(
    monkeypatch,
    validator,
    *,
    flashinfer_version="0.7.0.post1",
    cubin_version="0.7.0.post1",
    shim_version="0.7.0.post1+cu130",
    provider_versions=None,
    package_cubin_dir="/site-packages/flashinfer_cubin/cubins",
    selected_cubin_dir=None,
):
    versions = {
        "flashinfer-python": flashinfer_version,
        "flashinfer-cubin": cubin_version,
        "flashinfer-jit-cache": shim_version,
    }
    if provider_versions is None:
        provider_versions = {"flashinfer-jit-cache-sm90a": shim_version}
    distributions = [
        SimpleNamespace(metadata={"Name": name}, version=version)
        for name, version in provider_versions.items()
    ]
    modules = {
        "flashinfer_cubin": SimpleNamespace(get_cubin_dir=lambda: package_cubin_dir),
        "flashinfer.jit.env": SimpleNamespace(
            FLASHINFER_CUBIN_DIR=selected_cubin_dir or package_cubin_dir
        ),
    }

    monkeypatch.setattr(
        validator.importlib.metadata, "version", lambda name: versions[name]
    )
    monkeypatch.setattr(
        validator.importlib.metadata, "distributions", lambda: distributions
    )
    monkeypatch.setattr(
        validator.importlib, "import_module", lambda name: modules[name]
    )


def test_nightly_package_validation_accepts_matching_packages(monkeypatch) -> None:
    validator = _load_validator()
    _configure_packages(monkeypatch, validator)

    assert validator._validation_errors() == []


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {"cubin_version": "0.7.0"},
            "flashinfer-cubin version '0.7.0' does not match",
        ),
        (
            {"selected_cubin_dir": "/tmp/.cache/flashinfer/cubins"},
            "not the installed flashinfer-cubin directory",
        ),
        (
            {"shim_version": "0.7.0+cu130"},
            "flashinfer-jit-cache version '0.7.0+cu130' does not match",
        ),
        (
            {"shim_version": "0.7.0.post1+cpu"},
            "flashinfer-jit-cache version '0.7.0.post1+cpu' does not match",
        ),
        (
            {"provider_versions": {"flashinfer-jit-cache-sm90a": "0.7.0.post1+cu129"}},
            "flashinfer-jit-cache-sm90a version '0.7.0.post1+cu129' does not match",
        ),
    ],
)
def test_nightly_package_validation_rejects_mismatches(
    monkeypatch, overrides, expected
) -> None:
    validator = _load_validator()
    _configure_packages(monkeypatch, validator, **overrides)

    assert expected in "\n".join(validator._validation_errors())
