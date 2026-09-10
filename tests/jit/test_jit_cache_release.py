import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture
def cuda_config_module():
    name = "_test_validate_cuda_versions"
    module = _load_module(name, REPO_ROOT / "ci" / "validate_cuda_versions.py")
    try:
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def wheel_index_module():
    name = "_test_update_whl_index"
    module = _load_module(name, REPO_ROOT / "scripts" / "update_whl_index.py")
    try:
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def release_verifier_module():
    scripts_dir = REPO_ROOT / "scripts"
    name = "_test_verify_jit_cache_provider_release"
    sys.path.insert(0, str(scripts_dir))
    module = _load_module(name, scripts_dir / "verify_jit_cache_provider_release.py")
    try:
        yield module
    finally:
        sys.modules.pop(name, None)
        sys.modules.pop("jit_cache_provider_validation", None)
        sys.path.remove(str(scripts_dir))


def test_provider_release_matrices_are_explicit(cuda_config_module):
    config = json.loads((REPO_ROOT / "ci" / "cuda-versions.json").read_text())

    cuda_config_module.validate_cuda_config(config, REPO_ROOT)
    provider_matrix = cuda_config_module.build_jit_cache_provider_matrix(config)
    shim_matrix = cuda_config_module.build_jit_cache_shim_matrix(config)

    assert config["jit_cache_wheel_format"] == "legacy"
    assert len(provider_matrix) == 48
    assert len(shim_matrix) == 6
    assert all(entry["provider_tag"].startswith("sm") for entry in provider_matrix)
    assert {
        (entry["cuda_label"], entry["cpu_architecture"])
        for entry in provider_matrix
        if entry["provider_tag"] == "sm121a"
    } == {("cu130", "aarch64"), ("cu134", "aarch64")}
    assert sum(entry["provider_tag"] == "sm86" for entry in provider_matrix) == 6


def test_provider_release_matrix_rejects_duplicate_architectures(
    cuda_config_module,
):
    config = json.loads((REPO_ROOT / "ci" / "cuda-versions.json").read_text())
    invalid_config = deepcopy(config)
    invalid_config["jit_cache"][0]["x86_64_provider_architectures"].append("8.0")

    with pytest.raises(
        cuda_config_module.ConfigError, match="contains duplicate architectures"
    ):
        cuda_config_module.validate_cuda_config(invalid_config, REPO_ROOT)


def test_release_verifier_requires_exact_provider_set(release_verifier_module):
    shim = release_verifier_module.Wheel(
        path=Path(
            "flashinfer_jit_cache-0.6.16+cu130-cp39-abi3-manylinux_2_28_x86_64.whl"
        ),
        distribution="flashinfer-jit-cache",
        version="0.6.16+cu130",
        requirements=(
            "flashinfer-jit-cache-sm80==0.6.16+cu130",
            "flashinfer-jit-cache-sm90a==0.6.16+cu130",
        ),
        contents=("flashinfer_jit_cache-0.6.16.dist-info/METADATA",),
        metadata_path="flashinfer_jit_cache-0.6.16.dist-info/METADATA",
    )

    release_verifier_module.validate_shim(
        shim,
        "0.6.16+cu130",
        {"sm80", "sm90a"},
        "manylinux_2_28_x86_64",
    )
    with pytest.raises(ValueError, match="do not match"):
        release_verifier_module.validate_shim(
            shim,
            "0.6.16+cu130",
            {"sm80", "sm90a", "sm120f"},
            "manylinux_2_28_x86_64",
        )

    with pytest.raises(ValueError, match="does not use manylinux_2_28_aarch64"):
        release_verifier_module.validate_shim(
            shim,
            "0.6.16+cu130",
            {"sm80", "sm90a"},
            "manylinux_2_28_aarch64",
        )


@pytest.mark.parametrize(
    ("filename", "expected_package"),
    [
        (
            "flashinfer_jit_cache_sm90a-0.6.16+cu130-cp39-abi3-manylinux_2_28_x86_64.whl",
            "flashinfer-jit-cache-sm90a",
        ),
        (
            "flashinfer_jit_cache_sm121a-0.6.16.dev20260909+cu134-cp39-abi3-manylinux_2_28_aarch64.whl",
            "flashinfer-jit-cache-sm121a",
        ),
    ],
)
def test_wheel_index_recognizes_provider_distributions(
    wheel_index_module, filename, expected_package
):
    info = wheel_index_module.get_package_info(Path(filename))

    assert info == {
        "package": expected_package,
        "version": filename.split("-")[1],
        "cuda": "130" if "+cu130" in filename else "134",
    }
