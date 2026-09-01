from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch

from flashinfer.jit import cake_concat_mla_k as loader


def _clear_loader_caches() -> None:
    loader.get_cake_concat_mla_k_module_spec.cache_clear()
    loader.gen_cake_concat_mla_k_module.cache_clear()
    loader.load_cake_concat_mla_k_module.cache_clear()


@pytest.fixture(autouse=True)
def clear_loader_caches():
    _clear_loader_caches()
    yield
    _clear_loader_caches()


def test_cake_concat_mla_k_manifest_verifies_source_checkout():
    spec = loader.get_cake_concat_mla_k_module_spec()
    assert spec.module_ident == "cake_concat_mla_k_vector_copy_sm_103a_main"
    assert len(spec.closure_sha256) == 64
    assert spec.device_path.name == "cake_concat_mla_k_vector_copy_sm_103a.cu"
    assert spec.binding_path.name == "cake_concat_mla_k_vector_copy_sm_103a_binding.cu"
    assert spec.device_path.is_file()
    assert spec.binding_path.is_file()
    assert loader.get_cake_concat_mla_k_uri().endswith(spec.closure_sha256)


def test_cake_concat_mla_k_manifest_verifies_installed_sources(
    tmp_path: Path, monkeypatch
):
    checkout = Path(loader.__file__).resolve().parents[2] / "csrc" / "concat_mla"
    installed_root = tmp_path / "installed-csrc"
    shutil.copytree(checkout, installed_root / "concat_mla")
    monkeypatch.setattr(loader.jit_env, "FLASHINFER_CSRC_DIR", installed_root)

    spec = loader.get_cake_concat_mla_k_module_spec()
    assert spec.device_path.is_relative_to(installed_root)
    assert spec.binding_path.is_relative_to(installed_root)


def test_cake_concat_mla_k_manifest_rejects_source_drift(tmp_path: Path, monkeypatch):
    checkout = Path(loader.__file__).resolve().parents[2] / "csrc" / "concat_mla"
    installed_root = tmp_path / "installed-csrc"
    installed = installed_root / "concat_mla"
    shutil.copytree(checkout, installed)
    device = installed / "cake_concat_mla_k_vector_copy_sm_103a.cu"
    device.write_text(device.read_text() + "\n// unexpected drift\n")
    monkeypatch.setattr(loader.jit_env, "FLASHINFER_CSRC_DIR", installed_root)

    with pytest.raises(ValueError, match="sha256 mismatch"):
        loader.get_cake_concat_mla_k_module_spec()


def test_cake_concat_mla_k_jit_build_and_load():
    if not torch.cuda.is_available():
        pytest.skip("Cake concat MLA K JIT requires CUDA")
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Cake concat MLA K JIT requires exact SM103a")

    spec = loader.gen_cake_concat_mla_k_module()
    assert len(spec.sources) == 2
    assert spec.needs_device_linking is True
    assert [path.name for path in spec.sources] == [
        "cake_concat_mla_k_vector_copy_sm_103a.cu",
        "cake_concat_mla_k_vector_copy_sm_103a_binding.cu",
    ]
    module = loader.load_cake_concat_mla_k_module()
    assert hasattr(module, "run")
