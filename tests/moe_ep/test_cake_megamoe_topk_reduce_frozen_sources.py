"""Host-side identity gate for the frozen Cake MegaMoE TopK reducer exports.

These tests never build or launch a kernel: they verify that every published
per-arch export (``csrc/cake_megamoe_topk_reduce/<arch>/``) matches the
identity pinned in ``flashinfer.jit.cake_megamoe_topk_reduce`` and that the
arch selection follows the exact compute capability.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flashinfer.jit import cake_megamoe_topk_reduce as reducer

_ARCHS = ("sm_100a", "sm_103a")
_ARCH_DIRS = {"sm_100a": "sm100a", "sm_103a": "sm103a"}
_CAPABILITIES = {"sm_100a": (10, 0), "sm_103a": (10, 3)}


def _csrc_dir() -> Path:
    # Editable installs expose csrc through flashinfer/data/csrc; compare resolved paths.
    return reducer._get_csrc_dir().resolve()


def test_supported_capabilities_are_exact_blackwell_targets():
    assert reducer.supported_capabilities() == ((10, 0), (10, 3))
    assert tuple(reducer._ARCHS) == _ARCHS


@pytest.mark.parametrize("arch", _ARCHS)
def test_frozen_export_matches_pinned_identity(arch: str):
    source, manifest = reducer._program_source(arch)
    arch_dir = _csrc_dir() / _ARCH_DIRS[arch]
    assert source.resolve() == arch_dir / "cake_megamoe_topk_reduce_kernels.cu"
    source_bytes = source.read_bytes()
    assert (
        hashlib.sha256(source_bytes).hexdigest()
        == reducer._ARCHS[arch]["source_sha256"]
    )
    assert manifest["arch"] == arch
    assert manifest["source_sha256"] == reducer._ARCHS[arch]["source_sha256"]
    assert manifest["launch"] == {
        "block_threads": 256,
        "dynamic_smem_bytes": 0,
        "grid_x": "4 * num_tokens",
    }
    assert manifest["kernel_symbols"] == [reducer._KERNEL_SYMBOL]
    assert source_bytes.count(reducer._KERNEL_SYMBOL.encode()) == 1
    assert b"__launch_bounds__(256)" in source_bytes
    # The exported translation unit is arch-neutral: no SM100-only or
    # SM103-only instructions, so the same schedule serves both targets.
    assert b"tcgen05" not in source_bytes
    assert b"mbarrier" not in source_bytes
    assert (
        json.loads((arch_dir / "manifest.json").read_text(encoding="utf-8")) == manifest
    )


def test_frozen_exports_share_one_schedule():
    """Both per-arch drops are the same exporter output modulo nothing."""
    sources = {arch: reducer._program_source(arch)[0].read_bytes() for arch in _ARCHS}
    assert sources["sm_100a"] == sources["sm_103a"]


@pytest.mark.parametrize("arch", _ARCHS)
def test_binding_source_pins_exact_target(arch: str):
    binding = reducer._binding_source(arch)
    major, minor = _CAPABILITIES[arch]
    assert (
        f'#define CAKE_MEGAMOE_TOPK_REDUCE_BODY_FILE "{_ARCH_DIRS[arch]}/cake_megamoe_topk_reduce_kernels.cu"'
        in binding
    )
    assert f"#define CAKE_MEGAMOE_TOPK_REDUCE_CC_MAJOR {major}" in binding
    assert f"#define CAKE_MEGAMOE_TOPK_REDUCE_CC_MINOR {minor}" in binding
    assert "#define CAKE_MEGAMOE_TOPK_REDUCE_THREADS 256" in binding
    assert "#define CAKE_MEGAMOE_TOPK_REDUCE_SMEM_BYTES 0" in binding


@pytest.mark.parametrize("arch", _ARCHS)
def test_module_uri_is_arch_specific(arch: str):
    uri = reducer.get_cake_megamoe_topk_reduce_uri(arch)
    assert uri.startswith(f"cake_megamoe_topk_reduce_{_ARCH_DIRS[arch]}_")
    other = next(a for a in _ARCHS if a != arch)
    assert uri != reducer.get_cake_megamoe_topk_reduce_uri(other)


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((10, 0), "sm_100a"), ((10, 3), "sm_103a")],
)
def test_resolve_arch_maps_exact_capability(monkeypatch, capability, expected):
    import torch

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    assert reducer.resolve_arch() == expected
    assert reducer.is_cake_megamoe_topk_reduce_module_loaded() is (
        expected in reducer._LOADED_MODULES
    )


@pytest.mark.parametrize("capability", [(9, 0), (10, 1), (12, 0)])
def test_resolve_arch_rejects_other_capabilities(monkeypatch, capability):
    import torch

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    with pytest.raises(NotImplementedError, match="published for compute capabilities"):
        reducer.resolve_arch()
    assert reducer.is_cake_megamoe_topk_reduce_module_loaded() is False


def test_unknown_arch_is_rejected():
    with pytest.raises(ValueError, match="unknown frozen MegaMoE TopK reducer arch"):
        reducer._program_source("sm_90a")
