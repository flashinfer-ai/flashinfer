"""CPU-only integrity tests for vendored Blackwell MSA CUDA sources."""

import hashlib
import json
import re
from pathlib import Path


_CSRC_DIR = Path(__file__).resolve().parents[2] / "csrc" / "blackwell_msa"
_MANIFEST = json.loads((_CSRC_DIR / "route_manifest.json").read_text())
_INVENTORY = _MANIFEST["source_inventory"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_inventory_is_complete_and_hash_locked() -> None:
    entries = _INVENTORY["entries"]
    expected_paths = {
        Path(entry["target"]) / f"blackwell_msa_{entry['source_unit']}.cu"
        for entry in entries
    } | {
        Path(entry["target"])
        / f"blackwell_msa_{entry['source_unit']}_binding.cu"
        for entry in entries
    }
    actual_paths = {
        path.relative_to(_CSRC_DIR)
        for target in _INVENTORY["targets"]
        for path in (_CSRC_DIR / target).glob("*.cu")
    }

    assert len(entries) == 63
    assert set(_INVENTORY["targets"]) == {"sm100a", "sm103a"}
    assert len(actual_paths) == 126
    assert actual_paths == expected_paths
    for entry in entries:
        base = _CSRC_DIR / entry["target"] / f"blackwell_msa_{entry['source_unit']}"
        assert _sha256(base.with_suffix(".cu")) == entry["vendored_sha256"]
        assert _sha256(base.with_name(base.name + "_binding.cu")) == entry[
            "binding_sha256"
        ]
        assert entry["normalization_schema"] == _INVENTORY["normalization_schema"]
        assert len(entry["replacement_counts"]) == 2
        assert all(value > 0 for value in entry["replacement_counts"].values())


def test_each_binding_includes_its_device_source_and_exports_run() -> None:
    include_pattern = re.compile(r'#include "(blackwell_msa_[a-z0-9_]+\.cu)"')
    for entry in _INVENTORY["entries"]:
        unit = entry["source_unit"]
        target_dir = _CSRC_DIR / entry["target"]
        binding = (target_dir / f"blackwell_msa_{unit}_binding.cu").read_text()
        assert include_pattern.findall(binding) == [f"blackwell_msa_{unit}.cu"]
        assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run," in binding
        assert "CheckBlackwellMsaTarget" in binding
        assert "cudaLaunchKernel" in binding
        assert "EmbedCubin" not in binding
        assert "GetKernel" not in binding


def test_tma_parameters_are_passed_by_grid_constant_value() -> None:
    tma_parameter = re.compile(
        r"\bconst __grid_constant__ BlackwellMsaTensorMap [A-Za-z0-9_]+\b"
    )
    for entry in _INVENTORY["entries"]:
        source = (
            _CSRC_DIR
            / entry["target"]
            / f"blackwell_msa_{entry['source_unit']}.cu"
        ).read_text()
        if entry["source_unit"] == "topk":
            assert not tma_parameter.search(source)
        else:
            assert tma_parameter.search(source)
        assert "__device__ CUtensorMap" not in source
        assert "__constant__ CUtensorMap" not in source


def test_pdl_receipts_match_extended_launch_bindings() -> None:
    for entry in _INVENTORY["entries"]:
        binding = (
            _CSRC_DIR
            / entry["target"]
            / f"blackwell_msa_{entry['source_unit']}_binding.cu"
        ).read_text()
        if entry["programmatic_stream_serialization"]:
            assert "cudaLaunchKernelExC" in binding
            assert "cudaLaunchAttributeProgrammaticStreamSerialization" in binding
        else:
            assert "cudaLaunchKernelExC" not in binding


def test_sources_do_not_embed_device_images_or_device_tma_arenas() -> None:
    forbidden = (
        "static const unsigned char cubin",
        "cuModuleLoadData",
        "TmaDeviceArena",
        "TmaDeviceSlot",
        "cuMemAlloc",
        "cuMemcpyHtoD",
    )
    for path in _CSRC_DIR.glob("sm*/*.cu"):
        text = path.read_text()
        assert not [token for token in forbidden if token in text], path
