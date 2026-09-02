"""Host contracts for the stage-contiguous SM90 W4A8 payload layout."""

from importlib import resources as importlib_resources
from importlib.util import find_spec
from pathlib import Path

import pytest

from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.nvfp4_w4a8_gemm import (
    get_sm90_push_nvfp4_w4a8_gemm_uri,
)


_PACKAGE_NAME = "flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe"
_SOURCE_TREE_PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "flashinfer"
    / "moe_ep"
    / "kernel_src"
    / "sm90"
    / "push_style_megamoe"
)


def _package_text(*parts: str) -> str:
    source_tree = _SOURCE_TREE_PACKAGE_ROOT.joinpath(*parts)
    if source_tree.is_file():
        return source_tree.read_text(encoding="utf-8")

    resource = importlib_resources.files(_PACKAGE_NAME)
    for part in parts:
        resource = resource / part
    return resource.read_text(encoding="utf-8")


def test_w4a8_payload_layout_versions_have_distinct_modules() -> None:
    default = get_sm90_push_nvfp4_w4a8_gemm_uri()
    v3 = get_sm90_push_nvfp4_w4a8_gemm_uri(payload_layout=3)
    v4 = get_sm90_push_nvfp4_w4a8_gemm_uri(payload_layout=4)

    assert default == v4
    assert v3 != v4
    assert "_pv3_" in v3
    assert "_pv4_" in v4
    with pytest.raises(ValueError, match="payload_layout"):
        get_sm90_push_nvfp4_w4a8_gemm_uri(payload_layout=5)


def test_w4a8_v4_uses_one_stage_contiguous_payload_and_residual_tma() -> None:
    binding = "".join(_package_text("src", "nvfp4_w4a8_gemm", "binding.cu").split())
    kernel = "".join(_package_text("src", "nvfp4_w4a8_gemm", "kernel.cuh").split())

    assert "#ifW4A8_PAYLOAD_V4" in binding
    assert "global_dims[3]={static_cast<uint64_t>(kBlockK/2)" in binding
    assert "kElementsPerRow=kBlockK/kV3ResidualBlockK" in binding
    assert "rows_per_tma_row=scheme==ResidualScheme::kPow2?2:1" in binding
    assert "elements_per_tma_row=kElementsPerRow*rows_per_tma_row" in binding
    assert "#ifW4A8_PAYLOAD_V4" in kernel
    assert "kResidualRowsPerTmaRow=Scheme==ResidualScheme::kPow2?2:1" in kernel
    assert "task.n_begin/kResidualRowsPerTmaRow" in kernel
    assert "(n_local*4+k32_in_stage)*kV3PackedBytesPerRow" in kernel
    assert "n_local*8+k32_in_stage*kV3ResidualsPerPayloadTile" in kernel


def test_w4a8_stays_out_of_aot_prebuild() -> None:
    # The W4A8 grouped GEMM is JIT-only, matching every other moe_ep kernel:
    # aot.py must not pin a knob combination into the prebuilt jit-cache.
    spec = find_spec("flashinfer.aot")
    assert spec is not None and spec.origin is not None
    aot = Path(spec.origin).read_text(encoding="utf-8")

    assert "gen_sm90_push_nvfp4_w4a8_gemm_module" not in aot
