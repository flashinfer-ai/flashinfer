"""Host contracts for the stage-contiguous SM90 W4A8 payload layout."""

from pathlib import Path

import pytest

from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.nvfp4_w4a8_gemm import (
    get_sm90_push_nvfp4_w4a8_gemm_uri,
)


_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "flashinfer/moe_ep/kernel_src/sm90/push_style_megamoe/src/"
    "nvfp4_w4a8_gemm"
)


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
    binding = "".join((_SOURCE / "binding.cu").read_text().split())
    kernel = "".join((_SOURCE / "kernel.cuh").read_text().split())

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


def test_w4a8_aot_registers_only_the_v4_default() -> None:
    aot = (Path(__file__).resolve().parents[2] / "flashinfer" / "aot.py").read_text()

    assert "gen_sm90_push_nvfp4_w4a8_gemm_module(payload_layout=4)" in "".join(
        aot.split()
    )
    assert "gen_sm90_push_nvfp4_w4a8_gemm_module(payload_layout=3)" not in "".join(
        aot.split()
    )
