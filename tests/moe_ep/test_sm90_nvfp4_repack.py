"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch

from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.nvfp4_checkpoint import (
    NVFP4Checkpoint,
    reference_dequantize_nvfp4,
)
from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.nvfp4_repack import (
    NVFP4_SM90_LAYOUT_VERSION,
    NVFP4V3Manifest,
    repack_nvfp4_sm90_v3,
    unpack_nvfp4_sm90_v3,
)


def _nvfp4_v3_checkpoint() -> NVFP4Checkpoint:
    payload = (
        torch.arange(2 * 65 * 24, dtype=torch.int64)
        .mul(37)
        .remainder(256)
        .to(torch.uint8)
        .reshape(2, 65, 24)
    )
    scales = (
        torch.arange(2 * 65 * 3, dtype=torch.int64)
        .remainder(17)
        .to(torch.float32)
        .reshape(2, 65, 3)
        .to(torch.float8_e4m3fn)
    )
    return NVFP4Checkpoint(
        payload,
        scales,
        torch.tensor((0.5, 0.75), dtype=torch.float32),
        (2, 65, 47),
        (7, 11),
        "modelopt.nvfp4.test",
    )


@pytest.mark.parametrize("group_size", (32, 64, 128))
@pytest.mark.parametrize("residual_scheme", ("generic", "pow2"))
def test_nvfp4_repack_v3_roundtrip_and_manifest_contract(group_size, residual_scheme):
    checkpoint = _nvfp4_v3_checkpoint()
    view = repack_nvfp4_sm90_v3(
        checkpoint,
        group_size=group_size,
        residual_scheme=residual_scheme,
    )
    restored = unpack_nvfp4_sm90_v3(view)
    assert view.manifest.layout_version == NVFP4_SM90_LAYOUT_VERSION == 3
    assert set(view.manifest.to_dict()) == {
        "layout_version",
        "source_format_version",
        "sm_target",
        "group_size",
        "residual_scheme",
        "rounding_mode",
        "logical_shape",
        "padded_shape",
        "nibble_order",
        "byte_order",
        "global_layout",
        "w13_layout",
        "alpha_scope",
        "expert_mapping",
        "checksums",
    }
    torch.testing.assert_close(
        reference_dequantize_nvfp4(restored),
        reference_dequantize_nvfp4(checkpoint),
        rtol=0,
        atol=0,
    )


def test_nvfp4_repack_v3_manifest_tampering_is_rejected():
    view = repack_nvfp4_sm90_v3(
        _nvfp4_v3_checkpoint(),
        group_size=64,
        residual_scheme="generic",
    )
    lossy = view.manifest.to_dict()
    lossy["expert_mapping"] = [7, 11.9]
    with pytest.raises(TypeError, match="expert_mapping entries"):
        NVFP4V3Manifest.from_dict(lossy)

    rebound = view.manifest.to_dict()
    rebound["sm_target"] = "sm90"
    tampered_manifest = NVFP4V3Manifest.from_dict(rebound)
    tampered_view = type(view)(
        view.packed_e2m1,
        view.scale_e4m3_per16,
        view.promotion_group_scale,
        view.promotion_residual,
        view.global_alpha,
        tampered_manifest,
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        unpack_nvfp4_sm90_v3(tampered_view)
