# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

from flashinfer.conv.nvfp4_sm120 import _kernel_name


def test_conv3d_kernel_name_includes_max_active_clusters():
    common = {
        "input_shape": (1, 128, 4, 98, 135),
        "output_channels": 128,
        "fuse_alpha": True,
        "fuse_bias": True,
        "a_copy_bits": 128,
        "a_copy_layout": "coalesced",
        "a_producer_warps": 4,
        "n_pair": False,
        "swizzle_size": 2,
    }

    name_80 = _kernel_name(**common, max_active_clusters=80)
    name_120 = _kernel_name(**common, max_active_clusters=120)

    assert name_80.endswith("_mac80")
    assert name_120.endswith("_mac120")
    assert name_80 != name_120
