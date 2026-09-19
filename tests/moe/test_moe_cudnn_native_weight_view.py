# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native operand metadata and capsule ownership for expert-strided weights."""

import gc

import pytest
import torch


@pytest.mark.parametrize("pitch", [65536, 65552, 131072])
def test_native_blocked_view_retains_expert_stride_and_capsule(pitch):
    from cudnn import _pybind_module
    from cudnn.gemm.frost.compiler import _blocked_moe_weight_view

    # The caller keeps storage alive; each native wrapper owns its metadata.
    backing = torch.arange(2 * pitch, device="cuda", dtype=torch.int64).to(torch.uint8)
    tensor = backing.as_strided((2, 2, 2, 128, 128), (pitch, 32768, 16384, 128, 1))
    pack = _pybind_module.VariantPackNative(1)
    assert pack.read_operand(0, tensor)
    native = pack.operand(0, torch.cuda.current_device())
    flat = _blocked_moe_weight_view(native)
    assert flat.data_ptr() == tensor.data_ptr()
    assert tuple(flat.shape) == (2, 256, 256)
    assert tuple(flat.stride()) == (pitch, 256, 1)
    assert flat.dtype == native.dtype
    assert flat.__dlpack_device__() == native.__dlpack_device__()
    capsule = flat.permute(1, 2, 0).__dlpack__()
    del pack, native, flat
    gc.collect()
    imported = torch.utils.dlpack.from_dlpack(capsule)
    assert imported.data_ptr() == tensor.data_ptr()
    assert tuple(imported.stride()) == (256, 1, pitch)
    expected = tensor.view(2, 256, 256).permute(1, 2, 0)
    assert torch.equal(imported, expected)
    backing.bitwise_xor_(128)
    assert torch.equal(imported, expected)
