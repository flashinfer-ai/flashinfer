"""Dtype and layout regressions for the DLPack memory-view helpers."""

import pytest
import torch

from flashinfer.comm.dlpack_utils import create_dlpack_capsule, pack_strided_memory


@pytest.mark.parametrize(
    "dtype,code,bits",
    [
        (torch.bfloat16, 4, 16),
        (torch.float16, 2, 16),
        (torch.float32, 2, 32),
        (torch.int32, 0, 32),
    ],
)
def test_create_dlpack_capsule_dtype(dtype, code, bits):
    # Creating the descriptor does not access the pointer or require CUDA.
    wrapper = create_dlpack_capsule(0, 16, 32, 2, dtype, 0)
    dl_dtype = wrapper._managed_tensor.dl_tensor.dtype
    assert (dl_dtype.code, dl_dtype.bits, dl_dtype.lanes) == (code, bits, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32, torch.int32]
)
@pytest.mark.parametrize("padding", [0, 2])
def test_pack_strided_memory_dtype_and_values(dtype, padding):
    source = torch.arange(1, 2 * (4 + padding) + 1, dtype=dtype, device="cuda")
    source = source.reshape(2, 4 + padding)
    view = pack_strided_memory(
        source.data_ptr(),
        4 * source.element_size(),
        source.stride(0) * source.element_size(),
        source.size(0),
        dtype,
        source.device.index,
    )

    assert view.dtype == dtype
    assert view.device == source.device
    assert view.data_ptr() == source.data_ptr()
    assert view.stride() == source.stride()
    torch.testing.assert_close(view, source[:, :4], rtol=0, atol=0)
