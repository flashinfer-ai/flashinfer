import functools

import pytest
import torch

from flashinfer import mxfp8_dequantize_host, mxfp8_quantize
from flashinfer.utils import is_sm100a_supported


@pytest.mark.parametrize("m", [1, 1024])
@pytest.mark.parametrize("k", [1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_mxfp8_quantize_torch(m, k, dtype, is_sf_swizzled_layout, device):
    a = 16 * torch.randn([m, k], dtype=dtype).to(device).contiguous()

    if device == "cpu":
        a = a.float()

    a_fp8, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout)

    if device == "cuda":
        a_fp8 = a_fp8.cpu()
        a_sf = a_sf.cpu()

    a_pt = mxfp8_dequantize_host(
        a_fp8.view(torch.uint8),
        a_sf.view(torch.uint8).reshape(-1),
        is_sf_swizzled_layout,
    )

    if device == "cuda":
        a_pt = a_pt.cuda()

    torch.cuda.synchronize()

    def check_accuracy(a, b, atol, rtol, percent):
        if torch.any(torch.isnan(a)):
            raise Exception("NaN in a")
        if torch.any(torch.isnan(b)):
            raise Exception("NaN in b")
        assert a.shape == b.shape
        left = torch.abs(a - b)
        right = atol + rtol * torch.abs(b)
        count = torch.sum(left > right)
        mismatch_percent = count / a.numel()
        if mismatch_percent > 1 - percent:
            raise Exception(
                "Mismatch percentage is %f for rtol %f" % (mismatch_percent, rtol)
            )

    check_accuracy(a_pt, a, 8, 0, 0.999)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
