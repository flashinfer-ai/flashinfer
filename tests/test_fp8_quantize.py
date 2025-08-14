import pytest
import torch

from flashinfer import mxfp8_dequantize_host, mxfp8_quantize


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


def mxfp8_quantize_check_accuracy(a, b, atol, rtol, percent):
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


@pytest.mark.parametrize("m", [1, 2, 16, 1024])
@pytest.mark.parametrize("k", [512, 1024])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
def test_mxfp8_quantize_torch_host(m, k, dtype, is_sf_swizzled_layout):
    torch.random.manual_seed(0)
    a = (torch.randn([m, k], dtype=torch.float) * 16).cpu().contiguous()

    a_fp8, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout)

    a_pt = mxfp8_dequantize_host(
        a_fp8.view(torch.uint8), a_sf.view(torch.uint8), is_sf_swizzled_layout
    )

    torch.cuda.synchronize()

    mxfp8_quantize_check_accuracy(a_pt, a, 8, 0, 0.999)


@pytest.mark.parametrize("m", [1, 2, 16, 1024])
@pytest.mark.parametrize("k", [512, 1024])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
def test_mxfp8_quantize_torch_device(m, k, dtype, is_sf_swizzled_layout):
    torch.random.manual_seed(0)
    a = (torch.randn([m, k], dtype=torch.float) * 16).to(dtype).cuda().contiguous()

    a_fp8, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout, 32)
    a_pt = mxfp8_dequantize_host(
        a_fp8.cpu().view(torch.uint8),
        a_sf.cpu().view(torch.uint8),
        is_sf_swizzled_layout,
    )

    torch.cuda.synchronize()
    mxfp8_quantize_check_accuracy(
        a_pt.cpu().to(torch.float32), a.cpu().to(torch.float32), 8, 0, 0.999
    )


@pytest.mark.parametrize("m", [1, 2, 16, 1024])
@pytest.mark.parametrize("k", [1568])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("alignment", [64, 128])
def test_mxfp8_quantize_alignment_torch_device(
    m, k, dtype, is_sf_swizzled_layout, alignment
):
    torch.random.manual_seed(0)
    a = (torch.randn([m, k], dtype=torch.float) * 16).to(dtype).cuda().contiguous()
    padded_k = ((k + alignment - 1) // alignment) * alignment

    # Quantize it on device.
    a_fp8, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout, alignment)
    assert a_fp8.shape[1] == padded_k

    # Dequantize it on host.
    a_pt = mxfp8_dequantize_host(
        a_fp8.cpu().view(torch.uint8),
        a_sf.cpu().view(torch.uint8),
        is_sf_swizzled_layout,
    )

    # Check if the bits of paddings are zero.
    paddings = a_fp8.view(torch.int8)[:, k:]
    assert torch.all(paddings == 0), "Paddings should be zero"

    torch.cuda.synchronize()

    mxfp8_quantize_check_accuracy(
        a_pt[:, :k].cpu().to(torch.float32), a.cpu().to(torch.float32), 8, 0, 0.999
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
