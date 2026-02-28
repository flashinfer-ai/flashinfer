import pytest
import torch

from flashinfer import mxfp8_dequantize_host, mxfp8_quantize
from flashinfer.utils import get_compute_capability


@pytest.mark.parametrize("m", [1, 1024])
@pytest.mark.parametrize("k", [1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_mxfp8_quantize_torch(m, k, dtype, is_sf_swizzled_layout, device):
    if device == "cuda":
        major, _ = get_compute_capability(torch.device(device))
        if major < 10:
            pytest.skip(
                "mxfp8 quantization is not supported on compute capability < 10"
            )

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
    major, _ = get_compute_capability(torch.device("cuda:0"))
    if major < 10:
        pytest.skip("mxfp8 quantization is not supported on compute capability < 10")

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
    major, _ = get_compute_capability(torch.device("cuda:0"))
    if major < 10:
        pytest.skip("mxfp8 quantization is not supported on compute capability < 10")

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


@pytest.mark.parametrize("m", [1, 128, 2048])
@pytest.mark.parametrize("k", [1024])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
def test_mxfp8_quantize_denormal_inputs(m, k, dtype, is_sf_swizzled_layout):
    """Test that very small denormalized inputs do not produce NaN.

    This test covers a bug where inputs small enough to cause E8M0 scale factor
    underflow would result in NaN outputs due to 0 * infinity computations.
    """
    major, _ = get_compute_capability(torch.device("cuda:0"))
    if major < 10:
        pytest.skip("mxfp8 quantization is not supported on compute capability < 10")

    torch.random.manual_seed(42)

    # Create very small denormalized values (below float32 normal range ~1.17e-38)
    # These values caused NaN in the original buggy implementation
    a = (torch.randn([m, k], dtype=torch.float32) * 1e-38).to(dtype).cuda().contiguous()

    a_fp8, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout)

    # The primary check: no NaN values should be produced
    nan_count = torch.isnan(a_fp8.float()).sum().item()
    assert nan_count == 0, f"Found {nan_count} NaN values in output (expected 0)"

    # Secondary check: no Inf values should be produced
    inf_count = torch.isinf(a_fp8.float()).sum().item()
    assert inf_count == 0, f"Found {inf_count} Inf values in output (expected 0)"


@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
def test_mxfp8_quantize_all_zeros(dtype, is_sf_swizzled_layout):
    """Test that all-zero inputs produce all-zero outputs without NaN."""
    major, _ = get_compute_capability(torch.device("cuda:0"))
    if major < 10:
        pytest.skip("mxfp8 quantization is not supported on compute capability < 10")

    m, k = 128, 1024
    a = torch.zeros([m, k], dtype=dtype, device="cuda").contiguous()

    a_fp8, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout)

    # No NaN values
    assert not torch.isnan(a_fp8.float()).any(), "NaN found in output for zero input"

    # All outputs should be zero
    assert (a_fp8.float() == 0).all(), "Non-zero output for zero input"


@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
def test_mxfp8_quantize_mixed_magnitude(dtype, is_sf_swizzled_layout):
    """Test mixed inputs: some blocks with normal values, some with denormals.

    This mimics real-world scenarios where different regions of a tensor
    may have vastly different magnitudes.
    """
    major, _ = get_compute_capability(torch.device("cuda:0"))
    if major < 10:
        pytest.skip("mxfp8 quantization is not supported on compute capability < 10")

    torch.random.manual_seed(123)

    m, k = 256, 1024
    a = torch.randn([m, k], dtype=torch.float32)

    # Make some rows have very small values (denormals)
    # Rows 0-63: normal magnitude
    # Rows 64-127: very small (denormal range)
    # Rows 128-191: normal magnitude
    # Rows 192-255: extremely small
    a[64:128, :] *= 1e-38
    a[192:256, :] *= 1e-40

    a = a.to(dtype).cuda().contiguous()

    a_fp8, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout)

    # No NaN values should be produced anywhere
    nan_mask = torch.isnan(a_fp8.float())
    nan_count = nan_mask.sum().item()
    if nan_count > 0:
        nan_positions = torch.where(nan_mask)
        first_nan_row = nan_positions[0][0].item()
        first_nan_col = nan_positions[1][0].item()
        pytest.fail(
            f"Found {nan_count} NaN values. First NaN at row={first_nan_row}, col={first_nan_col}"
        )


@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
def test_mxfp8_quantize_single_denormal_in_block(dtype, is_sf_swizzled_layout):
    """Test a block where most values are normal but one is a tiny denormal.

    This specifically tests the scenario from the original bug report where
    a single float32 denormal value in a block would become NaN due to
    0 * infinity when FTZ mode flushes it to zero.
    """
    major, _ = get_compute_capability(torch.device("cuda:0"))
    if major < 10:
        pytest.skip("mxfp8 quantization is not supported on compute capability < 10")

    m, k = 64, 1024
    # Start with small but normal-range values
    a = torch.full([m, k], 1e-36, dtype=torch.float32)

    # Insert a few extremely small values (float32 denormals) at specific positions
    # These are the values that triggered NaN in the original bug
    denormal_positions = [(0, 498), (0, 911), (32, 100), (63, 512)]
    for row, col in denormal_positions:
        a[row, col] = 9.18e-40  # A float32 denormal value

    a = a.to(dtype).cuda().contiguous()

    a_fp8, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout)

    # Check that no NaN is produced
    nan_mask = torch.isnan(a_fp8.float())
    assert not nan_mask.any(), f"Found NaN at positions: {torch.where(nan_mask)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
