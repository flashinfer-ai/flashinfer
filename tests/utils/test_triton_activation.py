import pytest
import torch
import torch.nn.functional as F

from flashinfer.triton.activation import silu_and_mul


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [31, 512, 1025])
@pytest.mark.parametrize("column_stride", [0, 1, 2])
@pytest.mark.parametrize("scaled", [False, True])
def test_silu_and_mul_strided_input(dtype, dim, column_stride, scaled):
    """Strided columns match the reference with and without scaling."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is not supported on this GPU")
    torch.manual_seed(42)
    base = torch.randn(
        6, 2 * dim * max(column_stride, 1) + 1, dtype=dtype, device="cuda"
    )
    if column_stride == 0:
        x = base[::2, 1:2].expand(-1, 2 * dim)
    else:
        x = base[::2, 1::column_stride]
    x_scale = torch.tensor([0.25], device=x.device) if scaled else None
    o_scale = torch.tensor([1.75], device=x.device) if scaled else None

    y = silu_and_mul(x, x_scale=x_scale, o_scale=o_scale)
    x_ref = x.float() * (0.25 if scaled else 1.0)
    expected = F.silu(x_ref[:, :dim]) * x_ref[:, dim:]
    if scaled:
        expected = (expected * 1.75).clamp(
            min=torch.finfo(dtype).min, max=torch.finfo(dtype).max
        )
    torch.testing.assert_close(y, expected.to(dtype), rtol=2e-2, atol=2e-3)


def test_silu_and_mul_strided_scaled_saturation():
    """Scaled strided inputs clamp to the largest finite output value."""
    dim = 31
    base = torch.zeros(2, 4 * dim, dtype=torch.float16, device="cuda")
    x = base[:, ::2]
    x.fill_(1000)
    x_scale = torch.tensor([0.25], device=x.device)
    o_scale = torch.tensor([1.75], device=x.device)

    y = silu_and_mul(x, x_scale=x_scale, o_scale=o_scale)
    expected = torch.full((2, dim), 65504, dtype=x.dtype, device=x.device)
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
