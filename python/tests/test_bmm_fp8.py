import pytest
import torch
from flashinfer import bmm_fp8


@pytest.mark.parametrize("input_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("mat2_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
def test_bmm_fp8(input_dtype, mat2_dtype, res_dtype):
    if input_dtype == torch.float8_e5m2 and mat2_dtype == torch.float8_e5m2:
        pytest.skip("Invalid combination: both input and mat2 are e5m2")

    input = torch.randn([16, 48, 64], device="cuda", dtype=torch.bfloat16)
    input_fp8 = input.to(input_dtype)

    mat2 = torch.randn([16, 64, 80], device="cuda", dtype=torch.bfloat16)
    # mat2 row major -> column major
    mat2_fp8 = mat2.to(mat2_dtype).transpose(-1, -2).contiguous()
    # make original shape unchanged
    mat2_fp8 = mat2_fp8.transpose(-1, -2)

    res = torch.empty([16, 48, 80], device="cuda", dtype=res_dtype)
    bmm_fp8(input_fp8, mat2_fp8, res)

    res_ref = (input @ mat2).to(res_dtype)

    res_float = res.float().cpu()
    res_ref_float = res_ref.float().cpu()

    is_close = torch.isclose(res_float, res_ref_float, rtol=1e-1, atol=1e-1)

    total_elements = res_float.numel()
    unequal_elements = torch.sum(~is_close).item()
    unequal_percentage = (unequal_elements / total_elements) * 100
    assert unequal_percentage < 10


if __name__ == "__main__":
    pytest.main([__file__])
