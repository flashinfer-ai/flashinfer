import pytest
import torch
import torch.nn.functional as F

from flashinfer import fp4_quantize, mm_fp4


def quant_fp4(a):
    a_global_sf = (448 * 6) / a.float().abs().nan_to_num().max()
    sf_vec_size = 16

    a_fp4, a_sf = fp4_quantize(
        a.cuda(),
        a_global_sf.cuda(),
        sf_vec_size,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=True,
    )
    return a_fp4, a_sf, a_global_sf


@pytest.mark.parametrize("m", [48, 128])
@pytest.mark.parametrize("n", [128, 256])
@pytest.mark.parametrize("k", [128, 512])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
def test_mm_fp4(m, n, k, res_dtype):
    input = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)

    (input_fp4, input_inv_s, global_sf_input) = quant_fp4(input)
    (mat2_fp4, mat2_inv_s, global_sf_mat2) = quant_fp4(mat2)

    reference = torch.mm(input, mat2)

    alpha = 1.0 / (global_sf_input * global_sf_mat2)
    res = torch.empty([m, n], device="cuda", dtype=res_dtype)
    mm_fp4(input_fp4, mat2_fp4, input_inv_s, mat2_inv_s, alpha, res_dtype, res)

    cos_sim = F.cosine_similarity(reference.reshape(-1), res.reshape(-1), dim=0)
    assert cos_sim > 0.97


if __name__ == "__main__":
    pytest.main([__file__])
