import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, bmm_bf16
from flashinfer.utils import get_compute_capability


@pytest.mark.parametrize("b", [1, 16])
@pytest.mark.parametrize("m", [48, 128])
@pytest.mark.parametrize("n", [80, 64])
@pytest.mark.parametrize("k", [64, 256])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
def test_bmm_bf16(b, m, n, k, res_dtype):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    compute_capability_number = compute_capability[0] * 10 + compute_capability[1]
    if not bmm_bf16.is_compute_capability_supported(compute_capability_number):
        pytest.skip(
            f"bmm_bf16 not supported on current compute capability."
            f"Detected sm{compute_capability_number}."
        )
    torch.manual_seed(7)
    input = torch.randn([b, m, k], device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn([b, n, k], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    reference = torch.bmm(input, mat2)

    out = torch.empty([b, m, n], device="cuda", dtype=res_dtype)
    with autotune():
        bmm_bf16(input, mat2, out=out, out_dtype=res_dtype)

    cos_sim = F.cosine_similarity(reference.reshape(-1), out.reshape(-1), dim=0)
    assert cos_sim > 0.99


if __name__ == "__main__":
    pytest.main([__file__])
