import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, mm_bf16
from flashinfer.utils import get_compute_capability


@pytest.mark.parametrize("m", [1, 8, 16, 32, 64])
@pytest.mark.parametrize("n", [1024, 2048, 4096])
@pytest.mark.parametrize("k", [1024, 2048, 3072])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("enable_bias", [True, False])
@pytest.mark.parametrize("pdl", [True, False])
@pytest.mark.parametrize("backend", ["cutlass", "tgv"])
def test_mm_bf16(
    m: int,
    n: int,
    k: int,
    res_dtype: torch.dtype,
    enable_bias: bool,
    pdl: bool,
    backend: str,
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    compute_capability_number = compute_capability[0] * 10 + compute_capability[1]
    if not mm_bf16.is_compute_capability_supported(compute_capability_number):
        pytest.skip(
            f"mm_bf16 not supported on current compute capability."
            f"Detected sm{compute_capability_number}."
        )

    if backend == "cutlass" and (enable_bias or pdl):
        pytest.skip(
            "mm_bf16 with CUTLASS backend does not support bias or pdl arguments."
        )
    if res_dtype == torch.float16 and backend == "tgv":
        pytest.skip(
            "mm_bf16 with TGV backend does not support specifying non-bfloat16 result dtypes."
        )

    torch.manual_seed(42)
    input = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)

    if enable_bias:
        bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        reference = F.linear(input, mat2, bias)
    else:
        bias = None
        reference = torch.mm(input, mat2.T)

    out = torch.empty([m, n], device="cuda", dtype=res_dtype)
    with autotune():
        mm_bf16(input, mat2.T, bias, pdl, out, res_dtype, backend)

    cos_sim = F.cosine_similarity(reference.reshape(-1), out.reshape(-1), dim=0)
    assert cos_sim > 0.99


if __name__ == "__main__":
    pytest.main([__file__])
