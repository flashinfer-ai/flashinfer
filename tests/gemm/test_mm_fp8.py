from typing import Dict
from flashinfer.utils import get_compute_capability
import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, mm_fp8
from tests.utils_fp8 import to_float8
from flashinfer import prepare_low_latency_gemm_weights

_cache_permute_indices: Dict[torch.Size, torch.Tensor] = {}


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("n", [2560, 5120])
@pytest.mark.parametrize("k", [8192, 16384, 32768])
@pytest.mark.parametrize("input_dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("mat2_dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16])
def test_mm_fp8(
    m: int,
    n: int,
    k: int,
    input_dtype: torch.dtype,
    mat2_dtype: torch.dtype,
    res_dtype: torch.dtype,
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("mm_fp8 is only supported on Blackwell GPUs.")

    torch.manual_seed(123)
    input = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    input_fp8, input_inv_s = to_float8(input, dtype=input_dtype)

    mat2 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=mat2_dtype)

    res = torch.zeros([m, n], device="cuda", dtype=res_dtype)
    global_scale = input_inv_s * mat2_inv_s

    prepared_weights = prepare_low_latency_gemm_weights(
        mat2_fp8, _cache_permute_indices
    )
    with autotune():
        mm_fp8(
            input_fp8,
            prepared_weights,
            global_scale,
            out=res,
        )

    reference = torch.mm(input, mat2.transpose(-2, -1))
    cos_sim = F.cosine_similarity(reference.reshape(-1), res.reshape(-1), dim=0)
    assert cos_sim > 0.99


if __name__ == "__main__":
    pytest.main([__file__])
