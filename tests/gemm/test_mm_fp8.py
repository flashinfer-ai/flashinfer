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
@pytest.mark.parametrize(
    "backend",
    [
        "trtllm_low_latency",
        "cutedsl_low_latency",
        "cublas",
        "cudnn",
        "cutlass",
        "auto",
        "auto_prepared",
    ],
)
def test_mm_fp8(
    m: int,
    n: int,
    k: int,
    input_dtype: torch.dtype,
    mat2_dtype: torch.dtype,
    res_dtype: torch.dtype,
    backend: str,
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("mm_fp8 is only supported on Blackwell GPUs.")
    if backend == "cutedsl_low_latency":
        if compute_capability[1] not in (0, 3, 7):
            pytest.skip("cutedsl_low_latency mm_fp8 requires SM100/SM103/SM107")
        if m > 8:
            pytest.skip("cutedsl_low_latency mm_fp8 requires M <= 8")

    torch.manual_seed(123)
    input = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    input_fp8, input_inv_s = to_float8(input, dtype=input_dtype)

    mat2 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=mat2_dtype)

    res = torch.zeros([m, n], device="cuda", dtype=res_dtype)
    global_scale = input_inv_s * mat2_inv_s

    weights = (
        prepare_low_latency_gemm_weights(mat2_fp8, _cache_permute_indices)
        if backend in ("trtllm_low_latency", "auto_prepared")
        else mat2_fp8
    )
    with autotune():
        mm_fp8(
            input_fp8,
            weights,
            global_scale,
            out=res,
            backend="auto" if backend == "auto_prepared" else backend,
        )

    reference = torch.mm(input, mat2.transpose(-2, -1))
    cos_sim = F.cosine_similarity(reference.reshape(-1), res.reshape(-1), dim=0)
    assert cos_sim > 0.99


@pytest.mark.parametrize(
    "m,n,k", [(1, 5120, 8192), (16, 5120, 16384), (128, 2560, 8192)]
)
@pytest.mark.parametrize("prepared", [False, True])
def test_mm_fp8_auto_routing(m: int, n: int, k: int, prepared: bool):
    """Untuned backend="auto" follows the weight layout; cuBLAS first on SM107."""
    major, minor = get_compute_capability(torch.device(device="cuda"))
    if major != 10:
        pytest.skip("mm_fp8 is only supported on Blackwell GPUs.")
    torch.manual_seed(123)
    input = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    input_fp8, input_inv_s = to_float8(input, dtype=torch.float8_e4m3fn)
    mat2 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=torch.float8_e4m3fn)
    weights = (
        prepare_low_latency_gemm_weights(mat2_fp8, _cache_permute_indices)
        if prepared
        else mat2_fp8
    )

    res = mm_fp8(input_fp8, weights, input_inv_s * mat2_inv_s, backend="auto")

    picked = mm_fp8.suitable_auto_backends
    if prepared:
        assert picked == ["trtllm_low_latency"]
    else:
        assert "trtllm_low_latency" not in picked
        assert picked[0] == ("cublas" if minor == 7 else "cutlass_sm10x")
    reference = torch.mm(input, mat2.transpose(-2, -1))
    cos_sim = F.cosine_similarity(reference.reshape(-1), res.reshape(-1), dim=0)
    assert cos_sim > 0.99


def test_mm_fp8_trtllm_low_latency_rejects_unprocessed_weight():
    major, _ = get_compute_capability(torch.device(device="cuda"))
    if major != 10:
        pytest.skip("mm_fp8 is only supported on Blackwell GPUs.")
    a = torch.zeros([4, 256], device="cuda").to(torch.float8_e4m3fn)
    b = torch.zeros([256, 256], device="cuda").to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="prepare_low_latency_gemm_weights"):
        mm_fp8(a, b, backend="trtllm_low_latency")


if __name__ == "__main__":
    pytest.main([__file__])
