# NOTE for future contributors (incl. AI agents): keep this file a SMALL curated
# smoke set. New coverage (shapes, dtypes, backends, randomized breadth) belongs in
# tests/gemm/test_unified_gemm_fuzz.py -- extend an adapter/axis there. Add cases
# here only as deliberate regression anchors or for paths the fuzzer cannot express.

import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, bmm_fp8
from flashinfer.utils import get_compute_capability
from tests.utils_fp8 import to_float8


# Curated smoke set. Randomized breadth over {b,m,n,k} x {e4m3,e5m2} x backends
# (with a tight elementwise oracle, determinism and autotune-winner checks) lives in
# tests/gemm/test_unified_gemm_fuzz.py's bmm_fp8 adapter; this file keeps one
# deliberate case per backend / dtype-mix / autotune mode for fast bisection.
_SMOKE_CASES = [
    # b, m, n, k, input_dtype, mat2_dtype, res_dtype, backend, auto_tuning
    (
        16,
        48,
        80,
        256,
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
        torch.bfloat16,
        "cudnn",
        True,
    ),
    (
        1,
        128,
        10304,
        2688,
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
        torch.float16,
        "cublas",
        False,
    ),
    (
        16,
        1,
        64,
        2688,
        torch.float8_e5m2,
        torch.float8_e4m3fn,
        torch.bfloat16,
        "cudnn",
        False,
    ),
    (
        1,
        48,
        80,
        64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float16,
        "cublas",
        True,
    ),
    (
        16,
        128,
        80,
        256,
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
        torch.bfloat16,
        "cutlass",
        True,
    ),
    (
        1,
        1,
        10304,
        256,
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
        torch.bfloat16,
        "auto",
        False,
    ),
]


@pytest.mark.parametrize(
    "b,m,n,k,input_dtype,mat2_dtype,res_dtype,backend,auto_tuning", _SMOKE_CASES
)
def test_bmm_fp8(b, m, n, k, input_dtype, mat2_dtype, res_dtype, backend, auto_tuning):
    compute_capability = get_compute_capability(torch.device("cuda"))
    if backend == "cutlass" and compute_capability[0] not in [10, 11, 12]:
        pytest.skip(
            "bmm_fp8 with cutlass backend is only supported on SM100, SM110, and SM120/121 GPUs."
        )
    if input_dtype == torch.float8_e5m2 and mat2_dtype == torch.float8_e5m2:
        pytest.skip("Invalid combination: both input and mat2 are e5m2")
    if input_dtype == torch.float8_e5m2 or mat2_dtype == torch.float8_e5m2:
        if backend == "cutlass":
            pytest.skip("Invalid combination: cutlass does not support e5m2")
    if auto_tuning and backend not in ["cutlass", "cudnn", "cublas"]:
        pytest.skip(
            "Invalid combination: auto_tuning only supported for cutlass, cudnn, and cublas"
        )
    if compute_capability[0] == 11 and (
        input_dtype == torch.float8_e5m2 or mat2_dtype == torch.float8_e5m2
    ):
        pytest.skip(
            "Invalid combination: only cutlass supports SM110 which does not support e5m2"
        )
    input = torch.randn([b, m, k], device="cuda", dtype=torch.bfloat16)
    input_fp8, input_inv_s = to_float8(input, dtype=input_dtype)

    # mat2 row  major -> column major
    mat2 = torch.randn([b, n, k], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=mat2_dtype)
    reference = torch.bmm(input, mat2)

    res = torch.empty([b, m, n], device="cuda", dtype=res_dtype)

    with autotune(auto_tuning):
        bmm_fp8(
            input_fp8,
            mat2_fp8,
            input_inv_s,
            mat2_inv_s,
            res_dtype,
            res,
            backend=backend,
        )

    cos_sim = F.cosine_similarity(
        reference.reshape(-1).float(), res.reshape(-1).float(), dim=0
    )
    assert cos_sim > 0.99


if __name__ == "__main__":
    pytest.main([__file__])
