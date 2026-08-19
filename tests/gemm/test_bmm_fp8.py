import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, bmm_fp8
from flashinfer.utils import get_compute_capability
from tests.utils_fp8 import to_float8


@pytest.mark.parametrize("b", [1, 16])
# m=256 is the smallest value the SM107 cute-dsl backend can serve: every
# entry in SM107_AUTOTUNE_CONFIGS is 2-CTA with mma_tiler M=256, so the CTA
# tile needs m >= 256 (and n >= 128).  Without it that backend has no
# executable shape here at all.
@pytest.mark.parametrize("m", [1, 48, 128, 256])
@pytest.mark.parametrize("n", [64, 80, 10304])
@pytest.mark.parametrize("k", [64, 256, 2688])
@pytest.mark.parametrize("input_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("mat2_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["cudnn", "cublas", "cutlass", "cute-dsl", "auto"])
@pytest.mark.parametrize("auto_tuning", [True, False])
def test_bmm_fp8(b, m, n, k, input_dtype, mat2_dtype, res_dtype, backend, auto_tuning):
    compute_capability = get_compute_capability(torch.device("cuda"))
    if backend == "cutlass" and compute_capability[0] not in [10, 11, 12]:
        pytest.skip(
            "bmm_fp8 with cutlass backend is only supported on SM100, SM110, and SM120/121 GPUs."
        )
    # cute-dsl backend requirements
    if backend == "cute-dsl":
        if compute_capability != (10, 7):
            pytest.skip(
                "bmm_fp8 with cute-dsl backend is only supported on SM107 GPUs."
            )
        if m % 16 != 0 or n % 16 != 0 or k % 16 != 0:
            pytest.skip(
                "bmm_fp8 with cute-dsl backend requires m, n, k to be multiples of 16."
            )
        # Blackwell/Rubin kernel requires A and B to have the same dtype
        if input_dtype != mat2_dtype:
            pytest.skip(
                "bmm_fp8 with cute-dsl backend requires A and B to have the same dtype."
            )
        # All SM107 configs are 2-CTA with mma_tiler M=256, giving a CTA tile of
        # 256x128 once the 2x1 cluster is applied; a smaller problem cannot fill
        # one tile and bmm_fp8 raises rather than silently picking a bad tactic.
        if m < 256 or n < 128:
            pytest.skip(
                "bmm_fp8 with cute-dsl backend requires m >= 256 and n >= 128 "
                "(2-CTA tile); smaller problems have no valid SM107 config."
            )
    if (
        input_dtype == torch.float8_e5m2
        and mat2_dtype == torch.float8_e5m2
        and backend != "cute-dsl"
    ):
        pytest.skip("Invalid combination: both input and mat2 are e5m2")
    if input_dtype == torch.float8_e5m2 or mat2_dtype == torch.float8_e5m2:
        if backend == "cutlass":
            pytest.skip("Invalid combination: cutlass does not support e5m2")
    if auto_tuning and backend not in ["cutlass", "cudnn", "cublas", "cute-dsl"]:
        pytest.skip(
            "Invalid combination: auto_tuning only supported for cutlass, cudnn, "
            "cublas, and cute-dsl"
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
