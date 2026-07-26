import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, bmm_fp8
from flashinfer.utils import get_compute_capability
from tests.utils_fp8 import to_float8


@pytest.fixture
def bmm_fp8_inputs():
    input_fp8 = torch.randn((1, 16, 64), device="cuda", dtype=torch.float16).to(
        torch.float8_e4m3fn
    )
    mat2_fp8 = (
        torch.randn((1, 64, 64), device="cuda", dtype=torch.float16)
        .to(torch.float8_e4m3fn)
        .transpose(-2, -1)
    )
    scale = torch.ones((), device="cuda", dtype=torch.float32)
    return input_fp8, mat2_fp8, scale


@pytest.mark.parametrize("scale_name", ["A_scale", "B_scale"])
@pytest.mark.parametrize(
    ("invalid_scale", "error"),
    [
        (
            lambda: torch.ones((1, 2, 1), device="cuda", dtype=torch.float32),
            "must contain exactly one tensorwide scale value",
        ),
        (
            lambda: torch.ones((), device="cuda", dtype=torch.float16),
            "must be a float32 tensor",
        ),
        (
            lambda: torch.ones((), device="cpu", dtype=torch.float32),
            "must be on the same device",
        ),
    ],
)
def test_bmm_fp8_rejects_invalid_tensorwide_scale(
    bmm_fp8_inputs, scale_name, invalid_scale, error
):
    input_fp8, mat2_fp8, valid_scale = bmm_fp8_inputs
    scales = {"A_scale": valid_scale, "B_scale": valid_scale}
    scales[scale_name] = invalid_scale()

    with pytest.raises(ValueError, match=error):
        bmm_fp8(
            input_fp8,
            mat2_fp8,
            scales["A_scale"],
            scales["B_scale"],
            torch.bfloat16,
            backend="cublas",
        )


@pytest.mark.parametrize(
    ("invalid_scale", "error"),
    [
        (
            lambda: torch.ones((2,), device="cuda", dtype=torch.float32),
            "must contain exactly one tensorwide scale value",
        ),
        (
            lambda: torch.ones((), device="cuda", dtype=torch.float16),
            "must be float32",
        ),
        (
            lambda: torch.ones((), device="cpu", dtype=torch.float32),
            "must be a CUDA tensor",
        ),
    ],
)
def test_bmm_fp8_cublas_ffi_rejects_invalid_tensorwide_scale(
    bmm_fp8_inputs, invalid_scale, error
):
    input_fp8, mat2_fp8, valid_scale = bmm_fp8_inputs

    with pytest.raises(RuntimeError, match=error):
        bmm_fp8(
            input_fp8,
            mat2_fp8,
            invalid_scale(),
            valid_scale,
            torch.bfloat16,
            backend="cublas",
            skip_check=True,
        )


@pytest.mark.parametrize("shape", [(), (1,), (1, 1, 1)])
def test_bmm_fp8_accepts_single_value_scale_shapes(bmm_fp8_inputs, shape):
    input_fp8, mat2_fp8, _ = bmm_fp8_inputs
    scale = torch.ones(shape, device="cuda", dtype=torch.float32)

    result = bmm_fp8(
        input_fp8,
        mat2_fp8,
        scale,
        scale,
        torch.bfloat16,
        backend="cublas",
    )

    assert result.shape == (1, 16, 64)


@pytest.mark.parametrize("b", [1, 16])
@pytest.mark.parametrize("m", [1, 48, 128])
@pytest.mark.parametrize("n", [64, 80, 10304])
@pytest.mark.parametrize("k", [64, 256, 2688])
@pytest.mark.parametrize("input_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("mat2_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["cudnn", "cublas", "cutlass", "auto"])
@pytest.mark.parametrize("auto_tuning", [True, False])
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
