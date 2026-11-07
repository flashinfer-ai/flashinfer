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
    # cute-dsl (SM107 only).  Every entry in SM107_AUTOTUNE_CONFIGS is 2-CTA with
    # mma_tiler M=256, so the CTA tile is 256x128 and the problem must have
    # m >= 256 and n >= 128 to fill one; smaller shapes have no valid config.
    (
        1,
        256,
        10304,
        2688,
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
        torch.bfloat16,
        "cute-dsl",
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
    if backend == "cute-dsl":
        if compute_capability != (10, 7):
            pytest.skip(
                "bmm_fp8 with cute-dsl backend is only supported on SM107 GPUs."
            )
        # Imported here, not at module scope: on this branch the device is
        # SM107 and the cute-dsl backend was explicitly requested, so the DSL is
        # expected present. cute_dsl.utils imports cutlass at module scope and
        # re-exports this probe both before and after #4753.
        from flashinfer.cute_dsl.utils import is_rubin_cute_dsl_available

        if not is_rubin_cute_dsl_available():
            pytest.skip(
                "bmm_fp8 with cute-dsl backend requires CuTe DSL >= 4.8 "
                "(cutlass.utils.rubin_helpers)."
            )
        if m % 16 != 0 or n % 16 != 0 or k % 16 != 0:
            pytest.skip(
                "bmm_fp8 with cute-dsl backend requires m, n, k to be multiples of 16."
            )
        if input_dtype != mat2_dtype:
            pytest.skip(
                "bmm_fp8 with cute-dsl backend requires A and B to have the same dtype."
            )
        if m < 256 or n < 128:
            pytest.skip(
                "bmm_fp8 with cute-dsl backend requires m >= 256 and n >= 128 "
                "(2-CTA tile); smaller problems have no valid SM107 config."
            )
    if input_dtype == torch.float8_e5m2 and mat2_dtype == torch.float8_e5m2:
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


def _bias_case(
    res_dtype: torch.dtype, b: int = 2, m: int = 48
) -> tuple[torch.Tensor, ...]:
    major, minor = get_compute_capability(torch.device("cuda"))
    if not bmm_fp8.is_backend_supported("cublas", major * 10 + minor):
        pytest.skip("bmm_fp8 takes a bias only with the cublas backend (SM89+).")
    n, k = 80, 256
    input_fp8, input_inv_s = to_float8(
        torch.randn([b, m, k], device="cuda", dtype=torch.bfloat16)
    )
    # Scaled so the output is O(1) like the bias; column major as bmm_fp8 requires.
    mat2 = torch.randn([b, n, k], device="cuda", dtype=torch.bfloat16) / k**0.5
    mat2_fp8, mat2_inv_s = to_float8(mat2.transpose(-2, -1))
    bias = torch.randn(n, device="cuda", dtype=res_dtype)
    return input_fp8, mat2_fp8, input_inv_s, mat2_inv_s, bias


# The fuzzer has no bias axis, so the cuBLASLt bias epilogue is anchored here.
# Batch 1 has no batch stride, and M=1 is a decode step.
@pytest.mark.parametrize(("b", "m"), [(2, 48), (1, 1)])
@pytest.mark.parametrize("auto_tuning", [False, True])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["cublas", "auto"])
def test_bmm_fp8_bias(
    backend: str, res_dtype: torch.dtype, auto_tuning: bool, b: int, m: int
) -> None:
    input_fp8, mat2_fp8, input_inv_s, mat2_inv_s, bias = _bias_case(res_dtype, b, m)
    with autotune(auto_tuning):
        res = bmm_fp8(
            input_fp8,
            mat2_fp8,
            input_inv_s,
            mat2_inv_s,
            res_dtype,
            backend=backend,
            bias=bias,
        )
    no_bias = bmm_fp8(input_fp8, mat2_fp8, input_inv_s, mat2_inv_s, res_dtype)
    # Added once per output column of every batch.
    torch.testing.assert_close(
        res.float(), no_bias.float() + bias.float(), atol=0.1, rtol=0.02
    )


def test_bmm_fp8_bias_rejected() -> None:
    input_fp8, mat2_fp8, input_inv_s, mat2_inv_s, bias = _bias_case(torch.bfloat16)
    args = (input_fp8, mat2_fp8, input_inv_s, mat2_inv_s, torch.bfloat16)
    # A backend without the bias epilogue refuses the bias instead of dropping it.
    with pytest.raises(ValueError, match="cublas"):
        bmm_fp8(*args, backend="cudnn", bias=bias)
    # The batch-invariant class replaces the backends and takes no bias,
    # also when skip_check=True skips the problem-size check.
    for skip_check in (False, True):
        with pytest.raises(ValueError, match="batch_invariant"):
            bmm_fp8(
                *args,
                backend="auto",
                batch_invariant=True,
                bias=bias,
                skip_check=skip_check,
            )
    # One value per output column, in the output dtype.
    for wrong in (
        torch.randn(bias.numel() + 8, device="cuda").to(bias.dtype),
        bias.float(),
    ):
        with pytest.raises(ValueError, match="bias must have shape"):
            bmm_fp8(*args, bias=wrong)
    # A bias 2 bytes past an allocation has no cuBLASLt algo, so it is refused upfront.
    misaligned = torch.empty(bias.numel() + 1, device="cuda", dtype=bias.dtype)[1:]
    with pytest.raises(ValueError, match="aligned"):
        bmm_fp8(*args, bias=misaligned)


if __name__ == "__main__":
    pytest.main([__file__])
