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


@pytest.mark.parametrize(
    "b, m, n, k, layout",
    [
        (1, 1, 10304, 2688, "tma"),  # decode M, N not a multiple of 128
        (2, 300, 4096, 1024, "tma"),  # prefill M, batched
        (1, 4, 2048, 1000, "fallback"),  # K % 16 != 0
        (1, 4, 2044, 1024, "fallback"),  # N % 8 != 0
    ],
)
def test_bmm_fp8_cutlass_sm120_tactics(b, m, n, k, layout):
    # Regression anchor for the SM120 CUTLASS runner: every tactic of the per-tensor
    # kernel, and the groupwise fallback for inputs that kernel cannot take. The
    # fuzzer only runs the default tactic and the autotune winner.
    if get_compute_capability(torch.device("cuda"))[0] != 12:
        pytest.skip("SM120/SM121 only")
    from flashinfer.gemm.gemm_base import (
        DEFAULT_WORKSPACE_SIZE,
        _get_cache_buf,
        _sm120_fp8_gemm_tma_path_supported,
        get_gemm_sm120_module_cutlass_fp8,
    )

    torch.manual_seed(0)
    input = torch.randn([b, m, k], device="cuda", dtype=torch.bfloat16)
    input_fp8, input_inv_s = to_float8(input, dtype=torch.float8_e4m3fn)
    mat2 = torch.randn([b, n, k], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=torch.float8_e4m3fn)
    reference = torch.bmm(
        input_fp8.to(torch.bfloat16), mat2_fp8.to(torch.bfloat16)
    ).float() * (input_inv_s * mat2_inv_s)

    res = torch.empty([b, m, n], device="cuda", dtype=torch.bfloat16)
    workspace = _get_cache_buf("bmm_fp8_workspace", DEFAULT_WORKSPACE_SIZE, res.device)
    inputs = [input_fp8, mat2_fp8, input_inv_s, mat2_inv_s, res, workspace]
    assert _sm120_fp8_gemm_tma_path_supported(*inputs[:5]) == (layout == "tma")

    runner = get_gemm_sm120_module_cutlass_fp8().cutlass_fp8_gemm_runner()
    tactics = runner.get_valid_tactics(inputs, None)
    assert len(tactics) > 1 if layout == "tma" else tactics == [-1]
    for tactic in [-1] + tactics:
        res.fill_(float("nan"))
        runner.forward(inputs, tactic=tactic)
        torch.testing.assert_close(res.float(), reference, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    pytest.main([__file__])
