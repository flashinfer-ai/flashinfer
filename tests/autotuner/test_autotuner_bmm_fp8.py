import pytest
import torch

from flashinfer import autotune, bmm_fp8
from flashinfer.autotuner import AutoTuner
from flashinfer.gemm.gemm_base import (
    _FP8_GEMM_SM100_TUNING_CONFIG,
    _cudnn_gemm_fp8_runner,
    _get_cache_buf,
    DEFAULT_WORKSPACE_SIZE,
)
from flashinfer.utils import get_compute_capability
from tests.utils_fp8 import to_float8


@pytest.mark.parametrize(
    "pre_tune,tune_mode,expected_cache_hit",
    [
        (False, False, False),  # Cold inference: no cache hit
        (False, True, True),  # Tune in this call: cache hit
        (True, False, True),  # Warm cache then inference: cache hit
    ],
    ids=["cold_infer", "tune_now", "warm_then_infer"],
)
@pytest.mark.parametrize(
    "m,n,k",
    [
        # Test power-of-2 dimensions
        (128, 64, 256),
        (2048, 256, 512),
        # Test non power-of-2 dimensions
        (48, 80, 64),
        (200, 2048, 200),
    ],
)
def test_autotuner_gemm(pre_tune, tune_mode, expected_cache_hit, m, n, k):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    compute_capability_number = compute_capability[0] * 10 + compute_capability[1]
    if not bmm_fp8.is_compute_capability_supported(compute_capability_number):
        pytest.skip(
            f"bmm_fp8 not supported on current compute capability. "
            f"Detected sm{compute_capability_number}."
        )
    if not bmm_fp8.is_backend_supported("cudnn", compute_capability_number):
        pytest.skip("cudnn backend not supported on current compute capability.")

    autotuner = AutoTuner.get()
    # Keep each test independent from other parametrized runs.
    autotuner.clear_cache()

    input_dtype = torch.float8_e4m3fn
    res_dtype = torch.bfloat16
    b = 1

    input = torch.randn([b, m, k], device="cuda", dtype=torch.bfloat16)
    input_fp8, input_inv_s = to_float8(input, dtype=input_dtype)

    # mat2 row  major -> column major
    mat2 = torch.randn([b, n, k], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=input_dtype)

    # Output tensor
    res = torch.empty([b, m, n], device="cuda", dtype=res_dtype)

    if pre_tune:
        with autotune(tune_mode=True):
            bmm_fp8(
                input_fp8,
                mat2_fp8,
                input_inv_s,
                mat2_inv_s,
                res_dtype,
                res,
                backend="cudnn",
            )

    with autotune(tune_mode=tune_mode):
        # Using bmm_fp8 because it supports compute capability 89+.
        # This test can run on various compute capabilities.
        bmm_fp8(
            input_fp8,
            mat2_fp8,
            input_inv_s,
            mat2_inv_s,
            res_dtype,
            res,
            backend="cudnn",
        )

    assert res.isfinite().all()

    # bmm_fp8 is tuned through fp8_gemm_sm100 and uses these inputs internally.
    workspace_buffer = _get_cache_buf(
        "bmm_fp8_workspace", DEFAULT_WORKSPACE_SIZE, input_fp8.device
    )
    cache_inputs = [
        input_fp8,
        mat2_fp8,
        input_inv_s,
        mat2_inv_s,
        res,
        workspace_buffer,
    ]
    is_cache_hit, runner_id, tactic, stored_profile = autotuner.search_cache(
        "fp8_gemm",
        [_cudnn_gemm_fp8_runner()],
        (
            input_fp8.shape,
            mat2_fp8.shape,
            input_inv_s.shape,
            mat2_inv_s.shape,
            res.shape,
            workspace_buffer.shape,
        ),
        _FP8_GEMM_SM100_TUNING_CONFIG,
        inputs=cache_inputs,
    )

    assert is_cache_hit == expected_cache_hit
    if is_cache_hit:
        assert runner_id == 0
        # cuDNN FP8 runner now enumerates multiple execution plans, so the
        # autotuner can pick any valid plan index (>= 0). Tactic == -1 would
        # mean the fallback path was chosen (no hit), which contradicts the hit.
        assert tactic >= 0
        assert stored_profile is not None
