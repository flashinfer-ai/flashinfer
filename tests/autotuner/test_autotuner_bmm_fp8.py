import pytest
import torch

from flashinfer import autotune, bmm_fp8
from flashinfer.autotuner import AutoTuner
from flashinfer.gemm.gemm_base import (
    _FP8_GEMM_SM100_TUNING_CONFIG,
    _cudnn_gemm_fp8_runner,
    _get_cache_buf,
    get_gemm_module,
    DEFAULT_WORKSPACE_SIZE,
)
from flashinfer.gemm import gemm_base
from flashinfer.utils import get_compute_capability
from tests.utils_fp8 import to_float8


def _assert_engine_knob_tactic(tactic):
    """cuDNN tactics are stable (engine_id, knob_items) descriptors, not plan indices."""
    engine_id, knob_items = tactic
    assert isinstance(engine_id, int)
    assert all(len(item) == 2 for item in knob_items)


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
        _assert_engine_knob_tactic(tactic)
        assert stored_profile is not None


@pytest.mark.parametrize("backend", ["cudnn", "cublas"])
def test_autotuner_gemm_cross_bucket_m(backend):
    """Tune at one M bucket, then run inference at a non-bucket M.

    The cuBLASLt algo list is enumerated at the *real* shape, so an integer
    tactic tuned against a different (bucketed) M may be out of range at
    runtime. cuDNN tactics are stable engine/knob descriptors and are resolved
    against the runtime graph. Both paths must reuse the tuned bucket entry for
    the non-bucket M (i.e. the workspace scratch size does not leak into the
    cache key) and produce finite output.
    """
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    cc = compute_capability[0] * 10 + compute_capability[1]
    if not bmm_fp8.is_compute_capability_supported(cc):
        pytest.skip(f"bmm_fp8 not supported on sm{cc}.")
    if not bmm_fp8.is_backend_supported(backend, cc):
        pytest.skip(f"{backend} backend not supported on sm{cc}.")

    autotuner = AutoTuner.get()
    autotuner.clear_cache()

    input_dtype = torch.float8_e4m3fn
    res_dtype = torch.bfloat16
    b, n, k = 1, 64, 256
    # Tuning at M=256 profiles all buckets up to 256 (incl. 128). A runtime
    # M=100 rounds up to bucket 128, so real-M (100) != bucket-M (128).
    tune_m, run_m = 256, 100

    def make(m):
        inp = torch.randn([b, m, k], device="cuda", dtype=torch.bfloat16)
        inp_fp8, inp_s = to_float8(inp, dtype=input_dtype)
        mat2 = torch.randn([b, n, k], device="cuda", dtype=torch.bfloat16).transpose(
            -2, -1
        )
        mat2_fp8, mat2_s = to_float8(mat2, dtype=input_dtype)
        return inp_fp8, mat2_fp8, inp_s, mat2_s

    # 1) Tune over the bucket range.
    a8, b8, a_s, b_s = make(tune_m)
    with autotune(tune_mode=True):
        bmm_fp8(a8, b8, a_s, b_s, res_dtype, backend=backend)

    # 2) Run at a non-bucket M. Must not raise (out-of-range tactic is clamped)
    #    and must produce finite output.
    a8, b8, a_s, b_s = make(run_m)
    res = bmm_fp8(a8, b8, a_s, b_s, res_dtype, backend=backend)
    assert res.isfinite().all()

    # 3) The tuned bucket entry must be reused for the non-bucket M.
    runner = (
        _cudnn_gemm_fp8_runner()
        if backend == "cudnn"
        else get_gemm_module().cublas_fp8_gemm_runner()
    )
    workspace_buffer = _get_cache_buf(
        "bmm_fp8_workspace", DEFAULT_WORKSPACE_SIZE, a8.device
    )
    is_cache_hit, runner_id, tactic, stored_profile = autotuner.search_cache(
        "fp8_gemm",
        [runner],
        (
            a8.shape,
            b8.shape,
            a_s.shape,
            b_s.shape,
            res.shape,
            workspace_buffer.shape,
        ),
        _FP8_GEMM_SM100_TUNING_CONFIG,
        inputs=[a8, b8, a_s, b_s, res, workspace_buffer],
    )
    assert is_cache_hit
    if backend == "cudnn":
        _assert_engine_knob_tactic(tactic)
    else:
        assert tactic >= 0
    assert stored_profile is not None


def test_bmm_fp8_heuristic_gates_cudnn_on_sm12x_without_override_shape(monkeypatch):
    """On SM12x, cuDNN must be excluded from the ``auto`` candidate list when
    ``override_shape`` is unavailable (cuDNN backend < 9.23.1): the cuDNN FP8
    path pays an unbounded per-shape ``policy=ALL`` host-side graph build at
    serving time and surfaces an async CUDA fault past the tactic-level
    fallback (#3707 only catches synchronous exceptions). When
    ``override_shape`` is available, M is bucketed and the build is amortized,
    so cuDNN stays a candidate. See RFC #3920 (runner-contract rule 6).

    The SM version and cuDNN availability are monkeypatched so this runs on
    any host without SM12x hardware or a cuDNN install.
    """
    # _match_sm_version(device, versions) -> True only for the SM12x list,
    # so is_sm_supported (SM10x) is False and is_sm120_supported is True.
    monkeypatch.setattr(gemm_base, "_match_sm_version", lambda dev, vers: "120" in vers)

    a = torch.empty(1, 16, 32, dtype=torch.bfloat16)
    b = torch.empty(1, 64, 32, dtype=torch.bfloat16)

    def call(override_shape_available):
        monkeypatch.setattr(
            gemm_base, "CUDNN_AVAILABLE", True
        )  # exercise the cudnn branch
        monkeypatch.setattr(
            gemm_base,
            "_is_cudnn_override_shape_available",
            lambda: override_shape_available,
        )
        return gemm_base._heuristic_func_bmm_fp8(
            ["cudnn", "cublas", "cutlass"], a, b, a, b, torch.bfloat16, None, "auto"
        )

    # override_shape unavailable -> cuDNN excluded on SM12x.
    assert "cudnn" not in call(override_shape_available=False)

    # override_shape available -> cuDNN retained.
    assert "cudnn" in call(override_shape_available=True)
