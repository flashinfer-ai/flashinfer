# NOTE for future contributors (incl. AI agents): keep this file a SMALL curated
# smoke set. New coverage (shapes, dtypes, backends, randomized breadth) belongs in
# tests/gemm/test_unified_gemm_fuzz.py -- extend an adapter/axis there. Add cases
# here only as deliberate regression anchors or for paths the fuzzer cannot express.

import functools
import inspect
import typing

import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, mm_bf16
from flashinfer.autotuner import AutoTuner
from flashinfer.gemm.gemm_base import CUDNN_AVAILABLE
from flashinfer.gemm import is_cuda_tile_available
from flashinfer.utils import get_compute_capability, is_sm100a_supported


# Curated smoke set. Randomized breadth over {m,n,k} x out-dtype x backend (tight
# elementwise oracle, determinism, autotune-winner validation) lives in
# tests/gemm/test_unified_gemm_fuzz.py's mm_bf16 adapter. The bias and pdl epilogue
# axes are NOT fuzzed -- each supported backend keeps a bias=True and pdl=True case
# here; plus one plain case per backend / out dtype / autotune mode.
_SMOKE_CASES = [
    # m, n, k, res_dtype, enable_bias, pdl, backend, auto_tuning
    (1, 1024, 2048, torch.bfloat16, True, False, "cudnn", False),
    (16, 4096, 1024, torch.float16, False, True, "cudnn", True),
    (64, 2048, 3072, torch.float32, True, True, "cudnn", False),
    (8, 1024, 1024, torch.bfloat16, False, False, "cutlass", True),
    (32, 4096, 3072, torch.float32, False, False, "cutlass", False),
    (1, 2048, 1024, torch.bfloat16, True, True, "tgv", False),
    (64, 1024, 2048, torch.bfloat16, False, True, "tgv", True),
    (16, 2048, 2048, torch.bfloat16, False, False, "cublaslt", True),
    (16, 2048, 2048, torch.bfloat16, True, False, "cublaslt", True),
    (8, 4096, 3072, torch.float16, False, False, "cublaslt", False),
    (1, 1024, 3072, torch.bfloat16, True, False, "tinygemm", False),
    (32, 1024, 1024, torch.bfloat16, False, False, "cutile", False),
    (25, 2048, 1024, torch.bfloat16, False, False, "cute-dsl", False),
    # Warp Split-K: tuned low-M with bias and PDL; M > 16 (32-token tile, tail rows)
    # on the default tactic; K above its tile bound (served by the other runners).
    (8, 768, 7168, torch.bfloat16, True, True, "cute-dsl", True),
    (17, 2304, 1536, torch.bfloat16, False, False, "cute-dsl", False),
    (8, 768, 8320, torch.bfloat16, False, False, "cute-dsl", False),
    (64, 4096, 2048, torch.bfloat16, False, False, "auto", True),
    (16, 2048, 2048, torch.bfloat16, False, True, "auto", True),
    (1, 2048, 3072, torch.float16, False, False, "auto", False),
]


@pytest.mark.parametrize(
    "m,n,k,res_dtype,enable_bias,pdl,backend,auto_tuning", _SMOKE_CASES
)
def test_mm_bf16(
    m: int,
    n: int,
    k: int,
    res_dtype: torch.dtype,
    enable_bias: bool,
    pdl: bool,
    backend: str,
    auto_tuning: bool,
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    compute_capability_number = compute_capability[0] * 10 + compute_capability[1]
    if not mm_bf16.is_compute_capability_supported(compute_capability_number):
        pytest.skip(
            f"mm_bf16 not supported on current compute capability."
            f"Detected sm{compute_capability_number}."
        )
    if backend != "auto":
        if not mm_bf16.is_backend_supported(backend, compute_capability_number):
            pytest.skip(
                f"{backend} backend not supported on current compute capability."
            )

    if backend == "cudnn" and not CUDNN_AVAILABLE:
        pytest.skip("cuDNN is not available on this system.")

    if backend == "cutile":
        if not is_cuda_tile_available():
            pytest.skip(
                "cuda-tile / tileiras compiler not available in this environment."
            )

    if backend == "cute-dsl":
        if not is_sm100a_supported(torch.device("cuda")):
            pytest.skip("CuTeDSL low-M backend requires SM100/SM103 with CUDA 12.8+.")

        from flashinfer.cute_dsl.utils import is_cute_dsl_available

        if not is_cute_dsl_available():
            pytest.skip("nvidia-cutlass-dsl is not available.")
        if res_dtype != torch.bfloat16:
            pytest.skip("CuTeDSL low-M backend requires BF16 output.")

    if backend == "cutlass" and enable_bias:
        pytest.skip("mm_bf16 with CUTLASS backend does not support bias arguments.")
    if backend == "cutile" and (enable_bias or pdl):
        pytest.skip(
            "mm_bf16 with cuTile backend does not support bias or pdl arguments."
        )
    if res_dtype != torch.bfloat16 and backend == "tgv":
        pytest.skip(
            "mm_bf16 with TGV backend does not support specifying non-bfloat16 result dtypes."
        )
    if res_dtype != torch.bfloat16 and backend == "tinygemm":
        pytest.skip(
            "mm_bf16 with TinyGEMM backend does not support specifying non-bfloat16 result dtypes."
        )
    # cuDNN on SM103 does not support bf16 input -> fp16 output
    if (
        backend == "cudnn"
        and compute_capability_number == 103
        and res_dtype == torch.float16
    ):
        pytest.skip("cuDNN bf16 GEMM with fp16 output not supported on SM103.")

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
    with autotune(auto_tuning):
        mm_bf16(input, mat2.T, bias, pdl, out, res_dtype, backend)

    cos_sim = F.cosine_similarity(reference.reshape(-1), out.reshape(-1), dim=0)
    assert cos_sim > 0.99
    if backend == "cute-dsl":
        fp32_reference = F.linear(
            input.float(), mat2.float(), bias.float() if bias is not None else None
        ).to(res_dtype)
        torch.testing.assert_close(out, fp32_reference, rtol=2e-2, atol=5e-2)


@pytest.mark.parametrize("backend", ["auto", "cudnn"])
@pytest.mark.parametrize("padded", [False, True])
def test_mm_bf16_autotune_weight_layouts(monkeypatch, backend, padded):
    """Autotuning stages dense and padded weights in the cold-chain arena."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("The cold-chain BF16 tuning config is used on SM100-family.")
    if backend == "cudnn" and not CUDNN_AVAILABLE:
        pytest.skip("cuDNN is not available on this system.")

    monkeypatch.setattr(AutoTuner, "_instance", AutoTuner())
    torch.manual_seed(0)
    a = torch.randn(16, 512, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(256, 640 if padded else 512, device="cuda", dtype=torch.bfloat16)
    b = w[:, :512].T
    with autotune(True, tuning_buckets=[16]):
        out = mm_bf16(a, b, backend=backend)
    torch.testing.assert_close(out, a @ b, rtol=2e-2, atol=5e-2)


@pytest.mark.parametrize("n,k", [(1, 2560), (12, 5120)])
def test_mm_bf16_autotune_unaligned_n(monkeypatch, n, k):
    """Tuning auto must skip CUTLASS, not abort, when N breaks its 16-byte rows.

    N=1 is an MoE shared_expert_gate, N=12 a linear-attention in_proj_ba at TP8.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("The CUTLASS BF16 backend is used on SM100-family.")

    monkeypatch.setattr(AutoTuner, "_instance", AutoTuner())

    # Correctness alone does not pin this down: the tuner already caught
    # CUTLASS's host-side can_implement rejection and moved on, so the result
    # was right either way. Assert the shape never reaches CUTLASS at all.
    profiled = set()
    profile_single_kernel = AutoTuner._profile_single_kernel

    def record_runner(self, runner, *args, **kwargs):
        profiled.add(type(runner).__name__)
        return profile_single_kernel(self, runner, *args, **kwargs)

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", record_runner)

    torch.manual_seed(0)
    a = torch.randn(64, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    with autotune(True, tuning_buckets=[1, 32, 64]):
        mm_bf16(a, w.T, backend="auto")

    assert profiled, "autotuning profiled nothing, so the assertion below is vacuous"
    assert "CutlassBf16GemmRunner" not in profiled, profiled

    for m in (1, 32, 64):
        out = mm_bf16(a[:m], w.T, backend="auto")
        reference = (a[:m].float() @ w.float().T).to(torch.bfloat16)
        torch.testing.assert_close(out, reference, rtol=2e-2, atol=5e-2)


@pytest.mark.parametrize("pdl", [False, True])
def test_mm_bf16_tgv_single_element_output(pdl):
    """M = N = 1: the 1x1 output's batch dim also has stride 1."""
    compute_capability = get_compute_capability(torch.device("cuda"))
    cc_num = compute_capability[0] * 10 + compute_capability[1]
    if not mm_bf16.is_backend_supported("tgv", cc_num):
        pytest.skip("TGV backend not supported on current compute capability.")

    torch.manual_seed(0)
    a = torch.randn(1, 2560, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1, 2560, device="cuda", dtype=torch.bfloat16)
    out = mm_bf16(a, w.T, pdl=pdl, backend="tgv")
    reference = (a.float() @ w.float().T).to(torch.bfloat16)
    torch.testing.assert_close(out, reference, rtol=2e-2, atol=5e-2)


@pytest.mark.parametrize(
    "n,k,out_dtype,supported",
    [
        (12, 2560, torch.bfloat16, False),
        (16, 2564, torch.bfloat16, False),
        (12, 2560, torch.float32, True),
    ],
)
def test_mm_bf16_cutlass_alignment(n, k, out_dtype, supported):
    """CUTLASS needs K % 8 == 0 and 16-byte output rows; reject others up front."""
    compute_capability = get_compute_capability(torch.device("cuda"))
    cc_num = compute_capability[0] * 10 + compute_capability[1]
    if not mm_bf16.is_backend_supported("cutlass", cc_num):
        pytest.skip("cutlass backend not supported on current compute capability.")

    torch.manual_seed(0)
    a = torch.randn(16, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    if not supported:
        with pytest.raises(ValueError, match="CUTLASS backend requires K divisible"):
            mm_bf16(a, w.T, out_dtype=out_dtype, backend="cutlass")
        return
    out = mm_bf16(a, w.T, out_dtype=out_dtype, backend="cutlass")
    torch.testing.assert_close(out, a.float() @ w.float().T, rtol=2e-2, atol=5e-2)


def test_mm_bf16_cutile_rejects_bias_and_pdl():
    """The v1 cuTile path is alpha=1/beta=0 and ignores bias / pdl — must raise.

    Output dtype, on the other hand, supports bf16 / fp16 / fp32 via the
    polymorphic store epilogue, so it is exercised in the main parametrized
    matrix above rather than rejected here.
    """
    compute_capability = get_compute_capability(torch.device("cuda"))
    cc_num = compute_capability[0] * 10 + compute_capability[1]
    if not mm_bf16.is_backend_supported("cutile", cc_num):
        pytest.skip("cuTile backend not supported on current compute capability.")
    if not is_cuda_tile_available():
        pytest.skip("cuda-tile / tileiras compiler not available in this environment.")

    # a is (m, k) = (64, 1024); b is (n, k) = (2048, 1024); b.T is (k, n) = (1024, 2048)
    a = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(64, 2048, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="ignores `bias`"):
        mm_bf16(a, b.T, bias, False, out, torch.bfloat16, backend="cutile")
    with pytest.raises(ValueError, match="ignores `pdl`"):
        mm_bf16(a, b.T, None, True, out, torch.bfloat16, backend="cutile")


def test_mm_bf16_cutile_repeat_uses_tune_cache():
    """Second call on the same shape must reuse the cached autotune result.

    Cache hits reproduce the first call's output bit-for-bit (no kernel
    re-tune, no autotune RNG state to diverge). We use ``assert_close`` with
    rtol=atol=0 — a strict-inequality threshold like ``cos_sim > 0.999`` is
    unsafe in bf16 because the rhs rounds to 1.0 and ``1.0 > 1.0`` is False
    even when the kernel is correct.
    """
    compute_capability = get_compute_capability(torch.device("cuda"))
    cc_num = compute_capability[0] * 10 + compute_capability[1]
    if not mm_bf16.is_backend_supported("cutile", cc_num):
        pytest.skip("cuTile backend not supported on current compute capability.")
    if not is_cuda_tile_available():
        pytest.skip("cuda-tile / tileiras compiler not available in this environment.")

    # Shape (m, n, k) = (64, 2048, 1024). mm_bf16(a, b.T) → (m, n).
    a = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(64, 2048, device="cuda", dtype=torch.bfloat16)

    # First call: warms tune cache.
    mm_bf16(a, b.T, None, False, out, torch.bfloat16, backend="cutile")
    out_first = out.clone()
    # Second call: must produce the same result and not raise.
    mm_bf16(a, b.T, None, False, out, torch.bfloat16, backend="cutile")
    torch.testing.assert_close(
        out_first,
        out,
        rtol=0,
        atol=0,
        msg="tune-cache reuse produced divergent output",
    )


def test_cublaslt_bf16_runner_zero_algos(monkeypatch):
    """No candidate algorithms means nothing to profile; the default still runs."""
    from flashinfer.gemm.gemm_base import get_mm_bf16_cublaslt_module
    from flashinfer.utils import get_compute_capability

    compute_capability = get_compute_capability(torch.device("cuda"))
    cc_num = compute_capability[0] * 10 + compute_capability[1]
    if not mm_bf16.is_backend_supported("cublaslt", cc_num):
        pytest.skip("cublaslt backend not supported on this GPU")

    runner = get_mm_bf16_cublaslt_module().cublaslt_bf16_gemm_runner()

    m, n, k = 16, 1024, 1024
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    inputs = [a, b, None, None, out, workspace]

    # The runner is memoized; monkeypatch so the stub does not outlive the test.
    monkeypatch.setattr(runner, "_get_algos", lambda _inputs: [], raising=False)
    assert runner.get_valid_tactics(inputs, None) == []
    runner.forward(inputs)
    cos_sim = F.cosine_similarity((a @ b).reshape(-1), out.reshape(-1), dim=0)
    assert cos_sim > 0.99


@pytest.mark.parametrize("pdl", [False, True])
def test_mm_bf16_cute_dsl_direct_fallback(pdl: bool):
    """The no-autotune fallback uses direct in its measured crossover band."""
    if get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("CuTeDSL low-M backend requires SM100/SM103/SM107.")

    from flashinfer.cute_dsl.utils import is_cute_dsl_available

    if not is_cute_dsl_available():
        pytest.skip("nvidia-cutlass-dsl is not available.")

    m, n, k = 1, 256, 8192
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    out = mm_bf16(a, b.T, pdl=pdl, backend="cute-dsl")
    reference = a @ b.T
    cos_sim = F.cosine_similarity(reference.reshape(-1), out.reshape(-1), dim=0)
    assert cos_sim > 0.99


@pytest.mark.parametrize("enable_bias", [False, True])
def test_mm_bf16_cute_dsl_one_stage_splitk(enable_bias: bool):
    """K=128 exercises the shallow one-stage tensor-core pipeline."""
    if get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("CuTeDSL low-M backend requires SM100/SM103/SM107.")

    from flashinfer.cute_dsl.utils import is_cute_dsl_available

    if not is_cute_dsl_available():
        pytest.skip("nvidia-cutlass-dsl is not available.")

    m, n, k = 1, 1024, 128
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    bias = (
        torch.randn((n,), device="cuda", dtype=torch.bfloat16) if enable_bias else None
    )
    out = mm_bf16(a, b.T, bias=bias, pdl=True, backend="cute-dsl")
    reference = F.linear(a, b, bias)
    cos_sim = F.cosine_similarity(reference.reshape(-1), out.reshape(-1), dim=0)
    assert cos_sim > 0.99


@pytest.mark.parametrize("pdl", [False, True])
def test_mm_bf16_cute_dsl_low_m_only(monkeypatch, pdl):
    """backend="cute-dsl" serves M <= 32 only. Under "auto", a larger M still
    lists its low-M runners (one large-M warm-up tunes the M <= 32 buckets)
    and cuBLASLt only once, as its own backend."""
    if get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("CuTeDSL low-M backend requires SM100/SM103/SM107.")

    from flashinfer.cute_dsl.utils import is_cute_dsl_available
    from flashinfer.gemm import gemm_base

    if not is_cute_dsl_available():
        pytest.skip("nvidia-cutlass-dsl is not available.")

    a = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16).T
    with pytest.raises(ValueError, match="M <= 32"):
        mm_bf16(a, b, pdl=pdl, backend="cute-dsl")

    choose_one = AutoTuner.choose_one
    runner_lists = []

    def record(self, custom_op, runners, *args, **kwargs):
        if custom_op.startswith("bf16_gemm"):
            runner_lists.append(runners)
        return choose_one(self, custom_op, runners, *args, **kwargs)

    monkeypatch.setattr(AutoTuner, "choose_one", record)
    out = mm_bf16(a, b, pdl=pdl, backend="auto")
    (runners,) = runner_lists
    names = [type(runner).__name__ for runner in runners]
    assert sum("Cublaslt" in name for name in names) == 1, names
    assert any(isinstance(r, gemm_base._CuteDSLBf16Runner) for r in runners), names
    torch.testing.assert_close(out, a @ b, rtol=2e-2, atol=5e-2)


# What pdl=True does with a backend the caller names explicitly: "pdl" launches
# with PDL, "no-pdl" runs without it (cuDNN has no PDL launch), "raises" rejects
# it. Every backend runs with pdl=False.
_EXPLICIT_PDL_TRUE = {
    "cudnn": "no-pdl",
    "cutlass": "raises",
    "tgv": "pdl",
    "cublaslt": "raises",
    "tinygemm": "pdl",
    "cutile": "raises",
    "cute-dsl": "pdl",
}
# backend="auto" lists every backend except cuTile, and CUTLASS only without
# bias (no bias epilogue).
_AUTO_PDL_CASES = [
    pytest.param(winner, bias, id=f"{winner}-bias" if bias else winner)
    for winner in _EXPLICIT_PDL_TRUE
    if winner != "cutile"
    for bias in (False, True)
    if not (bias and winner == "cutlass")
]


def test_mm_bf16_pdl_matrix_covers_every_backend():
    """A new mm_bf16 backend must state above what it does with pdl=True."""
    backend_hint = inspect.signature(mm_bf16).parameters["backend"].annotation
    assert set(typing.get_args(backend_hint)) == {*_EXPLICIT_PDL_TRUE, "auto"}


def _skip_unless_supported(backend: str) -> None:
    major, minor = get_compute_capability(torch.device("cuda"))
    if not mm_bf16.is_backend_supported(backend, major * 10 + minor):
        pytest.skip(f"{backend} backend not supported on SM{major}{minor}.")


def _skip_unless_runnable(backend: str) -> None:
    _skip_unless_supported(backend)
    if backend == "cudnn" and not CUDNN_AVAILABLE:
        pytest.skip("cuDNN is not available on this system.")
    if backend == "cutile" and not is_cuda_tile_available():
        pytest.skip("cuda-tile / tileiras compiler not available in this environment.")
    if backend in ("tgv", "cute-dsl"):
        from flashinfer.cute_dsl.utils import is_cute_dsl_available

        if not is_cute_dsl_available():
            pytest.skip("nvidia-cutlass-dsl is not available.")
    if backend == "cute-dsl" and not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("CuTeDSL low-M backend requires SM100/SM103 with CUDA 12.8+.")


def _record_pdl_launches(monkeypatch, backend: str) -> list:
    """Record the pdl flag of every kernel launch of a backend with PDL."""
    from flashinfer.gemm import gemm_base

    launches = []

    def record(module, name, pdl_index):
        launch = getattr(module, name)

        def recorded(*args, **kwargs):
            launches.append(args[pdl_index])
            return launch(*args, **kwargs)

        monkeypatch.setattr(module, name, recorded)

    if backend == "tgv":
        from flashinfer.gemm.kernels import tgv_gemm_cute_ext

        record(tgv_gemm_cute_ext, "run_tgv_cute_ext", 4)
    elif backend == "tinygemm":
        tinygemm = gemm_base.get_tinygemm2_module()
        record(tinygemm, "tinygemm2_nobias_op", 3)
        record(tinygemm, "tinygemm2_op", 4)
    elif backend == "cute-dsl":
        from flashinfer.cute_dsl.availability import (
            is_cute_dsl_experimental_available,
        )
        from flashinfer.gemm.kernels import (
            dense_bf16_gemm_direct,
            dense_bf16_gemm_sm100_splitk,
        )

        record(dense_bf16_gemm_direct, "run_direct_dense", 3)
        record(dense_bf16_gemm_sm100_splitk, "run_splitk_dense", 4)
        if is_cute_dsl_experimental_available():
            from flashinfer.gemm.kernels import dense_bf16_gemm_warp_splitk

            record(dense_bf16_gemm_warp_splitk, "run_warp_splitk_dense", 3)
        # The runners bind their launchers when built: build fresh ones for
        # this test and keep the cached ones for the others.
        for name in (
            "_cute_dsl_direct_bf16_gemm_runner",
            "_cute_dsl_warp_splitk_bf16_gemm_runner",
            "_cute_dsl_splitk_bf16_gemm_runner",
        ):
            factory = getattr(gemm_base, name).__wrapped__
            monkeypatch.setattr(gemm_base, name, functools.cache(factory))
    return launches


def _runner_backend(runner) -> str:
    from flashinfer.gemm import gemm_base

    if isinstance(runner, gemm_base._CuteDSLBf16Runner):
        return "cute-dsl"
    name = type(runner).__name__
    for prefix, backend in (
        ("Cudnn", "cudnn"),
        ("Cublaslt", "cublaslt"),
        ("Cutlass", "cutlass"),
        ("TGV", "tgv"),
        ("TinyGemm", "tinygemm"),
    ):
        if name.startswith(prefix):
            return backend
    raise AssertionError(f"Unknown mm_bf16 runner {name}.")


def _pdl_problem(bias: bool = False):
    # Every backend serves this shape: M <= 32 (CuTeDSL), N % 16 == 0 (TinyGEMM).
    torch.manual_seed(0)
    a = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16).T
    c = torch.randn(2048, device="cuda", dtype=torch.bfloat16) if bias else None
    reference = a.float() @ b.float()
    if c is not None:
        reference += c.float()
    return a, b, c, reference.to(torch.bfloat16)


# None (accepted since #8032, shipped in 0.7.4rc1) must behave as False.
_PDL_VALUES = [False, True, None]


@pytest.mark.parametrize("pdl", _PDL_VALUES)
@pytest.mark.parametrize("backend", list(_EXPLICIT_PDL_TRUE))
def test_mm_bf16_explicit_backend_pdl(monkeypatch, backend, pdl):
    """A backend named explicitly launches with PDL exactly when it has a PDL
    launch and pdl=True. CUTLASS, cuBLASLt and cuTile reject pdl=True instead
    of silently running without PDL; cuDNN runs without it. pdl=None behaves
    as pdl=False on every backend."""
    _skip_unless_supported(backend)
    a, b, _, reference = _pdl_problem()
    if pdl and _EXPLICIT_PDL_TRUE[backend] == "raises":
        with pytest.raises(ValueError, match="(?i)pdl"):
            mm_bf16(a, b, pdl=True, backend=backend)
        return

    _skip_unless_runnable(backend)
    monkeypatch.setattr(AutoTuner, "_instance", AutoTuner())
    launches = _record_pdl_launches(monkeypatch, backend)
    out = mm_bf16(a, b, pdl=pdl, backend=backend)
    if _EXPLICIT_PDL_TRUE[backend] == "pdl":
        assert launches == [bool(pdl)]
    torch.testing.assert_close(out, reference, rtol=2e-2, atol=5e-2)


@pytest.mark.parametrize("pdl", _PDL_VALUES)
@pytest.mark.parametrize("winner,bias", _AUTO_PDL_CASES)
def test_mm_bf16_auto_pdl(monkeypatch, winner, bias, pdl):
    """pdl=True does not narrow backend="auto", with or without bias: every
    backend is a candidate once, also CUTLASS and cuBLASLt, which reject an
    explicit pdl=True. The call tunes under its own op, so a pdl=False winner
    (whose cuDNN / cuBLASLt / CUTLASS key does not record pdl) is never reused
    for it. Whichever candidate wins, TGV, TinyGEMM and CuTeDSL launch with
    PDL, the others without it."""
    _skip_unless_runnable(winner)
    launches = _record_pdl_launches(monkeypatch, winner)
    choose_one = AutoTuner.choose_one
    selections = []

    def select(self, custom_op, runners, tuning_config, inputs, **kwargs):
        if not custom_op.startswith("bf16_gemm"):
            return choose_one(self, custom_op, runners, tuning_config, inputs, **kwargs)
        selections.append((custom_op, inputs[3], runners))
        backends = [_runner_backend(runner) for runner in runners]
        assert winner in backends, backends
        # Run the winner's first runner with its default tactic, as a tuned
        # entry for it would.
        return runners[backends.index(winner)], -1

    monkeypatch.setattr(AutoTuner, "choose_one", select)
    a, b, c, reference = _pdl_problem(bias)
    out = mm_bf16(a, b, bias=c, pdl=pdl, backend="auto")
    ((custom_op, runner_pdl, runners),) = selections
    assert custom_op == ("bf16_gemm_pdl" if pdl else "bf16_gemm")
    assert runner_pdl is bool(pdl)
    names = [type(runner).__name__ for runner in runners]
    assert len(names) == len(set(names)), names
    if _EXPLICIT_PDL_TRUE[winner] == "pdl":
        assert launches == [bool(pdl)]
    torch.testing.assert_close(out, reference, rtol=2e-2, atol=5e-2)


def test_mm_bf16_pdl_tunes_under_its_own_op(monkeypatch):
    """pdl=True tunes under "bf16_gemm_pdl" on the real autotuner: a pdl=False
    winner is not reused for it, it profiles again when tuned, and
    skip_ops={"bf16_gemm"} still skips it."""
    _skip_unless_runnable("tgv")
    monkeypatch.setattr(AutoTuner, "_instance", AutoTuner())

    # Fake timings make TGV win, so a winner differs from the untuned
    # runners[0] (cuDNN or cuBLASLt).
    profiled = []

    def fake_profile(self, runner, inputs, tactic, *args, **kwargs):
        profiled.append(inputs[3])
        return 1.0 if _runner_backend(runner) == "tgv" else 2.0

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)

    choose_one = AutoTuner.choose_one
    selections = []

    def record_selection(self, custom_op, runners, tuning_config, inputs, **kwargs):
        runner, tactic = choose_one(
            self, custom_op, runners, tuning_config, inputs, **kwargs
        )
        if custom_op.startswith("bf16_gemm"):
            selections.append((custom_op, _runner_backend(runner), tactic, runners))
        return runner, tactic

    monkeypatch.setattr(AutoTuner, "choose_one", record_selection)
    a, b, _, reference = _pdl_problem()

    def call(pdl):
        profiled.clear()
        selections.clear()
        out = mm_bf16(a, b, pdl=pdl, backend="auto")
        torch.testing.assert_close(out, reference, rtol=2e-2, atol=5e-2)
        ((custom_op, backend, tactic, runners),) = selections
        return custom_op, backend, tactic, _runner_backend(runners[0])

    bucket = [a.shape[0]]
    with autotune(True, tuning_buckets=bucket):
        call(pdl=False)
    assert profiled and not any(profiled), profiled
    custom_op, backend, _, first = call(pdl=False)
    assert (custom_op, backend) == ("bf16_gemm", "tgv")
    assert first != "tgv", "TGV is runners[0], so a reused winner is undetectable"

    # Untuned pdl=True must not reuse the pdl=False winner.
    custom_op, backend, tactic, first = call(pdl=True)
    assert (custom_op, backend, tactic) == ("bf16_gemm_pdl", first, -1)
    assert not profiled

    # Skipping "bf16_gemm" also skips pdl=True calls.
    with autotune(True, tuning_buckets=bucket, skip_ops={"bf16_gemm"}):
        custom_op, backend, tactic, first = call(pdl=True)
    assert not profiled, profiled
    assert (custom_op, backend, tactic) == ("bf16_gemm", first, -1)

    # Tuning pdl=True profiles again, with PDL, and its winner is then used.
    with autotune(True, tuning_buckets=bucket):
        call(pdl=True)
    assert profiled and all(profiled), profiled
    custom_op, backend, _, _ = call(pdl=True)
    assert (custom_op, backend) == ("bf16_gemm_pdl", "tgv")


if __name__ == "__main__":
    pytest.main([__file__])
