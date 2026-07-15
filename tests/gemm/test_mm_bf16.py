# NOTE for future contributors (incl. AI agents): keep this file a SMALL curated
# smoke set. New coverage (shapes, dtypes, backends, randomized breadth) belongs in
# tests/gemm/test_unified_gemm_fuzz.py -- extend an adapter/axis there. Add cases
# here only as deliberate regression anchors or for paths the fuzzer cannot express.

import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, mm_bf16
from flashinfer.gemm.gemm_base import CUDNN_AVAILABLE
from flashinfer.gemm import is_cuda_tile_available
from flashinfer.utils import get_compute_capability


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
    (8, 4096, 3072, torch.float16, False, False, "cublaslt", False),
    (1, 1024, 3072, torch.bfloat16, True, False, "tinygemm", False),
    (32, 1024, 1024, torch.bfloat16, False, False, "cutile", False),
    (64, 4096, 2048, torch.bfloat16, False, False, "auto", True),
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

    if backend == "auto" and (enable_bias or pdl):
        pytest.skip("mm_bf16 with auto backend does not support bias or pdl arguments.")

    if backend == "cutlass" and (enable_bias or pdl):
        pytest.skip(
            "mm_bf16 with CUTLASS backend does not support bias or pdl arguments."
        )
    if backend == "cublaslt" and (enable_bias or pdl):
        pytest.skip(
            "mm_bf16 with cuBLASLt backend does not support bias or pdl arguments."
        )
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


def test_cublaslt_bf16_runner_zero_algos():
    """CublasltBf16GemmRunner.forward() must raise when heuristic returns 0 algorithms."""
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

    zero_algo_buf = torch.empty(0, dtype=torch.uint8, device="cpu")
    original_get_algos = runner._get_algos
    runner._get_algos = lambda _inputs: (zero_algo_buf, 0)
    try:
        with pytest.raises(RuntimeError, match="zero algorithms"):
            runner.forward(inputs)
    finally:
        runner._get_algos = original_get_algos


if __name__ == "__main__":
    pytest.main([__file__])
