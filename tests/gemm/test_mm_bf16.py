import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, mm_bf16
from flashinfer.gemm.gemm_base import CUDNN_AVAILABLE
from flashinfer.utils import get_compute_capability


@pytest.mark.parametrize("m", [1, 8, 16, 32, 64])
@pytest.mark.parametrize("n", [1024, 2048, 4096])
@pytest.mark.parametrize("k", [1024, 2048, 3072])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("enable_bias", [True, False])
@pytest.mark.parametrize("pdl", [True, False])
@pytest.mark.parametrize(
    "backend", ["cudnn", "cutlass", "tgv", "cublaslt", "tinygemm", "auto"]
)
@pytest.mark.parametrize("auto_tuning", [False, True])
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


def test_tinygemm_bf16_runner_multi_tactic():
    """TinyGemmBf16GemmRunner enumerates STAGES tactics and each must be correct."""
    from flashinfer.gemm.gemm_base import _tinygemm_bf16_gemm_runner
    from flashinfer.utils import get_compute_capability

    compute_capability = get_compute_capability(torch.device("cuda"))
    cc_num = compute_capability[0] * 10 + compute_capability[1]
    if not mm_bf16.is_backend_supported("tinygemm", cc_num):
        pytest.skip("tinygemm backend not supported on this GPU")

    runner = _tinygemm_bf16_gemm_runner()

    m, n, k = 8, 1024, 2048
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    b = weight.transpose(-2, -1)  # mm_bf16 convention: pass b as (k, n)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(0, device="cuda", dtype=torch.uint8)
    inputs = [a, b, None, False, out, workspace]

    reference = torch.mm(a, weight.T)

    tactics = runner.get_valid_tactics(inputs, None)
    # All tactics must be multiples of 4 in [4, 16]; at least one must exist.
    assert len(tactics) >= 1
    assert all(t % 4 == 0 and 4 <= t <= 16 for t in tactics)
    assert len(set(tactics)) == len(tactics)

    # The fallback (auto-selected depth) and every explicit tactic must be correct.
    for tactic in [-1, *tactics]:
        out.zero_()
        runner.forward(inputs, tactic=tactic)
        cos_sim = F.cosine_similarity(reference.reshape(-1), out.reshape(-1), dim=0)
        assert cos_sim > 0.99, f"tactic={tactic} produced incorrect result"


if __name__ == "__main__":
    pytest.main([__file__])
