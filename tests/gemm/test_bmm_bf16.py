import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, bmm_bf16
from flashinfer.gemm.gemm_base import CUDNN_AVAILABLE
from flashinfer.gemm import is_cuda_tile_available
from flashinfer.gemm.kernels.cutile.bmm_bf16_cutile import (
    bmm_bf16_cutile,
    make_bmm_bf16_tune_cache,
)
from flashinfer.utils import get_compute_capability


@pytest.mark.parametrize("b", [1, 16])
@pytest.mark.parametrize("m", [48, 128])
@pytest.mark.parametrize("n", [80, 64])
@pytest.mark.parametrize("k", [64, 256])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("backend", ["cutlass", "cudnn", "cutile", "auto"])
def test_bmm_bf16(b, m, n, k, res_dtype, backend):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    compute_capability_number = compute_capability[0] * 10 + compute_capability[1]
    if not bmm_bf16.is_compute_capability_supported(compute_capability_number):
        pytest.skip(
            f"bmm_bf16 not supported on current compute capability."
            f"Detected sm{compute_capability_number}."
        )
    if backend != "auto":
        if not bmm_bf16.is_backend_supported(backend, compute_capability_number):
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

    # cuDNN on SM103 does not support bf16 input -> fp16 output
    if (
        backend == "cudnn"
        and compute_capability_number == 103
        and res_dtype == torch.float16
    ):
        pytest.skip("cuDNN bf16 GEMM with fp16 output not supported on SM103.")
    torch.manual_seed(7)
    input = torch.randn([b, m, k], device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn([b, n, k], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    reference = torch.bmm(input, mat2)

    out = torch.empty([b, m, n], device="cuda", dtype=res_dtype)
    with autotune():
        bmm_bf16(input, mat2, out=out, out_dtype=res_dtype, backend=backend)

    cos_sim = F.cosine_similarity(reference.reshape(-1), out.reshape(-1), dim=0)
    assert cos_sim > 0.99


def test_bmm_bf16_cutile_repeat_uses_tune_cache():
    """Two back-to-back calls at the same shape must hit the cuTile tune cache.

    Uses :func:`make_bmm_bf16_tune_cache` to create an isolated cache so the
    test does not interfere with the module-level default cache and does not
    import private internals.
    """
    compute_capability = get_compute_capability(torch.device("cuda"))
    cc_num = compute_capability[0] * 10 + compute_capability[1]
    if not bmm_bf16.is_backend_supported("cutile", cc_num):
        pytest.skip("cuTile backend not supported on current compute capability.")
    if not is_cuda_tile_available():
        pytest.skip("cuda-tile / tileiras compiler not available in this environment.")

    torch.random.manual_seed(0)
    A = torch.randn(4, 128, 256, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(4, 256, 256, device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    out = torch.empty(4, 128, 256, device="cuda", dtype=torch.bfloat16)

    cache = make_bmm_bf16_tune_cache()  # fresh, isolated cache

    # First call: exhaustive_search runs and populates the cache.
    out1 = bmm_bf16_cutile(A, B, out.clone(), tune_cache=cache)
    assert len(cache) == 1, (
        f"first call should populate cache; got {len(cache)} entries"
    )

    # Second call at the same shape: must hit cache without re-tuning.
    out2 = bmm_bf16_cutile(A, B, out.clone(), tune_cache=cache)
    assert len(cache) == 1, f"second call must hit cache; got {len(cache)} entries"

    torch.testing.assert_close(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__])
