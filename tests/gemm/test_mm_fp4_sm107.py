"""SM107-specific correctness and alpha precision tests for block-scaled FP4 GEMM."""

import inspect
import pytest
import torch

from flashinfer import mm_fp4
from flashinfer.utils import get_compute_capability
from flashinfer.gemm.kernels import dense_blockscaled_gemm_sm107


def _is_sm107() -> bool:
    if not torch.cuda.is_available():
        return False
    return get_compute_capability(torch.device("cuda")) == (10, 7)


def test_sm107_kernel_alpha_preserves_fp32():
    """Verify that SM107 blockscaled kernels keep alpha in Float32 rather than rounding to c_dtype."""
    for kernel_cls in [
        dense_blockscaled_gemm_sm107.Sm107BlockScaledPersistentDenseGemmKernel,
        dense_blockscaled_gemm_sm107.Sm107BlockScaledPersistentDenseGemmMixedClustersKernel,
    ]:
        src = inspect.getsource(kernel_cls)
        # alpha must never be converted to c_dtype before epilogue
        assert "alpha[0].to(self.c_dtype)" not in src, (
            f"{kernel_cls.__name__} erroneously converts alpha to self.c_dtype"
        )
        assert "alpha[0].to(cutlass.Float32)" in src, (
            f"{kernel_cls.__name__} must convert alpha to cutlass.Float32"
        )


@pytest.mark.skipif(not _is_sm107(), reason="Requires an SM107 GPU")
@pytest.mark.parametrize("res_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("alpha_val", [2.0**-16, 2.0**-26, 68000.0])
def test_mm_fp4_sm107_alpha_extreme_values(res_dtype, alpha_val):
    """Verify that extreme alpha scaling factors do not underflow or overflow in SM107 epilogue."""
    m, n, k = 128, 256, 256
    torch.manual_seed(107)

    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((n, k), device="cuda", dtype=torch.float32)

    # Convert to FP4 dummy representation or test through mm_fp4
    from flashinfer import SfLayout, nvfp4_quantize

    a_global_sf = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    b_global_sf = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    a_fp4, a_sf = nvfp4_quantize(
        a.to(torch.bfloat16), a_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    b_fp4, b_sf = nvfp4_quantize(
        b.to(torch.bfloat16), b_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )

    res = torch.empty((m, n), device="cuda", dtype=res_dtype)
    mm_fp4(
        a_fp4,
        b_fp4.T,
        a_sf,
        b_sf.T,
        alpha=alpha_val,
        res_dtype=res_dtype,
        out=res,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="cute-dsl",
        use_nvfp4=True,
    )

    assert torch.isfinite(res).all(), f"Found non-finite values in output with alpha={alpha_val}"
    assert not (res == 0).all(), f"Output underflowed to all zeros with alpha={alpha_val}"
