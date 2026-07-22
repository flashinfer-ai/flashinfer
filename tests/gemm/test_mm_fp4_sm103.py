"""SM103-specific correctness tests for CUTLASS NVFP4 GEMM."""

import pytest
import torch
import torch.nn.functional as F

from flashinfer import SfLayout, autotune, mm_fp4, nvfp4_quantize
from flashinfer.utils import get_compute_capability


def _skip_if_not_sm103() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Requires an SM103 GPU")
    if get_compute_capability(torch.device("cuda")) != (10, 3):
        pytest.skip("Requires an SM103 GPU")


def test_mm_fp4_sm103_cutlass_epilogue() -> None:
    """The public API must produce accurate output through SM103 CUTLASS."""

    _skip_if_not_sm103()

    # K=1536 keeps the native K768 tactics eligible during public autotuning
    # while keeping the end-to-end API test compact.
    m, n, k = 512, 768, 1536
    torch.manual_seed(103768)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

    a_global_sf = (448 * 6) / a.float().abs().nan_to_num().max()
    b_global_sf = (448 * 6) / b.float().abs().nan_to_num().max()
    a_fp4, a_sf = nvfp4_quantize(
        a, a_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    b_fp4, b_sf = nvfp4_quantize(
        b, b_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    base_alpha = torch.reciprocal(a_global_sf * b_global_sf).float()
    reference = torch.mm(a, b.T).float()

    for out_dtype in (torch.bfloat16, torch.float16):
        unit_alpha_norm = None
        for alpha_factor in (1.0, 0.73125, -0.34375):
            alpha = base_alpha * alpha_factor
            output = torch.empty((m, n), device="cuda", dtype=out_dtype)

            # The optimized epilogue requires a 256-bit-aligned output.
            assert output.data_ptr() % 32 == 0
            assert output.stride(0) * output.element_size() % 32 == 0

            with autotune(True):
                result = mm_fp4(
                    a_fp4,
                    b_fp4.T,
                    a_sf,
                    b_sf.T,
                    alpha=alpha,
                    out_dtype=out_dtype,
                    out=output,
                    block_size=16,
                    use_8x4_sf_layout=False,
                    backend="cutlass",
                    use_nvfp4=True,
                )

            assert result.data_ptr() == output.data_ptr()
            assert torch.isfinite(output).all(), (
                f"non-finite output for out_dtype={out_dtype}, "
                f"alpha_factor={alpha_factor}"
            )

            cosine = F.cosine_similarity(
                (reference * alpha_factor).reshape(-1),
                output.float().reshape(-1),
                dim=0,
            )
            assert cosine > 0.98, (
                f"cosine={cosine.item():.6f} for out_dtype={out_dtype}, "
                f"alpha_factor={alpha_factor}"
            )

            output_norm = output.float().norm().item()
            if alpha_factor == 1.0:
                unit_alpha_norm = output_norm
            else:
                assert unit_alpha_norm is not None
                observed_scale = output_norm / unit_alpha_norm
                assert observed_scale == pytest.approx(
                    abs(alpha_factor), rel=0.01, abs=0.005
                )
