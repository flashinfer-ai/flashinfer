"""SM103-specific correctness tests for CUTLASS NVFP4 GEMM."""

import pytest
import torch
import torch.nn.functional as F

from flashinfer import SfLayout, nvfp4_quantize
from flashinfer.jit.gemm import gen_gemm_sm103_module_cutlass_fp4
from flashinfer.utils import get_compute_capability


# getConfigs() exposes seven tiles times nine cluster shapes and moves five
# generic tactics to the front. The K768 entries retain their original indices.
_EXPECTED_TACTIC_COUNT = 63
_GENERIC_REFERENCE_TACTIC = 0
_NATIVE_K768_TACTICS = (
    36,  # 128x128x768, cluster 1x1x1
    45,  # 128x192x768, cluster 1x1x1
    54,  # 128x256x768, cluster 1x1x1
    57,  # 128x256x768, cluster 2x2x1
)


def _skip_if_not_sm103() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Requires an SM103 GPU")
    if get_compute_capability(torch.device("cuda")) != (10, 3):
        pytest.skip("Requires an SM103 GPU")


def test_mm_fp4_sm103_native_k768_epilogue() -> None:
    """Native K768 tactics must match the generic epilogue bit for bit."""

    _skip_if_not_sm103()

    # N=768 is divisible by every native CTA-N tile under test. K=1536
    # exercises two native K768 mainloop tiles while keeping the test compact.
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
    a_sf = a_sf.view(torch.uint8)
    b_sf = b_sf.view(torch.uint8)

    module = gen_gemm_sm103_module_cutlass_fp4().build_and_load()
    tactic_count = int(module.fp4_gemm_tactic_num())
    assert tactic_count == _EXPECTED_TACTIC_COUNT, (
        "The SM103 tactic ordering changed; update the explicit native K768 "
        f"tactic mapping in this test (expected 63, got {tactic_count})."
    )

    workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    base_alpha = torch.reciprocal(a_global_sf * b_global_sf).reshape(1).float()
    reference = torch.mm(a, b.T).float()

    for out_dtype in (torch.bfloat16, torch.float16):
        for alpha_factor in (1.0, 0.73125, -0.34375):
            alpha = base_alpha * alpha_factor
            generic = torch.empty((m, n), device="cuda", dtype=out_dtype)
            module.fp4_gemm(
                a_fp4,
                b_fp4,
                a_sf,
                b_sf,
                alpha,
                generic,
                workspace,
                _GENERIC_REFERENCE_TACTIC,
            )

            for tactic in _NATIVE_K768_TACTICS:
                native = torch.empty_like(generic)
                # The optimized epilogue requires a 256-bit-aligned output.
                assert native.data_ptr() % 32 == 0
                assert native.stride(0) * native.element_size() % 32 == 0
                module.fp4_gemm(
                    a_fp4,
                    b_fp4,
                    a_sf,
                    b_sf,
                    alpha,
                    native,
                    workspace,
                    tactic,
                )

                assert torch.isfinite(native).all(), (
                    f"non-finite output for tactic={tactic}, "
                    f"out_dtype={out_dtype}, alpha_factor={alpha_factor}"
                )
                assert torch.equal(native, generic), (
                    f"native K768 tactic {tactic} differs from generic tactic "
                    f"for out_dtype={out_dtype}, alpha_factor={alpha_factor}"
                )

                cosine = F.cosine_similarity(
                    (reference * alpha_factor).reshape(-1),
                    native.float().reshape(-1),
                    dim=0,
                )
                assert cosine > 0.98, (
                    f"cosine={cosine.item():.6f} for tactic={tactic}, "
                    f"out_dtype={out_dtype}, alpha_factor={alpha_factor}"
                )
