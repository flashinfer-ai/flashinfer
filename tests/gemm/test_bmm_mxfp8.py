# NOTE for future contributors (incl. AI agents): keep this file a SMALL curated
# smoke set. New coverage (shapes, dtypes, backends, randomized breadth) belongs in
# tests/gemm/test_unified_gemm_fuzz.py -- extend an adapter/axis there. Add cases
# here only as deliberate regression anchors or for paths the fuzzer cannot express.

import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, bmm_mxfp8
from flashinfer.fp8_quantization import mxfp8_quantize
from flashinfer.utils import get_compute_capability


# Curated smoke set. Randomized breadth over {b,m,n,k} x backend (swizzled scales,
# tight elementwise oracle, determinism, autotune-winner validation, and the tracked
# #3604 b>1/M%128 ledger) lives in tests/gemm/test_unified_gemm_fuzz.py's bmm_mxfp8
# adapter; keep swizzled + linear scale layouts and autotune on/off per backend.
_SMOKE_CASES = [
    # b, m, n, k, is_sf_swizzled_layout, backend, auto_tuning
    (16, 128, 256, 1024, True, "cudnn", True),
    (1, 512, 128, 256, False, "cudnn", False),
    (16, 256, 512, 128, False, "cudnn", True),
    (16, 128, 128, 512, True, "cutlass", False),
    (1, 256, 512, 1024, True, "cutlass", True),
]


@pytest.mark.parametrize(
    "b,m,n,k,is_sf_swizzled_layout,backend,auto_tuning", _SMOKE_CASES
)
def test_bmm_mxfp8(b, m, n, k, is_sf_swizzled_layout, backend, auto_tuning):
    input_dtype = torch.bfloat16
    res_dtype = torch.bfloat16
    compute_capability = get_compute_capability(torch.device("cuda"))
    if backend == "cudnn" and compute_capability[0] != 10:
        pytest.skip("bmm_mxfp8 cudnn backend requires SM10x.")
    if backend == "cutlass" and compute_capability[0] != 12:
        pytest.skip("bmm_mxfp8 cutlass backend requires SM12x.")
    if backend == "cutlass" and not is_sf_swizzled_layout:
        pytest.skip("bmm_mxfp8 cutlass backend on SM12x only supports swizzled layout.")

    # Create inputs and quantize them to MXFP8 format
    input_mat = torch.randn([b, m, k], device="cuda", dtype=input_dtype)

    # input_mxfp8 dtype will be float8_e4m3fn
    # input_scale dtype will be uint8
    input_mxfp8, input_scale = mxfp8_quantize(input_mat, is_sf_swizzled_layout)

    # Block size is 32 in MXFP8
    assert input_mxfp8.numel() == (input_scale.numel() * 32)

    # The cuDNN override-shape path requires B as a COLUMN-major [b, k, n] VIEW
    # (K contiguous): quantize the contiguous [b, n, k] weight and pass the
    # transpose of the quantized tensor. A .contiguous() on the transpose flips
    # it back to row-major and is rejected once cudnn-frontend >= 1.24 enables
    # the override path (CI's older frontend hid this).
    weight = torch.randn([b, n, k], device="cuda", dtype=input_dtype)
    weight_mxfp8, mat2_scale = mxfp8_quantize(weight, is_sf_swizzled_layout)
    mat2_mxfp8 = weight_mxfp8.transpose(-2, -1)

    assert weight_mxfp8.numel() == (mat2_scale.numel() * 32)

    # Compute reference result
    reference = torch.bmm(input_mat, weight.transpose(-2, -1))

    # Create output tensor
    res = torch.empty([b, m, n], device="cuda", dtype=res_dtype)

    with autotune(auto_tuning):
        bmm_mxfp8(
            input_mxfp8,
            mat2_mxfp8,
            input_scale,
            mat2_scale,
            res_dtype,
            res,
            backend=backend,
        )

    # Verify output properties
    assert res.shape == (b, m, n), f"Expected shape {(b, m, n)}, got {res.shape}"
    assert res.dtype == res_dtype, f"Expected dtype {res_dtype}, got {res.dtype}"
    assert not torch.isnan(res).any(), "Output contains NaN values"

    # Use the same metric as in test_bmm_fp8
    min_cos_sim = 0.9  # TODO: check if can be increased
    cos_sim = F.cosine_similarity(reference.reshape(-1), res.reshape(-1), dim=0)
    assert cos_sim > min_cos_sim, (
        f"Cosine similarity {cos_sim:.4f} is too low (expected > {min_cos_sim})"
    )


if __name__ == "__main__":
    pytest.main([__file__])
