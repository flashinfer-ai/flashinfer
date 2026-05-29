import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, bmm_mxfp8
from flashinfer.fp8_quantization import mxfp8_quantize
from flashinfer.utils import get_compute_capability
from flashinfer.gemm.gemm_base import is_cudnn_override_shape_available


@pytest.mark.parametrize("b", [1, 16])
@pytest.mark.parametrize("m", [128, 256, 512])
@pytest.mark.parametrize("n", [128, 256, 512])
@pytest.mark.parametrize("k", [128, 256, 512, 1024])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16])
@pytest.mark.parametrize("backend", ["cudnn", "cutlass"])
@pytest.mark.parametrize("auto_tuning", [True, False])
def test_bmm_mxfp8(
    b, m, n, k, input_dtype, is_sf_swizzled_layout, res_dtype, backend, auto_tuning
):
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

    mat2 = (
        torch.randn([b, n, k], device="cuda", dtype=input_dtype)
        .transpose(-2, -1)
        .contiguous()
    )
    mat2_mxfp8, mat2_scale = mxfp8_quantize(mat2, is_sf_swizzled_layout)

    assert mat2_mxfp8.numel() == (mat2_scale.numel() * 32)

    # Compute reference result
    reference = torch.bmm(input_mat, mat2)

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


@pytest.mark.parametrize("m", [130, 200, 257, 384, 1000])
def test_bmm_mxfp8_cudnn_dynamic_m(m):
    """cuDNN mxfp8 must work for M that is not a multiple of 128.

    Regression for the override-shape path: the block-scale descale tensor is
    declared 3D ``[batch, dim_m, dim_k]`` in the graph, but the runtime scale
    buffer is 1D-flat.  Passing the flat ``.shape`` as the override made cuDNN
    reject every call (CUDNN_STATUS_NOT_SUPPORTED_INVALID_DYNAMIC_SHAPE) on
    cuDNN >= 9.21 -- even for 128-aligned M.  Existing tests only used
    128-aligned M on older cuDNN (non-override path), so this was masked.

    Uses the swizzled (128x4) scale layout, which is what the cuDNN graph's
    F8_128x4 reordering requires.  (Separate follow-ups: non-swizzled/linear SF
    is layout-incompatible with this graph at non-128-aligned M, and batched
    b>1 needs per-batch SF padding in mxfp8_quantize.)
    """
    is_sf_swizzled_layout = True
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] != 10:
        pytest.skip("bmm_mxfp8 cudnn backend requires SM10x.")
    if not is_cudnn_override_shape_available():
        pytest.skip("dynamic-M cuDNN mxfp8 requires the override-shape path.")

    b, n, k = 1, 256, 256  # b=1: batched (b>1) non-aligned-M is a separate fix.
    a = torch.randn([b, m, k], device="cuda", dtype=torch.bfloat16)
    mat2 = (
        torch.randn([b, n, k], device="cuda", dtype=torch.bfloat16)
        .transpose(-2, -1)
        .contiguous()
    )
    aq, asf = mxfp8_quantize(a, is_sf_swizzled_layout)
    bq, bsf = mxfp8_quantize(mat2, is_sf_swizzled_layout)
    res = torch.empty([b, m, n], device="cuda", dtype=torch.bfloat16)
    with autotune(False):
        bmm_mxfp8(aq, bq, asf, bsf, torch.bfloat16, res, backend="cudnn")
    assert torch.isfinite(res).all(), f"non-finite output at M={m}"
    cos = F.cosine_similarity(
        torch.bmm(a, mat2).reshape(-1).float(), res.reshape(-1).float(), dim=0
    )
    assert cos > 0.9, f"cos_sim {cos:.4f} too low at M={m}"


if __name__ == "__main__":
    pytest.main([__file__])
