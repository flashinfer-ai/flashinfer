import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, bmm_mxfp8
from flashinfer.fp8_quantization import mxfp8_quantize
from flashinfer.utils import get_compute_capability


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
    if backend == "cudnn" and not is_sf_swizzled_layout:
        pytest.skip(
            "bmm_mxfp8 cudnn backend requires 128x4-swizzled block scales "
            "(the cuDNN graph declares tensor_reordering.F8_128x4)."
        )

    def block_varying_randn(shape):
        # Vary each 32-element K-block's magnitude so the MXFP8 scale factors
        # are non-uniform; plain randn makes all scales roughly equal, which
        # masks scale-layout bugs.
        x = torch.randn(shape, device="cuda", dtype=torch.float32)
        e = torch.randint(-6, 7, (*shape[:-1], shape[-1] // 32), device="cuda")
        return torch.ldexp(x, e.repeat_interleave(32, dim=-1)).to(input_dtype)

    # Create inputs and quantize them to MXFP8 format
    input_mat = block_varying_randn([b, m, k])

    # input_mxfp8 dtype will be float8_e4m3fn
    # input_scale dtype will be uint8
    input_mxfp8, input_scale = mxfp8_quantize(input_mat, is_sf_swizzled_layout)

    # Block size is 32 in MXFP8
    assert input_mxfp8.numel() == (input_scale.numel() * 32)

    weight = block_varying_randn([b, n, k])
    weight_mxfp8, mat2_scale = mxfp8_quantize(weight, is_sf_swizzled_layout)
    mat2_mxfp8 = weight_mxfp8.transpose(-2, -1)

    assert mat2_mxfp8.numel() == (mat2_scale.numel() * 32)

    # Compute reference result
    reference = torch.bmm(input_mat.float(), weight.transpose(-2, -1).float())

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
    min_cos_sim = 0.98
    cos_sim = F.cosine_similarity(reference.reshape(-1), res.float().reshape(-1), dim=0)
    assert cos_sim > min_cos_sim, (
        f"Cosine similarity {cos_sim:.4f} is too low (expected > {min_cos_sim})"
    )


if __name__ == "__main__":
    pytest.main([__file__])
