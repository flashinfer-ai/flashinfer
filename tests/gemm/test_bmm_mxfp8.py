import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, bmm_mxfp8
from flashinfer.fp8_quantization import mxfp8_quantize
from flashinfer.utils import get_compute_capability, is_sm12x_supported


def _mxfp8_quantize_per_batch_swizzled(x):
    """Quantize each batch with independent 128-row swizzled scale padding."""
    b, m, k = x.shape
    m_pad = ((m + 127) // 128) * 128
    padded_k = ((k + 31) // 32) * 32
    sf_cols = (((padded_k // 32) + 3) // 4) * 4

    x_padded = torch.zeros((b, m_pad, k), device=x.device, dtype=x.dtype)
    x_padded[:, :m, :] = x
    x_q_padded, x_scale = mxfp8_quantize(
        x_padded.reshape(b * m_pad, k),
        is_sf_swizzled_layout=True,
    )
    x_q = x_q_padded.reshape(b, m_pad, padded_k)[:, :m, :].contiguous()
    x_scale = x_scale.reshape(b, m_pad, sf_cols).contiguous()
    return x_q, x_scale


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


@pytest.mark.parametrize("m", [17, 100])
def test_bmm_mxfp8_cutlass_non_aligned_m_per_batch_scales(m):
    """Verify CUTLASS BMM works with per-batch scales for non-128-aligned M."""
    if not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("bmm_mxfp8 cutlass backend requires SM12x.")

    b, n, k = 2, 128, 128
    input_dtype = torch.bfloat16
    out_dtype = torch.bfloat16

    input_mat = torch.randn([b, m, k], device="cuda", dtype=input_dtype)
    input_mxfp8, input_scale = _mxfp8_quantize_per_batch_swizzled(input_mat)

    mat2_row_major = torch.randn([b, n, k], device="cuda", dtype=input_dtype)
    mat2_mxfp8_row_major, mat2_scale = _mxfp8_quantize_per_batch_swizzled(
        mat2_row_major
    )
    mat2_mxfp8 = mat2_mxfp8_row_major.transpose(-2, -1).contiguous()

    reference = torch.bmm(input_mat, mat2_row_major.transpose(-2, -1))
    res = bmm_mxfp8(
        input_mxfp8,
        mat2_mxfp8,
        input_scale,
        mat2_scale,
        out_dtype,
        backend="cutlass",
    )

    min_cos_sim = 0.9
    cos_sim = F.cosine_similarity(reference.reshape(-1), res.reshape(-1), dim=0)
    assert cos_sim > min_cos_sim, (
        f"Cosine similarity {cos_sim:.4f} is too low (expected > {min_cos_sim})"
    )


@pytest.mark.parametrize("m", [17, 100])
def test_bmm_mxfp8_cutlass_rejects_combined_batch_scales(m):
    """Reject legacy combined-batch scales that can silently corrupt later batches."""
    if not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("bmm_mxfp8 cutlass backend requires SM12x.")

    b, n, k = 2, 128, 128
    input_dtype = torch.bfloat16
    out_dtype = torch.bfloat16

    input_mat = torch.randn([b, m, k], device="cuda", dtype=input_dtype)
    input_mxfp8, legacy_input_scale = mxfp8_quantize(
        input_mat,
        is_sf_swizzled_layout=True,
    )

    mat2_row_major = torch.randn([b, n, k], device="cuda", dtype=input_dtype)
    mat2_mxfp8_row_major, mat2_scale = _mxfp8_quantize_per_batch_swizzled(
        mat2_row_major
    )
    mat2_mxfp8 = mat2_mxfp8_row_major.transpose(-2, -1).contiguous()

    with pytest.raises(ValueError, match="legacy combined-batch swizzled layout"):
        bmm_mxfp8(
            input_mxfp8,
            mat2_mxfp8,
            legacy_input_scale,
            mat2_scale,
            out_dtype,
            backend="cutlass",
        )


if __name__ == "__main__":
    pytest.main([__file__])
