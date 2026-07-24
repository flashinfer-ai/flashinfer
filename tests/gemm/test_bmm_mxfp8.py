import warnings

import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, bmm_mxfp8
from flashinfer.fp8_quantization import mxfp8_quantize
from flashinfer.gemm import gemm_base
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
    if not is_sf_swizzled_layout:
        pytest.skip(
            "bmm_mxfp8 backends read scale factors in the F8_128x4 swizzled "
            "layout; linear-layout scales would be misinterpreted."
        )

    torch.manual_seed(42)

    # Create inputs and quantize them to MXFP8 format
    input_mat = torch.randn([b, m, k], device="cuda", dtype=input_dtype)

    # input_mxfp8 dtype will be float8_e4m3fn
    # input_scale dtype will be uint8
    input_mxfp8, input_scale = mxfp8_quantize(input_mat, is_sf_swizzled_layout)

    # Block size is 32 in MXFP8
    assert input_mxfp8.numel() == (input_scale.numel() * 32)

    weight = torch.randn([b, n, k], device="cuda", dtype=input_dtype)
    weight_mxfp8, weight_scale = mxfp8_quantize(weight, is_sf_swizzled_layout)
    mat2_mxfp8 = weight_mxfp8.transpose(-2, -1)

    assert mat2_mxfp8.shape == (b, k, n)
    assert mat2_mxfp8.stride(-2) == 1

    assert weight_mxfp8.numel() == (weight_scale.numel() * 32)

    # Compute reference result
    reference = torch.bmm(input_mat, weight.transpose(-2, -1))

    # Create output tensor
    res = torch.empty([b, m, n], device="cuda", dtype=res_dtype)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with autotune(auto_tuning):
            bmm_mxfp8(
                input_mxfp8,
                mat2_mxfp8,
                input_scale,
                weight_scale,
                res_dtype,
                res,
                backend=backend,
            )

    fallback_warnings = [
        warning
        for warning in caught
        if "falling back to default tactic=-1" in str(warning.message)
    ]
    assert not fallback_warnings

    # Verify output properties
    assert res.shape == (b, m, n), f"Expected shape {(b, m, n)}, got {res.shape}"
    assert res.dtype == res_dtype, f"Expected dtype {res_dtype}, got {res.dtype}"
    assert not torch.isnan(res).any(), "Output contains NaN values"

    min_cos_sim = 0.99
    cos_sim = F.cosine_similarity(
        reference.float().reshape(-1), res.float().reshape(-1), dim=0
    )
    assert cos_sim > min_cos_sim, (
        f"Cosine similarity {cos_sim:.4f} is too low (expected > {min_cos_sim})"
    )


def test_bmm_mxfp8_warns_for_row_major_weight(
    monkeypatch: pytest.MonkeyPatch,
):
    messages: list[str] = []
    monkeypatch.setattr(gemm_base.jit_logger, "warning_once", messages.append)

    a = torch.empty((1, 2, 32), dtype=torch.float8_e4m3fn)
    b = torch.empty((1, 32, 2), dtype=torch.float8_e4m3fn)
    gemm_base._warn_mxfp8_gemm_strides(  # pyright: ignore[reportPrivateUsage]
        a,
        b,
        "CUTLASS",
    )

    assert len(messages) == 1
    assert "CUTLASS" in messages[0]
    assert "B to be column-major" in messages[0]


def test_bmm_mxfp8_cudnn_warns_for_scale_length_mismatch(
    monkeypatch: pytest.MonkeyPatch,
):
    messages: list[str] = []
    monkeypatch.setattr(gemm_base.jit_logger, "warning_once", messages.append)

    a = torch.empty((1, 200, 256), dtype=torch.float8_e4m3fn)
    b = torch.empty((1, 256, 200), dtype=torch.float8_e4m3fn)
    linear_a_scale = torch.empty((1600,), dtype=torch.uint8)
    linear_b_scale = torch.empty((1600,), dtype=torch.uint8)

    gemm_base._warn_cudnn_bmm_mxfp8_scale_len(  # pyright: ignore[reportPrivateUsage]
        a,
        b,
        linear_a_scale,
        linear_b_scale,
    )

    assert len(messages) == 2
    assert "A_scale to contain 2048 elements" in messages[0]
    assert "B_scale to contain 2048 elements" in messages[1]
    assert all("out-of-bounds scale reads and NaN results" in msg for msg in messages)


def test_bmm_mxfp8_cudnn_scale_length_check_has_aligned_size_blind_spot(
    monkeypatch: pytest.MonkeyPatch,
):
    messages: list[str] = []
    monkeypatch.setattr(gemm_base.jit_logger, "warning_once", messages.append)

    a = torch.empty((1, 128, 256), dtype=torch.float8_e4m3fn)
    b = torch.empty((1, 256, 128), dtype=torch.float8_e4m3fn)
    indistinguishable_scale = torch.empty((1024,), dtype=torch.uint8)

    gemm_base._warn_cudnn_bmm_mxfp8_scale_len(  # pyright: ignore[reportPrivateUsage]
        a,
        b,
        indistinguishable_scale,
        indistinguishable_scale,
    )

    assert not messages


@pytest.mark.parametrize("m", [130, 200, 257, 384, 1000])
def test_bmm_mxfp8_cudnn_dynamic_m(m: int):
    if get_compute_capability(torch.device("cuda"))[0] != 10:
        pytest.skip("bmm_mxfp8 cudnn backend requires SM10x.")
    override_shape_available = (
        gemm_base._is_cudnn_override_shape_available()  # pyright: ignore[reportPrivateUsage]
    )
    if not override_shape_available:
        pytest.skip("Dynamic-M regression requires cuDNN override-shape support.")

    torch.manual_seed(42)
    n = k = 256
    input_mat = torch.randn((1, m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((1, n, k), device="cuda", dtype=torch.bfloat16)
    input_mxfp8, input_scale = mxfp8_quantize(input_mat, is_sf_swizzled_layout=True)
    weight_mxfp8, weight_scale = mxfp8_quantize(weight, is_sf_swizzled_layout=True)

    with autotune(False):
        result = bmm_mxfp8(
            input_mxfp8,
            weight_mxfp8.transpose(-2, -1),
            input_scale,
            weight_scale,
            torch.bfloat16,
            backend="cudnn",
        )

    reference = torch.bmm(input_mat, weight.transpose(-2, -1))
    cos_sim = F.cosine_similarity(
        reference.float().reshape(-1), result.float().reshape(-1), dim=0
    )
    assert torch.isfinite(result).all()
    assert cos_sim > 0.99


if __name__ == "__main__":
    pytest.main([__file__])
