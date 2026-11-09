# NOTE for future contributors (incl. AI agents): keep this file a SMALL curated
# smoke set. New coverage (shapes, dtypes, backends, randomized breadth) belongs in
# tests/gemm/test_unified_gemm_fuzz.py -- extend an adapter/axis there. Add cases
# here only as deliberate regression anchors or for paths the fuzzer cannot express.

import warnings

import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, bmm_mxfp8
from flashinfer.fp8_quantization import mxfp8_quantize
from flashinfer.gemm import gemm_base
from flashinfer.utils import get_compute_capability


# Curated smoke set. Randomized breadth over {b,m,n,k} x backend (swizzled scales,
# tight elementwise oracle, determinism, autotune-winner validation) lives in
# tests/gemm/test_unified_gemm_fuzz.py's bmm_mxfp8 adapter; keep autotune on/off per
# backend. Linear scale metadata cannot currently be represented by the public 3D BMM
# API, so every executable row uses swizzled scales.
_SMOKE_CASES = [
    # b, m, n, k, is_sf_swizzled_layout, backend, auto_tuning
    (16, 128, 256, 1024, True, "cudnn", True),
    (1, 512, 128, 256, True, "cudnn", False),
    (16, 256, 512, 128, True, "cudnn", True),
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
    torch.manual_seed(42)

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
    weight_mxfp8, weight_scale = mxfp8_quantize(weight, is_sf_swizzled_layout)
    mat2_mxfp8 = weight_mxfp8.transpose(-2, -1)

    assert mat2_mxfp8.shape == (b, k, n)
    assert mat2_mxfp8.stride(-2) == 1

    assert weight_mxfp8.numel() == (weight_scale.numel() * 32)

    # Create output tensor
    res = torch.empty([b, m, n], device="cuda", dtype=res_dtype)

    if not is_sf_swizzled_layout:
        # mxfp8_quantize always returns 1D scales for 3D inputs regardless of
        # is_sf_swizzled_layout, so the API cannot distinguish swizzled from
        # non-swizzled layouts by shape alone. Passing the wrong layout produces
        # silently incorrect results; there is no way to enforce this at the API
        # level without out-of-band metadata.
        pytest.skip(
            "bmm_mxfp8 cannot detect non-swizzled 1D scales from shape alone; "
            "layout validation is not possible for 3D inputs."
        )

    # Compute reference result
    reference = torch.bmm(input_mat, weight.transpose(-2, -1))

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


def _mxfp8_quantize_per_batch(x):
    return mxfp8_quantize(x, True, rank_preserving=True)


def _per_batch_cos_sim(res, reference):
    return F.cosine_similarity(
        reference.float().flatten(1), res.float().flatten(1), dim=1
    )


# #3604 regression anchor: b>1 with m or n not a multiple of 128 needs a per-batch scale.
@pytest.mark.parametrize("b,m,n,k", [(4, 130, 256, 256), (2, 200, 160, 512), (2, 130, 130, 160), (3, 64, 200, 288)])
@pytest.mark.parametrize("scale_ndim", [3, 1])
@pytest.mark.parametrize("backend", ["cudnn", "cutlass"])
def test_bmm_mxfp8_per_batch_scale(b, m, n, k, scale_ndim, backend):
    compute_capability = get_compute_capability(torch.device("cuda"))
    if backend == "cudnn" and compute_capability[0] != 10:
        pytest.skip("bmm_mxfp8 cudnn backend requires SM10x.")
    if backend == "cutlass" and compute_capability[0] != 12:
        pytest.skip("bmm_mxfp8 cutlass backend requires SM12x.")
    torch.manual_seed(42)
    input_mat = torch.randn([b, m, k], device="cuda", dtype=torch.bfloat16)
    weight = torch.randn([b, n, k], device="cuda", dtype=torch.bfloat16)
    input_mxfp8, input_scale = _mxfp8_quantize_per_batch(input_mat)
    weight_mxfp8, weight_scale = _mxfp8_quantize_per_batch(weight)
    assert input_scale.shape == (b, (m + 127) // 128 * 128, (k // 32 + 3) // 4 * 4)
    if scale_ndim == 1:
        input_scale = input_scale.flatten()
        weight_scale = weight_scale.flatten()

    with autotune(False):
        res = bmm_mxfp8(
            input_mxfp8,
            weight_mxfp8.transpose(-2, -1),
            input_scale,
            weight_scale,
            torch.bfloat16,
            backend=backend,
        )

    reference = torch.bmm(input_mat, weight.transpose(-2, -1))
    assert torch.isfinite(res).all()
    cos_sim = _per_batch_cos_sim(res, reference)
    assert (cos_sim > 0.99).all(), f"per-batch cosine similarity {cos_sim.tolist()}"


def test_bmm_mxfp8_rejects_folded_scale_for_unaligned_batches():
    b, m, n, k = 4, 130, 256, 256
    a = torch.empty((b, m, k), dtype=torch.float8_e4m3fn)
    bt = torch.empty((b, k, n), dtype=torch.float8_e4m3fn)
    folded_a_scale = torch.empty(
        ((b * m + 127) // 128 * 128 * (k // 32),), dtype=torch.uint8
    )
    b_scale = torch.empty((b * n * (k // 32),), dtype=torch.uint8)
    with pytest.raises(ValueError, match="A_scale to contain 8192 elements"):
        gemm_base._check_bmm_mxfp8_scale_shape(  # pyright: ignore[reportPrivateUsage]
            a, bt, folded_a_scale, b_scale
        )
    wrong_3d_scale = torch.empty((b, m, k // 32), dtype=torch.uint8)
    with pytest.raises(ValueError, match=r"3D A_scale of shape \(4, 256, 8\)"):
        gemm_base._check_bmm_mxfp8_scale_shape(  # pyright: ignore[reportPrivateUsage]
            a, bt, wrong_3d_scale, b_scale
        )
    gemm_base._check_bmm_mxfp8_scale_shape(  # pyright: ignore[reportPrivateUsage]
        a, bt, torch.empty((b, 256, k // 32), dtype=torch.uint8), b_scale
    )


def test_bmm_mxfp8_raises_for_row_major_weight():
    a = torch.empty((1, 2, 32), dtype=torch.float8_e4m3fn)
    b = torch.empty((1, 32, 2), dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="CUTLASS.*B to be column-major"):
        gemm_base._check_mxfp8_gemm_strides(  # pyright: ignore[reportPrivateUsage]
            a,
            b,
            "CUTLASS",
        )


def test_bmm_mxfp8_raises_for_scale_length_mismatch():
    a = torch.empty((1, 200, 256), dtype=torch.float8_e4m3fn)
    b = torch.empty((1, 256, 200), dtype=torch.float8_e4m3fn)
    linear_a_scale = torch.empty((1600,), dtype=torch.uint8)
    linear_b_scale = torch.empty((1600,), dtype=torch.uint8)

    with pytest.raises(ValueError, match="A_scale to contain 2048 elements"):
        gemm_base._check_bmm_mxfp8_scale_shape(  # pyright: ignore[reportPrivateUsage]
            a,
            b,
            linear_a_scale,
            linear_b_scale,
        )


def test_bmm_mxfp8_scale_length_check_has_aligned_size_blind_spot():
    a = torch.empty((1, 128, 256), dtype=torch.float8_e4m3fn)
    b = torch.empty((1, 256, 128), dtype=torch.float8_e4m3fn)
    indistinguishable_scale = torch.empty((1024,), dtype=torch.uint8)

    # Scale has the right element count for the expected swizzled layout,
    # so the check passes even if the data is not actually swizzled.
    gemm_base._check_bmm_mxfp8_scale_shape(  # pyright: ignore[reportPrivateUsage]
        a,
        b,
        indistinguishable_scale,
        indistinguishable_scale,
    )


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
