import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, bmm_bf16
from flashinfer.gemm.gemm_base import (
    CUDNN_AVAILABLE,
    get_blackwell_bf16_bmm_module,
)
from flashinfer.gemm import is_cuda_tile_available
from flashinfer.gemm.kernels.cutile.bmm_bf16_cutile import (
    bmm_bf16_cutile,
    make_bmm_bf16_tune_cache,
)
from flashinfer.utils import get_compute_capability


def _skip_unless_cake_bf16_bmm_supported():
    compute_capability = get_compute_capability(torch.device("cuda"))
    compute_capability_number = compute_capability[0] * 10 + compute_capability[1]
    if compute_capability not in {
        (10, 0),
        (10, 3),
    } or not bmm_bf16.is_backend_supported("cake", compute_capability_number):
        pytest.skip("CAKE BF16 BMM requires SM100 or SM103.")


@pytest.mark.parametrize("b", [1, 16])
@pytest.mark.parametrize("m", [48, 128])
@pytest.mark.parametrize("n", [80, 64])
@pytest.mark.parametrize("k", [64, 256])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "backend", ["cutlass", "cudnn", "cutile", "tgv", "cake", "auto"]
)
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

    # The TGV (cute_ext) backend only supports bfloat16 output.
    if backend == "tgv" and res_dtype != torch.bfloat16:
        pytest.skip("bmm_bf16 with TGV backend only supports bfloat16 output.")

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


@pytest.mark.parametrize(
    "shape,res_dtype,expected_route",
    [
        ((1, 48, 80, 64), torch.float32, 0),
        ((1, 48, 80, 256), torch.float16, 1),
        ((1, 16, 64, 1024), torch.float16, 2),
        ((16, 128, 80, 256), torch.bfloat16, 3),
        ((16, 128, 80, 256), torch.float16, 4),
        ((16, 128, 80, 256), torch.float32, 5),
        ((16, 128, 64, 256), torch.bfloat16, 6),
        ((16, 128, 64, 256), torch.float16, 7),
        ((16, 128, 64, 256), torch.float32, 8),
        ((4, 16, 1024, 1024), torch.bfloat16, 9),
        ((4, 16, 1024, 1024), torch.float16, 10),
        ((4, 16, 1024, 1024), torch.float32, 11),
        ((2, 8, 1024, 1024), torch.bfloat16, 12),
        ((2, 8, 1024, 1024), torch.float32, 2),
    ],
)
def test_bmm_bf16_cake_routes_and_output_identity(shape, res_dtype, expected_route):
    _skip_unless_cake_bf16_bmm_supported()

    b, m, n, k = shape
    torch.manual_seed(7)
    a = torch.randn((b, m, k), device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn((b, n, k), device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    out = torch.empty((b, m, n), device="cuda", dtype=res_dtype)
    expected = torch.bmm(a.float(), mat2.float()).to(res_dtype)

    result = bmm_bf16(
        a,
        mat2,
        out=out,
        out_dtype=res_dtype,
        backend="cake",
    )

    assert result is out
    torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)
    assert get_blackwell_bf16_bmm_module().route_of(a, mat2, out) == expected_route


def test_bmm_bf16_cake_repeat_reuses_module_and_output():
    _skip_unless_cake_bf16_bmm_supported()

    torch.manual_seed(0)
    a = torch.randn((4, 128, 256), device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn((4, 256, 256), device="cuda", dtype=torch.bfloat16).transpose(
        -2, -1
    )
    out = torch.empty((4, 128, 256), device="cuda", dtype=torch.bfloat16)
    expected = torch.bmm(a.float(), mat2.float()).to(torch.bfloat16)

    module = get_blackwell_bf16_bmm_module(a.device)
    first = bmm_bf16(a, mat2, out=out, backend="cake")
    second = bmm_bf16(a, mat2, out=out, backend="cake")

    assert get_blackwell_bf16_bmm_module(a.device) is module
    assert first is out
    assert second is out
    torch.testing.assert_close(second, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("k", [56, 128])
def test_bmm_bf16_cake_rejects_unsupported_k(k):
    _skip_unless_cake_bf16_bmm_supported()

    a = torch.randn((1, 16, k), device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn((1, 32, k), device="cuda", dtype=torch.bfloat16).transpose(
        -2, -1
    )
    with pytest.raises(ValueError, match="K to be 64, 256, or 1024"):
        bmm_bf16(a, mat2, backend="cake")


def test_bmm_bf16_cake_rejects_non_transposed_b():
    _skip_unless_cake_bf16_bmm_supported()

    a = torch.randn((1, 16, 64), device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn((1, 64, 32), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="exact column-major"):
        bmm_bf16(a, mat2, backend="cake")


def test_bmm_bf16_cake_rejects_odd_n():
    _skip_unless_cake_bf16_bmm_supported()

    a = torch.randn((1, 16, 64), device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn((1, 65, 64), device="cuda", dtype=torch.bfloat16).transpose(
        -2, -1
    )
    with pytest.raises(ValueError, match="N to be divisible by 8"):
        bmm_bf16(a, mat2, backend="cake")


@pytest.mark.parametrize("misaligned", ["A", "B", "out"])
def test_bmm_bf16_cake_rejects_misaligned_data_pointer(misaligned):
    _skip_unless_cake_bf16_bmm_supported()

    b, m, n, k = 1, 16, 32, 64
    a = torch.randn((b, m, k), device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn((b, n, k), device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    out = torch.empty((b, m, n), device="cuda", dtype=torch.bfloat16)
    if misaligned == "A":
        a = torch.empty(a.numel() + 1, device="cuda", dtype=a.dtype)[1:].view_as(a)
    elif misaligned == "B":
        b_storage = torch.empty(b * n * k + 1, device="cuda", dtype=torch.bfloat16)[
            1:
        ].view(b, n, k)
        mat2 = b_storage.transpose(-2, -1)
    else:
        out = torch.empty(out.numel() + 1, device="cuda", dtype=out.dtype)[1:].view_as(
            out
        )

    expected_error = f"{misaligned} data pointer must be 16-byte aligned"
    with pytest.raises(ValueError, match=expected_error):
        get_blackwell_bf16_bmm_module().route_of(a, mat2, out)
    with pytest.raises(ValueError, match=expected_error):
        bmm_bf16(a, mat2, out=out, backend="cake")


def test_bmm_bf16_cake_rejects_output_input_overlap():
    _skip_unless_cake_bf16_bmm_supported()

    a = torch.randn((1, 16, 64), device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn((1, 64, 64), device="cuda", dtype=torch.bfloat16).transpose(
        -2, -1
    )
    with pytest.raises(ValueError, match="out must not overlap A"):
        get_blackwell_bf16_bmm_module().route_of(a, mat2, a)
    with pytest.raises(ValueError, match="out must not overlap A"):
        bmm_bf16(a, mat2, out=a, backend="cake")

    out_overlapping_b = mat2.transpose(-2, -1).view(-1)[: 16 * 64].view(1, 16, 64)
    with pytest.raises(ValueError, match="out must not overlap B"):
        get_blackwell_bf16_bmm_module().route_of(a, mat2, out_overlapping_b)
    with pytest.raises(ValueError, match="out must not overlap B"):
        bmm_bf16(a, mat2, out=out_overlapping_b, backend="cake")


@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_bmm_bf16_cake_fresh_output_allocation(res_dtype):
    _skip_unless_cake_bf16_bmm_supported()

    a = torch.randn((1, 16, 64), device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn((1, 32, 64), device="cuda", dtype=torch.bfloat16).transpose(
        -2, -1
    )
    expected = torch.bmm(a.float(), mat2.float()).to(res_dtype)

    first = bmm_bf16(a, mat2, out_dtype=res_dtype, backend="cake")
    second = bmm_bf16(a, mat2, out_dtype=res_dtype, backend="cake")

    assert first.dtype == res_dtype
    assert second.dtype == res_dtype
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, expected, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(second, expected, atol=1e-2, rtol=1e-2)


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
