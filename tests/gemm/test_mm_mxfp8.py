import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, mm_mxfp8
from flashinfer.fp8_quantization import mxfp8_quantize
from flashinfer.utils import get_compute_capability


def _get_min_cosine_sim(
    is_sf_swizzled_layout: bool, scale: float | None = None
) -> float:
    if is_sf_swizzled_layout:
        return 0.98

    # Lower accuracy for non-swizzled layout
    if scale is not None:
        if scale < 0.5 or scale > 10.0:
            # For very small or large scales, we expect lower accuracy
            return 0.8
    return 0.84


def _assert_cosine_similarity(
    reference: torch.Tensor,
    result: torch.Tensor,
    is_sf_swizzled_layout: bool,
    *,
    use_float: bool = False,
    context: str = "",
) -> float:
    min_cos_sim = _get_min_cosine_sim(is_sf_swizzled_layout)
    if use_float:
        reference = reference.float()
        result = result.float()

    # Check cosine similarity between reference and result
    cos_sim = F.cosine_similarity(
        reference.reshape(-1), result.reshape(-1), dim=0
    ).item()

    if context:
        message = (
            f"{context} Cosine similarity {cos_sim:.4f} is too low "
            f"(expected > {min_cos_sim}, {is_sf_swizzled_layout=})."
        )
    else:
        message = (
            f"Cosine similarity {cos_sim:.4f} is too low "
            f"(expected > {min_cos_sim}, {is_sf_swizzled_layout=})."
        )
    assert cos_sim > min_cos_sim, message
    return cos_sim


def _skip_if_unsupported(backend: str = "cutlass"):
    if backend == "auto":
        backend = "cutlass"
    compute_capability = get_compute_capability(torch.device("cuda"))
    compute_capability_number = compute_capability[0] * 10 + compute_capability[1]
    if not mm_mxfp8.is_backend_supported(backend, compute_capability_number):
        pytest.skip(
            "Skipping test because mm_mxfp8 cutlass is not supported on compute "
            f"capability {compute_capability_number}."
        )


def _run_mm_mxfp8(
    m,
    n,
    k,
    input_dtype,
    is_sf_swizzled_layout,
    out_dtype,
    backend,
    auto_tuning,
    provide_out,
):
    _skip_if_unsupported(backend)

    input = torch.randn([m, k], device="cuda", dtype=input_dtype)
    mat2 = torch.randn([n, k], device="cuda", dtype=input_dtype)

    input_mxfp8, mat2_mxfp8, input_descale, mat2_descale = _prepare_mxfp8_tensors(
        input, mat2, is_sf_swizzled_layout
    )
    reference = torch.mm(input, mat2.T)

    res = torch.empty([m, n], device="cuda", dtype=out_dtype) if provide_out else None

    with autotune(auto_tuning):
        res = mm_mxfp8(
            input_mxfp8,
            mat2_mxfp8.T,  # mm_mxfp8 expects mat2.T (transposed)
            input_descale,
            mat2_descale,
            out=res,
            out_dtype=out_dtype,
            backend=backend,
        )

    assert res.shape == (m, n)
    assert res.dtype == out_dtype
    assert res.device.type == "cuda"
    assert torch.isfinite(res).all(), "Output contains NaN/Inf values"

    _assert_cosine_similarity(reference, res, is_sf_swizzled_layout)


def _prepare_descales(input_scale, weight_scale, m, n, k, is_sf_swizzled_layout):
    if is_sf_swizzled_layout:
        return input_scale, weight_scale
    input_descale = input_scale.view(m, k // 32)
    weight_descale = weight_scale.view(n, k // 32).t()
    return input_descale, weight_descale


def _prepare_mxfp8_tensors(input_bf16, weight_bf16, is_sf_swizzled_layout):
    m, k = input_bf16.shape
    n = weight_bf16.shape[0]
    input_mxfp8, input_scale = mxfp8_quantize(
        input_bf16, is_sf_swizzled_layout=is_sf_swizzled_layout
    )
    weight_mxfp8, weight_scale = mxfp8_quantize(
        weight_bf16, is_sf_swizzled_layout=is_sf_swizzled_layout
    )
    input_descale, weight_descale = _prepare_descales(
        input_scale, weight_scale, m, n, k, is_sf_swizzled_layout
    )
    return input_mxfp8, weight_mxfp8, input_descale, weight_descale


@pytest.mark.parametrize("m", [128, 256, 512, 1024])
@pytest.mark.parametrize("n", [128, 256, 512, 1024])
@pytest.mark.parametrize("k", [128, 256, 512, 1024, 2048, 2560, 3200])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["cutlass"])
@pytest.mark.parametrize("auto_tuning", [True, False])
def test_mm_mxfp8(
    m, n, k, input_dtype, is_sf_swizzled_layout, out_dtype, backend, auto_tuning
):
    _run_mm_mxfp8(
        m,
        n,
        k,
        input_dtype,
        is_sf_swizzled_layout,
        out_dtype,
        backend,
        auto_tuning,
        provide_out=True,
    )


@pytest.mark.parametrize("m", [128, 256, 1024, 2048, 4096])
@pytest.mark.parametrize("n", [2688, 5376, 8192, 12288, 16384])
@pytest.mark.parametrize("k", [4096, 8192])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
@pytest.mark.parametrize("backend", ["cutlass", "auto"])
def test_mm_mxfp8_large_dimensions(
    m, n, k, input_dtype, is_sf_swizzled_layout, out_dtype, backend
):
    _run_mm_mxfp8(
        m,
        n,
        k,
        input_dtype,
        is_sf_swizzled_layout,
        out_dtype,
        backend,
        auto_tuning=False,
        provide_out=True,
    )


@pytest.mark.parametrize(
    "m,n,k",
    [
        (4, 6144, 4096),
        (8, 6144, 4096),
        (16, 6144, 4096),
        (32, 2688, 1856),
        (32, 1856, 2688),
        (32, 2688, 4096),
        (32, 5376, 4096),
    ],
)
def test_mm_mxfp8_small_m(m, n, k):
    _run_mm_mxfp8(
        m,
        n,
        k,
        torch.bfloat16,
        True,  # swizzled scales are the intended fast path
        torch.bfloat16,
        "cutlass",
        auto_tuning=False,
        provide_out=True,
    )


def test_mm_mxfp8_invalid_input_dtype():
    _skip_if_unsupported()
    m, n, k = 128, 128, 128
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([k, n], device="cuda", dtype=torch.bfloat16)
    a_scale = torch.empty([m * (k // 32)], device="cuda", dtype=torch.uint8)
    b_scale = torch.empty([n * (k // 32)], device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        mm_mxfp8(a, b, a_scale, b_scale, out_dtype=torch.bfloat16, backend="cutlass")


def test_mm_mxfp8_invalid_ndim():
    _skip_if_unsupported()
    m, n, k = 128, 128, 128
    a = torch.randn([1, m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([k, n], device="cuda", dtype=torch.bfloat16)
    a_scale = torch.empty([m * (k // 32)], device="cuda", dtype=torch.uint8)
    b_scale = torch.empty([n * (k // 32)], device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match="accepts 2d tensors"):
        mm_mxfp8(a, b, a_scale, b_scale, out_dtype=torch.bfloat16, backend="cutlass")

    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([k, n], device="cuda", dtype=torch.bfloat16)
    a_mx, a_scale = mxfp8_quantize(a, is_sf_swizzled_layout=True)
    b_mx, b_scale = mxfp8_quantize(b.T.contiguous(), is_sf_swizzled_layout=True)
    a_descale = a_scale.view(1, -1, 1)
    b_descale = b_scale.view(1, -1, 1)
    with pytest.raises(
        ValueError,
        match=r"a_descale must be 1D \(swizzled\) or 2D \(non-swizzled\)",
    ):
        mm_mxfp8(
            a_mx,
            b_mx,
            a_descale,
            b_descale,
            out_dtype=torch.bfloat16,
            backend="cutlass",
        )


@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
def test_mm_mxfp8_find_minimum_cosine_similarity(is_sf_swizzled_layout):
    """Sweep value scales and enforce a minimum cosine similarity."""
    _skip_if_unsupported()

    m, n, k = 256, 4096, 4096

    value_scales = [0.001, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]

    results = []
    for value_scale in value_scales:
        input_data = (
            torch.randn([m, k], device="cuda", dtype=torch.bfloat16) * value_scale
        )
        mat2 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16) * value_scale

        input_mxfp8, mat2_mxfp8, input_descale, mat2_descale = _prepare_mxfp8_tensors(
            input_data, mat2, is_sf_swizzled_layout
        )

        reference = torch.mm(input_data, mat2.T)

        result = mm_mxfp8(
            input_mxfp8,
            mat2_mxfp8.T,
            input_descale,
            mat2_descale,
            out_dtype=torch.bfloat16,
            backend="cutlass",
        )

        cos_sim = F.cosine_similarity(
            reference.reshape(-1).float(), result.reshape(-1).float(), dim=0
        ).item()

        results.append((value_scale, cos_sim))

    print("\n" + "=" * 60)
    print(f"MXFP8 Cosine Similarity vs Value Scale Summary ({is_sf_swizzled_layout=})")
    print("=" * 60)

    fail_test: bool = False
    for scale, sim in results:
        min_cosine_sim = _get_min_cosine_sim(is_sf_swizzled_layout, scale)
        fail = sim < min_cosine_sim

        status = "[OK]" if not fail else "[FAIL]"
        print(f"  {status} Scale={scale:8.3f}: cos_sim={sim:.4f}")
        fail_test |= fail

    print("=" * 60)

    # Assert minimum acceptable similarity
    assert not fail_test, "One or more cosine similarities are too low"


@pytest.mark.parametrize("m", [256, 512, 1024])  # Skip M=128 (edge case issues)
@pytest.mark.parametrize("n", [4096, 14336])
@pytest.mark.parametrize("k", [4096])  # Focus on common hidden_size
@pytest.mark.parametrize(
    "input_std,weight_std",
    [
        (0.1, 0.02),  # Typical trained model statistics
        (0.5, 0.1),  # Larger activations
        (1.0, 1.0),  # Random normal (baseline)
    ],
)
def test_mm_mxfp8_realistic_model_statistics(m, n, k, input_std, weight_std):
    """Test accuracy for typical activation/weight statistics."""
    _skip_if_unsupported()

    torch.manual_seed(42)  # Reproducibility

    input_data = torch.randn([m, k], device="cuda", dtype=torch.bfloat16) * input_std
    mat2 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16) * weight_std

    reference = torch.mm(input_data, mat2.T)

    input_mxfp8, mat2_mxfp8, input_descale, mat2_descale = _prepare_mxfp8_tensors(
        input_data, mat2, True
    )

    result = mm_mxfp8(
        input_mxfp8,
        mat2_mxfp8.T,
        input_descale,
        mat2_descale,
        out_dtype=torch.bfloat16,
        backend="cutlass",
    )

    # Check for NaN/Inf
    if not torch.isfinite(result).all():
        pytest.fail(
            f"Output contains NaN/Inf for M={m}, N={n}, K={k}, "
            f"input_std={input_std}, weight_std={weight_std}"
        )

    cos_sim = F.cosine_similarity(
        reference.reshape(-1).float(), result.reshape(-1).float(), dim=0
    ).item()

    # Should maintain high accuracy across all realistic value ranges
    assert cos_sim > 0.95, (
        f"Accuracy too low for M={m}, N={n}, K={k}, "
        f"input_std={input_std}, weight_std={weight_std}: cos_sim={cos_sim:.4f}"
    )


def test_mm_mxfp8_llm_full_layer_simulation():
    """Simulate a transformer layer forward pass with multiple MM calls."""
    _skip_if_unsupported()

    torch.manual_seed(42)
    m = 256  # Batch size
    hidden_size = 4096
    intermediate_size = 14336
    qkv_size = 6144
    gate_up_size = 28672  # gate + up combined

    hidden_states = (
        torch.randn([m, hidden_size], device="cuda", dtype=torch.bfloat16) * 0.1
    )

    weights = {
        "qkv": torch.randn([qkv_size, hidden_size], device="cuda", dtype=torch.bfloat16)
        * 0.02,
        "o_proj": torch.randn(
            [hidden_size, hidden_size], device="cuda", dtype=torch.bfloat16
        )
        * 0.02,
        "gate_up": torch.randn(
            [gate_up_size, hidden_size], device="cuda", dtype=torch.bfloat16
        )
        * 0.02,
        "down": torch.randn(
            [hidden_size, intermediate_size], device="cuda", dtype=torch.bfloat16
        )
        * 0.02,
    }

    results = {}

    for name, weight in weights.items():
        n, k = weight.shape

        if name == "down":
            layer_input = (
                torch.randn([m, intermediate_size], device="cuda", dtype=torch.bfloat16)
                * 0.1
            )
        else:
            layer_input = hidden_states

        reference = torch.mm(layer_input, weight.T)

        input_mxfp8, weight_mxfp8, input_descale, weight_descale = (
            _prepare_mxfp8_tensors(layer_input, weight, True)
        )

        result = mm_mxfp8(
            input_mxfp8,
            weight_mxfp8.T,
            input_descale,
            weight_descale,
            out_dtype=torch.bfloat16,
            backend="cutlass",
        )

        cos_sim = F.cosine_similarity(
            reference.reshape(-1).float(), result.reshape(-1).float(), dim=0
        ).item()

        results[name] = cos_sim
        print(
            f"  {name}: input=[{m}, {layer_input.shape[1]}] @ weight=[{n}, {k}].T -> cos_sim={cos_sim:.6f}"
        )

    for name, cos_sim in results.items():
        assert cos_sim > 0.98, f"Layer {name} has low accuracy: cos_sim={cos_sim:.4f}"

    print(
        f"\n  All layers passed with average cos_sim={sum(results.values()) / len(results):.6f}"
    )


def test_mm_mxfp8_scale_contiguity_requirement():
    """Test behavior with non-contiguous scale tensors."""
    _skip_if_unsupported()

    m, n, k = 256, 4096, 4096

    input_bf16 = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    weight_bf16 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)

    input_fp8, input_scale = mxfp8_quantize(input_bf16, is_sf_swizzled_layout=False)
    weight_fp8, weight_scale = mxfp8_quantize(weight_bf16, is_sf_swizzled_layout=False)

    input_descale = input_scale.view(m, k // 32)

    weight_scale_2d = weight_scale.view(n, k // 32)
    weight_descale_noncontig = weight_scale_2d.t()  # Non-contiguous!

    assert not weight_descale_noncontig.is_contiguous(), (
        "Expected non-contiguous tensor"
    )

    output = mm_mxfp8(
        input_fp8,
        weight_fp8.T,
        input_descale,
        weight_descale_noncontig,
        out_dtype=torch.bfloat16,
        backend="cutlass",
    )
    assert torch.isfinite(output).all()

    weight_descale_contig = weight_descale_noncontig.contiguous()
    assert weight_descale_contig.is_contiguous()

    output = mm_mxfp8(
        input_fp8,
        weight_fp8.T,
        input_descale,
        weight_descale_contig,
        out_dtype=torch.bfloat16,
        backend="cutlass",
    )
    assert torch.isfinite(output).all(), "Output with contiguous scale should be valid"


@pytest.mark.parametrize("m", [128, 256, 512, 1024, 2048, 4096, 8192, 16384])
def test_mm_mxfp8_scale_1d_tensor_interpretation(m):
    """Check that 1D swizzled scales have the expected size."""
    _skip_if_unsupported()

    n, k = 4096, 4096

    input_bf16 = torch.randn([m, k], device="cuda", dtype=torch.bfloat16) * 0.1
    weight_bf16 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16) * 0.02

    input_fp8, weight_fp8, input_descale, weight_descale = _prepare_mxfp8_tensors(
        input_bf16, weight_bf16, True
    )

    input_scale = input_descale
    # Verify scale tensor properties
    assert input_scale.ndim == 1, (
        f"Swizzled scale should be 1D, got {input_scale.ndim}D"
    )
    assert input_scale.is_contiguous(), "Swizzled scale must be contiguous"

    padded_m = ((m + 127) // 128) * 128
    k_scale_cols = k // 32
    padded_k_scale = ((k_scale_cols + 3) // 4) * 4
    expected_input_scale_size = padded_m * padded_k_scale

    assert input_scale.numel() == expected_input_scale_size, (
        f"Input scale size mismatch: got {input_scale.numel()}, "
        f"expected {expected_input_scale_size} for M={m}, K={k} "
        f"(padded_m={padded_m}, padded_k_scale={padded_k_scale})"
    )

    output = mm_mxfp8(
        input_fp8,
        weight_fp8.T,
        input_descale,
        weight_descale,
        out_dtype=torch.bfloat16,
        backend="cutlass",
    )

    assert output.shape == (m, n)
    assert torch.isfinite(output).all()
