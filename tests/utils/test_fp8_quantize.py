import pytest
import torch

from flashinfer import mxfp8_grouped_quantize, mxfp8_quantize, SfLayout
from flashinfer.utils import get_compute_capability
from tests.utils_fp8 import (
    assert_mxfp8_quantize_exact as _assert_mxfp8_quantize_exact,
    mxfp8_quantize_reference,
)


def is_cute_dsl_available():
    """Check if CuTe-DSL is available."""
    try:
        from flashinfer.cute_dsl import is_cute_dsl_available as _is_available

        return _is_available()
    except ImportError:
        return False


def is_cutile_available():
    """Check if the cuTile backend is available for grouped MXFP8 quantization."""
    try:
        from flashinfer.cutile import is_cuda_tile_available

        return is_cuda_tile_available()
    except ImportError:
        return False


def _is_mxfp8_supported(device: torch.device) -> bool:
    """Check if MXFP8 quantization is supported on this device.

    The public ``mxfp8_quantize`` and ``mxfp8_grouped_quantize`` APIs gate on
    "SM100 or newer" (the grouped wrapper raises for ``major < 10``), so use a
    forward-compatible minimum-compute-capability check.
    """
    return get_compute_capability(device)[0] >= 10


def _unswizzle_mxfp8_scales_128x4(
    sf: torch.Tensor,
    row: int,
    col: int,
) -> torch.Tensor:
    scale_vec_size = 32
    factor = scale_vec_size * 4
    num_m_tiles = (row + 128 - 1) // 128
    num_k_tiles = (col + factor - 1) // factor
    sf_reshaped = sf.view(num_m_tiles, num_k_tiles, 32, 4, 4)
    sf_unswizzled = sf_reshaped.transpose(1, 3)
    sf_unswizzled = sf_unswizzled.reshape(num_m_tiles * 32 * 4, num_k_tiles * 4)
    return sf_unswizzled[:row, : (col // scale_vec_size)].contiguous()


@pytest.mark.parametrize("m", [1, 3, 16, 64, 1024])
@pytest.mark.parametrize("k", [128, 1024, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("backend", ["cuda", "cute-dsl"])
def test_mxfp8_quantize_torch(m, k, dtype, is_sf_swizzled_layout, device, backend):
    if device == "cuda" and not _is_mxfp8_supported(torch.device(device)):
        pytest.skip("mxfp8 quantization is not supported on compute capability < 10")

    # Skip cute-dsl backend for CPU or if not available
    if backend == "cute-dsl":
        if device == "cpu":
            pytest.skip("cute-dsl backend only supports CUDA")
        if not is_cute_dsl_available():
            pytest.skip("CuTe-DSL is not available")

    a = 16 * torch.randn([m, k], dtype=dtype).to(device).contiguous()

    if device == "cpu":
        a = a.float()

    a_fp8, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout, backend=backend)

    _assert_mxfp8_quantize_exact(
        a,
        a_fp8,
        a_sf,
        is_sf_swizzled_layout=is_sf_swizzled_layout,
    )

    if device == "cuda":
        torch.cuda.synchronize()


@pytest.mark.parametrize("batch_shape", [(1, 120, 64), (2, 128, 128), (3, 256, 160)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_mxfp8_grouped_quantize(batch_shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not _is_mxfp8_supported(torch.device("cuda:0")):
        pytest.skip("mxfp8 grouped quantization requires compute capability >= 10")
    if not is_cutile_available():
        pytest.skip("cuda.tile is not available")

    torch.manual_seed(0)
    b, m, k = batch_shape
    x = (torch.randn(batch_shape, dtype=torch.float32, device="cuda") * 16).to(dtype)
    x = x.contiguous()
    mask = torch.randint(low=1, high=m + 1, size=(b,), dtype=torch.int32, device="cuda")

    out, out_scale = mxfp8_grouped_quantize(x, mask)
    out = out.permute(2, 0, 1)
    out_scale = out_scale.permute(5, 2, 4, 0, 1, 3)

    padded_m = (m + 127) // 128 * 128
    padded_k = (k + 127) // 128 * 128
    assert out.shape == (b, m, padded_k)
    assert out.dtype == torch.float8_e4m3fn
    assert out_scale.shape == (b, padded_m // 128, padded_k // 128, 32, 4, 4)
    assert out_scale.dtype == torch.uint8

    for i in range(b):
        mask_i = int(mask[i].item())
        single_out, single_scale = mxfp8_quantize_reference(
            x[i],
            alignment=128,
            sf_swizzle_layout=SfLayout.layout_128x4,
        )
        torch.testing.assert_close(
            out[i, :mask_i].contiguous().view(torch.uint8),
            single_out[:mask_i].contiguous().view(torch.uint8),
            rtol=0,
            atol=0,
        )

        scale_ref = _unswizzle_mxfp8_scales_128x4(single_scale, m, padded_k)
        scale_ans = _unswizzle_mxfp8_scales_128x4(out_scale[i], m, padded_k)
        torch.testing.assert_close(
            scale_ans[:mask_i],
            scale_ref[:mask_i],
            rtol=0,
            atol=0,
        )

    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_mxfp8_grouped_quantize_empty_group(dtype):
    """A zero-token group (mask=0) must not corrupt its non-empty neighbors."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not _is_mxfp8_supported(torch.device("cuda:0")):
        pytest.skip("mxfp8 grouped quantization requires compute capability >= 10")
    if not is_cutile_available():
        pytest.skip("cuda.tile is not available")

    torch.manual_seed(0)
    b, m, k = 4, 160, 256
    x = (torch.randn((b, m, k), dtype=torch.float32, device="cuda") * 16).to(dtype)
    x = x.contiguous()
    # Groups 0 and 2 are empty; groups 1 and 3 are full.
    mask = torch.tensor([0, m, 0, m], dtype=torch.int32, device="cuda")

    # Must not raise even with empty groups interleaved.
    out, out_scale = mxfp8_grouped_quantize(x, mask)
    out = out.permute(2, 0, 1)
    out_scale = out_scale.permute(5, 2, 4, 0, 1, 3)

    padded_m = (m + 127) // 128 * 128
    padded_k = (k + 127) // 128 * 128
    assert out.shape == (b, m, padded_k)
    assert out_scale.shape == (b, padded_m // 128, padded_k // 128, 32, 4, 4)

    for i in range(b):
        mask_i = int(mask[i].item())
        if mask_i == 0:
            continue
        single_out, single_scale = mxfp8_quantize_reference(
            x[i],
            alignment=128,
            sf_swizzle_layout=SfLayout.layout_128x4,
        )
        torch.testing.assert_close(
            out[i, :mask_i].contiguous().view(torch.uint8),
            single_out[:mask_i].contiguous().view(torch.uint8),
            rtol=0,
            atol=0,
        )
        scale_ref = _unswizzle_mxfp8_scales_128x4(single_scale, m, padded_k)
        scale_ans = _unswizzle_mxfp8_scales_128x4(out_scale[i], m, padded_k)
        torch.testing.assert_close(
            scale_ans[:mask_i],
            scale_ref[:mask_i],
            rtol=0,
            atol=0,
        )

    torch.cuda.synchronize()


@pytest.mark.parametrize("batch_shape", [(2, 128, 256), (3, 256, 128)])
@torch.inference_mode()
def test_mxfp8_grouped_quantize_cuda_graph(batch_shape):
    """Grouped MXFP8 quantize must be CUDA-graph capturable and replay correctly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not _is_mxfp8_supported(torch.device("cuda:0")):
        pytest.skip("mxfp8 grouped quantization requires compute capability >= 10")
    if not is_cutile_available():
        pytest.skip("cuda.tile is not available")

    torch.manual_seed(0)
    b, m, k = batch_shape
    x = torch.randn(batch_shape, dtype=torch.bfloat16, device="cuda")
    mask = torch.full((b,), m, dtype=torch.int32, device="cuda")

    # Eager warmup: triggers the cuTile JIT compile so the first launch does not
    # happen mid-capture. The prefix scratch buffer is allocated per call from
    # the stream-ordered caching allocator, so there is no buffer to prepopulate.
    out_eager, sf_eager = mxfp8_grouped_quantize(x, mask)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_graph, sf_graph = mxfp8_grouped_quantize(x, mask)

    g.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        out_graph.contiguous().view(torch.uint8),
        out_eager.contiguous().view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(sf_graph, sf_eager, rtol=0, atol=0)


@pytest.mark.parametrize("batch_shape", [(2, 128, 256), (3, 256, 128)])
@torch.inference_mode()
def test_mxfp8_grouped_quantize_cuda_graph_pool_reuse(batch_shape):
    """Replay a captured graph many times with interleaved allocator churn.

    The per-call ``tile_offsets`` scratch is allocated and freed inside the
    capture region, so it returns to the graph private pool during capture.
    This stresses graph-pool isolation: the recorded prefix table must stay
    reserved across replays even when unrelated allocations and frees happen in
    between.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not _is_mxfp8_supported(torch.device("cuda:0")):
        pytest.skip("mxfp8 grouped quantization requires compute capability >= 10")
    if not is_cutile_available():
        pytest.skip("cuda.tile is not available")

    from flashinfer.quantization.kernels.cutile.mxfp8_grouped_quantize_cutile import (
        MAX_GROUPS_FUSED,
    )

    torch.manual_seed(0)
    b, m, k = batch_shape
    x = torch.randn(batch_shape, dtype=torch.bfloat16, device="cuda")
    mask = torch.full((b,), m, dtype=torch.int32, device="cuda")

    # Eager reference. Also triggers the cuTile JIT compile before capture.
    out_eager, sf_eager = mxfp8_grouped_quantize(x, mask)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_graph, sf_graph = mxfp8_grouped_quantize(x, mask)

    scratch_numel = MAX_GROUPS_FUSED + 1
    for _ in range(8):
        # Churn the caching allocator with same-sized blocks as the prefix
        # scratch (the most likely candidates to grab a non-isolated address)
        # plus a large block, writing a sentinel so any reuse would corrupt the
        # next replay rather than silently match.
        churn = [
            torch.full((scratch_numel,), -1, dtype=torch.int32, device="cuda")
            for _ in range(32)
        ]
        churn.append(torch.full((1 << 22,), -1, dtype=torch.int32, device="cuda"))
        del churn

        g.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(
            out_graph.contiguous().view(torch.uint8),
            out_eager.contiguous().view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(sf_graph, sf_eager, rtol=0, atol=0)


@torch.inference_mode()
def test_mxfp8_grouped_quantize_concurrent_streams():
    """Two streams on one device must not share prefix-schedule scratch.

    The op allocates its ``tile_offsets`` prefix table per call. This test
    guards against regressing to a per-device cached buffer, which would be
    shared across streams and race: one launch prefix table could overwrite the
    other between the build kernel and the persistent read (see
    flashinfer-ai/flashinfer#3618).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not _is_mxfp8_supported(torch.device("cuda:0")):
        pytest.skip("mxfp8 grouped quantization requires compute capability >= 10")
    if not is_cutile_available():
        pytest.skip("cuda.tile is not available")

    torch.manual_seed(0)
    dtype = torch.bfloat16

    # Distinct group counts, shapes, and masks so a prefix table built by one
    # stream would mis-tile the other stream output.
    shape_a, mask_vals_a = (3, 256, 256), [256, 128, 200]
    shape_b, mask_vals_b = (5, 128, 512), [128, 64, 96, 32, 100]
    x_a = torch.randn(shape_a, dtype=dtype, device="cuda")
    x_b = torch.randn(shape_b, dtype=dtype, device="cuda")
    mask_a = torch.tensor(mask_vals_a, dtype=torch.int32, device="cuda")
    mask_b = torch.tensor(mask_vals_b, dtype=torch.int32, device="cuda")

    def valid_views(q, sf, masks, m, k):
        padded_k = (k + 127) // 128 * 128
        q_groups = q.permute(2, 0, 1)
        sf_groups = sf.permute(5, 2, 4, 0, 1, 3)
        q_valid = []
        sf_valid = []
        for i, mask_i in enumerate(masks):
            q_valid.append(q_groups[i, :mask_i].contiguous().view(torch.uint8))
            sf_un = _unswizzle_mxfp8_scales_128x4(sf_groups[i], m, padded_k)
            sf_valid.append(sf_un[:mask_i].contiguous())
        return q_valid, sf_valid

    # Per-stream eager references. Also triggers the cuTile JIT compile so the
    # concurrent loop does not compile on a side stream.
    ref_qa, ref_sa = mxfp8_grouped_quantize(x_a, mask_a)
    ref_qb, ref_sb = mxfp8_grouped_quantize(x_b, mask_b)
    torch.cuda.synchronize()
    ref_qa_v, ref_sa_v = valid_views(
        ref_qa, ref_sa, mask_vals_a, shape_a[1], shape_a[2]
    )
    ref_qb_v, ref_sb_v = valid_views(
        ref_qb, ref_sb, mask_vals_b, shape_b[1], shape_b[2]
    )

    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    # Interleave launches on both streams without syncing inside the loop so
    # the two launch sequences actually overlap on the device.
    outs_a = []
    outs_b = []
    for _ in range(32):
        with torch.cuda.stream(stream_a):
            outs_a.append(mxfp8_grouped_quantize(x_a, mask_a))
        with torch.cuda.stream(stream_b):
            outs_b.append(mxfp8_grouped_quantize(x_b, mask_b))
    torch.cuda.synchronize()

    def assert_stream_matches(outs, ref_q_v, ref_s_v, masks, m, k):
        for q, sf in outs:
            q_v, s_v = valid_views(q, sf, masks, m, k)
            for got, ref in zip(q_v, ref_q_v, strict=True):
                torch.testing.assert_close(got, ref, rtol=0, atol=0)
            for got, ref in zip(s_v, ref_s_v, strict=True):
                torch.testing.assert_close(got, ref, rtol=0, atol=0)

    assert_stream_matches(
        outs_a, ref_qa_v, ref_sa_v, mask_vals_a, shape_a[1], shape_a[2]
    )
    assert_stream_matches(
        outs_b, ref_qb_v, ref_sb_v, mask_vals_b, shape_b[1], shape_b[2]
    )


@pytest.mark.parametrize("batch_shape", [(2, 256, 512), (3, 200, 160), (4, 128, 2048)])
@torch.inference_mode()
def test_mxfp8_grouped_quantize_matches_gemm_sfa_layout(batch_shape):
    """Contract test for the grouped MXFP8 quantizer -> masked grouped GEMM seam.

    ``grouped_gemm_nt_masked`` consumes ``(A, SFA)`` in a specific physical
    layout (``A`` logical ``(m, k, l)`` / physical ``(l, m, k)``; ``SFA``
    logical ``(m32, m4, rm, k4, rk, l)`` / physical ``(l, rm, rk, m32, m4, k4)``).
    This test asserts that ``mxfp8_grouped_quantize`` emits the same layout.

    The oracle is the GEMM module's canonical scale builder
    ``create_scale_factor_tensor``, so a failure points unambiguously at the
    quantizer's output format rather than GEMM math. The per-element swizzle
    values are covered separately by ``test_mxfp8_grouped_quantize``.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not _is_mxfp8_supported(torch.device("cuda:0")):
        pytest.skip("mxfp8 grouped quantization requires compute capability >= 10")
    if not is_cutile_available():
        pytest.skip("cuda.tile is not available")
    if not is_cute_dsl_available():
        pytest.skip("nvidia-cutlass-dsl is not available")

    from flashinfer.cute_dsl.utils import get_cutlass_dtype
    from flashinfer.gemm import create_scale_factor_tensor

    torch.manual_seed(0)
    b, m, k = batch_shape
    sf_vec_size = 32
    device = torch.device("cuda:0")

    x = torch.randn(batch_shape, dtype=torch.bfloat16, device=device)
    mask = torch.full((b,), m, dtype=torch.int32, device=device)

    out, sf = mxfp8_grouped_quantize(x, mask)

    # Activation contract: A is logical (m, padded_k, l), physical (l, m, padded_k),
    # i.e. K-contiguous, matching grouped_gemm_nt_masked's a_major="k".
    padded_k = (k + 127) // 128 * 128
    assert out.dtype == torch.float8_e4m3fn
    assert out.shape == (m, padded_k, b)
    assert out.stride() == (padded_k, 1, m * padded_k)

    # Scale contract: match the GEMM's own canonical SFA tensor for the same
    # (l, m, k, sf_vec_size). The batch/group dimension l maps to b.
    _, _, sfa_oracle = create_scale_factor_tensor(
        b, m, k, sf_vec_size, get_cutlass_dtype("float8_e8m0fnu"), device
    )
    assert sf.shape == tuple(sfa_oracle.shape)
    assert sf.stride() == tuple(sfa_oracle.stride())
    # E8M0 scales are byte-sized: the quantizer stores them as uint8 while the
    # GEMM types them as float8_e8m0fnu, so only the element size is compared.
    # The call site reinterprets uint8 <-> float8_e8m0fnu via a dtype view.
    assert sf.element_size() == sfa_oracle.element_size()


@pytest.mark.parametrize(
    "batch_shape", [(1, 120, 64), (2, 128, 128), (3, 256, 160), (4, 200, 2048)]
)
def test_mxfp8_grouped_quantize_fake_op_metadata(batch_shape):
    """Portable metadata check for the grouped MXFP8 fake (meta) op.

    The fake op only allocates empty tensors and applies the layout
    permutes, so it runs on the meta device and needs no GPU, cuTile, or
    SM100. This validates the metadata contract independently of the kernel.
    """
    from flashinfer.quantization.fp8_quantization import (
        get_mxfp8_grouped_quantization_module,
    )

    fake_op = get_mxfp8_grouped_quantization_module()._fake_mxfp8_grouped_quantize

    b, m, k = batch_shape
    a = torch.empty((b, m, k), dtype=torch.bfloat16, device="meta")
    mask = torch.empty((b,), dtype=torch.int32, device="meta")

    out, sf = fake_op(a, mask)

    padded_k = (k + 127) // 128 * 128
    padded_m = (m + 127) // 128 * 128
    assert out.shape == (m, padded_k, b)
    assert out.dtype == torch.float8_e4m3fn
    assert out.stride() == (padded_k, 1, m * padded_k)
    assert sf.shape == (32, 4, padded_m // 128, 4, padded_k // 128, b)
    assert sf.dtype == torch.uint8


@pytest.mark.parametrize("batch_shape", [(2, 128, 128), (3, 256, 160)])
@torch.inference_mode()
def test_mxfp8_grouped_quantize_fake_op_matches_real(batch_shape):
    """The fake op's output metadata must match the real op's (drift guard).

    This test pins the meta kernel to the cuTile kernel:
    a failure means the two have diverged in shape/dtype/stride.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not _is_mxfp8_supported(torch.device("cuda:0")):
        pytest.skip("mxfp8 grouped quantization requires compute capability >= 10")
    if not is_cutile_available():
        pytest.skip("cuda.tile is not available")

    from flashinfer.quantization.fp8_quantization import (
        get_mxfp8_grouped_quantization_module,
    )

    torch.manual_seed(0)
    b, m, k = batch_shape
    x = torch.randn(batch_shape, dtype=torch.bfloat16, device="cuda")
    mask = torch.full((b,), m, dtype=torch.int32, device="cuda")

    module = get_mxfp8_grouped_quantization_module()
    out_real, sf_real = module.mxfp8_grouped_quantize_impl(x, mask)
    out_fake, sf_fake = module._fake_mxfp8_grouped_quantize(x, mask)

    assert out_fake.shape == out_real.shape
    assert out_fake.dtype == out_real.dtype
    assert out_fake.stride() == out_real.stride()
    assert sf_fake.shape == sf_real.shape
    assert sf_fake.dtype == sf_real.dtype
    assert sf_fake.stride() == sf_real.stride()


@pytest.mark.parametrize("m", [1, 2, 16, 1024])
@pytest.mark.parametrize("k", [512, 1024])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
def test_mxfp8_quantize_torch_host(m, k, dtype, is_sf_swizzled_layout):
    torch.random.manual_seed(0)
    a = (torch.randn([m, k], dtype=torch.float) * 16).cpu().contiguous()

    a_fp8, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout)

    _assert_mxfp8_quantize_exact(
        a, a_fp8, a_sf, is_sf_swizzled_layout=is_sf_swizzled_layout
    )


@pytest.mark.parametrize("m", [1, 2, 3, 16, 64, 1024])
@pytest.mark.parametrize("k", [128, 512, 1024, 8192])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("backend", ["cuda", "cute-dsl"])
def test_mxfp8_quantize_torch_device(m, k, dtype, is_sf_swizzled_layout, backend):
    if not _is_mxfp8_supported(torch.device("cuda:0")):
        pytest.skip("mxfp8 quantization is not supported on compute capability < 10")

    if backend == "cute-dsl" and not is_cute_dsl_available():
        pytest.skip("CuTe-DSL is not available")

    torch.random.manual_seed(0)
    a = (torch.randn([m, k], dtype=torch.float) * 16).to(dtype).cuda().contiguous()

    a_fp8, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout, 32, backend=backend)

    _assert_mxfp8_quantize_exact(
        a, a_fp8, a_sf, is_sf_swizzled_layout=is_sf_swizzled_layout
    )
    torch.cuda.synchronize()


@pytest.mark.parametrize("m", [1, 2, 16, 1024])
@pytest.mark.parametrize("k", [1568])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("alignment", [64, 128])
@pytest.mark.parametrize("backend", ["cuda", "cute-dsl"])
def test_mxfp8_quantize_alignment_torch_device(
    m, k, dtype, is_sf_swizzled_layout, alignment, backend
):
    if not _is_mxfp8_supported(torch.device("cuda:0")):
        pytest.skip("mxfp8 quantization is not supported on compute capability < 10")

    if backend == "cute-dsl" and not is_cute_dsl_available():
        pytest.skip("CuTe-DSL is not available")

    torch.random.manual_seed(0)
    a = (torch.randn([m, k], dtype=torch.float) * 16).to(dtype).cuda().contiguous()
    padded_k = ((k + alignment - 1) // alignment) * alignment

    # Quantize it on device.
    a_fp8, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout, alignment, backend=backend)
    assert a_fp8.shape[1] == padded_k

    # Check if the bits of paddings are zero.
    paddings = a_fp8.view(torch.int8)[:, k:]
    assert torch.all(paddings == 0), "Paddings should be zero"

    _assert_mxfp8_quantize_exact(
        a,
        a_fp8,
        a_sf,
        is_sf_swizzled_layout=is_sf_swizzled_layout,
        alignment=alignment,
    )
    torch.cuda.synchronize()


@pytest.mark.parametrize("m", [1, 3, 128, 2048])
@pytest.mark.parametrize("k", [128, 1024])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("backend", ["cuda", "cute-dsl"])
def test_mxfp8_quantize_denormal_inputs(m, k, dtype, is_sf_swizzled_layout, backend):
    """Test that very small denormalized inputs do not produce NaN.

    This test covers a bug where inputs small enough to cause E8M0 scale factor
    underflow would result in NaN outputs due to 0 * infinity computations.
    """
    if not _is_mxfp8_supported(torch.device("cuda:0")):
        pytest.skip("mxfp8 quantization is not supported on compute capability < 10")

    if backend == "cute-dsl" and not is_cute_dsl_available():
        pytest.skip("CuTe-DSL is not available")

    torch.random.manual_seed(42)

    # Create very small denormalized values (below float32 normal range ~1.17e-38)
    # These values caused NaN in the original buggy implementation
    a = (torch.randn([m, k], dtype=torch.float32) * 1e-38).to(dtype).cuda().contiguous()

    a_fp8, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout, backend=backend)

    # The primary check: no NaN values should be produced
    nan_count = torch.isnan(a_fp8.float()).sum().item()
    assert nan_count == 0, f"Found {nan_count} NaN values in output (expected 0)"

    # Secondary check: no Inf values should be produced
    inf_count = torch.isinf(a_fp8.float()).sum().item()
    assert inf_count == 0, f"Found {inf_count} Inf values in output (expected 0)"

    _assert_mxfp8_quantize_exact(
        a, a_fp8, a_sf, is_sf_swizzled_layout=is_sf_swizzled_layout
    )


@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("backend", ["cuda", "cute-dsl"])
def test_mxfp8_quantize_all_zeros(dtype, is_sf_swizzled_layout, backend):
    """Test that all-zero inputs produce all-zero outputs without NaN."""
    if not _is_mxfp8_supported(torch.device("cuda:0")):
        pytest.skip("mxfp8 quantization is not supported on compute capability < 10")

    if backend == "cute-dsl" and not is_cute_dsl_available():
        pytest.skip("CuTe-DSL is not available")

    m, k = 128, 1024
    a = torch.zeros([m, k], dtype=dtype, device="cuda").contiguous()

    a_fp8, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout, backend=backend)

    # No NaN values
    assert not torch.isnan(a_fp8.float()).any(), "NaN found in output for zero input"

    # All outputs should be zero
    assert (a_fp8.float() == 0).all(), "Non-zero output for zero input"

    _assert_mxfp8_quantize_exact(
        a, a_fp8, a_sf, is_sf_swizzled_layout=is_sf_swizzled_layout
    )


@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("backend", ["cuda", "cute-dsl"])
def test_mxfp8_quantize_mixed_magnitude(dtype, is_sf_swizzled_layout, backend):
    """Test mixed inputs: some blocks with normal values, some with denormals.

    This mimics real-world scenarios where different regions of a tensor
    may have vastly different magnitudes.
    """
    if not _is_mxfp8_supported(torch.device("cuda:0")):
        pytest.skip("mxfp8 quantization is not supported on compute capability < 10")

    if backend == "cute-dsl" and not is_cute_dsl_available():
        pytest.skip("CuTe-DSL is not available")

    torch.random.manual_seed(123)

    m, k = 256, 1024
    a = torch.randn([m, k], dtype=torch.float32)

    # Make some rows have very small values (denormals)
    # Rows 0-63: normal magnitude
    # Rows 64-127: very small (denormal range)
    # Rows 128-191: normal magnitude
    # Rows 192-255: extremely small
    a[64:128, :] *= 1e-38
    a[192:256, :] *= 1e-40

    a = a.to(dtype).cuda().contiguous()

    a_fp8, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout, backend=backend)

    # No NaN values should be produced anywhere
    nan_mask = torch.isnan(a_fp8.float())
    nan_count = nan_mask.sum().item()
    if nan_count > 0:
        nan_positions = torch.where(nan_mask)
        first_nan_row = nan_positions[0][0].item()
        first_nan_col = nan_positions[1][0].item()
        pytest.fail(
            f"Found {nan_count} NaN values. First NaN at row={first_nan_row}, col={first_nan_col}"
        )

    _assert_mxfp8_quantize_exact(
        a, a_fp8, a_sf, is_sf_swizzled_layout=is_sf_swizzled_layout
    )


@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("backend", ["cuda", "cute-dsl"])
def test_mxfp8_quantize_single_denormal_in_block(dtype, is_sf_swizzled_layout, backend):
    """Test a block where most values are normal but one is a tiny denormal.

    This specifically tests the scenario from the original bug report where
    a single float32 denormal value in a block would become NaN due to
    0 * infinity when FTZ mode flushes it to zero.
    """
    if not _is_mxfp8_supported(torch.device("cuda:0")):
        pytest.skip("mxfp8 quantization is not supported on compute capability < 10")

    if backend == "cute-dsl" and not is_cute_dsl_available():
        pytest.skip("CuTe-DSL is not available")

    m, k = 64, 1024
    # Start with small but normal-range values
    a = torch.full([m, k], 1e-36, dtype=torch.float32)

    # Insert a few extremely small values (float32 denormals) at specific positions
    # These are the values that triggered NaN in the original bug
    denormal_positions = [(0, 498), (0, 911), (32, 100), (63, 512)]
    for row, col in denormal_positions:
        a[row, col] = 9.18e-40  # A float32 denormal value

    a = a.to(dtype).cuda().contiguous()

    a_fp8, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout, backend=backend)

    # Check that no NaN is produced
    nan_mask = torch.isnan(a_fp8.float())
    assert not nan_mask.any(), f"Found NaN at positions: {torch.where(nan_mask)}"

    _assert_mxfp8_quantize_exact(
        a, a_fp8, a_sf, is_sf_swizzled_layout=is_sf_swizzled_layout
    )


@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("backend", ["cuda", "cute-dsl"])
def test_mxfp8_quantize_extreme_scale_inputs(dtype, is_sf_swizzled_layout, backend):
    if not _is_mxfp8_supported(torch.device("cuda:0")):
        pytest.skip("mxfp8 quantization is not supported on compute capability < 10")

    if backend == "cute-dsl" and not is_cute_dsl_available():
        pytest.skip("CuTe-DSL is not available")

    a = torch.zeros((2, 128), dtype=dtype, device="cuda")
    a[:, 32:64] = float("inf")
    a[:, 64:96] = 448.0
    a[:, 96:128] = -448.0

    a_fp8, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout, backend=backend)

    _assert_mxfp8_quantize_exact(
        a, a_fp8, a_sf, is_sf_swizzled_layout=is_sf_swizzled_layout
    )


# =============================================================================
# CuTe-DSL Compilation Cache Tests
# =============================================================================


@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
def test_cute_dsl_compilation_cache_m_agnostic(is_sf_swizzled_layout):
    """
    Test that the CuTe-DSL compilation cache is M-agnostic.

    Different M values with the same K should reuse the cached kernel,
    meaning no recompilation occurs when only M changes.
    """
    if not _is_mxfp8_supported(torch.device("cuda:0")):
        pytest.skip("mxfp8 quantization is not supported on compute capability < 10")

    if not is_cute_dsl_available():
        pytest.skip("CuTe-DSL is not available")

    from flashinfer.quantization.kernels.mxfp8_quantize import (
        _get_compiled_kernel_mxfp8_linear,
        _get_compiled_kernel_mxfp8_swizzled,
    )

    # Get the appropriate cache based on layout
    if is_sf_swizzled_layout:
        cache_fn = _get_compiled_kernel_mxfp8_swizzled
    else:
        cache_fn = _get_compiled_kernel_mxfp8_linear

    # Clear the cache to start fresh
    cache_fn.cache_clear()

    # Fixed parameters for this test
    K = 1024
    dtype = torch.float16

    # First call with M=1 - should compile
    a1 = torch.randn([1, K], dtype=dtype, device="cuda")
    mxfp8_quantize(a1, is_sf_swizzled_layout, backend="cute-dsl")
    cache_info_after_m1 = cache_fn.cache_info()
    assert cache_info_after_m1.misses == 1, "First call should be a cache miss"
    assert cache_info_after_m1.hits == 0, "First call should have no hits"

    # Second call with M=16 (different M, same K) - should reuse cached kernel
    a2 = torch.randn([16, K], dtype=dtype, device="cuda")
    mxfp8_quantize(a2, is_sf_swizzled_layout, backend="cute-dsl")
    cache_info_after_m16 = cache_fn.cache_info()
    assert cache_info_after_m16.misses == 1, (
        "Second call with different M should still be 1 miss"
    )
    assert cache_info_after_m16.hits == 1, (
        "Second call should be a cache hit (M-agnostic)"
    )

    # Third call with M=1024 (different M again, same K) - should reuse cached kernel
    a3 = torch.randn([1024, K], dtype=dtype, device="cuda")
    mxfp8_quantize(a3, is_sf_swizzled_layout, backend="cute-dsl")
    cache_info_after_m1024 = cache_fn.cache_info()
    assert cache_info_after_m1024.misses == 1, (
        "Third call with different M should still be 1 miss"
    )
    assert cache_info_after_m1024.hits == 2, (
        "Third call should be a cache hit (M-agnostic)"
    )

    # Clean up
    cache_fn.cache_clear()


@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
def test_cute_dsl_compilation_cache_k_specific(is_sf_swizzled_layout):
    """
    Test that the CuTe-DSL compilation cache is K-specific.

    Different K values should create separate cached kernels,
    meaning recompilation occurs when K changes.
    """
    if not _is_mxfp8_supported(torch.device("cuda:0")):
        pytest.skip("mxfp8 quantization is not supported on compute capability < 10")

    if not is_cute_dsl_available():
        pytest.skip("CuTe-DSL is not available")

    from flashinfer.quantization.kernels.mxfp8_quantize import (
        _get_compiled_kernel_mxfp8_linear,
        _get_compiled_kernel_mxfp8_swizzled,
    )

    # Get the appropriate cache based on layout
    if is_sf_swizzled_layout:
        cache_fn = _get_compiled_kernel_mxfp8_swizzled
    else:
        cache_fn = _get_compiled_kernel_mxfp8_linear

    # Clear the cache to start fresh
    cache_fn.cache_clear()

    dtype = torch.float16
    M = 16  # Fixed M

    # First call with K=1024 - should compile
    a1 = torch.randn([M, 1024], dtype=dtype, device="cuda")
    mxfp8_quantize(a1, is_sf_swizzled_layout, backend="cute-dsl")
    cache_info_after_k1024 = cache_fn.cache_info()
    assert cache_info_after_k1024.misses == 1, "First call should be a cache miss"

    # Second call with K=2048 (different K) - should compile new kernel
    a2 = torch.randn([M, 2048], dtype=dtype, device="cuda")
    mxfp8_quantize(a2, is_sf_swizzled_layout, backend="cute-dsl")
    cache_info_after_k2048 = cache_fn.cache_info()
    assert cache_info_after_k2048.misses == 2, (
        "Second call with different K should be a cache miss"
    )

    # Third call with K=1024 again - should hit cache
    a3 = torch.randn([M, 1024], dtype=dtype, device="cuda")
    mxfp8_quantize(a3, is_sf_swizzled_layout, backend="cute-dsl")
    cache_info_after_k1024_again = cache_fn.cache_info()
    assert cache_info_after_k1024_again.misses == 2, (
        "Third call with same K=1024 should not add miss"
    )
    assert cache_info_after_k1024_again.hits >= 1, (
        "Third call with same K=1024 should hit cache"
    )

    # Clean up
    cache_fn.cache_clear()


# =============================================================================
# Backend-parity tests across all SF layouts (128x4 / 8x4 / linear)
# =============================================================================

MXFP8_SF_LAYOUTS = [
    SfLayout.layout_128x4,
    SfLayout.layout_8x4,
    SfLayout.layout_linear,
]


@pytest.mark.parametrize("m", [1, 3, 16, 64, 1024])
@pytest.mark.parametrize("k", [128, 1024, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sf_layout", MXFP8_SF_LAYOUTS)
def test_mxfp8_quantize_layout_backend_parity(m, k, dtype, sf_layout):
    """CUDA and CuTe-DSL backends must exactly match element-wise."""
    if not _is_mxfp8_supported(torch.device("cuda:0")):
        pytest.skip("mxfp8 quantization is not supported on compute capability < 10")
    if not is_cute_dsl_available():
        pytest.skip("CuTe-DSL is not available")

    torch.random.manual_seed(0)
    a = (torch.randn([m, k], dtype=torch.float) * 16).to(dtype).cuda().contiguous()

    a_fp8_cuda, a_sf_cuda = mxfp8_quantize(
        a, sf_swizzle_layout=sf_layout, backend="cuda"
    )
    a_fp8_cute, a_sf_cute = mxfp8_quantize(
        a, sf_swizzle_layout=sf_layout, backend="cute-dsl"
    )

    assert a_fp8_cuda.shape == a_fp8_cute.shape, (
        f"Quantized output shape mismatch for {sf_layout.name}: "
        f"cuda={a_fp8_cuda.shape}, cute={a_fp8_cute.shape}"
    )
    if not torch.equal(a_fp8_cuda, a_fp8_cute):
        mismatch = a_fp8_cuda != a_fp8_cute
        mismatch_count = int(mismatch.sum().item())
        first_index = tuple(
            int(x) for x in torch.nonzero(mismatch, as_tuple=False)[0].cpu()
        )
        raise AssertionError(
            f"Quantized output element mismatch for {sf_layout.name}: "
            f"{mismatch_count}/{a_fp8_cuda.numel()} elements differ; "
            f"first mismatch at index {first_index}: "
            f"cuda={a_fp8_cuda[first_index].float().item()}, "
            f"cute={a_fp8_cute[first_index].float().item()}"
        )

    assert a_sf_cuda.shape == a_sf_cute.shape, (
        f"Scale factor shape mismatch for {sf_layout.name}: "
        f"cuda={a_sf_cuda.shape}, cute={a_sf_cute.shape}"
    )
    if not torch.equal(a_sf_cuda, a_sf_cute):
        mismatch = a_sf_cuda != a_sf_cute
        mismatch_count = int(mismatch.sum().item())
        first_index = tuple(
            int(x) for x in torch.nonzero(mismatch, as_tuple=False)[0].cpu()
        )
        raise AssertionError(
            f"Scale factor element mismatch for {sf_layout.name}: "
            f"{mismatch_count}/{a_sf_cuda.numel()} elements differ; "
            f"first mismatch at index {first_index}: "
            f"cuda={int(a_sf_cuda[first_index].item())}, "
            f"cute={int(a_sf_cute[first_index].item())}"
        )


@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
def test_cute_dsl_compilation_cache_dtype_specific(is_sf_swizzled_layout):
    """
    Test that the CuTe-DSL compilation cache is dtype-specific.

    Different dtypes (fp16 vs bf16) should create separate cached kernels.
    """
    if not _is_mxfp8_supported(torch.device("cuda:0")):
        pytest.skip("mxfp8 quantization is not supported on compute capability < 10")

    if not is_cute_dsl_available():
        pytest.skip("CuTe-DSL is not available")

    from flashinfer.quantization.kernels.mxfp8_quantize import (
        _get_compiled_kernel_mxfp8_linear,
        _get_compiled_kernel_mxfp8_swizzled,
    )

    # Get the appropriate cache based on layout
    if is_sf_swizzled_layout:
        cache_fn = _get_compiled_kernel_mxfp8_swizzled
    else:
        cache_fn = _get_compiled_kernel_mxfp8_linear

    # Clear the cache to start fresh
    cache_fn.cache_clear()

    K = 1024
    M = 16

    # First call with float16 - should compile
    a1 = torch.randn([M, K], dtype=torch.float16, device="cuda")
    mxfp8_quantize(a1, is_sf_swizzled_layout, backend="cute-dsl")
    cache_info_after_fp16 = cache_fn.cache_info()
    assert cache_info_after_fp16.misses == 1, "First call (fp16) should be a cache miss"

    # Second call with bfloat16 (different dtype, same K) - should compile new kernel
    a2 = torch.randn([M, K], dtype=torch.bfloat16, device="cuda")
    mxfp8_quantize(a2, is_sf_swizzled_layout, backend="cute-dsl")
    cache_info_after_bf16 = cache_fn.cache_info()
    assert cache_info_after_bf16.misses == 2, (
        "Second call (bf16) should be a cache miss (dtype-specific)"
    )

    # Third call with float16 again - should hit cache
    a3 = torch.randn([M, K], dtype=torch.float16, device="cuda")
    mxfp8_quantize(a3, is_sf_swizzled_layout, backend="cute-dsl")
    cache_info_after_fp16_again = cache_fn.cache_info()
    assert cache_info_after_fp16_again.misses == 2, (
        "Third call (fp16 again) should not add miss"
    )
    assert cache_info_after_fp16_again.hits >= 1, (
        "Third call (fp16 again) should hit cache"
    )

    # Clean up
    cache_fn.cache_clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
